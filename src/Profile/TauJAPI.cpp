#include <jni.h>
#include "Profile/Profiler.h"
#include "Profile/TauJAPI.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#ifdef TAU_ANDROID

#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>

#include "Profile/adb.h"
#include "Profile/jdwp.h"

extern "C" {
    jint android_log(const char *message);
}
extern void CreateTopLevelRoutine(char *name, char *type, char *groupname, int tid);

#ifdef TAU_PTHREAD_WRAP

typedef int (*pcreate_t)(pthread_t*, const pthread_attr_t*, void *(*)(void*), void*);

pcreate_t pcreate;

typedef struct {
    void *(*start_routing)(void*);
    void *arg;
} arg_t;

static void
cleanup_handler(void *arg)
{
    printf("thread %d terminate\n", gettid());
    Tau_stop_top_level_timer_if_necessary();
}

static void*
thread_wrap(void *arg)
{
    void *rv;
    arg_t *arg_wrap = (arg_t*)arg;
    printf("thread %d start\n", gettid());

    JNIThreadLayer::RegisterThread(NULL);
    Tau_create_top_level_timer_if_necessary();

    pthread_cleanup_push(cleanup_handler, NULL);

    rv = arg_wrap->start_routing(arg_wrap->arg);

    pthread_cleanup_pop(1);

    return rv;
}

/*
 * pthread_create() wrap
 */
int
pthread_create(pthread_t *thread, const pthread_attr_t *attr,
	       void *(*start_routing)(void*), void *arg)
{
    arg_t *arg_wrap = (arg_t*)malloc(sizeof(*arg_wrap));

    if (pcreate == NULL) {
	void *ptr = dlsym(RTLD_NEXT, "pthread_create");
	pcreate = reinterpret_cast<pcreate_t>(reinterpret_cast<long>(ptr)) ;
    }

    if (RtsLayer::TheUsingJNI()) {
	arg_wrap->start_routing = start_routing;
	arg_wrap->arg           = arg;

	printf("pthread_create: %p %p %p %p\n", pcreate, arg_wrap, start_routing, arg);
	return pcreate(thread, attr, thread_wrap, arg_wrap);
    } else {
	return pcreate(thread, attr, start_routing, arg);
    }
}

#endif


jlong &TheLastJDWPEventThreadID()
{
    static jlong jid = 1;

    return jid;
}

static int dalvik_vm_running = 1;

static void*
dalvik_thread_monitor(void *arg)
{
    jdwp_ctx_t jdwp;
    jdwp_cmd_t *cmd;

    jlong jid = 9;
    std::map<uint64_t, jlong> java_threads;

    jdwp_init(&jdwp);

    jdwp_set_event_request(&jdwp, E_THREAD_START, SUSPEND_EVENT_THREAD);
    jdwp_set_event_request(&jdwp, E_THREAD_END, SUSPEND_NONE);

    while (1) {
	/* is there any pending events in backlog? */
	if (jdwp.events == NULL) {
	    /* Nope! Let's wait for new events coming */
	    cmd = (jdwp_cmd_t*)jdwp_recv_pkt(&jdwp);

	    /* something really bad happened, end of watch */
	    if (cmd == NULL) {
		fprintf(stderr, "Error: JDWP: disconnect...\n");
		break;
	    }

	    /* put the events into backlog */
	    jdwp_event_backlog(&jdwp, cmd);
	} else {
	    /* Yep! Let's deal with them first */
	    jdwp_event_t *event = jdwp.events;

	    if (jdwp.events->next == jdwp.events) {
		jdwp.events = NULL;
	    } else {
		jdwp.events->next->prev = jdwp.events->prev;
		jdwp.events->prev->next = jdwp.events->next;
		jdwp.events             = jdwp.events->next;
	    }

	    if (event->eventKind == E_THREAD_START) {
		/* get thread name */
		char *tname = jdwp_get_thread_name(&jdwp, event->threadID);
		if (tname == NULL) {
		    break;
		}

		uint64_t grpID = jdwp_get_thread_group(&jdwp, event->threadID);
		char *gname = jdwp_get_thread_group_name(&jdwp, grpID);

		java_threads[event->threadID] = jid;
		TheLastJDWPEventThreadID()    = jid;
		int tid = JNIThreadLayer::RegisterThread(jid);

		CreateTopLevelRoutine(strchr(tname, ' ')+1, (char*)" ", gname, tid);

		jid++;

		free(tname);
		free(gname);
	    }

	    if (event->eventKind == E_THREAD_END) {
		if (java_threads.find(event->threadID) != java_threads.end()) {
		    TheLastJDWPEventThreadID() = java_threads[event->threadID];
		    java_threads.erase(event->threadID);

		    TAU_PROFILE_EXIT("END...");
		}
	    }

	    if (event->eventKind == E_VM_DEATH) {
		dalvik_vm_running = 0;
		printf(" *** dalvik dead\n");
		break;
	    }

	    /* resume thread */
	    if (event->suspendPolicy != SUSPEND_NONE) {
		jdwp_resume_thread(&jdwp, event->threadID);
	    }

	    /* we are done with this event */
	    free(event);
	}
    }

    if (!adb_is_active(jdwp.adb)) {
	fprintf(stderr, "Error: JDWP: connection closed\n");
    }

    return NULL;
}

#endif

/*
 * The VM calls JNI_OnLoad() when the native library is loaded
 */
FILE *tau_verbose_fp;
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    printf("TAU: JNI_OnLoad\n");

    tau_verbose_fp = stderr;

    /*
     * This is a good point to attach your gdb on JVM to debug TAU
     */
    //getchar();

    RtsLayer::TheUsingJNI() = true;
    JNIThreadLayer::tauVM = vm;

    /*
     * thread ID of Java main() is 1.
     *
     * NOTE: This is not a portable implementation as we made this asusmption.
     *       See dalvik_thread_monitor() for more details.
     */
    JNIThreadLayer::RegisterThread(1);

#ifdef TAU_ANDROID
    pthread_t thr;
    printf("TAU: start thread monitor\n");
    pthread_create(&thr, NULL, dalvik_thread_monitor, NULL);
#endif

    return JNI_VERSION_1_6;
}

// Java: Thread.currentThread().getId();
jlong get_java_thread_id(void)
{
    JavaVM *vm = JNIThreadLayer::tauVM;
    JNIEnv *env;

    if (!dalvik_vm_running) {
	return 1;
    }

    if (vm == NULL) {
	return -1;
    }

    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
	return -1;
    }

    jclass thread = env->FindClass("java/lang/Thread");
    if (thread == NULL) {
	return -1;
    }

    jmethodID currentThread = env->GetStaticMethodID(thread, "currentThread", "()Ljava/lang/Thread;");
    if (currentThread == NULL) {
	return -1;
    }

    jobject thisThread = env->CallStaticObjectMethod(thread, currentThread);
    if (thisThread == NULL) {
	return -1;
    }

    jmethodID getId = env->GetMethodID(thread, "getId", "()J");
    if (getId == NULL) {
	return -1;
    }

    jlong id = env->CallLongMethod(thisThread, getId);

    /*
     * LocalRef should be deleted after use, otherwise it may overflow
     * Java native method's local reference table
     */
    env->DeleteLocalRef(thread);
    env->DeleteLocalRef(thisThread);

    return id;
}

jint android_log(const char *message)
{
    JavaVM *vm = JNIThreadLayer::tauVM;
    JNIEnv *env;

    printf("android_log(\"%s\")\n", message);

    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
	printf("can't GetEnv\n");
	return -1;
    }
 
    jclass log = env->FindClass("android/util/Log");
    if (log == NULL) {
	printf("can't FindClass(\"android/util/Log\")\n");
	return -1;
    }

    jstring tag = env->NewStringUTF("TAU");
    if (tag == NULL) {
	printf("can't NewStringUTF(\"TAU\")\n");
	return -1;
    }
    jstring msg = env->NewStringUTF(message);
    if (msg == NULL) {
	printf("can't NewStringUTF(\"%s\")\n", message);
	return -1;
    }

    jmethodID logV = env->GetStaticMethodID(log, "v", "(Ljava/lang/String;Ljava/lang/String;)I");
    if (logV == NULL) {
	printf("cant GetStaticMethodID\n");
	return -1;
    }

    printf("going to call static method\n");

    return env->CallStaticIntMethod(log, logV, tag, msg);
}


/*
 * Class:     Profile
 * Method:    NativeProfile
 * Signature: (Ljava/lang/String;Ljava/lang/String;J)V
 */

JNIEXPORT void JNICALL Java_edu_uoregon_TAU_Profile_NativeProfile
  (JNIEnv *env, jobject obj, jstring name, jstring type, jstring groupname, 
	jlong group)
{

  /* Get name and type strings from the JVM */
  const char *blockName = env->GetStringUTFChars(name, 0);
  const char *blockType = env->GetStringUTFChars(type, 0);
  const char *blockGroup = env->GetStringUTFChars(groupname, 0);
  /* create a new FunctionInfo object by passing these to it */
  FunctionInfo *f = new FunctionInfo(blockName, blockType, (TauGroup_t) group, 
	blockGroup, true);
  /* true indicates InitData will ensure that all data is clean */


  /* Now release the strings back to the JVM */
  env->ReleaseStringUTFChars(name, blockName);
  env->ReleaseStringUTFChars(type, blockType);
  env->ReleaseStringUTFChars(groupname, blockGroup);

  /* Find the field FuncInfoPtr in the Profile class where we need to store 
     the address of the FunctionInfo object just created */

  jclass cls = env->GetObjectClass(obj);
  jfieldID fid = env->GetFieldID(cls, "FuncInfoPtr", "J");


  /* Check if new was successful */

  if (f == (FunctionInfo *) NULL)
  {
    cout << "ERROR: FunctionInfo new returns NULL: Memory problem"<<endl;
  }

  /* Store the address of f in the Java class field where it can be accessed
     by successive JNI calls such as Start and Stop */

  env->SetLongField(obj, fid, (jlong) f); 
  DEBUGPROFMSG("Java_Profile_NativeProfile: FunctionInfoPtr set to "<<f<<endl);

}


/*
 * Class:     Profile
 * Method:    NativeStart
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_uoregon_TAU_Profile_NativeStart
  (JNIEnv *env, jobject obj)
{

  /* Find the FunctionInfo Pointer associated with this method*/
  jclass cls = env->GetObjectClass(obj);
  jfieldID fid;
  FunctionInfo *f; 

  fid = env->GetFieldID(cls, "FuncInfoPtr", "J");

  f = (FunctionInfo *) env->GetLongField(obj, fid);

  TAU_PROFILE_START(f);
}




/*
 * Class:     Profile
 * Method:    NativeStop
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_uoregon_TAU_Profile_NativeStop
  (JNIEnv * env, jobject obj) {
 TAU_GLOBAL_TIMER_STOP();
}

/* EOF Profile.cpp */

/***************************************************************************
 * $RCSfile: TauJAPI.cpp,v $   $Author: amorris $
 * $Revision: 1.3 $   $Date: 2009/02/19 20:08:29 $
 * TAU_VERSION_ID: $Id: TauJAPI.cpp,v 1.3 2009/02/19 20:08:29 amorris Exp $
 ***************************************************************************/

