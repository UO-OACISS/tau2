#include <jni.h>
#include "Profile/Profiler.h"
#include "Profile/TauJAPI.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#ifndef TAU_ANDROID
#define LOGV(...) printf(__VA_ARGS__)
#else
#include <sys/types.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>

#include <android/log.h>

#include "Profile/adb.h"
#include "Profile/jdwp.h"
#include "Profile/ddm.h"

#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "TAU", __VA_ARGS__)

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
    Tau_stop_top_level_timer_if_necessary();
}

static void*
thread_wrap(void *arg)
{
    void *rv;
    arg_t *arg_wrap = (arg_t*)arg;
    static jlong jid = 0;

    JNIThreadLayer::RegisterThread(jid++, "thread-"+gettid());
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

static char *
utf16_to_ascii(char *utf16, int len)
{
    int i;
    char *ascii;

    ascii = (char*)malloc(len+1);
    if (ascii == NULL) {
	return NULL;
    }

    for (i=0; i<len; i++) {
	ascii[i] = ntohs(((short*)utf16)[i]) & 0xff;
    }

    ascii[i] = 0;

    return ascii;
}

static int dalvik_vm_running = 1;

static jlong
get_jid_from_lid(uint32_t lid)
{
    static int sys_count = 0;  // how many system thread has appeared?
    static jlong jid = 0;

    jid++;

    /* this is a system thread, jid == lid */
    if (lid < 9) {
	sys_count++;
	return (jlong)lid;
    }

    /*
     * this is not a system thread, and not all system thread has been
     * populated, jid == lid
     */
    if (sys_count < 8) {
	return (jlong)lid;
    }

    return jid;
}

static int
handle_ddm_event(jdwp_ctx_t *jdwp, ddm_trunk_t *trunk)
{
    int i;
    ddm_thcr_t *thcr;
    ddm_thde_t *thde;
    ddm_thst_t *thst;

    int tid;         // tau internal thread id, +1 for each new thread
    int sid = -1;    // system thread id, i.e. gettid()
    jlong jid = 0;   // java thread id, +1 for each new thread
    char *tname;

    // dalvik vm-local thread id, free after thread death, reuseable
    uint32_t lid;  

    static map<uint32_t, char*> java_thread_name;  // lid ==> tname
    static map<uint32_t, jlong> java_thread_jid;   // lid ==> jid

    switch (ntohl(trunk->type)) {
    case DDM_THCR:
	thcr  = (ddm_thcr_t*)trunk;
	lid   = ntohl(thcr->lid);
	tname = utf16_to_ascii(thcr->tname, ntohl(thcr->tname_len));

	jid = get_jid_from_lid(lid);

	/* setup mapping between lid and tname */
	if (java_thread_name.find(lid) != java_thread_name.end()) {
	    free(java_thread_name[lid]);
	}
	java_thread_name[lid] = tname;

	/*
	 * lid 1  : main
	 * lid 2~8: dalvik internel threads
	 * lid 9~ : user threads
	 */
	if ((lid == 1) || (lid >= 9)) {
	    /* find out sid of this thread, we will need this for sampling */
	    thst = ddm_thst(jdwp);
	    for (i=0; i<ntohs(thst->count); i++) {
		TAU_VERBOSE(" *** THST: %d: %08x -> %08x, %d -> %d\n", i,
			    thst->thst[i].lid, thst->thst[i].sid,
			    ntohl(thst->thst[i].lid), ntohl(thst->thst[i].sid));
		if (lid == ntohl(thst->thst[i].lid)) {
		    sid = ntohl(thst->thst[i].sid);

		    TAU_VERBOSE(" *** going to map jid %lld to sid %d, i=%d\n",
				jid, sid, i);
		    break;
		}
	    }
	    free(thst);

	    /*
	     * Try to register this new thread.
	     *
	     * Note that this is an async event, hence
	     * 1. the thread may be running and already registered itself. We
	     *    shall handle this in RegisterThread().
	     * 2. the thread may already dead, in which case sid will be -1.
	     *    Let's don't bother to register it.
	     */
	    if (sid != -1) {
		/* setup mapping between lid and jid */
		java_thread_jid[lid]  = jid;

		TheLastJDWPEventThreadID() = jid;
		tid = JNIThreadLayer::RegisterThread(jid, sid, tname);

		LOGV(" *** DDM THCR <%d> %s -- Registered\n", lid, tname);
	    } else {
		LOGV(" *** DDM THCR <%d> %s\n", lid, tname);
	    }
	} else {
	    LOGV(" *** DDM THCR <%d> %s\n", lid, tname);
	}

	break;

    case DDM_THDE:
	thde  = (ddm_thde_t*)trunk;
	lid   = ntohl(thde->lid);
	tname = java_thread_name[lid];

	LOGV(" *** DDM THDE <%d> %s\n", lid, tname);
	free(tname);
	java_thread_name.erase(lid);

	if (lid == 1) {
	    dalvik_vm_running = 0;
	}

	if (java_thread_jid.find(lid) != java_thread_jid.end()) {
	    TheLastJDWPEventThreadID() = java_thread_jid[lid];
	    java_thread_jid.erase(lid);

	    TAU_PROFILE_EXIT("END...");
	}

	break;

    defalt:
	LOGV("Error: DTM: ignore DDM event %08x\n", ntohl(trunk->type));
	break;
    }

    return 0;
}

static void*
dalvik_thread_monitor(void *arg)
{
    jdwp_ctx_t jdwp;
    jdwp_cmd_t *cmd;

    if (jdwp_init(&jdwp) < 0) {
	LOGV(" *** Error: DTM: jdwp failed to init\n");
	return NULL;
    }

    ddm_helo(&jdwp);
    ddm_then(&jdwp);

    while (1) {
	/* is there any pending events in backlog? */
	if (jdwp.events == NULL) {
	    /* Nope! Let's wait for new events coming */
	    cmd = (jdwp_cmd_t*)jdwp_recv_pkt(&jdwp);

	    /* something really bad happened, end of watch */
	    if (cmd == NULL) {
		LOGV("Error: JDWP: disconnect...\n");
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

	    switch ((event->cmd->cmd_set << 8) | event->cmd->command) {
	    case DDM_TRUNK:
		handle_ddm_event(&jdwp, (ddm_trunk_t*)event->cmd->data);
		free(event->cmd);
		free(event);
		break;
	    case EVENT_COMPOSIT:
		LOGV("Error: DTM: ignore JDWP EVENT COMPOSIT\n");
		break;
	    default:
		LOGV("Error: DTM: ignore unknown JDWP event\n");
		break;
	    }
	}
    }

    if (!adb_is_active(jdwp.adb)) {
	LOGV("Error: JDWP: connection closed\n");
    }

    return NULL;
}

#include <sys/stat.h>
#include <fcntl.h>
static void
dump_proc_self_maps(void)
{
    char buf[128];

    int ifd = open("/proc/self/maps", O_RDONLY);
    if (ifd < 0) {
	LOGV(" *** open maps: %s", strerror(errno));
	return;
    }

    int ofd = open("/sdcard/self_maps", O_WRONLY);
    if (ofd < 0) {
	LOGV(" *** open sdcard maps: %s", strerror(errno));
	return;
    }

    while (1) {
	int rv = read(ifd, buf, sizeof(buf));

	if (rv > 0) {
	    write(ofd, buf, rv);
	} else {
	    break;
	}
    }

    close(ifd);
    close(ofd);
}

#endif

/*
 * The VM calls JNI_OnLoad() when the native library is loaded
 */
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    LOGV(" *** JNI_OnLoad");

    //dump_proc_self_maps();

    /*
     * This is a good point to attach your gdb on JVM to debug TAU
     */
    //getchar();

    /* do we have a JNI Env pointer? */
    JNIEnv *env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
	LOGV(" *** can not get env pointer in JNI_OnLoad\n");
    } else {
	LOGV(" *** env pointer in JNI_OnLoad is %p\n", env);
    }

    RtsLayer::TheUsingJNI() = true;
    JNIThreadLayer::tauVM = vm;

    /*
     * thread ID of Java main() is 1.
     *
     * NOTE: This is not a portable implementation as we made this asusmption.
     *       See dalvik_thread_monitor() for more details.
     */
    //JNIThreadLayer::RegisterThread(1, "main");

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

    /*
     * Note that we may still running even after dalvik vm is dead, in which
     * case the jid should be 1, i.e. the "main" thread.
     */
    if (!dalvik_vm_running) {
	return 1;
    }

    /* sanity check */
    if (vm == NULL) {
	return -1;
    }

    /*
     * Note that DTM(Dalvik Monitor Thread) is just a pthread. It's not attached
     * to dalvik vm, i.e. not a java thread. So there is no env pointer, and it
     * doesn't have a java thread id.
     */
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
	return -1;
    }

    jclass thread = env->FindClass("java/lang/Thread");
    if (thread == NULL) {
	return -1;
    }

    jmethodID currentThread = env->GetStaticMethodID(thread, "currentThread",
						     "()Ljava/lang/Thread;");
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

// Java: Thread.currentThread().getName();
char *get_java_thread_name(void)
{
    JavaVM *vm = JNIThreadLayer::tauVM;
    JNIEnv *env;

    /*
     * Note that we may still running even after dalvik vm is dead, in which
     * case the jid should be 1, i.e. the "main" thread.
     */
    if (!dalvik_vm_running) {
	return NULL;
    }

    /* sanity check */
    if (vm == NULL) {
	return NULL;
    }

    /*
     * Note that DTM(Dalvik Monitor Thread) is just a pthread. It's not attached
     * to dalvik vm, i.e. not a java thread. So there is no env pointer, and it
     * doesn't have a java thread id.
     */
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
	return NULL;
    }

    jclass thread = env->FindClass("java/lang/Thread");
    if (thread == NULL) {
	return NULL;
    }

    jmethodID currentThread = env->GetStaticMethodID(thread, "currentThread",
						     "()Ljava/lang/Thread;");
    if (currentThread == NULL) {
	return NULL;
    }

    jobject thisThread = env->CallStaticObjectMethod(thread, currentThread);
    if (thisThread == NULL) {
	return NULL;
    }

    jmethodID getName = env->GetMethodID(thread, "getName", "()Ljava/lang/String;");
    if (getName == NULL) {
	return NULL;
    }

    jstring jstr = (jstring) env->CallObjectMethod(thisThread, getName);

    const char *jname = env->GetStringUTFChars(jstr, NULL);

    char *name = strdup(jname);

    env->ReleaseStringUTFChars(jstr, jname);

    /*
     * LocalRef should be deleted after use, otherwise it may overflow
     * Java native method's local reference table
     */
    env->DeleteLocalRef(thread);
    env->DeleteLocalRef(thisThread);

    return name;
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

