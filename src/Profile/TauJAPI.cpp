#include <jni.h>
#include "Profile/Profiler.h"
#include "Profile/TauJAPI.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <sys/types.h>


#ifndef TAU_ANDROID
#define LOGV(...) printf(__VA_ARGS__)
#define LOGF(...) printf(__VA_ARGS__)
pid_t gettid(void);
static int dalvik_vm_running = 0;
static pid_t finalizer;
#else
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

#define LOGV(...) //__android_log_print(ANDROID_LOG_VERBOSE, "TAU", __VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL, "TAU", __VA_ARGS__)

static pid_t finalizer;

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

static int
handle_ddm_event(jdwp_ctx_t *jdwp, ddm_trunk_t *trunk)
{
    int i;
    ddm_thcr_t *thcr;
    ddm_thde_t *thde;
    ddm_thst_t *thst;
    ddm_trunk_t query;

    char *tname;
    pid_t sid = -1;    // system thread id, i.e. gettid()

    /*
     * Dalvik vm-local thread id, free after thread death, reuseable
     * See <android>/dalvik/vm/Thread.cpp
     */
    uint32_t lid;  

    static map<uint32_t, char*> java_thread_name;  // lid ==> tname
    static map<uint32_t, pid_t> java_thread_sid;   // lid ==> sid

    static map<uint32_t, pid_t> thst_map;          // lid ==> sid

    /* list of <lid, live?> */
    static list< pair <uint32_t, bool> > tstates;

    /*
     * How does this work:
     *
     *  - Each thread is represened as a lid
     *  - We send THST as soon as we recv THCR for a lid to query for
     *    the sid
     *  - We don't play any fancy here, meaning that we will not cache
     *    or reuse lid-sid map returned by THCR. We do not do that
     *    because it is not reliable, Read Dalvik source code to see
     *    why.
     *  - If THDE for a lid comes before corresponding THST, we say the
     *    lid is ephemeral. We do not register ephemeral lid as the THST
     *    returns later is not reliable.
     *
     * In short: we only register the lid if its THST comes before THDE.
     * We ignore all other lids.
     */
    switch (ntohl(trunk->type)) {
    case DDM_THCR:
	thcr  = (ddm_thcr_t*)trunk;
	lid   = ntohl(thcr->lid);
	tname = utf16_to_ascii(thcr->tname, ntohl(thcr->tname_len));

	/* setup mapping between lid and tname */
	if (java_thread_name.find(lid) != java_thread_name.end()) {
	    free(java_thread_name[lid]);
	}
	java_thread_name[lid] = tname;

	/*
	 * lid:
	 * <1> main
	 * <2> GC                         \
	 * <3> Signal Catcher              |
	 * <4> JDWP                        |
	 * <5> Compiler                    +> Dalvik internal threads
	 * <6> ReferenceQueueDaemon        |
	 * <7> FinalizerDaemon             |
	 * <8> FinalizerWatchdogDaemon    /
	 * <9> ~ : user threads
	 *
	 * We should monitor main and all user threads. We should also watch the
	 * finalizer daemons as they may call user provided finalize() methods
	 */
	if ((lid >= 2) && (lid <=8 ) && (lid != 7)) {
	    break;
	}

	query.type   = htonl(DDM_THST);
	query.length = htonl(0);

	jdwp_send_pkt(jdwp, DDM_TRUNK, (char*)&query, sizeof(query));

	LOGV(" *** DDM THCR <%d> %s send THST", lid, tname);

	tstates.push_back(pair<uint32_t,bool>(lid, true));

	break;

    case DDM_THST:
	thst = (ddm_thst_t*)trunk;

	/* decode lid-sid map */
	thst_map.clear();
	for (i=0; i<ntohs(thst->count); i++) {
	    lid = ntohl(thst->thst[i].lid);
	    sid = ntohl(thst->thst[i].sid);

	    LOGV(" *** DDM THST: %d: %d -> %d\n", i, lid, sid);
	    thst_map[lid] = sid;
	}

	/* tstates list must not be empty as we received a THST */
	if (tstates.begin()->second == true) {
	    /* the thread is still alive, let's register it */
	    lid = tstates.begin()->first;
	    if (lid == 7) {
		/*
		 * FinalizerDaemon will call Java class finalizers. Thus we
		 * must register FinalizerDaemon if we are going to inject
		 * the finalizer. However, this in practice usually doesn't
		 * work. Dalvik will throw an error message telling that a
		 * timeout occurs in finalize(), then it aborts the App.
		 * Therefore we choose not to inject finalizer().
		 *
		 * Note that we don't know which method will be called by
		 * finalizer(), so we save sid of FinalizerDaemon here and
		 * avoid any TAU API call in that thread.
		 */
		finalizer = thst_map[lid];
		tstates.pop_front();
		LOGV(" *** Ignore finalizer daemon sid = %d", finalizer);
		break;
	    }

	    if (thst_map.find(lid) != thst_map.end()) {
		java_thread_sid[lid] = thst_map[lid];

		JNIThreadLayer::SuThread(java_thread_sid[lid], java_thread_name[lid]);
		JNIThreadLayer::RegisterThread(java_thread_sid[lid], java_thread_name[lid]);
	    }

	    tstates.pop_front();
	} else {
	    /* the thread is already dead */
	    tstates.pop_front();
	}

	break;

    case DDM_THDE:
	thde  = (ddm_thde_t*)trunk;
	lid   = ntohl(thde->lid);
	tname = java_thread_name[lid];

	if (lid == 1) {
	    dalvik_vm_running = 0;
	}

	if (java_thread_sid.find(lid) != java_thread_sid.end()) {
	    /* yes, we are registered, now unregester */
	    sid = java_thread_sid[lid];

	    JNIThreadLayer::SuThread(sid, tname);
	    Tau_stop_all_timers(RtsLayer::myThread());

	    java_thread_sid.erase(lid);

	    LOGV(" *** DDM THDE <%d> %s unregistered", lid, tname);
	} else {
	    /* no, we are not registered yet, remove from waiting list */
	    list< pair<uint32_t, bool> >::iterator itor;
	    for (itor=tstates.begin(); itor!=tstates.end(); itor++) {
		if (itor->first == lid) {
		    itor->second = false; // mark of death
		    LOGV(" *** DDM THDE <%d> %s mark as death", lid, tname);
		    break;
		}
	    }

	    if (itor == tstates.end()) {
		LOGF(" *** DDM THDE <%d> %s not regestered and not in waiting list\n", lid, tname);
	    }
	}

	free(tname);
	java_thread_name.erase(lid);

	break;

    defalt:
	LOGF(" *** DTM: ignore DDM event %08x\n", ntohl(trunk->type));
	break;
    }

    return 0;
}

static void*
dalvik_thread_monitor(void *arg)
{
    jdwp_ctx_t jdwp;
    jdwp_cmd_t *cmd;

    JNIThreadLayer::IgnoreThisThread();

    if (jdwp_init(&jdwp) < 0) {
	LOGF(" *** Error: DTM: jdwp failed to init\n");
	return NULL;
    }

    ddm_helo(&jdwp);
    ddm_then(&jdwp);

    LOGV(" *** (S%d) DTM started\n", gettid());

    while (1) {
	/* is there any pending events in backlog? */
	if (jdwp.events == NULL) {
	    /* Nope! Let's wait for new events coming */
	    cmd = (jdwp_cmd_t*)jdwp_recv_pkt(&jdwp);

	    /* something really bad happened, end of watch */
	    if (cmd == NULL) {
		LOGF("Error: JDWP: disconnect...\n");
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

	    /* this should be a THST response */
	    if (event->cmd->flags == 0x80) {
		handle_ddm_event(&jdwp, (ddm_trunk_t*)event->cmd->data);
	    }

	    /* this should be a THCR/THDE notification */
	    if (((event->cmd->cmd_set << 8) | event->cmd->command) == DDM_TRUNK) {
		handle_ddm_event(&jdwp, (ddm_trunk_t*)event->cmd->data);
	    }

	    free(event->cmd);
	    free(event);
	}
    }

    if (!adb_is_active(jdwp.adb)) {
	LOGF("Error: JDWP: connection closed\n");
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

    RtsLayer::TheUsingJNI() = true;
    JNIThreadLayer::tauVM = vm;

#ifdef TAU_ANDROID
    pthread_t thr;
    pthread_create(&thr, NULL, dalvik_thread_monitor, NULL);
#endif

    return JNI_VERSION_1_6;
}

/*
 * Java: Thread.currentThread().getId();
 * This ID (jid) simply count upwards, so each Thread has a unique ID.
 * See <android>/libcore/libdvm/src/main/java/java/lang/Thread.java
 */
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

/*
 * Java: Thread.currentThread().getName();
 */
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
    if (jstr == NULL) {
	return NULL;
    }

    const char *jname = env->GetStringUTFChars(jstr, NULL);
    if (jname == NULL) {
	return NULL;
    }

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
  if (gettid() == finalizer) {
    return;
  }
  JNIThreadLayer::WaitForDTM();

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
  if (gettid() == finalizer) {
    return;
  }

  /* Find the FunctionInfo Pointer associated with this method*/
  jclass cls = env->GetObjectClass(obj);
  jfieldID fid;
  FunctionInfo *f; 

  JNIThreadLayer::WaitForDTM();

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
  if (gettid() == finalizer) {
    return;
  }

  TAU_GLOBAL_TIMER_STOP();
}

/* EOF Profile.cpp */

/***************************************************************************
 * $RCSfile: TauJAPI.cpp,v $   $Author: amorris $
 * $Revision: 1.3 $   $Date: 2009/02/19 20:08:29 $
 * TAU_VERSION_ID: $Id: TauJAPI.cpp,v 1.3 2009/02/19 20:08:29 amorris Exp $
 ***************************************************************************/

