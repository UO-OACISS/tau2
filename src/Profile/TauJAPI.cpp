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

#include "Profile/adb.h"
#include "Profile/jdwp.h"

extern "C" {
    jint android_log(const char *message);
}

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

typedef struct jdwp_event {
    char eventKind;
    long long threadID;
    struct jdwp_event *next;
    struct jdwp_event *prev;
} jdwp_event_t;

static int
event_backlog(jdwp_event_t **backlog, jdwp_cmd_t *cmd)
{
    int i, offset;
    char *data;
    char suspendPolicy;
    int  eventCount;

    jdwp_event_t *event;

    /* this must be a command, not a reply */
    if (cmd->flags == 0x80) {
	fprintf(stderr, "Error: JDWP: ignore a reply pkt.\n");
	return -1;
    }

    /* and the command must be EVENT_COMPOSIT */
    if (((cmd->cmd_set << 8) | cmd->command) != EVENT_COMPOSIT) {
	fprintf(stderr, "Error: JDWP: ignore a command pkt (%d, %d)\n",
		cmd->cmd_set, cmd->command);
	return -1;
    }

    data          = cmd->data;
    suspendPolicy = data[0];
    eventCount    = ntohl(*(int*)(data+1));
    offset        = 5;

    printf("Ender: eventCount = %d\n", eventCount);
    /* link them as a ring */
    for (i=0; i<eventCount; i++) {
	event = (jdwp_event_t*)malloc(sizeof(jdwp_event_t));
	if (event == NULL) {
	    /* FIXME: memory leak */
	    return -1;
	}

	if (*backlog == NULL) {
	    *backlog          = event;
	    event->next       = event;
	    event->prev       = event;
	} else {
	    event->next       = *backlog;
	    event->prev       = (*backlog)->prev;
	    event->next->prev = event;
	    event->prev->next = event;
	}

	event->eventKind = data[offset++];

	switch (event->eventKind) {
	case E_THREAD_START:
	case E_THREAD_END:
	case E_VM_START:
	    memcpy(&event->threadID, data+offset+4, sizeof(long long));
	    offset += 4 + 8; //requestID + threadID
	    break;
	case E_VM_DEATH:
	    offset += 4; // requestID
	    break;
	default:  // this shouldn't happen
	    fprintf(stderr, "Error: JDWP: ignore event %d\n", event->eventKind);
	    /* FIXME: memory leak */
	    free(event);
	    return -1;
	}
    }

    return 0;
}

static void*
dalvik_thread_monitor(void *arg)
{
    adb_ctx_t *ctx;
    jdwp_cmd_t *cmd;
    jdwp_event_t *jdwpEvents = NULL;

    int i, offset;
    char *data;
    char suspendPolicy;
    int  events;
    char eventKind;

    ctx = adb_open(getpid());

    jdwp_handshake(ctx);

    jdwp_set_event_request(ctx, E_THREAD_START, 1);
    jdwp_set_event_request(ctx, E_THREAD_END, 0);

    //jdwp_read_events(ctx);

    while (1) {
	/* is there any pending events in backlog? */
	if (jdwpEvents == NULL) {
	    /* Nope! Let's wait for new events coming */
	    cmd = (jdwp_cmd_t*)jdwp_recv_pkt(ctx);

	    /* something really bad happened, time to end of watch */
	    if (cmd == NULL) {
		fprintf(stderr, "Error: JDWP: disconnect...\n");
		break;
	    }

	    /* put the events into backlog */
	    event_backlog(&jdwpEvents, cmd);	    
	} else {
	    /* Yep! Let's deal with them first */
	    jdwp_reply_t *reply;
	    jdwp_event_t *event = jdwpEvents;

	    if (jdwpEvents->next = jdwpEvents) {
		jdwpEvents = NULL;
	    } else {
		jdwpEvents->next->prev = jdwpEvents->prev;
		jdwpEvents->prev->next = jdwpEvents->next;
		jdwpEvents             = jdwpEvents->next;
	    }

	    /* get thread name */

	    jdwp_send_pkt(ctx, THREADREF_NAME, (char*)&event->threadID,
			  sizeof(event->threadID));
	    while (1) {
		reply = (jdwp_reply_t*)jdwp_recv_pkt(ctx);

		if (reply == NULL) {
		    printf("Ender: reply == NULL\n");
		    break;
		}

		if (reply->flags == 0x80) {
		    /* okay, most likely we get our ack */
		    break;
		}

		/*
		 * dalvik send us some events, put them into backlog for
		 * now, then continue to wait for our ack
		 */
		event_backlog(&jdwpEvents, (jdwp_cmd_t*)reply);
	    }

	    if (reply->error_code != 0) {
		fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
			reply->error_code);
		return NULL;
	    }

	    /* reply->data[]: 4-byte length followed by a non-NULL-terminated string */
	    int len = ntohl(*(int*)(reply->data));
	    char *name = (char*)malloc(len + 1);
	    if (name == NULL) {
		return NULL;
	    }
	    memcpy(name, reply->data+4, len);
	    name[len] = 0;
	    printf("Thread Name : %s\n", name);
	    free(name);
	    free(reply);

	    /* resume thread */

	    jdwp_send_pkt(ctx, THREADREF_RESUME, (char*)&event->threadID,
			  sizeof(event->threadID));
	    while (1) {
		reply = (jdwp_reply_t*)jdwp_recv_pkt(ctx);

		/* okay, most likely we get our ack */
		if (reply->flags == 0x80) {
		    break;
		}

		/*
		 * dalvik send us some events, put them into backlog for
		 * now, then continue to wait for our ack
		 */
		event_backlog(&jdwpEvents, (jdwp_cmd_t*)reply);
	    }

	    if (reply->error_code != 0) {
		fprintf(stderr, "Error: JDWP: get reply with error code %d\n",
			reply->error_code);
		return NULL;
	    }

	    free(reply);

	    /* we are done with this event */
	    free(event);
	}
    }

    return NULL;
}

#endif

/*
 * The VM calls JNI_OnLoad() when the native library is loaded
 */
FILE *ender;
jint JNI_OnLoad(JavaVM *vm, void *reserved)
{
    printf("TAU: JNI_OnLoad\n");

    ender = stderr;

    /*
     * This is a good point to attach your gdb on JVM to debug TAU
     */
    //getchar();

    RtsLayer::TheUsingJNI() = true;
    JNIThreadLayer::tauVM = vm;

#ifdef TAU_ANDROID
    pthread_t thr;
    printf("TAU: start thread monitor\n");
    pthread_create(&thr, NULL, dalvik_thread_monitor, NULL);
#endif

    return JNI_VERSION_1_6;
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

