/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.acl.lanl.gov/tau                        **
*****************************************************************************
**    Copyright 1997                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : TauJava.cpp                                     **
**      Description     : TAU interface for JVMPI                         **
**      Author          : Sameer Shende                                   **
**      Contact         : sameer@cs.uoregon.edu sameer@acl.lanl.gov       **
**      Flags           : Compile with                                    **
**                        -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**                        -DPROFILE_STATS for Std. Deviation of Excl Time **
**                        -DSGI_HW_COUNTERS for using SGI counters        **
**                        -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**                        -DTULIP_TIMERS for non-sgi Platform             **
**                        -DPOOMA_STDSTL for using STD STL in POOMA src   **
**                        -DPOOMA_TFLOP for Intel Teraflop at SNL/NM      **
**                        -DPOOMA_KAI for KCC compiler                    **
**                        -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**      Documentation   : See http://www.acl.lanl.gov/tau                 **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#include <jvmpi.h>
#include <Profile/Profiler.h>
#include <Profile/TauJava.h>
#include <unistd.h>


static JVMPI_Interface *tau_jvmpi_interface;
#define CALL(routine) (tau_jvmpi_interface->routine)

// profiler agent entry point
extern "C" {
  JNIEXPORT jint JNICALL JVM_OnLoad(JavaVM *jvm, char *options, void *reserved)
{
#ifdef DEBUG_PROF
    fprintf(stderr, "TAU> initializing ..... \n");
#endif /* DEBUG_PROF */
    // get jvmpi interface pointer
    if ((jvm->GetEnv((void **)&tau_jvmpi_interface, JVMPI_VERSION_1)) < 0) {
      fprintf(stderr, "TAU> error in obtaining jvmpi interface pointer\n"
);
      return JNI_ERR;
    }

    // initialize jvmpi interface
    tau_jvmpi_interface->NotifyEvent = TauJavaLayer::NotifyEvent;
    // enabling class load event notification
    tau_jvmpi_interface->EnableEvent(JVMPI_EVENT_CLASS_LOAD, NULL);
    tau_jvmpi_interface->EnableEvent(JVMPI_EVENT_METHOD_ENTRY, NULL);
    tau_jvmpi_interface->EnableEvent(JVMPI_EVENT_METHOD_EXIT, NULL);
    tau_jvmpi_interface->EnableEvent(JVMPI_EVENT_THREAD_START, NULL);
    tau_jvmpi_interface->EnableEvent(JVMPI_EVENT_THREAD_END, NULL);

    if ((sbrk(1024*1024*4)) == (void *) -1) {
      fprintf(stderr, "TAU>ERROR: sbrk failed\n");
      exit(1);
    }
#ifdef DEBUG_PROF 
    fprintf(stderr, "TAU> .... ok \n\n");
#endif /* DEBUG_PROF */
    return JNI_OK;
  }
}


// function for handling event notification
void TauJavaLayer::NotifyEvent(JVMPI_Event *event) {
  switch(event->event_type) {
  case JVMPI_EVENT_CLASS_LOAD:
    TauJavaLayer::ClassLoad(event);
    break;
  case JVMPI_EVENT_METHOD_ENTRY:
    TauJavaLayer::MethodEntry(event);
    break;
  case JVMPI_EVENT_METHOD_EXIT:
    TauJavaLayer::MethodExit(event);
    break;
  case JVMPI_EVENT_THREAD_START:
    TauJavaLayer::ThreadStart(event);
    break;
  case JVMPI_EVENT_THREAD_END:
    TauJavaLayer::ThreadEnd(event);
    break;
  /* Use Monitor contended enter, entered and exit events as well */
  default:
    fprintf(stderr, "TAU> Event not registered\n");
    break;
  }
}

void TauJavaLayer::ClassLoad(JVMPI_Event *event)
{
  int i;
#ifdef DEBUG_PROF
    fprintf(stderr, "TAU> Class Load : %s\n", event->u.class_load.class_name);
#endif /* DEBUG_PROF */
  static int j = TAU_PROFILE_SET_NODE(0);

  for (i = 0; i < event->u.class_load.num_methods; i++)
  {
    /* Create FunctionInfo objects for each of these methods */

    TAU_MAPPING_CREATE(event->u.class_load.methods[i].method_name, 
		       event->u.class_load.methods[i].method_signature,
	  (long)  event->u.class_load.methods[i].method_id, 
		     event->u.class_load.class_name); 
#ifdef DEBUG_PROF 
    printf("TAU> method: %s, sig: %s, id: %ld, class: %s\n",
		event->u.class_load.methods[i].method_name,
        	event->u.class_load.methods[i].method_signature,
           		event->u.class_load.methods[i].method_id,
                       event->u.class_load.class_name);
#endif /* DEBUG_PROF */
    /* name, type, key, group name  are the four arguments above */
  }
	
}

void TauJavaLayer::MethodEntry(JVMPI_Event *event)
{
  TAU_MAPPING_OBJECT(TauMethodName);
  TAU_MAPPING_LINK(TauMethodName, (long) event->u.method.method_id);
  
  TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName);
  TAU_MAPPING_PROFILE_START(TauTimer);

#ifdef DEBUG_PROF 
  fprintf(stderr, "TAU> Method Entry %s %s:%ld \n", 
  		TauMethodName->GetName(), TauMethodName->GetType(), 
	 	(long) event->u.method.method_id);
#endif /* DEBUG_PROF */
}

void TauJavaLayer::MethodExit(JVMPI_Event *event)
{

  TAU_MAPPING_PROFILE_STOP();

#ifdef DEBUG_PROF
  fprintf(stderr, "TAU> Method Exit : %ld\n",
	 	(long) event->u.method.method_id);
#endif /* DEBUG_PROF */
}

void TauJavaLayer::ThreadStart(JVMPI_Event *event)
{
#ifdef DEBUG_PROF
    fprintf(stderr, "TAU> Thread Start : \n");
#endif /* DEBUG_PROF */

}

void TauJavaLayer::ThreadEnd(JVMPI_Event *event)
{
#ifdef DEBUG_PROF
    fprintf(stderr, "TAU> Thread End : \n");
#endif /* DEBUG_PROF */
//    TAU_PROFILE_EXIT("END...");
}

/* EOF : TauJava.cpp */
