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
int TauJavaLayer::NumThreads = 0;
#define CALL(routine) (tau_jvmpi_interface->routine)

// profiler agent entry point
extern "C" {
  JNIEXPORT jint JNICALL JVM_OnLoad(JavaVM *jvm, char *options, void *reserved)
{
#ifdef DEBUG_PROF
    fprintf(stdout, "TAU> initializing ..... \n");
#endif /* DEBUG_PROF */
    // get jvmpi interface pointer
    if ((jvm->GetEnv((void **)&tau_jvmpi_interface, JVMPI_VERSION_1)) < 0) {
      fprintf(stdout, "TAU> error in obtaining jvmpi interface pointer\n"
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
    tau_jvmpi_interface->EnableEvent(JVMPI_EVENT_JVM_SHUT_DOWN, NULL);


    if ((sbrk(1024*1024*4)) == (void *) -1) {
      fprintf(stdout, "TAU>ERROR: sbrk failed\n");
      exit(1);
    }


#ifdef DEBUG_PROF 
    fprintf(stdout, "TAU> .... ok \n\n");
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
  case JVMPI_EVENT_JVM_SHUT_DOWN:
    TauJavaLayer::ShutDown(event);
    break;
  /* Use Monitor contended enter, entered and exit events as well */
  default:
    fprintf(stdout, "TAU> Event not registered\n");
    break;
  }
}

void TauJavaLayer::ClassLoad(JVMPI_Event *event)
{
  char funcname[2048], groupname[1024];
  int i;
#ifdef DEBUG_PROF
  int k;
    fprintf(stdout, "TAU> Class Load : %s\n", event->u.class_load.class_name);
#endif /* DEBUG_PROF */
  static int j = TAU_PROFILE_SET_NODE(0);
  

#ifdef DEBUG_PROF
  printf("Loading Class: %s\n", event->u.class_load.class_name);
/*
  for(k = 0; k < event->u.class_load.num_instance_fields; k++)
  {
    printf("Class : %s Instances : field %s sig %s\n", 
	event->u.class_load.class_name,
	event->u.class_load.instances[k].field_name, 
	event->u.class_load.instances[k].field_signature);
  }
*/
#endif /* DEBUG_PROF */

  for (i = 0; i < event->u.class_load.num_methods; i++)
  {
    /* Create FunctionInfo objects for each of these methods */

    sprintf(funcname, "%s  %s", event->u.class_load.class_name, 
	event->u.class_load.methods[i].method_name); 
    sprintf(groupname, "%s", event->u.class_load.class_name); 

    TAU_MAPPING_CREATE(funcname, " ",
	  (long)  event->u.class_load.methods[i].method_id, 
		     groupname); 
#ifdef DEBUG_PROF 
    printf("TAU> %s, id: %ld group:  %s\n", funcname,
		event->u.class_load.methods[i].method_id, groupname);
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
  fprintf(stdout, "TAU> Method Entry %s %s:%ld ", 
  		TauMethodName->GetName(), TauMethodName->GetType(), 
	 	(long) event->u.method.method_id);
#endif /* DEBUG_PROF */
  int *tid = (int *) CALL(GetThreadLocalStorage)(event->env_id);
  if (tid == (int *) NULL) 
       tid = RegisterThread(event);
#ifdef DEBUG_PROF 
  printf(" TID = %d\n", (*tid));
#endif /* DEBUG_PROF */
}

void TauJavaLayer::MethodExit(JVMPI_Event *event)
{
  TAU_MAPPING_PROFILE_STOP();

  int *tid = (int *) CALL(GetThreadLocalStorage)(event->env_id);
#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> Method Exit : %ld, TID = %d\n",
	 	(long) event->u.method.method_id, (*tid));
#endif /* DEBUG_PROF */
}

int * TauJavaLayer::RegisterThread(JVMPI_Event *event)
{ /* returns tid pointer */
  JVMPI_RawMonitor numthreads_lock = CALL(RawMonitorCreate)("num threads lock");
  /* Set up the thread id */
  int *tid = (int *) malloc (sizeof (int));
  CALL(RawMonitorEnter)(numthreads_lock);
  (*tid) = TauJavaLayer::NumThreads++;
  CALL(RawMonitorExit)(numthreads_lock);
  CALL(SetThreadLocalStorage)(event->env_id, tid); 
  return tid;

}
void TauJavaLayer::ThreadStart(JVMPI_Event *event)
{
  int *tid = RegisterThread(event);

#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> Thread Start : id = %d, name = %s, group = %s\n", 
	(*tid), event->u.thread_start.thread_name, 
	event->u.thread_start.group_name);
#endif /* DEBUG_PROF */

}

void TauJavaLayer::ThreadEnd(JVMPI_Event *event)
{
  int *tid = (int *) CALL(GetThreadLocalStorage)(event->env_id);
#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> Thread End : id = %d \n", (*tid));
#endif /* DEBUG_PROF */
    TAU_PROFILE_EXIT("END...");
}

void TauJavaLayer::ShutDown(JVMPI_Event *event)
{
  int *tid = (int *) CALL(GetThreadLocalStorage)(event->env_id);
#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> JVM SHUT DOWN : id = %d \n", (*tid));
#endif 
  
}
/* EOF : TauJava.cpp */


