/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 1997-2000                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : TauJava.cpp                                     **
**      Description     : TAU interface for JVMPI                         **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**      Documentation   : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF
#include <jvmpi.h>
#include <Profile/Profiler.h>
#include <Profile/TauJava.h>
#if (!defined(TAU_WINDOWS))
#include <unistd.h>
#include <stdlib.h>
#endif //TAU_WINDOWS


#define CALL(routine) (JavaThreadLayer::tau_jvmpi_interface->routine)

extern "C" void Tau_profile_exit_all_threads(void);
extern "C" int Tau_stop_timer(void* function_info, int tid);
// Should we exclude all methods using -XrunTAU:nomethods flag? 
bool& TheTauExcludeMethodsFlag()
{
  static bool flag = false; 
  return flag;
}

// profiler agent entry point
extern "C" {
  JNIEXPORT jint JNICALL JVM_OnLoad(JavaVM *jvm, char *options, void *reserved) {
#ifdef DEBUG_PROF
    fprintf(stdout, "TAU> initializing ..... \n");
#endif /* DEBUG_PROF */
    // get jvmpi interface pointer
    if ((jvm->GetEnv((void **)&JavaThreadLayer::tau_jvmpi_interface, JVMPI_VERSION_1)) < 0) {
      fprintf(stdout, "TAU> error in obtaining jvmpi interface pointer\n"
);
      return JNI_ERR;
    }

    // Store the VM identity
    JavaThreadLayer::tauVM = jvm; 

    // initialize jvmpi interface
    JavaThreadLayer::tau_jvmpi_interface->NotifyEvent = 
      TauJavaLayer::NotifyEvent;

    TauJavaLayer::Init(options);

    // enabling class load event notification
    if (!TheTauExcludeMethodsFlag()) {
      CALL(EnableEvent)(JVMPI_EVENT_CLASS_LOAD, NULL);
      CALL(EnableEvent)(JVMPI_EVENT_METHOD_ENTRY, NULL);
      CALL(EnableEvent)(JVMPI_EVENT_METHOD_EXIT, NULL);
    }
    CALL(EnableEvent)(JVMPI_EVENT_THREAD_START, NULL);
    CALL(EnableEvent)(JVMPI_EVENT_THREAD_END, NULL);
    CALL(EnableEvent)(JVMPI_EVENT_JVM_SHUT_DOWN, NULL);
    CALL(EnableEvent)(JVMPI_EVENT_DATA_DUMP_REQUEST, NULL);
    CALL(EnableEvent)(JVMPI_EVENT_DATA_RESET_REQUEST, NULL);
    CALL(EnableEvent)(JVMPI_EVENT_GC_START, NULL);
    CALL(EnableEvent)(JVMPI_EVENT_GC_FINISH, NULL);

// Give TAU some room for its data structures. 
#if (!defined(TAU_WINDOWS))

    if (sizeof(void*) < 8) {
      if ((sbrk(1024*1024*4)) == (void *) -1) {
	fprintf(stdout, "TAU>ERROR: sbrk failed\n");
	CALL(ProfilerExit)(1);
      }
    }
#endif //TAU_WINDOWS

    TauJavaLayer::Init("exclude=TAU/Profile,TAU.Profile");

#ifdef DEBUG_PROF 
    fprintf(stdout, "TAU> .... ok \n\n");
#endif /* DEBUG_PROF */
    return JNI_OK;
  }
}


static char **TauExcludeList=NULL;
static int TauExcludeListSize = 0;
void TauJavaLayer::Init(char *options) {
  char *token;
  static int num_tokens = 0; /* This can be called more than once! TauJAPI */
  int token_len;
  char *s1;
  char *s2;
  
#ifdef DEBUG_PROF
  printf("Inside TauJavaLayer::Init options = %s\n",options);
#endif // DEBUG_PROF
  
  if(options && strlen(options)) {
#ifdef DEBUG_PROF
    fprintf(stdout,"Init: options = %s, len=%d\n", options, strlen(options));
#endif // DEBUG_PROF

    // There are problems with strtok. Since this is called twice,
    // we allocate new strings before using with strtok. Crashes otherwise 	
    if (num_tokens == 0) {
      s1 = new char [strlen (options) + 1];
      strcpy(s1, options);
      token=strtok(s1, "=,");
    } else {
      s2 = new char [strlen (options) + 1];
      strcpy(s2, options);
      token=strtok(s2, "=,");
    }

    if (strcmp(token,"nomethods")==0) {
      TheTauExcludeMethodsFlag() = true;
    } 

    if (strcmp(token,"exclude")==0) {
      if (TauExcludeList == (char **) NULL) {
        TauExcludeList = (char **) malloc(256*sizeof(char*)); // 256 items max in exclude list
	// This ensures that if it is coming here again then the list is not blank
      }
      while(token=strtok(NULL,"=,")) {
	token_len = strlen(token);
	TauExcludeList[num_tokens]=(char *)malloc(token_len+1);
	strcpy(TauExcludeList[num_tokens], token);
        num_tokens++;
      }
      TauExcludeListSize = num_tokens;
    }
    
#ifdef DEBUG_PROF
    for(int i = 0; i < TauExcludeListSize; i++) {
      printf("ExcludeList[%d] = %s\n", i, TauExcludeList[i]);
    }
#endif // DEBUG_PROF
  }
}

// function for handling event notification
void TauJavaLayer::NotifyEvent(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
#ifndef TAU_MPI
  static int j = 0;
  if (j == 0) {
    j=1;
    TAU_PROFILE_SET_NODE(0);
  }
#else  /* TAU_MPI */
  //static int j = TAU_MAPPING_PROFILE_SET_NODE(getpid(), tid);
#endif /* TAU_MPI */

  switch(event->event_type) {
  case JVMPI_EVENT_CLASS_LOAD:
    if (!TheTauExcludeMethodsFlag())
      TauJavaLayer::ClassLoad(event);
    break;
  case JVMPI_EVENT_METHOD_ENTRY:
    if (!TheTauExcludeMethodsFlag())
      TauJavaLayer::MethodEntry(event);
    break;
  case JVMPI_EVENT_METHOD_EXIT:
    if (!TheTauExcludeMethodsFlag())
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
  case JVMPI_EVENT_DATA_DUMP_REQUEST:
    TauJavaLayer::DataDump(event);
    break;
  case JVMPI_EVENT_DATA_RESET_REQUEST:
    TauJavaLayer::DataPurge(event);
    break;
  case JVMPI_EVENT_GC_START:
#ifdef DEBUG_PROF
    printf("TAU>JVMPI_EVENT_GC_START\n");
#endif 
    break;
  case JVMPI_EVENT_GC_FINISH:
#ifdef DEBUG_PROF
    printf("TAU>JVMPI_EVENT_GC_FINISH\n");
#endif
    break;
  /* Use Monitor contended enter, entered and exit events as well */
  default:
    fprintf(stdout, "TAU> Event not registered\n");
    break;
  }
}

void TauJavaLayer::ClassLoad(JVMPI_Event *event) {
  char funcname[2048], classname[1024];
  char *groupname;
  int i;
#ifdef DEBUG_PROF
    fprintf(stdout, "TAU> Class Load : %s\n", event->u.class_load.class_name);
#endif /* DEBUG_PROF */
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
/* Do this for single threaded appls that don't have PROFILE_SET_NODE */

int origkey = 1;
  
  if (TauExcludeListSize) {
    for (i=0; i < TauExcludeListSize; i++) {
      if (strncmp(event->u.class_load.class_name, 
		  TauExcludeList[i], strlen(TauExcludeList[i])) == 0) {
#ifdef DEBUG_PROF
        printf("Excluding %s\n", event->u.class_load.class_name);
#endif // DEBUG_PROF
	origkey = 0;
      }
    }
  }
  for (i = 0; i < event->u.class_load.num_methods; i++) {
    /* Create FunctionInfo objects for each of these methods */

    if (TauEnv_get_tracing()) {
      sprintf(funcname, "%s  %s", event->u.class_load.class_name, 
	      event->u.class_load.methods[i].method_name); 
    } else {
      sprintf(funcname, "%s  %s %s", event->u.class_load.class_name, 
	      event->u.class_load.methods[i].method_name, 
	      event->u.class_load.methods[i].method_signature); 
    }
    
    sprintf(classname, "%s", event->u.class_load.class_name); 
    groupname = strtok(classname, " /=");

/* old
    TAU_MAPPING_CREATE(funcname, " ",
	  (long)  event->u.class_load.methods[i].method_id, 
		     groupname, tid); 
*/
    if (origkey == 1) {
      TAU_MAPPING_CREATE(funcname, " ",
			 (long)  event->u.class_load.methods[i].method_id, 
			 groupname, tid); 
    } else {
      /* NO NEED TO CREATE THIS OBJECT! */ 
      /*
	TAU_MAPPING_CREATE1(funcname, " ",
	(long)  event->u.class_load.methods[i].method_id, origkey,
	groupname, tid); 
      */
    }
    
    
#ifdef DEBUG_PROF 
    printf("TAU> %s, id: %ld group:  %s\n", funcname,
	   event->u.class_load.methods[i].method_id, groupname);
#endif /* DEBUG_PROF */
    /* name, type, key, group name  are the four arguments above */
  }
}

void TauJavaLayer::MethodEntry(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
  TAU_MAPPING_OBJECT(TauMethodName=NULL);
  TAU_MAPPING_LINK(TauMethodName, (long) event->u.method.method_id);
  
  // If the method is not masked, (has non-zero group id), execute it
  if (TauMethodName) { 
    TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
    TAU_MAPPING_PROFILE_START(TauTimer, tid);

#ifdef DEBUG_PROF 
  /*fprintf(stdout, "TAU> Method Entry %s %s:%ld TID = %d\n", 
  		TauMethodName->ThisFunction()->GetName(), TauMethodName->ThisFunction()->GetType(), 
	 	(long) event->u.method.method_id, tid);
*/
#endif /* DEBUG_PROF */
  }
}

void TauJavaLayer::MethodExit(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);

  TAU_MAPPING_OBJECT(TauMethodName=NULL);
  TAU_MAPPING_LINK(TauMethodName, (long) event->u.method.method_id);
  
  if (TauMethodName) { // stop only if it has a finite group 
    TAU_MAPPING_PROFILE_STOP_TIMER(TauMethodName,tid);
#ifdef DEBUG_PROF
    fprintf(stdout, "TAU> Method Exit : %ld, TID = %d\n",
	    (long) event->u.method.method_id, tid);
#endif /* DEBUG_PROF */
  }
}

void TauJavaLayer::CreateTopLevelRoutine(char *name, char *type, char *groupname, 
			int tid) {
#ifdef DEBUG_PROF
  fprintf(stdout, "Inside CreateTopLevelRoutine: name = %s, type = %s, group = %s, tid = %d\n",
	  name, type, groupname, tid); 
#endif
  /* Create a top-level routine that is always called. Use the thread name in it */
  TAU_MAPPING_CREATE(name, type, 1, groupname, tid); 
  
  TAU_MAPPING_OBJECT(TauMethodName);
  TAU_MAPPING_LINK(TauMethodName, (long) 1);
  
  TAU_MAPPING_PROFILE_TIMER(TauTimer, TauMethodName, tid);
  TAU_MAPPING_PROFILE_START(TauTimer, tid);
}

void TauJavaLayer::ThreadStart(JVMPI_Event *event) {
  int * ptid = JavaThreadLayer::RegisterThread(event->env_id);
  int tid = *ptid;

#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> Thread Start : id = %d, name = %s, group = %s\n", 
	tid, event->u.thread_start.thread_name, 
	event->u.thread_start.group_name);
#endif /* DEBUG_PROF */
  char thread_name[256];
  sprintf(thread_name, "THREAD=%s; THREAD GROUP=%s", event->u.thread_start.thread_name,
	  event->u.thread_start.group_name); 
  CreateTopLevelRoutine(thread_name, " ", "THREAD", tid); 
}

void TauJavaLayer::ThreadEnd(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> Thread End : id = %d \n", tid);
#endif /* DEBUG_PROF */
  // TAU_MAPPING_PROFILE_STOP(tid);
    TAU_MAPPING_PROFILE_EXIT("END...", tid);
}

void TauJavaLayer::ShutDown(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
  JVMPI_RawMonitor shutdown_lock = CALL(RawMonitorCreate)("Shutdown lock");
#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> JVM SHUT DOWN : id = %d \n", tid);
#endif 

  //This call is bogus and should be removed..
  //TAU_MAPPING_PROFILE_EXIT only exits the current thread regardless
  //of the value of i. See TauCAPI.cpp:460.
  CALL(RawMonitorEnter)(shutdown_lock);
  Tau_profile_exit_all_threads();
/*
  for(int i = 0; i < JavaThreadLayer::TotalThreads(); i++) {
    TAU_MAPPING_PROFILE_EXIT("Forcing Shutdown of Performance Data",i);
  }
*/
  CALL(RawMonitorExit)(shutdown_lock);
}


void TauJavaLayer::DataDump(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
  JVMPI_RawMonitor dump_lock = CALL(RawMonitorCreate)("Dump lock");
  
#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> JVM DUMP : id = %d \n", tid);
#endif 

  CALL(RawMonitorEnter)(dump_lock);
  for(int i = 0; i < JavaThreadLayer::TotalThreads(); i++) {
    TAU_MAPPING_DB_DUMP(i);
  }
  CALL(RawMonitorExit)(dump_lock);
}

void TauJavaLayer::DataPurge(JVMPI_Event *event) {
  int tid = JavaThreadLayer::GetThreadId(event->env_id);
  JVMPI_RawMonitor purge_lock = CALL(RawMonitorCreate)("Purge lock");

#ifdef DEBUG_PROF
  fprintf(stdout, "TAU> JVM PURGE : id = %d \n", tid);
#endif 

  CALL(RawMonitorEnter)(purge_lock);
  for(int i = 0; i < JavaThreadLayer::TotalThreads(); i++) {
    /* Disabled to retain profile DB integrity on Ctrl-\ (DUMP/PURGE)
       TAU_MAPPING_DB_PURGE(i);
    */
  }
  CALL(RawMonitorExit)(purge_lock);
}

/* EOF : TauJava.cpp */


/***************************************************************************
 * $RCSfile: TauJava.cpp,v $   $Author: khuck $
 * $Revision: 1.33 $   $Date: 2009/03/13 00:46:56 $
 * TAU_VERSION_ID: $Id: TauJava.cpp,v 1.33 2009/03/13 00:46:56 khuck Exp $
 ***************************************************************************/

