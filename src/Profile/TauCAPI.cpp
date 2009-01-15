/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauCAPI.C					  **
**	Description 	: TAU Profiling Package API wrapper for C	  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifdef TAU_DOT_H_LESS_HEADERS 
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include "Profile/Profiler.h"
#include <stdio.h>
#include <stdlib.h>

#if (!defined(TAU_WINDOWS)) 
/* Needed for fork */
#include <sys/types.h>
#include <unistd.h>
#endif /* TAU_WINDOWS */


///////////////////////////////////////////////////////////////////////////
// Wrappers for corresponding C++ functions follow

/* Note: Our old scheme relied on getting a profiler object. This doesn't 
   work well with threads, all threads start/stop the same profiler & 
   profiler is supposed to be for each invocation (thread) as it has a single
   scalar for storing StartTime. So, we changed this so that each 
   Tau_get_profiler returns a functionInfo object which can then have 
   independent profilers associated with it. Each start/stop timer call creates 
   and destroys a profiler object now. Since the Fortran layer is built atop 
   the C layer, it remains unchanged. However, we should probably change the 
   name of this method to Tau_get_functioninfo or something. */
///////////////////////////////////////////////////////////////////////////
extern "C" void * Tau_get_profiler(const char *fname, const char *type, TauGroup_t group, const char *gr_name)
{
  FunctionInfo *f;
  //Profiler *p;

  DEBUGPROFMSG("Inside get_profiler group = " << group<<endl;);

  // since we're using new, we should set InitData to true in FunctionInfoInit
  if (group == TAU_MESSAGE)
  {
    if (gr_name && strcmp(gr_name, "TAU_MESSAGE") == 0)
      f = new FunctionInfo(fname, type, group, "MPI", true);
    else 
      f = new FunctionInfo(fname, type, group, gr_name, true);
  }
  else 
    f = new FunctionInfo(fname, type, group, gr_name, true);
//  p = new Profiler(f, group, true);

  return (void *) f;
}



#define STACK_DEPTH_INCREMENT 100
static Profiler *Tau_global_stack[TAU_MAX_THREADS];
static int Tau_global_stackdepth[TAU_MAX_THREADS];
static int Tau_global_stackpos[TAU_MAX_THREADS];


extern "C" void Tau_stack_initialization() {
  int i;
  for (i=0; i<TAU_MAX_THREADS; i++) {
    Tau_global_stackdepth[i] = 0;
    Tau_global_stackpos[i] = -1;
    Tau_global_stack[i] = NULL;
  }
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_start_timer(void *functionInfo, int phase) {
  int tid = RtsLayer::myThread();
  FunctionInfo *fi = (FunctionInfo *) functionInfo; 

  if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
    return; /* group is disabled */
  }

  // move the stack pointer
  Tau_global_stackpos[tid]++; /* push */


  if (Tau_global_stackpos[tid] >= Tau_global_stackdepth[tid]) {
    int oldDepth = Tau_global_stackdepth[tid];
    int newDepth = oldDepth + STACK_DEPTH_INCREMENT;
    Profiler *newStack = (Profiler *) malloc(sizeof(Profiler)*newDepth);
    memcpy(newStack, Tau_global_stack[tid], oldDepth*sizeof(Profiler));
    Tau_global_stack[tid] = newStack;
    Tau_global_stackdepth[tid] = newDepth;
  }

  Profiler *p = &(Tau_global_stack[tid][Tau_global_stackpos[tid]]);

  p->MyProfileGroup_ = fi->GetProfileGroup();
  p->ThisFunction = fi; 

  
#ifdef TAU_MPITRACE
  p->RecordEvent = false; /* by default, we don't record this event */
#endif /* TAU_MPITRACE */

  
#ifdef TAU_PROFILEPHASE
  if (phase) {
    p->SetPhase(true);
  } else {
    p->SetPhase(false);
  }
#endif /* TAU_PROFILEPHASE */

  p->Start();
}


///////////////////////////////////////////////////////////////////////////
static void reportOverlap (FunctionInfo *stack, FunctionInfo *caller) {
  printf("[%d:%d-%d] TAU: Runtime overlap: found %s (%p) on the stack, but stop called on %s (%p)\n", 
	 RtsLayer::getPid(), RtsLayer::getTid(), RtsLayer::myThread(),
	 stack->GetName(), stack, caller->GetName(), caller);
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_stop_timer(void *function_info) {
  FunctionInfo *functionInfo = (FunctionInfo *) function_info; 
  int tid = RtsLayer::myThread();
  Profiler *profiler;

  if (Tau_global_stackpos[tid] < 0) return 0;

  if (!(functionInfo->GetProfileGroup() & RtsLayer::TheProfileMask())) {
    return 0; /* group is disabled */
  }
  
  profiler = &(Tau_global_stack[tid][Tau_global_stackpos[tid]]);
  Tau_global_stackpos[tid]--; /* pop */
  
  if (profiler->ThisFunction != functionInfo) { /* Check for overlapping timers */
    reportOverlap (profiler->ThisFunction, functionInfo);
  }
  profiler->Stop();
  return 0;
}


///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_stop_current_timer() {
  FunctionInfo *functionInfo;
  Profiler *profiler;
  int tid;

  tid = RtsLayer::myThread();

  if (Tau_global_stackpos[tid] < 0) return 0;

  profiler = &(Tau_global_stack[tid][Tau_global_stackpos[tid]]);
  functionInfo = profiler->ThisFunction;

  if (functionInfo->GetProfileGroup() & RtsLayer::TheProfileMask()) {
    profiler->Stop();
    Tau_global_stackpos[tid]--;
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////



extern "C" int Tau_profile_exit() {
  int tid = RtsLayer::myThread();
  while (Tau_global_stackpos[tid] >= 0) {
    Profiler *p = &(Tau_global_stack[tid][Tau_global_stackpos[tid]]);
    p->Stop();
    Tau_global_stackpos[tid]--;
  }
  return 0;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_exit(char * msg) {
  tau::Profiler::ProfileExit(msg);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_init_ref(int* argc, char ***argv) {
  RtsLayer::ProfileInit(*argc, *argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_init(int argc, char **argv) {
  RtsLayer::ProfileInit(argc, argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_node(int node) {
  RtsLayer::setMyNode(node);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_context(int context) {
  RtsLayer::setMyContext(context);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_thread(int thread) {
  RtsLayer::setMyThread(thread);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_profile_callstack(void) {
  /* removed */
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump(void) {
  tau::Profiler::DumpData();
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_prefix(char *prefix) {
 tau::Profiler::DumpData(false, RtsLayer::myThread(), prefix);
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_incr(void) {
  tau::Profiler::DumpData(true);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_purge(void) {
  tau::Profiler::PurgeData();
}

extern "C" void Tau_the_function_list(const char ***functionList, int *num) {
  tau::Profiler::theFunctionList(functionList, num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_names() {
  tau::Profiler::dumpFunctionNames();
}
  
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_counter_names(const char ***counterList, int *num) {
  tau::Profiler::theCounterList(counterList, num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_function_values(const char **inFuncs, int numOfFuncs,
					double ***counterExclusiveValues,
					double ***counterInclusiveValues,
					int **numOfCalls, int **numOfSubRoutines,
					const char ***counterNames, int *numOfCounters) {
   tau::Profiler::getFunctionValues(inFuncs,numOfFuncs,counterExclusiveValues,counterInclusiveValues,
				    numOfCalls,numOfSubRoutines,counterNames,numOfCounters);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_event_names(const char ***eventList, int *num) {
  tau::Profiler::getUserEventList(eventList, num);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_event_vals(const char **inUserEvents, int numUserEvents,
				  int **numEvents, double **max, double **min,
				  double **mean, double **sumSqr) {
  tau::Profiler::getUserEventValues(inUserEvents, numUserEvents, numEvents, max, min,
		     mean, sumSqr);
}



///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_values(const char **functionList, int num) {
  tau::Profiler::dumpFunctionValues(functionList,num);;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_values_incr(const char **functionList, int num) {
  tau::Profiler::dumpFunctionValues(functionList,num,true);;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_register_thread(void) {
  RtsLayer::RegisterThread();;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_register_fork(int nodeid, enum TauFork_t opcode) {
  RtsLayer::RegisterFork(nodeid, opcode);;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_instrumentation(void) {
  RtsLayer::TheEnableInstrumentation() = true;;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_instrumentation(void) {
  RtsLayer::TheEnableInstrumentation() = false;;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_shutdown(void) {
  if (!TheUsingCompInst()) {
    RtsLayer::TheShutdown() = true;
    RtsLayer::TheEnableInstrumentation() = false;;
  }
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_enable_group_name(char * group) {
  return RtsLayer::enableProfileGroupName(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_disable_group_name(char * group) {
  return RtsLayer::disableProfileGroupName(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_get_profile_group(char * group) {
  return RtsLayer::getProfileGroup(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_group(TauGroup_t group) {
  RtsLayer::enableProfileGroup(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_group(TauGroup_t group) {
  RtsLayer::disableProfileGroup(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_enable_all_groups(void) {
  return RtsLayer::enableAllGroups();
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_disable_all_groups(void) {
  return RtsLayer::disableAllGroups();
}



///////////////////////////////////////////////////////////////////////////
extern "C" int& tau_totalnodes(int set_or_get, int value)
{
  static int nodes = 1;
  if (set_or_get == 1)
    {
      nodes = value;
    }
  return nodes;
}


#ifdef TAU_MPI
#define TAU_GEN_EVENT(e, msg) TauUserEvent* e () { \
	static TauUserEvent u(msg); return &u; } 

TAU_GEN_EVENT(TheSendEvent,"Message size sent to all nodes")
TAU_GEN_EVENT(TheRecvEvent,"Message size received from all nodes")
TAU_GEN_EVENT(TheBcastEvent,"Message size for broadcast")
TAU_GEN_EVENT(TheReduceEvent,"Message size for reduce")
TAU_GEN_EVENT(TheReduceScatterEvent,"Message size for reduce-scatter")
TAU_GEN_EVENT(TheScanEvent,"Message size for scan")
TAU_GEN_EVENT(TheAllReduceEvent,"Message size for all-reduce")
TAU_GEN_EVENT(TheAlltoallEvent,"Message size for all-to-all")
TAU_GEN_EVENT(TheScatterEvent,"Message size for scatter")
TAU_GEN_EVENT(TheGatherEvent,"Message size for gather")
TAU_GEN_EVENT(TheAllgatherEvent,"Message size for all-gather")

TauUserEvent**& TheMsgVolEvent()
{
  static TauUserEvent **u = 0; 
  return u;
}

int register_events(void)
{
#ifdef TAU_EACH_SEND
  char str[256];
  int i;

  TheMsgVolEvent() = (TauUserEvent **) malloc(sizeof(TauUserEvent *)*tau_totalnodes(0,0));
  for (i =0; i < tau_totalnodes(0,0); i++)
  {
    sprintf(str, "Message size sent to node %d", i);
    TheMsgVolEvent()[i] = (TauUserEvent *) new TauUserEvent((const char *)str);
  }
#endif /* TAU_EACH_SEND */
  return 0;
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length)
{
  static int initialize = register_events();
#ifdef TAU_PROFILEPARAM
#ifndef TAU_DISABLE_PROFILEPARAM_IN_MPI
  static string s("message size");
  TAU_PROFILE_PARAM1L(length, s);
#endif /* TAU_DISABLE_PROFILEPARAM_IN_MPI */
#endif  /* TAU_PROFILEPARAM */
#ifdef DEBUG_PROF
  printf("Node %d: Tau_trace_sendmsg: type %d dest %d len %d\n", 
        RtsLayer::myNode(), type, destination, length);
#endif /* DEBUG_PROF */
  TAU_EVENT(TheSendEvent(), length);
#ifdef TAU_EACH_SEND
  TheMsgVolEvent()[destination]->TriggerEvent(length, RtsLayer::myThread());
#endif /* TAU_EACH_SEND */
  if (destination >= 0)
    TAU_TRACE_SENDMSG(type, destination, length);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg(int type, int source, int length)
{
  TAU_EVENT(TheRecvEvent(), length);
#ifdef TAU_PROFILEPARAM
#ifndef TAU_DISABLE_PROFILEPARAM_IN_MPI
  static string s("message size");
  TAU_PROFILE_PARAM1L(length, s);
#endif /* TAU_DISABLE_PROFILEPARAM_IN_MPI */
#endif  /* TAU_PROFILEPARAM */
  if (source >= 0) 
    TAU_TRACE_RECVMSG(type, source, length);
}

extern "C" void Tau_bcast_data(int data)
{
  TAU_EVENT(TheBcastEvent(), data);
}

extern "C" void Tau_reduce_data(int data)
{
  TAU_EVENT(TheReduceEvent(), data);
}

extern "C" void Tau_alltoall_data(int data)
{
  TAU_EVENT(TheAlltoallEvent(), data);
}

extern "C" void Tau_scatter_data(int data)
{
  TAU_EVENT(TheScatterEvent(), data);
}

extern "C" void Tau_gather_data(int data)
{
  TAU_EVENT(TheGatherEvent(), data);
}

extern "C" void Tau_allgather_data(int data)
{
  TAU_EVENT(TheAllgatherEvent(), data);
}

extern "C" void Tau_allreduce_data(int data)
{
  TAU_EVENT(TheGatherEvent(), data);
}

extern "C" void Tau_scan_data(int data)
{
  TAU_EVENT(TheScanEvent(), data);
}

extern "C" void Tau_reducescatter_data(int data)
{
  TAU_EVENT(TheReduceScatterEvent(), data);
}
#else /* !TAU_MPI */
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length)
{
  TAU_TRACE_SENDMSG(type, destination, length);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg(int type, int source, int length)
{
  TAU_TRACE_RECVMSG(type, source, length);
}
#endif /* TAU_MPI */

///////////////////////////////////////////////////////////////////////////
// User Defined Events 
///////////////////////////////////////////////////////////////////////////
extern "C" void * Tau_get_userevent(char *name) {
  TauUserEvent *ue;
  ue = new TauUserEvent(name);
  return (void *) ue;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_userevent(void *ue, double data) {
  TauUserEvent *t = (TauUserEvent *) ue;
  t->TriggerEvent(data);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_context_userevent(void **ptr, char *name)
{
  
  if (*ptr == 0) {
    RtsLayer::LockEnv();

    if (*ptr == 0) {
      TauContextUserEvent *ue;
      ue = new TauContextUserEvent(name);
      *ptr = (void*) ue;
    }

    RtsLayer::UnLockEnv();
  }
  return;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_context_userevent(void *ue, double data)
{
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->TriggerEvent(data);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_event_name(void *ue, char *name)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetEventName(name);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_report_statistics(void) {
  TauUserEvent::ReportStatistics();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_report_thread_statistics(void) {
  TauUserEvent::ReportStatistics(true);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_min(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableMin(true);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_max(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableMax(true);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_mean(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableMean(true);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_stddev(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableStdDev(true);
} 

///////////////////////////////////////////////////////////////////////////

extern "C" void Tau_profile_c_timer(void **ptr, char *fname, char *type, TauGroup_t group, 
	char *group_name)
{
#ifdef DEBUG_PROF
  printf("Inside Tau_profile_timer_ fname=%s *ptr = %x\n", fname, *ptr);
#endif /* DEBUG_PROF */


  if (*ptr == 0) {
    
    RtsLayer::LockEnv();
    
    if (*ptr == 0) {  
      // remove garbage characters from the end of name
      for(int i=0; i<strlen(fname); i++) {
	if (!isprint(fname[i])) {
	  fname[i] = '\0';
	  break;
	}
      }
      *ptr = Tau_get_profiler(fname, type, group, group_name);
    }
    RtsLayer::UnLockEnv();
  }
  
#ifdef DEBUG_PROF
  printf("get_profiler returns %x\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}

///////////////////////////////////////////////////////////////////////////


/* We need a routine that will create a top level parent profiler and give
 * it a dummy name for the application, if just the MPI wrapper interposition
 * library is used without any instrumentation in main */
extern "C" void Tau_create_top_level_timer_if_necessary(void)
{
  int disabled = 0;
#ifdef TAU_VAMPIRTRACE
  disabled = 1;
#endif
#ifdef TAU_EPILOG
  disabled = 1;
#endif
  if (disabled) {
    return;
  }

  static bool initialized = false;
  static bool initthread[TAU_MAX_THREADS];
  if (!initialized) {
    RtsLayer::LockDB();
    if (!initialized) {
      for (int i=0; i<TAU_MAX_THREADS; i++) {
	initthread[i] = false;
      }
    }
    RtsLayer::UnLockDB();
    initialized = true;
  }
  int tid = RtsLayer::myThread();
  if (initthread[tid] == true) {
    return;
  }
  FunctionInfo *ptr;
  if (Profiler::CurrentProfiler[tid] == NULL) {
    initthread[tid] = true;
    ptr = (FunctionInfo *) Tau_get_profiler(".TAU application", " ", TAU_DEFAULT, "TAU_DEFAULT");
    if (ptr) {
      Tau_start_timer(ptr, 0);
    }
  }
}


extern "C" void Tau_stop_top_level_timer_if_necessary(void)
{
  int tid = RtsLayer::myThread();
  if (Profiler::CurrentProfiler[tid] && 
      Profiler::CurrentProfiler[tid]->ParentProfiler == NULL && 
      strcmp(Profiler::CurrentProfiler[tid]->ThisFunction->GetName(), ".TAU application") == 0)
  {
    DEBUGPROFMSG("Found top level .TAU application timer"<<endl;);  
    TAU_GLOBAL_TIMER_STOP();
  }
}


extern "C" void Tau_disable_context_event(void *event) {
  TauContextUserEvent *e = (TauContextUserEvent *) event;
  e->SetDisableContext(true);
}

extern "C" void Tau_enable_context_event(void *event) {
  TauContextUserEvent *e = (TauContextUserEvent *) event;
  e->SetDisableContext(false);
}



extern "C" void Tau_track_memory(void) {
  TauTrackMemoryUtilization(true);
}


extern "C" void Tau_track_memory_here(void) {
  TauTrackMemoryHere();
}


extern "C" void Tau_track_memory_headroom(void) {
  TauTrackMemoryUtilization(false);
}


extern "C" void Tau_track_memory_headroom_here(void) {
  TauTrackMemoryHeadroomHere();
}


extern "C" void Tau_track_muse_events(void) {
  TauTrackMuseEvents();
}


extern "C" void Tau_enable_tracking_memory(void) {
  TauEnableTrackingMemory();
}


extern "C" void Tau_disable_tracking_memory(void) {
  TauDisableTrackingMemory();
}


extern "C" void Tau_enable_tracking_memory_headroom(void) {
  TauEnableTrackingMemoryHeadroom();
}


extern "C" void Tau_disable_tracking_memory_headroom(void) {
  TauDisableTrackingMemoryHeadroom();
}


extern "C" void Tau_enable_tracking_muse_events(void) {
  TauEnableTrackingMuseEvents();
}



extern "C" void Tau_disable_tracking_muse_events(void) {
  TauDisableTrackingMuseEvents();
}


extern "C" void Tau_set_interrupt_interval(int value) {
  TauSetInterruptInterval(value);
}


extern "C" void Tau_global_stop(void) {
  Tau_stop_current_timer();;
}

///////////////////////////////////////////////////////////////////////////
extern "C" char * Tau_phase_enable(const char *group) {
#ifdef TAU_PROFILEPHASE
  char *newgroup = new char[strlen(group)+16];
  sprintf(newgroup, "%s | TAU_PHASE", group);
  return newgroup;
#else /* TAU_PROFILEPHASE */
  return (char *) group;
#endif /* TAU_PROFILEPHASE */
} 

///////////////////////////////////////////////////////////////////////////
extern "C" char * Tau_phase_enable_once(const char *group, void **ptr) {
  /* We don't want to parse the group name string every time this is invoked.
     we compare a pointer and if necessary, perform the string operations  
     on the group name (adding  | TAU_PHASE to it). */
  if (*ptr == 0) {
    return Tau_phase_enable(group);
  } else {
    return (char *) NULL;
  }
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_mark_group_as_phase(void *ptr) {
  FunctionInfo *fptr = (FunctionInfo *) ptr;
  char *newgroup = Tau_phase_enable(fptr->GetAllGroups()); 
  fptr->SetPrimaryGroupName(newgroup); 
}

///////////////////////////////////////////////////////////////////////////
extern "C" char * Tau_append_iteration_to_name(int iteration, char *name) {
  char tau_iteration_number[128];
  sprintf(tau_iteration_number, " [%d]", iteration);
  string iterationName = string(name)+string(tau_iteration_number);
  char *newName = strdup(iterationName.c_str());
  return newName;
}

///////////////////////////////////////////////////////////////////////////

extern "C" void Tau_profile_dynamic_auto(int iteration, void **ptr, char *fname, char *type, TauGroup_t group, char *group_name, int isPhase)
{ /* This routine creates dynamic timers and phases by embedding the
     iteration number in the name. isPhase argument tells whether we
     choose phases or timers. */

  char *newName = Tau_append_iteration_to_name(iteration, fname);

  /* create the pointer. */
  Tau_profile_c_timer(ptr, newName, type, group, group_name);

  /* annotate it as a phase if it is */
  if (isPhase)
    Tau_mark_group_as_phase(ptr);
  free(newName);

}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_profile_param1l(long data, const char *dataname) {
  string dname(dataname);
#ifdef TAU_PROFILEPARAM
  tau::Profiler::AddProfileParamData(data, dataname);
#endif
}


/*
  The following is for supporting pure and elemental fortran subroutines
*/


map<string, FunctionInfo *>& ThePureMap() {
  static map<string, FunctionInfo *> pureMap;
  return pureMap;
}

map<string, int *>& TheIterationMap() {
  static map<string, int *> iterationMap;
  return iterationMap;
}


extern "C" void Tau_pure_start(const char *name) {
  FunctionInfo *fi = 0;
  string n = string(name);
  map<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    tauCreateFI(&fi,n,"",TAU_USER,"TAU_USER");
    ThePureMap()[n] = fi;
  } else {
    fi = (*it).second;
  }
  Tau_start_timer(fi,0);
}

extern "C" void Tau_pure_stop(const char *name) {
  FunctionInfo *fi;
  string n = string(name);
  map<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    fprintf (stderr, "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\nTAU Error: You will likely get an overlapping timer message next\n\n", name);
  } else {
    fi = (*it).second;
    Tau_stop_timer(fi);
  }
}

extern "C" void Tau_static_phase_start(char *name) {
  FunctionInfo *fi = 0;
  string n = string(name);
  map<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    tauCreateFI(&fi,n,"",TAU_USER,"TAU_USER");
    Tau_mark_group_as_phase(fi);
    ThePureMap()[n] = fi;
  } else {
    fi = (*it).second;
  }   Tau_start_timer(fi,1);
}

extern "C" void Tau_static_phase_stop(char *name) {
  FunctionInfo *fi;
  string n = string(name);
  map<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    fprintf (stderr, "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\nTAU Error: You will likely get an overlapping timer message next\n\n", name);
  } else {
    fi = (*it).second;
    Tau_stop_timer(fi);
  }
}


static int *getIterationList(char *name) {
  string searchName(name);
  map<string, int *>::iterator iit = TheIterationMap().find(searchName);
  if (iit == TheIterationMap().end()) {
    RtsLayer::LockEnv();
    int *iterationList = new int[TAU_MAX_THREADS];
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      iterationList[i] = 0;
    }
    TheIterationMap()[searchName] = iterationList;
    RtsLayer::UnLockEnv();
  }
  return TheIterationMap()[searchName];
}

/* isPhase argument is 1 for phase and 0 for timer */
extern "C" void Tau_dynamic_start(char *name, int isPhase) {
#ifndef TAU_PROFILEPHASE
  isPhase = 0;
#endif

  int *iterationList = getIterationList(name);

  int tid = RtsLayer::myThread();
  int itcount = iterationList[tid];

  FunctionInfo *fi = NULL;
  char *newName = Tau_append_iteration_to_name(itcount, name);
  string n (newName);
  free(newName);
#ifdef DEBUG_PROF
  printf("Checking for %s: iteration = %d\n", n.c_str(), iteration);
#endif /* DEBUG_PROF */
  
  RtsLayer::LockDB();
  map<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    tauCreateFI(&fi,n,"",TAU_USER,"TAU_USER");
    if (isPhase) {
      Tau_mark_group_as_phase(fi);
    }
    ThePureMap()[n] = fi;
  } else {
    fi = (*it).second;
  }   
  RtsLayer::UnLockDB();
  Tau_start_timer(fi,isPhase);
}


/* isPhase argument is ignored in Tau_dynamic_stop. For consistency with
   Tau_dynamic_start. */
extern "C" void Tau_dynamic_stop(char *name, int isPhase) {
  
  int *iterationList = getIterationList(name);

  int tid = RtsLayer::myThread();
  int itcount = iterationList[tid];

  // increment the counter
  iterationList[tid]++;
  
  FunctionInfo *fi = NULL;   
  char *newName = Tau_append_iteration_to_name(itcount, name);
  string n (newName);
  free(newName);
  RtsLayer::LockDB();
  map<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    fprintf (stderr, "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\nTAU Error: You will likely get an overlapping timer message next\n\n", name);
    RtsLayer::UnLockDB();
    return;
  } else {
    fi = (*it).second;
  }
  RtsLayer::UnLockDB();
  Tau_stop_timer(fi);
}


#if (!defined(TAU_WINDOWS))
extern "C" pid_t tau_fork() {
  pid_t pid;

  pid = fork();

#ifdef TAU_WRAP_FORK
  if (pid == 0) {
    TAU_REGISTER_FORK(getpid(), TAU_EXCLUDE_PARENT_DATA);
//     fprintf (stderr, "[%d] Registered Fork!\n", getpid());
  } else {
    /* nothing */
  }
#endif

  return pid;
}
#endif /* TAU_WINDOWS */


//////////////////////////////////////////////////////////////////////
// Snapshot related routines
//////////////////////////////////////////////////////////////////////

extern "C" void Tau_profile_snapshot_1l(char *name, int number) {
  char buffer[4096];
  sprintf (buffer, "%s %d", name, number);
  Profiler::Snapshot(buffer);
}

extern "C" void Tau_profile_snapshot(char *name) {
  Profiler::Snapshot(name);
}

extern "C" double* TheTauTraceBeginningOffset() {
  static double offset = 0.0;
  return &offset;
}
extern "C" int* TheTauTraceSyncOffsetSet() {
  static int value = 0;
  return &value;
}

extern "C" double* TheTauTraceSyncOffset() {
  static double offset = -1.0;
  return &offset;
}

double TauSyncAdjustTimeStamp(double timestamp) {
  if (*TheTauTraceSyncOffsetSet() == 0) {
    // return 0 until sync'd
    return 0.0;
  }
  timestamp = timestamp - *TheTauTraceBeginningOffset() + *TheTauTraceSyncOffset();
  return timestamp;
}

extern "C" double TAUClockTime(int tid) {
#ifdef TAU_MULTIPLE_COUNTERS
  // counter 0 is the one we use
  double value = MultipleCounterLayer::getSingleCounter(tid, 0);
#else
  double value = RtsLayer::getUSecD(tid);
#endif
  return value;
}

static int Tau_usesMPI = 0;
extern "C" void Tau_set_usesMPI(int value) {
  Tau_usesMPI = value;
}

extern "C" int Tau_get_usesMPI() {
  return Tau_usesMPI;
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_calls(void *handle, long *values, int tid)
{
  FunctionInfo *ptr = (FunctionInfo *)handle;

  values[0] = (long) ptr->GetCalls(tid);
  return;
}

//////////////////////////////////////////////////////////////////////
void Tau_get_child_calls(void *handle, long* values, int tid)
{
  FunctionInfo *ptr = (FunctionInfo *)handle;

  values[0] = (long) ptr->GetSubrs(tid);
  return;
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_inclusive_values(void *handle, double* values, int tid)
{
  FunctionInfo *ptr = (FunctionInfo *)handle;
  
  if (ptr)
    ptr->getInclusiveValues(tid, values);
  return;
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_exclusive_values(void *handle, double* values, int tid)
{
  FunctionInfo *ptr = (FunctionInfo *)handle;
 
  if (ptr)
    ptr->getExclusiveValues(tid, values);
  return;
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_counter_info(const char ***counterlist, int *numcounters)
{

#ifndef TAU_MULTIPLE_COUNTERS
  Profiler::theCounterList(counterlist, numcounters);
#else
  bool *tmpCounterUsedList; // not used
  MultipleCounterLayer::theCounterListInternal(counterlist,
                                               numcounters,
                                               &tmpCounterUsedList);
#endif
}

//////////////////////////////////////////////////////////////////////
extern "C" int Tau_get_tid(void)
{
  return RtsLayer::myThread();
}

// this routine is called by the destructors of our static objects
// ensuring that the profiles are written out while the objects are still valid
void Tau_destructor_trigger() {
  if ((TheUsingDyninst() || TheUsingCompInst()) && TheSafeToDumpData()) {
    //printf ("FIvector destructor\n");
#ifndef TAU_VAMPIRTRACE
    TAU_PROFILE_EXIT("FunctionDB destructor");
    TheSafeToDumpData() = 0;
#endif
  }
}

//////////////////////////////////////////////////////////////////////
// Sometimes we may link in a library that needs the POMP stuff
// Even when we're not using opari
//////////////////////////////////////////////////////////////////////

#ifdef TAU_SICORTEX
#define TAU_FALSE_POMP
#endif

#ifdef TAU_FALSE_POMP

#pragma weak POMP_MAX_ID=TAU_POMP_MAX_ID
int TAU_POMP_MAX_ID = 0;
#pragma weak pomp_rd_table=tau_pomp_rd_table
int *tau_pomp_rd_table = 0;

// #pragma weak omp_get_thread_num=tau_omp_get_thread_num
// extern "C" int tau_omp_get_thread_num() {
//   return 0;
// }

// #ifdef TAU_OPENMP
// extern "C" {
//   void pomp_parallel_begin();
//   void pomp_parallel_fork_();
//   void POMP_Parallel_fork();
//   void _tau_pomp_will_not_be_called() {
//     pomp_parallel_begin();
//     pomp_parallel_fork_();
//     POMP_Parallel_fork();
//   }
// }
// #endif
#endif

                    

/***************************************************************************
 * $RCSfile: TauCAPI.cpp,v $   $Author: amorris $
 * $Revision: 1.96 $   $Date: 2009/01/15 19:30:40 $
 * VERSION: $Id: TauCAPI.cpp,v 1.96 2009/01/15 19:30:40 amorris Exp $
 ***************************************************************************/

