/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
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
extern "C" void * Tau_get_profiler(char *fname, char *type, TauGroup_t group, char *gr_name)
{
  FunctionInfo *f;
  //Profiler *p;

  DEBUGPROFMSG("Inside get_profiler group = " << group<<endl;);

  // since we're using new, we should set InitData to true in FunctionInfoInit
  if (group == TAU_MESSAGE)
    f = new FunctionInfo(fname, type, group, "MPI", true);
  else 
    f = new FunctionInfo(fname, type, group, gr_name, true);
//  p = new Profiler(f, group, true);

  return (void *) f;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_start_timer(void * function_info)
{
  FunctionInfo *f = (FunctionInfo *) function_info; 
  TauGroup_t gr = f->GetProfileGroup();
  if (gr & RtsLayer::TheProfileMask())
  {
    Profiler *p = new Profiler(f, gr, true);
/*
#pragma omp critical
  printf("START tid = %d, profiler= %x\n", RtsLayer::myThread(), p);
*/

    p->Start();
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_stop_timer(void * function_info)
{
  FunctionInfo *f = (FunctionInfo *) function_info; 
  if (f->GetProfileGroup() & RtsLayer::TheProfileMask())
  {
    Profiler *p = Profiler::CurrentProfiler[RtsLayer::myThread()];
/*
#pragma omp critical
  printf("STOP tid = %d, profiler= %x\n", RtsLayer::myThread(), p);
*/
    p->Stop();
    delete p;
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_exit(char * msg)
{
  TAU_PROFILE_EXIT(msg);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_init_ref(int* argc, char ***argv)
{
  TAU_INIT(argc, argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_init(int argc, char **argv)
{
  TAU_PROFILE_INIT(argc, argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_node(int node)
{
  TAU_PROFILE_SET_NODE(node);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_context(int context)
{
  TAU_PROFILE_SET_CONTEXT(context);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_callstack(void)
{
  TAU_PROFILE_CALLSTACK();
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump(void)
{
  return TAU_DB_DUMP();
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_incr(void)
{
  return TAU_DB_DUMP_INCR();
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_purge(void)
{
  TAU_DB_PURGE();
}

extern "C" void Tau_the_function_list(const char ***functionList, int *num)
{
  TAU_GET_FUNC_NAMES(*functionList, *num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_names()
{
  TAU_DUMP_FUNC_NAMES();
}
  
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_the_counter_names(const char **counterList, int num)
{
  TAU_GET_COUNTER_NAMES(counterList, num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_function_values(const char **inFuncs, int numOfFuncs,
					double ***counterExclusiveValues,
					double ***counterInclusiveValues,
					int **numOfCalls, int **numOfSubRoutines,
					const char ***counterNames, int *numOfCounters)
{
  TAU_GET_FUNC_VALS(inFuncs,numOfFuncs,*counterExclusiveValues,*counterInclusiveValues,
		    *numOfCalls,*numOfSubRoutines,*counterNames,*numOfCounters);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_values(const char **functionList, int num)
{
  TAU_DUMP_FUNC_VALS(functionList,num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_values_incr(const char **functionList, int num)
{
  TAU_DUMP_FUNC_VALS_INCR(functionList,num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_register_thread(void)
{
  TAU_REGISTER_THREAD();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_register_fork(int nodeid, enum TauFork_t opcode)
{
  TAU_REGISTER_FORK(nodeid, opcode);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_instrumentation(void)
{
  TAU_ENABLE_INSTRUMENTATION();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_instrumentation(void)
{
  TAU_DISABLE_INSTRUMENTATION();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_shutdown(void)
{
  RtsLayer::TheShutdown() = true;
  TAU_DISABLE_INSTRUMENTATION();
}

///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_enable_group_name(char * group)
{
  return TAU_ENABLE_GROUP_NAME(group);
}

///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_disable_group_name(char * group)
{
  return TAU_DISABLE_GROUP_NAME(group);
}

///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_get_profile_group(char * group)
{
  return TAU_GET_PROFILE_GROUP(group);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_group(TauGroup_t group)
{
  TAU_ENABLE_GROUP(group);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_group(TauGroup_t group)
{
  TAU_DISABLE_GROUP(group);
}

///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_enable_all_groups(void)
{
  return TAU_ENABLE_ALL_GROUPS();
}

///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_disable_all_groups(void)
{
  return TAU_DISABLE_ALL_GROUPS();
}

/* TAU's totalnodes implementation follows */
///////////////////////////////////////////////////////////////////////////
extern "C" int tau_totalnodes(int set_or_get, int value)
{
  static int nodes = 1;
  if (set_or_get == 1) /* SET (in is 1) , GET (out is 0) */
  {
    nodes = value;
  }
  return nodes;
}

#ifdef TAU_MPI
TAU_REGISTER_EVENT(sendevent,"Message size sent to all nodes");
TauUserEvent **u;
int register_events(void)
{
  char str[256];
  int i;
  u = (TauUserEvent **) malloc(sizeof(TauUserEvent *)*tau_totalnodes(0,0));
  for (i =0; i < tau_totalnodes(0,0); i++)
  {
    sprintf(str, "Message size sent to node %d", i);
    u[i] = (TauUserEvent *) new TauUserEvent((const char *)str);
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length)
{
  static int initialize = register_events();
#ifdef DEBUG_PROF
  printf("Node %d: Tau_trace_sendmsg: type %d dest %d len %d\n", 
        RtsLayer::myNode(), type, destination, length);
#endif /* DEBUG_PROF */
  TAU_EVENT(sendevent, length);
  u[destination]->TriggerEvent(length, RtsLayer::myThread());
  TAU_TRACE_SENDMSG(type, destination, length);
}

#else /* !TAU_MPI */
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length)
{
  TAU_TRACE_SENDMSG(type, destination, length);
}
#endif /* TAU_MPI */
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg(int type, int source, int length)
{
  TAU_TRACE_RECVMSG(type, source, length);
}

///////////////////////////////////////////////////////////////////////////
// User Defined Events 
///////////////////////////////////////////////////////////////////////////
extern "C" void * Tau_get_userevent(char *name)
{
  TauUserEvent *ue;
  ue = new TauUserEvent(name);
  return (void *) ue;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_userevent(void *ue, double data)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->TriggerEvent(data);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_event_name(void *ue, char *name)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetEventName(name);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_report_statistics(void)
{
  TAU_REPORT_STATISTICS();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_report_thread_statistics(void)
{
  TAU_REPORT_THREAD_STATISTICS();
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
  if (*ptr == 0)
  {  // remove garbage characters from the end of name
    for(int i=0; i<strlen(fname); i++)
    {
      if (!isprint(fname[i]))
      {
        fname[i] = '\0';
        break;
      }
    }
    *ptr = Tau_get_profiler(fname, type, group, group_name);
  }

#ifdef DEBUG_PROF
  printf("get_profiler returns %x\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}



/***************************************************************************
 * $RCSfile: TauCAPI.cpp,v $   $Author: bertie $
 * $Revision: 1.31 $   $Date: 2002/11/28 00:53:00 $
 * VERSION: $Id: TauCAPI.cpp,v 1.31 2002/11/28 00:53:00 bertie Exp $
 ***************************************************************************/

