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

///////////////////////////////////////////////////////////////////////////
// Wrappers for corresponding C++ functions follow

/* Note: Our old scheme relied on getting a profiler object. This doesn't 
   work well with threads, all threads start/stop the same profiler & 
   profiler is supposed to be for each invocation (thread) as it has a single
   scalar for storing StartTime. So, we changed this so that each 
   tau_get_profiler returns a functionInfo object which can then have 
   independent profilers associated with it. Each start/stop timer call creates 
   and destroys a profiler object now. Since the Fortran layer is built atop 
   the C layer, it remains unchanged. However, we should probably change the 
   name of this method to tau_get_functioninfo or something. */
///////////////////////////////////////////////////////////////////////////
extern "C" void * tau_get_profiler(char *fname, char *type, TauGroup_t group)
{
  FunctionInfo *f;
  //Profiler *p;

  DEBUGPROFMSG("Inside get_profiler group = " << group<<endl;);

  // since we're using new, we should set InitData to true in FunctionInfoInit
  if (group == TAU_MESSAGE)
    f = new FunctionInfo(fname, type, group, "MPI", true);
  else 
    f = new FunctionInfo(fname, type, group, fname, true);
//  p = new Profiler(f, group, true);

  return (void *) f;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_start_timer(void * function_info)
{
  FunctionInfo *f = (FunctionInfo *) function_info; 
  Profiler *p = new Profiler(f, f->GetProfileGroup(), true);
/*
#pragma omp critical
  printf("START tid = %d, profiler= %x\n", RtsLayer::myThread(), p);
*/

  p->Start();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_stop_timer(void * f)
{
  Profiler *p = Profiler::CurrentProfiler[RtsLayer::myThread()];
/*
#pragma omp critical
  printf("STOP tid = %d, profiler= %x\n", RtsLayer::myThread(), p);
*/
  p->Stop();
  delete p;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_exit(char * msg)
{
  TAU_PROFILE_EXIT(msg);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_init(int argc, char **argv)
{
  TAU_PROFILE_INIT(argc, argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_set_node(int node)
{
  TAU_PROFILE_SET_NODE(node);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_set_context(int context)
{
  TAU_PROFILE_SET_CONTEXT(context);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_callstack(void)
{
  TAU_PROFILE_CALLSTACK();
}


///////////////////////////////////////////////////////////////////////////
extern "C" void tau_register_thread(void)
{
  TAU_REGISTER_THREAD();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_register_fork(int nodeid, enum TauFork_t opcode)
{
  TAU_REGISTER_FORK(nodeid, opcode);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_trace_sendmsg(int type, int destination, int length)
{
  TAU_TRACE_SENDMSG(type, destination, length);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_trace_recvmsg(int type, int source, int length)
{
  TAU_TRACE_RECVMSG(type, source, length);
}

///////////////////////////////////////////////////////////////////////////
// User Defined Events 
///////////////////////////////////////////////////////////////////////////
extern "C" void * tau_get_userevent(char *name)
{
  TauUserEvent *ue;
  ue = new TauUserEvent(name);
  return (void *) ue;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_userevent(void *ue, double data)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->TriggerEvent(data);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_report_statistics(void)
{
  TAU_REPORT_STATISTICS();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_report_thread_statistics(void)
{
  TAU_REPORT_THREAD_STATISTICS();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_event_disable_min(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableMin(true);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_event_disable_max(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableMax(true);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_event_disable_mean(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableMean(true);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_event_disable_stddev(void *ue)
{
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetDisableStdDev(true);
} 

///////////////////////////////////////////////////////////////////////////

extern "C" void tau_profile_c_timer(void **ptr, char *fname, char *type, TauGroup_t group)
{
#ifdef DEBUG_PROF
  printf("Inside tau_profile_timer_ fname=%s *ptr = %x\n", fname, *ptr);
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
    *ptr = tau_get_profiler(fname, type, group);
  }

#ifdef DEBUG_PROF
  printf("get_profiler returns %x\n", *ptr);
#endif /* DEBUG_PROF */

  return;
}



/***************************************************************************
 * $RCSfile: TauCAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.13 $   $Date: 2000/10/11 21:21:22 $
 * POOMA_VERSION_ID: $Id: TauCAPI.cpp,v 1.13 2000/10/11 21:21:22 sameer Exp $
 ***************************************************************************/

