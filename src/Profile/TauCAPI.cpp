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

#include <iostream.h>
#include "Profile/Profiler.h"

///////////////////////////////////////////////////////////////////////////
// Wrappers for corresponding C++ functions follow

///////////////////////////////////////////////////////////////////////////
extern "C" void * tau_get_profiler(char *fname, char *type, unsigned int group)
{
  FunctionInfo *f;
  Profiler *p;

  DEBUGPROFMSG("Inside get_profiler group = " << group<<endl;);


  f = new FunctionInfo(fname, type, group);
  p = new Profiler(f, group, true);

  return (void *) p;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_start_timer(void *profiler)
{
  Profiler *p = (Profiler *) profiler;

  p->Start();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void tau_stop_timer(void *profiler)
{
  Profiler *p = (Profiler *) profiler;
  p->Stop();
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


/***************************************************************************
 * $RCSfile: TauCAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.3 $   $Date: 1999/04/06 22:52:04 $
 * POOMA_VERSION_ID: $Id: TauCAPI.cpp,v 1.3 1999/04/06 22:52:04 sameer Exp $
 ***************************************************************************/

