/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauFAPI.cpp					  **
**	Description 	: TAU Profiling Package wrapper for F77/F90	  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

/* Fortran Wrapper layer for TAU Portable Profiling */

/*****************************************************************************
* The following routines are called by the Fortran program and they in turn
* invoke the corresponding C routines. 
*****************************************************************************/

typedef unsigned int TauGroup_t;

extern "C" {
void tau_profile_timer_(void **ptr, char *fname, int *flen, char *type, int *tlen, unsigned int *group)
{
}


void tau_profile_start_(void **profiler)
{ 
}

void tau_profile_stop_(void **profiler)
{
}

void tau_profile_exit_(char *msg)
{
}

void tau_profile_init_(int *argc, char **argv)
{
}

void tau_profile_set_node_(int *node)
{
} 

void tau_profile_set_context_(int *context)
{
}

void tau_register_thread_(void)
{
}

/* Cray F90 specific extensions */
void TAU_REGISTER_THREAD(void)
{
}

void tau_trace_sendmsg_(int *type, int *destination, int *length)
{
}

void tau_trace_recvmsg_(int *type, int *source, int *length)
{
}

void tau_register_event_(void **ptr, char *event_name, int *flen)
{
}

void tau_event_(void **ptr, double *data)
{
}

void tau_report_statistics_(void)
{
}

void tau_report_thread_statistics_(void)
{
}

/* Cray F90 specific extensions */
void _main();
void TAU_PROFILE_TIMER(void **ptr, char *fname, int *flen)
{
}

void TAU_PROFILE_START(void **profiler)
{
}

void TAU_PROFILE_STOP(void **profiler)
{
}

void TAU_PROFILE_EXIT(char *msg)
{
}

void TAU_PROFILE_INIT()
{
  _main();
}

void TAU_PROFILE_SET_NODE(int *node)
{
}

void TAU_PROFILE_SET_CONTEXT(int *context)
{
}

void TAU_TRACE_SENDMSG(int *type, int *destination, int *length)
{
}

void TAU_TRACE_RECVMSG(int *type, int *source, int *length)
{
}

void TAU_REGISTER_EVENT(void **ptr, char *event_name, int *flen)
{
}

void TAU_EVENT(void **ptr, double *data)
{
}

void TAU_REPORT_STATISTICS(void)
{
}

void TAU_REPORT_THREAD_STATISTICS(void)
{
}

////////////////////////////////////////////////////////////
// Dummy C wrappers
////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
void * tau_get_profiler(char *fname, char *type, TauGroup_t group)
{
  return 0;
}

///////////////////////////////////////////////////////////////////////////
void tau_start_timer(void *profiler)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_stop_timer(void *profiler)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_exit(char * msg)
{
}
///////////////////////////////////////////////////////////////////////////
void tau_init(int argc, char **argv)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_set_node(int node)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_set_context(int context)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_callstack(void)
{
}


///////////////////////////////////////////////////////////////////////////
void tau_register_thread(void)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_trace_sendmsg(int type, int destination, int length)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_trace_recvmsg(int type, int source, int length)
{
}

///////////////////////////////////////////////////////////////////////////
// User Defined Events 
///////////////////////////////////////////////////////////////////////////
void * tau_get_userevent(char *name)
{
  return 0;
}

///////////////////////////////////////////////////////////////////////////
void tau_userevent(void *ue, double data)
{
} 

///////////////////////////////////////////////////////////////////////////
void tau_report_statistics(void)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_report_thread_statistics(void)
{
}

///////////////////////////////////////////////////////////////////////////
void tau_event_disable_min(void *ue)
{
} 

///////////////////////////////////////////////////////////////////////////
void tau_event_disable_max(void *ue)
{
} 

///////////////////////////////////////////////////////////////////////////
void tau_event_disable_mean(void *ue)
{
} 

///////////////////////////////////////////////////////////////////////////
void tau_event_disable_stddev(void *ue)
{
} 
///////////////////////////////////////////////////////////////////////////


} /* extern "C" */

/***************************************************************************
 * $RCSfile: TauDisable.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 1999/06/18 21:51:55 $
 * POOMA_VERSION_ID: $Id: TauDisable.cpp,v 1.1 1999/06/18 21:51:55 sameer Exp $ 
 ***************************************************************************/
