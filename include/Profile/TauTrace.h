/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2008  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauTrace.h 			        	   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : TAU Tracing                                      **
**                                                                         **
****************************************************************************/

#ifndef _TAU_TRACE_H_
#define _TAU_TRACE_H_

#ifdef TRACING_ON

#include "tau_types.h"


#if defined (__cplusplus) || defined (__STDC__) || defined (_AIX) || (defined (__mips) && defined (_SYSTYPE_SVR4))
#define SIGNAL_TYPE	void
#define SIGNAL_ARG_TYPE	int
#else	/* Not ANSI C.  */
#define SIGNAL_TYPE	int
#define SIGNAL_ARG_TYPE
#endif	/* ANSI C */


/* -- tau tracer events ------------------- */
#define TAU_EV_INIT         60000
#define TAU_EV_FLUSH_ENTER  60001
#define TAU_EV_FLUSH_EXIT   60002
#define TAU_EV_CLOSE        60003
#define TAU_EV_INITM        60004
#define TAU_EV_WALL_CLOCK   60005
#define TAU_EV_CONT_EVENT   60006
#define TAU_MESSAGE_SEND     60007
#define TAU_MESSAGE_RECV     60008



#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
  
  /* -- event record buffer descriptor ------------------------- */
  typedef struct {
    x_int32  ev;    /* -- event id        -- */
    x_uint16 nid;   /* -- node id         -- */
    x_uint16 tid;   /* -- thread id       -- */
    x_int64  par;   /* -- event parameter -- */
    x_uint64 ti;    /* -- time [us]?      -- */
  } TAU_EV;
  
#ifdef TAU_LIBRARY_SOURCE
  
  /* -- pointer to next free element of event record buffer ---- */
  extern TAU_EV *tau_ev_ptr;
  
  /* -- pointer to last available element of event record buffer */
  extern TAU_EV *tau_ev_max;
  
#endif /* TAU_LIBRARY_SOURCE */
  
#ifdef __cplusplus
  void tautrace_EvInit (char *n);
  void tautrace_Event (long int e, x_int64 p);
  void tautrace_LongEvent (long int e, int l, char *p);
  void tautrace_EvClose ();
  void tautrace_EvFlush ();
  /* New tracing interface */
  int TraceEvInit(int tid);
  void TraceUnInitialize(int tid);
  void TraceReinitialize(int oldid, int newid, int tid);
  void TraceEventOnly(long int ev, x_int64 par, int tid);
  void TraceEvFlush(int tid);
  void TraceEvent(long int ev, x_int64 par, int tid, x_uint64 ts = 0L, int use_ts = 0);
  void TraceEvClose(int tid);
  void SetFlushEvents(int tid);
  int  GetFlushEvents(int tid);
}
#else
extern void tautrace_EvInit(char *n);
extern void tautrace_Event(long int e, long long p);
extern void tautrace_LongEvent(long int e, int l, char *p);
extern void tautrace_EvClose ();
extern void tautrace_EvFlush ();
#endif /* __cplusplus */

#else

#define tautrace_EvInit(n)
#define tautrace_Event(e, p)
#define tautrace_LongEvent(e, l, p)
#define tautrace_EvClose()
#define tautrace_EvFlush()

#endif /* TRACING_ON */

#endif /* _TAU_TRACE_H_ */
