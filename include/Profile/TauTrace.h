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


extern unsigned long int pcxx_ev_class;


/* -- pcxx tracer events ------------------- */
#define PCXX_EV_INIT         60000
#define PCXX_EV_FLUSH_ENTER  60001
#define PCXX_EV_FLUSH_EXIT   60002
#define PCXX_EV_CLOSE        60003
#define PCXX_EV_INITM        60004
#define PCXX_EV_WALL_CLOCK   60005
#define PCXX_EV_CONT_EVENT   60006
#define TAU_MESSAGE_SEND     60007
#define TAU_MESSAGE_RECV     60008

/* -- the following two events are only the ----- */
/* -- base numbers, actually both represent ----- */
/* -- 64 events (60[1234]00 to 60[1234]64)  ----- */
#define PCXX_WTIMER_CLEAR    60199
#define PCXX_WTIMER_START    60100
#define PCXX_WTIMER_STOP     60200
#define PCXX_UTIMER_CLEAR    60399
#define PCXX_UTIMER_START    60300
#define PCXX_UTIMER_STOP     60400

/* from pcxx_machines.h */
#define PCXX_MAXPROCS 4096
#define PCXX_MALLOC malloc
#define PCXX_SUGGESTED_MSG_SIZE -1
#define PCXX_MYNODE   RtsLayer::myNode()
#define PCXX_MYTHREAD 0

#ifdef TAU_LIBRARY_SOURCE

#ifndef PCXX_BUFSIZE
#define PCXX_BUFSIZE 65536  /* -- 64 K -- */
#endif

/* -- event classes ----------------------------- */
#ifndef PCXX_EC_NULL
#define PCXX_EC_NULL       0x00000000
#define PCXX_EC_TRACER     0x00000001
#define PCXX_EC_TIMER      0x00000002
#define PCXX_EC_ALL        0xFFFFFFFF
#endif

#endif /* TAU_LIBRARY_SOURCE */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
  
  /* -- event record buffer descriptor ------------------------- */
  typedef struct {
    x_int32            ev;    /* -- event id        -- */
    x_uint16           nid;   /* -- node id         -- */
    x_uint16           tid;   /* -- thread id       -- */
    x_int64            par;   /* -- event parameter -- */
    x_uint64           ti;    /* -- time [us]?      -- */
  } PCXX_EV;
  
#ifdef TAU_LIBRARY_SOURCE
  
  /* -- pointer to next free element of event record buffer ---- */
  extern PCXX_EV *pcxx_ev_ptr;
  
  /* -- pointer to last available element of event record buffer */
  extern PCXX_EV *pcxx_ev_max;
  
#endif /* TAU_LIBRARY_SOURCE */
  
  /* -- pcxx monitor routines ------------------------------------ */
#define PCXX_EVENT(c,e,p) if (pcxx_ev_class&(c)) pcxx_Event(e,p)
#define PCXX_LONG_EVENT(c,e,l,p) if (pcxx_ev_class&(c)) pcxx_LongEvent(e,l,p)
#ifdef __cplusplus
  void pcxx_EvInit (char *n);
  void pcxx_Event (long int e, x_int64 p);
  void pcxx_LongEvent (long int e, int l, char *p);
  void pcxx_EvClose ();
  void pcxx_EvFlush ();
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
extern void pcxx_EvInit(char *n);
extern void pcxx_Event(long int e, long long p);
extern void pcxx_LongEvent(long int e, int l, char *p);
extern void pcxx_EvClose ();
extern void pcxx_EvFlush ();
#endif /* __cplusplus */

#else

#define PCXX_EVENT(c, e, p)
#define PCXX_LONG_EVENT(c, e, l, p)
#define pcxx_EvInit(n)
#define pcxx_Event(e, p)
#define pcxx_LongEvent(e, l, p)
#define pcxx_EvClose()
#define pcxx_EvFlush()

#endif /* TRACING_ON */

#endif /* _TAU_TRACE_H_ */
