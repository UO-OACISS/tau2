/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2009  						   	   **
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

#include <tau_internal.h>


/* TAU tracer events */
#define TAU_EV_INIT         60000
#define TAU_EV_FLUSH_ENTER  60001
#define TAU_EV_FLUSH_EXIT   60002
#define TAU_EV_CLOSE        60003
#define TAU_EV_INITM        60004
#define TAU_EV_WALL_CLOCK   60005
#define TAU_EV_CONT_EVENT   60006
#define TAU_MESSAGE_SEND    60007
#define TAU_MESSAGE_RECV    60008


/* event record description */
typedef struct {
  x_int32  ev;    /* event id                    */
  x_uint16 nid;   /* node id                     */
  x_uint16 tid;   /* thread id                   */
  x_int64  par;   /* event parameter             */
  x_uint64 ti;    /* timestamp (in microseconds) */
} TAU_EV;

/* structure to hold clocksync offset info */
typedef struct {
  int enabled;
  double beginOffset;
  double syncOffset;
} TauTraceOffsetInfo;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
  
  
  int  TauTraceInit(int tid);
  void TauTraceUnInitialize(int tid);
  void TauTraceReinitialize(int oldid, int newid, int tid);
  void TauTraceEventOnly(long int ev, x_int64 par, int tid);
  void TauTraceFlushBuffer(int tid);
  void TauTraceEventSimple(long int ev, x_int64 par, int tid);
  void TauTraceEvent(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts);
  void TauTraceClose(int tid);
  void TauTraceSetFlushEvents(int value);
  int  TauTraceGetFlushEvents();
  double TauTraceGetTime(int tid);
  int TauTraceDumpEDF(int tid);
  int TauTraceMergeAndConvertTracesIfNecessary(void);

  void TauTraceSendMsg(int type, int destination, int length);
  void TauTraceRecvMsg(int type, int source, int length);


  /* Returns a pointer to the (singleton) offset info struct */
  TauTraceOffsetInfo *TheTauTraceOffsetInfo();

#ifdef __cplusplus
}
#endif /* __cplusplus */




#endif /* _TAU_TRACE_H_ */
