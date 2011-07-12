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
#ifdef TAU_ENABLED
#include <Profile/TauGpu.h>
#endif // TAU_ENABLED

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

/* TAU one-side events (for TAU_CUCDA) */
#define TAU_ONESIDED_MESSAGE_SEND 70000
#define TAU_ONESIDED_MESSAGE_RECV 70001
#define TAU_ONESIDED_MESSAGE_ID_1 70002
#define TAU_ONESIDED_MESSAGE_ID_2 70003

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
  
  
  int  TAUDECL TauTraceInit(int tid);
  void TAUDECL TauTraceUnInitialize(int tid);
  void TAUDECL TauTraceReinitialize(int oldid, int newid, int tid);
  void TAUDECL TauTraceEventOnly(long int ev, x_int64 par, int tid);
  void TAUDECL TauTraceFlushBuffer(int tid);
  void TAUDECL TauTraceEventSimple(long int ev, x_int64 par, int tid);
  void TAUDECL TauTraceEvent(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts);
  void TAUDECL TauTraceClose(int tid);
  void TAUDECL TauTraceSetFlushEvents(int value);
  int  TAUDECL TauTraceGetFlushEvents();
  double TAUDECL TauTraceGetTime(int tid);
  int TAUDECL TauTraceDumpEDF(int tid);
  int TAUDECL TauTraceMergeAndConvertTracesIfNecessary(void);

  void TAUDECL TauTraceSendMsg(int type, int destination, int length);
  void TAUDECL TauTraceRecvMsg(int type, int source, int length);
  void TAUDECL TauTraceSendMsgRemote(int type, int destination, int length, int remoteid);
  void TAUDECL TauTraceRecvMsgRemote(int type, int source, int length, int remoteid);
#if TAU_GPU
  void TAUDECL TauTraceOneSidedMsg(bool type, gpuId *gpu, int length, int thread);
#endif // TAU_GPU
  /* Returns a pointer to the (singleton) offset info struct */
  TauTraceOffsetInfo* TAUDECL TheTauTraceOffsetInfo();

#ifdef __cplusplus
}
#endif /* __cplusplus */




#endif /* _TAU_TRACE_H_ */
