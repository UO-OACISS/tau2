/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 1997-2017  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauTraceOFT2.h 			        	   **
**	Description 	: TAU Tracing Support for native OTF2 generation 	   **
**  Author      : Nicholas Chaimov             **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**                                                                         **
****************************************************************************/

#ifndef _TAU_TRACE_OTF2_H_
#define _TAU_TRACE_OTF2_H_

#include <tau_internal.h>
#ifdef TAU_ENABLED
#include <Profile/TauGpu.h>
#endif // TAU_ENABLED

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
  
  
  int  TAUDECL TauTraceOTF2Init(int tid);
  void TAUDECL TauTraceOTF2InitShmem();
  int  TAUDECL TauTraceOTF2InitTS(int tid, x_uint64 ts);
  void TAUDECL TauTraceOTF2UnInitialize(int tid);
  void TAUDECL TauTraceOTF2Reinitialize(int oldid, int newid, int tid);
  void TAUDECL TauTraceOTF2EventOnly(long int ev, x_int64 par, int tid);
  void TAUDECL TauTraceOTF2FlushBuffer(int tid);
  void TAUDECL TauTraceOTF2EventSimple(long int ev, x_int64 par, int tid, int kind);
  void TAUDECL TauTraceOTF2Event(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts, int kind);
  void TAUDECL TauTraceOTF2EventWithNodeId(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts, int node_id, int kind);
  void TAUDECL TauTraceOTF2Msg(int send_or_recv, int type, int other_id, int length, x_uint64 ts, int use_ts, int node_id);
  void TAUDECL TauTraceOTF2BarrierAllStart(int tag);
  void TAUDECL TauTraceOTF2BarrierAllEnd(int tag);
  void TAUDECL TauTraceOTF2RMACollectiveBegin(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root);
  void TAUDECL TauTraceOTF2RMACollectiveEnd(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root);
  void TAUDECL TauTraceOTF2ShutdownComms(int tid);
  void TAUDECL TauTraceOTF2Close(int tid);


#ifdef __cplusplus
}
#endif /* __cplusplus */




#endif /* _TAU_TRACE_H_ */
