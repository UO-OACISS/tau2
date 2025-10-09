/****************************************************************************
 * TAU - Perfetto Tracer
 ****************************************************************************/

#ifndef _TAU_TRACE_PERFETTO_H_
#define _TAU_TRACE_PERFETTO_H_

#include <tau_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

int  TAUDECL TauTracePerfettoInit(int tid);
int  TAUDECL TauTracePerfettoInitTS(int tid, x_uint64 ts);
void TAUDECL TauTracePerfettoUnInitialize(int tid);
void TAUDECL TauTracePerfettoReinitialize(int oldid, int newid, int tid);

void TAUDECL TauTracePerfettoEvent(long int ev, x_int64 par, int tid,
                                   x_uint64 ts, int use_ts, int kind);
void TAUDECL TauTracePerfettoEventWithNodeId(long int ev, x_int64 par, int tid,
                                   x_uint64 ts, int use_ts, int node_id, int kind);

void TAUDECL TauTracePerfettoMsg(int send_or_recv, int type, int other_id,
                                 int length, x_uint64 ts, int use_ts, int node_id);

void TAUDECL TauTracePerfettoBarrierAllStart(int tag);
void TAUDECL TauTracePerfettoBarrierAllEnd(int tag);
void TAUDECL TauTracePerfettoRMACollectiveBegin(int tag, int type, int start,
                                 int stride, int size, int data_in, int data_out, int root);
void TAUDECL TauTracePerfettoRMACollectiveEnd(int tag, int type, int start,
                                 int stride, int size, int data_in, int data_out, int root);

void TAUDECL TauTracePerfettoMetadata(const char* name, const char* value, int tid);

void TAUDECL TauTracePerfettoFlushBuffer(int tid);
void TAUDECL TauTracePerfettoShutdownComms(int tid);
void TAUDECL TauTracePerfettoClose(int tid);

#ifdef __cplusplus
}
#endif

#endif /* _TAU_TRACE_PERFETTO_H_ */
