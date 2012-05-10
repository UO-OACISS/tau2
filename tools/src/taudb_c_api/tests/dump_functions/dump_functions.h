#include "taudb_api.h"

extern void dump_metadata(TAUDB_PRIMARY_METADATA *metadata, int count);
extern void dump_trial(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_timers(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_metrics(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_threads(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_timer_callpaths(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_timer_stats(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_timer_values(PGconn* connection, TAUDB_TRIAL* filter);

