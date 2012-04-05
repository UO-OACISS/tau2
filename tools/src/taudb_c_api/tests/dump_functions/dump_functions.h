#include "taudb_api.h"

extern void dump_metadata(TAUDB_PRIMARY_METADATA *metadata, int count);
extern void dump_trial(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_trial_timers(PGconn* connection, TAUDB_TRIAL* filter);
extern void dump_trial_metrics(PGconn* connection, TAUDB_TRIAL* filter);
//extern void dump_trial_threads(PGconn* connection, TAUDB_TRIAL* filter);
//extern void dump_trial_values(PGconn* connection, TAUDB_TRIAL* filter);

