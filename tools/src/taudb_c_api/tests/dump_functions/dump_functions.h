#include "taudb_api.h"

extern void dump_metadata(TAUDB_PRIMARY_METADATA *metadata);
extern void dump_secondary_metadata(TAUDB_SECONDARY_METADATA *metadata);
extern void dump_trial(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
extern void dump_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);

