#include "taudb_api.h"

#ifdef __cplusplus
extern "C" {
#endif

void dump_metadata(TAUDB_PRIMARY_METADATA *metadata);
void dump_secondary_metadata(TAUDB_SECONDARY_METADATA *metadata);
void dump_trial(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);
void dump_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial);

#ifdef __cplusplus
}
#endif
