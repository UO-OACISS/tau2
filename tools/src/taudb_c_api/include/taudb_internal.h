#ifndef TAUDB_INTERNAL_H
#define TAUDB_INTERNAL_H 1

#include "taudb_api.h"

/* parse configuration files */
extern TAUDB_CONFIGURATION* taudb_parse_config_file(char* config_file_name);

/* to connect to the database */
extern TAUDB_CONNECTION* taudb_private_connect(char* host, char* port, char* database,
                         char* login, char* password);
extern int taudb_check_schema_version(TAUDB_CONNECTION* connection);

/* when there is an error, disconnect and quit. */
extern void taudb_exit_nicely(TAUDB_CONNECTION* connection);

/************************************************/
/* memory functions */
/************************************************/

/* allocators */
extern PERFDMF_APPLICATION*      perfdmf_create_applications(int count);
extern PERFDMF_EXPERIMENT*       perfdmf_create_experiments(int count);
extern TAUDB_CONFIGURATION*      taudb_create_configuration();
extern TAUDB_CONNECTION*         taudb_create_connection();
extern TAUDB_DATA_SOURCE*        taudb_create_data_sources(int count);
extern TAUDB_METRIC*             taudb_create_metrics(int count);
extern TAUDB_TIME_RANGE*         taudb_create_time_ranges(int count);
extern TAUDB_THREAD*             taudb_create_threads(int count);
extern TAUDB_SECONDARY_METADATA* taudb_create_secondary_metadata(int count);
extern TAUDB_PRIMARY_METADATA*   taudb_create_primary_metadata(int count);
extern TAUDB_PRIMARY_METADATA*   taudb_resize_primary_metadata(int count, TAUDB_PRIMARY_METADATA* old_primary_metadata);
extern TAUDB_COUNTER*            taudb_create_counters(int count);
extern TAUDB_COUNTER_VALUE*      taudb_create_counter_values(int count);
extern TAUDB_TIMER*              taudb_create_timers(int count);
extern TAUDB_TIMER_PARAMETER*    taudb_create_timer_parameters(int count);
extern TAUDB_TIMER_GROUP*        taudb_create_timer_groups(int count);
extern TAUDB_TIMER_GROUP*        taudb_resize_timer_groups(int count, TAUDB_TIMER_GROUP* old_groups);
extern TAUDB_TIMER_CALLPATH*     taudb_create_timer_callpaths(int count);
extern TAUDB_TIMER_CALL_DATA*    taudb_create_timer_call_data(int count);
extern TAUDB_TIMER_VALUE*        taudb_create_timer_values(int count);
extern char*                     taudb_create_and_copy_string(const char* in_string);
extern char*                     taudb_create_hash_key_2(int thread, const char* timer);
extern char*                     taudb_create_hash_key_3(int thread, const char* timer, const char* metric);

/* freers */
extern void perfdmf_delete_applications(PERFDMF_APPLICATION* applications, int count);
extern void perfdmf_delete_experiments(PERFDMF_EXPERIMENT* experiments, int count);
extern void taudb_delete_trials(TAUDB_TRIAL* trials, int count);
extern void taudb_delete_data_sources(TAUDB_DATA_SOURCE* data_sources);
extern void taudb_delete_metrics(TAUDB_METRIC* metrics);
extern void taudb_delete_threads(TAUDB_THREAD* threads);
extern void taudb_delete_secondary_metadata(TAUDB_SECONDARY_METADATA* metadata);
extern void taudb_delete_primary_metadata(TAUDB_PRIMARY_METADATA* metadata);
extern void taudb_delete_counters(TAUDB_COUNTER* counters);
extern void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values);
extern void taudb_delete_timers(TAUDB_TIMER* timers);
extern void taudb_delete_timer_parameters(TAUDB_TIMER_PARAMETER* timer_parameters);
extern void taudb_delete_timer_groups(TAUDB_TIMER_GROUP* timer_groups);
extern void taudb_delete_timer_callpaths(TAUDB_TIMER_CALLPATH* timer_callpath);
extern void taudb_delete_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data);
extern void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values);

/* internal database functions */

extern void taudb_begin_transaction(TAUDB_CONNECTION *connection);
extern void taudb_execute_query(TAUDB_CONNECTION *connection, char* my_query);
extern int taudb_get_num_columns(TAUDB_CONNECTION *connection);
extern int taudb_get_num_rows(TAUDB_CONNECTION *connection);
extern char* taudb_get_column_name(TAUDB_CONNECTION *connection, int column);
extern char* taudb_get_value(TAUDB_CONNECTION *connection, int row, int column);
extern char* taudb_get_binary_value(TAUDB_CONNECTION *connection, int row, int column);
extern void taudb_clear_result(TAUDB_CONNECTION *connection);
extern void taudb_close_transaction(TAUDB_CONNECTION *connection);

#endif /* TAUDB_INTERNAL_H */
