#ifndef TAUDB_API_H
#define TAUDB_API_H 1

#include "taudb_structs.h"

/* when a "get" function is called, this global has the number of 
   top-level objects that are returned. */
extern int taudb_numItems;

/* the database version */
extern enum taudb_database_schema_version taudb_version;

/* parse configuration files */
extern TAUDB_CONFIGURATION* taudb_parse_config_file(char* config_file_name);

/* to connect to the database */
extern TAUDB_CONNECTION* taudb_connect_config(char* config_name);
extern TAUDB_CONNECTION* taudb_connect_config_file(char* config_file_name);
extern TAUDB_CONNECTION* taudb_private_connect(char* host, char* port, char* database,
                         char* login, char* password);
extern int taudb_check_schema_version(TAUDB_CONNECTION* connection);

/* test the connection status */
extern int taudb_check_connection(TAUDB_CONNECTION* connection);

/* disconnect from the database */
extern int taudb_disconnect(TAUDB_CONNECTION* connection);

/* when there is an error, disconnect and quit. */
extern void taudb_exit_nicely(TAUDB_CONNECTION* connection);

/************************************************/
/* query functions */
/************************************************/

/* functions to support the old database schema */
extern PERFDMF_APPLICATION* perfdmf_query_applications(TAUDB_CONNECTION* connection);
extern PERFDMF_EXPERIMENT*  perfdmf_query_experiments(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application);
extern PERFDMF_APPLICATION* perfdmf_query_application(TAUDB_CONNECTION* connection, char* name);
extern PERFDMF_EXPERIMENT*  perfdmf_query_experiment(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application, char* name);
extern TAUDB_TRIAL*         perfdmf_query_trials(TAUDB_CONNECTION* connection, PERFDMF_EXPERIMENT* experiment);

/* get the data sources */
extern TAUDB_DATA_SOURCE* taudb_query_data_sources(TAUDB_CONNECTION* connection);
extern TAUDB_DATA_SOURCE* taudb_get_data_source_by_id(TAUDB_DATA_SOURCE* data_sources, const int id);
extern TAUDB_DATA_SOURCE* taudb_get_data_source_by_name(TAUDB_DATA_SOURCE* data_sources, const char* name);

/* using the properties set in the filter, find a set of trials */
extern TAUDB_TRIAL* taudb_query_trials(TAUDB_CONNECTION* connection, boolean complete, TAUDB_TRIAL* filter);
extern TAUDB_PRIMARY_METADATA* taudb_query_primary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter);
extern TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter);

/* get the threads for a trial */
extern TAUDB_THREAD* taudb_query_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_THREAD* taudb_query_derived_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);

/* get the metrics for a trial */
extern TAUDB_METRIC* taudb_query_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_METRIC* taudb_get_metric_by_name(TAUDB_METRIC* metrics, const char* name);
extern TAUDB_METRIC* taudb_get_metric_by_id(TAUDB_METRIC* metrics, const int id);

/* get the time_ranges for a trial */
extern TAUDB_TIME_RANGE* taudb_query_time_range(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIME_RANGE* taudb_get_time_range(TAUDB_TIME_RANGE* time_ranges, const int id);

/* get the timers for a trial */
extern TAUDB_TIMER* taudb_query_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER* taudb_get_timer_by_id(TAUDB_TIMER* timers, int id);
extern TAUDB_TIMER* taudb_get_timer_by_name(TAUDB_TIMER* timers, const char* id);
extern TAUDB_TIMER_GROUP* taudb_query_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern void taudb_parse_timer_group_names(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, char* group_names);
extern TAUDB_TIMER_GROUP* taudb_get_timer_group_by_name(TAUDB_TIMER_GROUP* timers, const char* name);
extern TAUDB_TIMER_CALLPATH* taudb_query_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer);
extern TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern char* taudb_get_callpath_string(TAUDB_TIMER_CALLPATH* timer_callpath);

/* get the counters for a trial */
extern TAUDB_COUNTER* taudb_query_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);

/* get the timer call data for a trial */
extern TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread);
extern TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread);
extern TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);

/* get the timer values for a trial */
extern TAUDB_TIMER_VALUE* taudb_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER_VALUE* taudb_query_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER_VALUE* taudb_query_all_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_VALUE* taudb_query_all_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_VALUE* taudb_get_timer_value(TAUDB_TIMER_VALUE* timer_values, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric);

/* find main */
extern TAUDB_TIMER* taudb_query_main_timer(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);

/************************************************/
/* memory functions */
/************************************************/

/* allocators */
extern PERFDMF_APPLICATION*      perfdmf_create_applications(int count);
extern PERFDMF_EXPERIMENT*       perfdmf_create_experiments(int count);
extern TAUDB_CONFIGURATION*      taudb_create_configuration();
extern TAUDB_DATA_SOURCE*        taudb_create_data_sources(int count);
extern TAUDB_TRIAL*              taudb_create_trials(int count);
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
extern void taudb_delete_data_sources(TAUDB_DATA_SOURCE* data_sources, int count);
extern void taudb_delete_metrics(TAUDB_METRIC* metrics, int count);
extern void taudb_delete_threads(TAUDB_THREAD* threads, int count);
extern void taudb_delete_secondary_metadata(TAUDB_SECONDARY_METADATA* metadata, int count);
extern void taudb_delete_primary_metadata(TAUDB_PRIMARY_METADATA* metadata, int count);
extern void taudb_delete_counters(TAUDB_COUNTER* counters, int count);
extern void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values, int count);
extern void taudb_delete_timers(TAUDB_TIMER* timers, int count);
extern void taudb_delete_timer_parameters(TAUDB_TIMER_PARAMETER* timer_parameters, int count);
extern void taudb_delete_timer_groups(TAUDB_TIMER_GROUP* timer_groups, int count);
extern void taudb_delete_timer_callpaths(TAUDB_TIMER_CALLPATH* timer_callpath, int count);
extern void taudb_delete_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data, int count);
extern void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values, int count);

/* internal database functions */

extern void taudb_begin_transaction(TAUDB_CONNECTION *connection);
extern void* taudb_execute_query(TAUDB_CONNECTION *connection, char* my_query);
extern int taudb_get_num_columns(void* result);
extern int taudb_get_num_rows(void* result);
extern char* taudb_get_column_name(void* result, int column);
extern char* taudb_get_value(void* result, int row, int column);
extern void taudb_clear_result(void* result);
extern void taudb_close_transaction(TAUDB_CONNECTION *connection);

#endif /* TAUDB_API_H */
