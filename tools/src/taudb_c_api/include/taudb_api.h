#ifndef TAUDB_API_H
#define TAUDB_API_H 1

#include "taudb_structs.h"

/* when a "get" function is called, this global has the number of 
   top-level objects that are returned. */
extern int taudb_numItems;

/* the database version */
extern enum taudb_database_schema_version taudb_version;

/* to connect to the database */
extern TAUDB_CONNECTION* taudb_connect_config(char* config_name);
extern TAUDB_CONNECTION* taudb_connect_config_file(char* config_file_name);

/* test the connection status */
extern int taudb_check_connection(TAUDB_CONNECTION* connection);

/* disconnect from the database */
extern int taudb_disconnect(TAUDB_CONNECTION* connection);

/************************************************/
/* query functions */
/************************************************/

/* functions to support the old database schema - avoid these if you can */
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
extern TAUDB_PRIMARY_METADATA* taudb_get_primary_metadata_by_name(TAUDB_PRIMARY_METADATA* primary_metadata, const char* name);
extern TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter);

/* get the threads for a trial */
extern TAUDB_THREAD* taudb_query_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_THREAD* taudb_query_derived_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_THREAD* taudb_get_thread(TAUDB_THREAD* threads, int thread_index);

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
extern TAUDB_TIMER_CALLPATH* taudb_get_timer_callpath_by_id(TAUDB_TIMER_CALLPATH* timers, int id);
extern TAUDB_TIMER_CALLPATH* taudb_get_timer_callpath_by_name(TAUDB_TIMER_CALLPATH* timers, const char* id);
extern TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern char* taudb_get_callpath_string(TAUDB_TIMER_CALLPATH* timer_callpath);

/* get the counters for a trial */
extern TAUDB_COUNTER* taudb_query_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_COUNTER* taudb_get_counter_by_id(TAUDB_COUNTER* counters, int id);
extern TAUDB_COUNTER* taudb_get_counter_by_name(TAUDB_COUNTER* counters, const char* id);
extern TAUDB_COUNTER_VALUE* taudb_query_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
TAUDB_COUNTER_VALUE* taudb_get_counter_value(TAUDB_COUNTER_VALUE* counter_values, TAUDB_COUNTER* counter, TAUDB_THREAD* thread, TAUDB_TIMER_CALLPATH* context, char* timestamp);

/* get the timer call data for a trial */
extern TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread);
extern TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread);
extern TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data_by_id(TAUDB_TIMER_CALL_DATA* timer_call_data, int id);
extern TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data_by_key(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_TIMER_CALLPATH* callpath, TAUDB_THREAD* thread, char* timestamp);

/* get the timer values for a trial */
extern TAUDB_TIMER_VALUE* taudb_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER_VALUE* taudb_query_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER_VALUE* taudb_query_all_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_VALUE* taudb_query_all_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_VALUE* taudb_get_timer_value(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_METRIC* metric);

/* find main */
extern TAUDB_TIMER* taudb_query_main_timer(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);

/************************************************/
/* memory functions */
/************************************************/

extern TAUDB_TRIAL* taudb_create_trials(int count);
extern void taudb_delete_trials(TAUDB_TRIAL* trials, int count);

/* Profile parsers */
extern TAUDB_TRIAL* taudb_parse_tau_profiles(const char* directory_name);

#endif /* TAUDB_API_H */
