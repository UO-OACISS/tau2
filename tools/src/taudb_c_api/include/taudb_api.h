#ifndef TAUDB_API_H
#define TAUDB_API_H 1

#include "taudb_structs.h"

#ifdef __cplusplus
extern "C" {
#endif
/* when a "get" function is called, this global has the number of
   top-level objects that are returned. */
//int taudb_numItems;

/* the database version */
extern enum taudb_database_schema_version taudb_version;

/* to connect to the database */
TAUDB_CONNECTION* taudb_connect_config(char* config_name, int*);
TAUDB_CONNECTION* taudb_connect_config_file(char* config_file_name, int*);
TAUDB_CONNECTION* taudb_try_connect_config(char* config_name, taudb_error* err, int*);
TAUDB_CONNECTION* taudb_try_connect_config_file(char* config_file_name, taudb_error* err, int*);

/* test the connection status */
int taudb_check_connection(TAUDB_CONNECTION* connection);

/* disconnect from the database */
int taudb_disconnect(TAUDB_CONNECTION* connection);

/* return a string describing an error */
const char * taudb_error_str(taudb_error err);

/************************************************/
/* query functions */
/************************************************/

/* functions to support the old database schema - avoid these if you can */
PERFDMF_APPLICATION* perfdmf_query_applications(TAUDB_CONNECTION* connection,int*);
PERFDMF_EXPERIMENT*  perfdmf_query_experiments(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application, int*);
PERFDMF_APPLICATION* perfdmf_query_application(TAUDB_CONNECTION* connection, char* name);
PERFDMF_EXPERIMENT*  perfdmf_query_experiment(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application, char* name);
TAUDB_TRIAL*         perfdmf_query_trials(TAUDB_CONNECTION* connection, PERFDMF_EXPERIMENT* experiment, int*);

/* get the data sources */
TAUDB_DATA_SOURCE* taudb_query_data_sources(TAUDB_CONNECTION* connection, int*);
TAUDB_DATA_SOURCE* taudb_get_data_source_by_id(TAUDB_DATA_SOURCE* data_sources, const int id);
TAUDB_DATA_SOURCE* taudb_get_data_source_by_name(TAUDB_DATA_SOURCE* data_sources, const char* name);

/* using the properties set in the filter, find a set of trials */
TAUDB_TRIAL* taudb_query_trials(TAUDB_CONNECTION* connection, boolean complete, TAUDB_TRIAL* filter, int*);
TAUDB_PRIMARY_METADATA* taudb_query_primary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, int*);
TAUDB_PRIMARY_METADATA* taudb_get_primary_metadata_by_name(TAUDB_PRIMARY_METADATA* primary_metadata, const char* name);
TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, int*);

/* get the threads for a trial */
TAUDB_THREAD* taudb_query_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial,int*);
TAUDB_THREAD* taudb_query_derived_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial,int*);
TAUDB_THREAD* taudb_get_thread(TAUDB_THREAD* threads, int thread_index);
int taudb_get_total_threads(TAUDB_THREAD* threads);

/* get the metrics for a trial */
TAUDB_METRIC* taudb_query_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_METRIC* taudb_get_metric_by_name(TAUDB_METRIC* metrics, const char* name);
TAUDB_METRIC* taudb_get_metric_by_id(TAUDB_METRIC* metrics, const int id);


/* get the time_ranges for a trial */
TAUDB_TIME_RANGE* taudb_query_time_range(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial);
TAUDB_TIME_RANGE* taudb_get_time_range(TAUDB_TIME_RANGE* time_ranges, const int id);

/* get the timers for a trial */
TAUDB_TIMER* taudb_query_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial,int*);
TAUDB_TIMER* taudb_get_timer_by_id(TAUDB_TIMER* timers, int id);
TAUDB_TIMER* taudb_get_trial_timer_by_name(TAUDB_TIMER* timers, const char* id);
TAUDB_TIMER* taudb_get_trial_timer_by_name(TAUDB_TIMER* timers, const char* id);
TAUDB_TIMER_GROUP* taudb_query_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
void taudb_parse_timer_group_names(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, char* group_names);
TAUDB_TIMER_GROUP* taudb_get_timer_group_from_trial_by_name(TAUDB_TIMER_GROUP* timers, const char* name);
TAUDB_TIMER_GROUP* taudb_get_timer_group_from_timer_by_name(TAUDB_TIMER_GROUP* timers, const char* name);
TAUDB_TIMER_CALLPATH* taudb_query_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer,int*);
TAUDB_TIMER_CALLPATH* taudb_get_timer_callpath_by_id(TAUDB_TIMER_CALLPATH* timers, int id);
TAUDB_TIMER_CALLPATH* taudb_get_timer_callpath_by_name(TAUDB_TIMER_CALLPATH* timers, const char* id);
TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial,int*);
char* taudb_get_callpath_string(TAUDB_TIMER_CALLPATH* timer_callpath);

/* get the counters for a trial */
TAUDB_COUNTER* taudb_query_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_COUNTER* taudb_get_counter_by_id(TAUDB_COUNTER* counters, int id);
TAUDB_COUNTER* taudb_get_counter_by_name(TAUDB_COUNTER* counters, const char* id);
TAUDB_COUNTER_VALUE* taudb_query_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_COUNTER_VALUE* taudb_get_counter_value(TAUDB_COUNTER_VALUE* counter_values, TAUDB_COUNTER* counter, TAUDB_THREAD* thread, TAUDB_TIMER_CALLPATH* context, char* timestamp);

/* get the timer call data for a trial */
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, int*);
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, int*);
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data_by_id(TAUDB_TIMER_CALL_DATA* timer_call_data, int id);
TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data_by_key(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_TIMER_CALLPATH* callpath, TAUDB_THREAD* thread, char* timestamp);

/* get the timer values for a trial */
TAUDB_TIMER_VALUE* taudb_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric, int*);
TAUDB_TIMER_VALUE* taudb_query_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric, int*);
TAUDB_TIMER_VALUE* taudb_query_all_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_TIMER_VALUE* taudb_query_all_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);
TAUDB_TIMER_VALUE* taudb_get_timer_value(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_METRIC* metric);

/* find main */
TAUDB_TIMER* taudb_query_main_timer(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int*);

/* save everything */
void taudb_save_trial(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update, boolean cascade);
void taudb_save_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_time_ranges(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_timer_parameters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_primary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);
void taudb_save_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update);

/************************************************/
/* memory functions */
/************************************************/

char* taudb_strdup(const char* in_string);
TAUDB_TRIAL* taudb_create_trials(int count);
TAUDB_METRIC*             taudb_create_metrics(int count);
TAUDB_TIME_RANGE*         taudb_create_time_ranges(int count);
TAUDB_THREAD*             taudb_create_threads(int count);
TAUDB_SECONDARY_METADATA* taudb_create_secondary_metadata(int count);
TAUDB_PRIMARY_METADATA*   taudb_create_primary_metadata(int count);
TAUDB_PRIMARY_METADATA*   taudb_resize_primary_metadata(int count, TAUDB_PRIMARY_METADATA* old_primary_metadata);
TAUDB_COUNTER*            taudb_create_counters(int count);
TAUDB_COUNTER_VALUE*      taudb_create_counter_values(int count);
TAUDB_TIMER*              taudb_create_timers(int count);
TAUDB_TIMER_PARAMETER*    taudb_create_timer_parameters(int count);
TAUDB_TIMER_GROUP*        taudb_create_timer_groups(int count);
TAUDB_TIMER_GROUP*        taudb_resize_timer_groups(int count, TAUDB_TIMER_GROUP* old_groups);
TAUDB_TIMER_CALLPATH*     taudb_create_timer_callpaths(int count);
TAUDB_TIMER_CALL_DATA*    taudb_create_timer_call_data(int count);
TAUDB_TIMER_VALUE*        taudb_create_timer_values(int count);

void taudb_delete_trials(TAUDB_TRIAL* trials, int count);

/************************************************/
/* Adding objects to the hierarchy */
/************************************************/

void taudb_add_metric_to_trial(TAUDB_TRIAL* trial, TAUDB_METRIC* metric);
void taudb_add_time_range_to_trial(TAUDB_TRIAL* trial, TAUDB_TIME_RANGE* time_range);
void taudb_add_thread_to_trial(TAUDB_TRIAL* trial, TAUDB_THREAD* thread);
void taudb_add_secondary_metadata_to_trial(TAUDB_TRIAL* trial, TAUDB_SECONDARY_METADATA* secondary_metadata);
void taudb_add_secondary_metadata_to_secondary_metadata(TAUDB_SECONDARY_METADATA* parent, TAUDB_SECONDARY_METADATA* child);
void taudb_add_primary_metadata_to_trial(TAUDB_TRIAL* trial, TAUDB_PRIMARY_METADATA* primary_metadata);
void taudb_add_counter_to_trial(TAUDB_TRIAL* trial, TAUDB_COUNTER* counter);
void taudb_add_counter_value_to_trial(TAUDB_TRIAL* trial, TAUDB_COUNTER_VALUE* counter_value);
void taudb_add_timer_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER* timer);
void taudb_add_timer_parameter_to_timer(TAUDB_TIMER* timer, TAUDB_TIMER_PARAMETER* timer_parameter);
void taudb_add_timer_group_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER_GROUP* timer_group);
void taudb_add_timer_to_timer_group(TAUDB_TIMER_GROUP* timer_group, TAUDB_TIMER* timer);
void taudb_add_timer_callpath_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath);
void taudb_add_timer_call_data_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER_CALL_DATA* timer_call_data);
void taudb_add_timer_value_to_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_TIMER_VALUE* timer_value);

/* Profile parsers */
TAUDB_TRIAL* taudb_parse_tau_profiles(const char* directory_name);

/* Analysis routines */
void taudb_compute_statistics(TAUDB_TRIAL* trial);

/* iterators */
TAUDB_DATA_SOURCE* taudb_next_data_source_by_name_from_connection(TAUDB_DATA_SOURCE* current);
TAUDB_DATA_SOURCE* taudb_next_data_source_by_id_from_connection(TAUDB_DATA_SOURCE* current);
TAUDB_THREAD* taudb_next_thread_by_index_from_trial(TAUDB_THREAD* current);
TAUDB_METRIC* taudb_next_metric_by_name_from_trial(TAUDB_METRIC* current);
TAUDB_METRIC* taudb_next_metric_by_id_from_trial(TAUDB_METRIC* current);
TAUDB_TIME_RANGE* taudb_next_time_range_by_id_from_trial(TAUDB_TIME_RANGE* current);
TAUDB_TIMER* taudb_next_timer_by_name_from_trial(TAUDB_TIMER* current);
TAUDB_TIMER* taudb_next_timer_by_id_from_trial(TAUDB_TIMER* current);
TAUDB_TIMER* taudb_next_timer_by_name_from_group(TAUDB_TIMER* current);
TAUDB_TIMER_GROUP* taudb_next_timer_group_by_name_from_trial(TAUDB_TIMER_GROUP* current);
TAUDB_TIMER_GROUP* taudb_next_timer_group_by_name_from_timer(TAUDB_TIMER_GROUP* current);
TAUDB_TIMER_PARAMETER* taudb_next_timer_parameter_by_name_from_timer(TAUDB_TIMER_PARAMETER* current);
TAUDB_TIMER_CALLPATH* taudb_next_timer_callpath_by_name_from_trial(TAUDB_TIMER_CALLPATH* current);
TAUDB_TIMER_CALLPATH* taudb_next_timer_callpath_by_id_from_trial(TAUDB_TIMER_CALLPATH* current);
TAUDB_TIMER_CALL_DATA* taudb_next_timer_call_data_by_key_from_trial(TAUDB_TIMER_CALL_DATA* current);
TAUDB_TIMER_CALL_DATA* taudb_next_timer_call_data_by_id_from_trial(TAUDB_TIMER_CALL_DATA* current);
TAUDB_TIMER_VALUE* taudb_next_timer_value_by_metric_from_timer_call_data(TAUDB_TIMER_VALUE* current);
TAUDB_COUNTER* taudb_next_counter_by_name_from_trial(TAUDB_COUNTER* current);
TAUDB_COUNTER* taudb_next_counter_by_id_from_trial(TAUDB_COUNTER* current);
TAUDB_COUNTER_VALUE* taudb_next_counter_value_by_key_from_trial(TAUDB_COUNTER_VALUE* current);
TAUDB_PRIMARY_METADATA* taudb_next_primary_metadata_by_name_from_trial(TAUDB_PRIMARY_METADATA* current);
TAUDB_SECONDARY_METADATA* taudb_next_secondary_metadata_by_key_from_trial(TAUDB_SECONDARY_METADATA* current);
TAUDB_SECONDARY_METADATA* taudb_next_secondary_metadata_by_id_from_trial(TAUDB_SECONDARY_METADATA* current);

#ifdef __cplusplus
}
#endif

#endif /* TAUDB_API_H */
