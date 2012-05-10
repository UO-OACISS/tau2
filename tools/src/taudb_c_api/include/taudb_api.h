#ifndef TAUDB_API_H
#define TAUDB_API_H 1

#include "libpq-fe.h"
#include "taudb_structs.h"

// when a "get" function is called, this global has the number of 
// top-level objects that are returned. 
extern int taudb_numItems;

// the database version
extern enum taudb_database_schema_version taudb_version;

// parse configuration files
extern TAUDB_CONFIGURATION* taudb_parse_config_file(char* config_file_name);

// to connect to the database
extern PGconn* taudb_connect_config(char* config_name);
extern PGconn* taudb_connect_config_file(char* config_file_name);
extern PGconn* taudb_private_connect(char* host, char* port, char* database,
                         char* login, char* password);
extern int taudb_check_schema_version(PGconn* connection);

// test the connection status
extern int taudb_check_connection(PGconn* connection);

// disconnect from the database
extern int taudb_disconnect(PGconn* connection);

// when there is an error, disconnect and quit.
extern void taudb_exit_nicely(PGconn* connection);

// functions to support the old database schema
extern PERFDMF_APPLICATION* perfdmf_query_applications(PGconn* connection);
extern PERFDMF_EXPERIMENT*  perfdmf_query_experiments(PGconn* connection, PERFDMF_APPLICATION* application);
extern PERFDMF_APPLICATION* perfdmf_query_application(PGconn* connection, char* name);
extern PERFDMF_EXPERIMENT*  perfdmf_query_experiment(PGconn* connection, PERFDMF_APPLICATION* application, char* name);
extern TAUDB_TRIAL*         perfdmf_query_trials(PGconn* connection, PERFDMF_EXPERIMENT* experiment);

// using the properties set in the filter, find a set of trials
extern TAUDB_TRIAL* taudb_query_trials(PGconn* connection, boolean complete, TAUDB_TRIAL* filter);
extern TAUDB_PRIMARY_METADATA* taudb_query_primary_metadata(PGconn* connection, TAUDB_TRIAL* filter);
extern TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(PGconn* connection, TAUDB_TRIAL* filter);

// get the threads for a trial
extern TAUDB_THREAD* taudb_query_threads(PGconn* connection, TAUDB_TRIAL* trial);
extern TAUDB_THREAD* taudb_query_derived_threads(PGconn* connection, TAUDB_TRIAL* trial);

// get the metrics for a trial
extern TAUDB_METRIC* taudb_query_metrics(PGconn* connection, TAUDB_TRIAL* trial);

// get the timers for a trial
extern TAUDB_TIMER* taudb_query_timers(PGconn* connection, TAUDB_TRIAL* trial);

// get the counters for a trial
extern TAUDB_COUNTER* taudb_query_counters(PGconn* connection, TAUDB_TRIAL* trial);

// get the timer callpath data for a trial
extern TAUDB_TIMER_CALLPATH* taudb_query_timer_callpaths(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread);
extern TAUDB_TIMER_CALLPATH* taudb_query_timer_callpath_stats(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread);
extern TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpaths(PGconn* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpath_stats(PGconn* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_CALLPATH* taudb_get_timer_callpath(TAUDB_TIMER_CALLPATH* timer_callpaths, TAUDB_TIMER* timer, TAUDB_THREAD* thread);

// get the timer values for a trial
extern TAUDB_TIMER_VALUE* taudb_query_timer_values(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER_VALUE* taudb_query_timer_stats(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER_VALUE* taudb_query_all_timer_values(PGconn* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_VALUE* taudb_query_all_timer_stats(PGconn* connection, TAUDB_TRIAL* trial);
extern TAUDB_TIMER_VALUE* taudb_get_timer_value(TAUDB_TIMER_VALUE* timer_values, TAUDB_TIMER* timer, TAUDB_THREAD* thread, TAUDB_METRIC* metric);

// find main
extern TAUDB_TIMER* taudb_query_main_timer(PGconn* connection, TAUDB_TRIAL* trial);

// allocators
extern PERFDMF_APPLICATION*      perfdmf_create_applications(int count);
extern PERFDMF_EXPERIMENT*       perfdmf_create_experiments(int count);
extern TAUDB_CONFIGURATION*      taudb_create_configuration();
extern TAUDB_TRIAL*              taudb_create_trials(int count);
extern TAUDB_METRIC*             taudb_create_metrics(int count);
extern TAUDB_THREAD*             taudb_create_threads(int count);
extern TAUDB_SECONDARY_METADATA* taudb_create_secondary_metadata(int count);
extern TAUDB_PRIMARY_METADATA*   taudb_create_primary_metadata(int count);
extern TAUDB_PRIMARY_METADATA*   taudb_resize_primary_metadata(int count, TAUDB_PRIMARY_METADATA* old_primary_metadata);
extern TAUDB_COUNTER*            taudb_create_counters(int count);
extern TAUDB_COUNTER_GROUP*      taudb_create_counter_groups(int count);
extern TAUDB_COUNTER_GROUP*      taudb_resize_counter_groups(int count, TAUDB_COUNTER_GROUP* old_groups);
extern TAUDB_COUNTER_VALUE*      taudb_create_counter_values(int count);
extern TAUDB_TIMER*              taudb_create_timers(int count);
extern TAUDB_TIMER_PARAMETER*    taudb_create_timer_parameters(int count);
extern TAUDB_TIMER_GROUP*        taudb_create_timer_groups(int count);
extern TAUDB_TIMER_GROUP*        taudb_resize_timer_groups(int count, TAUDB_TIMER_GROUP* old_groups);
extern TAUDB_TIMER_CALLPATH*     taudb_create_timer_callpaths(int count);
extern TAUDB_TIMER_VALUE*        taudb_create_timer_values(int count);
extern char*                     taudb_create_and_copy_string(const char* in_string);
extern char*                     taudb_create_hash_key_2(int thread, const char* timer);
extern char*                     taudb_create_hash_key_3(int thread, const char* timer, const char* metric);

// freers
extern void perfdmf_delete_applications(PERFDMF_APPLICATION* applications, int count);
extern void perfdmf_delete_experiments(PERFDMF_EXPERIMENT* experiments, int count);
extern void taudb_delete_trials(TAUDB_TRIAL* trials, int count);
extern void taudb_delete_metrics(TAUDB_METRIC* metrics, int count);
extern void taudb_delete_threads(TAUDB_THREAD* threads, int count);
extern void taudb_delete_secondary_metadata(TAUDB_SECONDARY_METADATA* metadata, int count);
extern void taudb_delete_primary_metadata(TAUDB_PRIMARY_METADATA* metadata, int count);
extern void taudb_delete_counters(TAUDB_COUNTER* counters, int count);
extern void taudb_delete_counter_groups(TAUDB_COUNTER_GROUP* groups, int count);
extern void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values, int count);
extern void taudb_delete_timers(TAUDB_TIMER* timers, int count);
extern void taudb_delete_timer_parameters(TAUDB_TIMER_PARAMETER* timer_parameters, int count);
extern void taudb_delete_timer_groups(TAUDB_TIMER_GROUP* timer_groups, int count);
extern void taudb_delete_timer_callpath(TAUDB_TIMER_CALLPATH* timer_callpath, int count);
extern void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values, int count);

#endif // TAUDB_API_H
