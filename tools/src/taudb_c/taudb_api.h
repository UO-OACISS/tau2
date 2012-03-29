#ifndef TAUDB_API_H
#define TAUDB_API_H 1

#include "libpq-fe.h"
#include "taudb_structs.h"

// when a "get" function is called, this global has the number of 
// top-level objects that are returned. 
extern int taudb_numItems;

// the database connection
extern PGconn* _taudb_connection;

// the database version
extern enum taudb_database_schema_version taudb_version;

// to connect to the database
extern int taudb_connect(char* host, char* port, char* database,
                         char* login, char* password);

// test the connection status
extern int taudb_check_connection(void);

// disconnect from the database
extern int taudb_disconnect(void);

// when there is an error, disconnect and quit.
extern void taudb_exit_nicely(void);

// functions to support the old database schema
#ifdef TAUDB_PERFDMF
extern PERFDMF_APPLICATION* taudb_get_applications();
extern PERFDMF_EXPERIMENT*  taudb_get_experiments(PERFDMF_APPLICATION* application);
extern PERFDMF_APPLICATION* taudb_get_application(char* name);
extern PERFDMF_EXPERIMENT*  taudb_get_experiment(PERFDMF_APPLICATION* application, char* name);
extern TAUDB_TRIAL*         taudb_get_trials(PERFDMF_EXPERIMENT* experiment);
#endif

#ifdef TAUDB_PERFDMF
extern PERFDMF_APPLICATION*      taudb_create_applications(int count);
extern PERFDMF_EXPERIMENT*       taudb_create_experiments(int count);
#endif
extern TAUDB_TRIAL*              taudb_create_trials(int count);
extern TAUDB_METRIC*             taudb_create_metrics(int count);
extern TAUDB_THREAD*             taudb_create_threads(int count);
extern TAUDB_SECONDARY_METADATA* taudb_create_secondary_metadata(int count);
extern TAUDB_PRIMARY_METADATA*   taudb_create_primary_metadata(int count);
extern TAUDB_COUNTER*            taudb_create_counters(int count);
extern TAUDB_COUNTER_GROUP*      taudb_create_counter_groups(int count);
extern TAUDB_COUNTER_GROUP*      taudb_resize_counter_groups(int count, TAUDB_COUNTER_GROUP* old_groups);
extern TAUDB_COUNTER_VALUE*      taudb_create_counter_values(int count);
extern TAUDB_TIMER*              taudb_create_timers(int count);
extern TAUDB_TIMER_PARAMETER*    taudb_create_timer_parameters(int count);
extern TAUDB_TIMER_GROUP*        taudb_create_timer_groups(int count);
extern TAUDB_TIMER_GROUP*        taudb_resize_timer_groups(int count, TAUDB_TIMER_GROUP* old_groups);
extern TAUDB_TIMER_VALUE*        taudb_create_timer_values(int count);

#ifdef TAUDB_PERFDMF
extern void taudb_delete_applications(PERFDMF_APPLICATION* applications, int count);
extern void taudb_delete_experiments(PERFDMF_EXPERIMENT* experiments, int count);
#endif
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
extern void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values, int count);

// using the properties set in the filter, find a set of trials
extern TAUDB_TRIAL* taudb_find_trials(boolean complete, TAUDB_TRIAL* filter);
// get the threads for a trial
extern TAUDB_THREAD* taudb_get_threads(TAUDB_TRIAL* trial);
// get the metrics for a trial
extern TAUDB_METRIC* taudb_get_metrics(TAUDB_TRIAL* trial);
// get the timers for a trial
extern TAUDB_TIMER* taudb_get_timers(TAUDB_TRIAL* trial);
// get the counters for a trial
extern TAUDB_COUNTER* taudb_get_counters(TAUDB_TRIAL* trial);
// get the timer values for a trial
extern TAUDB_TIMER_VALUE* taudb_get_timer_values(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread, TAUDB_METRIC* metric);
extern TAUDB_TIMER* taudb_get_main_timer(TAUDB_TRIAL* trial);

#endif // TAUDB_API_H
