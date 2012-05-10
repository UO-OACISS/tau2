#ifndef TAUDB_STRUCTS_H
#define TAUDB_STRUCTS_H 1

#include "time.h"
#include "uthash.h"

#ifndef boolean
#define TRUE  1
#define FALSE 0
typedef int boolean;
#endif

enum taudb_database_schema_version {
  TAUDB_2005_SCHEMA,
  TAUDB_2012_SCHEMA
};

typedef struct taudb_configuration {
  char* jdbc_db_type;
  char* db_hostname;
  char* db_portnum;
  char* db_dbname;
  char* db_schemaprefix;
  char* db_username;
  char* db_password;
  char* db_schemafile;
} TAUDB_CONFIGURATION;

typedef struct taudb_data_source {
 int id;
 char* name;
 char*description;
} TAUDB_DATA_SOURCE;

typedef struct taudb_timer_value {
 int id; // link back to database
 int timer; // link back to database
 int metric; // link back to database
 int thread; // link back to database, roundabout way
 char *key; // hash table key
 double inclusive;
 double exclusive;
 double inclusive_percentage;
 double exclusive_percentage;
 double sum_exclusive_squared;
 char* timestamp;
 UT_hash_handle hh;
} TAUDB_TIMER_VALUE;

typedef struct taudb_timer_callpath {
 int id; // link back to database
 int timer; // link back to database
 int thread; // link back to database, roundabout way
 char *key; // hash table key
 int calls;
 int subroutines;
 char *parent_key; // hash table key
 char* timestamp;
 UT_hash_handle hh;
} TAUDB_TIMER_CALLPATH;

/* timers groups are the groups such as tau_default,
   mpi, openmp, tau_phase, tau_callpath, tau_param, etc. 
   this mapping table allows for nxn mappings between timers
   and groups */

typedef struct taudb_timer_group {
 int id;
 int timer;
 char* name;
} TAUDB_TIMER_GROUP;

/* timer parameters are parameter based profile values. 
   an example is foo (x,y) where x=4 and y=10. in that example,
   timer would be the index of the timer with the
   name 'foo (x,y) <x>=<4> <y>=<10>'. this table would have two
   entries, one for the x value and one for the y value.
*/

typedef struct taudb_timer_parameter {
 int id;
 char* name;
 char* value;
} TAUDB_TIMER_PARAMETER;

/* timers are interval timers, capturing some interval value.  for callpath or
   phase profiles, the parent refers to the calling function or phase. */

typedef struct taudb_timer {
 int id;
 int trial;
 char* name;
 char* short_name;
 char* source_file;
 int line_number;
 int line_number_end;
 int column_number;
 int column_number_end;
 int child_count;
 int group_count;
 int parameter_count;
 struct taudb_timer* children; // self-referencing pointer
 struct taudb_timer* parent; // self-referencing pointer, only one parent
 TAUDB_TIMER_GROUP* groups;
 TAUDB_TIMER_PARAMETER* parameters;
 TAUDB_TIMER_VALUE* values;
} TAUDB_TIMER;

/* counters are atomic counters, not just interval timers */

typedef struct taudb_counter_value {
 int id;
 int sample_count;
 double maximum_value;
 double minimum_value;
 double mean_value;
 double standard_deviation;
} TAUDB_COUNTER_VALUE;

/* counter groups are the groups of counters. This table
   allows for NxN mappings of counters to groups. */

typedef struct taudb_counter_group {
 int id;
 int counter;
 char*  name;
} TAUDB_COUNTER_GROUP;

/* counters measure some counted value. */

typedef struct taudb_counter {
 int id;
 int trial;
 char* name;
 char* source_file;
 int line_number;
 int group_count;
 int value_count;
 TAUDB_TIMER* parent;
 TAUDB_COUNTER_GROUP* groups;
 TAUDB_COUNTER_VALUE* values;
} TAUDB_COUNTER;

static int TAUDB_MEAN_WITHOUT_NULLS = -1;
static int TAUDB_TOTAL = -2;
static int TAUDB_STDDEV_WITHOUT_NULLS = -3;
static int TAUDB_MIN = -4;
static int TAUDB_MAX = -5;
static int TAUDB_MEAN_WITH_NULLS = -6;
static int TAUDB_STDDEV_WITH_NULLS = -7;

/* primary metadata is metadata that is not nested, does not
   contain unique data for each thread. */

typedef struct taudb_primary_metadata {
 char* name;
 char* value;
} TAUDB_PRIMARY_METADATA;

/* primary metadata is metadata that could be nested, could
   contain unique data for each thread, and could be an array. */

typedef struct taudb_secondary_metadata {
 int id; // link back to database
 int trial; // link back to database
 int timer; // link back to database
 int thread; // link back to database
 char* name;
 char** value;
 int num_values;
 int child_count;
 struct taudb_secondary_metadata* children; // self-referencing 
} TAUDB_SECONDARY_METADATA;

typedef struct taudb_thread {
 int id;
 int trial;
 int node_rank;
 int context_rank;
 int thread_rank;
 int index;
 int secondary_metadata_count;
 TAUDB_SECONDARY_METADATA* secondary_metadata;
} TAUDB_THREAD;

/* metrics are things like num_calls, num_subroutines, TIME, PAPI
   counters, and derived metrics. */

typedef struct taudb_metric {
 int id;
 int trial;
 char* name;
 boolean derived;
} TAUDB_METRIC;

/* trials are the top level structure */

typedef struct taudb_trial {
 int id;
 char* name;
 char* collection_date;
 int data_source;
 int node_count;
 int contexts_per_node;
 int threads_per_context;
 int metric_count;
 int thread_count;
 int timer_count;
 int callpath_count;
 int value_count;
 int callpath_stat_count;
 int value_stat_count;
 int counter_count;
 int primary_metadata_count;
 int secondary_metadata_count;
 TAUDB_METRIC* metrics;
 TAUDB_THREAD* threads;
 TAUDB_TIMER* timers;
 TAUDB_COUNTER* counters;
 TAUDB_TIMER_CALLPATH* timer_callpaths;
 TAUDB_TIMER_VALUE* timer_values;
 TAUDB_TIMER_CALLPATH* timer_callpath_stats;
 TAUDB_TIMER_VALUE* timer_value_stats;
 TAUDB_COUNTER_VALUE* counter_values;
 TAUDB_PRIMARY_METADATA* primary_metadata;
 TAUDB_SECONDARY_METADATA* secondary_metadata;
} TAUDB_TRIAL;

typedef struct perfdmf_experiment {
 int id;
 char* name;
 int primary_metadata_count;
 TAUDB_PRIMARY_METADATA* primary_metadata;
} PERFDMF_EXPERIMENT;

typedef struct perfdmf_application {
 int id;
 char* name;
 int primary_metadata_count;
 TAUDB_PRIMARY_METADATA* primary_metadata;
} PERFDMF_APPLICATION;

#endif // TAUDB_STRUCTS_H
