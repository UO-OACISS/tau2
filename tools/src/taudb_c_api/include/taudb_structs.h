#ifndef TAUDB_STRUCTS_H
#define TAUDB_STRUCTS_H 1

#include "time.h"
#include "uthash.h"

#ifndef boolean
#define TRUE  1
#define FALSE 0
typedef int boolean;
#endif

/* forward declarations to ease objects that need to know about each other 
 * and have doubly-linked relationships */

struct taudb_timer_call_data;
struct taudb_timer_value;
struct taudb_timer_callpath;
struct taudb_timer_group;
struct taudb_timer_parameter;
struct taudb_timer;
struct taudb_counter_value;
struct taudb_counter;
struct taudb_primary_metadata;
struct taudb_secondary_metadata;
struct taudb_thread;
struct taudb_metric;
struct taudb_trial;
struct perfdmf_experiment;
struct perfdmf_application;

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

enum taudb_database_schema_version {
  TAUDB_2005_SCHEMA,
  TAUDB_2012_SCHEMA
};

typedef struct taudb_data_source {
 int id;
 char* name;
 char*description;
} TAUDB_DATA_SOURCE;

/* these are the derived thread indexes. */

static int TAUDB_MEAN_WITHOUT_NULLS = -1;
static int TAUDB_TOTAL = -2;
static int TAUDB_STDDEV_WITHOUT_NULLS = -3;
static int TAUDB_MIN = -4;
static int TAUDB_MAX = -5;
static int TAUDB_MEAN_WITH_NULLS = -6;
static int TAUDB_STDDEV_WITH_NULLS = -7;

/* trials are the top level structure */

typedef struct taudb_trial {
 // actual data from the database
 int id;
 char* name;
 char* collection_date;
 struct taudb_data_source* data_source;
 int node_count;
 int contexts_per_node;
 int threads_per_context;
 // array sizes
 int metric_count;
 int thread_count;
 int derived_thread_count;
 int timer_count;
 int timer_group_count;
 int timer_callpath_count;
 int timer_call_data_count;
 int counter_count;
 int counter_value_count;
 int primary_metadata_count;
 int secondary_metadata_count;
 // arrays of data for this trial
 struct taudb_metric* metrics;
 struct taudb_thread* threads;
 struct taudb_timer* timers;
 struct taudb_timer_group* timer_groups;
 struct taudb_timer_callpath* timer_callpaths;
 struct taudb_timer_call_data* timer_call_data;
 struct taudb_counter* counters;
 struct taudb_counter_value* counter_values;
 struct taudb_primary_metadata* primary_metadata;
 struct taudb_secondary_metadata* secondary_metadata;
} TAUDB_TRIAL;

/*********************************************/
/* data dimensions */
/*********************************************/

typedef struct taudb_thread {
 int id; // database id, also key to hash
 struct taudb_trial* trial;
 int node_rank;
 int context_rank;
 int thread_rank;
 int index;
 int secondary_metadata_count;
 struct taudb_secondary_metadata* secondary_metadata;
 UT_hash_handle hh;
} TAUDB_THREAD;

/* metrics are things like TIME, PAPI counters, and derived metrics. */

typedef struct taudb_metric {
 int id; // database value, also key to hash
 char* name;
 boolean derived;
 UT_hash_handle hh;
} TAUDB_METRIC;

/* timers are interval timers, capturing some interval value.  for callpath or
   phase profiles, the parent refers to the calling function or phase. */

typedef struct taudb_timer {
 int id; // database value, also key to hash
 struct taudb_trial* trial;
 char* name;
 char* short_name;
 char* source_file;
 int line_number;
 int line_number_end;
 int column_number;
 int column_number_end;
 int group_count;
 int parameter_count;
 struct taudb_timer_group* groups;
 struct taudb_timer_parameter* parameters;
 UT_hash_handle hh;
} TAUDB_TIMER;

/*********************************************/
/* timer related structures  */
/*********************************************/

/* timer groups are the groups such as tau_default,
   mpi, openmp, tau_phase, tau_callpath, tau_param, etc. 
   this mapping table allows for nxn mappings between timers
   and groups */

typedef struct taudb_timer_group {
 int id; // database reference, and hash key
 char* name;
 int timer_count;
 struct taudb_timer* timers;
 UT_hash_handle hh;
} TAUDB_TIMER_GROUP;

/* timer parameters are parameter based profile values. 
   an example is foo (x,y) where x=4 and y=10. in that example,
   timer would be the index of the timer with the
   name 'foo (x,y) <x>=<4> <y>=<10>'. this table would have two
   entries, one for the x value and one for the y value.
*/

typedef struct taudb_timer_parameter {
 int id; // database reference, and hash key
 char* name;
 char* value;
 UT_hash_handle hh;
} TAUDB_TIMER_PARAMETER;

/* callpath objects contain the merged dynamic callgraph tree seen
 * during execution */

typedef struct taudb_timer_callpath {
 int id; // link back to database, and hash key
 struct taudb_timer* timer; // which timer is this?
 struct taudb_timer_callpath *parent; // callgraph parent
 UT_hash_handle hh;
} TAUDB_TIMER_CALLPATH;

/* timer_call_data objects are observations of a node of the callgraph
   for one of the threads. */

typedef struct taudb_timer_call_data {
 int id; // link back to database
 struct taudb_timer_callpath *timer_callpath; // link back to database
 struct taudb_thread *thread; // link back to database, roundabout way
 char *key; // hash table key - thread:timer_string (all names)
 int calls;
 int subroutines;
 char* timestamp;
 int timer_value_count;
 struct taudb_timer_value* timer_values;
 UT_hash_handle hh;
} TAUDB_TIMER_CALL_DATA;

/* finally, timer_values are specific measurements during one of the
   observations of the node of the callgraph on a thread. */

typedef struct taudb_timer_value {
 struct taudb_metric* metric; 
 double inclusive;
 double exclusive;
 double inclusive_percentage;
 double exclusive_percentage;
 double sum_exclusive_squared;
 char *key; // hash table key - thread:timer_string:metric (all names)
 UT_hash_handle hh;
} TAUDB_TIMER_VALUE;

/*********************************************/
/* counter related structures  */
/*********************************************/

/* counters measure some counted value. */

typedef struct taudb_counter {
 int id; // database reference
 struct taudb_trial* trial;
 char* name;
} TAUDB_COUNTER;

/* counters are atomic counters, not just interval timers */

typedef struct taudb_counter_value {
 struct taudb_counter* counter; // the counter we are measuring
 struct taudb_thread* thread;   // where this measurement is
 struct taudb_timer_callpath* context; // the calling context (can be null)
 char* timestamp; // timestamp in case we are in a snapshot or something
 int sample_count;
 double maximum_value;
 double minimum_value;
 double mean_value;
 double standard_deviation;
} TAUDB_COUNTER_VALUE;

/*********************************************/
/* metadata related structures  */
/*********************************************/

/* primary metadata is metadata that is not nested, does not
   contain unique data for each thread. */

typedef struct taudb_primary_metadata {
 char* name;
 char* value;
 UT_hash_handle hh; // uses the name as the key
} TAUDB_PRIMARY_METADATA;

/* primary metadata is metadata that could be nested, could
   contain unique data for each thread, and could be an array. */

typedef struct taudb_secondary_metadata {
 int id; // link back to database
 struct taudb_timer_call_data* timer_call_data; 
 struct taudb_thread* thread; 
 struct taudb_secondary_metadata* parent; // self-referencing 
 int num_values; // can have arrays of data
 char* name;
 char** value;
 int child_count;
 struct taudb_secondary_metadata* children; // self-referencing 
 char* key;
 UT_hash_handle hh; // uses the key as a compound key
} TAUDB_SECONDARY_METADATA;

/* these are for supporting the older schema */

typedef struct perfdmf_experiment {
 int id;
 char* name;
 int primary_metadata_count;
 struct taudb_primary_metadata* primary_metadata;
} PERFDMF_EXPERIMENT;

typedef struct perfdmf_application {
 int id;
 char* name;
 int primary_metadata_count;
 struct taudb_primary_metadata* primary_metadata;
} PERFDMF_APPLICATION;

#endif // TAUDB_STRUCTS_H
