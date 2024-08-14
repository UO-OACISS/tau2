#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <uuid/uuid.h>

PERFDMF_APPLICATION* perfdmf_create_applications(int count){ 
 PERFDMF_APPLICATION* applications = (PERFDMF_APPLICATION*) (calloc (count, sizeof (PERFDMF_APPLICATION)));
 return applications;
}

PERFDMF_EXPERIMENT* perfdmf_create_experiments(int count){ 
 PERFDMF_EXPERIMENT* experiments = (PERFDMF_EXPERIMENT*) (calloc (count, sizeof (PERFDMF_EXPERIMENT)));
 return experiments;
}

TAUDB_CONFIGURATION* taudb_create_configuration(){ 
 TAUDB_CONFIGURATION* config = (TAUDB_CONFIGURATION*) (calloc (1, sizeof (TAUDB_CONFIGURATION)));
 return config;
}

TAUDB_CONNECTION* taudb_create_connection(){ 
 TAUDB_CONNECTION* config = (TAUDB_CONNECTION*) (calloc (1, sizeof (TAUDB_CONNECTION)));
 return config;
}

TAUDB_DATA_SOURCE* taudb_create_data_sources(int count){ 
 TAUDB_DATA_SOURCE* data_sources = (TAUDB_DATA_SOURCE*) (calloc (count, sizeof (TAUDB_DATA_SOURCE)));
 return data_sources;
}

TAUDB_TRIAL* taudb_create_trials(int count){ 
 TAUDB_TRIAL* trials = (TAUDB_TRIAL*) (calloc (count, sizeof (TAUDB_TRIAL)));
 return trials;
}

TAUDB_METRIC* taudb_create_metrics(int count){ 
 TAUDB_METRIC* metrics = (TAUDB_METRIC*) (calloc (count, sizeof (TAUDB_METRIC)));
 return metrics;
}

TAUDB_TIME_RANGE* taudb_create_time_ranges(int count){ 
 TAUDB_TIME_RANGE* time_ranges = (TAUDB_TIME_RANGE*) (calloc (count, sizeof (TAUDB_TIME_RANGE)));
 return time_ranges;
}

TAUDB_THREAD* taudb_create_threads(int count){ 
 TAUDB_THREAD* threads = (TAUDB_THREAD*) (calloc (count, sizeof (TAUDB_THREAD)));
 return threads;
}

TAUDB_SECONDARY_METADATA* taudb_create_secondary_metadata(int count){ 
 TAUDB_SECONDARY_METADATA* metadata = (TAUDB_SECONDARY_METADATA*) (calloc (count, sizeof (TAUDB_SECONDARY_METADATA)));
 metadata->child_count = 0;
 metadata->num_values = 0;
 uuid_t uuid;
 uuid_generate(uuid);
#ifdef __X86_64_LINUX__
 char out[37]; // 36 bytes, plus null terminator
 uuid_unparse_upper(uuid, out);
 metadata->id = taudb_strdup(out);
#else
 uuid_string_t out;
 uuid_unparse_upper(uuid, out);
 metadata->id = taudb_strdup((char*)out);
#endif
 return metadata;
}

TAUDB_PRIMARY_METADATA* taudb_create_primary_metadata(int count){ 
 TAUDB_PRIMARY_METADATA* metadata = (TAUDB_PRIMARY_METADATA*) (calloc (count, sizeof (TAUDB_PRIMARY_METADATA)));
 return metadata;
}

TAUDB_PRIMARY_METADATA* taudb_resize_primary_metadata(int count, TAUDB_PRIMARY_METADATA* old_primary_metadata){ 
 TAUDB_PRIMARY_METADATA* primary_metadata = (TAUDB_PRIMARY_METADATA*) (realloc (old_primary_metadata, count * sizeof (TAUDB_PRIMARY_METADATA)));
 return primary_metadata;
}

TAUDB_COUNTER* taudb_create_counters(int count){ 
 TAUDB_COUNTER* counters = (TAUDB_COUNTER*) (calloc (count, sizeof (TAUDB_COUNTER)));
 return counters;
}

TAUDB_COUNTER_VALUE* taudb_create_counter_values(int count){ 
 TAUDB_COUNTER_VALUE* counter_values = (TAUDB_COUNTER_VALUE*) (calloc (count, sizeof (TAUDB_COUNTER_VALUE)));
 return counter_values;
}

TAUDB_TIMER* taudb_create_timers(int count){ 
 TAUDB_TIMER* timers = (TAUDB_TIMER*) (calloc (count, sizeof (TAUDB_TIMER)));
 return timers;
}

TAUDB_TIMER_PARAMETER* taudb_create_timer_parameters(int count){ 
 TAUDB_TIMER_PARAMETER* timer_parameters = (TAUDB_TIMER_PARAMETER*) (calloc (count, sizeof (TAUDB_TIMER_PARAMETER)));
 return timer_parameters;
}

TAUDB_TIMER_GROUP* taudb_create_timer_groups(int count){ 
 TAUDB_TIMER_GROUP* timer_groups = (TAUDB_TIMER_GROUP*) (calloc (count, sizeof (TAUDB_TIMER_GROUP)));
 return timer_groups;
}

TAUDB_TIMER_GROUP* taudb_resize_timer_groups(int count, TAUDB_TIMER_GROUP* old_groups){ 
 TAUDB_TIMER_GROUP* timer_groups = (TAUDB_TIMER_GROUP*) (realloc ((void*)(old_groups), count * (sizeof (TAUDB_TIMER_GROUP))));
 return timer_groups;
}

TAUDB_TIMER_VALUE* taudb_create_timer_values(int count){ 
 TAUDB_TIMER_VALUE* timer_values = (TAUDB_TIMER_VALUE*) (calloc (count, sizeof (TAUDB_TIMER_VALUE)));
 return timer_values;
}

TAUDB_TIMER_CALLPATH* taudb_create_timer_callpaths(int count){ 
 TAUDB_TIMER_CALLPATH* timer_callpaths = (TAUDB_TIMER_CALLPATH*) (calloc (count, sizeof (TAUDB_TIMER_CALLPATH)));
 return timer_callpaths;
}

TAUDB_TIMER_CALL_DATA* taudb_create_timer_call_data(int count){ 
 TAUDB_TIMER_CALL_DATA* timer_call_data = (TAUDB_TIMER_CALL_DATA*) (calloc (count, sizeof (TAUDB_TIMER_CALL_DATA)));
 return timer_call_data;
}

char* taudb_strdup(const char* in_string) {
  // add one more character for the null terminator
  int length = strlen(in_string) + 1;
  char* new_string = (char*)calloc(length, sizeof(char));
  strncpy(new_string,  in_string, length); 
  return new_string;
}

char* taudb_create_hash_key_2(int thread, const char* timer) {
  char str_thread[15];
  snprintf(str_thread, sizeof(str_thread),  "%d", thread);
  // add colon character and null terminator
  int length = strlen(str_thread) + strlen(timer) + 2;
  char* key = (char*)calloc(length, sizeof(char));
  snprintf(key, length,  "%d:%s", thread, timer);
  return key;
}

char* taudb_create_hash_key_3(int thread, const char* timer, const char* metric) {
  char str_thread[15];
  snprintf(str_thread, sizeof(str_thread),  "%d", thread);
  // add colon characters and null terminator
  int length = strlen(str_thread) + strlen(timer) + strlen(metric) + 3;
  char* key = (char*)calloc(length, sizeof(char));
  snprintf(key, length,  "%d:%s:%s", thread, timer, metric);
  return key;
}


