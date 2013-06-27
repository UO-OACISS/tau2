#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include "dump_functions.h"

int main (int argc, char** argv) {
  printf("Looking for profile files in: %s\n", argv[1]);
  // get the config files in the home directory
  TAUDB_TRIAL* trial = taudb_parse_tau_profiles(argv[1]);
   
  int count = 0;
  TAUDB_METRIC* metrics = trial->metrics_by_name;
  // iterate over the hash
  TAUDB_METRIC* metric;
  for (metric = metrics ; metric != NULL ; 
       metric = taudb_next_metric_by_name_from_trial(metric)) {
    printf("METRIC: %s\n", metric->name);
	count = count + 1;
  }
  printf("%d METRICs found\n", count);

  TAUDB_THREAD* threads = trial->threads;
  // iterate over the hash
  TAUDB_THREAD* thread;
  count = 0;
  for (thread = threads ; thread != NULL ; thread = taudb_next_thread_by_index_from_trial(thread)) {
    //printf("THREAD: %d %d %d %d\n", thread->node_rank, thread->context_rank, thread->thread_rank, thread->index);
	count = count + 1;
  }
  printf("%d THREADs found\n", count);

  printf("%d TIME_RANGEs found\n", HASH_CNT(hh, trial->time_ranges));

  TAUDB_TIMER* timers = trial->timers_by_name;
  // iterate over the hash
  TAUDB_TIMER* timer;
  count = 0;
  for (timer = timers ; timer != NULL ; timer = taudb_next_timer_by_name_from_trial(timer)) {
    //printf("TIMER: %s\n'%s', '%s', %d, %d, %d, %d\n", timer->name, timer->short_name, timer->source_file, timer->line_number, timer->line_number_end, timer->column_number, timer->column_number_end);
	count = count + 1;
  }
  printf("%d TIMERs found\n", count);

  printf("%d TIMER_GROUPs found\n", HASH_CNT(trial_hash_by_name, trial->timer_groups));

  TAUDB_TIMER_CALLPATH* timer_callpaths = trial->timer_callpaths_by_name;
  // iterate over the hash
  TAUDB_TIMER_CALLPATH* timer_callpath;
  count = 0;
  for (timer_callpath = timer_callpaths ; timer_callpath != NULL ; timer_callpath = taudb_next_timer_callpath_by_name_from_trial(timer_callpath)) {
    //printf("TIMER_CALLPATH: '%s'\n", timer_callpath->name);
	count = count + 1;
  }
  printf("%d TIMER_CALLPATHs found\n", count);

  TAUDB_TIMER_CALL_DATA* timer_call_datas = trial->timer_call_data_by_key;
  // iterate over the hash
  TAUDB_TIMER_CALL_DATA* timer_call_data;
  count = 0;
  int count2 = 0;
  for (timer_call_data = timer_call_datas ; timer_call_data != NULL ; timer_call_data = taudb_next_timer_call_data_by_key_from_trial(timer_call_data)) {
	count = count + 1;
    TAUDB_TIMER_VALUE* timer_values = timer_call_data->timer_values;
    // iterate over the hash
    TAUDB_TIMER_VALUE* timer_value;
    for (timer_value = timer_values ; timer_value != NULL ; timer_value = taudb_next_timer_value_by_metric_from_timer_call_data(timer_value)) {
	  count2 = count2 + 1;
    }
  }
  printf("%d TIMER_CALL_DATAs found\n", count);

  printf("%d TIMER_VALUEs found\n", count2);

  printf("%d COUNTERs found\n", HASH_CNT(hh2, trial->counters_by_name));
  printf("%d COUNTER_VALUEs found\n", HASH_CNT(hh1, trial->counter_values));
  printf("%d PRIMARY_METADATAs found\n", HASH_CNT(hh, trial->primary_metadata));
  printf("%d SECONDARY_METADATAs found\n", HASH_CNT(hh2, trial->secondary_metadata_by_key));

  printf("Done.\n");
  return 0;
}
