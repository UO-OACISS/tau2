#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void perfdmf_delete_applications(PERFDMF_APPLICATION* applications, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_delete_applications(%p,%d)\n", applications, count);
#endif
  if (count == 0 || applications == NULL) return;

  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_primary_metadata(applications[i].primary_metadata, applications[i].primary_metadata_count);
  }
  free(applications);
}

void perfdmf_delete_experiments(PERFDMF_EXPERIMENT* experiments, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_delete_experiments(%p,%d)\n", experiments, count);
#endif
  if (count == 0 || experiments == NULL) return;

  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_primary_metadata(experiments[i].primary_metadata, experiments[i].primary_metadata_count);
  }
  free(experiments);
}

void taudb_delete_trials(TAUDB_TRIAL* trials, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_trials(%p,%d)\n", trials, count);
#endif
  if (count == 0 || trials == NULL) return;

  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(trials[i].secondary_metadata, trials[i].secondary_metadata_count);
    taudb_delete_primary_metadata(trials[i].primary_metadata, trials[i].primary_metadata_count);
    taudb_delete_counter_values(trials[i].counter_values, trials[i].counter_value_count);
    taudb_delete_counters(trials[i].counters, trials[i].counter_count);
    taudb_delete_timer_call_data(trials[i].timer_call_data, trials[i].timer_call_data_count);
    taudb_delete_timer_callpaths(trials[i].timer_callpaths, trials[i].timer_callpath_count);
    taudb_delete_timer_groups(trials[i].timer_groups, trials[i].timer_group_count);
    taudb_delete_timers(trials[i].timers, trials[i].timer_count);
    taudb_delete_threads(trials[i].threads, trials[i].thread_count);
    taudb_delete_metrics(trials[i].metrics, trials[i].metric_count);
    free(trials[i].name);
    free(trials[i].collection_date);
  }
  free(trials);
}

void taudb_delete_metrics(TAUDB_METRIC* metrics, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_metrics(%p,%d)\n", metrics, count);
#endif
  if (count == 0 || metrics == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(metrics[i].name);
  }
  free(metrics);
}

void taudb_delete_threads(TAUDB_THREAD* threads, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_threads(%p,%d)\n", threads, count);
#endif
  if (count == 0 || threads == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(threads[i].secondary_metadata, threads[i].secondary_metadata_count);
  }
  free(threads);
}

void taudb_delete_secondary_metadata(TAUDB_SECONDARY_METADATA* metadata, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_secondary_metadata(%p,%d)\n", metadata, count);
#endif
  if (count == 0 || metadata == NULL) return;
  int i,j;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(metadata[i].children, metadata[i].child_count);
    free(metadata[i].name);
	for (j = metadata[i].num_values ; j >= 0 ; j--) {
      free(metadata[i].value[j]);
	}
  }
  free(metadata);
}

void taudb_delete_primary_metadata(TAUDB_PRIMARY_METADATA* metadata, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_primary_metadata(%p,%d)\n", metadata, count);
#endif
  if (count == 0 || metadata == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(metadata[i].name);
    free(metadata[i].value);
  }
  free(metadata);
}

void taudb_delete_counters(TAUDB_COUNTER* counters, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counters(%p,%d)\n", counters, count);
#endif
  if (count == 0 || counters == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(counters[i].name);
  }
  free(counters);
}

void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counter_values(%p,%d)\n", counter_values, count);
#endif
  if (count == 0 || counter_values == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(counter_values[i].timestamp);
  }
  free(counter_values);
}

void taudb_delete_timers(TAUDB_TIMER* timers, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timers(%p,%d)\n", timers, count);
#endif
  if (count == 0 || timers == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_timer_parameters(timers[i].parameters, timers[i].parameter_count);
    taudb_delete_timer_groups(timers[i].groups, timers[i].group_count);
    free(timers[i].name);
    free(timers[i].short_name);
    free(timers[i].source_file);
  }
  free(timers);
}

void taudb_delete_timer_parameters(TAUDB_TIMER_PARAMETER* timer_parameters, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_parameters(%p,%d)\n", timer_parameters, count);
#endif
  if (count == 0 || timer_parameters == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(timer_parameters[i].name);
    free(timer_parameters[i].value);
  }
  free(timer_parameters);
}

void taudb_delete_timer_groups(TAUDB_TIMER_GROUP* timer_groups, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_groups(%p,%d)\n", timer_groups, count);
#endif
  if (count == 0 || timer_groups == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    // the list of timers in this group SHOULD get deleted when the trial
	// deletes its array of timers...
    free(timer_groups[i].timers);
    free(timer_groups[i].name);
  }
  free(timer_groups);
}

void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_values(%p,%d)\n", timer_values, count);
#endif
  if (count == 0 || timer_values == NULL) return;
  free(timer_values);
}

void taudb_delete_timer_callpaths(TAUDB_TIMER_CALLPATH* timer_callpaths, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_callpaths(%p,%d)\n", timer_callpaths, count);
#endif
  if (count == 0 || timer_callpaths == NULL) return;
  free(timer_callpaths);
}

void taudb_delete_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_call_data(%p,%d)\n", timer_call_data, count);
#endif
  if (count == 0 || timer_call_data == NULL) return;
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(timer_call_data[i].timestamp);
	taudb_delete_timer_values(timer_call_data[i].timer_values, timer_call_data[i].timer_value_count);
  }
  free(timer_call_data);
}

