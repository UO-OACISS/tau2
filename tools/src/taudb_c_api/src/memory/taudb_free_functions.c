#include "taudb_internal.h"
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
    taudb_delete_primary_metadata(applications[i].primary_metadata);
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
    taudb_delete_primary_metadata(experiments[i].primary_metadata);
  }
  free(experiments);
}

void taudb_delete_trials(TAUDB_TRIAL* trials, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_trials(%p,%d)\n", trials, count);
#endif
  if (count == 0 || trials == NULL) return;

  // until we can do this properly, don't do it.
  return;

  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(trials[i].secondary_metadata);
    taudb_delete_primary_metadata(trials[i].primary_metadata);
    taudb_delete_counter_values(trials[i].counter_values);
    taudb_delete_counters(trials[i].counters_by_id);
    taudb_delete_timer_call_data(trials[i].timer_call_data_by_id);
    taudb_delete_timer_callpaths(trials[i].timer_callpaths_by_id);
    taudb_delete_timer_groups(trials[i].timer_groups);
    taudb_delete_timers(trials[i].timers_by_id);
    taudb_delete_threads(trials[i].threads);
    taudb_delete_metrics(trials[i].metrics_by_id);
    free(trials[i].name);
  }
  free(trials);
}

void taudb_delete_metrics(TAUDB_METRIC* metrics) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_metrics(%p)\n", metrics);
#endif
  TAUDB_METRIC *current, *tmp;
  HASH_ITER(hh1, metrics, current, tmp) {
    HASH_DELETE(hh1, metrics, current);
    free(current->name);
    free(current);
  }
}

void taudb_delete_data_sources(TAUDB_DATA_SOURCE* data_sources) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_data_sources(%p)\n", data_sources);
#endif
  if (data_sources == NULL) return;
  TAUDB_DATA_SOURCE *current, *tmp;
  HASH_ITER(hh1, data_sources, current, tmp) {
    HASH_DELETE(hh1, data_sources, current);
    free(current->name);
    free(current->description);
    free(current);
  }
}

void taudb_delete_threads(TAUDB_THREAD* threads) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_threads(%p)\n", threads);
#endif
  if (threads == NULL) return;
  TAUDB_THREAD *current, *tmp;
  HASH_ITER(hh, threads, current, tmp) {
    HASH_DELETE(hh, threads, current);
    free(current);
  }
}

void taudb_delete_secondary_metadata(TAUDB_SECONDARY_METADATA* metadata) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_secondary_metadata(%p)\n", metadata);
#endif
  if (metadata == NULL) return;
  int j;
  TAUDB_SECONDARY_METADATA *current, *tmp;
  HASH_ITER(hh, metadata, current, tmp) {
    HASH_DELETE(hh, metadata, current);
    taudb_delete_secondary_metadata(current->children);
    free(current->key.name);
	for (j = current->num_values ; j >= 0 ; j--) {
      free(current->value[j]);
	}
    free(current->id);
    free(current);
  }
}

void taudb_delete_primary_metadata(TAUDB_PRIMARY_METADATA* metadata) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_primary_metadata(%p)\n", metadata);
#endif
  if (metadata == NULL) return;
  TAUDB_PRIMARY_METADATA *current, *tmp;
  HASH_ITER(hh, metadata, current, tmp) {
    HASH_DELETE(hh, metadata, current);
    free(current->name);
    free(current->value);
    free(current);
  }
}

void taudb_delete_counters(TAUDB_COUNTER* counters) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counters(%p)\n", counters);
#endif
  if (counters == NULL) return;
  TAUDB_COUNTER *current, *tmp;
  HASH_ITER(hh1, counters, current, tmp) {
    HASH_DELETE(hh1, counters, current);
    free(current->name);
    free(current);
  }
}

void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counter_values(%p)\n", counter_values);
#endif
  if (counter_values == NULL) return;
  TAUDB_COUNTER_VALUE *current, *tmp;
  HASH_ITER(hh1, counter_values, current, tmp) {
    HASH_DELETE(hh1, counter_values, current);
    free(current->key.timestamp);
    free(current);
  }
}

void taudb_delete_timers(TAUDB_TIMER* timers) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timers(%p)\n", timers);
#endif
  if (timers == NULL) return;
  TAUDB_TIMER *current, *tmp;
  HASH_ITER(trial_hash_by_id, timers, current, tmp) {
    HASH_DELETE(trial_hash_by_id, timers, current);
    taudb_delete_timer_parameters(current->parameters);
    free(current->groups); // these will be deleted by the trial, later
    free(current->name);
    free(current->short_name);
    free(current->source_file);
    free(current);
  }
}

void taudb_delete_timer_parameters(TAUDB_TIMER_PARAMETER* timer_parameters) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_parameters(%p)\n", timer_parameters);
#endif
  if (timer_parameters == NULL) return;
  TAUDB_TIMER_PARAMETER *current, *tmp;
  HASH_ITER(hh, timer_parameters, current, tmp) {
    HASH_DELETE(hh, timer_parameters, current);
    free(current->name);
    free(current->value);
    free(current);
  }
}

void taudb_delete_timer_groups(TAUDB_TIMER_GROUP* timer_groups) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_groups(%p)\n", timer_groups);
#endif
  if (timer_groups == NULL) return;
  TAUDB_TIMER_GROUP *current, *tmp;
  HASH_ITER(trial_hash_by_name, timer_groups, current, tmp) {
    HASH_DELETE(trial_hash_by_name, timer_groups, current);
    free(current->name);
    free(current);
  }
}

void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_values(%p)\n", timer_values);
#endif
  if (timer_values == NULL) return;
  TAUDB_TIMER_VALUE *current, *tmp;
  HASH_ITER(hh, timer_values, current, tmp) {
    HASH_DELETE(hh, timer_values, current);
    free(current);
  }
}

void taudb_delete_timer_callpaths(TAUDB_TIMER_CALLPATH* timer_callpaths) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_callpaths(%p)\n", timer_callpaths);
#endif
  if (timer_callpaths == NULL) return;
  TAUDB_TIMER_CALLPATH *current, *tmp;
  HASH_ITER(hh1, timer_callpaths, current, tmp) {
    HASH_DELETE(hh1, timer_callpaths, current);
    free(current->name);
    free(current);
  }
}

void taudb_delete_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_call_data(%p)\n", timer_call_data);
#endif
  if (timer_call_data == NULL) return;
  TAUDB_TIMER_CALL_DATA *current, *tmp;
  HASH_ITER(hh1, timer_call_data, current, tmp) {
    HASH_DELETE(hh1, timer_call_data, current);
    taudb_delete_timer_values(current->timer_values);
    free(current->key.timestamp);
    free(current);
  }
}

