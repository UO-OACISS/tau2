#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void taudb_delete_trials(TAUDB_TRIAL* trials, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(trials[i].secondary_metadata, trials[i].secondary_metadata_count);
    taudb_delete_primary_metadata(trials[i].primary_metadata, trials[i].primary_metadata_count);
	int value_count = trials[i].thread_count * trials[i].counter_count;
    taudb_delete_counter_values(trials[i].counter_values, value_count);
	value_count = trials[i].thread_count * trials[i].metric_count * trials[i].timer_count;
    taudb_delete_counters(trials[i].counters, trials[i].counter_count);
    taudb_delete_timer_values(trials[i].timer_values, value_count);
    taudb_delete_timers(trials[i].timers, trials[i].timer_count);
    taudb_delete_metrics(trials[i].metrics, trials[i].metric_count);
    taudb_delete_threads(trials[i].threads, trials[i].thread_count);
    free(&(trials[i]));
  }
}

void taudb_delete_metrics(TAUDB_METRIC* metrics, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(metrics[i]));
  }
}

void taudb_delete_threads(TAUDB_THREAD* threads, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(threads[i].secondary_metadata, threads[i].secondary_metadata_count);
    free(&(threads[i]));
  }
}

void taudb_delete_secondary_metadata(TAUDB_SECONDARY_METADATA* metadata, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_secondary_metadata(metadata[i].children, metadata[i].child_count);
    free(&(metadata[i]));
  }
}

void taudb_delete_primary_metadata(TAUDB_PRIMARY_METADATA* metadata, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(metadata[i]));
  }
}

void taudb_delete_counters(TAUDB_COUNTER* counters, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_counter_groups(counters[i].groups, counters[i].group_count);
    taudb_delete_counter_values(counters[i].values, counters[i].value_count);
    free(&(counters[i]));
  }
}

void taudb_delete_counter_groups(TAUDB_COUNTER_GROUP* counter_groups, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(counter_groups[i]));
  }
}

void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(counter_values[i]));
  }
}

void taudb_delete_timers(TAUDB_TIMER* timers, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    taudb_delete_timer_parameters(timers[i].parameters, timers[i].parameter_count);
    taudb_delete_timer_groups(timers[i].groups, timers[i].group_count);
    //taudb_delete_timer_children(timer_groups[i].children, timer_groups[i].child_count);
    free(&(timers[i]));
  }
}

void taudb_delete_timer_parameters(TAUDB_TIMER_PARAMETER* timer_parameters, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(timer_parameters[i]));
  }
}

void taudb_delete_timer_groups(TAUDB_TIMER_GROUP* timer_groups, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(timer_groups[i]));
  }
}

void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values, int count) {
  int i = 0;
  for (i = count-1 ; i >= 0 ; i++) {
    free(&(timer_values[i]));
  }
}

