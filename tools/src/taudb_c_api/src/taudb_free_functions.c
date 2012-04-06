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
  for (i = 0 ; i < count ; i++) {
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
  for (i = 0 ; i < count ; i++) {
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
  for (i = 0 ; i < count ; i++) {
    taudb_delete_secondary_metadata(trials[i].secondary_metadata, trials[i].secondary_metadata_count);
    taudb_delete_primary_metadata(trials[i].primary_metadata, trials[i].primary_metadata_count);
    taudb_delete_counter_values(trials[i].counter_values, trials[i].counter_value_count);
    taudb_delete_counters(trials[i].counters, trials[i].counter_count);
    taudb_delete_timer_values(trials[i].timer_values, trials[i].timer_value_count);
    taudb_delete_timers(trials[i].timers, trials[i].timer_count);
    taudb_delete_metrics(trials[i].metrics, trials[i].metric_count);
    taudb_delete_threads(trials[i].threads, trials[i].thread_count);
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
  for (i = 0 ; i < count ; i++) {
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
  for (i = 0 ; i < count ; i++) {
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
  for (i = 0 ; i < count ; i++) {
    taudb_delete_secondary_metadata(metadata[i].children, metadata[i].child_count);
    if (metadata[i].name != NULL) free(metadata[i].name);
    if (metadata[i].num_values > 0) {
		for (j = 0 ; j < metadata[i].num_values ; j++ ) {
			if (metadata[i].values[j] != NULL) free(metadata[i].values[j]);
		}
    	free(metadata[i].values);
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
  for (i = 0 ; i < count ; i++) {
    if (metadata[i].name != NULL) free(metadata[i].name);
    if (metadata[i].value != NULL) free(metadata[i].value);
  }
  free(metadata);
}

void taudb_delete_counters(TAUDB_COUNTER* counters, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counters(%p,%d)\n", counters, count);
#endif
  if (count == 0 || counters == NULL) return;
  int i = 0;
  for (i = 0 ; i < count ; i++) {
    taudb_delete_counter_groups(counters[i].groups, counters[i].group_count);
    taudb_delete_counter_values(counters[i].values, counters[i].value_count);
    free(counters[i].short_name);
    free(counters[i].full_name);
    free(counters[i].source_file);
  }
  free(counters);
}

void taudb_delete_counter_groups(TAUDB_COUNTER_GROUP* counter_groups, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counter_groups(%p,%d)\n", counter_groups, count);
#endif
  if (count == 0 || counter_groups == NULL) return;
  int i = 0;
  for (i = 0 ; i < count ; i++) {
    free(counter_groups[i].name);
  }
  free(counter_groups);
}

void taudb_delete_counter_values(TAUDB_COUNTER_VALUE* counter_values, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_counter_values(%p,%d)\n", counter_values, count);
#endif
  if (count == 0 || counter_values == NULL) return;
  free(counter_values);
}

void taudb_delete_timers(TAUDB_TIMER* timers, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timers(%p,%d)\n", timers, count);
#endif
  if (count == 0 || timers == NULL) return;
  int i = 0;
  for (i = 0 ; i < count ; i++) {
    taudb_delete_timer_parameters(timers[i].parameters, timers[i].parameter_count);
    taudb_delete_timer_groups(timers[i].groups, timers[i].group_count);
    //taudb_delete_timer_children(timer_groups[i].children, timer_groups[i].child_count);
    free(timers[i].short_name);
    free(timers[i].full_name);
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
  for (i = 0 ; i < count ; i++) {
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
  for (i = 0 ; i < count ; i++) {
    free(timer_groups[i].name);
  }
  free(timer_groups);
}

void taudb_delete_timer_values(TAUDB_TIMER_VALUE* timer_values, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_values(%p,%d)\n", timer_values, count);
#endif
  if (count == 0 || timer_values == NULL) return;
  // clean up the hash
  if (timer_values[0].key != NULL) {
    TAUDB_TIMER_VALUE *current, *tmp;
    HASH_ITER(hh, timer_values, current, tmp) {
      HASH_DEL(timer_values,current);  /* delete; users advances to next */
    }
  }
  // free the array of pointers
  free(timer_values);
}

void taudb_delete_timer_callpaths(TAUDB_TIMER_CALLPATH* timer_callpaths, int count) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_timer_callpaths(%p,%d)\n", timer_callpaths, count);
#endif
  if (count == 0 || timer_callpaths == NULL) return;
  int i = 0;
  for (i = 0 ; i < count ; i++) {
    if (timer_callpaths[i].parent_key != NULL) free(timer_callpaths[i].parent_key); 
    if (timer_callpaths[i].timestamp != NULL) free(timer_callpaths[i].timestamp); 
  }
  // clean up the hash
  if (timer_callpaths[0].key != NULL) {
  TAUDB_TIMER_CALLPATH *current, *tmp;
    HASH_ITER(hh, timer_callpaths, current, tmp) {
      HASH_DEL(timer_callpaths,current);  /* delete; users advances to next */
    }
  }
  free(timer_callpaths);
}

void taudb_delete_configuration(TAUDB_CONFIGURATION* config) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_configuration(%p)\n", config);
#endif
  if (config == NULL) return;
  free(config->jdbc_db_type);
  free(config->db_hostname);
  free(config->db_portnum);
  free(config->db_dbname);
  free(config->db_schemaprefix);
  free(config->db_username);
  free(config->db_password);
  free(config->db_schemafile);
  free(config);
}

void taudb_delete_data_source (TAUDB_DATA_SOURCE* data_source) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_delete_data_source(%p)\n", data_source);
#endif
  if (data_source == NULL) return;
  free(data_source->name);
  free(data_source->description);
  free(data_source);
}

