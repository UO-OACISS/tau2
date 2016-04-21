#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include "dump_functions.h"

int main (int argc, char** argv) {
  TAUDB_CONNECTION* connection = NULL;
  int items=0;
  int* taudb_numItems=&items;
  if (argc >= 2) {
    connection = taudb_connect_config(argv[1],taudb_numItems);
  } else {
    fprintf(stderr, "Please specify a TAUdb config file.\n");
    exit(1);
  }
  printf("Checking connection...\n");
  taudb_check_connection(connection);

  // create a trial
  TAUDB_TRIAL* trial = taudb_create_trials(1);
  trial->name = taudb_strdup("TEST TRIAL");
  // set the data source to "other"
  trial->data_source = taudb_get_data_source_by_id(taudb_query_data_sources(connection,taudb_numItems), 999);
  
  // create some metadata
  TAUDB_PRIMARY_METADATA* pm = taudb_create_primary_metadata(1);
  pm->name = taudb_strdup("Application");
  pm->value = taudb_strdup("Test Application");
  taudb_add_primary_metadata_to_trial(trial, pm);

  pm = taudb_create_primary_metadata(1);
  pm->name = taudb_strdup("Start Time");
  pm->value = taudb_strdup("2012-11-07 12:30:00");
  taudb_add_primary_metadata_to_trial(trial, pm);

  // alternatively, you can allocate the primary metadata in blocks
  pm = taudb_create_primary_metadata(10);
  pm[0].name = taudb_strdup("ClientID");
  pm[0].value = taudb_strdup("joe_user");
  taudb_add_primary_metadata_to_trial(trial, &(pm[0]));
  pm[1].name = taudb_strdup("hostname");
  pm[1].value = taudb_strdup("hopper04");
  taudb_add_primary_metadata_to_trial(trial, &(pm[1]));
  pm[2].name = taudb_strdup("Operating System");
  pm[2].value = taudb_strdup("Linux");
  taudb_add_primary_metadata_to_trial(trial, &(pm[2]));
  pm[3].name = taudb_strdup("Release");
  pm[3].value = taudb_strdup("2.6.32.36-0.5-default");
  taudb_add_primary_metadata_to_trial(trial, &(pm[3]));
  pm[4].name = taudb_strdup("Machine");
  pm[4].value = taudb_strdup("Hopper.nersc.gov");
  taudb_add_primary_metadata_to_trial(trial, &(pm[4]));
  pm[5].name = taudb_strdup("CPU Cache Size");
  pm[5].value = taudb_strdup("512 KB");
  taudb_add_primary_metadata_to_trial(trial, &(pm[5]));
  pm[6].name = taudb_strdup("CPU Clock Frequency");
  pm[6].value = taudb_strdup("800.000 MHz");
  taudb_add_primary_metadata_to_trial(trial, &(pm[6]));
  pm[7].name = taudb_strdup("CPU Model");
  pm[7].value = taudb_strdup("Quad-Core AMD Opteron(tm) Processor 8378");
  taudb_add_primary_metadata_to_trial(trial, &(pm[7]));

  // create a metric
  TAUDB_METRIC* metric = taudb_create_metrics(1);
  metric->name = taudb_strdup("TIME");
  taudb_add_metric_to_trial(trial, metric);

  // create a thread
  TAUDB_THREAD* thread = taudb_create_threads(1);
  thread->node_rank = 1;
  thread->context_rank = 1;
  thread->thread_rank = 1;
  thread->index = 1;
  taudb_add_thread_to_trial(trial, thread);

  // create a timer, timer_callpath, timer_call_data, timer_value
  TAUDB_TIMER_GROUP* timer_group = taudb_create_timer_groups(1);
  TAUDB_TIMER* timer = taudb_create_timers(1);
  // the timer parameter is optional
  TAUDB_TIMER_PARAMETER* timer_parameter = taudb_create_timer_parameters(2);
  TAUDB_TIMER_CALLPATH* timer_callpath = taudb_create_timer_callpaths(1);
  TAUDB_TIMER_CALL_DATA* timer_call_data = taudb_create_timer_call_data(1);
  TAUDB_TIMER_VALUE* timer_value = taudb_create_timer_values(1);

  timer->name = taudb_strdup("int main(int, char **) [{kernel.c} {134,1}-{207,1}]");
  timer->short_name = taudb_strdup("main");
  timer->source_file = taudb_strdup("kernel.c");
  timer->line_number = 134;
  timer->column_number = 1;
  timer->line_number_end = 207;
  timer->column_number_end = 1;
  taudb_add_timer_to_trial(trial, timer);

  timer_group->name = taudb_strdup("TAU_DEFAULT");
  taudb_add_timer_group_to_trial(trial, timer_group);
  taudb_add_timer_to_timer_group(timer_group, timer);

  // timer parameters are optional
  timer_parameter[0].name = taudb_strdup("argc");
  timer_parameter[0].value = taudb_strdup("1");
  taudb_add_timer_parameter_to_timer(timer, (&timer_parameter[0]));
  timer_parameter[1].name = taudb_strdup("argv");
  timer_parameter[1].value = taudb_strdup("myProgramName");
  taudb_add_timer_parameter_to_timer(timer, (&timer_parameter[1]));

  timer_callpath->timer = timer;
  timer_callpath->parent = NULL;
  taudb_add_timer_callpath_to_trial(trial, timer_callpath);

  timer_call_data->key.timer_callpath = timer_callpath;
  timer_call_data->key.thread = thread;
  timer_call_data->calls = 1;
  timer_call_data->subroutines = 0;
  taudb_add_timer_call_data_to_trial(trial, timer_call_data);

  timer_value->metric = metric;
  timer_value->inclusive = 5000000; // 5 seconds, or 5 million microseconds
  timer_value->exclusive = 5000000;
  timer_value->inclusive_percentage = 100.0;
  timer_value->exclusive_percentage = 100.0;
  timer_value->sum_exclusive_squared = 0.0;
  taudb_add_timer_value_to_timer_call_data(timer_call_data, timer_value);

  // compute stats
  printf("Computing Stats...\n");
  taudb_compute_statistics(trial);

  // create secondary metadata, specific to this thread
  TAUDB_SECONDARY_METADATA * sm = taudb_create_secondary_metadata(1);
  sm->key.timer_callpath = NULL; // no timer call associated with this value
  sm->key.thread = thread;
  sm->key.parent = NULL; // no nested metadata in this example
  sm->key.time_range = NULL; // no time range in this example
  sm->key.name  = taudb_strdup("HOSTNAME");
  sm->num_values = 1;
  sm->child_count = 0; // no nested metadata in this example
  sm->children = NULL; // no nested metadata in this example
  sm->value = (char**)malloc(sizeof(char*)); // allocate an array of 1 char*
  sm->value[0] = taudb_strdup("cn114");
  taudb_add_secondary_metadata_to_trial(trial, sm);

  // save the trial!
  printf("Testing inserts...\n");
  boolean update = FALSE;
  boolean cascade = TRUE;
  taudb_save_trial(connection, trial, update, cascade);
	
	free(trial->name);
	trial->name = taudb_strdup("FOO TEST TRIAL");
  TAUDB_METRIC* metric2 = taudb_create_metrics(1);
  metric2->name = taudb_strdup("L2_CACHE_MISS");
  taudb_add_metric_to_trial(trial, metric2);
	pm[4].value = taudb_strdup("aciss.uoregon.edu");
	update = TRUE;
	taudb_save_trial(connection, trial, update, cascade);
	
  
  printf("Disconnecting...\n");
  taudb_disconnect(connection);
  printf("Done.\n");
  return 0;
}
