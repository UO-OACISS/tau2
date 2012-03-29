#include "taudb_api.h"
#include <stdio.h>
#include <string.h>

int main (int argc, char** argv) {
   printf("Connecting...\n");
   char* host = "paratools03.rrp.net";
   char* port = "5432";
#ifdef TAUDB_PERFDMF
   char* database = "facets";
#else
   char* database = "taudb";
#endif
   char* login = "khuck";
   char* password = "kevin huck perfdmf facets";
   taudb_connect(host, port, database, login, password);
   printf("Checking connection...\n");
   taudb_checkConnection();
   printf("Testing queries...\n");

   int i = 0;
   int j = 0;
#ifdef TAUDB_PERFDMF
/*
   PERFDMF_APPLICATION* applications = taudb_get_applications();
   printf("%d applications:\n", taudb_numItems);
   for (i = 0 ; i < taudb_numItems ; i = i+1) {
     printf("  Application name: %s, id: %d\n", applications[i].name, applications[i].id);
   }
   PERFDMF_EXPERIMENT* experiments = taudb_get_experiments(&(applications[0]));
   printf("%d experiments:\n", taudb_numItems);
   for (i = 0 ; i < taudb_numItems ; i = i+1) {
     printf("  Experiment name: %s, id: %d\n", experiments[i].name, experiments[i].id);
   }
   TAUDB_TRIAL* trials = taudb_get_trials(&(experiments[0]));
   printf("%d trials:\n", taudb_numItems);
   for (i = 0 ; i < taudb_numItems ; i = i+1) {
     printf("  Trial name: %s, id: %d\n", trials[i].name, trials[i].id);
     printf("%d metadata fields:\n", trials[i].primary_metadata_count);
     for (j = 0 ; j < trials[i].primary_metadata_count ; j = j+1) {
	   printf("  Trial %s = %s\n", trials[i].primary_metadata[j].name, trials[i].primary_metadata[j].value);
     }
   }

   TAUDB_THREAD* threads = taudb_get_threads(&(trials[0]));
   printf("%d threads:\n", taudb_numItems);
   for (i = 0 ; i < taudb_numItems ; i = i+1) {
	 printf("  node %d, context %d, thread, %d\n", threads[i].node_rank, threads[i].context_rank, threads[i].thread_rank);
   }

   TAUDB_METRIC* metrics = taudb_get_metrics(&(trials[0]));
   printf("%d metrics:\n", taudb_numItems);
   for (i = 0 ; i < taudb_numItems ; i = i+1) {
	 printf("  id %d, name %s\n", metrics[i].id, metrics[i].name);
   }
   */

   PERFDMF_APPLICATION* application = taudb_get_application("ACISS Regression");
   printf("Application name: %s, id: %d\n", application->name, application->id);
   PERFDMF_EXPERIMENT* experiment = taudb_get_experiment(application, "2012-03-06");
   printf("Experiment name: %s, id: %d\n", experiment->name, experiment->id);
   TAUDB_TRIAL* trials = taudb_get_trials(experiment);
   // test the "find trials" method to populate the trial
   TAUDB_TRIAL* filter = taudb_create_trials(1);
   filter->id = 216;
   TAUDB_TRIAL* trial = taudb_find_trials(FALSE, filter);
   TAUDB_TIMER* timer = taudb_get_main_timer(trial);
   printf("Trial name: '%s', id: %d, main: '%s'\n", trials->name, trials->id, timer->name);
   TAUDB_METRIC* metrics = taudb_get_metrics(trial);
   printf("Found %d metrics\n", taudb_numItems);
   TAUDB_TIMER_VALUE* timer_values = taudb_get_timer_values(trial, timer, NULL, NULL);
   printf("Found %d values\n", taudb_numItems);
#else
   taudb_iterate_tables();
#endif

   printf("Disconnecting...\n");
   taudb_disconnect();
   printf("Done.\n");
   return 0;
}
