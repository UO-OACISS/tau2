#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include "dump_functions.h"

int main (int argc, char** argv) {
   printf("Connecting...\n");
   TAUDB_CONNECTION* connection = NULL;
   int items=0;  int* taudb_numItems=&items;
   if (argc >= 2) {
     connection = taudb_connect_config(argv[1],taudb_numItems);
   } else {
     fprintf(stderr, "Please specify a TAUdb config file.\n");
     exit(1);
   }
   printf("Checking connection...\n");
   taudb_check_connection(connection);
   printf("Testing queries...\n");

   int a, e, t;

   if (taudb_version == TAUDB_2005_SCHEMA) {
     PERFDMF_APPLICATION* applications = perfdmf_query_applications(connection,taudb_numItems);
     printf("%d applications:\n", (*taudb_numItems));
     int numApplications = *taudb_numItems;
     for (a = 0 ; a < numApplications ; a = a+1) {
       printf("  Application name: %s, id: %d\n", applications[a].name, applications[a].id);
	   dump_metadata(applications[a].primary_metadata);
       PERFDMF_EXPERIMENT* experiments = perfdmf_query_experiments(connection, &(applications[a]),taudb_numItems);
       printf("%d experiments:\n", (*taudb_numItems));
       int numExperiments = *taudb_numItems;
       for (e = 0 ; e < numExperiments ; e = e+1) {
         printf("  Experiment name: %s, id: %d\n", experiments[e].name, experiments[e].id);
	     dump_metadata(experiments[e].primary_metadata);
         TAUDB_TRIAL* trials = perfdmf_query_trials(connection, &(experiments[e]),taudb_numItems);
         printf("%d trials:\n", *taudb_numItems);
         int numTrials = *taudb_numItems;
         for (t = 0 ; t < numTrials ; t = t+1) {
           printf("  Trial name: '%s', id: %d\n", trials[t].name, trials[t].id);
	       dump_metadata(trials[t].primary_metadata);
           dump_trial(connection, &(trials[t]), TRUE);
         }
       }
	   //perfdmf_delete_experiments(experiments, numExperiments);
     }
	 //perfdmf_delete_applications(applications, numApplications);
   } else {
     // test the "find trials" method to populate the trial
     TAUDB_TRIAL* filter = taudb_create_trials(1);
     //filter->id = 216;
     filter->id = 1;
     TAUDB_TRIAL* trials = taudb_query_trials(connection, TRUE, filter,taudb_numItems);
     int numTrials = *taudb_numItems;
     for (t = 0 ; t < numTrials ; t = t+1) {
        printf("  Trial name: '%s', id: %d\n", trials[t].name, trials[t].id);
	    dump_metadata(trials[t].primary_metadata);
        dump_trial(connection, &(trials[t]), TRUE);
     }
   }

   printf("Disconnecting...\n");
   taudb_disconnect(connection);
   printf("Done.\n");
   return 0;
}
