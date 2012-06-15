#include "taudb_api.h"
#include <stdio.h>
#include <string.h>

void dump_counter_values(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, FALSE, filter);
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_THREAD* threads = taudb_query_threads(connection, trial);
   int numThreads = taudb_numItems;
   printf("Found %d threads\n", numThreads);

   TAUDB_COUNTER* counters = taudb_query_counters(connection, trial);
   int numCounters = taudb_numItems;
   printf("Found %d counters\n", numCounters);

   printf("%s\n", counters[1].name);
   printf("%s\n", counters[2].name);
   printf("%s\n", counters[3].name);
   exit(0);

   TAUDB_COUNTER_VALUE* counter_values = taudb_query_all_counter_values(connection, trial);
   int numValues = taudb_numItems;
   printf("Found %d values\n\n", numValues);

   printf("%s\n", counters[2].name);

   int e, t, total = 0;
   for (e = 0 ; e < numCounters ; e++) {
      for (t = 0 ; t < numThreads ; t++) {
         TAUDB_COUNTER_VALUE* counter_value = taudb_get_counter_value(counter_values, &(counters[e]), &(threads[t]));
         //TAUDB_COUNTER_VALUE* counter_value = taudb_query_counter_values(connection, trial, &(counters[e]), &(threads[t]));
         if (counter_value) {
             printf("counter %s, thread %d - num_samples: %d, mean %f\n", counters[e].name, t, counter_value->sample_count, counter_value->mean_value);
             total++;
         } else {
		    printf("\n'%d:%s' not found\n\n", threads[t].index, counters[e].name);
         }
      }
   }
   printf("Found %d objects in the hash.\n\n", total);

   if (numValues != total) {
     printf("ERROR!!! %d != %d - MISSING ITEMS!\n\n", taudb_numItems, total);
	 exit(1);
   }


   taudb_delete_trials(trial, 1);
}

int main (int argc, char** argv) {
   printf("Connecting...\n");
   PGconn* connection = NULL;
   connection = taudb_connect_config("facets");
   printf("Checking connection...\n");
   taudb_check_connection(connection);
   printf("Testing queries...\n");

   int i = 0;
   int j = 0;
   int a, e, t;

   if (taudb_version == TAUDB_2005_SCHEMA) {
     // test the "find trials" method to populate the trial
     TAUDB_TRIAL* filter = taudb_create_trials(1);
     //filter->id = 216;
     filter->id = 209;
     TAUDB_TRIAL* trials = taudb_query_trials(connection, FALSE, filter);
     int numTrials = taudb_numItems;
     for (t = 0 ; t < numTrials ; t = t+1) {
        printf("  Trial name: '%s', id: %d\n", trials[t].name, trials[t].id);
        dump_counter_values(connection, &(trials[t]));
     }
   }

   printf("Disconnecting...\n");
   taudb_disconnect(connection);
   printf("Done.\n");
return 0;
}


