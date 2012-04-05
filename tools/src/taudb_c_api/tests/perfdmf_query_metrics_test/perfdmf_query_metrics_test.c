#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include "dump_functions.h"

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
	    dump_metadata(trials[t].primary_metadata, trials[t].primary_metadata_count);
        dump_trial_metrics(connection, &(trials[t]));
     }
   }

   printf("Disconnecting...\n");
   taudb_disconnect(connection);
   printf("Done.\n");
return 0;
}
