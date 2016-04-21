#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include "dump_functions.h"

int main (int argc, char** argv) {
   printf("Connecting...\n");
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
   printf("Testing queries...\n");

   int t;

   //if (taudb_version == TAUDB_2012_SCHEMA) {
     // test the "find trials" method to populate the trial
     TAUDB_TRIAL* filter = taudb_create_trials(1);
     // filter->id = atoi(argv[2]);
	 // Find trials executed on a Cray: TAU Architecture = craycnl
     TAUDB_PRIMARY_METADATA* pm = taudb_create_primary_metadata(1);
     pm->name = taudb_strdup("TAU Architecture");
     pm->value = taudb_strdup("craycnl");
     taudb_add_primary_metadata_to_trial(filter, pm);

     TAUDB_PRIMARY_METADATA* pm2 = taudb_create_primary_metadata(1);
     pm2->name = taudb_strdup("username");
     pm2->value = taudb_strdup("khuck");
     taudb_add_primary_metadata_to_trial(filter, pm2);

     TAUDB_PRIMARY_METADATA* pm3 = taudb_create_primary_metadata(1);
     pm3->name = taudb_strdup("Local Time");
	 // this is a wildcard search - get any trials loaded on this date
     pm3->value = taudb_strdup("2012-10-17%");
     taudb_add_primary_metadata_to_trial(filter, pm3);

     TAUDB_TRIAL* trials = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
     int numTrials = *taudb_numItems;
     for (t = 0 ; t < numTrials ; t = t+1) {
        printf("  Trial name: '%s', id: %d\n", trials[t].name, trials[t].id);
	    dump_metadata(trials[t].primary_metadata);
        dump_trial(connection, &(trials[t]), TRUE);
     }
   //}

   printf("Disconnecting...\n");
   taudb_disconnect(connection);
   printf("Done.\n");
return 0;
}
