#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

TAUDB_PRIMARY_METADATA* test_primary_metadata(int count) {
   TAUDB_PRIMARY_METADATA* metadata = taudb_create_primary_metadata(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     metadata[i].name = taudb_create_string(count);
	 metadata[i].value = taudb_create_string(count);
   }
   return metadata;
}

TAUDB_SECONDARY_METADATA* test_secondary_metadata(int count) {
   TAUDB_SECONDARY_METADATA* metadata = taudb_create_secondary_metadata(count);
   int i,j;
   for (i = 0 ; i < count ; i++ ) {
     metadata[i].name = taudb_create_string(count);
     metadata[i].values = (char**) calloc(count, sizeof(char*));
     for (j = 0 ; j < count ; j++ ) {
    	 metadata[i].values[j] = taudb_create_string(count);
     }
   }
   return metadata;
}

PERFDMF_APPLICATION* test_applications(int count) {
   PERFDMF_APPLICATION* applications = perfdmf_create_applications(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     applications[i].primary_metadata = test_primary_metadata(count);
	 applications[i].primary_metadata_count = count;
   }
   return applications;
}

PERFDMF_EXPERIMENT* test_experiments(int count) {
   PERFDMF_EXPERIMENT* experiments = perfdmf_create_experiments(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     experiments[i].primary_metadata = test_primary_metadata(count);
	 experiments[i].primary_metadata_count = count;
   }
   return experiments;
}

TAUDB_TRIAL* test_trials(int count) {
   TAUDB_TRIAL* trials = taudb_create_trials(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     trials[i].primary_metadata = test_primary_metadata(count);
	 trials[i].primary_metadata_count = count;
     trials[i].secondary_metadata = test_secondary_metadata(count);
	 trials[i].secondary_metadata_count = count;
   }
   return trials;
}

TAUDB_THREAD* test_threads(int count) {
   TAUDB_THREAD* threads = taudb_create_threads(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     threads[i].secondary_metadata = test_secondary_metadata(count);
	 threads[i].secondary_metadata_count = count;
   }
   return threads;
}

TAUDB_METRIC* test_metrics(int count) {
   TAUDB_METRIC* metrics = taudb_create_metrics(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     metrics[i].name = taudb_create_string(count);
   }
   return metrics;
}

TAUDB_TIMER_VALUE* test_timer_values(int count) {
   TAUDB_TIMER_VALUE* timer_values = taudb_create_timer_values(count);
   return timer_values;
}

TAUDB_TIMER_CALLPATH* test_timer_callpaths(int count) {
   TAUDB_TIMER_CALLPATH* timer_callpaths = taudb_create_timer_callpaths(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     timer_callpaths[i].parent_key = taudb_create_string(count);
     timer_callpaths[i].timestamp = taudb_create_string(count);
   }
   return timer_callpaths;
}

TAUDB_TIMER_GROUP* test_timer_groups(int count) {
   TAUDB_TIMER_GROUP* timer_groups = taudb_create_timer_groups(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     timer_groups[i].name = taudb_create_string(count);
   }
   return timer_groups;
}

TAUDB_TIMER_PARAMETER* test_timer_parameters(int count) {
   TAUDB_TIMER_PARAMETER* timer_parameters = taudb_create_timer_parameters(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     timer_parameters[i].name = taudb_create_string(count);
     timer_parameters[i].value = taudb_create_string(count);
   }
   return timer_parameters;
}

TAUDB_TIMER* test_timers(int count) {
   TAUDB_TIMER* timers = taudb_create_timers(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     timers[i].short_name = taudb_create_string(count);
     timers[i].full_name = taudb_create_string(count);
     timers[i].source_file = taudb_create_string(count);
     timers[i].groups = test_timer_groups(count);
	 timers[i].group_count = count;
     timers[i].parameters = test_timer_parameters(count);
	 timers[i].parameter_count = count;
     timers[i].values = test_timer_values(count);
   }
   return timers;
}

TAUDB_COUNTER_VALUE* test_counter_values(int count) {
   TAUDB_COUNTER_VALUE* counter_values = taudb_create_counter_values(count);
   return counter_values;
}

TAUDB_COUNTER_GROUP* test_counter_groups(int count) {
   TAUDB_COUNTER_GROUP* counter_groups = taudb_create_counter_groups(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     counter_groups[i].name = taudb_create_string(count);
   }
   return counter_groups;
}

TAUDB_COUNTER* test_counters(int count) {
   TAUDB_COUNTER* counters = taudb_create_counters(count);
   int i;
   for (i = 0 ; i < count ; i++ ) {
     counters[i].full_name = taudb_create_string(count);
     counters[i].short_name = taudb_create_string(count);
     counters[i].source_file = taudb_create_string(count);
     counters[i].groups = test_counter_groups(count);
	 counters[i].group_count = count;
     counters[i].values = test_counter_values(count);
   }
   return counters;
}

int main (int argc, char** argv) {
   int count = 10;

// test all metadata 

   TAUDB_PRIMARY_METADATA* primary_metadata = test_primary_metadata(count);
   taudb_delete_primary_metadata(primary_metadata, count);
   TAUDB_SECONDARY_METADATA* secondary_metadata = test_secondary_metadata(count);
   taudb_delete_secondary_metadata(secondary_metadata, count);

   PERFDMF_APPLICATION* applications = test_applications(count);
   perfdmf_delete_applications(applications, count);
   PERFDMF_EXPERIMENT* experiments = test_experiments(count);
   perfdmf_delete_experiments(experiments, count);

   TAUDB_THREAD* threads = test_threads(count);
   taudb_delete_threads(threads, count);
   TAUDB_METRIC* metrics = test_metrics(count);
   taudb_delete_metrics(metrics, count);

// test timers

   TAUDB_TIMER_VALUE* timer_values = test_timer_values(count);
   taudb_delete_timer_values(timer_values, count);
   TAUDB_TIMER_CALLPATH* timer_callpaths = test_timer_callpaths(count);
   taudb_delete_timer_callpaths(timer_callpaths, count);
   TAUDB_TIMER_GROUP* timer_groups = test_timer_groups(count);
   taudb_delete_timer_groups(timer_groups, count);
   TAUDB_TIMER_PARAMETER* timer_parameters = test_timer_parameters(count);
   taudb_delete_timer_parameters(timer_parameters, count);
   TAUDB_TIMER* timers = test_timers(count);
   taudb_delete_timers(timers, count);

// test counters

   TAUDB_COUNTER_VALUE* counter_values = test_counter_values(count);
   taudb_delete_counter_values(counter_values, count);
   TAUDB_COUNTER_GROUP* counter_groups = test_counter_groups(count);
   taudb_delete_counter_groups(counter_groups, count);
   TAUDB_COUNTER* counters = test_counters(count);
   taudb_delete_counters(counters, count);
   
// test everything, nested

   TAUDB_TRIAL* trials = test_trials(count);
   taudb_delete_trials(trials, count);

   printf("Done.\n");
   return 0;
}
