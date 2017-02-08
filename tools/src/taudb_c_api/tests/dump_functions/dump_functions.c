#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include "dump_functions.h"

void dump_metadata(TAUDB_PRIMARY_METADATA *metadata) {
   printf("%d metadata fields:\n", HASH_COUNT(metadata));
   TAUDB_PRIMARY_METADATA * current;
   for (current = metadata; current != NULL; 
		current = taudb_next_primary_metadata_by_name_from_trial(current)) {
     printf("  %s = %s\n", current->name, current->value);
   }
}

void dump_secondary_metadata(TAUDB_SECONDARY_METADATA *metadata) {
   printf("%d secondary metadata fields:\n", HASH_COUNT(metadata));
   TAUDB_SECONDARY_METADATA * current;
   for (current = metadata; current != NULL; 
	    current = taudb_next_secondary_metadata_by_key_from_trial(current)) {
     printf("  %s = %s\n", current->key.name, current->value[0]);
   }
}

void dump_trial(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }

   TAUDB_TIMER* timer = taudb_query_main_timer(connection, trial,taudb_numItems);
   printf("Trial name: '%s', id: %d, main: '%s'\n\n", trial->name, trial->id, timer->name);
}

void dump_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_TIMER* timers = taudb_query_timers(connection, trial,taudb_numItems);
   int numTimers = *taudb_numItems;
   printf("Found %d timers\n", numTimers);

   // iterate over the hash
   TAUDB_TIMER* timer;
   for (timer = timers; timer != NULL; 
        timer = taudb_next_timer_by_name_from_trial(timer)) {
     printf("%s\n", timer->name);
   }

   //taudb_delete_timers(timers, 1);
}

void dump_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_COUNTER* counters = taudb_query_counters(connection, trial,taudb_numItems);
   int numTimers = *taudb_numItems;
   printf("Found %d counters\n", numTimers);

   // iterate over the hash
   TAUDB_COUNTER* counter;
   for (counter = counters ; counter != NULL ; 
        counter = taudb_next_counter_by_name_from_trial(counter)) {
     printf("%s\n", counter->name);
   }

   //taudb_delete_counters(timers, 1);
}

void dump_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_COUNTER* counters = taudb_query_counters(connection, trial,taudb_numItems);
   int numTimers = *taudb_numItems;
   printf("Found %d counters\n", numTimers);
   TAUDB_THREAD* threads = taudb_query_threads(connection, trial,taudb_numItems);
   int numThreads = *taudb_numItems;
   printf("Found %d threads\n", numThreads);
   TAUDB_COUNTER_VALUE* counter_values = taudb_query_counter_values(connection, trial,taudb_numItems);

   // iterate over the hash
   TAUDB_COUNTER* counter;
   TAUDB_THREAD* thread;
   for (thread = threads ; thread != NULL ; 
        thread = taudb_next_thread_by_index_from_trial(thread)) {
     printf("\n\n");
     for (counter = counters ; counter != NULL ; 
          counter = taudb_next_counter_by_name_from_trial(counter)) {
       TAUDB_COUNTER_VALUE* counter_value = taudb_get_counter_value(counter_values, counter, thread, NULL, NULL);
       printf("Thread %d, %s, %f\n", thread->index, counter->name, counter_value->mean_value);
     }
   }

   //taudb_delete_counters(timers, 1);
}

void dump_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_METRIC* metrics = taudb_query_metrics(connection, trial,taudb_numItems);
   int numMetrics = *taudb_numItems;
   printf("Found %d metrics\n", numMetrics);

   // iterate over the hash
   TAUDB_METRIC* metric;
   for (metric = metrics ; metric != NULL ; 
        metric = taudb_next_metric_by_name_from_trial(metric)) {
     printf("%s\n", metric->name);
   }

   //taudb_delete_metrics(metrics, 1);
}

void dump_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_THREAD* threads = taudb_query_threads(connection, trial,taudb_numItems);
   int numThreads = *taudb_numItems;
   printf("Found %d threads\n", numThreads);

   // iterate over the hash
   TAUDB_THREAD* thread;
   for (thread = threads ; thread != NULL ; 
        thread = taudb_next_thread_by_index_from_trial(thread)) {
     printf("%d %d %d %d\n", thread->index, thread->node_rank, thread->context_rank, thread->thread_rank);
   }

   //taudb_delete_threads(threads, 1);
}

void dump_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   dump_threads(connection, trial, TRUE);
   dump_timers(connection, trial, TRUE);

   TAUDB_TIMER_CALLPATH* callpaths = taudb_query_timer_callpaths(connection, trial, NULL,taudb_numItems);
   int numCallpaths = *taudb_numItems;
   printf("Found %d callpaths\n", numCallpaths);

   int total = 0;
   TAUDB_TIMER_CALLPATH* callpath;
   for (callpath = callpaths ; callpath != NULL ; 
        callpath = taudb_next_timer_callpath_by_name_from_trial(callpath)) {
     if (callpath->parent == NULL) {
       printf("timer '%s', parent: (nil)\n", callpath->timer->name);
     } else {
       printf("timer '%s', parent: '%s'\n", callpath->timer->name, callpath->parent->timer->name);
     }
     total++;
   }
   printf("Found %d objects in the hash.\n\n", total);

   taudb_delete_trials(trial, 1);

   if (numCallpaths != total) {
     printf("ERROR!!! %d != %d - MISSING ITEMS!\n\n", numCallpaths, total);
     exit(1);
   }
   return;
}

void dump_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   dump_timer_callpaths(connection, trial, TRUE);

   //TAUDB_TIMER_CALL_DATA* call_datas = taudb_query_timer_call_data(connection, trial, NULL, NULL);
   taudb_query_timer_call_data(connection, trial, NULL, NULL,taudb_numItems);
   int numCallpaths = *taudb_numItems;
   printf("Found %d call_datas\n", numCallpaths);

   int total = 0;
   TAUDB_THREAD* thread;
   TAUDB_TIMER_CALLPATH* callpath;
   TAUDB_TIMER_CALL_DATA* call_data;
   printf("Calls, Subroutines, Callpath\n");
   for (thread = trial->threads ; thread != NULL ; 
        thread = taudb_next_thread_by_index_from_trial(thread)) {
     for (callpath = trial->timer_callpaths_by_name ; callpath != NULL ; 
          callpath = taudb_next_timer_callpath_by_name_from_trial(callpath)) {
       call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, callpath, thread, NULL);
       if (call_data != NULL) {
         printf("%d, %d, %d, '%s'\n", call_data->id, call_data->calls, call_data->subroutines, callpath->name);
         total++;
       }
     }
   }
   printf("Found %d objects in the hash.\n\n", total);

   taudb_delete_trials(trial, 1);

   if (numCallpaths != total) {
     printf("ERROR!!! %d != %d - MISSING ITEMS!\n\n", numCallpaths, total);
     exit(1);
   }
   return;
}


void dump_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   int items=0;  int* taudb_numItems=&items;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter,taudb_numItems);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   dump_timer_call_data(connection, trial, TRUE);
   dump_metrics(connection, trial, TRUE);

   //TAUDB_TIMER_VALUE* values = taudb_query_all_timer_values(connection, trial);
   taudb_query_all_timer_values(connection, trial,taudb_numItems);
   int numValues = *taudb_numItems;
   printf("Found %d timer values\n", numValues);

   int total = 0;
   TAUDB_THREAD* thread;
   TAUDB_TIMER_CALLPATH* callpath;
   TAUDB_TIMER_CALL_DATA* call_data;
   TAUDB_TIMER_VALUE* value;
   TAUDB_METRIC* metric;
   printf("Thread, Calls, Subroutines, Metric, Inclusive, Exclusive, Callpath\n");
   for (thread = trial->threads ; thread != NULL ; 
        thread = taudb_next_thread_by_index_from_trial(thread)) {
     for (callpath = trial->timer_callpaths_by_name ; callpath != NULL ; 
          callpath = taudb_next_timer_callpath_by_name_from_trial(callpath)) {
       call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_id, callpath, thread, NULL);
       if (call_data != NULL) {
         for (metric = trial->metrics_by_name ; metric != NULL ; 
              metric = taudb_next_metric_by_name_from_trial(metric)) {
           value = taudb_get_timer_value(call_data, metric);
		   if (value != NULL) {
             printf("%d, %d, %d, %s, %f, %f, '%s'\n", thread->index, call_data->calls, call_data->subroutines, metric->name, value->inclusive, value->exclusive, callpath->name);
           total++;
			}
         }
       }
     }
   }
   printf("Found %d objects in the hash.\n\n", total);

   taudb_delete_trials(trial, 1);

   if (numValues != total) {
     printf("ERROR!!! %d != %d - MISSING ITEMS!\n\n", numValues, total);
     exit(1);
   }
   return;
}



