#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include "dump_functions.h"

void dump_metadata(TAUDB_PRIMARY_METADATA *metadata) {
   printf("%d metadata fields:\n", HASH_COUNT(metadata));
   TAUDB_PRIMARY_METADATA * cur;
   for(cur = metadata; cur != NULL; cur = (TAUDB_PRIMARY_METADATA*)cur->hh.next) {
     printf("  %s = %s\n", cur->name, cur->value);
   }
}


void dump_trial(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }

   TAUDB_TIMER* timer = taudb_query_main_timer(connection, trial);
   printf("Trial name: '%s', id: %d, main: '%s'\n\n", trial->name, trial->id, timer->name);
}

void dump_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_TIMER* timers = taudb_query_timers(connection, trial);
   int numTimers = taudb_numItems;
   printf("Found %d timers\n", numTimers);

   // iterate over the hash
   TAUDB_TIMER* timer;
   for (timer = timers ; timer != NULL ; timer=(TAUDB_TIMER*)timer->hh1.next) {
     printf("%s\n", timer->name);
   }

   //taudb_delete_timers(timers, 1);
}

void dump_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_COUNTER* counters = taudb_query_counters(connection, trial);
   int numTimers = taudb_numItems;
   printf("Found %d counters\n", numTimers);

   // iterate over the hash
   TAUDB_COUNTER* counter;
   for (counter = counters ; counter != NULL ; counter=(TAUDB_COUNTER*)counter->hh1.next) {
     printf("%s\n", counter->name);
   }

   //taudb_delete_counters(timers, 1);
}

void dump_counter_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_COUNTER* counters = taudb_query_counters(connection, trial);
   int numTimers = taudb_numItems;
   printf("Found %d counters\n", numTimers);
   TAUDB_THREAD* threads = taudb_query_threads(connection, trial);
   int numThreads = taudb_numItems;
   printf("Found %d threads\n", numThreads);
   TAUDB_COUNTER_VALUE* counter_values = taudb_query_counter_values(connection, trial);

   // iterate over the hash
   TAUDB_COUNTER* counter;
   TAUDB_THREAD* thread;
   for (thread = threads ; thread != NULL ; thread=(TAUDB_THREAD*)thread->hh.next) {
     printf("\n\n");
     for (counter = counters ; counter != NULL ; counter=(TAUDB_COUNTER*)counter->hh1.next) {
       TAUDB_COUNTER_VALUE* counter_value = taudb_get_counter_value(counter_values, counter, thread, NULL, NULL);
       printf("Thread %d, %s, %f\n", thread->index, counter->name, counter_value->mean_value);
     }
   }

   //taudb_delete_counters(timers, 1);
}

void dump_metrics(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_METRIC* metrics = taudb_query_metrics(connection, trial);
   int numMetrics = taudb_numItems;
   printf("Found %d metrics\n", numMetrics);

   // iterate over the hash
   TAUDB_METRIC* metric;
   for (metric = metrics ; metric != NULL ; metric=(TAUDB_METRIC*)metric->hh1.next) {
     printf("%s\n", metric->name);
   }

   //taudb_delete_metrics(metrics, 1);
}

void dump_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_THREAD* threads = taudb_query_threads(connection, trial);
   int numThreads = taudb_numItems;
   printf("Found %d threads\n", numThreads);

   // iterate over the hash
   TAUDB_THREAD* thread;
   for (thread = threads ; thread != NULL ; thread=(TAUDB_THREAD*)thread->hh.next) {
     printf("%d %d %d %d\n", thread->index, thread->node_rank, thread->context_rank, thread->thread_rank);
   }

   //taudb_delete_threads(threads, 1);
}

void dump_timer_callpaths(TAUDB_CONNECTION* connection, TAUDB_TRIAL* filter, boolean haveTrial) {
   TAUDB_TRIAL* trial;
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   dump_threads(connection, trial, TRUE);
   dump_timers(connection, trial, TRUE);

   TAUDB_TIMER_CALLPATH* callpaths = taudb_query_timer_callpaths(connection, trial, NULL);
   int numCallpaths = taudb_numItems;
   printf("Found %d callpaths\n", numCallpaths);

   int total = 0;
   TAUDB_TIMER_CALLPATH* callpath;
   for (callpath = callpaths ; callpath != NULL ; callpath=(TAUDB_TIMER_CALLPATH*)callpath->hh1.next) {
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
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   dump_timer_callpaths(connection, trial, TRUE);

   //TAUDB_TIMER_CALL_DATA* call_datas = taudb_query_timer_call_data(connection, trial, NULL, NULL);
   taudb_query_timer_call_data(connection, trial, NULL, NULL);
   int numCallpaths = taudb_numItems;
   printf("Found %d call_datas\n", numCallpaths);

   int total = 0;
   TAUDB_THREAD* thread;
   TAUDB_TIMER_CALLPATH* callpath;
   TAUDB_TIMER_CALL_DATA* call_data;
   printf("Calls, Subroutines, Callpath\n");
   for (thread = trial->threads ; thread != NULL ; thread=(TAUDB_THREAD*)thread->hh.next) {
     for (callpath = trial->timer_callpaths_by_id ; callpath != NULL ; callpath=(TAUDB_TIMER_CALLPATH*)callpath->hh1.next) {
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
   if (haveTrial) {
     trial = filter;
   } else {
     trial = taudb_query_trials(connection, FALSE, filter);
   }
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   dump_timer_call_data(connection, trial, TRUE);
   dump_metrics(connection, trial, TRUE);

   //TAUDB_TIMER_VALUE* values = taudb_query_all_timer_values(connection, trial);
   taudb_query_all_timer_values(connection, trial);
   int numValues = taudb_numItems;
   printf("Found %d timer values\n", numValues);

   int total = 0;
   TAUDB_THREAD* thread;
   TAUDB_TIMER_CALLPATH* callpath;
   TAUDB_TIMER_CALL_DATA* call_data;
   TAUDB_TIMER_VALUE* value;
   TAUDB_METRIC* metric;
   printf("Thread, Calls, Subroutines, Metric, Inclusive, Exclusive, Callpath\n");
   for (thread = trial->threads ; thread != NULL ; thread=(TAUDB_THREAD*)thread->hh.next) {
     for (callpath = trial->timer_callpaths_by_id ; callpath != NULL ; callpath=(TAUDB_TIMER_CALLPATH*)callpath->hh1.next) {
       call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_id, callpath, thread, NULL);
       if (call_data != NULL) {
         for (metric = trial->metrics_by_id ; metric != NULL ; metric=(TAUDB_METRIC*)metric->hh1.next) {
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



