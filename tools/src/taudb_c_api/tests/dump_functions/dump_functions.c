#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include "dump_functions.h"

void dump_metadata(TAUDB_PRIMARY_METADATA *metadata, int count) {
   int j;
   printf("%d metadata fields:\n", count);
   for (j = 0 ; j < count ; j = j+1) {
      printf("  %s = %s\n", metadata[j].name, metadata[j].value);
   }
}

void dump_trial(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, FALSE, filter);

   TAUDB_TIMER* timer = taudb_query_main_timer(connection, trial);
   printf("Trial name: '%s', id: %d, main: '%s'\n\n", trial->name, trial->id, timer->name);
}

void dump_timers(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, FALSE, filter);
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_TIMER* timers = taudb_query_timers(connection, trial);
   int numTimers = taudb_numItems;
   printf("Found %d timers\n", numTimers);

   int e;
   int total = 0;
   for (e = 0 ; e < numTimers ; e++) {
     printf("%s\n", timers[e].name);
   }

   //taudb_delete_timers(timers, 1);
}

void dump_metrics(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, FALSE, filter);
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_METRIC* metrics = taudb_query_metrics(connection, trial);
   int numMetrics = taudb_numItems;
   printf("Found %d metrics\n", numMetrics);

   int m;
   int total = 0;
   for (m = 0 ; m < numMetrics ; m++) {
     printf("%s\n", metrics[m].name);
   }

   //taudb_delete_metrics(metrics, 1);
}

void dump_threads(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, FALSE, filter);
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_THREAD* threads = taudb_query_threads(connection, trial);
   int numThreads = taudb_numItems;
   printf("Found %d threads\n", numThreads);

   int t;
   int total = 0;
   for (t = 0 ; t < numThreads ; t++) {
     printf("%d %d %d %d\n", threads[t].index, threads[t].node_rank, threads[t].context_rank, threads[t].thread_rank);
   }

   //taudb_delete_threads(threads, 1);
}

void dump_timer_callpaths(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, FALSE, filter);
   printf("Trial name: '%s', id: %d\n\n", trial->name, trial->id);

   TAUDB_THREAD* threads = taudb_query_threads(connection, trial);
   int numThreads = taudb_numItems;
   printf("Found %d threads\n", numThreads);
   TAUDB_TIMER* timers = taudb_query_timers(connection, trial);
   int numTimers = taudb_numItems;
   printf("Found %d timers\n", numTimers);
   TAUDB_TIMER_CALLPATH* callpaths = taudb_query_timer_callpaths(connection, trial, NULL, NULL);
   int numCallpaths = taudb_numItems;
   printf("Found %d callpaths\n", numCallpaths);

   int e, t, m;
   int total = 0;
   for (e = 0 ; e < numTimers ; e++) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("'%s'\n", timers[e].name);
#endif
      for (t = 0 ; t < numThreads ; t++) {
         TAUDB_TIMER_CALLPATH* timer_callpath = taudb_get_timer_callpath(callpaths, &(timers[e]), &(threads[t]));
         //TAUDB_TIMER_CALLPATH* timer_callpath = taudb_query_timer_callpaths(connection, trial, &(timers[e]), &(threads[t]));
         if (timer_callpath) {
           printf("timer %s, thread %d - calls: %d\n", timers[e].name, threads[t].index, timer_callpath->calls);
           total++;
         } else {
           printf("ERROR!!! key '%d:%s' not found.\n", threads[t].index, timers[e].name);
         }
      }
   }
   printf("Found %d objects in the hash.\n\n", total);

   if (numCallpaths != total) {
     printf("ERROR!!! %d != %d - MISSING ITEMS!\n\n", numCallpaths, total);
   }

   //taudb_delete_trials(trial, 1);
}

void dump_timer_values(PGconn* connection, TAUDB_TRIAL* filter) {
   TAUDB_TRIAL* trial = taudb_query_trials(connection, TRUE, filter);

   TAUDB_TIMER* timer = taudb_query_main_timer(connection, trial);
   printf("Trial name: '%s', id: %d, main: '%s'\n\n", trial->name, trial->id, timer->name);

   int numMetrics = trial->metric_count;
   int numTimers = trial->timer_count;
   int numThreads = trial->thread_count;
   int numCallpaths = trial->callpath_count;
   printf("Found %d threads\n", numThreads);
   printf("Found %d timers\n", numTimers);
   printf("Found %d callpaths\n", numCallpaths);
   printf("Found %d metrics\n\n", numMetrics);

   int e, t, m;
   int total = 0;

   TAUDB_TIMER_VALUE* timer_values = taudb_query_all_timer_values(connection, trial);
   printf("Found %d values\n\n", taudb_numItems);

   total = 0;
   for (e = 0 ; e < numTimers ; e++) {
      for (t = 0 ; t < numThreads ; t++) {
         for (m = 0 ; m < numMetrics ; m++) {
           TAUDB_TIMER_VALUE* timer_value = taudb_get_timer_value(timer_values, &(trial->timers[e]), &(trial->threads[t]), &(trial->metrics[m]));
           if (timer_value) {
             //printf("timer %s, metric %s, thread %d - inclusive: %f, exclusive %f\n", timers[e].name, metrics[m].name, t, timer_value->inclusive, timer_value->exclusive);
             total++;
           }
         }
      }
   }
   printf("Found %d objects in the hash.\n\n", total);

   //taudb_delete_trials(trial, 1);
}

