#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_VALUE* taudb_private_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric, boolean derived) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_values(%p,%p,%p,%p)\n", trial, timer_callpath, thread, metric);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  // validate inputs
  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

/*
  //if the Trial already has the data, return it.
  if (trial->timer_values != NULL && trial->value_count > 0) {
    taudb_numItems = trial->value_count;
    return trial->timer_values;
  }
  */

  taudb_begin_transaction(connection);

// select ilp.*, ie.name as event_name, m.name as metric_name from interval_location_profile ilp left outer join interval_event ie on ilp.interval_event = ie.id left outer join metric m on ilp.metric = m.id where ie.trial = 206;

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[1024];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"select ilp.*, ie.name as timer_name, ie.metric from interval_location_profile ilp inner join interval_event ie on ilp.interval_event = ie.id");
    char* conjoiner = "where";
    if (trial != NULL) {
      sprintf(my_query,"%s where ie.trial = %d", my_query, trial->id);
      conjoiner = "and";
    } 
    if (timer_callpath != NULL) {
      fprintf(stderr, "TODO: THE CALLPATH IS NOT VALID IN PERFDMF\n");
      sprintf(my_query,"%s %s ie.id = %d", my_query, conjoiner, timer_callpath->id);
      conjoiner = "and";
    }
    if (metric != NULL) {
      if ((strcmp(metric->name, "calls") == 0) ||
          (strcmp(metric->name, "subroutines") == 0)) {
          // we need just one metric, but from this trial
        sprintf(my_query,"%s %s m.id = (select max(id) from metric where trial = %d)", my_query, conjoiner, trial->id);
      } else {
        sprintf(my_query,"%s %s m.id = %d", my_query, conjoiner, metric->id);
      }
      conjoiner = "and";
    }
    if (thread != NULL) {
      sprintf(my_query,"%s %s node = %d and context = %d and thread = %d", my_query, conjoiner, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
  } else {
    //sprintf(my_query,"select * from timer where trial = %d", trial->id);
    sprintf(my_query,"select tv.*, h.thread_index as index from timer_value tv inner join timer_call_data td on tv.timer_call_data = td.id inner join timer_callpath tc on td.timer_callpath = tc.id left outer join thread h on tv.thread = h.id");
    char* conjoiner = "where";
    if (trial != NULL) {
      sprintf(my_query,"%s where t.trial = %d", my_query, trial->id);
      conjoiner = "and";
    } 
    if (timer_callpath != NULL) {
      sprintf(my_query,"%s %s tc.id = %d", my_query, conjoiner, timer_callpath->id);
      conjoiner = "and";
    }
    if (metric != NULL) {
      sprintf(my_query,"%s %s tv.metric = %d", my_query, conjoiner, metric->id);
      conjoiner = "and";
    }
    if (thread != NULL) {
      sprintf(my_query,"%s %s h.thread_index = %d ", my_query, conjoiner, thread->index);
      conjoiner = "and";
    }
    if (derived) {
      sprintf(my_query,"%s %s h.thread_index < 0 order by h.thread_index desc", my_query, conjoiner);
    } else {
      sprintf(my_query,"%s %s h.thread_index > -1 order by h.thread_index asc", my_query, conjoiner);
    }
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  //TAUDB_TIMER_VALUE* timer_values = taudb_create_timer_values(nRows);
  TAUDB_TIMER_VALUE* timer_values = NULL;
  taudb_numItems = nRows;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
    int index = 0;
    int metric_id;
    char* timer_str;
    TAUDB_TIMER_VALUE* timer_value = taudb_create_timer_values(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
        //timer_value->id = atoi(taudb_get_value(res, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(res, j), "interval_event") == 0) {
        //timer_value->timer = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "timer") == 0) {
        //timer_value->timer = atoi(taudb_get_value(res, i, j));

      } else if (strcmp(taudb_get_column_name(res, j), "node") == 0) {
        node = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "context") == 0) {
        context = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "thread") == 0) {
        thread = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "index") == 0) {
        index = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "metric") == 0) {
        metric_id = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "timer_name") == 0) {
        timer_str = taudb_get_value(res, i, j);
      } else if (strcmp(taudb_get_column_name(res, j), "metric") == 0) {
        if (metric != NULL)
          timer_value->metric = metric;
        else {
          timer_value->metric = taudb_get_metric_by_id(trial->metrics, atoi(taudb_get_value(res, i, j)));
        }
// these two are the same
      } else if (strcmp(taudb_get_column_name(res, j), "inclusive_percentage") == 0) {
        timer_value->inclusive_percentage = atof(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "inclusive_percent") == 0) {
        timer_value->inclusive_percentage = atof(taudb_get_value(res, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(res, j), "exclusive_percentage") == 0) {
        timer_value->exclusive_percentage = atof(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "exclusive_percent") == 0) {
        timer_value->exclusive_percentage = atof(taudb_get_value(res, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(res, j), "inclusive") == 0) {
        timer_value->inclusive = atof(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "inclusive_value") == 0) {
        timer_value->inclusive = atof(taudb_get_value(res, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(res, j), "exclusive") == 0) {
        timer_value->exclusive = atof(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "exclusive_value") == 0) {
        timer_value->exclusive = atof(taudb_get_value(res, i, j));

      } else if (strcmp(taudb_get_column_name(res, j), "sum_exclusive_squared") == 0) {
        timer_value->sum_exclusive_squared = atof(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "inclusive_per_call") == 0) {
        // ignore this
        continue;
      } else if (strcmp(taudb_get_column_name(res, j), "call") == 0) {
        // ignore this
        continue;
      } else if (strcmp(taudb_get_column_name(res, j), "subroutines") == 0) {
        // ignore this
        continue;
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(res, j));
        taudb_exit_nicely(connection);
      }
    } 
    /*
    if (node < 0) {
      timer_value->thread = index;
    } else {
      timer_value->thread = (node * (trial->contexts_per_node * trial->threads_per_context)) +
                           (context * (trial->threads_per_context)) + 
                           thread;
    }
    timer_value->thread = taudb_get_thread(trial->threads, index);
    */

    //timer_value->key = taudb_create_hash_key_3(timer_value->thread, timer_str, metric_str);
    //HASH_ADD_KEYPTR(hh, timer_values, timer_value->key, strlen(timer_value->key), timer_value);
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return (timer_values);
}

TAUDB_TIMER_VALUE* taudb_query_all_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_all_timer_values(%p)\n", trial);
#endif
  return taudb_private_query_timer_values(connection, trial, NULL, NULL, NULL, FALSE);
}

TAUDB_TIMER_VALUE* taudb_query_all_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_all_timer_values(%p)\n", trial);
#endif
  return taudb_private_query_timer_values(connection, trial, NULL, NULL, NULL, TRUE);
}

TAUDB_TIMER_VALUE* taudb_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timer_values(%p,%p,%p,%p)\n", trial, timer_callpath, thread, metric);
#endif
  return taudb_private_query_timer_values(connection, trial, timer_callpath, thread, metric, FALSE);
}

TAUDB_TIMER_VALUE* taudb_query_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timer_values(%p,%p,%p,%p)\n", trial, timer_callpath, thread, metric);
#endif
  return taudb_private_query_timer_values(connection, trial, timer_callpath, thread, metric, TRUE);
}

TAUDB_TIMER_VALUE* taudb_get_timer_value(TAUDB_TIMER_VALUE* timer_values, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_value(%p,%p,%p,%p)\n", timer_values, timer_callpath, thread, metric);
#endif
  if (timer_values == NULL) {
    fprintf(stderr, "Error: timer_values parameter null. Please provide a valid set of timer_values.\n");
    return NULL;
  }
  if (timer_callpath == NULL) {
    fprintf(stderr, "Error: timer_callpath parameter null. Please provide a valid timer_callpath.\n");
    return NULL;
  }
  if (thread == NULL) {
    fprintf(stderr, "Error: thread parameter null. Please provide a valid thread.\n");
    return NULL;
  }
  if (metric == NULL) {
    fprintf(stderr, "Error: metric parameter null. Please provide a valid metric.\n");
    return NULL;
  }
  
  char *key = taudb_create_hash_key_3(thread->index, taudb_get_callpath_string(timer_callpath), metric->name);
  //printf("%s\n", key);

  TAUDB_TIMER_VALUE* timer_value = NULL;
  HASH_FIND_STR(timer_values, key, timer_value);
  return timer_value;
}
