#include "taudb_internal.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_CALL_DATA* taudb_private_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, boolean derived) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_call_data(%p,%p,%p)\n", trial, timer_callpath, thread);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  
  // if the Trial already has the callpath data, return it.
  if (trial->timer_call_data_by_id != NULL) {
    taudb_numItems = trial->timer_callpath_count;
    return trial->timer_call_data_by_id;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[1024];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"select distinct node, context, thread, call, subroutines, interval_event from interval_location_profile ilp left outer join interval_event ie on ilp.interval_event = ie.id");
    sprintf(my_query,"%s where ie.trial = %d", my_query, trial->id);
    if (timer_callpath != NULL) {
      sprintf(my_query,"%s and ie.id = %d", my_query, timer_callpath->id);
    }
    if (thread != NULL) {
      sprintf(my_query,"%s and node = %d and context = %d and thread = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
  } else {
    //sprintf(my_query,"select * from timer where trial = %d", trial->id);
    sprintf(my_query,"select h.node_rank as node, h.context_rank as context, h.thread_rank as thread, h.thread_index as index, td.calls as call, td.subroutines as subroutines, td.timer_callpath from timer_call_data td inner join thread h on tc.thread = h.id");
    sprintf(my_query,"%s where t.trial = %d", my_query, trial->id);
    if (timer_callpath != NULL) {
      sprintf(my_query,"%s and td.timer_callpath = %d", my_query, timer_callpath->id);
    }
    if (thread != NULL) {
      sprintf(my_query,"%s and h.node_rank = %d and h.context_rank = %d and h.thread_rank = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
    if (derived) {
      sprintf(my_query,"%s and h.thread_index < 0 order by h.thread_index desc", my_query);
    } else {
      sprintf(my_query,"%s and h.thread_index > -1 order by h.thread_index asc", my_query);
    }
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  taudb_numItems = nRows;
  trial->timer_call_data_count = taudb_numItems;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
    int index = 0;
    TAUDB_TIMER_CALL_DATA* timer_call_datum = calloc(1, sizeof(TAUDB_TIMER_CALL_DATA));
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
        timer_call_datum->id = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "interval_event") == 0) {
        timer_call_datum->key.timer_callpath = taudb_get_timer_callpath_by_id(trial->timer_callpaths_by_id, atoi(taudb_get_value(res, i, j)));
      } else if (strcmp(taudb_get_column_name(res, j), "timer_callpath") == 0) {
        timer_call_datum->key.timer_callpath = taudb_get_timer_callpath_by_id(trial->timer_callpaths_by_id, atoi(taudb_get_value(res, i, j)));
      } else if (strcmp(taudb_get_column_name(res, j), "node") == 0) {
        node = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "context") == 0) {
        context = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "thread") == 0) {
        thread = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "index") == 0) {
        index = atoi(taudb_get_value(res, i, j));
        timer_call_datum->key.thread = taudb_get_thread(trial->threads, index);
      } else if (strcmp(taudb_get_column_name(res, j), "call") == 0) {
        timer_call_datum->calls = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "subroutines") == 0) {
        timer_call_datum->subroutines = atoi(taudb_get_value(res, i, j));
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(res, j));
        taudb_exit_nicely(connection);
      }
    } 
    HASH_ADD(hh1, trial->timer_call_data_by_id, id, sizeof(int), timer_call_datum);
    if (timer_call_datum->key.thread == NULL) {
      int thread_index = node * trial->contexts_per_node * trial->threads_per_context;
      thread_index += context * trial->threads_per_context;
      thread_index += thread;
      timer_call_datum->key.thread = taudb_get_thread(trial->threads, thread_index);
    }
    timer_call_datum->key.timestamp = NULL; // for now
    HASH_ADD(hh2, trial->timer_call_data_by_key, key, sizeof(TAUDB_TIMER_CALL_DATA_KEY), timer_call_datum);
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);
  
  return (trial->timer_call_data_by_id);
}

// convenience method
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
  return taudb_query_timer_call_data(connection, trial, NULL, NULL);
}

// convenience method
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
  return taudb_query_timer_call_data_stats(connection, trial, NULL, NULL);
}

// for getting call_datas for real threads
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread) {
  return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, FALSE);
} 

// for getting call_datas for derived threads
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread) {
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, TRUE);
  } else {
    return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, TRUE);
  }
}

// convenience method for indexing into the hash
TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data_by_key(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_TIMER_CALLPATH* callpath, TAUDB_THREAD* thread, char* timestamp) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_call_data(%p,%p,%p)\n", timer_call_data, callpath, thread);
#endif
  if (timer_call_data == NULL) {
    fprintf(stderr, "Error: timer_call_data parameter null. Please provide a valid set of timer_call_data.\n");
    return NULL;
  }
  if (callpath == NULL) {
    fprintf(stderr, "Error: callpath parameter null. Please provide a valid callpath.\n");
    return NULL;
  }
  if (thread == NULL) {
    fprintf(stderr, "Error: thread parameter null. Please provide a valid thread.\n");
    return NULL;
  }

  TAUDB_TIMER_CALL_DATA* timer_call_datum = NULL;

  TAUDB_TIMER_CALL_DATA_KEY key;
  memset(&key, 0, sizeof(TAUDB_TIMER_CALL_DATA_KEY));
  key.timer_callpath = callpath;
  key.thread = thread;
  key.timestamp = timestamp;
 
  HASH_FIND(hh2, timer_call_data, &key, sizeof(TAUDB_TIMER_CALL_DATA_KEY), timer_call_datum);
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer_call_datum == NULL) {
    TAUDB_TIMER_CALL_DATA *current, *tmp;
    HASH_ITER(hh2, timer_call_data, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("COUNTER VALUE: (%p,%p,%p)\n", current->key.timer_callpath, current->key.thread, current->key.timestamp);
#endif
      if ((current->key.timer_callpath->id == callpath->id) && (current->key.thread == thread) && ((current->key.timestamp == NULL && timestamp == NULL) || (strcmp(current->key.timestamp, timestamp) == 0))) {
        return current;
      }
    }
  }
  return timer_call_datum;
}

// convenience method for indexing into the hash
TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data_by_id(TAUDB_TIMER_CALL_DATA* timer_call_data, int id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_call_data(%p,%d)\n", timer_call_data, id);
#endif
  if (timer_call_data == NULL) {
    fprintf(stderr, "Error: timer_call_data parameter null. Please provide a valid set of timer_call_data.\n");
    return NULL;
  }
  if (id == 0) {
    fprintf(stderr, "Error: id parameter null. Please provide a valid id.\n");
    return NULL;
  }

  TAUDB_TIMER_CALL_DATA* timer_call_datum = NULL;

  HASH_FIND(hh1, timer_call_data, &id, sizeof(int), timer_call_datum);
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer_call_datum == NULL) {
    TAUDB_TIMER_CALL_DATA *current, *tmp;
    HASH_ITER(hh2, timer_call_data, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("COUNTER VALUE: (%p,%p,%p)\n", current->key.timer_callpath, current->key.thread, current->key.timestamp);
#endif
      if (current->id == id) {
        return current;
      }
    }
  }
  return timer_call_datum;
}
