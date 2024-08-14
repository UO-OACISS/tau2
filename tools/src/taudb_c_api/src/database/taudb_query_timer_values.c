#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_VALUE* taudb_private_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric, boolean derived, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_values(%p,%p,%p,%p)\n", trial, timer_callpath, thread, metric);
#endif
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
    *taudb_numItems = trial->value_count;
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
    snprintf(my_query, sizeof(my_query), "select ilp.*, ie.name as timer_name, ilp.metric from interval_location_profile ilp inner join interval_event ie on ilp.interval_event = ie.id");
    char* conjoiner = "where";
    if (trial != NULL) {
      snprintf(my_query, sizeof(my_query), "%s where ie.trial = %d", my_query, trial->id);
      conjoiner = "and";
    } 
    if (timer_callpath != NULL) {
      fprintf(stderr, "TODO: THE CALLPATH IS NOT VALID IN PERFDMF\n");
      snprintf(my_query, sizeof(my_query), "%s %s ie.id = %d", my_query, conjoiner, timer_callpath->id);
      conjoiner = "and";
    }
    if (metric != NULL) {
      if ((strcmp(metric->name, "calls") == 0) ||
          (strcmp(metric->name, "subroutines") == 0)) {
          // we need just one metric, but from this trial
        snprintf(my_query, sizeof(my_query), "%s %s m.id = (select max(id) from metric where trial = %d)", my_query, conjoiner, trial->id);
      } else {
        snprintf(my_query, sizeof(my_query), "%s %s m.id = %d", my_query, conjoiner, metric->id);
      }
      conjoiner = "and";
    }
    if (thread != NULL) {
      snprintf(my_query, sizeof(my_query), "%s %s node = %d and context = %d and thread = %d", my_query, conjoiner, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
  } else {
    //sprintf(my_query,"select * from timer where trial = %d", trial->id);
    snprintf(my_query, sizeof(my_query), "select tv.*, h.thread_index as index from timer_value tv inner join timer_call_data td on tv.timer_call_data = td.id inner join timer_callpath tc on td.timer_callpath = tc.id left outer join thread h on td.thread = h.id");
    char* conjoiner = "where";
    if (trial != NULL) {
      snprintf(my_query, sizeof(my_query), "%s where h.trial = %d", my_query, trial->id);
      conjoiner = "and";
    } 
    if (timer_callpath != NULL) {
      snprintf(my_query, sizeof(my_query), "%s %s tc.id = %d", my_query, conjoiner, timer_callpath->id);
      conjoiner = "and";
    }
    if (metric != NULL) {
      snprintf(my_query, sizeof(my_query), "%s %s tv.metric = %d", my_query, conjoiner, metric->id);
      conjoiner = "and";
    }
    if (thread != NULL) {
      snprintf(my_query, sizeof(my_query), "%s %s h.thread_index = %d ", my_query, conjoiner, thread->index);
      conjoiner = "and";
    }
    if (derived) {
      snprintf(my_query, sizeof(my_query), "%s %s h.thread_index < 0 order by h.thread_index desc", my_query, conjoiner);
    } else {
      snprintf(my_query, sizeof(my_query), "%s %s h.thread_index > -1 order by h.thread_index asc", my_query, conjoiner);
    }
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  //TAUDB_TIMER_VALUE* timer_values = taudb_create_timer_values(nRows);
  TAUDB_TIMER_VALUE* timer_values = NULL;
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
    int index = 0;
    int metric_id = 0;
    int timer_call_data_id = 0;
    int timer_id = 0;
    char* timer_str;
    TAUDB_TIMER_VALUE* timer_value = taudb_create_timer_values(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "timer_call_data") == 0) {
        timer_call_data_id = atoi(taudb_get_value(connection, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(connection, j), "interval_event") == 0) {
        timer_id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "timer") == 0) {
        timer_id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "node") == 0) {
        node = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "context") == 0) {
        context = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "thread") == 0) {
        thread = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "index") == 0) {
        index = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "metric") == 0) {
        metric_id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "timer_name") == 0) {
        timer_str = taudb_get_value(connection, i, j);
      } else if (strcmp(taudb_get_column_name(connection, j), "metric") == 0) {
        if (metric != NULL)
          timer_value->metric = metric;
        else {
          timer_value->metric = taudb_get_metric_by_id(trial->metrics_by_id, atoi(taudb_get_value(connection, i, j)));
        }
// these two are the same
      } else if (strcmp(taudb_get_column_name(connection, j), "inclusive_percentage") == 0) {
        timer_value->inclusive_percentage = atof(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "inclusive_percent") == 0) {
        timer_value->inclusive_percentage = atof(taudb_get_value(connection, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(connection, j), "exclusive_percentage") == 0) {
        timer_value->exclusive_percentage = atof(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "exclusive_percent") == 0) {
        timer_value->exclusive_percentage = atof(taudb_get_value(connection, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(connection, j), "inclusive") == 0) {
        timer_value->inclusive = atof(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "inclusive_value") == 0) {
        timer_value->inclusive = atof(taudb_get_value(connection, i, j));
// these two are the same
      } else if (strcmp(taudb_get_column_name(connection, j), "exclusive") == 0) {
        timer_value->exclusive = atof(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "exclusive_value") == 0) {
        timer_value->exclusive = atof(taudb_get_value(connection, i, j));

      } else if (strcmp(taudb_get_column_name(connection, j), "sum_exclusive_squared") == 0) {
        timer_value->sum_exclusive_squared = atof(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "inclusive_per_call") == 0) {
        // ignore this
        continue;
      } else if (strcmp(taudb_get_column_name(connection, j), "call") == 0) {
        // ignore this
        continue;
      } else if (strcmp(taudb_get_column_name(connection, j), "subroutines") == 0) {
        // ignore this
        continue;
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
    } 
    TAUDB_TIMER_CALL_DATA* timer_call_data = NULL;
    if (taudb_version == TAUDB_2005_SCHEMA) {
      index = (node * (trial->contexts_per_node * trial->threads_per_context)) +
              (context * (trial->threads_per_context)) + 
              thread;
      TAUDB_THREAD* thread = taudb_get_thread(trial->threads, index);
      timer_value->metric = taudb_get_metric_by_id(trial->metrics_by_id, metric_id);
      TAUDB_TIMER_CALLPATH* timer_callpath = taudb_get_timer_callpath_by_id(trial->timer_callpaths_by_id, timer_id);
      // find the timer_call_data object
      timer_call_data = taudb_get_timer_call_data_by_key(trial->timer_call_data_by_key, timer_callpath, thread, NULL);
    } else {
      timer_value->metric = taudb_get_metric_by_id(trial->metrics_by_id, metric_id);
      timer_call_data = taudb_get_timer_call_data_by_id(trial->timer_call_data_by_id, timer_call_data_id);
    }

	if (timer_call_data == NULL) {
	  printf("Failed to find timer_call_data %d\n", timer_call_data_id);
	} else {
      taudb_add_timer_value_to_timer_call_data(timer_call_data, timer_value);
	}
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (timer_values);
}

void taudb_add_timer_value_to_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_TIMER_VALUE* timer_value) {
  HASH_ADD_KEYPTR(hh, timer_call_data->timer_values, timer_value->metric->name, strlen(timer_value->metric->name), timer_value);
  //HASH_ADD_KEYPTR(hh, timer_call_data->timer_values, timer_value->metric, sizeof(timer_value->metric), timer_value);
}

TAUDB_TIMER_VALUE* taudb_query_all_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_all_timer_values(%p)\n", trial);
#endif
  return taudb_private_query_timer_values(connection, trial, NULL, NULL, NULL, FALSE, taudb_numItems);
}

TAUDB_TIMER_VALUE* taudb_query_all_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_all_timer_values(%p)\n", trial);
#endif
  return taudb_private_query_timer_values(connection, trial, NULL, NULL, NULL, TRUE, taudb_numItems);
}

TAUDB_TIMER_VALUE* taudb_query_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timer_values(%p,%p,%p,%p)\n", trial, timer_callpath, thread, metric);
#endif
  return taudb_private_query_timer_values(connection, trial, timer_callpath, thread, metric, FALSE, taudb_numItems);
}

TAUDB_TIMER_VALUE* taudb_query_timer_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, TAUDB_METRIC* metric, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timer_values(%p,%p,%p,%p)\n", trial, timer_callpath, thread, metric);
#endif
  return taudb_private_query_timer_values(connection, trial, timer_callpath, thread, metric, TRUE, taudb_numItems);
}

TAUDB_TIMER_VALUE* taudb_get_timer_value(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_METRIC* metric) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_value(%p,%p)\n", timer_call_data, metric);
#endif
  if (timer_call_data == NULL) {
    fprintf(stderr, "Error: timer_callpath parameter null. Please provide a valid timer_callpath.\n");
    return NULL;
  }
  if (metric == NULL) {
    fprintf(stderr, "Error: metric parameter null. Please provide a valid metric.\n");
    return NULL;
  }
  
  TAUDB_TIMER_VALUE* timer_value = NULL;
  HASH_FIND(hh, timer_call_data->timer_values, metric->name, strlen(metric->name), timer_value);
  //HASH_FIND(hh, timer_call_data->timer_values, metric, sizeof(metric), timer_value);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer_value == NULL) {
    TAUDB_TIMER_VALUE *current, *tmp;
    HASH_ITER(hh, timer_call_data->timer_values, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("METRIC NAME: (%s)\n", current->metric->name);
#endif
      if (strcmp(current->metric->name, metric->name) == 0) {
        return current;
      }
    }
  }
#endif
  
  return timer_value;
}

extern void taudb_save_timer_values(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
	  const char* my_query;
		const char* statement_name;
		int nParams;
	
		const char* insert_query = "insert into timer_value (timer_call_data, metric, inclusive_value, exclusive_value, "
			"inclusive_percent, exclusive_percent, sum_exclusive_squared) values ($1, $2, $3, $4, $5, $6, $7);";
    const char* insert_statement_name = "TAUDB_INSERT_TIMER_VALUE";
		const int insert_nParams = 7;
		const char* update_query = "update timer_value set inclusive_value=$3, exclusive_value=$4, inclusive_percent=$5, "
			"exclusive_percent=$6, sum_exclusive_squared=$7 where timer_call_data=$1 and metric=$2";
		const char* update_statement_name = "TAUDB_UPDATE_TIMER_VALUE";
		const int update_nParams = 7;
		
		if(update) {
			my_query = update_query;
			statement_name = update_statement_name;
			nParams = update_nParams;
		} else {
			my_query = insert_query;
			statement_name = insert_statement_name;
			nParams = insert_nParams;
		}
		
    taudb_prepare_statement(connection, statement_name, my_query, nParams);
	
	/* the timer value lists are stored inside taudb_timer_call_data */
    TAUDB_TIMER_CALL_DATA *timer_call_data, *tmp;
    HASH_ITER(hh2, trial->timer_call_data_by_key, timer_call_data, tmp) {
	  TAUDB_TIMER_VALUE *timer_value, *tmp2;
	  HASH_ITER(hh, timer_call_data->timer_values, timer_value, tmp2) {
	      const char* paramValues[7] = {0};
	      char timer_call_data_id[32] = {0};
	      snprintf(timer_call_data_id, sizeof(timer_call_data_id),  "%d", timer_call_data->id);
	      paramValues[0] = timer_call_data_id;
		  
		  char metric_id[32] = {0};
		  snprintf(metric_id, sizeof(metric_id),  "%d", timer_value->metric->id);
		  paramValues[1] = metric_id;
		  
		  char inclusive_value[64] = {};
		  snprintf(inclusive_value, sizeof(inclusive_value),  "%31.31f", timer_value->inclusive);
		  paramValues[2] = inclusive_value;
		  
		  char exclusive_value[64] = {};
		  snprintf(exclusive_value, sizeof(exclusive_value),  "%31.31f", timer_value->exclusive);
		  paramValues[3] = exclusive_value;
		  
		  char inclusive_percent[64] = {};
		  snprintf(inclusive_percent, sizeof(inclusive_percent),  "%31.31f", timer_value->inclusive_percentage);
		  paramValues[4] = inclusive_percent;
		  
		  char exclusive_percent[64] = {};
		  snprintf(exclusive_percent, sizeof(exclusive_percent),  "%31.31f", timer_value->exclusive_percentage);
		  paramValues[5] = exclusive_percent;
		  
		  char sum_exclusive_squared[64] = {};
		  snprintf(sum_exclusive_squared, sizeof(sum_exclusive_squared),  "%31.31f", timer_value->sum_exclusive_squared);
		  paramValues[6] = sum_exclusive_squared;

	    int rows = taudb_execute_statement(connection, statement_name, nParams, paramValues);
		  //printf("New Timer Value: (%d, %d)\n", timer_call_data->id, timer_value->metric->id);
			if(update && rows == 0) {
#ifdef TAUDB_DEBUG
				printf("Falling back to insert for update of timer value.\n");
#endif
				/* updated row didn't exist; insert instead */
				taudb_prepare_statement(connection, insert_statement_name, insert_query, insert_nParams);
				taudb_execute_statement(connection, insert_statement_name, insert_nParams, paramValues);
			}

			

		  /* timer_values don't have ids, so there's nothing to update */
	    }
	}
    taudb_clear_result(connection);
  
}

TAUDB_TIMER_VALUE* taudb_next_timer_value_by_metric_from_timer_call_data(TAUDB_TIMER_VALUE* current) {
  return current->hh.next;
}

