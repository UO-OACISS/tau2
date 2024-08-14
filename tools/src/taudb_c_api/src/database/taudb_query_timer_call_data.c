#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_CALL_DATA* taudb_private_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, boolean derived, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_call_data(%p,%p,%p)\n", trial, timer_callpath, thread);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  
  // if the Trial already has the callpath data, return it.
  if (trial->timer_call_data_by_id != NULL) {
    *taudb_numItems = HASH_CNT(hh1,trial->timer_callpaths_by_id);
    return trial->timer_call_data_by_id;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[1024];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    snprintf(my_query, sizeof(my_query), "select distinct node, context, thread, call, subroutines, interval_event from interval_location_profile ilp left outer join interval_event ie on ilp.interval_event = ie.id");
    snprintf(my_query, sizeof(my_query), "%s where ie.trial = %d", my_query, trial->id);
    if (timer_callpath != NULL) {
      snprintf(my_query, sizeof(my_query), "%s and ie.id = %d", my_query, timer_callpath->id);
    }
    if (thread != NULL) {
      snprintf(my_query, sizeof(my_query), "%s and node = %d and context = %d and thread = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
  } else {
    //sprintf(my_query,"select * from timer where trial = %d", trial->id);
    snprintf(my_query, sizeof(my_query), "select h.node_rank as node, h.context_rank as context, h.thread_rank as thread, h.thread_index as index, td.calls as call, td.subroutines as subroutines, td.timer_callpath, td.id from timer_call_data td inner join thread h on td.thread = h.id");
    snprintf(my_query, sizeof(my_query), "%s where h.trial = %d", my_query, trial->id);
    if (timer_callpath != NULL) {
      snprintf(my_query, sizeof(my_query), "%s and td.timer_callpath = %d", my_query, timer_callpath->id);
    }
    if (thread != NULL) {
      snprintf(my_query, sizeof(my_query), "%s and h.node_rank = %d and h.context_rank = %d and h.thread_rank = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
    if (derived) {
      snprintf(my_query, sizeof(my_query), "%s and h.thread_index < 0 order by h.thread_index desc", my_query);
    } else {
      snprintf(my_query, sizeof(my_query), "%s and h.thread_index > -1 order by h.thread_index asc", my_query);
    }
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
    int index = 0;
    TAUDB_TIMER_CALL_DATA* timer_call_datum = taudb_create_timer_call_data(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        timer_call_datum->id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "interval_event") == 0) {
        timer_call_datum->key.timer_callpath = taudb_get_timer_callpath_by_id(trial->timer_callpaths_by_id, atoi(taudb_get_value(connection, i, j)));
      } else if (strcmp(taudb_get_column_name(connection, j), "timer_callpath") == 0) {
        timer_call_datum->key.timer_callpath = taudb_get_timer_callpath_by_id(trial->timer_callpaths_by_id, atoi(taudb_get_value(connection, i, j)));
      } else if (strcmp(taudb_get_column_name(connection, j), "node") == 0) {
        node = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "context") == 0) {
        context = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "thread") == 0) {
        thread = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "index") == 0) {
        index = atoi(taudb_get_value(connection, i, j));
        timer_call_datum->key.thread = taudb_get_thread(trial->threads, index);
      } else if (strcmp(taudb_get_column_name(connection, j), "call") == 0) {
        timer_call_datum->calls = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "subroutines") == 0) {
        timer_call_datum->subroutines = atoi(taudb_get_value(connection, i, j));
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
    } 
    if (timer_call_datum->key.thread == NULL) {
      int thread_index = node * trial->contexts_per_node * trial->threads_per_context;
      thread_index += context * trial->threads_per_context;
      thread_index += thread;
      timer_call_datum->key.thread = taudb_get_thread(trial->threads, thread_index);
    }
    timer_call_datum->key.timestamp = NULL; // for now
    taudb_add_timer_call_data_to_trial(trial, timer_call_datum);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);
  
  return (trial->timer_call_data_by_id);
}

void taudb_add_timer_call_data_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER_CALL_DATA* timer_call_data) {
  if (timer_call_data->id > 0) {
    HASH_ADD(hh1, trial->timer_call_data_by_id, id, sizeof(int), timer_call_data);
  }
  HASH_ADD(hh2, trial->timer_call_data_by_key, key, sizeof(TAUDB_TIMER_CALL_DATA_KEY), timer_call_data);
}

// convenience method
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
  return taudb_query_timer_call_data(connection, trial, NULL, NULL, taudb_numItems);
}

// convenience method
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
  return taudb_query_timer_call_data_stats(connection, trial, NULL, NULL, taudb_numItems);
}

// for getting call_datas for real threads
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, int* taudb_numItems) {
  return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, FALSE, taudb_numItems);
} 

// for getting call_datas for derived threads
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data_stats(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, int* taudb_numItems) {
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, TRUE, taudb_numItems);
  } else {
    return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, TRUE, taudb_numItems);
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
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer_call_datum == NULL) {
    TAUDB_TIMER_CALL_DATA *current, *tmp;
    HASH_ITER(hh2, timer_call_data, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("TIMER CALL DATA KEY: '%s',%d,%p\n", current->key.timer_callpath->name, current->key.thread->index, current->key.timestamp);
#endif
      if ((current->key.timer_callpath == callpath) && 
	      (current->key.thread == thread) && 
		  ((current->key.timestamp == NULL && timestamp == NULL) || 
		   (strcmp(current->key.timestamp, timestamp) == 0))) {
        return current;
      }
    }
  }
#endif
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
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer_call_datum == NULL) {
    TAUDB_TIMER_CALL_DATA *current, *tmp;
    HASH_ITER(hh2, timer_call_data, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf("TIMER CALL DATA KEY: (%p,%p,%p)\n", current->key.timer_callpath, current->key.thread, current->key.timestamp);
#endif
      if (current->id == id) {
        return current;
      }
    }
  }
#endif
  return timer_call_datum;
}

extern void taudb_save_timer_call_data(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {	
		const char* my_query;
		const char* statement_name;
		int nParams;
		
		const char* insert_query = "insert into timer_call_data (timer_callpath, thread, calls, subroutines, time_range) values($1, $2, $3, $4, $5);";
    const char* insert_statement_name = "TAUDB_INSERT_TIMER_CALL_DATA";
		const int insert_nParams = 5;
		const char* update_query = "update timer_call_data set timer_callpath=$1, thread=$2, calls=$3, subroutines=$4, time_range=$5 where id=$6";
		const char* update_statement_name = "TAUDB_UPDATE_TIMER_CALL_DATA";
		const int update_nParams = 6;
		
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
    TAUDB_TIMER_CALL_DATA *timer_call_data, *tmp;
    HASH_ITER(hh2, trial->timer_call_data_by_key, timer_call_data, tmp) {
      const char* paramValues[6] = {0};
      char timer_callpath_id[32] = {0};
      snprintf(timer_callpath_id, sizeof(timer_callpath_id),  "%d", timer_call_data->key.timer_callpath->id);
      paramValues[0] = timer_callpath_id;
		char thread_id[32] = {0};
		snprintf(thread_id, sizeof(thread_id),  "%d", timer_call_data->key.thread->id);
		paramValues[1] = thread_id;
		char calls[32] = {0};
		snprintf(calls, sizeof(calls),  "%d", timer_call_data->calls);
		paramValues[2] = calls;
		char subroutines[32] = {0};
		snprintf(subroutines, sizeof(subroutines),  "%d", timer_call_data->subroutines);
		paramValues[3] = subroutines;
		paramValues[4] = NULL; // TODO: Update this when support for saving time ranges is added
	  
		char id[32] = {0};
		if(update && timer_call_data->id > 0) {
			snprintf(id, sizeof(id),  "%d", timer_call_data->id);
			paramValues[5] = id;
		}
		
	    int rows = taudb_execute_statement(connection, statement_name, nParams, paramValues);
			if(update && rows == 0) {
#ifdef TAUDB_DEBUG
				printf("Falling back to insert for update of timer call data.\n");
#endif
				/* updated row didn't exist; insert instead */
				timer_call_data->id = 0;
				taudb_prepare_statement(connection, insert_statement_name, insert_query, insert_nParams);
				taudb_execute_statement(connection, insert_statement_name, insert_nParams, paramValues);
			}
      
			if(!(update && timer_call_data->id > 0)) {
				taudb_execute_query(connection, "select currval('timer_call_data_id_seq');");
				
	      int nRows = taudb_get_num_rows(connection);
	      if (nRows == 1) {
	        timer_call_data->id = atoi(taudb_get_value(connection, 0, 0));
	        //printf("New Timer Call Data: %d\n", timer_call_data->id);
	      } else {
	        printf("Failed.\n");
	      }
	  	  taudb_close_query(connection);
	    }
		}
    taudb_clear_result(connection);
}

TAUDB_TIMER_CALL_DATA* taudb_next_timer_call_data_by_key_from_trial(TAUDB_TIMER_CALL_DATA* current) {
  return current->hh2.next;
}

TAUDB_TIMER_CALL_DATA* taudb_next_timer_call_data_by_id_from_trial(TAUDB_TIMER_CALL_DATA* current) {
  return current->hh2.next;
}

