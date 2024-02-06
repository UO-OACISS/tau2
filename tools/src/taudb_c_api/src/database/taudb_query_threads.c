#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_THREAD* taudb_query_threads_2005(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean derived, int* taudb_numItems) {
  if (derived) {
    *taudb_numItems = 2;
    TAUDB_THREAD* thread = taudb_create_threads(1);
    thread->id = 0;
    thread->trial = trial;
    thread->node_rank = TAUDB_MEAN_WITHOUT_NULLS;
    thread->context_rank = TAUDB_MEAN_WITHOUT_NULLS;
    thread->thread_rank = TAUDB_MEAN_WITHOUT_NULLS;
    thread->index = TAUDB_MEAN_WITHOUT_NULLS;;
	taudb_add_thread_to_trial(trial, thread);
    thread = taudb_create_threads(1);
    thread->id = 0;
    thread->trial = trial;
    thread->node_rank = TAUDB_TOTAL;
    thread->context_rank = TAUDB_TOTAL;
    thread->thread_rank = TAUDB_TOTAL;
    thread->index = TAUDB_TOTAL;
	taudb_add_thread_to_trial(trial, thread);
    return trial->threads;
  } else {
    int i, j, k;
    *taudb_numItems = trial->node_count * trial->contexts_per_node * trial->threads_per_context;
    int threadIndex = 0;
    for (i = 0; i < trial->node_count; i++)
    {
      for (j = 0; j < trial->contexts_per_node; j++)
      {
        for (k = 0; k < trial->threads_per_context; k++)
        {
          TAUDB_THREAD* thread = taudb_create_threads(1);
          thread->id = 0;
          thread->trial = trial;
          thread->node_rank = i;
          thread->context_rank = j;
          thread->thread_rank = k;
          thread->index = threadIndex;
	      taudb_add_thread_to_trial(trial, thread);
          threadIndex++;
        }
      } 
    }
    return trial->threads;
  }
}

TAUDB_THREAD* taudb_query_threads_2012(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean derived, int* taudb_numItems) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->threads != NULL) {
    *taudb_numItems = HASH_CNT(hh,trial->threads);
    return trial->threads;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  snprintf(my_query, sizeof(my_query), "select * from thread where trial = %d", trial->id);
  if (derived) {
    snprintf(my_query, sizeof(my_query), "%s and thread_index < 0 order by thread_index desc", my_query);
  } else {
    snprintf(my_query, sizeof(my_query), "%s and thread_index > -1 order by thread_index asc", my_query);
  }
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    TAUDB_THREAD* thread = taudb_create_threads(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        thread->id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "trial") == 0) {
        thread->trial = trial;
      } else if (strcmp(taudb_get_column_name(connection, j), "node_rank") == 0) {
        thread->node_rank = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "context_rank") == 0) {
        thread->context_rank = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "thread_rank") == 0) {
        thread->thread_rank = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "thread_index") == 0) {
        thread->index = atoi(taudb_get_value(connection, i, j));
      }
    } 
    HASH_ADD_INT(trial->threads, index, thread);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return trial->threads;
}

void taudb_add_thread_to_trial(TAUDB_TRIAL* trial, TAUDB_THREAD* thread) {
  HASH_ADD_INT(trial->threads, index, thread);
}

TAUDB_THREAD* taudb_query_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_query_threads_2005(connection, trial, FALSE,taudb_numItems);
  } else {
    return taudb_query_threads_2012(connection, trial, FALSE,taudb_numItems);
  }
}

TAUDB_THREAD* taudb_query_derived_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial,int* taudb_numItems) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_query_threads_2005(connection, trial, TRUE,taudb_numItems);
  } else {
    return taudb_query_threads_2012(connection, trial, TRUE,taudb_numItems);
  }
}

TAUDB_THREAD* taudb_get_thread(TAUDB_THREAD* threads, int index) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_thread(%p,%d)\n", threads, index);
#endif
  TAUDB_THREAD* thread;
  HASH_FIND_INT(threads, &(index), thread);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (thread == NULL) {
    TAUDB_THREAD *current, *tmp;
    HASH_ITER(hh, threads, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("THREAD: '%d'\n", current->index);
#endif
      if (current->index == index) {
        return current;
      }
    }
  }
#endif
  return thread;
}


void taudb_save_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  const char* my_query; 
	const char* statement_name;
  int nParams;
	
	const char * update_query = "update thread set trial=$1, node_rank=$2, context_rank=$3, thread_rank=$4, thread_index=$5 where id=$6;";
	const char * update_statement_name = "TAUDB_UPDATE_THREAD";
	const int update_nParams = 6;
	const char * insert_query = "insert into thread (trial, node_rank, context_rank, thread_rank, thread_index) values ($1, $2, $3, $4, $5);";
	const char * insert_statement_name = "TAUDB_INSERT_THREAD";
	const int insert_nParams = 5;
	
	if(update) {
		nParams = update_nParams;
	  my_query = update_query;
	  statement_name = update_statement_name;
	} else {
		nParams = insert_nParams;
	  my_query = insert_query;
	  statement_name = insert_statement_name;
	}
	
  taudb_prepare_statement(connection, statement_name, my_query, nParams);
  
  TAUDB_THREAD *thread, *tmp;
  HASH_ITER(hh, trial->threads, thread, tmp) {
    // make array of 6 character pointers
    const char* paramValues[6] = {0};
    char trialid[32] = {0};
    snprintf(trialid, sizeof(trialid),  "%d", trial->id);
    paramValues[0] = trialid;
    char node_rank[32] = {0};
    snprintf(node_rank, sizeof(node_rank),  "%d", thread->node_rank);
    paramValues[1] = node_rank;
    char context_rank[32] = {0};
    snprintf(context_rank, sizeof(context_rank),  "%d", thread->context_rank);
    paramValues[2] = context_rank;
    char thread_rank[32] = {0};
    snprintf(thread_rank, sizeof(thread_rank),  "%d", thread->thread_rank);
    paramValues[3] = thread_rank;
    char thread_index[32] = {0};
    snprintf(thread_index, sizeof(thread_index),  "%d", thread->index);
    paramValues[4] = thread_index;

	char id[32] = {0};
	if(update && thread->id > 0) {
		snprintf(id, sizeof(id),  "%d", thread->id);
		paramValues[5] = id;
	}

    int rows = taudb_execute_statement(connection, statement_name, nParams, paramValues);
		if(update && rows == 0) {
#ifdef TAUDB_DEBUG
			printf("Falling back to insert for update of thread.\n");
#endif
			/* updated row didn't exist; insert instead */
			thread->id = 0;
			taudb_prepare_statement(connection, insert_statement_name, insert_query, insert_nParams);
			taudb_execute_statement(connection, insert_statement_name, insert_nParams, paramValues);
		}
		
		if(!(update && thread->id >0)) {
	    taudb_execute_query(connection, "select currval('thread_id_seq');");
	    int nRows = taudb_get_num_rows(connection);
	    if (nRows == 1) {
	      thread->id = atoi(taudb_get_value(connection, 0, 0));
	      //printf("New Thread: %d\n", thread->id);
	    } else {
	      printf("Failed.\n");
	    }
		taudb_close_query(connection);
	  }
	}

  taudb_clear_result(connection);
}

TAUDB_THREAD* taudb_next_thread_by_index_from_trial(TAUDB_THREAD* current) {
  return current->hh.next;
}

