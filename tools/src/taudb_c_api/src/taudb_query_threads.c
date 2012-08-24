#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_THREAD* taudb_query_threads_2005(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean derived) {
  if (derived) {
    taudb_numItems = 2;
    TAUDB_THREAD* threads = taudb_create_threads(taudb_numItems);
    threads[0].id = 0;
    threads[0].trial = trial;
    threads[0].node_rank = TAUDB_MEAN_WITHOUT_NULLS;
    threads[0].context_rank = TAUDB_MEAN_WITHOUT_NULLS;
    threads[0].thread_rank = TAUDB_MEAN_WITHOUT_NULLS;
    threads[0].index = TAUDB_MEAN_WITHOUT_NULLS;;
	HASH_ADD_INT(threads, index, &(threads[0]));
    threads[1].id = 0;
    threads[1].trial = trial;
    threads[1].node_rank = TAUDB_TOTAL;
    threads[1].context_rank = TAUDB_TOTAL;
    threads[1].thread_rank = TAUDB_TOTAL;
    threads[1].index = TAUDB_TOTAL;
	HASH_ADD_INT(threads, index, &(threads[1]));
    return threads;
  } else {
    int i, j, k;
    taudb_numItems = trial->node_count * trial->contexts_per_node * trial->threads_per_context;
    TAUDB_THREAD* threads = taudb_create_threads(taudb_numItems);
    int threadIndex = 0;
    for (i = 0; i < trial->node_count; i++)
    {
      for (j = 0; j < trial->contexts_per_node; j++)
      {
        for (k = 0; k < trial->threads_per_context; k++)
        {
          TAUDB_THREAD* thread = &(threads[threadIndex]);
          thread->id = 0;
          thread->trial = trial;
          thread->node_rank = i;
          thread->context_rank = j;
          thread->thread_rank = k;
          thread->index = threadIndex;
	      //HASH_ADD_INT(threads, index, thread);
		  HASH_ADD(hh,threads,index,sizeof(int),thread);
          threadIndex++;
        }
      } 
    }
    return threads;
  }
}

TAUDB_THREAD* taudb_query_threads_2012(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean derived) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  void *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->threads != NULL && trial->thread_count > 0) {
    taudb_numItems = trial->thread_count;
    return trial->threads;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  sprintf(my_query,"select * from thread where trial = %d", trial->id);
  if (derived) {
    sprintf(my_query,"%s and thread_index < 0 order by thread_index desc", my_query);
  } else {
    sprintf(my_query,"%s and thread_index > -1 order by thread_index asc", my_query);
  }
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  TAUDB_THREAD* threads = taudb_create_threads(nRows);
  taudb_numItems = nRows;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    TAUDB_THREAD* thread = &(threads[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
        thread->id = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "trial") == 0) {
        thread->trial = trial;
      } else if (strcmp(taudb_get_column_name(res, j), "node_rank") == 0) {
        thread->node_rank = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "context_rank") == 0) {
        thread->context_rank = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "thread_rank") == 0) {
        thread->thread_rank = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "thread_index") == 0) {
        thread->index = atoi(taudb_get_value(res, i, j));
      }
    } 
    thread->secondary_metadata_count = 0;
	HASH_ADD_INT(threads, index, thread);
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return threads;
}

TAUDB_THREAD* taudb_query_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_query_threads_2005(connection, trial, FALSE);
  } else {
    return taudb_query_threads_2012(connection, trial, FALSE);
  }
}

TAUDB_THREAD* taudb_query_derived_threads(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_query_threads_2005(connection, trial, TRUE);
  } else {
    return taudb_query_threads_2012(connection, trial, TRUE);
  }
}

TAUDB_THREAD* taudb_get_thread(TAUDB_THREAD* threads, int index) {
  TAUDB_THREAD* thread;
  HASH_FIND_INT(threads, &(index), thread);
  return thread;
}
