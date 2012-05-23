#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_THREAD* taudb_query_threads_2005(PGconn* connection, TAUDB_TRIAL* trial, boolean derived) {
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
          threads[threadIndex].id = 0;
          threads[threadIndex].trial = trial;
          threads[threadIndex].node_rank = i;
          threads[threadIndex].context_rank = j;
          threads[threadIndex].thread_rank = k;
          threads[threadIndex].index = threadIndex;
	      HASH_ADD_INT(threads, index, &(threads[threadIndex]));
          threadIndex++;
        }
      } 
    }
    return threads;
  }
}

TAUDB_THREAD* taudb_query_threads_2012(PGconn* connection, TAUDB_TRIAL* trial, boolean derived) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  PGresult *res;
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

  /* Start a transaction block */
  res = PQexec(connection, "BEGIN");
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "BEGIN command failed: %s", PQerrorMessage(connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }

  /*
   * Should PQclear PGresult whenever it is no longer needed to avoid
   * memory leaks
   */
  PQclear(res);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from thread where trial = %d", trial->id);
  if (derived) {
    sprintf(my_query,"%s and thread_index < 0 order by thread_index desc", my_query);
  } else {
    sprintf(my_query,"%s and thread_index > -1 order by thread_index asc", my_query);
  }
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
#endif
  res = PQexec(connection, my_query);
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "DECLARE CURSOR failed: %s", PQerrorMessage(connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }
  PQclear(res);

  res = PQexec(connection, "FETCH ALL in myportal");
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    fprintf(stderr, "FETCH ALL failed: %s", PQerrorMessage(connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }

  int nRows = PQntuples(res);
  TAUDB_THREAD* threads = taudb_create_threads(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    TAUDB_THREAD* thread = &(threads[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        thread->id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "trial") == 0) {
        thread->trial = trial;
      } else if (strcmp(PQfname(res, j), "node_rank") == 0) {
        thread->node_rank = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "context_rank") == 0) {
        thread->context_rank = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread_rank") == 0) {
        thread->thread_rank = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread_index") == 0) {
        thread->index = atoi(PQgetvalue(res, i, j));
      }
    } 
    thread->secondary_metadata_count = 0;
	HASH_ADD_INT(threads, index, thread);
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return threads;
}

TAUDB_THREAD* taudb_query_threads(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_query_threads(%p)\n", trial);
#endif
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_query_threads_2005(connection, trial, FALSE);
  } else {
    return taudb_query_threads_2012(connection, trial, FALSE);
  }
}

TAUDB_THREAD* taudb_query_derived_threads(PGconn* connection, TAUDB_TRIAL* trial) {
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
