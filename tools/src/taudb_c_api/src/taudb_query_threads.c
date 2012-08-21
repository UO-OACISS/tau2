#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_THREAD* taudb_query_threads_2005(PGconn* connection, TAUDB_TRIAL* trial) {
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
        threads[threadIndex].trial = trial->id;
        threads[threadIndex].node_rank = i;
        threads[threadIndex].context_rank = j;
        threads[threadIndex].thread_rank = k;
        threads[threadIndex].process_id = 0; // should get this from the metadata
        threads[threadIndex].thread_id = 0; // should get this from the metadata
        threads[threadIndex].index = threadIndex;
        threadIndex++;
      }
    } 
  }
  return threads;
}

TAUDB_THREAD* taudb_query_threads_2012(PGconn* connection, TAUDB_TRIAL* trial) {
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
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        threads[i].id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "trial") == 0) {
        threads[i].trial = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "node_rank") == 0) {
        threads[i].node_rank = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "context_rank") == 0) {
        threads[i].context_rank = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread_rank") == 0) {
        threads[i].thread_rank = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "process_id") == 0) {
        threads[i].process_id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread_id") == 0) {
        threads[i].thread_id = atoi(PQgetvalue(res, i, j));
      }
    } 
    threads[i].index = i;
    threads[i].secondary_metadata_count = 0;
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
    return taudb_query_threads_2005(connection, trial);
  } else {
    return taudb_query_threads_2012(connection, trial);
  }
}
