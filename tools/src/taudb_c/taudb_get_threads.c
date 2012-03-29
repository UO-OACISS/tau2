#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef TAUDB_PERFDMF
TAUDB_THREAD* taudb_get_threads(TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_threads(%p)\n", trial);
#endif
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
		threadIndex++;
	  }
	} 
  }
  return threads;
}
#else

TAUDB_THREAD* taudb_get_threads(TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_threads(%d)\n", trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  /* Start a transaction block */
  res = PQexec(_taudb_connection, "BEGIN");
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "BEGIN command failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    taudb_exit_nicely();
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
  res = PQexec(_taudb_connection, my_query);
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "DECLARE CURSOR failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    taudb_exit_nicely();
  }
  PQclear(res);

  res = PQexec(_taudb_connection, "FETCH ALL in myportal");
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    fprintf(stderr, "FETCH ALL failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    taudb_exit_nicely();
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
	threads[i].secondary_metadata_count = 0;
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  
  return threads;
}

#endif
