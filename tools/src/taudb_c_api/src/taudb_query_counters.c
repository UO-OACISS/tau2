#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_COUNTER* taudb_query_counters(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_counters(%p)\n", trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  // if the Trial already has the data, return it.
  if (trial->counters != NULL && trial->counter_count > 0) {
    taudb_numItems = trial->counter_count;
    return trial->counters;
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
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select * from atomic_event where trial = %d", trial->id);
  } else {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select * from counter where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
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
  TAUDB_COUNTER* counters = taudb_create_counters(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    counters[i].id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "trial") == 0) {
	    counters[i].trial = trial;
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    //counters[i].name = PQgetvalue(res, i, j);
		counters[i].name = taudb_create_and_copy_string(PQgetvalue(res,i,j));
#ifdef TAUDB_DEBUG
        //printf("Got counter '%s'\n", counters[i].name);
#endif
	  } else {
	    printf("Error: unknown column '%s'\n", PQfname(res, j));
	    taudb_exit_nicely(connection);
	  }
	  // TODO - Populate the rest properly?
	} 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
res = PQexec(connection, "END");
  PQclear(res);
  
  return (counters);
}
