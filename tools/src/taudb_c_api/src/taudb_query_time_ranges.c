#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIME_RANGE* taudb_query_time_ranges(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_time_ranges(%p)\n", trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->time_ranges != NULL && trial->time_range_count > 0) {
    taudb_numItems = trial->time_range_count;
    return trial->time_ranges;
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
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from time_range where trial = %d", trial->id);
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
  taudb_numItems = nRows;

  TAUDB_TIME_RANGE* time_ranges = taudb_create_time_ranges(taudb_numItems);

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    TAUDB_TIME_RANGE* time_range = &(time_ranges[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    time_range->id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "trial") == 0) {
	    //time_range->trial = trial;
	  } else if (strcmp(PQfname(res, j), "iteration_start") == 0) {
	    time_range->iteration_start = atoi(PQgetvalue(res,i,j));
	  } else if (strcmp(PQfname(res, j), "iteration_end") == 0) {
	    time_range->iteration_end = atoi(PQgetvalue(res,i,j));
	  } else if (strcmp(PQfname(res, j), "time_start") == 0) {
	    time_range->time_start = atoll(PQgetvalue(res,i,j));
	  } else if (strcmp(PQfname(res, j), "time_end") == 0) {
	    time_range->time_end = atoll(PQgetvalue(res,i,j));
	  } else {
	    printf("Error: unknown column '%s'\n", PQfname(res, j));
	    taudb_exit_nicely(connection);
	  }
	} 
	HASH_ADD_INT(time_ranges, id, time_range);
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return (time_ranges);
}

TAUDB_TIME_RANGE* taudb_get_time_range(TAUDB_TIME_RANGE* time_ranges, const int id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_time_range(%p,%d)\n", time_ranges, id);
#endif
  if (time_ranges == NULL) {
    fprintf(stderr, "Error: time_range parameter null. Please provide a valid set of time_ranges.\n");
    return NULL;
  }
  if (id == 0) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIME_RANGE* time_range = NULL;
  HASH_FIND_INT(time_ranges, &id, time_range);
  return time_range;
}

