#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_METRIC* taudb_get_metrics(TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_metrics(%p)\n", trial);
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
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from metric where trial = %d", trial->id);
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
  int offset = 0;
  if (taudb_version == TAUDB_2005_SCHEMA) {
    taudb_numItems = nRows+2;
	offset = 2;
  } else {
    taudb_numItems = nRows;
  }

  TAUDB_METRIC* metrics = taudb_create_metrics(taudb_numItems);

  if (taudb_version == TAUDB_2005_SCHEMA) {
    metrics[0].id = 0;
    metrics[0].trial = 0;
    metrics[0].name = "calls";
    metrics[0].derived = FALSE;
    metrics[1].id = 0;
    metrics[2].trial = 0;
    metrics[3].name = "subroutines";
    metrics[4].derived = FALSE;
  }

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    metrics[i+offset].id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "trial") == 0) {
	    metrics[i+offset].trial = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    metrics[i+offset].name = PQgetvalue(res, i, j);
	  } else if (strcmp(PQfname(res, j), "derived") == 0) {
	    metrics[i+offset].derived = atoi(PQgetvalue(res, i, j));
	  } else {
	    printf("Error: unknown column '%s'\n", PQfname(res, j));
	    taudb_exit_nicely();
	  }
	} 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  
  return (metrics);
}
