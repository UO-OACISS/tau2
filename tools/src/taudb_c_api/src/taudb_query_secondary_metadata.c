#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_secondary_metadata(%d, %p)\n", full, trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  char my_query[1024];
  if (trial != NULL) { // the user wants a specific trial, so get it
    sprintf(my_query,"DECLARE myportal CURSOR FOR select sm.*, t.name as timer_name, h.thread_index from secondary_metadata sm left outer join timer t on sm.timer = t.id left outer join thread h on sm.thread = h.id where sm.trial = %d", trial->id);
  } else {
    fprintf(stderr, "You don't want all the metadata. Please specify a trial.\n");
  }

  /*
   * Our test case here involves using a cursor, for which we must be
   * inside a transaction block.  We could do the whole thing with a
   * single PQexec() of "select * from table_name", but that's too
   * trivial to make a good example.
   */

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
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
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
  TAUDB_SECONDARY_METADATA* pm = taudb_create_secondary_metadata(nRows);

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    boolean is_array = FALSE;
	pm[i].child_count = 0;
	pm[i].num_values = 1;
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        pm[i].id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "trial") == 0) {
        //pm[i].trial = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread") == 0) {
        //pm[i].thread = atoi(PQgetvalue(res, i, j));
		fprintf(stderr, "TODO: ASSIGN THE THREAD FOR THE SECONDARY METADATA\n");
      } else if (strcmp(PQfname(res, j), "timer_call_data") == 0) {
        //pm[i].timer_call_data = atoi(PQgetvalue(res, i, j));
		fprintf(stderr, "TODO: ASSIGN THE TIMER_CALL_DATA FOR THE SECONDARY METADATA\n");
      } else if (strcmp(PQfname(res, j), "timer_call_data") == 0) {
      } else if (strcmp(PQfname(res, j), "parent") == 0) {
        //pm[i].parent = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "is_array") == 0) {
	    if (strcmp(PQgetvalue(res, i, j), "t") == 0) {
          is_array = TRUE;
		  fprintf(stderr, "WARNING! Array metadata not yet supported...\n");
		}
      } else if (strcmp(PQfname(res, j), "timer_name") == 0) {
        //pm[i].timer_name = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res, i, j))));
        //strcpy(pm[i].name, PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread_index") == 0) {
        //pm[i].thread_index = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "name") == 0) {
        pm[i].name = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res, i, j))));
        strcpy(pm[i].name, PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "value") == 0) {
        pm[i].value = (char**)(malloc(sizeof(char*)));
        pm[i].value[0] = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res,i,j))));
        strcpy(pm[i].value[0], PQgetvalue(res, i, j));
      } else {
	    fprintf(stderr,"Unknown secondary_metadata column: %s\n", PQfname(res, j));
      }
    } 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  taudb_numItems = nFields;

  return pm;
}
