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
	    counters[i].trial = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    //counters[i].name = PQgetvalue(res, i, j);
		counters[i].name = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res,i,j))));
		strcpy(counters[i].name, PQgetvalue(res,i,j));
#ifdef TAUDB_DEBUG
        //printf("Got counter '%s'\n", counters[i].name);
#endif
	  } else if (strcmp(PQfname(res, j), "source_file") == 0) {
	    //counters[i].source_file = PQgetvalue(res, i, j);
		counters[i].source_file = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res,i,j))));
		strcpy(counters[i].source_file, PQgetvalue(res,i,j));
	  } else if (strcmp(PQfname(res, j), "line_number") == 0) {
	    counters[i].line_number = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "group_name") == 0) {
	    // tokenize the string, something like 'TAU_USER|MPI|...'
	    char* group_names = PQgetvalue(res, i, j);
		char* group = strtok(group_names, "|");
		if (group != NULL && strlen(group_names) > 0) {
#ifdef TAUDB_DEBUG
          //printf("Got counter groups '%s'\n", group_names);
#endif
		  counters[i].group_count = 1;
	      TAUDB_COUNTER_GROUP* groups = taudb_create_counter_groups(1);
		  groups[0].id = 0;
		  groups[0].counter = 0;
		  groups[0].name = group;
		  group = strtok(NULL, "|");
		  while (group != NULL) {
	        TAUDB_COUNTER_GROUP* groups = taudb_resize_counter_groups(counters[i].group_count+1, groups);
		    groups[counters[i].group_count].id = 0;
		    groups[counters[i].group_count].counter = 0;
		    groups[counters[i].group_count].name = group;
		    counters[i].group_count++;
		    group = strtok(NULL, "|");
		  }
		} else {
		  counters[i].group_count = 0;
		  counters[i].groups = NULL;
		}
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
