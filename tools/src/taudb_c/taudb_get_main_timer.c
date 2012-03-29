#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER* taudb_get_main_timer(TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_main_timer(%p)\n", trial);
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
  char my_query[1024];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select interval_location_profile.inclusive, interval_event.* from interval_location_profile left outer join interval_event on interval_location_profile.interval_event = interval_event.id where interval_event.trial = %d order by 1 desc limit 1", trial->id);
  } else {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select * from measurement where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG_DEBUG
  printf("Query: %s\n", my_query);
#endif
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
  TAUDB_TIMER* timers = taudb_create_timers(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    timers[i].id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "trial") == 0) {
	    timers[i].trial = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    timers[i].name = PQgetvalue(res, i, j);
#ifdef TAUDB_DEBUG
        //printf("Got timer '%s'\n", timers[i].name);
#endif
	  } else if (strcmp(PQfname(res, j), "source_file") == 0) {
	    timers[i].source_file = PQgetvalue(res, i, j);
	  } else if (strcmp(PQfname(res, j), "line_number") == 0) {
	    timers[i].line_number = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "line_number_end") == 0) {
	    timers[i].line_number_end = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "column_number") == 0) {
	    timers[i].column_number = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "column_number_end") == 0) {
	    timers[i].column_number_end = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "group_name") == 0) {
	    // tokenize the string, something like 'TAU_USER|MPI|...'
	    char* group_names = PQgetvalue(res, i, j);
		char* group = strtok(group_names, "|");
		if (group != NULL && (strlen(group_names) > 0)) {
#ifdef TAUDB_DEBUG
          //printf("Got timer groups '%s'\n", group_names);
#endif
		  timers[i].group_count = 1;
	      TAUDB_TIMER_GROUP* groups = taudb_create_timer_groups(1);
		  groups[0].id = 0;
		  groups[0].timer = 0;
		  groups[0].name = group;
		  group = strtok(NULL, "|");
		  while (group != NULL) {
	        TAUDB_TIMER_GROUP* groups = taudb_resize_timer_groups(timers[i].group_count+1, groups);
		    groups[timers[i].group_count].id = 0;
		    groups[timers[i].group_count].timer = 0;
		    groups[timers[i].group_count].name = group;
		    timers[i].group_count++;
		    group = strtok(NULL, "|");
		  }
		} else {
		  timers[i].group_count = 0;
		  timers[i].groups = NULL;
		}
	  } else if (strcmp(PQfname(res, j), "inclusive") == 0) {
	    continue;
	  } else {
	    printf("Error: unknown column '%s'\n", PQfname(res, j));
	    taudb_exit_nicely();
	  }
	  // TODO - Populate the rest properly?
	} 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  
  return (timers);
}
