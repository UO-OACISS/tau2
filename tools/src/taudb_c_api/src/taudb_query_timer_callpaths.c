#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_CALLPATH* taudb_query_timer_callpaths(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_callpaths(%p,%p)\n", trial, timer);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  
  // if the Trial already has the callpath data, return it.
  if (trial->timer_callpaths != NULL && trial->timer_callpath_count > 0) {
    taudb_numItems = trial->timer_callpath_count;
    return trial->timer_callpaths;
  }

  if (taudb_version == TAUDB_2005_SCHEMA) {
  /*
    sprintf(my_query,"DECLARE myportal CURSOR FOR select ilp.node, ilp.context, ilp.thread, ilp.call, ilp.subroutines, ie.name as timer_name from interval_location_profile ilp inner join interval_event ie on ilp.interval_event = ie.id left outer join metric m on ilp.metric = m.id");
    sprintf(my_query,"%s where ie.trial = %d", my_query, trial->id);
    if (timer != NULL) {
      sprintf(my_query,"%s and ie.id = %d", my_query, timer->id);
    }
    if (thread != NULL) {
      sprintf(my_query,"%s and node = %d and context = %d and thread = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
	// we need just one metric, but from this trial
    sprintf(my_query,"%s and m.id = (select max(id) from metric where trial = %d)", my_query, trial->id);
	*/
	fprintf(stderr, "TODO: BUILD THE CALLPATHS FROM THE INTERVAL_EVENT NAMES!\n");
	return NULL;
  } 

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
  char my_query[1024];
  //sprintf(my_query,"DECLARE myportal CURSOR FOR select * from timer where trial = %d", trial->id);
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from timer_callpath tc inner join timer t on tc.timer = t.id");
  if (timer != NULL) {
    sprintf(my_query,"%s inner join timer t on tc.timer = t.id", my_query);
  }
  sprintf(my_query,"%s where t.trial = %d", my_query, trial->id);
  if (timer != NULL) {
    sprintf(my_query,"%s and t.id = %d", my_query, timer->id);
  }
  sprintf(my_query,"%s order by parent", my_query, timer->id);
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
  TAUDB_TIMER_CALLPATH* timer_callpaths = taudb_create_timer_callpaths(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    TAUDB_TIMER_CALLPATH* timer_callpath = &(timer_callpaths[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        timer_callpath->id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "timer") == 0) {
        // timer_callpath->timer = atoi(PQgetvalue(res, i, j));
		fprintf(stderr, "TODO: need to lookup the timer for a reference!\n");
      } else if (strcmp(PQfname(res, j), "parent") == 0) {
        TAUDB_TIMER_CALLPATH* parent = NULL;
        int parent_id = atoi(PQgetvalue(res, i, j));
        HASH_FIND_INT(timer_callpaths, &(parent_id), parent);
		timer_callpath->parent = parent;
      } else {
        printf("Error: unknown column '%s'\n", PQfname(res, j));
        taudb_exit_nicely(connection);
      }
    } 

	HASH_ADD_INT(timer_callpaths, id, timer_callpath);
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return (timer_callpaths);
}

// convenience method
TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpaths(PGconn* connection, TAUDB_TRIAL* trial) {
  return taudb_query_timer_callpaths(connection, trial, NULL);
}

char* taudb_get_callpath_string(TAUDB_TIMER_CALLPATH* timer_callpath) {
  char* full_string = calloc(strlen(timer_callpath->timer->name) + 1, sizeof(char));
  strcpy(full_string, timer_callpath->timer->name);
  TAUDB_TIMER_CALLPATH* parent = timer_callpath->parent;
  while (parent != NULL) {
    // resize for "string -> string" with null terminator
    int new_length = strlen(full_string) + strlen(parent->timer->name) + 5;
    full_string = realloc(full_string, new_length);
	sprintf(full_string, "%s -> %s", parent->timer->name, full_string);
    parent = parent->parent;
  }
  return full_string;
}


