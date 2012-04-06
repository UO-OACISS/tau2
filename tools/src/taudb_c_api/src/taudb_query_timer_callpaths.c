#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// convenience method
TAUDB_TIMER_CALLPATH* taudb_query_all_timer_callpaths(PGconn* connection, TAUDB_TRIAL* trial) {
  return taudb_query_timer_callpaths(connection, trial, NULL, NULL);
}

TAUDB_TIMER_CALLPATH* taudb_query_timer_callpaths(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timer_callpaths(%p,%p,%p)\n", trial, timer, thread);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  
  // if the Trial already has the callpath data, return it.
  if (trial->timer_callpaths != NULL && trial->callpath_count > 0) {
    taudb_numItems = trial->callpath_count;
    return trial->timer_callpaths;
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
  char my_query[1024];
  if (taudb_version == TAUDB_2005_SCHEMA) {
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
  } else {
    //sprintf(my_query,"DECLARE myportal CURSOR FOR select * from measurement where trial = %d", trial->id);
    fprintf(stderr, "Error: 2012 schema not supported yet.\n");
    return NULL;
  }
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
  //TAUDB_TIMER_CALLPATH* timer_callpaths = taudb_create_timer_callpaths(nRows);
  TAUDB_TIMER_CALLPATH* timer_callpaths = NULL;
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
	char* timer_str;
    TAUDB_TIMER_CALLPATH* timer_callpath = taudb_create_timer_callpaths(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        timer_callpath->id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "interval_event") == 0) {
        timer_callpath->timer = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "node") == 0) {
        node = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "context") == 0) {
        context = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread") == 0) {
        thread = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "timer_name") == 0) {
        timer_str = PQgetvalue(res, i, j);
      } else if (strcmp(PQfname(res, j), "call") == 0) {
        timer_callpath->calls = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "subroutines") == 0) {
        timer_callpath->subroutines = atoi(PQgetvalue(res, i, j));
      } else {
        printf("Error: unknown column '%s'\n", PQfname(res, j));
        taudb_exit_nicely(connection);
      }
    } 
    timer_callpath->thread = (node * (trial->contexts_per_node * trial->threads_per_context)) +
                          (context * (trial->threads_per_context)) + 
                          thread;

    char tmp_thread[100];
	sprintf(tmp_thread, "%d", timer_callpath->thread);
	timer_callpath->key = taudb_create_string(strlen(tmp_thread) + strlen(timer_str) + 2);
    sprintf(timer_callpath->key, "%d:%s", timer_callpath->thread, timer_str);
#ifdef TAUDB_DEBUG_DEBUG
    printf("NEW KEY: '%s'\n",timer_callpath->key);
#endif
	HASH_ADD_KEYPTR(hh, timer_callpaths, timer_callpath->key, strlen(timer_callpath->key), timer_callpath);
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

// convenience method for indexing into the hash
TAUDB_TIMER_CALLPATH* taudb_get_timer_callpath(TAUDB_TIMER_CALLPATH* timer_callpaths, TAUDB_TIMER* timer, TAUDB_THREAD* thread) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_callpaths(%p,%p,%p)\n", timer_callpaths, timer, thread);
#endif
  if (timer_callpaths == NULL) {
    fprintf(stderr, "Error: timer_callpaths parameter null. Please provide a valid set of timer_callpaths.\n");
    return NULL;
  }
  if (timer == NULL) {
    fprintf(stderr, "Error: timer parameter null. Please provide a valid timer.\n");
    return NULL;
  }
  if (timer_callpaths == NULL) {
    fprintf(stderr, "Error: thread parameter null. Please provide a valid thread.\n");
    return NULL;
  }
  char tmp_thread[10];
  sprintf(tmp_thread, "%d", thread->index);
  char *key = taudb_create_string(strlen(tmp_thread) + strlen(timer->full_name) + 2);
  sprintf(key, "%d:%s", thread->index, timer->full_name);
#ifdef TAUDB_DEBUG_DEBUG
  printf("'%d', '%s', Looking for key: %s\n", thread->index, timer->full_name, key);
#endif

  TAUDB_TIMER_CALLPATH* timer_callpath = NULL;
  HASH_FIND_STR(timer_callpaths, key, timer_callpath);
  return timer_callpath;
}
