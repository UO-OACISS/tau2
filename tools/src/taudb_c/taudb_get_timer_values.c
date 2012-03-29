#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_VALUE* taudb_get_timer_values(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, TAUDB_THREAD* thread, TAUDB_METRIC* metric) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_timer_values(%p,%p,%p,%p)\n", trial, timer, thread, metric);
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
    sprintf(my_query,"DECLARE myportal CURSOR FOR select interval_location_profile.* from interval_location_profile");
	char* conjoiner = "where";
	if (trial != NULL) {
      sprintf(my_query,"%s inner join interval_event on interval_location_profile.interval_event = interval_event.id where interval_event.trial = %d", my_query, trial->id);
	  conjoiner = "and";
	} 
	if (timer != NULL) {
      sprintf(my_query,"%s %s interval_event.id = %d", my_query, conjoiner, timer->id);
	  conjoiner = "and";
	}
	if (metric != NULL) {
      sprintf(my_query,"%s %s metric.id = %d", my_query, conjoiner, metric->id);
	  conjoiner = "and";
	}
	if (thread != NULL) {
      sprintf(my_query,"%s %s node = %d and context = %d and thread = %d", my_query, conjoiner, thread->node_rank, thread->context_rank, thread->thread_rank);
	}
  } else {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select * from measurement where trial = %d", trial->id);
  }
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
  TAUDB_TIMER_VALUE* timer_values = taudb_create_timer_values(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    timer_values[i].id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "interval_event") == 0) {
	    timer_values[i].timer = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "node") == 0) {
	    node = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "context") == 0) {
	    context = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "thread") == 0) {
	    thread = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "metric") == 0) {
	    timer_values[i].metric = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "inclusive_percentage") == 0) {
	    timer_values[i].inclusive_percentage = atof(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "exclusive_percentage") == 0) {
	    timer_values[i].exclusive_percentage = atof(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "inclusive") == 0) {
	    timer_values[i].inclusive = atof(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "exclusive") == 0) {
	    timer_values[i].exclusive = atof(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "sum_exclusive_squared") == 0) {
	    timer_values[i].sum_exclusive_squared = atof(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "inclusive_per_call") == 0) {
	    // ignore this
	    continue;
	  } else if (strcmp(PQfname(res, j), "call") == 0) {
	    // TODO do something with it!
	    continue;
	  } else if (strcmp(PQfname(res, j), "subroutines") == 0) {
	    // TODO do something with it!
	    continue;
	  } else {
	    printf("Error: unknown column '%s'\n", PQfname(res, j));
	    taudb_exit_nicely();
	  }
	  timer_values[i].thread = (node * (trial->contexts_per_node * trial->threads_per_context)) +
	                           (context * (trial->threads_per_context)) + 
							   thread;
	} 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  
  return (timer_values);
}
