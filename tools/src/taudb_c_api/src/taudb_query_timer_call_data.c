#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_CALL_DATA* taudb_private_query_timer_call_data(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread, boolean derived) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_call_data(%p,%p,%p)\n", trial, timer_callpath, thread);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  
  // if the Trial already has the callpath data, return it.
  if (trial->timer_call_data != NULL && trial->timer_callpath_count > 0) {
    taudb_numItems = trial->timer_callpath_count;
    return trial->timer_call_data;
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
    if (timer_callpath != NULL) {
      sprintf(my_query,"%s and ie.id = %d", my_query, timer_callpath->id);
    }
    if (thread != NULL) {
      sprintf(my_query,"%s and node = %d and context = %d and thread = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
	// we need just one metric, but from this trial
    sprintf(my_query,"%s and m.id = (select max(id) from metric where trial = %d)", my_query, trial->id);
  } else {
    //sprintf(my_query,"DECLARE myportal CURSOR FOR select * from timer where trial = %d", trial->id);
    sprintf(my_query,"DECLARE myportal CURSOR FOR select h.node_rank as node, h.context_rank as context, h.thread_rank as thread, h.thread_index as index, td.calls as call, td.subroutines as subroutines, t.name as timer_name from timer_call_data td inner join timer_callpath tc on td.timer_callpath = tc.id inner join timer t on td.timer = t.id inner join thread h on tc.thread = h.id");
    sprintf(my_query,"%s where t.trial = %d", my_query, trial->id);
    if (timer_callpath != NULL) {
      sprintf(my_query,"%s and t.id = %d", my_query, timer_callpath->id);
    }
    if (thread != NULL) {
      sprintf(my_query,"%s and h.node_rank = %d and h.context_rank = %d and h.thread_rank = %d", my_query, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
    if (derived) {
      sprintf(my_query,"%s and h.thread_index < 0 order by h.thread_index desc", my_query);
    } else {
      sprintf(my_query,"%s and h.thread_index > -1 order by h.thread_index asc", my_query);
	}
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
  TAUDB_TIMER_CALL_DATA* timer_call_data = taudb_create_timer_call_data(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
	int index = 0;
	char* timer_str;
    TAUDB_TIMER_CALL_DATA* timer_call_datum = &(timer_call_data[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        timer_call_datum->id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "interval_event") == 0) {
        // timer_call_datum->timer = atoi(PQgetvalue(res, i, j));
		fprintf(stderr, "TODO: need to lookup the timer for a reference!\n");
      } else if (strcmp(PQfname(res, j), "node") == 0) {
        node = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "context") == 0) {
        context = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread") == 0) {
        thread = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "index") == 0) {
        index = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "timer_name") == 0) {
        timer_str = PQgetvalue(res, i, j);
      } else if (strcmp(PQfname(res, j), "call") == 0) {
        timer_call_datum->calls = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "subroutines") == 0) {
        timer_call_datum->subroutines = atoi(PQgetvalue(res, i, j));
      } else {
        printf("Error: unknown column '%s'\n", PQfname(res, j));
        taudb_exit_nicely(connection);
      }
    } 
	/*
	if (node < 0) {
	  timer_call_datum->thread = index;
	} else {
      timer_call_datum->thread = (node * (trial->contexts_per_node * trial->threads_per_context)) +
                            (context * (trial->threads_per_context)) + 
                            thread;
	}
	*/
	fprintf(stderr, "TODO: LOOKUP THE THREAD!\n");

    timer_call_datum->key = taudb_create_hash_key_2(timer_call_datum->thread->index, timer_str);
#ifdef TAUDB_DEBUG_DEBUG
    printf("NEW KEY: '%s'\n",timer_call_datum->key);
#endif
	HASH_ADD_KEYPTR(hh, timer_call_data, timer_call_datum->key, strlen(timer_call_datum->key), timer_call_datum);
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return (timer_call_data);
}

// convenience method
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data(PGconn* connection, TAUDB_TRIAL* trial) {
  return taudb_query_timer_call_data(connection, trial, NULL, NULL);
}

// convenience method
TAUDB_TIMER_CALL_DATA* taudb_query_all_timer_call_data_stats(PGconn* connection, TAUDB_TRIAL* trial) {
  return taudb_query_timer_call_data_stats(connection, trial, NULL, NULL);
}

// for getting call_datas for real threads
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread) {
  return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, FALSE);
} 

// for getting call_datas for derived threads
TAUDB_TIMER_CALL_DATA* taudb_query_timer_call_data_stats(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_TIMER_CALLPATH* timer_callpath, TAUDB_THREAD* thread) {
  if (taudb_version == TAUDB_2005_SCHEMA) {
    return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, TRUE);
  } else {
    return taudb_private_query_timer_call_data(connection, trial, timer_callpath, thread, TRUE);
  }
}

// convenience method for indexing into the hash
TAUDB_TIMER_CALL_DATA* taudb_get_timer_call_data(TAUDB_TIMER_CALL_DATA* timer_call_data, TAUDB_TIMER* timer, TAUDB_THREAD* thread) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_call_data(%p,%p,%p)\n", timer_call_data, timer, thread);
#endif
  if (timer_call_data == NULL) {
    fprintf(stderr, "Error: timer_call_data parameter null. Please provide a valid set of timer_call_data.\n");
    return NULL;
  }
  if (timer == NULL) {
    fprintf(stderr, "Error: timer parameter null. Please provide a valid timer.\n");
    return NULL;
  }
  if (timer_call_data == NULL) {
    fprintf(stderr, "Error: thread parameter null. Please provide a valid thread.\n");
    return NULL;
  }
  char *key = taudb_create_hash_key_2(thread->index, timer->name);
#ifdef TAUDB_DEBUG_DEBUG
  printf("'%d', '%s', Looking for key: %s\n", thread->index, timer->name, key);
#endif

  TAUDB_TIMER_CALL_DATA* timer_call_datum = NULL;
  HASH_FIND_STR(timer_call_data, key, timer_call_datum);
  return timer_call_datum;
}
