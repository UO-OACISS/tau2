#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_COUNTER_VALUE* taudb_query_all_counter_values(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_all_counter_values(%p)\n", trial);
#endif
  return taudb_query_counter_values(connection, trial, NULL, NULL);
}

TAUDB_COUNTER_VALUE* taudb_query_counter_values(PGconn* connection, TAUDB_TRIAL* trial, TAUDB_COUNTER* counter, TAUDB_THREAD* thread) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_counter_values(%p,%p,%p)\n", trial, counter, thread);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  // validate inputs
  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
	return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->counter_values != NULL && trial->counter_value_count > 0) {
    taudb_numItems = trial->counter_value_count;
    return trial->counter_values;
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
    sprintf(my_query,"DECLARE myportal CURSOR FOR select alp.*, ae.name as counter_name from atomic_location_profile alp inner join atomic_event ae on alp.atomic_event = ae.id");
    char* conjoiner = "where";
    if (trial != NULL) {
      sprintf(my_query,"%s where ae.trial = %d", my_query, trial->id);
      conjoiner = "and";
    } 
    if (counter != NULL) {
      sprintf(my_query,"%s %s ae.id = %d", my_query, conjoiner, counter->id);
      conjoiner = "and";
    }
    if (thread != NULL) {
      sprintf(my_query,"%s %s node = %d and context = %d and thread = %d", my_query, conjoiner, thread->node_rank, thread->context_rank, thread->thread_rank);
    }
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
  TAUDB_COUNTER_VALUE* counter_values = NULL;
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int node = 0;
    int context = 0;
    int thread = 0;
	char* counter_str;
    TAUDB_COUNTER_VALUE* counter_value = taudb_create_counter_values(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        counter_value->id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "atomic_event") == 0) {
        counter_value->counter = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "node") == 0) {
        node = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "context") == 0) {
        context = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "thread") == 0) {
        thread = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "counter_name") == 0) {
        counter_str = PQgetvalue(res, i, j);
      } else if (strcmp(PQfname(res, j), "sample_count") == 0) {
        counter_value->sample_count = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "maximum_value") == 0) {
        counter_value->maximum_value = atof(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "minimum_value") == 0) {
        counter_value->minimum_value = atof(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "mean_value") == 0) {
        counter_value->mean_value = atof(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "standard_deviation") == 0) {
        counter_value->standard_deviation = atof(PQgetvalue(res, i, j));
      } else {
        printf("Error: unknown column '%s'\n", PQfname(res, j));
        taudb_exit_nicely(connection);
      }
    } 
    counter_value->thread = (node * (trial->contexts_per_node * trial->threads_per_context)) +
                          (context * (trial->threads_per_context)) + 
                          thread;

    char tmp_thread[100];
	sprintf(tmp_thread, "%d", counter_value->thread);
	counter_value->key = taudb_create_string(strlen(tmp_thread) + strlen(counter_str) + 2);
    sprintf(counter_value->key, "%d:%s", counter_value->thread, counter_str);
	HASH_ADD_KEYPTR(hh, counter_values, counter_value->key, strlen(counter_value->key), counter_value);
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return (counter_values);
}

TAUDB_COUNTER_VALUE* taudb_get_counter_value(TAUDB_COUNTER_VALUE* counter_values, TAUDB_COUNTER* counter, TAUDB_THREAD* thread) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_counter_value(%p,%p,%p)\n", counter_values, counter, thread);
#endif
  if (counter_values == NULL) {
    fprintf(stderr, "Error: counter_values parameter null. Please provide a valid set of counter_values.\n");
    return NULL;
  }
  if (counter == NULL) {
    fprintf(stderr, "Error: counter parameter null. Please provide a valid counter.\n");
    return NULL;
  }
  if (thread == NULL) {
    fprintf(stderr, "Error: thread parameter null. Please provide a valid thread.\n");
    return NULL;
  }
  char tmp_thread[10];
  sprintf(tmp_thread, "%d", thread->index);
  char *key = taudb_create_string(strlen(tmp_thread) + strlen(counter->full_name) + 2);
  sprintf(key, "%d:%s", thread->index, counter->full_name);
  //printf("%s\n", key);

  TAUDB_COUNTER_VALUE* counter_value = NULL;
  HASH_FIND_STR(counter_values, key, counter_value);
  return counter_value;
}
