#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TRIAL* taudb_private_query_trials(PGconn* connection, boolean full, char* my_query) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_trials(%d, %s)\n", full, my_query);
#endif
  PGresult *res;
  int nFields;
  int i, j;

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
  TAUDB_TRIAL* trials = taudb_create_trials(nRows);

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int metaIndex = 0;
    //trials[i].primary_metadata = taudb_create_primary_metadata(nFields-6);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "id") == 0) {
        trials[i].id = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "name") == 0) {
        trials[i].name = taudb_create_and_copy_string(PQgetvalue(res,i,j));
      //} else if (strcmp(PQfname(res, j), "date") == 0) {
        //trials[i].collection_date = taudb_create_and_copy_string(PQgetvalue(res,i,j));
      } else if (strcmp(PQfname(res, j), "node_count") == 0) {
        trials[i].node_count = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "contexts_per_node") == 0) {
        trials[i].contexts_per_node = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "threads_per_context") == 0) {
        trials[i].threads_per_context = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "total_threads") == 0) {
        trials[i].total_threads = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "data_source") == 0) {
        trials[i].data_source = atoi(PQgetvalue(res, i, j));
      } else if (strcmp(PQfname(res, j), "xml_metadata") == 0) {
        // TODO we need to handle this!
        continue;
      } else if (strcmp(PQfname(res, j), "xml_metadata_gz") == 0) {
        // TODO we need to handle this!
        continue;
      } else {
        //trials[i].primary_metadata[metaIndex].name = taudb_create_and_copy_string(PQfname(res, j));
        //trials[i].primary_metadata[metaIndex].value = taudb_create_and_copy_string(PQgetvalue(res,i,j));
        //metaIndex++;
      }
    } 
    //trials[i].primary_metadata_count = metaIndex;
    trials[i].primary_metadata_count = 0;
 }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  for (i = 0 ; i < nRows ; i++) {
    if (taudb_version == TAUDB_2005_SCHEMA) {
	  fprintf(stderr,"Did not load the PerfDMF metadata...\n");
	} else {
      trials[i].primary_metadata = taudb_query_primary_metadata(connection, &(trials[i]));
      trials[i].primary_metadata_count = taudb_numItems;
	}
    if (full) {
      trials[i].threads = taudb_query_threads(connection, &(trials[i]));
      trials[i].thread_count = taudb_numItems;
      trials[i].timers = taudb_query_timers(connection, &(trials[i]));
      trials[i].timer_count = taudb_numItems;
      trials[i].timer_callpaths = taudb_query_all_timer_callpaths(connection, &(trials[i]));
      trials[i].timer_callpath_count = taudb_numItems;
      //trials[i].timer_callpath_stats = taudb_query_all_timer_callpath_stats(connection, &(trials[i]));
      //trials[i].callpath_stat_count = taudb_numItems;
      trials[i].metrics = taudb_query_metrics(connection, &(trials[i]));
      trials[i].metric_count = taudb_numItems;
      //trials[i].timer_values = taudb_query_all_timer_values(connection, &(trials[i]));
      //trials[i].value_count = taudb_numItems;
      //trials[i].counters = taudb_query_counters(&(trials[i]));
      //trials[i].counter_count = taudb_numItems;
      //taudb_query_counter_values(&(trials[i]));
      if (taudb_version == TAUDB_2012_SCHEMA) {
        trials[i].secondary_metadata = taudb_query_secondary_metadata(connection, &(trials[i]));
        trials[i].secondary_metadata_count = taudb_numItems;
      }
    }
  }
  taudb_numItems = nRows;

  return trials;
}

TAUDB_TRIAL* taudb_query_trials(PGconn* connection, boolean full, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_trials(%d, %p)\n", full, trial);
#endif
  char my_query[1024];
  if (trial->id > 0) { // the user wants a specific trial, so get it
    sprintf(my_query,"DECLARE myportal CURSOR FOR select * from trial where id = %d", trial->id);
  } else {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select * from trial where");
    if (trial->name != NULL) {
      sprintf(my_query,"%s name = '%s'", my_query, trial->name);
    } 
  }
  return taudb_private_query_trials(connection, full, my_query);
}

TAUDB_TRIAL* perfdmf_query_trials(PGconn* connection, PERFDMF_EXPERIMENT* experiment) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_trials(%p)\n", experiment);
#endif
  char my_query[256];
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from trial where experiment = %d", experiment->id);

  return taudb_private_query_trials(connection, FALSE, my_query);
}

