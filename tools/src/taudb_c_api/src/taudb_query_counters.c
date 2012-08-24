#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_COUNTER* taudb_query_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_counters(%p)\n", trial);
#endif
  void *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  // if the Trial already has the data, return it.
  if (trial->counters != NULL && trial->counter_count > 0) {
    taudb_numItems = trial->counter_count;
    return trial->counters;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"select * from atomic_event where trial = %d", trial->id);
  } else {
    sprintf(my_query,"select * from counter where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  TAUDB_COUNTER* counters = taudb_create_counters(nRows);
  taudb_numItems = nRows;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
	    counters[i].id = atoi(taudb_get_value(res, i, j));
	  } else if (strcmp(taudb_get_column_name(res, j), "trial") == 0) {
	    counters[i].trial = trial;
	  } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
	    //counters[i].name = taudb_get_value(res, i, j);
		counters[i].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
#ifdef TAUDB_DEBUG
        //printf("Got counter '%s'\n", counters[i].name);
#endif
	  } else {
	    printf("Error: unknown column '%s'\n", taudb_get_column_name(res, j));
	    taudb_exit_nicely(connection);
	  }
	  // TODO - Populate the rest properly?
	} 
  }
  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return (counters);
}
