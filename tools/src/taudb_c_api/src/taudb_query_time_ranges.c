#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIME_RANGE* taudb_query_time_ranges(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_time_ranges(%p)\n", trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->time_ranges != NULL && trial->time_range_count > 0) {
    taudb_numItems = trial->time_range_count;
    return trial->time_ranges;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  sprintf(my_query,"select * from time_range where trial = %d", trial->id);
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  taudb_numItems = nRows;

  TAUDB_TIME_RANGE* time_ranges = taudb_create_time_ranges(taudb_numItems);

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    TAUDB_TIME_RANGE* time_range = &(time_ranges[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
	    time_range->id = atoi(taudb_get_value(res, i, j));
	  } else if (strcmp(taudb_get_column_name(res, j), "trial") == 0) {
	    //time_range->trial = trial;
	  } else if (strcmp(taudb_get_column_name(res, j), "iteration_start") == 0) {
	    time_range->iteration_start = atoi(taudb_get_value(res,i,j));
	  } else if (strcmp(taudb_get_column_name(res, j), "iteration_end") == 0) {
	    time_range->iteration_end = atoi(taudb_get_value(res,i,j));
	  } else if (strcmp(taudb_get_column_name(res, j), "time_start") == 0) {
	    time_range->time_start = atoll(taudb_get_value(res,i,j));
	  } else if (strcmp(taudb_get_column_name(res, j), "time_end") == 0) {
	    time_range->time_end = atoll(taudb_get_value(res,i,j));
	  } else {
	    printf("Error: unknown column '%s'\n", taudb_get_column_name(res, j));
	    taudb_exit_nicely(connection);
	  }
	} 
	HASH_ADD_INT(time_ranges, id, time_range);
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return (time_ranges);
}

TAUDB_TIME_RANGE* taudb_get_time_range(TAUDB_TIME_RANGE* time_ranges, const int id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_time_range(%p,%d)\n", time_ranges, id);
#endif
  if (time_ranges == NULL) {
    fprintf(stderr, "Error: time_range parameter null. Please provide a valid set of time_ranges.\n");
    return NULL;
  }
  if (id == 0) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIME_RANGE* time_range = NULL;
  HASH_FIND_INT(time_ranges, &id, time_range);
  return time_range;
}

