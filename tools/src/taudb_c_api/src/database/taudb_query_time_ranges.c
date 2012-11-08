#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIME_RANGE* taudb_query_time_ranges(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_time_ranges(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->time_ranges != NULL) {
    taudb_numItems = HASH_COUNT(trial->time_ranges);
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
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  taudb_numItems = nRows;

  TAUDB_TIME_RANGE* time_ranges = taudb_create_time_ranges(taudb_numItems);

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    TAUDB_TIME_RANGE* time_range = &(time_ranges[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
	    time_range->id = atoi(taudb_get_value(connection, i, j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "trial") == 0) {
	    //time_range->trial = trial;
	  } else if (strcmp(taudb_get_column_name(connection, j), "iteration_start") == 0) {
	    time_range->iteration_start = atoi(taudb_get_value(connection,i,j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "iteration_end") == 0) {
	    time_range->iteration_end = atoi(taudb_get_value(connection,i,j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "time_start") == 0) {
	    time_range->time_start = atoll(taudb_get_value(connection,i,j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "time_end") == 0) {
	    time_range->time_end = atoll(taudb_get_value(connection,i,j));
	  } else {
	    printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
	    taudb_exit_nicely(connection);
	  }
	} 
	HASH_ADD_INT(time_ranges, id, time_range);
  }

  taudb_clear_result(connection);
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

void taudb_save_time_ranges(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  const char* my_query = "insert into time_range (iteration_start, iteration_end, time_start, time_end) values ($1, $2, $3, $4);";
  const char* statement_name = "TAUDB_INSERT_TIME_RANGE";
  taudb_prepare_statement(connection, statement_name, my_query, 4);
  TAUDB_TIME_RANGE *time_range, *tmp;
  HASH_ITER(hh, trial->time_ranges, time_range, tmp) {
    // make array of 6 character pointers
    const char* paramValues[4] = {0};
    char istart[32] = {0};
    sprintf(istart, "%d", time_range->iteration_start);
    paramValues[0] = istart;
    char iend[32] = {0};
    sprintf(iend, "%d", time_range->iteration_end);
    paramValues[1] = iend;
    char tstart[32] = {0};
    sprintf(tstart, "%lld", time_range->time_start);
    paramValues[2] = tstart;
    char tend[32] = {0};
    sprintf(tend, "%lld", time_range->time_end);
    paramValues[3] = tend;

    taudb_execute_statement(connection, statement_name, 4, paramValues);
    taudb_execute_query(connection, "select currval('time_range_id_seq');");

    int nRows = taudb_get_num_rows(connection);
    if (nRows == 1) {
      time_range->id = atoi(taudb_get_value(connection, 0, 0));
      //printf("New Time_Range: %d\n", time_range->id);
    } else {
      printf("Failed.\n");
    }
	taudb_close_query(connection);
  }
  taudb_clear_result(connection);
}
