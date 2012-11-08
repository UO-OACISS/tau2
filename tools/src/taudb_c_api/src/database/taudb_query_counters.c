#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_COUNTER* taudb_query_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_counters(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  // if the Trial already has the data, return it.
  if (trial->counters_by_id != NULL) {
    taudb_numItems = HASH_CNT(hh1,trial->counters_by_id);
    return trial->counters_by_id;
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
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  taudb_numItems = nRows;
#ifdef TAUDB_DEBUG
  printf("'%d' rows returned\n",nRows);
#endif

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    TAUDB_COUNTER* counter = taudb_create_counters(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
	    counter->id = atoi(taudb_get_value(connection, i, j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "trial") == 0) {
	    counter->trial = trial;
	  } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
	    //counter->name = taudb_get_value(connection, i, j);
		counter->name = taudb_strdup(taudb_get_value(connection,i,j));
#ifdef TAUDB_DEBUG
        //printf("Got counter '%s'\n", counter->name);
#endif
	  } else if (strcmp(taudb_get_column_name(connection, j), "source_file") == 0) {
            // do nothing
	  } else if (strcmp(taudb_get_column_name(connection, j), "line_number") == 0) {
            // do nothing
	  } else if (strcmp(taudb_get_column_name(connection, j), "group_name") == 0) {
            // do nothing
	  } else {
	    printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
	    taudb_exit_nicely(connection);
	  }
	  // TODO - Populate the rest properly?
	} 
    HASH_ADD(hh1, trial->counters_by_id, id, sizeof(int), counter);
    HASH_ADD_KEYPTR(hh2, trial->counters_by_name, counter->name, strlen(counter->name), counter);
  }
  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (trial->counters_by_id);
}

TAUDB_COUNTER* taudb_get_counter_by_id(TAUDB_COUNTER* counters, const int id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_counter_by_id(%p,%d)\n", counters, id);
#endif
  if (counters == NULL) {
    fprintf(stderr, "Error: counter parameter null. Please provide a valid set of counters.\n");
    return NULL;
  }
  if (id == 0) {
    fprintf(stderr, "Error: id parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_COUNTER* counter = NULL;
  HASH_FIND(hh1, counters, &id, sizeof(int), counter);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (counter == NULL) {
    TAUDB_COUNTER *current, *tmp;
    HASH_ITER(hh1, counters, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("COUNTER: '%s'\n", current->name);
#endif
      if (current->id == id) {
        return current;
      }
    }
  }
#endif
  return counter;
}

TAUDB_COUNTER* taudb_get_counter_by_name(TAUDB_COUNTER* counters, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_counter_by_name(%p,%s)\n", counters, name);
#endif
  if (counters == NULL) {
    fprintf(stderr, "Error: counter parameter null. Please provide a valid set of counters.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_COUNTER* counter = NULL;
  HASH_FIND(hh2, counters, name, strlen(name), counter);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (counter == NULL) {
    TAUDB_COUNTER *current, *tmp;
    HASH_ITER(hh2, counters, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("COUNTER: '%s'\n", current->name);
#endif
      if (strcmp(current->name, name) == 0) {
        return current;
      }
    }
  }
#endif
  return counter;
}


extern void taudb_save_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  printf("Counters not supported yet.\n");
}