#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_COUNTER* taudb_query_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
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
    *taudb_numItems = HASH_CNT(hh1,trial->counters_by_id);
    return trial->counters_by_id;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    snprintf(my_query, sizeof(my_query), "select * from atomic_event where trial = %d", trial->id);
  } else {
    snprintf(my_query, sizeof(my_query), "select * from counter where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  *taudb_numItems = nRows;
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
    taudb_add_counter_to_trial(trial, counter);
  }
  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (trial->counters_by_id);
}

void taudb_add_counter_to_trial(TAUDB_TRIAL* trial, TAUDB_COUNTER* counter) {
  if (counter->id > 0) {
    HASH_ADD(hh1, trial->counters_by_id, id, sizeof(int), counter);
  }
  HASH_ADD_KEYPTR(hh2, trial->counters_by_name, counter->name, strlen(counter->name), counter);
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

void taudb_save_counters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  const char* my_query;
	const char* statement_name;
	int nParams;
	
	const char* insert_query = "insert into counter (trial, name) values ($1, $2);";
  const char* insert_statement_name = "TAUDB_INSERT_COUNTER";
	const int insert_nParams = 2;
	const char* update_query = "update counter set trial = $1, name = $2 where id = $3;";
  const char* update_statement_name = "TAUDB_UPDATE_COUNTER";
	const int update_nParams = 3;
	
	if(update) {
		my_query = update_query;
		statement_name = update_statement_name;
		nParams = update_nParams;
	} else {
		my_query = insert_query;
		statement_name = insert_statement_name;
		nParams = insert_nParams;
	}
	
  taudb_prepare_statement(connection, statement_name, my_query, nParams);
  TAUDB_COUNTER *counter, *tmp;
  HASH_ITER(hh2, trial->counters_by_name, counter, tmp) {
    // make array of 3 character pointers
    const char* paramValues[3] = {0};
    char trialid[32] = {0};
    snprintf(trialid, sizeof(trialid),  "%d", trial->id);
    paramValues[0] = trialid;
    paramValues[1] = counter->name;
		
	char id[32] = {0};
	if(update && counter->id > 0) {
		snprintf(id, sizeof(id),  "%d", counter->id);
		paramValues[2] = id;
	}

    int rows = taudb_execute_statement(connection, statement_name, nParams, paramValues);
	if(update && rows == 0) {
#ifdef TAUDB_DEBUG
		printf("Falling back to insert for update of counter %s.\n", counter->name);
#endif
		/* updated row didn't exist; insert instead */
		counter->id = 0;
		taudb_prepare_statement(connection, insert_statement_name, insert_query, insert_nParams);
		taudb_execute_statement(connection, insert_statement_name, insert_nParams, paramValues);
	}
		
	if(!(update && counter->id > 0)) {
	    taudb_execute_query(connection, "select currval('counter_id_seq');");

	    int nRows = taudb_get_num_rows(connection);
	    if (nRows == 1) {
	      counter->id = atoi(taudb_get_value(connection, 0, 0));
	      //printf("New Counter: %d\n", counter->id);
	    } else {
	      printf("Failed.\n");
	    }
		taudb_close_query(connection);
	}
  }
  taudb_clear_result(connection);
}

TAUDB_COUNTER* taudb_next_counter_by_name_from_trial(TAUDB_COUNTER* current) {
  return current->hh1.next;
}

TAUDB_COUNTER* taudb_next_counter_by_id_from_trial(TAUDB_COUNTER* current) {
  return current->hh2.next;
}


