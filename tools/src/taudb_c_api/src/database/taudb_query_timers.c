#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern void taudb_parse_timer_group_names(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, char* group_names);
extern void taudb_trim(char * s);

TAUDB_TIMER* taudb_query_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial,int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timers(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->timers_by_id != NULL) {
    *taudb_numItems = HASH_CNT(trial_hash_by_id,trial->timers_by_id);
    return trial->timers_by_id;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    /* this odd-looking query will make sure that the flat profile timers
     * will be processed before the callpath timers. */
    // sprintf(my_query,"select *, 0 as order_rank from interval_event where trial = %d and name not like '%% => %%' union select *, 1 as order_rank from interval_event where trial = %d and name like '%% => %%' order by order_rank", trial->id, trial->id);
    snprintf(my_query, sizeof(my_query), "select * from interval_event where trial = %d and name not like '%% => %%'", trial->id);
  } else {
    snprintf(my_query, sizeof(my_query), "select * from timer where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    TAUDB_TIMER* timer = taudb_create_timers(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        timer->id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "trial") == 0) {
        timer->trial = trial;
      } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        timer->name = taudb_strdup(taudb_get_value(connection,i,j));
        taudb_trim(timer->name);
#ifdef TAUDB_DEBUG_DEBUG
        printf("Got timer '%s'\n", timer->name);
#endif
      } else if (strcmp(taudb_get_column_name(connection, j), "short_name") == 0) {
#ifdef TAUDB_DEBUG_DEBUG
        printf("Short Name: %s\n", taudb_get_value(connection,i,j));
#endif
        timer->short_name = taudb_strdup(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "source_file") == 0) {
        timer->source_file = taudb_strdup(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "line_number") == 0) {
        timer->line_number = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "line_number_end") == 0) {
        timer->line_number_end = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "column_number") == 0) {
        timer->column_number = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "column_number_end") == 0) {
        timer->column_number_end = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "group_name") == 0) {
        // tokenize the string, something like 'TAU_USER|MPI|...'
        char* group_names = taudb_get_value(connection, i, j);
        taudb_parse_timer_group_names(trial, timer, group_names);
      } else if (strcmp(taudb_get_column_name(connection, j), "order_rank") == 0) {
        continue; // ignore this synthetic value
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
    } 
    // save this in the hash
    taudb_add_timer_to_trial(trial, timer);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (trial->timers_by_id);
}

void taudb_add_timer_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER* timer) {
  if (timer->id > 0) {
    HASH_ADD(trial_hash_by_id, trial->timers_by_id, id, sizeof(int), timer);
  }
  HASH_ADD_KEYPTR(trial_hash_by_name, trial->timers_by_name, timer->name, strlen(timer->name), timer);
}

TAUDB_TIMER* taudb_get_timer_by_id(TAUDB_TIMER* timers, const int id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_by_id(%p,%d)\n", timers, id);
#endif
  if (timers == NULL) {
    fprintf(stderr, "Error: timer parameter null. Please provide a valid set of timers.\n");
    return NULL;
  }
  if (id == 0) {
    fprintf(stderr, "Error: id parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER* timer = NULL;
  HASH_FIND(trial_hash_by_id, timers, &id, sizeof(int), timer);
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer == NULL) {
    TAUDB_TIMER *current, *tmp;
    HASH_ITER(trial_hash_by_id, timers, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("TIMER: '%s'\n", current->name);
#endif
      if (current->id == id) {
        return current;
      }
    }
  }
  return timer;
}

TAUDB_TIMER* taudb_get_trial_timer_by_name(TAUDB_TIMER* timers, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_by_name(%p,%s)\n", timers, name);
#endif
  if (timers == NULL) {
    fprintf(stderr, "Error: timer parameter null. Please provide a valid set of timers.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER* timer = NULL;
  HASH_FIND(trial_hash_by_name, timers, name, strlen(name), timer);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer == NULL) {
#ifdef TAUDB_DEBUG
      printf ("TIMER not found, iterating...\n");
#endif
    TAUDB_TIMER *current, *tmp;
    HASH_ITER(trial_hash_by_name, timers, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("TIMER: '%s'\n", current->name);
#endif
      if (strcmp(current->name, name) == 0) {
        return current;
      }
    }
  }
#endif
  return timer;
}

TAUDB_TIMER* taudb_get_group_timer_by_name(TAUDB_TIMER* timers, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_by_name(%p,%s)\n", timers, name);
#endif
  if (timers == NULL) {
    fprintf(stderr, "Error: timer parameter null. Please provide a valid set of timers.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER* timer = NULL;
  HASH_FIND(group_hash_by_name, timers, name, strlen(name), timer);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer == NULL) {
#ifdef TAUDB_DEBUG
      printf ("TIMER not found, iterating...\n");
#endif
    TAUDB_TIMER *current, *tmp;
    HASH_ITER(group_hash_by_name, timers, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("TIMER: '%s'\n", current->name);
#endif
      if (strcmp(current->name, name) == 0) {
        return current;
      }
    }
  }
#endif
  return timer;
}

int taudb_count_bars(const char* instring) {
  int count = 0;
  int i;
  int length = strlen(instring);
  for (i = 0; i < length; i++) {
    if (instring[i] == '|') {
      count++;
	}
  }
  return count;
}


void taudb_parse_timer_group_names(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, char* group_names) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Got timer groups '%s'\n", group_names);
#endif
  if (strlen(group_names) > 0) {
    //int group_count = taudb_count_bars(group_names) + 1;
	//timer->groups = (TAUDB_TIMER_GROUP**)malloc(group_count * sizeof(TAUDB_TIMER_GROUP*));
    char* group_name = strtok(group_names, "|");
    while (group_name != NULL) {
      // see if the group exists
	  taudb_trim(group_name);
	  // this timer might have already been processed - does this timer already exist in this group?
      TAUDB_TIMER_GROUP* group = taudb_get_timer_group_from_timer_by_name(timer->groups, group_name);
      if (group == NULL) {
	    // does the group exist at all?
        group = taudb_get_timer_group_from_trial_by_name(trial->timer_groups, group_name);
        if (group == NULL) {
          group = taudb_create_timer_groups(1);
		  taudb_trim(group_name);
          group->name = taudb_strdup(group_name);
          // add the group to the trial
		  taudb_add_timer_group_to_trial(trial, group);
        }
	    taudb_add_timer_to_timer_group(group, timer);
	  }
      // get the next token
      group_name = strtok(NULL, "|");
    }
  }
}

void taudb_add_timer_to_timer_group(TAUDB_TIMER_GROUP* timer_group, TAUDB_TIMER* timer) {
  //printf("Adding timer '%s' to group '%s', %p\n", timer->name, timer_group->name, timer->group_hash_by_name.prev);
  //HASH_ADD_KEYPTR(group_hash_by_name, timer_group->timers, timer->name, strlen(timer->name), timer);
  //printf("Adding group '%s' to timer '%s', %p\n", timer_group->name, timer->name, timer_group->timer_hash_by_name.prev);
  HASH_ADD_KEYPTR(timer_hash_by_name, timer->groups, timer_group->name, strlen(timer_group->name), timer_group);
}

extern void taudb_process_timer_name(TAUDB_TIMER* timer);

void taudb_save_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  const char* my_query; 
	const char* statement_name;
	int nParams;
	
  const char* update_query = "update timer set trial=$1, name=$2, short_name=$3, source_file=$4, line_number=$5, line_number_end=$6, column_number=$7, column_number_end=$8 where id=$9;";
  const char* update_statement_name = "TAUDB_UPDATE_TIMER";
	const int update_nParams = 9;
  const char* insert_query = "insert into timer (trial, name, short_name, source_file, line_number, line_number_end, column_number, column_number_end) values ($1, $2, $3, $4, $5, $6, $7, $8);";
  const char* insert_statement_name = "TAUDB_INSERT_TIMER";
	const int insert_nParams = 8;
	
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
	
  TAUDB_TIMER *timer, *tmp;
  HASH_ITER(trial_hash_by_name, trial->timers_by_name, timer, tmp) {
    // make array of 9 character pointers
    const char* paramValues[9] = {0};
    char trialid[32] = {0};
    snprintf(trialid, sizeof(trialid),  "%d", trial->id);
    paramValues[0] = trialid;
    paramValues[1] = timer->name;
		if (timer->short_name == NULL) {
		  taudb_process_timer_name(timer);
		}
    paramValues[2] = timer->short_name;
    paramValues[3] = timer->source_file;
    char line_number[32] = {0};
    snprintf(line_number, sizeof(line_number),  "%d", timer->line_number);
    paramValues[4] = line_number;
    char line_number_end[32] = {0};
    snprintf(line_number_end, sizeof(line_number_end),  "%d", timer->line_number_end);
    paramValues[5] = line_number_end;
    char column_number[32] = {0};
    snprintf(column_number, sizeof(column_number),  "%d", timer->column_number);
    paramValues[6] = column_number;
    char column_number_end[32] = {0};
    snprintf(column_number_end, sizeof(column_number_end),  "%d", timer->column_number_end);
    paramValues[7] = column_number_end;
		
	char id[32] = {0};
	if(update && timer->id > 0) {
		snprintf(id, sizeof(id),  "%d", timer->id);
		paramValues[8] = id;
	}

    int rows = taudb_execute_statement(connection, statement_name, nParams, paramValues);
			if(update && rows == 0) {
#ifdef TAUDB_DEBUG
				printf("Falling back to insert for update of timer.\n");
#endif
				/* updated row didn't exist; insert instead */
				timer->id = 0;
				taudb_prepare_statement(connection, insert_statement_name, insert_query, insert_nParams);
				taudb_execute_statement(connection, insert_statement_name, insert_nParams, paramValues);
			}
		
		if(!(update && timer->id > 0)) {
	    taudb_execute_query(connection, "select currval('timer_id_seq');");

	    int nRows = taudb_get_num_rows(connection);
	    if (nRows == 1) {
	      timer->id = atoi(taudb_get_value(connection, 0, 0));
	      //printf("New Timer: %d\n", timer->id);
	    } else {
	      printf("Failed.\n");
	    }
			taudb_close_query(connection);
		}
  }
  taudb_clear_result(connection);
}

TAUDB_TIMER* taudb_next_timer_by_name_from_trial(TAUDB_TIMER* current) {
  return current->trial_hash_by_name.next;
}

TAUDB_TIMER* taudb_next_timer_by_id_from_trial(TAUDB_TIMER* current) {
  return current->trial_hash_by_id.next;
}

TAUDB_TIMER* taudb_next_timer_by_name_from_group(TAUDB_TIMER* current) {
  return current->group_hash_by_name.next;
}

