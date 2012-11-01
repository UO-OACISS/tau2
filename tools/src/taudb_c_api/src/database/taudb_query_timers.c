#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern void taudb_parse_timer_group_names(TAUDB_TRIAL* trial, TAUDB_TIMER* timer, char* group_names);
extern void taudb_trim(char * s);

TAUDB_TIMER* taudb_query_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
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
    taudb_numItems = HASH_CNT(hh1,trial->timers_by_id);
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
    sprintf(my_query,"select * from interval_event where trial = %d and name not like '%% => %%'", trial->id);
  } else {
    sprintf(my_query,"select * from timer where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  taudb_numItems = nRows;

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
        timer->name = taudb_create_and_copy_string(taudb_get_value(connection,i,j));
        taudb_trim(timer->name);
#ifdef TAUDB_DEBUG_DEBUG
        printf("Got timer '%s'\n", timer->name);
#endif
      } else if (strcmp(taudb_get_column_name(connection, j), "short_name") == 0) {
#ifdef TAUDB_DEBUG_DEBUG
        printf("Short Name: %s\n", taudb_get_value(connection,i,j));
#endif
        timer->short_name = taudb_create_and_copy_string(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "source_file") == 0) {
        timer->source_file = taudb_create_and_copy_string(taudb_get_value(connection,i,j));
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
    //HASH_ADD(hh1, trial->timers_by_id, id, sizeof(int), timer);
    HASH_ADD_KEYPTR(hh2, trial->timers_by_name, timer->name, strlen(timer->name), timer);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (trial->timers_by_id);
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
  HASH_FIND(hh1, timers, &id, sizeof(int), timer);
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer == NULL) {
    TAUDB_TIMER *current, *tmp;
    HASH_ITER(hh1, timers, current, tmp) {
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

TAUDB_TIMER* taudb_get_timer_by_name(TAUDB_TIMER* timers, const char* name) {
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
  HASH_FIND(hh2, timers, name, strlen(name), timer);
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer == NULL) {
#ifdef TAUDB_DEBUG
      printf ("TIMER not found, iterating...\n");
#endif
    TAUDB_TIMER *current, *tmp;
    HASH_ITER(hh2, timers, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("TIMER: '%s'\n", current->name);
#endif
      if (strcmp(current->name, name) == 0) {
        return current;
      }
    }
  }
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
    int group_count = taudb_count_bars(group_names) + 1;
	timer->groups = (TAUDB_TIMER_GROUP**)malloc(group_count * sizeof(TAUDB_TIMER_GROUP*));
    char* group_name = strtok(group_names, "|");
    while (group_name != NULL) {
      timer->group_count++;
      // see if the group exists
	  taudb_trim(group_name);
      TAUDB_TIMER_GROUP* group = taudb_get_timer_group_by_name(trial->timer_groups, group_name);
      if (group != NULL) {
#ifdef TAUDB_DEBUG_DEBUG
        printf("FOUND GROUP: %s\n", group_name);
#endif
      } else {
        group = taudb_create_timer_groups(1);
		taudb_trim(group_name);
        group->name = taudb_create_and_copy_string(group_name);
        // add the group to the trial
        HASH_ADD_KEYPTR(hh, trial->timer_groups, group->name, strlen(group->name), group);
      }
      // add this timer group to our timer
      timer->groups[(timer->group_count)-1] = group;
      // add this timer to the list of timers in the group
#if 0
      group->timer_count++;
      group->timers = (TAUDB_TIMER**)realloc(group->timers, (group->timer_count * sizeof(TAUDB_TIMER*)));
      group->timers[(group->timer_count)-1] = timer;
#endif
      // get the next token
      group_name = strtok(NULL, "|");
    }
  } else {
    timer->group_count = 0;
    timer->groups = NULL;
  }
}


