#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_GROUP* taudb_query_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_group(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->timer_groups != NULL) {
    *taudb_numItems = HASH_CNT(trial_hash_by_name,trial->timer_groups);
    return trial->timer_groups;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    snprintf(my_query, sizeof(my_query), "select group_name from interval_event where trial = %d", trial->id);
	fprintf(stderr, "WARNING - NOT TESTED!\n");
  } else {
    snprintf(my_query, sizeof(my_query), "select distinct tg.group_name as name from timer_group tg inner join timer t on tg.timer = t.id where t.trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  TAUDB_TIMER_GROUP* timer_groups = NULL;
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    TAUDB_TIMER_GROUP* timer_group = taudb_create_timer_groups(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        timer_group->name = taudb_strdup(taudb_get_value(connection,i,j));
#ifdef TAUDB_DEBUG_DEBUG
        printf("Got group '%s'\n", timer_group->name);
#endif
      } else if (strcmp(taudb_get_column_name(connection, j), "group_name") == 0) {
	    // do nothing - this is handled elsewhere.
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
    } 
    taudb_add_timer_group_to_trial(trial, timer_group);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);
 
  return (timer_groups);
}

void taudb_add_timer_group_to_trial(TAUDB_TRIAL* trial, TAUDB_TIMER_GROUP* timer_group) {
    HASH_ADD_KEYPTR(trial_hash_by_name, trial->timer_groups, timer_group->name, strlen(timer_group->name), timer_group);
}

TAUDB_TIMER_GROUP* taudb_get_timer_group_from_trial_by_name(TAUDB_TIMER_GROUP* timer_groups, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_group_by_name(%p,'%s')\n", timer_groups, name);
#endif
  if (timer_groups == NULL) {
    // the hash isn't populated yet
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER_GROUP* timer_group = NULL;
  HASH_FIND(trial_hash_by_name, timer_groups, name, strlen(name), timer_group);
  return timer_group;
}

TAUDB_TIMER_GROUP* taudb_get_timer_group_from_timer_by_name(TAUDB_TIMER_GROUP* timer_groups, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_group_by_name(%p,'%s')\n", timer_groups, name);
#endif
  if (timer_groups == NULL) {
    // the hash isn't populated yet
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER_GROUP* timer_group = NULL;
  HASH_FIND(timer_hash_by_name, timer_groups, name, strlen(name), timer_group);
  return timer_group;
}

void taudb_save_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  const char* my_query = "insert into timer_group (timer, group_name) select $1, $2::varchar where not exists (select 1 from timer_group where timer=$1 and group_name=$2);";
  const char* statement_name = "TAUDB_INSERT_TIMER_GROUP";
	int nParams = 2;
	
  taudb_prepare_statement(connection, statement_name, my_query, nParams);
  TAUDB_TIMER_GROUP *group, *tmp;
  TAUDB_TIMER *timer, *tmp2;
#if 0
  // iterate over the timer_groups
  HASH_ITER(trial_hash_by_name, trial->timer_groups, group, tmp) {
    // iterate over the timers in the timer_group
    HASH_ITER(group_hash_by_name, group->timers, timer, tmp2) {
#else
  // iterate over the timers
  HASH_ITER(trial_hash_by_name, trial->timers_by_name, timer, tmp2) {
    // iterate over the timer_groups in the timer
    HASH_ITER(timer_hash_by_name, timer->groups, group, tmp) {
#endif
      // make array of 6 character pointers
      const char* paramValues[2] = {0};
      char timerid[32] = {0};
      snprintf(timerid, sizeof(timerid),  "%d", timer->id);
      paramValues[0] = timerid;
      paramValues[1] = group->name;
      taudb_execute_statement(connection, statement_name, nParams, paramValues);
    }
  }
  taudb_clear_result(connection);
}

TAUDB_TIMER_GROUP* taudb_next_timer_group_by_name_from_trial(TAUDB_TIMER_GROUP* current) {
  return current->trial_hash_by_name.next;
}

TAUDB_TIMER_GROUP* taudb_next_timer_group_by_name_from_timer(TAUDB_TIMER_GROUP* current) {
  return current->timer_hash_by_name.next;
}


