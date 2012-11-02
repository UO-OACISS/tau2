#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_GROUP* taudb_query_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
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
    taudb_numItems = HASH_CNT(hh,trial->timer_groups);
    return trial->timer_groups;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"select group_name from interval_event where trial = %d", trial->id);
	fprintf(stderr, "WARNING - NOT TESTED!\n");
  } else {
    sprintf(my_query,"select distinct tg.name as name from timer_group tg inner join timer t on tg.timer = t.id where t.trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  TAUDB_TIMER_GROUP* timer_groups = NULL;
  taudb_numItems = nRows;

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
	  /*
        // tokenize the string, something like 'TAU_USER|MPI|...'
        char* group_names = taudb_get_value(connection, i, j);
        char* group = strtok(group_names, "|");
        if (group != NULL && (strlen(group_names) > 0)) {
#ifdef TAUDB_DEBUG
          //printf("Got group groups '%s'\n", group_names);
#endif
          groups[i].group_count = 1;
          TAUDB_TIMER_GROUP_TIMER_GROUP* groups = taudb_create_group_groups(1);
          groups[0].id = 0;
          groups[0].group = 0;
          groups[0].name = taudb_strdup(group);
          group = strtok(NULL, "|");
          while (group != NULL) {
            TAUDB_TIMER_GROUP_TIMER_GROUP* groups = taudb_resize_group_groups(groups[i].group_count+1, groups);
            groups[groups[i].group_count].id = 0;
            groups[groups[i].group_count].group = 0;
            groups[groups[i].group_count].name = taudb_strdup(group);
            groups[i].group_count++;
            group = strtok(NULL, "|");
          }
        } else {
          groups[i].group_count = 0;
          groups[i].groups = NULL;
        }
		  */
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
      // TODO - Populate the rest properly?
      timer_group->timer_count = 0;
      timer_group->timers = NULL;
    } 
    HASH_ADD_KEYPTR(hh, timer_groups, timer_group->name, strlen(timer_group->name), timer_group);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);
 
  return (timer_groups);
}

TAUDB_TIMER_GROUP* taudb_get_timer_group_by_name(TAUDB_TIMER_GROUP* timer_groups, const char* name) {
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
  HASH_FIND_STR(timer_groups, name, timer_group);
  return timer_group;
}

extern void taudb_save_timer_groups(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  printf("Timer groups not supported yet.\n");
}
