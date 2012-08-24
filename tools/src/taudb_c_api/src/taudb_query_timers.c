#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER* taudb_query_timers(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_timers(%p)\n", trial);
#endif
  void *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->timers != NULL && trial->timer_count > 0) {
    taudb_numItems = trial->timer_count;
    return trial->timers;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"select * from interval_event where trial = %d", trial->id);
  } else {
    sprintf(my_query,"select * from timer where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  TAUDB_TIMER* timers = taudb_create_timers(nRows);
  taudb_numItems = nRows;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
        timers[i].id = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "trial") == 0) {
        timers[i].trial = trial;
      } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
        timers[i].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
#ifdef TAUDB_DEBUG_DEBUG
        printf("Got timer '%s'\n", timers[i].name);
#endif
      } else if (strcmp(taudb_get_column_name(res, j), "short_name") == 0) {
        printf("Short Name: %s\n", taudb_get_value(res,i,j));
        timers[i].short_name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
      } else if (strcmp(taudb_get_column_name(res, j), "source_file") == 0) {
        timers[i].source_file = taudb_create_and_copy_string(taudb_get_value(res,i,j));
      } else if (strcmp(taudb_get_column_name(res, j), "line_number") == 0) {
        timers[i].line_number = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "line_number_end") == 0) {
        timers[i].line_number_end = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "column_number") == 0) {
        timers[i].column_number = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "column_number_end") == 0) {
        timers[i].column_number_end = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "group_name") == 0) {
        // tokenize the string, something like 'TAU_USER|MPI|...'
        char* group_names = taudb_get_value(res, i, j);
        char* group = strtok(group_names, "|");
        if (group != NULL && (strlen(group_names) > 0)) {
#ifdef TAUDB_DEBUG
          //printf("Got timer groups '%s'\n", group_names);
#endif
          timers[i].group_count = 1;
          TAUDB_TIMER_GROUP* groups = taudb_create_timer_groups(1);
          groups[0].id = 0;
          groups[0].name = taudb_create_and_copy_string(group);
          group = strtok(NULL, "|");
          while (group != NULL) {
            TAUDB_TIMER_GROUP* groups = taudb_resize_timer_groups(timers[i].group_count+1, groups);
            groups[timers[i].group_count].id = 0;
            groups[timers[i].group_count].name = taudb_create_and_copy_string(group);
            timers[i].group_count++;
            group = strtok(NULL, "|");
          }
        } else {
          timers[i].group_count = 0;
          timers[i].groups = NULL;
        }
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(res, j));
        taudb_exit_nicely(connection);
      }
      // TODO - Populate the rest properly?
      timers[i].parameter_count = 0;
    } 
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return (timers);
}
