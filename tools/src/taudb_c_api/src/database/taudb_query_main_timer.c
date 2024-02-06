#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER* taudb_query_main_timer(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_main_timer(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[1024];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    snprintf(my_query, sizeof(my_query), "select interval_location_profile.inclusive, interval_event.* from interval_location_profile left outer join interval_event on interval_location_profile.interval_event = interval_event.id where interval_event.trial = %d order by 1 desc limit 1", trial->id);
  } else {
    snprintf(my_query, sizeof(my_query), "select * from timer where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  TAUDB_TIMER* timers = taudb_create_timers(nRows);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        timers[i].id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "trial") == 0) {
        timers[i].trial = trial;
      } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        //timers[i].name = taudb_get_value(connection, i, j);
        timers[i].name = taudb_strdup(taudb_get_value(connection,i,j));
#ifdef TAUDB_DEBUG
        //printf("Got timer '%s'\n", timers[i].name);
#endif
      } else if (strcmp(taudb_get_column_name(connection, j), "short_name") == 0) {
        timers[i].short_name = taudb_strdup(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "source_file") == 0) {
        //timers[i].source_file = taudb_get_value(connection, i, j);
        timers[i].source_file = taudb_strdup(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "line_number") == 0) {
        timers[i].line_number = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "line_number_end") == 0) {
        timers[i].line_number_end = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "column_number") == 0) {
        timers[i].column_number = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "column_number_end") == 0) {
        timers[i].column_number_end = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "group_name") == 0) {
        // tokenize the string, something like 'TAU_USER|MPI|...'
        char* group_names = taudb_get_value(connection, i, j);
        if (strlen(group_names) > 0) {
          taudb_parse_timer_group_names(trial, &(timers[i]), group_names);
        } else {
          timers[i].groups = NULL;
        }
      } else if (strcmp(taudb_get_column_name(connection, j), "inclusive") == 0) {
        continue;
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
      // TODO - Populate the rest properly?
    } 
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (timers);
}
