#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER* taudb_query_main_timer(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_main_timer(%p)\n", trial);
#endif
  void *res;
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
    sprintf(my_query,"select interval_location_profile.inclusive, interval_event.* from interval_location_profile left outer join interval_event on interval_location_profile.interval_event = interval_event.id where interval_event.trial = %d order by 1 desc limit 1", trial->id);
  } else {
    sprintf(my_query,"select * from timer where trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
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
	    //timers[i].name = taudb_get_value(res, i, j);
		timers[i].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
#ifdef TAUDB_DEBUG
        //printf("Got timer '%s'\n", timers[i].name);
#endif
	  } else if (strcmp(taudb_get_column_name(res, j), "short_name") == 0) {
		timers[i].short_name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
	  } else if (strcmp(taudb_get_column_name(res, j), "source_file") == 0) {
	    //timers[i].source_file = taudb_get_value(res, i, j);
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
		  groups[0].name = group;
		  group = strtok(NULL, "|");
		  while (group != NULL) {
	        TAUDB_TIMER_GROUP* groups = taudb_resize_timer_groups(timers[i].group_count+1, groups);
		    groups[timers[i].group_count].id = 0;
		    groups[timers[i].group_count].name = group;
		    timers[i].group_count++;
		    group = strtok(NULL, "|");
		  }
		} else {
		  timers[i].group_count = 0;
		  timers[i].groups = NULL;
		}
	  } else if (strcmp(taudb_get_column_name(res, j), "inclusive") == 0) {
	    continue;
	  } else {
	    printf("Error: unknown column '%s'\n", taudb_get_column_name(res, j));
	    taudb_exit_nicely(connection);
	  }
	  // TODO - Populate the rest properly?
	} 
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return (timers);
}
