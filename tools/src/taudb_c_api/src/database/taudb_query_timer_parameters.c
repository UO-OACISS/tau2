#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern TAUDB_TIMER_PARAMETER* taudb_process_parameter_timer(TAUDB_TRIAL* trial, TAUDB_TIMER_PARAMETER* timer_parameter);
extern void taudb_trim(char * s);

TAUDB_TIMER_PARAMETER* taudb_query_timer_parameters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_TIMER* timer, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_private_query_timer_parameters(%p,%p)\n", trial, timer);
#endif
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }
  
  // if the Trial already has the parameter data, return it.
  if (timer != NULL && timer->parameters != NULL) {
    *taudb_numItems = HASH_CNT(hh,timer->parameters);
    return timer->parameters;
  }

  taudb_begin_transaction(connection);
  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[1024];
  //sprintf(my_query,"select * from timer where trial = %d", trial->id);
  snprintf(my_query, sizeof(my_query), "select tp.*, t.id as timerid from timer_parameter tp inner join timer t on tc.timer = t.id");
  snprintf(my_query, sizeof(my_query), "%s where t.trial = %d", my_query, trial->id);
  if (timer != NULL) {
    snprintf(my_query, sizeof(my_query), "%s and t.id = %d", my_query, timer->id);
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
    TAUDB_TIMER* tmp_timer = NULL;
    TAUDB_TIMER_PARAMETER* timer_parameter = taudb_create_timer_parameters(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        timer_parameter->name = taudb_strdup(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "timerid") == 0) {
        tmp_timer = taudb_get_timer_by_id(trial->timers_by_id, atoi(taudb_get_value(connection, i, j)));
        //timer_parameter->timer = tmp_timer;
      } else if (strcmp(taudb_get_column_name(connection, j), "value") == 0) {
        timer_parameter->value = taudb_strdup(taudb_get_value(connection, i, j));
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
    } 
	taudb_add_timer_parameter_to_timer(tmp_timer, timer_parameter);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  if (timer != NULL) {
    return (timer->parameters);
  }
  return NULL;
}

// convenience method
TAUDB_TIMER_PARAMETER* taudb_query_all_timer_parameters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
  return taudb_query_timer_parameters(connection, trial, NULL, taudb_numItems);
}

void taudb_add_timer_parameter_to_timer(TAUDB_TIMER* timer, TAUDB_TIMER_PARAMETER* timer_parameter) {
  HASH_ADD_KEYPTR(hh, timer->parameters, timer_parameter->name, strlen(timer_parameter->name), timer_parameter);
}

TAUDB_TIMER_PARAMETER* taudb_get_timer_parameter_by_name(TAUDB_TIMER_PARAMETER* timer_parameters, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_parameter_by_name(%p,%s)\n", timer_parameters, name);
#endif
  if (timer_parameters == NULL) {
    fprintf(stderr, "Error: timer_parameter parameter null. Please provide a valid set of timer_parameters.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER_PARAMETER* timer_parameter = NULL;
  HASH_FIND(hh, timer_parameters, name, strlen(name), timer_parameter);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (timer_parameter == NULL) {
    TAUDB_TIMER_PARAMETER *current, *tmp;
    HASH_ITER(hh, timer_parameters, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("TIMER_PARAMETER: '%s'\n", current->name);
#endif
      if (strcmp(current->name, name) == 0) {
        return current;
      }
    }
  }
#endif
  return timer_parameter;
}

void taudb_save_timer_parameters(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {	
  const char* my_query = "insert into timer_parameter (timer, parameter_name, parameter_value) select $1, $2::varchar, $3::varchar where not exists (select 1 from timer_parameter where timer=$1 and parameter_name=$2 and parameter_value=$3);";
  const char* statement_name = "TAUDB_INSERT_TIMER_PARAMETER";
  taudb_prepare_statement(connection, statement_name, my_query, 3);
  TAUDB_TIMER *timer, *tmp;
  TAUDB_TIMER_PARAMETER *parameter, *tmp2;
  HASH_ITER(trial_hash_by_name, trial->timers_by_name, timer, tmp) {
    HASH_ITER(hh, timer->parameters, parameter, tmp2) {
      // make array of 6 character pointers
      const char* paramValues[3] = {0};
      char timerid[32] = {0};
      snprintf(timerid, sizeof(timerid),  "%d", timer->id);
      paramValues[0] = timerid;
      paramValues[1] = parameter->name;
      paramValues[2] = parameter->value;
      taudb_execute_statement(connection, statement_name, 3, paramValues);
    }
  }
  taudb_clear_result(connection);
}

TAUDB_TIMER_PARAMETER* taudb_next_timer_parameter_by_name_from_timer(TAUDB_TIMER_PARAMETER* current) {
  return current->hh.next;
}


