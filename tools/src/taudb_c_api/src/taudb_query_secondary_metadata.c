#include "taudb_internal.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_secondary_metadata(%p)\n", trial);
#endif
  void *res;
  int nFields;
  int i, j;

  char my_query[1024];
  if (trial != NULL) { // the user wants a specific trial, so get it
    sprintf(my_query,"select sm.*, t.name as timer_name, h.thread_index from secondary_metadata sm left outer join timer t on sm.timer = t.id left outer join thread h on sm.thread = h.id where sm.trial = %d", trial->id);
  } else {
    fprintf(stderr, "You don't want all the metadata. Please specify a trial.\n");
  }

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  TAUDB_SECONDARY_METADATA* pm = taudb_create_secondary_metadata(nRows);

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    boolean is_array = FALSE;
	pm[i].child_count = 0;
	pm[i].num_values = 1;
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
        pm[i].id = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "trial") == 0) {
        //pm[i].trial = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "thread") == 0) {
        //pm[i].thread = atoi(taudb_get_value(res, i, j));
		fprintf(stderr, "TODO: ASSIGN THE THREAD FOR THE SECONDARY METADATA\n");
      } else if (strcmp(taudb_get_column_name(res, j), "timer_call_data") == 0) {
        //pm[i].timer_call_data = atoi(taudb_get_value(res, i, j));
		fprintf(stderr, "TODO: ASSIGN THE TIMER_CALL_DATA FOR THE SECONDARY METADATA\n");
      } else if (strcmp(taudb_get_column_name(res, j), "timer_call_data") == 0) {
      } else if (strcmp(taudb_get_column_name(res, j), "parent") == 0) {
        //pm[i].parent = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "is_array") == 0) {
	    if (strcmp(taudb_get_value(res, i, j), "t") == 0) {
          is_array = TRUE;
		  fprintf(stderr, "WARNING! Array metadata not yet supported...\n");
		}
      } else if (strcmp(taudb_get_column_name(res, j), "timer_name") == 0) {
        //pm[i].timer_name = (char*)(malloc(sizeof(char)*strlen(taudb_get_value(res, i, j))));
        //strcpy(pm[i].name, taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "thread_index") == 0) {
        //pm[i].thread_index = atoi(taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
        pm[i].name = (char*)(malloc(sizeof(char)*strlen(taudb_get_value(res, i, j))));
        strcpy(pm[i].name, taudb_get_value(res, i, j));
      } else if (strcmp(taudb_get_column_name(res, j), "value") == 0) {
        pm[i].value = (char**)(malloc(sizeof(char*)));
        pm[i].value[0] = (char*)(malloc(sizeof(char)*strlen(taudb_get_value(res,i,j))));
        strcpy(pm[i].value[0], taudb_get_value(res, i, j));
      } else {
	    fprintf(stderr,"Unknown secondary_metadata column: %s\n", taudb_get_column_name(res, j));
      }
    } 
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  taudb_numItems = nFields;

  return pm;
}
