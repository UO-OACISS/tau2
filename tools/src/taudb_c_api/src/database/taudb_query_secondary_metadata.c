#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_secondary_metadata(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  char my_query[1024];
  if (trial != NULL) { // the user wants a specific trial, so get it
    sprintf(my_query,"select sm.* from secondary_metadata sm left outer join timer_callpath tcp on sm.timer_callpath = tcp.id left outer join thread h on sm.thread = h.id where sm.trial = %d", trial->id);
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
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < nRows; i++)
  {
    boolean is_array = FALSE;
	int tmpID = 0;
	char *tmpStringID = NULL;
    TAUDB_SECONDARY_METADATA* secondary_metadata = taudb_create_secondary_metadata(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        secondary_metadata->id = taudb_create_and_copy_string(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "trial") == 0) {
      } else if (strcmp(taudb_get_column_name(connection, j), "thread") == 0) {
        tmpID = atoi(taudb_get_value(connection, i, j));
		if (tmpID > 0) {
		  secondary_metadata->key.thread = taudb_get_thread(trial->threads, tmpID);
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "timer_callpath") == 0) {
        tmpID = atoi(taudb_get_value(connection, i, j));
		if (tmpID > 0) {
          secondary_metadata->key.timer_callpath = taudb_get_timer_callpath_by_id(trial->timer_callpaths_by_id, tmpID);
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "time_range") == 0) {
        tmpID = atoi(taudb_get_value(connection, i, j));
		if (tmpID > 0) {
          secondary_metadata->key.time_range = taudb_get_time_range(trial->time_ranges, tmpID);
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "parent") == 0) {
	    // don't need to copy this, we will only need it temporarily
        tmpStringID = taudb_get_value(connection, i, j);
		if (tmpStringID > 0) {
          //secondary_metadata->parent = taudb_get_secondary_metadata_by_id(trial->secondary_metadata, tmpID);
	      //secondary_metadata->parent->child_count++;
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "is_array") == 0) {
	    if (strcmp(taudb_get_value(connection, i, j), "t") == 0) {
          is_array = TRUE;
		  fprintf(stderr, "WARNING! Array metadata not yet supported...\n");
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        secondary_metadata->key.name = taudb_create_and_copy_string(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "value") == 0) {
	    // just get the whole thing now, TODO: split into array later?
	    secondary_metadata->value = (char**)calloc(1,sizeof(char*));
        secondary_metadata->value[0] = taudb_create_and_copy_string(taudb_get_value(connection,i,j));
        secondary_metadata->num_values = 1;
      } else {
	    fprintf(stderr,"Unknown secondary_metadata column: %s\n", taudb_get_column_name(connection, j));
      }
    } 
	HASH_ADD_KEYPTR(hh, trial->secondary_metadata, secondary_metadata->id, strlen(secondary_metadata->id), secondary_metadata);
	HASH_ADD_KEYPTR(hh2, trial->secondary_metadata, &(secondary_metadata->key), sizeof(secondary_metadata->key), secondary_metadata);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  taudb_numItems = nFields;

  return trial->secondary_metadata;
}
