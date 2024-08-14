#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern TAUDB_SECONDARY_METADATA* taudb_get_secondary_metadata_by_id(TAUDB_SECONDARY_METADATA* secondary_metadatas, const char* id);

TAUDB_SECONDARY_METADATA* taudb_query_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_secondary_metadata(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  char my_query[1024];
  if (trial != NULL) { // the user wants a specific trial, so get it
    snprintf(my_query, sizeof(my_query), "select sm.* from secondary_metadata sm left outer join timer_callpath tcp on sm.timer_callpath = tcp.id left outer join thread h on sm.thread = h.id where sm.trial = %d", trial->id);
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
        secondary_metadata->id = taudb_strdup(taudb_get_value(connection, i, j));
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
          secondary_metadata->key.parent = taudb_get_secondary_metadata_by_id(trial->secondary_metadata, tmpStringID);
	      if (secondary_metadata->key.parent != NULL) {
	        secondary_metadata->key.parent->child_count++;
	      }
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "is_array") == 0) {
	    if (strcmp(taudb_get_value(connection, i, j), "t") == 0) {
          is_array = TRUE;
		  fprintf(stderr, "WARNING! Array metadata not yet supported...\n");
		}
      } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        secondary_metadata->key.name = taudb_strdup(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "value") == 0) {
	    // just get the whole thing now, TODO: split into array later?
	    secondary_metadata->value = (char**)calloc(1,sizeof(char*));
        secondary_metadata->value[0] = taudb_strdup(taudb_get_value(connection,i,j));
        secondary_metadata->num_values = 1;
      } else {
	    fprintf(stderr,"Unknown secondary_metadata column: %s\n", taudb_get_column_name(connection, j));
      }
    } 
	if (secondary_metadata->key.parent != NULL) {
	  taudb_add_secondary_metadata_to_secondary_metadata(secondary_metadata->key.parent, secondary_metadata);
	} else {
	  taudb_add_secondary_metadata_to_trial(trial, secondary_metadata);
	}
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  *taudb_numItems = nFields;

  return trial->secondary_metadata;
}

void taudb_add_secondary_metadata_to_secondary_metadata(TAUDB_SECONDARY_METADATA* parent, TAUDB_SECONDARY_METADATA* child) {
  HASH_ADD_KEYPTR(hh2, parent->children, &(child->key), sizeof(child->key), child);
}

void taudb_add_secondary_metadata_to_trial(TAUDB_TRIAL* trial, TAUDB_SECONDARY_METADATA* secondary_metadata) {
  HASH_ADD_KEYPTR(hh, trial->secondary_metadata, secondary_metadata->id, strlen(secondary_metadata->id), secondary_metadata);
  HASH_ADD_KEYPTR(hh2, trial->secondary_metadata_by_key, &(secondary_metadata->key), sizeof(secondary_metadata->key), secondary_metadata);
}

TAUDB_SECONDARY_METADATA* taudb_get_secondary_metadata_by_id(TAUDB_SECONDARY_METADATA* secondary_metadatas, const char* id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_secondary_metadata_by_id(%p,%s)\n", secondary_metadatas, id);
#endif
  if (secondary_metadatas == NULL) {
    // not an error, the list could be empty
    //fprintf(stderr, "Error: secondary_metadata parameter null. Please provide a valid set of secondary_metadatas.\n");
    return NULL;
  }
  if (id == NULL) {
    fprintf(stderr, "Error: id parameter null. Please provide a valid id.\n");
    return NULL;
  }

  TAUDB_SECONDARY_METADATA* secondary_metadata = NULL;
  HASH_FIND(hh, secondary_metadatas, id, strlen(id), secondary_metadata);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (secondary_metadata == NULL) {
#ifdef TAUDB_DEBUG
      printf ("SECONDARY_METADATA not found, iterating...\n");
#endif
    TAUDB_SECONDARY_METADATA *current, *tmp;
    HASH_ITER(hh, secondary_metadatas, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("SECONDARY_METADATA: '%s'\n", current->id);
#endif
      if (strcmp(current->id, id) == 0) {
        return current;
      }
    }
  }
#endif
  return secondary_metadata;
}

void taudb_private_save_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, TAUDB_SECONDARY_METADATA* secondary_metadata, TAUDB_SECONDARY_METADATA* parent, const char* statement_name, boolean update) {
	const char* insert_query = "insert into secondary_metadata (id, trial, thread, timer_callpath, time_range, parent, name, value, is_array) values ($1, $2, $3, $4, $5, $6, $7, $8, $9);";
  	const char* insert_statement_name = "TAUDB_INSERT_SECONDARY_METADATA";
	const int insert_nParams = 9;
	// make array of 6 character pointers
    const char* paramValues[9] = {0};
    paramValues[0] = secondary_metadata->id;
    char trialid[32] = {0};
    snprintf(trialid, sizeof(trialid),  "%d", trial->id);
    paramValues[1] = trialid;
    char thread[32] = {0};
    snprintf(thread, sizeof(thread),  "%d", secondary_metadata->key.thread->id);
    paramValues[2] = thread;
    char timer_callpath[32] = {0};
	if (secondary_metadata->key.timer_callpath != NULL) {
      snprintf(timer_callpath, sizeof(timer_callpath),  "%d", secondary_metadata->key.timer_callpath->id);
      paramValues[3] = timer_callpath;
	}
    char time_range[32] = {0};
	if (secondary_metadata->key.time_range != NULL) {
      snprintf(time_range, sizeof(time_range),  "%d", secondary_metadata->key.time_range->id);
      paramValues[4] = time_range;
	}
	if (parent != NULL) {
      paramValues[5] = parent->id;
	}

    paramValues[6] = secondary_metadata->key.name;
	if (secondary_metadata->num_values > 1) {
	  int i = 0;
	  int length = 0;
	  for (i = 0 ; i < secondary_metadata->num_values ; i++) {
	    length = length + strlen(secondary_metadata->value[i]) + 1;
	  }
	  char *tmpstr = calloc(length, sizeof(char));
	  snprintf(tmpstr, length,  "[%s", secondary_metadata->value[i]);
	  for (i = 1 ; i < secondary_metadata->num_values ; i++) {
	    snprintf(tmpstr, length,  "%s,%s", tmpstr, secondary_metadata->value[i]);
	  }
	  snprintf(tmpstr, length,  "%s]", tmpstr);
      paramValues[7] = tmpstr;
      paramValues[8] = "TRUE";
	} else {
      paramValues[7] = secondary_metadata->value[0];
      paramValues[8] = "FALSE";
	}
    int rows = taudb_execute_statement(connection, statement_name, 9, paramValues);

		if(update && rows == 0) {
#ifdef TAUDB_DEBUG
			printf("Falling back to insert for update of seconday metadata.\n");
#endif
			/* updated row didn't exist; insert instead */
			taudb_prepare_statement(connection, insert_statement_name, insert_query, insert_nParams);
			taudb_execute_statement(connection, insert_statement_name, insert_nParams, paramValues);
		}

    TAUDB_SECONDARY_METADATA *child, *tmp;
    HASH_ITER(hh2, secondary_metadata->children, child, tmp) {
      taudb_private_save_secondary_metadata(connection, trial, child, secondary_metadata, statement_name, update);
	}
}

void taudb_save_secondary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  const char* my_query;
	const char* statement_name;
	int nParams;
	
	const char* insert_query = "insert into secondary_metadata (id, trial, thread, timer_callpath, time_range, parent, name, value, is_array) values ($1, $2, $3, $4, $5, $6, $7, $8, $9);";
  	const char* insert_statement_name = "TAUDB_INSERT_SECONDARY_METADATA";
	const int insert_nParams = 9;
	const char* update_query = "update secondary_metadata set trial=$2, thread=$3, timer_callpath=$4, time_range=$5, parent=$6, name=$7, value=$8, is_array=$9 where id=$1;";
	const char* update_statement_name = "TAUDB_UPDATE_SECONDARY_METADATA";
	const int update_nParams = 9;
	
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
  TAUDB_SECONDARY_METADATA *secondary_metadata, *tmp;
  HASH_ITER(hh2, trial->secondary_metadata_by_key, secondary_metadata, tmp) {
    taudb_private_save_secondary_metadata(connection, trial, secondary_metadata, NULL, statement_name, update);
  }
  taudb_clear_result(connection);
}

TAUDB_SECONDARY_METADATA* taudb_next_secondary_metadata_by_key_from_trial(TAUDB_SECONDARY_METADATA* current) {
  return current->hh2.next;
}

TAUDB_SECONDARY_METADATA* taudb_next_secondary_metadata_by_id_from_trial(TAUDB_SECONDARY_METADATA* current) {
  return current->hh.next;
}

