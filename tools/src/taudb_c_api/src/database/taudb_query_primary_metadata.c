#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_PRIMARY_METADATA* taudb_query_primary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_primary_metadata(%p)\n", trial);
#endif
  int nFields;
  int i, j;

  char my_query[1024];
  if (trial != NULL) { // the user wants a specific trial, so get it
    sprintf(my_query,"select name, value from primary_metadata where trial = %d", trial->id);
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
  //int metaIndex = trial->primary_metadata_count;
  //TAUDB_PRIMARY_METADATA* pm = taudb_resize_primary_metadata(nRows+trial->primary_metadata_count, trial->primary_metadata);
  // the resize should do this, but just in case...
  //for (i = 0; i < trial->primary_metadata_count; i++) {
    //pm[i].name = trial->primary_metadata[i].name;
    //pm[i].value = trial->primary_metadata[i].value;
  //}

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    TAUDB_PRIMARY_METADATA* pm = taudb_create_primary_metadata(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        pm->name = taudb_create_and_copy_string(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "value") == 0) {
        pm->value = taudb_create_and_copy_string(taudb_get_value(connection,i,j));
      } else {
	    fprintf(stderr,"Unknown primary_metadata column: %s\n", taudb_get_column_name(connection, j));
      }
    } 
    HASH_ADD_KEYPTR(hh, trial->primary_metadata, pm->name, strlen(pm->name), pm);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  taudb_numItems = nRows;

  return trial->primary_metadata;
}

TAUDB_PRIMARY_METADATA* taudb_get_primary_metadata_by_name(TAUDB_PRIMARY_METADATA* primary_metadatas, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_primary_metadata_by_id(%p,%s)\n", primary_metadatas, name);
#endif
  if (primary_metadatas == NULL) {
    fprintf(stderr, "Error: primary_metadata parameter null. Please provide a valid set of primary_metadatas.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_PRIMARY_METADATA* primary_metadata = NULL;
  HASH_FIND(hh, primary_metadatas, name, strlen(name), primary_metadata);
#ifdef ITERATE_ON_FAILURE
  // HASH_FIND is not working so well... now we iterate. Sigh.
  if (primary_metadata == NULL) {
#ifdef TAUDB_DEBUG
      printf ("PRIMARY_METADATA not found, iterating...\n");
#endif
    TAUDB_PRIMARY_METADATA *current, *tmp;
    HASH_ITER(hh, primary_metadatas, current, tmp) {
#ifdef TAUDB_DEBUG_DEBUG
      printf ("PRIMARY_METADATA: '%s'\n", current->name);
#endif
      if (strcmp(current->name, name) == 0) {
        return current;
      }
    }
  }
#endif
  return primary_metadata;
}


