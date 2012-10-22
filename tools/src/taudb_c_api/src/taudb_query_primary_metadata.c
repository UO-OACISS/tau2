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
