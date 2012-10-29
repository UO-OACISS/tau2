#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_PRIMARY_METADATA* taudb_query_primary_metadata(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_primary_metadata(%p)\n", trial);
#endif
  void *res;
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
  res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  int metaIndex = trial->primary_metadata_count;
  TAUDB_PRIMARY_METADATA* pm = taudb_resize_primary_metadata(nRows+trial->primary_metadata_count, trial->primary_metadata);
  // the resize should do this, but just in case...
  //for (i = 0; i < trial->primary_metadata_count; i++) {
    //pm[i].name = trial->primary_metadata[i].name;
    //pm[i].value = trial->primary_metadata[i].value;
  //}

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
        pm[metaIndex].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
      } else if (strcmp(taudb_get_column_name(res, j), "value") == 0) {
        pm[metaIndex].value = taudb_create_and_copy_string(taudb_get_value(res,i,j));
        metaIndex++;
      } else {
	    fprintf(stderr,"Unknown primary_metadata column: %s\n", taudb_get_column_name(res, j));
      }
    } 
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  taudb_numItems = metaIndex;

  return pm;
}
