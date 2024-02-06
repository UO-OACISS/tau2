#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_EXPERIMENT* perfdmf_query_experiments(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_experiments(%p)\n", application);
#endif
  int nFields;
  int i, j;

  /*
   * Our test case here involves using a cursor, for which we must be
   * inside a transaction block.  We could do the whole thing with a
   * single PQexec() of "select * from table_name", but that's too
   * trivial to make a good example.
   */

  /* Start a transaction block */
  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  snprintf(my_query, sizeof(my_query), "select * from experiment where application = %d", application->id);
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  PERFDMF_EXPERIMENT* experiments = perfdmf_create_experiments(nRows);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
	    experiments[i].id = atoi(taudb_get_value(connection, i, j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
		experiments[i].name = taudb_strdup(taudb_get_value(connection,i,j));
	  } else {
    	TAUDB_PRIMARY_METADATA* primary_metadata = taudb_create_primary_metadata(nFields);
	    primary_metadata->name = taudb_strdup(taudb_get_column_name(connection, j));
	    primary_metadata->value = taudb_strdup(taudb_get_value(connection, i, j));
        HASH_ADD(hh, experiments[i].primary_metadata, name, (strlen(primary_metadata->name)), primary_metadata);
	  }
	} 
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);
  
  return experiments;
}
