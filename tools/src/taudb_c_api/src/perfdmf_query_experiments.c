#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_EXPERIMENT* perfdmf_query_experiments(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application) {
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
  sprintf(my_query,"select * from experiment where application = %d", application->id);
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  void* res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  PERFDMF_EXPERIMENT* experiments = perfdmf_create_experiments(nRows);
  taudb_numItems = nRows;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    int metaIndex = 0;
    experiments[i].primary_metadata = taudb_create_primary_metadata(nFields);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
	    experiments[i].id = atoi(taudb_get_value(res, i, j));
	  } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
	    //experiments[i].name = taudb_get_value(res, i, j);
		experiments[i].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
	  } else {
	    experiments[i].primary_metadata[metaIndex].name = taudb_get_column_name(res, j);
	    experiments[i].primary_metadata[metaIndex].value = taudb_get_value(res, i, j);
		metaIndex++;
	  }
	} 
    experiments[i].primary_metadata_count = metaIndex;
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);
  
  return experiments;
}
