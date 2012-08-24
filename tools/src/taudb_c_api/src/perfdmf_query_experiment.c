#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_EXPERIMENT* perfdmf_query_experiment(TAUDB_CONNECTION* connection, PERFDMF_APPLICATION* application, char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_experiment(%p, '%s')\n", application, name);
#endif
  int                     nFields;
  int                     i, j;

  taudb_begin_transaction(connection);
  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256];
  sprintf(my_query, "select * from experiment where application = %d and name = '%s'", application->id, name);
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  void* res = taudb_execute_query(connection, my_query);

  PERFDMF_EXPERIMENT* experiment = perfdmf_create_experiments(1);

  /* first, print out the attribute names */
  nFields = taudb_get_num_columns(res);
  experiment->primary_metadata = taudb_create_primary_metadata(nFields);

  /* next, print out the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    int metaIndex = 0;
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
	    experiment->id = atoi(taudb_get_value(res, i, j));
	  } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
	    //experiment->name = taudb_get_value(res, i, j);
		experiment->name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
	  } else {
	    experiment->primary_metadata[metaIndex].name = taudb_create_and_copy_string(taudb_get_column_name(res,j));
	    experiment->primary_metadata[metaIndex].value = taudb_create_and_copy_string(taudb_get_value(res,i,j));
	  }
	} 
    experiment->primary_metadata_count = metaIndex;
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);
  
  return experiment;
}
