#include "taudb_internal.h"
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
  snprintf(my_query, sizeof(my_query),  "select * from experiment where application = %d and name = '%s'", application->id, name);
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  taudb_execute_query(connection, my_query);

  PERFDMF_EXPERIMENT* experiment = perfdmf_create_experiments(1);

  /* first, print out the attribute names */
  nFields = taudb_get_num_columns(connection);

  /* next, print out the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
	    experiment->id = atoi(taudb_get_value(connection, i, j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
	    //experiment->name = taudb_get_value(connection, i, j);
		experiment->name = taudb_strdup(taudb_get_value(connection,i,j));
	  } else {
	  	TAUDB_PRIMARY_METADATA* primary_metadata = taudb_create_primary_metadata(1);
	    primary_metadata->name = taudb_strdup(taudb_get_column_name(connection,j));
	    primary_metadata->value = taudb_strdup(taudb_get_value(connection,i,j));
		HASH_ADD(hh, experiment->primary_metadata, name, (strlen(primary_metadata->name)), primary_metadata);
	  }
	} 
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);
  
  return experiment;
}
