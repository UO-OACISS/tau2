#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_APPLICATION* perfdmf_query_application(TAUDB_CONNECTION* connection, char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_application('%s')\n", name);
#endif
  int                     nFields;
  int                     i, j;

  /* Start a transaction block */
  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256] = "select * from application where name = '";
  strcat(my_query, name);
  strcat(my_query, "'");
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif
  PERFDMF_APPLICATION* application = perfdmf_create_applications(1);

  taudb_execute_query(connection, my_query);

  /* first, print out the attribute names */
  nFields = taudb_get_num_columns(connection);

  /* next, print out the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
	    application->id = atoi(taudb_get_value(connection, i, j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
	    //application->name = taudb_get_value(connection, i, j);
		application->name = taudb_strdup(taudb_get_value(connection,i,j));

	  } else {
  		TAUDB_PRIMARY_METADATA* primary_metadata = taudb_create_primary_metadata(1);
	    primary_metadata->name = taudb_strdup(taudb_get_column_name(connection,j));
	    primary_metadata->value = taudb_strdup(taudb_get_value(connection,i,j));
		HASH_ADD(hh, application->primary_metadata, name, (strlen(primary_metadata->name)), primary_metadata);
	  }
	} 
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);
 
  return application;
}
