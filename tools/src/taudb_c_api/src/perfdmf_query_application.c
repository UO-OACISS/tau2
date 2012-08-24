#include "taudb_api.h"
#include "libpq-fe.h"
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

  void* res = taudb_execute_query(connection, my_query);

  /* first, print out the attribute names */
  nFields = taudb_get_num_columns(res);
  application->primary_metadata = taudb_create_primary_metadata(nFields);

  /* next, print out the rows */
  for (i = 0; i < taudb_get_num_rows(res); i++)
  {
    int metaIndex = 0;
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
	    application->id = atoi(taudb_get_value(res, i, j));
	  } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
	    //application->name = taudb_get_value(res, i, j);
		application->name = taudb_create_and_copy_string(taudb_get_value(res,i,j));

	  } else {
	    application->primary_metadata[metaIndex].name = taudb_create_and_copy_string(taudb_get_column_name(res,j));
	    application->primary_metadata[metaIndex].value = taudb_create_and_copy_string(taudb_get_value(res,i,j));
		metaIndex++;
	  }
	} 
    application->primary_metadata_count = metaIndex;
  }

  taudb_clear_result(res);
  taudb_close_transaction(res);
 
  return application;
}
