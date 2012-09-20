#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_APPLICATION* perfdmf_query_applications(TAUDB_CONNECTION* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_applications()\n");
#endif
  int                     nFields;
  int                     i, j;

  /* Start a transaction block */
  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256] = "select * from application";
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
#endif

  void* res = taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(res);
  PERFDMF_APPLICATION* applications = perfdmf_create_applications(nRows);
  taudb_numItems = nRows;

  nFields = taudb_get_num_columns(res);

  /* the rows */
  for (i = 0; i < nRows; i++)
  {
    int metaIndex = 0;
    applications[i].primary_metadata = taudb_create_primary_metadata(nFields);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(res, j), "id") == 0) {
	    applications[i].id = atoi(taudb_get_value(res, i, j));
	  } else if (strcmp(taudb_get_column_name(res, j), "name") == 0) {
	    //applications[i].name = taudb_get_value(res, i, j);
		applications[i].name = taudb_create_and_copy_string(taudb_get_value(res,i,j));
	  } else {
	    applications[i].primary_metadata[metaIndex].name = taudb_get_column_name(res, j);
	    applications[i].primary_metadata[metaIndex].value = taudb_get_value(res, i, j);
		metaIndex++;
	  }
	} 
    applications[i].primary_metadata_count = metaIndex;
  }

  taudb_clear_result(res);
  taudb_close_transaction(connection);

  return applications;
}
