#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_APPLICATION* perfdmf_query_applications(TAUDB_CONNECTION* connection,int* taudb_numItems) {
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

  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  PERFDMF_APPLICATION* applications = perfdmf_create_applications(nRows);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < nRows; i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
	    applications[i].id = atoi(taudb_get_value(connection, i, j));
	  } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
	    //applications[i].name = taudb_get_value(connection, i, j);
		applications[i].name = taudb_strdup(taudb_get_value(connection,i,j));
	  } else {
        TAUDB_PRIMARY_METADATA* primary_metadata = taudb_create_primary_metadata(nFields);
	    primary_metadata->name = taudb_strdup(taudb_get_column_name(connection, j));
	    primary_metadata->value = taudb_strdup(taudb_get_value(connection, i, j));
		HASH_ADD(hh, applications[i].primary_metadata, name, (strlen(primary_metadata->name)), primary_metadata);
	  }
	} 
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return applications;
}
