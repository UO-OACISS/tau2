#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_PRIMARY_METADATA* taudb_query_primary_metadata(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_primary_metadata(%d, %p)\n", full, trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  char my_query[1024];
  if (trial != NULL) { // the user wants a specific trial, so get it
    sprintf(my_query,"DECLARE myportal CURSOR FOR select name, value from primary_metadata where trial = %d", trial->id);
  } else {
    fprintf(stderr, "You don't want all the metadata. Please specify a trial.\n");
  }

  /*
   * Our test case here involves using a cursor, for which we must be
   * inside a transaction block.  We could do the whole thing with a
   * single PQexec() of "select * from table_name", but that's too
   * trivial to make a good example.
   */

  /* Start a transaction block */
  res = PQexec(connection, "BEGIN");
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "BEGIN command failed: %s", PQerrorMessage(connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }

  /*
   * Should PQclear PGresult whenever it is no longer needed to avoid
   * memory leaks
   */
  PQclear(res);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
  res = PQexec(connection, my_query);
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "DECLARE CURSOR failed: %s", PQerrorMessage(connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }
  PQclear(res);

  res = PQexec(connection, "FETCH ALL in myportal");
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    fprintf(stderr, "FETCH ALL failed: %s", PQerrorMessage(connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }

  int nRows = PQntuples(res);
  int metaIndex = trial->primary_metadata_count;
  TAUDB_PRIMARY_METADATA* pm = taudb_resize_primary_metadata(nRows+trial->primary_metadata_count, trial->primary_metadata);
  // the resize should do this, but just in case...
  //for (i = 0; i < trial->primary_metadata_count; i++) {
    //pm[i].name = trial->primary_metadata[i].name;
    //pm[i].value = trial->primary_metadata[i].value;
  //}

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "name") == 0) {
        pm[metaIndex].name = taudb_create_and_copy_string(PQfname(res, j));
      } else if (strcmp(PQfname(res, j), "value") == 0) {
        pm[metaIndex].value = taudb_create_and_copy_string(PQgetvalue(res,i,j));
        metaIndex++;
      } else {
	    fprintf(stderr,"Unknown primary_metadata column: %s\n", PQfname(res, j));
      }
    } 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  taudb_numItems = metaIndex;

  return pm;
}
