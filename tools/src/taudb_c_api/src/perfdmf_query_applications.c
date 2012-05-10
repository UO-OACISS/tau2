#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_APPLICATION* perfdmf_query_applications(PGconn* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_applications()\n");
#endif
  PGresult   *res;
  int                     nFields;
  int                     i, j;

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
  char my_query[256] = "DECLARE myportal CURSOR FOR select * from application";
#ifdef TAUDB_DEBUG
  printf("'%s'\n",my_query);
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
  PERFDMF_APPLICATION* applications = perfdmf_create_applications(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int metaIndex = 0;
    applications[i].primary_metadata = taudb_create_primary_metadata(nFields);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    applications[i].id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    //applications[i].name = PQgetvalue(res, i, j);
		applications[i].name = taudb_create_and_copy_string(PQgetvalue(res,i,j));
	  } else {
	    applications[i].primary_metadata[metaIndex].name = PQfname(res, j);
	    applications[i].primary_metadata[metaIndex].value = PQgetvalue(res, i, j);
		metaIndex++;
	  }
	} 
    applications[i].primary_metadata_count = metaIndex;
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return applications;
}
