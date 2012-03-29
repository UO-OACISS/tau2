#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_APPLICATION* taudb_get_application(char* name) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_application('%s')\n", name);
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
  res = PQexec(_taudb_connection, "BEGIN");
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "BEGIN command failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    taudb_exit_nicely();
  }

  /*
   * Should PQclear PGresult whenever it is no longer needed to avoid
   * memory leaks
   */
  PQclear(res);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256] = "DECLARE myportal CURSOR FOR select * from application where name = '";
  strcat(my_query, name);
  strcat(my_query, "'");
  res = PQexec(_taudb_connection, my_query);
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "DECLARE CURSOR failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    taudb_exit_nicely();
  }
  PQclear(res);

  res = PQexec(_taudb_connection, "FETCH ALL in myportal");
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    fprintf(stderr, "FETCH ALL failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    taudb_exit_nicely();
  }

  PERFDMF_APPLICATION* application = taudb_create_applications(1);

  /* first, print out the attribute names */
  nFields = PQnfields(res);
  application->primary_metadata = taudb_create_primary_metadata(nFields);

  /* next, print out the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int metaIndex = 0;
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    application->id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    application->name = PQgetvalue(res, i, j);
	  } else {
	    application->primary_metadata[metaIndex].name = PQfname(res, j);
	    application->primary_metadata[metaIndex].value = PQgetvalue(res, i, j);
		metaIndex++;
	  }
	} 
    application->primary_metadata_count = metaIndex;
  }


  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  
  return application;
}
