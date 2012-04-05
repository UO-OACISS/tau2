#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_EXPERIMENT* perfdmf_query_experiment(PGconn* connection, PERFDMF_APPLICATION* application, char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling perfdmf_query_experiment(%p, '%s')\n", application, name);
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
  char my_query[256];
  sprintf(my_query, "DECLARE myportal CURSOR FOR select * from experiment where application = %d and name = '%s'", application->id, name);
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

  PERFDMF_EXPERIMENT* experiment = perfdmf_create_experiments(1);

  /* first, print out the attribute names */
  nFields = PQnfields(res);
  experiment->primary_metadata = taudb_create_primary_metadata(nFields);

  /* next, print out the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int metaIndex = 0;
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    experiment->id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    //experiment->name = PQgetvalue(res, i, j);
		experiment->name = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res,i,j))));
		strcpy(experiment->name, PQgetvalue(res,i,j));
	  } else {
	    experiment->primary_metadata[metaIndex].name = (char*)(malloc(sizeof(char)*strlen(PQfname(res,j))));
		strcpy(experiment->primary_metadata[metaIndex].name, PQfname(res, j));
	    experiment->primary_metadata[metaIndex].value = (char*)(malloc(sizeof(char)*strlen(PQgetvalue(res,i,j))));
		strcpy(experiment->primary_metadata[metaIndex].value, PQgetvalue(res, i, j));
	  }
	} 
    experiment->primary_metadata_count = metaIndex;
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return experiment;
}
