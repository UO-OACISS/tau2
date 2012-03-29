#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

PERFDMF_EXPERIMENT* taudb_get_experiments(PERFDMF_APPLICATION* application) {
#ifdef TAUDB_DEBUG
  printf("Calling taudb_get_experiments(%p)\n", application);
#endif
  PGresult *res;
  int nFields;
  int i, j;

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
  char my_query[256];
  sprintf(my_query,"DECLARE myportal CURSOR FOR select * from experiment where application = %d", application->id);
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

  int nRows = PQntuples(res);
  PERFDMF_EXPERIMENT* experiments = taudb_create_experiments(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    int metaIndex = 0;
    experiments[i].primary_metadata = taudb_create_primary_metadata(nFields);
    /* the columns */
    for (j = 0; j < nFields; j++) {
	  if (strcmp(PQfname(res, j), "id") == 0) {
	    experiments[i].id = atoi(PQgetvalue(res, i, j));
	  } else if (strcmp(PQfname(res, j), "name") == 0) {
	    experiments[i].name = PQgetvalue(res, i, j);
	  } else {
	    experiments[i].primary_metadata[metaIndex].name = PQfname(res, j);
	    experiments[i].primary_metadata[metaIndex].value = PQgetvalue(res, i, j);
		metaIndex++;
	  }
	} 
    experiments[i].primary_metadata_count = metaIndex;
  }


  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  
  return experiments;
}
