#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int taudb_check_schema_version(PGconn* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_check_schema_version()\n");
#endif
  PGresult *res;
  int nFields;

  /*
   * Our test case here involves using a cursor, for which we must be
   * inside a transaction block.  We could do the whole thing with a
   * single PQexec() of "select * from table_name", but that's too
   * trivial to make a good example.
   */

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  char my_query[256] = "select * from application";
  res = PQexec(connection, my_query);
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    taudb_version = TAUDB_2012_SCHEMA;
    //fprintf(stderr, "SELECT failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    //taudb_exit_nicely();
	return 0;
  }

  int nRows = PQntuples(res);

  if (nRows > 0) {
    taudb_version = TAUDB_2005_SCHEMA;
  }

  PQclear(res);

  return 0;
}
