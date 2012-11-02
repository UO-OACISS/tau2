#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int taudb_check_schema_version(TAUDB_CONNECTION* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_check_schema_version()\n");
#endif

  char my_query[256] = "select * from application";
#ifdef __TAUDB_POSTGRESQL__
  PGresult *res;
  res = PQexec(connection->connection, my_query);
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    taudb_version = TAUDB_2012_SCHEMA;
    connection->schema_version = TAUDB_2012_SCHEMA;
    //fprintf(stderr, "SELECT failed: %s", PQerrorMessage(_taudb_connection));
    PQclear(res);
    //taudb_exit_nicely();
	return 0;
  }

  int nRows = PQntuples(res);

  if (nRows > 0) {
    taudb_version = TAUDB_2005_SCHEMA;
    connection->schema_version = TAUDB_2005_SCHEMA;
  }

  PQclear(res);
#endif
  return 0;
}
