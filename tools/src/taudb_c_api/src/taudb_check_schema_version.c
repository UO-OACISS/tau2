#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined __TAUDB_SQLITE__
static int taudb_sqlite_callback(void *NotUsed, int argc, char **argv, char **azColName) {
  if (argc > 0) {
    taudb_version = TAUDB_2005_SCHEMA;
  }
  return 0;
}
#endif

int taudb_check_schema_version(TAUDB_CONNECTION* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_check_schema_version()\n");
#endif

  char my_query[256] = "select * from application";
#if defined __TAUDB_POSTGRESQL__
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
#elif defined __TAUDB_SQLITE__
  char *zErrMsg = 0;
  connection->rc = sqlite3_exec(connection->connection, my_query, taudb_sqlite_callback, 0, &zErrMsg);
  if( connection->rc!=SQLITE_OK ){
    taudb_version = TAUDB_2012_SCHEMA;
    connection->schema_version = TAUDB_2012_SCHEMA;
    sqlite3_free(zErrMsg);
  } else {
    // taudb_version gets set in the callback function
    connection->schema_version = taudb_version;
  }
#endif
  return 0;
}
