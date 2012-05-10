#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef TAUDB_PERFDMF
int num_tables = 11;
char* taudb_tables[] = {
  "application",
  "experiment",
  "trial",
  "metric",
  "interval_event",
  "atomic_event",
  "interval_location_profile",
  "atomic_location_profile",
  "interval_total_summary",
  "interval_mean_summary",
  "machine_thread_map"
};
#else
int num_tables = 13;
char* taudb_tables[] = {
   "data_source",
   "trial",
   "thread",
   "primary_metadata",
   "secondary_metadata",
   "metric",
   "timer",
   "timer_group",
   "timer_parameter",
   "timer_callpath",
   "counter",
   "counter_group",
   "timer_value",
   "counter_value"
};
#endif

int taudb_api_test(PGconn* connection, char* table_name) {
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
  printf("Selecting from %s...\n", table_name);
  char my_query[256] = "DECLARE myportal CURSOR FOR select * from ";
  strcat(my_query, table_name);
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

  /* first, print out the attribute names */
  nFields = PQnfields(res);
  for (i = 0; i < nFields; i++)
    printf("%-15s", PQfname(res, i));
  printf("\n\n");

  /* next, print out the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    for (j = 0; j < nFields; j++)
            printf("%-15s", PQgetvalue(res, i, j));
    printf("\n");
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  return 0;
}

void taudb_iterate_tables(PGconn* connection) {
   int i;
   for (i = 0 ; i < num_tables ; i = i+1) {
     taudb_api_test(connection, taudb_tables[i]);
   }
}


