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
   "measurement",
   "measurement_group",
   "measurement_parameter",
   "counter",
   "counter_group",
   "measurement_value",
   "counter_value"
};
#endif

int taudb_iterate_tables() {
   int i;
   for (i = 0 ; i < num_tables ; i = i+1) {
     taudb_api_test(taudb_tables[i]);
   }
}

int taudb_api_test(char* table_name) {
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
  printf("Selecting from %s...\n", table_name);
  char my_query[256] = "DECLARE myportal CURSOR FOR select * from ";
  strcat(my_query, table_name);
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
  res = PQexec(_taudb_connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(_taudb_connection, "END");
  PQclear(res);
  return 0;
}