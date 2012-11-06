#include "taudb_internal.h"
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

int taudb_api_test(TAUDB_CONNECTION* connection, char* table_name) {
  int                     nFields;
  int                     i, j;

  taudb_begin_transaction(connection);

  /*
   * Fetch rows from table_name, the system catalog of databases
   */
  printf("Selecting from %s...\n", table_name);
  char my_query[256] = "select * from ";
  strcat(my_query, table_name);
  taudb_execute_query(connection, my_query);

  /* first, print out the attribute names */
  nFields = taudb_get_num_columns(connection);
  for (i = 0; i < nFields; i++) {
    printf("%-15s", taudb_get_column_name(connection, i));
  }
  printf("\n\n");

  /* next, print out the rows */
  for (i = 0; i < taudb_get_num_rows(connection); i++)
  {
    for (j = 0; j < nFields; j++)
            printf("%-15s", taudb_get_value(connection, i, j));
    printf("\n");
  }
  taudb_clear_result(connection);
  taudb_close_transaction(connection);
  return 0;
}

void taudb_iterate_tables(TAUDB_CONNECTION* connection) {
   int i;
   for (i = 0 ; i < num_tables ; i = i+1) {
     taudb_api_test(connection, taudb_tables[i]);
   }
}


