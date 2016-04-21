#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include "dump_functions.h"

int main (int argc, char** argv) {
  TAUDB_CONNECTION* connection = NULL;
  int items=0;  int* taudb_numItems=&items;
  if (argc >= 3) {
    connection = taudb_connect_config(argv[1],taudb_numItems);
  } else {
    fprintf(stderr, "Please specify a TAUdb config file and a directory of profiles.\n");
    exit(1);
  }
  printf("Checking connection...\n");
  taudb_check_connection(connection);

  printf("Parsing file...\n");
  TAUDB_TRIAL* trial = taudb_parse_tau_profiles(argv[2]);
  trial->name = taudb_strdup("TEST TRIAL");
  trial->data_source = taudb_get_data_source_by_id(taudb_query_data_sources(connection,taudb_numItems), 1);

  boolean update = FALSE;
  boolean cascade = TRUE;
  printf("Testing inserts...\n");
  taudb_save_trial(connection, trial, update, cascade);
  
  printf("Disconnecting...\n");
  taudb_disconnect(connection);
  printf("Done.\n");
  return 0;
}
