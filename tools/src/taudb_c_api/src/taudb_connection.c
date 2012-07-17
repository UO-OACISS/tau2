#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>

int taudb_numItems = 0;
enum taudb_database_schema_version taudb_version = TAUDB_2005_SCHEMA;

void taudb_exit_nicely(PGconn* connection) {
  PQfinish(connection);
  exit (1);
}

PGconn* taudb_connect(char* host, char* port, char* database, char* login, char* password) {
  char* pgoptions = NULL;
  char* pgtty = NULL;
  PGconn* connection;
  connection = PQsetdbLogin(host, port, pgoptions, pgtty, database, login, password);
  printf("Connecting to host: %s, port: %s, db: %s, login: %s", host, port, database, login);
  /* Check to see that the backend connection was successfully made */
  if (PQstatus(connection) != CONNECTION_OK)
  {
    fprintf(stderr, "Connection to database failed: %s",
           PQerrorMessage(connection));
    taudb_exit_nicely(connection);
  }

  // what version of the schema do we have?
  taudb_check_schema_version(connection);

  return connection;
}

int taudb_check_connection(PGconn* connection) {
  char* feedback;
  int status = PQstatus(connection);
  switch(status)
  {
    case CONNECTION_OK:
        feedback = "Connection OK";
        break;
    case CONNECTION_BAD:
        feedback = "Connection bad";
        break;
    case CONNECTION_STARTED:
        feedback = "Connection started";
        break;
    case CONNECTION_MADE:
        feedback = "Connected to server";
        break;
    case CONNECTION_AWAITING_RESPONSE:
        feedback = "Waiting for a response from the server.";
        break;
    case CONNECTION_AUTH_OK:
        feedback = "Received authentication; waiting for backend start-up to finish.";
        break;
    case CONNECTION_SSL_STARTUP:
        feedback = "Negotiating SSL encryption.";
        break;
    case CONNECTION_SETENV:
        feedback = "Negotiating environment-driven parameter settings.";
        break;
    case CONNECTION_NEEDED:
        feedback = "Internal status - connect() needed.";
        break;
    default:
        feedback = "Connecting...";
        break;
  }

  PQerrorMessage(connection);
  printf("%d : %s\n", status, feedback);
  return 0;
}

int taudb_disconnect(PGconn* connection) {
  PQfinish(connection);
  return 0;
}

PGconn* taudb_connect_config(char* config_name) {
   const char* config_prefix = "perfdmf.cfg";
   const char* home = getenv("HOME");
   char config_file[256];
   sprintf(config_file, "%s/.ParaProf/%s.%s", home, config_prefix, config_name);
   return taudb_connect_config_file(config_file);
}

PGconn* taudb_connect_config_file(char* config_file_name) {
  TAUDB_CONFIGURATION* config = taudb_parse_config_file(config_file_name);
  return taudb_connect(config->db_hostname, config->db_portnum, config->db_dbname, config->db_username, config->db_password);
}

