#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>

PGconn* _taudb_connection;
int taudb_numItems = 0;
enum taudb_database_schema_version taudb_version = TAUDB_2005_SCHEMA;

void taudb_exit_nicely(void) {
  PQfinish(_taudb_connection);
  exit (1);
}

int taudb_connect(char* host, char* port, char* database, char* login, char* password) {
  char* pgoptions = NULL;
  char* pgtty = NULL;
  _taudb_connection = PQsetdbLogin(host, port, pgoptions, pgtty, database, login, password);
  /* Check to see that the backend connection was successfully made */
  if (PQstatus(_taudb_connection) != CONNECTION_OK)
  {
    fprintf(stderr, "Connection to database failed: %s",
           PQerrorMessage(_taudb_connection));
    taudb_exit_nicely();
  }

  return 0;
}

int taudb_checkConnection(void) {
  char* feedback;
  int status = PQstatus(_taudb_connection);
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
  }

  PQerrorMessage(_taudb_connection);
  printf("%d : %s\n", status, feedback);
  return 0;
}

int taudb_disconnect(void) {
  PQfinish(_taudb_connection);
  return 0;
}