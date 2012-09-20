#include "taudb_internal.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include "zlib.h"

int taudb_numItems = 0;
enum taudb_database_schema_version taudb_version = TAUDB_2005_SCHEMA;

void taudb_exit_nicely(TAUDB_CONNECTION* connection) {
#ifdef __TAUDB_POSTGRESQL__
  PQfinish(connection->connection);
#endif
  exit (1);
}

TAUDB_CONNECTION* taudb_connect(char* host, char* port, char* database, char* login, char* password) {
  TAUDB_CONNECTION* taudb_connection = (TAUDB_CONNECTION*)malloc(sizeof (TAUDB_CONNECTION));
#ifdef __TAUDB_POSTGRESQL__
  char* pgoptions = NULL;
  char* pgtty = NULL;
  PGconn* connection;
  connection = PQsetdbLogin(host, port, pgoptions, pgtty, database, login, password);
  printf("Connecting to host: %s, port: %s, db: %s, login: %s\n", host, port, database, login);
  /* Check to see that the backend connection was successfully made */
  if (PQstatus(connection) != CONNECTION_OK)
  {
    fprintf(stderr, "Connection to database failed: %s\n",
           PQerrorMessage(connection));
    taudb_exit_nicely(taudb_connection);
  }
  taudb_connection->connection = connection;
#endif

  /* what version of the schema do we have? */
  taudb_check_schema_version(taudb_connection);

  /* get the data sources, if available */
  if (taudb_connection->schema_version == TAUDB_2012_SCHEMA) {
    taudb_query_data_sources(taudb_connection);
  }

  return taudb_connection;
}

int taudb_check_connection(TAUDB_CONNECTION* connection) {
#ifdef __TAUDB_POSTGRESQL__
  char* feedback;
  int status = PQstatus(connection->connection);
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

  PQerrorMessage(connection->connection);
  printf("%d : %s\n", status, feedback);
#endif
  return 0;
}

int taudb_disconnect(TAUDB_CONNECTION* connection) {
#ifdef __TAUDB_POSTGRESQL__
  PQfinish(connection->connection);
#endif
  return 0;
}

TAUDB_CONNECTION* taudb_connect_config(char* config_name) {
   const char* config_prefix = "perfdmf.cfg";
   const char* home = getenv("HOME");
   char config_file[256];
   sprintf(config_file, "%s/.ParaProf/%s.%s", home, config_prefix, config_name);
   return taudb_connect_config_file(config_file);
}

TAUDB_CONNECTION* taudb_connect_config_file(char* config_file_name) {
  TAUDB_CONFIGURATION* config = taudb_parse_config_file(config_file_name);
  TAUDB_CONNECTION* connection = taudb_connect(config->db_hostname, config->db_portnum, config->db_dbname, config->db_username, config->db_password);
  connection->configuration = config;
  return connection;
}

void taudb_begin_transaction(TAUDB_CONNECTION *connection) {
#ifdef __TAUDB_POSTGRESQL__
  PGresult   *res;
  res = PQexec(connection->connection, "BEGIN");
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "BEGIN command failed: %s", PQerrorMessage(connection->connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }
  /*
   * Should PQclear PGresult whenever it is no longer needed to avoid
   * memory leaks
   */
  PQclear(res);
#endif
}

void* taudb_execute_query(TAUDB_CONNECTION *connection, char* my_query) {
  void* result;
  const char* portal_string = "DECLARE myportal CURSOR FOR";
#ifdef __TAUDB_POSTGRESQL__
  char* full_query = malloc(sizeof(char) * (strlen(my_query) + strlen(portal_string) + 2));
  sprintf(full_query, "%s %s", portal_string, my_query);
  PGresult   *res;
  res = PQexec(connection->connection, full_query);
  if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "DECLARE CURSOR failed: %s", PQerrorMessage(connection->connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }
  PQclear(res);

  res = PQexec(connection->connection, "FETCH ALL in myportal");
  if (PQresultStatus(res) != PGRES_TUPLES_OK)
  {
    fprintf(stderr, "FETCH ALL failed: %s", PQerrorMessage(connection->connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }

  /* return the result to be used by the calling code */
  result = ((void*)res);
#endif
  return (result);
}

int taudb_get_num_columns(void* result) {
  int columns = 0;
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res = (PGresult*)result;
  columns = PQnfields(res);
#endif
  return (columns);
}

int taudb_get_num_rows(void* result) {
  int rows = 0;
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res = (PGresult*)result;
  rows = PQntuples(res);
#endif
  return (rows);
}

char* taudb_get_column_name(void* result, int column) {
  char* name;
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res = (PGresult*)result;
  name = PQfname(res, column);
#endif
  return (name);
}

char* taudb_get_value(void* result, int row, int column) {
  char* value;
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res = (PGresult*)result;
  value = PQgetvalue(res, row, column);
#endif
  return (value);
}

void taudb_clear_result(void* result) {
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res = (PGresult*)result;
  PQclear(res);
#endif
  return;
}

void taudb_close_transaction(TAUDB_CONNECTION *connection) {
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res;
  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection->connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection->connection, "END");
  PQclear(res);
#endif
}

boolean gzipInflate( char* compressedBytes, int length, char** uncompressedBytes ) {  
  
  if ( length == 0 ) {  
    uncompressedBytes[0] = '\0';  
    return TRUE ;  
  }  
  
  unsigned int full_length = length ;  
  unsigned int half_length = full_length / 2;  
  
  unsigned int uncompLength = full_length ;  
  char* uncomp = (char*) calloc( sizeof(char), uncompLength );  
  
  z_stream strm;  
  strm.next_in = (Bytef *) compressedBytes;  
  strm.avail_in = length ;  
  strm.total_out = 0;  
  strm.zalloc = Z_NULL;  
  strm.zfree = Z_NULL;  
  
  boolean done = FALSE ;  
  
  if (inflateInit2(&strm, (32+MAX_WBITS)) != Z_OK) {  
    free( uncomp );
	fprintf(stderr, "Unable to inflateInit2\n");  
    return FALSE;  
  }  
  
  while (!done) {  
    // If our output buffer is too small  
    if (strm.total_out >= uncompLength ) {  
      // Increase size of output buffer  
      char* uncomp2 = (char*) calloc( sizeof(char), uncompLength + half_length );  
      memcpy( uncomp2, uncomp, uncompLength );  
      uncompLength += half_length ;  
      free( uncomp );  
      uncomp = uncomp2 ;  
    }  
  
    strm.next_out = (Bytef *) (uncomp + strm.total_out);  
    strm.avail_out = uncompLength - strm.total_out;  
  
    // Inflate another chunk.  
    int err = inflate (&strm, Z_SYNC_FLUSH);  
    if (err == Z_STREAM_END) done = TRUE;  
    else if (err != Z_OK)  {  
	  printf("%s\n", strm.msg);
      break;  
    }  
  }  
  
  if (inflateEnd (&strm) != Z_OK) {  
    free( uncomp );  
    return FALSE;  
  }  
  
  char * result = malloc(strm.total_out * sizeof(char));
  memcpy(result, uncomp, strm.total_out * sizeof(char));
  *uncompressedBytes = result; 
  
  free( uncomp );  
  return TRUE ;  
}  

char* taudb_get_binary_value(void* result, int row, int column) {
  char* value;
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res = (PGresult*)result;
  value = PQgetvalue(res, row, column);
/* The binary representation of BYTEA is a bunch of bytes, which could
 * include embedded nulls so we have to pay attention to field length.
 */
  int blen = PQgetlength(res, row, column);
  printf("tuple %d: got\n", row);
  printf(" XMAL_METADATA_GZ = (%d bytes) ", blen);
 /*
  * It turns out that Postgres doesn't return raw bytes; it returns a
  * string consisting of '\x' followed by the characters of the hex
  * representation of the bytes, so we need to convert back to bytes.
  */	 
  size_t length = 0;
  unsigned char * unescaped = PQunescapeBytea(value, &length);
  char* expanded = NULL;
  gzipInflate(unescaped, length, &expanded);
  printf("%s\n", expanded);
#endif
  return (value);
}


