#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include "zlib.h"

//int taudb_numItems = 0;
enum taudb_database_schema_version taudb_version = TAUDB_2005_SCHEMA;

void taudb_exit_nicely(TAUDB_CONNECTION* connection) {
#if defined __TAUDB_POSTGRESQL__
  PQfinish(connection->connection);
#elif defined __TAUDB_SQLITE__
  sqlite3_close_v2(connection->connection);
#endif
  exit (1);
}

TAUDB_CONNECTION* taudb_try_connect(char* host, char* port, char* database, char* login, char* password, taudb_error * err, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_connect()\n");
#endif
  TAUDB_CONNECTION* taudb_connection = taudb_create_connection();
  taudb_connection->inTransaction = FALSE;
  taudb_connection->inPortal = FALSE;
#if defined __TAUDB_POSTGRESQL__
  char* pgoptions = NULL;
  char* pgtty = NULL;
  taudb_connection->res = NULL;
  PGconn* connection;
  connection = PQsetdbLogin(host, port, pgoptions, pgtty, database, login, password);
  printf("Connecting to host: %s, port: %s, db: %s, login: %s\n", host, port, database, login);
  /* Check to see that the backend connection was successfully made */
  if (PQstatus(connection) != CONNECTION_OK)
  {
		*err = TAUDB_CONNECTION_FAILED;
    fprintf(stderr, "Connection to database failed: %s\n",
    PQerrorMessage(connection));
	} else {
		*err = TAUDB_OK;
	}
#elif defined __TAUDB_SQLITE__
  sqlite3 *connection;
  // get HOME
  
  int rc = sqlite3_open_v2(database, &connection);
#endif
  taudb_connection->connection = connection;

  /* what version of the schema do we have? */
  taudb_check_schema_version(taudb_connection);

  /* get the data sources, if available */
  if (taudb_connection->schema_version == TAUDB_2012_SCHEMA) {
    taudb_query_data_sources(taudb_connection, taudb_numItems);
  }

  return taudb_connection;
}

TAUDB_CONNECTION* taudb_connect(char* host, char* port, char* database, char* login, char* password, int* taudb_numItems) {
	taudb_error err;
	TAUDB_CONNECTION * taudb_connection = taudb_try_connect(host, port, database, login, password, &err, taudb_numItems);
	if(err != TAUDB_OK) {
    taudb_exit_nicely(taudb_connection);
	}
	return taudb_connection;
}


int taudb_check_connection(TAUDB_CONNECTION* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_check_connection()\n");
#endif
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
#elif defined __TAUDB_SQLITE__
  //?
#endif
  return 0;
}

int taudb_disconnect(TAUDB_CONNECTION* connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_disconnect()\n");
#endif
#if defined __TAUDB_POSTGRESQL__
  PQfinish(connection->connection);
#elif defined __TAUDB_SQLITE__
  sqlite3_close_v2(connection->connection);
#endif
  return 0;
}

TAUDB_CONNECTION* taudb_try_connect_config(char* config_name, taudb_error* err, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_connect_config()\n");
#endif
   const char* config_prefix = "perfdmf.cfg";
   const char* home = getenv("HOME");
   char config_file[256];
   snprintf(config_file, sizeof(config_file),  "%s/.ParaProf/%s.%s", home, config_prefix, config_name);
   return taudb_try_connect_config_file(config_file, err, taudb_numItems);
}

TAUDB_CONNECTION* taudb_try_connect_config_file(char* config_file_name, taudb_error *err, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_connect_config_file()\n");
#endif
  TAUDB_CONFIGURATION* config = taudb_parse_config_file(config_file_name);
  TAUDB_CONNECTION* connection = taudb_try_connect(config->db_hostname, config->db_portnum, config->db_dbname, config->db_username, config->db_password, err, taudb_numItems);
  connection->configuration = config;
  return connection;
}

TAUDB_CONNECTION* taudb_connect_config(char* config_name, int* taudb_numItems) {
	taudb_error err;
	return taudb_try_connect_config(config_name, &err, taudb_numItems);
}

TAUDB_CONNECTION * taudb_connect_config_file(char* config_file_name, int* taudb_numItems) {
	taudb_error err;
	return taudb_try_connect_config_file(config_file_name, &err, taudb_numItems);
}

void taudb_begin_transaction(TAUDB_CONNECTION *connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_begin_transaction()\n");
#endif
  if (connection->inTransaction) {
    printf("already in transaction!\n");
  }
#ifdef TAUDB_DEBUG_DEBUG
  printf("QUERY: '%s'\n", "BEGIN");
#endif

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
  connection->inTransaction = TRUE;
#endif
}

void taudb_execute_query(TAUDB_CONNECTION *connection, char* my_query) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_execute_query()\n");
#endif
  const char* portal_string = "DECLARE myportal CURSOR FOR";

#ifdef __TAUDB_POSTGRESQL__
  const int query_size = sizeof(char) * (strlen(my_query) + strlen(portal_string) + 2);
  char* full_query = (char*)malloc(query_size);
  snprintf(full_query, query_size,  "%s %s", portal_string, my_query);
#ifdef TAUDB_DEBUG_DEBUG
  printf("QUERY: '%s'\n", full_query);
#endif
  connection->res = PQexec(connection->connection, full_query);
  if (PQresultStatus(connection->res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "DECLARE CURSOR failed: %s", PQerrorMessage(connection->connection));
    PQclear(connection->res);
    taudb_exit_nicely(connection);
  }
  PQclear(connection->res);
  connection->inPortal = TRUE;

#ifdef TAUDB_DEBUG_DEBUG
  printf("QUERY: '%s'\n", "FETCH ALL in myportal");
#endif
  connection->res = PQexec(connection->connection, "FETCH ALL in myportal");
  if (PQresultStatus(connection->res) != PGRES_TUPLES_OK)
  {
    fprintf(stderr, "FETCH ALL failed: %s", PQerrorMessage(connection->connection));
    PQclear(connection->res);
    taudb_exit_nicely(connection);
  }
#elif defined __TAUDB_SQLITE__
  int rc = sqlite3_prepare_v2(connection->connection, my_query, strlen(my_query), &(connection->ppStmt), NULL);
  if( rc!=SQLITE_OK ){
    taudb_exit_nicely(connection);
  } 
#endif
  return;
}

int taudb_get_num_columns(TAUDB_CONNECTION *connection) {
  int columns = 0;
#ifdef __TAUDB_POSTGRESQL__
  columns = PQnfields(connection->res);
#endif
  return (columns);
}

int taudb_get_num_rows(TAUDB_CONNECTION *connection) {
  int rows = 0;
#ifdef __TAUDB_POSTGRESQL__
  rows = PQntuples(connection->res);
#endif
  return (rows);
}

char* taudb_get_column_name(TAUDB_CONNECTION *connection, int column) {
  char* name;
#ifdef __TAUDB_POSTGRESQL__
  name = PQfname(connection->res, column);
#endif
  return (name);
}

char* taudb_get_value(TAUDB_CONNECTION *connection, int row, int column) {
  char* value;
#ifdef __TAUDB_POSTGRESQL__
  value = PQgetvalue(connection->res, row, column);
#endif
  return (value);
}

void taudb_clear_result(TAUDB_CONNECTION *connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_clear_result()\n");
#endif
#ifdef __TAUDB_POSTGRESQL__
  if (connection->res != NULL) {
    PQclear(connection->res);
    connection->res = NULL;
  }
#endif
  return;
}

void taudb_close_transaction(TAUDB_CONNECTION *connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_close_transaction()\n");
#endif
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res;
  /* close the portal ... we don't bother to check for errors ... */
#ifdef TAUDB_DEBUG_DEBUG
  printf("QUERY: '%s'\n", "CLOSE myportal");
#endif
  if (connection->inPortal) {
    res = PQexec(connection->connection, "CLOSE myportal");
    PQclear(res);
    connection->inPortal = FALSE;
  }

  /* end the transaction */
#ifdef TAUDB_DEBUG_DEBUG
  printf("QUERY: '%s'\n", "END");
#endif
  res = PQexec(connection->connection, "END");
  PQclear(res);
  connection->inTransaction = FALSE;
#endif
}

void taudb_close_query(TAUDB_CONNECTION *connection) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_close_query()\n");
#endif
#ifdef __TAUDB_POSTGRESQL__
  PGresult* res;
  /* close the portal ... we don't bother to check for errors ... */
#ifdef TAUDB_DEBUG_DEBUG
  printf("QUERY: '%s'\n", "CLOSE myportal");
#endif
  if (connection->inPortal) {
    res = PQexec(connection->connection, "CLOSE myportal");
    PQclear(res);
    connection->inPortal = FALSE;
  }
#endif
}

boolean taudb_gzip_inflate( unsigned char* compressedBytes, int length, char** uncompressedBytes ) {  
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_gzip_inflate()\n");
#endif
  
  if ( length == 0 ) {
	/* return null pointer if there's nothing to decompress */
    *uncompressedBytes = 0;  
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
  
  char * result = (char*)malloc(strm.total_out * sizeof(char));
  memcpy(result, uncomp, strm.total_out * sizeof(char));
  *uncompressedBytes = result; 
  
  free( uncomp );  
  return TRUE ;  
}  

char* taudb_get_binary_value(TAUDB_CONNECTION *connection, int row, int column) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_get_binary_value()\n");
#endif
  char* value, * retVal;
#ifdef __TAUDB_POSTGRESQL__
  value = PQgetvalue(connection->res, row, column);
/* The binary representation of BYTEA is a bunch of bytes, which could
 * include embedded nulls so we have to pay attention to field length.
 */
#ifdef TAUDB_DEBUG
  int blen = PQgetlength(connection->res, row, column);
  printf("tuple %d: got\n", row);
  printf(" XML_METADATA_GZ = (%d bytes) ", blen);
#endif
 /*
  * It turns out that Postgres doesn't return raw bytes; it returns a
  * string consisting of '\x' followed by the characters of the hex
  * representation of the bytes, so we need to convert back to bytes.
  */	 
  size_t length = 0;
  unsigned char * unescaped = PQunescapeBytea((const unsigned char *)value, &length);
  char* expanded = NULL;
  taudb_gzip_inflate(unescaped, length, &expanded);
  PQfreemem(expanded);
#endif

#ifdef TAUDB_DEBUG_DEBUG
  //printf("%s\n\n", expanded);
#endif
  retVal = strdup(expanded);
  return (retVal);
}

void taudb_prepare_statement(TAUDB_CONNECTION* connection, const char* statement_name, const char* statement, int nParams) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_prepare_statement('%s')\n", statement);
#endif

#ifdef __TAUDB_POSTGRESQL__
  // have we prepared this statement already?
  TAUDB_PREPARED_STATEMENT* prepared_statement = NULL;
  HASH_FIND(hh, connection->statements, statement_name, strlen(statement_name), prepared_statement);
  if (prepared_statement == NULL) {
    PGresult* res;
    res = PQprepare(connection->connection, statement_name, statement, nParams, NULL);
    if (PQresultStatus(res) != PGRES_COMMAND_OK)
    {
      fprintf(stderr, "Preparing statement failed: %s", PQerrorMessage(connection->connection));
      PQclear(res);
      taudb_exit_nicely(connection);
    } else {
      PQclear(res);
	  prepared_statement = (TAUDB_PREPARED_STATEMENT*)malloc(sizeof(TAUDB_PREPARED_STATEMENT));
	  prepared_statement->name = taudb_strdup(statement_name);
      HASH_ADD_KEYPTR(hh, connection->statements, statement_name, strlen(statement_name), prepared_statement);
    }
  }
#endif
}

int taudb_execute_statement(TAUDB_CONNECTION* connection, const char* statement_name, int nParams, const char ** paramValues) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("calling taudb_execute_statement()\n");
	printf("Will execute statement %s with %d parameters.\n", statement_name, nParams);
	int i;
	for(i = 0; i < nParams; ++i) {
		printf("Param %d = %s\n", i+1, paramValues[i]);
	}
#endif

#ifdef __TAUDB_POSTGRESQL__
  PGresult* res;
  res = PQexecPrepared(connection->connection, statement_name, nParams, paramValues, NULL, NULL, 0);
	if (PQresultStatus(res) != PGRES_COMMAND_OK)
  {
    fprintf(stderr, "Executing statement %s failed: %s", statement_name, PQerrorMessage(connection->connection));
    PQclear(res);
    taudb_exit_nicely(connection);
  }
  char * rows_changed_str;
	rows_changed_str = PQcmdTuples(res); /* PQclear frees this */
	int rows_changed = atoi(rows_changed_str);
	return rows_changed;
#endif
	return 0;
}

const char * taudb_error_str(taudb_error err) {
	switch(err) {
		case TAUDB_OK: return "";
		case TAUDB_CONNECTION_FAILED: return "Connection failed.";
		default: return "Invalid error.";
	}
}
