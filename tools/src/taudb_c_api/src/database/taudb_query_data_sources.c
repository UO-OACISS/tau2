#include "taudb_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void taudb_add_data_source_to_connection(TAUDB_CONNECTION* connection, TAUDB_DATA_SOURCE* data_source) {
  if (data_source->id > 0) {
    HASH_ADD(hh1, connection->data_sources_by_id, id, sizeof(int), data_source);
  }
  HASH_ADD_KEYPTR(hh2, connection->data_sources_by_name, data_source->name, strlen(data_source->name), data_source);
}

TAUDB_DATA_SOURCE* taudb_query_data_sources(TAUDB_CONNECTION* connection, int* taudb_numItems) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_data_sources()\n");
#endif
  int nFields;
  int i, j;

  //if the connection already has the data, return it.
  if (connection->data_sources_by_id != NULL) {
    *taudb_numItems = HASH_CNT(hh1, connection->data_sources_by_id);
    return connection->data_sources_by_id;
  }

  taudb_begin_transaction(connection);
  char my_query[256];
  snprintf(my_query, sizeof(my_query), "select * from data_source");
#ifdef TAUDB_DEBUG
  printf("Query: %s\n", my_query);
#endif
  taudb_execute_query(connection, my_query);

  int nRows = taudb_get_num_rows(connection);
  *taudb_numItems = nRows;

  nFields = taudb_get_num_columns(connection);

  /* the rows */
  for (i = 0; i < nRows; i++)
  {
    TAUDB_DATA_SOURCE* data_source = taudb_create_data_sources(1);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(taudb_get_column_name(connection, j), "id") == 0) {
        data_source->id = atoi(taudb_get_value(connection, i, j));
      } else if (strcmp(taudb_get_column_name(connection, j), "name") == 0) {
        data_source->name = taudb_strdup(taudb_get_value(connection,i,j));
      } else if (strcmp(taudb_get_column_name(connection, j), "description") == 0) {
        data_source->description = taudb_strdup(taudb_get_value(connection, i, j));
      } else {
        printf("Error: unknown column '%s'\n", taudb_get_column_name(connection, j));
        taudb_exit_nicely(connection);
      }
    } 
    taudb_add_data_source_to_connection(connection, data_source);
  }

  taudb_clear_result(connection);
  taudb_close_transaction(connection);

  return (connection->data_sources_by_id);
}

TAUDB_DATA_SOURCE* taudb_get_data_source_by_id(TAUDB_DATA_SOURCE* data_sources, const int id) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_data_source(%p,%d)\n", data_sources, id);
#endif
  if (data_sources == NULL) {
    fprintf(stderr, "Error: data_source parameter null. Please provide a valid set of data_sources.\n");
    return NULL;
  }
  if (id == 0) {
    fprintf(stderr, "Error: id parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_DATA_SOURCE* data_source = NULL;
  HASH_FIND(hh1, data_sources, &(id), sizeof(int), data_source);
  return data_source;
}

TAUDB_DATA_SOURCE* taudb_get_data_source_by_name(TAUDB_DATA_SOURCE* data_sources, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_data_source(%p,%s)\n", data_sources, name);
#endif
  if (data_sources == NULL) {
    fprintf(stderr, "Error: data_source parameter null. Please provide a valid set of data_sources.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_DATA_SOURCE* data_source = NULL;
  HASH_FIND(hh2, data_sources, name, strlen(name), data_source);
  return data_source;
}

extern void taudb_save_data_sources(TAUDB_CONNECTION* connection, TAUDB_TRIAL* trial, boolean update) {
  printf("Data sources not supported yet.\n");
}

TAUDB_DATA_SOURCE* taudb_next_data_source_by_name_from_connection(TAUDB_DATA_SOURCE* current) {
  return current->hh2.next;
}

TAUDB_DATA_SOURCE* taudb_next_data_source_by_id_from_connection(TAUDB_DATA_SOURCE* current) {
  return current->hh2.next;
}

