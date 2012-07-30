#include "taudb_api.h"
#include "libpq-fe.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

TAUDB_TIMER_GROUP* taudb_query_groups(PGconn* connection, TAUDB_TRIAL* trial) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_query_group(%p)\n", trial);
#endif
  PGresult *res;
  int nFields;
  int i, j;

  if (trial == NULL) {
    fprintf(stderr, "Error: trial parameter null. Please provide a valid trial.\n");
    return NULL;
  }

  //if the Trial already has the data, return it.
  if (trial->timer_groups != NULL && trial->timer_group_count > 0) {
    taudb_numItems = trial->timer_group_count;
    return trial->timer_groups;
  }

  /* Start a transaction block */
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
  char my_query[256];
  if (taudb_version == TAUDB_2005_SCHEMA) {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select group_name from interval_event where trial = %d", trial->id);
	fprintf(stderr, "WARNING - NOT TESTED!\n");
  } else {
    sprintf(my_query,"DECLARE myportal CURSOR FOR select distinct tg.name as name from timer_group tg inner join timer t on tg.timer = t.id where t.trial = %d", trial->id);
  }
#ifdef TAUDB_DEBUG
  printf("%s\n", my_query);
#endif
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

  int nRows = PQntuples(res);
  TAUDB_TIMER_GROUP* timer_groups = taudb_create_timer_groups(nRows);
  taudb_numItems = nRows;

  nFields = PQnfields(res);

  /* the rows */
  for (i = 0; i < PQntuples(res); i++)
  {
    TAUDB_TIMER_GROUP* timer_group = &(timer_groups[i]);
    /* the columns */
    for (j = 0; j < nFields; j++) {
      if (strcmp(PQfname(res, j), "name") == 0) {
        timer_group->name = taudb_create_and_copy_string(PQgetvalue(res,i,j));
#ifdef TAUDB_DEBUG_DEBUG
        printf("Got group '%s'\n", timer_group->name);
#endif
      } else if (strcmp(PQfname(res, j), "group_name") == 0) {
	  /*
        // tokenize the string, something like 'TAU_USER|MPI|...'
        char* group_names = PQgetvalue(res, i, j);
        char* group = strtok(group_names, "|");
        if (group != NULL && (strlen(group_names) > 0)) {
#ifdef TAUDB_DEBUG
          //printf("Got group groups '%s'\n", group_names);
#endif
          groups[i].group_count = 1;
          TAUDB_TIMER_GROUP_TIMER_GROUP* groups = taudb_create_group_groups(1);
          groups[0].id = 0;
          groups[0].group = 0;
          groups[0].name = taudb_create_and_copy_string(group);
          group = strtok(NULL, "|");
          while (group != NULL) {
            TAUDB_TIMER_GROUP_TIMER_GROUP* groups = taudb_resize_group_groups(groups[i].group_count+1, groups);
            groups[groups[i].group_count].id = 0;
            groups[groups[i].group_count].group = 0;
            groups[groups[i].group_count].name = taudb_create_and_copy_string(group);
            groups[i].group_count++;
            group = strtok(NULL, "|");
          }
        } else {
          groups[i].group_count = 0;
          groups[i].groups = NULL;
        }
		  */
      } else {
        printf("Error: unknown column '%s'\n", PQfname(res, j));
        taudb_exit_nicely(connection);
      }
      // TODO - Populate the rest properly?
      timer_group->timer_count = 0;
      timer_group->timers = NULL;
      HASH_ADD_KEYPTR(hh, timer_groups, timer_group->name, strlen(timer_group->name), timer_group);
    } 
  }

  PQclear(res);

  /* close the portal ... we don't bother to check for errors ... */
  res = PQexec(connection, "CLOSE myportal");
  PQclear(res);

  /* end the transaction */
  res = PQexec(connection, "END");
  PQclear(res);
  
  return (timer_groups);
}

TAUDB_TIMER_GROUP* taudb_get_timer_group(TAUDB_TIMER_GROUP* timer_groups, const char* name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_get_timer_group(%p,%s)\n", timer_groups, name);
#endif
  if (timer_groups == NULL) {
    fprintf(stderr, "Error: timer_groups parameter null. Please provide a valid set of timer_groups.\n");
    return NULL;
  }
  if (name == NULL) {
    fprintf(stderr, "Error: name parameter null. Please provide a valid name.\n");
    return NULL;
  }

  TAUDB_TIMER_GROUP* timer_group = NULL;
  HASH_FIND_STR(timer_groups, name, timer_group);
  return timer_group;
}

