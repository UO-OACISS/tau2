#include "taudb_internal.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>

#define MAX_RECORD_LENGTH 256

void taudb_trim(char * s) {
    char * p = s;
    int l = strlen(p);

    while((l > 0) && isspace(p[l - 1])) p[--l] = 0;
    while(* p && isspace(* p)) ++p, --l;

    memmove(s, p, l + 1);
}

TAUDB_CONFIGURATION* taudb_parse_config_file(char* config_name) {
#ifdef TAUDB_DEBUG_DEBUG
  printf("Calling taudb_parse_config(%s)\n", config_name);
#endif

  // validate the config file name
  if (config_name == NULL || strlen(config_name) == 0) {
    fprintf(stderr, "ERROR: empty configuration file name.\n");
	return NULL;
  }

  // open the config file
  FILE* ifp = fopen (config_name, "r");
  if (ifp == NULL) {
    fprintf(stderr, "ERROR: could not parse configuration file %s\n", config_name);
	return NULL;
  }

  char name[32];
  char value[128];
  char line[MAX_RECORD_LENGTH];

  TAUDB_CONFIGURATION* config = taudb_create_configuration();

  // parse the config file, one line at a time
  while (!feof(ifp)) {
	fgets(line, MAX_RECORD_LENGTH, ifp);
	taudb_trim(line);
    if (strlen(line) == 0) {
	  continue;
	} else if (strncmp(line, "#", 1) == 0) {
	  continue;
	} else {
      char* tmp = strtok(line, ":");
      if (tmp != NULL && (strlen(tmp) > 0)) {
	    strncpy(name,  tmp, sizeof(name)); 
	  }
      tmp = strtok(NULL, ":");
      if (tmp != NULL && (strlen(tmp) > 0)) {
	    strncpy(value,  tmp, sizeof(value)); 
	  }
	  if (strcmp(name, "jdbc_db_type") == 0) {
	    config->jdbc_db_type = taudb_strdup(value);
	  } else if (strcmp(name, "db_hostname") == 0) {
	    config->db_hostname = taudb_strdup(value);
	  } else if (strcmp(name, "db_portnum") == 0) {
	    config->db_portnum = taudb_strdup(value);
	  } else if (strcmp(name, "db_dbname") == 0) {
	    config->db_dbname = taudb_strdup(value);
	  } else if (strcmp(name, "db_schemaprefix") == 0) {
	    config->db_schemaprefix = taudb_strdup(value);
	  } else if (strcmp(name, "db_username") == 0) {
	    config->db_username = taudb_strdup(value);
	  } else if (strcmp(name, "db_password") == 0) {
	    config->db_password = taudb_strdup(value);
	  } else if (strcmp(name, "db_schemafile") == 0) {
	    config->db_schemafile = taudb_strdup(value);
	  }
	}
  }

  return config;
}
