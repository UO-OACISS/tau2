#include "taudb_api.h"
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include "dump_functions.h"

int main (int argc, char** argv) {
   const char* config_prefix = "perfdmf.cfg";
   // get the home directory path
   const char* home = getenv("HOME");
   printf("Found $HOME as: %s\n", home);
   char config_dir[256];
   snprintf(config_dir, sizeof(config_dir),  "%s/.ParaProf", home);
   printf("Looking for config files in: %s\n", config_dir);
   // get the config files in the home directory
   DIR *dp = NULL;
   struct dirent *ep = NULL;
   char config_file[256];
   dp = opendir (config_dir);
   if (dp != NULL) {
     ep = readdir(dp);
     while (ep != NULL) {
	   if (strncmp(ep->d_name, config_prefix, 11) == 0) {
          snprintf(config_file, sizeof(config_file),  "%s/%s", config_dir, ep->d_name);
          printf("Parsing config file %s...\n", config_file);
          TAUDB_CONFIGURATION* config = taudb_parse_config_file(config_file);
		  printf ("Database: %s, %s, %s\n", config->db_hostname, config->db_portnum, config->db_dbname);
	   }
       ep = readdir(dp);
	 }
	 closedir(dp);
   } else {
     printf("No TAUdb config files found.\n");
   }

   printf("Done.\n");
   return 0;
}
