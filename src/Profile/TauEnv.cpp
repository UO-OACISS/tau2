/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2008  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauEnv.cpp 			        	   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Handle environment variables                     **
**                                                                         **
****************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

#include <Profile/TauEnv.h>

extern "C" {


  static int env_synchronize_clocks = 0;
  static int env_verbose = 0;


  static int parse_bool(char *str) {
    if (str == NULL) {
      return 0;
    }
    static char strbuf[128];
    char* ptr = strbuf;
    strncpy(strbuf, str, 128);
    while (*ptr) {
      *ptr = tolower(*ptr);
      ptr++;
    }
    if (strcmp(strbuf, "yes") == 0  || strcmp(strbuf, "true") == 0 || strcmp(strbuf, "1") == 0) {
      return 1;
    } else {
      return 0;
    }
  }

  int TauEnv_get_synchronize_clocks() {
    return env_synchronize_clocks;
  }

  int TauEnv_get_verbose() {
    return env_verbose;
  }


  void TauEnv_initialize() {
    static int initialized = 0;

    if (!initialized) {
      char *tmp;

      tmp = getenv("TAU_VERBOSE");
      if (parse_bool(tmp)) {
	env_verbose = 1;
      } else {
	env_verbose = 0;
      }

      TAU_VERBOSE("TAU: Initialized TAU (TAU_VERBOSE=1)\n");
      
      tmp = getenv("TAU_SYNCHRONIZE_CLOCKS");
      if (parse_bool(tmp)) {
	env_synchronize_clocks = 1;
	TAU_VERBOSE("TAU: Clock Synchronization Enabled\n");
      } else {
	env_synchronize_clocks = 0;
      }
      
    }
    
  }


}
