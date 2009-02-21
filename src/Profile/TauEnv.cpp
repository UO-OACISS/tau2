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
#ifndef TAU_WINDOWS
#include <strings.h>
#else
#define strcasecmp(X,Y)  stricmp(X,Y)
#define unsetenv(X)
#endif
#include <ctype.h>
#include <stdio.h>
#include <stdarg.h>

#include <Profile/TauEnv.h>

/* We should throttle if number n > a && percall < b .a and b are given below */
#define TAU_THROTTLE_NUMCALLS_DEFAULT 100000
#define TAU_THROTTLE_PERCALL_DEFAULT  10
#define TAU_CALLPATH_DEPTH_DEFAULT  2

/* If TAU is built with -PROFILECALLPATH, we turn callpath profiling on by default */
#ifdef TAU_CALLPATH
# define TAU_CALLPATH_DEFAULT 1
#else
# define TAU_CALLPATH_DEFAULT 0
#endif

#ifdef TRACING_ON
# define TAU_TRACING_DEFAULT 1
#else
# define TAU_TRACING_DEFAULT 0
#endif

#ifdef PROFILING_ON
# define TAU_PROFILING_DEFAULT 1
#else
# define TAU_PROFILING_DEFAULT 0
#endif

#define TAU_THROTTLE_DEFAULT 1
#ifdef TAU_MPI
  #define TAU_SYNCHRONIZE_CLOCKS_DEFAULT 1
#else
  #define TAU_SYNCHRONIZE_CLOCKS_DEFAULT 0
#endif /* TAU_MPI */

extern "C" {

  static int env_synchronize_clocks = 0;
  static int env_verbose = 0;
  static int env_throttle = 0;
  static int env_callpath = 0;
  static int env_profiling = 0;
  static int env_tracing = 0;
  static int env_callpath_depth = 0;
  static int env_profile_format = TAU_FORMAT_PROFILE;
  static double env_throttle_numcalls = 0;
  static double env_throttle_percall = 0;
  static const char *env_profiledir = NULL;
  static const char *env_tracedir = NULL;

  double TauEnv_get_throttle_numcalls();
  double TauEnv_get_throttle_percall();


  void TAU_VERBOSE(const char *format, ...) {
    va_list args;
    if (env_verbose != 1) {
      return;
    }
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
  }
 
  static int parse_bool(char *str, int default_value = 0) {
    if (str == NULL) {
      return default_value;
    }
    static char strbuf[128];
    char* ptr = strbuf;
    strncpy(strbuf, str, 128);
    while (*ptr) {
      *ptr = tolower(*ptr);
      ptr++;
    }
    if (strcmp(strbuf, "yes") == 0  || 
	strcmp(strbuf, "true") == 0 || 
	strcmp(strbuf, "1") == 0) {
      return 1;
    } else {
      return 0;
    }
  }


  const char *TauEnv_get_profiledir() {
    return env_profiledir;
  }

  const char *TauEnv_get_tracedir() {
    return env_tracedir;
  }

  int TauEnv_get_synchronize_clocks() {
    return env_synchronize_clocks;
  }

  int TauEnv_get_verbose() {
    return env_verbose;
  }

  int TauEnv_get_throttle() {
    return env_throttle;
  }

  int TauEnv_get_callpath() {
    return env_callpath;
  }

  int TauEnv_get_profiling() {
    return env_profiling;
  }

  int TauEnv_get_tracing() {
    return env_tracing;
  }

  int TauEnv_get_callpath_depth() {
    return env_callpath_depth;
  }

  double TauEnv_get_throttle_numcalls() {
    return env_throttle_numcalls;
  }

  double TauEnv_get_throttle_percall() {
    return env_throttle_percall;
  }


  int TauEnv_get_profile_format() {
    return env_profile_format;
  }

  void TauEnv_initialize() {

    // unset LD_PRELOAD so that vt_unify and elg_unify work
    unsetenv("LD_PRELOAD");

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
      if (parse_bool(tmp, TAU_SYNCHRONIZE_CLOCKS_DEFAULT)) {
	env_synchronize_clocks = 1;
      } else {
	env_synchronize_clocks = 0;
      }

#ifndef TAU_MPI
      /* If there is no MPI, there can't be any sync, so forget it */
      env_synchronize_clocks = 0;
      TAU_VERBOSE("TAU: Clock Synchronization Disabled (MPI not available)\n");
#else
      if (env_synchronize_clocks) {
	TAU_VERBOSE("TAU: Clock Synchronization Enabled\n");
      } else {
	TAU_VERBOSE("TAU: Clock Synchronization Disabled\n");
      }
#endif

      if ((env_profiledir = getenv("PROFILEDIR")) == NULL) {
	env_profiledir = "."; // current directory
      }
      TAU_VERBOSE("TAU: PROFILEDIR is \"%s\"\n", env_profiledir);

      if ((env_tracedir = getenv("TRACEDIR")) == NULL) {
	env_tracedir = "."; // current directory
      }
      TAU_VERBOSE("TAU: TRACEDIR is \"%s\"\n", env_tracedir);

      // callpath
      tmp = getenv("TAU_CALLPATH");
      if (parse_bool(tmp, TAU_CALLPATH_DEFAULT)) {
	env_callpath = 1;
	TAU_VERBOSE("TAU: Callpath Profiling Enabled\n");
      } else {
	env_callpath = 0;
	TAU_VERBOSE("TAU: Callpath Profiling Disabled\n");
      }

      // profiling
      tmp = getenv("TAU_PROFILING");
      if (parse_bool(tmp, TAU_PROFILING_DEFAULT)) {
	env_profiling = 1;
	TAU_VERBOSE("TAU: Profiling Enabled\n");
      } else {
	env_profiling = 0;
	TAU_VERBOSE("TAU: Profiling Disabled\n");
      }

      // tracing
      tmp = getenv("TAU_TRACING");
      if (parse_bool(tmp, TAU_TRACING_DEFAULT)) {
	env_tracing = 1;
	TAU_VERBOSE("TAU: Tracing Enabled\n");
      } else {
	env_tracing = 0;
	TAU_VERBOSE("TAU: Tracing Disabled\n");
      }

      // callpath depth
      char *depth = getenv("TAU_CALLPATH_DEPTH"); 
      env_callpath_depth = TAU_CALLPATH_DEPTH_DEFAULT;
      if (depth) {
	env_callpath_depth = atoi(depth);
	if (env_callpath_depth <= 1) {
	  env_callpath_depth = TAU_CALLPATH_DEPTH_DEFAULT;
	}
      }
      if (env_callpath) {
	TAU_VERBOSE("TAU: Callpath Depth = %d\n", env_callpath_depth);
      }


      // Throttle
      tmp = getenv("TAU_THROTTLE");
      if (parse_bool(tmp, TAU_THROTTLE_DEFAULT)) {
	env_throttle = 1;
	TAU_VERBOSE("TAU: Throttling Enabled\n");
      } else {
	env_throttle = 0;
	TAU_VERBOSE("TAU: Throttling Disabled\n");
      }

      char *percall = getenv("TAU_THROTTLE_PERCALL"); 
      env_throttle_percall = TAU_THROTTLE_PERCALL_DEFAULT;
      if (percall) {
	env_throttle_percall = strtod(percall,0); 
      }

      char *numcalls = getenv("TAU_THROTTLE_NUMCALLS"); 
      env_throttle_numcalls = TAU_THROTTLE_NUMCALLS_DEFAULT;
      if (numcalls) {
	env_throttle_numcalls = strtod(numcalls,0); 
      }

      if (env_throttle) {
	TAU_VERBOSE("TAU: Throttle PerCall = %g\n", env_throttle_percall);
	TAU_VERBOSE("TAU: Throttle NumCalls = %g\n", env_throttle_numcalls);
      }

      char *profileFormat = getenv("TAU_PROFILE_FORMAT");
      if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "snapshot")) {
	env_profile_format = TAU_FORMAT_SNAPSHOT;
      } else if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "merged")) {
	env_profile_format = TAU_FORMAT_MERGED;
      } else {
	env_profile_format = TAU_FORMAT_PROFILE;
      }
    }
  }
}
