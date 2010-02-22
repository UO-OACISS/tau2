/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2008                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File            : TauEnv.cpp                                       **
**	Description     : TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
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
#define strcasecmp(X, Y)  stricmp(X, Y)
#define unsetenv(X)
#endif
#include <ctype.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>

#include <Profile/TauEnv.h>
#include <TAU.h>

#define MAX_LN_LEN 2048

/* We should throttle if number n > a && percall < b .a and b are given below */
#define TAU_THROTTLE_NUMCALLS_DEFAULT 100000
#define TAU_THROTTLE_PERCALL_DEFAULT  10
#define TAU_CALLPATH_DEPTH_DEFAULT  2

#define TAU_DEPTH_LIMIT_DEFAULT INT_MAX

/* If TAU is built with -PROFILECALLPATH, we turn callpath profiling on by default */
#ifdef TAU_CALLPATH
# define TAU_CALLPATH_DEFAULT 1
#else
# define TAU_CALLPATH_DEFAULT 0
#endif

/* if we are doing EBS sampling, set the default sampling frequency */
#define TAU_EBS_FREQUENCY_DEFAULT 1000
/* if we are doing EBS sampling, set whether we want inclusive samples */
/* that is, main->foo->mpi_XXX is a sample for main, foo and mpi_xxx */
#define TAU_EBS_INCLUSIVE_DEFAULT 0

#define TAU_EBS_SOURCE_DEFAULT "itimer"


#ifdef TAU_COMPENSATE
# define TAU_COMPENSATE_DEFAULT 1
#else
# define TAU_COMPENSATE_DEFAULT 0
#endif

#if (defined(MPI_TRACE) || defined(TRACING_ON))
# define TAU_TRACING_DEFAULT 1
#else
# define TAU_TRACING_DEFAULT 0
#endif

#ifdef PROFILING_ON
# define TAU_PROFILING_DEFAULT 1
#else
# define TAU_PROFILING_DEFAULT 0
#endif

#ifdef TAU_EACH_SEND
# define TAU_COMM_MATRIX_DEFAULT 1
#else
# define TAU_COMM_MATRIX_DEFAULT 0
#endif

#define TAU_TRACK_MESSAGE_DEFAULT 0

#define TAU_THROTTLE_DEFAULT 1
#ifdef TAU_MPI
  #define TAU_SYNCHRONIZE_CLOCKS_DEFAULT 1
#else
  #define TAU_SYNCHRONIZE_CLOCKS_DEFAULT 0
#endif /* TAU_MPI */

/************************** tau.conf stuff, adapted from Scalasca ***********/

/*********************************************************************
 * Tau configuration record definition
 ********************************************************************/
typedef struct {
  char *key;
  char *val;
} tauConf_data;

static tauConf_data *tauConf_vals = 0;
static int tauConf_numVals = 0;
static int tauConf_maxVals = 0;

/*********************************************************************
 * Syntax checker
 ********************************************************************/
static void TauConf_check_syntax(char *val, char *epos, const char *fname) {
  char *tmp = val;
  while (isspace(*val)) {
    val++;
  }
  if (val != epos) {
    TAU_VERBOSE("TAU: Warning, Syntax error in %s::%s", fname, tmp);
  }
}

/*********************************************************************
 * Format configuration value
 ********************************************************************/
static char *TauConf_format(char *val) {
  char *it;

  while (isspace(*val)) {
    val++;
  }

  if (*val == 0) {
    return NULL;
  }

  it = val + strlen(val) - 1;
  while (isspace(*it)) {
    it--;
  }
  *(++it) = 0;
  return val;
}

/*********************************************************************
 * Set a configuration value
 ********************************************************************/
static void TauConf_setval(const char *key, const char *val) {
  int newIdx = tauConf_numVals;

  if (newIdx + 1 > tauConf_maxVals) {
    tauConf_maxVals += 100;
    tauConf_vals = (tauConf_data *)realloc(tauConf_vals, tauConf_maxVals * sizeof(tauConf_data));
  }

  tauConf_vals[newIdx].key = strdup(key);
  tauConf_vals[newIdx].val = strdup(val);

  tauConf_numVals = tauConf_numVals + 1;
}

/*********************************************************************
 * Get a configuration value
 ********************************************************************/
static const char *TauConf_getval(const char *key) {
  int i;
  for (i = 0; i < tauConf_numVals; i++) {
    if (!strcmp(key, tauConf_vals[i].key)) {
      return tauConf_vals[i].val;
    }
  }
  return NULL;
}

/*********************************************************************
 * Parse a tau.conf file
 ********************************************************************/
static int TauConf_parse(FILE *cfgFile, const char *fname) {
  char buf[MAX_LN_LEN], *it, *val;

  TAU_VERBOSE("TAU: Reading configuration file: %s\n", fname);

  while (fgets(buf, MAX_LN_LEN, cfgFile)) {
    if ((strlen(buf) == MAX_LN_LEN - 1) && (buf[MAX_LN_LEN - 1] != '\n')) {
      TAU_VERBOSE("TAU: Warning, syntax error in %s::%s (Skipped parsing at overlong line)\n", fname, buf);
      break;
    } else {
      it = buf;
      while (*it && isspace(*it)) {   /* Skip until either end of string or char  */
        it++;
      }
      if (*it == '#') {
        continue;           /* If it is a comment, skip the line */
      }
      while (*it && *it != '=') { /* Skip until end of string or = or # */
        it++;
      }
      if (*it != '=') {
        *--it = 0;
        TauConf_check_syntax(buf, it, fname);
        continue;
      }
      *it++ = 0;
      val = it;
      while (*it  && *it != '#') { /* Skip until either end of string or # */
        it++;
      }
      *it = 0;
      TauConf_setval(TauConf_format(buf), TauConf_format(val));
    }
  }
  return 0;
}

/*********************************************************************
 * Read configuration file
 ********************************************************************/
static int TauConf_read() {
  const char *tmp;

  tmp = getenv("TAU_CONF");
  if (tmp == NULL) {
    tmp = "tau.conf";
  }
  FILE *cfgFile = fopen(tmp, "r");
  if (cfgFile) {
    TauConf_parse(cfgFile, tmp);
    fclose(cfgFile);
  }
  return 0;
}

/*********************************************************************
 * Local getconf routine
 ********************************************************************/
static const char *getconf(const char *key) {
  const char *val = TauConf_getval(key);
  if (val) {
    return val;
  }
  return getenv(key);
}

/****************************************************************************/

extern "C" { /* C linkage */
static int env_synchronize_clocks = 0;
static int env_verbose = 0;
static int env_throttle = 0;
static int env_callpath = 0;
static int env_compensate = 0;
static int env_profiling = 0;
static int env_tracing = 0;
static int env_callpath_depth = 0;
static int env_depth_limit = 0;
static int env_track_message = 0;
static int env_comm_matrix = 0;
static int env_track_memory_heap = 0;
static int env_track_memory_headroom = 0;
static int env_extras = 0;
static int env_ebs_frequency = 0;
static int env_ebs_inclusive = 0;
static const char *env_ebs_source = "itimer";

static int env_profile_format = TAU_FORMAT_PROFILE;
static double env_throttle_numcalls = 0;
static double env_throttle_percall = 0;
static const char *env_profiledir = NULL;
static const char *env_tracedir = NULL;
static const char *env_metrics = NULL;

/*********************************************************************
 * Write to stderr if verbose mode is on
 ********************************************************************/
void TAU_VERBOSE(const char *format, ...) {
  va_list args;
  if (env_verbose != 1) {
    return;
  }
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}

/*********************************************************************
 * Parse a boolean value
 ********************************************************************/
static int parse_bool(const char *str, int default_value = 0) {
  if (str == NULL) {
    return default_value;
  }
  static char strbuf[128];
  char *ptr = strbuf;
  strncpy(strbuf, str, 128);
  while (*ptr) {
    *ptr = tolower(*ptr);
    ptr++;
  }
  if (strcmp(strbuf, "yes") == 0  ||
      strcmp(strbuf, "true") == 0 ||
      strcmp(strbuf, "on") == 0 ||
      strcmp(strbuf, "1") == 0) {
    return 1;
  } else {
    return 0;
  }
}

const char *TauEnv_get_metrics() {
  return env_metrics;
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

int TauEnv_get_compensate() {
  return env_compensate;
}

int TauEnv_get_comm_matrix() {
  return env_comm_matrix;
}

int TauEnv_get_track_message() {
  return env_track_message;
}

int TauEnv_get_track_memory_heap() {
  return env_track_memory_heap;
}

int TauEnv_get_track_memory_headroom() {
  return env_track_memory_headroom;
}

int TauEnv_get_extras() {
  return env_extras;
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

int TauEnv_get_depth_limit() {
  return env_depth_limit;
}

void TauEnv_set_depth_limit(int value) {
  env_depth_limit = value;
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

int TauEnv_get_ebs_frequency() {
  return env_ebs_frequency;
}

int TauEnv_get_ebs_inclusive() {
  return env_ebs_inclusive;
}

const char *TauEnv_get_ebs_source() {
  return env_ebs_source;
}

/*********************************************************************
 * Initialize the TauEnv module, get configuration values
 ********************************************************************/
void TauEnv_initialize() {
  char tmpstr[512];

  /* unset LD_PRELOAD so that vt_unify and elg_unify work */
  unsetenv("LD_PRELOAD");

  static int initialized = 0;

  if (!initialized) {
    const char *tmp;

    tmp = getenv("TAU_VERBOSE");
    if (parse_bool(tmp)) {
      env_verbose = 1;
    } else {
      env_verbose = 0;
    }

    /* Read the configuration file */
    TauConf_read();

    TAU_VERBOSE("TAU: Initialized TAU (TAU_VERBOSE=1)\n");

    /*** Options that can be used with Scalasca and VampirTrace ***/
    tmp = getconf("TAU_TRACK_HEAP");
    if (parse_bool(tmp, env_track_memory_heap)) {
      TAU_VERBOSE("TAU: Entry/Exit Memory tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_HEAP", "on");
      env_track_memory_heap = 1;
      env_extras = 1;
    } else {
      TAU_METADATA("TAU_TRACK_HEAP", "off");
      env_track_memory_heap = 0;
    }

    tmp = getconf("TAU_TRACK_HEADROOM");
    if (parse_bool(tmp, env_track_memory_headroom)) {
      TAU_VERBOSE("TAU: Entry/Exit Headroom tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_HEADROOM", "on");
      env_track_memory_headroom = 1;
      env_extras = 1;
    } else {
      TAU_METADATA("TAU_TRACK_HEADROOM", "off");
      env_track_memory_headroom = 0;
    }

    /*** Options that can be used with Scalasca and VampirTrace need to go above this line ***/
#ifdef TAU_EPILOG
    TAU_VERBOSE("TAU: Epilog/Scalasca active! (TAU measurement disabled)\n");
    return;
#endif

#ifdef TAU_VAMPIRTRACE
    TAU_VERBOSE("[%d] TAU: VampirTrace active! (TAU measurement disabled)\n", getpid());
    return;
#endif

    if ((env_profiledir = getconf("PROFILEDIR")) == NULL) {
      env_profiledir = ".";   /* current directory */
    }
    TAU_VERBOSE("TAU: PROFILEDIR is \"%s\"\n", env_profiledir);

    if ((env_tracedir = getconf("TRACEDIR")) == NULL) {
      env_tracedir = ".";   /* current directory */
    }
    TAU_VERBOSE("TAU: TRACEDIR is \"%s\"\n", env_tracedir);

    int profiling_default = TAU_PROFILING_DEFAULT;
    /* tracing */
    tmp = getconf("TAU_TRACE");
    if (parse_bool(tmp, TAU_TRACING_DEFAULT)) {
      env_tracing = 1;
      env_track_message = 1;
      profiling_default = 0;
      TAU_VERBOSE("TAU: Tracing Enabled\n");
      TAU_METADATA("TAU_TRACE", "on");
    } else {
      env_tracing = 0;
      env_track_message = TAU_TRACK_MESSAGE_DEFAULT;
      TAU_VERBOSE("TAU: Tracing Disabled\n");
      TAU_METADATA("TAU_TRACE", "off");
    }

    /* profiling */
    tmp = getconf("TAU_PROFILE");
    if (parse_bool(tmp, profiling_default)) {
      env_profiling = 1;
      TAU_VERBOSE("TAU: Profiling Enabled\n");
      TAU_METADATA("TAU_PROFILE", "on");
    } else {
      env_profiling = 0;
      TAU_VERBOSE("TAU: Profiling Disabled\n");
      TAU_METADATA("TAU_PROFILE", "off");
    }

    if (env_profiling) {
      /* callpath */
      tmp = getconf("TAU_CALLPATH");
      if (parse_bool(tmp, TAU_CALLPATH_DEFAULT)) {
        env_callpath = 1;
        TAU_VERBOSE("TAU: Callpath Profiling Enabled\n");
        TAU_METADATA("TAU_CALLPATH", "on");
      } else {
        env_callpath = 0;
        TAU_VERBOSE("TAU: Callpath Profiling Disabled\n");
        TAU_METADATA("TAU_CALLPATH", "off");
      }

      /* compensate */
      tmp = getconf("TAU_COMPENSATE");
      if (parse_bool(tmp, TAU_COMPENSATE_DEFAULT)) {
        env_compensate = 1;
        env_extras = 1;
        TAU_VERBOSE("TAU: Overhead Compensation Enabled\n");
        TAU_METADATA("TAU_COMPENSATE", "on");
      } else {
        env_compensate = 0;
        TAU_VERBOSE("TAU: Overhead Compensation Disabled\n");
        TAU_METADATA("TAU_COMPENSATE", "off");
      }
    }

#ifdef TAU_MPI
    /* track comm (opposite of old -nocomm option) */
    tmp = getconf("TAU_TRACK_MESSAGE");
    if (parse_bool(tmp, env_track_message)) {
      env_track_message = 1;
    } else {
      env_track_message = 0;
    }

    /* comm matrix */
    tmp = getconf("TAU_COMM_MATRIX");
    if (parse_bool(tmp, TAU_COMM_MATRIX_DEFAULT)) {
      env_comm_matrix = 1;
      env_track_message = 1;
      TAU_VERBOSE("TAU: Comm Matrix Enabled\n");
      TAU_METADATA("TAU_COMM_MATRIX", "on");
    } else {
      env_comm_matrix = 0;
      TAU_VERBOSE("TAU: Comm Matrix Disabled\n");
      TAU_METADATA("TAU_COMM_MATRIX", "off");
    }

    if (env_track_message) {
      TAU_VERBOSE("TAU: Message Tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_MESSAGE", "on");
    } else {
      TAU_VERBOSE("TAU: Message Tracking Disabled\n");
      TAU_METADATA("TAU_TRACK_MESSAGE", "off");
    }
#endif

    /* clock synchronization */
    if (env_tracing == 0) {
      env_synchronize_clocks = 0;
    } else {
      tmp = getconf("TAU_SYNCHRONIZE_CLOCKS");
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
        TAU_METADATA("TAU_SYNCHRONIZE_CLOCKS", "on");
      } else {
        TAU_VERBOSE("TAU: Clock Synchronization Disabled\n");
        TAU_METADATA("TAU_SYNCHRONIZE_CLOCKS", "off");
      }
#endif
    }

    /* callpath depth */
    const char *depth = getconf("TAU_CALLPATH_DEPTH");
    env_callpath_depth = TAU_CALLPATH_DEPTH_DEFAULT;
    if (depth) {
      env_callpath_depth = atoi(depth);
/*      if (env_callpath_depth <= 1) { */
/*        env_callpath_depth = TAU_CALLPATH_DEPTH_DEFAULT; */
/*      } */
      if (env_callpath_depth < 0) {
        env_callpath_depth = TAU_CALLPATH_DEPTH_DEFAULT;
      }
    }
    if (env_callpath) {
      TAU_VERBOSE("TAU: Callpath Depth = %d\n", env_callpath_depth);
    }
    sprintf(tmpstr, "%d", env_callpath_depth);
    TAU_METADATA("TAU_CALLPATH_DEPTH", tmpstr);

#ifdef TAU_DEPTH_LIMIT
    /* depthlimit depth */
    tmp = getconf("TAU_DEPTH_LIMIT");
    env_depth_limit = TAU_DEPTH_LIMIT_DEFAULT;
    if (tmp) {
      env_depth_limit = atoi(tmp);
    }
    TAU_VERBOSE("TAU: Depth Limit = %d\n", env_depth_limit);
    sprintf(tmpstr, "%d", env_depth_limit);
    TAU_METADATA("TAU_DEPTH_LIMIT", tmpstr);
#endif /* TAU_DEPTH_LIMIT */

    /* Throttle */
    tmp = getconf("TAU_THROTTLE");
    if (parse_bool(tmp, TAU_THROTTLE_DEFAULT)) {
      env_throttle = 1;
      TAU_VERBOSE("TAU: Throttling Enabled\n");
      TAU_METADATA("TAU_THROTTLE", "on");
    } else {
      env_throttle = 0;
      TAU_VERBOSE("TAU: Throttling Disabled\n");
      TAU_METADATA("TAU_THROTTLE", "off");
    }

    const char *percall = getconf("TAU_THROTTLE_PERCALL");
    env_throttle_percall = TAU_THROTTLE_PERCALL_DEFAULT;
    if (percall) {
      env_throttle_percall = strtod(percall, 0);
    }

    const char *numcalls = getconf("TAU_THROTTLE_NUMCALLS");
    env_throttle_numcalls = TAU_THROTTLE_NUMCALLS_DEFAULT;
    if (numcalls) {
      env_throttle_numcalls = strtod(numcalls, 0);
    }

    if (env_throttle) {
      TAU_VERBOSE("TAU: Throttle PerCall = %g\n", env_throttle_percall);
      TAU_VERBOSE("TAU: Throttle NumCalls = %g\n", env_throttle_numcalls);

      sprintf(tmpstr, "%g", env_throttle_percall);
      TAU_METADATA("TAU_THROTTLE_PERCALL", tmpstr);
      sprintf(tmpstr, "%g", env_throttle_numcalls);
      TAU_METADATA("TAU_THROTTLE_NUMCALLS", tmpstr);
    }

    const char *profileFormat = getconf("TAU_PROFILE_FORMAT");
    if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "snapshot")) {
      env_profile_format = TAU_FORMAT_SNAPSHOT;
      TAU_VERBOSE("TAU: Output Format: snapshot\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "snapshot");
    } else if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "merged")) {
      env_profile_format = TAU_FORMAT_MERGED;
      TAU_VERBOSE("TAU: Output Format: merged\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "merged");
    } else {
      env_profile_format = TAU_FORMAT_PROFILE;
      TAU_VERBOSE("TAU: Output Format: profile\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "profile");
    }

    if ((env_metrics = getconf("TAU_METRICS")) == NULL) {
      env_metrics = "";   /* default to 'time' */
      TAU_VERBOSE("TAU: METRICS is not set\n", env_metrics);
    } else {
      TAU_VERBOSE("TAU: METRICS is \"%s\"\n", env_metrics);
    }

#ifdef TAU_EXP_SAMPLING
    /* TAU_EXP_SAMPLING frequency */
    const char *ebs_frequency = getconf("TAU_EBS_FREQUENCY");
    env_ebs_frequency = TAU_EBS_FREQUENCY_DEFAULT;
    if (ebs_frequency) {
      env_ebs_frequency = atoi(ebs_frequency);
      if (env_ebs_frequency < 0) {
        env_ebs_frequency = TAU_EBS_FREQUENCY_DEFAULT;
      }
    }
    TAU_VERBOSE("TAU: EBS frequency = %d usec\n", env_ebs_frequency);
    sprintf(tmpstr, "%d usec", env_ebs_frequency);
    TAU_METADATA("TAU_EBS_FREQUENCY", tmpstr);

    const char *ebs_inclusive = getconf("TAU_EBS_INCLUSIVE");
    env_ebs_inclusive = TAU_EBS_INCLUSIVE_DEFAULT;
    if (ebs_inclusive) {
      env_ebs_inclusive = atoi(ebs_inclusive);
      if (env_ebs_inclusive < 0) {
        env_ebs_inclusive = TAU_EBS_INCLUSIVE_DEFAULT;
      }
    }
    TAU_VERBOSE("TAU: EBS inclusive = %d usec\n", env_ebs_inclusive);
    sprintf(tmpstr, "%d usec", env_ebs_inclusive);
    TAU_METADATA("TAU_EBS_INCLUSIVE", tmpstr);


    if ((env_ebs_source = getconf("TAU_EBS_SOURCE")) == NULL) {
      env_ebs_source = "itimer";
    }
    TAU_VERBOSE("TAU: EBS Source: %s\n", env_ebs_source);

    env_callpath = 1;
    env_callpath_depth = 300;
    TAU_VERBOSE("TAU: EBS Overriding callpath settings, callpath enabled, depth = 300\n");

#endif

  }
}
} /* C linkage */
