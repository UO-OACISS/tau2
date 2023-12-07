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
#include <sys/stat.h>
#include <sys/types.h>

#ifndef TAU_WINDOWS
#include <strings.h>
#else
#define strcasecmp(X, Y)  stricmp(X, Y)
#define unsetenv(X)
#endif

#ifdef TAU_ANDROID
#include <android/log.h>
#endif

#include <iostream>
#include <ctype.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include <time.h>
#include <Profile/TauEnv.h>
#include <Profile/TauHandler.h>
#include <TAU.h>
#include <tauroot.h>
#include <tauarch.h>
#include <fcntl.h>
#include <string>
#include <set>

#include <iostream>
#include <sstream>
using namespace std;

#ifndef TAU_BGP
//#include <pwd.h>
#endif /* TAU_BGP */

#ifdef TAU_WINDOWS
/* We are on Windows which doesn't have strtok_s */
# define strtok_r strtok_s
#define TAU_LIB_DIR "tau2/win32/lib"
#endif


#define MAX_LN_LEN 2048

/* We should throttle if number n > a && percall < b .a and b are given below */
#define TAU_THROTTLE_NUMCALLS_DEFAULT 100000
#define TAU_THROTTLE_PERCALL_DEFAULT  10
#define TAU_CALLPATH_DEPTH_DEFAULT  2

#define TAU_DEPTH_LIMIT_DEFAULT INT_MAX

#define TAU_DISABLE_INSTRUMENTATION_DEFAULT 0

/* If TAU is built with -PROFILECALLPATH, we turn callpath profiling on by default */
#ifdef TAU_CALLPATH
# define TAU_CALLPATH_DEFAULT 1
#else
# define TAU_CALLPATH_DEFAULT 0
#endif

#define TAU_ENABLE_THREAD_CONTEXT_DEFAULT 0

#define TAU_CALLSITE_DEFAULT 0
#define TAU_CALLSITE_DEPTH_DEFAULT 1 /* default to be local */

#ifdef __PGI
#define TAU_CALLSITE_OFFSET_DEFAULT 6 /* PGI needs 6 */
#else /* __PGI */
#define TAU_CALLSITE_OFFSET_DEFAULT 2 /* otherwise 2 */
#endif /* __PGI */

/* If we are using OpenMP and the collector API or OMPT */
#define TAU_OPENMP_RUNTIME_DEFAULT 1
#define TAU_OPENMP_RUNTIME_STATES_DEFAULT 0
#define TAU_OPENMP_RUNTIME_EVENTS_DEFAULT 1
#define TAU_OPENMP_RUNTIME_CONTEXT_TIMER "timer"
#define TAU_OPENMP_RUNTIME_CONTEXT_REGION "region"
#define TAU_OPENMP_RUNTIME_CONTEXT_NONE "none"

/* OMPT magic strings */
#define TAU_OMPT_SUPPORT_LEVEL_BASIC "basic"
#define TAU_OMPT_SUPPORT_LEVEL_LOWOVERHEAD "lowoverhead"
#define TAU_OMPT_SUPPORT_LEVEL_FULL "full"

/* if we are doing EBS sampling, set the default sampling period */
#define TAU_EBS_DEFAULT 0
#define TAU_EBS_DEFAULT_TAU 0
#define TAU_EBS_KEEP_UNRESOLVED_ADDR_DEFAULT 0
#if (defined (TAU_BGL) || defined(TAU_BGP))
#define TAU_EBS_PERIOD_DEFAULT 20000 // Kevin made this bigger,
#else
#if (defined (TAU_CRAYCNL) || defined(TAU_BGQ))
#define TAU_EBS_PERIOD_DEFAULT 50000 // Sameer made this bigger,
#else
#define TAU_EBS_PERIOD_DEFAULT 10000 // Kevin made this bigger,

#ifdef TAU_PYTHON
#undef TAU_EBS_PERIOD_DEFAULT
#define TAU_EBS_PERIOD_DEFAULT 30000 // Sameer made this bigger,
#endif /* TAU_PYTHON */

#endif /* CRAYCNL */
#endif
// because smaller causes problems sometimes.
/* if we are doing EBS sampling, set whether we want inclusive samples */
/* that is, main->foo->mpi_XXX is a sample for main, foo and mpi_xxx */
#define TAU_EBS_INCLUSIVE_DEFAULT 0

#define TAU_EBS_SOURCE_DEFAULT "itimer"
#define TAU_EBS_UNWIND_DEFAULT 0
#ifdef TAU_USE_BACKTRACE
#define TAU_EBS_UNWIND_DEPTH_DEFAULT 0
#else
#define TAU_EBS_UNWIND_DEPTH_DEFAULT 10
#endif

#define TAU_EBS_RESOLUTION_STR_LINE "line"
#define TAU_EBS_RESOLUTION_STR_FILE "file"
#define TAU_EBS_RESOLUTION_STR_FUNCTION "function"
#define TAU_EBS_RESOLUTION_STR_FUNCTION_LINE "function_line"

/* Experimental feature - pre-computation of statistics */
//#if (defined(TAU_UNIFY) && defined(TAU_MPI))
#if defined(TAU_UNIFY)
#define TAU_PRECOMPUTE_DEFAULT 1
#endif /* TAU_UNIFY && TAU_MPI */

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

#define TAU_TRACE_FORMAT_DEFAULT TAU_TRACE_FORMAT_TAU

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

#define TAU_LITE_DEFAULT 0

#define TAU_TRACK_IO_PARAMS_DEFAULT 0

#define TAU_TRACK_SIGNALS_DEFAULT 0
/* In TAU_TRACK_SIGNALS operations, do we invoke gdb? */
#define TAU_SIGNALS_GDB_DEFAULT 0
/* Also dump backtrace to stderr */
#define TAU_ECHO_BACKTRACE_DEFAULT 0

#define TAU_SUMMARY_DEFAULT 0

#define TAU_IBM_BG_HWP_COUNTERS 0

#define TAU_THROTTLE_DEFAULT 1
#ifdef TAU_MPI
  #define TAU_SYNCHRONIZE_CLOCKS_DEFAULT 1
#else
  #define TAU_SYNCHRONIZE_CLOCKS_DEFAULT 0
#endif /* TAU_MPI */

#define TAU_CUPTI_API_DEFAULT "runtime"
#define TAU_CUDA_DEVICE_NAME_DEFAULT NULL
#define TAU_TRACK_CUDA_INSTRUCTIONS_DEFAULT ""
#define TAU_TRACK_CUDA_CDP_DEFAULT 0
#define TAU_TRACK_UNIFIED_MEMORY_DEFAULT 0
#define TAU_TRACK_CUDA_SASS_DEFAULT 0
#define TAU_SASS_TYPE_DEFAULT "kernel"
#define TAU_OUTPUT_CUDA_CSV_DEFAULT 0
#define TAU_TRACK_CUDA_ENV_DEFAULT 0

#define TAU_MIC_OFFLOAD_DEFAULT 0

#define TAU_BFD_LOOKUP 1

// Memory debugging environment variable defaults
#define TAU_MEMDBG_PROTECT_ABOVE_DEFAULT  0
#define TAU_MEMDBG_PROTECT_BELOW_DEFAULT  0
#define TAU_MEMDBG_PROTECT_FREE_DEFAULT   0
#define TAU_MEMDBG_PROTECT_GAP_DEFAULT    0
#define TAU_MEMDBG_FILL_GAP_DEFAULT       0 // 0 => undefined, not zero
#define TAU_MEMDBG_ALLOC_MIN_DEFAULT      0 // 0 => undefined, not zero
#define TAU_MEMDBG_ALLOC_MAX_DEFAULT      0 // 0 => undefined, not zero
#define TAU_MEMDBG_OVERHEAD_DEFAULT       0 // 0 => undefined, not zero
#ifdef TAU_BGQ
#define TAU_MEMDBG_ALIGNMENT_DEFAULT      64
#else
#define TAU_MEMDBG_ALIGNMENT_DEFAULT      sizeof(long)
#endif
#define TAU_MEMDBG_ZERO_MALLOC_DEFAULT    0
#define TAU_MEMDBG_ATTEMPT_CONTINUE_DEFAULT 0

// pthread stack size default
#define TAU_PTHREAD_STACK_SIZE_DEFAULT    0

#define TAU_MERGE_METADATA_DEFAULT 0
#define TAU_DISABLE_METADATA_DEFAULT 0

#define TAU_MEM_CALLPATH_DEFAULT 0
#define TAU_REGION_ADDRESSES_DEFAULT 0

/* Thread recycling */
#define TAU_RECYCLE_THREADS_DEFAULT 0

// forward declartion of cuserid. need for c++ compilers on Cray.
extern "C" char *cuserid(char *);

#ifdef TAU_MPI
extern "C" void Tau_set_usesMPI(int value);
#endif /* TAU_MPI */

#ifdef TAU_ENABLE_ROCTRACER
extern void Tau_roctracer_start_tracing(void);
#endif /* TAU_ENABLE_ROCTRACER */

/************************** tau.conf stuff, adapted from Scalasca ***********/

extern "C" {

static int env_synchronize_clocks = 0;
static int env_verbose = 0;
static int env_verbose_file = 0;
static int env_verbose_rank = -1;
static int env_throttle = 0;
static double env_evt_threshold = 0.0;
static int env_interval = 0;
static int env_disable_instrumentation = 0;
static double env_max_records = 64*1024;
static int env_callpath = 0;
static int env_thread_context = 0;
static int env_callsite = 0;
static int env_callsite_depth = 0;
static int env_callsite_offset = TAU_CALLSITE_OFFSET_DEFAULT;
static int env_compensate = 0;
static int env_profiling = 0;
static int env_tracing = 0;
static int env_thread_per_gpu_stream = 0;
static int env_trace_format = TAU_TRACE_FORMAT_DEFAULT;
static int env_callpath_depth = 0;
static int env_depth_limit = 0;
static int env_track_message = 0;
static int env_comm_matrix = 0;
static int env_track_memory_heap = 0;
static int env_track_power = 0;
static int env_track_memory_footprint = 0;
static int env_show_memory_functions = 0;
static int env_track_load = 0;
static int env_tau_lite = 0;
static int env_tau_anonymize = 0;
static int env_track_memory_leaks = 0;
static int env_track_memory_headroom = 0;
static int env_track_io_params = 0;
static int env_track_signals = TAU_TRACK_SIGNALS_DEFAULT;
static int env_signals_gdb = TAU_SIGNALS_GDB_DEFAULT;
static int env_echo_backtrace = TAU_ECHO_BACKTRACE_DEFAULT;
static int env_track_mpi_t_pvars = 0;
static int env_mpi_t_enable_user_tuning_policy = 0;
static int env_summary_only = 0;
static int env_ibm_bg_hwp_counters = 0;
/* This is a malleable default */
static int env_ebs_keep_unresolved_addr = 0;
static int env_ebs_period = 0;
static int env_ebs_inclusive = 0;
static int env_ompt_resolve_address_eagerly = 1;
static int env_ompt_support_level = 0;
static int env_ompt_force_finalize = 1;
static int env_openmp_runtime_enabled = 1;
static int env_openmp_runtime_states_enabled = 0;
static int env_openmp_runtime_events_enabled = 1;
static int env_openmp_runtime_context = 1;
static int env_ebs_enabled = 0;
static int env_ebs_enabled_tau = 0;
static const char *env_ebs_source = "itimer";
static const char *env_ebs_source_orig = "itimer";
static int env_ebs_unwind_enabled = 0;
static int env_ebs_unwind_depth = TAU_EBS_UNWIND_DEPTH_DEFAULT;
static int env_ebs_resolution = TAU_EBS_RESOLUTION_LINE;

static int env_stat_precompute = 0;
static int env_child_forkdirs = 0;
static int env_l0_api_tracing = 0;

static int env_profile_format = TAU_FORMAT_PROFILE;
static const char *env_profile_prefix = NULL;
static double env_throttle_numcalls = 0;
static double env_throttle_percall = 0;
static const char *env_profiledir = NULL;
static const char *env_tracedir = NULL;
static const char *env_metrics = NULL;
static const char *env_cvar_metrics = NULL;
static const char *env_mpi_t_comm_metric_values = NULL;
static const char *env_cvar_values = NULL;
static const char *env_plugins_path = NULL;
static const char *env_plugins = NULL;
static int env_plugins_enabled = 0;
static int env_track_mpi_t_comm_metric_values = 0;
static const char *env_select_file = NULL;
static const char *env_cupti_api = TAU_CUPTI_API_DEFAULT;
static const char * env_cuda_device_name = TAU_CUDA_DEVICE_NAME_DEFAULT;
static int env_sigusr1_action = TAU_ACTION_DUMP_PROFILES;
static const char *env_track_cuda_instructions = TAU_TRACK_CUDA_INSTRUCTIONS_DEFAULT;
static int env_track_cuda_cdp = TAU_TRACK_CUDA_CDP_DEFAULT;
static int env_track_unified_memory = TAU_TRACK_UNIFIED_MEMORY_DEFAULT;
static int env_track_cuda_sass = TAU_TRACK_CUDA_SASS_DEFAULT;
static const char* env_sass_type = TAU_SASS_TYPE_DEFAULT;
static int env_output_cuda_csv = TAU_OUTPUT_CUDA_CSV_DEFAULT;
static const char *env_binaryexe = NULL;
static int env_track_cuda_env = TAU_TRACK_CUDA_ENV_DEFAULT;
static int env_current_timer_exit_params = 0;

static int env_node_set = -1;

static int env_cudatotalthreads = 0;
static int env_taucuptiavail = 0;
static int env_nodenegoneseen = 0;
static int env_mic_offload = 0;
static int env_bfd_lookup = 0;

static int env_memdbg = 0;
static int env_memdbg_protect_above = TAU_MEMDBG_PROTECT_ABOVE_DEFAULT;
static int env_memdbg_protect_below = TAU_MEMDBG_PROTECT_BELOW_DEFAULT;
static int env_memdbg_protect_free = TAU_MEMDBG_PROTECT_FREE_DEFAULT;
static int env_memdbg_protect_gap = TAU_MEMDBG_PROTECT_GAP_DEFAULT;
// All values of env_memdbg_fill_gap_value are valid fill patterns
static int env_memdbg_fill_gap = TAU_MEMDBG_FILL_GAP_DEFAULT;
static unsigned char env_memdbg_fill_gap_value = 0xAB;
// All values of env_memdbg_alloc_min are valid limits
static int env_memdbg_alloc_min = TAU_MEMDBG_ALLOC_MIN_DEFAULT;
static size_t env_memdbg_alloc_min_value = 0;
// All values of env_memdbg_alloc_max are valid limits
static int env_memdbg_alloc_max = TAU_MEMDBG_ALLOC_MAX_DEFAULT;
static size_t env_memdbg_alloc_max_value = 0;
// All values of env_memdbg_overhead are valid limits
static int env_memdbg_overhead = TAU_MEMDBG_OVERHEAD_DEFAULT;
static size_t env_memdbg_overhead_value = 0;
static size_t env_memdbg_alignment = TAU_MEMDBG_ALIGNMENT_DEFAULT;
static int env_memdbg_zero_malloc = TAU_MEMDBG_ZERO_MALLOC_DEFAULT;
static int env_memdbg_attempt_continue = TAU_MEMDBG_ATTEMPT_CONTINUE_DEFAULT;
static int env_merge_metadata = 1;
static int env_disable_metadata = 0;

static int env_pthread_stack_size = TAU_PTHREAD_STACK_SIZE_DEFAULT;
static int env_papi_multiplexing = 0;

#ifdef TAU_ANDROID
static int env_alfred_port = 6113;
#endif

static int env_mem_callpath = 0;
static int env_mem_all = 0;
static const char *env_mem_classes = NULL;
static std::set<std::string> * env_mem_classes_set = NULL;
static int env_region_addresses = TAU_REGION_ADDRESSES_DEFAULT;
static int env_recycle_threads = TAU_RECYCLE_THREADS_DEFAULT;

static const char *env_tau_exec_args = NULL;
static const char *env_tau_exec_path = NULL;

} // extern "C"

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

extern int Tau_util_readFullLine(char *line, FILE *fp);
/*********************************************************************
 * Get executable directory name: /usr/local/foo will return /usr/local
 ********************************************************************/
static char * Tau_get_cwd_of_exe()
{
  char * retval = NULL;

  FILE * f = fopen("/proc/self/cmdline", "r");
  if (f) {
    char * line = (char*)malloc(4096);
    line[0] ='\0';
    if (Tau_util_readFullLine(line, f)) {
      int pos = strlen(line) - 1;
      while(pos >= 0) {
        char c = line[pos];
        if (c == '/' || c == '\\') break;
        --pos;
      }
      if (pos >= 0) {
        line[pos] = '\0';
        retval = strdup(line);
      }
      free((void*)line);
    }
    fclose(f); // close the file if it is not null
  }
  return retval;
}

/*********************************************************************
 * Replace part of a string with another. similar to <algorithm> replace.
 ********************************************************************/
void Tau_util_replaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

/* A C implementation, for crappy compilers. */
void Tau_util_replaceStringInPlaceC(char * subject, const char search,
                          const char replace) {
    size_t pos = 0;
    size_t len = strlen(subject);
    while (pos < len) {
         if (subject[pos] == search) {
            subject[pos] = replace;
         }
         pos++;
    }
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

/*********************************************************************
 * Parse an integer value
 ********************************************************************/
static int parse_int(const char *str, int default_value = 0) {
  if (str == NULL) {
    return default_value;
  }
  int tmp = atoi(str);
  if (tmp < 0) {
    return default_value;
  }
  return tmp;
}

/*********************************************************************
 * Read configuration file
 ********************************************************************/
static int TauConf_read()
{
  const char *tmp;
  char conf_file_name[1024];

/* Eagerly get some settings needed for configuring verbose output */

  tmp = getenv("TAU_VERBOSE");
  if (parse_bool(tmp)) {
    env_verbose = 1;
    tmp = getenv("TAU_VERBOSE_FILE");
    if (parse_bool(tmp,env_verbose_file)) {
        env_verbose_file = 1;
    }
    tmp = getenv("TAU_VERBOSE_RANK");
    if (parse_int(tmp,env_verbose_rank)) {
        env_verbose_rank = Tau_get_node();
    }
    if ((env_profiledir = getenv("PROFILEDIR")) == NULL) {
      env_profiledir = ".";   /* current directory */
#ifdef TAU_GPI
      // if exe is /usr/local/foo, this will return /usr/local where profiles
      // may be stored if PROFILEDIR is not specified
      char const * cwd = Tau_get_cwd_of_exe();
      if (cwd) {
        env_profiledir = strdup(cwd);
      }
#endif /* TAU_GPI */
    }
  } else {
    env_verbose = 0;
  }

  tmp = getenv("TAU_CONF");
  if (tmp == NULL) {
#ifdef TAU_ANDROID
    tmp = "/sdcard/tau.conf";
#else
    tmp = "tau.conf";
#endif
  }
  FILE * cfgFile = fopen(tmp, "r");
  if (!cfgFile) {
    char const * exedir = Tau_get_cwd_of_exe();
    if (!exedir) {
      sprintf(conf_file_name, "./tau.conf");
    } else {
      sprintf(conf_file_name, "%s/tau.conf", exedir);
      free((void*)exedir);
    }
    TAU_VERBOSE("Trying %s\n", conf_file_name);
    cfgFile = fopen(conf_file_name, "r");
  }
  if (cfgFile) {
    TauConf_parse(cfgFile, tmp);
    fclose(cfgFile);
  } else {
    sprintf(conf_file_name, "%s/tau_system_defaults/tau.conf", TAUROOT);
    cfgFile = fopen(conf_file_name, "r");
    if (cfgFile) {
      TauConf_parse(cfgFile, tmp);
      fclose(cfgFile);
      TAU_VERBOSE("TAU: Read systemwide default configuration settings from %s\n", conf_file_name);
    }
  }
  return 0;
}

/*********************************************************************
 * TAU's getconf routine
 ********************************************************************/
const char *getconf(const char *key) {
  const char *val = TauConf_getval(key);
  //TAU_VERBOSE("%s=%s\n", key, val);
  if (val) {
    return val;
  }
  return getenv(key);
}

/*********************************************************************
 * Local Tau_check_dirname routine
 ********************************************************************/
char * Tau_check_dirname(const char * dir)
{
  if (strcmp(dir, "$TAU_LOG_DIR") == 0) {
    TAU_VERBOSE("Using PROFILEDIR=%s\n", dir);
    const char *logdir = getconf("TAU_LOG_PATH");
    const char *jobid = getconf("COBALT_JOBID");
    if (jobid == (const char *)NULL) jobid = strdup("0");
    TAU_VERBOSE("jobid = %s\n", jobid);
    time_t theTime = time(NULL);
    struct tm *thisTime = gmtime(&theTime);
    thisTime = localtime(&theTime);
    char user[1024];
    int ret;

    char logfiledir[2048];
    char scratchdir[2048];
#if (defined (TAU_BGL) || defined(TAU_BGP) || defined(TAU_BGQ) || (defined(__linux__) && !defined(TAU_ANDROID)))
    if (cuserid(user) == NULL) {
      sprintf(user,"unknown");
    }
#else

#ifdef TAU_WINDOWS
    char *temp = "unknown";
#else
    /*    struct passwd *pwInfo = getpwuid(geteuid());
     if ((pwInfo != NULL) &&
     (pwInfo->pw_name != NULL)) {
     strcpy(user, pwInfo->pw_name);
     */
    char *temp = getlogin();
    TAU_VERBOSE("TAU: cuserid returns %s\n", temp);
#endif // TAU_WINDOWS
    if (temp != NULL) {
      sprintf(user, "%s", temp);
    } else {
      sprintf(user, "unknown");
    }
    free(temp);
#endif /* TAU_BGP */
    ret = sprintf(logfiledir, "%s/%d/%d/%d/%s_id%s_%d-%d-%d", logdir, (thisTime->tm_year + 1900),
        (thisTime->tm_mon + 1), thisTime->tm_mday, user, jobid, (thisTime->tm_mon + 1), thisTime->tm_mday,
        (thisTime->tm_hour * 60 * 60 + thisTime->tm_min * 60 + thisTime->tm_sec));
#ifndef TAU_WINDOWS
	if (ret < 0) { TAU_VERBOSE("sprintf failed! %s %s %s", __func__, __FILE__, __LINE__); }
#else
        if (ret < 0) { TAU_VERBOSE("sprintf failed! %s %s", __FILE__, __LINE__); }
#endif  /* TAU_WINDOWS */
    TAU_VERBOSE("Using logdir = %s\n", logfiledir);
    if (RtsLayer::myNode() < 1) {
#ifdef TAU_WINDOWS
      mkdir(logfiledir);
#else

      mode_t oldmode;
      oldmode = umask(0);
      mkdir(logdir, S_IRWXU | S_IRGRP | S_IWGRP | S_IXGRP | S_IRWXO);
      sprintf(scratchdir, "%s/%d", logdir, (thisTime->tm_year + 1900));
      mkdir(scratchdir, S_IRWXU | S_IRGRP | S_IWGRP | S_IXGRP | S_IRWXO);
      sprintf(scratchdir, "%s/%d/%d", logdir, (thisTime->tm_year + 1900), (thisTime->tm_mon + 1));
      mkdir(scratchdir, S_IRWXU | S_IRGRP | S_IWGRP | S_IXGRP | S_IRWXO);
      sprintf(scratchdir, "%s/%d/%d/%d", logdir, (thisTime->tm_year + 1900), (thisTime->tm_mon + 1), thisTime->tm_mday);
      mkdir(scratchdir, S_IRWXU | S_IRGRP | S_IWGRP | S_IXGRP | S_IRWXO);
      TAU_VERBOSE("mkdir %s\n", scratchdir);

      mkdir(logfiledir, S_IRWXU | S_IRGRP | S_IXGRP | S_IXGRP | S_IRWXO);
      TAU_VERBOSE("mkdir %s\n", logfiledir);
      umask(oldmode);
#endif
    }
    return strdup(logfiledir);
  }
  return (char *)dir;

}



/****************************************************************************/

class Tau_logfile_t {
public:
    FILE * pfile;
    Tau_logfile_t() : pfile(stderr) {
        if (env_verbose_file == 1 &&
            env_verbose_rank == Tau_get_node()) {
            std::stringstream ss;
            ss << env_profiledir << "/tau." <<
            (Tau_get_node() < 0 ? 0 : Tau_get_node())
            << ".log";
            std::string tmp(ss.str());
            pfile = fopen(tmp.c_str(),"w");
        }
    }
    ~Tau_logfile_t() {
        if (env_verbose_file == 1 &&
            env_verbose_rank == Tau_get_node()) {
            fclose(pfile);
        }
    }
};

extern "C" { /* C linkage */

#ifdef TAU_GPI
#include <GPI.h>
#include <GpiLogger.h>
#endif /* TAU_GPI */
/*********************************************************************
 * Write to stderr if verbose mode is on
 ********************************************************************/
extern "C" x_uint64 TauMetrics_getTimeOfDay();
void TAU_VERBOSE(const char *format, ...)
{
  if (env_verbose == 1) {
    static Tau_logfile_t foo;
    TauInternalFunctionGuard protects_this_function;
    va_list args;

#ifdef TAU_ANDROID

    va_start(args, format);

    __android_log_vprint(ANDROID_LOG_VERBOSE, "TAU", format, args);

    //vasprintf(&str, format, args);

    va_end(args);

#else

    //fprintf(foo.pfile, "%llu : ", TauMetrics_getTimeOfDay());
    va_start(args, format);

#ifdef TAU_GPI
    gpi_vprintf(format, args);
#else
    vfprintf(foo.pfile, format, args);
#endif
    va_end(args);
    fflush(foo.pfile);

#endif
  } // END inside TAU
}

const char *TauEnv_get_metrics() {
  return env_metrics;
}

extern "C" const char *TauEnv_get_cvar_metrics() {
  return env_cvar_metrics;
}

extern "C" const char *TauEnv_get_mpi_t_comm_metric_values() {
  return env_mpi_t_comm_metric_values;
}

extern "C" const char *TauEnv_get_plugins_path() {
  return env_plugins_path;
}

extern "C" const char *TauEnv_get_plugins() {
  return env_plugins;
}

extern "C" int TauEnv_get_plugins_enabled() {
  return env_plugins_enabled;
}

extern "C" int TauEnv_get_track_mpi_t_comm_metric_values() {
  return env_track_mpi_t_comm_metric_values;
}

extern "C" int TauEnv_get_set_node() {
  return env_node_set;
}

extern "C" const char *TauEnv_get_cvar_values() {
  return env_cvar_values;
}

extern "C" const char *TauEnv_get_profiledir() {
  return env_profiledir;
}

extern "C" const char *TauEnv_get_tracedir() {
  return env_tracedir;
}

extern "C" void TauEnv_set_profiledir(const char * new_profiledir) {
    env_profiledir = strdup(new_profiledir);
}

extern "C" void TauEnv_set_tracedir(const char * new_tracedir) {
    env_tracedir = strdup(new_tracedir);
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

void TauEnv_set_throttle(int throttle) {
  env_throttle = throttle;
}

int TauEnv_get_disable_instrumentation() {
  return env_disable_instrumentation;
}

double TauEnv_get_max_records() {
  return env_max_records;
}

double TauEnv_get_evt_threshold() {
  return env_evt_threshold;
}

int TauEnv_get_interval() {
  return env_interval;
}

int TauEnv_get_callpath() {
  return env_callpath;
}

int TauEnv_get_threadContext() {
  return env_thread_context;
}

int TauEnv_get_callsite() {
  return env_callsite;
}

int TauEnv_get_callsite_depth() {
  return env_callsite_depth;
}

int TauEnv_get_callsite_offset() {
  return env_callsite_offset;
}

int TauEnv_get_compensate() {
  return env_compensate;
}

int TauEnv_get_level_zero_enable_api_tracing() {
  return env_l0_api_tracing;
}

int TauEnv_get_comm_matrix() {
  return env_comm_matrix;
}

int TauEnv_get_current_timer_exit_params() {
  return env_current_timer_exit_params;
}

int TauEnv_get_ompt_resolve_address_eagerly() {
  return env_ompt_resolve_address_eagerly;
}

int TauEnv_get_ompt_support_level() {
  return env_ompt_support_level;
}

int TauEnv_get_ompt_force_finalize() {
  return env_ompt_force_finalize;
}

int TauEnv_get_track_mpi_t_pvars() {
  return env_track_mpi_t_pvars;
}

int TauEnv_get_mpi_t_enable_user_tuning_policy() {
  return env_mpi_t_enable_user_tuning_policy;
}

int TauEnv_set_track_mpi_t_pvars(int value) {
  env_track_mpi_t_pvars = value;
  return env_track_mpi_t_pvars;
}

int TauEnv_set_ompt_resolve_address_eagerly(int value) {
  env_ompt_resolve_address_eagerly = value;
  return env_ompt_resolve_address_eagerly;
}

int TauEnv_set_ompt_support_level(int value) {
  env_ompt_support_level = value;
  return env_ompt_support_level;
}

int TauEnv_get_track_signals() {
  return env_track_signals;
}

int TauEnv_get_signals_gdb() {
  return env_signals_gdb;
}

int TauEnv_get_echo_backtrace() {
  return env_echo_backtrace;
}

int TauEnv_get_track_message() {
  return env_track_message;
}

int TauEnv_get_track_memory_heap() {
  return env_track_memory_heap;
}

int TauEnv_get_track_power() {
  return env_track_power;
}

int TauEnv_get_track_memory_footprint() {
  return env_track_memory_footprint;
}

int TauEnv_get_show_memory_functions() {
  return env_show_memory_functions;
}

int TauEnv_get_track_load() {
  return env_track_load;
}

int TauEnv_get_track_memory_leaks() {
  return env_track_memory_leaks;
}

int TauEnv_get_track_memory_headroom() {
  return env_track_memory_headroom;
}

int TauEnv_get_papi_multiplexing() {
  return env_papi_multiplexing;
}

int TauEnv_get_region_addresses() {
  return env_region_addresses;
}

int TauEnv_get_recycle_threads() {
  return env_recycle_threads;
}

int TauEnv_get_track_io_params() {
  return env_track_io_params;
}

int TauEnv_get_summary_only() {
  return env_summary_only;
}

int TauEnv_get_ibm_bg_hwp_counters() {
  return env_ibm_bg_hwp_counters;
}

int TauEnv_get_profiling() {
  return env_profiling;
}

int TauEnv_get_tracing() {
  return env_tracing;
}

int TauEnv_get_thread_per_gpu_stream() {
  return env_thread_per_gpu_stream;
}

int TauEnv_get_trace_format() {
  return env_trace_format;
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

const char* TauEnv_get_profile_prefix() {
  return env_profile_prefix;
}

int TauEnv_get_sigusr1_action() {
  return env_sigusr1_action;
}

int TauEnv_get_ebs_keep_unresolved_addr() {
  return env_ebs_keep_unresolved_addr;
}

  // *CWL* Only to be used by TAU whenever the desired ebs period violates
  //       system-supported thresholds.
void TauEnv_force_set_ebs_period(int period) {
  char tmpstr[512];
  env_ebs_period = period;
  sprintf(tmpstr, "%d", env_ebs_period);
  TAU_METADATA("TAU_EBS_PERIOD (FORCED)", tmpstr);
}

int TauEnv_get_ebs_period() {
  return env_ebs_period;
}

int TauEnv_get_ebs_inclusive() {
  return env_ebs_inclusive;
}

int TauEnv_get_ebs_enabled() {
  return env_ebs_enabled;
}

int TauEnv_get_ebs_enabled_tau() {
  return env_ebs_enabled_tau;
}

int TauEnv_get_ebs_resolution() {
  return env_ebs_resolution;
}

int TauEnv_get_openmp_runtime_enabled() {
  return env_openmp_runtime_enabled;
}

int TauEnv_get_openmp_runtime_states_enabled() {
  return env_openmp_runtime_states_enabled;
}

int TauEnv_get_openmp_runtime_events_enabled() {
  const char *tmp;

  tmp = getconf("TAU_OPENMP_RUNTIME_EVENTS");
  if (parse_bool(tmp, TAU_OPENMP_RUNTIME_EVENTS_DEFAULT)) {
    env_openmp_runtime_events_enabled = 1;
  } else {
    env_openmp_runtime_events_enabled = 0;
  }

  return env_openmp_runtime_events_enabled;
}

int TauEnv_get_openmp_runtime_context() {
  return env_openmp_runtime_context;
}

int TauEnv_get_ebs_unwind() {
  return env_ebs_unwind_enabled;
}

int TauEnv_get_ebs_unwind_depth() {
  return env_ebs_unwind_depth;
}

const char *TauEnv_get_ebs_source() {
  return env_ebs_source;
}

const char *TauEnv_get_ebs_source_orig() {
  return env_ebs_source_orig;
}

void TauEnv_override_ebs_source(const char *newName) {
  env_ebs_source = strdup(newName);
  TAU_METADATA("TAU_EBS_SOURCE (Override)", newName);
}

int TauEnv_get_stat_precompute() {
  return env_stat_precompute;
}

int TauEnv_get_child_forkdirs(){
  return env_child_forkdirs;
}

const char* TauEnv_get_cupti_api(){
  return env_cupti_api;
}

const char* TauEnv_get_cuda_device_name(){
  return env_cuda_device_name;
}

const char* TauEnv_get_cuda_instructions(){
  return env_track_cuda_instructions;
}

int TauEnv_get_cuda_track_cdp(){
  return env_track_cuda_cdp;
}

int TauEnv_get_cuda_track_unified_memory(){
  return env_track_unified_memory;
}

int TauEnv_get_cuda_track_sass(){
  return env_track_cuda_sass;
}

const char* TauEnv_get_cuda_sass_type(){
  return env_sass_type;
}

int TauEnv_get_cuda_csv_output(){
  return env_output_cuda_csv;
}

const char* TauEnv_get_cuda_binary_exe(){
  return env_binaryexe;
}

int TauEnv_get_cuda_track_env(){
  return env_track_cuda_env;
}

void TauEnv_set_cudaTotalThreads(int nthreads) {
    env_cudatotalthreads = nthreads;
}
int TauEnv_get_cudaTotalThreads() {
    return env_cudatotalthreads;
}

void TauEnv_set_tauCuptiAvail(int off) {
    env_taucuptiavail = off;
}
int TauEnv_get_tauCuptiAvail() {
    return env_taucuptiavail;
}

void TauEnv_set_nodeNegOneSeen(int nthreads) {
    env_nodenegoneseen = nthreads;
}
int TauEnv_get_nodeNegOneSeen() {
    return env_nodenegoneseen;
}

int TauEnv_get_mic_offload(){
  return env_mic_offload;
}

int TauEnv_get_bfd_lookup(){
  return env_bfd_lookup;
}

int TauEnv_get_lite_enabled() {
  return env_tau_lite;
}

int TauEnv_get_anonymize_enabled() {
  return env_tau_anonymize;
}

int TauEnv_get_memdbg() {
  return env_memdbg;
}

int TauEnv_get_memdbg_protect_above() {
  return env_memdbg_protect_above;
}
void TauEnv_set_memdbg_protect_above(int value) {
  env_memdbg_protect_above = value;
  env_memdbg = (env_memdbg_protect_above ||
                env_memdbg_protect_below ||
                env_memdbg_protect_free);
}

int TauEnv_get_memdbg_protect_below() {
  return env_memdbg_protect_below;
}
void TauEnv_set_memdbg_protect_below(int value) {
  env_memdbg_protect_below = value;
  env_memdbg = (env_memdbg_protect_above ||
                env_memdbg_protect_below ||
                env_memdbg_protect_free);
}

int TauEnv_get_memdbg_protect_free() {
  return env_memdbg_protect_free;
}
void TauEnv_set_memdbg_protect_free(int value) {
  env_memdbg_protect_free = value;
  env_memdbg = (env_memdbg_protect_above ||
                env_memdbg_protect_below ||
                env_memdbg_protect_free);
}

int TauEnv_get_memdbg_protect_gap() {
  return env_memdbg_protect_gap;
}

int TauEnv_get_memdbg_fill_gap() {
  return env_memdbg_fill_gap;
}

unsigned char TauEnv_get_memdbg_fill_gap_value() {
  return env_memdbg_fill_gap_value;
}

int TauEnv_get_memdbg_alloc_min() {
  return env_memdbg_alloc_min;
}

size_t TauEnv_get_memdbg_alloc_min_value() {
  return env_memdbg_alloc_min_value;
}

int TauEnv_get_memdbg_alloc_max() {
  return env_memdbg_alloc_max;
}

size_t TauEnv_get_memdbg_alloc_max_value() {
  return env_memdbg_alloc_max_value;
}

int TauEnv_get_memdbg_overhead() {
  return env_memdbg_overhead;
}

size_t TauEnv_get_memdbg_overhead_value() {
  return env_memdbg_overhead_value;
}

size_t TauEnv_get_memdbg_alignment() {
  return env_memdbg_alignment;
}

int TauEnv_get_memdbg_zero_malloc() {
  return env_memdbg_zero_malloc;
}

int TauEnv_get_memdbg_attempt_continue() {
  return env_memdbg_attempt_continue;
}

int TauEnv_get_pthread_stack_size() {
  return env_pthread_stack_size;
}

int TauEnv_get_merge_metadata(){
  return env_merge_metadata;
}

int TauEnv_get_disable_metadata(){
  return env_disable_metadata;
}

#ifdef TAU_ANDROID
int TauEnv_get_alfred_port() {
  return env_alfred_port;
}
#endif

int TauEnv_get_mem_callpath() {
  return env_mem_callpath;
}

const char *TauEnv_get_mem_classes() {
  return env_mem_classes;
}

int TauEnv_get_mem_class_present(const char * name) {
    if(env_mem_all) {
        return 1;
    }
    if(env_mem_classes_set == NULL) {
        return 0;
    }
    return env_mem_classes_set->count(name);
}

const char * TauEnv_get_tau_exec_args() {
  return env_tau_exec_args;
}

const char * TauEnv_get_tau_exec_path() {
  return env_tau_exec_path;
}


/*********************************************************************
 * Initialize the TauEnv module, get configuration values
 ********************************************************************/
void TauEnv_initialize()
{
  char tmpstr[512];

  /* unset LD_PRELOAD so that vt_unify and elg_unify work */
  unsetenv("LD_PRELOAD");

  static int initialized = 0;

  if (!initialized) {
    const char *tmp;
    char *saveptr;
    const char *key;
    const char *val;

    /* Read the configuration file */
    TauConf_read();

    tmp = getconf("TAU_VERBOSE");
    if (parse_bool(tmp,env_verbose)) {
      TAU_VERBOSE("TAU: VERBOSE enabled\n");
      TAU_METADATA("TAU_VERBOSE", "on");
      env_verbose = 1;
    }

    tmp = getconf("TAU_VERBOSE_FILE");
    if (parse_bool(tmp,env_verbose_file)) {
      TAU_VERBOSE("TAU: VERBOSE to file enabled\n");
      TAU_METADATA("TAU_VERBOSE_FILE", "on");
      env_verbose_file = 1;
    }

    tmp = getconf("TAU_VERBOSE_RANK");
    if (parse_int(tmp,env_verbose_rank)) {
      TAU_VERBOSE("TAU: VERBOSE RANK enabled\n");
      sprintf(tmpstr, "%d", env_verbose_rank);
      TAU_METADATA("TAU_VERBOSE_RANK", tmpstr);
      env_verbose_rank = Tau_get_node();
    }

    //sprintf(tmpstr, "%d", TAU_MAX_THREADS);//TODO: DYNATHREAD
    TAU_VERBOSE("TAU: Supporting dynamic allocation of threads\n");//, TAU_MAX_THREADS);
    //TAU_METADATA("TAU_MAX_THREADS", tmpstr);


    /*** Options that can be used with Scalasca and VampirTrace ***/
    tmp = getconf("TAU_LITE");
    if (parse_bool(tmp,env_tau_lite)) {
      TAU_VERBOSE("TAU: LITE measurement enabled\n");
      TAU_METADATA("TAU_LITE", "on");
      env_tau_lite = 1;
    }


    const char *interval = getconf("TAU_INTERRUPT_INTERVAL");
    env_interval = TAU_INTERRUPT_INTERVAL_DEFAULT;;
    if (interval) {
      int interval_value = 0;
      sscanf(interval,"%d",&interval_value);
      env_interval = interval_value;
      sprintf(tmpstr, "%d", env_interval);
      TAU_SET_INTERRUPT_INTERVAL(interval_value);
      TAU_METADATA("TAU_INTERRUPT_INTERVAL", tmpstr);
    }

    tmp = getconf("TAU_TRACK_POWER");
    if (parse_bool(tmp, env_track_power)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: Power tracking disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_POWER", "disabled (disabled memory management)");
#else
      TAU_VERBOSE("TAU: Power tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_POWER", "on");
      TauEnableTrackingPower();
#endif
    }

    tmp = getconf("TAU_TRACK_LOAD");
    if (parse_bool(tmp, env_track_load)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: system load tracking disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_LOAD", "disabled (disabled memory management)");
#else
      TAU_VERBOSE("TAU: system load tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_LOAD", "on");
      TauEnableTrackingLoad();
#endif
    }

#ifdef TAU_MPI_T
    if ((env_mpi_t_comm_metric_values = getconf("TAU_MPI_T_COMM_METRIC_VALUES")) == NULL) {
      env_mpi_t_comm_metric_values = "";   /* Not set */
      env_track_mpi_t_comm_metric_values=0;
    } else {
      env_track_mpi_t_comm_metric_values=1;
      TAU_VERBOSE("TAU: TAU_MPI_T_COMM_METRIC_VALUES is \"%s\"\n", env_mpi_t_comm_metric_values);
      TAU_METADATA("TAU_MPI_T_COMM_METRIC_VALUES", env_mpi_t_comm_metric_values);
    }

    tmp = getconf("TAU_TRACK_MPI_T_PVARS");
    if (parse_bool(tmp, env_track_mpi_t_pvars)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: MPI_T PVARS tracking disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_MPI_T_PVARS", "disabled (disabled memory management)");
#else
      env_track_mpi_t_pvars = 1;
      TAU_VERBOSE("TAU: MPI_T PVARS tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_MPI_T_PVARS", "on");
      TAU_VERBOSE("TAU: Checking for performance variables from MPI_T\n");
#endif
    } else {
      TAU_METADATA("TAU_TRACK_MPI_T_PVARS", "off");
    }

    tmp = getconf("TAU_MPI_T_ENABLE_USER_TUNING_POLICY");
    if (parse_bool(tmp, env_mpi_t_enable_user_tuning_policy)) {
      env_mpi_t_enable_user_tuning_policy = 1;
      TAU_VERBOSE("TAU: MPI_T enable user tuning policy enabled\n");
      TAU_METADATA("TAU_MPI_T_ENABLE_USER_TUNING_POLICY", "on");
      TAU_VERBOSE("TAU: Enabling user CVAR tuning policy\n");
    } else {
      TAU_METADATA("TAU_MPI_T_ENABLE_USER_TUNING_POLICY", "off");
    }

#endif /* TAU_MPI_T */

#if defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
    tmp = getconf("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY");
    if (parse_bool(tmp, env_ompt_resolve_address_eagerly)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: OMPT resolving addresses eagerly disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "disabled (disabled memory management)");
#else
      env_ompt_resolve_address_eagerly = 1;
      TAU_VERBOSE("TAU: OMPT resolving addresses eagerly Enabled\n");
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "on");
      TAU_VERBOSE("TAU: Resolving OMPT addresses eagerly\n");
#endif /*  TAU_DISABLE_MEM_MANAGER */
    } else {
      env_ompt_resolve_address_eagerly = 0;
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "off");
    }

    env_ompt_support_level = 0; // Basic OMPT support is the default
    const char *omptSupportLevel = getconf("TAU_OMPT_SUPPORT_LEVEL");
    if (omptSupportLevel != NULL && 0 == strcasecmp(omptSupportLevel, TAU_OMPT_SUPPORT_LEVEL_BASIC)) {
      env_ompt_support_level = 0;
      TAU_VERBOSE("TAU: OMPT support will be basic - only required events supported\n");
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "basic");
    } else if (omptSupportLevel != NULL && 0 == strcasecmp(omptSupportLevel, TAU_OMPT_SUPPORT_LEVEL_LOWOVERHEAD)) {
      env_ompt_support_level = 1;
      TAU_VERBOSE("TAU: OMPT support will be for all required events along with optional low overhead events\n");
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "lowoverhead");
    } else if (omptSupportLevel != NULL && 0 == strcasecmp(omptSupportLevel, TAU_OMPT_SUPPORT_LEVEL_FULL)) {
      env_ompt_support_level = 2;
      TAU_VERBOSE("TAU: OMPT support will be full - all events will be supported\n");
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "full");
    }
#endif /* defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */

    tmp = getconf("TAU_OMPT_FORCE_FINALIZE");
    if (parse_bool(tmp, env_ompt_force_finalize)) {
      TAU_VERBOSE("TAU: OMPT Finalize Tool Enabled\n");
      env_ompt_force_finalize = 1;
    } else {
      TAU_VERBOSE("TAU: OMPT Finalize Tool Disabled\n");
      env_ompt_force_finalize = 0;
    }

    tmp = getconf("TAU_TRACK_HEAP");
    if (parse_bool(tmp, env_track_memory_heap)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: Entry/Exit Memory tracking disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_HEAP", "disabled (memory management disabled)");
#else
      TAU_VERBOSE("TAU: Entry/Exit Memory tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_HEAP", "on");
      env_track_memory_heap = 1;
#endif
    } else {
      TAU_METADATA("TAU_TRACK_HEAP", "off");
      env_track_memory_heap = 0;
    }

    tmp = getconf("TAU_SHOW_MEMORY_FUNCTIONS");
    if (parse_bool(tmp, env_show_memory_functions)) {
      TAU_VERBOSE("TAU: Show memory functions Enabled\n");
      TAU_METADATA("TAU_SHOW_MEMORY_FUNCTIONS", "on");
      env_show_memory_functions = 1;
    } else {
      TAU_METADATA("TAU_SHOW_MEMORY_FUNCTIONS", "off");
      env_show_memory_functions = 0;
    }

    tmp = getconf("TAU_TRACK_MEMORY_FOOTPRINT");
    if (parse_bool(tmp, env_track_memory_footprint)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: TAU_TRACK_MEMORY_FOOTPRINT VmRSS and VmHWM tracking disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_MEMORY_FOOTPRINT", "disabled (disabled memory management)");
#else
      TAU_VERBOSE("TAU: TAU_TRACK_MEMORY_FOOTPRINT VmRSS and VmHWM tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_MEMORY_FOOTPRINT", "on");

	  env_track_memory_footprint = 1;
      TAU_TRACK_MEMORY_FOOTPRINT();
#endif
    } else {
      TAU_METADATA("TAU_TRACK_MEMORY_FOOTPRINT", "off");
      env_track_memory_footprint = 0;
    }

    tmp = getconf("TAU_TRACK_HEADROOM");
    if (parse_bool(tmp, env_track_memory_headroom)) {
    /*
      TAU_VERBOSE("TAU: Entry/Exit Headroom tracking Enabled\n");
      TAU_METADATA("TAU_TRACK_HEADROOM", "on");
      env_track_memory_headroom = 1;
      */
      TAU_VERBOSE("NOTE: Entry/Exit Headroom tracking is permanently disabled!\n");
      TAU_METADATA("TAU_TRACK_HEADROOM", "off");
      env_track_memory_headroom = 0;
    } else {
      TAU_METADATA("TAU_TRACK_HEADROOM", "off");
      env_track_memory_headroom = 0;
    }

    tmp = getconf("TAU_TRACK_MEMORY_LEAKS");
    if (parse_bool(tmp, env_track_memory_leaks)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: Memory tracking disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_MEMORY_LEAKS", "disabled (disabled memory management)");
#else
      TAU_VERBOSE("TAU: Memory tracking enabled\n");
      TAU_METADATA("TAU_TRACK_MEMORY_LEAKS", "on");
      env_track_memory_leaks = 1;
#endif
    } else {
      TAU_METADATA("TAU_TRACK_MEMORY_LEAKS", "off");
      env_track_memory_leaks = 0;
    }

    tmp = getconf("TAU_PAPI_MULTIPLEXING");
    if (parse_bool(tmp, env_papi_multiplexing)) {
      TAU_VERBOSE("TAU: PAPI multiplexing Enabled\n");
      TAU_METADATA("TAU_PAPI_MULTIPLEXING", "on");
      env_papi_multiplexing = 1;
    } else {
      TAU_METADATA("TAU_PAPI_MULTIPLEXING", "off");
      env_papi_multiplexing = 0;
    }

    tmp = getconf("TAU_REGION_ADDRESSES");
    if (parse_bool(tmp, env_region_addresses)) {
      TAU_VERBOSE("TAU: Region addresses Enabled\n");
      TAU_METADATA("TAU_REGION_ADDRESSES", "on");
      env_region_addresses = 1;
    } else {
      TAU_METADATA("TAU_REGION_ADDRESSES", "off");
      env_region_addresses = 0;
    }

    tmp = getconf("TAU_RECYCLE_THREADS");
    if (parse_bool(tmp, TAU_RECYCLE_THREADS_DEFAULT)) {
      TAU_VERBOSE("TAU: Region addresses Enabled\n");
      TAU_METADATA("TAU_RECYCLE_THREADS", "on");
      env_recycle_threads = 1;
    } else {
      TAU_METADATA("TAU_RECYCLE_THREADS", "off");
      env_recycle_threads = 0;
    }

    // Setting TAU_MEMDBG_PROTECT_{ABOVE,BELOW,FREE} enables memory debugging.

    tmp = getconf("TAU_MEMDBG_PROTECT_ABOVE");
    env_memdbg_protect_above = parse_bool(tmp, env_memdbg_protect_above);
    if(env_memdbg_protect_above) {
      env_memdbg = 1;
      TAU_VERBOSE("TAU: Bounds checking enabled on array end\n");
      TAU_METADATA("TAU_MEMDBG_PROTECT_ABOVE", "on");
    } else {
      TAU_METADATA("TAU_MEMDBG_PROTECT_ABOVE", "off");
    }

    tmp = getconf("TAU_MEMDBG_PROTECT_BELOW");
    env_memdbg_protect_below = parse_bool(tmp, env_memdbg_protect_below);
    if(env_memdbg_protect_below) {
      env_memdbg = 1;
      TAU_VERBOSE("TAU: Bounds checking enabled on array beginning\n");
      TAU_METADATA("TAU_MEMDBG_PROTECT_BELOW", "on");
    } else {
      TAU_METADATA("TAU_MEMDBG_PROTECT_BELOW", "off");
    }

    tmp = getconf("TAU_MEMDBG_PROTECT_FREE");
    env_memdbg_protect_free = parse_bool(tmp, env_memdbg_protect_free);
    if(env_memdbg_protect_free) {
      env_memdbg = 1;
      TAU_VERBOSE("TAU: Checking for free memory reuse errors\n");
      TAU_METADATA("TAU_MEMDBG_PROTECT_FREE", "on");
    } else {
      TAU_METADATA("TAU_MEMDBG_PROTECT_FREE", "off");
    }

    if(env_memdbg) {

      size_t page_size = Tau_page_size();
      sprintf(tmpstr, "%ld", page_size);
      TAU_METADATA("Virtual Memory Page Size", tmpstr);

      env_track_signals = 1;

      tmp = getconf("TAU_MEMDBG_PROTECT_GAP");
      env_memdbg_protect_gap = parse_bool(tmp, env_memdbg_protect_gap);
      if(env_memdbg_protect_gap) {
        TAU_VERBOSE("TAU: Bounds checking enabled in memory gap\n");
        TAU_METADATA("TAU_MEMDBG_PROTECT_GAP", "on");
      } else {
        TAU_METADATA("TAU_MEMDBG_PROTECT_GAP", "off");
      }

      tmp = getconf("TAU_MEMDBG_FILL_GAP");
      if (tmp) {
        env_memdbg_fill_gap = 1;
        env_memdbg_fill_gap_value = parse_int(tmp, env_memdbg_fill_gap_value);
        TAU_VERBOSE("TAU: Initializing memory gap to %d\n", tmp);
        TAU_METADATA("TAU_MEMDBG_FILL_GAP", tmp);
      }

      tmp = getconf("TAU_MEMDBG_ALLOC_MIN");
      if (tmp) {
        env_memdbg_alloc_min = 1;
        env_memdbg_alloc_min_value = atol(tmp);
        TAU_VERBOSE("TAU: Minimum allocation size for bounds checking is %d\n", env_memdbg_alloc_min_value);
        TAU_METADATA("TAU_MEMDBG_ALLOC_MIN", tmp);
      }

      tmp = getconf("TAU_MEMDBG_ALLOC_MAX");
      if (tmp) {
        env_memdbg_alloc_max = 1;
        env_memdbg_alloc_max_value = atol(tmp);
        TAU_VERBOSE("TAU: Maximum allocation size for bounds checking is %d\n", env_memdbg_alloc_max_value);
        TAU_METADATA("TAU_MEMDBG_ALLOC_MAX", tmp);
      }

      tmp = getconf("TAU_MEMDBG_OVERHEAD");
      if (tmp) {
        env_memdbg_overhead = 1;
        env_memdbg_overhead_value = atol(tmp);
        TAU_VERBOSE("TAU: Maximum bounds checking overhead is %d\n", env_memdbg_overhead_value);
        TAU_METADATA("TAU_MEMDBG_OVERHEAD", tmp);
      }

      tmp = getconf("TAU_MEMDBG_ALIGNMENT");
      if (tmp) {
        env_memdbg_alignment = parse_int(tmp, env_memdbg_alignment);
      }
      if ((int)env_memdbg_alignment != ((int)env_memdbg_alignment & -(int)env_memdbg_alignment)) {
        TAU_VERBOSE("TAU: ERROR - Memory debugging alignment is not a power of two: %ld\n", env_memdbg_alignment);
      } else {
        TAU_VERBOSE("TAU: Memory debugging alignment: %ld\n", env_memdbg_alignment);
      }
      sprintf(tmpstr, "%ld", env_memdbg_alignment);
      TAU_METADATA("TAU_MEMDBG_ALIGNMENT", tmpstr);

      tmp = getconf("TAU_MEMDBG_ZERO_MALLOC");
      env_memdbg_zero_malloc = parse_bool(tmp, env_memdbg_zero_malloc);
      if(env_memdbg_zero_malloc) {
        TAU_VERBOSE("TAU: Zero-size malloc will be accepted\n");
        TAU_METADATA("TAU_MEMDBG_ZERO_MALLOC", "on");
      } else {
        TAU_VERBOSE("TAU: Zero-size malloc will be flagged as error\n");
        TAU_METADATA("TAU_MEMDBG_ZERO_MALLOC", "off");
      }

      tmp = getconf("TAU_MEMDBG_ATTEMPT_CONTINUE");
      env_memdbg_attempt_continue = parse_bool(tmp, env_memdbg_attempt_continue);
      if(env_memdbg_attempt_continue) {
        TAU_VERBOSE("TAU: Attempt to resume execution after memory error\n");
        TAU_METADATA("TAU_MEMDBG_ATTEMPT_CONTINUE", "on");
      } else {
        TAU_VERBOSE("TAU: The first memory error will halt execution and generate a backtrace\n");
        TAU_METADATA("TAU_MEMDBG_ATTEMPT_CONTINUE", "off");
      }

    } // if (env_memdbg)

    tmp = getconf("TAU_PTHREAD_STACK_SIZE");
    if (tmp) {
      env_pthread_stack_size = atoi(tmp);
      if (env_pthread_stack_size) {
        TAU_VERBOSE("TAU: pthread stack size = %d\n", env_pthread_stack_size);
        TAU_METADATA("TAU_PTHREAD_STACK_SIZE", tmp);
      }
    }

    tmp = getconf("TAU_TRACK_IO_PARAMS");
    if (parse_bool(tmp, env_track_io_params)) {
      TAU_VERBOSE("TAU: POSIX I/O wrapper parameter tracking enabled\n");
      TAU_METADATA("TAU_TRACK_IO_PARAMS", "on");
      env_track_io_params = 1;
    } else {
      TAU_METADATA("TAU_TRACK_IO_PARAMS", "off");
      env_track_io_params = 0;
    }

    tmp = getconf("TAU_TRACK_SIGNALS");
    if (parse_bool(tmp, env_track_signals)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      TAU_VERBOSE("TAU: Tracking SIGNALS disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_TRACK_SIGNALS", "disabled (disabled memory management)");
#else
      TAU_VERBOSE("TAU: Tracking SIGNALS enabled\n");
      TAU_METADATA("TAU_TRACK_SIGNALS", "on");
      env_track_signals = 1;
      tmp = getconf("TAU_SIGNALS_GDB");
      if (parse_bool(tmp, env_signals_gdb)) {
        TAU_VERBOSE("TAU: SIGNALS GDB output enabled\n");
        TAU_METADATA("TAU_SIGNALS_GDB", "on");
        env_signals_gdb = 1;
      } else {
        TAU_METADATA("TAU_SIGNALS_GDB", "off");
        env_signals_gdb = 0;
      }
      tmp = getconf("TAU_ECHO_BACKTRACE");
      if (parse_bool(tmp, env_echo_backtrace)) {
        TAU_VERBOSE("TAU: Backtrace will be echoed to stderr\n");
        TAU_METADATA("TAU_ECHO_BACKTRACE", "on");
        env_echo_backtrace = 1;
      } else {
        TAU_METADATA("TAU_ECHO_BACKTRACE", "off");
        env_echo_backtrace = 0;
      }
#endif
    } else {
      TAU_METADATA("TAU_TRACK_SIGNALS", "off");
      TAU_METADATA("TAU_SIGNALS_GDB", "off");
      env_track_signals = 0;
    }

    tmp = getconf("TAU_IBM_BG_HWP_COUNTERS");
    if (parse_bool(tmp, env_ibm_bg_hwp_counters)) {
      TAU_VERBOSE("TAU: IBM UPC HWP counter data collection enabled\n");
      TAU_METADATA("TAU_IBM_BG_HWP_COUNTERS", "on");
      env_ibm_bg_hwp_counters = 1;
    } else {
      TAU_METADATA("TAU_IBM_BG_HWP_COUNTERS", "off");
      env_ibm_bg_hwp_counters = 0;
    }



    /*** Options that can be used with Scalasca and VampirTrace need to go above this line ***/
#ifdef TAU_EPILOG
    TAU_VERBOSE("TAU: Epilog/Scalasca active! (TAU measurement disabled)\n");
    return;
#endif

#ifdef TAU_VAMPIRTRACE
    TAU_VERBOSE("[%d] TAU: VampirTrace active! (TAU measurement disabled)\n", RtsLayer::getPid());
    return;
#endif

#ifdef TAU_SCOREP
    TAU_VERBOSE("[%d] TAU: SCOREP active! (TAU measurement disabled)\n", RtsLayer::getPid());
    //return;
    //if we return here, the other TAU variables such as TAU_SELECT_FILE are not read!
#endif

    if ((env_profile_prefix = getconf("TAU_PROFILE_PREFIX")) == NULL) {
      TAU_VERBOSE("TAU: PROFILE PREFIX is \"%s\"\n", env_profile_prefix);
    }

    if ((env_profiledir = getconf("PROFILEDIR")) == NULL) {
      env_profiledir = ".";   /* current directory */
#ifdef TAU_GPI
      // if exe is /usr/local/foo, this will return /usr/local where profiles
      // may be stored if PROFILEDIR is not specified
      char const * cwd = Tau_get_cwd_of_exe();
      if (cwd) {
        env_profiledir = strdup(cwd);
        TAU_VERBOSE("ENV_PROFILEDIR = %s\n", env_profiledir);
      }
#endif /* TAU_GPI */
    }
    TAU_VERBOSE("TAU: PROFILEDIR is \"%s\"\n", env_profiledir);

    if ((env_tracedir = getconf("TRACEDIR")) == NULL) {
      env_tracedir = ".";   /* current directory */
#ifdef TAU_GPI
      // if exe is /usr/local/foo, this will return /usr/local where profiles
      // may be stored if PROFILEDIR is not specified
      char const * cwd = Tau_get_cwd_of_exe();
      if (cwd) {
        env_tracedir = strdup(cwd);
        TAU_VERBOSE("ENV_TRACEDIR = %s\n", env_tracedir);
      }
#endif /* TAU_GPI */
    }
    TAU_VERBOSE("TAU: TRACEDIR is \"%s\"\n", env_tracedir);

    int profiling_default = TAU_PROFILING_DEFAULT;
    /* tracing */
    tmp = getconf("TAU_TRACE");
    if (parse_bool(tmp, TAU_TRACING_DEFAULT)) {
      env_tracing = 1;
      env_thread_per_gpu_stream = 1;
      env_track_message = 1;
      profiling_default = 0;
      TAU_VERBOSE("TAU: Tracing Enabled\n");
      TAU_METADATA("TAU_TRACE", "on");
      if (TauEnv_get_callsite_depth() > 1) {
        printf("WARNING: TAU_CALLSITE_DEPTH > 1 is not supported with tracing.\n");
      }
    } else {
      env_tracing = 0;
      env_thread_per_gpu_stream = 0;
      env_track_message = TAU_TRACK_MESSAGE_DEFAULT;
      TAU_VERBOSE("TAU: Tracing Disabled\n");
      TAU_METADATA("TAU_TRACE", "off");
    }
    /* trace format */
    tmp = getconf("TAU_TRACE_FORMAT");
    if(tmp != NULL) {
      if(strcasecmp(tmp, "otf2") == 0) {
#ifdef TAU_OTF2
        env_trace_format = TAU_TRACE_FORMAT_OTF2;
#else
        fprintf(stderr, "TAU: Warning: requested OTF2 trace but TAU built without OTF2, using default instead.\n");
#endif
      } else if(strcasecmp(tmp, "tau") == 0) {
        env_trace_format = TAU_TRACE_FORMAT_TAU;
      } else {
        fprintf(stderr, "TAU: Warning: unrecognized trace format %s, using default instead.\n", tmp);
      }
    }
    if(env_trace_format == TAU_TRACE_FORMAT_TAU) {
      TAU_VERBOSE("TAU: Trace format is tau\n");
      TAU_METADATA("TAU_TRACE_FORMAT", "tau")
    } else if(env_trace_format == TAU_TRACE_FORMAT_OTF2) {
      TAU_VERBOSE("TAU: Trace format is otf2\n");
      TAU_METADATA("TAU_TRACE_FORMAT", "otf2")
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

    tmp = getconf("TAU_CURRENT_TIMER_EXIT_PARAMS");
    if (parse_bool(tmp, profiling_default)) {
      env_current_timer_exit_params = 1;
      TAU_VERBOSE("TAU: Current Timer Exit Enabled\n");
      TAU_METADATA("TAU_CURRENT_TIMER_EXIT_PARAMS", "on");
    } else {
      env_current_timer_exit_params = 0;
      TAU_VERBOSE("TAU: Current Timer Exit Disabled\n");
      TAU_METADATA("TAU_CURRENT_TIMER_EXIT_PARAMS", "off");
    }

    /* Switched this from env_profiling to !env_tracing.
     * If we are using alternative outputs (ADIOS2, SQLITE, SOS)
     * we want to disable profile wrting at the end of execution
     * but we don't want to disable callpaths.
     */
    if (!env_tracing) {
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

      /* thread context */
      tmp = getconf("TAU_ENABLE_THREAD_CONTEXT");
      if (parse_bool(tmp, TAU_ENABLE_THREAD_CONTEXT_DEFAULT)) {
        env_thread_context = 1;
        TAU_VERBOSE("TAU: Thread Context Enabled\n");
        TAU_METADATA("TAU_ENABLE_THREAD_CONTEXT", "on");
      } else {
        env_thread_context = 0;
        TAU_VERBOSE("TAU: Thread Context Disabled\n");
        TAU_METADATA("TAU_ENABLE_THREAD_CONTEXT", "off");
      }

      /* compensate */
      tmp = getconf("TAU_COMPENSATE");
      if (parse_bool(tmp, TAU_COMPENSATE_DEFAULT)) {
        env_compensate = 1;
        TAU_VERBOSE("TAU: Overhead Compensation Enabled\n");
        TAU_METADATA("TAU_COMPENSATE", "on");
      } else {
        env_compensate = 0;
        TAU_VERBOSE("TAU: Overhead Compensation Disabled\n");
        TAU_METADATA("TAU_COMPENSATE", "off");
      }

      tmp = getconf("TAU_THREAD_PER_GPU_STREAM");
      if (parse_bool(tmp, 0)) {
        env_thread_per_gpu_stream = 1;
        TAU_VERBOSE("TAU: Enabling new thread for every GPU stream\n");
        TAU_METADATA("TAU_THREAD_PER_GPU_STREAM", "on");
      }
    }

    tmp = getconf("TAU_CALLSITE");
    if (parse_bool(tmp, TAU_CALLSITE_DEFAULT)) {
      env_callsite = 1;
      TAU_VERBOSE("TAU: Callsite Discovery via Unwinding Enabled\n");
      TAU_METADATA("TAU_CALLSITE", "on");
    }

    const char *callsiteDepth = getconf("TAU_CALLSITE_DEPTH");
    env_callsite_depth = TAU_CALLSITE_DEPTH_DEFAULT;
    if (callsiteDepth) {
      env_callsite_depth = atoi(callsiteDepth);
      if (env_callsite_depth < 0) {
        env_callsite_depth = TAU_CALLSITE_DEPTH_DEFAULT;
      }
    }
    TAU_VERBOSE("TAU: Callsite Depth Limit = %d\n", env_callsite_depth);
    sprintf(tmpstr, "%d", env_callsite_depth);
    TAU_METADATA("TAU_CALLSITE_DEPTH", tmpstr);

    const char *callsiteOffset = getconf("TAU_CALLSITE_OFFSET");
    env_callsite_offset = TAU_CALLSITE_OFFSET_DEFAULT;
    if (callsiteOffset) {
      env_callsite_offset = atoi(callsiteOffset);
      if (env_callsite_offset < 0) {
        env_callsite_offset = TAU_CALLSITE_OFFSET_DEFAULT;
      }
      sprintf(tmpstr, "%d", env_callsite_offset);
      TAU_METADATA("TAU_CALLSITE_OFFSET", tmpstr);
      TAU_VERBOSE("TAU: Callsite Offset = %d\n", env_callsite_offset);
    }


#if (defined(TAU_MPI) || defined(TAU_SHMEM) || defined(TAU_DMAPP) || defined(TAU_UPC) || defined(TAU_GPI))
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


    const char *max_records = getconf("TAU_MAX_RECORDS");
    env_max_records = TAU_MAX_RECORDS;
    if (max_records) {
      env_max_records = strtod(max_records, 0);
      TAU_VERBOSE("TAU: TAU_MAX_RECORDS = %g\n", env_max_records);
    }



#ifdef TAU_MPI
    tmp = getconf("TAU_SET_NODE");
    if (tmp) {
      int node_id = 0;
      sscanf(tmp,"%d",&node_id);
      env_node_set=node_id;
      TAU_VERBOSE("TAU: Setting node value forcibly to (TAU_SET_NODE): %d\n", node_id);
      TAU_PROFILE_SET_NODE(node_id);
      Tau_set_usesMPI(1);
      TAU_METADATA("TAU_SET_NODE", tmp);
    }
#endif /* TAU_MPI */


#endif /* TAU_MPI || TAU_SHMEM || TAU_DMAPP || TAU_UPC || TAU_GPI */

    /* clock synchronization */
    if (env_tracing == 0) {
      env_synchronize_clocks = 0;
    } else {
#ifndef TAU_MPI
      /* If there is no MPI, there can't be any sync, so forget it */
      env_synchronize_clocks = 0;
      TAU_VERBOSE("TAU: Clock Synchronization Disabled (MPI not available)\n");
      TAU_METADATA("TAU_SYNCHRONIZE_CLOCKS", "off");
#else
      tmp = getconf("TAU_SYNCHRONIZE_CLOCKS");
      if (parse_bool(tmp, TAU_SYNCHRONIZE_CLOCKS_DEFAULT)) {
        env_synchronize_clocks = 1;
        TAU_VERBOSE("TAU: Clock Synchronization Enabled\n");
        TAU_METADATA("TAU_SYNCHRONIZE_CLOCKS", "on");
      } else {
        env_synchronize_clocks = 0;
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

    /* Throttle */
    tmp = getconf("TAU_DISABLE_INSTRUMENTATION");
    if (parse_bool(tmp, TAU_DISABLE_INSTRUMENTATION_DEFAULT)) {
      env_disable_instrumentation = 1;
      TAU_DISABLE_INSTRUMENTATION();
      TAU_VERBOSE("TAU: Instrumentation Disabled\n");
      TAU_METADATA("TAU_DISABLE_INSTRUMENTATION", "on");
    } else { /* default: instrumentation is enabled */
      env_disable_instrumentation = 0;
    }

    const char *percall = getconf("TAU_THROTTLE_PERCALL");
    env_throttle_percall = TAU_THROTTLE_PERCALL_DEFAULT;
    if (percall) {
      env_throttle_percall = strtod(percall, 0);
    }

    const char *l0_api_tracing = getconf("ZE_ENABLE_API_TRACING");
    if (l0_api_tracing) {
      env_l0_api_tracing =1 ; /* Intel OneAPI Level Zero API TRACING */
    }

    const char *evt_threshold = getconf("TAU_EVENT_THRESHOLD");
    env_evt_threshold = TAU_EVENT_THRESHOLD_DEFAULT;
    if (evt_threshold) {
      double evt_value = 0.0;
      sscanf(evt_threshold,"%lg",&evt_value);
      env_evt_threshold = evt_value;
      TAU_METADATA("TAU_EVENT_THRESHOLD", evt_threshold);
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

    const char *sigusr1Action = getconf("TAU_SIGUSR1_ACTION");
    if (sigusr1Action != NULL && 0 == strcasecmp(sigusr1Action, "backtraces")) {
      env_sigusr1_action = TAU_ACTION_DUMP_BACKTRACES;
      TAU_VERBOSE("TAU: SIGUSR1 Action: dump backtraces\n");
    } else if (sigusr1Action != NULL && 0 == strcasecmp(sigusr1Action, "callpaths")) {
      env_sigusr1_action = TAU_ACTION_DUMP_CALLPATHS;
      TAU_VERBOSE("TAU: SIGUSR1 Action: dump callpaths\n");
    } else {
      TAU_VERBOSE("TAU: SIGUSR1 Action: dump profiles\n");
	}

    const char *profileFormat = getconf("TAU_PROFILE_FORMAT");
    if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "snapshot")) {
      env_profile_format = TAU_FORMAT_SNAPSHOT;
      TAU_VERBOSE("TAU: Output Format: snapshot\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "snapshot");
    } else if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "merged")) {
//#ifdef TAU_MPI
      env_profile_format = TAU_FORMAT_MERGED;
      TAU_VERBOSE("TAU: Output Format: merged\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "merged");
//#else
      //env_profile_format = TAU_FORMAT_PROFILE;
      //TAU_VERBOSE("TAU: Output Format: merged format not supported without MPI, reverting to profile\n");
      //TAU_METADATA("TAU_PROFILE_FORMAT", "profile");
//#endif /* TAU_MPI */
    } else if (profileFormat != NULL && 0 == strcasecmp(profileFormat, "none")) {
      env_profile_format = TAU_FORMAT_NONE;
      TAU_VERBOSE("TAU: Output Format: none\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "none");
    } else {
      env_profile_format = TAU_FORMAT_PROFILE;
      TAU_VERBOSE("TAU: Output Format: profile\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "profile");
    }

    tmp = getconf("TAU_ANONYMIZE");
    if (parse_bool(tmp,env_tau_anonymize)) {
      TAU_VERBOSE("TAU: Anonymize enabled\n");
      TAU_METADATA("TAU_ANONYMIZE", "on");
      env_tau_anonymize = 1;
      env_profile_format = TAU_FORMAT_MERGED;
      TAU_VERBOSE("TAU: Output Format: merged\n");
      TAU_METADATA("TAU_PROFILE_FORMAT", "merged");
    }


    tmp = getconf("TAU_SUMMARY");
    if (parse_bool(tmp, env_summary_only)) {
#ifdef TAU_MPI
			if (env_profile_format == TAU_FORMAT_MERGED) {
				TAU_VERBOSE("TAU: Generating only summary data: TAU_SUMMARY enabled\n");
				TAU_METADATA("TAU_SUMMARY", "on");
				env_summary_only = 1;
			} else {
      	TAU_VERBOSE("TAU: Summary requires merged format, reverting non-summary profiling.\n");
				TAU_METADATA("TAU_SUMMARY", "off");
				env_summary_only = 0;
			}
#else
      TAU_VERBOSE("TAU: Summary requires merged format, which is not supported without MPI, reverting non-summary profiling.\n");
      TAU_METADATA("TAU_SUMMARY", "off");
      env_summary_only = 0;
#endif /* TAU_MPI */
		}

    if ((env_metrics = getconf("TAU_METRICS")) == NULL) {
      env_metrics = "";   /* default to 'time' */
      TAU_VERBOSE("TAU: METRICS is not set\n", env_metrics);
    } else {
      TAU_VERBOSE("TAU: METRICS is \"%s\"\n", env_metrics);
    }

    if ((env_cvar_metrics = getconf("TAU_MPI_T_CVAR_METRICS")) == NULL) {
      env_cvar_metrics = "";   /* default to 'time' */
      TAU_VERBOSE("TAU: MPI_T_CVAR_METRICS is not set\n", env_cvar_metrics);
    } else {
      TAU_VERBOSE("TAU: MPI_T_CVAR_METRICS is \"%s\"\n", env_cvar_metrics);
    }

    if ((env_cvar_values = getconf("TAU_MPI_T_CVAR_VALUES")) == NULL) {
      env_cvar_values = "";   /* default to 'time' */
      TAU_VERBOSE("TAU: MPI_T_CVAR_VALUES is not set\n", env_cvar_values);
    } else {
      TAU_VERBOSE("TAU: MPI_T_CVAR_VALUES is \"%s\"\n", env_cvar_values);
    }

    if ((env_plugins_path = getconf("TAU_PLUGINS_PATH")) == NULL) {
      env_plugins_path = NULL;
      TAU_VERBOSE("TAU: TAU_PLUGINS_PATH is not set\n", env_plugins_path);
    } else {
      TAU_VERBOSE("TAU: TAU_PLUGINS_PATH is \"%s\"\n", env_plugins_path);
    }

    if ((env_plugins = getconf("TAU_PLUGINS")) == NULL) {
      env_plugins = NULL;
      TAU_VERBOSE("TAU: TAU_PLUGINS is not set\n", env_plugins);
    } else {
      TAU_VERBOSE("TAU: TAU_PLUGINS is \"%s\"\n", env_plugins);
    }

    if((env_select_file = getconf("TAU_SELECT_FILE")) == NULL) {
      env_select_file = NULL;
    } else {
      if ((env_plugins == NULL) && (env_plugins_path == NULL)) {
        TAU_VERBOSE("TAU: TAU_SELECT_FILE is set to %s when TAU plugins are not initialized\n", env_select_file);
          env_plugins_path=strdup(TAU_LIB_DIR);
          TAU_VERBOSE("TAU: TAU_PLUGINS_PATH is now %s, TAU_LIB_DIR=%s\n", env_plugins_path, TAU_LIB_DIR);
          //sprintf(env_plugins,"libTAU-filter-plugin.so(%s)", env_select_file);
          char *plugins = (char *) malloc(1024);
	  char *filename = strdup(env_select_file);
          sprintf(plugins, "libTAU-filter-plugin.so(%s)", filename);
          env_plugins = plugins;
          TAU_VERBOSE("TAU: TAU plugin is now %s\n", env_plugins);
	  TAU_METADATA("TAU_SELECT_FILE", filename);
      } else {
        TAU_VERBOSE("TAU: Ignoring TAU_SELECT_FILE because TAU_PLUGINS and/or TAU_PLUGINS_PATH is set.\nPlease use export TAU_PLUGINS_PATH=%s and export TAU_PLUGINS=\"libTAU-filter-plugin.so(%s)\"\n",
			strdup(TAU_LIB_DIR),
			strdup(env_select_file));
      }
    }

    // Check for these paths after TAU_SELECT_FILE sets them
    if (env_plugins_path != NULL && env_plugins != NULL) {
        env_plugins_enabled = 1;
    }

    tmp = getconf("TAU_OPENMP_RUNTIME");
    if (parse_bool(tmp, TAU_OPENMP_RUNTIME_DEFAULT)) {
      env_openmp_runtime_enabled = 1;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Enabled\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME", "on");
    } else {
      env_openmp_runtime_enabled = 0;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Disabled\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME", "off");
    }

    tmp = getconf("TAU_OPENMP_RUNTIME_STATES");
    if (parse_bool(tmp, TAU_OPENMP_RUNTIME_STATES_DEFAULT)) {
      env_openmp_runtime_states_enabled = 1;
      TAU_VERBOSE("TAU: OpenMP Runtime Support States Enabled\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_STATES", "on");
    } else {
      env_openmp_runtime_states_enabled = 0;
      TAU_VERBOSE("TAU: OpenMP Runtime Support States Disabled\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_STATES", "off");
    }

    tmp = getconf("TAU_OPENMP_RUNTIME_EVENTS");
    if (parse_bool(tmp, TAU_OPENMP_RUNTIME_EVENTS_DEFAULT)) {
      env_openmp_runtime_events_enabled = 1;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Events Enabled\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_EVENTS", "on");
    } else {
      env_openmp_runtime_events_enabled = 0;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Events Disabled\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_EVENTS", "off");
    }

    env_openmp_runtime_context = 2; // the region is the default
    const char *apiContext = getconf("TAU_OPENMP_RUNTIME_CONTEXT");
    if (apiContext != NULL && 0 == strcasecmp(apiContext, TAU_OPENMP_RUNTIME_CONTEXT_TIMER)) {
      env_openmp_runtime_context = 1;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Context will be the current timer\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_CONTEXT", "timer");
    } else if (apiContext != NULL && 0 == strcasecmp(apiContext, TAU_OPENMP_RUNTIME_CONTEXT_REGION)) {
      env_openmp_runtime_context = 2;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Context will be the current parallel region\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_CONTEXT", "region");
    } else if (apiContext != NULL && 0 == strcasecmp(apiContext, TAU_OPENMP_RUNTIME_CONTEXT_NONE)) {
      env_openmp_runtime_context = 0;
      TAU_VERBOSE("TAU: OpenMP Runtime Support Context none\n");
      TAU_METADATA("TAU_OPENMP_RUNTIME_CONTEXT", "none");
    }

    tmp = getconf("TAU_MEASURE_TAU");
    if (parse_bool(tmp, TAU_EBS_DEFAULT_TAU)) {
      env_ebs_enabled = 1; // enable samping too?
      env_ebs_enabled_tau = 1;
      TAU_VERBOSE("TAU: Sampling TAU overhead\n");
      TAU_METADATA("TAU_SAMPLING", "on");
      TAU_METADATA("TAU_MEASURE_TAU", "on");
    } else {
      env_ebs_enabled_tau = 0;
      TAU_VERBOSE("TAU: Not sampling TAU overhead\n");
      TAU_METADATA("TAU_MEASURE_TAU", "off");
    }

    tmp = getconf("TAU_SAMPLING");
    // We should disable sampling if tracing has been enabled!
    if (parse_bool(tmp, TAU_EBS_DEFAULT) && (env_tracing == 0)) {
#ifdef TAU_DISABLE_MEM_MANAGER
      env_ebs_enabled = 0;
      TAU_VERBOSE("TAU: Sampling Disabled - memory management was disabled at configuration!\n");
      TAU_METADATA("TAU_SAMPLING", "disabled (memory management disabled)");
#else
      env_ebs_enabled = 1;
      TAU_VERBOSE("TAU: Sampling Enabled\n");
      TAU_METADATA("TAU_SAMPLING", "on");
#endif
    } else {
      env_ebs_enabled = 0;
      TAU_VERBOSE("TAU: Sampling Disabled\n");
      TAU_METADATA("TAU_SAMPLING", "off");
    }

    tmp = getconf("TAU_EBS_KEEP_UNRESOLVED_ADDR");
    if (parse_bool(tmp, TAU_EBS_KEEP_UNRESOLVED_ADDR_DEFAULT)) {
      env_ebs_keep_unresolved_addr = 1;
      TAU_METADATA("TAU_EBS_KEEP_UNRESOLVED_ADDR", "on");
    } else {
      env_ebs_keep_unresolved_addr = 0;
      TAU_METADATA("TAU_EBS_KEEP_UNRESOLVED_ADDR", "off");
    }

    if (TauEnv_get_ebs_enabled()) {

      // *CWL* Acquire the sampling source. This has to be done first
      //       because the default EBS_PERIOD will depend on whether
      //       the specified source relies on timer interrupts or
      //       PAPI overflow interrupts or some other future
      //       mechanisms for triggering samples. The key problem with
      //       EBS_PERIOD defaults are that they are source-semantic
      //       sensitive (ie. 1000 microseconds is fine for timer
      //       interrutps, but 1000 PAPI_TOT_CYC is way too small).
      if ((env_ebs_source = getconf("TAU_EBS_SOURCE")) == NULL) {
        env_ebs_source = "itimer";
      }
      env_ebs_source_orig = strdup(env_ebs_source);
      sprintf(tmpstr, "%s", env_ebs_source);
      TAU_METADATA("TAU_EBS_SOURCE", tmpstr);

      TAU_VERBOSE("TAU: EBS Source: %s\n", env_ebs_source);

      /* TAU sampling period */
      const char *ebs_period = getconf("TAU_EBS_PERIOD");
      int default_ebs_period = TAU_EBS_PERIOD_DEFAULT;
      // *CWL* - adopting somewhat saner period values for PAPI-based
      //         EBS sample sources. The code obviously has to be more
      //         adaptive to account for the widely-varying semantics,
      //         but we will use a one-size-fits-all mid-sized prime
      //         number for now. The reason for a prime number? So we
      //         do not get into cyclical sampling problems on sources
      //         like L1 cache misses.
      //
      //         The check for PAPI sources will be extremely naive for
      //         now.
      if (strncmp(env_ebs_source, "PAPI", 4) == 0) {
        default_ebs_period = 133337;
      }
      env_ebs_period = default_ebs_period;
      if (ebs_period) {
        // Try setting it to the user value.
        env_ebs_period = atoi(ebs_period);
        // *CWL* - 0 is not a valid ebs_period. Plus atoi() returns 0
        //         if the string is not a number.
        if (env_ebs_period <= 0) {
          // go back to default on failure or bad value.
          env_ebs_period = default_ebs_period;
        }
      }
      TAU_VERBOSE("TAU: EBS period = %d \n", env_ebs_period);
      sprintf(tmpstr, "%d", env_ebs_period);
      TAU_METADATA("TAU_EBS_PERIOD", tmpstr);

      bool ebs_period_forced = false;
#ifdef EBS_CLOCK_RES
      if (strcmp(env_ebs_source, "itimer") != 0) {
        // *CWL* - force the clock period to be of a sane value
        //         if the desired (or default) value is not
        //         supported by the machine. ONLY valid for "itimer"
        //         EBS_SOURCE.
        if (env_ebs_period < EBS_CLOCK_RES) {
          env_ebs_period = EBS_CLOCK_RES;
          ebs_period_forced = true;
        }
      }
#endif
      if (ebs_period_forced) {
        sprintf(tmpstr, "%d", env_ebs_period);
        TAU_METADATA("TAU_EBS_PERIOD (FORCED)", tmpstr);
      }

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

#ifdef TAU_UNWIND
      tmp = getconf("TAU_EBS_UNWIND");
      if (parse_bool(tmp, TAU_EBS_UNWIND_DEFAULT)) {
        env_ebs_unwind_enabled = 1;
        TAU_METADATA("TAU_EBS_UNWIND", "on");
      } else {
        env_ebs_unwind_enabled = 0;
        TAU_METADATA("TAU_EBS_UNWIND", "off");
      }

      if (env_ebs_unwind_enabled == 1) {
        const char *depth = getconf("TAU_EBS_UNWIND_DEPTH");
        env_ebs_unwind_depth = TAU_EBS_UNWIND_DEPTH_DEFAULT;
        if (depth) {
          env_ebs_unwind_depth = atoi(depth);
          if (env_ebs_unwind_depth < 0) {
            env_ebs_unwind_depth = TAU_CALLPATH_DEPTH_DEFAULT;
          }
        }
		if (env_ebs_unwind_depth == 0) {
          sprintf(tmpstr, "auto");
        } else {
          sprintf(tmpstr, "%d", env_ebs_unwind_depth);
		}
        TAU_METADATA("TAU_EBS_UNWIND_DEPTH", tmpstr);
      }
#endif /* TAU_UNWIND */

      const char *ebs_resolution = getconf("TAU_EBS_RESOLUTION");
      if (ebs_resolution) {
          if (strcmp(ebs_resolution, TAU_EBS_RESOLUTION_STR_FILE) == 0) {
              env_ebs_resolution = TAU_EBS_RESOLUTION_FILE;
              TAU_METADATA("TAU_EBS_RESOLUTION", TAU_EBS_RESOLUTION_STR_FILE);
          } else if (strcmp(ebs_resolution, TAU_EBS_RESOLUTION_STR_FUNCTION) == 0) {
              env_ebs_resolution = TAU_EBS_RESOLUTION_FUNCTION;
              TAU_METADATA("TAU_EBS_RESOLUTION", TAU_EBS_RESOLUTION_STR_FUNCTION);
          } else if (strcmp(ebs_resolution, TAU_EBS_RESOLUTION_STR_FUNCTION_LINE) == 0) {
              env_ebs_resolution = TAU_EBS_RESOLUTION_FUNCTION_LINE;
              TAU_METADATA("TAU_EBS_RESOLUTION", TAU_EBS_RESOLUTION_STR_FUNCTION_LINE);
          } else if (strcmp(ebs_resolution, TAU_EBS_RESOLUTION_STR_LINE) == 0) { // otherwise, it's the default - line.
              env_ebs_resolution = TAU_EBS_RESOLUTION_LINE;
              TAU_METADATA("TAU_EBS_RESOLUTION", TAU_EBS_RESOLUTION_STR_LINE);
          }
      }

      if (TauEnv_get_tracing()) {
        env_callpath = 1;
        env_callpath_depth = 300;
        TAU_VERBOSE("TAU: EBS Overriding callpath settings, callpath enabled, depth = 300\n");
      }
    }

//#if (defined(TAU_UNIFY) && defined(TAU_MPI))
#if defined(TAU_UNIFY)
    tmp = getconf("TAU_STAT_PRECOMPUTE");
    if (parse_bool(tmp, TAU_PRECOMPUTE_DEFAULT)) {
      env_stat_precompute = 1;
      TAU_VERBOSE("TAU: Precomputation of statistics Enabled\n");
      /* *CWL* PRECOMPUTE only makes sense in the context of merged output */
      //      TAU_METADATA("TAU_PRECOMPUTE", "on");
    } else {
      env_stat_precompute = 0;
      TAU_VERBOSE("TAU: Precomputation of statistics Disabled\n");
      //      TAU_METADATA("TAU_PRECOMPUTE", "off");
    }
#endif /* TAU_UNIFY && TAU_MPI */

    /* child fork directory */
    tmp = getconf("TAU_CHILD_FORKDIRS");
    if (parse_bool(tmp, 0)) {
      env_child_forkdirs = 1;
      TAU_VERBOSE("TAU: Child-Fork Directories Enabled\n");
      /*TAU_METADATA("TAU_PROFILE", "on");*/
    } else {
      env_child_forkdirs = 0;
      /*TAU_VERBOSE("TAU: Profiling Disabled\n");
        TAU_METADATA("TAU_PROFILE", "off");*/
    }

    env_cupti_api = getconf("TAU_CUPTI_API");
    if (env_cupti_api == NULL || 0 == strcasecmp(env_cupti_api, "")) {
      env_cupti_api = TAU_CUPTI_API_DEFAULT;
      TAU_VERBOSE("TAU: CUPTI API tracking: %s\n", env_cupti_api);
      TAU_METADATA("TAU_CUPTI_API", env_cupti_api);
    }
    else {
      TAU_VERBOSE("TAU: CUPTI API tracking: %s\n", env_cupti_api);
      TAU_METADATA("TAU_CUPTI_API", env_cupti_api);
		}
    env_cuda_device_name = getconf("TAU_CUDA_DEVICE_NAME");
    if (!env_cuda_device_name || 0 == strcasecmp(env_cuda_device_name, "")) {
        env_cuda_device_name = TAU_CUDA_DEVICE_NAME_DEFAULT;
    } else {
        TAU_VERBOSE("TAU: CUDA device: %s\n", env_cuda_device_name);
        TAU_METADATA("TAU_CUDA_DEVICE", env_cuda_device_name);
    }
    env_track_cuda_instructions = getconf("TAU_TRACK_CUDA_INSTRUCTIONS");
    if (env_track_cuda_instructions == NULL || 0 == strcasecmp(env_track_cuda_instructions, "")) {
      env_track_cuda_instructions = TAU_TRACK_CUDA_INSTRUCTIONS_DEFAULT;
      TAU_VERBOSE("TAU: tracking CUDA instructions: %s\n", env_track_cuda_instructions);
      TAU_METADATA("TAU_TRACK_CUDA_INSTRUCTIONS", env_track_cuda_instructions);
    }
    else {
      TAU_VERBOSE("TAU: tracking CUDA instructions: %s\n", env_track_cuda_instructions);
      TAU_METADATA("TAU_TRACK_CUDA_INSTRUCTIONS", env_track_cuda_instructions);
		}
    tmp = getconf("TAU_TRACK_CUDA_CDP");
    if (parse_bool(tmp, TAU_TRACK_CUDA_CDP_DEFAULT)) {
      env_track_cuda_cdp = 1;
      TAU_VERBOSE("TAU: tracking CUDA CDP kernels Enabled\n");
      TAU_METADATA("TAU_TRACK_CUDA_CDP", "on");
    } else {
      TAU_VERBOSE("TAU: tracking CUDA CDP kernels Disabled\n");
      TAU_METADATA("TAU_TRACK_CUDA_CDP", "off");
    }
    tmp = getconf("TAU_TRACK_UNIFIED_MEMORY");
    if (parse_bool(tmp, TAU_TRACK_UNIFIED_MEMORY_DEFAULT)) {
      env_track_unified_memory = 1;
      TAU_VERBOSE("TAU: tracking CUDA UNIFIED MEMORY Enabled\n");
      TAU_METADATA("TAU_TRACK_UNIFIED_MEMORY", "on");
    } else {
      TAU_VERBOSE("TAU: tracking CUDA UNIFIED MEMORY Disabled\n");
      TAU_METADATA("TAU_TRACK_UNIFIED_MEMORY", "off");
    }
    tmp = getconf("TAU_TRACK_CUDA_SASS");
    if (parse_bool(tmp, TAU_TRACK_CUDA_SASS_DEFAULT)) {
      env_track_cuda_sass = 1;
      TAU_VERBOSE("TAU: tracking CUDA SASS Enabled\n");
      TAU_METADATA("TAU_TRACK_CUDA_SASS", "on");
      // get arg of sass type
      const char *sass_type = getconf("TAU_SASS_TYPE");
      if (sass_type) {
	env_sass_type = sass_type;
      }
      TAU_VERBOSE("TAU: SASS type = %s \n", env_sass_type);
      sprintf(tmpstr, "%s", env_sass_type);
      TAU_METADATA("TAU_SASS_TYPE", tmpstr);

    } else {
      TAU_VERBOSE("TAU: tracking CUDA SASS Disabled\n");
      TAU_METADATA("TAU_TRACK_CUDA_SASS", "off");
    }
    tmp = getconf("TAU_OUTPUT_CUDA_CSV");
    if (parse_bool(tmp, TAU_OUTPUT_CUDA_CSV_DEFAULT)) {
      env_output_cuda_csv = 1;
      TAU_VERBOSE("TAU: output CUDA CSV Enabled\n");
      TAU_METADATA("TAU_OUTPUT_CUDA_CSV", "on");
    } else {
      TAU_VERBOSE("TAU: output CUDA CSV Disabled\n");
      TAU_METADATA("TAU_OUTPUT_CUDA_CSV", "off");
    }
    env_binaryexe = getconf("TAU_CUDA_BINARY_EXE");
    if (env_binaryexe == NULL || 0 == strcasecmp(env_binaryexe, "")) {
      env_binaryexe = "";
      TAU_VERBOSE("TAU: CUDA binary exe not provided: %s\n", env_binaryexe);
      TAU_METADATA("TAU_CUDA_BINARY_EXE", env_binaryexe);
    }
    else {
      TAU_VERBOSE("TAU: CUDA binary exe: %s\n", env_binaryexe);
      TAU_METADATA("TAU_CUDA_BINARY_EXE", env_binaryexe);
    }
    tmp = getconf("TAU_TRACK_CUDA_ENV");
    if (parse_bool(tmp, TAU_TRACK_CUDA_ENV_DEFAULT)) {
      env_track_cuda_env = 1;
      TAU_VERBOSE("TAU: tracking CUDA Environment Enabled\n");
      TAU_METADATA("TAU_TRACK_CUDA_ENV", "on");
    } else {
      TAU_VERBOSE("TAU: tracking CUDA Environment Disabled\n");
      TAU_METADATA("TAU_TRACK_CUDA_ENV", "off");
    }
    tmp = getconf("TAU_MIC_OFFLOAD");
    if (parse_bool(tmp, TAU_MIC_OFFLOAD_DEFAULT)) {
      env_mic_offload = 1;
      TAU_VERBOSE("TAU: MIC offloading Enabled\n");
      TAU_METADATA("TAU_MIC_OFFLOAD", "on");
		}

    tmp = getconf("TAU_BFD_LOOKUP");
    if (parse_bool(tmp, TAU_BFD_LOOKUP)) {
      env_bfd_lookup = 1;
      TAU_VERBOSE("TAU: BFD Lookup Enabled\n");
      TAU_METADATA("TAU_BFD_LOOKUP", "on");
    } else {
      env_bfd_lookup = 0;
      TAU_VERBOSE("TAU: BFD Lookup Disabled\n");
      TAU_METADATA("TAU_BFD_LOOKUP", "off");
    }

    tmp = getconf("TAU_MERGE_METADATA");
    if (parse_bool(tmp, TAU_MERGE_METADATA_DEFAULT)) {
      env_merge_metadata = 1;
      TAU_VERBOSE("TAU: Metadata Merging Enabled\n");
    } else {
      env_merge_metadata = 0;
      TAU_VERBOSE("TAU: Metadata Merging Disabled\n");
    }

    tmp = getconf("TAU_DISABLE_METADATA");
    if (parse_bool(tmp, TAU_DISABLE_METADATA_DEFAULT)) {
      env_disable_metadata = 1;
      TAU_VERBOSE("TAU: Metadata Disabled\n");
    } else {
      env_disable_metadata = 0;
      TAU_VERBOSE("TAU: Metadata Enabled\n");
    }

#if defined(TAU_TBB_SUPPORT) && defined(TAU_MPI)
    if (env_profile_format != TAU_FORMAT_MERGED) {
      std::cerr << "TAU: WARNING: TAU_PROFILE_FORMAT=merged is recommended when profiling TBB and MPI." << std::endl;
    }
#endif

#ifdef TAU_ANDROID
    tmp = getconf("TAU_ALFRED_PORT");
    if (tmp) {
	env_alfred_port = atoi(tmp);
    }
    TAU_VERBOSE("TAU: Alfred will listen on port %d\n", env_alfred_port);
#endif

    tmp = getconf("TAU_MEM_CONTEXT");
    if (parse_bool(tmp, TAU_MEM_CALLPATH_DEFAULT)) {
      env_mem_callpath = 1;
      TAU_VERBOSE("TAU: Memory Class Tracking Callpath Enabled\n");
    } else {
      env_mem_callpath = 0;
      TAU_VERBOSE("TAU: Memory Class Tracking Callpath Disabled\n");
    }

    if ((env_mem_classes = getconf("TAU_MEM_CLASSES")) == NULL) {
      env_mem_classes = "";
      TAU_VERBOSE("TAU: MEM_CLASSES is not set\n");
    } else {
      TAU_VERBOSE("TAU: MEM_CLASSES is \"%s\"\n", env_mem_classes);
      if(strcmp(env_mem_classes, "all") == 0) {
        env_mem_all = 1;
        TAU_VERBOSE("TAU: Tracking All Class Allocations\n");
      }
      env_mem_classes_set = new std::set<std::string>();
      char * next_mem_class = strtok_r((char *)env_mem_classes, ":,", &saveptr);
      while(next_mem_class != NULL) {
        env_mem_classes_set->insert(next_mem_class);
        next_mem_class = strtok_r(NULL, ":,", &saveptr);
      }
    }

    if ((env_tau_exec_args = getconf("TAU_EXEC_ARGS")) == NULL) {
      env_tau_exec_args = "";
      TAU_VERBOSE("TAU: TAU_EXEC_ARGS is not set\n");
    } else {
      TAU_VERBOSE("TAU: TAU_EXEC_ARGS is \"%s\"\n", env_tau_exec_args);
    }

    if ((env_tau_exec_path = getconf("TAU_EXEC_PATH")) == NULL) {
      env_tau_exec_path = "";
      TAU_VERBOSE("TAU: TAU_EXEC_PATH is not set\n");
    } else {
      TAU_VERBOSE("TAU: TAU_EXEC_PATH is \"%s\"\n", env_tau_exec_path);
    }

    initialized = 1;
    TAU_VERBOSE("TAU: Initialized TAU (TAU_VERBOSE=1)\n");

/* Add metadata in the form of "<key1=val1:key2=val2:key3=val3>" */
    char *metadata = (char *) getconf("TAU_METADATA");

#ifndef TAU_WINDOWS
    // export TAU_METADATA="<key1=val1:key2=val2:key3=val3>"
    if (metadata) {
      key = strtok_r(metadata, "<=,>", &saveptr);
      while (key != (char *) NULL) {
        val = strtok_r(NULL, ":>", &saveptr);
        TAU_VERBOSE("TAU_METADATA %s = %s \n", key, val);
        TAU_METADATA(key, val);
        key = strtok_r(NULL, "=", &saveptr); // get the next pair
      }
    }
#endif /* TAU_WINDOWS - use strtok under Windows */
    /* Now that we have set all the options, start the one signal
     * handler that will handle load, power, memory, headroom, etc.
     */
    //TauSetupHandler();
  }

#ifdef TAU_ENABLE_ROCTRACER
  TAU_VERBOSE("Calling TAU_ROCTRACER...\n");
  Tau_roctracer_start_tracing();
#endif /* TAU_ENABLE_ROCTRACER */
}

} /* C linkage */
