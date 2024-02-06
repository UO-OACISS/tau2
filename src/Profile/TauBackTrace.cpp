#include <TAU.h>
#include <Profile/TauBfd.h>
#include <Profile/TauBacktrace.h>
#include <vector>
#include <cstdarg>

using namespace std;

#define TAU_MAX_STACK 1024

#if !defined(_AIX) && !defined(__sun) && !defined(TAU_WINDOWS) && !defined(TAU_ANDROID) && !defined(TAU_NEC_SX)
#include <execinfo.h>
#define TAU_EXECINFO 1
#endif

#ifdef __APPLE__
#include <dlfcn.h>
#ifdef TAU_HAVE_CORESYMBOLICATION
#include "CoreSymbolication.h"
#endif
#endif /* __APPLE__ */

extern "C" void finalizeCallSites_if_necessary();

struct BacktraceFrame
{
  char const * funcname;
  char const * filename;
  char const * mapname;
  int lineno;
};

static vector<int> iteration;//[TAU_MAX_THREADS] = { 0 };
std::mutex ItVectorMutex;
inline void checkItVector(int tid){
      std::lock_guard<std::mutex> guard(ItVectorMutex);
      while (iteration.size()<=tid){
          iteration.push_back(0);
      }
}
static inline int& getIterationRef(int tid){
    checkItVector(tid);
    return iteration[tid];
}

static int getBacktraceFromExecinfo(int trim, BacktraceFrame ** oframes)
{
#ifdef TAU_EXECINFO

  static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

  // Initialize BFD
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit();
  }

  // Get the backtrace
  BacktraceFrame * frames = NULL;
  void * addrs[TAU_MAX_STACK];
  int naddrs = backtrace(addrs, TAU_MAX_STACK);
  if (naddrs) {
    TAU_VERBOSE("TAU: Backtrace has %d addresses:\n", naddrs);
    frames = (BacktraceFrame*)calloc(naddrs, sizeof(BacktraceFrame));

    for (int i=trim+1, j=0; i<naddrs; ++i, ++j) {
      unsigned long addr = (unsigned long)addrs[i];

      // Get source information from BFD
      TauBfdInfo info;
#if defined(__APPLE__)
#if defined(TAU_HAVE_CORESYMBOLICATION)
      static CSSymbolicatorRef symbolicator = CSSymbolicatorCreateWithPid(getpid());
      CSSourceInfoRef source_info = CSSymbolicatorGetSourceInfoWithAddressAtTime(symbolicator, (vm_address_t)addr, kCSNow);
      if(!CSIsNull(source_info)) {
          CSSymbolRef symbol = CSSourceInfoGetSymbol(source_info);
          info.filename = strdup(CSSourceInfoGetPath(source_info));
          info.funcname = strdup(CSSymbolGetName(symbol));
          info.lineno = CSSourceInfoGetLineNumber(source_info);
      }
#else
      Dl_info dlinfo;
      int rc = dladdr((const void *)addr, &dlinfo);
      if (rc != 0) {
        info.filename = strdup(dlinfo.dli_fname);
        info.funcname = strdup(dlinfo.dli_sname);
        info.lineno = 0; // Apple doesn't give us line numbers.
      }
#endif
#else
      Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, info);

      // Try to get the name of the memory map containing the address
      TauBfdAddrMap const * map = Tau_bfd_getAddressMap(bfdUnitHandle, addr);
      char const * mapname = map ? map->name : "unknown";
      // Record information
      frames[j].mapname = mapname;
#endif
      // Record information
      frames[j].funcname = info.funcname;
      frames[j].filename = info.filename;
      frames[j].lineno = info.lineno;
    }
  } else {
    TAU_VERBOSE("TAU: ERROR: Backtrace not available!\n");
  }
  *oframes = frames;
  return naddrs - (trim+1);

#else
  TAU_VERBOSE("TAU: ERROR: execinfo not available for backtrace\n");
  *oframes = NULL;
  return 0;
#endif
}

static int getBacktraceFromGDB(int trim, BacktraceFrame ** oframes)
{
  // This obviously needs work...

  char cmd[8192];
  char path[4096];
  char gdb_in_file[128];
  char gdb_out_file[128];

  path[readlink("/proc/self/exe", path, sizeof(path)-1)] = '\0';

  snprintf(gdb_in_file, sizeof(gdb_in_file),  "tau_gdb_cmds_%d.txt", RtsLayer::getPid());
  snprintf(gdb_out_file, sizeof(gdb_out_file),  "tau_gdb_out_%d.txt", RtsLayer::getPid());

  FILE * gdb_fp = fopen(gdb_in_file, "w+");
  fprintf(gdb_fp, "set logging on %s\nbt\nq\n", gdb_out_file);
  fclose(gdb_fp);

  snprintf(cmd, sizeof(cmd),  "gdb -batch -x %s %s -p %d >/dev/null\n", gdb_in_file, path, RtsLayer::getPid());
  TAU_VERBOSE("Calling: str=%s\n", cmd);
  int systemRet = system(cmd);

  // Success returns the pid. We check for failure (-1) here.
  if (systemRet == -1) {
    TAU_VERBOSE("TAU: ERROR - Call failed executing %s\n", cmd);
  }

  * oframes = NULL;
  return 0;
}


extern "C"
void Tau_print_simple_backtrace(int tid)
{
  BacktraceFrame * frames;
  int nframes;

  if (TauEnv_get_signals_gdb()) {
    nframes = getBacktraceFromGDB(0, &frames);
  } else {
    nframes = getBacktraceFromExecinfo(0, &frames);
  }

  if (nframes) {
    char metadata[128];
    char field[4096];
    for (int i=0; i<nframes; ++i) {
      BacktraceFrame const & info = frames[i];
      snprintf(metadata, sizeof(metadata),  "BACKTRACE(%5d) %3d", tid, i+1);
      snprintf(field, sizeof(field),  "[%s] [%s:%d] [%s]", info.funcname, info.filename, info.lineno, info.mapname);
      fprintf(stderr, "%s | %s\n", metadata, field);
    }
    delete[] frames;
  } else {
    printf("No frames!");
  }
}

extern "C"
int Tau_backtrace_record_backtrace(int trim)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  int & iter = getIterationRef(RtsLayer::myThread());
  ++iter;

  BacktraceFrame * frames;
  int nframes;

  if (TauEnv_get_signals_gdb()) {
    nframes = getBacktraceFromGDB(trim+1, &frames);
  } else {
    nframes = getBacktraceFromExecinfo(trim+1, &frames);
  }

  if (nframes) {
    char metadata[128];
    char field[4096];
    bool echo = TauEnv_get_echo_backtrace();
    for (int i=0; i<nframes; ++i) {
      BacktraceFrame const & info = frames[i];
      snprintf(metadata, sizeof(metadata),  "BACKTRACE(%5d) %3d", iter, i+1);
      snprintf(field, sizeof(field),  "[%s] [%s:%d] [%s]", info.funcname, info.filename, info.lineno, info.mapname);
      TAU_METADATA(metadata, field);
      if (echo) {
        fprintf(stderr, "%s | %s\n", metadata, field);
      }
    }
    delete[] frames;
  } else {
    printf("No frames!");
  }

  return iter;
}

extern "C"
void Tau_backtrace_exit_with_backtrace(int trim, char const * fmt, ...)
{
  va_list args;

  // Don't decrement this.  We're exiting so TAU's internal data structures
  // are being destroyed from here on out.  Recording new events will segfault.
  Tau_global_incr_insideTAU();

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_callsite()) {
    finalizeCallSites_if_necessary();
  }

  Tau_MemMgr_finalizeIfNecessary();

  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_finalize_if_necessary(Tau_get_local_tid());
  }
#endif /* _AIX */
#endif

  // Increment trim to exclude this function from the backtrace
  Tau_backtrace_record_backtrace(trim+1);

  // Record profiles
  TAU_PROFILE_EXIT("BACKTRACE");

  // Print the message
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);

  // Give the other tasks some time to process the handler and exit
  sleep(5);
  exit(1);
}
