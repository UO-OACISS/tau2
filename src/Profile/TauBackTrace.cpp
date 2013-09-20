#include <TAU.h>
#include <Profile/TauBfd.h>
#include <Profile/TauBacktrace.h>
#include <vector>
#include <cstdarg>

#ifdef __GNUC__
#include <cxxabi.h>
#endif /* __GNUC__ */

using namespace std;

#define TAU_MAX_STACK 1024

#if !defined(_AIX) && !defined(__sun) && !defined(TAU_WINDOWS) && !defined(TAU_ANDROID)
#include <execinfo.h>
#define TAU_EXECINFO 1
#endif

extern "C" void finalizeCallSites_if_necessary();

struct BacktraceFrame
{
  char const * funcname;
  char const * filename;
  char const * mapname;
  int lineno;
};

static int iteration[TAU_MAX_THREADS] = { 0 };

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
      Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, info);

      // Try to get the name of the memory map containing the address
      TauBfdAddrMap const * map = Tau_bfd_getAddressMap(bfdUnitHandle, addr);
      char const * mapname = map ? map->name : "unknown";

      // Record information
      frames[j].funcname = info.funcname;
      frames[j].filename = info.filename;
      frames[j].mapname = mapname;
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

  sprintf(gdb_in_file, "tau_gdb_cmds_%d.txt", getpid());
  sprintf(gdb_out_file, "tau_gdb_out_%d.txt", getpid());

  FILE * gdb_fp = fopen(gdb_in_file, "w+");
  fprintf(gdb_fp, "set logging on %s\nbt\nq\n", gdb_out_file);
  fclose(gdb_fp);

  sprintf(cmd, "gdb -batch -x %s %s -p %d >/dev/null\n", gdb_in_file, path, getpid());
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
int Tau_backtrace_record_backtrace(int trim)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  int & iter = iteration[RtsLayer::getTid()];
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
    for (int i=0; i<nframes; ++i) {
      BacktraceFrame const & info = frames[i];
      sprintf(metadata, "BACKTRACE(%d) %3d", iter, i+1);
      sprintf(field, "[%s] [%s:%d] [%s]", info.funcname, info.filename, info.lineno, info.mapname);
      TAU_METADATA(metadata, field);
    }
    delete[] frames;
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

#if !defined(TAU_WINDOWS) && !defined(TAU_ANDROID)
  if (TauEnv_get_callsite()) {
    finalizeCallSites_if_necessary();
  }

  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_finalize_if_necessary();
  }
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
