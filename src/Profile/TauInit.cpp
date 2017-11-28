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
 **	File 		: TauInit.cpp 			        	   **
 **	Description 	: TAU Profiling Package				   **
 **	Author		: Alan Morris					   **
 **	Contact		: tau-bugs@cs.uoregon.edu               	   **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
 **                                                                         **
 **      Description     : TAU initialization                               **
 **                                                                         **
 ****************************************************************************/

#ifdef __APPLE__
#define _XOPEN_SOURCE 700 /* Single UNIX Specification, Version 3 */
#endif /* __APPLE__ */

#include <TAU.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
//#if !defined(TAU_WINDOWS) && !defined(TAU_ANDROID)
//#include <ucontext.h>
//#endif //TAU_WINDOWS
#include <string.h>

#ifndef TAU_WINDOWS
#include <unistd.h>
#endif

#include <Profile/TauEnv.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauSampling.h>
#include <Profile/TauSnapshot.h>
#include <Profile/TauMetaData.h>
#include <Profile/TauInit.h>
#include <Profile/TauMemory.h>
#include <Profile/TauBacktrace.h>
#include <Profile/TauUtil.h>
#include "Profile/TauSOS.h"

#ifdef TAU_VAMPIRTRACE 
#include <Profile/TauVampirTrace.h>
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
#include "elg_trc.h"
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_SCOREP
#include <Profile/TauSCOREP.h>
#endif

#ifdef TAU_ANDROID

#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>

/* There are something underhood TAU_VERBOSE which we don't want in Alfred */
#include <android/log.h>
#define LOGV(...) //__android_log_print(ANDROID_LOG_VERBOSE, "TAU", __VA_ARGS__)

#endif

#include <Profile/TauPlugin.h>
#include <Profile/TauPluginInternals.h>

using namespace std;

#ifndef TAU_WINDOWS
typedef void (*tau_sighandler_t)(int, siginfo_t*, void*);
#endif

#if defined(TAU_STRSIGNAL_OK)
#if defined(__PGI)
extern "C" char *strsignal(int sig) __THROW;
#else
extern "C" char *strsignal(int sig);
#endif /* __PGI */
#endif /* TAU_STRSIGNAL_OK */

extern "C" void Tau_stack_initialization();
extern "C" int Tau_compensate_initialization();
extern "C" int Tau_profiler_initialization();
extern "C" void Tau_profile_exit_all_threads();
extern "C" int Tau_dump_callpaths();
//extern "C" int Tau_initialize_collector_api(void);

extern "C" int Tau_show_profiles();


// True if TAU is fully initialized
int tau_initialized = 0;

// True if TAU is initializing
int initializing = 0;

// Rely on the dl auditor (src/wrapper/taupreload) to set dl_initialized
// if the audit feature is available (GLIBC version 2.4 or greater).
// DO NOT declare static!
#ifdef TAU_TRACK_LD_LOADER
int dl_initialized = 0;
#else
int dl_initialized = 1;
#endif

#ifndef TAU_DISABLE_SIGUSR

static void tauSignalHandler(int sig)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  if (TauEnv_get_sigusr1_action() == TAU_ACTION_DUMP_CALLPATHS) {
    fprintf(stderr, "Caught SIGUSR1, dumping TAU callpath data\n");
    Tau_dump_callpaths();
  } else if (TauEnv_get_sigusr1_action() == TAU_ACTION_DUMP_BACKTRACES) {
    fprintf(stderr, "Caught SIGUSR1, dumping backtrace data\n");
  } else {
    fprintf(stderr, "Caught SIGUSR1, dumping TAU profile data\n");
    TAU_DB_DUMP_PREFIX("profile");
  }
}

static void tauToggleInstrumentationHandler(int sig)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  fprintf(stderr, "Caught SIGUSR2, toggling TAU instrumentation\n");
  if (RtsLayer::TheEnableInstrumentation()) {
    RtsLayer::TheEnableInstrumentation() = false;
  } else {
    RtsLayer::TheEnableInstrumentation() = true;
  }
}

static void tauBacktraceHandler(int sig, siginfo_t *si, void *context)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  // Trigger a context event and record metadata
  char eventname[1024];
  sprintf(eventname, "TAU_SIGNAL (%s)", strsignal(sig));
  TAU_REGISTER_CONTEXT_EVENT(evt, eventname);
  TAU_CONTEXT_EVENT(evt, 1);
  TAU_METADATA("SIGNAL", strsignal(sig));

  Tau_backtrace_exit_with_backtrace(1,
      "TAU: Caught signal %d (%s), dumping profile with stack trace: [rank=%d, thread=%d, pid=%d, tid=%d]... \n",
      sig, strsignal(sig), RtsLayer::myNode(), RtsLayer::myThread(), RtsLayer::getPid(), RtsLayer::getTid());
}

static void tauMemdbgHandler(int sig, siginfo_t *si, void *context)
{
  // Use the backtrace handler if this SIGSEGV wasn't due to invalid memory access
  if (sig == SIGSEGV && si->si_code != SEGV_ACCERR) {
    tauBacktraceHandler(sig, si, context);
    return;
  }

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  TAU_REGISTER_CONTEXT_EVENT(evt, "Invalid memory access");

  // Try to find allocation information for the address
  void * ptr = si->si_addr;
  TauAllocation * alloc = TauAllocation::FindContaining(ptr);

  // If allocation info was found, be more informative and maybe attempt to continue
  if (alloc && TauEnv_get_memdbg_attempt_continue()) {
    typedef TauAllocation::addr_t addr_t;

    // Unprotect range so we can resume
    size_t size = Tau_page_size();
    addr_t addr = (addr_t)((size_t)ptr & ~(size-1));
    if (TauAllocation::Unprotect(addr, size)) {
      Tau_backtrace_exit_with_backtrace(1,
          "TAU: Memory debugger caught invalid memory access and cannot continue. "
          "Dumping profile with stack trace: [rank=%d, pid=%d, tid=%d]... \n",
          RtsLayer::myNode(), RtsLayer::getPid(), RtsLayer::getTid());
    }

    // Trigger the context event and record a backtrace
    TAU_CONTEXT_EVENT(evt, 1);
    Tau_backtrace_record_backtrace(1);

  } else {
    // Trigger the context event and record a backtrace
    TAU_CONTEXT_EVENT(evt, 1);
    Tau_backtrace_exit_with_backtrace(1,
        "TAU: Memory debugger caught invalid memory access. "
        "Dumping profile with stack trace: [rank=%d, pid=%d, tid=%d]... \n",
        RtsLayer::myNode(), RtsLayer::getPid(), RtsLayer::getTid());
  }

  // Exit the handler and return to the instruction that raised the signal
}


static int tauAddSignal(int sig, tau_sighandler_t handler = tauBacktraceHandler)
{
  int ret = 0;

  struct sigaction act;
  memset(&act, 0, sizeof(struct sigaction));
  ret = sigemptyset(&act.sa_mask);
  if (ret != 0) {
    printf("TAU: Signal error: %s\n", strerror(ret));
    return -1;
  }

  ret = sigaddset(&act.sa_mask, sig);
  if (ret != 0) {
    printf("TAU: Signal error: %s\n", strerror(ret));
    return -1;
  }
  act.sa_sigaction = handler;
#if defined(TAU_BGL) || defined(TAU_BGP) || defined(TAU_BGQ)
  act.sa_flags = SA_SIGINFO;
#else
  act.sa_flags = SA_SIGINFO | SA_ONSTACK;
#endif

  ret = sigaction(sig, &act, NULL);
  if (ret != 0) {
    printf("TAU: error adding signal in sigaction: %s\n", strerror(ret));
    return -1;
  }

  return ret;
}
#endif //TAU_DISABLE_SIGUSR


#ifdef TAU_VAMPIRTRACE
//////////////////////////////////////////////////////////////////////
// Initialize VampirTrace Tracing package
//////////////////////////////////////////////////////////////////////
int Tau_init_vampirTrace(void) {
  vt_open();
  return 0;
}
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_EPILOG 
//////////////////////////////////////////////////////////////////////
// Initialize EPILOG Tracing package
//////////////////////////////////////////////////////////////////////
int Tau_init_epilog(void) {
  esd_open();
  return 0;
}
#endif /* TAU_EPILOG */

extern "C"
int Tau_init_check_initialized()
{
  return tau_initialized;
}


extern "C"
int Tau_init_initializingTAU()
{
  return initializing - tau_initialized;
}

extern "C"
void Tau_init_dl_initialized()
{
  dl_initialized = 1;
}

extern "C"
int Tau_init_check_dl_initialized()
{
  return dl_initialized;
}

extern "C"
int Tau_profile_exit_scorep()
{
  Tau_profile_exit_all_threads();
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Initialize signal handling routines
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_signal_initialization()
{
#ifndef TAU_DISABLE_SIGUSR
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  if (TauEnv_get_track_signals()) {
    TAU_VERBOSE("TAU: Enable signal tracking\n");

    tauAddSignal(SIGILL);
    tauAddSignal(SIGINT);
    tauAddSignal(SIGQUIT);
    tauAddSignal(SIGTERM);
    tauAddSignal(SIGPIPE);
    tauAddSignal(SIGABRT);
    tauAddSignal(SIGFPE);
    if (TauEnv_get_memdbg()) {
      tauAddSignal(SIGBUS, tauMemdbgHandler);
      tauAddSignal(SIGSEGV, tauMemdbgHandler);
    } else {
      tauAddSignal(SIGBUS);
      tauAddSignal(SIGSEGV);
    }
  }

#endif // TAU_DISABLE_SIGUSR
  return 0;
}

#ifdef TAU_ANDROID

static void
alfred_handle_command(int fd, char *cmd, int len)
{
    if (strncasecmp(cmd, "DUMP", 4) == 0) {
	Tau_profile_exit_all_threads();
	write(fd, "OKAY\n", 5);
    }

    if (strncasecmp(cmd, "PROFILERS", 9) == 0) {
	Tau_show_profiles();
	write(fd, "OKAY\n", 5);
    }
}

static void*
alfred(void *arg)
{
    int rv;
    int sfd;
    struct sockaddr_in saddr;

    JNIThreadLayer::IgnoreThisThread();

    sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
	LOGV(" *** Alfred failed to start: %s", strerror(errno));
	return NULL;
    }

    int on = 1;
    rv = setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, (const char *) &on, sizeof(on));
    if (rv < 0) {
	LOGV(" *** Alfred failed to reuse socket: %s", strerror(errno));
    }

    saddr.sin_family = AF_INET;
    saddr.sin_port   = htons(TauEnv_get_alfred_port());
    saddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sfd, (struct sockaddr *)&saddr, sizeof(saddr)) < 0) {
	LOGV(" *** Alfred failed to start: %s", strerror(errno));
	close(sfd);
	return NULL;
    }

    if (listen(sfd, 1) < 0) {
	LOGV(" *** Alfred failed to start: %s", strerror(errno));
	close(sfd);
	return NULL;
    }

    while (1) {
	char cmd[18];
	int fd;

	LOGV(" *** (S%d) Alfred at your service\n", gettid());

	fd = accept(sfd, NULL, NULL);
	if (fd < 0) {
	    LOGV(" *** Alfred failed to accept new connection: %s", strerror(errno));
	    continue;
	}

	while (1) {
	    LOGV(" *** Alfred waiting for command\n");
	    rv = read(fd, cmd, sizeof(cmd));
	    if (rv < 0) {
		LOGV(" *** Alfred failed to read data: %s", strerror(errno));
		break;
	    }

	    if (rv == sizeof(cmd)) {
		LOGV(" *** Alfred can't understand the command");
		write(fd, "UNKNOWN\n", 8);
		close(fd);
		break;
	    }

	    if (rv == 0) { // peer disconnected
		break;
	    }

	    alfred_handle_command(fd, cmd, rv);
	}
    }

    return NULL;
}
#endif

extern "C" int Tau_init_initializeTAU()
{

  //protect against reentrancy
  if (initializing) return 0;
  initializing = 1;

  //Initialize locks.
  RtsLayer::Initialize();

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  /* initialize the memory debugger */
  Tau_memory_initialize();

  /* initialize the Profiler stack */
  Tau_stack_initialization();

  /* initialize environment variables */
  TauEnv_initialize();

  /*Initialize the plugin system only if both plugin path and plugins are specified*/
  if(TauEnv_get_plugins_path() && TauEnv_get_plugins()) {
    TAU_VERBOSE("TAU INIT: Initializing plugin system...\n");
    if(!Tau_initialize_plugin_system()) {
      TAU_VERBOSE("TAU INIT: Successfully Initialized the plugin system.\n");
    } else {
      printf("TAU INIT: Error initializing the plugin system\n");
    }
  }

#ifdef TAU_EPILOG
  /* no more initialization necessary if using epilog/scalasca */
  Tau_init_epilog();
  return 0;
#endif

#ifdef TAU_SCOREP
  /* no more initialization necessary if using SCOREP */
  SCOREP_Tau_InitMeasurement();
  SCOREP_Tau_RegisterExitCallback(Tau_profile_exit_scorep);
  return 0;
#endif

#ifdef TAU_VAMPIRTRACE
  /* no more initialization necessary if using vampirtrace */
  Tau_init_vampirTrace();
  return 0;
#endif

  /* we need the timestamp of the "start" */
  Tau_snapshot_initialization();

#ifndef TAU_DISABLE_SIGUSR
  if (signal(SIGUSR1, tauSignalHandler) == SIG_ERR) {
    perror("failed to register TAU profile dump signal handler");
  }

  if (signal(SIGUSR2, tauToggleInstrumentationHandler) == SIG_ERR) {
    perror("failed to register TAU instrumentation toggle signal handler");
  }
#endif

  Tau_profiler_initialization();

  /* initialize the metrics we will be counting */
  TauMetrics_init();

  Tau_signal_initialization();

  /* initialize compensation */
  if (TauEnv_get_compensate()) {
    Tau_compensate_initialization();
  }

  /* initialize sampling if requested */
#if !defined(TAU_MPI) && !defined(TAU_WINDOWS)
  if (TauEnv_get_ebs_enabled()) {
    // Work-around for MVAPHICH 2 to move sampling initialization to after MPI_Init()
    Tau_sampling_init_if_necessary();
  }
#endif /* TAU_MPI && TAU_WINDOWS */

#ifdef TAU_PGI
  sbrk(102400);
#endif /* TAU_PGI */

#ifndef TAU_DISABLE_METADATA
  Tau_metadata_fillMetaData();
#endif

#ifdef TAU_OPENMP
  //Tau_initialize_collector_api();
#endif

  // Mark initialization complete so calls below can start timers
  tau_initialized = 1;

#ifdef __MIC__
  if (TauEnv_get_mic_offload()) {
    TAU_PROFILE_SET_NODE(0);
  }
#endif

#if defined(TAU_SOS) && !defined(TAU_MPI)
  TAU_SOS_init_simple();
#endif

  Tau_create_top_level_timer_if_necessary();

  Tau_memory_wrapper_enable();

#ifdef TAU_ANDROID
  pthread_t thr;
  pthread_create(&thr, NULL, alfred, NULL);
#endif

  return 0;
}
