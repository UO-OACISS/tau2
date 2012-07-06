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

#include <TAU.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>

#ifndef TAU_WINDOWS
#include <unistd.h>
#endif

#if !defined(_AIX) && !defined(__sun) && !defined(TAU_WINDOWS)
#include <execinfo.h>
#define TAU_EXECINFO 1
#endif /* _AIX */

#include <Profile/TauEnv.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauSampling.h>
#include <Profile/TauSnapshot.h>
#include <Profile/TauMetaData.h>

// Moved from header file
using namespace std;



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


#define TAU_MAXSTACK 1024

extern "C" void Tau_stack_initialization();
extern "C" int Tau_compensate_initialization();
extern "C" int Tau_profiler_initialization();
extern "C" int Tau_profile_exit_all_threads(); 

// **CWL** 20110912 NOTE: strsignal is NOT portable!
//         A number of systems has strsignal declared in string.h and this 
//         will conflict with an explicit extern declaration of this nature.
//
//         This is now determined at configure time.
#if defined(TAU_STRSIGNAL_OK)
extern "C" char *strsignal(int sig);
#endif /* TAU_STRSIGNAL_OK */

/* -- signal catching to flush event buffers ----------------- */
#if defined (__cplusplus) || defined (__STDC__) || defined (_AIX) || (defined (__mips) && defined (_SYSTYPE_SVR4))
#define SIGNAL_TYPE	void
#define SIGNAL_ARG_TYPE	int
#else	/* Not ANSI C.  */
#define SIGNAL_TYPE	int
#define SIGNAL_ARG_TYPE
#endif	/* ANSI C */
# ifndef NSIG
#   define NSIG 32
# endif
static SIGNAL_TYPE (*sighdlr[NSIG])(SIGNAL_ARG_TYPE);

static void wrap_up(int sig) {
  void *array[10];
  size_t size;

#ifdef TAU_EXECINFO
  // get void*'s for all entries on the stack
  size = backtrace(array, 10);
#endif /* TAU_EXECINFO = !(_AIX || sun || windows) */

  // print out all the frames to stderr
  fprintf (stderr, "TAU: signal %d on %d - calling TAU_PROFILE_EXIT()...\n", sig, RtsLayer::myNode());

#ifdef TAU_EXECINFO
  backtrace_symbols_fd(array, size, 2);
#endif /* TAU_EXECINFO */
  TAU_PROFILE_EXIT("signal");
  fprintf (stderr, "TAU: done.\n");
  exit (1);
}

static void TauInitialize_kill_handlers() {
# ifdef SIGINT
  sighdlr[SIGINT ] = signal (SIGINT , wrap_up);
# endif
# ifdef SIGQUIT
  sighdlr[SIGQUIT] = signal (SIGQUIT, wrap_up);
# endif
# ifdef SIGILL
  sighdlr[SIGILL ] = signal (SIGILL , wrap_up);
# endif
# ifdef SIGFPE
  sighdlr[SIGFPE ] = signal (SIGFPE , wrap_up);
# endif
# ifdef SIGBUS
  sighdlr[SIGBUS ] = signal (SIGBUS , wrap_up);
# endif
# ifdef SIGTERM
  sighdlr[SIGTERM] = signal (SIGTERM, wrap_up);
# endif
# ifdef SIGABRT
  sighdlr[SIGABRT] = signal (SIGABRT, wrap_up);
# endif
# ifdef SIGSEGV
  sighdlr[SIGSEGV] = signal (SIGSEGV, wrap_up);
# endif
# ifdef SIGCHLD
  sighdlr[SIGCHLD] = signal (SIGCHLD, wrap_up);
# endif
}

extern "C" int Tau_get_backtrace_off_by_one_correction(void) {
  return 1; /* offset address by one */
}

// **CWL** Added to be consistent for operation with Comp_gnu.cpp
//#ifndef TAU_XLC
extern int Tau_Backtrace_writeMetadata(int i, char *token1, unsigned long addr);
//#endif /* TAU_XLC */

#ifndef TAU_DISABLE_SIGUSR

//static void tauBacktraceHandler(int sig) {
extern "C" void finalizeCallSites_if_necessary();
void tauBacktraceHandler(int sig, siginfo_t *si, void *context) {
          char str[100+4096];
          char path[4096];
          char gdb_in_file[256];
          char gdb_out_file[256];

	  if (TauEnv_get_callsite()) {
	    finalizeCallSites_if_necessary();
	  }

#ifndef TAU_WINDOWS
	  if (TauEnv_get_ebs_enabled()) {
	    // *CWL* - If sampling is active, get it to stop and finalize immediately,
	    //         we are about to halt execution!
	    //	    int tid = RtsLayer::myThread();
	    //	    Tau_sampling_finalize(tid);
	    Tau_sampling_finalize_if_necessary();
	  }
#endif /* TAU_WINDOWS */

  // Start by triggering a context event
  char eventname[1024];
  sprintf(eventname,"TAU_SIGNAL (%s)", strsignal(sig)); 
  TAU_REGISTER_CONTEXT_EVENT(evt, eventname);
  TAU_CONTEXT_EVENT(evt, 1); 

  // Attempt to generate backtrace text information via GDB
  // *CWL* - Shouldn't we make use of the information (via parsing?) as an alternative
  //         to backtrace in case the latter fails (as in the case with PrgEnv-cray)?
  //         Something to keep in mind for later.
  if (TauEnv_get_signals_gdb()) {
          path[readlink("/proc/self/exe", path, -1+ sizeof(path))] = '\0';
          //sprintf(str, "echo 'bt\ndetach\nquit\n' | gdb -batch -x /dev/stdin %s -p %d \n",
                  //path, (int)getpid() );
          sprintf(gdb_in_file, "tau_gdb_cmds_%d.txt", getpid());
          sprintf(gdb_out_file, "tau_gdb_out_%d.txt", getpid());
          FILE *gdb_fp=fopen(gdb_in_file, "w+"); 
          fprintf(gdb_fp, "set logging on %s\nbt\nq\n", gdb_out_file);
          fclose(gdb_fp);
	  //sprintf(str,"echo set logging on %s\nbt\nq > %s",gdb_out_file, gdb_in_file);
          //system(str); // create gdbcmds
          sprintf(str, "gdb -batch -x %s %s -p %d >/dev/null\n", gdb_in_file,
                  path, (int)getpid() );
          TAU_VERBOSE("Calling: str=%s\n", str);

	  int systemRet = 0;
          systemRet = system(str);
	  // Success returns the pid. We check for failure (-1) here.
	  if (systemRet == -1) {
	    // We still want to output the TAU profile.
	    TAU_VERBOSE("tauBacktraceHandler: Call failed executing %s\n",
			str);
	    // give the other tasks some time to process the handler and exit
	    fprintf(stderr, "TAU: Caught signal %d (%s), dumping profile without stack trace (GDB failure): [rank=%d, pid=%d, tid=%d]... \n", sig, strsignal(sig), RtsLayer::myNode(), getpid(), Tau_get_tid(), sig);
	    TAU_METADATA("SIGNAL", strsignal(sig));
	    TAU_PROFILE_EXIT("none");
	    sleep(4);  
	    exit(1);     
	    
	  }
  }

  // Attempt to generate metadata information about backtraces if the backtrace calls are
  //   supported on the system.
#ifdef TAU_EXECINFO
  static void *addresses[TAU_MAXSTACK];
  int n = backtrace( addresses, TAU_MAXSTACK );

  if (n < 2) {
    // For dealing with badly implemented backtrace calls
    TAU_VERBOSE("TAU: ERROR: Backtrace not available!\n" );
  } else {
    TAU_VERBOSE("TAU: ::BACKTRACE:: has %d addresses:\n", n);
    char **names = backtrace_symbols( addresses, n );
    for ( int i = 2; i < n; i++ )
    {
      TAU_VERBOSE("**STACKTRACE** Entry %d = %s\n", i, names[i]);
      char *temp = NULL;
      char *token = NULL;
      unsigned long addr;
      //  UPDATE 11/20: token2 is required if we encounter shared code in which
      //         case the format is <path>(<func_name>+<offset>) [<addr>].
      //         This is extremely fragile code and is at the mercy of backtrace
      //         not changing its output formats over a wide variety of machines.
      //         We must keep an eye on this. This has the potential to (it has
      //         already) blow up in our faces repeatedly.
      //
      //  For now, the correct way to do this is to look for the last token.
      temp = strtok(names[i],"[]");
      while (temp != NULL) {
	token = temp;
	temp = strtok(NULL, "[]");
      }
      if (token == NULL) {
	// If the backtrace string is completely invalid, then set to 0 and allow
	//   tauPrintAddr to fail. Issue a verbose warning.
	TAU_VERBOSE("No valid token in backtrace string!\n");
	addr = 0;
      } else {
	TAU_VERBOSE("Found Address Token = %s\n", token);
	sscanf(token,"%lx", &addr);
	TAU_VERBOSE("Backtrace Address determined to be %p\n", addr);
	if (i > 2) { /* first address is correct */
	  // Backtrace messes up and gives you the address of the next instruction.
	  // We subtract one to compensate for the off-by-one error.
	  addr=addr-Tau_get_backtrace_off_by_one_correction(); 
	}
      }
// **CWL** For correct operation with Comp_gnu.cpp
//#ifndef TAU_XLC
      // Map the addresses found in backtrace to actual code symbols and line information
      //   for addition to TAU_METADATA.
      Tau_Backtrace_writeMetadata(i, names[i], addr);
      //#endif /* TAU_XLC */
    }
    free(names);
  }
#endif /* TAU_EXECINFO */ 

  // **CWL** We must always allow the handler to write out data and invoke exit.
  fprintf(stderr, "TAU: Caught signal %d (%s), dumping profile with stack trace: [rank=%d, pid=%d, tid=%d]... \n", sig, strsignal(sig), RtsLayer::myNode(), getpid(), Tau_get_tid(), sig);
  TAU_METADATA("SIGNAL", strsignal(sig));
  TAU_PROFILE_EXIT("none");
  sleep(4);  // give the other tasks some time to process the handler and exit
  exit(1);     
}

static void tauSignalHandler(int sig) {
  fprintf (stderr, "Caught SIGUSR1, dumping TAU profile data\n");
  TAU_DB_DUMP_PREFIX("profile");
}

static void tauToggleInstrumentationHandler(int sig) {
  fprintf (stderr, "Caught SIGUSR2, toggling TAU instrumentation\n");
  if (RtsLayer::TheEnableInstrumentation()) {
    RtsLayer::TheEnableInstrumentation() = false;
  } else {
    RtsLayer::TheEnableInstrumentation() = true;
  }
}

#endif //TAU_DISABLE_SIGUSR

static int tau_initialized = 0;

extern "C" int Tau_init_check_initialized() {
  return tau_initialized;
}

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

#ifndef TAU_DISABLE_SIGUSR

int Tau_add_signal(int alarmType) {
    int ret = 0;

    struct sigaction act;
    memset(&act, 0, sizeof(struct sigaction));
    ret = sigemptyset(&act.sa_mask);
    if (ret != 0) {
      printf("TAU: Signal error: %s\n", strerror(ret));
      return -1;
    }

    ret = sigaddset(&act.sa_mask, alarmType);
    if (ret != 0) {
      printf("TAU: Signal error: %s\n", strerror(ret));
      return -1;
    }
    act.sa_sigaction = tauBacktraceHandler;
    act.sa_flags     = SA_SIGINFO;

    ret = sigaction(alarmType, &act, NULL);
    if (ret != 0) {
      printf("TAU: error adding signal in sigaction: %s\n", strerror(ret));
      return -1;
    }

    return ret;
}
//////////////////////////////////////////////////////////////////////
// Initialize signal handling routines
//////////////////////////////////////////////////////////////////////
extern "C" int Tau_signal_initialization() {
  if (TauEnv_get_track_signals()) {
    TAU_VERBOSE("TAU: Enable tracking of signals\n");

    Tau_add_signal(SIGILL);
    Tau_add_signal(SIGINT);
    Tau_add_signal(SIGQUIT);
    Tau_add_signal(SIGTERM);
    Tau_add_signal(SIGPIPE);
    Tau_add_signal(SIGSEGV);
    Tau_add_signal(SIGABRT);
    Tau_add_signal(SIGFPE);
    Tau_add_signal(SIGBUS);

  } /* TAU_TRACK_SIGNALS=1 */
  return 0; 
}

#endif // TAU_DISABLE_SIGUSR
static int initializing = 0;

// used if other components want to guard against operating while TAU is being
// initialized.
extern "C" int Tau_init_initializingTAU() {
	return initializing - tau_initialized;
}

extern "C" int Tau_init_initializeTAU() {

	//protect against reentrancy
  if (initializing) {
    return 0;
  }

  Tau_global_incr_insideTAU();
  
  initializing = 1;

  /* initialize the Profiler stack */
  Tau_stack_initialization();

  /* initialize environment variables */
  TauEnv_initialize();

#ifdef TAU_EPILOG
  /* no more initialization necessary if using epilog/scalasca */
  initializing = 1;
  Tau_init_epilog();
  return 0;
#endif


#ifdef TAU_SCOREP
  /* no more initialization necessary if using SCOREP */
  initializing = 1;
  SCOREP_Tau_InitMeasurement();
  SCOREP_Tau_RegisterExitCallback(Tau_profile_exit_all_threads); 
  return 0;
#endif

#ifdef TAU_VAMPIRTRACE
  /* no more initialization necessary if using vampirtrace */
  initializing = 1;
  Tau_init_vampirTrace();
  return 0;
#endif
  
  /* we need the timestamp of the "start" */
  Tau_snapshot_initialization();


#ifndef TAU_DISABLE_SIGUSR
  /* register SIGUSR1 handler */
  if (signal(SIGUSR1, tauSignalHandler) == SIG_ERR) {
    perror("failed to register TAU profile dump signal handler");
  }

  if (signal(SIGUSR2, tauToggleInstrumentationHandler) == SIG_ERR) {
    perror("failed to register TAU instrumentation toggle signal handler");
  }
#endif



  Tau_profiler_initialization();

  /********************************************/
  /* other initialization code should go here */
  /********************************************/

  /* initialize the metrics we will be counting */
  TauMetrics_init();

#ifndef TAU_DISABLE_SIGUSR
  Tau_signal_initialization();
#endif // TAU_DISABLE_SIGUSR

  /* TAU must me marked as initialized BEFORE Tau_compensate_initialize is called
     Otherwise re-entry to this function will take place and bad things will happen */
  initializing = 1;

  /* initialize compensation */
  if (TauEnv_get_compensate()) {
    Tau_compensate_initialization();
  }

  /* initialize signal handlers to flush the trace buffer */
  if (TauEnv_get_tracing()) {
    TauInitialize_kill_handlers();
  }
  
  //TauInitialize_kill_handlers();

  /* initialize sampling if requested */
  if (TauEnv_get_ebs_enabled()) {
    /* Work-around for MVAPHICH 2 to move sampling initialization to 
       after MPI_Init()
    */
#if !defined(TAU_MPI) && !defined(TAU_WINDOWS)
    Tau_sampling_init_if_necessary();
#endif /* TAU_MPI && TAU_WINDOWS */
  }
#ifdef TAU_PGI
  sbrk(102400);
#endif /* TAU_PGI */

#ifndef TAU_DISABLE_METADATA
	Tau_metadata_fillMetaData();
#endif

  tau_initialized = 1;
  Tau_global_decr_insideTAU();
  return 0;
}

//Rely on the dl auditor (src/wrapper/taupreload) to set dl_initialized if the audit feature is//available (GLIBC version 2.4 or greater).
#ifdef TAU_TRACK_LD_LOADER
static int dl_initialized = 0;
#else
static int dl_initialized = 1;
#endif

extern "C" void Tau_init_dl_initialized() {
	dl_initialized = 1;
}
extern "C" int Tau_init_check_dl_initialized() {
	return dl_initialized;
}

