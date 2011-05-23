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
}

extern int tauPrintAddr(int i, char *token1, unsigned long addr);

#ifndef TAU_DISABLE_SIGUSR

//static void tauBacktraceHandler(int sig) {
void tauBacktraceHandler(int sig, siginfo_t *si, void *context) {

#ifdef TAU_EXECINFO
  static void *addresses[TAU_MAXSTACK];
  int n = backtrace( addresses, TAU_MAXSTACK );

  if (n < 2){
    printf("TAU: Backtrace not available!\n" );
  } else {
    TAU_VERBOSE("TAU: Backtrace:\n");
    char **names = backtrace_symbols( addresses, n );
    for ( int i = 2; i < n; i++ )
    {
      TAU_VERBOSE("stacktrace %s\n",names[i]);
      char *token1 = strtok(names[i],"[]");
      TAU_VERBOSE("found it: token1 = %s\n", token1);
      char *token2 = strtok(NULL,"[]");
      unsigned long addr;
      sscanf(token2,"%lx", &addr);
      TAU_VERBOSE("found it: addr = %lx\n", addr);
      tauPrintAddr(i, token1, addr);
    }
    fprintf(stderr, "TAU: Caught signal %d (%s), dumping profile with stack trace: [rank=%d, pid=%d, tid=%d]... \n", sig, strsignal(sig), RtsLayer::myNode(), getpid(), Tau_get_tid(), sig);
    TAU_METADATA("SIGNAL", strsignal(sig));
    TAU_PROFILE_EXIT("none");

    free(names);
    exit(1);
  }
#endif /* TAU_EXECINFO */ 
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
}
//////////////////////////////////////////////////////////////////////
// Initialize signal handling routines
//////////////////////////////////////////////////////////////////////
int Tau_signal_initialization() {
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
}

#endif // TAU_DISABLE_SIGUSR

extern "C" int Tau_init_initializeTAU() {
  static int initialized = 0;

  if (initialized) {
    return 0;
  }

  Tau_global_incr_insideTAU();
  
  initialized = 1;

  /* initialize the Profiler stack */
  Tau_stack_initialization();

  /* initialize environment variables */
  TauEnv_initialize();

#ifdef TAU_EPILOG
  /* no more initialization necessary if using epilog/scalasca */
  initialized = 1;
  Tau_init_epilog();
  return 0;
#endif


#ifdef TAU_SCOREP
  /* no more initialization necessary if using SCOREP */
  initialized = 1;
  SCOREP_Tau_InitMeasurement();
  SCOREP_Tau_RegisterExitCallback(Tau_profile_exit_all_threads); 
  return 0;
#endif

#ifdef TAU_VAMPIRTRACE
  /* no more initialization necessary if using vampirtrace */
  initialized = 1;
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
  initialized = 1;

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
    Tau_sampling_init(0);
#endif /* TAU_MPI && TAU_WINDOWS */
  }
#ifdef TAU_PGI
  sbrk(102400);
#endif /* TAU_PGI */

  tau_initialized = 1;
  Tau_global_decr_insideTAU();
  return 0;
}
