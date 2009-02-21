/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2009  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: Tracer.cpp 			        	   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : TAU Tracing                                      **
**                                                                         **
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>

#include <tau_library.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauTrace.h>

/* Magic number, parameter for certain events */
#define INIT_PARAM 3

/* -- event record buffer ------------------------------------ */
#define TAU_MAX_RECORDS 64*1024
#define TAU_BUFFER_SIZE sizeof(TAU_EV)*TAU_MAX_RECORDS

/* -- buffer that holds the events before they are flushed to disk -- */
static TAU_EV *TraceBuffer[TAU_MAX_THREADS]; 

/* -- current record pointer for each thread -- */
static int TauCurrentEvent[TAU_MAX_THREADS] = {0}; 

/* -- event trace file descriptor ---------------------------- */
static int TraceFd[TAU_MAX_THREADS] = {0};

/* -- do event files need to be re-written ------------------- */
static int FlushEvents[TAU_MAX_THREADS] = {0};

/* -- initialization status flags ---------------------------- */
static int TauTraceInitialized[TAU_MAX_THREADS] = {0};
static int TraceFileInitialized[TAU_MAX_THREADS] = {0};

#ifdef TAU_MULTIPLE_COUNTERS
static double tracerValues[MAX_TAU_COUNTERS] = {0};
#endif // TAU_MULTIPLE_COUNTERS


double TauSyncAdjustTimeStamp(double timestamp) {
  TauTraceOffsetInfo *offsetInfo = TheTauTraceOffsetInfo();

  if (offsetInfo->enabled == 1) {
    timestamp = timestamp - offsetInfo->beginOffset + offsetInfo->syncOffset;
    return timestamp;
  } else {
    // return 0 until sync'd
    return 0.0;
  }
}


/* -- Use Profiling interface for time -- */
x_uint64 TauTraceGetTimeStamp(int tid) { 
  // If you're modifying the behavior of this routine, note that in 
  // Profiler::Start and Stop, we obtain the timestamp for tracing explicitly. 
  // The same changes would have to be made there as well (e.g., using COUNTER1
  // for tracing in multiplecounters case) for consistency.
#ifdef TAU_MULTIPLE_COUNTERS
  //In the presence of multiple counters, the system always
  //assumes that counter1 contains the tracing metric.
  //Thus, if you want gettimeofday, make sure that you
  //define counter1 to be GETTIMEOFDAY.
  //Just return values[0] as that is the position of counter1 (whether it
  //is active or not).

// THE SLOW WAY!
//   RtsLayer::getUSecD(tid, tracerValues);
//   double value = tracerValues[0];

  x_uint64 value = MultipleCounterLayer::getSingleCounter(tid, 0);

#else //TAU_MULTIPLE_COUNTERS
  x_uint64 value = (x_uint64) RtsLayer::getUSecD(tid);
#endif // TAU_MULTIPLE_COUNTERS

  if (TauEnv_get_synchronize_clocks()) {
    return (x_uint64) TauSyncAdjustTimeStamp(value);
  } else {
    return (x_uint64) value;
  }
}

/* -- write event to buffer only [without overflow check] ---- */
void TauTraceEventOnly(long int ev, x_int64 par, int tid) {
  TAU_EV * tau_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]] ;  
  tau_ev_ptr->ev   = ev;
  tau_ev_ptr->ti   = TauTraceGetTimeStamp(tid);
  tau_ev_ptr->par  = par;
  tau_ev_ptr->nid  = RtsLayer::myNode();
  tau_ev_ptr->tid  = tid;
  TauCurrentEvent[tid] ++;
}

/* -- Set the flag to flush the EDF file --------------------- */
void TauTraceSetFlushEvents(int tid) {
  FlushEvents[tid] = 1;
} 

/* -- Get the flag to flush the EDF file. 1 means flush edf file. ------ */
int TauTraceGetFlushEvents(int tid) {
  return FlushEvents[tid];
}

static int checkTraceFileInitialized(int tid) {
  if ( !(TraceFileInitialized[tid]) && (RtsLayer::myNode() > -1)) { 
    TraceFileInitialized[tid] = 1;
    const char *dirname;
    char tracefilename[1024];
    dirname = TauEnv_get_tracedir();
    sprintf(tracefilename, "%s/tautrace.%d.%d.%d.trc",dirname, 
	    RtsLayer::myNode(), RtsLayer::myContext(), tid);
    if ((TraceFd[tid] = open (tracefilename, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND|O_BINARY|LARGEFILE_OPTION, 0600)) < 0) {
      fprintf (stderr, "TAU: TauTraceInit[open]: ");
      perror (tracefilename);
      exit (1);
    }

    if (TraceBuffer[tid][0].ev == TAU_EV_INIT) { 
      /* first record is init */
      for (int iter = 0; iter < TauCurrentEvent[tid]; iter ++) {
	TraceBuffer[tid][iter].nid = RtsLayer::myNode();
      }
    }
  }
  return 0;
}

/* -- write event buffer to file ----------------------------- */
void TauTraceFlush(int tid) {
  checkTraceFileInitialized(tid);
/*
  static TAU_EV flush_end = { TAU_EV_FLUSH_EXIT, 0, 0, 0L };
*/
  int ret;
  if (TraceFd[tid] == 0) {
    printf("Error: TauTraceFlush(%d): Fd is -1. Trace file not initialized \n", tid);
    if (RtsLayer::myNode() == -1) {
      fprintf (stderr, "ERROR in configuration. Trace file not initialized. If this is an MPI application, please ensure that TAU MPI wrapper library is linked. If not, please ensure that TAU_PROFILE_SET_NODE(id); is called in the program (0 for sequential).\n");
      exit(1);
    }

  }

  if (FlushEvents[tid]) { 
    /* Dump the EDF file before writing trace data */
    RtsLayer::DumpEDF(tid);
    FlushEvents[tid]=0;
  }
  
  int numEventsToBeFlushed = TauCurrentEvent[tid]; /* starting from 0 */
  DEBUGPROFMSG("Tid "<<tid<<": TauTraceFlush()"<<endl;);
  if (numEventsToBeFlushed != 0) {
    /*-- there are a finite no. of records --*/
    ret = write (TraceFd[tid], TraceBuffer[tid], (numEventsToBeFlushed) * sizeof(TAU_EV));
    if (ret < 0) {
#ifdef DEBUG_PROF
      printf("Error: TraceFd[%d] = %d, numEvents = %d ", tid, TraceFd[tid], 
	numEventsToBeFlushed);
      perror("Write Error in TauTraceFlush()");
#endif
    }
  }
  TauCurrentEvent[tid] = 0;
}

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
  fprintf (stderr, "TAU: signal %d on %d - flushing event buffer...\n", sig, RtsLayer::myNode());
  TAU_PROFILE_EXIT("signal");
  fprintf (stderr, "TAU: done.\n");
  exit (1);
}

static void init_wrap_up() {
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



/* -- current record pointer for each thread -- */

bool *TauBufferAllocated() {
  static bool flag = true;
  static bool allocated[TAU_MAX_THREADS];
  if (flag) {
    for (int i=0; i < TAU_MAX_THREADS; i++) {
      allocated[i] = false;
    }
    flag = false;
  }
  return allocated;
}

/* -- initialize SW monitor and open trace file(s) ----------- */
/* -- TauTraceInit should be called in every trace routine to ensure that 
   the trace file is initialized -- */
int TauTraceInit(int tid) {
   if (!TauBufferAllocated()[tid]) {
     TraceBuffer[tid] = (TAU_EV*) malloc(TAU_BUFFER_SIZE);
     TauBufferAllocated()[tid] = true;
   }
  int retvalue = 0; 
  /* by default this is what is returned. No trace records were generated */
   
  if ( !(TauTraceInitialized[tid]) && (RtsLayer::myNode() > -1)) {
  /* node has been set*/ 
    /* done with initialization */
    TauTraceInitialized[tid] = 1;

    init_wrap_up ();
  
    /* there may be some records in tau_ev_ptr already. Make sure that the
       first record has node id set properly */
    if (TraceBuffer[tid][0].ev == TAU_EV_INIT) { 
      /* first record is init */
      for (int iter = 0; iter < TauCurrentEvent[tid]; iter ++) {
        TraceBuffer[tid][iter].nid = RtsLayer::myNode();
      }
    } else {
      /* either the first record is blank - in which case we should
	 put INIT record, or it is an error */
      if (TauCurrentEvent[tid] == 0) { 
        TauTraceEventSimple(TAU_EV_INIT, INIT_PARAM, tid);
        retvalue ++; /* one record generated */
      } else { 
	/* error */ 
        printf("Warning: TauTraceInit(%d): First record is not INIT\n", tid);
      }
    } /* first record was not INIT */
    
    /* generate a wallclock time record */
    TauTraceEventSimple (TAU_EV_WALL_CLOCK, time((time_t *)0), tid);
    retvalue ++;
  }
  return retvalue; 
}

 /* This routine is typically invoked when multiple SET_NODE calls are 
    encountered for a multi-threaded program */ 
void TauTraceReinitialize(int oldid, int newid, int tid) {
#ifndef TAU_SETNODE0
  printf("Inside TauTraceReinitialize : oldid = %d, newid = %d, tid = %d\n",
	oldid, newid, tid);
#endif
  /* We should put a record in the trace that says that oldid is mapped to newid this 
     way and have an offline program clean and transform it. Otherwise if we do it 
     online, we'd have to lock the multithreaded execution, and do if for all threads
     and this may perturb the application */

  return ;
}

void tau_EvInit(char *name) { 
  /*-- dummy function for compatibility with the earlier ver. Remove later -- */ 
  TauTraceInit(RtsLayer::myThread());
} 

/* -- Reset the trace  --------------------------------------- */
void TauTraceUnInitialize(int tid) {
/* -- to set the trace as uninitialized and clear the current buffers (for forked
      child process, trying to clear its parent records) -- */
   TauTraceInitialized[tid] = 0;
   TauCurrentEvent[tid] = 0;
   TauTraceEventOnly(TAU_EV_INIT, INIT_PARAM, tid);
}



/* -- write event to buffer ---------------------------------- */
void TauTraceEventSimple(long int ev, x_int64 par, int tid) {
  TauTraceEvent(ev, par, tid, 0, 0);
}


void TauTraceEvent(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts) {
  int i;
  int records_created = TauTraceInit(tid);
  TAU_EV *event = &TraceBuffer[tid][TauCurrentEvent[tid]];  

  if (TauEnv_get_synchronize_clocks()) {
    ts = (x_uint64) TauSyncAdjustTimeStamp((double)ts);
  }

  if (records_created) {
    /* one or more records were created in TauTraceInit. We must initialize
    the timestamps of those records to the current timestamp. */
    if (use_ts) {
      /* we're asked to use the timestamp. Initialize with this ts */
      /* Initialize only records just above the current record! */
      for (i = 0; i < records_created; i++) {
	/* set the timestamp accordingly */
        TraceBuffer[tid][TauCurrentEvent[tid]-1-i].ti = ts; 
      }
    }
  }

  if (!(TauTraceInitialized[tid]) && (TauCurrentEvent[tid] == 0)) {
  /* not initialized  and its the first time */
    if (ev != TAU_EV_INIT) {
	/* we need to ensure that INIT is the first event */
      event->ev = TAU_EV_INIT; 
      /* Should we use the timestamp provided to us? */
      if (use_ts) {
        event->ti = ts;
      } else {
        event->ti = TauTraceGetTimeStamp(tid);
      }
      event->par = INIT_PARAM; /* init event */ 
      /* probably the nodeid is not set yet */
      event->nid = RtsLayer::myNode();
      event->tid = tid;
 
      TauCurrentEvent[tid] ++;
      event = &TraceBuffer[tid][TauCurrentEvent[tid]];
    } 
  } 
        
  event->ev  = ev;
  if (use_ts) {
    event->ti = ts;
  } else {
    event->ti = TauTraceGetTimeStamp(tid);
  }
  event->par = par;
  event->nid = RtsLayer::myNode();
  event->tid = tid ;
  TauCurrentEvent[tid]++;

  if (TauCurrentEvent[tid] >= TAU_MAX_RECORDS-1) {
    TauTraceFlush(tid); 
  }
}

/* -- terminate SW tracing ----------------------------------- */
void TauTraceClose(int tid) {
  TauTraceEventSimple (TAU_EV_CLOSE, 0, tid);
  TauTraceEventSimple (TAU_EV_WALL_CLOCK, time((time_t *)0), tid);
  TauTraceFlush (tid);
  //close (TraceFd[tid]); 
  // Just in case the same thread writes to this file again, don't close it.
  // for OpenMP.
#ifndef TAU_OPENMP
  if ((RtsLayer::myNode() == 0) && (RtsLayer::myThread() == 0)) {
    close(TraceFd[tid]);
  }
#endif /* TAU_OPENMP */
}

//////////////////////////////////////////////////////////////////////
// TraceCallStack is a recursive function that looks at the current
// Profiler and requests that all the previous profilers be traced prior
// to tracing the current profiler
//////////////////////////////////////////////////////////////////////
void TraceCallStack(int tid, Profiler *current) {
  if (current == 0) {
    return;
  } else {
    // Trace all the previous records before tracing self
    TraceCallStack(tid, current->ParentProfiler);
    TauTraceEventSimple(current->ThisFunction->GetFunctionId(), 1, tid);
    DEBUGPROFMSG("TRACE CORRECTED: "<<current->ThisFunction->GetName()<<endl;);
  }
}



double TauTraceGetTime(int tid) {
#ifdef TAU_MULTIPLE_COUNTERS
  // counter 0 is the one we use
  double value = MultipleCounterLayer::getSingleCounter(tid, 0);
#else
  double value = RtsLayer::getUSecD(tid);
#endif
  return value;
}


TauTraceOffsetInfo *TheTauTraceOffsetInfo() {
  static int init = 1;
  static TauTraceOffsetInfo offsetInfo;
  if (init) {
    init = 0;
    offsetInfo.enabled = 0;
    offsetInfo.beginOffset = 0.0;
    offsetInfo.syncOffset = -1.0;
  }
  return &offsetInfo;
}




