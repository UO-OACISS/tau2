/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1994                  */
/*  Indiana University  University of Oregon  University of Rennes   */
/*********************************************************************/

/*
 * pcxx_event.c: simple SW monitor routines
 *
 * (c) 1994 Jerry Manic Saftware
 *
 * Version 3.0
 */

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <sys/types.h>
# include <fcntl.h>
# include <signal.h>

#ifdef TAU_WINDOWS
  #include <io.h>
  typedef __int64 x_int64;
  typedef unsigned __int64 x_uint64;
#else
  #include <unistd.h>
  #define O_BINARY 0
  typedef long long x_int64;
  typedef unsigned long long x_uint64;
#endif
# include <time.h>
# include <Profile/Profiler.h>


#ifdef TAU_LARGEFILE
  #define LARGEFILE_OPTION O_LARGEFILE
#else
  #define LARGEFILE_OPTION 0
#endif

#ifdef TAU_SYNCHRONIZE_CLOCKS
extern double TauSyncAdjustTimeStamp(double timestamp);
#endif

# define PCXX_EVENT_SRC 
# include "Profile/pcxx_events.h"

/* extern "C" time_t time(time_t * t);
 use time.h */

unsigned long int pcxx_ev_class = PCXX_EC_TRACER | PCXX_EC_TIMER;

/* -- event record buffer ------------------------------------ */
#define TAU_MAX_RECORDS 64*1024
#define TAU_BUFFER_SIZE sizeof(PCXX_EV)*TAU_MAX_RECORDS

/* -- buffer that holds the events before they are flushed to disk -- */
//static PCXX_EV TraceBuffer[TAU_MAX_THREADS][TAU_MAX_RECORDS]; 
static PCXX_EV *TraceBuffer[TAU_MAX_THREADS]; 
/* The second dimension shouldn't be TAU_BUFFER_SIZE ! */

/* -- id of the last record for each thread --- */
/* -- pointer to last available element of event record buffer */
/* -- need one place for flush event => - 1 ------------------ */
// static int  TauEventMax[TAU_MAX_THREADS] = {TAU_MAX_RECORDS - 1 };

/* -- current record pointer for each thread -- */
static int TauCurrentEvent[TAU_MAX_THREADS] = {0}; 

/* -- event trace file descriptor ---------------------------- */
static int TraceFd[TAU_MAX_THREADS] = {0};

/* -- do event files need to be re-written ------------------- */
static int FlushEvents[TAU_MAX_THREADS] = {0};

/* -- initialization status flags ---------------------------- */
static int TraceInitialized[TAU_MAX_THREADS] = {0};

#ifdef TAU_MULTIPLE_COUNTERS
static double tracerValues[MAX_TAU_COUNTERS] = {0};
#endif // TAU_MULTIPLE_COUNTERS


/* -- Use Profiling interface for time -- */
x_uint64 pcxx_GetUSecLong(int tid)
{ 
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
  x_uint64 value = RtsLayer::getUSecD(tid);
#endif // TAU_MULTIPLE_COUNTERS

#ifdef TAU_SYNCHRONIZE_CLOCKS
  return (x_uint64) TauSyncAdjustTimeStamp(value);
#else 
  return (x_uint64) value;
#endif
}

/* -- write event to buffer only [without overflow check] ---- */
void TraceEventOnly(long int ev, x_int64 par, int tid)
{
  PCXX_EV * pcxx_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]] ;  
  pcxx_ev_ptr->ev   = ev;
  pcxx_ev_ptr->ti   = pcxx_GetUSecLong(tid);
  pcxx_ev_ptr->par  = par;
  pcxx_ev_ptr->nid  = RtsLayer::myNode();
  pcxx_ev_ptr->tid  = tid;
  TauCurrentEvent[tid] ++;
}

/* -- Set the flag to flush the EDF file --------------------- */
void SetFlushEvents(int tid)
{
  FlushEvents[tid] = 1;
} 

/* -- Get the flag to flush the EDF file. 1 means flush edf file. ------ */
int GetFlushEvents(int tid)
{
  return FlushEvents[tid];
}

/* -- write event buffer to file ----------------------------- */
void TraceEvFlush(int tid)
{
/*
  static PCXX_EV flush_end = { PCXX_EV_FLUSH_EXIT, 0, 0, 0L };
*/
  int ret;
  if (TraceFd[tid] == 0)
  {
    printf("Error: TraceEvFlush(%d): Fd is -1. Trace file not initialized \n", tid);
    if (RtsLayer::myNode() == -1)
    {
      fprintf (stderr, "ERROR in configuration. Trace file not initialized. If this is an MPI application, please ensure that TAU MPI wrapper library is linked. If not, please ensure that TAU_PROFILE_SET_NODE(id); is called in the program (0 for sequential).\n");
      exit(1);
    }

  }

//#ifdef TRACEMONITORING
// Do this by default. 
  if (FlushEvents[tid])
  { /* Dump the EDF file before writing trace data: Monitoring */
    RtsLayer::DumpEDF(tid);
    FlushEvents[tid]=0;
  }
//#endif /* TRACEMONITORING */
  
  int numEventsToBeFlushed = TauCurrentEvent[tid]; /* starting from 0 */
  DEBUGPROFMSG("Tid "<<tid<<": TraceEvFlush()"<<endl;);
  if ( numEventsToBeFlushed != 0) /*-- there are a finite no. of records --*/
  {
    ret = write (TraceFd[tid], TraceBuffer[tid], (numEventsToBeFlushed) * sizeof(PCXX_EV));
    if (ret < 0) 
    {
#ifdef DEBUG_PROF
      printf("Error: TraceFd[%d] = %d, numEvents = %d ", tid, TraceFd[tid], 
	numEventsToBeFlushed);
      perror("Write Error in TraceEvFlush()");
#endif
    }
  }
  TauCurrentEvent[tid] = 0;
}

/* -- signal catching to flush event buffers ----------------- */
# ifndef NSIG
#   define NSIG 32
# endif
static SIGNAL_TYPE (*sighdlr[NSIG])(SIGNAL_ARG_TYPE);

static void wrap_up(int sig)
{
  fprintf (stderr, "signal %d on %d - flushing event buffer...\n", sig, RtsLayer::myNode());
  TAU_PROFILE_EXIT("signal");
  /* TraceEvFlush (RtsLayer::myThread());
   * */
  fprintf (stderr, "done.\n");
  /* if ( sighdlr[sig] != SIG_IGN ) (* sighdlr)(sig);
   */
  exit (1);
}

static void init_wrap_up()
{
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
/* -- TraceEvInit should be called in every trace routine to ensure that 
   the trace file is initialized -- */
int TraceEvInit(int tid)
{
   if (!TauBufferAllocated()[tid]) {
     TraceBuffer[tid] = (PCXX_EV*) malloc(TAU_BUFFER_SIZE);
     TauBufferAllocated()[tid] = true;
   }
  int retvalue = 0; 
  /* by default this is what is returned. No trace records were generated */
   
  if ( !(TraceInitialized[tid]) && (RtsLayer::myNode() > -1)) 
  /* node has been set*/ 
  { 
    /* done with initialization */
    TraceInitialized[tid] = 1;

    char *dirname, tracefilename[1024];
    if ((dirname = getenv("TRACEDIR")) == NULL) {
      // Use default directory name .
      dirname  = new char[8];
      strcpy (dirname,".");
    }
    sprintf(tracefilename, "%s/tautrace.%d.%d.%d.trc",dirname, 
	    RtsLayer::myNode(), RtsLayer::myContext(), tid);

    init_wrap_up ();

    if ((TraceFd[tid] = open (tracefilename, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND|O_BINARY|LARGEFILE_OPTION, 0600)) < 0)
    {
      fprintf (stderr, "TraceEvInit[open]: ");
      perror (tracefilename);
      exit (1);
    }
/* there may be some records in pcxx_ev_ptr already. Make sure that the
   first record has node id set properly */
    if (TraceBuffer[tid][0].ev == PCXX_EV_INIT)
    { /* first record is init */
      for (int iter = 0; iter < TauCurrentEvent[tid]; iter ++)
      {
        TraceBuffer[tid][iter].nid = RtsLayer::myNode();
      }
    }
    else
    { /* either the first record is blank - in which case we should
           put INIT record, or it is an error */
      if (TauCurrentEvent[tid] == 0) 
      { 
        TraceEvent(PCXX_EV_INIT, pcxx_ev_class, tid);
        retvalue ++; /* one record generated */
      }
      else
      { /* error */ 
        printf("Warning: TraceEvInit(%d): First record is not INIT\n", tid);
      }
    } /* first record was not INIT */
    
    if ( pcxx_ev_class & PCXX_EC_TRACER )
    { /* generate a wallclock time record */
      TraceEvent (PCXX_EV_WALL_CLOCK, time((time_t *)0), tid);
      retvalue ++;
    }
  }
  return retvalue; 
}

 /* This routine is typically invoked when multiple SET_NODE calls are 
    encountered for a multi-threaded program */ 
void TraceReinitialize(int oldid, int newid, int tid)
{
  printf("Inside TraceReinitialize : oldid = %d, newid = %d, tid = %d\n",
	oldid, newid, tid);
  /* We should put a record in the trace that says that oldid is mapped to newid this 
     way and have an offline program clean and transform it. Otherwise if we do it 
     online, we'd have to lock the multithreaded execution, and do if for all threads
     and this may perturb the application */

  return ;
}

void pcxx_EvInit(char *name)
{ /*-- dummy function for compatibility with the earlier ver. Remove later -- */ 
  TraceEvInit(RtsLayer::myThread());
} 

/* -- Reset the trace  --------------------------------------- */
void TraceUnInitialize(int tid)
{
/* -- to set the trace as uninitialized and clear the current buffers (for forked
      child process, trying to clear its parent records) -- */
   TraceInitialized[tid] = 0;
   TauCurrentEvent[tid] = 0;
   TraceEventOnly(PCXX_EV_INIT, pcxx_ev_class, tid);
}



/* -- write event to buffer ---------------------------------- */
void TraceEvent(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts)
{
  int i;
  int records_created = TraceEvInit(tid);
  PCXX_EV * pcxx_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]] ;  

#ifdef TAU_SYNCHRONIZE_CLOCKS
  ts = (x_uint64) TauSyncAdjustTimeStamp((double)ts);
#endif

  if (records_created)
  {
#ifdef DEBUG
    printf("TraceEvent(): TID %d records_created in TraceEvInit = %d\n",
	RtsLayer::myThread(), records_created);
#endif /* DEBUG */
    /* one or more records were created in TraceEvInit. We must initialize
    the timestamps of those records to the current timestamp. */
    if (use_ts)
    { /* we're asked to use the timestamp. Initialize with this ts */
      /* Initialize only records just above the current record! */
      for (i = 0; i < records_created; i++)
      { /* set the timestamp accordingly */
        TraceBuffer[tid][TauCurrentEvent[tid]-1-i].ti = ts; 
      }
    }
  }
  if (!(TraceInitialized[tid]) && (TauCurrentEvent[tid] == 0)) 
  /* not initialized  and its the first time */
  { 
    if (ev != PCXX_EV_INIT) 
    {
	/* we need to ensure that INIT is the first event */
      pcxx_ev_ptr->ev = PCXX_EV_INIT; 
      /* Should we use the timestamp provided to us? */
      if (use_ts)
      {
        pcxx_ev_ptr->ti   = ts;
      }
      else 
      {
        pcxx_ev_ptr->ti   = pcxx_GetUSecLong(tid);
      }
      pcxx_ev_ptr->par  = pcxx_ev_class; /* init event */ 
      /* probably the nodeid is not set yet */
      pcxx_ev_ptr->nid  = RtsLayer::myNode();
      pcxx_ev_ptr->tid  = tid;
 
      TauCurrentEvent[tid] ++;
      pcxx_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]];
    } 
  } 
        
  pcxx_ev_ptr->ev   = ev;
  if (use_ts)
  {
    pcxx_ev_ptr->ti   = ts;
  }
  else
  {
    pcxx_ev_ptr->ti   = pcxx_GetUSecLong(tid);
  }
  pcxx_ev_ptr->par  = par;
  pcxx_ev_ptr->nid  = RtsLayer::myNode();
  pcxx_ev_ptr->tid  = tid ;
  TauCurrentEvent[tid] ++;

  if ( TauCurrentEvent[tid] >= TAU_MAX_RECORDS - 1 ) TraceEvFlush(tid); 
}

void pcxx_Event(long int ev, x_int64 par)
{
  TraceEvent(ev, par, RtsLayer::myThread());
}
/* -- terminate SW tracing ----------------------------------- */
void TraceEvClose(int tid)
{
    if ( pcxx_ev_class & PCXX_EC_TRACER )
    {
      TraceEvent (PCXX_EV_CLOSE, 0, tid);
      TraceEvent (PCXX_EV_WALL_CLOCK, time((time_t *)0), tid);
    }
    TraceEvFlush (tid);
    //close (TraceFd[tid]); 
    // Just in case the same thread writes to this file again, don't close it.
    // for OpenMP.
#ifndef TAU_OPENMP
    if ((RtsLayer::myNode() == 0) && (RtsLayer::myThread() == 0))
      close(TraceFd[tid]);
#endif /* TAU_OPENMP */
}

void pcxx_EvClose(void)
{
  TraceEvClose(RtsLayer::myThread());
}

//////////////////////////////////////////////////////////////////////
// TraceCallStack is a recursive function that looks at the current
// Profiler and requests that all the previous profilers be traced prior
// to tracing the current profiler
//////////////////////////////////////////////////////////////////////
void TraceCallStack(int tid, Profiler *current)
{
  if (current == 0)
    return;
  else
  {
     // Trace all the previous records before tracing self
     TraceCallStack(tid, current->ParentProfiler);
     TraceEvent(current->ThisFunction->GetFunctionId(), 1, tid);
     DEBUGPROFMSG("TRACE CORRECTED: "<<current->ThisFunction->GetName()<<endl;);
  }
}


#if defined( TRACING_ON ) && defined( ARIADNE_SUPPORT )
/* Function to trace the events of Ariadne. */
void pcxx_AriadneTrace (long int event_class, long int event, int pid, int oid, int rwtype, int mtag, long long par)
{
/* This routine writes the ariadne events to the trace file */
long long trace_value = 0L; /* the first parameter to be traced */
long long parameter = 0L; /* dummy to shift the par by 32 bits */ 
/* Even for pC++ events we use U as the event rwtype and PCXX_... as the utag */
/* This way we can keep the old format for tracing :
	parameter (32), pid (10), oid (10), rwtype (4) , utag (8) 
for 64 bit long int */ 
  parameter = (long long) par; 

  if (sizeof (long long) == 8) 
  { /* This is true of SGI8K  */

    /* care has to be taken to ensure that mtag is 8 bits long */
  trace_value = (parameter << 32) | (pid << 22) | (oid << 12) | (rwtype << 8) | mtag;

  /*
  printf("Tracing ec = %lx, ev = %lx, pid = %d, oid = %d, mtag = %d, rwtype = %d, parameter = %d, trace_value = %ld\n", event_class, event, pid, oid, mtag, rwtype, parameter, trace_value);	
  */

  PCXX_EVENT(event_class, event, trace_value);
  } 
	
}

#endif  /* defined( TRACING_ON ) && defined( ARIADNE_SUPPORT ) */

/* eof */
