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
# include <unistd.h>
# include <Profile/Profiler.h>


# define PCXX_EVENT_SRC 
# include "Profile/pcxx_events.h"

extern "C" time_t time(time_t * t);

unsigned long int pcxx_ev_class = PCXX_EC_TRACER | PCXX_EC_TIMER;

/* -- event record buffer ------------------------------------ */
#define TAU_MAX_RECORDS 64*1024
#define TAU_BUFFER_SIZE sizeof(PCXX_EV)*TAU_MAX_RECORDS

/* -- buffer that holds the events before they are flushed to disk -- */
static PCXX_EV TraceBuffer[TAU_MAX_THREADS][TAU_MAX_RECORDS]; 
/* The second dimension shouldn't be TAU_BUFFER_SIZE ! */

/* -- id of the last record for each thread --- */
/* -- pointer to last available element of event record buffer */
/* -- need one place for flush event => - 1 ------------------ */
// static int  TauEventMax[TAU_MAX_THREADS] = {TAU_MAX_RECORDS - 1 };

/* -- current record pointer for each thread -- */
static int  TauCurrentEvent[TAU_MAX_THREADS] = {0}; 

/* -- event trace file descriptor ---------------------------- */
static int TraceFd[TAU_MAX_THREADS] = {0};

/* -- initialization status flags ---------------------------- */
static int TraceInitialized[PCXX_MAXPROCS] = {0};


/* -- Use Profiling interface for time -- */
unsigned long long pcxx_GetUSecLong(int tid)
{ 
  return (unsigned long long) RtsLayer::getUSecD(tid); 
}

/* -- write event to buffer only [without overflow check] ---- */
void TraceEventOnly(long int ev,long long par, int tid)
{
  PCXX_EV * pcxx_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]] ;  
  pcxx_ev_ptr->ev   = ev;
  pcxx_ev_ptr->ti   = pcxx_GetUSecLong(tid);
  pcxx_ev_ptr->par  = par;
  pcxx_ev_ptr->nid  = RtsLayer::myNode();
  pcxx_ev_ptr->tid  = tid;
  TauCurrentEvent[tid] ++;
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
    TraceEvInit(tid);
  }
  
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
  TraceEvFlush (RtsLayer::myThread());
  fprintf (stderr, "done.\n");
  if ( sighdlr[sig] != SIG_IGN ) (* sighdlr)(sig);
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
/* -- initialize SW monitor and open trace file(s) ----------- */
/* -- TraceEvInit should be called in every trace routine to ensure that 
   the trace file is initialized -- */
void TraceEvInit(int tid)
{
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

    if ((TraceFd[tid] = open (tracefilename, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND, 0600)) < 0)
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
      }
      else
      { /* error */ 
        printf("Warning: TraceEvInit(%d): First record is not INIT\n", tid);
      }
    } /* first record was not INIT */
    
    if ( pcxx_ev_class & PCXX_EC_TRACER )
      TraceEvent (PCXX_EV_WALL_CLOCK, time((time_t *)0), tid);
  }
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
void TraceEvent(long int ev, long long par, int tid)
{
  TraceEvInit(tid);
  PCXX_EV * pcxx_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]] ;  
  if (!(TraceInitialized[tid]) && (TauCurrentEvent[tid] == 0)) 
  /* not initialized  and its the first time */
  { 
    if (ev != PCXX_EV_INIT) 
    {
	/* we need to ensure that INIT is the first event */
      pcxx_ev_ptr->ev = PCXX_EV_INIT; 
      pcxx_ev_ptr->ti   = pcxx_GetUSecLong(tid);
      pcxx_ev_ptr->par  = pcxx_ev_class; /* init event */ 
      /* probably the nodeid is not set yet */
      pcxx_ev_ptr->nid  = RtsLayer::myNode();
      pcxx_ev_ptr->tid  = tid;
 
      TauCurrentEvent[tid] ++;
      pcxx_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]];
    } 
  } 
        
  pcxx_ev_ptr->ev   = ev;
  pcxx_ev_ptr->ti   = pcxx_GetUSecLong(tid);
  pcxx_ev_ptr->par  = par;
  pcxx_ev_ptr->nid  = RtsLayer::myNode();
  pcxx_ev_ptr->tid  = tid ;
  TauCurrentEvent[tid] ++;

  if ( TauCurrentEvent[tid] >= TAU_MAX_RECORDS - 1 ) TraceEvFlush(tid); 
}

void pcxx_Event(long int ev, long long par)
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
