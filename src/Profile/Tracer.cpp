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

#include <tau_internal.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauTrace.h>
#include <Profile/TauMetrics.h>

#include <iostream>

using namespace std;

/* Magic number, parameter for certain events */
#define INIT_PARAM 3

/* Trace buffer settings */
#define TAU_MAX_RECORDS 64*1024
#define TAU_BUFFER_SIZE sizeof(TAU_EV)*TAU_MAX_RECORDS

/* Trace buffer */
static TAU_EV *TraceBuffer[TAU_MAX_THREADS]; 

/* Trace buffer pointer for each threads */
static int TauCurrentEvent[TAU_MAX_THREADS] = {0}; 

/* Trace file descriptors */
static int TauTraceFd[TAU_MAX_THREADS] = {0};

/* Flags for whether or not EDF files need to be rewritten when this thread's
   trace buffer is flushed.  Because any thread can introduce new functions and
   need to be flushed, we can't always wait for thread 0 */
static int TauTraceFlushEvents = 0;


/* Initialization status flags */
static int TauTraceInitialized[TAU_MAX_THREADS] = {0};
static int TraceFileInitialized[TAU_MAX_THREADS] = {0};

static double tracerValues[TAU_MAX_COUNTERS] = {0};


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


x_uint64 TauTraceGetTimeStamp(int tid) { 
  // If you're modifying the behavior of this routine, note that in 
  // Profiler::Start and Stop, we obtain the timestamp for tracing explicitly. 
  // The same changes would have to be made there as well (e.g., using COUNTER1
  // for tracing in multiplecounters case) for consistency.

  // In the presence of multiple counters, the system always
  // assumes that COUNTER1 contains the tracing metric.
  // Thus, if you want gettimeofday, make sure that you
  // define counter1 to be GETTIMEOFDAY.
  // Just return values[0] as that is the position of counter1 (whether it
  // is active or not).
  
  // THE SLOW WAY!
  //   RtsLayer::getUSecD(tid, tracerValues);
  //   double value = tracerValues[0];

  x_uint64 value = (x_uint64) TauMetrics_getTraceMetricValue(tid);

  if (TauEnv_get_synchronize_clocks()) {
    return (x_uint64) TauSyncAdjustTimeStamp(value);
  } else {
    return (x_uint64) value;
  }
}


/* Write event to buffer only [without overflow check] */
void TauTraceEventOnly(long int ev, x_int64 par, int tid) {
  TAU_EV * tau_ev_ptr = &TraceBuffer[tid][TauCurrentEvent[tid]] ;  
  tau_ev_ptr->ev   = ev;
  tau_ev_ptr->ti   = TauTraceGetTimeStamp(tid);
  tau_ev_ptr->par  = par;
  tau_ev_ptr->nid  = RtsLayer::myNode();
  tau_ev_ptr->tid  = tid;
  TauCurrentEvent[tid] ++;
}

/* Set the flag for flushing the EDF file, 1 means flush edf file. */
void TauTraceSetFlushEvents(int value) {
  RtsLayer::LockDB();
  TauTraceFlushEvents = value;
  RtsLayer::UnLockDB();
} 

/* Get the flag for flushing the EDF file, 1 means flush edf file. */
int TauTraceGetFlushEvents() {
  int val;
  RtsLayer::LockDB();
  val = TauTraceFlushEvents;
  RtsLayer::UnLockDB();
  return val;
}

/* Check that the trace file is initialized */
static int checkTraceFileInitialized(int tid) {
  if ( !(TraceFileInitialized[tid]) && (RtsLayer::myNode() > -1)) { 
    TraceFileInitialized[tid] = 1;
    const char *dirname;
    char tracefilename[1024];
    dirname = TauEnv_get_tracedir();
    sprintf(tracefilename, "%s/tautrace.%d.%d.%d.trc",dirname, 
	    RtsLayer::myNode(), RtsLayer::myContext(), tid);
    if ((TauTraceFd[tid] = open (tracefilename, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND|O_BINARY|LARGEFILE_OPTION, 0666)) < 0) {
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

/* Flush the trace buffer */
void TauTraceFlushBuffer(int tid) {
  checkTraceFileInitialized(tid);

  int ret;
  if (TauTraceFd[tid] == 0) {
    printf("Error: TauTraceFlush(%d): Fd is -1. Trace file not initialized \n", tid);
    if (RtsLayer::myNode() == -1) {
      fprintf (stderr, "ERROR in configuration. Trace file not initialized. If this is an MPI application, please ensure that TAU MPI wrapper library is linked. If not, please ensure that TAU_PROFILE_SET_NODE(id); is called in the program (0 for sequential).\n");
      exit(1);
    }
  }

  if (TauTraceGetFlushEvents()) { 
    /* Dump the EDF file before writing trace data */
    TauTraceDumpEDF(tid);
    TauTraceSetFlushEvents(0);
  }
  
  int numEventsToBeFlushed = TauCurrentEvent[tid]; /* starting from 0 */
  DEBUGPROFMSG("Tid "<<tid<<": TauTraceFlush()"<<endl;);
  if (numEventsToBeFlushed != 0) {
    ret = write (TauTraceFd[tid], TraceBuffer[tid], (numEventsToBeFlushed) * sizeof(TAU_EV));
    if (ret < 0) {
#ifdef DEBUG_PROF
      printf("Error: TauTraceFd[%d] = %d, numEvents = %d ", tid, TauTraceFd[tid], numEventsToBeFlushed);
      perror("Write Error in TauTraceFlush()");
#endif
    }
  }
  TauCurrentEvent[tid] = 0;
}


/* static list of flags specifying if the buffer has been allocated for a given thread */
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

/* Initialize tracing. TauTraceInit should be called in every trace routine to ensure that 
   the trace file is initialized */
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

/* Reset the trace */
void TauTraceUnInitialize(int tid) {
  /* to set the trace as uninitialized and clear the current buffers (for forked
     child process, trying to clear its parent records) */
  TauTraceInitialized[tid] = 0;
  TauCurrentEvent[tid] = 0;
  TauTraceEventOnly(TAU_EV_INIT, INIT_PARAM, tid);
}



/* Write event to buffer */
void TauTraceEventSimple(long int ev, x_int64 par, int tid) {
  TauTraceEvent(ev, par, tid, 0, 0);
}


/* Write event to buffer */
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
    TauTraceFlushBuffer(tid); 
  }
}

/* Close the trace */
void TauTraceClose(int tid) {
  TauTraceEventSimple (TAU_EV_CLOSE, 0, tid);
  TauTraceEventSimple (TAU_EV_WALL_CLOCK, time((time_t *)0), tid);
  TauTraceDumpEDF(tid);
  TauTraceFlushBuffer (tid);
  //close (TauTraceFd[tid]); 
  // Just in case the same thread writes to this file again, don't close it.
  // for OpenMP.
#ifndef TAU_OPENMP
  if ((RtsLayer::myNode() == 0) && (RtsLayer::myThread() == 0)) {
    close(TauTraceFd[tid]);
  }
#endif /* TAU_OPENMP */

  TauTraceMergeAndConvertTracesIfNecessary();
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



extern "C" double TauTraceGetTime(int tid) {
  // counter 0 is the one we use
  double value = TauMetrics_getTraceMetricValue(tid);
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


/* Write event definition file (EDF) */
int TauTraceDumpEDF(int tid) {
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator uit;
  char filename[1024], errormsg[1024];
  const char *dirname;
  FILE* fp;
  int  numEvents, numExtra;
  
  RtsLayer::LockDB();

  if (tid != 0) { 
    if (TauTraceGetFlushEvents() == 0) {
      RtsLayer::UnLockDB();
      return 1; 
    }
  }

  dirname = TauEnv_get_tracedir();
  
  sprintf(filename,"%s/events.%d.edf",dirname, RtsLayer::myNode());
  if ((fp = fopen (filename, "w+")) == NULL) {
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    RtsLayer::UnLockDB();
    return -1;
  }
  
  // Data Format 
  // <no.> events
  // # or \n ignored
  // %s %s %d "%s %s" %s 
  // id group tag "name type" parameters
  
  numEvents = TheFunctionDB().size() + TheEventDB().size();
#ifdef TAU_GPU 
  numExtra = 13; // Added four ONESIDED msg events
#else
  numExtra = 9; // Number of extra events
#endif	
  numEvents += numExtra;
  
  fprintf(fp,"%d dynamic_trace_events\n", numEvents);
  
  fprintf(fp,"# FunctionId Group Tag \"Name Type\" Parameters\n");
  
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    fprintf(fp, "%ld %s 0 \"%s %s\" EntryExit\n", (*it)->GetFunctionId(),
	    (*it)->GetPrimaryGroup(), (*it)->GetName(), (*it)->GetType() );
  }
  
  /* Now write the user defined event */
  for (uit = TheEventDB().begin(); uit != TheEventDB().end(); uit++) {
    int monoInc = 0; 
    if ((*uit)->GetMonotonicallyIncreasing()) { 
      monoInc = 1;
    }
    fprintf(fp, "%ld TAUEVENT %d \"%s\" TriggerValue\n", (*uit)->GetEventId(), monoInc, (*uit)->GetEventName());
  }

  // Now add the nine extra events 
  fprintf(fp,"%ld TRACER 0 \"EV_INIT\" none\n", (long) TAU_EV_INIT); 
  fprintf(fp,"%ld TRACER 0 \"FLUSH_ENTER\" none\n", (long) TAU_EV_FLUSH_ENTER); 
  fprintf(fp,"%ld TRACER 0 \"FLUSH_EXIT\" none\n", (long) TAU_EV_FLUSH_EXIT); 
  fprintf(fp,"%ld TRACER 0 \"FLUSH_CLOSE\" none\n", (long) TAU_EV_CLOSE); 
  fprintf(fp,"%ld TRACER 0 \"FLUSH_INITM\" none\n", (long) TAU_EV_INITM); 
  fprintf(fp,"%ld TRACER 0 \"WALL_CLOCK\" none\n", (long) TAU_EV_WALL_CLOCK); 
  fprintf(fp,"%ld TRACER 0 \"CONT_EVENT\" none\n", (long) TAU_EV_CONT_EVENT); 
  fprintf(fp,"%ld TAU_MESSAGE -7 \"MESSAGE_SEND\" par\n", (long) TAU_MESSAGE_SEND); 
  fprintf(fp,"%ld TAU_MESSAGE -8 \"MESSAGE_RECV\" par\n", (long) TAU_MESSAGE_RECV); 

#ifdef TAU_GPU
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE_SEND\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_SEND); 
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE_RECV\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_RECV); 
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE_ID_TriggerValueT1\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_ID_1); 
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE_ID_TriggerValueT2\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_ID_2); 
#endif
  
  fclose(fp);
  RtsLayer::UnLockDB();
  return 0;
}



//////////////////////////////////////////////////////////////////////
// MergeAndConvertTracesIfNecessary does just that!
//////////////////////////////////////////////////////////////////////

int TauTraceMergeAndConvertTracesIfNecessary(void) { 
  char *outfile;

  outfile = getenv("TAU_TRACEFILE");

  if (outfile == NULL) {
    /* output file not defined, just exit normally */
    return 0;
  }

  /* output file is defined. We need to merge the traces */
  /* Now, who does the merge and conversion? */
  if ((RtsLayer::myNode() != 0) || (RtsLayer::myThread() != 0)) {
    /* only node/thread 0 should do this */
    return 0;
  }
  
  const char *outdir;
  char *keepfiles;
  char cmd[1024];
  char rmcmd[256]; 
  char cdcmd[1024];
  const char *tauroot=TAUROOT;
  const char *tauarch=TAU_ARCH;
  const char *conv="tau2vtf";
  char converter[1024] = {0}; 
  FILE *in;
  
  /* If we can't find tau2vtf, use tau_convert! */
  sprintf(converter, "%s/%s/bin/%s",tauroot, tauarch, conv);
  if ((in = fopen(converter, "r")) == NULL) {
    sprintf(converter, "%s/%s/bin/tau_convert", tauroot, tauarch);
  } else {
    fclose(in);
  }
  
  /* Should we get rid of intermediate trace files? */
  if ((keepfiles = getenv("TAU_KEEP_TRACEFILES")) == NULL) {
    strcpy(rmcmd, "/bin/rm -f app12345678.trc tautrace.*.trc tau.edf events.*.edf");
  } else { 
    strcpy(rmcmd," "); /* NOOP */
  }
  
  /* Next, look for trace directory */
  outdir = TauEnv_get_tracedir();
  sprintf(cdcmd, "cd %s;", outdir);
  
  /* create the command */
  sprintf(cmd, "%s /bin/rm -f app12345678.trc; %s/%s/bin/tau_merge tautrace.*.trc app12345678.trc; %s app12345678.trc tau.edf %s; %s", cdcmd,tauroot, tauarch, converter, outfile, rmcmd);
#ifdef DEBUG_PROF
  printf("The merge/convert cmd is: %s\n", cmd);
#endif /* DEBUG_PROF */
  
  /* and execute it */
#ifndef TAU_CATAMOUNT
  /* NOTE: BGL will not execute this code as well because the compute node 
     kernels cannot fork tasks. So, on BGL, nothing will happen when the 
     following system command executes */
  system(cmd);
#endif /* TAU_CATAMOUNT */

  return 0;
}

#ifdef TAU_GPU

void TauTraceOneSidedMsg(bool type, gpuId *gpu, int length, int threadId)
{
		/* there are three user events that make up a one-sided msg */
		if (type == MESSAGE_SEND)
    	TauTraceEventSimple(TAU_ONESIDED_MESSAGE_SEND, length, threadId); 
		else
    	TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, length, threadId); 
    TauTraceEventSimple(TAU_ONESIDED_MESSAGE_ID_1, gpu->id_p1(), threadId); 
    TauTraceEventSimple(TAU_ONESIDED_MESSAGE_ID_2, gpu->id_p2(), threadId); 
}

#endif

//////////////////////////////////////////////////////////////////////
// TraceSendMsg traces the message send
//////////////////////////////////////////////////////////////////////
void TauTraceSendMsg(int type, int destination, int length) {
  x_int64 parameter;
  x_uint64 xother, xtype, xlength, xcomm;

  if (RtsLayer::isEnabled(TAU_MESSAGE)) {
    parameter = 0;
    /* for send, othernode is receiver or destination */
    xtype = type;
    xlength = length;
    xother = destination;
    xcomm = 0;

    /* Format for parameter is
       63 ..... 56 55 ..... 48 47............. 32
          other       type          length

       These are the high order bits, below are the low order bits

       31 ..... 24 23 ..... 16 15..............0
          other       type          length       

       e.g.

       xtype = 0xAABB;
       xother = 0xCCDD;
       xlength = 0xDEADBEEF;
       result = 0xccaaDEADdddbbBEEF

     parameter = ((xlength >> 16) << 32) | 
       ((xtype >> 8 & 0xFF) << 48) |
       ((xother >> 8 & 0xFF) << 56) |
       (xlength & 0xFFFF) | 
       ((xtype & 0xFF)  << 16) | 
       ((xother & 0xFF) << 24);

     */

    parameter = (xlength >> 16 << 54 >> 22) |
      ((xtype >> 8 & 0xFF) << 48) |
      ((xother >> 8 & 0xFF) << 56) |
      (xlength & 0xFFFF) | 
      ((xtype & 0xFF)  << 16) | 
      ((xother & 0xFF) << 24) |
      (xcomm << 58 >> 16);

    TauTraceEventSimple(TAU_MESSAGE_SEND, parameter, RtsLayer::myThread()); 
  } 
}

  
//////////////////////////////////////////////////////////////////////
// TraceRecvMsg traces the message recv
//////////////////////////////////////////////////////////////////////
void TauTraceRecvMsg(int type, int source, int length) {
  x_int64 parameter;
  x_uint64 xother, xtype, xlength, xcomm;

  if (RtsLayer::isEnabled(TAU_MESSAGE)) {
    parameter = 0;
    /* for recv, othernode is sender or source*/
    xtype = type;
    xlength = length;
    xother = source;
    xcomm = 0;

    // see TraceSendMsg for documentation

    parameter = (xlength >> 16 << 54 >> 22) |
      ((xtype >> 8 & 0xFF) << 48) |
      ((xother >> 8 & 0xFF) << 56) |
      (xlength & 0xFFFF) | 
      ((xtype & 0xFF)  << 16) | 
      ((xother & 0xFF) << 24) |
      (xcomm << 58 >> 16);

    TauTraceEventSimple(TAU_MESSAGE_RECV, parameter, RtsLayer::myThread()); 
  }
}

