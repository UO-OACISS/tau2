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
#include <Profile/TauTraceOTF2.h>
#include <Profile/TauMetrics.h>

#include <iostream>
#include <vector>
#include <atomic>

using namespace std;
using namespace tau;

/* Magic number, parameter for certain events */
#define INIT_PARAM 3

/* Trace buffer settings */
#define TAU_BUFFER_SIZE sizeof(TAU_EV)*TAU_MAX_RECORDS

extern "C" int Tau_get_usesMPI(void);
static unsigned long long TauMaxTraceRecords = 0;
static int TauBufferSize = 0;

struct TraceThreadData
{
	TAU_EV *TraceBuffer=NULL;
	unsigned int TauCurrentEvent=0;
	int TauTraceFd=0;
	int TauTraceInitialized=0;
	int TraceFileInitialized=0;
	bool allocated=false;
};

struct TraceThreadVector : vector<TraceThreadData *>{
    TraceThreadVector() {
        // nothing
    }

    virtual ~TraceThreadVector(){
        //destructed=true;
        Tau_destructor_trigger();
    }
};


struct TraceThreadVector_local : vector<TraceThreadData *>{
    TraceThreadVector_local() {
        // nothing
    }

    virtual ~TraceThreadVector_local(){
        //destructed=true;
        //Tau_destructor_trigger();
    }
};

std::mutex TraceThreadListMutex;
static TraceThreadVector ThreadList;
static thread_local TraceThreadVector_local  ThreadListCache; 

static TraceThreadData* getTracerVector(int tid){

  TraceThreadData* TVOut=NULL;
  static thread_local const unsigned int local_tid = RtsLayer::myThread();
  
  if(tid!=0 && (tid == local_tid)){// use_metric_tls && !destructed && !destructed_local) {
        if(ThreadListCache.size() > tid) {
            TVOut = ThreadListCache.operator[](tid);
            if(TVOut != NULL) {
                return TVOut;
            }
        }
    }
   //if(destructed_local || destructed){return MOut;}
  
  
      // Not in thread-local cache, or cache not searched.
    // Create a new FunctionMetrics instance.
  std::lock_guard<std::mutex> guard(TraceThreadListMutex);
	while(ThreadList.size()<=tid){
		ThreadList.push_back(new TraceThreadData());
	}
    
    TVOut=ThreadList[tid];

    // Use thread-local optimization if the current thread is requesting its own metrics.
    if(tid !=0 && (tid == local_tid)) { //use_metric_tls && 
        // Ensure the FMetricList vector is long enough to accomodate the new cached item.
        while(ThreadListCache.size() <= tid) {
            ThreadListCache.push_back(NULL);
        }    
        // Store the FunctionMetrics pointer in the thread-local cache
        ThreadListCache.operator[](tid) = TVOut;
    } 
 return TVOut;
}

/* Trace buffer */


//static TAU_EV *TraceBuffer[TAU_MAX_THREADS]; 
static inline TAU_EV* getTraceBuffer(int tid){
	//checkTracerVector(tid);
	return getTracerVector(tid)->TraceBuffer;
}
static inline void setTraceBuffer(int tid, TAU_EV* value){
	//checkTracerVector(tid);
	getTracerVector(tid)->TraceBuffer=value;
}

/* Trace buffer pointer for each threads */
//static unsigned int TauCurrentEvent[TAU_MAX_THREADS] = {0}; 
static inline unsigned int getTauCurrentEvent(int tid){
	//checkTracerVector(tid);
	return getTracerVector(tid)->TauCurrentEvent;
}
static inline void incrementTauCurrentEvent(int tid){
	//checkTracerVector(tid);
	getTracerVector(tid)->TauCurrentEvent++;
}
static inline void resetTauCurrentEvent(int tid){
	//checkTracerVector(tid);
	getTracerVector(tid)->TauCurrentEvent=0;
}

/* Trace file descriptors */
//static int TauTraceFd[TAU_MAX_THREADS] = {0};
static inline int setTauTraceFd(int tid, int value){
	//checkTracerVector(tid);
	getTracerVector(tid)->TauTraceFd=value;
	return value;
}
static inline int getTauTraceFd(int tid){
	//checkTracerVector(tid);
	return getTracerVector(tid)->TauTraceFd;
}

/* Flags for whether or not EDF files need to be rewritten when this thread's
   trace buffer is flushed.  Because any thread can introduce new functions and
   need to be flushed, we can't always wait for thread 0 */
static std::atomic<int> TauTraceFlushEvents{0};


/* Initialization status flags */
//static int TauTraceInitialized[TAU_MAX_THREADS] = {0};
static inline int getTauTraceInitialized(int tid){
	//checkTracerVector(tid);
	return getTracerVector(tid)->TauTraceInitialized;
}
static inline void setTauTraceInitialized(int tid, int value){
	//checkTracerVector(tid);
	getTracerVector(tid)->TauTraceInitialized=value;
}
//static int TraceFileInitialized[TAU_MAX_THREADS] = {0};
static inline int getTraceFileInitialized(int tid){
	//checkTracerVector(tid);
	return getTracerVector(tid)->TraceFileInitialized;
}
static inline void setTraceFileInitialized(int tid){
	//checkTracerVector(tid);
	getTracerVector(tid)->TraceFileInitialized=1;
}
//static double tracerValues[TAU_MAX_COUNTERS] = {0};


double TauSyncAdjustTimeStamp(double timestamp)
{
  TauTraceOffsetInfo *offsetInfo = TheTauTraceOffsetInfo();

  if (offsetInfo->enabled) {
    timestamp = timestamp - offsetInfo->beginOffset + offsetInfo->syncOffset;
  }
  return timestamp;
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

  x_uint64 value = (x_uint64)TauMetrics_getTraceMetricValue(tid);

  if (TauEnv_get_synchronize_clocks()) {
    value = (x_uint64)TauSyncAdjustTimeStamp(value);
  }

  TAU_ASSERT(value > 0, "Zero timestamp value found.");

  return value;
}

extern "C" x_uint64 TauTraceGetTimeStamp() {
	return TauTraceGetTimeStamp(0);
}


/* Write event to buffer only [without overflow check] */
void TauTraceEventOnly(long int ev, x_int64 par, int tid) {
  TAU_EV * tau_ev_ptr = &getTraceBuffer(tid)[getTauCurrentEvent(tid)] ;
  tau_ev_ptr->ev   = ev;
  tau_ev_ptr->ti   = TauTraceGetTimeStamp(tid);
  tau_ev_ptr->par  = par;
  tau_ev_ptr->nid  = RtsLayer::myNode();
  tau_ev_ptr->tid  = tid;
  incrementTauCurrentEvent(tid);
}

/* Set the flag for flushing the EDF file, 1 means flush edf file. */
void TauTraceSetFlushEvents(int value) {
  TauTraceFlushEvents = value;
}

/* Get the flag for flushing the EDF file, 1 means flush edf file. */
int TauTraceGetFlushEvents() {
  return TauTraceFlushEvents;
}

/* Check that the trace file is initialized */
static int checkTraceFileInitialized(int tid) {
  if ( !(getTraceFileInitialized(tid))){
    if(RtsLayer::myNode() <= -1) {
      fprintf (stderr,"ERROR: TAU is creating a trace file on a node less than 0.\n");
    } 
    setTraceFileInitialized(tid);
    const char *dirname;
    char tracefilename[1024];
    dirname = TauEnv_get_tracedir();
    snprintf(tracefilename, sizeof(tracefilename),  "%s/tautrace.%d.%d.%d.trc",dirname,
	    RtsLayer::myNode(), RtsLayer::myContext(), tid);
    if ((setTauTraceFd(tid, open (tracefilename, O_WRONLY|O_CREAT|O_TRUNC|O_APPEND|O_BINARY|LARGEFILE_OPTION, 0666))) < 0) {
      fprintf (stderr, "TAU: TauTraceInit[open]: ");
      perror (tracefilename);
      exit (1);
    }

    //    printf("checkTraceFileInitialized [%d]: TauTraceFd[%d] for [%s] is %d\n", RtsLayer::myNode(),
    //	   tid, tracefilename, TauTraceFd[tid]);

	  TAU_EV* TB = getTraceBuffer(tid);
    if (TB[0].ev == TAU_EV_INIT) {
      /* first record is init */
      for (unsigned int iter = 0; iter < getTauCurrentEvent(tid); iter ++) {
        int mynodeid = RtsLayer::myNode();
	if ((mynodeid > 0) && (TB[iter].nid == 0)) {
          TB[iter].nid = RtsLayer::myNode();
        } else {
            if ((mynodeid > 0) && (TB[iter].nid > 0))
	      break;
        }
      }
    }
  }
  return 0;
}

/* Flush the trace buffer */
extern "C" void finalizeCallSites_if_necessary();
void TauTraceFlushBuffer(int tid)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
    TauTraceOTF2FlushBuffer(tid);
    return;
  }
#endif

  checkTraceFileInitialized(tid);

  int ret;
  if (getTauTraceFd(tid) == -1) {
    printf("Error: TauTraceFlush(%d): Fd is -1. Trace file not initialized \n", tid);
    if (RtsLayer::myNode() == -1) {
      fprintf(stderr, "TAU: ERROR in configuration. Trace file not initialized.\n"
          "TAU: If this is an MPI application, please ensure that TAU MPI wrapper library is linked.\n"
          "TAU: If not, please ensure that TAU_PROFILE_SET_NODE(id); is called in the program (0 for sequential).\n");
      exit(1);
    }
  }

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_callsite()) {
    finalizeCallSites_if_necessary();
  }
#endif /* TAU_WINDOWS */
#endif /* _AIX */

  if (TauTraceGetFlushEvents()) {
    /* Dump the EDF file before writing trace data */
    TauTraceDumpEDF(tid);
    TauTraceSetFlushEvents(0);
  }

  int numEventsToBeFlushed = getTauCurrentEvent(tid); /* starting from 0 */
  DEBUGPROFMSG("Tid "<<tid<<": TauTraceFlush()"<<endl;);
  if (numEventsToBeFlushed != 0) {
#ifdef TAU_MPI
#ifndef TAU_SHMEM
   if (Tau_get_usesMPI())
#endif /* TAU_SHMEM */
   {
#endif /* TAU_MPI */

    ret = write(getTauTraceFd(tid), getTraceBuffer(tid), (numEventsToBeFlushed) * sizeof(TAU_EV));
    if (ret < 0) {
#ifdef DEBUG_PROF
      TAU_VERBOSE("Error: TauTraceFd[%d] = %d, numEvents = %d ", tid, getTauTraceFd(tid), numEventsToBeFlushed);
      TAU_VERBOSE("Write Error in TauTraceFlush()");
#endif
    }
#ifdef TAU_MPI
   }
#ifndef TAU_SHMEM
   else {
     // do nothing.
     return;
   }
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */
  }
  resetTauCurrentEvent(tid);
}


/* static list of flags specifying if the buffer has been allocated for a given thread 
bool *TauBufferAllocated() {
  static bool flag = true;
  static bool allocated[TAU_MAX_THREADS]; //TODO: DYNATHREAD
  if (flag) {
    for (int i=0; i < TAU_MAX_THREADS; i++) {
      allocated[i] = false;
    }
    flag = false;
  }
  return allocated;
}*/ //TODO: Remove once validated

static inline bool getTauBufferAllocated(int tid){
	//checkTracerVector(tid);
	return getTracerVector(tid)->allocated;
}

static inline void setTauBufferAllocated(int tid, bool value){
	//checkTracerVector(tid);
	getTracerVector(tid)->allocated=value;
}

/* Initialize tracing. TauTraceInit should be called in every trace routine to ensure that
   the trace file is initialized */
int TauTraceInit(int tid)
{
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      return TauTraceOTF2Init(tid);
  }
#endif
   if (!getTauBufferAllocated(tid)) {
     TauMaxTraceRecords = (unsigned long long) TauEnv_get_max_records(); 
     TauBufferSize = sizeof(TAU_EV)*TauMaxTraceRecords; 
     setTraceBuffer(tid,(TAU_EV*) malloc(TauBufferSize));
     if (getTraceBuffer(tid) == (TAU_EV *) NULL) {
       fprintf(stderr, 
          "TAU: FATAL Error: Trace buffer malloc failed.\n"
          "TAU: Please rerun the application with the TAU_MAX_RECORDS environment variable set to a smaller value\n");
       exit(1);
     }
     setTauBufferAllocated(tid, true);
   }
  int retvalue = 0;
  /* by default this is what is returned. No trace records were generated */
   
  if ( !(getTauTraceInitialized(tid)) && (RtsLayer::myNode() > -1)) {
  /* node has been set*/ 
    /* done with initialization */
    setTauTraceInitialized(tid,1);

    /* there may be some records in tau_ev_ptr already. Make sure that the
       first record has node id set properly */
	TAU_EV* TB = getTraceBuffer(tid);
    if (TB[0].ev == TAU_EV_INIT) {
      /* first record is init */
      for (unsigned int iter = 0; iter < getTauCurrentEvent(tid); iter ++) {
        TB[iter].nid = RtsLayer::myNode();
      }
    } else {
      /* either the first record is blank - in which case we should
	 put INIT record, or it is an error */
      if (getTauCurrentEvent(tid) == 0) {
        TauTraceEventSimple(TAU_EV_INIT, INIT_PARAM, tid, TAU_TRACE_EVENT_KIND_FUNC);
        retvalue ++; /* one record generated */
      } else {
	/* error */
        printf("Warning: TauTraceInit(%d): First record is not INIT\n", tid);
      }
    } /* first record was not INIT */

    /* generate a wallclock time record */
    TauTraceEventSimple (TAU_EV_WALL_CLOCK, time((time_t *)0), tid, TAU_TRACE_EVENT_KIND_FUNC);
    retvalue ++;
  }
  return retvalue;
}

/* This routine is typically invoked when multiple SET_NODE calls are
   encountered for a multi-threaded program */
void TauTraceReinitialize(int oldid, int newid, int tid) {
#ifndef TAU_SETNODE0
  TAU_VERBOSE("Inside TauTraceReinitialize : oldid = %d, newid = %d, tid = %d\n",
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

#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
    TauTraceOTF2UnInitialize(tid);
    return;
  }
#endif
  /* to set the trace as uninitialized and clear the current buffers (for forked
     child process, trying to clear its parent records) */
  setTauTraceInitialized(tid,0);
  resetTauCurrentEvent(tid);
  TauTraceEventOnly(TAU_EV_INIT, INIT_PARAM, tid);
}



/* Write event to buffer */
void TauTraceEventSimple(long int ev, x_int64 par, int tid, int kind) {
  TauTraceEvent(ev, par, tid, 0, 0, kind);
}


/* Write event to buffer */
void TauTraceEventWithNodeId(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts, int node_id, int kind)
{
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
    TauTraceOTF2EventWithNodeId(ev, par, tid, ts, use_ts, node_id, kind);
    return;
  }
#endif

  int i;
  int records_created = TauTraceInit(tid);
  x_uint64 timestamp;
  TAU_EV *event = &getTraceBuffer(tid)[getTauCurrentEvent(tid)];

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
        getTraceBuffer(tid)[getTauCurrentEvent(tid)-1-i].ti = ts; 
      }
    }
  }

  if (!(getTauTraceInitialized(tid)) && (getTauCurrentEvent(tid) == 0)) {
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
 
      incrementTauCurrentEvent(tid);
      event = &getTraceBuffer(tid)[getTauCurrentEvent(tid)];
    }
  }

  event->ev  = ev;
  if (use_ts) {
    event->ti = ts;
    timestamp = ts;
  } else {
    timestamp = TauTraceGetTimeStamp(tid);
    event->ti = timestamp;
  }
  event->par = par;
  event->nid = node_id;
  event->tid = tid ;
  incrementTauCurrentEvent(tid);

  if (getTauCurrentEvent(tid) >= TauMaxTraceRecords-2) {
    //TauTraceEventSimple (TAU_EV_FLUSH, 0, tid);
    event = &getTraceBuffer(tid)[getTauCurrentEvent(tid)];
    event->ev = TAU_EV_FLUSH;  event->ti = timestamp; event->par = 1;
    event->nid = node_id; event->tid = tid;
    incrementTauCurrentEvent(tid);

    // Flush the buffer!
    TauTraceFlushBuffer(tid);

    //TauTraceEventSimple (TAU_EV_FLUSH, 0, tid);
    timestamp = TauTraceGetTimeStamp(tid);
    event = &getTraceBuffer(tid)[getTauCurrentEvent(tid)];
    event->ev = TAU_EV_FLUSH;  event->ti = timestamp; event->par = -1;
    event->nid = node_id; event->tid = tid;
    incrementTauCurrentEvent(tid);
  }
}

/* Write event to buffer */
void TauTraceEvent(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts, int kind) {
  TauTraceEventWithNodeId(ev, par, tid, ts, use_ts, RtsLayer::myNode(), kind);
}

/* Close the trace */
void TauTraceClose(int tid) {

#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      TauTraceOTF2Close(tid);
      return;
  }
#endif

  TauTraceEventSimple (TAU_EV_CLOSE, 0, tid, TAU_TRACE_EVENT_KIND_FUNC);
  TauTraceEventSimple (TAU_EV_WALL_CLOCK, time((time_t *)0), tid, TAU_TRACE_EVENT_KIND_FUNC);
  TauTraceDumpEDF(tid);
  TauTraceFlushBuffer (tid);
  //close (TauTraceFd[tid]);
  // Just in case the same thread writes to this file again, don't close it.
  // for OpenMP.
#ifndef TAU_OPENMP
  if ((RtsLayer::myNode() == 0) && (RtsLayer::myThread() == 0)) {
    close(getTauTraceFd(tid));
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
  if (current) {
    // Trace all the previous records before tracing self
    TraceCallStack(tid, current->ParentProfiler);
    TauTraceEventSimple(current->ThisFunction->GetFunctionId(), 1, tid, TAU_TRACE_EVENT_KIND_FUNC);
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
  AtomicEventDB::iterator uit;
  char filename[1024], errormsg[1064];
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

#ifdef TAU_MPI
#ifndef TAU_SHMEM
  if (Tau_get_usesMPI())
#endif /* TAU_SHMEM */
  {
#endif /* TAU_MPI */
    snprintf(filename, sizeof(filename), "%s/events.%d.edf",dirname, RtsLayer::myNode());
    if ((fp = fopen (filename, "w+")) == NULL) {
      snprintf(errormsg, sizeof(errormsg), "Error: Could not create %s",filename);
      perror(errormsg);
      RtsLayer::UnLockDB();
      return -1;
    }
#ifdef TAU_MPI
  }
#ifndef TAU_SHMEM
  else {
    RtsLayer::UnLockDB();
    return -1;
  }
#endif /* TAU_SHMEM */
#endif /* TAU_MPI */

  // Data Format
  // <no.> events
  // # or \n ignored
  // %s %s %d "%s %s" %s
  // id group tag "name type" parameters

  numEvents = TheFunctionDB().size() + TheEventDB().size();
#ifdef TAU_GPU
  numExtra = 16; // Added seven ONESIDED msg events
#else
  numExtra = 9; // Number of extra events
#endif
  numEvents += numExtra;

  fprintf(fp,"%d dynamic_trace_events\n", numEvents+1);

  fprintf(fp,"# FunctionId Group Tag \"Name Type\" Parameters\n");

  fprintf(fp,"0 TAUEVENT 0 \".TAU <unknown event>\" TriggerValue\n");

  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    fprintf(fp, "%ld %s 0 \"%s %s\" EntryExit\n", (long)((*it)->GetFunctionId()),
	    (*it)->GetPrimaryGroup(), (*it)->GetName(), (*it)->GetType() );
  }

  /* Now write the user defined event */
  for (uit = TheEventDB().begin(); uit != TheEventDB().end(); uit++) {
    int monoInc = (*uit)->IsMonotonicallyIncreasing() ? 1 : 0;
    fprintf(fp, "%ld TAUEVENT %d \"%s\" TriggerValue\n", (long)(*uit)->GetId(), monoInc, (*uit)->GetName().c_str());
  }

  // Now add the nine extra events
  fprintf(fp,"%ld TRACER 0 \"EV_INIT\" none\n", (long) TAU_EV_INIT);
  fprintf(fp,"%ld TRACER 0 \"FLUSH\" EntryExit\n", (long) TAU_EV_FLUSH);
//  fprintf(fp,"%ld TRACER 0 \"FLUSH_EXIT\" none\n", (long) TAU_EV_FLUSH_EXIT);
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
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE_RECIPROCAL_SEND\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_RECIPROCAL_SEND);
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE_RECIPROCAL_RECV\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_RECIPROCAL_RECV);
  fprintf(fp,"%ld TAUEVENT 0 \"ONESIDED_MESSAGE\" TriggerValue\n", (long)
	TAU_ONESIDED_MESSAGE_UNKNOWN);
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
  char cmd[4096];
  char rmcmd[256];
  char cdcmd[1024];
  const char *tauroot=TAUROOT;
  const char *tauarch=TAU_ARCH;
  const char *conv="tau2vtf";
  char converter[1024] = {0};
  FILE *in;
  int status_code = 0;

  /* If we can't find tau2vtf, use tau_convert! */
  snprintf(converter, sizeof(converter),  "%s/%s/bin/%s",tauroot, tauarch, conv);
  if ((in = fopen(converter, "r")) == NULL) {
    snprintf(converter, sizeof(converter),  "%s/%s/bin/tau_convert", tauroot, tauarch);
  } else {
    fclose(in);
  }

  /* Should we get rid of intermediate trace files? */
  keepfiles = getenv("TAU_KEEP_TRACEFILES");
  if (keepfiles == NULL) {
    strncpy(rmcmd,  "/bin/rm -f app12345678.trc tautrace.*.trc tau.edf events.*.edf", sizeof(rmcmd)); 
  } else {
    strncpy(rmcmd, " ", sizeof(rmcmd));  /* NOOP */
  }

  /* Next, look for trace directory */
  outdir = TauEnv_get_tracedir();
  snprintf(cdcmd, sizeof(cdcmd),  "cd %s;", outdir);

  /* create the command */
  snprintf(cmd, sizeof(cmd),  "%s /bin/rm -f app12345678.trc; %s/%s/bin/tau_merge tautrace.*.trc app12345678.trc; %s app12345678.trc tau.edf %s; %s", cdcmd,tauroot, tauarch, converter, outfile, rmcmd);
#ifdef DEBUG_PROF
  TAU_VERBOSE("The merge/convert cmd is: %s\n", cmd);
#endif /* DEBUG_PROF */

  /* and execute it */
#ifndef TAU_CATAMOUNT
  /* NOTE: BGL will not execute this code as well because the compute node
     kernels cannot fork tasks. So, on BGL, nothing will happen when the
     following system command executes */
  status_code = system(cmd);
  if (status_code != 0) {
    TAU_VERBOSE("Warning: unable to execute command: '%s'\n", cmd);
  }
#endif /* TAU_CATAMOUNT */

  return 0;
}

#ifdef TAU_GPU

void TauTraceOneSidedMsg(int type, GpuEvent *gpu, int length, int threadId, x_uint64 ts = 0UL)
{
    int use_ts = 0;
    if (ts > 0UL) {
        use_ts = 1;
    }
    /* there are three user events that make up a one-sided msg */
    if (type == MESSAGE_SEND) {
        TauTraceEvent(TAU_ONESIDED_MESSAGE_SEND, length, threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
    } else if (type == MESSAGE_RECV) {
        TauTraceEvent(TAU_ONESIDED_MESSAGE_RECV, length, threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
    } else if (type == MESSAGE_RECIPROCAL_SEND) {
        TauTraceEvent(TAU_ONESIDED_MESSAGE_RECIPROCAL_SEND, length, threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
    } else if (type == MESSAGE_RECIPROCAL_RECV) {
        TauTraceEvent(TAU_ONESIDED_MESSAGE_RECIPROCAL_RECV, length, threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
    } else {
        TauTraceEvent(TAU_ONESIDED_MESSAGE_UNKNOWN, length, threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
    }
    TauTraceEvent(TAU_ONESIDED_MESSAGE_ID_1, gpu->id_p1(), threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
    TauTraceEvent(TAU_ONESIDED_MESSAGE_ID_2, gpu->id_p2(), threadId, ts, use_ts, TAU_TRACE_EVENT_KIND_COMM);
}

#endif


extern "C" void TauTraceMsg(int send_or_recv, int type, int other_id, int length, x_uint64 ts, int use_ts, int node_id) {

#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      TauTraceOTF2Msg(send_or_recv, type, other_id, length, ts, use_ts, node_id);
  }
#endif

  x_int64 parameter;
  x_uint64 xother, xtype, xlength, xcomm;

  if (RtsLayer::isEnabled(TAU_MESSAGE)) {
    parameter = 0;
    /* for recv, othernode is sender or source*/
    xtype = type;
    xlength = length;
    xother = other_id;
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

    TauTraceEventWithNodeId(send_or_recv, parameter, RtsLayer::myThread(), ts, use_ts, node_id, TAU_TRACE_EVENT_KIND_COMM);
  }
}

//////////////////////////////////////////////////////////////////////
// TauTraceRecvMsg traces the message recv
//////////////////////////////////////////////////////////////////////
void TauTraceRecvMsg(int type, int source, int length) {
  TauTraceMsg(TAU_MESSAGE_RECV, type, source, length, 0, 0, RtsLayer::myNode());
  /* 0, 0 is for ts and use_ts so TAU generates the timestamp */
}

//////////////////////////////////////////////////////////////////////
// TraceSendMsg traces the message send
//////////////////////////////////////////////////////////////////////
void TauTraceSendMsg(int type, int destination, int length) {
  TauTraceMsg(TAU_MESSAGE_SEND, type, destination, length, 0, 0, RtsLayer::myNode());
  /* 0, 0 is for ts and use_ts so TAU generates the timestamp */
}


//////////////////////////////////////////////////////////////////////
// TauTraceRecvMsgRemote traces the message recv for an RMA operation
//////////////////////////////////////////////////////////////////////
void TauTraceRecvMsgRemote(int type, int source, int length, int remote_id) {
  TauTraceMsg(TAU_MESSAGE_RECV, type, source, length, 0, 0, remote_id);
  /* 0, 0 is for ts and use_ts so TAU generates the timestamp */
}

//////////////////////////////////////////////////////////////////////
// TraceSendMsgRemote traces the message send for an RMA operation
//////////////////////////////////////////////////////////////////////
void TauTraceSendMsgRemote(int type, int destination, int length, int remote_id) {
  TauTraceMsg(TAU_MESSAGE_SEND, type, destination, length, 0, 0, remote_id);
  /* 0, 0 is for ts and use_ts so TAU generates the timestamp */
}

void TauTraceBarrierAllStart(int tag) {
#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      TauTraceOTF2BarrierAllStart(tag);
  }
#endif
}

void TauTraceBarrierAllEnd(int tag) {
#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      TauTraceOTF2BarrierAllEnd(tag);
  }
#endif
}


void TauTraceRMACollectiveBegin(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root) {
#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      TauTraceOTF2RMACollectiveBegin(tag, type, start, stride, size, data_in, data_out, root);
  }
#endif
}

void TauTraceRMACollectiveEnd(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root) {
#ifdef TAU_OTF2
  if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
      TauTraceOTF2RMACollectiveEnd(tag, type, start, stride, size, data_in, data_out, root);
  }
#endif
}

void TauTraceOTF2InitShmem_if_necessary() {
#ifdef TAU_OTF2
    TauTraceOTF2InitShmem();
#endif
}

void TauTraceOTF2ShutdownComms_if_necessary(int tid) {
#ifdef TAU_OTF2
    TauTraceOTF2ShutdownComms(tid);
#endif
}
