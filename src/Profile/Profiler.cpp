/*****************************************************************************
 **      TAU Portable Profiling Package          **
 **      http://www.cs.uoregon.edu/research/tau              **
 *****************************************************************************
 **    Copyright 1997-2009                   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/*****************************************************************************
 **  File     : Profiler.cpp              **
 **  Description   : TAU Profiling Package            **
 **  Author    : Sameer Shende              **
 **  Contact    : tau-bugs@cs.uoregon.edu                       **
 **  Documentation  : See http://www.cs.uoregon.edu/research/tau        **
 ****************************************************************************/

#include <Profile/Profiler.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauSnapshot.h>
#include <Profile/TauTrace.h>
#include <Profile/TauMetaData.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetaDataMerge.h>
#include <Profile/TauPluginInternals.h>

// Define DEBUG_PROF if you want to debug timers
// #define DEBUG_PROF 1

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#include <stack>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#include <stack.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#include <string>
#include <vector>
//#include <mutex>

#ifdef TAU_VAMPIRTRACE
#include <Profile/TauVampirTrace.h>
#endif

#ifdef RENCI_STFF
#include <Profile/RenciSTFF.h>
#endif

#ifdef TAUKTAU
#include <Profile/KtauProfiler.h>
#endif

#ifdef KTAU_NG
#include <Profile/KtauNGProfiler.h>
#endif

#ifdef TAU_PERFSUITE
#include <pshwpc.h>
extern "C" int ps_hwpc_xml_write(const char *filename);
extern "C" int ps_hwpc_reset();
#endif

#ifdef TAU_EPILOG
#include <elg_trc.h>

#ifdef TAU_SCALASCA
extern "C" void esd_enter (elg_ui4 rid);
extern "C" void esd_exit (elg_ui4 rid);
#endif /* TAU_SCALASCA */
#endif /* TAU_EPILOG */

#ifdef TAU_WINDOWS
double TauWindowsUsecD(void);
#endif

#ifdef CUPTI
#ifdef __GNUC__
#include "cupti_version.h"
#include "cupti_events.h"
#include "cupti_metrics.h"
#include <cuda_runtime_api.h>
#endif //__GNUC__
#endif //CUPTI

#ifdef TAU_ENABLE_ROCTRACER
extern void Tau_roctracer_stop_tracing(void);
#endif /* TAU_ENABLE_ROCTRACER */

#ifdef TAU_SHMEM
#include "shmem.h"
extern "C" void  __real_shmem_finalize() ;
#endif /* TAU_SHMEM */
extern "C" int Tau_get_usesSHMEM();


#ifdef TAU_TRACK_IDLE_THREADS
extern "C" bool Tau_check_Stopping_All_Threads();
#endif /* TAU_TRACK_IDLE_THREADS */

using namespace std;
using namespace tau;

// This would be more useful in a utility header somewhere, but the way people slap 'extern "C"'
// on everything means we'll probably wind up with an C-linked template at some point...
template < typename T >
struct ScopedArray {
    ScopedArray(size_t count) :
        size(count*sizeof(T)), ptr(new T[count]) {}
    ~ScopedArray() {
        if(ptr) delete[] ptr;
    }
    operator T*() const {
        return ptr;
    }
    size_t size;
    T * const ptr;
};

//////////////////////////////////////////////////////////////////////
// Explicit Instantiations for templated entities needed for ASCI Red
//////////////////////////////////////////////////////////////////////

#ifdef PGI
template void vector<FunctionInfo *>::insert_aux(vector<FunctionInfo *>::pointer, FunctionInfo *const &);
template FunctionInfo** copy_backward(FunctionInfo**,FunctionInfo**,FunctionInfo**);
template FunctionInfo** uninitialized_copy(FunctionInfo**,FunctionInfo**,FunctionInfo**);
#endif /* PGI */

static int writeUserEvents(FILE *fp, int tid);
static int matchFunction(FunctionInfo *fi, const char **inFuncs, int numFuncs);

extern "C" int Tau_get_usesMPI();
extern "C" void Tau_shutdown(void);
extern "C" void Tau_profile_exit_most_threads();
extern "C" int TauCompensateInitialized(void);
extern "C" void Tau_ompt_resolve_callsite(FunctionInfo &fi, char * resolved_address);

/*#ifdef TAU_ENABLE_ROCM
extern void TauFlushRocmEventsIfNecessary(int thread_id);
#endif*/ /* TAU_ENABLE_ROCM */

x_uint64 Tau_get_firstTimeStamp();
struct ProfilerData{
    int profileWriteCount=0;
    #ifdef TAU_TRACK_IDLE_THREADS
        double TheLastTimeStamp[TAU_MAX_COUNTERS];
    #endif /* TAU_TRACK_IDLE_THREADS */
};
struct ProfThreadList : vector<ProfilerData*>{
    ProfThreadList (const ProfThreadList&) = delete;
    ProfThreadList& operator= (const ProfThreadList&) = delete;
    ProfThreadList(){
         //printf("Creating ProfilerThreadList at %p\n", this);
      }
     virtual ~ProfThreadList(){
         //printf("Destroying ProfilerThreadList at %p, with size %ld\n", this, this->size());
         Tau_destructor_trigger();
     }
   };

static ProfThreadList & TheProfilerThreadList(){
	static ProfThreadList profThreads;
	return profThreads;
}
//static std::lock_guard<std::mutex> guard(ProfilerVectorMutex);
static std::mutex ProfilerVectorMutex;
inline void checkProfilerVector(int tid){
    //if(TheProfilerThreadList().size()<=tid){
    //  static std::mutex ProfilerVectorMutex;
    //  std::lock_guard<std::mutex> guard(ProfilerVectorMutex);
	while(TheProfilerThreadList().size()<=tid){
        //RtsLayer::LockDB();
		TheProfilerThreadList().push_back(new ProfilerData());
        //ProfilerThreadList.back()->profileWriteCount=0;
        //RtsLayer::UnLockDB();
	}
   // }
    //printf("Write count for tid: %d, post-check: %d\n",tid,ProfilerThreadList[tid]->profileWriteCount);
}


static thread_local int local_tid = RtsLayer::myThread();
static thread_local ProfilerData* PD_cache=0;

static inline ProfilerData& getProfilerData(int tid){

    if(tid == local_tid){
        if(PD_cache!=0){
            return *PD_cache;
        }
    }

    //printf("CACHE MISSED seeking %d on %d!!!\n",tid,local_tid);
    std::lock_guard<std::mutex> guard(ProfilerVectorMutex);
    checkProfilerVector(tid);
    ProfilerData* PDOut=TheProfilerThreadList()[tid];
    if(tid == local_tid){
        if(PD_cache==0){
            PD_cache=PDOut;
        }
    }
    return *PDOut;
}

//////////////////////////////////////////////////////////////////////
// For OpenMP
//////////////////////////////////////////////////////////////////////
#ifdef TAU_TRACK_IDLE_THREADS
//double TheLastTimeStamp[TAU_MAX_THREADS][TAU_MAX_COUNTERS]; //TODO: DYNATHREAD
inline void setLastTimeStamp(int tid, int counter, double value){
    //printf("SLT: TID: %d, CID: %d\n",tid,counter);

    //printf("SLT: Checked\n");
    //if(ProfilerThreadList()[tid]->TheLastTimeStamp==0||)
    //{
    //    printf("SLT: Invalid!\n");
    //}

    getProfilerData(tid).TheLastTimeStamp[counter]=value;
}
inline double getLastTimeStamp(int tid, int counter){

    return getProfilerData(tid).TheLastTimeStamp[counter];
}
#endif /* TAU_TRACK_IDLE_THREADS */

//////////////////////////////////////////////////////////////////////
// Get the string containing the counter name
//////////////////////////////////////////////////////////////////////
const char *TauGetCounterString(void)
{
#ifdef SGI_HW_COUNTERS
  return "templated_functions_hw_counters";
#elif (defined (TAU_PAPI) \
       || defined(TAU_PAPI_WALLCLOCKTIME)  \
       || defined(TAU_PAPI_VIRTUAL))
  char *tau_env = NULL;

#ifdef TAU_PAPI
  tau_env = getenv("PAPI_EVENT");
#endif /* TAU_PAPI */
  if (tau_env) {
    char *header = new char[1024];
    snprintf(header, 1024,  "templated_functions_MULTI_%s", tau_env);
    return header;
  } else {
#ifdef TAU_PAPI_WALLCLOCKTIME
    return "templated_functions_MULTI_P_WALL_CLOCK_TIME";
#endif /* TAU_PAPI_WALLCLOCKTIME */

#ifdef TAU_PAPI_VIRTUAL
    return "templated_functions_MULTI_P_VIRTUAL_TIME";
#endif /* TAU_PAPI_VIRTUAL */
    return "templated_functions_hw_counters";
  }
#else  // ! (TAU_PAPI) => SGI_TIMERS
  return "templated_functions";
#endif // ALL options
}

#ifdef TAU_MPITRACE
void TauProfiler_EnableAllEventsOnCallStack(int tid, Profiler * current)
{
  /* Go up the callstack and enable all events on it */
  if (current) {
    DEBUGPROFMSG(RtsLayer::myNode() << " This func = " << current->ThisFunction->GetName() << " RecordEvent = " << current->RecordEvent << endl);
    if (!current->RecordEvent) {
      DEBUGPROFMSG(RtsLayer::myNode() << " Enabling event " << current->ThisFunction->GetName() << endl);
      current->RecordEvent = true;
      TauProfiler_EnableAllEventsOnCallStack(tid, current->ParentProfiler);
      /* process the current event */
      DEBUGPROFMSG(RtsLayer::myNode() << " Processing EVENT " << current->ThisFunction->GetName() << endl);
      TauTraceEvent(current->ThisFunction->GetFunctionId(), 1, tid, (x_uint64)current->StartTime[0], 1, TAU_TRACE_EVENT_KIND_FUNC);
      TauMetrics_triggerAtomicEvents((x_uint64)current->StartTime[0], current->StartTime, tid);
    }
  }
}
#endif /* TAU_MPITRACE */

#ifdef TAU_PERFSUITE
static x_uint64 getTimeStamp()
{
  x_uint64 timestamp;
#ifdef TAU_WINDOWS
  timestamp = TauWindowsUsecD();
#else
  struct timeval tp;
  gettimeofday(&tp, 0);
  timestamp = (x_uint64)tp.tv_sec * (x_uint64)1e6 + (x_uint64)tp.tv_usec;
#endif
  return timestamp;
}
#endif /* TAU_PERFSUITE */


//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class Profiler
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

void Profiler::Start(int tid)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE( "[%d:%d-%d] Profiler::Start for %s (%p), node %d\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction, RtsLayer::myNode());
#endif
  ParentProfiler = TauInternal_ParentProfiler(tid);

  /********************************************************************************/
  /*** Phase Profiling ***/
  /********************************************************************************/
#ifdef TAU_PROFILEPHASE
  if (ParentProfiler == (Profiler *) NULL) {
    string AllGroups = ThisFunction->AllGroups;
    if (AllGroups.find("TAU_PHASE", 0) == string::npos) {
      AllGroups.append("|TAU_PHASE");
      free(ThisFunction->AllGroups);
      ThisFunction->AllGroups = strdup(AllGroups.c_str());
    }
  }

#ifdef TAU_PERFSUITE
  if (GetPhase()) {
    static int perfsuiteInit = 0;
    if (perfsuiteInit == 0) {
      perfsuiteInit = 1;
      int ierr = ps_hwpc_init();
      if (ierr != 0) {
        printf ("Error on ps_hwpc_init: %d\n", ierr);
      }
    }

    printf ("tau-perfsuite: starting\n");
    int ierr = ps_hwpc_start();
    if (ierr != 0) {
      printf ("Error on ps_hwpc_start: %d\n", ierr);
    }
  }
#endif /* TAU_PERFSUITE */
#endif /* TAU_PROFILEPHASE */
  /********************************************************************************/
  /*** Phase Profiling ***/
  /********************************************************************************/

  /* Get the current metric values */
  x_uint64 TimeStamp;
  // Record metrics in reverse order so wall clock metrics are recorded after PAPI, etc.
  RtsLayer::getUSecD(tid, StartTime, 1);

  TimeStamp = (x_uint64)StartTime[0];    // USE COUNTER1 for tracing
  // This can happen when starting .TAU application on "virtual" GPU threads.
  // The GPU timestamp isn't availble yet, so start is bogus.  Instead,
  // get the timers read just after initialization.
#ifndef TAU_SCOREP
  if (TimeStamp == 0L) {
#if !defined(TAU_USE_OMPT_5_0) && !defined(TAU_GPU) // this can happen with OMPT async threads
    printf("Got a bogus start! %d %s\n", tid, ThisFunction->GetName());
#endif
    TauMetrics_getDefaults(tid, StartTime, 1);
    TimeStamp = (x_uint64)StartTime[0];    // USE COUNTER1 for tracing
    if (TimeStamp == 0L) {
      fprintf(stderr, "Still got a bogus start! %d %s\n", tid, ThisFunction->GetName());
      abort();
    }
  }
#endif /* !TAU_SCOREP */

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

  /*** Profile Compensation ***/
  if (TauEnv_get_compensate()) {
    SetNumChildren(0); /* for instrumentation perturbation compensation */
  }

  // An initialization of sorts. Call Paths (if any) will update this.
#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_callsite() == 1) {
    CallSiteAddPath(NULL, tid);
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_callsite() == 1) {
    CallSiteStart(tid, TimeStamp);
  }
#endif /* _AIX */
#endif

  if (TauEnv_get_callpath()) {
    CallPathStart(tid);
  }

#ifdef TAU_PROFILEPARAM
  ProfileParamFunction = NULL;
  if (ParentProfiler && ParentProfiler->ProfileParamFunction ) {
    ParentProfiler->ProfileParamFunction->IncrNumSubrs(tid);
  }
#endif /* TAU_PROFILEPARAM */

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/

#ifdef TAU_MPITRACE
  if (RecordEvent) {
#endif /* TAU_MPITRACE */
  if (TauEnv_get_tracing()) {
    TauTraceEvent(ThisFunction->GetFunctionId(), 1 /* entry */, tid, TimeStamp, 1 /* use supplied timestamp */, TAU_TRACE_EVENT_KIND_FUNC);
    TauMetrics_triggerAtomicEvents(TimeStamp, StartTime, tid);
  }
#ifdef TAU_MPITRACE
}
#endif /* TAU_MPITRACE */

#ifdef TAU_MPITRACE
  if (MyProfileGroup_ & TAU_MESSAGE) {
    /* if we're in the group, we must first enable all the other events
     * on the callstack */
    DEBUGPROFMSG(RtsLayer::myNode()<< " Function is enabled: "<<ThisFunction->GetName()<<endl;);
    TauProfiler_EnableAllEventsOnCallStack(tid, this);
  }
#endif /* TAU_MPITRACE */

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/

  // Inncrement the number of calls
  ThisFunction->IncrNumCalls(tid);

  // Increment the parent's NumSubrs()
  if (ParentProfiler != 0) {
    ParentProfiler->ThisFunction->IncrNumSubrs(tid);
    if (TauEnv_get_callsite()) {
      if (ParentProfiler->CallSiteFunction != NULL) {
        ParentProfiler->CallSiteFunction->IncrNumSubrs(tid);
      }
    }
  }

  // If this function is not already on the call stack, put it
  if (ThisFunction->GetAlreadyOnStack(tid) == false) {
    AddInclFlag = true;
    // We need to add Inclusive time when it gets over as
    // it is not already on callstack.

    ThisFunction->SetAlreadyOnStack(true, tid);    // it is on callstack now
  } else {
    // the function is already on callstack, no need to add inclusive time
    AddInclFlag = false;
  }

  /********************************************************************************/
  /*** KTAU Code ***/
  /********************************************************************************/
#if defined(TAUKTAU)
  ThisKtauProfiler = KtauProfiler::GetKtauProfiler();
  ThisKtauProfiler->Start(this);
#endif /* TAUKTAU */

  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.function_entry) {
    Tau_plugin_event_function_entry_data_t plugin_data;
    plugin_data.timer_name = ThisFunction->GetName();
    plugin_data.func_id = ThisFunction->GetFunctionId();
    plugin_data.timer_group = ThisFunction->GetAllGroups();
    plugin_data.tid = tid;
    plugin_data.timestamp = TimeStamp;
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_FUNCTION_ENTRY, ThisFunction->GetName(), &plugin_data);
  }
}

void Profiler::Stop(int tid, bool useLastTimeStamp)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE( "[%d:%d-%d] Profiler::Stop  for %s (%p), node %d\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction, RtsLayer::myNode());
#endif

/* It is possible that when the event stack gets deep, and has to be
 * reallocated, the pointers in the event stack get messed up. This
 * fixes the parent pointer for flat profiles, but I don't know if it
 * is a robust fix for all scenarios! - Kevin */

#if 0
  if (ParentProfiler != TauInternal_ParentProfiler(tid)) {
    ParentProfiler = TauInternal_ParentProfiler(tid);
    //printf ("%d: Warning! ParentProfiler pointer was bogus!\n", tid);
  }
#endif

  /********************************************************************************/
  /*** PerfSuite Integration Code ***/
  /********************************************************************************/
#ifdef TAU_PROFILEPHASE
#ifdef TAU_PERFSUITE
  if (GetPhase()) {
    static int sequence=0;
    char annotation[4096];
    snprintf (annotation, sizeof(annotation),  "TAU^seq^%d^phase^%s^nct^%d:%d:%d^timestamp^%lld^start^%lld^", sequence,
        ThisFunction->GetName(), RtsLayer::myNode(), RtsLayer::myContext(), RtsLayer::myThread(),
        getTimeStamp(), Tau_get_firstTimeStamp());
    printf ("tau-perfsuite: stopping %s\n", ThisFunction->GetName());
    setenv("PS_HWPC_ANNOTATION", annotation, 1);
    char seqstring[256];
    snprintf (seqstring, sizeof(seqstring),  "TAU.%d", sequence);
    int ierr;

    ierr = ps_hwpc_suspend();
    if (ierr != 0) {
      printf ("Error on ps_hwpc_suspend: %d\n", ierr);
    }

    ierr = ps_hwpc_xml_write(seqstring);
    if (ierr != 0) {
      printf ("Error on ps_hwpc_xml_write: %d\n", ierr);
    }

    ierr = ps_hwpc_reset();
    if (ierr != 0) {
      printf ("Error on ps_hwpc_reset: %d\n", ierr);
    }
    sequence++;
  }
#endif /* TAU_PERFSUITE */
#endif /* TAU_PROFILEPHASE */
  /********************************************************************************/
  /*** PerfSuite Integration Code ***/
  /********************************************************************************/

  // first initialize the CurrentTime
  double CurrentTime[TAU_MAX_COUNTERS] = { 0 };
  double TotalTime[TAU_MAX_COUNTERS] = { 0 };

#ifdef TAU_TRACK_IDLE_THREADS
  int i;
  if (useLastTimeStamp) {
    /* for openmp parallel regions */
    /* .TAU Application needs to be stopped */
    for (i = 0; i < TAU_MAX_COUNTERS; i++) {
      CurrentTime[i] = getLastTimeStamp(tid,i);
    }
  } else {
    /* use the usual mechanism */
    RtsLayer::getUSecD(tid, CurrentTime);
  }
  for (i = 0; i < TAU_MAX_COUNTERS; i++) {
    setLastTimeStamp(tid,i,CurrentTime[i]);
  }
#else
  RtsLayer::getUSecD(tid, CurrentTime);
#endif /* TAU_TRACK_IDLE_THREADS */
  //printf("In Stop: CurrentTime[0] = %f\n", CurrentTime[0]);

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_event_stop(tid, CurrentTime);
  }
#endif /* _AIX */
#endif

#if defined(TAUKTAU)
#ifdef KTAU_DEBUGPROF
  printf("Profiler::Stop: --EXIT-- %s \n", TauInternal_CurrentProfiler(tid)->ThisFunction->GetName());
#endif /*KTAU_DEBUGPROF*/
  ThisKtauProfiler->Stop(this, AddInclFlag);
#endif /* TAUKTAU */

  // this happens during early initialization, before program load
  //if (CurrentTime[0] == 0.0) { CurrentTime[0] = TauMetrics_getTimeOfDay(); }

  // It's ok if CurrentTime is 0, because that means StartTime is too.
  // However, if CurrentTime is not 0, we need to fix a timer that was read
  // before we were done initializing metrics.
//   // This code SHOULDN'T be needed any more.  but things slip through the cracks.
//   if (CurrentTime[0] != 0.0 && StartTime[0] == 0.0) {
//     abort();
//     // get the CurrentTime again, but use the thread 0 context
//     double CurrentTime_0[TAU_MAX_COUNTERS] = { 0 };
//     RtsLayer::getUSecD(0, CurrentTime_0);
//     // ...because the default values were captured by thread 0
//     TauMetrics_getDefaults(tid, StartTime, 0);
//     // ...and what we really care about is that the delta is correct.
//     for (int k = 0; k < Tau_Global_numCounters; k++) {
//       TotalTime[k] = CurrentTime_0[k] - StartTime[k];
//     }
//   } else {
//     for (int k = 0; k < Tau_Global_numCounters; k++) {
//       TotalTime[k] = CurrentTime[k] - StartTime[k];
// #ifdef DEBUG_PROF
//       printf("CurrentTime[%d] = %f\n", k, CurrentTime[k]);
//       printf("StartTime[%d]   = %f\n", k, StartTime[k]);
//       printf("TotalTime[%d]   = %f\n", k, TotalTime[k]);
// #endif /* DEBUG_PROF */
//     }
//   }
  if (CurrentTime[0] != 0.0 && StartTime[0] == 0.0) {
    TauMetrics_getDefaults(tid, StartTime, 0);
  }
  for (int k = 0; k < Tau_Global_numCounters; k++) {
    TotalTime[k] = CurrentTime[k] - StartTime[k];
  }

  x_uint64 TimeStamp = 0L;
  TimeStamp = (x_uint64)CurrentTime[0];    // USE COUNTER1

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

  /*** Profile Compensation ***/
  if (TauEnv_get_compensate()) {
    double *tover, *tnull;
    tover = TauGetTimerOverhead(TauFullTimerOverhead);
    tnull = TauGetTimerOverhead(TauNullTimerOverhead);

    for (int k = 0; k < Tau_Global_numCounters; k++) {
      /* To compensate for timing overhead, shrink the totaltime! */
      TotalTime[k] = TotalTime[k] - tnull[k] - GetNumChildren() * tover[k];
      if (TotalTime[k] < 0) {
        TotalTime[k] = 0;
        DEBUGPROFMSG("TotalTime[" <<k<<"] negative in "<<ThisFunction->GetName()<<endl;);
      }
    }
  }

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/

#ifdef TAU_MPITRACE
  if (RecordEvent) {
#endif /* TAU_MPITRACE */
  if (TauEnv_get_tracing()) {
    TauTraceEvent(ThisFunction->GetFunctionId(), -1 /* exit */, tid, TimeStamp, 1 /* use supplied timestamp */, TAU_TRACE_EVENT_KIND_FUNC);
    TauMetrics_triggerAtomicEvents(TimeStamp, CurrentTime, tid);
  }
#ifdef TAU_MPITRACE
}
#endif /* TAU_MPITRACE */

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_callsite()) {
    CallSiteStop(TotalTime, tid, TimeStamp);
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/

  if (TauEnv_get_callpath()) {
    CallPathStop(TotalTime, tid);
  }

#ifdef RENCI_STFF
  if (TauEnv_get_callpath()) {
    RenciSTFF::recordValues(CallPathFunction, TimeStamp, TotalTime, tid);
  } else {
    RenciSTFF::recordValues(ThisFunction, TimeStamp, TotalTime, tid);
  }
#endif //RENCI_STFF
#ifdef TAU_PROFILEPARAM
  ProfileParamStop(TotalTime, tid);
  if (ParentProfiler && ParentProfiler->ProfileParamFunction) {
    /* Increment the parent's NumSubrs and decrease its exclude time */
    ParentProfiler->ProfileParamFunction->ExcludeTime(TotalTime, tid);
  }
#endif /* TAU_PROFILEPARAM */

  if (AddInclFlag == true) {    // The first time it came on call stack
    ThisFunction->SetAlreadyOnStack(false, tid);    // while exiting

    // And its ok to add both excl and incl times
    ThisFunction->AddInclTime(TotalTime, tid);
  }
  // If its already on call stack, don't change AlreadyOnStack
  ThisFunction->AddExclTime(TotalTime, tid);
  // In either case we need to add time to the exclusive time.

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
#ifdef KTAU_DEBUGPROF
  /* Checking to see if kern_time < tot-time (user+kern time) */
  ThisKtauProfiler->VerifyMerge(ThisFunction);
#endif /*KTAU_DEBUGPROF*/
#endif /*TAUKTAU && TAUKTAU_MERGE*/

  if (TauEnv_get_compensate()) {
    ThisFunction->ResetExclTimeIfNegative(tid);

    if (TauEnv_get_callpath()) {
      if (ParentProfiler != NULL) {
        CallPathFunction->ResetExclTimeIfNegative(tid);
      }
    }
    if (TauEnv_get_callsite()) {
      if (ParentProfiler != NULL) {
        if (CallSiteFunction != NULL) {
          CallSiteFunction->ResetExclTimeIfNegative(tid);
        }
      }
    }

#ifdef TAU_PROFILEPARAM
    if (ProfileParamFunction != NULL) {
      ProfileParamFunction->ResetExclTimeIfNegative(tid);
    }
#endif /* TAU_PROFILEPARAM */
  }

  if (ParentProfiler != NULL) {
    ParentProfiler->ThisFunction->ExcludeTime(TotalTime, tid);

    if (TauEnv_get_compensate()) {
      ParentProfiler->AddNumChildren(GetNumChildren() + 1);
      /* Add 1 and my children to my parents total number of children */
    }
  }

  /********************************************************************************/
  /*** Throttling Code ***/
  /********************************************************************************/
  if (TauEnv_get_throttle()) {
    /* if the frequency of events is high, disable them */
    long numCalls = ThisFunction->GetCalls(tid);
    double inclusiveTime = ThisFunction->GetInclTimeForCounter(tid, 0);
    /* here we get the array of double values representing the double
     metrics. We choose the first counter */

    /* Putting AddInclFlag means we can't throttle recursive calls */
    if (AddInclFlag &&
        (numCalls > TauEnv_get_throttle_numcalls()) &&
        (inclusiveTime / numCalls < TauEnv_get_throttle_percall()))
    {
      RtsLayer::LockDB();
      ThisFunction->SetProfileGroup(TAU_DISABLE);
      ThisFunction->SetPrimaryGroupName("TAU_DISABLE");
      ThisFunction->SetType("[THROTTLED]");
      RtsLayer::UnLockDB();
      TAU_VERBOSE("TAU<%d,%d>: Throttle: Disabling %s\n",
          RtsLayer::myNode(), RtsLayer::myThread(), ThisFunction->GetName());
    }
  }
  /********************************************************************************/
  /*** Throttling Code ***/
  /********************************************************************************/

  if (( TauEnv_get_recycle_threads() && (!ParentProfiler && tid == 0)) ||
      (!TauEnv_get_recycle_threads() && !ParentProfiler)) {
    /*** Profile Compensation ***/
    // If I am still compensating, I do not expect a top level timer. Just pretend
    // this never happened.
    if (TauEnv_get_compensate() && !TauCompensateInitialized()) return;

    /* Should we detect memory leaks here? */
    if (TheSafeToDumpData() && !RtsLayer::isCtorDtor(ThisFunction->GetName())) {
      Tau_detect_memory_leaks();
      /* the last event should be before final exit */
    }

    /* On Crays with -iowrapper, rank 0 is spawned by the clone syscall. This
     creates a parent thread (rank = -1) that tries to write data at the end
     of execution and crashes. This fixes it and disables profile output from
     rank -1. */
#if (defined (TAU_MPI) && defined(TAU_CRAYCNL))
    if (RtsLayer::myNode() == -1) TheSafeToDumpData() = 0;
#endif /* TAU_MPI && TAU_CRAYCNL */

    // For Dyninst. tcf gets called after main and all the data structures may not be accessible
    // after main exits. Still needed on Linux - we use TauProgramTermination()
    if (strcmp(ThisFunction->GetName(), "_fini") == 0) {
      TheSafeToDumpData() = 0;
    }
#ifdef TAU_GPU
    //Stop all other running tasks.
    if (tid == 0) {
      Tau_profile_exit_most_threads();
    }
#endif
    if (TheSafeToDumpData()) {
      if (!RtsLayer::isCtorDtor(ThisFunction->GetName())) {
        // Not a destructor of a static object - its a function like main

        // Write profile data
#ifndef TAU_SCOREP
        TauProfiler_StoreData(tid);
#endif // TAU_SCOREP 
        TAU_VERBOSE("TAU: <Node=%d.Thread=%d>:<pid=%d>: %s initiated TauProfiler_StoreData\n", RtsLayer::myNode(),
            RtsLayer::myThread(), RtsLayer::getPid(), ThisFunction->GetName());
// Be careful here, we can not disable instrumentation in multithreaded
// application because that will cause profilers on any other stack to never get
// stopped.
#if defined(TAU_DMAPP) && TAU_MAX_THREADS == 1
        if (RtsLayer::myThread() == 0) {
          TAU_DISABLE_INSTRUMENTATION();
        }
#endif /* TAU_DMAPP */

#if defined(TAUKTAU)
        //AN Removed - New func inside
        //ThisKtauProfiler->KernProf.DumpKProfile();
        ThisKtauProfiler->KernProf.DumpKProfileOut();
#endif /*TAUKTAU */

#ifdef TAU_TRACK_IDLE_THREADS /* Check if we need to shut off .TAU applications on other tids */
	//Disable if already stopping all threads
        if ((tid == 0) && (!Tau_check_Stopping_All_Threads())) {
          int i;
          for (i = 1; i < TheProfilerThreadList().size(); i++) {
            /* for all other threads */
            Profiler *cp = TauInternal_CurrentProfiler(i);
            if (cp && strncmp(cp->ThisFunction->GetName(), ".TAU", 4) == 0) {
              bool uselasttimestamp = true;
              cp->Stop(i, uselasttimestamp); /* force it to write the data*/
            }
          }
        }
#endif /* TAU_TRACK_IDLE_THREADS */
      }
    }
  }

  /********************************************************************************/
  /*** KTAU Code ***/
  /********************************************************************************/
#if defined(PROFILING_ON) && defined(TAUKTAU)
  KtauProfiler::PutKtauProfiler();
#endif /* TAUKTAU */

  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.function_exit) {
    Tau_plugin_event_function_exit_data_t plugin_data;
    plugin_data.timer_name = ThisFunction->GetName();
    plugin_data.func_id = ThisFunction->GetFunctionId();
    plugin_data.timer_group = ThisFunction->GetAllGroups();
    plugin_data.tid = tid;
    plugin_data.timestamp = TimeStamp;
    plugin_data.metrics = &(TotalTime[0]);
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_FUNCTION_EXIT, ThisFunction->GetName(), &plugin_data);
  }
}

//////////////////////////////////////////////////////////////////////

void TauProfiler_theFunctionList(const char ***inPtr, int *numFuncs, bool addName, const char * inString)
{
  TauInternalFunctionGuard protects_this_function;

  static int numberOfFunctions = 0;

  if (addName) {
    numberOfFunctions++;
  } else {
    //We do not want to pass back internal pointers.
    *inPtr = (char const **)malloc(sizeof(char *) * numberOfFunctions);

  // don't iterate over the FunctionInfo vector without the lock!
  RtsLayer::LockDB();
    for (int i = 0; i < numberOfFunctions; i++) {
      (*inPtr)[i] = TheFunctionDB()[i]->GetName();
    }
  RtsLayer::UnLockDB();
    *numFuncs = numberOfFunctions;
  }
}

void TauProfiler_dumpFunctionNames()
{
  TauInternalFunctionGuard protects_this_function;
  if(!TheSafeToDumpData()) {
    return;
  }

  int numFuncs;
  const char ** functionList;

  TauProfiler_theFunctionList(&functionList, &numFuncs);

  const char *dirname = TauEnv_get_profiledir();

  //Create temp write to file.
  char filename[1024];
  snprintf(filename, sizeof(filename),  "%s/temp.%d.%d.%d", dirname, RtsLayer::myNode(), RtsLayer::myContext(), RtsLayer::myThread());

  FILE* fp;
  if ((fp = fopen(filename, "w+")) == NULL) {
    char errormsg[1048];
    snprintf(errormsg, sizeof(errormsg),  "Error: Could not create %s", filename);
    perror(errormsg);
    return;
  }

  //Write data, and close.
  fprintf(fp, "number of functions %d\n", numFuncs);
  for (int i = 0; i < numFuncs; i++) {
    fprintf(fp, "%s\n", functionList[i]);
  }
  fclose(fp);

  //Rename from the temp filename.
  char dumpfile[1024];
  snprintf(dumpfile, sizeof(dumpfile),  "%s/dump_functionnames_n,c,t.%d.%d.%d", dirname, RtsLayer::myNode(), RtsLayer::myContext(),
      RtsLayer::myThread());
  rename(filename, dumpfile);
}

void TauProfiler_getUserEventList(const char ***inPtr, int *numUserEvents)
{
  TauInternalFunctionGuard protects_this_function;

  *numUserEvents = 0;

  AtomicEventDB::iterator eit;

  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    (*numUserEvents)++;
  }

  *inPtr = (char const **)malloc(sizeof(char*) * *numUserEvents);

  for (int i = 0; i < *numUserEvents; i++) {
    (*inPtr)[i] = TheEventDB()[i]->GetName().c_str();
  }
}

void TauProfiler_getUserEventValues(const char **inUserEvents, int numUserEvents, int **numEvents, double **max,
    double **min, double **mean, double **sumSqr, int tid)
{
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("TAU_GET_EVENT_VALUES()", " ", TAU_IO);

  *numEvents = (int*)malloc(sizeof(int) * numUserEvents);
  *max = (double *)malloc(sizeof(double) * numUserEvents);
  *min = (double *)malloc(sizeof(double) * numUserEvents);
  *mean = (double *)malloc(sizeof(double) * numUserEvents);
  *sumSqr = (double *)malloc(sizeof(double) * numUserEvents);

  RtsLayer::LockDB();

  int idx = 0;
  AtomicEventDB::iterator eit;

  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    for (int i = 0; i < numUserEvents; i++) {
      if (inUserEvents && (*eit)->GetName() == inUserEvents[i]) {
        (*numEvents)[idx] = (*eit)->GetNumEvents(tid);
        (*max)[idx] = (*eit)->GetMax(tid);
        (*min)[idx] = (*eit)->GetMin(tid);
        (*mean)[idx] = (*eit)->GetMean(tid);
        (*sumSqr)[idx] = (*eit)->GetSumSqr(tid);
        idx++;
        break;
      }
    }
  }

  RtsLayer::UnLockDB();

}

double *Profiler::getStartValues()
{
  return StartTime;
}

void TauProfiler_theCounterList(const char ***inPtr, int *numCounters)
{
  TauInternalFunctionGuard protects_this_function;

  *inPtr = (const char **)malloc(sizeof(const char **) * 1);
  const char *tmpChar = "default counter";
  (*inPtr)[0] = tmpChar;
  *numCounters = 1;
}

static bool helperIsFunction(FunctionInfo *fi, Profiler *profiler)
{
  if (TauEnv_get_callpath()) {
    if (fi == profiler->ThisFunction || fi == profiler->CallPathFunction) {
      return true;
    }
  } else {
    if (fi == profiler->ThisFunction) {
      return true;
    }
  }
  return false;
}

void TauProfiler_getFunctionValues(const char **inFuncs, int numFuncs, double ***counterExclusiveValues,
    double ***counterInclusiveValues, int **numCalls, int **numSubr, const char ***counterNames, int *numCounters,
    int tid)
{
  TauInternalFunctionGuard protects_this_function;

  //TAU_PROFILE("TAU_GET_FUNC_VALS()", " ", TAU_IO);

  vector<FunctionInfo*>::iterator it;

  TauMetrics_getCounterList(counterNames, numCounters);

  // allocate memory for the lists
  *counterExclusiveValues = (double **)malloc(sizeof(double *) * numFuncs);
  *counterInclusiveValues = (double **)malloc(sizeof(double *) * numFuncs);
  for (int i = 0; i < numFuncs; i++) {
    (*counterExclusiveValues)[i] = (double *)malloc(sizeof(double) * Tau_Global_numCounters);
    (*counterInclusiveValues)[i] = (double *)malloc(sizeof(double) * Tau_Global_numCounters);
  }
  *numCalls = (int *)malloc(sizeof(int) * numFuncs);
  *numSubr = (int *)malloc(sizeof(int) * numFuncs);

  TauProfiler_updateIntermediateStatistics(tid);

  // don't iterate over the FunctionInfo vector without the lock!
  RtsLayer::LockDB();

  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;

    int funcPos = matchFunction(fi, inFuncs, numFuncs);

    if (funcPos == -1) {    // skip this function
      continue;
    }

    (*numCalls)[funcPos] = fi->GetCalls(tid);
    (*numSubr)[funcPos] = fi->GetSubrs(tid);

    int posCounter = 0;
    for (int m = 0; m < Tau_Global_numCounters; m++) {
      (*counterInclusiveValues)[funcPos][posCounter] = fi->getDumpInclusiveValues(tid)[m];
      (*counterExclusiveValues)[funcPos][posCounter] = fi->getDumpExclusiveValues(tid)[m];
      posCounter++;
    }
  }
  RtsLayer::UnLockDB();
}

static void finalizeTrace(int tid)
{

#ifdef TAU_VAMPIRTRACE
  DEBUGPROFMSG("Calling vt_close()"<<endl;);
  if (RtsLayer::myThread() == 0) {
    vt_close();
  }
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
  DEBUGPROFMSG("Calling elg_close()"<<endl;);
  if (RtsLayer::myThread() == 0) {
    esd_close();
  }
#else /* TAU_EPILOG */

  if (TauEnv_get_tracing()) {
    TauTraceClose(tid);
  }
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
}

void TauProfiler_PurgeData(int tid)
{
  TauInternalFunctionGuard protects_this_function;

  vector<FunctionInfo*>::iterator it;
  AtomicEventDB::iterator eit;
  Profiler *curr;

  DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
  // don't iterate over the FunctionInfo vector without the lock!
  RtsLayer::LockDB();

  // Reset The Function Database
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    (*it)->SetCalls(tid, 0);
    (*it)->SetSubrs(tid, 0);
    (*it)->SetExclTimeZero(tid);
    (*it)->SetInclTimeZero(tid);
  }

  // Reset the Atomit/User Event Database
  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    (*eit)->ResetData(tid);
  }

  if (TauInternal_CurrentProfiler(tid) == NULL) {
    // There are no active timers, we are finished!
    RtsLayer::UnLockDB();
    return;
  }

  // Now Re-register callstack entries
  curr = TauInternal_CurrentProfiler(tid);
  curr->ThisFunction->IncrNumCalls(tid);

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    curr->StartTime[i] = 0;
  }
  RtsLayer::getUSecD(tid, curr->StartTime);

  curr = curr->ParentProfiler;

  while (curr != 0) {
    curr->ThisFunction->IncrNumCalls(tid);
    curr->ThisFunction->IncrNumSubrs(tid);
    for (int i = 0; i < Tau_Global_numCounters; i++) {
      curr->StartTime[i] = 0;
    }
    RtsLayer::getUSecD(tid, curr->StartTime);
    curr = curr->ParentProfiler;
  }

  RtsLayer::UnLockDB();
}

/////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//  Profiler::GetNumChildren()
//  Description: Returns the total number of child timers (including
//  children of children) executed under the given profiler.
//////////////////////////////////////////////////////////////////////
long Profiler::GetNumChildren(void)
{
  return NumChildren;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::SetNumChildren(value)
//  Description: Sets the total number of child timers.
//////////////////////////////////////////////////////////////////////
void Profiler::SetNumChildren(long value)
{
  NumChildren = value;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::AddNumChildren(value)
//  Description: increments by value the number of child timers.
//////////////////////////////////////////////////////////////////////
void Profiler::AddNumChildren(long value)
{
  NumChildren += value;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::GetPhase(void)
//  Description: Returns if a profiler is a phase or not
//////////////////////////////////////////////////////////////////////
#ifdef TAU_PROFILEPHASE
bool Profiler::GetPhase(void) {
  return PhaseFlag;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::GetPhase(bool flag)
//  Description: SetPhase sets a phase to be true or false based on the
//               parameter flag
//////////////////////////////////////////////////////////////////////
void Profiler::SetPhase(bool flag) {
  PhaseFlag = flag;
}
#endif /* TAU_PROFILEPHASE */

bool startsWith(const char *pre, const char *str)
{
    size_t lenpre = strlen(pre),
           lenstr = strlen(str);
    return lenstr < lenpre ? false : memcmp(pre, str, lenpre) == 0;
}

// writes user events to the file
static int writeUserEvents(FILE *fp, int tid)
{
  AtomicEventDB::iterator it;

  fprintf(fp, "0 aggregates\n");    // For now there are no aggregates

  // Print UserEvent Data if any
  int numEvents = 0;
  for (it = TheEventDB().begin(); it != TheEventDB().end(); ++it) {
    if ((*it) && (*it)->GetNumEvents(tid) == 0 && !startsWith("CUDA", (*it)->GetName().c_str())) {    // skip user events with no calls
      continue;
    }
    if ((*it)->GetWriteAsMetric()) { //skip events that are written out as metrics.
      printf("skipping: %s.\n", (*it)->GetName().c_str());
      continue;
    }
    if ((*it)) {
      numEvents++;
    }
  }

  if (numEvents > 0) {
    // Data format
    // # % userevents
    // # name numsamples max min mean sumsqr
    fprintf(fp, "%d userevents\n", numEvents);
    fprintf(fp, "# eventname numevents max min mean sumsqr\n");

    for (it = TheEventDB().begin(); it != TheEventDB().end(); ++it) {
      if ((*it) && (*it)->GetNumEvents(tid) == 0 && !startsWith("CUDA", (*it)->GetName().c_str())) continue;
      if ((*it) && (*it)->GetWriteAsMetric()) continue;
      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", (*it)->GetName().c_str(), (*it)->GetNumEvents(tid),
          (*it)->GetMax(tid), (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
    }
  }
  return 0;
}

static int writeHeader(FILE *fp, int numFunc, char *metricName)
{
  char header[2096];
  snprintf(header, sizeof(header),  "%d %s\n", numFunc, metricName);
  strcat(header, "# Name Calls Subrs Excl Incl ");

  strcat(header, "ProfileCalls");
  fprintf(fp, "%s", header);
  return 0;
}

extern "C" int TauProfiler_updateAllIntermediateStatistics()
{
  TAU_VERBOSE("Updating Intermediate Stats for All %d Threads\n", RtsLayer::getTotalThreads());
  RtsLayer::LockDB();
  for (int tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
    TauProfiler_updateIntermediateStatistics(tid);
  }
  RtsLayer::UnLockDB();

  return 0;
}

// This is a very important function, it must be called before writing function data to disk.
// This function fills in the values that will be dumped to disk.
// It performs the calculations for timers that are still on the stack.
int TauProfiler_updateIntermediateStatistics(int tid)
{

  // get current values for all counters
  double currentTime[TAU_MAX_COUNTERS];
  RtsLayer::getCurrentValues(tid, currentTime);

  // an index for iterating over counters
  int c;

  // don't iterate over the FunctionInfo vector without the lock!
  RtsLayer::LockDB();
  // iterate over all functions in the database.
  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    if(fi==NULL){
	    TAU_VERBOSE("WARNING: NULL FunctionInfoPointer!");
	    continue;
    }
    // get the current "dump" profile for this timer
    double *incltime = fi->getDumpInclusiveValues(tid);
    double *excltime = fi->getDumpExclusiveValues(tid);

    // update the "dump" profile with the currently stored values...
    // ...including timers that are still running!
    fi->getInclusiveValues(tid, incltime);
    fi->getExclusiveValues(tid, excltime);

    if (fi->GetAlreadyOnStack(tid)) {
      // this routine is currently on the callstack
      // we will have to compute the exclusive and inclusive time it has accumulated

      // Start with the data already accumulated
      // Then walk the entire stack, when this function the function in question is found we do two things
      // 1) Compute the current amount that should be added to the inclusive time.
      //    This is simply the current time minus the start time of our function.
      //    If a routine is in the callstack twice, only the highest (top-most) value
      //    will be retained, this is correct.
      // 2) Add to the exclusive value by subtracting the start time of the current
      //    child (if there is one) from the duration of this function so far.

      // *CWL* - This is stupid, but I do not currently have the time nor energy to
      //         attempt to refactor, especially since both forms are actively used.
      //         a) incltime and excltime are used for non-threaded programs
      //            Note that getDump*Values(tid) grants pointer-access to the
      //            internal structures stored in the FunctionInfo object.
      //         b) InclTime and ExclTime are used for threaded programs.
      //            Note that InclTime and ExclTime allocates memory.

      // get the pointers to the current profile for this timer
      //double *InclTime = fi->GetInclTime(tid);
      //double *ExclTime = fi->GetExclTime(tid);

      // allocate some placeholders
      double inclusiveToAdd[TAU_MAX_COUNTERS] = {0.0};
      double prevStartTime[TAU_MAX_COUNTERS] = {0.0};

      //TAU_VERBOSE("fi: %s\n", fi->GetName()); fflush(stderr);
      // Iterate over the current timer stack, starting at the innermost function
      for (Profiler *current = TauInternal_CurrentProfiler(tid); current != 0; current = current->ParentProfiler) {
        //TAU_VERBOSE("\tcurrent: %s\n", current->ThisFunction->GetName()); fflush(stderr);
	// is this the current function we are processing?
        if (helperIsFunction(fi, current)) {
	  // iterate over the counters
          for (c = 0; c < Tau_Global_numCounters; c++) {
            // first, get the amount of time elapsed since this function started
            inclusiveToAdd[c] = currentTime[c] - current->getStartValues()[c];
            // second, update the "dump" profile exclusive value
            excltime[c] += inclusiveToAdd[c] - prevStartTime[c];
            // *CWL* - followup to the data structure insanity issues
	    // *KAH* - this is probably a bad idea?
            //ExclTime[c] += inclusiveToAdd[c] - prevStartTime[c];

            /*
             TAU_VERBOSE("\t[%d] %s:\n\t    currentTime=%f\n\t    startValue=%f\n\t    prevStartTime=%f\n\t    excltime=%f\n\t    incltime=%f\n",
             tid, current->ThisFunction->GetName(), currentTime[c], current->getStartValues()[c],
             prevStartTime[c]/1000000.0, excltime[c]/1000000.0, incltime[c]/1000000.0); fflush(stderr);
             */
          }
	  // done with this function, exit the stack loop
	  break;
        } else {
	  // get the start time for this "child" function.
	  // The goal is to get the start time for the immediate child of "fi".
          for (c = 0; c < Tau_Global_numCounters; c++) {
            prevStartTime[c] = currentTime[c] - current->getStartValues()[c];
          }
	}
      }
      for (c = 0; c < Tau_Global_numCounters; c++) {
        incltime[c] += inclusiveToAdd[c];
        // *CWL* - followup to the data structure insanity issues
	// *KAH* - this is probably a bad idea?!!
        //InclTime[c] += inclusiveToAdd[c];
      }

      // *CWL* - followup to the data structure insanity issues
	// *KAH* - this is probably a bad idea?!!
      //fi->SetInclTime(tid, InclTime);
      //fi->SetExclTime(tid, ExclTime);
      //free(InclTime);
      //free(ExclTime);
    }
  }
  RtsLayer::UnLockDB();
  return 0;
}

// Checks if a function matches in a list of strings
// returns -1 if not found, otherwise, returns the index that it is found at
// if inFuncs is NULL, or numFuncs is 0, it returns 0
static int matchFunction(FunctionInfo *fi, const char **inFuncs, int numFuncs)
{
  if (numFuncs == 0 || inFuncs == NULL) {
    return 0;
  }
  const char *tmpFunctionName = fi->GetName();
  for (int i = 0; i < numFuncs; i++) {
    if ((inFuncs != NULL) && (strcmp(inFuncs[i], tmpFunctionName) == 0)) {
      return i;
    }
  }
  return -1;
}


// Writes function event data
static int writeFunctionData(FILE *fp, int tid, int metric, const char **inFuncs, int numFuncs)
{
  // don't iterate over the FunctionInfo vector without the lock!
  RtsLayer::LockDB();
    for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
        FunctionInfo & fi = **it;

        if (!fi.GetCalls(tid) || -1 == matchFunction(&fi, inFuncs, numFuncs)) {
            continue;
        }

        bool found_one = false;
        char const * const atomic_metric = TauMetrics_getMetricAtomic(metric);
        if (atomic_metric) {
            for (AtomicEventDB::iterator it2 = TheEventDB().begin(); it2 != TheEventDB().end(); ++it2) {
                TauUserEvent *ue = *it2;

                char const * str = ue->GetName().c_str();
                char const * suffix = fi.GetName();
                if (!str || !suffix) continue;

                size_t lenstr = strlen(str);
                size_t lensuffix = strlen(suffix);
                if (lensuffix > lenstr) continue;

                //printf("testing: %s vs. %s.\n", atomic_metric, str);
                if (strncmp(str+lenstr-lensuffix, suffix, lensuffix) == 0 &&
                    strncmp(atomic_metric, str, strlen(atomic_metric)) == 0)
                {
                    double excltime = ue->GetMean(tid);
                    double incltime = excltime;
                    int calls = fi.GetCalls(tid);
                    fprintf(fp, "\"%s\" %ld %ld %.16G %.16G ", suffix, (long int) calls, 0L, excltime, incltime);
                    fprintf(fp, "0 ");    // Indicating that profile calls is turned off
                    fprintf(fp, "GROUP=\"%s\" \n", fi.GetAllGroups());
                    found_one = true;
                    break; // it2
                }
            } // for (it2)
        } // if (atomic_metric)

        if (found_one) continue;

#ifdef CUPTI
        // Is a Cupti metric
        if (TauMetrics_getIsCuptiMetric(metric) == TAU_METRIC_CUPTI_METRIC) {
            char const * const metric_name = TauMetrics_getMetricName(metric);
            char const * const tau_cuda_device_name = TauEnv_get_cuda_device_name();

            // Iterate over devices... seems wrong.  Probably should prefix device name to metric in TAU_METRICS
            int deviceCount;
            CUresult result = cuDeviceGetCount(&deviceCount);
            if (result != CUDA_SUCCESS) {
                if (result == CUDA_ERROR_NOT_INITIALIZED) {
                    cuInit(0);
                    result = cuDeviceGetCount(&deviceCount);
                }
                if (result != CUDA_SUCCESS) {
                    char const * err_str;
                    cuGetErrorString(result, &err_str);
                    fprintf(stderr, "cuDeviceGetCount failed: %s\n", err_str);
                    // no device found.
                    continue;
                }
            }
            for(int dev=0; dev<deviceCount; ++dev) {
                CUptiResult result;
                CUdevice device;
                cudaDeviceProp deviceProps;
                if (cuDeviceGet(&device, dev) != CUDA_SUCCESS) {
                    fprintf(stderr, "Could not get device %d.\n", dev);
                    continue;
                }

                // Check if metric is a CUPTI metric we can calculate on this device
                CUpti_MetricID metricID;
                result = cuptiMetricGetIdFromName(device, metric_name, &metricID);
                if (result != CUPTI_SUCCESS) {
                    cout << "TAU: NOTE: Cannot calculate '" << metric_name << "' on device " << dev << endl;
                    continue;
                }

                // Get the device name to be used in the event name below
                cudaGetDeviceProperties(&deviceProps, dev);
                std::string device_name = deviceProps.name;
                //std::replace(device_name.begin(), device_name.end(), ' ', '_');
                // PGI compiler has some issues with c++11.
                Tau_util_replaceStringInPlace(device_name, " ", "_");
                if (tau_cuda_device_name && strcmp(tau_cuda_device_name, device_name.c_str())) {
                    continue;
                }

                // Get the list of events required to calculate this metric on this device
                uint32_t numMetricEvents;
                result = cuptiMetricGetNumEvents(metricID, &numMetricEvents);
                if (result != CUPTI_SUCCESS) {
                    fprintf(stderr, "cuptiMetricGetNumEvents failed on device %d\n", dev);
                    continue;
                }
                ScopedArray<CUpti_EventID> metricEvents(numMetricEvents);
                result = cuptiMetricEnumEvents(metricID, &metricEvents.size, metricEvents);
                if (result != CUPTI_SUCCESS) {
                    fprintf(stderr, "cuptiMetricEnumEvents failed on device %d\n", dev);
                    continue;
                }

                // Get the values of required events
                ScopedArray<uint64_t> eventValues(numMetricEvents);
                memset(eventValues, 0, eventValues.size);
                for (int i = 0; i < numMetricEvents; i++) {
                    int eventIndex = TauMetrics_getEventIndex(metricEvents[i]);
                    char const * const event_name = TauMetrics_getMetricName(eventIndex);

                    for (AtomicEventDB::iterator it2 = TheEventDB().begin(); it2 != TheEventDB().end(); ++it2) {
                        TauUserEvent *ue = *it2;

                        const char *str = ue->GetName().c_str();
                        const char *suffix = fi.GetName();
                        if (!str || !suffix) continue;

                        size_t lenstr = strlen(str);
                        size_t lensuffix = strlen(suffix);
                        if (lensuffix > lenstr) continue;

                        if (strncmp(str+lenstr-lensuffix, suffix, lensuffix) == 0 &&
                            strncmp(str, event_name, strlen(event_name)) == 0)
                        {
                            eventValues[i] = ue->GetMean(tid);
                            break;
                        }
                    }
                } // for (i)

                // Get inclusive time for CUPTI metric calculation
                double incltime = fi.getDumpInclusiveValues(tid)[TauMetrics_getTimeMetric()];

                // Calculate value of Cupti metric
                CUpti_MetricValue inclmetric;
                cuptiMetricGetValue(device, metricID, metricEvents.size, metricEvents,
                                    eventValues.size, eventValues, incltime, &inclmetric);
                CUpti_MetricValue exclmetric = inclmetric;

                fprintf(fp, "\"%s", fi.GetName());
                if (strlen(fi.GetType()) > 0)
                    fprintf(fp, " %s", fi.GetType());
                fprintf(fp, "\" %ld %ld %.16G %.16G ", fi.GetCalls(tid), fi.GetSubrs(tid),
                        exclmetric.metricValueDouble, inclmetric.metricValueDouble);
            } // for (dev)
        } else {
#endif // CUPTI
            double incltime = fi.getDumpInclusiveValues(tid)[metric];
            double excltime = fi.getDumpExclusiveValues(tid)[metric];

            /*Do not resolve addresses if they have already been resolved eagerly*/
            if(strstr(fi.GetName(), " ADDR <") != NULL && !TauEnv_get_ompt_resolve_address_eagerly()) {
              char resolved_address[10240] = "";
              Tau_ompt_resolve_callsite(fi, resolved_address);
              fprintf(fp, "\"%s", resolved_address);
            } else {
              fprintf(fp, "\"%s", fi.GetName());
            }
            if (strlen(fi.GetType()) > 0)
                fprintf(fp, " %s", fi.GetType());
            if(fi.StartAddr || fi.StopAddr) {
                fprintf(fp, "[{%#lx}-{%#lx}]", fi.StartAddr, fi.StopAddr);
            }
            fprintf(fp, "\" %ld %ld %.16G %.16G ", fi.GetCalls(tid), fi.GetSubrs(tid), excltime, incltime);
#ifdef CUPTI
        }
#endif //CUPTI

        fprintf(fp, "0 ");    // Indicating that profile calls is turned off
        fprintf(fp, "GROUP=\"%s\" \n", fi.GetAllGroups());

    } // for (it)
  RtsLayer::UnLockDB();

    return 0;
}

// Writes function event data
static int getTrueFunctionCount(int count, int tid, const char **inFuncs, int numFuncs, int metric)
{
  int trueCount = count;

  AtomicEventDB::iterator it2;
  const char *metricName = TauMetrics_getMetricAtomic(metric);

  // don't iterate over the FunctionInfo vector without the lock!
  RtsLayer::LockDB();
  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;

    if (-1 == matchFunction(*it, inFuncs, numFuncs)) {    // skip this function
      trueCount--;
    } else if (fi->GetCalls(tid) == 0) {
      trueCount--;
    }
    if (metricName != NULL)
    {
        int tempCount = 0;
        for (it2 = TheEventDB().begin(); it2 != TheEventDB().end(); ++it2) {
          TauUserEvent *ue = *it2;

          const char *str = ue->GetName().c_str();
          const char *suffix = fi->GetName();

          if (!str || !suffix)
              continue;
          size_t lenstr = strlen(str);
          size_t lensuffix = strlen(suffix);
          if (lensuffix >  lenstr) {
              continue;
          }
          //printf("testing: %s vs. %s.\n", TauMetrics_getMetricAtomic(metric), str);
          if (strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0 &&
              strncmp(TauMetrics_getMetricAtomic(metric), str, strlen(TauMetrics_getMetricAtomic(metric))) == 0)
          {
              tempCount++;
          }
          /*if (tempCount > 0)
          {
            trueCount += (tempCount - 1)*2; //account for noncontext event and two functions per event (callpath/noncallpath).
          }*/
        }

    }
  }
  RtsLayer::UnLockDB();
  return trueCount;
}

// Writes a single profile file
static int writeProfile(FILE *fp, char *metricName, int tid, int metric, const char **inFuncs, int numFuncs)
{
  int trueCount = getTrueFunctionCount(TheFunctionDB().size(), tid, inFuncs, numFuncs, metric);
  //writeHeader(fp, TheFunctionDB().size(), metricName);
  writeHeader(fp, trueCount, metricName);
  fprintf(fp, " # ");
  Tau_metadata_writeMetaData(fp, metric, tid);
  fprintf(fp, "\n");
  fflush(fp);
  writeFunctionData(fp, tid, metric, inFuncs, numFuncs);
  writeUserEvents(fp, tid);
  fclose(fp);
  return 0;
}

//static int profileWriteCount[TAU_MAX_THREADS];
inline void setProfileWriteCount(int tid, int val){//TODO: DYNATHREAD

    getProfilerData(tid).profileWriteCount=val;
}
inline void incProfileWriteCount(int tid){

    //printf("Write count for tid: %d, pre increment: %d\n",tid,ProfilerThreadList[tid]->profileWriteCount);
    //printf("Vector for tid: %d: %p\n",tid,ProfilerThreadList);
    //printf("Pointer for tid: %d, pre increment: %p\n",tid,ProfilerThreadList[tid]);
    //printf("ProfilerThreadList size: %d, checking tid: %d\n",ProfilerThreadList.size(),tid);
    getProfilerData(tid).profileWriteCount++;

}
inline int getProfileWriteCount(int tid){//TODO: DYNATHREAD

    return getProfilerData(tid).profileWriteCount;
}
static int profileWriteWarningPrinted = 0;

extern "C" int Tau_profiler_initialization()
{
  int i;
  for (i = 1; i < TheProfilerThreadList().size(); i++) {
    setProfileWriteCount(i,0);
  }
  profileWriteWarningPrinted = 0;
  return 0;
}

extern "C" int Tau_print_metadata_for_traces(int tid) {

  MetaDataRepo *localRepo = NULL;
    localRepo = &(Tau_metadata_getMetaData(tid));

   for (MetaDataRepo::iterator it = (*localRepo).begin(); it != (*localRepo).end(); it++) {
      string metadata_str(it->first.name + string(" | ") + string(it->second->data.cval));
      Tau_trigger_userevent(metadata_str.c_str(), 1.0);
  }
  return 0;
}

bool& Tau_is_destroyed(void);

// Store profile data at the end of execution (when top level timer stops)
extern "C" void finalizeCallSites_if_necessary();
int TauProfiler_StoreData(int tid)
{
  TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData\n", RtsLayer::myNode(), tid);
  if(!TheSafeToDumpData()) {
    return -1;
  }
  /* If TAU has already shut down and written data, return.
   * This can happen if a thread outlives thread 0. */
  if (RtsLayer::myThread() > 0 && Tau_is_destroyed()) {
    return -1;
  }
/*#ifdef TAU_ENABLE_ROCM
  TauFlushRocmEventsIfNecessary(tid);
#endif *//* TAU_ENABLE_ROCM */
  TauMetrics_finalize();

#ifndef TAU_MPI
  /*Invoke plugins only if both plugin path and plugins are specified
   *Do this first, because the plugin can write TAU_METADATA as recommendations to the user*/
  if(RtsLayer::myThread() == 0 && tid == 0 && Tau_plugins_enabled.pre_end_of_execution) {
    Tau_plugin_event_pre_end_of_execution_data_t plugin_data;
    plugin_data.tid = tid;
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION, "*", &plugin_data);
  }
#endif

  //TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData 1\n", RtsLayer::myNode(), tid);
  if (TauEnv_get_tracing() && (tid == 0) && (TauEnv_get_trace_format() != TAU_TRACE_FORMAT_OTF2)) {
    Tau_print_metadata_for_traces(tid);
  }

#ifdef TAU_SCOREP
  Tau_write_metadata_records_in_scorep(tid);
#endif /* TAU_SCOREP */
  incProfileWriteCount(tid);
  // if ((tid != 0) && (profileWriteCount[tid] > 1)) return 0;
#if !defined(PTHREADS)
  // Rob:  Needed to evaluate for kernels to show in profiles (ignore dreaded #2 thread)!
  if ((tid != 0) && (getProfileWriteCount(tid) > 1)) {
    TAU_VERBOSE("[Profiler]: TauProfiler_StoreData: returning, tid: %i, profileWriteCount[%i]: %i\n", tid, tid, getProfileWriteCount(tid));
    return 0;
  }
#endif
  //TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData 2\n", RtsLayer::myNode(), tid);
  if (getProfileWriteCount(tid) == 10) {
    RtsLayer::LockDB();
    if (profileWriteWarningPrinted == 0) {
      profileWriteWarningPrinted = 1;
      fprintf(stderr, "TAU: Warning: Profile data for at least one thread has been written out more than 10 times!\n"
          "TAU: This could cause extreme overhead and be due to an error\n"
          "TAU: in instrumentation (lack of top level timer).\n"
          "TAU: If using OpenMP, make sure -opari is enabled.\n");
    }
    RtsLayer::UnLockDB();
  }
  finalizeTrace(tid);
#if defined(TAU_SHMEM)
  // If we are using SHMEM, we have to delay finalization until TAU has finalized traces,
  // as OTF2 must communicate over SHMEM in order to write the global definitions file,
  // so the wrapper's version of shmem_finalize skips finalization and we do it here instead.
  // We first check the Tau_get_usesSHMEM() flag, which is set in the wrapper's shmem_init,
  // to avoid calling __real_shmem_finalize twice if the wrapper was not used.
  if(Tau_get_usesSHMEM() && !(TauEnv_get_profile_format() == TAU_FORMAT_MERGED)) {
    __real_shmem_finalize();
  }
#endif

  //TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData 3\n", RtsLayer::myNode(), tid);

  Tau_MemMgr_finalizeIfNecessary();

#ifndef TAU_WINDOWS
// #ifndef _AIX
  if (TauEnv_get_callsite()) {
    TAU_VERBOSE("finalizeCallSites_if_necessary: Total threads = %d\n", RtsLayer::getTotalThreads());
    finalizeCallSites_if_necessary();
  }

  if (TauEnv_get_ebs_enabled()) {
    // Tau_sampling_finalize(tid);
    Tau_sampling_finalize_if_necessary(tid);
  }
// #endif /* _AIX */
#endif
  if (TauEnv_get_profiling()) {
    if (TauEnv_get_profile_format() == TAU_FORMAT_SNAPSHOT) {
      Tau_snapshot_writeFinal("final");
	}
    if (TauEnv_get_profile_format() == TAU_FORMAT_PROFILE) {
      TauProfiler_DumpData(false, tid, "profile");
	}
  }
  /* If we have thread recycling enabled, threads won't write
   * their profiles when they exit.  So thread 0 has to do it,
   * even in cases where CUDA is used without pthread or openmp
   * support.  For some reason, thread 0 is getting its myThread()
   * value changed from 0, still need to investigate that. */
    if (RtsLayer::myThread() == 0 && tid == 0) {
        if (TauEnv_get_recycle_threads()) {
            /* clean up other threads? */
            for (int i = 1; i < RtsLayer::getTotalThreads(); i++) {
                TAU_VERBOSE("Thread 0 checking other threads... i = %d\n", i);
                if (TauInternal_CurrentProfiler(i)) {
                    TAU_VERBOSE("Thread 0 writing data for thread %d\n", i);
                    TauProfiler_StoreData(i);
                }
            }
        }
#ifndef TAU_MPI
#ifndef TAU_SHMEM
	/* Only thread 0 should create a merged profile. */
    if (TauEnv_get_profile_format() == TAU_FORMAT_MERGED) {
      if(TauEnv_get_merge_metadata()) {
        Tau_metadataMerge_mergeMetaData();
      }
      /* Create a merged profile if requested */
      Tau_mergeProfiles_MPI();
	}
#endif
#endif
  }
  //TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData 4\n", RtsLayer::myNode(), tid);

#if defined(TAU_SHMEM) && !defined(TAU_MPI)
  if (TauEnv_get_profile_format() == TAU_FORMAT_MERGED) {
    Tau_global_setLightsOut();
    if(TauEnv_get_merge_metadata()) {
      Tau_metadataMerge_mergeMetaData_SHMEM();
    }
    Tau_mergeProfiles_SHMEM();
    __real_shmem_finalize();
  }
#endif /* TAU_SHMEM */

  //TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData 5\n", RtsLayer::myNode(), tid);
  /*Invoke plugins only if both plugin path and plugins are specified
   *Do this first, because the plugin can write TAU_METADATA as recommendations to the user*/
  if(RtsLayer::myThread() == 0 && tid == 0 && Tau_plugins_enabled.end_of_execution) {
    Tau_plugin_event_end_of_execution_data_t plugin_data;
    plugin_data.tid = tid;
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_END_OF_EXECUTION, "*", &plugin_data);
  }
  ////TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData 6\n", RtsLayer::myNode(), tid);
/* static dtors cause a crash. This could fix it */
#ifdef TAU_SCOREP
  TAU_VERBOSE("TAU<%d,%d>: Turning off the lights... \n", RtsLayer::myNode(), tid);
  Tau_global_setLightsOut();
#endif /* TAU_SCOREP */


#ifdef TAU_ENABLE_ROCTRACER
  Tau_roctracer_stop_tracing();
#endif /* TAU_ENABLE_ROCTRACER */

  return 1;
}

// Returns directory name for the location of a particular metric
static int getProfileLocation(int metric, char *str)
{
    char const * profiledir;
#if defined(KTAU_NG)
    int written_bytes = 0;
    unsigned int profile_dir_len = KTAU_NG_PREFIX_LEN + HOSTNAME_LEN;
    profiledir = new char[profile_dir_len];
    written_bytes = snprintf(profiledir, profile_dir_len,  "%s.", KTAU_NG_PREFIX);
    gethostname(profiledir + written_bytes, profile_dir_len - written_bytes);
#else
    profiledir = TauEnv_get_profiledir();
#endif

    if (Tau_Global_numCounters - Tau_Global_numGPUCounters <= 1) {
        sprintf(str, "%s", profiledir);
    } else {
#ifdef DEBUGPROF
        cout << "metric: " << metric << endl;
#endif /* DEBUGPROF */
        string metricStr = string(TauMetrics_getMetricName(metric));
#ifdef DEBUGPROF
        cout << "metricStr: " << metricStr << endl;
#endif /* DEBUGPROF */

        //sanitize metricName before creating a directory name from it.
        string illegalChars("/\\?%*:|\"<>= ");
        size_t found = metricStr.find_first_of(illegalChars, 0);
        while (found != string::npos) {
            metricStr[found] = '_';
            found = metricStr.find_first_of(illegalChars, found+1);
        }
        sprintf(str, "%s/MULTI__%s", profiledir, metricStr.c_str());
    }
    return 0;
}

int TauProfiler_DumpData(bool increment, int tid, const char *prefix)
{
  TAU_VERBOSE("TAU<%d,%d>: TauProfiler_DumpData\n", RtsLayer::myNode(), tid);
  if(!TheSafeToDumpData()) {
    return -1;
  }

  int rc = TauProfiler_writeData(tid, prefix, increment);

  return rc;
}

void getMetricHeader(int i, char *header)
{
  sprintf(header, "templated_functions_MULTI_%s", RtsLayer::getCounterName(i));
}

// Stores profile data
int TauProfiler_writeData(int tid, const char *prefix, bool increment, const char **inFuncs, int numFuncs)
{
  if(!TheSafeToDumpData()) {
    return -1;
  }

  TauProfiler_updateIntermediateStatistics(tid);

  RtsLayer::LockDB();

  //If we haven't created any directories yet go ahead and keep checking until we have. Otherwise we may give up before initializing metrics
  static bool createdDirectories=false;
  bool createFlag=false;

  if(!createdDirectories){
    createFlag = TauProfiler_createDirectories();
      if (createFlag) {
        createdDirectories=true;
      }
   }
//#ifdef CUPTI
//  CUdevice device;
//  int retval;
//  int er, err;
//  int dev, deviceCount;
//  int metricid;
//  int numEvents;
//  CUpti_EventID *eventIdArray;
//  size_t eventValueArraySizeBytes, eventIdArraySizeBytes;
//  uint64_t *eventValueArray, timeDuration;
//  CUpti_MetricValue *metricValue;
//
//  er = cuDeviceGetCount(&deviceCount);
//  if (er == CUDA_ERROR_NOT_INITIALIZED) {
//    cuInit(0);
//    er = cuDeviceGetCount(&deviceCount);
//  }
//  if (er == CUDA_SUCCESS) {
//    dev = 0;
//    {
//      retval = cuDeviceGet(&device, dev);
//      if(retval != CUDA_SUCCESS) {
//        fprintf(stderr, "Could not get device %d.\n", dev);
//      }
//    }
//  }
//#endif //CUPTI

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    if (TauMetrics_getMetricUsed(i)) {

      char metricHeader[1024];
      char profileLocation[1024];
      FILE* fp;

      getMetricHeader(i, metricHeader);
      if (TauMetrics_getIsCuptiMetric(i) == TAU_METRIC_CUPTI_METRIC) continue;
      if (TauMetrics_getIsCuptiMetric(i) == TAU_METRIC_CUPTI_EVENT) continue;

      //cout << "metric name: " << metricHeader << endl;
#ifdef CUPTI
      // Is a Cupti event, do not record
      //if(TauMetrics_getIsCuptiMetric(i) == 1) continue;
      // Is a Cupti metric
      //if(TauMetrics_getIsCuptiMetric(i) == 2)
      //{
      //  cout << TauMetrics_getMetricName(i) << endl;
      //  cuptiMetricGetIdFromName(device, TauMetrics_getMetricName(i), &metricid); // Get metric id
      //  // Get events
      //  cuptiMetricGetNumEvents(metricid, &numEvents);
      //  eventIdArraySizeBytes = numEvents * sizeof(CUpti_EventID);
      //  eventIdArray = (CUpti_EventID *) malloc(numEvents*sizeof(CUpti_EventID));
      //  cuptiMetricEnumEvents(metricid, &eventIdArraySizeBytes, eventIdArray);
      //  eventValueArraySizeBytes = numEvents*sizeof(uint64_t);
      //  // Calculate value of Cupti metric
      //  cuptiMetricGetValuea(device, CUpti_MetricID metricid,
      //                                   eventIdArraySizeBytes,
      //                                   eventIdArray,
      //                                   eventValueArraySizeBytes,
      //                                   eventValueArray,
      //                                   NULL, //uint64_t timeDuration,
      //                                   &metricValue);
      //}
#endif //CUPTI
      getProfileLocation(i, profileLocation);
//       sprintf(filename, "%s/temp.%d.%d.%d", profileLocation,
//         RtsLayer::myNode(), RtsLayer::myContext(), tid);

      const char *selectivePrefix = "";
      if (numFuncs > 0) {
        selectivePrefix = "sel_";
      }

      char dumpfile[1128];
      if (increment) {
        // place date and time in the filename
        time_t theTime = time(NULL);
        char *stringTime = ctime(&theTime);
        //tm *structTime = localtime(&theTime);
        char *day = strtok(stringTime, " ");
        char *month = strtok(NULL, " ");
        char *dayInt = strtok(NULL, " ");
        char *time = strtok(NULL, " ");
        char *year = strtok(NULL, " ");
        // remove newline
        year[4] = '\0';
        char newStringTime[64];
        snprintf(newStringTime, sizeof(newStringTime),  "%s-%s-%s-%s-%s", day, month, dayInt, time, year);

        snprintf(dumpfile, sizeof(dumpfile),  "%s/%s%s__%s__.%d.%d.%d", profileLocation, selectivePrefix, prefix, newStringTime,
            RtsLayer::myNode(), RtsLayer::myContext(), tid);

        if ((fp = fopen(dumpfile, "w+")) == NULL) {
          char errormsg[1256];
          snprintf(errormsg, sizeof(errormsg),  "Error: Could not create %s", dumpfile);
          perror(errormsg);
          return 0;
        }

        char cwd[1024];
        char *tst = getcwd(cwd, 1024);
		if (tst == NULL) {
          char errormsg[1024];
          snprintf(errormsg, sizeof(errormsg),  "Error: Could not get current working directory");
          perror(errormsg);
          return 0;
		}
        TAU_VERBOSE("[pid=%d], TAU: Writing A profile %s, cwd = %s\n", RtsLayer::getPid(), dumpfile, cwd);
      } else {
        int flags = O_CREAT | O_EXCL | O_WRONLY;
#ifdef TAU_DISABLE_SIGUSR
        int mode = 0;
#else
        int mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
#endif
        int node = RtsLayer::myNode();

        snprintf(dumpfile, sizeof(dumpfile),  "%s/%s%s.%d.%d.%d", profileLocation, selectivePrefix, prefix, node, RtsLayer::myContext(),
            tid);

        int sicortex = 0;
#ifdef TAU_SICORTEX
        sicortex = 1;
#endif

        if (sicortex && !Tau_get_usesMPI()) {
          int test = open(dumpfile, flags, mode);
          while (test == -1 && node < 99999) {
            node++;
            snprintf(dumpfile, sizeof(dumpfile),  "%s/%s%s.%d.%d.%d", profileLocation, selectivePrefix, prefix, node, RtsLayer::myContext(),
                tid);
            test = open(dumpfile, flags, mode);
          }
          if ((fp = fdopen(test, "w")) == NULL) {
            char errormsg[1256];
            snprintf(errormsg, sizeof(errormsg),  "Error: Could not create %s", dumpfile);
            perror(errormsg);
            return 0;
          }

        } else {
#ifdef TAU_MPI
#ifndef TAU_SHMEM
        if (Tau_get_usesMPI())
#endif /* TAU_SHMEM */
        {
#endif /* TAU_MPI */
          if ((fp = fopen(dumpfile, "w+")) == NULL) {
            char errormsg[1256];
            snprintf(errormsg, sizeof(errormsg),  "Error: Could not create %s", dumpfile);
            perror(errormsg);
            return 0;
          }
#ifdef TAU_MPI
        }
#endif /* TAU_MPI */
          char cwd[1024];
          char *tst = getcwd(cwd, 1024);
		  if (tst == NULL) {
            char errormsg[1024];
            snprintf(errormsg, sizeof(errormsg),  "Error: Could not get current working directory");
            perror(errormsg);
            return 0;
		  }
          TAU_VERBOSE("[pid=%d], TAU: Writing B profile %s, cwd = %s\n", RtsLayer::getPid(), dumpfile, cwd);
        }
      }
      TAU_VERBOSE("[pid=%d], TAU: Uses MPI Rank=%d\n", RtsLayer::getPid(), RtsLayer::myNode());
      writeProfile(fp, metricHeader, tid, i, inFuncs, numFuncs);
    }
  }

  RtsLayer::UnLockDB();

  return 0;
}

int TauProfiler_dumpFunctionValues(const char **inFuncs, int numFuncs, bool increment, int tid, const char *prefix)
{
  TauInternalFunctionGuard protects_this_function;
  if(!TheSafeToDumpData()) {
    return -1;
  }

  TAU_PROFILE("TAU_DUMP_FUNC_VALS()", " ", TAU_IO);

  TAU_VERBOSE("TAU<%d,%d>: TauProfiler_dumpFunctionValues\n", RtsLayer::myNode(), RtsLayer::myThread());
  TauProfiler_writeData(tid, prefix, increment, inFuncs, numFuncs);
  return 0;
}

bool TauProfiler_createDirectories()
{
    char newdirname[1024];
    int countDirs=0;
    TAU_VERBOSE("Creating Directories\n");
#ifdef KTAU_NG
    getProfileLocation(0, newdirname);
    mkdir(newdirname, S_IRWXU | S_IRGRP | S_IXGRP);
#else
    for (int i = 0; i < Tau_Global_numCounters; i++) {
        if (TauMetrics_getMetricUsed(i)) {
            getProfileLocation(i, newdirname);
	    countDirs++;
#ifdef TAU_WINDOWS
            mkdir(newdirname);
#else
            mkdir(newdirname, S_IRWXU | S_IRGRP | S_IXGRP);
#endif
        }
    }
#endif
    if(countDirs==0)
    {
	return false;
    }
    return true;
}

/***************************************************************************
 * $RCSfile: Profiler.cpp,v $   $Author: sameer $
 * $Revision: 1.271 $   $Date: 2010/05/25 23:06:19 $
 * VERSION_ID: $Id: Profiler.cpp,v 1.271 2010/05/25 23:06:19 sameer Exp $
 ***************************************************************************/
