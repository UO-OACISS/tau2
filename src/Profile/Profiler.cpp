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

using namespace std;
using namespace tau;

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
extern "C" int Tau_profile_exit_all_tasks();
extern "C" int TauCompensateInitialized(void);

void Tau_unwind_unwindTauContext(Profiler *myProfiler);
x_uint64 Tau_get_firstTimeStamp();

//////////////////////////////////////////////////////////////////////
// For OpenMP
//////////////////////////////////////////////////////////////////////
#ifdef TAU_TRACK_IDLE_THREADS
double TheLastTimeStamp[TAU_MAX_THREADS][TAU_MAX_COUNTERS];
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
    sprintf(header, "templated_functions_MULTI_%s", tau_env);
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
      TauTraceEvent(current->ThisFunction->GetFunctionId(), 1, tid, (x_uint64)current->StartTime[0], 1);
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
  fprintf (stderr, "[%d:%d-%d] Profiler::Start for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);
#endif
//  TAU_VERBOSE("[%d:%d-%d] Profiler::Start for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);
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
  RtsLayer::getUSecD(tid, StartTime);
  TimeStamp = (x_uint64)StartTime[0];    // USE COUNTER1 for tracing

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

  /*** Profile Compensation ***/
  if (TauEnv_get_compensate()) {
    SetNumChildren(0); /* for instrumentation perturbation compensation */
  }

  // An initialization of sorts. Call Paths (if any) will update this.
#ifndef TAU_WINDOWS
  if (TauEnv_get_callsite() == 1) {
    CallSiteAddPath(NULL, tid);
  }
#endif /* TAU_WINDOWS */

  if (TauEnv_get_callpath()) {
    CallPathStart(tid);
  }

#ifndef TAU_WINDOWS
  if (TauEnv_get_callsite() == 1) {
    CallSiteStart(tid);
  }
#endif

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
    TauTraceEvent(ThisFunction->GetFunctionId(), 1 /* entry */, tid, TimeStamp, 1 /* use supplied timestamp */);
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

}

void Profiler::Stop(int tid, bool useLastTimeStamp)
{
#ifdef DEBUG_PROF
  fprintf (stderr, "[%d:%d-%d] Profiler::Stop  for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);
#endif
//  TAU_VERBOSE("[%d:%d-%d] Profiler::Stop  for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);

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
    sprintf (annotation, "TAU^seq^%d^phase^%s^nct^%d:%d:%d^timestamp^%lld^start^%lld^", sequence,
        ThisFunction->GetName(), RtsLayer::myNode(), RtsLayer::myContext(), RtsLayer::myThread(),
        getTimeStamp(), Tau_get_firstTimeStamp());
    printf ("tau-perfsuite: stopping %s\n", ThisFunction->GetName());
    setenv("PS_HWPC_ANNOTATION", annotation, 1);
    char seqstring[256];
    sprintf (seqstring, "TAU.%d", sequence);
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
      CurrentTime[i] = TheLastTimeStamp[tid][i];
    }
  } else {
    /* use the usual mechanism */
    RtsLayer::getUSecD(tid, CurrentTime);
  }
  for (i = 0; i < TAU_MAX_COUNTERS; i++) {
    TheLastTimeStamp[tid][i] = CurrentTime[i];
  }
#else
  RtsLayer::getUSecD(tid, CurrentTime);
#endif /* TAU_TRACK_IDLE_THREADS */

#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_event_stop(tid, CurrentTime);
  }
#endif

#if defined(TAUKTAU)
#ifdef KTAU_DEBUGPROF
  printf("Profiler::Stop: --EXIT-- %s \n", TauInternal_CurrentProfiler(tid)->ThisFunction->GetName());
#endif /*KTAU_DEBUGPROF*/
  ThisKtauProfiler->Stop(this, AddInclFlag);
#endif /* TAUKTAU */

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
    TauTraceEvent(ThisFunction->GetFunctionId(), -1 /* exit */, tid, TimeStamp, 1 /* use supplied timestamp */);
    TauMetrics_triggerAtomicEvents(TimeStamp, CurrentTime, tid);
  }
#ifdef TAU_MPITRACE
}
#endif /* TAU_MPITRACE */

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/

  if (TauEnv_get_callpath()) {
    CallPathStop(TotalTime, tid);
  }

#ifndef TAU_WINDOWS
  if (TauEnv_get_callsite()) {
    CallSiteStop(TotalTime, tid);
  }
#endif /* TAU_WINDOWS */

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
    double inclusiveTime;
    inclusiveTime = ThisFunction->GetInclTimeForCounter(tid, 0);
    /* here we get the array of double values representing the double 
     metrics. We choose the first counter */

    if ((ThisFunction->GetCalls(tid) > TauEnv_get_throttle_numcalls())
        && (inclusiveTime / ThisFunction->GetCalls(tid) < TauEnv_get_throttle_percall()) && AddInclFlag) {
      RtsLayer::LockDB();
      /* Putting AddInclFlag means we can't throttle recursive calls */
      ThisFunction->SetProfileGroup(TAU_DISABLE);
      ThisFunction->SetPrimaryGroupName("TAU_DISABLE");
      //const char *func_type = ThisFunction->GetType();
      string ftype(string("[THROTTLED]"));
      ThisFunction->SetType(ftype);
      //cout <<"TAU<"<<RtsLayer::myNode()<<">: Throttle: Disabling "<<ThisFunction->GetName()<<endl;
      TAU_VERBOSE("TAU<%d,%d>: Throttle: Disabling %s\n", RtsLayer::myNode(), RtsLayer::myThread(),
          ThisFunction->GetName());
      RtsLayer::UnLockDB();
    }
  }
  /********************************************************************************/
  /*** Throttling Code ***/
  /********************************************************************************/

  if (!ParentProfiler) {
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
    if (tid == 0) {
      Tau_profile_exit_all_tasks();
    }
#ifdef TAU_GPU
    //Stop all other running tasks.
    if (tid == 0) {
      //printf("exiting all tasks....\n");
      Tau_profile_exit_all_tasks();
    }
#endif
#ifndef TAU_WINDOWS
    if (tid == 0) {
      atexit(Tau_shutdown);
    }
#endif //TAU_WINDOWS
    if (TheSafeToDumpData()) {
      if (!RtsLayer::isCtorDtor(ThisFunction->GetName())) {
        // Not a destructor of a static object - its a function like main

        // Write profile data
        TauProfiler_StoreData(tid);
#ifndef TAU_WINDOWS
        // getpid() not available on Windows
        TAU_VERBOSE("TAU: <Node=%d.Thread=%d>:<pid=%d>: %s initiated TauProfiler_StoreData\n", RtsLayer::myNode(),
            RtsLayer::myThread(), getpid(), ThisFunction->GetName());
#endif
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
        if (tid == 0) {
          int i;
          for (i = 1; i < TAU_MAX_THREADS; i++) {
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

    for (int i = 0; i < numberOfFunctions; i++) {
      (*inPtr)[i] = TheFunctionDB()[i]->GetName();
    }
    *numFuncs = numberOfFunctions;
  }
}

void TauProfiler_dumpFunctionNames()
{
  TauInternalFunctionGuard protects_this_function;

  int numFuncs;
  const char ** functionList;

  TauProfiler_theFunctionList(&functionList, &numFuncs);

  const char *dirname = TauEnv_get_profiledir();

  //Create temp write to file.
  char filename[1024];
  sprintf(filename, "%s/temp.%d.%d.%d", dirname, RtsLayer::myNode(), RtsLayer::myContext(), RtsLayer::myThread());

  FILE* fp;
  if ((fp = fopen(filename, "w+")) == NULL) {
    char errormsg[1024];
    sprintf(errormsg, "Error: Could not create %s", filename);
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
  sprintf(dumpfile, "%s/dump_functionnames_n,c,t.%d.%d.%d", dirname, RtsLayer::myNode(), RtsLayer::myContext(),
      RtsLayer::myThread());
  rename(filename, dumpfile);
}

void TauProfiler_getUserEventList(const char ***inPtr, int *numUserEvents)
{
  TauInternalFunctionGuard protects_this_function;

  *numUserEvents = 0;

  vector<TauUserEvent*>::iterator eit;

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
  vector<TauUserEvent*>::iterator eit;

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

  TAU_PROFILE("TAU_GET_FUNC_VALS()", " ", TAU_IO);

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
  vector<TauUserEvent*>::iterator eit;
  Profiler *curr;

  DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
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

// writes user events to the file
static int writeUserEvents(FILE *fp, int tid)
{
  vector<TauUserEvent*>::iterator it;

  fprintf(fp, "0 aggregates\n");    // For now there are no aggregates

  // Print UserEvent Data if any
  int numEvents = 0;
  for (it = TheEventDB().begin(); it != TheEventDB().end(); ++it) {
    if ((*it) && (*it)->GetNumEvents(tid) == 0) {    // skip user events with no calls
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
      if ((*it) && (*it)->GetNumEvents(tid) == 0) continue;
      if ((*it) && (*it)->GetWriteAsMetric()) continue;
      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", (*it)->GetName().c_str(), (*it)->GetNumEvents(tid),
          (*it)->GetMax(tid), (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
    }
  }
  return 0;
}

static int writeHeader(FILE *fp, int numFunc, char *metricName)
{
  char header[256];
  sprintf(header, "%d %s\n", numFunc, metricName);
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

  // get current values
  double currentTime[TAU_MAX_COUNTERS];

  RtsLayer::getCurrentValues(tid, currentTime);
  int c;

  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;

    double *incltime = fi->getDumpInclusiveValues(tid);
    double *excltime = fi->getDumpExclusiveValues(tid);

    // get currently stored values
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
      double *InclTime = fi->GetInclTime(tid);
      double *ExclTime = fi->GetExclTime(tid);

      double inclusiveToAdd[TAU_MAX_COUNTERS];
      double prevStartTime[TAU_MAX_COUNTERS];

      for (c = 0; c < Tau_Global_numCounters; c++) {
        inclusiveToAdd[c] = 0;
        prevStartTime[c] = 0;
      }

      for (Profiler *current = TauInternal_CurrentProfiler(tid); current != 0; current = current->ParentProfiler) {
        if (helperIsFunction(fi, current)) {
          for (c = 0; c < Tau_Global_numCounters; c++) {
            inclusiveToAdd[c] = currentTime[c] - current->getStartValues()[c];
            excltime[c] += inclusiveToAdd[c] - prevStartTime[c];
            // *CWL* - followup to the data structure insanity issues
            ExclTime[c] += inclusiveToAdd[c] - prevStartTime[c];
            /*
             TAU_VERBOSE("[%d] currentTime=%f startValue=%f prevStartTime=%f excltime=%f ExclTime=%f!\n",
             tid, currentTime[c], current->getStartValues()[c],
             prevStartTime[c], excltime[c], ExclTime[c]);
             */
          }
        }
        for (c = 0; c < Tau_Global_numCounters; c++) {
          prevStartTime[c] = currentTime[c] - current->getStartValues()[c];
        }
      }
      for (c = 0; c < Tau_Global_numCounters; c++) {
        incltime[c] += inclusiveToAdd[c];
        // *CWL* - followup to the data structure insanity issues
        InclTime[c] += inclusiveToAdd[c];
      }

      // *CWL* - followup to the data structure insanity issues
      fi->SetInclTime(tid, InclTime);
      fi->SetExclTime(tid, ExclTime);
      free(InclTime);
      free(ExclTime);
    }
  }
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
  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;

    if (-1 == matchFunction(*it, inFuncs, numFuncs)) {    // skip this function
      continue;
    }

    if (fi->GetCalls(tid) == 0) {    // skip this function
      continue;
    }

    bool found_one = false;

    if (TauMetrics_getMetricAtomic(metric) != NULL)
    {
      vector<TauUserEvent*>::iterator it2;

      // Print UserEvent Data if any
      for (it2 = TheEventDB().begin(); it2 != TheEventDB().end() && !found_one; ++it2) {
        TauUserEvent *ue = *it2;
        //printf("testing %s vs %s.\n", fi->GetName(), ue->GetName().c_str());

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

          double excltime = ue->GetMean(tid);
          //double excltime = ue->GetMean(tid) * ue->GetNumEvents(tid);
          double incltime = excltime;
          int calls = fi->GetCalls(tid);
       
          //std::string name = ue->GetName();

          //size_t del = name.find(std::string(":"));

          fprintf(fp, "\"%s\" %ld %ld %.16G %.16G ", suffix, (long int)calls, 0L, excltime,
              incltime);
          fprintf(fp, "0 ");    // Indicating that profile calls is turned off
          fprintf(fp, "GROUP=\"%s\" \n", fi->GetAllGroups());
          
          found_one = true;
        }
      }
       /* 
      if (!found_one) {
        fprintf(fp, "\"%s\" %ld %ld %.16G %.16G ", fi->GetName(), fi->GetCalls(tid), 0, 0.0, 0.0);
        fprintf(fp, "0 ");    // Indicating that profile calls is turned off
        fprintf(fp, "GROUP=\"%s\" \n", fi->GetAllGroups());
      }*/
      //found_one = false;
/*
      if (numEvents > 0) {
        // Data format 
        // # % userevents
        // # name numsamples max min mean sumsqr 
        fprintf(fp, "%d userevents\n", numEvents);
        fprintf(fp, "# eventname numevents max min mean sumsqr\n");

        for (it = TheEventDB().begin(); it != TheEventDB().end(); ++it) {
          if ((*it) && (*it)->GetNumEvents(tid) == 0) continue;
          fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", (*it)->GetName().c_str(), (*it)->GetNumEvents(tid),
              (*it)->GetMax(tid), (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
        }
      }
*/
    }

    if (!found_one)
    {

      // get currently stored values
      double incltime = fi->getDumpInclusiveValues(tid)[metric];
      double excltime = fi->getDumpExclusiveValues(tid)[metric];

      if (strlen(fi->GetType()) > 0) {
        fprintf(fp, "\"%s %s\" %ld %ld %.16G %.16G ", fi->GetName(), fi->GetType(), fi->GetCalls(tid), fi->GetSubrs(tid),
            excltime, incltime);
      } else {
        fprintf(fp, "\"%s\" %ld %ld %.16G %.16G ", fi->GetName(), fi->GetCalls(tid), fi->GetSubrs(tid), excltime,
            incltime);
      }

      fprintf(fp, "0 ");    // Indicating that profile calls is turned off
      fprintf(fp, "GROUP=\"%s\" \n", fi->GetAllGroups());
      
    }
  }

  return 0;
}

// Writes function event data
static int getTrueFunctionCount(int count, int tid, const char **inFuncs, int numFuncs, int metric)
{
  int trueCount = count;

  vector<TauUserEvent*>::iterator it2;
  const char *metricName = TauMetrics_getMetricAtomic(metric);

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

static int profileWriteCount[TAU_MAX_THREADS];
static int profileWriteWarningPrinted = 0;

extern "C" int Tau_profiler_initialization()
{
  int i;
  for (i = 1; i < TAU_MAX_THREADS; i++) {
    profileWriteCount[i] = 0;
  }
  profileWriteWarningPrinted = 0;
  return 0;
}

// Store profile data at the end of execution (when top level timer stops)
extern "C" void finalizeCallSites_if_necessary();
int TauProfiler_StoreData(int tid)
{
  TAU_VERBOSE("TAU<%d,%d>: TauProfiler_StoreData\n", RtsLayer::myNode(), tid);

#ifdef TAU_SCOREP
  Tau_write_metadata_records_in_scorep(tid);
#endif /* TAU_SCOREP */
  profileWriteCount[tid]++;
  if ((tid != 0) && (profileWriteCount[tid] > 1)) return 0;

  if (profileWriteCount[tid] == 10) {
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

#ifndef TAU_WINDOWS
  if (TauEnv_get_callsite()) {
    finalizeCallSites_if_necessary();
  }

  if (TauEnv_get_ebs_enabled()) {
    // Tau_sampling_finalize(tid);
    Tau_sampling_finalize_if_necessary();
  }
#endif
  if (TauEnv_get_profiling()) {
    Tau_snapshot_writeFinal("final");
    if (TauEnv_get_profile_format() == TAU_FORMAT_PROFILE) {
      TauProfiler_DumpData(false, tid, "profile");
	}
  }
#if defined(PTHREADS) || defined(TAU_OPENMP)
  if (RtsLayer::myThread() == 0 && tid == 0) {
    /* clean up other threads? */
    for (int i = 1; i < TAU_MAX_THREADS; i++) {
      if (TauInternal_ParentProfiler(i) != (Profiler *)NULL) {
        TauProfiler_StoreData(i);
      }
    }
#ifndef TAU_MPI
	/* Only thread 0 should create a merged profile. */
    if (TauEnv_get_profile_format() == TAU_FORMAT_MERGED) {
      Tau_metadataMerge_mergeMetaData();
      /* Create a merged profile if requested */
      Tau_mergeProfiles();
	}
#endif
  }
#endif /* PTHREADS */

// this doesn't work... apparently "getTotalThreads() lies to us.
// Is there a reliable way to get the number of threads seen by
// OpenMP???
#if 0
#ifndef TAU_SCOREP
#if defined(TAU_OPENMP)
  fprintf(stderr, "Total Threads: %d\n", RtsLayer::getTotalThreads());
  if (RtsLayer::getTotalThreads() == 1) {
    // issue a warning, because this is a multithreaded config,
    // and we saw no threads other than 0!
    fprintf(stderr,
        "\nTAU: WARNING! TAU did not detect more than one thread.\n"
        "If running an OpenMP application with tau_exec and you expected\n"
        "more than one thread, try using the '-T pthread' configuration,\n"
        "or instrument your code with TAU.\n\n");
  }
#endif /* OPENMP */
#endif /* SCOREP */
#endif
  return 1;
}

// Returns directory name for the location of a particular metric
static int getProfileLocation(int metric, char *str)
{
  const char *profiledir;
  profiledir = TauEnv_get_profiledir();
#if defined(KTAU_NG)
  if(profiledir == NULL) {
    int written_bytes = 0;
    unsigned int profile_dir_len = KTAU_NG_PREFIX_LEN + HOSTNAME_LEN;
    profiledir = new char[profile_dir_len];
    written_bytes = sprintf(profiledir, "%s.", KTAU_NG_PREFIX);
    gethostname(profiledir + written_bytes, profile_dir_len - written_bytes);
  }
#else
  profiledir = TauEnv_get_profiledir();
#endif

  if (Tau_Global_numCounters <= 1) {
    sprintf(str, "%s", profiledir);
  } else {
    string metricStr = string(TauMetrics_getMetricName(metric));

    //sanitize metricName before creating a directory name from it.
    string illegalChars("/\\?%*:|\"<> ");
    size_t found;
    found = metricStr.find_first_of(illegalChars, 0);
    while (found != string::npos) {
      metricStr[found] = '_';
      found = metricStr.find_first_of(illegalChars, found + 1);
    }
    sprintf(str, "%s/MULTI__%s", profiledir, metricStr.c_str());
  }

  return 0;
}

int TauProfiler_DumpData(bool increment, int tid, const char *prefix)
{
  TAU_VERBOSE("TAU<%d,%d>: TauProfiler_DumpData\n", RtsLayer::myNode(), tid);
  return TauProfiler_writeData(tid, prefix, increment);
}

void getMetricHeader(int i, char *header)
{
  sprintf(header, "templated_functions_MULTI_%s", RtsLayer::getCounterName(i));
}

// Stores profile data
int TauProfiler_writeData(int tid, const char *prefix, bool increment, const char **inFuncs, int numFuncs)
{

  TauProfiler_updateIntermediateStatistics(tid);

  RtsLayer::LockDB();

  static bool createFlag = TauProfiler_createDirectories();
  if (createFlag) {
    TAU_VERBOSE ("Profile directories created\n");
  }

  for (int i = 0; i < Tau_Global_numCounters; i++) {
    if (TauMetrics_getMetricUsed(i)) {

      char metricHeader[1024];
      char profileLocation[1024];
      FILE* fp;

      getMetricHeader(i, metricHeader);
      getProfileLocation(i, profileLocation);
//       sprintf(filename, "%s/temp.%d.%d.%d", profileLocation, 
//         RtsLayer::myNode(), RtsLayer::myContext(), tid);

      const char *selectivePrefix = "";
      if (numFuncs > 0) {
        selectivePrefix = "sel_";
      }

      char dumpfile[1024];
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
        char newStringTime[1024];
        sprintf(newStringTime, "%s-%s-%s-%s-%s", day, month, dayInt, time, year);

        sprintf(dumpfile, "%s/%s%s__%s__.%d.%d.%d", profileLocation, selectivePrefix, prefix, newStringTime,
            RtsLayer::myNode(), RtsLayer::myContext(), tid);

        if ((fp = fopen(dumpfile, "w+")) == NULL) {
          char errormsg[1024];
          sprintf(errormsg, "Error: Could not create %s", dumpfile);
          perror(errormsg);
          return 0;
        }

        char cwd[1024];
        char *tst = getcwd(cwd, 1024);
		if (tst == NULL) {
          char errormsg[1024];
          sprintf(errormsg, "Error: Could not get current working directory");
          perror(errormsg);
          return 0;
		}
#ifndef TAU_WINDOWS
        TAU_VERBOSE("[pid=%d], TAU: Writing profile %s, cwd = %s\n", getpid(), dumpfile, cwd);
#endif

      } else {
        int flags = O_CREAT | O_EXCL | O_WRONLY;
#ifdef TAU_DISABLE_SIGUSR
        int mode = 0;
#else
        int mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
#endif
        int node = RtsLayer::myNode();

        sprintf(dumpfile, "%s/%s%s.%d.%d.%d", profileLocation, selectivePrefix, prefix, node, RtsLayer::myContext(),
            tid);

        int sicortex = 0;
#ifdef TAU_SICORTEX
        sicortex = 1;
#endif

        if (sicortex && !Tau_get_usesMPI()) {
          int test = open(dumpfile, flags, mode);
          while (test == -1 && node < 99999) {
            node++;
            sprintf(dumpfile, "%s/%s%s.%d.%d.%d", profileLocation, selectivePrefix, prefix, node, RtsLayer::myContext(),
                tid);
            test = open(dumpfile, flags, mode);
          }
          if ((fp = fdopen(test, "w")) == NULL) {
            char errormsg[1024];
            sprintf(errormsg, "Error: Could not create %s", dumpfile);
            perror(errormsg);
            return 0;
          }

        } else {
          if ((fp = fopen(dumpfile, "w+")) == NULL) {
            char errormsg[1024];
            sprintf(errormsg, "Error: Could not create %s", dumpfile);
            perror(errormsg);
            return 0;
          }
          char cwd[1024];
          char *tst = getcwd(cwd, 1024);
		  if (tst == NULL) {
            char errormsg[1024];
            sprintf(errormsg, "Error: Could not get current working directory");
            perror(errormsg);
            return 0;
		  }
#ifndef TAU_WINDOWS
          TAU_VERBOSE("[pid=%d], TAU: Writing profile %s, cwd = %s\n", getpid(), dumpfile, cwd);
#endif
        }
      }
      writeProfile(fp, metricHeader, tid, i, inFuncs, numFuncs);
    }
  }

  RtsLayer::UnLockDB();

  return 0;
}

int TauProfiler_dumpFunctionValues(const char **inFuncs, int numFuncs, bool increment, int tid, const char *prefix)
{
  TauInternalFunctionGuard protects_this_function;

  TAU_PROFILE("TAU_DUMP_FUNC_VALS()", " ", TAU_IO);

  TAU_VERBOSE("TAU<%d,%d>: TauProfiler_dumpFunctionValues\n", RtsLayer::myNode(), RtsLayer::myThread());
  TauProfiler_writeData(tid, prefix, increment, inFuncs, numFuncs);
  return 0;
}

bool TauProfiler_createDirectories()
{

  static bool flag = true;
  if (flag && Tau_Global_numCounters > 1) {
    for (int i = 0; i < Tau_Global_numCounters; i++) {
      if (TauMetrics_getMetricUsed(i)) {
        char *newdirname = new char[1024];
        char *mkdircommand = new char[1024];
        getProfileLocation(i, newdirname);
        sprintf(mkdircommand, "mkdir -p %s", newdirname);

        //system(rmdircommand);
        //system(mkdircommand);
        /* On IBM BGL, system command doesn't execute. So, we need to create
         these directories using our mkdir syscall instead. */
        /* OLD: mkdir(newdirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); */
#ifdef TAU_WINDOWS
        mkdir(newdirname);
#else
        mkdir(newdirname, S_IRWXU | S_IRGRP | S_IXGRP);
#endif
      }
    }
    flag = false;
  } else {
#ifdef KTAU_NG
    char *newdirname = new char[1024];
    getProfileLocation(Tau_Global_numCounters, newdirname);
    mkdir(newdirname, S_IRWXU | S_IRGRP | S_IXGRP);
#endif
    flag = false;
  }
  return true;
}

/***************************************************************************
 * $RCSfile: Profiler.cpp,v $   $Author: sameer $
 * $Revision: 1.271 $   $Date: 2010/05/25 23:06:19 $
 * VERSION_ID: $Id: Profiler.cpp,v 1.271 2010/05/25 23:06:19 sameer Exp $ 
 ***************************************************************************/
