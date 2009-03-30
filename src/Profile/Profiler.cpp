/*****************************************************************************
 **			TAU Portable Profiling Package			    **
 **			http://www.cs.uoregon.edu/research/tau	            **
 *****************************************************************************
 **    Copyright 1997-2009					   	    **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/*****************************************************************************
 **	File 		: Profiler.cpp					    **
 **	Description 	: TAU Profiling Package				    **
 **	Author		: Sameer Shende					    **
 **	Contact		: tau-bugs@cs.uoregon.edu                 	    **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau        **
 ****************************************************************************/

//#define DEBUG_PROF
#include <Profile/Profiler.h>
#include <Profile/TauMetrics.h>

//#include <tau_internal.h>

#ifdef TAU_PERFSUITE
  #include <pshwpc.h>
  extern "C" int ps_hwpc_xml_write(const char *filename);
  extern "C" int ps_hwpc_reset();
#endif

#ifdef TAU_WINDOWS
double TauWindowsUsecD(void);
#endif

//#ifndef TAU_WINDOWS
extern "C" void Tau_shutdown(void);
//#endif //TAU_WINDOWS

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#include <stack>
using namespace std;
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

// #ifdef TAU_WINDOWS
// #ifndef TAU_DISABLE_METADATA
// #include <sys/utsname.h> // for host identification (uname)
// #endif
// #endif //TAU_WINDOWS

#ifdef TAU_VAMPIRTRACE
#include "Profile/TauVampirTrace.h"
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_EPILOG
#include "elg_trc.h"

#ifdef TAU_SCALASCA
extern "C" {
void esd_enter (elg_ui4 rid);
void esd_exit (elg_ui4 rid);
}
#endif /* SCALASCA */
#endif /* TAU_EPILOG */

#include <Profile/TauTrace.h>


#ifdef RENCI_STFF
#include "Profile/RenciSTFF.h"
#endif // RENCI_STFF


int Tau_writeProfileMetaData(FILE *fp, int counter);

static int writeUserEvents(FILE *fp, int tid);
static int matchFunction(FunctionInfo *fi, const char **inFuncs, int numFuncs);
extern "C" int Tau_get_usesMPI();

//////////////////////////////////////////////////////////////////////
//Initialize static data
//////////////////////////////////////////////////////////////////////

// No need to initialize FunctionDB. using TheFunctionDB() instead.
// vector<FunctionInfo*> FunctionInfo::FunctionDB[TAU_MAX_THREADS] ;
//Profiler * Profiler::CurrentProfiler[] = {0}; // null to start with

#if defined(TAUKTAU)
#include <Profile/KtauProfiler.h>
#endif /* TAUKTAU */

// The rest of CurrentProfiler entries are initialized to null automatically
//TauGroup_t RtsLayer::ProfileMask = TAU_DEFAULT;

// Default value of Node.
//int RtsLayer::Node = -1;

#ifdef TAU_OPENMP 
#define TAU_TRACK_IDLE_THREADS
#endif

#ifdef TAU_TRACK_PTHREAD_IDLE
#define TAU_TRACK_IDLE_THREADS
#endif

//////////////////////////////////////////////////////////////////////
// For OpenMP
//////////////////////////////////////////////////////////////////////
#ifdef TAU_TRACK_IDLE_THREADS
#ifndef TAU_MULTIPLE_COUNTERS
double TheLastTimeStamp[TAU_MAX_THREADS]; 
#else /* FOR MULTIPLE COUNTERS */
double TheLastTimeStamp[TAU_MAX_THREADS][MAX_TAU_COUNTERS]; 
#endif /* MULTIPLE_COUNTERS */
#endif /* TAU_TRACK_IDLE_THREADS */
//////////////////////////////////////////////////////////////////////
// Explicit Instantiations for templated entities needed for ASCI Red
//////////////////////////////////////////////////////////////////////

#ifdef PGI
template
void vector<FunctionInfo *>::insert_aux(vector<FunctionInfo *>::pointer, FunctionInfo *const &);
// need a few other function templates instantiated
template
FunctionInfo** copy_backward(FunctionInfo**,FunctionInfo**,FunctionInfo**);
template
FunctionInfo** uninitialized_copy(FunctionInfo**,FunctionInfo**,FunctionInfo**);
//template <>
//std::basic_ostream<char, std::char_traits<char> > & std::operator<< (std::basic_ostream<char, std::char_traits<char> > &, const char * );
#endif /* PGI */


//////////////////////////////////////////////////////////////////////
// Shutdown routine which calls TAU's shutdown
//////////////////////////////////////////////////////////////////////
void TauAppShutdown(void) {
  Tau_shutdown();
}
 
//////////////////////////////////////////////////////////////////////
// Get the string containing the counter name
//////////////////////////////////////////////////////////////////////
const char *TauGetCounterString(void) {
#ifdef SGI_HW_COUNTERS
  return "templated_functions_hw_counters";
#elif (defined (TAU_PAPI) \
       || defined(TAU_PAPI_WALLCLOCKTIME)	\
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
//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class Profiler
//////////////////////////////////////////////////////////////////////

#ifdef TAU_MPITRACE
//////////////////////////////////////////////////////////////////////
void TauProfiler_EnableAllEventsOnCallStack(int tid, Profiler *current) {
  /* Go up the callstack and enable all events on it */
  if (current != (Profiler *) NULL) {
    DEBUGPROFMSG(RtsLayer::myNode()<<" This func = "<<current->ThisFunction->GetName()<<" RecordEvent = "<<current->RecordEvent<<endl;);
    if (!current->RecordEvent) { 
      DEBUGPROFMSG(RtsLayer::myNode()<< " Enabling event "<<current->ThisFunction->GetName()<<endl;);
      current->RecordEvent = true;
      TauProfiler_EnableAllEventsOnCallStack(tid, current->ParentProfiler);
      /* process the current event */
      DEBUGPROFMSG(RtsLayer::myNode()<<" Processing EVENT "<<current->ThisFunction->GetName()<<endl;);
      TauTraceEvent(current->ThisFunction->GetFunctionId(), 1, tid, (x_uint64) current->StartTime, 1); 
#ifdef TAU_MULTIPLE_COUNTERS 
      MultipleCounterLayer::triggerCounterEvents((x_uint64) current->StartTime[0], current->StartTime, tid);
#endif /* TAU_MULTIPLE_COUNTERS */
    }
  }
}

#endif /* TAU_MPITRACE */
//////////////////////////////////////////////////////////////////////


void Profiler::Start(int tid) { 
#ifdef DEBUG_PROF
  fprintf (stderr, "[%d:%d-%d] Profiler::Start for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);
#endif

#ifdef TAU_TRACK_IDLE_THREADS
  /* If we are performing idle thread tracking, we start a top level timer */
  if (tid != 0) {
    Tau_create_top_level_timer_if_necessary();
  }
#endif

  ParentProfiler = TauInternal_ParentProfiler(tid);

  

  /********************************************************************************/
  /*** Phase Profiling ***/
  /********************************************************************************/
#ifdef TAU_PROFILEPHASE
  if (ParentProfiler == (Profiler *) NULL) {
    string AllGroups = ThisFunction->AllGroups;
    if (AllGroups.find("TAU_PHASE", 0) == string::npos) {
      AllGroups.append(" | TAU_PHASE");
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

  
#ifdef TAU_PROFILEMEMORY
  ThisFunction->GetMemoryEvent()->TriggerEvent(TauGetMaxRSS());
#endif /* TAU_PROFILEMEMORY */

#ifdef TAU_PROFILEHEADROOM
  ThisFunction->GetHeadroomEvent()->TriggerEvent((double)TauGetFreeMemory());
#endif /* TAU_PROFILEHEADROOM */

#ifdef TAU_COMPENSATE
  SetNumChildren(0); /* for instrumentation perturbation compensation */
#endif /* TAU_COMPENSATE */


  x_uint64 TimeStamp;

#ifndef TAU_MULTIPLE_COUNTERS 
  StartTime = RtsLayer::getUSecD(tid);
  TimeStamp = (x_uint64) StartTime;
#else //TAU_MULTIPLE_COUNTERS
  RtsLayer::getUSecD(tid, StartTime);	  
  TimeStamp = (x_uint64) StartTime[0]; // USE COUNTER1 for tracing
#endif//TAU_MULTIPLE_COUNTERS
  
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
  if (MyProfileGroup_ & TAU_MESSAGE) {
    /* if we're in the group, we must first enable all the other events
     * on the callstack */
    DEBUGPROFMSG(RtsLayer::myNode()<< " Function is enabled: "<<ThisFunction->GetName()<<endl;);
    TauProfiler_EnableAllEventsOnCallStack(tid, this);
  }
#else /* TAU_MPITRACE */
#ifdef TAU_VAMPIRTRACE 
  TimeStamp = vt_pform_wtime();
  vt_enter((uint64_t *) &TimeStamp, ThisFunction->GetFunctionId());
#else /* TAU_VAMPITRACE */
#ifdef TAU_EPILOG
  esd_enter(ThisFunction->GetFunctionId());
#else /* TAU_EPILOG */
  if (TauEnv_get_tracing()) {
    TauTraceEvent(ThisFunction->GetFunctionId(), 1 /* entry */, tid, TimeStamp, 1 /* use supplied timestamp */); 
#ifdef TAU_MULTIPLE_COUNTERS 
    MultipleCounterLayer::triggerCounterEvents(TimeStamp, StartTime, tid);
#endif /* TAU_MULTIPLE_COUNTERS */
  }
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif /* TAU_MPITRACE */
  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/


  // Inncrement the number of calls
  ThisFunction->IncrNumCalls(tid);
  
  // Increment the parent's NumSubrs()
  if (ParentProfiler != 0) {
    ParentProfiler->ThisFunction->IncrNumSubrs(tid);	
  }
  
  // If this function is not already on the call stack, put it
  if (ThisFunction->GetAlreadyOnStack(tid) == false) { 
    AddInclFlag = true; 
    // We need to add Inclusive time when it gets over as 
    // it is not already on callstack.
    
    ThisFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
  } else { 
    // the function is already on callstack, no need to add inclusive time
    AddInclFlag = false;
  }
    
  /********************************************************************************/
  /*** KTAU Code ***/
  /********************************************************************************/
#if defined(TAUKTAU)
  ThisKtauProfiler->Start(this);
#endif /* TAUKTAU */

}

//////////////////////////////////////////////////////////////////////

x_uint64 Tau_get_firstTimeStamp();
static x_uint64 getTimeStamp() {
  x_uint64 timestamp;
#ifdef TAU_WINDOWS
  timestamp = TauWindowsUsecD();
#else
  struct timeval tp;
  gettimeofday (&tp, 0);
  timestamp = (x_uint64)tp.tv_sec * (x_uint64)1e6 + (x_uint64)tp.tv_usec;
#endif
  return timestamp;
}


void Profiler::Stop(int tid, bool useLastTimeStamp) {
#ifdef DEBUG_PROF
  fprintf (stderr, "[%d:%d-%d] Profiler::Stop  for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);
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


#ifdef TAU_COMPENSATE
#ifndef TAU_MULTIPLE_COUNTERS 
  double tover = TauGetTimerOverhead(TauFullTimerOverhead);
  double tnull = TauGetTimerOverhead(TauNullTimerOverhead);
#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* TAU_COMPENSATE */
  
#ifndef TAU_MULTIPLE_COUNTERS
  double CurrentTime; 
  
#ifdef TAU_TRACK_IDLE_THREADS
  if (useLastTimeStamp) {
    /* for openmp parallel regions */
    /* .TAU Application needs to be stopped */
    CurrentTime = TheLastTimeStamp[tid]; 
  } else { /* use the usual mechanism */
    CurrentTime = RtsLayer::getUSecD(tid);
  }
  TheLastTimeStamp[tid] = CurrentTime;
#else /* TAU_TRACK_IDLE_THREADS */
  CurrentTime = RtsLayer::getUSecD(tid);
#endif /* TAU_TRACK_IDLE_THREADS */
  
  
#if defined(TAUKTAU)
  ThisKtauProfiler->Stop(this, AddInclFlag);
#endif /* TAUKTAU */
  
  double TotalTime = CurrentTime - StartTime;
  
  
#if (defined(TAU_COMPENSATE) && defined(PROFILING_ON))
  /* To compensate for timing overhead, shrink the totaltime! */
  TotalTime = TotalTime - tnull - GetNumChildren() * tover; 
  if (TotalTime < 0 ) {
    TotalTime = 0;
    DEBUGPROFMSG("TotalTime negative in "<<ThisFunction->GetName()<<endl;);
  }
#endif /* TAU_COMPENSATE && PROFILING_ON */

#else //TAU_MULTIPLE_COUNTERS
  // first initialize the CurrentTime
  double CurrentTime[MAX_TAU_COUNTERS];
  double TotalTime[MAX_TAU_COUNTERS];

#ifdef TAU_TRACK_IDLE_THREADS
  int i;
  if (useLastTimeStamp) {
    /* for openmp parallel regions */
    /* .TAU Application needs to be stopped */
    for (i=0; i < MAX_TAU_COUNTERS; i++) {
      CurrentTime[i] = TheLastTimeStamp[tid][i]; 
    }
  } else { 
    /* use the usual mechanism */
    RtsLayer::getUSecD(tid, CurrentTime);
  }
  for (i=0; i < MAX_TAU_COUNTERS; i++) {
    TheLastTimeStamp[tid][i] = CurrentTime[i]; 
  }
#else
  RtsLayer::getUSecD(tid, CurrentTime);
#endif /* TAU_TRACK_IDLE_THREADS */

#if defined(TAUKTAU)
#ifdef KTAU_DEBUGPROF
  printf("Profiler::Stop: --EXIT-- %s \n", TauInternal_CurrentProfiler(tid)->ThisFunction->GetName());
#endif /*KTAU_DEBUGPROF*/
  ThisKtauProfiler->Stop(this, AddInclFlag);
#endif /* TAUKTAU */
    

#ifdef PROFILING_ON
#ifdef TAU_COMPENSATE
  double *tover = TauGetTimerOverhead(TauFullTimerOverhead);
  double *tnull = TauGetTimerOverhead(TauNullTimerOverhead);
#endif /* TAU_COMPENSATE */


  for (int k=0; k<Tau_Global_numCounters; k++) {
    TotalTime[k] = CurrentTime[k] - StartTime[k];
#ifdef TAU_COMPENSATE 
    /* To compensate for timing overhead, shrink the totaltime! */
    TotalTime[k] = TotalTime[k] - tnull[k] - GetNumChildren() * tover[k]; 
    if (TotalTime[k] < 0 ) {
      TotalTime[k] = 0;
      DEBUGPROFMSG("TotalTime[" <<k<<"] negative in "<<ThisFunction->GetName()<<endl;);
    }
#endif /* TAU_COMPENSATE */
  }
#endif // PROFILING_ON
    
#endif//TAU_MULTIPLE_COUNTERS

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/
  x_uint64 TimeStamp = 0L; 

#ifdef TAU_MULTIPLE_COUNTERS 
  TimeStamp = (x_uint64) CurrentTime[0]; // USE COUNTER1
#else
  TimeStamp = (x_uint64) CurrentTime; 
#endif /* TAU_MULTIPLE_COUNTERS */

#ifdef TAU_VAMPIRTRACE
  TimeStamp = vt_pform_wtime();
  DEBUGPROFMSG("Calling vt_exit(): "<< ThisFunction->GetName() << "With Timestamp = " << TimeStamp<<endl;);
  vt_exit((uint64_t *)&TimeStamp);
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
  DEBUGPROFMSG("Calling elg_exit(): "<< ThisFunction->GetName()<<endl;);
  esd_exit(ThisFunction->GetFunctionId());
#else /* TAU_EPILOG */
#ifdef TAU_MPITRACE
  if (RecordEvent) {
#endif /* TAU_MPITRACE */
    if (TauEnv_get_tracing()) {
      TauTraceEvent(ThisFunction->GetFunctionId(), -1 /* exit */, tid, TimeStamp, 1 /* use supplied timestamp */); 
#ifdef TAU_MULTIPLE_COUNTERS 
      MultipleCounterLayer::triggerCounterEvents(TimeStamp, CurrentTime, tid);
#endif /* TAU_MULTIPLE_COUNTERS */
    }
#ifdef TAU_MPITRACE
  }
#endif /* TAU_MPITRACE */
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */


  /* What should we do while exiting when profiling is off, tracing is on and 
     throttling is on? */
#ifndef PROFILING_ON
  if (TauEnv_get_throttle() && AddInclFlag) {
    ThisFunction->SetAlreadyOnStack(false, tid); // while exiting
      
    // Next, compute inclusive time for counter 0
#ifdef TAU_MULTIPLE_COUNTERS
    double TimeTaken = CurrentTime[0] - StartTime[0];
    ThisFunction->AddInclTimeForCounter(TimeTaken, tid, 0);
#else /* single counter */
    double TimeTaken = CurrentTime - StartTime;
    ThisFunction->AddInclTime(TimeTaken, tid);
#endif /* TAU_MULTIPLE_COUNTERS */ /* we only maintain inclusive time for counter 0 */
  }
#endif /* PROFILING is off */

  /********************************************************************************/
  /*** Tracing ***/
  /********************************************************************************/
    
    
#ifdef PROFILING_ON  // Calculations relevent to profiling only 
    
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

  if (AddInclFlag == true) { // The first time it came on call stack
    ThisFunction->SetAlreadyOnStack(false, tid); // while exiting
      
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
    
#ifdef TAU_COMPENSATE
  ThisFunction->ResetExclTimeIfNegative(tid); 

  if (TauEnv_get_callpath()) {
    if (ParentProfiler != NULL) {
      CallPathFunction->ResetExclTimeIfNegative(tid); 
    }
  }
#ifdef TAU_PROFILEPARAM
  if (ParentProfiler != NULL) {
    ProfileParamFunction->ResetExclTimeIfNegative(tid);
  }
#endif /* TAU_PROFILEPARAM */
#endif /* TAU_COMPENSATE */

    
  
  if (ParentProfiler != NULL) {
      ParentProfiler->ThisFunction->ExcludeTime(TotalTime, tid);
    
#ifdef TAU_COMPENSATE
      ParentProfiler->AddNumChildren(GetNumChildren()+1);
      /* Add 1 and my children to my parents total number of children */
#endif /* TAU_COMPENSATE */
  }

#endif //PROFILING_ON

  /********************************************************************************/
  /*** Throttling Code ***/
  /********************************************************************************/
  if (TauEnv_get_throttle()) {
    /* if the frequency of events is high, disable them */
    double inclusiveTime; 
#ifdef TAU_MULTIPLE_COUNTERS
    inclusiveTime = ThisFunction->GetInclTimeForCounter(tid, 0); 
    /* here we get the array of double values representing the double 
       metrics. We choose the first counter */
#else  /* TAU_MULTIPLE_COUNTERS */
    inclusiveTime = ThisFunction->GetInclTime(tid); 
    /* when multiple counters are not used, it is a single metric or double */
#endif /* MULTIPLE_COUNTERS */
    
    if ((ThisFunction->GetCalls(tid) > TauEnv_get_throttle_numcalls()) 
	&& (inclusiveTime/ThisFunction->GetCalls(tid) < TauEnv_get_throttle_percall()) 
	&& AddInclFlag) { 
      RtsLayer::LockDB();
      /* Putting AddInclFlag means we can't throttle recursive calls */
      ThisFunction->SetProfileGroup(TAU_DISABLE, tid);
      ThisFunction->SetPrimaryGroupName("TAU_DISABLE");
      //cout <<"TAU<"<<RtsLayer::myNode()<<">: Throttle: Disabling "<<ThisFunction->GetName()<<endl;
      TAU_VERBOSE("TAU<%d>: Throttle: Disabling %s\n", RtsLayer::myNode(), ThisFunction->GetName());
      RtsLayer::UnLockDB();
    }
  }
  /********************************************************************************/
  /*** Throttling Code ***/
  /********************************************************************************/
    
  if (ParentProfiler == (Profiler *) NULL) {
    /* Should we detect memory leaks here? */
    if (TheSafeToDumpData() && !RtsLayer::isCtorDtor(ThisFunction->GetName())) {
      TauDetectMemoryLeaks(); /* the last event should be before final exit */
    }

    // For Dyninst. tcf gets called after main and all the data structures may not be accessible
    // after main exits. Still needed on Linux - we use TauProgramTermination()
    if (strcmp(ThisFunction->GetName(), "_fini") == 0) {
      TheSafeToDumpData() = 0;
    }

#ifndef TAU_WINDOWS
    atexit(TauAppShutdown);
#endif //TAU_WINDOWS

    if (TheSafeToDumpData()) {
      if (!RtsLayer::isCtorDtor(ThisFunction->GetName())) {
	// Not a destructor of a static object - its a function like main

	// Write profile data
	TauProfiler_StoreData(tid);
	  
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
	    if (cp && strncmp(cp->ThisFunction->GetName(),".TAU", 4) == 0) {
	      bool uselasttimestamp = true;
	      cp->Stop(i,uselasttimestamp); /* force it to write the data*/
	    }
	  }
	}
#endif /* TAU_TRACK_IDLE_THREADS */
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////

void TauProfiler_theFunctionList(const char ***inPtr, int *numFuncs, bool addName, const char * inString) {
  static int numberOfFunctions = 0;

  if (addName) {
    numberOfFunctions++;
  } else {
    //We do not want to pass back internal pointers.
    *inPtr = ( char const **) malloc( sizeof(char *) * numberOfFunctions);

    for(int i=0;i<numberOfFunctions;i++) {
      (*inPtr)[i] = TheFunctionDB()[i]->GetName();
    }
    *numFuncs = numberOfFunctions;
  }
}

void TauProfiler_dumpFunctionNames() {

  int numFuncs;
  const char ** functionList;

  TauProfiler_theFunctionList(&functionList, &numFuncs);

  const char *dirname = TauEnv_get_profiledir();

  //Create temp write to file.
  char filename[1024];
  sprintf(filename,"%s/temp.%d.%d.%d",dirname, RtsLayer::myNode(),
	  RtsLayer::myContext(), RtsLayer::myThread());

  FILE* fp;
  if ((fp = fopen (filename, "w+")) == NULL) {
    char errormsg[1024];
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    return;
  }

  //Write data, and close.
  fprintf(fp, "number of functions %d\n", numFuncs);
  for (int i =0; i<numFuncs; i++) {
    fprintf(fp, "%s\n", functionList[i]);
  }
  fclose(fp);
  
  //Rename from the temp filename.
  char dumpfile[1024];
  sprintf(dumpfile,"%s/dump_functionnames_n,c,t.%d.%d.%d",dirname, RtsLayer::myNode(),
	  RtsLayer::myContext(), RtsLayer::myThread());
  rename(filename, dumpfile);
}

void TauProfiler_getUserEventList(const char ***inPtr, int *numUserEvents) {

  *numUserEvents = 0;

  vector<TauUserEvent*>::iterator eit;
  
  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    (*numUserEvents)++;
  }
  
  *inPtr = (char const **) malloc(sizeof(char*) * *numUserEvents);

  for(int i=0;i<*numUserEvents;i++) {
    (*inPtr)[i] = TheEventDB()[i]->GetEventName();
  }
}


void TauProfiler_getUserEventValues(const char **inUserEvents, int numUserEvents,
				  int **numEvents, double **max, double **min,
				  double **mean, double **sumSqr, int tid) {

  TAU_PROFILE("TAU_GET_EVENT_VALUES()", " ", TAU_IO);

#ifdef PROFILING_ON

  *numEvents = (int*) malloc (sizeof(int) * numUserEvents);
  *max = (double *) malloc (sizeof(double) * numUserEvents);
  *min = (double *) malloc (sizeof(double) * numUserEvents);
  *mean = (double *) malloc (sizeof(double) * numUserEvents);
  *sumSqr = (double *) malloc (sizeof(double) * numUserEvents);

  RtsLayer::LockDB();

  int idx = 0;
  vector<TauUserEvent*>::iterator eit;

  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    for (int i=0;i<numUserEvents;i++) {
      if ((inUserEvents != 0) && (strcmp(inUserEvents[i], (*eit)->GetEventName()) == 0)) {
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
#endif //PROFILING_ON
  
}

double *Profiler::getStartValues() {
#ifdef TAU_MULTIPLE_COUNTERS
  return StartTime;
#else
  return &StartTime;
#endif
}

void TauProfiler_theCounterList(const char ***inPtr, int *numCounters) {
  *inPtr = (const char **) malloc(sizeof(const char **) * 1);
  const char *tmpChar = "default counter";
  (*inPtr)[0] = tmpChar;
  *numCounters = 1;
}

static bool helperIsFunction(FunctionInfo *fi, Profiler *profiler) {
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

void TauProfiler_getFunctionValues(const char **inFuncs,
				 int numFuncs,
				 double ***counterExclusiveValues,
				 double ***counterInclusiveValues,
				 int **numCalls,
				 int **numSubr,
				 const char ***counterNames,
				 int *numCounters,
				 int tid) {
  TAU_PROFILE("TAU_GET_FUNC_VALS()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  

  int tmpNumberOfCounters;
  const char ** tmpCounterList;

#ifndef TAU_MULTIPLE_COUNTERS
  TauProfiler_theCounterList(&tmpCounterList,
			   &tmpNumberOfCounters);
#else
  bool *tmpCounterUsedList; // not used
  MultipleCounterLayer::theCounterListInternal(&tmpCounterList,
					       &tmpNumberOfCounters,
					       &tmpCounterUsedList);
#endif

  *numCounters = tmpNumberOfCounters;
  *counterNames = tmpCounterList;

  // allocate memory for the lists
  *counterExclusiveValues = (double **) malloc(sizeof(double *) * numFuncs);
  *counterInclusiveValues = (double **) malloc(sizeof(double *) * numFuncs);
  for (int i=0; i<numFuncs; i++) {
    (*counterExclusiveValues)[i] = (double *) malloc( sizeof(double) * tmpNumberOfCounters);
    (*counterInclusiveValues)[i] = (double *) malloc( sizeof(double) * tmpNumberOfCounters);
  }
  *numCalls = (int *) malloc(sizeof(int) * numFuncs);
  *numSubr = (int *) malloc(sizeof(int) * numFuncs);

  TauProfiler_updateIntermediateStatistics(tid);

  RtsLayer::LockDB();
  
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;

    int funcPos = matchFunction(fi, inFuncs, numFuncs);

    if (funcPos == -1) { // skip this function
      continue;
    }

    (*numCalls)[funcPos] = fi->GetCalls(tid);
    (*numSubr)[funcPos] = fi->GetSubrs(tid);
    
    int posCounter = 0;
    for (int m=0; m<MAX_TAU_COUNTERS; m++) {
      if (RtsLayer::getCounterUsed(m)) {
	(*counterInclusiveValues)[funcPos][posCounter] = fi->getDumpInclusiveValues(tid)[m];
	(*counterExclusiveValues)[funcPos][posCounter] = fi->getDumpExclusiveValues(tid)[m];
	posCounter++;
      }
    }
  }
  RtsLayer::UnLockDB();
#endif //PROFILING_ON
}



static void finalizeTrace(int tid) {

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

void TauProfiler_PurgeData(int tid) {
  
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;
  Profiler *curr;
  
  DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
  RtsLayer::LockDB();

  // Reset The Function Database
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    (*it)->SetCalls(tid,0);
    (*it)->SetSubrs(tid,0);
    (*it)->SetExclTimeZero(tid);
    (*it)->SetInclTimeZero(tid);
  }
  
  // Reset the Atomit/User Event Database
  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    (*eit)->LastValueRecorded[tid] = 0;
    (*eit)->NumEvents[tid] = 0L;
    (*eit)->MinValue[tid] = 9999999;
    (*eit)->MaxValue[tid] = -9999999;
    (*eit)->SumSqrValue[tid] = 0;
    (*eit)->SumValue[tid] = 0;
  }

  if (TauInternal_CurrentProfiler(tid) == NULL) {
    // There are no active timers, we are finished!
    RtsLayer::UnLockDB();
    return;	
  }

  // Now Re-register callstack entries
  curr = TauInternal_CurrentProfiler(tid);
  curr->ThisFunction->IncrNumCalls(tid);

#ifdef TAU_MULTIPLE_COUNTERS 
    for (int i=0;i<MAX_TAU_COUNTERS;i++) {
      curr->StartTime[i]=0;
    }
    RtsLayer::getUSecD(tid, curr->StartTime);	  
#else
    curr->StartTime = RtsLayer::getUSecD(tid) ;
#endif

  curr = curr->ParentProfiler;

  while (curr != 0) {
    curr->ThisFunction->IncrNumCalls(tid);
    curr->ThisFunction->IncrNumSubrs(tid);
#ifdef TAU_MULTIPLE_COUNTERS 
    for (int i=0;i<MAX_TAU_COUNTERS;i++) {
      curr->StartTime[i]=0;
    }
    RtsLayer::getUSecD(tid, curr->StartTime);	  
#else
    curr->StartTime = RtsLayer::getUSecD(tid) ;
#endif
    curr = curr->ParentProfiler;
  }
  
  RtsLayer::UnLockDB();
}


/////////////////////////////////////////////////////////////////////////


#ifdef TAU_COMPENSATE
//////////////////////////////////////////////////////////////////////
//  Profiler::GetNumChildren()
//  Description: Returns the total number of child timers (including 
//  children of children) executed under the given profiler. 
//////////////////////////////////////////////////////////////////////
long Profiler::GetNumChildren(void) {
  return NumChildren;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::SetNumChildren(value)
//  Description: Sets the total number of child timers.
//////////////////////////////////////////////////////////////////////
void Profiler::SetNumChildren(long value) {
  NumChildren = value;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::AddNumChildren(value)
//  Description: increments by value the number of child timers.
//////////////////////////////////////////////////////////////////////
void Profiler::AddNumChildren(long value) {
  NumChildren += value;
}
#endif /* TAU_COMPENSATE */

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
static int writeUserEvents(FILE *fp, int tid) {
  vector<TauUserEvent*>::iterator it;

  fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
  
  // Print UserEvent Data if any
  int numEvents = 0;
  for (it = TheEventDB().begin(); it != TheEventDB().end(); ++it) {
    if ((*it)->GetNumEvents(tid) == 0) { // skip user events with no calls
      continue;
    }
    numEvents++;
  }
  
  if (numEvents > 0) {
    // Data format 
    // # % userevents
    // # name numsamples max min mean sumsqr 
    fprintf(fp, "%d userevents\n", numEvents);
    fprintf(fp, "# eventname numevents max min mean sumsqr\n");
    
    for(it = TheEventDB().begin(); it != TheEventDB().end(); ++it) {
      if ((*it)->GetNumEvents(tid) == 0) { // skip user events with no calls
	continue;
      }
      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
	      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
    }
  }
  return 0;
}

static int writeHeader(FILE *fp, int numFunc, char *metricName) {
  char header[256];
  sprintf(header,"%d %s\n", numFunc, metricName);
  strcat(header,"# Name Calls Subrs Excl Incl ");
  
  strcat(header,"ProfileCalls");
  fprintf(fp, "%s", header);	
  return 0;
}


// This is a very important function, it must be called before writing function data to disk.
// This function fills in the values that will be dumped to disk.
// It performs the calculations for timers that are still on the stack.
int TauProfiler_updateIntermediateStatistics(int tid) {
  
  // get current values
  double currentTime[MAX_TAU_COUNTERS];

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

      double inclusiveToAdd[MAX_TAU_COUNTERS];
      double prevStartTime[MAX_TAU_COUNTERS];
      for (c=0; c<MAX_TAU_COUNTERS; c++) {
	inclusiveToAdd[c] = 0;
	prevStartTime[c] = 0;
      }
      
      for (Profiler *current = TauInternal_CurrentProfiler(tid); current != 0; current = current->ParentProfiler) {
	if (helperIsFunction(fi, current)) {
	  for (c=0; c<MAX_TAU_COUNTERS; c++) {
	    inclusiveToAdd[c] = currentTime[c] - current->getStartValues()[c]; 
	    excltime[c] += inclusiveToAdd[c] - prevStartTime[c];
	  }
	}
	for (c=0; c<MAX_TAU_COUNTERS; c++) {
	  prevStartTime[c] = currentTime[c] - current->getStartValues()[c];  
	}
      }
      for (c=0; c<MAX_TAU_COUNTERS; c++) {
	incltime[c] += inclusiveToAdd[c];
      }
    }
  }
  return 0;
}

// Checks if a function matches in a list of strings
// returns -1 if not found, otherwise, returns the index that it is found at
// if inFuncs is NULL, or numFuncs is 0, it returns 0
static int matchFunction(FunctionInfo *fi, const char **inFuncs, int numFuncs) {
  if (numFuncs == 0 || inFuncs == NULL) {
    return 0;
  }
  const char *tmpFunctionName = fi->GetName();
  for (int i=0; i<numFuncs; i++) {
    if ((inFuncs != NULL) && (strcmp(inFuncs[i], tmpFunctionName) == 0)) {
      return i;
    }
  }
  return -1;
}

// Writes function event data
static int writeFunctionData(FILE *fp, int tid, int metric, const char **inFuncs, int numFuncs) {
  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;

    if (-1 == matchFunction(*it, inFuncs, numFuncs)) { // skip this function
      continue;
    } 

    // get currently stored values
    double incltime = fi->getDumpInclusiveValues(tid)[metric];
    double excltime = fi->getDumpExclusiveValues(tid)[metric];

    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", fi->GetName(), 
	    fi->GetType(), fi->GetCalls(tid), fi->GetSubrs(tid), 
	    excltime, incltime);
    
    fprintf(fp,"0 "); // Indicating that profile calls is turned off
    fprintf(fp,"GROUP=\"%s\" \n", fi->GetAllGroups());
  }

  return 0;
}

// Writes a single profile file
static int writeProfile(FILE *fp, char *metricName, int tid, int metric, 
			const char **inFuncs, int numFuncs) {
  writeHeader(fp, TheFunctionDB().size(), metricName);
  fprintf(fp, " # ");	
  Tau_writeProfileMetaData(fp, metric);
  fprintf(fp, "\n");
  fflush(fp);
  writeFunctionData(fp, tid, metric, inFuncs, numFuncs);
  writeUserEvents(fp, tid);
  fclose(fp);
  return 0;
}

// Store profile data at the end of execution (when top level timer stops)
int TauProfiler_StoreData(int tid) {

  finalizeTrace(tid);

  if (TauEnv_get_profiling()) {
    TauProfiler_Snapshot("final", true, tid);
    
    if (TauEnv_get_profile_format() == TAU_FORMAT_PROFILE) {
      TauProfiler_DumpData(false, tid, "profile");
    }
  }
  return 1;
} 


// Returns directory name for the location of a particular metric
static int getProfileLocation(int metric, char *str) {
  const char *profiledir = TauEnv_get_profiledir();
#ifdef TAU_MULTIPLE_COUNTERS

  if (Tau_Global_numCounters <= 1) { 
    sprintf (str, "%s", profiledir);
  } else {
    const char *metricName = TauMetrics_getMetricName(metric);
    sprintf (str, "%s/MULTI__%s", profiledir, metricName);
  }
#else
  sprintf (str, "%s", profiledir);
#endif
  return 0;
}


int TauProfiler_DumpData(bool increment, int tid, const char *prefix) {
  return TauProfiler_writeData(tid, prefix, increment);
}


void getMetricHeader(int i, char *header) {
#ifdef TAU_MULTIPLE_COUNTERS
  sprintf(header, "templated_functions_MULTI_%s", RtsLayer::getCounterName(i));
#else
  sprintf(header, "%s", TauGetCounterString());
#endif
}


// Stores profile data
int TauProfiler_writeData(int tid, const char *prefix, bool increment, const char **inFuncs, int numFuncs) {
  
  TauProfiler_updateIntermediateStatistics(tid);

#ifdef PROFILING_ON 
  RtsLayer::LockDB();

  static bool createFlag = TauProfiler_createDirectories();

  for (int i=0;i<MAX_TAU_COUNTERS;i++) {
    if (TauMetrics_getMetricUsed(i)) {
      
      char metricHeader[1024];
      char profileLocation[1024];
      FILE* fp;

      getMetricHeader(i, metricHeader);
      getProfileLocation(i, profileLocation);
//       sprintf(filename, "%s/temp.%d.%d.%d", profileLocation, 
// 	      RtsLayer::myNode(), RtsLayer::myContext(), tid);

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
	char *day = strtok(stringTime," ");
	char *month = strtok(NULL," ");
	char *dayInt = strtok(NULL," ");
	char *time = strtok(NULL," ");
	char *year = strtok(NULL," ");
	// remove newline
	year[4] = '\0';
	char newStringTime[1024];
	sprintf(newStringTime, "%s-%s-%s-%s-%s", day, month, dayInt, time, year);
	
	sprintf(dumpfile, "%s/%s%s__%s__.%d.%d.%d", profileLocation, selectivePrefix,
		prefix, newStringTime,
		RtsLayer::myNode(), RtsLayer::myContext(), tid);

	if ((fp = fopen (dumpfile, "w+")) == NULL) {
	  char errormsg[1024];
	  sprintf(errormsg,"Error: Could not create %s",dumpfile);
	  perror(errormsg);
	  return 0;
	}

	char cwd[1024];
	char *tst = getcwd(cwd, 1024);
	TAU_VERBOSE("TAU: Writing profile %s, cwd = %s\n", dumpfile, cwd);

      } else {
	int flags = O_CREAT | O_EXCL | O_WRONLY;
	int mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
	int node = RtsLayer::myNode();
	sprintf(dumpfile,"%s/%s%s.%d.%d.%d", profileLocation, selectivePrefix, prefix, 
		node, RtsLayer::myContext(), tid);

	int sicortex = 0;
#ifdef TAU_SICORTEX
	sicortex = 1;
#endif

	if (sicortex && !Tau_get_usesMPI()) {
	  int test = open(dumpfile, flags, mode);
	  while (test == -1 && node < 99999) {
	    node++;
	    sprintf(dumpfile,"%s/%s%s.%d.%d.%d", profileLocation, selectivePrefix, prefix, 
		    node, RtsLayer::myContext(), tid);
	    test = open(dumpfile, flags, mode);
	  }
	  if ((fp = fdopen (test, "w")) == NULL) {
	    char errormsg[1024];
	    sprintf(errormsg,"Error: Could not create %s",dumpfile);
	    perror(errormsg);
	    return 0;
	  }

	} else {
	  if ((fp = fopen (dumpfile, "w+")) == NULL) {
	    char errormsg[1024];
	    sprintf(errormsg,"Error: Could not create %s",dumpfile);
	    perror(errormsg);
	    return 0;
	  }
	  char cwd[1024];
	  char *tst = getcwd(cwd, 1024);
	  TAU_VERBOSE("TAU: Writing profile %s, cwd = %s\n", dumpfile, cwd);
	}
      }
      writeProfile(fp, metricHeader, tid, i, inFuncs, numFuncs);
    }
  }
  
  RtsLayer::UnLockDB();
#endif //PROFILING_ON
  
  return 0;
}



int TauProfiler_dumpFunctionValues(const char **inFuncs,
				 int numFuncs,
				 bool increment,
				 int tid, const char *prefix) {
  
  TAU_PROFILE("TAU_DUMP_FUNC_VALS()", " ", TAU_IO);

  TauProfiler_writeData(tid, prefix, increment, inFuncs, numFuncs);
  return 0;
}


bool TauProfiler_createDirectories() {

#ifdef TAU_MULTIPLE_COUNTERS
  static bool flag = true;
  if (flag && Tau_Global_numCounters > 1) {
    for (int i=0;i<MAX_TAU_COUNTERS;i++) {
      if (TauMetrics_getMetricUsed(i)) {
	const char * tmpChar = TauMetrics_getMetricName(i);
	char *newdirname = new char[1024];
	//char *rmdircommand = new char[1024];
	char *mkdircommand = new char[1024];
	
	const char *dirname = TauEnv_get_profiledir();
	
	sprintf(newdirname,"%s/MULTI__%s",dirname,tmpChar);
	//sprintf(rmdircommand,"rm -rf %s",newdirname);
	sprintf(mkdircommand,"mkdir -p %s",newdirname);
	
	//system(rmdircommand);
	//system(mkdircommand);
	/* On IBM BGL, system command doesn't execute. So, we need to create
	   these directories using our mkdir syscall instead. */
	/* OLD: mkdir(newdirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); */
	mkdir(newdirname, S_IRWXU | S_IRGRP | S_IXGRP);
      }
    }
    flag = false;
  }

#endif
  return true;
}

/***************************************************************************
 * $RCSfile: Profiler.cpp,v $   $Author: amorris $
 * $Revision: 1.236 $   $Date: 2009/03/30 21:51:20 $
 * VERSION_ID: $Id: Profiler.cpp,v 1.236 2009/03/30 21:51:20 amorris Exp $ 
 ***************************************************************************/
