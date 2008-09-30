/*****************************************************************************
 **			TAU Portable Profiling Package			    **
 **			http://www.cs.uoregon.edu/research/tau	            **
 *****************************************************************************
 **    Copyright 1999  						   	    **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/*****************************************************************************
 **	File 		: Profiler.cpp					    **
 **	Description 	: TAU Profiling Package				    **
 **	Author		: Sameer Shende					    **
 **	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	    **
 **	Flags		: Compile with				            **
 **			  -DPROFILING_ON to enable profiling (ESSENTIAL)    **
 **			  -DPROFILE_STATS for Std. Deviation of Excl Time   **
 **			  -DSGI_HW_COUNTERS for using SGI counters 	    **
 **			  -DPROFILE_CALLS  for trace of each invocation     **
 **                        -DSGI_TIMERS  for SGI fast nanosecs timer        **
 **			  -DTULIP_TIMERS for non-sgi Platform	 	    **
 **			  -DPOOMA_STDSTL for using STD STL in POOMA src     **
 **			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	    **
 **			  -DPOOMA_KAI for KCC compiler 			    **
 **			  -DDEBUG_PROF  for internal debugging messages     **
 **                        -DPROFILE_CALLSTACK to enable callstack traces   **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau        **
 ****************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF // For Debugging Messages from Profiler.cpp
#include "Profile/Profiler.h"
#include "tauarch.h"

#ifdef TAU_WINDOWS
typedef __int64 x_int64;
typedef unsigned __int64 x_uint64;
double TauWindowsUsecD(void);
#include <io.h>
#include <direct.h> /* for getcwd */
#define S_IRUSR 0
#define S_IWUSR 0
#define S_IRGRP 0
#define S_IWGRP 0
#define S_IROTH 0
#define S_IWOTH 0
#else
typedef long long x_int64;
typedef unsigned long long x_uint64;
#endif

//#ifndef TAU_WINDOWS
extern "C" void Tau_shutdown(void);
//#endif //TAU_WINDOWS

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <stdio.h> 
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>

#if (!defined(TAU_WINDOWS))
#include <sys/types.h>
#include <sys/stat.h>
#ifndef TAU_DISABLE_METADATA
#include <sys/utsname.h> // for host identification (uname)
#endif
#include <unistd.h>

#if (defined(POOMA_TFLOP) || !defined(TULIP_TIMERS))
#include <sys/time.h>
#else
#ifdef TULIP_TIMERS 
#include "Profile/TulipTimers.h"
#endif //TULIP_TIMERS 
#endif //POOMA_TFLOP

#endif //TAU_WINDOWS

#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
#include "Profile/TauVampirTrace.h"
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
#include "elg_trc.h"

#ifdef TAU_SCALASCA
extern "C" {
void esd_enter (elg_ui4 rid);
void esd_exit (elg_ui4 rid);
}
#endif /* SCALASCA */

#else /* TAU_EPILOG */
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 

#ifdef RENCI_STFF
#include "Profile/RenciSTFF.h"
#endif // RENCI_STFF


int Tau_writeProfileMetaData(FILE *fp, int counter);

static int writeUserEvents(FILE *fp, int tid);
static int matchFunction(FunctionInfo *fi, const char **inFuncs, int numFuncs);
extern "C" int Tau_get_usesMPI();

//#define PROFILE_CALLS // Generate Excl Incl data for each call 

//////////////////////////////////////////////////////////////////////
//Initialize static data
//////////////////////////////////////////////////////////////////////

// No need to initialize FunctionDB. using TheFunctionDB() instead.
// vector<FunctionInfo*> FunctionInfo::FunctionDB[TAU_MAX_THREADS] ;
Profiler * Profiler::CurrentProfiler[] = {0}; // null to start with

#if defined(TAUKTAU)
#include <Profile/KtauProfiler.h>
#endif /* TAUKTAU */

// The rest of CurrentProfiler entries are initialized to null automatically
//TauGroup_t RtsLayer::ProfileMask = TAU_DEFAULT;

// Default value of Node.
//int RtsLayer::Node = -1;

//////////////////////////////////////////////////////////////////////
// For OpenMP
//////////////////////////////////////////////////////////////////////
#ifdef TAU_OPENMP 
#ifndef TAU_MULTIPLE_COUNTERS
double TheLastTimeStamp[TAU_MAX_THREADS]; 
#else /* FOR MULTIPLE COUNTERS */
double TheLastTimeStamp[TAU_MAX_THREADS][MAX_TAU_COUNTERS]; 
#endif /* MULTIPLE_COUNTERS */
#endif /* TAU_OPENMP */
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
// TAU_DEPTH_LIMIT 
//////////////////////////////////////////////////////////////////////
int& TauGetDepthLimit(void) {
  static int depth = 0;
  char *depthvar; 
  if (depth == 0) {
    depthvar = getenv("TAU_DEPTH_LIMIT"); 
    if (depthvar == (char *) NULL) {
      depth = INT_MAX; 
    } else {
      depth = atoi(depthvar);
    }
  } 
  return depth; 
}



//////////////////////////////////////////////////////////////////////
// Shutdown routine which calls TAU's shutdown
//////////////////////////////////////////////////////////////////////
void TauAppShutdown(void) {
  Tau_shutdown();
}
 
//////////////////////////////////////////////////////////////////////
// Get the string containing the counter name
//////////////////////////////////////////////////////////////////////
char *TauGetCounterString(void) {
#ifdef SGI_HW_COUNTERS
  return "templated_functions_hw_counters";
#elif (defined (TAU_PAPI) || defined (TAU_PCL)	\
       || defined(TAU_PAPI_WALLCLOCKTIME)	\
       || defined(TAU_PAPI_VIRTUAL))
  char *tau_env = NULL;

#ifdef TAU_PAPI
  tau_env = getenv("PAPI_EVENT");
#else  /* TAU_PAPI */
#ifdef TAU_PCL
  tau_env = getenv("PCL_EVENT");
#endif /* TAU_PCL */
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
#else  // ! (TAU_PAPI/PCL) => SGI_TIMERS, TULIP_TIMERS 
#ifdef TAU_MUSE
  return "templated_functions_hw_counters";
#endif /* TAU_MUSE */
  return "templated_functions";
#endif // ALL options
}
//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class Profiler
//////////////////////////////////////////////////////////////////////

#ifdef TAU_MPITRACE

//////////////////////////////////////////////////////////////////////
void Profiler::EnableAllEventsOnCallStack(int tid, Profiler *current) {
  /* Go up the callstack and enable all events on it */
  if (current != (Profiler *) NULL) {
    DEBUGPROFMSG(RtsLayer::myNode()<<" This func = "<<current->ThisFunction->GetName()<<" RecordEvent = "<<current->RecordEvent<<endl;);
    if (!current->RecordEvent) { 
      DEBUGPROFMSG(RtsLayer::myNode()<< " Enabling event "<<current->ThisFunction->GetName()<<endl;);
      current->RecordEvent = true;
      EnableAllEventsOnCallStack(tid, current->ParentProfiler);
      /* process the current event */
      DEBUGPROFMSG(RtsLayer::myNode()<<" Processing EVENT "<<current->ThisFunction->GetName()<<endl;);
      TraceEvent(current->ThisFunction->GetFunctionId(), 1, tid, (x_uint64) current->StartTime, 1); 
#ifdef TAU_MULTIPLE_COUNTERS 
      MultipleCounterLayer::triggerCounterEvents((x_uint64) current->StartTime[0], current->StartTime, tid);
#endif /* TAU_MULTIPLE_COUNTERS */
    }
  }
}

#endif /* TAU_MPITRACE */
//////////////////////////////////////////////////////////////////////

void Profiler::Start(int tid) { 
//    fprintf (stderr, "[%d:%d-%d] Profiler::Start for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);

#ifdef TAU_OPENMP
  if (tid != 0) {
    Tau_create_top_level_timer_if_necessary();
  }
#endif

  ParentProfiler = CurrentProfiler[tid]; // Timers


#ifdef TAU_DEPTH_LIMIT
  int userspecifieddepth = TauGetDepthLimit();
  if (ParentProfiler) {
    SetDepthLimit(ParentProfiler->GetDepthLimit()+1);
  } else {
    SetDepthLimit(1);
  }
  int mydepth = GetDepthLimit();
  DEBUGPROFMSG("Start: Name: "<< ThisFunction->GetName()<<" mydepth = "<<mydepth<<", userspecifieddepth = "<<userspecifieddepth<<endl;);
  if (mydepth > userspecifieddepth) { 
    /* set the profiler */
    CurrentProfiler[tid] = this;
    return; 
  }
#endif /* TAU_DEPTH_LIMIT */
  
#ifdef TAU_PROFILEPHASE
  if (ParentProfiler == (Profiler *) NULL) {
    if (ThisFunction->AllGroups.find("TAU_PHASE", 0) == string::npos) {
      ThisFunction->AllGroups.append(" | TAU_PHASE"); 
    }
  }
#endif /* TAU_PROFILEPHASE */
  
  x_uint64 TimeStamp = 0L;
  
  DEBUGPROFMSG("Profiler::Start: MyProfileGroup_ = " << MyProfileGroup_ 
	       << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
  if ((MyProfileGroup_ & RtsLayer::TheProfileMask()) 
      && RtsLayer::TheEnableInstrumentation()) {
    if (ThisFunction == (FunctionInfo *) NULL) return; // Mapping
    DEBUGPROFMSG("Profiler::Start Entering " << ThisFunction->GetName()<<endl;);
    
#ifdef TAU_PROFILEMEMORY
    ThisFunction->GetMemoryEvent()->TriggerEvent(TauGetMaxRSS());
#endif /* TAU_PROFILEMEMORY */
#ifdef TAU_PROFILEHEADROOM
    ThisFunction->GetHeadroomEvent()->TriggerEvent((double)TauGetFreeMemory());
#endif /* TAU_PROFILEHEADROOM */
#ifdef TAU_COMPENSATE
    SetNumChildren(0); /* for instrumentation perturbation compensation */
#endif /* TAU_COMPENSATE */
    // Initialization is over, now record the time it started
#ifndef TAU_MULTIPLE_COUNTERS 
    StartTime =  RtsLayer::getUSecD(tid) ;
    TimeStamp += (x_uint64) StartTime;
#else //TAU_MULTIPLE_COUNTERS
    //Initialize the array to zero, as some of the elements will
    //not be set by counting functions.
    for (int i=0;i<MAX_TAU_COUNTERS;i++) {
      StartTime[i]=0;
    }
    //Now get the start times.
    RtsLayer::getUSecD(tid, StartTime);	  
    TimeStamp += (unsigned long long) StartTime[0]; // USE COUNTER1 for tracing
#endif//TAU_MULTIPLE_COUNTERS
    
#ifdef TAU_CALLPATH
    CallPathStart(tid);
#endif // TAU_CALLPATH
#ifdef TAU_PROFILEPARAM
    ProfileParamFunction = NULL;
    if (ParentProfiler && ParentProfiler->ProfileParamFunction ) {
      ParentProfiler->ProfileParamFunction->IncrNumSubrs(tid);
    }
#endif /* TAU_PROFILEPARAM */
    
#ifdef TRACING_ON
#ifdef TAU_MPITRACE
    if (MyProfileGroup_ & TAU_MESSAGE) {
      /* if we're in the group, we must first enable all the other events
       * on the callstack */
      DEBUGPROFMSG(RtsLayer::myNode()<< " Function is enabled: "<<ThisFunction->GetName()<<endl;);
      EnableAllEventsOnCallStack(tid, this);
    }
#else /* TAU_MPITRACE */
#ifdef TAU_VAMPIRTRACE 
    TimeStamp = vt_pform_wtime();
    
    DEBUGPROFMSG("Calling vt_enter: ["<<ThisFunction->GetFunctionId()<<"] "
		 << ThisFunction->GetName()<<" Time" <<TimeStamp<<endl;);
    vt_enter((uint64_t *) &TimeStamp, ThisFunction->GetFunctionId());
#else /* TAU_VAMPITRACE */
#ifdef TAU_EPILOG
    DEBUGPROFMSG("Calling elg_enter: ["<<ThisFunction->GetFunctionId()<<"] "
		 << ThisFunction->GetName()<<endl;);
    esd_enter(ThisFunction->GetFunctionId());
#else /* TAU_EPILOG */
    TraceEvent(ThisFunction->GetFunctionId(), 1, tid, TimeStamp, 1); 
    // 1 is for entry in second parameter and for use TimeStamp in last
    DEBUGPROFMSG("Start TimeStamp for Tracing = "<<TimeStamp<<endl;);
#ifdef TAU_MULTIPLE_COUNTERS 
    MultipleCounterLayer::triggerCounterEvents(TimeStamp, StartTime, tid);
#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif /* TAU_MPITRACE */
#endif /* TRACING_ON */

    /* What do we maintain if PROFILING is turned off and tracing is turned on and
       throttling is not disabled? We need to maintain enough info to generate 
       inclusive time and keep information about AddInclFlag */
#ifndef PROFILING_ON
#ifdef  TRACING_ON
#ifndef TAU_DISABLE_THROTTLE
    if (TauEnv_get_throttle() && (ThisFunction->GetAlreadyOnStack(tid)== false)) {
      /* Set the callstack flag */
      AddInclFlag = true; 
      ThisFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
      
      // Next, increment the number of calls
      ThisFunction->IncrNumCalls(tid);
    }
#endif /* TAU_DISABLE_THROTTLE is off */
#endif /* TRACING is on */
#endif /* PROFILING is off */
    
#ifdef PROFILING_ON
    // First, increment the number of calls
    ThisFunction->IncrNumCalls(tid);
    
    // now increment parent's NumSubrs()
    if (ParentProfiler != 0) {
      ParentProfiler->ThisFunction->IncrNumSubrs(tid);	
    }
    
    // Next, if this function is not already on the call stack, put it
    if (ThisFunction->GetAlreadyOnStack(tid) == false) { 
      AddInclFlag = true; 
      // We need to add Inclusive time when it gets over as 
      // it is not already on callstack.

      ThisFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
    } else { // the function is already on callstack, no need to add inclusive time
      AddInclFlag = false;
    }
    
    DEBUGPROFMSG("Start Time = "<< StartTime<<endl;);
#endif // PROFILING_ON
    
    
    DEBUGPROFMSG("nct "<< RtsLayer::myNode() << "," 
		 << RtsLayer::myContext() << ","  << tid 
		 << " Profiler::Start (tid)  : Name : " 
		 << ThisFunction->GetName() <<" Type : " << ThisFunction->GetType() 
		 << endl; );
    
    CurrentProfiler[tid] = this;
    if (ParentProfiler != 0) {
      DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
		   << RtsLayer::myContext() << ","  << tid
		   << " Inside "<< ThisFunction->GetName()<< " Setting ParentProfiler "
		   << ParentProfiler->ThisFunction->GetName()<<endl
		   << " ParentProfiler = "<<ParentProfiler << " CurrProf = "
		   << CurrentProfiler[tid] << " = this = "<<this<<endl;);
    }
    
#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
    ExclTimeThisCall = 0;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
    
    /********* KTAU CODE *************************/
    
#if defined(TAUKTAU)
    ThisKtauProfiler->Start(this);
#endif /* TAUKTAU */

  } else { 
    /* If instrumentation is disabled, set the CurrentProfiler */
    ParentProfiler = CurrentProfiler[tid] ;
    CurrentProfiler[tid] = this;
  } /* this is so Stop can access CurrentProfiler as well */
}

//////////////////////////////////////////////////////////////////////

Profiler::Profiler( FunctionInfo * function, TauGroup_t ProfileGroup, 
		    bool StartStop, int tid) {
#if defined(TAUKTAU) 
  ThisKtauProfiler = KtauProfiler::GetKtauProfiler(tid);
#endif /* defined(TAUKTAU) */
  
  StartStopUsed_ = StartStop; // will need it later in ~Profiler
  MyProfileGroup_ = function->GetProfileGroup(tid) ;
  //MyProfileGroup_ = ProfileGroup;
  /* Get the latest profile group from the function. For throttling. */
  ThisFunction = function ; 
#ifdef TAU_MPITRACE
  RecordEvent = false; /* by default, we don't record this event */
#endif /* TAU_MPITRACE */
#ifdef TAU_PROFILEPHASE
  SetPhase(false); /* By default it is not in phase */
#endif /* TAU_PROFILEPHASE */ 
  DEBUGPROFMSG("Profiler::Profiler: MyProfileGroup_ = " << MyProfileGroup_ 
	       << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
  
  if(!StartStopUsed_) { // Profiler ctor/dtor interface used
    Start(tid); 
  }
}


//////////////////////////////////////////////////////////////////////

Profiler::Profiler( const Profiler& X)
  : ThisFunction(X.ThisFunction),
    ParentProfiler(X.ParentProfiler),
    MyProfileGroup_(X.MyProfileGroup_),
    StartStopUsed_(X.StartStopUsed_) {
#if defined(TAUKTAU)
  ThisKtauProfiler = KtauProfiler::GetKtauProfiler();
#endif /* defined(TAUKTAU) */
  
#ifndef TAU_MULTIPLE_COUNTERS	
  StartTime = X.StartTime;
#else //TAU_MULTIPLE_COUNTERS
  
  for (int i=0;i<MAX_TAU_COUNTERS;i++) {
    StartTime[i] = X.StartTime[i];
  }
#endif//TAU_MULTIPLE_COUNTERS

#ifdef TAU_PROFILEPHASE
  PhaseFlag = X.PhaseFlag;
#endif /* TAU_PROFILEPHASE */ 

  CurrentProfiler[RtsLayer::myThread()] = this;
  
#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
  ExclTimeThisCall = X.ExclTimeThisCall;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
}

//////////////////////////////////////////////////////////////////////

Profiler& Profiler::operator= (const Profiler& X) {
#ifndef TAU_MULTIPLE_COUNTERS	
  StartTime = X.StartTime;
#else //TAU_MULTIPLE_COUNTERS
  for (int i=0;i<MAX_TAU_COUNTERS;i++) {
    StartTime[i] = X.StartTime[i];
  }
#endif//TAU_MULTIPLE_COUNTERS
  
  ThisFunction = X.ThisFunction;
  ParentProfiler = X.ParentProfiler; 
  MyProfileGroup_ = X.MyProfileGroup_;
  StartStopUsed_ = X.StartStopUsed_;
#ifdef TAU_PROFILEPHASE
  PhaseFlag = X.PhaseFlag;
#endif /* TAU_PROFILEPHASE */ 
  
  DEBUGPROFMSG(" Profiler& Profiler::operator= (const Profiler& X)" <<endl;);
  
#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
  ExclTimeThisCall = X.ExclTimeThisCall;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
  return (*this) ;
}

//////////////////////////////////////////////////////////////////////

void Profiler::Stop(int tid, bool useLastTimeStamp) {
//    fprintf (stderr, "[%d:%d-%d] Profiler::Stop  for %s (%p)\n", RtsLayer::getPid(), RtsLayer::getTid(), tid, ThisFunction->GetName(), ThisFunction);
  x_uint64 TimeStamp = 0L; 
  if (CurrentProfiler[tid] == NULL) return;
  
#ifdef TAU_DEPTH_LIMIT
  int userspecifieddepth = TauGetDepthLimit();
  int mydepth = GetDepthLimit(); 
  if (mydepth > userspecifieddepth) {
    CurrentProfiler[tid] = ParentProfiler; 
    DEBUGPROFMSG("Stop: mydepth = "<<mydepth<<", userspecifieddepth = "<<userspecifieddepth<<endl;);
    return;
  }
#endif /* TAU_DEPTH_LIMIT */

  DEBUGPROFMSG("Profiler::Stop: MyProfileGroup_ = " << MyProfileGroup_ 
	       << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
  if ((MyProfileGroup_ & RtsLayer::TheProfileMask()) 
      && RtsLayer::TheEnableInstrumentation()) {
    if (ThisFunction == (FunctionInfo *) NULL) return; // Mapping
    DEBUGPROFMSG("Profiler::Stop for routine = " << ThisFunction->GetName()<<endl;);
    
#ifdef TAU_COMPENSATE 
    DEBUGPROFMSG("Profiler::Stop - "<<ThisFunction->GetName()<<" NumChildren = "
		 <<GetNumChildren()<<endl;);
#endif /* TAU_COMPENSATE */
    
#ifdef TAU_COMPENSATE
#ifndef TAU_MULTIPLE_COUNTERS 
    double tover = TauGetTimerOverhead(TauFullTimerOverhead);
    double tnull = TauGetTimerOverhead(TauNullTimerOverhead);
#endif /* TAU_MULTIPLE_COUNTERS */
#endif /* TAU_COMPENSATE */
    
#ifndef TAU_MULTIPLE_COUNTERS
    double CurrentTime; 
    if (useLastTimeStamp) {
      /* for openmp parallel regions */
      /* .TAU Application needs to be stopped */
#ifdef TAU_OPENMP 
      CurrentTime = TheLastTimeStamp[tid]; 
#endif /* TAU_OPENMP */
    } else { /* use the usual mechanism */
      CurrentTime = RtsLayer::getUSecD(tid);
    }
    
#if defined(TAUKTAU)
#ifdef KTAU_DEBUGPROF
    printf("Profiler::Stop: --EXIT-- %s \n", CurrentProfiler[tid]->ThisFunction->GetName());
#endif /*KTAU_DEBUGPROF*/
    ThisKtauProfiler->Stop(this, AddInclFlag);
#endif /* TAUKTAU */
    
    double TotalTime = CurrentTime - StartTime;
    TimeStamp += (x_uint64) CurrentTime; 
    
#ifdef TAU_OPENMP
    TheLastTimeStamp[tid] = CurrentTime;
#endif /* TAU_OPENMP */
    
#if (defined(TAU_COMPENSATE ) && defined(PROFILING_ON))
    /* To compensate for timing overhead, shrink the totaltime! */
    TotalTime = TotalTime - tnull - GetNumChildren() * tover; 
    if (TotalTime < 0 ) {
      TotalTime = 0;
      DEBUGPROFMSG("TotalTime negative in "<<ThisFunction->GetName()<<endl;);
    }
#endif /* TAU_COMPENSATE && PROFILING_ON */
#else //TAU_MULTIPLE_COUNTERS
    // first initialize the CurrentTime
    int i;
    for (i=0; i < MAX_TAU_COUNTERS; i++) {
      CurrentTime[i] = 0;
    }
    //Get the current counter values.
    if (useLastTimeStamp) {
      /* for openmp parallel regions */
      /* .TAU Application needs to be stopped */
#ifdef TAU_OPENMP 
      for (i=0; i < MAX_TAU_COUNTERS; i++) {
	CurrentTime[i] = TheLastTimeStamp[tid][i]; 
      }
#endif /* TAU_OPENMP */
    } else { 
      /* use the usual mechanism */
      RtsLayer::getUSecD(tid, CurrentTime);
    }
#ifdef TAU_OPENMP
    for (i=0; i < MAX_TAU_COUNTERS; i++) {
      TheLastTimeStamp[tid][i] = CurrentTime[i]; 
    }
#endif /* TAU_OPENMP */

#if defined(TAUKTAU)
#ifdef KTAU_DEBUGPROF
    printf("Profiler::Stop: --EXIT-- %s \n", CurrentProfiler[tid]->ThisFunction->GetName());
#endif /*KTAU_DEBUGPROF*/
    ThisKtauProfiler->Stop(this, AddInclFlag);
#endif /* TAUKTAU */
    
#ifdef PROFILING_ON
#ifdef TAU_COMPENSATE
    double *tover = TauGetTimerOverhead(TauFullTimerOverhead);
    double *tnull = TauGetTimerOverhead(TauNullTimerOverhead);
#endif /* TAU_COMPENSATE */
    for (int k=0;k<MAX_TAU_COUNTERS;k++) {
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
    TimeStamp += (unsigned long long) CurrentTime[0]; // USE COUNTER1
    
#endif//TAU_MULTIPLE_COUNTERS
    
    /* Should we detect memory leaks here? */
    if (ParentProfiler == (Profiler *) NULL && TheSafeToDumpData() && !RtsLayer::isCtorDtor(ThisFunction->GetName())) {
      TauDetectMemoryLeaks(); /* the last event should be before final exit */
    }
#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
    TimeStamp = vt_pform_wtime();
    DEBUGPROFMSG("Calling vt_exit(): "<< ThisFunction->GetName()<<
		 "With Timestamp = "<<TimeStamp<<endl;);
    vt_exit((uint64_t *)&TimeStamp);
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
    DEBUGPROFMSG("Calling elg_exit(): "<< ThisFunction->GetName()<<endl;);
    esd_exit(ThisFunction->GetFunctionId());
#else /* TAU_EPILOG */
#ifdef TAU_MPITRACE
    if (RecordEvent) {
#endif /* TAU_MPITRACE */
      TraceEvent(ThisFunction->GetFunctionId(), -1, tid, TimeStamp, 1); 
      // -1 is for exit, 1 is for use TimeStamp in the last argument
      DEBUGPROFMSG("Stop TimeStamp for Tracing = "<<TimeStamp<<endl;);
#ifdef TAU_MULTIPLE_COUNTERS 
      MultipleCounterLayer::triggerCounterEvents(TimeStamp, CurrentTime, tid);
#endif /* TAU_MULTIPLE_COUNTERS */
#ifdef TAU_MPITRACE
    }
#endif /* TAU_MPITRACE */
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif //TRACING_ON
    
    /* What should we do while exiting when profiling is off, tracing is on and 
       throttling is on? */
#ifndef PROFILING_ON
#ifdef  TRACING_ON
#ifndef TAU_DISABLE_THROTTLE
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
#endif /* TAU_DISABLE_THROTTLE is off */
#endif /* TRACING is on */
#endif /* PROFILING is off */
    
#ifdef PROFILING_ON  // Calculations relevent to profiling only 
    
#ifdef TAU_CALLPATH
    CallPathStop(TotalTime, tid);
#endif // TAU_CALLPATH
    
#ifdef RENCI_STFF
#ifdef TAU_CALLPATH
    RenciSTFF::recordValues(CallPathFunction, TimeStamp, TotalTime, tid);
#endif //TAU_CALLPATH
    RenciSTFF::recordValues(ThisFunction, TimeStamp, TotalTime, tid);
#endif //RENCI_STFF

#ifdef TAU_PROFILEPARAM
    ProfileParamStop(TotalTime, tid);
    if (ParentProfiler && ParentProfiler->ProfileParamFunction) {
      /* Increment the parent's NumSubrs and decrease its exclude time */
      ParentProfiler->ProfileParamFunction->ExcludeTime(TotalTime, tid);
    }
#endif /* TAU_PROFILEPARAM */

    DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << "," 
		 << RtsLayer::myContext() << "," << tid 
		 << " Profiler::Stop() : Name : "<< ThisFunction->GetName() 
		 << " Start : " <<StartTime <<" TotalTime : " << TotalTime
		 << " AddInclFlag : " << AddInclFlag << endl;);
    
    if (AddInclFlag == true) { // The first time it came on call stack
      ThisFunction->SetAlreadyOnStack(false, tid); // while exiting
      
      DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << "," 
		   << RtsLayer::myContext() << "," << tid  << " "  
		   << "STOP: After SetAlreadyOnStack Going for AddInclTime" <<endl; );
      
      // And its ok to add both excl and incl times
      ThisFunction->AddInclTime(TotalTime, tid);
      DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
		   << RtsLayer::myContext() << "," << tid
		   << " AddInclFlag true in Stop Name: "<< ThisFunction->GetName()
		   << " Type: " << ThisFunction->GetType() << endl; );
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
#ifdef TAU_CALLPATH
    if (ParentProfiler != NULL) {
      CallPathFunction->ResetExclTimeIfNegative(tid); 
    }
#endif /* TAU_CALLPATH */
#ifdef TAU_PROFILEPARAM
    if (ParentProfiler != NULL) {
      ProfileParamFunction->ResetExclTimeIfNegative(tid);
    }
#endif /* TAU_PROFILEPARAM */
#endif /* TAU_COMPENSATE */

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS)|| defined(PROFILE_CALLSTACK) )
    ExclTimeThisCall += TotalTime;
    DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
		 << RtsLayer::myContext() << "," << tid  << " " 
		 << "Profiler::Stop() : Name " 
		 << ThisFunction->GetName() << " ExclTimeThisCall = "
		 << ExclTimeThisCall << " InclTimeThisCall " << TotalTime << endl;);
    
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
    
#ifdef PROFILE_CALLS
    ThisFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
#ifdef TAU_CALLPATH
    if (CallPathFunction) {
      CallPathFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
    }
#endif /* TAU_CALLPATH */
#ifdef TAU_PROFILEPARAM
    if (ProfileParamFunction) {
      ProfileParamFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
    }
#endif /* TAU_PROFILEPARAM */
#endif // PROFILE_CALLS

#ifdef PROFILE_STATS
    ThisFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
#ifdef TAU_CALLPATH
    if (CallPathFunction) {
      CallPathFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
    }
#endif /* TAU_CALLPATH */
#ifdef TAU_PROFILEPARAM
    if (ProfileParamFunction) {
      ProfileParamFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
    }
#endif /* TAU_PROFILEPARAM */
#endif // PROFILE_STATS
    
    if (ParentProfiler != (Profiler *) NULL) {
      
      DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
		   << RtsLayer::myContext() << "," << tid  
		   << " Profiler::Stop(): ParentProfiler Function Name : " 
		   << ParentProfiler->ThisFunction->GetName() << endl;);
      DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
		   << RtsLayer::myContext() << "," << tid
		   << " Exiting from "<<ThisFunction->GetName() << " Returning to "
		   << ParentProfiler->ThisFunction->GetName() << endl;);
      
      if (ParentProfiler->ThisFunction != (FunctionInfo *) NULL)
	ParentProfiler->ThisFunction->ExcludeTime(TotalTime, tid);
      else {
	cout <<"ParentProfiler's Function info is NULL" <<endl;
      }
      
#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
      ParentProfiler->ExcludeTimeThisCall(TotalTime);
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
      
#ifdef TAU_COMPENSATE
      ParentProfiler->AddNumChildren(GetNumChildren()+1);
      /* Add 1 and my children to my parents total number of children */
#endif /* TAU_COMPENSATE */

    }

#endif //PROFILING_ON

    /* if the frequency of events is high, disable them */
#ifndef TAU_DISABLE_THROTTLE /* unless we are overriding the throttle */
    double inclusiveTime; 
#ifdef TAU_MULTIPLE_COUNTERS
    inclusiveTime = ThisFunction->GetInclTimeForCounter(tid, 0); 
    /* here we get the array of double values representing the double 
       metrics. We choose the first counter */
#else  /* TAU_MULTIPLE_COUNTERS */
    inclusiveTime = ThisFunction->GetInclTime(tid); 
    /* when multiple counters are not used, it is a single metric or double */
#endif /* MULTIPLE_COUNTERS */
    DEBUGPROFMSG("Calls = "<<ThisFunction->GetCalls(tid)
		 <<" inclusiveTime = "<<inclusiveTime<<endl);
    if (TauEnv_get_throttle() && (ThisFunction->GetCalls(tid) > TauEnv_get_throttle_numcalls()) && (inclusiveTime/ThisFunction->GetCalls(tid) < TauEnv_get_throttle_percall()) && AddInclFlag) { 
      /* Putting AddInclFlag means we can't throttle recursive calls */
      ThisFunction->SetProfileGroup(TAU_DISABLE, tid);
      ThisFunction->SetPrimaryGroupName("TAU_DISABLE");
      //cout <<"TAU<"<<RtsLayer::myNode()<<">: Throttle: Disabling "<<ThisFunction->GetName()<<endl;
      TAU_VERBOSE("TAU<%d>: Throttle: Disabling %s\n", RtsLayer::myNode(), ThisFunction->GetName());
    }
#endif /* TAU_DISABLE_THROTTLE */
    
    
    // First check if timers are overlapping.
    if (CurrentProfiler[tid] != this) {
      DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
		   << RtsLayer::myContext() << "," << tid
		   << " ERROR: Timers Overlap. Illegal operation Profiler::Stop " 
		   << ThisFunction->GetName() << " " 
		   << ThisFunction->GetType() <<endl;);
      if (CurrentProfiler[tid] != (Profiler *) NULL) {
	if (CurrentProfiler[tid]->ThisFunction != (FunctionInfo *)NULL) {
#ifdef TAU_OPENMP
#pragma omp critical
#endif /* TAU_OPENMP */
	  cout << "Overlapping function = "
	       << CurrentProfiler[tid]->ThisFunction->GetName () << " " 
	       << CurrentProfiler[tid]->ThisFunction->GetType() 
	       << " Other function " << this->ThisFunction->GetName()
	       << this->ThisFunction->GetType()<< " Tid = "<<tid<<endl;
	} else {
	  cout <<"CurrentProfiler is not Null but its FunctionInfo is"<<endl;
	}
      }
    }
    // While exiting, reset value of CurrentProfiler to reflect the parent
    CurrentProfiler[tid] = ParentProfiler;
    DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
		 << RtsLayer::myContext() << "," << tid
		 << " Stop: " << ThisFunction->GetName() 
		 << " TheSafeToDumpData() = " << TheSafeToDumpData()
		 << " CurrProf = "<<CurrentProfiler[tid] << " this = "
		 << this<<endl;);
    
    if (ParentProfiler == (Profiler *) NULL) {
      // For Dyninst. tcf gets called after main and all the data structures may not be accessible
      // after main exits. Still needed on Linux - we use TauProgramTermination()
      if (strcmp(ThisFunction->GetName(), "_fini") == 0) TheSafeToDumpData() = 0;
#ifndef TAU_WINDOWS
      atexit(TauAppShutdown);
#endif //TAU_WINDOWS
      if (TheSafeToDumpData()) {
	if (!RtsLayer::isCtorDtor(ThisFunction->GetName())) {
	  // Not a destructor of a static object - its a function like main
	  DEBUGPROFMSG("nct " << RtsLayer::myNode() << "," 
		       << RtsLayer::myContext() << "," << tid  << " "
		       << "Profiler::Stop() : Reached top level function: dumping data"
		       << ThisFunction->GetName() <<endl;);
	  
	  StoreData(tid);
	  
#if defined(TAUKTAU) 
	  //AN Removed - New func inside 
	  //ThisKtauProfiler->KernProf.DumpKProfile();
	  ThisKtauProfiler->KernProf.DumpKProfileOut();
#endif /*TAUKTAU */

#ifdef TAU_OPENMP /* Check if we need to shut off .TAU applications on other tids */
	  if (tid == 0) {
	    int i; 
	    for (i = 1; i < TAU_MAX_THREADS; i++) {  
	      /* for all other threads */
	      Profiler *cp = CurrentProfiler[i];
	      if (cp && strncmp(cp->ThisFunction->GetName(),".TAU", 4) == 0) {
		bool uselasttimestamp = true;
		cp->Stop(i,uselasttimestamp); /* force it to write the data*/
	      }
	    }
	  }
	      
#endif /* TAU_OPENMP */
	      
	}
	// dump data here. Dump it only at the exit of top level profiler.
      }
    }
  } else { // if TheProfileMask() 
    /* set current profiler properly */
    CurrentProfiler[tid] = ParentProfiler; 
  }
}

//////////////////////////////////////////////////////////////////////

Profiler::~Profiler() {
  if (!StartStopUsed_) {
    Stop();
  } // If ctor dtor interface is used then call Stop. 
  // If the Profiler object is going out of scope without Stop being
  // called, call it now!
#if defined(TAUKTAU)
  KtauProfiler::PutKtauProfiler();
#endif /* TAUKTAU */
}

//////////////////////////////////////////////////////////////////////

void Profiler::ProfileExit(const char *message, int tid) {
  Profiler *current;
  
  current = CurrentProfiler[tid];
  
  DEBUGPROFMSG("nct "<< RtsLayer::myNode() << " RtsLayer::ProfileExit called :"
	       << message << endl;);
  if (current == 0) {   
    DEBUGPROFMSG("Current is NULL, No need to store data TID = " << tid << endl;);
    //StoreData(tid);
  } else {  
    while (current != NULL) {
      DEBUGPROFMSG("Thr "<< RtsLayer::myNode() << " ProfileExit() calling Stop:"
		   << current->ThisFunction->GetName() << " " 
		   << current->ThisFunction->GetType() << endl;);
      current->Stop(tid); // clean up 
      
      if (current->ParentProfiler == 0) {
        if (!RtsLayer::isCtorDtor(current->ThisFunction->GetName())) {
	  // Not a destructor of a static object - its a function like main
	  DEBUGPROFMSG("Thr " << RtsLayer::myNode()
		       << " ProfileExit() : Reached top level function - dumping data"
		       << endl;);
        }
      }
      
#if defined(TAUKTAU)
      KtauProfiler::PutKtauProfiler();
#endif /* TAUKTAU */
      
      current = CurrentProfiler[tid]; // Stop should set it
    }
  }

#ifdef RENCI_STFF  
  RenciSTFF::cleanup();
#endif // RENCI_STFF  
}

//////////////////////////////////////////////////////////////////////

void Profiler::theFunctionList(const char ***inPtr, int *numFuncs, bool addName, const char * inString) {
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

void Profiler::dumpFunctionNames() {

  int numFuncs;
  const char ** functionList;

  Profiler::theFunctionList(&functionList, &numFuncs);

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

void Profiler::getUserEventList(const char ***inPtr, int *numUserEvents) {

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


void Profiler::getUserEventValues(const char **inUserEvents, int numUserEvents,
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

void Profiler::theCounterList(const char ***inPtr, int *numCounters) {
  *inPtr = (const char **) malloc(sizeof(const char **) * 1);
  const char *tmpChar = "default counter";
  (*inPtr)[0] = tmpChar;
  *numCounters = 1;
}

static bool helperIsFunction(FunctionInfo *fi, Profiler *profiler) {
#ifdef TAU_CALLPATH
  if (fi == profiler->ThisFunction || fi == profiler->CallPathFunction) {
    return true;
  }
#else
  if (fi == profiler->ThisFunction) { 
    return true;
  }
#endif
  return false;
}

void Profiler::getFunctionValues(const char **inFuncs,
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
  Profiler::theCounterList(&tmpCounterList,
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

  updateIntermediateStatistics(tid);

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

#ifdef TRACING_ON
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
  TraceEvClose(tid);
  RtsLayer::DumpEDF(tid);
  RtsLayer::MergeAndConvertTracesIfNecessary();
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 
}

void Profiler::PurgeData(int tid) {
  
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;
  Profiler *curr;
  
  DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
  RtsLayer::LockDB();
  
  // Reset The Function Database (save callstack entries)
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    // May be able to recycle fns which never get called again??
    (*it)->SetCalls(tid,0);
    (*it)->SetSubrs(tid,0);
    (*it)->SetExclTimeZero(tid);
    (*it)->SetInclTimeZero(tid);
  }
  // Now Re-register callstack entries
  curr = CurrentProfiler[tid];
  curr->ThisFunction->IncrNumCalls(tid);
  curr = curr->ParentProfiler;
  while (curr != 0) {
    curr->ThisFunction->IncrNumCalls(tid);
    curr->ThisFunction->IncrNumSubrs(tid);
    curr = curr->ParentProfiler;
  }
  
  // Reset the Event Database
  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    (*eit)->LastValueRecorded[tid] = 0;
    (*eit)->NumEvents[tid] = 0L;
    (*eit)->MinValue[tid] = 9999999;
    (*eit)->MaxValue[tid] = -9999999;
    (*eit)->SumSqrValue[tid] = 0;
    (*eit)->SumValue[tid] = 0;
  }
  
  RtsLayer::UnLockDB();
}


#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
int Profiler::ExcludeTimeThisCall(double t) {
  ExclTimeThisCall -= t;
  return 1;
}
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

/////////////////////////////////////////////////////////////////////////

#ifdef PROFILE_CALLSTACK

//////////////////////////////////////////////////////////////////////
//  Profiler::CallStackTrace()
//
//  Author:  Mike Kaufman
//           mikek@cs.uoregon.edu
//  output stack of active Profiler objects
//////////////////////////////////////////////////////////////////////
void Profiler::CallStackTrace(int tid) {
  char      *dirname;            // directory name of output file
  char      fname[1024];         // output file name 
  char      errormsg[1024];      // error message buffer
  FILE      *fp;
  Profiler  *curr;               // current Profiler object in stack traversal
  double    now;                 // current wallclock time 
  double    totalTime;           // now - profiler's start time
  double    prevTotalTime;       // inclusive time of last Profiler object 
  //   stack
  static int ncalls = 0;         // number of times CallStackTrace()
  //   has been called
  
  // get wallclock time
  now = RtsLayer::getUSecD(tid);  
  
  DEBUGPROFMSG("CallStackTrace started at " << now << endl;);
  
  // increment num of calls to trace
  ncalls++;

  // set up output file
  dirname = TauEnv_get_profiledir();
  
  // create file name string
  sprintf(fname, "%s/callstack.%d.%d.%d", dirname, RtsLayer::myNode(),
	  RtsLayer::myContext(), tid);
  
  // traverse stack and set all FunctionInfo's *_cs fields to zero
  curr = CurrentProfiler[tid];
  while (curr != 0) {
    curr->ThisFunction->ExclTime_cs = curr->ThisFunction->GetExclTime(tid);
    curr = curr->ParentProfiler;
  }  

  prevTotalTime = 0;
  // calculate time info
  curr = CurrentProfiler[tid];
  while (curr != 0 ) {
    totalTime = now - curr->StartTime;
 
    // set profiler's inclusive time
    curr->InclTime_cs = totalTime;

    // calc Profiler's exclusive time
    curr->ExclTime_cs = totalTime + curr->ExclTimeThisCall
      - prevTotalTime;
     
    if (curr->AddInclFlag == true) {
      // calculate inclusive time for profiler's FunctionInfo
      curr->ThisFunction->InclTime_cs = curr->ThisFunction->GetInclTime(tid)  
	+ totalTime;
    }
    
    // calculate exclusive time for each profiler's FunctionInfo
    curr->ThisFunction->ExclTime_cs += totalTime - prevTotalTime;
    
    // keep total of inclusive time
    prevTotalTime = totalTime;
    
    // next profiler
    curr = curr->ParentProfiler;
  }
 
  // open file
  if (ncalls == 1) {
    fp = fopen(fname, "w+");
  } else {
    fp = fopen(fname, "a");
  }
  if (fp == NULL) {
    // error opening file
    sprintf(errormsg, "Error:  Could not create %s", fname);
    perror(errormsg);
    return;
  }

  if (ncalls == 1) {
    fprintf(fp,"%s%s","# Name Type Calls Subrs Prof-Incl ",
	    "Prof-Excl Func-Incl Func-Excl\n");
    fprintf(fp, 
	    "# -------------------------------------------------------------\n");
  } else {
    fprintf(fp, "\n");
  }

  // output time of callstack dump
  fprintf(fp, "%.16G\n", now);
  // output call stack info
  curr = CurrentProfiler[tid];
  while (curr != 0) {
    fprintf(fp, "\"%s %s\" %ld %ld %.16G %.16G %.16G %.16G\n",
	    curr->ThisFunction->GetName(),  curr->ThisFunction->GetType(),
	    curr->ThisFunction->GetCalls(tid),curr->ThisFunction->GetSubrs(tid),
	    curr->InclTime_cs, curr->ExclTime_cs,
	    curr->ThisFunction->InclTime_cs, curr->ThisFunction->ExclTime_cs);
    curr = curr->ParentProfiler;
  } 

  // close file
  fclose(fp);
}
#endif //PROFILE_CALLSTACK
/*-----------------------------------------------------------------*/

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

#ifdef TAU_DEPTH_LIMIT 
//////////////////////////////////////////////////////////////////////
//  Profiler::GetDepthLimit(void)
//  Description: GetDepthLimit returns the callstack depth beyond which
//               all instrumentation is disabled
//////////////////////////////////////////////////////////////////////
int Profiler::GetDepthLimit(void) {
  return profiledepth;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::SetDepthLimit(int value)
//  Description: SetDepthLimit sets the callstack instrumentation depth
//////////////////////////////////////////////////////////////////////
void Profiler::SetDepthLimit(int value) {
  profiledepth = value;
}
#endif /* TAU_DEPTH_LIMIT */ 


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
  
#ifdef PROFILE_STATS
  strcat(header,"SumExclSqr ");
#endif
  strcat(header,"ProfileCalls");
  fprintf(fp, "%s", header);	
  return 0;
}


// This is a very important function, it must be called before writing function data to disk.
// This function fills in the values that will be dumped to disk.
// It performs the calculations for timers that are still on the stack.
int Profiler::updateIntermediateStatistics(int tid) {
  
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
      
      for (Profiler *current = Profiler::CurrentProfiler[tid]; current != 0; current = current->ParentProfiler) {
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
    
#ifdef PROFILE_STATS 
    fprintf(fp,"%.16G ", fi->GetSumExclSqr(tid));
#endif
    
#ifdef PROFILE_CALLS
    long listSize = (long) fi->ExclInclCallList->size(); 
    long numCalls = fi->GetCalls(tid);
    fprintf(fp,"%ld\n", listSize); // number of records to follow
    list<pair<double,double> >::iterator iter;
    for (iter = fi->ExclInclCallList->begin(); iter != fi->ExclInclCallList->end(); iter++) {
      fprintf(fp,"%G %G\n", (*iter).first , (*iter).second);
    }
#else
    fprintf(fp,"0 "); // Indicating that profile calls is turned off
    fprintf(fp,"GROUP=\"%s\" \n", fi->GetAllGroups());
#endif
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
int Profiler::StoreData(int tid) {

#ifdef TRACING_ON
  finalizeTrace(tid);
#endif // TRACING_ON 

  Snapshot("final", true, tid);

  if (TauEnv_get_profile_format() == TAU_FORMAT_PROFILE) {
    DumpData(false, tid, "profile");
  }
  return 1;
} 


// Returns directory name for the location of a particular metric
static int getProfileLocation(int metric, char *str) {
  const char *profiledir = TauEnv_get_profiledir();
#ifdef TAU_MULTIPLE_COUNTERS
  char *metricName = MultipleCounterLayer::getCounterNameAt(metric);
  sprintf (str, "%s/MULTI__%s", profiledir, metricName);
#else
  sprintf (str, "%s", profiledir);
#endif
  return 0;
}


int Profiler::DumpData(bool increment, int tid, const char *prefix) {
  return writeData(tid, prefix, increment);
}


void getMetricHeader(int i, char *header) {
#ifdef TAU_MULTIPLE_COUNTERS
  sprintf(header, "templated_functions_MULTI_%s", RtsLayer::getCounterName(i));
#else
  sprintf(header, "%s", TauGetCounterString());
#endif
}


// Stores profile data
int Profiler::writeData(int tid, const char *prefix, bool increment, const char **inFuncs, int numFuncs) {
  
  updateIntermediateStatistics(tid);


#ifdef PROFILING_ON 
  RtsLayer::LockDB();

  static bool createFlag = createDirectories();

  for (int i=0;i<MAX_TAU_COUNTERS;i++) {
    if (RtsLayer::getCounterUsed(i)) {
      
      char metricHeader[1024];
      char profileLocation[1024];
      char filename[1024];
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

	if ((fp = fopen (filename, "w+")) == NULL) {
	  char errormsg[1024];
	  sprintf(errormsg,"Error: Could not create %s",filename);
	  perror(errormsg);
	  return 0;
	}

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
	    sprintf(errormsg,"Error: Could not create %s",filename);
	    perror(errormsg);
	    return 0;
	  }

	} else {
	  if ((fp = fopen (dumpfile, "w+")) == NULL) {
	    char errormsg[1024];
	    sprintf(errormsg,"Error: Could not create %s",filename);
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



int Profiler::dumpFunctionValues(const char **inFuncs,
				 int numFuncs,
				 bool increment,
				 int tid, char *prefix) {
  
  TAU_PROFILE("TAU_DUMP_FUNC_VALS()", " ", TAU_IO);

  writeData(tid, prefix, increment, inFuncs, numFuncs);
  return 0;
}


bool Profiler::createDirectories() {

#ifdef TAU_MULTIPLE_COUNTERS
  static bool flag = true;
  if (flag) {
    for (int i=0;i<MAX_TAU_COUNTERS;i++) {
      if (MultipleCounterLayer::getCounterUsed(i)) {
	char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
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
 * $Revision: 1.191 $   $Date: 2008/09/30 19:03:22 $
 * POOMA_VERSION_ID: $Id: Profiler.cpp,v 1.191 2008/09/30 19:03:22 amorris Exp $ 
 ***************************************************************************/
