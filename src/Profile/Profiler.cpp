/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.cpp					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

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
#else /* TAU_EPILOG */
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 

#ifdef RENCI_STFF
#include "Profile/RenciSTFF.h"
#endif // RENCI_STFF


int Tau_writeProfileMetaData(FILE *fp);


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
int& TauGetDepthLimit(void)
{
  static int depth = 0;
  char *depthvar; 

  if (depth == 0)
  {
    depthvar = getenv("TAU_DEPTH_LIMIT"); 
    if (depthvar == (char *) NULL)
    {
      depth = INT_MAX; 
    }
    else
    {
      depth = atoi(depthvar);
    }
  } 
	
  return depth; 
}



//////////////////////////////////////////////////////////////////////
// Shutdown routine which calls TAU's shutdown
//////////////////////////////////////////////////////////////////////
void TauAppShutdown(void)
{
  Tau_shutdown();
}
 
//////////////////////////////////////////////////////////////////////
// Get the string containing the counter name
//////////////////////////////////////////////////////////////////////
char * TauGetCounterString(void)
{
  char *header = new char[64];
#ifdef SGI_HW_COUNTERS
  return "templated_functions_hw_counters";
#elif (defined (TAU_PAPI) || defined (TAU_PCL) \
		|| defined(TAU_PAPI_WALLCLOCKTIME) \
		|| defined(TAU_PAPI_VIRTUAL))
  char *tau_env = NULL;

#ifdef TAU_PAPI
  tau_env = getenv("PAPI_EVENT");
#else  /* TAU_PAPI */
#ifdef TAU_PCL
  tau_env = getenv("PCL_EVENT");
#endif /* TAU_PCL */
#endif /* TAU_PAPI */
  if (tau_env)
  {
    sprintf(header, "templated_functions_MULTI_%s", tau_env);
    return header;
  }
  else
  {
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
void Profiler::EnableAllEventsOnCallStack(int tid, Profiler *current)
{
  /* Go up the callstack and enable all events on it */
	if (current != (Profiler *) NULL)
	{
	  DEBUGPROFMSG(RtsLayer::myNode()<<" This func = "<<current->ThisFunction->GetName()<<" RecordEvent = "<<current->RecordEvent<<endl;);
	  if (!current->RecordEvent)
	  { 
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

void Profiler::Start(int tid)
{ 
//   fprintf (stderr, "[%d:%d-%d] Profiler::Start for %s\n", getpid(), gettid(), tid, ThisFunction->GetName());
      ParentProfiler = CurrentProfiler[tid]; // Timers
#ifdef TAU_DEPTH_LIMIT
      int userspecifieddepth = TauGetDepthLimit();
      if (ParentProfiler)
      {
	SetDepthLimit(ParentProfiler->GetDepthLimit()+1);
      }
      else
      {
	SetDepthLimit(1);
      }
      int mydepth = GetDepthLimit();
      DEBUGPROFMSG("Start: Name: "<< ThisFunction->GetName()<<" mydepth = "<<mydepth<<", userspecifieddepth = "<<userspecifieddepth<<endl;);
      if (mydepth > userspecifieddepth) 
      { /* set the profiler */
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
	for(int i=0;i<MAX_TAU_COUNTERS;i++){
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
	if (MyProfileGroup_ & TAU_MESSAGE)
	{
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
	elg_enter(ThisFunction->GetFunctionId());
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
        if (TheTauThrottle() && (ThisFunction->GetAlreadyOnStack(tid)== false))
        {
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
	if (ParentProfiler != 0)
          ParentProfiler->ThisFunction->IncrNumSubrs(tid);	

	// Next, if this function is not already on the call stack, put it
	if (ThisFunction->GetAlreadyOnStack(tid) == false)   { 
	  AddInclFlag = true; 
	  // We need to add Inclusive time when it gets over as 
	  // it is not already on callstack.

	  ThisFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
	}
	else { // the function is already on callstack, no need to add
	       // inclusive time
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

      }  
      else
      { /* If instrumentation is disabled, set the CurrentProfiler */

	ParentProfiler = CurrentProfiler[tid] ;
	CurrentProfiler[tid] = this;
      } /* this is so Stop can access CurrentProfiler as well */
}

//////////////////////////////////////////////////////////////////////

Profiler::Profiler( FunctionInfo * function, TauGroup_t ProfileGroup, 
	bool StartStop, int tid)
{
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
  StartStopUsed_(X.StartStopUsed_)
{
#if defined(TAUKTAU)
	ThisKtauProfiler = KtauProfiler::GetKtauProfiler();
#endif /* defined(TAUKTAU) */

#ifndef TAU_MULTIPLE_COUNTERS	
  StartTime = X.StartTime;
#else //TAU_MULTIPLE_COUNTERS

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    StartTime[i] = X.StartTime[i];
  }
#endif//TAU_MULTIPLE_COUNTERS

#ifdef TAU_PROFILEPHASE
  PhaseFlag = X.PhaseFlag;
#endif /* TAU_PROFILEPHASE */ 


  DEBUGPROFMSG("Profiler::Profiler(const Profiler& X)"<<endl;);
  
  CurrentProfiler[RtsLayer::myThread()] = this;

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall = X.ExclTimeThisCall;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

}

//////////////////////////////////////////////////////////////////////

Profiler& Profiler::operator= (const Profiler& X)
{
#ifndef TAU_MULTIPLE_COUNTERS	
  StartTime = X.StartTime;
#else //TAU_MULTIPLE_COUNTERS
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
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

void Profiler::Stop(int tid, bool useLastTimeStamp)
{
//   fprintf (stderr, "[%d:%d-%d] Profiler::Stop for %s\n", getpid(), gettid(), tid, ThisFunction->GetName());
      x_uint64 TimeStamp = 0L; 
      if (CurrentProfiler[tid] == NULL) return;

#ifdef TAU_DEPTH_LIMIT
      int userspecifieddepth = TauGetDepthLimit();
      int mydepth = GetDepthLimit(); 
      if (mydepth > userspecifieddepth) 
      {
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
	if (useLastTimeStamp) /* for openmp parallel regions */
        { /* .TAU Application needs to be stopped */
#ifdef TAU_OPENMP 
          CurrentTime = TheLastTimeStamp[tid]; 
#endif /* TAU_OPENMP */
        }
	else
	{ /* use the usual mechanism */
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
	for (i=0; i < MAX_TAU_COUNTERS; i++)
	{
	  CurrentTime[i] = 0;
	}
	//Get the current counter values.
	if (useLastTimeStamp) /* for openmp parallel regions */
        { /* .TAU Application needs to be stopped */
#ifdef TAU_OPENMP 
	  for (i=0; i < MAX_TAU_COUNTERS; i++)
            CurrentTime[i] = TheLastTimeStamp[tid][i]; 
#endif /* TAU_OPENMP */
        }
	else
	{ /* use the usual mechanism */
	  RtsLayer::getUSecD(tid, CurrentTime);
        }
#ifdef TAU_OPENMP
        for (i=0; i < MAX_TAU_COUNTERS; i++)
        {
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
	for(int k=0;k<MAX_TAU_COUNTERS;k++){
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

#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
        TimeStamp = vt_pform_wtime();
        DEBUGPROFMSG("Calling vt_exit(): "<< ThisFunction->GetName()<<
		"With Timestamp = "<<TimeStamp<<endl;);
        vt_exit((uint64_t *)&TimeStamp);
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
        DEBUGPROFMSG("Calling elg_exit(): "<< ThisFunction->GetName()<<endl;);
	elg_exit();
#else /* TAU_EPILOG */
#ifdef TAU_MPITRACE
	if (RecordEvent)
	{
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
        if (TheTauThrottle() && AddInclFlag)
        {
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
            if (ParentProfiler && ParentProfiler->ProfileParamFunction)
            { /* Increment the parent's NumSubrs and decrease its exclude time */
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
	if (CallPathFunction)
	  CallPathFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
#endif /* TAU_CALLPATH */
#ifdef TAU_PROFILEPARAM
	if (ProfileParamFunction)
	  ProfileParamFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
#endif /* TAU_PROFILEPARAM */
#endif // PROFILE_CALLS

#ifdef PROFILE_STATS
	ThisFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
#ifdef TAU_CALLPATH
	if (CallPathFunction)
	  CallPathFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
#endif /* TAU_CALLPATH */
#ifdef TAU_PROFILEPARAM
	if (ProfileParamFunction)
	  ProfileParamFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
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
          <<" inclusiveTime = "<<inclusiveTime
          <<" TheTauThrottle = "<<TheTauThrottle() 
          <<" ThrCalls = "<<TheTauThrottleNumCalls()
	  <<" PerCall = " <<TheTauThrottlePerCall()<<endl;);
        if (TheTauThrottle() && (ThisFunction->GetCalls(tid) > TheTauThrottleNumCalls()) && (inclusiveTime/ThisFunction->GetCalls(tid) < TheTauThrottlePerCall()) && AddInclFlag)
	{ /* Putting AddInclFlag means we can't throttle recursive calls */
	  ThisFunction->SetProfileGroup(TAU_DISABLE, tid);
	  ThisFunction->SetPrimaryGroupName("TAU_DISABLE");
	  cout <<"TAU<"<<RtsLayer::myNode()<<">: Throttle: Disabling "<<ThisFunction->GetName()<<endl;
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
	      Snapshot("final", true, tid);

#if defined(TAUKTAU) 
	      //AN Removed - New func inside 
	      //ThisKtauProfiler->KernProf.DumpKProfile();
	      ThisKtauProfiler->KernProf.DumpKProfileOut();
#endif /*TAUKTAU */

#ifdef TAU_OPENMP /* Check if we need to shut off .TAU applications on other tids */
              if (tid == 0) 
              {
                 int i; 
                 for (i = 1; i < TAU_MAX_THREADS; i++)
		 {  /* for all other threads */
	           Profiler *cp = CurrentProfiler[i];
                   if (cp && strncmp(cp->ThisFunction->GetName(),".TAU", 4) == 0)
		   {
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

      } // if TheProfileMask() 
      else 
      { /* set current profiler properly */
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

void Profiler::ProfileExit(const char *message, int tid)
{
  Profiler *current;

  current = CurrentProfiler[tid];

  DEBUGPROFMSG("nct "<< RtsLayer::myNode() << " RtsLayer::ProfileExit called :"
    << message << endl;);
  if (current == 0) 
  {   
     DEBUGPROFMSG("Current is NULL, No need to store data TID = " << tid << endl;);
     //StoreData(tid);
  }
  else 
  {  
    while (current != 0) {
      DEBUGPROFMSG("Thr "<< RtsLayer::myNode() << " ProfileExit() calling Stop:"        << current->ThisFunction->GetName() << " " 
        << current->ThisFunction->GetType() << endl;);
      current->Stop(tid); // clean up 
  
      if (current->ParentProfiler == 0) {
        if (!RtsLayer::isCtorDtor(current->ThisFunction->GetName())) {
         // Not a destructor of a static object - its a function like main
           DEBUGPROFMSG("Thr " << RtsLayer::myNode()
             << " ProfileExit() : Reached top level function - dumping data"
             << endl;);
  
        //    StoreData(tid); // static now. Don't need current. 
        // The above Stop should call StoreData. We needn't do it again.
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

void Profiler::theFunctionList(const char ***inPtr, int *numOfFunctions, bool addName, const char * inString)
{
  //static const char *const functionList[START_SIZE];
  static int numberOfFunctions = 0;

  if(addName){
    numberOfFunctions++;
  }
  else{
    //We do not want to pass back internal pointers.
    *inPtr = ( char const **) malloc( sizeof(char *) * numberOfFunctions);

    for(int i=0;i<numberOfFunctions;i++)
    {
	    /*
      (*inPtr)[i] = functionList[i]; //Need the () in (*inPtr)[i] or the dereferrencing is
    //screwed up!
	    */
      (*inPtr)[i] = TheFunctionDB()[i]->GetName(); //Need the () in (*inPtr)[i] or the dereferrencing is
    }

    *numOfFunctions = numberOfFunctions;
  }
}

void Profiler::dumpFunctionNames()
{
  char *filename, *dumpfile, *errormsg;
  char *dirname;
  FILE* fp;

  int numOfFunctions;
  const char ** functionList;

  Profiler::theFunctionList(&functionList, &numOfFunctions);

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  //Create temp write to file.
  filename = new char[1024];
  sprintf(filename,"%s/temp.%d.%d.%d",dirname, RtsLayer::myNode(),
	  RtsLayer::myContext(), RtsLayer::myThread());
  if ((fp = fopen (filename, "w+")) == NULL) {
    errormsg = new char[1024];
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    return;
  }

  //Write data, and close.
  fprintf(fp, "number of functions %d\n", numOfFunctions);
  for(int i =0;i<numOfFunctions;i++){
    fprintf(fp, "%s\n", functionList[i]);
  }
  fclose(fp);
  
  //Rename from the temp filename.
  dumpfile = new char[1024];
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
      if ((inUserEvents != 0) && (strcmp(inUserEvents[i], (*eit)->GetEventName()) == 0)){
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


#ifndef TAU_MULTIPLE_COUNTERS
void Profiler::theCounterList(const char ***inPtr, int *numOfCounters)
{
  *inPtr = ( char const **) malloc( sizeof(char *) * 1);
  char *tmpChar = "default counter";
  (*inPtr)[0] = tmpChar; //Need the () in (*inPtr)[j] or the dereferrencing is
  //screwed up!
  *numOfCounters = 1;
}

void Profiler::getFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 double ***counterExclusiveValues,
				 double ***counterInclusiveValues,
				 int **numOfCalls,
				 int **numOfSubRoutines,
				 const char ***counterNames,
				 int *numOfCounters,
				 int tid)
{
  TAU_PROFILE("TAU_GET_FUNCTION_VALUES()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  
  bool functionCheck = false;
  int currentFuncPos = -1;
  const char *tmpFunctionName = NULL;

  int tmpNumberOfCounters;
  const char ** tmpCounterList;

  Profiler::theCounterList(&tmpCounterList,
			   &tmpNumberOfCounters);

  *numOfCounters = tmpNumberOfCounters;
  *counterNames = tmpCounterList;

  //Allocate memory for the lists.
  *counterExclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  *counterInclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  for(int memAlloc=0;memAlloc<numOfFuncs;memAlloc++){
    (*counterExclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * 1);
    (*counterInclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * 1);
  }
  *numOfCalls = (int *) malloc(sizeof(int) * numOfFuncs);
  *numOfSubRoutines = (int *) malloc(sizeof(int) * numOfFuncs);

  double tmpDoubleExcl;
  double tmpDoubleIncl;

  double currenttime = 0;
  double prevtime = 0;
  double total = 0;

  currenttime = RtsLayer::getUSecD(tid);

  RtsLayer::LockDB();
  
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
    //Check to see that it is one of the requested functions.
    functionCheck = false;
    currentFuncPos = -1;
    tmpFunctionName = (*it)->GetName();
    for(int fc=0;fc<numOfFuncs;fc++){
      if((inFuncs != 0) && (strcmp(inFuncs[fc], tmpFunctionName) == 0)){
	functionCheck = true;
	currentFuncPos = fc;
	break;
      }
    }

    if(functionCheck){
      if ((*it)->GetAlreadyOnStack(tid)){
	/* it is on the callstack. We need to do some processing. */
	/* Calculate excltime, incltime */
	Profiler *current;
	/* Traverse the Callstack */
	current = CurrentProfiler[tid];
	
	if (current == 0){ /* current is null */
	  DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	}
	else{ /* current is not null */
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);
	  
	  //Initialize what gets added for
	  //reducing from the parent profile
	  prevtime = 0;
	  total = 0;
	  
	  while (current != 0){
	    /* Traverse the stack */ 
#ifdef TAU_CALLPATH
	    if ((*it) == current->ThisFunction || (*it) == current->CallPathFunction) {
#else
	    if ((*it) == current->ThisFunction) { 
#endif
	      /* Match! */
	      DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			   <<endl;);
	      total = currenttime - current->StartTime;
	      tmpDoubleExcl += total - prevtime;
	      /* prevtime is the inclusive time of the subroutine that should
		 be subtracted from the current exclusive time */ 
	      /* If there is no instance of this function higher on the 	
		 callstack, we should add the total to the inclusive time */
	    }
	    prevtime = currenttime - current->StartTime;  
	    
	    /* to calculate exclusive time */
	    current = current->ParentProfiler; 
	  } /* We've reached the top! */
	  tmpDoubleIncl += total;//add this to the inclusive time
	  //prevtime and incltime are calculated
	} /* Current is not null */
      } /* On call stack */
      else{ /* it is not on the callstack. */
	tmpDoubleExcl = (*it)->GetExclTime(tid);
	tmpDoubleIncl = (*it)->GetInclTime(tid);
      }// Not on the Callstack

      //Copy the data.
      (*numOfCalls)[currentFuncPos] = (*it)->GetCalls(tid);
      (*numOfSubRoutines)[currentFuncPos] = (*it)->GetSubrs(tid);
      
      (*counterInclusiveValues)[currentFuncPos][0] = tmpDoubleIncl;
      (*counterExclusiveValues)[currentFuncPos][0] = tmpDoubleExcl;
    }
  }
  RtsLayer::UnLockDB();
#endif //PROFILING_ON
}

int Profiler::dumpFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 bool increment,
				 int tid, char *prefix){
  
  TAU_PROFILE("GET_FUNC_VALS()", " ", TAU_IO);
#ifdef PROFILING_ON
	vector<FunctionInfo*>::iterator it;
  	vector<TauUserEvent*>::iterator eit;
	char *filename, *dumpfile, *errormsg, *header;
	char *dirname;
	FILE* fp;
 	int numFunc = numOfFuncs; 
	int numEvents;

	bool functionCheck = false;
	const char *tmpFunctionName = NULL;

#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS
 	double excltime, incltime; 
	double currenttime, prevtime, total;

	DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);

#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
	DEBUGPROFMSG("Calling vt_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  vt_close();
#else /* VAMPIRTRACE */
#ifdef TAU_EPILOG 
	DEBUGPROFMSG("Calling elg_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  elg_close();
#else /* TAU_EPILOG */
	TraceEvClose(tid);
	RtsLayer::DumpEDF(tid);
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	currenttime = RtsLayer::getUSecD(tid); 
	RtsLayer::LockDB();
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	filename = new char[1024];
	sprintf(filename,"%s/temp.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	DEBUGPROFMSG("Creating " << filename << endl;);
	/* Changed: TRUNCATE dump file */ 
	if ((fp = fopen (filename, "w+")) == NULL) {
	 	errormsg = new char[1024];
		sprintf(errormsg,"Error: Could not create %s",filename);
		perror(errormsg);
		return 0;
	}

	// Data format :
	// %d templated_functions
	// "%s %s" %ld %G %G  
	//  funcname type numcalls Excl Incl
	// %d aggregates
	// <aggregate info>
       
	// Recalculate number of funcs using ProfileGroup. Static objects 
        // constructed before setting Profile Groups have entries in FuncDB 
	// (TAU_DEFAULT) even if they are not supposed to be there.
	/*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	    numFunc++;
	  }
	}
	*/
	//numFunc = TheFunctionDB().size();
	header = new char[256];

	sprintf(header,"%d %s\n", numFunc, TauGetCounterString());

	// Send out the format string
	strcat(header,"# Name Calls Subrs Excl Incl ");
#ifdef PROFILE_STATS
	strcat(header,"SumExclSqr ");
#endif //PROFILE_STATS
	strcat(header,"ProfileCalls");

	int sz = strlen(header);
	int ret = fprintf(fp, "%s",header);	
	fprintf(fp, " # ");	
	Tau_writeProfileMetaData(fp);
	fprintf(fp, "\n");	

	ret = fflush(fp);
	/*
	if (ret != sz) {
	  cout <<"ret not equal to strlen "<<endl;
 	}
        cout <<"Header: "<< tid << " : bytes " <<ret <<":"<<header ;
	*/
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
          /* if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask())
	     { 
	  */
	  
	  //Check to see that it is one of the requested functions.
	  functionCheck = false;
	  tmpFunctionName = (*it)->GetName();
	  for(int fc=0;fc<numOfFuncs;fc++){
            if((inFuncs != 0) && (strcmp(inFuncs[fc], tmpFunctionName) == 0)){
	      functionCheck = true;
	      break;
	    }
	  }
	  if(functionCheck){
	    if ((*it)->GetAlreadyOnStack(tid)) { 
	      /* it is on the callstack. We need to do some processing. */
	      /* Calculate excltime, incltime */
	      Profiler *current; 
	      /* Traverse the Callstack */
	      current = CurrentProfiler[tid];
	      
	      if (current == 0){ /* current is null */
		DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	      }
	      else{ /* current is not null */
		incltime = (*it)->GetInclTime(tid); 
		excltime = (*it)->GetExclTime(tid); 
		total = 0;  /* Initialize what gets added */
		prevtime = 0; /* for reducing from the parent profiler */
		while (current != 0){
		  /* Traverse the stack */ 

#ifdef TAU_CALLPATH
		  if ((*it) == current->ThisFunction || (*it) == current->CallPathFunction) {
#else
		  if ((*it) == current->ThisFunction) { 
#endif
		    /* Match! */
		    
		    DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
				 <<endl;);
		    total = currenttime - current->StartTime; 
		    excltime += total - prevtime; 
		    /* prevtime is the inclusive time of the subroutine that should
		       be subtracted from the current exclusive time */ 
		    /* If there is no instance of this function higher on the 	
		       callstack, we should add the total to the inclusive time */
		  }
		  prevtime = currenttime - current->StartTime;  
		  /* to calculate exclusive time */
		  
		  current = current->ParentProfiler; 
		} /* We've reached the top! */
		incltime += total; /* add this to the inclusive time */ 
		/* prevtime and incltime are calculated */
	      } /* Current is not null */
	    } /* On call stack */
	    else{ /* it is not on the callstack. */ 
	      excltime = (*it)->GetExclTime(tid);
	      incltime = (*it)->GetInclTime(tid); 
	    } // Not on the Callstack
	    
	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
			 << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
			 << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid) 
			 << " Excl : " << excltime << " Incl : " << incltime << endl;);
	    
	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
		    (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
		    excltime, incltime); 
	    
	    fprintf(fp,"0 "); // Indicating - profile calls is turned off
	    fprintf(fp,"GROUP=\"%s\" \n", (*it)->GetAllGroups());
	    /*
	      } // ProfileGroup test 
	    */
	  }
	} // for loop. End of FunctionInfo data
	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
	// Change this when aggregate profiling in introduced in Pooma 
	
	// Print UserEvent Data if any
	
	numEvents = 0;
 	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	  {
	    if ((*eit)->GetNumEvents(tid)) { 
	      numEvents++;
	    }
	  }
	
	if (numEvents > 0) {
	  // Data format 
	  // # % userevents
	  // # name numsamples max min mean sumsqr 
    	  fprintf(fp, "%d userevents\n", numEvents);
    	  fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	  
    	  vector<TauUserEvent*>::iterator it;
    	  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
	  {
	      
	    if ((*it)->GetNumEvents(tid)) 
            { 
	      DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
			   (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid) 
			   << "\n Max " << (*it)->GetMax(tid) << "\n Mean " 
			   << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid) 
			   << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	      
	      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
		      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
		      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
            }
	  }
	}
	// End of userevents data 
	
	RtsLayer::UnLockDB();
	fclose(fp);
	dumpfile = new char[1024];
	if(increment){
	  //Place the date and time to the dumpfile name:
	  time_t theTime = time(NULL);
	  char *stringTime = ctime(&theTime);
	  tm *structTime = localtime(&theTime);
	  char *day = strtok(stringTime," ");
	  char *month = strtok(NULL," ");
	  char *dayInt = strtok(NULL," ");
	  char *time = strtok(NULL," ");
	  char *year = strtok(NULL," ");
	  //Get rid of the mewline.
	  year[4] = '\0';
	  char *newStringTime = new char[1024];
	  sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	  
	  sprintf(dumpfile,"%s/sel_%s__%s__.%d.%d.%d",dirname, prefix,
		  newStringTime,
		  RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	}
	else{
	  sprintf(dumpfile,"%s/%s.%d.%d.%d",dirname, prefix, RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	} 
	
	
#endif //PROFILING_ON
	return 1;
}


// This is ::StoreData for only a single metric
int Profiler::StoreData(int tid)
{
#ifdef PROFILING_ON 
	vector<FunctionInfo*>::iterator it;
  	vector<TauUserEvent*>::iterator eit;
	char *filename, *errormsg, *header;
	char *dirname;
	FILE* fp;
 	int numFunc, numEvents;
#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	long listSize, numCalls;
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS
#ifdef TAUKTAU
	KtauProfiler* CurrentKtauProfiler = KtauProfiler::GetKtauProfiler(tid);
#endif /* TAUKTAU */

	DEBUGPROFMSG("Profiler::StoreData( tid = "<<tid <<" ) "<<endl;);
	TauDetectMemoryLeaks();

#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
	DEBUGPROFMSG("Calling vt_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  vt_close();
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG 
	DEBUGPROFMSG("Calling elg_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  elg_close();
#else /* TAU_EPILOG */
	TraceEvClose(tid);
	RtsLayer::DumpEDF(tid);
	RtsLayer::MergeAndConvertTracesIfNecessary();
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	RtsLayer::LockDB();
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	filename = new char[1024];

	sprintf(filename,"%s/profile.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);

	DEBUGPROFMSG("Creating " << filename << endl;);
	if ((fp = fopen (filename, "w+")) == NULL) {
	 	errormsg = new char[1024];
		sprintf(errormsg,"Error: Could not create %s",filename);
		perror(errormsg);
		return 0;
	}

#ifdef TAUKTAU_MERGE
	FILE* ktau_fp = NULL;
	if((ktau_fp = KtauProfiler::OpenOutStream(dirname, RtsLayer::myNode(), RtsLayer::myContext(), tid)) == NULL) {
		return 0;
	}
#endif

	// Data format :
	// %d templated_functions
	// "%s %s" %ld %G %G  
	//  funcname type numcalls Excl Incl
	// %d aggregates
	// <aggregate info>
       
	// Recalculate number of funcs using ProfileGroup. Static objects 
        // constructed before setting Profile Groups have entries in FuncDB 
	// (TAU_DEFAULT) even if they are not supposed to be there.
	/*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	    numFunc++;
	  }
	}
	*/
	numFunc = TheFunctionDB().size();
	header = new char[256];

#if defined(TAUKTAU_MERGE)
	char* ktau_header = NULL;
	ktau_header = new char[256];
	sprintf(ktau_header,"%d %s\n", (numFunc * 11)+(CurrentKtauProfiler->KernProf.GetNumKProfileFunc()+10), 
			TauGetCounterString());
#endif /*TAUKTAU_MERGE*/
	sprintf(header,"%d %s\n", numFunc, TauGetCounterString());

	// Send out the format string
	strcat(header,"# Name Calls Subrs Excl Incl ");
#if defined(TAUKTAU_MERGE)
	// Send out the format string
	strcat(ktau_header,"# Name Calls Subrs Excl Incl ");
#endif /* TAUKTAU_MERGE */

#ifdef PROFILE_STATS
	strcat(header,"SumExclSqr ");
#endif //PROFILE_STATS
	strcat(header,"ProfileCalls");
	int sz = strlen(header);
	int ret = fprintf(fp, "%s",header);	
	fprintf(fp, " # ");	
	Tau_writeProfileMetaData(fp);
	fprintf(fp, "\n");	
	ret = fflush(fp);

#if defined(TAUKTAU_MERGE)
	strcat(ktau_header,"ProfileCalls");
	int ktau_sz = strlen(ktau_header);
	int ktau_ret = fprintf(ktau_fp, "%s",ktau_header);	
	fprintf(ktau_fp, " # ");	
	Tau_writeProfileMetaData(ktau_fp);
	fprintf(ktau_fp, "\n");	

	ktau_ret = fflush(ktau_fp);
	bool top = 1;
#endif /* TAUKTAU_MERGE */

	/*
	if (ret != sz) {
	  cout <<"ret not equal to strlen "<<endl;
 	}
        cout <<"Header: "<< tid << " : bytes " <<ret <<":"<<header ;
	*/
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
	/*
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	  */
  
  	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
  	      << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
              << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid) 
  	      << " Excl : " << (*it)->GetExclTime(tid) << " Incl : " 
  	      << (*it)->GetInclTime(tid) << endl;);
  	
  	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
  	      (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
  	      (*it)->GetExclTime(tid), (*it)->GetInclTime(tid));

#ifdef PROFILE_STATS 
  	    fprintf(fp,"%.16G ", (*it)->GetSumExclSqr(tid));
#endif //PROFILE_STATS
  
#ifdef PROFILE_CALLS
  	    listSize = (long) (*it)->ExclInclCallList->size(); 
  	    numCalls = (*it)->GetCalls(tid);
  	    // Sanity check
  	    if (listSize != numCalls) 
  	    {
  	      fprintf(fp,"0 \n"); // don't write any invocation data
  	      DEBUGPROFMSG("Error *** list (profileCalls) size mismatch size "
  	        << listSize << " numCalls " << numCalls << endl;);
  	    }
  	    else { // List is maintained correctly
  	      fprintf(fp,"%ld \n", listSize); // no of records to follow
  	      for (iter = (*it)->ExclInclCallList->begin(); 
  	        iter != (*it)->ExclInclCallList->end(); iter++)
  	      {
  	        DEBUGPROFMSG("Node: " << RtsLayer::myNode() <<" Name "
  	          << (*it)->GetName() << " " << (*it)->GetType()
  	          << " ExclThisCall : "<< (*iter).first <<" InclThisCall : " 
  	          << (*iter).second << endl; );
  	        fprintf(fp,"%G %G\n", (*iter).first , (*iter).second);
  	      }
            } // sanity check 
#else  // PROFILE_CALLS
  	    fprintf(fp,"0 "); // Indicating - profile calls is turned off
	    fprintf(fp,"GROUP=\"%s\" \n", (*it)->GetAllGroups());
#endif // PROFILE_CALLS

#ifdef TAUKTAU_MERGE
	    KtauFuncInfo* pKFunc = (*it)->GetKtauFuncInfo(tid);
	    if(pKFunc) {
		    if(top) {
			    top = 0;
			    double dZERO = 0;
			    long lZERO = 0;
			    for(int i=0; i<NO_MERGE_GRPS; i++) {
				    fprintf(ktau_fp,"\"%s\" %ld %ld %.16G %.16G ",  
				      merge_grp_name[i], (long)KtauFuncInfo::kernelGrpCalls[tid][i], lZERO, 
				      KtauFuncInfo::kernelGrpExcl[tid][i]/KTauGetMHz(), KtauFuncInfo::kernelGrpIncl[tid][i]/KTauGetMHz());
				    fprintf(ktau_fp,"0 "); // Indicating - profile calls is turned off
				    fprintf(ktau_fp,"GROUP=\"%s\" \n", "KERNEL_GROUPS | TAU_KERNEL_MERGE");
			    }

		    }
		    fprintf(ktau_fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
		      (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
		      (*it)->GetExclTime(tid) - (pKFunc->GetExclTicks(0)/KTauGetMHz()), (*it)->GetInclTime(tid));
		    fprintf(ktau_fp,"0 "); // Indicating - profile calls is turned off
		    fprintf(ktau_fp,"GROUP=\"%s\" ", (*it)->GetAllGroups());
		    fprintf(ktau_fp,"%.16G \n", pKFunc->GetExclTicks(0)/KTauGetMHz());

		    double dZERO = 0;
		    long lZERO = 0;
		    for(int i=0; i<NO_MERGE_GRPS; i++) {
			    string grp_string = string((*it)->GetAllGroups());
			    grp_string += " | KERNEL_GROUPS | TAU_KERNEL_MERGE | ";
			    grp_string += (*it)->GetName();

			    fprintf(ktau_fp,"\"%s %s => %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
			      (*it)->GetType() ,merge_grp_name[i], (long)pKFunc->GetExclCalls(i), lZERO, 
			      pKFunc->GetExclKExcl(i)/KTauGetMHz(), pKFunc->GetExclTicks(i)/KTauGetMHz());
			    fprintf(ktau_fp,"0 "); // Indicating - profile calls is turned off
			    fprintf(ktau_fp,"GROUP=\"%s\" \n", grp_string.c_str());//(*it)->GetAllGroups());
		    }

	    }
#endif /* TAUKTAU_MERGE */

	    /*
	  } // ProfileGroup test 
	  */
	} // for loop. End of FunctionInfo data

#if defined(TAUKTAU_MERGE)
	/*Merging templated_function*/
	CurrentKtauProfiler->KernProf.MergingKProfileFunc(ktau_fp);	
#endif /*TAUKTAU_MERGE*/


	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
#if defined(TAUKTAU_MERGE)
	/*Merging aggregates*/
	fprintf(ktau_fp,"0 aggregates\n"); // For now there are no aggregates
#endif /*TAUKTAU_MERGE*/
	RtsLayer::UnLockDB();
	// Change this when aggregate profiling in introduced in Pooma 

	// Print UserEvent Data if any
	
	numEvents = 0;
 	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
          if ((*eit)->GetNumEvents(tid)) { 
	    numEvents++;
	  }
	}

	if (numEvents > 0) {
    	// Data format 
    	// # % userevents
    	// # name numsamples max min mean sumsqr 
#if defined(TAUKTAU_MERGE)
    	  fprintf(ktau_fp, "%d userevents\n", numEvents+(CurrentKtauProfiler->KernProf.GetNumKProfileEvent()));
    	  fprintf(ktau_fp, "# eventname numevents max min mean sumsqr\n");
#endif /*TAUKTAU_MERGE*/
    	  fprintf(fp, "%d userevents\n", numEvents);

    	  fprintf(fp, "# eventname numevents max min mean sumsqr\n");

    	  vector<TauUserEvent*>::iterator it;
    	  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
    	  {
      
	    if ((*it)->GetNumEvents(tid)) 
            { 
	      DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
                (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid) 
                << "\n Max " << (*it)->GetMax(tid) << "\n Mean " 
	        << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid) 
	        << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);

     	      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
	        (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	        (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));

#if defined(TAUKTAU_MERGE)
     	      fprintf(ktau_fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
	        (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	        (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
#endif /*TAUKTAU_MERGE*/

            }
    	  }
#if defined(TAUKTAU_MERGE)
	/*Merging events*/
	CurrentKtauProfiler->KernProf.MergingKProfileEvent(ktau_fp);	
#endif /*TAUKTAU_MERGE*/

	}
	// End of userevents data 

	fclose(fp);
#if defined(TAUKTAU_MERGE)
	KtauProfiler::CloseOutStream(ktau_fp);
#endif /*TAUKTAU_MERGE*/

#ifdef TAUKTAU
	KtauProfiler::PutKtauProfiler(tid);
	CurrentKtauProfiler = NULL;
#endif /* TAUKTAU */

#endif //PROFILING_ON
	return 1;
}

// This is ::DumpData for only a single metric
int Profiler::DumpData(bool increment, int tid, char *prefix)
{
  	TAU_PROFILE("TAU_DB_DUMP()", " ", TAU_IO);
#ifdef PROFILING_ON
	vector<FunctionInfo*>::iterator it;
  	vector<TauUserEvent*>::iterator eit;
	char *filename, *dumpfile, *errormsg, *header;
	char *dirname;
	FILE* fp;
 	int numFunc, numEvents;
#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS
 	double excltime, incltime; 
	double currenttime, prevtime, total;

	DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);

#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
	DEBUGPROFMSG("Calling vt_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  vt_close();
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG 
	DEBUGPROFMSG("Calling elg_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  elg_close();
#else /* TAU_EPILOG */
	TraceEvFlush(tid);
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	currenttime = RtsLayer::getUSecD(tid); 
	RtsLayer::LockDB();
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	filename = new char[1024];
	sprintf(filename,"%s/temp.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	DEBUGPROFMSG("Creating " << filename << endl;);
	/* Changed: TRUNCATE dump file */ 
	if ((fp = fopen (filename, "w+")) == NULL) {
	 	errormsg = new char[1024];
		sprintf(errormsg,"Error: Could not create %s",filename);
		perror(errormsg);
		return 0;
	}

	// Data format :
	// %d templated_functions
	// "%s %s" %ld %G %G  
	//  funcname type numcalls Excl Incl
	// %d aggregates
	// <aggregate info>
       
	// Recalculate number of funcs using ProfileGroup. Static objects 
        // constructed before setting Profile Groups have entries in FuncDB 
	// (TAU_DEFAULT) even if they are not supposed to be there.
	/*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	    numFunc++;
	  }
	}
	*/
	numFunc = TheFunctionDB().size();
	header = new char[256];

	sprintf(header,"%d %s\n", numFunc, TauGetCounterString());

	// Send out the format string
	strcat(header,"# Name Calls Subrs Excl Incl ");
#ifdef PROFILE_STATS
	strcat(header,"SumExclSqr ");
#endif //PROFILE_STATS
	strcat(header,"ProfileCalls");
	int sz = strlen(header);
	int ret = fprintf(fp, "%s",header);	
	fprintf(fp, " # ");	
	Tau_writeProfileMetaData(fp);
	fprintf(fp, "\n");	

	ret = fflush(fp);
	/*
	if (ret != sz) {
	  cout <<"ret not equal to strlen "<<endl;
 	}
        cout <<"Header: "<< tid << " : bytes " <<ret <<":"<<header ;
	*/
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          /* if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask())
	  { 
	  */

	    if ((*it)->GetAlreadyOnStack(tid)) 
	    { 
	      /* it is on the callstack. We need to do some processing. */
	      /* Calculate excltime, incltime */
	      Profiler *current; 
	      /* Traverse the Callstack */
	      current = CurrentProfiler[tid];
	  
	      if (current == 0)
	      { /* current is null */
		DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	      }
	      else 
	      { /* current is not null */
		incltime = (*it)->GetInclTime(tid); 
		excltime = (*it)->GetExclTime(tid); 
		total = 0;  /* Initialize what gets added */
		prevtime = 0; /* for reducing from the parent profiler */
		while (current != 0) 
		{
		  /* Traverse the stack */ 
		  
#ifdef TAU_CALLPATH
		  if ((*it) == current->ThisFunction || (*it) == current->CallPathFunction) {
#else
		  if ((*it) == current->ThisFunction) { 
#endif
		    /* Match! */
		    DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName() << endl;);
		    
		    total = currenttime - current->StartTime; 
		    excltime += total - prevtime; 
		    /* prevtime is the inclusive time of the subroutine that should
		       be subtracted from the current exclusive time */ 
		    /* If there is no instance of this function higher on the 	
			callstack, we should add the total to the inclusive time */
		  }
        	  prevtime = currenttime - current->StartTime;  
		  /* to calculate exclusive time */

	          current = current->ParentProfiler; 
	        } /* We've reached the top! */
		incltime += total; /* add this to the inclusive time */ 
		/* prevtime and incltime are calculated */
	      } /* Current is not null */
	    } /* On call stack */
 	    else 
	    { /* it is not on the callstack. */ 
	      excltime = (*it)->GetExclTime(tid);
	      incltime = (*it)->GetInclTime(tid); 
	    } // Not on the Callstack

	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
  	      << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
              << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid) 
  	      << " Excl : " << excltime << " Incl : " << incltime << endl;);
  	
  	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
  	      (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
	      excltime, incltime); 

  	    fprintf(fp,"0 "); // Indicating - profile calls is turned off
	    fprintf(fp,"GROUP=\"%s\" \n", (*it)->GetAllGroups());
	    /*
	  } // ProfileGroup test 
	  */
	} // for loop. End of FunctionInfo data
	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
	// Change this when aggregate profiling in introduced in Pooma 

	// Print UserEvent Data if any
	
	numEvents = 0;
 	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
          if ((*eit)->GetNumEvents(tid)) { 
	    numEvents++;
	  }
	}

	if (numEvents > 0) {
    	// Data format 
    	// # % userevents
    	// # name numsamples max min mean sumsqr 
    	  fprintf(fp, "%d userevents\n", numEvents);
    	  fprintf(fp, "# eventname numevents max min mean sumsqr\n");

    	  vector<TauUserEvent*>::iterator it;
    	  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
    	  {
      
            if ((*it)->GetNumEvents(tid)) { 
	      DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
                (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid) 
                << "\n Max " << (*it)->GetMax(tid) << "\n Mean " 
	        << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid) 
	        << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);

     	      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
	      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
            }
    	  }
	}
	// End of userevents data 

	RtsLayer::UnLockDB();
	fclose(fp);
	dumpfile = new char[1024];
	if(increment){
	  //Place the date and time to the dumpfile name:
	  time_t theTime = time(NULL);
	  char *stringTime = ctime(&theTime);
	  tm *structTime = localtime(&theTime);
	  char *day = strtok(stringTime," ");
	  char *month = strtok(NULL," ");
	  char *dayInt = strtok(NULL," ");
	  char *time = strtok(NULL," ");
	  char *year = strtok(NULL," ");
	  //Get rid of the mewline.
	  year[4] = '\0';
	  char *newStringTime = new char[1024];
	  sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	  
	  sprintf(dumpfile,"%s/%s__%s__.%d.%d.%d",dirname, prefix, 
		  newStringTime,
		  RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	}
	else{
	  sprintf(dumpfile,"%s/%s.%d.%d.%d",dirname, prefix, RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	} 


#endif //PROFILING_ON
	return 1;
}

void Profiler::PurgeData(int tid)
{
	vector<FunctionInfo*>::iterator it;
	vector<TauUserEvent*>::iterator eit;
	Profiler *curr;

	DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
	RtsLayer::LockDB();

	// Reset The Function Database (save callstack entries)
	for(it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
// May be able to recycle fns which never get called again??
	    (*it)->SetCalls(tid,0);
	    (*it)->SetSubrs(tid,0);
	    (*it)->SetExclTime(tid,0);
	    (*it)->SetInclTime(tid,0);
#ifdef PROFILE_STATS
	    (*it)->SetSumExclSqr(tid,0);
#endif //PROFILE_STATS
#ifdef PROFILE_CALLS
	    (*it)->ExclInclCallList->clear();
#endif // PROFILE_CALLS
/*
	  }
*/
	}
	// Now Re-register callstack entries
	curr = CurrentProfiler[tid];
	curr->ThisFunction->IncrNumCalls(tid);
	curr = curr->ParentProfiler;
	while(curr != 0) {
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
//////////////////////////////////////////////////////////////////////
#else //TAU_MULTIPLE_COUNTERS


bool Profiler::createDirectories(){

  char *dirname;
  static bool flag = true;
  RtsLayer::LockDB();
  if (flag) {
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      char *newdirname = new char[1024];
      //char *rmdircommand = new char[1024];
      char *mkdircommand = new char[1024];
      
      if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	dirname  = new char[8];
	strcpy (dirname,".");
      }
      
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
  RtsLayer::UnLockDB();
  return true;
}

void Profiler::getFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 double ***counterExclusiveValues,
				 double ***counterInclusiveValues,
				 int **numOfCalls,
				 int **numOfSubRoutines,
				 const char ***counterNames,
				 int *numOfCounters,
				 int tid)
{
  TAU_PROFILE("TAU_GET_FUNCTION_VALUES()", " ", TAU_IO);

#ifdef PROFILING_ON

  vector<FunctionInfo*>::iterator it;

  bool functionCheck = false;
  int currentFuncPos = -1;
  const char *tmpFunctionName = NULL;
  bool memAllocated = false; //Used to help with memory cleanup.

  int tmpNumberOfCounters;
  bool * tmpCounterUsedList;
  const char ** tmpCounterList;

  MultipleCounterLayer::theCounterListInternal(&tmpCounterList,
					       &tmpNumberOfCounters,
					       &tmpCounterUsedList);

  *numOfCounters = tmpNumberOfCounters;
  *counterNames = tmpCounterList;

  //Allocate memory for the lists.
  *counterExclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  *counterInclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  for(int memAlloc=0;memAlloc<numOfFuncs;memAlloc++){
    (*counterExclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * tmpNumberOfCounters);
    (*counterInclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * tmpNumberOfCounters);
  }
  *numOfCalls = (int *) malloc(sizeof(int) * numOfFuncs);
  *numOfSubRoutines = (int *) malloc(sizeof(int) * numOfFuncs);

  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

  double currenttime[MAX_TAU_COUNTERS];
  double prevtime[MAX_TAU_COUNTERS];
  double total[MAX_TAU_COUNTERS];

  for(int a=0;a<MAX_TAU_COUNTERS;a++){
    currenttime[a]=0;
    prevtime[a]=0;
    total[a]=0;
  }

  RtsLayer::getUSecD(tid, currenttime);

  RtsLayer::LockDB();
  
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
    //Check to see that it is one of the requested functions.
    functionCheck = false;
    currentFuncPos = -1;
    tmpFunctionName = (*it)->GetName();
    for(int fc=0;fc<numOfFuncs;fc++){
      if((inFuncs != 0) && (strcmp(inFuncs[fc], tmpFunctionName) == 0)){
	functionCheck = true;
	currentFuncPos = fc;
	break;
      }
    }
    if(functionCheck){
      if ((*it)->GetAlreadyOnStack(tid)){
	/* it is on the callstack. We need to do some processing. */
	/* Calculate excltime, incltime */
	Profiler *current;
	/* Traverse the Callstack */
	current = CurrentProfiler[tid];
	
	if (current == 0){ /* current is null */
	  DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	}
	else{ /* current is not null */
	  //These calls return pointers to new memory.
	  //Remember to free this memory after use!!!
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);
	  memAllocated = true;
	  
	  //Initialize what gets added for
	  //reducing from the parent profile
	  for(int j=0;j<MAX_TAU_COUNTERS;j++){
	    prevtime[j]=0;
	    total[j]=0;
	  }
	  
	  while (current != 0){
	    /* Traverse the stack */


#ifdef TAU_CALLPATH
	      if ((*it) == current->ThisFunction || (*it) == current->CallPathFunction) {
#else
	      if ((*it) == current->ThisFunction) { 
#endif
		/* Match! */
 
	      DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			   <<endl;);
	      
	      for(int k=0;k<MAX_TAU_COUNTERS;k++){
		total[k] = currenttime[k] - current->StartTime[k];
		tmpDoubleExcl[k] += total[k] - prevtime[k];
	      }
	      /* prevtime is the inclusive time of the subroutine that should
		 be subtracted from the current exclusive time */ 
	      /* If there is no instance of this function higher on the 	
		 callstack, we should add the total to the inclusive time */
	    }
	    for(int l=0;l<MAX_TAU_COUNTERS;l++){
	      prevtime[l] = currenttime[l] - current->StartTime[l];  
	    }
	    /* to calculate exclusive time */
	    current = current->ParentProfiler; 
	  } /* We've reached the top! */
	  for(int m=0;m<MAX_TAU_COUNTERS;m++){
	    tmpDoubleIncl[m] += total[m];//add this to the inclusive time
	    //prevtime and incltime are calculated
	  }
	} /* Current is not null */
      } /* On call stack */
      else{ /* it is not on the callstack. */
	//These calls return pointers to new memory.
	//Remember to free this memory after use!!!
	tmpDoubleExcl = (*it)->GetExclTime(tid);
	tmpDoubleIncl = (*it)->GetInclTime(tid);
	memAllocated = true;
      }// Not on the Callstack

      //Copy the data.
      (*numOfCalls)[currentFuncPos] = (*it)->GetCalls(tid);
      (*numOfSubRoutines)[currentFuncPos] = (*it)->GetSubrs(tid);
      
      int posCounter = 0;
      if(memAllocated){
	for(int copyData=0;copyData<MAX_TAU_COUNTERS;copyData++){
	  if(tmpCounterUsedList[copyData]){
	    (*counterInclusiveValues)[currentFuncPos][posCounter] = tmpDoubleIncl[copyData];
	    (*counterExclusiveValues)[currentFuncPos][posCounter] = tmpDoubleExcl[copyData];
	    posCounter++;
	  }
	}
      }
      else{
	for(int copyData=0;copyData<MAX_TAU_COUNTERS;copyData++){
	  if(tmpCounterUsedList[copyData]){
	    (*counterInclusiveValues)[currentFuncPos][posCounter] = 0;
	    (*counterExclusiveValues)[currentFuncPos][posCounter] = 0;
	    posCounter++;
	  }
	}
      }
      //Free up the memory if it was allocated.
      if(memAllocated){
	free(tmpDoubleIncl);
	free(tmpDoubleExcl);
      }
    }
  }
  RtsLayer::UnLockDB();
#endif //PROFILING_ON
}

int Profiler::dumpFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 bool increment,
				 int tid, char *prefix){
  
  TAU_PROFILE("GET_FUNC_VALS()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;

  bool functionCheck = false;
  const char *tmpFunctionName = NULL;

  FILE* fp;
  char *dirname, *dumpfile;
  int numFunc = numOfFuncs; 
  int numEvents;

  bool memAllocated = false; //Used to help with memory cleanup.
  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

  double currenttime[MAX_TAU_COUNTERS];
  double prevtime[MAX_TAU_COUNTERS];
  double total[MAX_TAU_COUNTERS];

  for(int a=0;a<MAX_TAU_COUNTERS;a++){
    currenttime[a]=0;
    prevtime[a]=0;
    total[a]=0;
  }

  RtsLayer::getUSecD(tid, currenttime);

  DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);


  //Create directories for storage.
  static bool createFlag = createDirectories();

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      RtsLayer::LockDB();

      char *newdirname = new char[1024];
      char *filename = new char[1024];
      char *errormsg = new char[1024];
      char *header = new char[1024];

      sprintf(newdirname,"%s/MULTI__%s",dirname,tmpChar);

      sprintf(filename,"%s/temp.%d.%d.%d",newdirname, RtsLayer::myNode(),
	      RtsLayer::myContext(), tid);

      DEBUGPROFMSG("Creating " << filename << endl;);
      if ((fp = fopen (filename, "w+")) == NULL) {
      errormsg = new char[1024];
      sprintf(errormsg,"Error: Could not create %s",filename);
      perror(errormsg);
      return 0;
      }

      // Data format :
      // %d templated_functions
      // "%s %s" %ld %G %G  
      //  funcname type numcalls Excl Incl
      // %d aggregates
      // <aggregate info>
      
      // Recalculate number of funcs using ProfileGroup. Static objects 
      // constructed before setting Profile Groups have entries in FuncDB 
      // (TAU_DEFAULT) even if they are not supposed to be there.
      /*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
	if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	numFunc++;
	}
	}
      */

      //numFunc = TheFunctionDB().size();

      //Setting the header to the correct name.
      sprintf(header,"%d templated_functions_MULTI_%s\n", numFunc, tmpChar);
  
      strcat(header,"# Name Calls Subrs Excl Incl ");

      strcat(header,"ProfileCalls");
      int sz = strlen(header);
      int ret = fprintf(fp, "%s",header);
      fprintf(fp, " # ");	
      Tau_writeProfileMetaData(fp);
      fprintf(fp, "\n");	
      ret = fflush(fp);

      for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
	//Check to see that it is one of the requested functions.
	functionCheck = false;
	tmpFunctionName = (*it)->GetName();
	for(int fc=0;fc<numOfFuncs;fc++){
          if((inFuncs != 0) && (strcmp(inFuncs[fc], tmpFunctionName) == 0)){
	    functionCheck = true;
	    break;
	  }
	}
	if(functionCheck){
	  if ((*it)->GetAlreadyOnStack(tid)){
	    /* it is on the callstack. We need to do some processing. */
	    /* Calculate excltime, incltime */
	    Profiler *current;
	    /* Traverse the Callstack */
	    current = CurrentProfiler[tid];
	    
	    if (current == 0){ /* current is null */
	      DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	    }
	    else{ /* current is not null */
	      //These calls return pointers to new memory.
	      //Remember to free this memory after use!!!
	      tmpDoubleExcl = (*it)->GetExclTime(tid);
	      tmpDoubleIncl = (*it)->GetInclTime(tid);
	      memAllocated = true;
	      
	      //Initialize what gets added for
	      //reducing from the parent profile
	      for(int j=0;j<MAX_TAU_COUNTERS;j++){
		prevtime[j]=0;
		total[j]=0;
	      }
	      
	      while (current != 0){
		/* Traverse the stack */ 

#ifdef TAU_CALLPATH
		if ((*it) == current->ThisFunction || (*it) == current->CallPathFunction) {
#else
		if ((*it) == current->ThisFunction) { 
#endif
		  /* Match! */
		  DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			       <<endl;);
		  
		  for(int k=0;k<MAX_TAU_COUNTERS;k++){
		    total[k] = currenttime[k] - current->StartTime[k];
		    tmpDoubleExcl[k] += total[k] - prevtime[k];
		  }
		  /* prevtime is the inclusive time of the subroutine that should
		     be subtracted from the current exclusive time */ 
		  /* If there is no instance of this function higher on the 	
		     callstack, we should add the total to the inclusive time */
		}
		for(int l=0;l<MAX_TAU_COUNTERS;l++){
		  prevtime[l] = currenttime[l] - current->StartTime[l];  
		}
		/* to calculate exclusive time */
		current = current->ParentProfiler; 
	      } /* We've reached the top! */
	      for(int m=0;m<MAX_TAU_COUNTERS;m++){
		tmpDoubleIncl[m] += total[m];//add this to the inclusive time
		//prevtime and incltime are calculated
	      }
	    } /* Current is not null */
	  } /* On call stack */
	  else{ /* it is not on the callstack. */
	    //These calls return pointers to new memory.
	    //Remember to free this memory after use!!!
	    tmpDoubleExcl = (*it)->GetExclTime(tid);
	    tmpDoubleIncl = (*it)->GetInclTime(tid);
	    memAllocated = true;
	  } // Not on the Callstack
	  
	  if(memAllocated){
	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
			 << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
			 << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
			 << " Excl : " << tmpDoubleExcl[i] << " Incl : "
			 << tmpDoubleIncl[i] << endl;);
	    
	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		    (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		    tmpDoubleExcl[i], tmpDoubleIncl[i]);
	  }
	  else{
	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
			 << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
			 << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
			 << " Excl : " << 0 << " Incl : "
			 << 0 << endl;);
	    
	    fprintf(fp,"\"%s %s\" %ld %ld 0 0 ", (*it)->GetName(),
	    (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid));
	  }
	  fprintf(fp,"0 "); // Indicating - profile calls is turned off
	  fprintf(fp,"GROUP=\"%s\" \n", (*it)->GetAllGroups());
	  //Free up the memory if it was allocated.
	  if(memAllocated){
	    free(tmpDoubleIncl);
	    free(tmpDoubleExcl);
	  }
	}
      }
      fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
      
      numEvents = 0;
      for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
	  if ((*eit)->GetNumEvents(tid)) {
	    numEvents++;
	  }
	}
      
      if (numEvents > 0) {
	// Data format
	// # % userevents
	// # name numsamples max min mean sumsqr
	fprintf(fp, "%d userevents\n", numEvents);
	fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	
	vector<TauUserEvent*>::iterator it;
	for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++){
	  
	  if ((*it)->GetNumEvents(tid)) {
	    DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
              (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid)
	      << "\n Max " << (*it)->GetMax(tid) << "\n Mean "
	      << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid)
              << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	  
	    fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n",
	      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
          }
	}
      }
      
      // End of userevents data
      RtsLayer::UnLockDB();
      fclose(fp);
      
      dumpfile = new char[1024];
      if(increment){
	//Place the date and time to the dumpfile name:
	time_t theTime = time(NULL);
	char *stringTime = ctime(&theTime);
	tm *structTime = localtime(&theTime);
	char *day = strtok(stringTime," ");
	char *month = strtok(NULL," ");
	char *dayInt = strtok(NULL," ");
	char *time = strtok(NULL," ");
	char *year = strtok(NULL," ");
	//Get rid of the mewline.
	year[4] = '\0';
	char *newStringTime = new char[1024];
	sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	
	sprintf(dumpfile,"%s/sel_%s__%s__.%d.%d.%d",newdirname, prefix,
		newStringTime,
		RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	
	rename(filename, dumpfile);
      }
      else{
	sprintf(dumpfile,"%s/sel_%s.%d.%d.%d",newdirname, prefix, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	rename(filename, dumpfile);
      }
    }
  }
#endif //PROFILING_ON
  return 1;
}



// This is ::StoreData for multiple metrics
int Profiler::StoreData(int tid){
#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;
  FILE* fp;
  char *dirname;
  int numFunc, numEvents;
  
  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

#endif //PROFILING_ON

  DEBUGPROFMSG("Profiler::StoreData( tid = "<<tid <<" ) "<<endl;);
  TauDetectMemoryLeaks();

#ifdef TRACING_ON
#ifdef TAU_EPILOG 
	DEBUGPROFMSG("Calling elg_close()"<<endl;);
 	if (RtsLayer::myThread() == 0)
	  elg_close();
#else /* TAU_EPILOG */
	TraceEvClose(tid);
	RtsLayer::DumpEDF(tid);
	RtsLayer::MergeAndConvertTracesIfNecessary();
#endif /* TAU_EPILOG */
#endif // TRACING_ON

#ifdef PROFILING_ON
  
  //Create directories for storage.
  static bool createFlag = createDirectories();

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      RtsLayer::LockDB();
      
      char *newdirname = new char[1024];
      char *filename = new char[1024];
      char *errormsg = new char[1024];
      char *header = new char[1024];
      

      sprintf(newdirname,"%s/MULTI__%s",dirname,tmpChar);

      sprintf(filename,"%s/profile.%d.%d.%d",newdirname, RtsLayer::myNode(),
	      RtsLayer::myContext(), tid);

      DEBUGPROFMSG("Creating " << filename << endl;);
      if ((fp = fopen (filename, "w+")) == NULL) {
      errormsg = new char[1024];
      sprintf(errormsg,"Error: Could not create %s",filename);
      perror(errormsg);
      return 0;
      }
      
      // Data format :
      // %d templated_functions
      // "%s %s" %ld %G %G
      //  funcname type numcalls Excl Incl
      // %d aggregates
      // <aggregate info>
      
      numFunc = TheFunctionDB().size();
      
      //Setting the header to the correct name.
      sprintf(header,"%d templated_functions_MULTI_%s\n", numFunc, tmpChar);
  
      strcat(header,"# Name Calls Subrs Excl Incl ");

      strcat(header,"ProfileCalls");
      int sz = strlen(header);
      int ret = fprintf(fp, "%s",header);
      fprintf(fp, " # ");	
      Tau_writeProfileMetaData(fp);
      fprintf(fp, "\n");	
      ret = fflush(fp);

      

      for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
        {
	  //These calls return pointers to new memory.
	  //Remember to free this memory after use!!! 
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);

	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
		       << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
		       << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
		       << " Excl : " << tmpDoubleExcl[i] << " Incl : "
		       << tmpDoubleIncl[i] << endl;);
	  
	  fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		  (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		  tmpDoubleExcl[i], tmpDoubleIncl[i]);

	  fprintf(fp,"0 "); // Indicating - profile calls is turned off
	  fprintf(fp,"GROUP=\"%s\" \n", (*it)->GetAllGroups());
	  
	  //Free up the memory.
	  free(tmpDoubleIncl);
	  free(tmpDoubleExcl);
	}

      fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
  
      RtsLayer::UnLockDB();
      
      numEvents = 0;
      for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
	  if ((*eit)->GetNumEvents(tid)) {
	    numEvents++;
	  }
	}
      
      if (numEvents > 0) {
	// Data format
	// # % userevents
	// # name numsamples max min mean sumsqr
	fprintf(fp, "%d userevents\n", numEvents);
	fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	
	vector<TauUserEvent*>::iterator it;
	for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++){
	  
	  if ((*it)->GetNumEvents(tid)) {
	    DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
	      (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid)
	      << "\n Max " << (*it)->GetMax(tid) << "\n Mean "
	      << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid)
	      << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	  
	    fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n",
	      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
          }
	}
      }
      
      // End of userevents data

      fclose(fp);
    }
  }
#endif //PROFILING_ON
 
  return 1;
}

// This is ::DumpData for multiple metrics
int Profiler::DumpData(bool increment, int tid, char *prefix){

  TAU_PROFILE("TAU_DB_DUMP()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;

  FILE* fp;
  char *dirname, *dumpfile;
  int numFunc, numEvents;

  bool memAllocated = false; //Used to help with memory cleanup.
  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

  double currenttime[MAX_TAU_COUNTERS];
  double prevtime[MAX_TAU_COUNTERS];
  double total[MAX_TAU_COUNTERS];

  for(int a=0;a<MAX_TAU_COUNTERS;a++){
    currenttime[a]=0;
    prevtime[a]=0;
    total[a]=0;
  }

  RtsLayer::getUSecD(tid, currenttime);

  DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);


  //Create directories for storage.
  static bool createFlag = createDirectories();

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      RtsLayer::LockDB();

      char *newdirname = new char[1024];
      char *filename = new char[1024];
      char *errormsg = new char[1024];
      char *header = new char[1024];

      sprintf(newdirname,"%s/MULTI__%s",dirname,tmpChar);

      sprintf(filename,"%s/temp.%d.%d.%d",newdirname, RtsLayer::myNode(),
	      RtsLayer::myContext(), tid);

      DEBUGPROFMSG("Creating " << filename << endl;);
      if ((fp = fopen (filename, "w+")) == NULL) {
	errormsg = new char[1024];
	sprintf(errormsg,"Error: Could not create %s",filename);
	perror(errormsg);
	return 0;
      }

      // Data format :
      // %d templated_functions
      // "%s %s" %ld %G %G  
      //  funcname type numcalls Excl Incl
      // %d aggregates
      // <aggregate info>
      
      // Recalculate number of funcs using ProfileGroup. Static objects 
      // constructed before setting Profile Groups have entries in FuncDB 
      // (TAU_DEFAULT) even if they are not supposed to be there.
      /*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
	if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	numFunc++;
	}
	}
      */

      numFunc = TheFunctionDB().size();

      //Setting the header to the correct name.
      sprintf(header,"%d templated_functions_MULTI_%s\n", numFunc, tmpChar);
  
      strcat(header,"# Name Calls Subrs Excl Incl ");

      strcat(header,"ProfileCalls");
      int sz = strlen(header);
      int ret = fprintf(fp, "%s",header);
      fprintf(fp, " # ");	
      Tau_writeProfileMetaData(fp);
      fprintf(fp, "\n");	
      ret = fflush(fp);

      for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
	if ((*it)->GetAlreadyOnStack(tid)){
	  /* it is on the callstack. We need to do some processing. */
	  /* Calculate excltime, incltime */
	  Profiler *current;
	  /* Traverse the Callstack */
	  current = CurrentProfiler[tid];

	  if (current == 0){ /* current is null */
	    DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	  }
	  else{ /* current is not null */
	    //These calls return pointers to new memory.
	    //Remember to free this memory after use!!!
	    tmpDoubleExcl = (*it)->GetExclTime(tid);
	    tmpDoubleIncl = (*it)->GetInclTime(tid);
	    memAllocated = true;
	    
	    //Initialize what gets added for
	    //reducing from the parent profile
	    for(int j=0;j<MAX_TAU_COUNTERS;j++){
	      prevtime[j]=0;
	      total[j]=0;
	    }

	    while (current != 0){
	      /* Traverse the stack */ 

#ifdef TAU_CALLPATH
	      if ((*it) == current->ThisFunction || (*it) == current->CallPathFunction) {
#else
		if ((*it) == current->ThisFunction) { 
#endif
		  /* Match! */
		DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			     <<endl;);

		for(int k=0;k<MAX_TAU_COUNTERS;k++){
		  total[k] = currenttime[k] - current->StartTime[k];
		  tmpDoubleExcl[k] += total[k] - prevtime[k];
		}
		/* prevtime is the inclusive time of the subroutine that should
		   be subtracted from the current exclusive time */ 
		/* If there is no instance of this function higher on the 	
		   callstack, we should add the total to the inclusive time */
	      }
	      for(int l=0;l<MAX_TAU_COUNTERS;l++){
		prevtime[l] = currenttime[l] - current->StartTime[l];  
	      }
	      /* to calculate exclusive time */
	      current = current->ParentProfiler; 
	    } /* We've reached the top! */
	    for(int m=0;m<MAX_TAU_COUNTERS;m++){
	      tmpDoubleIncl[m] += total[m];//add this to the inclusive time
	      //prevtime and incltime are calculated
	    }
	  } /* Current is not null */
	} /* On call stack */
	else{ /* it is not on the callstack. */
	  //These calls return pointers to new memory.
	  //Remember to free this memory after use!!!
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);
	  memAllocated = true;
	} // Not on the Callstack

	if(memAllocated){
	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
		       << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
		       << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
		       << " Excl : " << tmpDoubleExcl[i] << " Incl : "
		       << tmpDoubleIncl[i] << endl;);

	  fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		  (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		  tmpDoubleExcl[i], tmpDoubleIncl[i]);
	} else {
	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
		       << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
		       << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
		       << " Excl : " << 0 << " Incl : "
		       << 0 << endl;);
	  
	  fprintf(fp,"\"%s %s\" %ld %ld 0 0 ", (*it)->GetName(),
		  (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid));
	}
	fprintf(fp,"0 "); // Indicating - profile calls is turned off
	fprintf(fp,"GROUP=\"%s\" \n", (*it)->GetAllGroups());
	 //Free up the memory if it was allocated.
	if(memAllocated){
	  free(tmpDoubleIncl);
	  free(tmpDoubleExcl);
	}
      }
      fprintf(fp,"0 aggregates\n"); // For now there are no aggregates

      numEvents = 0;
      for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
	  if ((*eit)->GetNumEvents(tid)) {
	    numEvents++;
	  }
	}
      
      if (numEvents > 0) {
	// Data format
	// # % userevents
	// # name numsamples max min mean sumsqr
	fprintf(fp, "%d userevents\n", numEvents);
	fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	
	vector<TauUserEvent*>::iterator it;
	for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++){
	  	  
	  if ((*it)->GetNumEvents(tid)) {
	    DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
	      (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid)
	      << "\n Max " << (*it)->GetMax(tid) << "\n Mean "
	      << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid)
	      << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	  
	    fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n",
	      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
          }
	}
      }
      
      // End of userevents data
      RtsLayer::UnLockDB();
      fclose(fp);

      dumpfile = new char[1024];
      	if(increment){
	  //Place the date and time to the dumpfile name:
	  time_t theTime = time(NULL);
	  char *stringTime = ctime(&theTime);
	  tm *structTime = localtime(&theTime);
	  char *day = strtok(stringTime," ");
	  char *month = strtok(NULL," ");
	  char *dayInt = strtok(NULL," ");
	  char *time = strtok(NULL," ");
	  char *year = strtok(NULL," ");
	  //Get rid of the mewline.
	  year[4] = '\0';
	  char *newStringTime = new char[1024];
	  sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	  
	  sprintf(dumpfile,"%s/dump__%s__.%d.%d.%d",newdirname,
		  newStringTime,
		  RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);

	  rename(filename, dumpfile);
	}
	else{
	  sprintf(dumpfile,"%s/%s.%d.%d.%d",newdirname, prefix, RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	}
    }
  }
#elif TRACING_ON
#ifndef TAU_EPILOG
#ifndef TAU_VAMPIRTRACE
  TraceEvFlush(tid);
#endif /* TAU_VAMPIRTRACE */
#endif /* TAU_EPILOG */
#endif //PROFILING_ON
  return 1;
}

void Profiler::PurgeData(int tid){
  
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;
  Profiler *curr;
  
  DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
  RtsLayer::LockDB();
  
  // Reset The Function Database (save callstack entries)
  for(it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
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
  while(curr != 0) {
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
#endif//TAU_MULTIPLE_COUNTERS

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
int Profiler::ExcludeTimeThisCall(double t)
{
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
void Profiler::CallStackTrace(int tid)
{
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
  if ((dirname = getenv("PROFILEDIR")) == NULL)
  {
    dirname = new char[8];
    strcpy (dirname, ".");
  }
  
  // create file name string
  sprintf(fname, "%s/callstack.%d.%d.%d", dirname, RtsLayer::myNode(),
	  RtsLayer::myContext(), tid);
  
  // traverse stack and set all FunctionInfo's *_cs fields to zero
  curr = CurrentProfiler[tid];
  while (curr != 0)
  {
    curr->ThisFunction->ExclTime_cs = curr->ThisFunction->GetExclTime(tid);
    curr = curr->ParentProfiler;
  }  

  prevTotalTime = 0;
  // calculate time info
  curr = CurrentProfiler[tid];
  while (curr != 0 )
  {
    totalTime = now - curr->StartTime;
 
    // set profiler's inclusive time
    curr->InclTime_cs = totalTime;

    // calc Profiler's exclusive time
    curr->ExclTime_cs = totalTime + curr->ExclTimeThisCall
                      - prevTotalTime;
     
    if (curr->AddInclFlag == true)
    {
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
  if (ncalls == 1)
    fp = fopen(fname, "w+");
  else
    fp = fopen(fname, "a");
  if (fp == NULL)  // error opening file
  {
    sprintf(errormsg, "Error:  Could not create %s", fname);
    perror(errormsg);
    return;
  }

  if (ncalls == 1)
  {
    fprintf(fp,"%s%s","# Name Type Calls Subrs Prof-Incl ",
            "Prof-Excl Func-Incl Func-Excl\n");
    fprintf(fp, 
            "# -------------------------------------------------------------\n");
  }
  else
    fprintf(fp, "\n");

  // output time of callstack dump
  fprintf(fp, "%.16G\n", now);
  // output call stack info
  curr = CurrentProfiler[tid];
  while (curr != 0 )
  {
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
/*-----------------------------------------------------------------*/
#endif //PROFILE_CALLSTACK

#ifdef TAU_COMPENSATE
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
#endif /* TAU_COMPENSATE */

//////////////////////////////////////////////////////////////////////
//  Profiler::GetPhase(void)
//  Description: Returns if a profiler is a phase or not 
//////////////////////////////////////////////////////////////////////
#ifdef TAU_PROFILEPHASE
bool Profiler::GetPhase(void)
{
  return PhaseFlag;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::GetPhase(bool flag)
//  Description: SetPhase sets a phase to be true or false based on the 
//               parameter flag
//////////////////////////////////////////////////////////////////////
void Profiler::SetPhase(bool flag)
{
  PhaseFlag = flag;
}

#endif /* TAU_PROFILEPHASE */

#ifdef TAU_DEPTH_LIMIT 
//////////////////////////////////////////////////////////////////////
//  Profiler::GetDepthLimit(void)
//  Description: GetDepthLimit returns the callstack depth beyond which
//               all instrumentation is disabled
//////////////////////////////////////////////////////////////////////
int Profiler::GetDepthLimit(void)
{
  return profiledepth;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::SetDepthLimit(int value)
//  Description: SetDepthLimit sets the callstack instrumentation depth
//////////////////////////////////////////////////////////////////////
void Profiler::SetDepthLimit(int value)
{
  profiledepth = value;
}
#endif /* TAU_DEPTH_LIMIT */ 

//////////////////////////////////////////////////////////////////////
//  TauGetThrottle(void)
//  Description: Returns whether throttling is enabled or disabled 
//////////////////////////////////////////////////////////////////////
bool TauGetThrottle(void)
{
  if (getenv("TAU_THROTTLE") == (char *) NULL)
   return false;
  else
   return true;
}


//////////////////////////////////////////////////////////////////////
//  TauGetThrottleNumCalls(void)
//  Description: Returns (as a double) the number of calls for throttle
//               based control of instrumentation. 
//////////////////////////////////////////////////////////////////////
double TauGetThrottleNumCalls(void)
{
  char *numcalls = getenv("TAU_THROTTLE_NUMCALLS"); 
  double d = TAU_THROTTLE_NUMCALLS_DEFAULT;  /* default numcalls */
  if (numcalls)
  {
    d = strtod(numcalls,0); 
  }
  DEBUGPROFMSG("TauGetThrottleNumCalls: Returning "<<d <<" as numcalls value"<<endl;);
  return d;
}

//////////////////////////////////////////////////////////////////////
//  TauGetThrottlePerCall(void)
//  Description: Returns (as a double) the per call value for throttle
//               based control of instrumentation. 
//////////////////////////////////////////////////////////////////////
double TauGetThrottlePerCall(void)
{
  char *percall = getenv("TAU_THROTTLE_PERCALL"); 
  double d = TAU_THROTTLE_PERCALL_DEFAULT;  /* default numcalls */
  if (percall)
  {
    d = strtod(percall,0); 
  }
  DEBUGPROFMSG("TauGetThrottlePerCall: Returning "<<d <<" as per-call value"<<endl;);
  return d;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::TheTauThrottle(void)
//  Description: Returns whether throttling is enabled or disabled
//////////////////////////////////////////////////////////////////////
bool& Profiler::TheTauThrottle(void)
{
  static bool throttle = TauGetThrottle();
  return throttle;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::TauGetThrottlePerCall(void)
//  Description: Returns the number of calls for throttling
//////////////////////////////////////////////////////////////////////
double& Profiler::TheTauThrottleNumCalls(void)
{
  static double throttleNumcalls = TauGetThrottleNumCalls();
  return throttleNumcalls;
}

//////////////////////////////////////////////////////////////////////
//  Profiler::TauGetThrottlePerCall(void)
//  Description: Returns the per call value for throttling
//////////////////////////////////////////////////////////////////////
double& Profiler::TheTauThrottlePerCall(void)
{
  static double throttlePercall = TauGetThrottlePerCall();
  return throttlePercall;
}











/***************************************************************************
 * $RCSfile: Profiler.cpp,v $   $Author: amorris $
 * $Revision: 1.166 $   $Date: 2007/05/21 23:33:42 $
 * POOMA_VERSION_ID: $Id: Profiler.cpp,v 1.166 2007/05/21 23:33:42 amorris Exp $ 
 ***************************************************************************/

	





