/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.h					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/
#ifndef PROFILER_H
#define PROFILER_H

#if (defined(TAU_WINDOWS))
#pragma warning( disable : 4786 )
#endif /* TAU_WINDOWS */

#ifdef __cplusplus 

#include <Profile/ProfileGroups.h>

#if (defined(PTHREADS) || defined(TULIPTHREADS) || defined(JAVA) || defined(TAU_WINDOWS) || defined (TAU_OPENMP))
#define TAU_MAX_THREADS 128
#else
#define TAU_MAX_THREADS 1
#endif /* PTHREADS || TULIPTHREADS || JAVA || TAU_WINDOWS*/

#include <Profile/TauAPI.h>

#if (defined(PROFILING_ON) || defined(TRACING_ON))

#include <Profile/ProfileHeaders.h>

#include <Profile/PthreadLayer.h>

#include <Profile/TulipThreadLayer.h>

#include <Profile/JavaThreadLayer.h>

#include <Profile/RtsLayer.h>


#include <Profile/FunctionInfo.h>
			  
#include <Profile/UserEvent.h>

#include <Profile/PclLayer.h>

#include <Profile/PapiLayer.h>

#include <Profile/WindowsThreadLayer.h>

//////////////////////////////////////////////////////////////////////
//
// class Profiler
//
// This class is intended to be instantiated once per function
// (or other code block to be timed) as an auto variable.
//
// It will be constructed each time the block is entered
// and destroyed when the block is exited.  The constructor
// turns on the timer, and the destructor turns it off.
//
//////////////////////////////////////////////////////////////////////
class Profiler
{
public:
	Profiler(FunctionInfo * fi, TauGroup_t ProfileGroup = TAU_DEFAULT, 
	  bool StartStop = false, int tid = RtsLayer::myThread());

	void Start(int tid = RtsLayer::myThread());
	Profiler(const Profiler& X);
	Profiler& operator= (const Profiler& X);
	// Clean up data from this invocation.
	void Stop(int tid = RtsLayer::myThread());
	~Profiler();
  	static void ProfileExit(const char *message=0, 
	  int tid = RtsLayer::myThread());
	static int StoreData(int tid = RtsLayer::myThread()); 
	static int DumpData(int tid = RtsLayer::myThread()); 
	static void PurgeData(int tid = RtsLayer::myThread());

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) ) 
	int ExcludeTimeThisCall(double t);
	double ExclTimeThisCall; // for this invocation of the function
#endif /* PROFILE_CALLS || PROFILE_STATS */

	static Profiler * CurrentProfiler[TAU_MAX_THREADS];
	double StartTime;
	FunctionInfo * ThisFunction;
	Profiler * ParentProfiler; 



#ifdef PROFILE_CALLSTACK
  	double InclTime_cs;
  	double ExclTime_cs;
  	static void CallStackTrace(int tid = RtsLayer::myThread());
#endif /* PROFILE_CALLSTACK  */

private:
	TauGroup_t MyProfileGroup_;
	bool	StartStopUsed_;
	bool 	AddInclFlag; 
	// There is a class that will do some initialization
	// of FunctionStack that can't be done with
	// just the constructor.
	//friend class ProfilerInitializer;
};


#endif /* PROFILING_ON || TRACING_ON */
#include <Profile/TauMapping.h>
// included after class Profiler is defined.
#else /* __cplusplus */
#include <Profile/TauCAPI.h> /* For C program */
#endif /* __cplusplus */

#endif /* PROFILER_H */
/***************************************************************************
 * $RCSfile: Profiler.h,v $   $Author: sameer $
 * $Revision: 1.31 $   $Date: 2001/03/08 23:52:01 $
 * POOMA_VERSION_ID: $Id: Profiler.h,v 1.31 2001/03/08 23:52:01 sameer Exp $ 
 ***************************************************************************/
