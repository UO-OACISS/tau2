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

#ifdef __cplusplus 

#include <Profile/ProfileGroups.h>

#define TAU_MAX_THREADS 64

#include <Profile/TauAPI.h>

#if (defined(PROFILING_ON) || defined(TRACING_ON))

#include <Profile/ProfileHeaders.h>

#include <Profile/PthreadLayer.h>

#include <Profile/TulipThreadLayer.h>

#include <Profile/RtsLayer.h>


#include <Profile/FunctionInfo.h>
			  
#include <Profile/UserEvent.h>

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
	Profiler(FunctionInfo * fi, unsigned int ProfileGroup = TAU_DEFAULT, 
	  bool StartStop = false);

	void Start(void);
	Profiler(const Profiler& X);
	Profiler& operator= (const Profiler& X);
	// Clean up data from this invocation.
	void Stop(void);
	~Profiler();
  	static void ProfileExit(const char *message=0);
	int StoreData(int tid); 

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
  	static void CallStackTrace();
#endif /* PROFILE_CALLSTACK  */

private:
	unsigned int MyProfileGroup_;
	bool	StartStopUsed_;
	bool 	AddInclFlag; 
	// There is a class that will do some initialization
	// of FunctionStack that can't be done with
	// just the constructor.
	//friend class ProfilerInitializer;
};

#endif /* PROFILING_ON || TRACING_ON */
#else /* __cplusplus */
#include <Profile/TauCAPI.h> /* For C program */
#endif /* __cplusplus */

#endif /* PROFILER_H */
/***************************************************************************
 * $RCSfile: Profiler.h,v $   $Author: sameer $
 * $Revision: 1.17 $   $Date: 1998/08/14 15:33:30 $
 * POOMA_VERSION_ID: $Id: Profiler.h,v 1.17 1998/08/14 15:33:30 sameer Exp $ 
 ***************************************************************************/
