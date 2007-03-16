/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Papi_ThreadLayer.h				  **
**	Description 	: TAU Profiling Package Papi Thread Support Layer **
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
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


#ifndef _PAPI_THREADLAYER_H_
#define _PAPI_THREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class Papi_ThreadLayer
//
// This class is used for supporting papi_threads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef TAU_PAPI_THREADS

#include <papi.h>

class PapiThreadLayer 
{ // Layer for RtsLayer to interact with papi_threads 
  public:
 	
 	PapiThreadLayer () { }  // defaults
	~PapiThreadLayer () { } 

	static int RegisterThread(void); // called before any profiling code
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
        static int InitializeEnvMutexData(void);     // init tauEnv mutex
	static int GetThreadId(void); 	 // gets 0..N-1 thread id
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex
	static int LockEnv(void);	 // locks the tauEnvMutex
	static int UnLockEnv(void);	 // unlocks the tauEnvMutex
  private:
	
};
#endif // TAU_PAPI_THREADS 

#endif // _PAPI_THREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: PapiThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.1 $   $Date: 2007/03/16 23:48:09 $
 * POOMA_VERSION_ID: $Id: PapiThreadLayer.h,v 1.1 2007/03/16 23:48:09 amorris Exp $
 ***************************************************************************/


