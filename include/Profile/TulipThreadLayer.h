/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TulipThreadLayer.h				  **
**	Description 	: TAU Profiling Package TulipThread Support Layer **
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

#ifndef _TULIPTHREADLAYER_H_
#define _TULIPTHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class TulipThreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef TULIPTHREADS
#include <Tulip_Mutex.h>
class TulipThreadLayer 
{ // Layer for RtsLayer to interact with pthreads 
  public:
 	
 	TulipThreadLayer () { }  // defaults
	~TulipThreadLayer () { } 

	static int RegisterThread(void); // called before any profiling code
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
	static int GetThreadId(void); 	 // gets 0..N-1 thread id
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex

  private:
	static Tulip_Mutex	   tauDBMutex;  // to protect TheFunctionDB
	
};
#endif // TULIPTHREADS 

#endif // _TULIPTHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: TulipThreadLayer.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 1998/08/18 14:29:32 $
 * POOMA_VERSION_ID: $Id: TulipThreadLayer.h,v 1.1 1998/08/18 14:29:32 sameer Exp $
 ***************************************************************************/


