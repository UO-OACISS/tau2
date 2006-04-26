/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: PthreadLayer.h				  **
**	Description 	: TAU Profiling Package Pthread Support Layer	  **
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

#ifndef _PTHREADLAYER_H_
#define _PTHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class PthreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef PTHREADS

#ifndef TAU_LIBRARY_SOURCE
/* pthread_create wrapper */
#include <pthread.h>
#define pthread_create(thread, attr, function, arg) \
        tau_pthread_create(thread, attr, function, arg)

extern "C" int tau_pthread_create (pthread_t *__restrict __threadp,
			__const pthread_attr_t *__restrict __attr,
			void *(*__start_routine) (void *),
			void *__restrict __arg);
#endif /* TAU_LIBRARY_SOURCE */

#ifdef TAU_CHARM
extern "C" {
#include <cpthreads.h>
}
#else 
#include <pthread.h>
#endif

class PthreadLayer 
{ // Layer for RtsLayer to interact with pthreads 
  public:
 	
 	PthreadLayer () { }  // defaults
	~PthreadLayer () { } 

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
	static pthread_key_t 	   tauPthreadId; // tid 
	static pthread_mutex_t     tauThreadcountMutex; // to protect counter
	static pthread_mutexattr_t tauThreadcountAttr; // count attribute 
	static int 		   tauThreadCount;     // counter
	static pthread_mutex_t	   tauDBMutex;  // to protect TheFunctionDB
	static pthread_mutex_t	   tauEnvMutex;  // to protect TheFunctionDB
	static pthread_mutexattr_t tauDBAttr;   // DB mutex attribute
	
};
#endif // PTHREADS 

#endif // _PTHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: PthreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2006/04/26 01:25:21 $
 * POOMA_VERSION_ID: $Id: PthreadLayer.h,v 1.6 2006/04/26 01:25:21 amorris Exp $
 ***************************************************************************/


