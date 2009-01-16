/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: PthreadLayer.h				  **
**	Description 	: TAU Profiling Package Pthread Support Layer	  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
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
	static void SetThreadId(int); 	 // gets 0..N-1 thread id
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
 * $Revision: 1.9 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: PthreadLayer.h,v 1.9 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


