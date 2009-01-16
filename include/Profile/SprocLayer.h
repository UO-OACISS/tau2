/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: SprocLayer.h				          **
**	Description 	: TAU Profiling Package Sproc Support Layer	  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _SPROCLAYER_H_
#define _SPROCLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class SprocLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef TAU_SPROC
#include <ulocks.h>
class SprocLayer 
{ // Layer for RtsLayer to interact with pthreads 
  public:
 	
 	SprocLayer () { }  // defaults
	~SprocLayer () { } 

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
        static usptr_t * tauArena; // Shared memory arena /dev/zero on SGI
	static int 	 tauThreadCount;     // counter
	static usema_t * tauThreadCountMutex;  // to protect tauThreadCount
	static usema_t * tauDBMutex;  // to protect TheFunctionDB
	static usema_t * tauEnvMutex;  // second mutex
	
};
#endif // TAU_SPROC 

#endif // _SPROCLAYER_H_

	

/***************************************************************************
 * $RCSfile: SprocLayer.h,v $   $Author: amorris $
 * $Revision: 1.4 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: SprocLayer.h,v 1.4 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


