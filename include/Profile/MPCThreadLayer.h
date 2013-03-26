/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: MPCThreadLayer.h				  **
**	Description 	: TAU Profiling Package Pthread Support Layer	  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


#ifndef _MPCTHREADLAYER_H_
#define _MPCTHREADLAYER_H_

#include <mpc.h>
//////////////////////////////////////////////////////////////////////
//
// class MPCThreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

class MPCThreadLayer 
{ // Layer for RtsLayer to interact with pthreads 
  public:
 	
 	MPCThreadLayer () { }  // defaults
	~MPCThreadLayer () { } 

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
        static mpc_thread_mutex_t tauThreadCountMutex; 
        static mpc_thread_mutex_t tauDBMutex; 
        static mpc_thread_mutex_t tauEnvMutex; 
        static int tauThreadCount; 	
};
	



#endif /* _MPCTHREADLAYER_H_ */

	

/***************************************************************************
 * $RCSfile: MPCThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.9 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: MPCThreadLayer.h,v 1.9 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


