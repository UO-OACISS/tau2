/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009					   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TulipThreadLayer.h				  **
**	Description 	: TAU Profiling Package TulipThread Support Layer **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
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
#ifdef SMARTS
#include <Mutex.h>
using Smarts::Mutex;
#else  // SMARTS
#include <Tulip_Mutex.h>
#endif // SMARTS
class TulipThreadLayer 
{ // Layer for RtsLayer to interact with pthreads 
  public:
 	
 	TulipThreadLayer () { }  // defaults
	~TulipThreadLayer () { } 

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
#ifdef SMARTS
	static Mutex	   	tauDBMutex;  // to protect TheFunctionDB
	static Mutex	   	tauEnvMutex;  // second mutex
#else // SMARTS
	static Tulip_Mutex	   tauDBMutex;  // to protect TheFunctionDB
	static Tulip_Mutex	   tauEnvMutex;  // second mutex
#endif // SMARTS
	
};
#endif // TULIPTHREADS 

#endif // _TULIPTHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: TulipThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: TulipThreadLayer.h,v 1.6 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


