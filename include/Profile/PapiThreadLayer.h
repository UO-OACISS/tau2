/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Papi_ThreadLayer.h				   **
**	Description 	: TAU Profiling Package Papi Thread Support Layer  **
**	Contact		: tau-team@cs.uoregon.edu 		 	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
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
 * $Revision: 1.2 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: PapiThreadLayer.h,v 1.2 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


