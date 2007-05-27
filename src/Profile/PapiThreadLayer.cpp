/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: PapiThreadLayer.cpp				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting PAPI threads 			  **
**	Author		: Alan Morris					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <map.h>
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include "Profile/Profiler.h"



////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. This should be the first routine a 
// thread should invoke from its wrapper. Note: main() thread shouldn't
// call this routine. 
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::RegisterThread(void) {
  // do nothing
  return 0;
}


////////////////////////////////////////////////////////////////////////
// GetThreadId returns an id in the range 0..N-1 by looking at the 
// thread specific data. Since a getspecific has to be preceeded by a 
// setspecific (that all threads besides main do), we get a null for the
// main thread that lets us identify it as thread 0. It is the only 
// thread that doesn't do a PapiThreadLayer::RegisterThread(). 
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::GetThreadId(void) {

  int retval;
  int data;

  retval = PAPI_get_thr_specific(PAPI_USR1_TLS, (void**)&data);

  // we're making the (perhaps incorrect) assumption that the value will be zero 
  // if uninitialzed
  if (data == 0) {
    LockDB();
    static int tauThreadCount = 0;
    tauThreadCount++;
    data = tauThreadCount;
    UnLockDB();
    retval = PAPI_set_thr_specific(PAPI_USR1_TLS, (void*)data);
  }


  return data-1;
}

////////////////////////////////////////////////////////////////////////
// InitializeThreadData is called before any thread operations are performed. 
// It sets the default values for static private data members of the 
// PapiThreadLayer class.
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::InitializeThreadData(void) {
  return 0;
}

////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::InitializeDBMutexData(void) {
  return 0;
}

////////////////////////////////////////////////////////////////////////
// LockDB locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::LockDB(void) {
  PAPI_lock(PAPI_USR1_LOCK);
  return 1;
}  

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::UnLockDB(void) {
  PAPI_unlock(PAPI_USR1_LOCK);
  return 1;
}  

////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::InitializeEnvMutexData(void) {
  // false - can't lock since we're using PAPI for locking
  static int flag = PapiLayer::initializePapiLayer(false);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockEnv locks the mutex protecting TheFunctionEnv() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::LockEnv(void) {
  static int initflag=InitializeEnvMutexData();
  PAPI_lock(PAPI_USR2_LOCK);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int PapiThreadLayer::UnLockEnv(void) {
  PAPI_unlock(PAPI_USR2_LOCK);
  return 1;
}  


/***************************************************************************
 * $RCSfile: PapiThreadLayer.cpp,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 2007/05/27 19:25:22 $
 * POOMA_VERSION_ID: $Id: PapiThreadLayer.cpp,v 1.2 2007/05/27 19:25:22 sameer Exp $
 ***************************************************************************/


