/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TulipThread.cpp				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting Tulip Threads 			  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF
#include "Profile/Profiler.h"
#ifdef SMARTS
#include <Context.h>
#include <Thread.h>
#include <Mutex.h>
using namespace NAMESPACE;
#else // SMARTS
#include <Tulip_Context.h>
#include <Tulip_Thread.h>
#include <Tulip_Mutex.h>
#endif // SMARTS



/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class TulipThread
// This allows us to get thread ids from 0..N-1 
/////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////
// Define the static private members of PthreadLayer  
/////////////////////////////////////////////////////////////////////////

#ifdef SMARTS
Mutex     TulipThreadLayer::tauDBMutex;  
Mutex     TulipThreadLayer::tauEnvMutex;  
#else // SMARTS
Tulip_Mutex     TulipThreadLayer::tauDBMutex;  
Tulip_Mutex     TulipThreadLayer::tauEnvMutex;  
#endif // SMARTS


////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. This should be the first routine a 
// thread should invoke from its wrapper. Note: main() thread shouldn't
// call this routine. 
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::RegisterThread(void)
{
/* Do nothing */
  return 1;
}


////////////////////////////////////////////////////////////////////////
// GetThreadId returns an id in the range 0..N-1 by looking at the 
// Tulip_Context::thread and getting the id from it. Note: Tulip numbers
// the threads from 1..N so we need to decrement it by 1.
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::GetThreadId(void) 
{
  DEBUGPROFMSG("TulipThreadLayer::GetThreadId() : "<<flush);
#ifdef SMARTS
  int tid = Context::getThreadID();
/*
  int tid;
  Thread *mythr = Context::thread();
  DEBUGPROFMSG("Tid = "<<flush);
  if (mythr == NULL)
  {
    DEBUGPROFMSG("Thread NULL " <<flush<<endl;);
    tid = MYID;
  }
  else
  {
    tid = mythr ->getThreadID() ;
  }
*/
/* old...
  int tid = Context::thread()->getThreadID() - 1;
*/ 
#else // SMARTS
  int tid = Tulip_Context::thread()->getThreadID() - 1;
#endif // SMARTS
  DEBUGPROFMSG(" " << tid <<endl;);

  return tid;
}

////////////////////////////////////////////////////////////////////////
// InitializeThreadData is called before any thread operations are performed. 
// It sets the default values for static private data members of the 
// PthreadLayer class.
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::InitializeThreadData(void)
{
  // Do we need to initialize anything?
  return 1;
}

////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::InitializeDBMutexData(void)
{
  // For locking functionDB 
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockDB locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::LockDB(void)
{
  static int initflag=InitializeDBMutexData();
  // Lock the functionDB mutex
  tauDBMutex.lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::UnLockDB(void)
{
  // Unlock the functionDB mutex
  tauDBMutex.unlock();
  return 1;
}  

////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::InitializeEnvMutexData(void)
{
  // For locking functionEnv
  return 1;
}


////////////////////////////////////////////////////////////////////////
// LockEnv locks the mutex protecting TheFunctionEnv() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::LockEnv(void)
{
  static int initflag=InitializeEnvMutexData();
  // Lock the functionEnv mutex
  tauEnvMutex.lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int TulipThreadLayer::UnLockEnv(void)
{
  // Unlock the functionEnv mutex
  tauEnvMutex.unlock();
  return 1;
}  
/***************************************************************************
 * $RCSfile: TulipThreadLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.5 $   $Date: 2009/01/16 00:46:53 $
 * POOMA_VERSION_ID: $Id: TulipThreadLayer.cpp,v 1.5 2009/01/16 00:46:53 amorris Exp $
 ***************************************************************************/


