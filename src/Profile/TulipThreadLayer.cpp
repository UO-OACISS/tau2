/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TulipThread.cpp				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting Tulip Threads 			  **
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


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF
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
#include "Profile/Profiler.h"



/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class TulipThread
// This allows us to get thread ids from 0..N-1 
/////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////
// Define the static private members of PthreadLayer  
/////////////////////////////////////////////////////////////////////////

#ifdef SMARTS
Mutex     TulipThreadLayer::tauDBMutex;  
#else // SMARTS
Tulip_Mutex     TulipThreadLayer::tauDBMutex;  
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


/***************************************************************************
 * $RCSfile: TulipThreadLayer.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 1998/08/27 19:23:52 $
 * POOMA_VERSION_ID: $Id: TulipThreadLayer.cpp,v 1.1 1998/08/27 19:23:52 sameer Exp $
 ***************************************************************************/


