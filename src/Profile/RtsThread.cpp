/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: RtsThread.cpp				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting threads 			  **
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

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include "Profile/Profiler.h"

//////////////////////////////////////////////////////////////////////
// myNode() returns the current node id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myNode(void)
{
  return TheNode();
}


//////////////////////////////////////////////////////////////////////
// myContext() returns the current context id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myContext(void)
{
  return TheContext(); 
}

//////////////////////////////////////////////////////////////////////
// myNode() returns the current node id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myThread(void)
{
#ifdef PTHREADS
  return PthreadLayer::GetThreadId();
#elif  TAU_WINDOWS
  return WindowsThreadLayer::GetThreadId();
#elif  TULIPTHREADS
  return TulipThreadLayer::GetThreadId();
#elif JAVA
  // Java should not call this routine. tids should be in TauJava.cpp layer
  // only in Java 
  //cout <<"ERROR : Java shouldn't call RtsLayer::myThread() returns -1 \n";
  //return -1;
  return 0; // Be forgiving. This way a C++ app can use the .so as well.
#else  // if no other thread package is available 
  return 0;
#endif // PTHREADS
}


//////////////////////////////////////////////////////////////////////
// RegisterThread is called before any other profiling function in a 
// thread that is spawned off
//////////////////////////////////////////////////////////////////////
void RtsLayer::RegisterThread()
{
#ifdef PTHREADS
  PthreadLayer::RegisterThread();
#elif  TAU_WINDOWS
  WindowsThreadLayer::RegisterThread();
#elif  TULIPTHREADS
  TulipThreadLayer::RegisterThread();
#endif // PTHREADS
  return;
}

//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionDB (global) is locked while updating
//////////////////////////////////////////////////////////////////////

void RtsLayer::LockDB(void)
{
#ifdef PTHREADS
  PthreadLayer::LockDB();
#elif  TAU_WINDOWS
  WindowsThreadLayer::LockDB();
#elif  TULIPTHREADS
  TulipThreadLayer::LockDB();
#endif // PTHREADS
  return ; // do nothing if threads are not used
}


//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionDB (global) is locked while updating
//////////////////////////////////////////////////////////////////////
void RtsLayer::UnLockDB(void)
{
#ifdef PTHREADS
  PthreadLayer::UnLockDB();
#elif  TAU_WINDOWS
  WindowsThreadLayer::UnLockDB();
#elif  TULIPTHREADS
  TulipThreadLayer::UnLockDB();
#endif // PTHREADS
  return;
}



/***************************************************************************
 * $RCSfile: RtsThread.cpp,v $   $Author: bertie $
 * $Revision: 1.8 $   $Date: 1999/10/27 21:16:38 $
 * POOMA_VERSION_ID: $Id: RtsThread.cpp,v 1.8 1999/10/27 21:16:38 bertie Exp $
 ***************************************************************************/


