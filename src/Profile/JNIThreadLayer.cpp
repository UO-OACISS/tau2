/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009 					      	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: JNIThreadLayer.cpp				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting Java Threads 			  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include <Profile/Profiler.h>
#include <stdlib.h>

#include <jni.h>
#include <map>
#include <mutex>

#include "Profile/TauJAPI.h"

#include <android/log.h>
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "TAU", __VA_ARGS__)

/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class JNIThreadLayer
// This allows us to get thread ids from 0..N-1 
/////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////
// Define the static private members of JNIThreadLayer  
/////////////////////////////////////////////////////////////////////////

int                JNIThreadLayer::tauThreadCount = 0; 
map<jlong, int>    JNIThreadLayer::tauTidMap; // jid ==> tid
map<jlong, int>    JNIThreadLayer::tauSidMap; // jid ==> sid
JavaVM*            JNIThreadLayer::tauVM;  // init in JNI_OnLoad()
recursive_mutex    JNIThreadLayer::tauNumThreadsLock;
recursive_mutex    JNIThreadLayer::tauDBMutex;
recursive_mutex    JNIThreadLayer::tauEnvMutex;

static thread_local int tid = 0;

extern void CreateTopLevelRoutine(char *name, char *type, char *groupname, int tid);

int JNIThreadLayer::GetSidFromJid(jlong jid)
{
    return tauSidMap[jid];
}

////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. 
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::RegisterThread(jlong jid, int sid, char *tname)
{
  static int initflag = JNIThreadLayer::InitializeThreadData();

  // Lock the mutex guarding the thread count before incrementing it.
  tauNumThreadsLock.lock();

  if (tauThreadCount == TAU_MAX_THREADS)
  {
    fprintf(stderr, "TAU>ERROR number of threads exceeds TAU_MAX_THREADS\n");
    fprintf(stderr, "Change TAU_MAX_THREADS parameter in <tau>/include/Profile/Profiler.h\n");
    fprintf(stderr, "And make install. Current value is %d\n", tauThreadCount);
    fprintf(stderr, "******************************************************************\n");
    exit(1);
  }

  /* register only if it's not registered yet */
  if (tauTidMap.find(jid) == tauTidMap.end()) {
      tid = tauThreadCount;
      tauTidMap[jid] = tauThreadCount++;

      LOGV(" *** map jid %lld to sid %d\n", jid, sid);
      tauSidMap[jid] = sid;

      RtsLayer::setMyNode(0, tid);

      /* create top level profiler for this thread */
      CreateTopLevelRoutine(tname, (char*)"<ThreadEvents>", (char*)"DTM", tid);
  }

  // Unlock it now 
  tauNumThreadsLock.unlock();

  DEBUGPROFMSG("Thread id "<< *threadId << " Created! "<<endl);

  // A thread should call this routine exactly once. 

  return tauTidMap[jid];
}

////////////////////////////////////////////////////////////////////////
// GetThreadId wrapper to be used when we don't have the environment 
// pointer (JNIEnv *) that we get from JVMPI. Typically called by entry/exit
// of a non-Java layer. 
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::GetThreadId(void) 
{
    jlong jid = get_java_thread_id();
    if (jid == -1) {
	jid = TheLastJDWPEventThreadID();
//	printf(" *** %d: last jid = %lld, tid = %d\n", gettid(), jid, tauTidMap[jid]);
    } else {
	/* if this is a java thread and not get registered, register itself */
	if (tauTidMap.find(jid) == tauTidMap.end()) {
	    char *tname = get_java_thread_name();
	    RegisterThread(jid, gettid(), tname);
	    free(tname);
	}
	
//	printf(" *** %d: java jid = %lld, tid = %d\n", gettid(), jid, tauTidMap[jid]);
    }

    return tauTidMap[jid];
}

////////////////////////////////////////////////////////////////////////
// InitializeThreadData is called before any thread operations are performed. 
// It sets the default values for static private data members of the 
// JNIThreadLayer class.
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::InitializeThreadData(void)
{
  return 1;
}

////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::InitializeDBMutexData(void)
{
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockDB locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::LockDB(void)
{
  static int initflag=InitializeDBMutexData();
  // Lock the functionDB mutex
  tauDBMutex.lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::UnLockDB(void)
{
  // Unlock the functionDB mutex
  tauDBMutex.unlock();
  return 1;
}  

////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::InitializeEnvMutexData(void)
{
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockEnv locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::LockEnv(void)
{
  static int initflag=InitializeEnvMutexData();
  // Lock the Env mutex
  tauEnvMutex.lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::UnLockEnv(void)
{
  // Unlock the Env mutex
  tauEnvMutex.unlock();
  return 1;
}  
////////////////////////////////////////////////////////////////////////
// TotalThreads returns the number of active threads 
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::TotalThreads(void)
{
  int count;
  // For synchronization, we lock the thread count mutex. If we had a 
  // set and increment operation, we wouldn't need this. Optimization for
  // the future.

  tauNumThreadsLock.lock();
  count = tauThreadCount;
  tauNumThreadsLock.unlock();

  return count;
}

// Use JVMPI to get per thread cpu time (microseconds)
jlong JNIThreadLayer::getCurrentThreadCpuTime(void) {
  /* unimplemented */
  return 0;
}
  
// EOF JNIThreadLayer.cpp 


/***************************************************************************
 * $RCSfile: JNIThreadLayer.cpp,v $   $Author: khuck $
 * $Revision: 1.8 $   $Date: 2009/03/13 00:46:56 $
 * TAU_VERSION_ID: $Id: JNIThreadLayer.cpp,v 1.8 2009/03/13 00:46:56 khuck Exp $
 ***************************************************************************/


