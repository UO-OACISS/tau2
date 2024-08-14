/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2014 					      	   **
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
#include <unistd.h>
#include <sys/syscall.h>

#include <jni.h>
#include <map>
#include <mutex>

#include "Profile/TauJAPI.h"

#ifdef TAU_ANDROID 
#include <android/log.h>
#define LOGV(...) //__android_log_print(ANDROID_LOG_VERBOSE, "TAU", __VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL, "TAU", __VA_ARGS__)
#else
#define LOGV(...) 
#define LOGF(...)
#endif /* TAU_ANDROID */

/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class JNIThreadLayer
// This allows us to get thread ids from 0..N-1 
/////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////
// Define the static private members of JNIThreadLayer  
/////////////////////////////////////////////////////////////////////////

int                JNIThreadLayer::tauThreadCount = 0; 
map<pid_t, int>    JNIThreadLayer::tauTidMap; // sid ==> tid
JavaVM*            JNIThreadLayer::tauVM;  // init in JNI_OnLoad()
mutex              JNIThreadLayer::tauNumThreadsLock;
map<pid_t, mutex>  JNIThreadLayer::tauDBMutex;
map<pid_t, mutex>  JNIThreadLayer::tauEnvMutex;

map<pid_t, mutex>  JNIThreadLayer::tauDTMLock;

/* C++11 thread local variables */
static thread_local pid_t _sid   = 0;
static thread_local const char *_tname = NULL;
static thread_local bool  _isJavaThread = true;

extern void CreateTopLevelRoutine(char *name, char *type, char *groupname, int tid);

/* make sure this thread has been registered by dalvik_thread_monitor() */
void JNIThreadLayer::WaitForDTM(void)
{
    pid_t sid = GetThreadSid();

    tauNumThreadsLock.lock();

    LOGV(" *** (S%d) WaitForDTM: S%d: start\n", gettid(), sid);

    if (tauTidMap.find(sid) == tauTidMap.end()) {
	/* ok, DTM didn't register this thread yet */
	tauDTMLock[sid].lock();

	/* give DTM a chance to register this thread */
	tauNumThreadsLock.unlock();

	LOGV(" *** (S%d) WaitForDTM: S%d: block\n", gettid(), sid);

	/* block until unlocked by DTM */
	tauDTMLock[sid].lock();

	/* we don't need this lock anymore */
	tauDTMLock.erase(sid);

	LOGV(" *** (S%d) WaitForDTM: S%d: finish\n", gettid(), sid);
	return;
    }

    LOGV(" *** (S%d) WaitForDTM: S%d: finish\n", gettid(), sid);
    tauNumThreadsLock.unlock();
}

/*
 * dalvik_thread_monitor() may start the top level profiler for new threads.
 * To make things go smoothly, dalvik_thread_monitor() shall "su" to that thread.
 * See also GetThreadSid()/GetThreadName() below.
 */
void JNIThreadLayer::SuThread(pid_t sid, char *tname)
{
    _sid   = sid;
    _tname = tname;
}

void JNIThreadLayer::SuThread(pid_t tid)
{
    map<pid_t, int>::iterator it;
    for(it=tauTidMap.begin(); it!=tauTidMap.end(); it++) {
	if (it->second == tid) {
	    _sid = it->first;
	    break;
	}
    }

    if (it == tauTidMap.end()) {
	_sid = -1;
    }

    _tname = "_no_use_";
}

pid_t JNIThreadLayer::GetThreadSid(void)
{
    pid_t sid;

    if (_sid != 0) {        // we are in dalvik_thread_monitor()
	sid = _sid;
    } else {                // we are in the java thread or alfred
	sid = syscall(__NR_gettid);
    }

    return sid;
}

char *JNIThreadLayer::GetThreadName(void)
{
    if (_tname != NULL) {   // we are in the dalvik_thread_monitor()
	return strdup(_tname);
    } else {                // we are in the java thread or alfred
	return get_java_thread_name();
    }
}

/* mark this thread as not a java thread */
void JNIThreadLayer::IgnoreThisThread(void)
{
    _isJavaThread = false;
}

/* Management threads include Alfred and DTM */
bool JNIThreadLayer::IsMgmtThread(void)
{
    return (_isJavaThread == false);
}

////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. 
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::RegisterThread(pid_t sid, char *tname)
{
    static int initflag = JNIThreadLayer::InitializeThreadData();

    // Lock the mutex guarding the thread count before incrementing it.
    tauNumThreadsLock.lock();
/*
    if (tauThreadCount == TAU_MAX_THREADS)
    {
	fprintf(stderr, "TAU>ERROR number of threads exceeds TAU_MAX_THREADS\n");
	fprintf(stderr, "Change TAU_MAX_THREADS parameter in <tau>/include/Profile/Profiler.h\n");
	fprintf(stderr, "And make install. Current value is %d\n", tauThreadCount);
	fprintf(stderr, "******************************************************************\n");
	exit(1);
    }
*/
    int tid = tauThreadCount;

    tauTidMap[sid] = tid;

    RtsLayer::setMyNode(0, tid);

    LOGV(" *** (S%d) Register S%d %s to T%d\n", gettid(), sid, tname, tid);

    /*
     * Create top level profiler for this thread
     * This should only be done after updating tauTidMap
     */
    CreateTopLevelRoutine(tname, (char*)"<ThreadEvents>", (char*)"DTM", tid);

    LOGV(" *** (S%d) Register done\n", gettid());

    tauThreadCount++;

    if (tauDTMLock.find(sid) != tauDTMLock.end()) {
	LOGV(" *** (S%d) Unlock S%d\n", gettid(), sid);
	tauDTMLock[sid].unlock();
    }

    // Unlock it now 
    tauNumThreadsLock.unlock();

    return tid;
}

int JNIThreadLayer::GetThreadId(void) 
{
    pid_t sid = GetThreadSid();

    if (tauTidMap.find(sid) == tauTidMap.end()) {
	LOGF(" *** GetThreadId: %d never regiested!", sid);
    }

    /* note that for DTM and Alfred, this will return 0 */
    return tauTidMap[sid];
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
  pid_t sid = GetThreadSid();
  // Lock the functionDB mutex
  tauDBMutex[sid].lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::UnLockDB(void)
{
  // Unlock the functionDB mutex
  pid_t sid = GetThreadSid();
  tauDBMutex[sid].unlock();
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
  pid_t sid = GetThreadSid();
  // Lock the Env mutex
  tauEnvMutex[sid].lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::UnLockEnv(void)
{
  // Unlock the Env mutex
  pid_t sid = GetThreadSid();
  tauEnvMutex[sid].unlock();
  return 1;
}  
////////////////////////////////////////////////////////////////////////
// TotalThreads returns the number of active threads 
////////////////////////////////////////////////////////////////////////
int JNIThreadLayer::TotalThreads(void)
{
  lock_guard<mutex> tidMapGuard(tauNumThreadsLock);

  return tauThreadCount;
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


