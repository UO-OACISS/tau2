/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009 					      	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: JavaThreadLayer.cpp				  **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting Java Threads 			  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
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

#include <Profile/Profiler.h>
#include <Profile/JavaThreadLayer.h>
#include <Profile/TauJVMTI.h>
#include <stdlib.h>


/////////////////////////////////////////////////////////////////////////
// Define the static private members of JavaThreadLayer  
/////////////////////////////////////////////////////////////////////////

int 	  JavaThreadLayer::tauThreadCount = 0; 
jrawMonitorID JavaThreadLayer::tauNumThreadsLock ;
jrawMonitorID JavaThreadLayer::tauDBMutex ;
jrawMonitorID JavaThreadLayer::tauEnvMutex ;
jvmtiEnv  * JavaThreadLayer::jvmti = NULL;


extern GlobalAgentData * get_global_data();
////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. 
////////////////////////////////////////////////////////////////////////
int * JavaThreadLayer::RegisterThread(jthread this_thread)
{
  static int initflag = JavaThreadLayer::InitializeThreadData();
  // if its in here the first time, setup mutexes etc.

  int *threadId = new int;

  // Lock the mutex guarding the thread count before incrementing it.
  jvmti->RawMonitorEnter(tauNumThreadsLock); 

  if (tauThreadCount == TAU_MAX_THREADS)
  {
    fprintf(stderr, "TAU>ERROR number of threads exceeds TAU_MAX_THREADS\n");
    fprintf(stderr, "Change TAU_MAX_THREADS parameter in <tau>/include/Profile/Profiler.h\n");
    fprintf(stderr, "And make install. Current value is %d\n", tauThreadCount);
    fprintf(stderr, "******************************************************************\n");
    exit(1);
  }

  // Increment the number of threads present (after assignment)
  (*threadId) = tauThreadCount ++;

  // Unlock it now 
  jvmti->RawMonitorExit(tauNumThreadsLock); 
  // A thread should call this routine exactly once. 

  // Make this a thread specific data structure with the thread environment.
  jvmti->SetThreadLocalStorage(this_thread, (void * )threadId); 

  DEBUGPROFMSG("Thread id " <<  *threadId << " Created!\n";);

  return threadId;
}

////////////////////////////////////////////////////////////////////////
// ThreadEnd Cleans up the thread.
// Needs to be issued before the thread is killed.
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::ThreadEnd(jthread this_thread){
  int * tidp;
  jvmti->GetThreadLocalStorage(this_thread, (void **) &tidp);

  DEBUGPROFMSG("Thread id " << *tidp << " End!\n";);
  //It seems that the threadLocalStorage is automatically deallocated by the JVM.
  //delete tidp;
}

////////////////////////////////////////////////////////////////////////
// GetThreadId returns an id in the range 0..N-1 by looking at the 
// thread specific data.
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::GetThreadId(jthread this_thread) 
{
  int *tidp ;
  static GlobalAgentData * gdata = get_global_data();

  //The ThreadLocalStorage is not properly initialized to zero
  //during shutdown/startup.
  if(!gdata->vm_is_initialized or gdata->vm_is_dead)
    return 0;
   
  jvmti->GetThreadLocalStorage(this_thread, (void **) &tidp);
  // The thread id is stored in a thread specific storage

  if (tidp == (int *) NULL)
  { // This thread needs to be registered
    DEBUGPROFMSG("This thread dosn't apear to be registered.\n";);

    tidp = JavaThreadLayer::RegisterThread(this_thread);
  }
  return (*tidp);

}


////////////////////////////////////////////////////////////////////////
// InitializeThreadData is called before any thread operations are performed. 
// It sets the default values for static private data members of the 
// JavaThreadLayer class.
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::InitializeThreadData(void)
{
  // Initialize the mutex
  jvmtiError             error;
  static bool initialized = false;
  if(!initialized){
    error = jvmti->CreateRawMonitor("num threads lock", &tauNumThreadsLock);
    check_jvmti_error(jvmti, error, "Cannot Create raw monitor");
    initialized = true;
  }
  
  return 1;
}

////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::InitializeDBMutexData(void)
{
  jvmtiError error;
  static bool initialized = false;
  // For locking functionDB 
  DEBUGPROFMSG("InitializeDBMutexData.\n";);

  if(!initialized){
    error = jvmti->CreateRawMonitor("FuncDB lock", &tauDBMutex);
    check_jvmti_error(jvmti, error, "Cannot create raw monitor");
    initialized = true;
  }
		 
   return 1;
}

////////////////////////////////////////////////////////////////////////
// LockDB locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::LockDB(void)
{
  jvmtiError error;
  InitializeDBMutexData();

  // Lock the functionDB mutex
  error = jvmti->RawMonitorEnter(tauDBMutex);
  check_jvmti_error(jvmti, error, "Cannot enter with raw monitor");
		 
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::UnLockDB(void)
{
  jvmtiError error;

  // Unlock the functionDB mutex
  error = jvmti->RawMonitorExit(tauDBMutex);
  check_jvmti_error(jvmti, error, "Cannot exit with raw monitor");
  return 1;
}  

////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::InitializeEnvMutexData(void)
{
  jvmtiError             error;
  static bool initialized = false;
  if (jvmti == NULL) {
    fprintf (stderr,"Error, TAU's jvmpi interface was not initialized properly (java -XrunTAU ...)\n");
    fprintf (stderr,"When TAU is configured with -jdk=<dir>, it can only profile Java Programs!\n");
    exit(-1);
  }
  //For locking LockEnv
  DEBUGPROFMSG("InitializeEnvMutex.\n";);

  if(!initialized){
    error = jvmti->CreateRawMonitor("Env lock", &tauEnvMutex);
    check_jvmti_error(jvmti, error, "Cannot create raw monitor");
    initialized = true;
  }
  
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockEnv locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::LockEnv(void)
{
  jvmtiError error;
  InitializeEnvMutexData();
  // Lock the Env mutex
  error = jvmti->RawMonitorEnter(tauEnvMutex);
  check_jvmti_error(jvmti, error, "Cannot enter tauEnv with raw monitor");
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::UnLockEnv(void)
{
  jvmtiError error;
  // Unlock the Env mutex
  error = jvmti->RawMonitorExit(tauEnvMutex);
  check_jvmti_error(jvmti, error, "Cannot exit with raw monitor");

  return 1;
}  
////////////////////////////////////////////////////////////////////////
// TotalThreads returns the number of active threads 
////////////////////////////////////////////////////////////////////////
int JavaThreadLayer::TotalThreads(void)
{
  int count;
  // For synchronization, we lock the thread count mutex. If we had a 
  // set and increment operation, we wouldn't need this. Optimization for
  // the future.

  jvmti->RawMonitorEnter(tauNumThreadsLock);
  count = tauThreadCount;
  jvmti->RawMonitorExit(tauNumThreadsLock);

  return count;
}

// Use JVMTI to get per thread cpu time (microseconds)
jlong JavaThreadLayer::getCurrentThreadCpuTime(void) {
  jlong thread_time;
  jvmti->GetCurrentThreadCpuTime(&thread_time);
  return thread_time;
}
  
// EOF JavaThreadLayer.cpp 


/***************************************************************************
 * $RCSfile: JavaThreadLayer.cpp,v $   $Author: khuck $
 * $Revision: 1.8 $   $Date: 2009/03/13 00:46:56 $
 * TAU_VERSION_ID: $Id: JavaThreadLayer.cpp,v 1.8 2009/03/13 00:46:56 khuck Exp $
 ***************************************************************************/


