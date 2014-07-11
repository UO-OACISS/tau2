/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997  						   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: MPCThreadLayer.cpp				  **
 **	Description 	: TAU Profiling Package RTS Layer definitions     **
 **			  for supporting pthreads 			  **
 **	Contact		: tau-team@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <Profile/Profiler.h>
#include <Profile/MPCThreadLayer.h>
#include <Profile/TauInit.h>

#include <stdlib.h>

using namespace std;
using namespace tau;

extern __thread void * tls_args;

/////////////////////////////////////////////////////////////////////////
// Define the static private members of MPCThreadLayer
/////////////////////////////////////////////////////////////////////////

int MPCThreadLayer::tauThreadCount = 0;
mpc_thread_once_t MPCThreadLayer::initFlag = MPC_THREAD_ONCE_INIT;
mpc_thread_key_t MPCThreadLayer::tauThreadId;
mpc_thread_mutex_t MPCThreadLayer::tauThreadCountMutex;
mpc_thread_mutex_t MPCThreadLayer::tauDBMutex;
mpc_thread_mutex_t MPCThreadLayer::tauEnvMutex;

#if defined (TAU_OPENMP)
#include "omp.h"
__thread int __openmp_thread_id = -1;
__thread int __openmp_thread_count = 0;
omp_lock_t tauRegistermutex;
#endif

////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. This should be the first routine a 
// thread should invoke from its wrapper. Note: main() thread shouldn't
// call this routine. 
////////////////////////////////////////////////////////////////////////

int MPCThreadLayer::RegisterThread(void)
{
  InitializeThreadData();
#if defined (TAU_OPENMP)
  if (__openmp_thread_id == -1) {
    mpc_thread_mutex_lock(&tauThreadCountMutex);
    omp_set_lock(&tauRegistermutex);
    __openmp_thread_id = __openmp_thread_count++;
    omp_unset_lock(&tauRegistermutex);
    mpc_thread_mutex_unlock(&tauThreadCountMutex);
  }
  return __openmp_thread_id;
#else
  int * id = (int*)mpc_thread_getspecific(tauThreadId);
  if (!id) {
    id = new int;
    mpc_thread_setspecific(tauThreadId, id);
    mpc_thread_mutex_lock(&tauThreadCountMutex);
    // Which should it be?
    // *id = RtsLayer::_createThread() - 1;
    // Or this?
    *id = tauThreadCount++;
    mpc_thread_mutex_unlock(&tauThreadCountMutex);
  }
  tls_args = (void*)id;
  return *id;
#endif
}

////////////////////////////////////////////////////////////////////////
// GetThreadId returns an id in the range 0..N-1 by looking at the 
// thread specific data. Since a getspecific has to be preceeded by a 
// setspecific (that all threads besides main do), we get a null for the
// main thread that lets us identify it as thread 0. It is the only 
// thread that doesn't do a MPCThreadLayer::RegisterThread(). 
////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::GetThreadId(void)
{
  InitializeThreadData();

#if defined (TAU_OPENMP)
  if (__openmp_thread_id == -1) {
    MPCThreadLayer::RegisterThread();
  }
  return __openmp_thread_id;
#else
  int * id = (int*)mpc_thread_getspecific(tauThreadId);
  if (id) {
    return *id;
  }
  return 0; // main() thread
#endif
}

////////////////////////////////////////////////////////////////////////
// InitializeThreadData is called before any thread operations are performed. 
// It sets the default values for static private data members of the 
// PthreadLayer class.
////////////////////////////////////////////////////////////////////////
extern "C"
void mpc_init_once(void)
{
  mpc_thread_key_create(&MPCThreadLayer::tauThreadId, NULL);
  mpc_thread_mutex_init(&MPCThreadLayer::tauThreadCountMutex, NULL);
  mpc_thread_mutex_init(&MPCThreadLayer::tauDBMutex, NULL);
  mpc_thread_mutex_init(&MPCThreadLayer::tauEnvMutex, NULL);
}

int MPCThreadLayer::InitializeThreadData(void)
{
  // Do this exactly once.  Checking a static flag is a race condition so
  // use pthread_once with a callback friend function.
  mpc_thread_once(&initFlag, mpc_init_once);
  return 0;
}

////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::InitializeDBMutexData(void)
{
  // Initialized in mpc_init_once
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockDB locks the mutex protecting TheFunctionDB() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::LockDB(void)
{
  InitializeThreadData();
  mpc_thread_mutex_lock(&tauDBMutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::UnLockDB(void)
{
  mpc_thread_mutex_unlock(&tauDBMutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::InitializeEnvMutexData(void)
{
  // Initialized in mpc_init_once
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockEnv locks the mutex protecting TheFunctionEnv() global database of 
// functions. This is required to ensure that push_back() operation 
// performed on this is atomic (and in the case of tracing this is 
// followed by a GetFunctionID() ). This is used in 
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::LockEnv(void)
{
  InitializeThreadData();
  mpc_thread_mutex_lock(&tauEnvMutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int MPCThreadLayer::UnLockEnv(void)
{
  mpc_thread_mutex_unlock(&tauEnvMutex);
  return 1;
}

extern "C" void MPC_Process_hook(void)
{
  int process;
  MPC_Process_rank(&process);
}

extern "C" void MPC_Task_hook(int rank)
{
  MPCThreadLayer::RegisterThread();
}

extern "C" int TauGetMPCProcessRank(void) {
  int rank;
  MPC_Process_rank(&rank);
  return rank;
}

