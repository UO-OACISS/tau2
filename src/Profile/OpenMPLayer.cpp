/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997  						   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: OpenMPLayer.cpp				  **
 **	Description 	: TAU Profiling Package RTS Layer definitions     **
 **			  for supporting OpenMP Threads			  **
 **	Contact		: tau-team@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <math.h>
#include <Profile/Profiler.h>
#include <Profile/OpenMPLayer.h>
#include <atomic>

using namespace std;

/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class OpenMPLayer
// This allows us to get thread ids from 0..N-1 and lock and unlock DB
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Define the static private members of OpenMPLayer
/////////////////////////////////////////////////////////////////////////

std::mutex OpenMPLayer::tauDBmutex;
std::mutex OpenMPLayer::tauEnvmutex;
std::mutex OpenMPLayer::tauRegistermutex;

thread_local int _tau_thread_id = -1;
atomic<int> _thread_count{0};

////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. This should be the first routine a
// thread should invoke from its wrapper. Note: main() thread shouldn't
// call this routine.
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::RegisterThread(void)
{
  /* There's some circular recursion going on...*/
  static bool avoid_reentry = false;
  if (avoid_reentry == true) return 0;
#ifdef TAU_OPENMP

  /* We have to lock here and use the unsafe thread creation routine
   * Using the safe creation routine generates a call to GetTauThreadId()
   * which would still detect _tau_thread_id as being -1 and try to create
   * the thread again */
  // if this thread has not been registered, then it does not have a TLS value for the ID
  if (_tau_thread_id == -1) {
    /* This is temporary, just in case we re-enter this function.
     * Technically, it's a race condition on _thread_count, but this
     * code only really matters when thread 0 is initializing.  The
     * call two lines later to Initialize() will call this function,
     * and a single thread can get registered twice. */
    if (_thread_count == 0) {
        _tau_thread_id = 0;
        avoid_reentry = true;
    }
    /* end temporary settings */
    Tau_global_incr_insideTAU();
    {
        if (_thread_count > 0) {
            const std::lock_guard<std::mutex> lock(tauRegistermutex);
            /* Process is already locked, call the unsafe thread creation routine. */
            _tau_thread_id = RtsLayer::_createThread();
        } else {
            _tau_thread_id = 0;
        }
        _thread_count = _thread_count + 1;
    }
    Tau_global_decr_insideTAU();
	// TAU may not be done initializing yet! So don't start the timer for thread 0
	if (_tau_thread_id > 0)
      Tau_create_top_level_timer_if_necessary_task(_tau_thread_id);
  }
  avoid_reentry = false;
  return _tau_thread_id;
#else /* TAU_OPENMP */
  avoid_reentry = false;
  return 0;
#endif /* TAU_OPENMP */
}

int OpenMPLayer::numThreads()
{
  return omp_get_max_threads();
}

////////////////////////////////////////////////////////////////////////
// GetThreadId maps the id in the thread specific data to the acutal TAU thread
// ID. Since a getspecific has to be preceeded by a
// setspecific (that all threads besides main do), we get a null for the
// main thread that lets us identify it as thread 0. It is the only
// thread that doesn't do a OpenMPLayer::RegisterThread().
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::GetTauThreadId(void)
{
#ifdef TAU_OPENMP
  // if this thread has not been registered, then it does not have a TLS value for the ID
  if (_tau_thread_id == -1) {
      RtsLayer::RegisterThread();
  }
  return _tau_thread_id;
#else
  return 0;
#endif /* TAU_OPENMP */
}

int OpenMPLayer::GetThreadId(void)
{
#ifdef TAU_OPENMP

  if (_tau_thread_id == -1) {
	// call the function above, which will register the thread
	// and assign the TLS value which we will use henceforth
    return GetTauThreadId();
  } else {
    return _tau_thread_id;
  }
  /* the code below shouldn't be called, but just in case... */
  int omp_thread_id = omp_get_thread_num();
#ifdef TAU_OPENMP_NESTED
  int level = omp_get_level();
  int width = omp_get_team_size(level);
  for (--level; level >= 0; --level) {
    omp_thread_id += omp_get_ancestor_thread_num(level) * width;
    width *= omp_get_team_size(level);
  }
#else
  if (omp_get_nested()) {
    //OPENMP thread identification not supported by compiler.
    printf("ERROR: OpenMP nesting not supported. Please use a compiler that supports OMP specification >= 3.0 or rerun with OMP_NESTED=FALSE.\n");
    exit(1);
  }
#endif /* TAU_OPENMP_NESTED */
  return omp_thread_id;
#else
  return 0;
#endif /* TAU_OPENMP */
}

////////////////////////////////////////////////////////////////////////
// TotalThreads returns the total number of threads running
// The user typically sets this by setting the environment variable
// OMP_NUM_THREADS or by using the routine omp_set_num_threads(int);
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::TotalThreads(void)
{
#ifdef TAU_OPENMP
  // Note: this doesn't work for nested parallelism
  return omp_get_num_threads();
#else
  return 0;
#endif /* TAU_OPENMP */

}

////////////////////////////////////////////////////////////////////////
// LockDB locks the mutex protecting TheFunctionDB() global database of
// functions. This is required to ensure that push_back() operation
// performed on this is atomic (and in the case of tracing this is
// followed by a GetFunctionID() ). This is used in
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::LockDB(void)
{
#if 0
  static int owner = 0;
  int acquired = 0;
  int tries = 0;
  do {
      acquired = omp_test_lock(&OpenMPLayer::tauDBmutex);
      if (++tries > 10000000) {
          printf("DEADLOCK! I am %d, lock held by %d\n", _tau_thread_id, owner);
          abort();
      }
  } while(acquired == 0);
  owner = _tau_thread_id;
#else
  OpenMPLayer::tauDBmutex.lock();
#endif
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockDB(void)
{
  OpenMPLayer::tauDBmutex.unlock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// LockEnv locks the mutex protecting TheFunctionEnv() global database of
// functions. This is required to ensure that push_back() operation
// performed on this is atomic (and in the case of tracing this is
// followed by a GetFunctionID() ). This is used in
// FunctionInfo::FunctionInfoInit().
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::LockEnv(void)
{
  OpenMPLayer::tauEnvmutex.lock();
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockEnv(void)
{
  OpenMPLayer::tauEnvmutex.unlock();
  return 1;
}

/***************************************************************************
 * $RCSfile: OpenMPLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2009/01/16 00:46:52 $
 * POOMA_VERSION_ID: $Id: OpenMPLayer.cpp,v 1.6 2009/01/16 00:46:52 amorris Exp $
 ***************************************************************************/

