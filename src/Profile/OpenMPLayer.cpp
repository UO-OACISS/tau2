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

using namespace std;

/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class OpenMPLayer
// This allows us to get thread ids from 0..N-1 and lock and unlock DB
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Define the static private members of OpenMPLayer  
/////////////////////////////////////////////////////////////////////////

omp_lock_t OpenMPLayer::tauDBmutex;
omp_lock_t OpenMPLayer::tauEnvmutex;
omp_lock_t OpenMPLayer::tauRegistermutex;

struct OpenMPMap : public std::map<int, int>
{
  virtual ~OpenMPMap() {
    Tau_destructor_trigger();
  }
};

OpenMPMap & TheOMPMap()
{
  static OpenMPMap omp_map;
  return omp_map;
}

/* This is Thread Local Storage (TLS) for the thread ID.
 * Using this is MUCH faster than computing it every time we need it.
 * HOWEVER, it might not be supported everywhere. */
#if defined (TAU_OPENMP) && defined(TAU_USE_FAST_THREADID)
__thread int _tau_thread_id = -1;
#endif

////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. This should be the first routine a 
// thread should invoke from its wrapper. Note: main() thread shouldn't
// call this routine. 
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::RegisterThread(void)
{
  return RtsLayer::createThread();
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

#ifdef TAU_USE_FAST_THREADID
  // if this thread has been registered, then it has a TLS value for the ID
  if (_tau_thread_id > -1)
    return _tau_thread_id;
#endif

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
    //OpenMP thread identification not supported by compiler.
    printf("ERROR: OpenMP nesting not supported. Please use a compiler that supports OMP specification >= 3.0 or rerun with OMP_NESTED=FALSE.\n");
    exit(1);
  }
#endif /* TAU_OPENMP_NESTED */

  int tau_thread_id;
  if (omp_thread_id == 0) {
    tau_thread_id = omp_thread_id;
  } else {
    Initialize();
    omp_set_lock(&OpenMPLayer::tauRegistermutex);
    OpenMPMap & ompMap = TheOMPMap();
    OpenMPMap::iterator it = ompMap.find(omp_thread_id);
    if (it == ompMap.end()) {
    /* Process is already locked, call the unsafe thread creation routine. */
      tau_thread_id = RtsLayer::_createThread();
      ompMap[omp_thread_id] = tau_thread_id;
    } else {
      tau_thread_id = it->second;
    }
    omp_unset_lock(&OpenMPLayer::tauRegistermutex);

// do this now so the thread ID is set before starting timers
#ifdef TAU_USE_FAST_THREADID
    _tau_thread_id = tau_thread_id;
#endif

    Tau_create_top_level_timer_if_necessary_task(tau_thread_id);
  }

#ifdef TAU_USE_FAST_THREADID
  _tau_thread_id = tau_thread_id;
#endif

  return tau_thread_id;
#else
  return 0;
#endif /* TAU_OPENMP */
}

int OpenMPLayer::GetThreadId(void)
{
#ifdef TAU_OPENMP

#ifdef TAU_USE_FAST_THREADID
  if (_tau_thread_id == -1) {
	// call the function above, which will register the thread
	// and assign the TLS value which we will use henceforth
    return GetTauThreadId();  
  } else {
    return _tau_thread_id;
  }
#endif

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

#ifdef TAU_USE_FAST_THREADID
  _tau_thread_id = omp_thread_id;
#endif

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
// InitializeThreadData is called before any thread operations are performed. 
// It sets the default values for static private data members of the 
// OpenMPLayer class.
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::InitializeThreadData(void)
{
  return 1;
}

void OpenMPLayer::Initialize(void)
{
  // ONLY INITIALIZE THE LOCK ONCE!
  static int registerInitFlag = InitializeRegisterMutexData();
  static int dbInitFlag = InitializeDBMutexData();
  static int envInitFlag = InitializeEnvMutexData();
}

////////////////////////////////////////////////////////////////////////
int OpenMPLayer::InitializeDBMutexData(void)
{
  // For locking functionDB 
  omp_init_lock(&OpenMPLayer::tauDBmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
int OpenMPLayer::InitializeRegisterMutexData(void)
{
  // For locking thread registration process 
  omp_init_lock(&OpenMPLayer::tauRegistermutex);
  return 1;
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
  Initialize();
  omp_set_lock(&OpenMPLayer::tauDBmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockDB(void)
{
  omp_unset_lock(&OpenMPLayer::tauDBmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
int OpenMPLayer::InitializeEnvMutexData(void)
{
  // For locking functionEnv 
  omp_init_lock(&OpenMPLayer::tauEnvmutex);
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
  Initialize();
  omp_set_lock(&OpenMPLayer::tauEnvmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockEnv(void)
{
  omp_unset_lock(&OpenMPLayer::tauEnvmutex);
  return 1;
}

/***************************************************************************
 * $RCSfile: OpenMPLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2009/01/16 00:46:52 $
 * POOMA_VERSION_ID: $Id: OpenMPLayer.cpp,v 1.6 2009/01/16 00:46:52 amorris Exp $
 ***************************************************************************/

