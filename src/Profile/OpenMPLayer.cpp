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

static bool initialized = false;

/* This is Thread Local Storage (TLS) for the thread ID.
 * Using this is MUCH faster than computing it every time we need it.
 * HOWEVER, it might not be supported everywhere. */
#if defined (TAU_OPENMP)
#if defined (TAU_USE_TLS)
__thread int _tau_thread_id = -1;
int _thread_count = 0;
#elif defined(TAU_USE_DTLS)
__declspec(thread) int _tau_thread_id = -1;
int _thread_count = 0;
#elif defined(TAU_USE_PGS)
#include "TauPthreadGlobal.h"
int _thread_count = 0;
#endif
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
#ifdef TAU_OPENMP

  /* We have to lock here and use the unsafe thread creation routine
   * Using the safe creation routine generates a call to GetTauThreadId()
   * which would still detect _tau_thread_id as being -1 and try to create
   * the thread again */
#if defined (TAU_USE_TLS) || defined (TAU_USE_DTLS)
  // if this thread has not been registered, then it does not have a TLS value for the ID
  if (_tau_thread_id == -1) {
    Tau_global_incr_insideTAU();
    Initialize();
    if (initialized) omp_set_lock(&OpenMPLayer::tauRegistermutex);
    if (_thread_count > 0) {
      /* Process is already locked, call the unsafe thread creation routine. */
      _tau_thread_id = RtsLayer::_createThread();
    } else {
      _tau_thread_id = 0;
    }
    _thread_count = _thread_count + 1;
    if (initialized) omp_unset_lock(&OpenMPLayer::tauRegistermutex);
    Tau_global_decr_insideTAU();
	// TAU may not be done initializing yet! So don't start the timer for thread 0
	if (_tau_thread_id > 0) 
      Tau_create_top_level_timer_if_necessary_task(_tau_thread_id);
  }
  return _tau_thread_id;
#elif defined (TAU_USE_PGS)
  struct _tau_global_data *tmp = TauGlobal::getInstance().getValue();
  // if this thread has not been registered, then it does not have a TLS value for the ID
  if (tmp->threadID == -1) {
    Tau_global_incr_insideTAU();
    Initialize();
    if (initialized) omp_set_lock(&OpenMPLayer::tauRegistermutex);
    if (_thread_count > 0) {
      /* Process is already locked, call the unsafe thread creation routine. */
      tmp->threadID = RtsLayer::_createThread();
    } else {
      tmp->threadID = 0;
    }
    _thread_count = _thread_count + 1;
    if (initialized) omp_unset_lock(&OpenMPLayer::tauRegistermutex);
    Tau_global_decr_insideTAU();
	// TAU may not be done initializing yet! So don't start the timer for thread 0
	if (tmp->threadID > 0)
      Tau_create_top_level_timer_if_necessary_task(tmp->threadID);
  }
  return tmp->threadID;
#else // TAU_USE_TLS

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
    if (initialized) omp_set_lock(&OpenMPLayer::tauRegistermutex);
    OpenMPMap & ompMap = TheOMPMap();
    OpenMPMap::iterator it = ompMap.find(omp_thread_id);
    if (it == ompMap.end()) {
    /* Process is already locked, call the unsafe thread creation routine. */
      tau_thread_id = RtsLayer::_createThread();
      ompMap[omp_thread_id] = tau_thread_id;
    } else {
      tau_thread_id = it->second;
    }
    if (initialized) omp_unset_lock(&OpenMPLayer::tauRegistermutex);

    Tau_create_top_level_timer_if_necessary_task(tau_thread_id);
  }

  return tau_thread_id;
#endif // TAU_USE_TLS
#else
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

#if defined (TAU_USE_TLS) || defined (TAU_USE_DTLS)
  // if this thread has not been registered, then it does not have a TLS value for the ID
  if (_tau_thread_id == -1) {
      RtsLayer::RegisterThread();
  }
  return _tau_thread_id;
#elif defined (TAU_USE_PGS)
  struct _tau_global_data *tmp = TauGlobal::getInstance().getValue();
  // if this thread has not been registered, then it does not have a TLS value for the ID
  if (tmp->threadID == -1) {
      RtsLayer::RegisterThread();
  }
  return tmp->threadID;
#else // TAU_USE_TLS

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
    if (initialized) omp_set_lock(&OpenMPLayer::tauRegistermutex);
    OpenMPMap & ompMap = TheOMPMap();
    OpenMPMap::iterator it = ompMap.find(omp_thread_id);
    if (it == ompMap.end()) {
        if (initialized) omp_unset_lock(&OpenMPLayer::tauRegistermutex);
        tau_thread_id = OpenMPLayer::RegisterThread();
        if (initialized) omp_set_lock(&OpenMPLayer::tauRegistermutex);
      ompMap[omp_thread_id] = tau_thread_id;
    /* Activating Sampling here since we had to use OpenMPLayer::RegisterThread instead of RtsLayer::RegisterThread. */
#ifndef TAU_WINDOWS
#ifndef _AIX
      if (TauEnv_get_ebs_enabled()) {
          Tau_sampling_init_if_necessary();
      }
#endif /* _AIX */
#endif /* TAU_WINDOWS */
    } else {
      tau_thread_id = it->second;
    }
    if (initialized) omp_unset_lock(&OpenMPLayer::tauRegistermutex);

    Tau_create_top_level_timer_if_necessary_task(tau_thread_id);
  }

  return tau_thread_id;
#endif // TAU_USE_TLS
#else
  return 0;
#endif /* TAU_OPENMP */
}

int OpenMPLayer::GetThreadId(void)
{
#ifdef TAU_OPENMP

#if defined (TAU_USE_TLS) || defined (TAU_USE_DTLS)
  if (_tau_thread_id == -1) {
	// call the function above, which will register the thread
	// and assign the TLS value which we will use henceforth
    return GetTauThreadId();  
  } else {
    return _tau_thread_id;
  }
#elif defined (TAU_USE_PGS)
  struct _tau_global_data *tmp = TauGlobal::getInstance().getValue();
  if (tmp->threadID == -1) {
	// call the function above, which will register the thread
	// and assign the TLS value which we will use henceforth
    return GetTauThreadId();  
  } else {
    return tmp->threadID;
  }
#else //TAU_USE_TLS

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
#endif //TAU_USE_TLS
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
  static int initializing_or_initialized = false;
  if (initializing_or_initialized) { return; }
  initializing_or_initialized = true;
  // ONLY INITIALIZE THE LOCK ONCE!
  static int registerInitFlag = InitializeRegisterMutexData();
  static int dbInitFlag = InitializeDBMutexData();
  static int envInitFlag = InitializeEnvMutexData();
  // use the flags so that the compiler doesn't complain
  if (registerInitFlag && dbInitFlag && envInitFlag) {};
  initialized = true;
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
  if (initialized) omp_set_lock(&OpenMPLayer::tauDBmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockDB(void)
{
  if (initialized) omp_unset_lock(&OpenMPLayer::tauDBmutex);
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
  if (initialized) omp_set_lock(&OpenMPLayer::tauEnvmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockEnv(void)
{
  if (initialized) omp_unset_lock(&OpenMPLayer::tauEnvmutex);
  return 1;
}

/***************************************************************************
 * $RCSfile: OpenMPLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2009/01/16 00:46:52 $
 * POOMA_VERSION_ID: $Id: OpenMPLayer.cpp,v 1.6 2009/01/16 00:46:52 amorris Exp $
 ***************************************************************************/

