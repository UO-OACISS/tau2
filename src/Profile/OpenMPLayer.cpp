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

//#define DEBUG_PROF
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;

#include <math.h>

#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include "Profile/Profiler.h"
#include "Profile/OpenMPLayer.h"



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

#ifdef TAU_OPENMP_NESTED
static int threadId = -1;
#pragma omp threadprivate(threadId)
#endif /* TAU_OPENMP_NESTED */


////////////////////////////////////////////////////////////////////////
// RegisterThread() should be called before any profiling routines are
// invoked. This routine sets the thread id that is used by the code in
// FunctionInfo and Profiler classes. This should be the first routine a 
// thread should invoke from its wrapper. Note: main() thread shouldn't
// call this routine. 
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::RegisterThread(void)
{
	int id = RtsLayer::createThread();
//	printf("OpenMP: registering thread, id = %d.\n", id);
	return id;
  // Not needed for OpenMP programs! 
  //return 0;
}

int OpenMPLayer::numThreads()
{
	return omp_get_max_threads(); 
}

map<int, int>& TheOMPMap()
{
	static map<int, int> omp_map;
	return omp_map;
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
	int omp_thread_id, tau_thread_id;
#ifdef TAU_OPENMP
#ifdef TAU_OPENMP_NESTED
	threadId = omp_get_thread_num();
	int level = omp_get_level() - 1;
	int depth = 0;
	int width = omp_get_team_size(level+1);
	while (level >= 0)
	{
		depth++;
		threadId += omp_get_ancestor_thread_num(level)*width;
		width *= omp_get_team_size(level);
		level--;
	}
  omp_thread_id = threadId;
#else
	if (omp_get_nested())
	{
		//OPENMP thread identification not supported by compiler.
		printf("ERROR: OpenMP nesting not supported. Please use a compiler that supports OMP specification >= 3.0 or rerun with OMP_NESTED=FALSE.\n");
		exit(1);
	}
  omp_thread_id = omp_get_thread_num();
#endif /* TAU_OPENMP_NESTED */

	if (omp_thread_id == 0)
	{
		tau_thread_id = omp_thread_id;
	}
	else
	{
		Initialize();
		//fprintf(stderr, "Thread %d locking register\n", omp_thread_id);
        omp_set_lock(&OpenMPLayer::tauRegistermutex);

		map<int, int>::iterator it = TheOMPMap().find(omp_thread_id);
		if (it == TheOMPMap().end())
		{
			//TheOMPMap()[omp_thread_id] = OpenMPLayer::RegisterThread();

		    // unlock register
            omp_unset_lock(&OpenMPLayer::tauRegistermutex);
		    tau_thread_id = OpenMPLayer::RegisterThread();
			// relock register
            omp_set_lock(&OpenMPLayer::tauRegistermutex);
			TheOMPMap()[omp_thread_id] = tau_thread_id;
		}	
		else
		{
			tau_thread_id = it->second;
		}
		
        omp_unset_lock(&OpenMPLayer::tauRegistermutex);
		//fprintf(stderr, "Thread %d unlocking register\n", omp_thread_id);
	}
	return omp_thread_id;
#endif /* TAU_OPENMP */
}

int OpenMPLayer::GetThreadId(void) 
{
#ifdef TAU_OPENMP
#ifdef TAU_OPENMP_NESTED
	threadId = omp_get_thread_num();
	int level = omp_get_level() - 1;
	int depth = 0;
	int width = omp_get_team_size(level+1);
	while (level >= 0)
	{
		depth++;
		threadId += omp_get_ancestor_thread_num(level)*width;
		width *= omp_get_team_size(level);
		level--;
	}
  return threadId;
#else
	if (omp_get_nested())
	{
		//OPENMP thread identification not supported by compiler.
		printf("ERROR: OpenMP nesting not supported. Please use a compiler that supports OMP specification >= 3.0 or rerun with OMP_NESTED=FALSE.\n");
		exit(1);
	}
 	return omp_get_thread_num();
#endif /* TAU_OPENMP_NESTED */
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
  // Initialize the mutex
  omp_init_lock(&OpenMPLayer::tauDBmutex);
  //cout <<" Initialized the functionDB Mutex data " <<endl;
  return 1;
}

////////////////////////////////////////////////////////////////////////
int OpenMPLayer::InitializeRegisterMutexData(void)
{
  // For locking thread registration process 
  // Initialize the mutex
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
  // Lock the functionDB mutex
  //fprintf(stderr, "Thread %d locking DB\n", omp_get_thread_num());
  omp_set_lock(&OpenMPLayer::tauDBmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockDB() unlocks the mutex tauDBMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockDB(void)
{
  // Unlock the functionDB mutex
  //fprintf(stderr, "Thread %d unlocking DB\n", omp_get_thread_num());
  omp_unset_lock(&OpenMPLayer::tauDBmutex);
  return 1;
}  

////////////////////////////////////////////////////////////////////////
int OpenMPLayer::InitializeEnvMutexData(void)
{
  // For locking functionEnv 
  // Initialize the mutex
  omp_init_lock(&OpenMPLayer::tauEnvmutex);
  //cout <<" Initialized the functionEnv Mutex data " <<endl;
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
  // Lock the functionEnv mutex
  //fprintf(stderr, "Thread %d locking Env\n", omp_get_thread_num());
  omp_set_lock(&OpenMPLayer::tauEnvmutex);
  return 1;
}

////////////////////////////////////////////////////////////////////////
// UnLockEnv() unlocks the mutex tauEnvMutex used by the above lock operation
////////////////////////////////////////////////////////////////////////
int OpenMPLayer::UnLockEnv(void)
{
  // Unlock the functionEnv mutex
  //fprintf(stderr, "Thread %d unlocking Env\n", omp_get_thread_num());
  omp_unset_lock(&OpenMPLayer::tauEnvmutex);
  return 1;
}  


/***************************************************************************
 * $RCSfile: OpenMPLayer.cpp,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2009/01/16 00:46:52 $
 * POOMA_VERSION_ID: $Id: OpenMPLayer.cpp,v 1.6 2009/01/16 00:46:52 amorris Exp $
 ***************************************************************************/


