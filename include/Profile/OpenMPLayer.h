/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: OpenMPLayer.h  				  **
**	Description 	: TAU Profiling Package OpenMP Support Layer	  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _TAU_OPENMP_LAYER_H_
#define _TAU_OPENMP_LAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class OpenMPLayer
//
// This class is used for supporting OpenMP Threads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef TAU_OPENMP
extern "C" {
#include <omp.h>
#ifndef _OPENMP
#define _OPENMP
#endif /* _OPENMP */
}
class OpenMPLayer 
{ // Layer for RtsLayer to interact with OpenMP
  public:
 	
 	OpenMPLayer () { }  // defaults
	~OpenMPLayer () { } 

	static int RegisterThread(void); // called before any profiling code
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
        static int InitializeEnvMutexData(void);     // init tauEnv mutex
	static int GetThreadId(void); 	 // gets 0..N-1 thread id
	static int TotalThreads(void);   // gets number of threads
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex
	static int LockEnv(void);	 // locks the tauEnvMutex
	static int UnLockEnv(void);	 // unlocks the tauEnvMutex

  private:
	static omp_lock_t tauDBmutex;  // to protect TheFunctionDB
	static omp_lock_t tauEnvmutex;  // second lock
	
};
#endif // TAU_OPENMP

#endif // _TAU_OPENMP_LAYER_H_

	

/***************************************************************************
 * $RCSfile: OpenMPLayer.h,v $   $Author: amorris $
 * $Revision: 1.6 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: OpenMPLayer.h,v 1.6 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


