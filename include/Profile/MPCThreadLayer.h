/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997-2009					   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: MPCThreadLayer.h				  **
 **	Description 	: TAU Profiling Package MPC Support Layer	  **
 *	Contact		: tau-team@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

#ifndef _MPCTHREADLAYER_H_
#define _MPCTHREADLAYER_H_

#include <mpc.h>

extern "C" void mpc_init_once(void);

class MPCThreadLayer
{
public:

  MPCThreadLayer() {
    InitializeThreadData();
  }

  static int RegisterThread(void);    // called before any profiling code
  static int InitializeThreadData(void);     // init thread mutexes
  static int InitializeDBMutexData(void);     // init tauDB mutex
  static int InitializeEnvMutexData(void);     // init tauEnv mutex
  static int GetThreadId(void); 	 // gets 0..N-1 thread id
  static int LockDB(void);    // locks the tauDBMutex
  static int UnLockDB(void);    // unlocks the tauDBMutex
  static int LockEnv(void);    // locks the tauEnvMutex
  static int UnLockEnv(void);    // unlocks the tauEnvMutex

private:
  static int tauThreadCount;
  static mpc_thread_once_t initFlag;
  static pthread_key_t tauThreadId;
  static sctk_thread_mutex_t tauThreadCountMutex;
  static sctk_thread_mutex_t tauDBMutex;
  static sctk_thread_mutex_t tauEnvMutex;

  friend void mpc_init_once(void);
};

#endif /* _MPCTHREADLAYER_H_ */

/***************************************************************************
 * $RCSfile: MPCThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.9 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: MPCThreadLayer.h,v 1.9 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/

