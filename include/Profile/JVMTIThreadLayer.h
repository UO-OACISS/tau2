/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009          				   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: JVMTIThreadLayer.h				  **
**	Description 	: TAU Profiling Package Java Thread Support Layer **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _JVMTITHREADLAYER_H_
#define _JVMTITHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class JVMTIThreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef JAVA

#include <jvmti.h>
class  JVMTIThreadLayer
{ // Layer for JVMTIThreadLayer to interact with Java Threads 
  public:
 	
 	JVMTIThreadLayer () { }  // defaults
	~JVMTIThreadLayer () { } 

	static int * RegisterThread(jthread this_thread=NULL); 
	static int ThreadEnd(jthread this_thread);
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
        static int InitializeEnvMutexData(void);     // init tauEnv mutex
	static int GetThreadId(jthread this_thread=NULL); 	 	 // gets 0..N-1 thread id
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex
	static int LockEnv(void);	 // locks the tauEnvMutex
	static int UnLockEnv(void);	 // unlocks the tauEnvMutex
	static int TotalThreads(void); 	 // returns the thread count
        // return the current thread's cpu time, in nanoseconds (as reported by jvmpi)
        static jlong getCurrentThreadCpuTime(void); 
	static jvmtiEnv 	    *jvmti;
  private:
        static int		    tauThreadCount;  // Number of threads
	static jrawMonitorID     tauNumThreadsLock; // to protect counter
	static jrawMonitorID     tauDBMutex; // to protect counter
	static jrawMonitorID     tauEnvMutex; // second mutex
};

#endif // JAVA 
#endif // _JVMTITHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: JVMTIThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.5 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: JVMTIThreadLayer.h,v 1.5 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


