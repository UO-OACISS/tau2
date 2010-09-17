/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009          				   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: JavaThreadLayer.h				  **
**	Description 	: TAU Profiling Package Java Thread Support Layer **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _JAVATHREADLAYER_H_
#define _JAVATHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class PthreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef JAVA
#ifndef TAU_JVMTI
#include <jvmpi.h>
class  JavaThreadLayer
{ // Layer for JavaThreadLayer to interact with Java Threads 
  public:
 	
 	JavaThreadLayer () { }  // defaults
	~JavaThreadLayer () { } 

	static int * RegisterThread(JNIEnv * env_id); 
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
        static int InitializeEnvMutexData(void);     // init tauEnv mutex
	static int GetThreadId(void); 	 	 // gets 0..N-1 thread id
	static int GetThreadId(JNIEnv *env_id);	 // gets 0..N-1 thread id
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex
	static int LockEnv(void);	 // locks the tauEnvMutex
	static int UnLockEnv(void);	 // unlocks the tauEnvMutex
	static int TotalThreads(void); 	 // returns the thread count
        // return the current thread's cpu time, in nanoseconds (as reported by jvmpi)
        static jlong getCurrentThreadCpuTime(void); 

	static JVMPI_Interface 	    *tau_jvmpi_interface;
	static JavaVM 	   	    *tauVM; 	     // Virtual machine 
  private:
        static int		    tauThreadCount;  // Number of threads
	static JVMPI_RawMonitor     tauNumThreadsLock; // to protect counter
	static JVMPI_RawMonitor     tauDBMutex; // to protect counter
	static JVMPI_RawMonitor     tauEnvMutex; // second mutex
};

#else //TAU_JVMTI
#include <jvmti.h>
class  JavaThreadLayer
{ // Layer for JavaThreadLayer to interact with Java Threads 
  public:
 	
 	JavaThreadLayer () { }  // defaults
	~JavaThreadLayer () { } 

	static int * RegisterThread(jthread this_thread); 
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

#endif // TAU_JVMTI
#endif // JAVA 
#endif // _JAVATHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: JavaThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.5 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: JavaThreadLayer.h,v 1.5 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


