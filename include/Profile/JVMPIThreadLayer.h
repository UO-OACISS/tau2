/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009          				   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: JVMPIThreadLayer.h				  **
**	Description 	: TAU Profiling Package Java Thread Support Layer **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _JVMPITHREADLAYER_H_
#define _JVMPITHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class JVMPIThreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef JAVA

#include <jvmpi.h>
class  JVMPIThreadLayer
{ // Layer for JVMPIThreadLayer to interact with Java Threads 
  public:
 	
 	JVMPIThreadLayer () { }  // defaults
	~JVMPIThreadLayer () { } 

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

#endif // JAVA 
#endif // _JVMPITHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: JVMPIThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.5 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: JVMPIThreadLayer.h,v 1.5 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


