/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009          				   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: JNIThreadLayer.h				  **
**	Description 	: TAU Profiling Package Java Thread Support Layer **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _JNITHREADLAYER_H_
#define _JNITHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class JNIThreadLayer
//
// This class is used for supporting pthreads in RtsLayer class.
//////////////////////////////////////////////////////////////////////

#ifdef JAVA

#include <sys/types.h>
#include <jni.h>
#include <map>
#include <mutex>
class  JNIThreadLayer
{ // Layer for JNIThreadLayer to interact with Java Threads 
  public:
 	
 	JNIThreadLayer () { }  // defaults
	~JNIThreadLayer () { } 

	static void WaitForDTM(void);
	static void SuThread(pid_t sid, char *tname);
	static void SuThread(pid_t tid);
	static void IgnoreThisThread(void);
	static bool IsMgmtThread(void);
	static int RegisterThread(int sid, char *tname);
	static char *GetThreadName(void);
	static pid_t GetThreadSid(void);
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
        static int InitializeEnvMutexData(void);     // init tauEnv mutex
	static int GetThreadId(void); 	 	 // gets 0..N-1 thread id
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex
	static int LockEnv(void);	 // locks the tauEnvMutex
	static int UnLockEnv(void);	 // unlocks the tauEnvMutex
	static int TotalThreads(void); 	 // returns the thread count
        // return the current thread's cpu time, in nanoseconds (as reported by jvmpi)
        static jlong getCurrentThreadCpuTime(void); 

	static JavaVM 	   	    *tauVM; 	     // Virtual machine 
  private:
        static int                      tauThreadCount;  // Number of threads
	static std::map<pid_t, int>     tauTidMap;
	static std::mutex               tauNumThreadsLock; // to protect counter
	static std::map<pid_t, std::mutex> tauDBMutex; // to protect counter
	static std::map<pid_t, std::mutex> tauEnvMutex; // second mutex
	static std::map<pid_t, std::mutex> tauDTMLock;
};

#endif // JAVA 
#endif // _JNITHREADLAYER_H_

	

/***************************************************************************
 * $RCSfile: JNIThreadLayer.h,v $   $Author: amorris $
 * $Revision: 1.5 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: JNIThreadLayer.h,v 1.5 2009/01/16 00:46:32 amorris Exp $
 ***************************************************************************/


