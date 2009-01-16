/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: WindowsThreadLayer.h				  **
**	Description 	: Microsoft Windows Thread Support Layer	  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#ifndef _WINDOWSTHREADLAYER_H_
#define _WINDOWSTHREADLAYER_H_

//////////////////////////////////////////////////////////////////////
//
// class WindowsThreadLayer
//
// This class is used for supporting Microsoft Windows' threads in RtsLayer class.
//////////////////////////////////////////////////////////////////////


#ifdef TAU_WINDOWS

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////
#include <windows.h>


class WindowsThreadLayer 
{ // Layer for RtsLayer to interact with Microsoft Windows' threads. 
  public:
 	
 	WindowsThreadLayer () { }  // defaults
	~WindowsThreadLayer () { } 

	static int RegisterThread(void); // called before any profiling code
        static int InitializeThreadData(void);     // init thread mutexes
        static int InitializeDBMutexData(void);     // init tauDB mutex
        static int InitializeEnvMutexData(void);     // init tauDB mutex
	static int GetThreadId(void); 	 // gets 0..N-1 thread id
	static int LockDB(void);	 // locks the tauDBMutex
	static int UnLockDB(void);	 // unlocks the tauDBMutex
	static int LockEnv(void);	 // locks the tauEnvMutex
	static int UnLockEnv(void);	 // unlocks the tauEnvMutex

  private:
	static DWORD		 	   tauWindowsthreadId; // tid 
	static HANDLE			   tauThreadcountMutex; // to protect counter 
	static int 				   tauThreadCount;     // counter
	static HANDLE			   tauDBMutex;  // to protect TheFunctionDB
	static HANDLE			   tauEnvMutex;  // to protect TheFunctionDB
	
};

#endif //TAU_WINDOWS

#endif // _WINDOWSTHREADLAYER_H_
