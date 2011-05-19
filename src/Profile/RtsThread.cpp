/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: RtsThread.cpp				          **
**	Description 	: TAU Profiling Package RTS Layer definitions     **
**			  for supporting threads 			  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include "Profile/Profiler.h"
#include "Profile/OpenMPLayer.h"

#include <Profile/TauTrace.h>
void TraceCallStack(int tid, Profiler *current);

#include <stdio.h>

#ifdef TAUKTAU
#include <Profile/KtauProfiler.h>
#ifdef TAUKTAU_MERGE
#include <Profile/KtauFuncInfo.h>
#endif //TAUKTAU_MERGE
#ifdef TAUKTAU_SHCTR
#include <Profile/KtauCounters.h>
#endif //TAUKTAU_SHCTR
#endif //TAUKTAU

#include <Profile/TauSampling.h>


int RtsLayer::lockDBcount[TAU_MAX_THREADS];
int RtsLayer::lockEnvCount[TAU_MAX_THREADS];

//////////////////////////////////////////////////////////////////////
// myNode() returns the current node id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myNode(void)
{
#ifdef TAU_PID_AS_NODE
  return getpid();
#endif
  return TheNode();
}


//////////////////////////////////////////////////////////////////////
// myContext() returns the current context id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myContext(void)
{
  return TheContext(); 
}

extern "C" int Tau_RtsLayer_myThread(void) {
  return RtsLayer::myThread();
}


//////////////////////////////////////////////////////////////////////
// myNode() returns the current node id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myThread(void)
{
#ifdef PTHREADS
  return PthreadLayer::GetThreadId();
#elif  TAU_SPROC
  return SprocLayer::GetThreadId();
#elif  TAU_WINDOWS
  return WindowsThreadLayer::GetThreadId();
#elif  TULIPTHREADS
  return TulipThreadLayer::GetThreadId();
#elif JAVA
  return JavaThreadLayer::GetThreadId(); 
	// C++ app shouldn't use this unless there's a VM
#elif TAU_OPENMP
  return OpenMPLayer::GetThreadId();
#elif TAU_PAPI_THREADS
  return PapiThreadLayer::GetThreadId();
#else  // if no other thread package is available 
  return 0;
#endif // PTHREADS
}

int RtsLayer::setMyThread(int tid) {
#ifdef PTHREADS
  PthreadLayer::SetThreadId(tid);
#endif // PTHREADS
  return 0;
}

int RtsLayer::getNumThreads() {
  return *(RtsLayer::numThreads());
}

int *RtsLayer::numThreads() {
  static int numthreads = 1;
  return &numthreads;
}


//////////////////////////////////////////////////////////////////////
// RegisterThread is called before any other profiling function in a 
// thread that is spawned off
//////////////////////////////////////////////////////////////////////
int RtsLayer::RegisterThread() {
  /* Check the size of threads */
  LockEnv();
  int numthreads = *(RtsLayer::numThreads());
  numthreads ++;
  if (numthreads >= TAU_MAX_THREADS) {
    fprintf(stderr, "TAU: RtsLayer: Max thread limit (%d) exceeded. Please re-configure TAU with -useropt=-DTAU_MAX_THREADS=<higher limit>\n", numthreads);
  }
  UnLockEnv();
 
#ifndef TAU_WINDOWS 
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init(numthreads-1);
  }
#endif

#ifdef PTHREADS
  PthreadLayer::RegisterThread();
#elif TAU_SPROC
  SprocLayer::RegisterThread();
#elif  TAU_WINDOWS
  WindowsThreadLayer::RegisterThread();
#elif  TULIPTHREADS
  TulipThreadLayer::RegisterThread();
#elif TAU_OPENMP
  OpenMPLayer::RegisterThread();
#elif TAU_PAPI_THREADS
  PapiThreadLayer::RegisterThread();
#endif // PTHREADS
// Note: Java thread registration is done at the VM layer in TauJava.cpp

  *(RtsLayer::numThreads()) = *(RtsLayer::numThreads()) + 1;
  return numthreads;
}


//////////////////////////////////////////////////////////////////////
// RegisterFork is called before any other profiling function in a 
// process that is forked off (child process)
//////////////////////////////////////////////////////////////////////
void RtsLayer::RegisterFork(int nodeid, enum TauFork_t opcode) {
#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  Profiler *current;
#endif // PROFILING_ON

#ifdef TAUKTAU
  //If KTAU profiling (esp. merged, but even non-merged) is on
  //then we ALWAYS do EXCLUDE_PARENT - i.e. KTAU doesnt currently 
  //support INCLUDE_PARENT. Unlike in the case of TAU, there is a
  // LOT of extra work that needs to be done in KTAU for INCLUDE.
  // - TODO. : AN
  opcode = TAU_EXCLUDE_PARENT_DATA;
  DEBUGPROFMSG("KTAU Profiling On. Currently only supports EXCLUDE-PARENT on RegisterFork." << endl;);
#endif //TAUKTAU

#ifdef TAU_PAPI
  // PAPI must be reinitialized in the child
  PapiLayer::reinitializePAPI();
#endif

#ifdef TAUKTAU_SHCTR
     KtauCounters::RegisterFork(opcode);//forking needs to be tested further
#endif	// TAUKTAU_SHCTR

  TAU_PROFILE_SET_NODE(nodeid);
  // First, set the new id 
  if (opcode == TAU_EXCLUDE_PARENT_DATA) {
  // If opcode is TAU_EXCLUDE_PARENT_DATA then we clear out the 
  // previous values in the TheFunctionDB()

  // Get the current time
     double CurrentTimeOrCounts[TAU_MAX_COUNTERS];
     for(int i=0;i<Tau_Global_numCounters;i++){
       CurrentTimeOrCounts[i]=0;
     }
     getUSecD(myThread(), CurrentTimeOrCounts);
     for (int tid = 0; tid < TAU_MAX_THREADS; tid++) { 
       // For each thread of execution 
#ifdef PROFILING_ON
       for(it=TheFunctionDB().begin(); it!=TheFunctionDB().end(); it++) { 
	 // Iterate through each FunctionDB item 
	 // Clear all values 
	 (*it)->SetCalls(tid, 0);
	 (*it)->SetSubrs(tid, 0);
         (*it)->SetExclTimeZero(tid);
         (*it)->SetInclTimeZero(tid);
	/* Do we need to change AlreadyOnStack? No*/
	DEBUGPROFMSG("FI Zap: Inside "<< (*it)->GetName() <<endl;);
#ifdef TAUKTAU_MERGE
	DEBUGPROFMSG("RtsLayer::RegisterFork: GetKtauFuncInfo(tid)->ResetAllCounters(tid): Func:"<< (*it)->GetName() <<endl;);
	(*it)->GetKtauFuncInfo(tid)->ResetAllCounters(tid);
#endif //TAUKTAU_MERGE
       }
#ifdef TAUKTAU_MERGE
       DEBUGPROFMSG("RtsLayer::RegisterFork: KtauFuncInfo::ResetAllGrpTotals(tid)"<<endl;);
       KtauFuncInfo::ResetAllGrpTotals(tid);
#endif //TAUKTAU_MERGE
       DEBUGPROFMSG("RtsLayer::RegisterFork: Running-Up Stack\n");
       // Now that the FunctionDB is cleared, we need to add values to it 
       //	corresponding to the present state.
       current = TauInternal_CurrentProfiler(tid);
       while (current != 0) { 
	 // Iterate through each profiler on the callstack and 
	 // fill Values in it 
	 DEBUGPROFMSG("P Correct: Inside "<< current->ThisFunction->GetName() 
		      <<endl;);
	 current->ThisFunction->IncrNumCalls(tid);
	 if (current->ParentProfiler != 0) { 
	   // Increment the number of called functions in its parent
	   current->ParentProfiler->ThisFunction->IncrNumSubrs(tid);
	 }
	 for(int j=0;j<Tau_Global_numCounters;j++){
	   current->StartTime[j] = CurrentTimeOrCounts[j];
	 }
	 current = current->ParentProfiler;
       } // Until the top of the stack
#endif   // PROFILING_ON
       
       
       if (TauEnv_get_tracing()) {
	 DEBUGPROFMSG("Tracing Correct: "<<endl;);
	 TauTraceUnInitialize(tid); // Zap the earlier contents of the trace buffer  
	 TraceCallStack(tid, TauInternal_CurrentProfiler(tid)); 
       }
       
#ifdef TAUKTAU
       DEBUGPROFMSG("RtsLayer::RegisterFork: CurrentProfiler:"<<TauInternal_CurrentProfiler(tid)<<endl;);
       if (TauInternal_CurrentProfiler(tid) != NULL) {
	 TauInternal_CurrentProfiler(tid)->ThisKtauProfiler->RegisterFork(TauInternal_CurrentProfiler(tid), tid, nodeid, opcode);
       }
#endif //TAUKTAU

     } // for tid loop
     // DONE! 
   }
   // If it is TAU_INCLUDE_PARENT_DATA then there's no need to do anything.
   // fork would copy over all the parent data as it is. 
}

bool RtsLayer::initLocks(void) {
  threadLockDB();
  for (int i=0; i<TAU_MAX_THREADS; i++) {
    lockDBcount[i] = 0;
  }
  threadUnLockDB();
  return true;
}

bool RtsLayer::initEnvLocks(void) {
  threadLockEnv();
  for (int i=0; i<TAU_MAX_THREADS; i++) {
    lockEnvCount[i] = 0;
  }
  threadUnLockEnv();
  return true;
}

//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionDB (global) is locked while updating
//////////////////////////////////////////////////////////////////////

extern "C" void Tau_RtsLayer_LockDB() {
  RtsLayer::LockDB();
}

extern "C" void Tau_RtsLayer_UnLockDB() {
  RtsLayer::UnLockDB();
}

void RtsLayer::LockDB(void) {
  static bool init = initLocks();
  int tid=myThread();
  if (lockDBcount[tid] == 0) {
    threadLockDB();
  }
  lockDBcount[tid]++;
  return;
}

void RtsLayer::UnLockDB(void) {
  int tid=myThread();
  lockDBcount[tid]--;
  if (lockDBcount[tid] == 0) {
    threadUnLockDB();
  }
}

void RtsLayer::threadLockDB(void) {
#ifdef PTHREADS
  PthreadLayer::LockDB();
#elif TAU_SPROC
  SprocLayer::LockDB();
#elif  TAU_WINDOWS
  WindowsThreadLayer::LockDB();
#elif  TULIPTHREADS
  TulipThreadLayer::LockDB();
#elif  JAVA
  JavaThreadLayer::LockDB();
#elif TAU_OPENMP
  OpenMPLayer::LockDB();
#elif TAU_PAPI_THREADS
  PapiThreadLayer::LockDB();
#endif
  return ; // do nothing if threads are not used
}



//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionDB (global) is locked while updating
//////////////////////////////////////////////////////////////////////
void RtsLayer::threadUnLockDB(void) {
#ifdef PTHREADS
  PthreadLayer::UnLockDB();
#elif TAU_SPROC
  SprocLayer::UnLockDB();
#elif  TAU_WINDOWS
  WindowsThreadLayer::UnLockDB();
#elif  TULIPTHREADS
  TulipThreadLayer::UnLockDB();
#elif JAVA
  JavaThreadLayer::UnLockDB();
#elif TAU_OPENMP
  OpenMPLayer::UnLockDB();
#elif TAU_PAPI_THREADS
  PapiThreadLayer::UnLockDB();
#endif
  return;
}

void RtsLayer::LockEnv(void) {
  static bool init = initEnvLocks();
  int tid=myThread();
  if (lockEnvCount[tid] == 0) {
    threadLockEnv();
  }
  lockEnvCount[tid]++;
  return;
}

void RtsLayer::UnLockEnv(void) {
  int tid=myThread();
  lockEnvCount[tid]--;
  if (lockEnvCount[tid] == 0) {
    threadUnLockEnv();
  }
}

//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionEnv (global) is locked while updating
//////////////////////////////////////////////////////////////////////

void RtsLayer::threadLockEnv(void)
{
#ifdef PTHREADS
  PthreadLayer::LockEnv();
#elif TAU_SPROC
  SprocLayer::LockEnv();
#elif  TAU_WINDOWS
  WindowsThreadLayer::LockEnv();
#elif  TULIPTHREADS
  TulipThreadLayer::LockEnv();
#elif  JAVA
  JavaThreadLayer::LockEnv();
#elif TAU_OPENMP
  OpenMPLayer::LockEnv();
#elif TAU_PAPI_THREADS
  PapiThreadLayer::LockEnv();
#endif // PTHREADS
  return ; // do nothing if threads are not used
}


//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionEnv (global) is locked while updating
//////////////////////////////////////////////////////////////////////
void RtsLayer::threadUnLockEnv(void)
{
#ifdef PTHREADS
  PthreadLayer::UnLockEnv();
#elif TAU_SPROC
  SprocLayer::UnLockEnv();
#elif  TAU_WINDOWS
  WindowsThreadLayer::UnLockEnv();
#elif  TULIPTHREADS
  TulipThreadLayer::UnLockEnv();
#elif JAVA
  JavaThreadLayer::UnLockEnv();
#elif TAU_OPENMP
  OpenMPLayer::UnLockEnv();
#elif TAU_PAPI_THREADS
  PapiThreadLayer::UnLockEnv();
#endif // PTHREADS
  return;
}


/***************************************************************************
 * $RCSfile: RtsThread.cpp,v $   $Author: amorris $
 * $Revision: 1.42 $   $Date: 2010/04/08 23:08:13 $
 * VERSION: $Id: RtsThread.cpp,v 1.42 2010/04/08 23:08:13 amorris Exp $
 ***************************************************************************/


