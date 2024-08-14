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

#include <stdio.h>
#include <stdlib.h>

#ifndef TAU_WINDOWS
#include <unistd.h>
#include <sys/syscall.h>

#ifdef SYS_gettid
pid_t tid = syscall(SYS_gettid);
#define gettid() syscall(SYS_gettid)
#else
#error "SYS_gettid unavailable on this system"
#endif
#endif /*TAU_WINDOWS*/

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <thread>

#include <Profile/Profiler.h>
#include <Profile/OpenMPLayer.h>
#include <Profile/TauTrace.h>
#include <Profile/TauSampling.h>

#ifdef TAUKTAU
#include <Profile/KtauProfiler.h>
#ifdef TAUKTAU_MERGE
#include <Profile/KtauFuncInfo.h>
#endif //TAUKTAU_MERGE
#ifdef TAUKTAU_SHCTR
#include <Profile/KtauCounters.h>
#endif //TAUKTAU_SHCTR
#endif //TAUKTAU

#ifdef TAU_MPC
#include <Profile/MPCThreadLayer.h>
#endif /* TAU_MPC */

// This is used for printing the stack trace when debugging locks
//#define DEBUG_LOCK_PROBLEMS
#ifdef DEBUG_LOCK_PROBLEMS
//#define DEBUG_LOCK_PROBLEMS_disabled
#include <execinfo.h>
thread_local bool verbose{false};
#endif //DEBUG_LOCK_PROBLEMS

// This is a hack for all the deadlocks in TAU.
// A new threading layer is being built that will do away with
// this locking model and make this hack obsolete.
#ifndef TAU_ENVLOCK_IS_DBLOCK
#define TAU_ENVLOCK_IS_DBLOCK
#endif

using namespace std;
using namespace tau;

#include <Profile/TauPin.h>

int& RtsLayer::lockDBCount() {
    static thread_local int count{0};
    return count;
}


int& RtsLayer::lockEnvCount() {
    static thread_local int count{0};
    return count;
}

/*
inline void decrementDBLock(int tid){
        RtsLayer::lockDBCounty[tid]--;
}

*/
void TraceCallStack(int tid, Profiler *current);


//////////////////////////////////////////////////////////////////////
// Thread struct
//////////////////////////////////////////////////////////////////////
class RtsThread
{
private:
    static const std::thread::id MAIN_THREAD_ID;
public:

	static int num_threads;
	int thread_rank;
	bool recyclable;
	bool active;
	int next_available;

	RtsThread()
	{
		thread_rank = ++num_threads;
		recyclable = false;
		active = true;
		next_available = thread_rank + 1;
	  //printf("creating new thread obj, rank: %d, next: %d.\n", thread_rank,
			//next_available);
	}

    // Use whatever means necessary to see if this thread is the main thread
    static bool isMainThread() {
        return (std::this_thread::get_id() == MAIN_THREAD_ID);
    }
};

int RtsThread::num_threads = 0;
const std::thread::id RtsThread::MAIN_THREAD_ID{std::this_thread::get_id()};

bool RtsLayer::isMainThread() {
#if defined(__gnu_linux__) || defined(__linux) || defined(__linux__) || defined(linux)
    // if posix support availalbe, use that.
    return (RtsLayer::getPid() == RtsLayer::getTid());
#else
    // otherwise, use the C++ method and hope that main thread initialized our static var...
    return RtsThread::isMainThread();
#endif
}

class TauThreadList : public vector<RtsThread*> {
public:
    virtual ~TauThreadList() {
        Tau_destructor_trigger();
    }
};

TauThreadList & TheThreadList(void)
{
	static TauThreadList ThreadList;

	return ThreadList;
}


static unsigned int nextThread = 1;

int RtsLayer::_createThread()
{
	RtsThread* newThread;

	if (nextThread >= TheThreadList().size())
	{
		newThread = new RtsThread();
		TheThreadList().push_back(newThread);
		nextThread = newThread->next_available;
	}
	else
	{
		newThread = TheThreadList().at(nextThread);
		newThread->active = true;
		nextThread = newThread->next_available;
	}

	return newThread->thread_rank;
}

int RtsLayer::createThread()
{
  TauInternalFunctionGuard protects_this_function;

  threadLockEnv();

	int tid = RtsLayer::_createThread();

  threadUnLockEnv();

  int numThreads = getTotalThreads();

  /*if (numThreads > TAU_MAX_THREADS) {
    fprintf(stderr,
        "TAU Error: RtsLayer: [Max thread limit = %d] [Encountered = %d]. Please re-configure TAU with -useropt=-DTAU_MAX_THREADS=<higher limit> or set the environment variable TAU_RECYCLE_THREADS=1\n",
        TAU_MAX_THREADS, numThreads);
    exit(-1);
  }*/
  return tid;
}

extern "C" int Tau_RtsLayer_createThread() {
	return RtsLayer::createThread();
}

void RtsLayer::recycleThread(int id)
{
  TauInternalFunctionGuard protects_this_function;
  LockEnv();

	TheThreadList().at(id-1)->active = false;
	TheThreadList().at(id-1)->next_available = nextThread;
	nextThread = id-1;

	UnLockEnv();
}

int RtsLayer::localThreadId(void)
{
  TauInternalFunctionGuard protects_this_function;
  return RtsLayer::unsafeLocalThreadId();
}

int RtsLayer::unsafeLocalThreadId(void)
{
#ifdef TAU_MPC
  return MPCThreadLayer::GetThreadId();
#elif PTHREADS
  return PthreadLayer::GetThreadId();
#elif  TAU_SPROC
  return SprocLayer::GetThreadId();
#elif  TAU_WINDOWS
  return WindowsThreadLayer::GetThreadId();
#elif  TULIPTHREADS
  return TulipThreadLayer::GetThreadId();
#elif JAVA
  if (TheUsingJNI() == true) {
      return JNIThreadLayer::GetThreadId();
  } else {
      return JVMTIThreadLayer::GetThreadId();
  }
	// C++ app shouldn't use this unless there's a VM
#elif TAU_OPENMP
  return OpenMPLayer::GetThreadId();
#elif TAU_PAPI_THREADS
  return PapiThreadLayer::GetThreadId();
#else  // if no other thread package is available
  return 0;
#endif // PTHREADS
}

int RtsLayer::unsafeThreadId(void)
{
#ifdef TAU_MPC
  return MPCThreadLayer::GetThreadId();
#elif PTHREADS
  return PthreadLayer::GetThreadId();
#elif  TAU_SPROC
  return SprocLayer::GetThreadId();
#elif  TAU_WINDOWS
  return WindowsThreadLayer::GetThreadId();
#elif  TULIPTHREADS
  return TulipThreadLayer::GetThreadId();
#elif JAVA
  if (TheUsingJNI() == true) {
      return JNIThreadLayer::GetThreadId();
  } else {
      return JVMTIThreadLayer::GetThreadId();
  }
	// C++ app shouldn't use this unless there's a VM
#elif TAU_OPENMP
  return OpenMPLayer::GetTauThreadId();
#elif TAU_PAPI_THREADS
  return PapiThreadLayer::GetThreadId();
#else  // if no other thread package is available
  return 0;
#endif // PTHREADS
}

int RtsLayer::myThread(void)
{
  TauInternalFunctionGuard protects_this_function;
  return RtsLayer::unsafeThreadId();
}

extern "C" int Tau_RtsLayer_myThread(void) {
	return RtsLayer::myThread();
}

int RtsLayer::getTotalThreads()
{
  TauInternalFunctionGuard protects_this_function;
  LockEnv();
  // *CWL* - The Thread vector does NOT include the main thread!!
  int numThreads = TheThreadList().size() + 1;
  UnLockEnv();
  return numThreads;
}

#ifdef TAU_MPC
extern "C" int TauGetMPCProcessRank(void);
#endif /* TAU_MPC */

//////////////////////////////////////////////////////////////////////
// myNode() returns the current node id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myNode(void)
{
#ifdef TAU_MPC
  return TauGetMPCProcessRank();
#endif /* TAU_MPC */

#ifdef TAU_PID_AS_NODE
  return RtsLayer::getPid();
#endif
#ifdef KTAU_NG
#ifdef TAU_TID_AS_NODE
  return RtsLayer::getLinuxKernelTid(); //voorhees
#endif /* TAU_TID_AS_NODE */
#endif /* KTAU_NG */
  return TheNode();
}


//////////////////////////////////////////////////////////////////////
// myContext() returns the current context id (0..N-1)
//////////////////////////////////////////////////////////////////////
int RtsLayer::myContext(void)
{
#ifdef KTAU_NG
  return RtsLayer::getLinuxKernelTid(); //voorhees
#elif defined(__MIC__)
if (TauEnv_get_mic_offload())
{
	return RtsLayer::getPid();
}
else
#endif /* KTAU_NG */
  return TheContext();
}


//////////////////////////////////////////////////////////////////////
// RegisterThread is called before any other profiling function in a
// thread that is spawned off
//////////////////////////////////////////////////////////////////////
int RtsLayer::RegisterThread()
{
#ifdef TAU_MPC
  MPCThreadLayer::RegisterThread();
#elif PTHREADS
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

  // *CWL* - This is a fuzzy report. What is guaranteed is that AT LEAST ONE thread has
  //         pushed us over the limit with the last registration.
  //
  //         Because this is a guaranteed failure, we "gracefully" exit at this point
  //         rather than suffer a random segfault later.
  int numThreads = getTotalThreads();
  /*if (numThreads > TAU_MAX_THREADS) {
    fprintf(stderr,
        "TAU Error: RtsLayer: [Max thread limit = %d] [Encountered = %d]. Please re-configure TAU with -useropt=-DTAU_MAX_THREADS=<higher limit> or set the environment variable TAU_RECYCLE_THREADS=1\n",
        TAU_MAX_THREADS, numThreads);
    exit(-1);
  }*/

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init_if_necessary();
  }
#endif /* _AIX */
#endif

  return numThreads;
}


//////////////////////////////////////////////////////////////////////
// RegisterFork is called before any other profiling function in a
// process that is forked off (child process)
//////////////////////////////////////////////////////////////////////
void RtsLayer::RegisterFork(int nodeid, enum TauFork_t opcode) {
  TauInternalFunctionGuard protects_this_function;

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
     int threadCount=getTotalThreads();
     for (int tid = 0; tid < threadCount; tid++) {
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
void RtsLayer::Initialize(void) {
  return ; // do nothing if threads are not used
}

/* These functions should no longer be necessary with new thread local locck management
bool RtsLayer::initLocks(void) {
  threadLockDB();
  int threadCount=getLockVecSize();//TAU_MAX_THREADS;//getDBVecSize();//getTotalThreads();

  for (int i=0; i<threadCount; i++) {
    setDBLock(i, 0);
  }
  threadUnLockDB();
  return true;
}

bool RtsLayer::initEnvLocks(void) {
  threadLockEnv();
  int threadCount=getLockVecSize();//TAU_MAX_THREADS;//getEnvVecSize();//getTotalThreads();
  for (int i=0; i<threadCount; i++) {
	  setEnvLock(i,0);
  }
  threadUnLockEnv();
  return true;
}
*/
//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionDB (global) is locked while updating
//////////////////////////////////////////////////////////////////////

extern "C" void Tau_RtsLayer_LockDB() {
  RtsLayer::LockDB();
}

extern "C" void Tau_RtsLayer_UnLockDB() {
  RtsLayer::UnLockDB();
}

int RtsLayer::getNumDBLocks(void) {
  return lockDBCount();
}
#ifdef DEBUG_LOCK_PROBLEMS
constexpr int stack_depth=4;
#endif

int RtsLayer::LockDB(void) {
#ifdef DEBUG_LOCK_PROBLEMS
  thread_local static void* old_callstack[stack_depth];
  thread_local static int old_frames;
#endif
  // use the init value so the compiler doesn't complain
#ifndef TAU_WINDOWS
  int tid=gettid();
#else
  int tid=localThreadId();
#endif
/* This block of code is helpful in debugging deadlocks... see the top of this file */
	TAU_ASSERT(Tau_global_get_insideTAU() > 0, "Thread is trying for DB lock but it is not in TAU");
#ifdef DEBUG_LOCK_PROBLEMS
    int nid = RtsLayer::myNode();
  if (lockDBCount() > 0) {
    TAU_VERBOSE("WARNING! Thread %d,%d,%d has %d DB locks, trying for another DB lock\n", nid, tid, gettid(), lockDBCount());
    if(!TauEnv_get_ebs_enabled()) {
      int i;
      char** old_strs = backtrace_symbols(old_callstack, old_frames);
      TAU_VERBOSE("\n\n");
      TAU_VERBOSE("Lock %d: Old Callstack: \n", lockDBCount());
      for (i = 0; i < old_frames; ++i) {
        fprintf(stderr,"%d,%d: %s\n", nid, tid, old_strs[i]);
      }
      free(old_strs);
      void* callstack[stack_depth];
      int frames = backtrace(callstack, stack_depth);
      char** strs = backtrace_symbols(callstack, frames);
      TAU_VERBOSE("\n\n");
      TAU_VERBOSE("Lock %d: New Callstack: \n", lockDBCount());
      for (i = 0; i < frames; ++i) {
        TAU_VERBOSE("%d,%d: %s\n", nid, tid, strs[i]);
      }
      free(strs);
    }
    //abort();
  }
  old_frames = backtrace(old_callstack, stack_depth);
/*
  // check the OTHER lock
  if (lockEnvCount() > 0) {
    fprintf(stderr,"WARNING! Thread %d,%d has Env lock, trying for DB lock\n", nid, tid);
    if(!TauEnv_get_ebs_enabled()) {
      void* callstack[stack_depth];
      int i, frames = backtrace(callstack, stack_depth);
      char** strs = backtrace_symbols(callstack, frames);
      for (i = 0; i < frames; ++i) {
        fprintf(stderr,"%d,%d: %s\n", nid, tid, strs[i]);
      }
      free(strs);
    }
  }
*/
#endif

  if (lockDBCount() == 0) {
    threadLockDB();
  }
  lockDBCount()++;
/* This block of code is helpful in debugging deadlocks... see the top of this file */
#ifdef DEBUG_LOCK_PROBLEMS_disabled
      int i;
      void* callstack[stack_depth];
      int frames = backtrace(callstack, stack_depth);
      char** strs = backtrace_symbols(callstack, frames);
      fprintf(stderr,"\n\n");
      for (i = 0; i < frames; ++i) {
        fprintf(stderr,"%d,%d: %s\n", nid, tid, strs[i]);
      }
      free(strs);
#endif
#ifdef DEBUG_LOCK_PROBLEMS

  TAU_VERBOSE("Lock: THREAD %d,%d HAS %d DB LOCKS\n", RtsLayer::myNode(), tid, lockDBCount());
#endif
  return lockDBCount();
}

int RtsLayer::UnLockDB(void) {
#ifndef TAU_WINDOWS
  int tid=gettid();
#else
  int tid=localThreadId();
#endif
  lockDBCount()--;
  if (lockDBCount() == 0) {
    threadUnLockDB();
  } else {
    // protect against too many unlocks!
    if (lockDBCount() < 0) {
        lockDBCount() = 0;
    }
  }
/* This block of code is helpful in debugging deadlocks... see the top of this file */
#ifdef DEBUG_LOCK_PROBLEMS
  thread_local static void* old_callstack[stack_depth];
  thread_local static int old_frames;
  int nid = RtsLayer::myNode();
  if (lockDBCount() > 0) {
    if(!TauEnv_get_ebs_enabled()) {
      int i;
      char** old_strs = backtrace_symbols(old_callstack, old_frames);
      TAU_VERBOSE("\n\n");

      TAU_VERBOSE("Unlock %d: Old Callstack: \n", lockDBCount());
      for (i = 0; i < old_frames; ++i) {
        fprintf(stderr,"%d,%d: %s\n", nid, tid, old_strs[i]);
      }
      free(old_strs);
      void* callstack[stack_depth];
      int frames = backtrace(callstack, stack_depth);
      char** strs = backtrace_symbols(callstack, frames);
      TAU_VERBOSE("\n\n");
      TAU_VERBOSE("Unlock %d: New Callstack: \n", lockDBCount());
      for (i = 0; i < frames; ++i) {
        TAU_VERBOSE("%d,%d: %s\n", nid, tid, strs[i]);
      }
      free(strs);
    }
  }
  old_frames = backtrace(old_callstack, stack_depth);
  TAU_VERBOSE("Unlock: THREAD %d,%d HAS %d DB LOCKS\n", nid, tid, lockDBCount());
#endif
  return lockDBCount();
}

void RtsLayer::threadLockDB(void) {
#ifdef TAU_MPC
  MPCThreadLayer::LockDB();
#elif PTHREADS
  PthreadLayer::LockDB();
#elif TAU_SPROC
  SprocLayer::LockDB();
#elif  TAU_WINDOWS
  WindowsThreadLayer::LockDB();
#elif  TULIPTHREADS
  TulipThreadLayer::LockDB();
#elif  JAVA
  if (TheUsingJNI() == true) {
      JNIThreadLayer::LockDB();
  } else {
      JVMTIThreadLayer::LockDB();
  }
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
#ifdef TAU_MPC
  MPCThreadLayer::UnLockDB();
#elif PTHREADS
  PthreadLayer::UnLockDB();
#elif TAU_SPROC
  SprocLayer::UnLockDB();
#elif  TAU_WINDOWS
  WindowsThreadLayer::UnLockDB();
#elif  TULIPTHREADS
  TulipThreadLayer::UnLockDB();
#elif JAVA
  if (TheUsingJNI() == true) {
      JNIThreadLayer::UnLockDB();
  } else {
      JVMTIThreadLayer::UnLockDB();
  }
#elif TAU_OPENMP
  OpenMPLayer::UnLockDB();
#elif TAU_PAPI_THREADS
  PapiThreadLayer::UnLockDB();
#endif
  return;
}

int RtsLayer::getNumEnvLocks(void) {
  return lockEnvCount();
}

int RtsLayer::LockEnv(void)
{
#ifdef TAU_ENVLOCK_IS_DBLOCK
  return LockDB();
#else
  int tid=gettid();
	TAU_ASSERT(Tau_global_get_insideTAU() > 0, "Thread is trying for Env lock but it is not in TAU");
/* This block of code is helpful in debugging deadlocks... see the top of this file */
#ifdef DEBUG_LOCK_PROBLEMS
    int nid = RtsLayer::myNode();
  if (lockEnvCount() > 0) {
    TAU_VERBOSE("WARNING! Thread %d,%d has Env lock, trying for another Env lock\n", nid, tid);
    if(!TauEnv_get_ebs_enabled()) {
      void* callstack[stack_depth];
      int i, frames = backtrace(callstack, stack_depth);
      char** strs = backtrace_symbols(callstack, frames);
      for (i = 0; i < frames; ++i) {
        TAU_VERBOSE("%d,%d: %s\n", nid, tid, strs[i]);
      }
      free(strs);
    }
    abort();
  }
#endif
  //TAU_ASSERT(lockDBCount() == 0, "Thread has DB lock, trying for Env lock");
	if (lockEnvCount() == 0) {
    threadLockEnv();
  }
  lockEnvCount()++;
/* This block of code is helpful in debugging deadlocks... see the top of this file */
#ifdef DEBUG_LOCK_PROBLEMS_disabled
  fprintf(stderr,"THREAD %d,%d HAS %d ENV LOCKS (locking)\n", RtsLayer::myNode(), tid, lockEnvCount());
  fflush(stdout);
#endif
  return lockEnvCount();
#endif
}

int RtsLayer::UnLockEnv(void)
{
#ifdef TAU_ENVLOCK_IS_DBLOCK
  return UnLockDB();
#else
  int tid=gettid();
  lockEnvCount()--;
  if (lockEnvCount() == 0) {
    threadUnLockEnv();
  }
/* This block of code is helpful in debugging deadlocks... see the top of this file */
#ifdef DEBUG_LOCK_PROBLEMS_disabled
  TAU_VERBOSE("THREAD %d,%d HAS %d ENV LOCKS\n", RtsLayer::myNode(), tid, lockEnvCount());
#endif
  return lockEnvCount();
#endif
}

//////////////////////////////////////////////////////////////////////
// This ensure that the FunctionEnv (global) is locked while updating
//////////////////////////////////////////////////////////////////////

void RtsLayer::threadLockEnv(void)
{
#ifdef TAU_MPC
  MPCThreadLayer::LockEnv();
#elif PTHREADS
  PthreadLayer::LockEnv();
#elif TAU_SPROC
  SprocLayer::LockEnv();
#elif  TAU_WINDOWS
  WindowsThreadLayer::LockEnv();
#elif  TULIPTHREADS
  TulipThreadLayer::LockEnv();
#elif  JAVA
  if (TheUsingJNI() == true) {
      JNIThreadLayer::LockEnv();
  } else {
      JVMTIThreadLayer::LockEnv();
  }
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
#ifdef TAU_MPC
  MPCThreadLayer::UnLockEnv();
#elif PTHREADS
  PthreadLayer::UnLockEnv();
#elif TAU_SPROC
  SprocLayer::UnLockEnv();
#elif  TAU_WINDOWS
  WindowsThreadLayer::UnLockEnv();
#elif  TULIPTHREADS
  TulipThreadLayer::UnLockEnv();
#elif JAVA
  if (TheUsingJNI() == true) {
      JNIThreadLayer::UnLockEnv();
  } else {
      JVMTIThreadLayer::UnLockEnv();
  }
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


