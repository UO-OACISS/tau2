/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauCAPI.C					  **
**	Description 	: TAU Profiling Package API wrapper for C	  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#ifdef TAU_DOT_H_LESS_HEADERS 
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauSnapshot.h>
#include <Profile/TauTrace.h>

#if (!defined(TAU_WINDOWS))
/* Needed for fork */
#include <sys/types.h>
#include <unistd.h>
#endif /* TAU_WINDOWS */

#if defined(TAUKTAU)
#include <Profile/KtauProfiler.h>
#endif //TAUKTAU

#ifdef TAU_EPILOG
#include "elg_trc.h"

#ifdef TAU_SCALASCA
extern "C" {
void esd_enter (elg_ui4 rid);
void esd_exit (elg_ui4 rid);
}
#endif /* SCALASCA */
#endif /* TAU_EPILOG */

#ifdef TAU_VAMPIRTRACE
#include <Profile/TauVampirTrace.h>
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_SCOREP
#include <Profile/TauSCOREP.h>
#endif

#ifdef DEBUG_LOCK_PROBLEMS
#include <execinfo.h>
#endif
#if !defined(TAU_WINDOWS) && !defined(TAU_ANDROID)
#include <execinfo.h>
#endif

using namespace tau;

extern "C" void Tau_shutdown(void);

#define TAU_GEN_CONTEXT_EVENT(e, msg) TauContextUserEvent* e () { \
	static TauContextUserEvent ce(msg); return &ce; } 

TAU_GEN_CONTEXT_EVENT(TheHeapMemoryEntryEvent,"Heap Memory Used (KB) at Entry")
TAU_GEN_CONTEXT_EVENT(TheHeapMemoryExitEvent,"Heap Memory Used (KB) at Exit")
TAU_GEN_CONTEXT_EVENT(TheHeapMemoryIncreaseEvent,"Increase in Heap Memory (KB)")
TAU_GEN_CONTEXT_EVENT(TheHeapMemoryDecreaseEvent,"Decrease in Heap Memory (KB)")

extern "C" void * Tau_get_profiler(const char *fname, const char *type, TauGroup_t group, const char *gr_name)
{
  FunctionInfo *f;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  DEBUGPROFMSG("Inside get_profiler group = " << group<<endl;);

  // since we're using new, we should set InitData to true in FunctionInfoInit
  if (group == TAU_MESSAGE) {
    if (gr_name && strcmp(gr_name, "TAU_MESSAGE") == 0) {
      f = new FunctionInfo(fname, type, group, "MPI", true);
    } else {
      f = new FunctionInfo(fname, type, group, gr_name, true);
    }
  } else {
    f = new FunctionInfo(fname, type, group, gr_name, true);
  }

  return (void *)f;
}

/* An array of this struct is shared by all threads.
 * To make sure we don't have false sharing, the struct is 64 bytes in size,
 * so that it fits exactly in one (or two) cache lines. That way, when one
 * thread updates its data in the array, it won't invalidate the cache line
 * for other threads. This is very important with timers, as all threads are
 * entering timers at the same time, and every thread will invalidate the
 * cache line otherwise.
 */
#ifndef CRAYCC
union Tau_thread_status_flags
{
  /* Padding structures is tricky because compilers pad unexpectedly 
   * and word sizes differ.
   *
   * You can see this in this example program:
   *   struct A {
   *     char c;
   *     char d;
   *     int i;
   *   };
   *   struct B {
   *     char c;
   *     int i;
   *     char d;
   *   };
   *   int main() {
   *     cout << sizeof(A) << endl;
   *     cout << sizeof(B) << endl;
   *   }
   *
   * Depending on your compiler, you'll get two different sizes.
   * The only sure way to see this structure padded to 64 bytes is to calculate
   * the pad at compile time as below.
   *
   * Use an anonymous struct container and allow the compiler to place members
   * where it likes.  IT IS CRITICALLY IMPORTANT that the members are ordered
   * largest to smallest, i.e. doubles before floats.  The "int i" member of
   * struct B in the above example could be misaligned.  This idiom is very
   * dangerous in an I/O situation, but for this application it should be safe.
   */
  struct {
    Profiler * Tau_global_stack;
    int Tau_global_stackdepth;
    int Tau_global_stackpos;
    int Tau_global_insideTAU;
    int Tau_is_thread_fake_for_task_api;
    int lightsOut;
  };

  char _pad[64];
};
#else
struct Tau_thread_status_flags {
  Profiler * Tau_global_stack;
  int Tau_global_stackdepth;
  int Tau_global_stackpos;
  int Tau_global_insideTAU;
  int Tau_is_thread_fake_for_task_api;
  int lightsOut;
  // Not as elegant, but similar effect
  char _pad[64-sizeof(Profiler*)-5*sizeof(int)];
};
#endif

#define STACK_DEPTH_INCREMENT 100
/* This array is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
#if defined(__INTEL_COMPILER)
__declspec (align(64)) static Tau_thread_status_flags Tau_thread_flags[TAU_MAX_THREADS] = {0};
#else
#ifdef __GNUC__
static Tau_thread_status_flags Tau_thread_flags[TAU_MAX_THREADS] __attribute__ ((aligned(64))) = {{{0}}};
#else
static Tau_thread_status_flags Tau_thread_flags[TAU_MAX_THREADS] = {0};
#endif
#endif

#if defined (TAU_USE_TLS)
__thread int _Tau_global_insideTAU = 0;
__thread int lightsOut = 0;
#elif defined (TAU_USE_DTLS)
__declspec(thread) int _Tau_global_insideTAU = 0;
__declspec(thread) int lightsOut = 0;
#elif defined (TAU_USE_PGS)
#include "TauPthreadGlobal.h"
#endif


static void Tau_stack_checkInit() {
  static bool init = false;
  if (init) return;
  init = true;

#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  lightsOut = 0;
#elif defined(TAU_USE_PGS)
  TauGlobal::getInstance().getValue()->lightsOut = 0;
#else
  Tau_thread_flags[RtsLayer::unsafeLocalThreadId()].lightsOut = 0;
#endif

  for (int i=0; i<TAU_MAX_THREADS; i++) {
    Tau_thread_flags[i].Tau_global_stackdepth = 0;
    Tau_thread_flags[i].Tau_global_stackpos = -1;
    Tau_thread_flags[i].Tau_global_stack = NULL;
    Tau_thread_flags[i].Tau_global_insideTAU = 0;
    Tau_thread_flags[i].Tau_is_thread_fake_for_task_api = 0; /* by default all threads are real*/
  }
}

extern "C" int Tau_global_getLightsOut() {
  Tau_stack_checkInit();
#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  return lightsOut;
#elif defined(TAU_USE_PGS)
  return TauGlobal::getInstance().getValue()->lightsOut;
#else
  return Tau_thread_flags[RtsLayer::unsafeLocalThreadId()].lightsOut;
#endif
}

extern "C" void Tau_global_setLightsOut() {
  Tau_stack_checkInit();
  // Disable profiling from here on out
  Tau_global_incr_insideTAU();
#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  lightsOut = 1;
#elif defined(TAU_USE_PGS)
  TauGlobal::getInstance().getValue()->lightsOut = 1;
#else
  Tau_thread_flags[RtsLayer::unsafeLocalThreadId()].lightsOut = 1;
#endif
}

/* the task API does not have a real thread associated with the tid */
extern "C" int Tau_is_thread_fake(int tid) {
  return Tau_thread_flags[tid].Tau_is_thread_fake_for_task_api;
}

extern "C" void Tau_set_thread_fake(int tid) {
  Tau_thread_flags[tid].Tau_is_thread_fake_for_task_api = 1;
}

extern "C" void Tau_stack_initialization() {
  Tau_stack_checkInit();
}

extern "C" int Tau_global_get_insideTAU() {
  Tau_stack_checkInit();
  //printf("checking TAU: %d\n", _Tau_global_insideTAU); fflush(stdout);
#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  return _Tau_global_insideTAU;
#elif defined(TAU_USE_PGS)
  return (TauGlobal::getInstance().getValue())->insideTAU;
#else
  int tid = RtsLayer::unsafeLocalThreadId();
  return Tau_thread_flags[tid].Tau_global_insideTAU;
#endif
}

extern "C" int Tau_global_incr_insideTAU()
{
  Tau_stack_checkInit();
  Tau_memory_wrapper_disable();
#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  //printf("enter TAU: %d -> %d\n", _Tau_global_insideTAU, _Tau_global_insideTAU+1); fflush(stdout);
  return ++_Tau_global_insideTAU;
#elif defined(TAU_USE_PGS)
  struct _tau_global_data *tmp = TauGlobal::getInstance().getValue();
  return ++(tmp->insideTAU);
#else
  int tid = RtsLayer::unsafeLocalThreadId();

  volatile int * insideTAU = &Tau_thread_flags[tid].Tau_global_insideTAU;
  *insideTAU = *insideTAU + 1;
  return *insideTAU;
#endif
}

extern "C" int Tau_global_decr_insideTAU()
{
  int retval;
  Tau_stack_checkInit();
#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  //printf("exit TAU: %d -> %d\n", _Tau_global_insideTAU, _Tau_global_insideTAU-1); fflush(stdout);
  retval = --_Tau_global_insideTAU;
#elif defined(TAU_USE_PGS)
  struct _tau_global_data *tmp = TauGlobal::getInstance().getValue();
  retval = --(tmp->insideTAU);
#else
  int tid = RtsLayer::unsafeLocalThreadId();

  volatile int * insideTAU = &Tau_thread_flags[tid].Tau_global_insideTAU;
  *insideTAU = *insideTAU - 1;
  retval = *insideTAU;
#endif
  TAU_ASSERT(retval >= 0, "Thread has decremented the insideTAU counter past 0");
  if (retval == 0) {
    Tau_memory_wrapper_enable();
  }
  return retval;
}

extern "C" Profiler *TauInternal_CurrentProfiler(int tid) {
  int pos = Tau_thread_flags[tid].Tau_global_stackpos;
  if (pos < 0) {
    return NULL;
  }
  return &(Tau_thread_flags[tid].Tau_global_stack[pos]);
}

extern "C" Profiler *TauInternal_ParentProfiler(int tid) {
  int pos = Tau_thread_flags[tid].Tau_global_stackpos-1;
  if (pos < 0) {
    return NULL;
  }
  return &(Tau_thread_flags[tid].Tau_global_stack[pos]);
}

extern "C" char *TauInternal_CurrentCallsiteTimerName(int tid) {
  if(TauInternal_CurrentProfiler(tid) != NULL)
    if(TauInternal_CurrentProfiler(tid)->ThisFunction != NULL)
      return TauInternal_CurrentProfiler(tid)->ThisFunction->Name;
  return NULL;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_start_timer(void *functionInfo, int phase, int tid) {
  FunctionInfo *fi = (FunctionInfo *) functionInfo;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_init_if_necessary();
    Tau_sampling_suspend(tid);
  }
#endif

#ifdef TAU_TRACK_IDLE_THREADS
  /* If we are performing idle thread tracking, we start a top level timer */
  if (tid != 0) {
    Tau_create_top_level_timer_if_necessary_task(tid);
  }
#endif


#ifdef TAU_EPILOG
  esd_enter(fi->GetFunctionId());
#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
  }
#endif
  return;
#endif

#ifdef TAU_VAMPIRTRACE 
  uint64_t TimeStamp = vt_pform_wtime();
#ifdef TAU_VAMPIRTRACE_5_12_API
  vt_enter(VT_CURRENT_THREAD, (uint64_t *) &TimeStamp, fi->GetFunctionId());
#else
  vt_enter((uint64_t *) &TimeStamp, fi->GetFunctionId());
#endif /* TAU_VAMPIRTRACE_5_12_API */
#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
  }
#endif
  return;
#endif

#ifdef TAU_SCOREP
  SCOREP_Tau_EnterRegion(fi->GetFunctionId());
#endif


  // move the stack pointer
  Tau_thread_flags[tid].Tau_global_stackpos++; /* push */



  if (Tau_thread_flags[tid].Tau_global_stackpos >= Tau_thread_flags[tid].Tau_global_stackdepth) {
    int oldDepth = Tau_thread_flags[tid].Tau_global_stackdepth;
    int newDepth = oldDepth + STACK_DEPTH_INCREMENT;
    //printf("%d: NEW STACK DEPTH: %d\n", tid, newDepth); 
    //Profiler *newStack = (Profiler *) malloc(sizeof(Profiler)*newDepth);

    //A deep copy is necessary here to keep the profiler pointers up to date
    Profiler *newStack = (Profiler *) calloc(newDepth, sizeof(Profiler));
    memcpy(newStack, Tau_thread_flags[tid].Tau_global_stack, oldDepth*sizeof(Profiler));

    int tmpDepth=oldDepth;
    Profiler *fixP = &(newStack[oldDepth]);
    while(tmpDepth>0){
    	tmpDepth--;
    	fixP->ParentProfiler=&(newStack[tmpDepth]);
    	fixP=fixP->ParentProfiler;

    }

    free(Tau_thread_flags[tid].Tau_global_stack);

    Tau_thread_flags[tid].Tau_global_stack = newStack;
    Tau_thread_flags[tid].Tau_global_stackdepth = newDepth;
  }

  Profiler *p = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);

  p->MyProfileGroup_ = fi->GetProfileGroup();
  p->ThisFunction = fi;
  p->needToRecordStop = 0;

#ifdef TAU_MPITRACE
  p->RecordEvent = false; /* by default, we don't record this event */
#endif /* TAU_MPITRACE */

  
#ifdef TAU_PROFILEPHASE
  if (phase) {
    p->SetPhase(true);
  } else {
    p->SetPhase(false);
  }
#endif /* TAU_PROFILEPHASE */

#ifdef TAU_DEPTH_LIMIT
  static int userspecifieddepth = TauEnv_get_depth_limit();
  int mydepth = Tau_thread_flags[tid].Tau_global_stackpos;
  if (mydepth >= userspecifieddepth) { 
#ifndef TAU_WINDOWS
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif
    return; 
  }
#endif /* TAU_DEPTH_LIMIT */

  p->Start(tid);

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

  /*** Memory Profiling ***/
  if (TauEnv_get_track_memory_heap()) {
    double *heapmem = new double; 
    *heapmem = Tau_max_RSS();
    TAU_CONTEXT_EVENT(TheHeapMemoryEntryEvent(), *heapmem);
    p->extraInfo = heapmem;
   
  }

  if (TauEnv_get_track_memory_headroom()) {
    TAU_REGISTER_CONTEXT_EVENT(memEvent, "Memory Headroom Available (MB) at Entry");
    TAU_CONTEXT_EVENT(memEvent, Tau_estimate_free_memory());
  }

#ifdef TAU_PROFILEMEMORY
  p->ThisFunction->GetMemoryEvent()->TriggerEvent(Tau_max_RSS());
#endif /* TAU_PROFILEMEMORY */

#ifdef TAU_PROFILEHEADROOM
  p->ThisFunction->GetHeadroomEvent()->TriggerEvent(Tau_estimate_free_memory());
#endif /* TAU_PROFILEHEADROOM */

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
    // if the unwind depth should be "automatic", then get the stack for right now
    if (TauEnv_get_ebs_unwind_depth() == 0) {
      Tau_sampling_event_start(tid, p->address);
    }
  }
#endif
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_lite_start_timer(void *functionInfo, int phase)
{
  if (TauEnv_get_lite_enabled()) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    int tid = RtsLayer::myThread();
    // move the stack pointer
    Tau_thread_flags[tid].Tau_global_stackpos++; /* push */
    FunctionInfo *fi = (FunctionInfo *)functionInfo;
    Profiler *pp = TauInternal_ParentProfiler(tid);
    if (fi) {
      fi->IncrNumCalls(tid);    // increment number of calls
    }
    if (pp && pp->ThisFunction) {
      pp->ThisFunction->IncrNumSubrs(tid);    // increment parent's child calls
    }

    if (Tau_thread_flags[tid].Tau_global_stackpos >= Tau_thread_flags[tid].Tau_global_stackdepth) {
      int oldDepth = Tau_thread_flags[tid].Tau_global_stackdepth;
      int newDepth = oldDepth + STACK_DEPTH_INCREMENT;
      Profiler *newStack = (Profiler *)malloc(sizeof(Profiler) * newDepth);
      memcpy(newStack, Tau_thread_flags[tid].Tau_global_stack, oldDepth * sizeof(Profiler));
      Tau_thread_flags[tid].Tau_global_stack = newStack;
      Tau_thread_flags[tid].Tau_global_stackdepth = newDepth;
    }
    Profiler *p = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
    RtsLayer::getUSecD(tid, p->StartTime);

    p->MyProfileGroup_ = fi->GetProfileGroup();
    p->ThisFunction = fi;
    p->ParentProfiler = pp;

    // if this function is not already on the callstack, put it
    if (fi->GetAlreadyOnStack(tid) == false) {
      p->AddInclFlag = true;
      fi->SetAlreadyOnStack(true, tid);
    } else {
      p->AddInclFlag = false;
    }

  } else {    // not lite - default
    FunctionInfo *fi = (FunctionInfo *)functionInfo;
    if (RtsLayer::TheEnableInstrumentation() && (fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
      Tau_start_timer(functionInfo, phase, Tau_get_thread());
    }
  }
}
    




///////////////////////////////////////////////////////////////////////////
static void reportOverlap (FunctionInfo *stack, FunctionInfo *caller) {
  fprintf(stderr, "[%d:%d][%d:%d] TAU: Runtime overlap: found %s (%p) on the stack, but stop called on %s (%p)\n",
	 RtsLayer::getPid(), RtsLayer::getTid(), RtsLayer::myNode(), RtsLayer::myThread(),
	 stack->GetName(), stack, caller->GetName(), caller);
#if !defined(TAU_WINDOWS) && !defined(TAU_ANDROID)
     if(!TauEnv_get_ebs_enabled()) {
       void* callstack[128];
       int i, frames = backtrace(callstack, 128);
       char** strs = backtrace_symbols(callstack, frames);
       for (i = 0; i < frames; ++i) {
         fprintf(stderr,"%s\n", strs[i]);
       }
       free(strs);
     }
#endif
	 abort();
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_stop_timer(void *function_info, int tid ) {
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  FunctionInfo *fi = (FunctionInfo *) function_info; 
  double currentHeap = 0.0;
  bool enableHeapTracking;

  Profiler *profiler;
#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_suspend(tid);
  }
#endif

  //TAU_VERBOSE(" *** (S%d) going to stop T%d\n", gettid(), tid);

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/

  /*** Memory Profiling ***/
  enableHeapTracking = TauEnv_get_track_memory_heap();
  if (enableHeapTracking) {
    currentHeap = Tau_max_RSS();
    TAU_CONTEXT_EVENT(TheHeapMemoryExitEvent(), currentHeap);
  }

  if (TauEnv_get_track_memory_headroom()) {
    TAU_REGISTER_CONTEXT_EVENT(memEvent, "Memory Headroom Available (MB) at Exit");
    TAU_CONTEXT_EVENT(memEvent, Tau_estimate_free_memory());
  }

  /********************************************************************************/
  /*** Extras ***/
  /********************************************************************************/


#ifdef TAU_EPILOG
  esd_exit(fi->GetFunctionId());
#ifndef TAU_WINDOWS
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif
  return 0;
#endif

#ifdef TAU_VAMPIRTRACE 
  uint64_t TimeStamp = vt_pform_wtime();

#ifdef TAU_VAMPIRTRACE_5_12_API
  vt_exit(VT_CURRENT_THREAD, (uint64_t *)&TimeStamp);
#else 
  vt_exit((uint64_t *)&TimeStamp);
#endif /* TAU_VAMPIRTRACE_5_12_API */

#ifndef TAU_WINDOWS
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif
  return 0;
#endif

#ifdef TAU_SCOREP
  SCOREP_Tau_ExitRegion(fi->GetFunctionId());
#endif

  if (Tau_thread_flags[tid].Tau_global_stackpos < 0) {
#ifndef TAU_WINDOWS
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif
    return 0; 
  }

  profiler = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
  
  while (profiler->ThisFunction != fi) { /* Check for overlapping timers */
      /* We might have an inconstant stack because of throttling. If one thread
       * throttles a routine while it is on the top of the stack of another thread
       * it will remain there until a stop is called on its parent. Check for this
       * condition before printing a overlap error message. */
      if (!profiler->ThisFunction->GetProfileGroup() & RtsLayer::TheProfileMask())
      {
	  profiler->Stop();
	  Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */

	  profiler = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
      } else {
	  reportOverlap (profiler->ThisFunction, fi);
      }
  }


#ifdef TAU_DEPTH_LIMIT
  static int userspecifieddepth = TauEnv_get_depth_limit();
  int mydepth = Tau_thread_flags[tid].Tau_global_stackpos;
  if (mydepth >= userspecifieddepth) { 
    Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */
#ifndef TAU_WINDOWS
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif
    return 0; 
  }
#endif /* TAU_DEPTH_LIMIT */


  /* check memory */
  if (enableHeapTracking && profiler->extraInfo) {
    double *oldheap = (double *) (profiler->extraInfo);
    double difference = currentHeap - *oldheap; 
    if (difference > 0) {
      TAU_CONTEXT_EVENT(TheHeapMemoryIncreaseEvent(), difference);
    } else {
       if (difference < 0) {
        TAU_CONTEXT_EVENT(TheHeapMemoryDecreaseEvent(), (0 - difference));
       }
    }
			  
  }

  profiler->Stop(tid);

  Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */

  //TAU_VERBOSE(" *** (S%d) stop timer for T%d %s\n", gettid(), tid, profiler->ThisFunction->GetName());

#ifndef TAU_WINDOWS
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
  }
#endif
  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_lite_stop_timer(void *function_info)
{
  if (TauEnv_get_lite_enabled()) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    int tid = RtsLayer::myThread();
    double timeStamp[TAU_MAX_COUNTERS] = { 0 };
    double delta[TAU_MAX_COUNTERS] = { 0 };
    RtsLayer::getUSecD(tid, timeStamp);

    FunctionInfo *fi = (FunctionInfo *)function_info;
    Profiler *profiler;
    profiler = (Profiler *)&(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);

    for (int k = 0; k < Tau_Global_numCounters; k++) {
      delta[k] = timeStamp[k] - profiler->StartTime[k];
    }

    if (profiler && profiler->ThisFunction != fi) { /* Check for overlapping timers */
      reportOverlap(profiler->ThisFunction, fi);
    }
    if (profiler && profiler->AddInclFlag == true) {
      fi->SetAlreadyOnStack(false, tid);    // while exiting
      fi->AddInclTime(delta, tid);    // ok to add both excl and incl times
    } else {
      //printf("Couldn't add incl time: profiler= %p, profiler->AddInclFlag=%d\n", profiler, profiler->AddInclFlag);
    }
    fi->AddExclTime(delta, tid);
    Profiler *pp = TauInternal_ParentProfiler(tid);

    if (pp) {
      pp->ThisFunction->ExcludeTime(delta, tid);
    } else {
      //printf("Tau_lite_stop: parent profiler = 0x0: Function name = %s, StoreData?\n", fi->GetName()); 
      TauProfiler_StoreData(tid);
    }
    Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */
  } else {
    FunctionInfo *fi = (FunctionInfo *)function_info;
    if (RtsLayer::TheEnableInstrumentation() && (fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
      Tau_stop_timer(function_info, Tau_get_thread());
    }
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_stop_current_timer_task(int tid) 
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  if (Tau_thread_flags[tid].Tau_global_stackpos >= 0) {
    Profiler * profiler = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
    /* We might have an inconstant stack because of throttling. If one thread
     * throttles a routine while it is on the top of the stack of another thread
     * it will remain there until a stop is called on its parent. Check for this
     * condition before printing a overlap error message. */
    while (!profiler->ThisFunction->GetProfileGroup() & RtsLayer::TheProfileMask() && 
          (Tau_thread_flags[tid].Tau_global_stackpos >= 0))
    {
      profiler->Stop();
      Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */
      profiler = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
    }

    FunctionInfo * functionInfo = profiler->ThisFunction;
    return Tau_stop_timer(functionInfo, tid);
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_stop_current_timer() 
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  int tid = RtsLayer::myThread();
  return Tau_stop_current_timer_task(tid);
}

///////////////////////////////////////////////////////////////////////////

extern "C" void Tau_disable_collector_api();

extern "C" int Tau_profile_exit_all_tasks()
{
  // Stop the collector API. The main thread may exit with running
  // worker threads. When those threads try to exit, they will
  // try to stop timers that aren't running.
  RtsLayer::LockDB();
#ifdef TAU_OPENMP
  Tau_disable_collector_api();
#endif

#ifdef TAU_ANDROID
  bool su = JNIThreadLayer::IsMgmtThread();
#endif

  int tid = 1;
  while (tid < TAU_MAX_THREADS) {
#ifdef TAU_ANDROID
    if (su == true) {
      JNIThreadLayer::SuThread(tid);
    }
#endif
    while (Tau_thread_flags[tid].Tau_global_stackpos >= 0) {
      Profiler *p = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
      //Make sure even throttled routines are stopped.
      if (Tau_stop_timer(p->ThisFunction, tid)) {
        TAU_VERBOSE("Stopping timer on thread %d: %s\n", tid, p->ThisFunction->Name);
        p->Stop(tid);
        Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */
      }
    }
    tid++;
  }
  Tau_shutdown();
  RtsLayer::UnLockDB();
  return 0;
}

extern "C" int Tau_show_profiles()
{
  for (int tid=0; tid < TAU_MAX_THREADS; ++tid) {
      int pos = Tau_thread_flags[tid].Tau_global_stackpos;
      while (pos >= 0) {
	  Profiler * p = &(Tau_thread_flags[tid].Tau_global_stack[pos]);
	  TAU_VERBOSE(" *** Alfred Profile (%d:%d) :  %s\n", tid, pos, p->ThisFunction->Name);
	  pos--;
      }
  }

  return 0;
}

extern "C" int Tau_profile_exit_all_threads()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_ANDROID
  bool su = JNIThreadLayer::IsMgmtThread();
#endif

  for (int tid = 0; tid < TAU_MAX_THREADS; ++tid) {
#ifdef TAU_ANDROID
    if (su == true) {
      JNIThreadLayer::SuThread(tid);
    }
#endif
    while (Tau_thread_flags[tid].Tau_global_stackpos >= 0) {
      Profiler * p = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);

      TAU_VERBOSE(" *** Alfred (%d) : stop %s\n", tid, p->ThisFunction->Name);

      //Make sure even throttled routines are stopped.
      if (Tau_stop_timer(p->ThisFunction, tid)) {
        p->Stop(tid);
        Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */
      }
      // DO NOT pop. It is popped in stop above: Tau_thread_flags[tid].Tau_global_stackpos--;
    }
  }

  Tau_shutdown();
  return 0;
}


extern "C" int Tau_profile_exit()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  int tid = RtsLayer::myThread();
  while (Tau_thread_flags[tid].Tau_global_stackpos >= 0) {
    Profiler * p = &(Tau_thread_flags[tid].Tau_global_stack[Tau_thread_flags[tid].Tau_global_stackpos]);
    //Make sure even throttled routines are stopped.
    if (Tau_stop_timer(p->ThisFunction, tid)) {
      p->Stop(tid);
      Tau_thread_flags[tid].Tau_global_stackpos--; /* pop */
    }
    // DO NOT pop. It is popped in stop above: Tau_thread_flags[tid].Tau_global_stackpos--;
  }
  Tau_shutdown();
  return 0;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_exit(const char * msg) {
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_CUDA
	Tau_profile_exit_all_threads();
#else
  Tau_profile_exit();
#endif

#ifdef TAUKTAU
  KtauProfiler::PutKtauProfiler();
#endif /* TAUKTAU */
  
#ifdef RENCI_STFF  
  RenciSTFF::cleanup();
#endif // RENCI_STFF  
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_init_ref(int* argc, char ***argv) {
  TauInternalFunctionGuard protects_this_function;
  RtsLayer::ProfileInit(*argc, *argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_init(int argc, char **argv) {
  TauInternalFunctionGuard protects_this_function;
  RtsLayer::ProfileInit(argc, argv);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_node(int node) {
  TauInternalFunctionGuard protects_this_function;
  if (node >= 0) TheSafeToDumpData()=1;
  RtsLayer::setMyNode(node);
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_get_node(void) {
  TauInternalFunctionGuard protects_this_function;
  return RtsLayer::myNode();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_context(int context) {
  TauInternalFunctionGuard protects_this_function;
  RtsLayer::setMyContext(context);
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_get_context(void) {
  TauInternalFunctionGuard protects_this_function;
  return RtsLayer::myContext();
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_thread(int threadId) {
  // Placeholder, probably shouldn't be a legal operation
  // Recommend we deprecate TAU_SET_THREAD
  cerr << "TAU: ERROR: Unsafe and deprecated call to TAU_SET_THREAD!" << endl;
}

//////////////////////////////////////////////////////////////////////
extern "C" int Tau_get_thread(void) {
  TauInternalFunctionGuard protects_this_function;
  return RtsLayer::myThread();
}


///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump(void) {
  TauInternalFunctionGuard protects_this_function;
  TauProfiler_DumpData();
  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_prefix(const char *prefix) {
  TauInternalFunctionGuard protects_this_function;
  for (int i = 0 ; i < RtsLayer::getTotalThreads() ; i++)
    TauProfiler_DumpData(false, i, prefix);
  return 0;
}

extern x_uint64 TauTraceGetTimeStamp(int tid);

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_callpaths() {
  TauInternalFunctionGuard protects_this_function;

  int tid;
  long pos;
  int pid = RtsLayer::myNode();

  const char *dirname = TauEnv_get_profiledir();

  //Create temp write to file.
  char filename[1024];
  sprintf(filename,"%s/callpaths.%d",dirname, pid);

  FILE* fp;
  if ((fp = fopen (filename, "a+")) == NULL) {
    char errormsg[1024];
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    return 1;
  }

  fprintf(fp, "Thread\tStack\tCalls\tIncl.\tExcl.\tName\tTimestamp:\t%llu\n", TauTraceGetTimeStamp(0));
  for (tid = 0 ; tid < RtsLayer::getTotalThreads() ; tid++) {
    pos = Tau_thread_flags[tid].Tau_global_stackpos;
	TauProfiler_updateIntermediateStatistics(tid);
    while (pos >= 0) {
	  Profiler profiler = Tau_thread_flags[tid].Tau_global_stack[pos];
	  FunctionInfo *fi = profiler.ThisFunction;
      fprintf(fp, "%d\t%ld\t%ld\t%.f\t%.f\t\"%s\"\n", tid, pos, fi->GetCalls(tid), fi->getDumpInclusiveValues(tid)[0], fi->getDumpExclusiveValues(tid)[0], fi->Name);
      pos--;
    }
  }

  fclose(fp);
  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_prefix_task(const char *prefix, int taskid) {
  TauInternalFunctionGuard protects_this_function;
  TauProfiler_DumpData(false, taskid, prefix);
  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_incr(void) {
  TauInternalFunctionGuard protects_this_function;
  TauProfiler_DumpData(true);
  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_purge(void) {
  TauProfiler_PurgeData();
}

extern "C" void Tau_the_function_list(const char ***functionList, int *num) {
  TauProfiler_theFunctionList(functionList, num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_names() {
  TauProfiler_dumpFunctionNames();
}
  
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_counter_names(const char ***counterList, int *num) {
  TauProfiler_theCounterList(counterList, num);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_function_values(const char **inFuncs, int numOfFuncs,
					double ***counterExclusiveValues,
					double ***counterInclusiveValues,
					int **numOfCalls, int **numOfSubRoutines,
					const char ***counterNames, int *numOfCounters) {
  TauProfiler_getFunctionValues(inFuncs,numOfFuncs,counterExclusiveValues,counterInclusiveValues,
				numOfCalls,numOfSubRoutines,counterNames,numOfCounters);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_event_names(const char ***eventList, int *num) {
  TauProfiler_getUserEventList(eventList, num);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_event_vals(const char **inUserEvents, int numUserEvents,
				  int **numEvents, double **max, double **min,
				  double **mean, double **sumSqr) {
  TauProfiler_getUserEventValues(inUserEvents, numUserEvents, numEvents, max, min,
				 mean, sumSqr);
}



///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_values(const char **functionList, int num) {
  TauProfiler_dumpFunctionValues(functionList,num);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_dump_function_values_incr(const char **functionList, int num) {
  TauProfiler_dumpFunctionValues(functionList,num,true);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_register_thread(void) {
  RtsLayer::RegisterThread();
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_register_fork(int nodeid, enum TauFork_t opcode) {
  RtsLayer::RegisterFork(nodeid, opcode);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_instrumentation(void) {
  RtsLayer::TheEnableInstrumentation() = true;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_instrumentation(void) {
  RtsLayer::TheEnableInstrumentation() = false;
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_shutdown(void) {
  Tau_memory_wrapper_disable();
  if (!TheUsingCompInst()) {
    RtsLayer::TheShutdown() = true;
    RtsLayer::TheEnableInstrumentation() = false;
  }
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_enable_group_name(char const * group) {
  return RtsLayer::enableProfileGroupName(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_disable_group_name(char const * group) {
  return RtsLayer::disableProfileGroupName(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_profile_set_group_name(void *ptr, const char *groupname) {
  FunctionInfo *f = (FunctionInfo*)ptr;
  f->SetPrimaryGroupName(groupname);
}

extern "C" void Tau_profile_set_name(void *ptr, const char *name) {
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *f = (FunctionInfo*)ptr;
  f->Name = strdup(name);
}

extern "C" void Tau_profile_set_type(void *ptr, const char *type) {
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *f = (FunctionInfo*)ptr;
  f->Type = strdup(type);
}

extern "C" void Tau_profile_set_group(void *ptr, TauGroup_t group) {
  FunctionInfo *f = (FunctionInfo*)ptr;
  f->SetProfileGroup(group);
}

extern "C" const char *Tau_profile_get_group_name(void *ptr) {
  FunctionInfo *f = (FunctionInfo*)ptr;
  return f->GroupName;
}

extern "C" const char *Tau_profile_get_name(void *ptr) {
  FunctionInfo *f = (FunctionInfo*)ptr;
  return f->Name;
}

extern "C" const char *Tau_profile_get_type(void *ptr) {
  FunctionInfo *f = (FunctionInfo*)ptr;
  return f->Type;
}

extern "C" TauGroup_t Tau_profile_get_group(void *ptr) {
  FunctionInfo *f = (FunctionInfo*)ptr;
  return f->GetProfileGroup();
}

///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_get_profile_group(char * group) {
  return RtsLayer::getProfileGroup(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_group(TauGroup_t group) {
  RtsLayer::enableProfileGroup(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_group(TauGroup_t group) {
  RtsLayer::disableProfileGroup(group);
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_enable_all_groups(void) {
  return RtsLayer::enableAllGroups();
}


///////////////////////////////////////////////////////////////////////////
extern "C" TauGroup_t Tau_disable_all_groups(void) {
  return RtsLayer::disableAllGroups();
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_is_shutdown(void) {
  return RtsLayer::TheShutdown();
}



///////////////////////////////////////////////////////////////////////////
extern "C" int tau_totalnodes(int set_or_get, int value)
{
  static int nodes = 1;
  if (set_or_get == 1) {
    nodes = value;
  }
  return nodes;
}



#if (defined(TAU_MPI) || defined(TAU_SHMEM) || defined(TAU_DMAPP) || defined(TAU_UPC) || defined(TAU_GPI) )



#ifdef TAU_SYSTEMWIDE_TRACK_MSG_SIZE_AS_CTX_EVENT 
#define TAU_GEN_EVENT(e, msg) TauContextUserEvent* e () { \
        static TauContextUserEvent ce(msg); return &ce; }
#undef TAU_EVENT
#define TAU_EVENT(event,data) Tau_context_userevent(event, data);
#else
#define TAU_GEN_EVENT(e, msg) TauUserEvent* e () { \
	static TauUserEvent u(msg); return &u; } 
#endif /* TAU_SYSTEMWIDE_TRACK_MSG_SIZE_AS_CTX_EVENT */



TAU_GEN_EVENT(TheSendEvent,"Message size sent to all nodes")
TAU_GEN_EVENT(TheRecvEvent,"Message size received from all nodes")
TAU_GEN_EVENT(TheBcastEvent,"Message size for broadcast")
TAU_GEN_EVENT(TheReduceEvent,"Message size for reduce")
TAU_GEN_EVENT(TheReduceScatterEvent,"Message size for reduce-scatter")
TAU_GEN_EVENT(TheScanEvent,"Message size for scan")
TAU_GEN_EVENT(TheAllReduceEvent,"Message size for all-reduce")
TAU_GEN_EVENT(TheAlltoallEvent,"Message size for all-to-all")
TAU_GEN_EVENT(TheScatterEvent,"Message size for scatter")
TAU_GEN_EVENT(TheGatherEvent,"Message size for gather")
TAU_GEN_EVENT(TheAllgatherEvent,"Message size for all-gather")
TAU_GEN_CONTEXT_EVENT(TheWaitEvent,"Message size received in wait")

TauContextUserEvent & TheMsgVolSendContextEvent(int tid) {
    static TauContextUserEvent ** sendEvents = NULL;

    if(!sendEvents) {
        sendEvents = (TauContextUserEvent**)calloc(tau_totalnodes(0,0), sizeof(TauContextUserEvent*));
    }

    if(!sendEvents[tid]) {
        char buff[256];
        sprintf(buff, "Message size sent to node %d", tid);
        sendEvents[tid] = new TauContextUserEvent(buff);
    }

    return *(sendEvents[tid]);
}

TauContextUserEvent & TheMsgVolRecvContextEvent(int tid) {
    static TauContextUserEvent ** recvEvents = NULL;

    if(!recvEvents) {
        recvEvents = (TauContextUserEvent**)calloc(tau_totalnodes(0,0), sizeof(TauContextUserEvent*));
    }

    if(!recvEvents[tid]) {
        char buff[256];
        sprintf(buff, "Message size received from node %d", tid);
        recvEvents[tid] = new TauContextUserEvent(buff);
    }

    return *(recvEvents[tid]);
}

///////////////////////////////////////////////////////////////////////////
#ifdef TAU_SHMEM 
extern "C" int shmem_n_pes(void); 
#endif /* TAU_SHMEM */

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length) 
{
  if (!RtsLayer::TheEnableInstrumentation()) return; 

#ifdef TAU_PROFILEPARAM
#ifndef TAU_DISABLE_PROFILEPARAM_IN_MPI
  TAU_PROFILE_PARAM1L(length, "message size");
#endif /* TAU_DISABLE_PROFILEPARAM_IN_MPI */
#endif  /* TAU_PROFILEPARAM */

  TAU_EVENT(TheSendEvent(), length);

  if (TauEnv_get_comm_matrix()) {
    if (destination >= tau_totalnodes(0,0)) {
#ifdef TAU_SHMEM
      tau_totalnodes(1,shmem_n_pes());
#else /* TAU_SHMEM */
      fprintf(stderr, 
          "TAU Error: Comm Matrix destination %d exceeds node count %d. "
          "Was MPI_Init/shmem_init wrapper never called? "
          "Please disable TAU_COMM_MATRIX or add calls to the init function in your source code.\n", 
          destination, tau_totalnodes(0,0));
      exit(-1);
#endif /* TAU_SHMEM */
    }
    TheMsgVolSendContextEvent(destination).TriggerEvent(length, Tau_get_thread());
  }

  if (TauEnv_get_tracing()) {
    if (destination >= 0) {
      TauTraceSendMsg(type, destination, length);
    }
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg(int type, int source, int length)
{
#ifdef TAU_PROFILEPARAM
#ifndef TAU_DISABLE_PROFILEPARAM_IN_MPI
  TAU_PROFILE_PARAM1L(length, "message size");
#endif /* TAU_DISABLE_PROFILEPARAM_IN_MPI */
#endif  /* TAU_PROFILEPARAM */

  TAU_EVENT(TheRecvEvent(), length);

  if (TauEnv_get_tracing()) {
    if (source >= 0) {
      TauTraceRecvMsg(type, source, length);
    }
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg_remote(int type, int source, int length, int remoteid) 
{
  if (!RtsLayer::TheEnableInstrumentation()) return; 
  if (TauEnv_get_tracing()) {
    if (source >= 0) {
      TauTraceRecvMsgRemote(type, source, length, remoteid);
    }
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg_remote(int type, int destination, int length, int remoteid) 
{
  if (!RtsLayer::TheEnableInstrumentation()) return; 

  if (TauEnv_get_tracing()) {
    if (destination >= 0) {
      TauTraceSendMsgRemote(type, destination, length, remoteid);
    }
  }

  if (TauEnv_get_comm_matrix())  {

#ifdef TAU_PROFILEPARAM
#ifndef TAU_DISABLE_PROFILEPARAM_IN_MPI
    TAU_PROFILE_PARAM1L(length, "message size");
#endif /* TAU_DISABLE_PROFILEPARAM_IN_MPI */
#endif  /* TAU_PROFILEPARAM */

    if (TauEnv_get_comm_matrix()) {
      if (destination >= tau_totalnodes(0,0)) {
#ifdef TAU_SHMEM
        tau_totalnodes(1,shmem_n_pes());
#else /* TAU_SHMEM */
        fprintf(stderr, 
            "TAU Error: Comm Matrix destination %d exceeds node count %d. "
            "Was MPI_Init/shmem_init wrapper never called? "
            "Please disable TAU_COMM_MATRIX or add calls to the init function in your source code.\n", 
            destination, tau_totalnodes(0,0));
        exit(-1);
#endif /* TAU_SHMEM */
      }
      TheMsgVolRecvContextEvent(remoteid).TriggerEvent(length, Tau_get_thread());
    }

  }
}


extern "C" void Tau_bcast_data(int data) {
  TAU_EVENT(TheBcastEvent(), data);
}

extern "C" void Tau_reduce_data(int data) {
  TAU_EVENT(TheReduceEvent(), data);
}

extern "C" void Tau_alltoall_data(int data) {
  TAU_EVENT(TheAlltoallEvent(), data);
}

extern "C" void Tau_scatter_data(int data) {
  TAU_EVENT(TheScatterEvent(), data);
}

extern "C" void Tau_gather_data(int data) {
  TAU_EVENT(TheGatherEvent(), data);
}

extern "C" void Tau_allgather_data(int data) {
  TAU_EVENT(TheAllgatherEvent(), data);
}

extern "C" void Tau_wait_data(int data) {
  TAU_CONTEXT_EVENT(TheWaitEvent(), data);
}

extern "C" void Tau_allreduce_data(int data) {
  TAU_EVENT(TheAllReduceEvent(), data);
}

extern "C" void Tau_scan_data(int data) {
  TAU_EVENT(TheScanEvent(), data);
}

extern "C" void Tau_reducescatter_data(int data) {
  TAU_EVENT(TheReduceScatterEvent(), data);
}

#else /* !(TAU_MPI || TAU_SHMEM || TAU_DMAPP || TAU_GPI)*/

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length) {
  TauTraceSendMsg(type, destination, length);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg_remote(int type, int destination, int length, int remoteid) {
  TauTraceSendMsgRemote(type, destination, length, remoteid);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg(int type, int source, int length) {
  TauTraceRecvMsg(type, source, length);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_recvmsg_remote(int type, int source, int length, int remoteid) {
  TauTraceRecvMsgRemote(type, source, length, remoteid);
}

#endif /* TAU_MPI || TAU_SHMEM*/

///////////////////////////////////////////////////////////////////////////
// User Defined Events 
///////////////////////////////////////////////////////////////////////////
extern "C" void * Tau_get_userevent(char const * name) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent *ue;
  ue = new TauUserEvent(std::string(name));
  return (void *) ue;
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_userevent(void *ue, double data) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent *t = (TauUserEvent *) ue;
  t->TriggerEvent(data);
} 

extern "C" void Tau_userevent_thread(void *ue, double data, int tid) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent *t = (TauUserEvent *) ue;
  t->TriggerEvent(data, tid);
}

extern "C" void * Tau_return_context_userevent(const char *name) {
    TauInternalFunctionGuard protects_this_function;
    TauContextUserEvent * ue = new TauContextUserEvent(name);
    return (void*)ue;
}

///////////////////////////////////////////////////////////////////////////
// WARNING: the pointer passed into Tau_get_context_userevent must be declared
// static or intialized to NULL otherwise it could end up pointing to a random 
// piece of memory. See Tau_pure_context_userevent for a routine that does a
// name lookup.
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_context_userevent(void **ptr, const char *name)
{
  if (!*ptr) {
    TauInternalFunctionGuard protects_this_function;
    RtsLayer::LockEnv();
    if (!*ptr) {
      TauContextUserEvent * ue = new TauContextUserEvent(name);
      *ptr = (void*)ue;
    }
    RtsLayer::UnLockEnv();
  }
}

typedef TAU_HASH_MAP<string, TauContextUserEvent *> pure_atomic_map_t;
pure_atomic_map_t & ThePureAtomicMap() {
  static pure_atomic_map_t pureAtomicMap;
  return pureAtomicMap;
}

typedef TAU_HASH_MAP<string, TauUserEvent *> pure_userevent_atomic_map_t;
pure_userevent_atomic_map_t & ThePureUserEventAtomicMap() {
  static pure_userevent_atomic_map_t pureUserEventAtomicMap;
  return pureUserEventAtomicMap;
}

extern "C" void Tau_pure_context_userevent(void **ptr, const char* name)
{
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *ue = 0;
  RtsLayer::LockEnv();
  pure_atomic_map_t::iterator it = ThePureAtomicMap().find(string(name));
  if (it == ThePureAtomicMap().end()) {
    ue = new TauContextUserEvent(name); 
    ThePureAtomicMap()[string(name)] = ue;
  } else {
    ue = (*it).second;
  }
  RtsLayer::UnLockEnv();
  *ptr = (void *) ue;
}

extern "C" void Tau_pure_userevent(void **ptr, const char* name)
{
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent *ue = 0;
  RtsLayer::LockEnv();
  pure_userevent_atomic_map_t::iterator it = ThePureUserEventAtomicMap().find(string(name));
  if (it == ThePureUserEventAtomicMap().end()) {
    ue = new TauUserEvent(name); 
    ThePureUserEventAtomicMap()[string(name)] = ue;
  } else {
    ue = (*it).second;
  }
  RtsLayer::UnLockEnv();
  *ptr = (void *) ue;
}



///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_context_userevent(void *ue, double data) {
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->TriggerEvent(data);
} 


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trigger_context_event_thread(const char *name, double data, int tid) {
  TauInternalFunctionGuard protects_this_function;
  void *ue;
  Tau_pure_context_userevent(&ue, name);
  Tau_context_userevent_thread(ue, data, tid);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trigger_context_event(const char *name, double data) {
  TauInternalFunctionGuard protects_this_function;
  void *ue;
  Tau_pure_context_userevent(&ue, name);
  Tau_context_userevent(ue, data);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trigger_userevent(const char *name, double data) {
  TauInternalFunctionGuard protects_this_function;
  void *ue;
  Tau_pure_userevent(&ue, name);
  Tau_userevent(ue, data);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_context_userevent_thread(void *ue, double data, int tid) {
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->TriggerEvent(data, tid);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_event_name(void *ue, char *name) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetName(name);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_context_event_name(void *ue, const char *name) {
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->SetAllEventName(name);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_write_user_event_as_metric(void *ue) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetWriteAsMetric(true);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_report_statistics(void) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent::ReportStatistics();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_report_thread_statistics(void) {
  TauInternalFunctionGuard protects_this_function;
  TauUserEvent::ReportStatistics(true);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_min(void *ue) {
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetMinEnabled(false);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_max(void *ue) {
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetMaxEnabled(false);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_mean(void *ue) {
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetMeanEnabled(false);
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_event_disable_stddev(void *ue) {
  TauUserEvent *t = (TauUserEvent *) ue;
  t->SetStdDevEnabled(false);
} 

///////////////////////////////////////////////////////////////////////////

extern "C" void Tau_profile_c_timer(void **ptr, const char *name, const char *type, 
    TauGroup_t group, const char *group_name) 
{
  if (*ptr == 0) {
    TauInternalFunctionGuard protects_this_function;
    RtsLayer::LockDB();
    if (*ptr == 0) {  
      // remove garbage characters from the end of name
      unsigned int len=0;
      while(isprint(name[len])) {
        ++len;
      }

      char * fixedname = (char*)malloc(len+1);
      memcpy(fixedname, name, len);
      fixedname[len] = '\0';

      *ptr = Tau_get_profiler(fixedname, type, group, group_name);

      free((void*)fixedname);
    }
    RtsLayer::UnLockDB();
  }
}

///////////////////////////////////////////////////////////////////////////

static string& gTauApplication()
{
  static string g = string(".TAU application");
  return g;
}

// forward declare the function we need to use - it's defined later
extern void Tau_pure_start_task_string(const string name, int tid);

/* We need a routine that will create a top level parent profiler and give
 * it a dummy name for the application, if just the MPI wrapper interposition
 * library is used without any instrumentation in main */
extern "C" void Tau_create_top_level_timer_if_necessary_task(int tid)
{
#if ! (defined(TAU_VAMPIRTRACE) || defined(TAU_EPILOG) || defined(TAU_SCOREP))
  TauInternalFunctionGuard protects_this_function;

  /* After creating the ".TAU application" timer, we start it. In the
   timer start code, it will call this function, so in that case,
   return right away. */
  static bool initialized = false;
  static bool initializing[TAU_MAX_THREADS] = { false };
  static bool initthread[TAU_MAX_THREADS] = { false };

  if (!initialized && !initializing[tid]) {
    RtsLayer::LockDB();
    if (!initialized) {
      // whichever thread got here first, has the lock and will create the
      // FunctionInfo object for the top level timer.
      if (!TauInternal_CurrentProfiler(tid)) {
        initthread[tid] = true;
        initializing[tid] = true;
        Tau_pure_start_task_string(gTauApplication(), tid);
        atexit((void(*)(void))Tau_profile_exit_all_threads);
        initializing[tid] = false;
        initialized = true;
      }
    }
    RtsLayer::UnLockDB();
  }

  if (!initthread[tid]) {
    // if there is no top-level timer, create one - But only create one FunctionInfo object.
    // that should be handled by the Tau_pure_start_task call.
    if (!TauInternal_CurrentProfiler(tid)) {
      initthread[tid] = true;
      initializing[tid] = true;
      Tau_pure_start_task_string(gTauApplication(), tid);
      initializing[tid] = false;
    }
  }

#endif
}

extern "C" void Tau_create_top_level_timer_if_necessary(void) {
  return Tau_create_top_level_timer_if_necessary_task(Tau_get_thread());
}


extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid)
{
#ifdef TAU_SCOREP
//  printf("Returning from Tau_stop_top_level_timer_if_necessary_task");
  return;
#endif /* TAU_SCOREP */
  TauInternalFunctionGuard protects_this_function;

  if (TauInternal_CurrentProfiler(tid)
      && TauInternal_CurrentProfiler(tid)->ParentProfiler == NULL
      && strcmp(TauInternal_CurrentProfiler(tid)->ThisFunction->GetName(), ".TAU application") == 0)
  {
    DEBUGPROFMSG("Found top level .TAU application timer"<<endl;);
    TAU_GLOBAL_TIMER_STOP();
  }
}

extern "C" void Tau_stop_top_level_timer_if_necessary(void) {
   Tau_stop_top_level_timer_if_necessary_task(Tau_get_thread());
}


extern "C" void Tau_disable_context_event(void *event) {
  TauContextUserEvent *e = (TauContextUserEvent *) event;
  e->SetContextEnabled(false);
}

extern "C" void Tau_enable_context_event(void *event) {
  TauContextUserEvent *e = (TauContextUserEvent *) event;
  e->SetContextEnabled(true);
}



extern "C" void Tau_track_memory(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackMemoryUtilization(true);
}


extern "C" void Tau_track_memory_here(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackMemoryHere();
}

extern "C" void Tau_track_power(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackPower();
}


extern "C" void Tau_track_power_here(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackPowerHere();
}

extern "C" void Tau_track_memory_headroom(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackMemoryUtilization(false);
}


extern "C" void Tau_track_memory_headroom_here(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackMemoryHeadroomHere();
}


extern "C" void Tau_enable_tracking_memory(void) {
  TauEnableTrackingMemory();
}


extern "C" void Tau_disable_tracking_memory(void) {
  TauDisableTrackingMemory();
}

extern "C" void Tau_disable_tracking_power(void) {
  TauDisableTrackingPower();
}

extern "C" void Tau_enable_tracking_power(void) {
  TauEnableTrackingPower();
}


extern "C" void Tau_enable_tracking_memory_headroom(void) {
  TauEnableTrackingMemoryHeadroom();
}


extern "C" void Tau_disable_tracking_memory_headroom(void) {
  TauDisableTrackingMemoryHeadroom();
}

extern "C" void Tau_set_interrupt_interval(int value) {
  TauSetInterruptInterval(value);
}


extern "C" void Tau_global_stop(void) {
  Tau_stop_current_timer();
}

///////////////////////////////////////////////////////////////////////////
extern "C" char * Tau_phase_enable(const char *group) {
  TauInternalFunctionGuard protects_this_function;
#ifdef TAU_PROFILEPHASE
  char *newgroup = new char[strlen(group)+16];
  sprintf(newgroup, "%s|TAU_PHASE", group);
  return newgroup;
#else /* TAU_PROFILEPHASE */
  return (char *) group;
#endif /* TAU_PROFILEPHASE */
} 

///////////////////////////////////////////////////////////////////////////
extern "C" char * Tau_phase_enable_once(const char *group, void **ptr) {
  /* We don't want to parse the group name string every time this is invoked.
     we compare a pointer and if necessary, perform the string operations  
     on the group name (adding  | TAU_PHASE to it). */
  if (*ptr == 0) {
    return Tau_phase_enable(group);
  } else {
    return (char *) NULL;
  }
} 

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_mark_group_as_phase(void *ptr)
{
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fptr = (FunctionInfo *)ptr;
  char *newgroup = Tau_phase_enable(fptr->GetAllGroups());
  fptr->SetPrimaryGroupName(newgroup);
}

///////////////////////////////////////////////////////////////////////////
extern "C" char const * Tau_append_iteration_to_name(int iteration, char const * name, int slen) {
  TauInternalFunctionGuard protects_this_function;
  char * buff = (char*)malloc(slen+128);
  sprintf(buff, "%s[%d]", name, iteration);
  return buff;
}

///////////////////////////////////////////////////////////////////////////
/* This routine creates dynamic timers and phases by embedding the
     iteration number in the name. isPhase argument tells whether we
     choose phases or timers. */
extern "C" void Tau_profile_dynamic_auto(int iteration, void **ptr, char *fname, char *type, TauGroup_t group, char *group_name, int isPhase)
{
  TauInternalFunctionGuard protects_this_function;

  char const * newName = Tau_append_iteration_to_name(iteration, fname, strlen(fname));

  /* create the pointer. */
  Tau_profile_c_timer(ptr, newName, type, group, group_name);

  /* annotate it as a phase if it is */
  if (isPhase)
    Tau_mark_group_as_phase(ptr);
  free((void*)newName);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_profile_param1l(long data, const char *dataname) {
  string dname(dataname);
#ifdef TAU_PROFILEPARAM
  TauProfiler_AddProfileParamData(data, dataname);
#ifdef TAU_SCOREP
  SCOREP_Tau_ParamHandle handle = SCOREP_TAU_INIT_PARAM_HANDLE;
  SCOREP_Tau_Parameter_INT64(&handle, dataname, data);
#endif
#endif
}

//void Tau_clear_pure_map();

/*
  The following is for supporting pure and elemental fortran subroutines
*/

struct PureMap : public TAU_HASH_MAP<string, FunctionInfo *> {
  virtual ~PureMap() {
    Tau_destructor_trigger();
	//Tau_clear_pure_map();
  }
};

PureMap & ThePureMap() 
{
  static PureMap map;
  return map;
}

map<string, int *>& TheIterationMap() {
  static map<string, int *> iterationMap;
  return iterationMap;
}

void *Tau_pure_search_for_function(const char *name)
{
  FunctionInfo *fi = 0;
  RtsLayer::LockDB();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(name);
  if (it != pure.end()) {
    fi = it->second;
  }
  RtsLayer::UnLockDB();
  return (void *) fi;
}

/* DON'T REMOVE THIS FUNCTION! 
 * When processing samples, you cannot allocate memory!
 * That means you can't create strings!
 * Some compilers call the std::string() constructor, which then calls malloc.
 * Therefore, we need gTauApplication to be a static string.
 * I realize this duplicates code, but that's just too bad.
 * Everything that signaling support touches has to be signal safe...
 */
void Tau_pure_start_task_string(const string name, int tid)
{
  TauInternalFunctionGuard protects_this_function;

  FunctionInfo *fi = 0;
  RtsLayer::LockDB();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(name);
  if (it == pure.end()) {
    tauCreateFI_signalSafe((void**)&fi, name, "", TAU_USER, "TAU_USER");
    pure[name] = fi;
  } else {
    fi = it->second;
  }
  RtsLayer::UnLockDB();
  Tau_start_timer(fi,0, tid);
}


extern "C" void Tau_pure_start_task(const char * n, int tid)
{
  TauInternalFunctionGuard protects_this_function;
  string name = n; // this is VERY bad if called from signalling! see above ^
  FunctionInfo * fi = NULL;

  PureMap & pure = ThePureMap();
  int exists = pure.count(name);
  if (exists > 0) {
    PureMap::iterator it = pure.find(name);
    fi = it->second;
  } 
  if (fi == NULL) {
    RtsLayer::LockEnv();
    PureMap::iterator it = pure.find(name);
    if (it == pure.end()) {
      tauCreateFI((void**)&fi, name, "", TAU_USER, "TAU_USER");
      pure[name] = fi;
    } else {
      fi = it->second;
    }
    RtsLayer::UnLockEnv();
  }
  Tau_start_timer(fi, 0, tid);
}

extern FunctionInfo* Tau_make_openmp_timer(const char * n, const char * t)
{
  TauInternalFunctionGuard protects_this_function;
  string name(n+string(" ")+ string(t)); // this is VERY bad if called from signalling! see above ^
  string type = ""; // this is VERY bad if called from signalling! see above ^
  FunctionInfo * fi = NULL;

  //printf("Tau_make_openmp_timer: n=%s, t = %s, PureMapSize=%d\n", n, t, ThePureMap().size());
  PureMap & pure = ThePureMap();
  int exists = pure.count(name);
  if (exists > 0) {
    PureMap::iterator it = pure.find(name);
    fi = it->second;
  }
  if (fi == NULL) {
    RtsLayer::LockEnv();
    PureMap::iterator it = pure.find(name);
    if (it == pure.end()) {
      tauCreateFI((void**)&fi, name, type, TAU_USER, "OpenMP");
      pure[name] = fi;
    } else {
      fi = it->second;
    }
    RtsLayer::UnLockEnv();
  }
  return fi;
}

extern "C" void Tau_pure_start_openmp_task(const char * n, const char * t, int tid) {
  FunctionInfo * fi = Tau_make_openmp_timer(n, t);
  Tau_start_timer(fi, 0, tid);
}

// This function will return a timer for the Collector API OpenMP state, if available
// This is called by the OpenMP collector API wrapper initialization...
FunctionInfo * Tau_create_thread_state_if_necessary(const char *name)
{
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fi = NULL;
  std::string n = name;
  RtsLayer::LockEnv();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(n);
  if (it == pure.end()) {
    tauCreateFI_signalSafe((void**)&fi, n, "", TAU_USER, "TAU_OMP_STATE");
    pure[n] = fi;
  } else {
    fi = it->second;
  }
  RtsLayer::UnLockEnv();
  return fi;
}

// This function will return a timer for the Collector API OpenMP state, if available
FunctionInfo * Tau_create_thread_state_if_necessary_string(string const & name)
{
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fi = NULL;

  RtsLayer::LockEnv();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(name);
  if (it == pure.end()) {
    tauCreateFI_signalSafe((void**)&fi, name, "", TAU_USER, "TAU_OMP_STATE");
    pure[name] = fi;
  } else {
    fi = it->second;
  }
  RtsLayer::UnLockEnv();
  return fi;
}

extern "C" void Tau_pure_start(const char *name)
{
  Tau_pure_start_task(name, Tau_get_thread());
}

extern "C" void Tau_pure_stop_task(char const * n, int tid)
{
  TauInternalFunctionGuard protects_this_function;
  string name = n;
  FunctionInfo * fi = NULL;

  RtsLayer::LockDB();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(name);
  if (it == pure.end()) {
    fprintf(stderr,
        "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\n"
        "TAU Error: You will likely get an overlapping timer message next\n\n", n);
  } else {
    fi = it->second;
  }
  RtsLayer::UnLockDB();
  Tau_stop_timer(fi, tid);
}

extern "C" void Tau_pure_stop(const char *name)
{
  Tau_pure_stop_task(name, Tau_get_thread());
}

extern "C" void Tau_static_phase_start(char const * name)
{
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fi = 0;
  string n = name;

  RtsLayer::LockDB();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(n);
  if (it == pure.end()) {
    tauCreateFI((void**)&fi, n, "", TAU_USER, "TAU_USER");
    Tau_mark_group_as_phase(fi);
    pure[n] = fi;
  } else {
    fi = it->second;
  }
  RtsLayer::UnLockDB();
  Tau_start_timer(fi, 1, Tau_get_thread());
}

extern "C" void Tau_static_phase_stop(char const * name)
{
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fi;
  string n = name;

  RtsLayer::LockDB();
  PureMap & pure = ThePureMap();
  PureMap::iterator it = pure.find(n);
  if (it == pure.end()) {
    fprintf(stderr,
        "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\n"
        "TAU Error: You will likely get an overlapping timer message next\n\n",
        name);
    RtsLayer::UnLockDB();
  } else {
    fi = it->second;
    RtsLayer::UnLockDB();
    Tau_stop_timer(fi, Tau_get_thread());
  }
}


static int *getIterationList(char const * name) {
  string searchName(name);
  map<string, int *>::iterator iit = TheIterationMap().find(searchName);
  if (iit == TheIterationMap().end()) {
    RtsLayer::LockEnv();
    int *iterationList = new int[TAU_MAX_THREADS];
    for (int i=0; i<TAU_MAX_THREADS; i++) {
      iterationList[i] = 0;
    }
    TheIterationMap()[searchName] = iterationList;
    RtsLayer::UnLockEnv();
  }
  return TheIterationMap()[searchName];
}

/* isPhase argument is 1 for phase and 0 for timer */
extern "C" void Tau_dynamic_start(char const * name, int isPhase)
{
  TauInternalFunctionGuard protects_this_function;
#ifndef TAU_PROFILEPHASE
  isPhase = 0;
#endif

  int *iterationList = getIterationList(name);

  int tid = RtsLayer::myThread();
  int itcount = iterationList[tid];

  FunctionInfo *fi = NULL;
  char const * newName = Tau_append_iteration_to_name(itcount, name, strlen(name));
  string n(newName);
  free((void*)newName);

  RtsLayer::LockDB();
  TAU_HASH_MAP<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    tauCreateFI((void**)&fi, n, "", TAU_USER, "TAU_USER");
    if (isPhase) {
      Tau_mark_group_as_phase(fi);
    }
    ThePureMap()[n] = fi;
  } else {
    fi = (*it).second;
  }
  RtsLayer::UnLockDB();
  Tau_start_timer(fi, isPhase, Tau_get_thread());
}


/* isPhase argument is ignored in Tau_dynamic_stop. For consistency with
 Tau_dynamic_start. */
extern "C" void Tau_dynamic_stop(char const * name, int isPhase)
{
  TauInternalFunctionGuard protects_this_function;
  int *iterationList = getIterationList(name);

  int tid = RtsLayer::myThread();
  int itcount = iterationList[tid];

  // increment the counter
  iterationList[tid]++;

  FunctionInfo *fi = NULL;
  char const * newName = Tau_append_iteration_to_name(itcount, name, strlen(name));
  string n(newName);
  free((void*)newName);

  RtsLayer::LockDB();
  TAU_HASH_MAP<string, FunctionInfo *>::iterator it = ThePureMap().find(n);
  if (it == ThePureMap().end()) {
    fprintf(stderr,
        "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\nTAU Error: You will likely get an overlapping timer message next\n\n",
        name);
    RtsLayer::UnLockDB();
    return;
  } else {
    fi = (*it).second;
  }
  RtsLayer::UnLockDB();
  Tau_stop_timer(fi, Tau_get_thread());
}


#if (!defined(TAU_WINDOWS))
extern "C" pid_t tau_fork() {
  pid_t pid;

  pid = fork();

#ifdef TAU_WRAP_FORK
  if (pid == 0) {
    TAU_REGISTER_FORK(RtsLayer::getPid(), TAU_EXCLUDE_PARENT_DATA);
  }
#endif

  return pid;
}
#endif /* TAU_WINDOWS */


//////////////////////////////////////////////////////////////////////
// Snapshot related routines
//////////////////////////////////////////////////////////////////////

extern "C" void Tau_profile_snapshot_1l(const char *name, int number) {
  char buffer[4096];
  sprintf (buffer, "%s %d", name, number);
  Tau_snapshot_writeIntermediate(buffer);
}

extern "C" void Tau_profile_snapshot(const char *name) {
  Tau_snapshot_writeIntermediate(name);
}


static int Tau_usesMPI = 0;
extern "C" void Tau_set_usesMPI(int value) {
  Tau_usesMPI = value;
}

extern "C" int Tau_get_usesMPI() {
  return Tau_usesMPI;
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_calls(void *handle, long *values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  values[0] = (long) ptr->GetCalls(tid);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_calls(void *handle, long values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  ptr->SetCalls(tid, values);
}

//////////////////////////////////////////////////////////////////////
void Tau_get_child_calls(void *handle, long* values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  values[0] = (long) ptr->GetSubrs(tid);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_child_calls(void *handle, long values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  ptr->SetSubrs(tid, values);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_inclusive_values(void *handle, double* values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  if (ptr) ptr->getInclusiveValues(tid, values);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_inclusive_values(void *handle, double* values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  if (ptr) ptr->SetInclTime(tid, values);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_exclusive_values(void *handle, double* values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  if (ptr) ptr->getExclusiveValues(tid, values);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_exclusive_values(void *handle, double* values, int tid) {
  FunctionInfo *ptr = (FunctionInfo *)handle;
  if (ptr) ptr->SetExclTime(tid, values);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_get_counter_info(const char ***counterNames, int *numCounters) {
  TauMetrics_getCounterList(counterNames, numCounters);
}

//////////////////////////////////////////////////////////////////////
//Fast but DO NOT use this call when calling the FunctionInfo DB
//or Profiler stack. 
extern "C" int Tau_get_local_tid(void) {
  TauInternalFunctionGuard protects_this_function;
  return RtsLayer::unsafeLocalThreadId();
}

#ifdef TAU_OPENMP
extern "C" void Tau_finalize_collector_api(void);
#endif

// this routine is called by the destructors of our static objects
// ensuring that the profiles are written out while the objects are still valid
void Tau_destructor_trigger() {
#ifdef TAU_OPENMP
  Tau_finalize_collector_api();
#endif
  Tau_memory_wrapper_disable();
//#ifndef JAVA
  Tau_stop_top_level_timer_if_necessary();
  Tau_global_setLightsOut();
//#endif
  if ((TheUsingDyninst() || TheUsingCompInst()) && TheSafeToDumpData()) {
#ifndef TAU_VAMPIRTRACE
    TAU_PROFILE_EXIT("FunctionDB destructor");
    TheSafeToDumpData() = 0;
#endif
  }
}

/*
   This is causing segfaults on exit.

   Destructors are called in any order as the application exits
   so we can't deallocate things from a destructor called on application exit.

   In any case, there's no need to deallocate memory as the application exits
   because the OS will do that.

void Tau_clear_pure_map(void) {
  // clear the hash map to eliminate memory leaks
  PureMap & mymap = ThePureMap();
  TAU_HASH_MAP<string, FunctionInfo *>::iterator it = mymap.begin();
  while ( it != mymap.end() ) {
	TAU_HASH_MAP<string, FunctionInfo *>::iterator eraseme = it;
	FunctionInfo * fi = eraseme->second;
    it++; // do this BEFORE the delete!
	// The top level timer is not allocated? Weird. IF you free it you'll get a segv.
	if (strcmp(fi->Name, ".TAU application") != 0)
		delete fi;
    //mymap.erase(eraseme);
  }
  mymap.clear(); 
}
*/

//////////////////////////////////////////////////////////////////////
extern "C" int Tau_create_task(void) {
  TauInternalFunctionGuard protects_this_function;

  int taskid;
  if (TAU_MAX_THREADS == 1) {
    printf("TAU: ERROR: Please re-configure TAU with -useropt=-DTAU_MAX_THREADS=100  and rebuild it to use the new TASK API\n");
  }
  taskid= RtsLayer::RegisterThread() - 1; /* it returns 1 .. N, we want 0 .. N-1 */
  /* specify taskid is a fake thread used in the Task API */
  Tau_thread_flags[taskid].Tau_is_thread_fake_for_task_api = 1; /* This thread is fake! */
 	//printf("create task with id: %d.\n", taskid); 
  return taskid;
}

/* because the OpenMP runtime collector API is a C file, and because
 * there are times when the main thread starts timers for the other
 * threads, we need to be able to lock the environment from C.
 */

extern "C" void Tau_lock_environment() {
  RtsLayer::LockEnv();
}

extern "C" void Tau_unlock_environment() {
  RtsLayer::UnLockEnv();
}

/**************************************************************************
 Query API allowing a program/library to query the TAU callstack
***************************************************************************/
void *Tau_query_current_event() {
  Profiler *profiler = TauInternal_CurrentProfiler(Tau_get_thread());
  return (void*)profiler;
}

const char *Tau_query_event_name(void *event) {
  if (event == NULL) {
    return NULL;
  }
  Profiler *profiler = (Profiler*) event;
  return profiler->ThisFunction->Name;
}

void *Tau_query_parent_event(void *event) {
  int tid = Tau_get_thread();
  void *topOfStack = &(Tau_thread_flags[tid].Tau_global_stack[0]);
  if (event == topOfStack) {
    return NULL;
  } else {
    long loc = Tau_convert_ptr_to_long(event);
    return (void*)(loc - (sizeof(Profiler)));
  }
}



//////////////////////////////////////////////////////////////////////
// User definable clock
//////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_user_clock(double value) {
  int tid = Tau_get_thread();
  metric_write_userClock(tid, value);
}

extern "C" void Tau_set_user_clock_thread(double value, int tid) {
  metric_write_userClock(tid, value);
}

extern "C" long Tau_convert_ptr_to_long(void *ptr) {
  long long a = (long long) ptr;
  long ret = (long) a;
  return ret;
}

extern "C" unsigned long Tau_convert_ptr_to_unsigned_long(void *ptr) {
  unsigned long long a = (unsigned long long) ptr;
  unsigned long ret = (unsigned long) a;
  return ret;
}

//////////////////////////////////////////////////////////////////////
// Sometimes we may link in a library that needs the POMP stuff
// Even when we're not using opari
//////////////////////////////////////////////////////////////////////

#ifdef TAU_SICORTEX
#define TAU_FALSE_POMP
#endif

#ifdef TAU_FALSE_POMP

#pragma weak POMP_MAX_ID=TAU_POMP_MAX_ID
int TAU_POMP_MAX_ID = 0;
#pragma weak pomp_rd_table=tau_pomp_rd_table
int *tau_pomp_rd_table = 0;

// #pragma weak omp_get_thread_num=tau_omp_get_thread_num
// extern "C" int tau_omp_get_thread_num() {
//   return 0;
// }

// #ifdef TAU_OPENMP
// extern "C" {
//   void pomp_parallel_begin();
//   void pomp_parallel_fork_();
//   void POMP_Parallel_fork();
//   void _tau_pomp_will_not_be_called() {
//     pomp_parallel_begin();
//     pomp_parallel_fork_();
//     POMP_Parallel_fork();
//   }
// }
// #endif
#endif

#ifndef TAU_BGP
extern "C" void Tau_Bg_hwp_counters_start(int *error) {
}

extern "C" void Tau_Bg_hwp_counters_stop(int* numCounters, x_uint64 counters[], int* mode, int *error) {
}

extern "C" void Tau_Bg_hwp_counters_output(int* numCounters, x_uint64 counters[], int* mode, int* error) {
}
#endif /* TAU_BGP */
                    

/***************************************************************************
 * $RCSfile: TauCAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.158 $   $Date: 2010/05/28 17:45:49 $
 * VERSION: $Id: TauCAPI.cpp,v 1.158 2010/05/28 17:45:49 sameer Exp $
 ***************************************************************************/


