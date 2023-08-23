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
#include <sstream>
#include <iostream>
#include <vector>
#include <mutex>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <sstream.h>
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#include <mutex>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauSnapshot.h>
#include <Profile/TauTrace.h>
#include <Profile/TauBacktrace.h>
#include <ctype.h>

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

#ifdef TAU_BEACON
#include <Profile/TauBeacon.h>
#endif /* TAU_BEACON */

#ifdef DEBUG_LOCK_PROBLEMS
#include <execinfo.h>
#endif
#if !defined(TAU_WINDOWS) && !defined(TAU_ANDROID) && !defined(_AIX) && !defined(TAU_NEC_SX)
#include <execinfo.h>
#endif

#include <Profile/TauPin.h>
#include <Profile/TauPluginInternals.h>
#include <Profile/TauPluginCPPTypes.h>

#ifdef CUPTI
#include <Profile/CuptiLayer.h>
#endif
#include <atomic>

using namespace tau;

extern "C" void Tau_shutdown(void);
//extern "C" void Tau_disable_collector_api();
extern int Tau_get_count_for_pvar(int index);
extern "C" long Tau_get_message_send_path(void);
extern "C" long Tau_get_message_recv_path(void);
extern "C" size_t Tau_util_return_hash_of_string(const char *name);
extern "C" void Tau_util_invoke_async_callback(unsigned int id, void * data);

extern unsigned int plugin_id_counter;
extern std::list < std::string > regex_list;
#ifdef TAU_UNWIND
bool Tau_unwind_unwindTauContext(int tid, unsigned long *addresses);
#endif

/* These are needed so that TauGpu.cpp can let the rest of TAU know that
 * it has been initialized.  PthreadLayer.cpp needs to know whether a
 * CUDA/CUPTI thread should be monitored - if it starts before TAU is
 * initialized, then we could have problems.  This prevents that. */
bool _tau_gpu_initialized = false;
bool& Tau_gpu_initialized(void) { return _tau_gpu_initialized; }
void Tau_gpu_initialized(bool init) { _tau_gpu_initialized = init; }

#define TAU_GEN_CONTEXT_EVENT(e, msg) TauContextUserEvent* e () { \
	static TauContextUserEvent ce(msg); return &ce; }

TAU_GEN_CONTEXT_EVENT(TheHeapMemoryEntryEvent,"Heap Memory Used (KB) at Entry")
TAU_GEN_CONTEXT_EVENT(TheHeapMemoryExitEvent,"Heap Memory Used (KB) at Exit")
TAU_GEN_CONTEXT_EVENT(TheHeapMemoryIncreaseEvent,"Increase in Heap Memory (KB)")
TAU_GEN_CONTEXT_EVENT(TheHeapMemoryDecreaseEvent,"Decrease in Heap Memory (KB)")

extern "C" void * Tau_get_profiler(const char *fname, const char *type, TauGroup_t group, const char *gr_name);

/*The padding is probably irrelevant for the dynamic (vector) implementation. Keeping for reference.*/
/* An array of this struct is shared by all threads.
 * To make sure we don't have false sharing, the struct is 64 bytes in size,
 * so that it fits exactly in one (or two) cache lines. That way, when one
 * thread updates its data in the array, it won't invalidate the cache line
 * for other threads. This is very important with timers, as all threads are
 * entering timers at the same time, and every thread will invalidate the
 * cache line otherwise.
 */
//#if !(defined(CRAYCC) || defined(TAU_NEC_SX))
//union Tau_thread_status_flags
//{
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
/*  struct {
    Profiler * Tau_global_stack;
    int Tau_global_stackdepth;
    int Tau_global_stackpos;
    int Tau_global_insideTAU;
    int Tau_is_thread_fake_for_task_api;
    int lightsOut;
  };

  char _pad[64];
};
#else*/
struct Tau_thread_status_flags {
  Profiler * Tau_global_stack = NULL;
  int Tau_global_stackdepth = 0;
  int Tau_global_stackpos = -1;
  int Tau_global_insideTAU = 0;
  int Tau_is_thread_fake_for_task_api = 0;
  int Tau_fake_thread_uses_cpu_metric = 0;
  int lightsOut = 0;
};
//#endif

#define STACK_DEPTH_INCREMENT 100
 struct CAPIThreadList : vector<Tau_thread_status_flags *>{
      CAPIThreadList(){
         //printf("Creating CapiThreadList at %p\n", this);
      }
     virtual ~CAPIThreadList(){
         //printf("Destroying CapiThreadList at %p, with size %ld\n", this, this->size());
         Tau_destructor_trigger();
     }
   };

//static CAPIThreadList Tau_thread_flags;
CAPIThreadList & TheCAPIThreadList() {
    static CAPIThreadList threadList;
    return threadList;
}

//static thread_local bool locallock=false;
std::mutex CAPIVectorMutex;
void checkTCAPIVector(int tid){
    if(TheCAPIThreadList().size()<=tid){
      std::lock_guard<std::mutex> guard(CAPIVectorMutex);
      while(TheCAPIThreadList().size()<=tid){
        TheCAPIThreadList().push_back(new Tau_thread_status_flags());
      }
    }
 }

static thread_local int local_tid = RtsLayer::myThread();
static thread_local Tau_thread_status_flags* flag_cache=0;
static inline Tau_thread_status_flags& getTauThreadFlag(int tid){

    if(tid == local_tid){
        if(flag_cache!=0){
            return *flag_cache;
        }
    }

    checkTCAPIVector(tid);

    //Tau_thread_status_flags& test = *(TheCAPIThreadList()[tid]);
    //printf("stackpos: %d on tid: %d\n", test.Tau_global_stackpos,tid);
    std::lock_guard<std::mutex> guard(CAPIVectorMutex);
    Tau_thread_status_flags* FlagOut=TheCAPIThreadList()[tid];
    if(tid == local_tid){
        if(flag_cache==0){
            flag_cache=FlagOut;
        }
    }
    return *FlagOut;//*TheCAPIThreadList()[tid];
}
//static inline void SetTauThreadFlag

#if defined (TAU_USE_TLS)
__thread int _Tau_global_insideTAU = 0;
__thread int lightsOut = 0;
#elif defined (TAU_USE_DTLS)
__declspec(thread) int _Tau_global_insideTAU = 0;
__declspec(thread) int lightsOut = 0;
#elif defined (TAU_USE_PGS)
#include "TauPthreadGlobal.h"
#endif
/* This is the ONE flag that indicates whether thread 0 has exited. */
static bool tauIsDestroyed = false;

bool& Tau_is_destroyed(void) {
    return tauIsDestroyed;
}

static void Tau_stack_checkInit() {
  static bool init = false;
  if (init) return;
  init = true;

#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  lightsOut = 0;
#elif defined(TAU_USE_PGS)
  TauGlobal::getInstance().getValue()->lightsOut = 0;
#else
  getTauThreadFlag(RtsLayer::unsafeLocalThreadId()).lightsOut = 0;
#endif
}

extern "C" int Tau_global_getLightsOut() {
  Tau_stack_checkInit();
#if defined (TAU_USE_TLS) || (TAU_USE_DTLS)
  return lightsOut;
#elif defined(TAU_USE_PGS)
  return TauGlobal::getInstance().getValue()->lightsOut;
#else
  return getTauThreadFlag(RtsLayer::unsafeLocalThreadId()).lightsOut;
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
  getTauThreadFlag(RtsLayer::unsafeLocalThreadId()).lightsOut = 1;
#endif
}

/* the task API does not have a real thread associated with the tid */
extern "C" int Tau_is_thread_fake(int tid) {
  return getTauThreadFlag(tid).Tau_is_thread_fake_for_task_api;
}

extern "C" void Tau_set_thread_fake(int tid) {
  getTauThreadFlag(tid).Tau_is_thread_fake_for_task_api = 1;
  //printf("Thread %d is fake!\n", tid);
  //Tau_print_simple_backtrace(tid);
}

extern "C" int Tau_is_fake_thread_use_cpu_metric(int tid) {
  return getTauThreadFlag(tid).Tau_fake_thread_uses_cpu_metric;
}

extern "C" void Tau_set_fake_thread_use_cpu_metric(int tid) {
  getTauThreadFlag(tid).Tau_fake_thread_uses_cpu_metric = 1;
  //printf("Thread %d uses CPU metric!\n", tid);
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
  return getTauThreadFlag(tid).Tau_global_insideTAU;
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

  volatile int * insideTAU = &getTauThreadFlag(tid).Tau_global_insideTAU; //TODO: Confirm this is correct with the new getter function reutnring a reference
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

  volatile int * insideTAU = &getTauThreadFlag(tid).Tau_global_insideTAU;
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
  int pos = getTauThreadFlag(tid).Tau_global_stackpos;
  if (pos < 0) {
    return NULL;
  }
  return &(getTauThreadFlag(tid).Tau_global_stack[pos]);
}

extern "C" Profiler *TauInternal_ParentProfiler(int tid) {
  int pos = getTauThreadFlag(tid).Tau_global_stackpos-1;
  if (pos < 0) {
    return NULL;
  }
  return &(getTauThreadFlag(tid).Tau_global_stack[pos]);
}

extern "C" char *TauInternal_CurrentCallsiteTimerName(int tid) {
  if(TauInternal_CurrentProfiler(tid) != NULL)
    if(TauInternal_CurrentProfiler(tid)->ThisFunction != NULL)
      return TauInternal_CurrentProfiler(tid)->ThisFunction->Name;
  return NULL;
}

//#define REPORT_ENTRY_EXIT 1
#ifdef REPORT_ENTRY_EXIT
///////////////////////////////////////////////////////////////////////////
static void reportEntryExit (bool entry, FunctionInfo *caller, int tid) {
    // Tau_global_stackpos starts at -1.
    int position = getTauThreadFlag(tid).Tau_global_stackpos + 1;
    // This function is called by Tau_stop_timer before the timer is popped
    if (!entry) position--;
    std::string tabs{""};
    if (position > 0) {
        std::string _tabs(position, ' ');
        tabs = _tabs;
    }
    static std::mutex mtx_;
    std::lock_guard<std::mutex> lck (mtx_);
    TAU_VERBOSE("%06u %03d %02d %s%s: %s\n", RtsLayer::getTid(), tid, position, tabs.c_str(),
        (entry ? "Entry" : "Exit "), caller->GetName());
    fflush(stderr);
    //Tau_print_simple_backtrace(tid);
}
#endif

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_start_timer(void *functionInfo, int phase, int tid) {
  FunctionInfo *fi = (FunctionInfo *) functionInfo;
#ifdef REPORT_ENTRY_EXIT
  reportEntryExit(true, fi, tid);
#endif

  // Don't start throttled timers
  if (fi && fi->IsThrottled()) return;
  if (Tau_global_getLightsOut()) return;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_UNWIND
  if(TauEnv_get_region_addresses()) {
    unsigned long unwound_addresses[TAU_SAMP_NUM_ADDRESSES];
    if(fi->StartAddr == 0) {
      int unwind_ret = Tau_unwind_unwindTauContext(tid, unwound_addresses);
      if(unwind_ret && unwound_addresses[0] >= 2) {
          fi->StartAddr = unwound_addresses[2];
      }
    }
  }
#endif

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    // OK, this gets called WAY too much, just to make sure that TAU is initialized
    // before timers start
    //Tau_sampling_init_if_necessary();
    Tau_sampling_suspend(tid);
  }
#endif /* _AIX */
#endif

#ifdef TAU_TRACK_IDLE_THREADS
  /* If we are performing idle thread tracking, we start a top level timer */
  if (tid != 0) {
    Tau_create_top_level_timer_if_necessary_task(tid);
  }
#endif // TAU_TRACK_IDLE_THREADS


#ifdef TAU_EPILOG
  esd_enter(fi->GetFunctionId());
#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */
  return;
#endif // TAU_EPILOG

#ifdef TAU_VAMPIRTRACE
  uint64_t TimeStamp = vt_pform_wtime();
#ifdef TAU_VAMPIRTRACE_5_12_API
  vt_enter(VT_CURRENT_THREAD, (uint64_t *) &TimeStamp, fi->GetFunctionId());
#else
  vt_enter((uint64_t *) &TimeStamp, fi->GetFunctionId());
#endif /* TAU_VAMPIRTRACE_5_12_API */
#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */
  return;
#endif

#ifdef TAU_SCOREP
  SCOREP_Tau_EnterRegion(fi->GetFunctionId());
#endif


  // move the stack pointer
  //printf("Incrementing stack pointer at 401 for tid:%d\n",tid);
  getTauThreadFlag(tid).Tau_global_stackpos++; /* push */

  if (getTauThreadFlag(tid).Tau_global_stackpos >= getTauThreadFlag(tid).Tau_global_stackdepth) {
    int oldDepth = getTauThreadFlag(tid).Tau_global_stackdepth;
    int newDepth = oldDepth + STACK_DEPTH_INCREMENT;
    //printf("%d: NEW STACK DEPTH: %d\n", tid, newDepth);
    //Profiler *newStack = (Profiler *) malloc(sizeof(Profiler)*newDepth);

    //A deep copy is necessary here to keep the profiler pointers up to date
    Profiler *newStack = (Profiler *) calloc(newDepth, sizeof(Profiler));
    memcpy(newStack, getTauThreadFlag(tid).Tau_global_stack, oldDepth*sizeof(Profiler));
    TAU_VERBOSE("Growing stack: depth=%d, size=%ld\n", newDepth, newDepth*sizeof(Profiler));

    int tmpDepth=oldDepth;
    Profiler *fixP = &(newStack[oldDepth]);
    while(tmpDepth>0){
    	tmpDepth--;
    	fixP->ParentProfiler=&(newStack[tmpDepth]);
    	fixP=fixP->ParentProfiler;
    }

    free(getTauThreadFlag(tid).Tau_global_stack);

    getTauThreadFlag(tid).Tau_global_stack = newStack;
    getTauThreadFlag(tid).Tau_global_stackdepth = newDepth;
  }

  Profiler *p = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);

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
  int mydepth = getTauThreadFlag(tid).Tau_global_stackpos;
  if (mydepth >= userspecifieddepth) {
#ifndef TAU_WINDOWS
#ifndef _AIX
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif /* _AIX */
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
    double heapmem = Tau_max_RSS();
    TAU_CONTEXT_EVENT(TheHeapMemoryEntryEvent(), heapmem);
    p->heapmem = heapmem;
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
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
    // if the unwind depth should be "automatic", then get the stack for right now
    if (TauEnv_get_ebs_unwind_depth() == 0) {
      Tau_sampling_event_start(tid, p->address);
    }
  }
#endif /* _AIX */
#endif
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_lite_start_timer(void *functionInfo, int phase)
{
  FunctionInfo *fi = (FunctionInfo *)functionInfo;
  // Don't start throttled timers
  if (fi->IsThrottled()) return;
  if (Tau_global_getLightsOut()) return;

  if (TauEnv_get_lite_enabled()) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    int tid = RtsLayer::myThread();
    // move the stack pointer
    //printf("Incrementing stack pointer at 521 for tid:%d\n",tid);
    getTauThreadFlag(tid).Tau_global_stackpos++; /* push */
    Profiler *pp = TauInternal_ParentProfiler(tid);
    if (fi) {
      fi->IncrNumCalls(tid);    // increment number of calls
    }
    if (pp && pp->ThisFunction) {
      pp->ThisFunction->IncrNumSubrs(tid);    // increment parent's child calls
    }

    if (getTauThreadFlag(tid).Tau_global_stackpos >= getTauThreadFlag(tid).Tau_global_stackdepth) {
      int oldDepth = getTauThreadFlag(tid).Tau_global_stackdepth;
      int newDepth = oldDepth + STACK_DEPTH_INCREMENT;
      Profiler *newStack = (Profiler *)malloc(sizeof(Profiler) * newDepth);
      memcpy(newStack, getTauThreadFlag(tid).Tau_global_stack, oldDepth * sizeof(Profiler));
      getTauThreadFlag(tid).Tau_global_stack = newStack;
      getTauThreadFlag(tid).Tau_global_stackdepth = newDepth;
    }
    Profiler *p = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
    // Record metrics in reverse order so wall clock metrics are recorded after PAPI, etc.
    RtsLayer::getUSecD(tid, p->StartTime, 1);

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
    Tau_start_timer(functionInfo, phase, Tau_get_thread());
  }
}





///////////////////////////////////////////////////////////////////////////
static void reportOverlap (FunctionInfo *stack, FunctionInfo *caller, int tid) {
    fprintf(stderr, "[%d:%d][%d:%d] TAU: Runtime overlap: found %s (%p) on the stack, but stop called on %s (%p)\n",
    RtsLayer::getPid(), RtsLayer::getTid(), RtsLayer::myNode(), RtsLayer::myThread(),
    stack->GetName(), stack, caller->GetName(), caller);
#if !defined(TAU_WINDOWS) && !defined(TAU_ANDROID) && !defined(_AIX) && !defined(TAU_NEC_SX)
    if(TauEnv_get_ebs_enabled()) {
        Tau_sampling_stop_sampling();
    }
    void* callstack[128];
    int i, frames = backtrace(callstack, 128);
    char** strs = backtrace_symbols(callstack, frames);
    for (i = 0; i < frames; ++i) {
        fprintf(stderr,"%s\n", strs[i]);
    }
    free(strs);
#endif
    fprintf(stderr,"Timer Stack:\n");
    int position = getTauThreadFlag(tid).Tau_global_stackpos; /* pop */
    while (position > 0) {
        auto profiler = &(getTauThreadFlag(tid).Tau_global_stack[position]);
        auto fi = profiler->ThisFunction;
        fprintf(stderr,"%s\n", fi->GetName());
        position--;
     }
	 abort();
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_stop_timer(void *function_info, int tid ) {
  FunctionInfo *fi = (FunctionInfo *) function_info;
#ifdef REPORT_ENTRY_EXIT
  reportEntryExit(false, fi, tid);
#endif

  // Don't stop throttled timers
  if (fi->IsThrottled()) return;
  //if (Tau_global_getLightsOut()) return;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_UNWIND
  if(TauEnv_get_region_addresses()) {
    unsigned long unwound_addresses[TAU_SAMP_NUM_ADDRESSES];
    if(fi->StopAddr == 0) {
      int unwind_ret = Tau_unwind_unwindTauContext(tid, unwound_addresses);
      if(unwind_ret && unwound_addresses[0] >= 2) {
          fi->StopAddr = unwound_addresses[2];
      }
    }
  }
#endif

  double currentHeap = 0.0;
  bool enableHeapTracking;

  Profiler *profiler;
#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_suspend(tid);
  }
#endif /* _AIX */
#endif

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
#ifndef _AIX
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif /* _AIX */
#endif
  return;
#endif

#ifdef TAU_VAMPIRTRACE
  uint64_t TimeStamp = vt_pform_wtime();

#ifdef TAU_VAMPIRTRACE_5_12_API
  vt_exit(VT_CURRENT_THREAD, (uint64_t *)&TimeStamp);
#else
  vt_exit((uint64_t *)&TimeStamp);
#endif /* TAU_VAMPIRTRACE_5_12_API */

#ifndef TAU_WINDOWS
#ifndef _AIX
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif /* _AIX */
#endif
  return;
#endif

#ifdef TAU_SCOREP
  SCOREP_Tau_ExitRegion(fi->GetFunctionId());
#endif

  if (getTauThreadFlag(tid).Tau_global_stackpos < 0) {
#ifndef TAU_WINDOWS
#ifndef _AIX
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif /* _AIX */
#endif
    return;
  }

  profiler = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);

  /* Check for overlapping timers */
  while (profiler->ThisFunction != fi) {
    /* We might have an inconsistent stack because of throttling. If one thread
     * throttles a routine while it is on the top of the stack of another thread
     * it will remain there until a stop is called on its parent. Check for this
     * condition before printing a overlap error message. */
    if (profiler->ThisFunction->IsThrottled()) {
      profiler->Stop();
      getTauThreadFlag(tid).Tau_global_stackpos--; /* pop */
      profiler = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
    } else {
#ifdef __PIN__ /* PIN can't resolve exits very well - it is not guaranteed */
      TAU_VERBOSE("[%d:%d] PIN: Stopping %s instead of %s\n", RtsLayer::myNode(), RtsLayer::myThread(), profiler->ThisFunction->GetName(), fi->GetName());
      TAU_VERBOSE("[%d:%d] PIN: %s return not found by PIN, stopping it instead of %s\n", RtsLayer::myNode(), RtsLayer::myThread(), profiler->ThisFunction->GetName(), fi->GetName());
      profiler->ThisFunction->SetType("[IGNORE - Return not found by PIN]");
      profiler->Stop();
      getTauThreadFlag(tid).Tau_global_stackpos--; /* pop */
      profiler = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
#else
      reportOverlap(profiler->ThisFunction, fi, tid);
#endif
    }
  }

#ifdef TAU_DEPTH_LIMIT
  static int userspecifieddepth = TauEnv_get_depth_limit();
  int mydepth = getTauThreadFlag(tid).Tau_global_stackpos;
  if (mydepth >= userspecifieddepth) {
    getTauThreadFlag(tid).Tau_global_stackpos--; /* pop */
#ifndef TAU_WINDOWS
#ifndef _AIX
    if (TauEnv_get_ebs_enabled()) {
      Tau_sampling_resume(tid);
    }
#endif /* _AIX */
#endif
    return;
  }
#endif /* TAU_DEPTH_LIMIT */

  /* check memory */
  if (enableHeapTracking && profiler->heapmem) {
    double oldheap = profiler->heapmem;
    double difference = currentHeap - oldheap;
    if (difference > 0) {
      TAU_CONTEXT_EVENT(TheHeapMemoryIncreaseEvent(), difference);
    } else if (difference < 0) {
      TAU_CONTEXT_EVENT(TheHeapMemoryDecreaseEvent(), -difference);
    }
  }

  profiler->Stop(tid);

  getTauThreadFlag(tid).Tau_global_stackpos--;

#ifndef TAU_WINDOWS
#ifndef _AIX
  if (TauEnv_get_ebs_enabled()) {
    Tau_sampling_resume(tid);
  }
#endif /* _AIX */
#endif
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_lite_stop_timer(void *function_info)
{
  FunctionInfo *fi = (FunctionInfo *)function_info;
  // Don't stop throttled timers
  if (fi->IsThrottled()) {
      //...unless it is already on the stack.
      Profiler *profiler;
      int tid = RtsLayer::myThread();
      profiler = (Profiler *)&(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
      // if this timer isn't on the top of the stack, stop it.
      if (profiler && profiler->ThisFunction != fi) {
          return;
      }
  }
  if (Tau_global_getLightsOut()) return;

  if (TauEnv_get_lite_enabled()) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    int tid = RtsLayer::myThread();
    double timeStamp[TAU_MAX_COUNTERS] = { 0 };
    double delta[TAU_MAX_COUNTERS] = { 0 };
    RtsLayer::getUSecD(tid, timeStamp);

    Profiler *profiler;
    profiler = (Profiler *)&(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);

    for (int k = 0; k < Tau_Global_numCounters; k++) {
      delta[k] = timeStamp[k] - profiler->StartTime[k];
    }

    if (profiler && profiler->ThisFunction != fi) { /* Check for overlapping timers */
      reportOverlap(profiler->ThisFunction, fi, tid);
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
    getTauThreadFlag(tid).Tau_global_stackpos--; /* pop */
  } else {
    Tau_stop_timer(function_info, Tau_get_thread());
  }
}

extern "C" Profiler * Tau_get_current_profiler(void) {
    int tid = RtsLayer::myThread();
    return &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_stop_current_timer_task(int tid)
{
  if (getTauThreadFlag(tid).Tau_global_stackpos >= 0) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    Profiler * profiler = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
    /* We might have an inconstant stack because of throttling. If one thread
     * throttles a routine while it is on the top of the stack of another thread
     * it will remain there until a stop is called on its parent. Check for this
     * condition before printing a overlap error message. */
    while (!(profiler->ThisFunction->GetProfileGroup() & RtsLayer::TheProfileMask()) &&
            (getTauThreadFlag(tid).Tau_global_stackpos >= 0))
    {
      profiler->Stop();
      getTauThreadFlag(tid).Tau_global_stackpos--; /* pop */
      profiler = &(getTauThreadFlag(tid).Tau_global_stack[getTauThreadFlag(tid).Tau_global_stackpos]);
    }

    FunctionInfo * functionInfo = profiler->ThisFunction;
    Tau_stop_timer(functionInfo, tid);
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_stop_current_timer()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;
  Tau_stop_current_timer_task(RtsLayer::myThread());
}

///////////////////////////////////////////////////////////////////////////

extern "C" int Tau_show_profiles()
{
  for (int tid = 0; tid < TheCAPIThreadList().size(); ++tid) {
    int pos = getTauThreadFlag(tid).Tau_global_stackpos;
    while (pos >= 0) {
      Profiler * p = &(getTauThreadFlag(tid).Tau_global_stack[pos]);
      TAU_VERBOSE(" *** Alfred Profile (%d:%d:%d) :  %s\n", Tau_get_node(), tid, pos, p->ThisFunction->Name);
      pos--;
    }
  }

  return 0;
}

/* Used by SOS plugin to start all currently running timers
 * because "tracing" is not enabled until after some timers
 * are started. */
extern Profiler * Tau_get_timer_at_stack_depth(int pos) {
    return &(getTauThreadFlag(RtsLayer::myThread()).Tau_global_stack[pos]);
}

extern Profiler * Tau_get_timer_at_stack_depth_task(int pos, int tid) {
    return &(getTauThreadFlag(tid).Tau_global_stack[pos]);
}

extern "C" void Tau_stop_all_timers(int tid)
{
  TauInternalFunctionGuard protects_this_function;
  // prevent this thread from coming entering twice (from exit and from stop)
  static thread_local bool in_here{false};
  if (in_here) return;
  in_here = true;
  /* Thread 0 can stop other threads' timers, and they could try to do
     the same thing at the same time. So lock. */
  static std::mutex mtx;
  std::lock_guard<std::mutex> lck (mtx);

  //Make sure even throttled routines are stopped.
  while (getTauThreadFlag(tid).Tau_global_stackpos >= 0) {
    int stackpos = getTauThreadFlag(tid).Tau_global_stackpos;
    Profiler * p = getTauThreadFlag(tid).Tau_global_stack + stackpos;
    Tau_stop_timer(p->ThisFunction, tid);
    // Make sure the stack is shrinking
    // Throttling in multi-thread is very goofy right now so this can happen
    if (getTauThreadFlag(tid).Tau_global_stackpos == stackpos) {
      getTauThreadFlag(tid).Tau_global_stackpos--;
    }
  }
  in_here = false;
}


#ifdef TAU_TRACK_IDLE_THREADS
//Prevent Profile::Stop from internally stopping all threads when using OpenMP
//if TAU is already closing all threads
static bool thread_local tauStoppingAllThreads = false;

//Prevent Profile::Stop from internally stopping all threads when using OpenMP
//if TAU is already closing all threads
extern "C" bool Tau_check_Stopping_All_Threads(){
        return tauStoppingAllThreads;
}

#endif /* TAU_TRACK_IDLE_THREADS */


inline void Tau_profile_exit_threads(int begin_index)
{
  if(!TheSafeToDumpData()) {
    return;
  }
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;
  // Stop the collector API. The main thread may exit with running
  // worker threads. When those threads try to exit, they will
  // try to stop timers that aren't running.
#ifdef TAU_OPENMP
  //Tau_disable_collector_api();
#endif

//Prevent Profile::Stop from internally stopping all threads when using OpenMP
//if TAU is already closing all threads

#ifdef TAU_TRACK_IDLE_THREADS
  if(begin_index == 0)
      tauStoppingAllThreads = true;
#endif /* TAU_TRACK_IDLE_THREADS */
#ifdef TAU_ANDROID
  bool su = JNIThreadLayer::IsMgmtThread();
#endif

  for(int tid = begin_index; tid < TheCAPIThreadList().size(); ++tid) {
#ifdef TAU_ANDROID
    if (su) {
      JNIThreadLayer::SuThread(tid);
    }
#endif
    Tau_stop_all_timers(tid);
  }
}

extern "C" void Tau_profile_exit_most_threads()
{
  Tau_profile_exit_threads(1);
  // DO NOT call Tau_shutdown() - thread 0 is still active
}

extern "C" void Tau_profile_exit_all_threads()
{
  /* Set up a static flag to make sure we only do this once,
   * as it gets called from many, many destructors. */
  static bool done = false;
  if (done) { return; }
  // because this function gets called from the preload shutdown (static
  // global destructor) make sure CUPTI processing is finished.
  Tau_flush_gpu_activity();
  Tau_profile_exit_threads(0);
  Tau_shutdown();
  done = true;
}


extern "C" void Tau_profile_exit()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;
  int tid = RtsLayer::myThread();
  Tau_stop_all_timers(tid);
  if (tid == 0) {
    Tau_shutdown();
  }
}


///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_exit(const char * msg) {
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  /*Invoke plugins only if both plugin path and plugins are specified
  *    *Do this first, because the plugin can write TAU_METADATA as recommendations to the user*/
  if(Tau_plugins_enabled.function_finalize) {
    Tau_plugin_event_function_finalize_data_t plugin_data;
    plugin_data.junk = -1;
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_FUNCTION_FINALIZE, "*", &plugin_data);
  }

#if defined(TAU_OPENMP)
  Tau_profile_exit_most_threads();
#elif defined(TAU_CUDA)
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
extern "C" void Tau_post_init(void) {
  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.post_init) {
    Tau_plugin_event_post_init_data_t plugin_data;
    plugin_data.tid = Tau_get_thread();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_POST_INIT, "*", &plugin_data);
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_set_node(int node) {
  TauInternalFunctionGuard protects_this_function;
  if (node >= 0) TheSafeToDumpData()=1;
  RtsLayer::setMyNode(node);
  atexit(Tau_destructor_trigger);
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

/* Helper functions for fixing threading */
#if (!defined(TAU_WINDOWS))
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
bool validate_thread() {
    long pid = getpid();
    long tid = gettid();
    int tau_tid = RtsLayer::myThread();
    if (tau_tid == 0 && pid != tid) {
        TAU_VERBOSE("Registering thread! %ld != %ld, so need new thread\n", pid, tid);
        Tau_register_thread();
    }
    return true;
}
#else
bool validate_thread() {return false;} // do nothing
#endif

//////////////////////////////////////////////////////////////////////
extern "C" int Tau_get_thread(void) {
  TauInternalFunctionGuard protects_this_function;
  thread_local static bool do_once = validate_thread();
  return RtsLayer::myThread();
}

#ifdef CUPTI
class cupti_buffer_tracking {
public:
    cupti_buffer_tracking() {
        created = 0;
        processed = 0;
    }
    int open_buffers(void) {
        return created - processed;
    }
    int created;
    int processed;
};

cupti_buffer_tracking& Tau_get_cupti_buffer_tracker(void) {
    static cupti_buffer_tracking tracker;
    return tracker;
}

void Tau_cupti_buffer_created(void) {
    Tau_get_cupti_buffer_tracker().created++;
    //printf("BUFFERS! Created: %d, processed: %d\n", Tau_get_cupti_buffer_tracker().created, Tau_get_cupti_buffer_tracker().processed);
    //fflush(stdout);
}

void Tau_cupti_buffer_processed(void) {
    Tau_get_cupti_buffer_tracker().processed++;
    //printf("BUFFERS! Created: %d, processed: %d\n", Tau_get_cupti_buffer_tracker().created, Tau_get_cupti_buffer_tracker().processed);
    //fflush(stdout);
}
#endif

#ifdef TAU_ENABLE_ROCTRACER
extern void Tau_roctracer_flush_tracing(void);
#endif /* TAU_ENABLE_ROCTRACER */

#if defined(TAU_ENABLE_ROCPROFILER) || defined(TAU_ENABLE_ROCPROFILERV2)
extern void Tau_rocprofiler_pool_flush(void);
#endif
#ifdef TAU_USE_OMPT_5_0
extern void Tau_ompt_flush_trace(void);
#endif


#ifdef TAU_ENABLE_ROCM
extern void TauFlushRocmEventsIfNecessary(void);
#endif /* TAU_ENABLE_ROCM */


extern "C" void Tau_flush_gpu_activity(void) {
   TAU_VERBOSE("TAU: flushing asynchronous GPU events...\n");
#ifdef CUPTI
    static bool did_once = false;
    if (RtsLayer::myThread() != 0) return;
    if (Tau_init_check_initialized() &&
        !Tau_global_getLightsOut()) {
        if (Tau_get_cupti_buffer_tracker().created > Tau_get_cupti_buffer_tracker().processed) {
            if (RtsLayer::myNode() == 0) {
                if (did_once) {
                    TAU_VERBOSE("TAU: ...still flushing asynchronous CUDA events...\n");
                } else {
                    TAU_VERBOSE("TAU: flushing asynchronous CUDA events...\n");
                    did_once = true;
                }
            }
            cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
        }
    }
#endif
#if defined(TAU_ENABLE_ROCPROFILER) || defined(TAU_ENABLE_ROCPROFILERV2)
   Tau_rocprofiler_pool_flush();
#endif
#ifdef TAU_ENABLE_ROCTRACER
   TAU_VERBOSE("TAU: flushing asynchronous ROCM/HIP events...\n");
   Tau_roctracer_flush_tracing();
#endif /* TAU_ENABLE_ROCTRACER */
#ifdef TAU_ENABLE_ROCM
  TauFlushRocmEventsIfNecessary();
#endif /* TAU_ENABLE_ROCM */
#ifdef TAU_USE_OMPT_5_0
   Tau_ompt_flush_trace();
#endif
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump(void) {
  TauInternalFunctionGuard protects_this_function;

  Tau_flush_gpu_activity();
  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.dump) {
    Tau_plugin_event_dump_data_t plugin_data;
    plugin_data.tid = RtsLayer::myThread();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_DUMP, "*", &plugin_data);
  } else {
    TauProfiler_DumpData();
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_invoke_plugin_phase_entry(void *functionInfo) {
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fi = (FunctionInfo *) functionInfo;

  if(Tau_plugins_enabled.phase_entry) {
    Tau_plugin_event_phase_entry_data_t plugin_data;
    plugin_data.phase_name = fi->GetName();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_PHASE_ENTRY, fi->GetName(), &plugin_data);
  }

  return 0;

}

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_invoke_plugin_phase_exit(void *functionInfo) {
  TauInternalFunctionGuard protects_this_function;
  FunctionInfo *fi = (FunctionInfo *) functionInfo;

  if(Tau_plugins_enabled.phase_exit) {
    Tau_plugin_event_phase_exit_data_t plugin_data;
    plugin_data.phase_name = fi->GetName();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_PHASE_EXIT, fi->GetName(), &plugin_data);
  }

  return 0;

}

/* Plugin API */
extern "C" size_t Tau_create_trigger(const char *name) {
  static size_t trigger_counter = 0;
  TauInternalFunctionGuard protects_this_function;

  static std::mutex mtx;
  std::lock_guard<std::mutex> lck (mtx);
  size_t retval =  trigger_counter;
  trigger_counter++;

  return retval;
}

extern "C" void Tau_trigger(size_t id, void * data) {
  TauInternalFunctionGuard protects_this_function;
  Tau_util_invoke_callbacks_for_trigger_event(TAU_PLUGIN_EVENT_TRIGGER, id, data);
}

/* This mutex is used to control access to the plugin triggers */
std::mutex & TriggerMutex() {
  static std::mutex mtx;
  return mtx;
}

extern "C" void Tau_enable_plugin_for_specific_event(int ev, const char *name, unsigned int id)
{
  TauInternalFunctionGuard protects_this_function;
  size_t hash = Tau_util_return_hash_of_string(name);
  PluginKey key(ev, hash);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  Tau_get_plugins_for_named_specific_event()[key].insert(id);
  if(plugins_for_ompt_event[ev].is_ompt())
    plugins_for_ompt_event[ev].insert(id);

}

extern "C" void Tau_disable_plugin_for_specific_event(int ev, const char *name, unsigned int id)
{
  TauInternalFunctionGuard protects_this_function;
  size_t hash = Tau_util_return_hash_of_string(name);
  PluginKey key(ev, hash);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  Tau_get_plugins_for_named_specific_event()[key].erase(id);
  if(plugins_for_ompt_event[ev].is_ompt())
    plugins_for_ompt_event[ev].erase(id);

}

extern "C" void Tau_disable_all_plugins_for_specific_event(int ev, const char *name)
{
  TauInternalFunctionGuard protects_this_function;
  size_t hash = Tau_util_return_hash_of_string(name);
  PluginKey key(ev, hash);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  Tau_get_plugins_for_named_specific_event()[key].clear();
  if(plugins_for_ompt_event[ev].is_ompt())
    plugins_for_ompt_event[ev].clear();
}

extern "C" void Tau_enable_all_plugins_for_specific_event(int ev, const char *name)
{
  TauInternalFunctionGuard protects_this_function;
  size_t hash = Tau_util_return_hash_of_string(name);
  PluginKey key(ev, hash);

  std::lock_guard<std::mutex> lck (TriggerMutex());
  for(unsigned int i = 0 ; i < plugin_id_counter; i++) {
    Tau_get_plugins_for_named_specific_event()[key].insert(i);
  }

  if(plugins_for_ompt_event[ev].is_ompt()) {
    for(unsigned int i = 0 ; i < plugin_id_counter; i++) {
        plugins_for_ompt_event[ev].insert(i);
    }
  }

}

extern "C" void Tau_enable_plugin_for_trigger_event(int ev, size_t hash, unsigned int id)
{
  TauInternalFunctionGuard protects_this_function;
  PluginKey key(ev, hash);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  Tau_get_plugins_for_named_specific_event()[key].insert(id);

}

extern "C" void Tau_disable_plugin_for_trigger_event(int ev, size_t hash, unsigned int id)
{
  TauInternalFunctionGuard protects_this_function;
  PluginKey key(ev, hash);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  Tau_get_plugins_for_named_specific_event()[key].erase(id);

}

extern "C" void Tau_disable_all_plugins_for_trigger_event(int ev, size_t hash)
{
  TauInternalFunctionGuard protects_this_function;
  PluginKey key(ev, hash);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  Tau_get_plugins_for_named_specific_event()[key].clear();
}

extern "C" void Tau_enable_all_plugins_for_trigger_event(int ev, size_t hash)
{
  TauInternalFunctionGuard protects_this_function;
  PluginKey key(ev, hash);

  std::lock_guard<std::mutex> lck (TriggerMutex());
  for(unsigned int i = 0 ; i < plugin_id_counter; i++) {
    Tau_get_plugins_for_named_specific_event()[key].insert(i);
  }
}

extern "C" void Tau_add_regex(const char * r)
{
  TauInternalFunctionGuard protects_this_function;
  std::string s(r);
  std::lock_guard<std::mutex> lck (TriggerMutex());
  regex_list.push_back(s);
}

/* Plugin API */

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_dump_prefix(const char *prefix) {
  TauInternalFunctionGuard protects_this_function;
  for (int i = 0 ; i < RtsLayer::getTotalThreads() ; i++)
    TauProfiler_DumpData(false, i, prefix);
  return 0;
}

extern x_uint64 TauTraceGetTimeStamp(int tid);

///////////////////////////////////////////////////////////////////////////
extern "C" int Tau_get_current_stack_depth(int tid) {
  return getTauThreadFlag(tid).Tau_global_stackpos;
}
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
    char errormsg[1064];
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    return 1;
  }

  fprintf(fp, "Thread\tStack\tCalls\tIncl.\tExcl.\tName\tTimestamp:\t%llu\n", TauTraceGetTimeStamp(0));
  for (tid = 0 ; tid < RtsLayer::getTotalThreads() ; tid++) {
    pos = getTauThreadFlag(tid).Tau_global_stackpos;
	TauProfiler_updateIntermediateStatistics(tid);
    while (pos >= 0) {
	  Profiler profiler = getTauThreadFlag(tid).Tau_global_stack[pos];
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
//#if defined(PTHREADS)
  if (RtsLayer::myNode() != -1) {
    int tmp = RtsLayer::RegisterThread();
    TAU_VERBOSE("[TauCAPI]: Tau_register_thread, mynode %d, tid %d of %d\n", RtsLayer::myNode(), RtsLayer::myThread(), tmp);
  }
  else {
    TAU_VERBOSE("[TauCAPI]: Tau_register_thread, do not register thread, mynode %i, tid %i\n", RtsLayer::myNode(), RtsLayer::getTid());
  }

// #else
//   RtsLayer::RegisterThread();
// #endif
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
  TAU_VERBOSE("Tau_shutdown!\n");
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
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
extern "C" int __real__num_pes(void);
#else
extern "C" int __real_shmem_n_pes(void);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_sendmsg(int type, int destination, int length)
{
  if (!RtsLayer::TheEnableInstrumentation()) return;

#ifdef TAU_PROFILEPARAM
#ifndef TAU_DISABLE_PROFILEPARAM_IN_MPI
#ifdef TAU_PROFILE_PATHS
//  TAU_PROFILE_PARAM1L(Tau_get_message_send_path(), "message send path id");
#else
  TAU_PROFILE_PARAM1L(length, "message size");
#endif /* TAU_PROFILE_PATHS */
#endif /* TAU_DISABLE_PROFILEPARAM_IN_MPI */
#endif  /* TAU_PROFILEPARAM */

  TAU_EVENT(TheSendEvent(), length);

  if (TauEnv_get_comm_matrix()) {
    if (destination >= tau_totalnodes(0,0)) {
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
      tau_totalnodes(1,__real__num_pes());
#else
      tau_totalnodes(1,__real_shmem_n_pes());
#endif /* SHMEM_1_1 || SHMEM_1_2 */
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
#ifdef TAU_PROFILE_PATHS
  TAU_PROFILE_PARAM1L(Tau_get_message_recv_path(), "message receive path id");
#else
  TAU_PROFILE_PARAM1L(length, "message size");
#endif /* TAU_PROFILE_PATHS */
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
  if (RtsLayer::TheEnableInstrumentation() && TauEnv_get_tracing()) {
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
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
        tau_totalnodes(1,__real__num_pes());
#else
        tau_totalnodes(1,__real_shmem_n_pes());
#endif /* SHMEM_1_1 || SHMEM_1_2 */
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

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_barrier_all_start(int tag)
{
  if (RtsLayer::TheEnableInstrumentation() && TauEnv_get_tracing()) {
      TauTraceBarrierAllStart(tag);
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_barrier_all_end(int tag)
{
  if (RtsLayer::TheEnableInstrumentation() && TauEnv_get_tracing()) {
      TauTraceBarrierAllEnd(tag);
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_rma_collective_begin(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root)
{
  if (RtsLayer::TheEnableInstrumentation() && TauEnv_get_tracing()) {
      TauTraceRMACollectiveBegin(tag, type, start, stride, size, data_in, data_out, root);
  }
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_rma_collective_end(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root)
{
  if (RtsLayer::TheEnableInstrumentation() && TauEnv_get_tracing()) {
      TauTraceRMACollectiveEnd(tag, type, start, stride, size, data_in, data_out, root);
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
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_barrier_all_start(int tag) {
  TauTraceBarrierAllStart(tag);
}
///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_barrier_all_end(int tag) {
  TauTraceBarrierAllEnd(tag);
}
extern "C" void Tau_trace_rma_collective_begin(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root)
{
  TauTraceRMACollectiveBegin(tag, type, start, stride, size, data_in, data_out, root);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trace_rma_collective_end(int tag, int type, int start, int stride, int size, int data_in, int data_out, int root)
{
  TauTraceRMACollectiveEnd(tag, type, start, stride, size, data_in, data_out, root);
}

#endif /* TAU_MPI || TAU_SHMEM*/

///////////////////////////////////////////////////////////////////////////
// User Defined Events
///////////////////////////////////////////////////////////////////////////

class pure_context_userevent_map_t : public TAU_HASH_MAP<std::string, TauContextUserEvent*> {
private:
  int tid;
  static atomic<int> num_threads;
public:
  pure_context_userevent_map_t() : tid(RtsLayer::myThread()) { num_threads++; }
  virtual ~pure_context_userevent_map_t() {
    if ((tid == 0 || --num_threads == 0) && RtsLayer::isMainThread()) {
        Tau_destructor_trigger();
    }
  }
};

atomic<int> pure_context_userevent_map_t::num_threads{0};

TauContextUserEvent * Tau_find_context_userevent_internal(const char* name)
{
    static std::mutex mtx;
    static pure_context_userevent_map_t pureMap;
    static thread_local pure_context_userevent_map_t my_pureMap;
    TauInternalFunctionGuard protects_this_function;
    TauContextUserEvent *ue = nullptr;
    std::string tmp{name};
    /* First, check if this thread has seen this event before */
    pure_context_userevent_map_t::iterator it = my_pureMap.find(tmp);
    if (it != my_pureMap.end()) {
        ue = (*it).second;
        return ue;
    }
    /* if not, check the global map */
    std::lock_guard<std::mutex> lck (mtx);
    it = pureMap.find(tmp);
    if (it == pureMap.end()) {
        ue = new TauContextUserEvent(name);
        /* Add it to the global map */
        pureMap[tmp] = ue;
    } else {
        ue = (*it).second;
    }
    /* Add it to my local map */
    my_pureMap[tmp] = ue;
    return ue;
}

class pure_userevent_map_t : public TAU_HASH_MAP<std::string, TauUserEvent*> {
private:
  int tid;
  static atomic<int> num_threads;
public:
  pure_userevent_map_t() : tid(RtsLayer::myThread()) {
    num_threads++;
  }
  virtual ~pure_userevent_map_t() {
    if ((tid == 0 || --num_threads == 0) && RtsLayer::isMainThread()) {
        Tau_destructor_trigger();
    }
  }
};

atomic<int> pure_userevent_map_t::num_threads{0};

TauUserEvent * Tau_find_userevent_internal(const char* name) {
    static std::mutex mtx;
    static pure_userevent_map_t pureUserEventAtomicMap;
    static thread_local pure_userevent_map_t my_pureUserEventAtomicMap;
    TauInternalFunctionGuard protects_this_function;
    TauUserEvent *ue = nullptr;
    std::string tmp{name};
    /* First, check to see if it's in the thread's local map */
    pure_userevent_map_t::iterator it = my_pureUserEventAtomicMap.find(tmp);
    if (it != my_pureUserEventAtomicMap.end()) {
        ue = (*it).second;
        return ue;
    }
    /* Not in the local map, so check the global map */
    std::lock_guard<std::mutex> lck (mtx);
    it = pureUserEventAtomicMap.find(tmp);
    if (it == pureUserEventAtomicMap.end()) {
        ue = new TauUserEvent(name);
        /* Add it to the global map */
        pureUserEventAtomicMap[tmp] = ue;
    } else {
        ue = (*it).second;
    }
    /* Add it to the local map */
    my_pureUserEventAtomicMap[tmp] = ue;
    return ue;
}

extern "C" void * Tau_get_userevent(char const * name) {
    TauInternalFunctionGuard protects_this_function;
    TauUserEvent *ue = Tau_find_userevent_internal(name);
    return (void *) ue;
}

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
    void * ue = Tau_find_context_userevent_internal(name);
    return ue;
}

extern "C" void Tau_get_context_userevent(void **ptr, const char *name) {
    TauInternalFunctionGuard protects_this_function;
    *ptr = Tau_find_context_userevent_internal(name);
}

extern "C" void Tau_context_userevent(void *ue, double data) {
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->TriggerEvent(data);
}

extern "C" void Tau_trigger_context_event_thread(const char *name, double data, int tid) {
  TauInternalFunctionGuard protects_this_function;
  void *ue;
  Tau_get_context_userevent(&ue, name);
  Tau_context_userevent_thread(ue, data, tid);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trigger_context_event(const char *name, double data) {
  TauInternalFunctionGuard protects_this_function;
  void *ue;
  Tau_get_context_userevent(&ue, name);
  Tau_context_userevent(ue, data);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trigger_userevent(const char *name, double data) {
  TauInternalFunctionGuard protects_this_function;
  void *ue = Tau_get_userevent(name);
  Tau_userevent(ue, data);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_trigger_userevent_thread(const char *name, double data, int tid) {
  TauInternalFunctionGuard protects_this_function;
  void *ue = Tau_get_userevent(name);
  Tau_userevent_thread(ue, data, tid);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_context_userevent_thread(void *ue, double data, int tid) {
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->TriggerEvent(data, tid);
}

///////////////////////////////////////////////////////////////////////////
extern "C" void Tau_context_userevent_thread_ts(void *ue, double data, int tid, double ts) {
  TauInternalFunctionGuard protects_this_function;
  TauContextUserEvent *t = (TauContextUserEvent *) ue;
  t->TriggerEventTS(data, tid, ts);
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
  /* When PDT instruments a program, this is the FIRST TAU API call
   * so we need to make sure that TAU is initialized.  Otherwise,
   * race conditions can happen when requesting locks (i.e. in OpenMP)
   */
  static int do_this_once = Tau_init_initializeTAU();
  if (*ptr == 0) {
      TauInternalFunctionGuard protects_this_function;
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
}

///////////////////////////////////////////////////////////////////////////

static string& gTauApplication()
{
  static string g = string(".TAU application");
  return g;
}

// forward declare the function we need to use - it's defined later
extern void Tau_pure_start_task_string(const string name, int tid);

static inline void expandVector(vector<bool>* bv,int dex){
    while(bv->size()<=dex){
        bv->push_back(false);
    }
}

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
  static vector<bool> initializing;//[TAU_MAX_THREADS] = { false }; //TODO: DYNATHREAD
  static vector<bool> initthread;//[TAU_MAX_THREADS] = { false }; //TODO: DYNATHREAD

  static std::mutex mtx;
  if (!initialized && (initializing.size()<=tid || !initializing[tid])) {
    std::lock_guard<std::mutex> lck (mtx);
    expandVector(&initializing,tid);
    expandVector(&initthread,tid);
    if (!initialized) {
      // whichever thread got here first, has the lock and will create the
      // FunctionInfo object for the top level timer.
      if (!TauInternal_CurrentProfiler(tid)) {
#if defined(PTHREADS) && defined(TAU_GPU)
        if (tid != 0 && RtsLayer::myNode() == -1) {
          TAU_VERBOSE("Found node=-1\n");
        }
        else {
          initthread[tid] = true;
          initializing[tid] = true;
          Tau_pure_start_task_string(gTauApplication(), tid);
          atexit(Tau_profile_exit_all_threads);
          initializing[tid] = false;
          initialized = true;
        }
#else
        initthread[tid] = true;
        initializing[tid] = true;
        Tau_pure_start_task_string(gTauApplication(), tid);
        atexit(Tau_profile_exit_all_threads);
        initializing[tid] = false;
        initialized = true;
#endif
      }
    }
  }

  if (initthread.size()<=tid || !initthread[tid]) {
    std::lock_guard<std::mutex> lck (mtx);
    expandVector(&initializing,tid);
    expandVector(&initthread,tid);
    // if there is no top-level timer, create one - But only create one FunctionInfo object.
    // that should be handled by the Tau_pure_start_task call.
    if (!TauInternal_CurrentProfiler(tid)) {
#if defined(PTHREADS) && defined(TAU_GPU)
      if (tid != 0 && RtsLayer::myNode() == -1) {
        TAU_VERBOSE("Found node=-1\n");
      }
      else {
        initthread[tid] = true;
        initializing[tid] = true;
        Tau_pure_start_task_string(gTauApplication(), tid);
        initializing[tid] = false;
      }
#else
      initthread[tid] = true;
      initializing[tid] = true;
      Tau_pure_start_task_string(gTauApplication(), tid);
      initializing[tid] = false;
#endif
    }
  }

#endif
}

#ifdef TAU_ENABLE_ROCTRACER
extern void Tau_roctracer_start_tracing(void);
extern void Tau_roctracer_stop_tracing(void);
#endif /* TAU_ENABLE_ROCTRACER */
#ifdef TAU_ENABLE_ROCPROFILERV2
extern void Tau_rocprofv2_stop(void);
#endif /* TAU_ENABLE_ROCPROFILERV2 */



extern "C" void Tau_create_top_level_timer_if_necessary(void) {
  if ((RtsLayer::myNode() == -1) && (Tau_get_thread() != 0)) {
    TauEnv_set_nodeNegOneSeen(TauEnv_get_nodeNegOneSeen()+1);
  }
  return Tau_create_top_level_timer_if_necessary_task(Tau_get_thread());

}


extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid)
{
#ifdef TAU_SCOREP
//  printf("Returning from Tau_stop_top_level_timer_if_necessary_task");
  return;
#endif /* TAU_SCOREP */
  TauInternalFunctionGuard protects_this_function;

  Profiler * current = TauInternal_CurrentProfiler(tid);
  /*if(current){
    printf("current Profiler: %p\n",current);
  }*/
  if (current
      && current->ParentProfiler == NULL
      && strcmp(current->ThisFunction->GetName(), ".TAU application") == 0)
  {
    DEBUGPROFMSG("Found top level .TAU application timer"<<endl;);
    TAU_GLOBAL_TIMER_STOP();
  }
}

extern "C" const char * Tau_get_current_timer_name(int tid) {
   return TauInternal_CurrentProfiler(tid)->ThisFunction->GetName();
}

extern "C" void Tau_stop_top_level_timer_if_necessary(void) {
   static bool done = false;
   if(done){return;}
   Tau_stop_top_level_timer_if_necessary_task(Tau_get_thread());
#ifdef TAU_ENABLE_ROCTRACER
   Tau_roctracer_stop_tracing();
#endif /* TAU_ENABLE_ROCTRACER */
#ifdef TAU_ENABLE_ROCPROFILERV2
     Tau_rocprofv2_stop();
#endif /* TAU_ENABLE_ROCPROFILERV2 */

   done = true;
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

extern "C" void Tau_track_load(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackLoad();
}

extern "C" void Tau_track_memory_rss_and_hwm(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackMemoryFootPrint();
}

extern "C" void Tau_track_memory_rss_and_hwm_here(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackMemoryFootPrintHere();
}


extern "C" void Tau_track_power_here(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackPowerHere();
}

extern "C" void Tau_track_load_here(void) {
  TauInternalFunctionGuard protects_this_function;
  TauTrackLoadHere();
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

extern "C" void Tau_disable_tracking_load(void) {
  TauDisableTrackingLoad();
}

extern "C" void Tau_enable_tracking_load(void) {
  TauEnableTrackingLoad();
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

bool& Tau_is_pthread_tracking_enabled() {
    static bool enabled = true;
    return enabled;
}

extern "C" void Tau_disable_pthread_tracking(void) {
    Tau_is_pthread_tracking_enabled() = false;
}

extern "C" void Tau_enable_pthread_tracking(void) {
    Tau_is_pthread_tracking_enabled() = true;
}

extern "C" void Tau_global_stop(void) {
    /* Enable this check if you are getting lots of profiles writing
     * out early... */
    /*
  if (Tau_thread_flags[RtsLayer::myThread()].Tau_global_stackpos == 0) {
    abort();
  }
  */
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

/*
  The following is for supporting pure and elemental fortran subroutines
*/
struct PureMap : public TAU_HASH_MAP<string, FunctionInfo *> {
private:
  int tid;
  static atomic<int> num_threads;
public:
  PureMap() : tid(num_threads++) { }
  virtual ~PureMap() {
    if ((tid == 0 || --num_threads == 0) && RtsLayer::isMainThread()) {
        Tau_destructor_trigger();
    }
  }
};

atomic<int> PureMap::num_threads{0};

FunctionInfo * Tau_get_function_info_internal(
    string fname, const char *type,
    TauGroup_t group, const char *gr_name,
    bool create = true, bool phase = false, bool signal_safe = false)  {
  /* This is the thread_local cache map, so we can speed up lookups.  If we
   * have seen the timer before, we won't have to lock the main map to look
   * it up.  If we haven't another thread may have created it, so we'll
   * then check the main map.  If we don't find it, optionally create it
   * and insert into both maps. */
  static thread_local PureMap local_pure;
  FunctionInfo *fi = nullptr;

  /* First, check if the FI is in the thread local map */
  PureMap::iterator it = local_pure.find(fname);
  /* found? */
  if (it != local_pure.end()) {
    fi = it->second;
    return fi;
  }

  /* Not found, so check the shared map */
  static std::mutex mtx;
  static PureMap pure;
  /* Acquire control of the map */
  std::lock_guard<std::mutex> lck (mtx);
  it = pure.find(fname);
  /* Found? */
  if (it != pure.end()) {
    fi = it->second;
    /* Save to local map to speed up next search */
    local_pure[fname] = fi;
    return fi;
  }
  /* Not found, so we need to create it, if requested */
  if (create) {
    if (signal_safe) {
        tauCreateFI_signalSafe((void**)&fi, fname, type, group, gr_name);
    } else {
        tauCreateFI((void**)&fi, fname, type, group, gr_name);
    }
    pure[fname] = fi;
    local_pure[fname] = fi;
    if (phase) {
        Tau_mark_group_as_phase(fi);
    }
  }
  /* return the fi pointer, might be nullptr if not created. */
  return fi;
}

map<string, vector<int> *>& TheIterationMap() {
  static map<string, vector<int> *> iterationMap;
  return iterationMap;
}

extern "C" void * Tau_get_profiler(const char *fname, const char *type, TauGroup_t group, const char *gr_name)
{
  FunctionInfo *f;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();

  DEBUGPROFMSG("Inside get_profiler group = " << group<<endl;);
  string name(fname);
  // since we're using new, we should set InitData to true in FunctionInfoInit
  if (group == TAU_MESSAGE) {
    if (gr_name && strcmp(gr_name, "TAU_MESSAGE") == 0) {
      //f = new FunctionInfo(fname, type, group, "MPI", true);
      f = Tau_get_function_info_internal(name, type, group, "MPI");
    } else {
      //f = new FunctionInfo(fname, type, group, gr_name, true);
      f = Tau_get_function_info_internal(name, type, group, gr_name);
    }
  } else {
    //f = new FunctionInfo(fname, type, group, gr_name, true);
    f = Tau_get_function_info_internal(name, type, group, gr_name);
  }

  return (void *)f;
}

extern "C" void *Tau_pure_search_for_function(const char *name, int create)
{
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  string fname(name);
  FunctionInfo *fi = Tau_get_function_info_internal(fname, "", TAU_USER, "TAU_USER", create != 0);
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
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  FunctionInfo *fi = Tau_get_function_info_internal(name, "", TAU_DEFAULT, "TAU_DEFAULT");
  Tau_start_timer(fi,0, tid);
}


extern "C" void Tau_pure_start_task_group(const char * n, int tid, const char * group)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  string name = n; // this is VERY bad if called from signalling! see above ^
  FunctionInfo *fi = Tau_get_function_info_internal(name, "", TAU_USER, "TAU_USER");
  Tau_start_timer(fi, 0, tid);
}

extern "C" void Tau_pure_start_task(const char * n, int tid)
{
    Tau_pure_start_task_group(n, tid, "TAU_USER");
}

FunctionInfo* Tau_make_cupti_sample_timer(const char * filename, const char * function, int lineno)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  stringstream ss;
  ss << function << " [{" << filename << "}{" << lineno << "}]";

  string name = string(ss.str());
  //string name = string(function);
  //string dstream_name = string(ss.str());

  FunctionInfo *fi = Tau_get_function_info_internal(name, "", TAU_USER, "CUPTI_SAMPLES");
  return fi;
}

extern FunctionInfo* Tau_make_openmp_timer(const char * n, const char * t)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  /* Could be a new thread, make sure we don't deadlock */
  static thread_local int do_this_once_too = RtsLayer::RegisterThread();
  string name; // this is VERY bad if called from signalling! see above ^
  if (strcmp(t,"") == 0) {
    name = string(n); // this is VERY bad if called from signalling! see above ^
  } else {
    name = string(n) + string(" ") + string(t); // this is VERY bad if called from signalling! see above ^
  }
  FunctionInfo *fi = Tau_get_function_info_internal(name, "", TAU_USER, "OpenMP");
  return fi;
}

extern "C" void Tau_pure_start_openmp_task(const char * n, int tid) {
  FunctionInfo * fi = Tau_make_openmp_timer(n, "");
  Tau_start_timer(fi, 0, tid);
}

extern "C" void Tau_pure_stop_openmp_task(const char * n, int tid) {
  FunctionInfo * fi = Tau_make_openmp_timer(n, "");
  Tau_stop_timer(fi, tid);
}

// This function will return a timer for the Collector API OpenMP state, if available
// This is called by the OpenMP collector API wrapper initialization...
FunctionInfo * Tau_create_thread_state_if_necessary(const char *name)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  std::string fname = name;
  FunctionInfo *fi = Tau_get_function_info_internal(fname, "", TAU_USER, "TAU_OMP_STATE");
  return fi;
}

// This function will return a timer for the Collector API OpenMP state, if available
FunctionInfo * Tau_create_thread_state_if_necessary_string(string const & name)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  FunctionInfo *fi = Tau_get_function_info_internal(name, "", TAU_USER, "TAU_OMP_STATE", true, false, true);
  return fi;
}

extern "C" void Tau_pure_start(const char *name)
{
  Tau_pure_start_task(name, Tau_get_thread());
}

extern "C" void tau_print_entry(const char *name) {
  TAU_VERBOSE("TAU ENTRY: %s\n", name);
  Tau_pure_start(name);
}

extern "C" void Tau_pure_stop_task(char const * n, int tid)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  string name = n;
  FunctionInfo *fi = Tau_get_function_info_internal(name, "", TAU_USER, "", false);
  if (fi == nullptr) {
      fprintf(stderr,
          "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\n"
          "TAU Error: You will likely get an overlapping timer message next\n\n", n);
      return;
  }
  Tau_stop_timer(fi, tid);
}

extern "C" void Tau_pure_stop(const char *name)
{
  Tau_pure_stop_task(name, Tau_get_thread());
}

extern "C" void tau_print_exit(const char *name) {
  TAU_VERBOSE("TAU EXIT: %s\n", name);
  Tau_pure_stop(name);
}

extern "C" void Tau_static_phase_start(char const * name)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  string n = name;

  FunctionInfo *fi = Tau_get_function_info_internal(n, "", TAU_USER, "TAU_USER", true, true);
  Tau_start_timer(fi, 1, Tau_get_thread());
}

extern "C" void * Tau_get_function_info(const char *fname, const char *type, TauGroup_t group, const char *gr_name)  {
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  string n = fname;
  FunctionInfo *fi = Tau_get_function_info_internal(n, type, group, gr_name);
  return (void *) fi;
}

extern "C" void Tau_static_phase_stop(char const * name)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  string n = name;
  FunctionInfo *fi = Tau_get_function_info_internal(n, "", TAU_USER, "", false);
  if (fi == nullptr) {
      fprintf(stderr,
          "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\n"
          "TAU Error: You will likely get an overlapping timer message next\n\n",
          name);
      return;
  }
  Tau_stop_timer(fi, Tau_get_thread());
}

static vector<int> *getIterationList(char const * name) {
  static std::mutex mtx;
  //static map<string, int *> iterationMap;???
  string searchName(name);
  map<string, vector<int> *>::iterator iit = TheIterationMap().find(searchName);
  if (iit == TheIterationMap().end()) {
    std::lock_guard<std::mutex> lck (mtx);
    vector<int> *iterationList = new vector<int>;
    TheIterationMap()[searchName] = iterationList;
  }
  return TheIterationMap()[searchName];
}

/* isPhase argument is 1 for phase and 0 for timer */
extern "C" void Tau_dynamic_start(char const * name, int isPhase)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
#ifndef TAU_PROFILEPHASE
  isPhase = 0;
#endif
  vector<int> *iterationList = getIterationList(name);
  int tid = RtsLayer::myThread();
  while(iterationList->size()<=tid){
     iterationList->push_back(0);
  }
  int itcount = (*iterationList)[tid];
  const char * newName = Tau_append_iteration_to_name(itcount, name, strlen(name));
  string n(newName);
  free((void*)newName);
  FunctionInfo *fi = Tau_get_function_info_internal(n, "", TAU_USER, "", true, (isPhase != 0));
  Tau_start_timer(fi, isPhase, Tau_get_thread());
}


/* isPhase argument is ignored in Tau_dynamic_stop. For consistency with
 Tau_dynamic_start. */
extern "C" void Tau_dynamic_stop(char const * name, int isPhase)
{
  TauInternalFunctionGuard protects_this_function;
  /* We have to make sure we initialize TAU first - just in case we are
   * entering TAU for the first time, because Tau_get_function_info_internal (below) will
   * potentially construct a top level timer, which will recursively enter
   * this function. */
  static int do_this_once = Tau_init_initializeTAU();
  vector<int> *iterationList = getIterationList(name);

  int tid = RtsLayer::myThread();
  int itcount = (*iterationList)[tid];

  // increment the counter
  (*iterationList)[tid]++;

  char const * newName = Tau_append_iteration_to_name(itcount, name, strlen(name));
  string n(newName);
  free((void*)newName);
  FunctionInfo *fi = Tau_get_function_info_internal(n, "", TAU_USER, "", false);
  if (fi == nullptr) {
      fprintf(stderr,
          "\nTAU Error: Routine \"%s\" does not exist, did you misspell it with TAU_STOP()?\nTAU Error: You will likely get an overlapping timer message next\n\n",
          name);
      return;
  }
  Tau_stop_timer(fi, Tau_get_thread());

  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.dump) {
    Tau_plugin_event_dump_data_t plugin_data;
    plugin_data.tid = RtsLayer::myThread();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_DUMP, "*", &plugin_data);
  }

}

////

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

static int Tau_usesSHMEM = 0;
extern "C" void Tau_set_usesSHMEM(int value) {
  Tau_usesSHMEM = value;
}

extern "C" int Tau_get_usesSHMEM() {
  return Tau_usesSHMEM;
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
//extern "C" void Tau_finalize_collector_api(void);
#endif
#ifdef TAU_USE_OMPT_5_0
extern void Tau_ompt_finalize(void);
#endif

#ifdef TAU_OTF2
extern void TauTraceOTF2ToggleFlushAtExit(bool);
#endif

#ifdef TAU_ENABLE_LEVEL_ZERO
void TauL0DisableProfiling(void);
#endif


// this routine is called by the destructors of our static objects
// ensuring that the profiles are written out while the objects are still valid
void Tau_destructor_trigger() {
  //TAU_VERBOSE("calling Tau_destructor_trigger\n");
  /* Set up a static flag to make sure we only do this once,
   * as it gets called from many, many destructors. */
  static bool once = false;
  if (once) { return; }
  once = true;

  FunctionInfo::disable_metric_cache(); //TODO: This may not be needed with fixes to object locking!
  TAU_VERBOSE("Entering Tau_destructor_trigger...\n");

#ifndef TAU_WINDOWS
  // STOP ALL SAMPLING ON ALL THREADS!
  Tau_sampling_stop_sampling();
#endif
#ifdef TAU_OTF2
  TauTraceOTF2ToggleFlushAtExit(true);
#endif
  Tau_flush_gpu_activity();
  // make sure TAU doesn't profile the IO
  Tau_global_incr_insideTAU();
#ifdef TAU_USE_OMPT_5_0
  Tau_ompt_finalize();
#endif
#ifdef TAU_ENABLE_LEVEL_ZERO
  TauL0DisableProfiling();
#endif
  // prevent any threads from handling their own exit
  tauIsDestroyed = true;
// First, make sure all thread timers have stopped
  Tau_profile_exit_all_threads();
  Tau_memory_wrapper_disable();
//#ifndef JAVA
  Tau_stop_top_level_timer_if_necessary();
  Tau_global_setLightsOut();
  TheSafeToDumpData() = 0;
//#endif
  if ((TheUsingDyninst() || TheUsingCompInst()) && TheSafeToDumpData()) {
#ifndef TAU_VAMPIRTRACE
    Tau_exit("FunctionDB destructor");
    TheSafeToDumpData() = 0;
#endif
  }
  TAU_VERBOSE("Exiting Tau_destructor_trigger!\n");
}

//////////////////////////////////////////////////////////////////////
extern "C" int Tau_create_task(void) {
  TauInternalFunctionGuard protects_this_function;

  int taskid;
  /*if (TAU_MAX_THREADS == 1) { //TODO: DYNATHREAD
    printf("TAU: ERROR: Please re-configure TAU with -useropt=-DTAU_MAX_THREADS=100  and rebuild it to use the new TASK API\n");
  }*/
  taskid = Tau_RtsLayer_createThread();
  // taskid= RtsLayer::RegisterThread() - 1; /* it returns 1 .. N, we want 0 .. N-1 */
  /* specify taskid is a fake thread used in the Task API */

  Tau_set_thread_fake(taskid);
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
  void *topOfStack = &(getTauThreadFlag(tid).Tau_global_stack[0]);
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

#ifdef TAU_MPI_T

#include <mpi.h>

void Tau_fill_mpi_t_pvar_events(TauUserEvent*** event, int pvar_index, int pvar_count) {
  int return_val, namelen, verb, varclass, bind, threadsup, i;
  int readonly, continuous, atomic;
  char event_name[TAU_NAME_LENGTH + 1] = "";
  char concat_event_name[TAU_NAME_LENGTH + 1] = "";
  int desc_len;
  char description[TAU_NAME_LENGTH + 1] = "";
  MPI_Datatype datatype;
  MPI_T_enum enumtype;

  namelen = desc_len = TAU_NAME_LENGTH;
  return_val = MPI_T_pvar_get_info(pvar_index/*IN*/,
    event_name /*OUT*/,
    &namelen /*INOUT*/,
    &verb /*OUT*/,
    &varclass /*OUT*/,
    &datatype /*OUT*/,
    &enumtype /*OUT*/,
    description /*description: OUT*/,
    &desc_len /*desc_len: INOUT*/,
    &bind /*OUT*/,
    &readonly /*OUT*/,
    &continuous /*OUT*/,
    &atomic/*OUT*/);


  // clean up description for non-ascii characters and new lines
  for (i=0; i < strlen(description); i++) {
    if ((description[i] == '\n') || (!isprint(description[i]))) {
      description[i] = ' ';
    }
  }
  // clean up event_name for non-ascii characters and new lines
  for (i=0; i < strlen(event_name); i++) {
    if ((event_name[i] == '\n') || (!isprint(event_name[i]))) {
      event_name[i] = ' ';
    }
  }
  if(pvar_count == 1) {
    sprintf(concat_event_name, "%s (%s)", event_name, description);
    TAU_VERBOSE("Concat Event name = %s\n", concat_event_name);
    (*event)[0] = new TauUserEvent(concat_event_name);
  } else {
    for(i=0; i < pvar_count; i++) {
      sprintf(concat_event_name, "%s (%s)[%d]", event_name, description, i);
      TAU_VERBOSE("Concat Event name = %s\n", concat_event_name);
      (*event)[i] = new TauUserEvent(concat_event_name);
    }
  }

  /* Add a metadata field */
  sprintf(concat_event_name, "MPI_T PVAR[%d]: %s", pvar_index, event_name);
  TAU_METADATA(concat_event_name, description);
}

//Static variables with file scope
static TauUserEvent *** pvarEvents = NULL;
static char pvarnamearray[300];

TauUserEvent & ThePVarsMPIEvents(const int current_pvar_index, const int current_pvar_subindex, const int *tau_pvar_count, const int num_pvars) {
    /*All this routine does is to return the event at the current PVAR index and subindex*/

    return *(pvarEvents[current_pvar_index][current_pvar_subindex]);
}

TauUserEvent & PvarName(const int current_pvar_index, const int current_pvar_subindex) {
    /*All this routine does is to return the event at the current PVAR index and subindex*/

    return *(pvarEvents[current_pvar_index][current_pvar_subindex]);
    //return 0;
}

/*Allocate events to track PVARs*/
extern "C" void Tau_allocate_pvar_event(int num_pvars, const int *tau_pvar_count) {
    static int tau_previous_pvar_count = 0;
    int i,j;

    /* If this function is being invoked for the first time, allocate event buffers using malloc.
     * If the number of pvars changes during runtime, reallocate event buffers accordingly*/
    if(!pvarEvents) {
        pvarEvents = (TauUserEvent***)calloc(num_pvars, sizeof(TauUserEvent**));
        for(i=0; i < num_pvars; i++) {
          pvarEvents[i] = (TauUserEvent**)calloc(tau_pvar_count[i], sizeof(TauUserEvent*));
          Tau_fill_mpi_t_pvar_events(&(pvarEvents[i]), i, tau_pvar_count[i]);
        }
    } else if ((tau_previous_pvar_count > 0) && (num_pvars > tau_previous_pvar_count) ) {
        pvarEvents = (TauUserEvent***)realloc(pvarEvents, sizeof(TauUserEvent**)*num_pvars);
        for(j=tau_previous_pvar_count; j < num_pvars; j++) {
          pvarEvents[j] = NULL;
        }

        for(i=tau_previous_pvar_count; i < num_pvars; i++) {
          pvarEvents[i] = (TauUserEvent**)calloc(tau_pvar_count[i], sizeof(TauUserEvent*));
          Tau_fill_mpi_t_pvar_events(&(pvarEvents[i]), i, tau_pvar_count[i]);
        }
    }

    tau_previous_pvar_count = num_pvars;
}

extern "C" char * Tau_get_pvar_name(const int current_pvar_index, const int current_pvar_subindex) {

  char * pvarnamechar = const_cast<char*>(PvarName(current_pvar_index, current_pvar_subindex).GetName().c_str());

  strcpy(pvarnamearray,pvarnamechar);
  //return (char *) (PvarName(current_pvar_index, current_pvar_subindex).GetName().c_str());
  return pvarnamearray;
}

extern "C" void Tau_track_pvar_event(const int current_pvar_index, const int current_pvar_subindex, const int *tau_pvar_count, const int num_pvars, double data) {
  ThePVarsMPIEvents(current_pvar_index, current_pvar_subindex, tau_pvar_count, num_pvars).TriggerEvent(data, Tau_get_thread());
#ifdef TAU_BEACON
  if(getenv("BEACON_TOPOLOGY_SERVER_ADDR") != NULL)
    TauBeaconPublish(data, "counts", "MPI_T_PVAR", (char *) (ThePVarsMPIEvents(current_pvar_index, current_pvar_subindex, tau_pvar_count, num_pvars).GetName().c_str()));
#endif /* TAU_BEACON */
}

/* Consider a situation where a user configures TAU with ONLY MPI/MPIT
 * options, but is linking in TAU statically to an application
 * that does NOT use MPI. Meaning only libtau-* gets linked in.
 * Such a situation would lead to a linking failure of undefined symbol: "Tau_track_mpi_t_here"
 * inside TauHandler.o that does get built into libtau-*
 * This is ugly code, but atleast it would help some users who are new to configuring and using TAU
 * */
extern "C" int __attribute__ ((weak)) Tau_track_mpi_t_here(void) {
 return 0;
}

#ifdef TAU_SCOREP
/* If SCOREP is defined, there is TauMpi wrapper that typically contains this routine */
extern "C" int Tau_track_mpi_t_here(void) {
 // do nothing when MPI_T is not enabled
}
#endif /* TAU_SCOREP */

#else /* TAU_MPI_T */
extern "C" int Tau_track_mpi_t_here(void) {
 // do nothing when MPI_T is not enabled
 return 0;
}
#endif /* TAU_MPI_T */


//////////////////////////////////////////////////////////////////////
extern "C" void Tau_enable_tracking_mpi_t(void) {
  TauEnv_set_track_mpi_t_pvars(1);
}

//////////////////////////////////////////////////////////////////////
extern "C" void Tau_disable_tracking_mpi_t(void) {
  TauEnv_set_track_mpi_t_pvars(0);
}

#ifndef TAU_MPI_T
// stub function when MPI_T is not defined.
extern "C" int Tau_msg_init(void) {
  return 0;
}
extern "C" long Tau_get_message_send_path(void) {
  return 0L;
}

// stub function when MPI_T is not defined.
extern "C" long Tau_get_message_recv_path(void) {
  return 0L;
}

extern "C" int Tau_msg_send_prolog(void){
  return 0;
}

extern "C" int Tau_msg_recv_prolog(void){
  return 0;
}
#endif /* TAU_MPI_T */

size_t& tauGetAPITraceDepth() {
#ifdef thread_local
    thread_local static size_t depth{0};
#else
    static size_t depth = 0;
#endif
    return depth;
}

extern "C" void Tau_traced_api_call_enter() {
    tauGetAPITraceDepth()++;
}

extern "C" void Tau_traced_api_call_exit() {
    tauGetAPITraceDepth()--;
}

extern "C" int Tau_time_traced_api_call() {
    if (tauGetAPITraceDepth() > 0) { return 0;}
    return 1;
}

/***************************************************************************
 * $RCSfile: TauCAPI.cpp,v $   $Author: sameer $
 * $Revision: 1.158 $   $Date: 2010/05/28 17:45:49 $
 * VERSION: $Id: TauCAPI.cpp,v 1.158 2010/05/28 17:45:49 sameer Exp $
 ***************************************************************************/


