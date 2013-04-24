#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "omp_collector_api.h"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h> 
#include <string.h> 
#include "dlfcn.h" // for dynamic loading of symbols
#include "Profiler.h"
#ifdef TAU_USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

/* An array of this struct is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
struct Tau_collector_status_flags {
  int idle; // 4 bytes
  int busy; // 4 bytes
  int parallel; // 4 bytes
  int ordered_region_wait; // 4 bytes
  int ordered_region; // 4 bytes
  int task_exec; // 4 bytes
  char *timerContext; // 8 bytes(?)
  char *activeTimerContext; // 8 bytes(?)
  void *signal_message; // preallocated message for signal handling, 8 bytes
  char _pad[64-((sizeof(void*))+(2*sizeof(char*))+(6*sizeof(int)))];
};

/* This array is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
#if defined __INTEL__COMPILER
__declspec (align(64)) static struct Tau_collector_status_flags Tau_collector_flags[TAU_MAX_THREADS] = {0};
#elif defined __GNUC__
static struct Tau_collector_status_flags Tau_collector_flags[TAU_MAX_THREADS] __attribute__ ((aligned(64))) = {0};
#else
static struct Tau_collector_status_flags Tau_collector_flags[TAU_MAX_THREADS] = {0};
#endif

extern void Tau_fill_header(void *message, int sz, OMP_COLLECTORAPI_REQUEST rq, OMP_COLLECTORAPI_EC ec, int rsz, int append_zero);
  
static char* __UNKNOWN__ = "UNKNOWN";

extern const int OMP_COLLECTORAPI_HEADERSIZE;
char OMP_EVENT_NAME[35][50]= {
  "OMP_EVENT_FORK",
  "OMP_EVENT_JOIN",
  "OMP_EVENT_THR_BEGIN_IDLE",
  "OMP_EVENT_THR_END_IDLE",
  "OMP_EVENT_THR_BEGIN_IBAR",
  "OMP_EVENT_THR_END_IBAR",
  "OMP_EVENT_THR_BEGIN_EBAR",
  "OMP_EVENT_THR_END_EBAR",
  "OMP_EVENT_THR_BEGIN_LKWT",
  "OMP_EVENT_THR_END_LKWT",
  "OMP_EVENT_THR_BEGIN_CTWT",
  "OMP_EVENT_THR_END_CTWT",
  "OMP_EVENT_THR_BEGIN_ODWT",
  "OMP_EVENT_THR_END_ODWT",
  "OMP_EVENT_THR_BEGIN_MASTER",
  "OMP_EVENT_THR_END_MASTER",
  "OMP_EVENT_THR_BEGIN_SINGLE",
  "OMP_EVENT_THR_END_SINGLE",
  "OMP_EVENT_THR_BEGIN_ORDERED",
  "OMP_EVENT_THR_END_ORDERED",
  "OMP_EVENT_THR_BEGIN_ATWT",
  "OMP_EVENT_THR_END_ATWT",
/* new events created by UH */
  "OMP_EVENT_THR_BEGIN_CREATE_TASK",
  "OMP_EVENT_THR_END_CREATE_TASK_IMM",
  "OMP_EVENT_THR_END_CREATE_TASK_DEL",
  "OMP_EVENT_THR_BEGIN_SCHD_TASK",
  "OMP_EVENT_THR_END_SCHD_TASK",
  "OMP_EVENT_THR_BEGIN_SUSPEND_TASK",
  "OMP_EVENT_THR_END_SUSPEND_TASK",
  "OMP_EVENT_THR_BEGIN_STEAL_TASK",
  "OMP_EVENT_THR_END_STEAL_TASK",
  "OMP_EVENT_THR_FETCHED_TASK",
  "OMP_EVENT_THR_BEGIN_EXEC_TASK",
  "OMP_EVENT_THR_BEGIN_FINISH_TASK",
  "OMP_EVENT_THR_END_FINISH_TASK"
 };

const int OMP_COLLECTORAPI_HEADERSIZE=4*sizeof(int);

static int (*Tau_collector_api)(void*);

extern char * TauInternal_CurrentCallsiteTimerName(int tid);

void Tau_get_region_id(int tid) {
/* get the region ID */
  omp_collector_message req;
  int currentid_rsz = sizeof(long);
  void * message = (void *) calloc(OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz+sizeof(int), sizeof(char));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz, OMP_REQ_CURRENT_PRID, OMP_ERRCODE_OK, currentid_rsz, 1);
  long * rid = message + OMP_COLLECTORAPI_HEADERSIZE;
  int rc = (Tau_collector_api)(message);
  TAU_VERBOSE("Thread %d, region ID : %ld\n", tid, *rid);
  free(message);
  return;
}

#ifdef TAU_UNWIND
typedef struct {
  unsigned long pc;
  int moduleIdx;
  char *name;
} Tau_collector_api_CallSiteInfo;

extern struct CallSiteInfo * Tau_sampling_resolveCallSite(unsigned long address,
  const char *tag, const char *childName, char **newShortName, char addAddress);

char * show_backtrace (int tid) {
  char * location = NULL;
  unw_cursor_t cursor; unw_context_t uc;
  unw_word_t ip, sp;

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  int index = 0;
#if defined (TAU_OPEN64ORC)
  int depth = 1;
#else /* assume we are using gcc */
//#if defined (__GNUC__) && defined (__GNUC_MINOR__) && defined (__GNUC_PATCHLEVEL__)
  int depth = 4;
#endif /* (__GNUC__) && defined (__GNUC_MINOR__) && defined (__GNUC_PATCHLEVEL__) */
  while (unw_step(&cursor) > 0) {
    // we want to pop 3 or 4 levels of the stack:
    // - Tau_get_current_region_context()
    // - Tau_omp_event_handler()
    // - __ompc_event_callback() or fork()
	// - GOMP_parallel_begin() * maybe - only if using GOMP *
    // - ?? <- the source location we want
    if (index++ == depth) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      unw_get_reg(&cursor, UNW_REG_SP, &sp);
      char * newShort;
      void * tmpInfo = (void*)Tau_sampling_resolveCallSite(ip, "OPENMP", NULL, &newShort, 0);
      //void * tmpInfo = (void*)Tau_sampling_resolveCallSite(ip, "UNWIND", NULL, &newShort, 0);
      Tau_collector_api_CallSiteInfo * myInfo = (Tau_collector_api_CallSiteInfo*)(tmpInfo);
      //TAU_VERBOSE ("ip = %lx, sp = %lx, name= %s\n", (long) ip, (long) sp, myInfo->name);
      location = malloc(strlen(myInfo->name)+1);
      strcpy(location, myInfo->name);
      break;
    }
  }
  return location;
}
#endif

void Tau_get_current_region_context(int tid) {
  // Tau_get_region_id (tid);
  char * tmpStr = NULL;
#if defined(TAU_UNWIND) && defined(TAU_BFD) // need them both
  tmpStr = show_backtrace(tid); // find our source location
  if (tmpStr == NULL) {
    tmpStr = "UNKNOWN";
  }
#else
  tmpStr = TauInternal_CurrentCallsiteTimerName(tid); // find our top level timer
#endif
  if (tmpStr == NULL)
    tmpStr = "";
  if (Tau_collector_flags[tid].timerContext != NULL) {
    free(Tau_collector_flags[tid].timerContext);
  }
  Tau_collector_flags[tid].timerContext = malloc(strlen(tmpStr)+1);
  strcpy(Tau_collector_flags[tid].timerContext, tmpStr);
  //TAU_VERBOSE("Got timer: %s\n", Tau_collector_flags[tid].timerContext);
  //TAU_VERBOSE("Forking with %d threads\n", omp_get_max_threads());
  int i;
  for (i = 0 ; i < omp_get_max_threads() ; i++) {
    if (i == tid) continue; // don't mess with yourself
    if (Tau_collector_flags[i].timerContext != NULL) {
      free(Tau_collector_flags[i].timerContext);
    }
    Tau_collector_flags[i].timerContext = malloc(strlen(tmpStr)+1);
    strcpy(Tau_collector_flags[i].timerContext, tmpStr);
    // set the active timer context, too
    /*
    if (Tau_collector_flags[i].idle == 1) {
      if (Tau_collector_flags[i].activeTimerContext != NULL) {
        free(Tau_collector_flags[i].activeTimerContext);
      }
      Tau_collector_flags[i].activeTimerContext = malloc(strlen(Tau_collector_flags[tid].timerContext)+1);
      strcpy(Tau_collector_flags[i].activeTimerContext, Tau_collector_flags[tid].timerContext);
    }
    TAU_VERBOSE("Thread %d Got timer: %s\n", i, Tau_collector_flags[i].timerContext);
    TAU_VERBOSE("Thread %d Got active timer: %s\n", i, Tau_collector_flags[i].activeTimerContext);
    */
  }
  return;
}

/*__inline*/ void Tau_omp_start_timer(const char * state, int tid, int use_context) {
  char * regionIDstr = NULL;
  /* turns out the master thread wasn't updating it - so unlock and continue. */
  if (Tau_collector_flags[tid].timerContext == NULL) {
    regionIDstr = malloc(32);
  } else {
    regionIDstr = malloc(strlen(Tau_collector_flags[tid].timerContext) + 32);
  }
  if (use_context == 0) {
    //sprintf(regionIDstr, "OpenMP %s", state);
    sprintf(regionIDstr, "OpenMP_%s", state);
  } else {
    //sprintf(regionIDstr, "%s : OpenMP %s", Tau_collector_flags[tid].timerContext, state);
    sprintf(regionIDstr, "OpenMP_%s: %s", state, Tau_collector_flags[tid].timerContext);
    // it is safe to set the active timer context now.
    if (Tau_collector_flags[tid].activeTimerContext != NULL) {
      free(Tau_collector_flags[tid].activeTimerContext);
    }
	//TAU_VERBOSE("region ID: '%s'\n", regionIDstr);
    Tau_collector_flags[tid].activeTimerContext = malloc(strlen(Tau_collector_flags[tid].timerContext)+1);
    strcpy(Tau_collector_flags[tid].activeTimerContext, Tau_collector_flags[tid].timerContext);
  }
  //TAU_VERBOSE("%d starting: %s\n", tid, regionIDstr); fflush(stdout);
  Tau_pure_start_task(regionIDstr, tid);
  free(regionIDstr);
}

/*__inline*/ void Tau_omp_stop_timer(const char * state, int tid, int use_context) {
  char * regionIDstr = NULL;
  if (Tau_collector_flags[tid].activeTimerContext == NULL) {
    regionIDstr = malloc(32);
  } else {
    regionIDstr = malloc(strlen(Tau_collector_flags[tid].activeTimerContext) + 32);
  }
  if (use_context == 0) {
    //sprintf(regionIDstr, "OpenMP %s", state);
    sprintf(regionIDstr, "OpenMP_%s", state);
  } else {
    //sprintf(regionIDstr, "%s : OpenMP %s", Tau_collector_flags[tid].activeTimerContext, state);
    sprintf(regionIDstr, "OpenMP_%s: %s", state, Tau_collector_flags[tid].activeTimerContext);
  }
  //TAU_VERBOSE("%d stopping: %s\n", tid, regionIDstr); fflush(stdout);
  Tau_pure_stop_task(regionIDstr, tid);
  free(regionIDstr);
}

void Tau_omp_event_handler(OMP_COLLECTORAPI_EVENT event) {
  /* Never process anything internal to TAU */
  if (Tau_global_get_insideTAU() > 0) {
    return;
  }

  Tau_global_incr_insideTAU();

  int tid = Tau_get_tid();
  //TAU_VERBOSE("** Thread: %d, EVENT:%s **\n", tid, OMP_EVENT_NAME[event-1]); fflush(stdout);

  switch(event) {
    case OMP_EVENT_FORK:
      Tau_get_current_region_context(tid);
      Tau_omp_start_timer("PARALLEL_REGION", tid, 1);
      Tau_collector_flags[tid].parallel++;
      break;
    case OMP_EVENT_JOIN:
/*
      if (Tau_collector_flags[tid].idle == 1) {
        Tau_omp_stop_timer("IDLE", tid, 0);
        Tau_collector_flags[tid].idle = 0;
      }
*/
      Tau_omp_stop_timer("PARALLEL_REGION", tid, 1);
      Tau_collector_flags[tid].parallel--;
      break;
    case OMP_EVENT_THR_BEGIN_IDLE:
      // sometimes IDLE can be called twice in a row
      if (Tau_collector_flags[tid].idle == 1 && 
          Tau_collector_flags[tid].busy == 0) {
        break;
      }
      if (Tau_collector_flags[tid].busy == 1) {
        Tau_omp_stop_timer("PARALLEL_REGION", tid, 1);
        Tau_collector_flags[tid].busy = 0;
      }
/*
      Tau_omp_start_timer("IDLE", tid, 0);
      Tau_collector_flags[tid].idle = 1;
*/
      Tau_collector_flags[tid].idle = 1;
      break;
    case OMP_EVENT_THR_END_IDLE:
/*
      if (Tau_collector_flags[tid].idle == 1) {
        Tau_omp_stop_timer("IDLE", tid, 0);
        Tau_collector_flags[tid].idle = 0;
      }
*/
      // it is safe to set the active timer context now.
      if (Tau_collector_flags[tid].activeTimerContext != NULL) {
        free(Tau_collector_flags[tid].activeTimerContext);
      }
      if (Tau_collector_flags[tid].timerContext == NULL) {
        Tau_collector_flags[tid].timerContext = malloc(strlen(__UNKNOWN__)+1);
        strcpy(Tau_collector_flags[tid].timerContext, __UNKNOWN__);
      }
      Tau_collector_flags[tid].activeTimerContext = malloc(strlen(Tau_collector_flags[tid].timerContext)+1);
      strcpy(Tau_collector_flags[tid].activeTimerContext, Tau_collector_flags[tid].timerContext);
      Tau_omp_start_timer("PARALLEL_REGION", tid, 1);
      Tau_collector_flags[tid].busy = 1;
      Tau_collector_flags[tid].idle = 0;
      break;
    case OMP_EVENT_THR_BEGIN_IBAR:
      Tau_omp_start_timer("IMPLICIT_BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_END_IBAR:
      Tau_omp_stop_timer("IMPLICIT_BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_EBAR:
      Tau_omp_start_timer("EXPLICIT_BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_END_EBAR:
      Tau_omp_stop_timer("EXPLICIT_BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_LKWT:
      Tau_omp_start_timer("LOCK_WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_END_LKWT:
      Tau_omp_stop_timer("LOCK_WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_CTWT:
      Tau_omp_start_timer("CRITICAL_SECTION_WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_END_CTWT:
      Tau_omp_stop_timer("CRITICAL_SECTION_WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_ODWT:
      // for some reason, the ordered region wait is entered twice for some threads.
      if (Tau_collector_flags[tid].ordered_region_wait == 0) {
        Tau_omp_start_timer("ORDERED_REGION_WAIT", tid, 1);
      }
      Tau_collector_flags[tid].ordered_region_wait = 1;
      break;
    case OMP_EVENT_THR_END_ODWT:
      if (Tau_collector_flags[tid].ordered_region_wait == 1) {
        Tau_omp_stop_timer("ORDERED_REGION_WAIT", tid, 1);
      }
      Tau_collector_flags[tid].ordered_region_wait = 0;
      break;
    case OMP_EVENT_THR_BEGIN_MASTER:
      Tau_omp_start_timer("MASTER_REGION", tid, 1);
      break;
    case OMP_EVENT_THR_END_MASTER:
      Tau_omp_stop_timer("MASTER_REGION", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_SINGLE:
      Tau_omp_start_timer("SINGLE_REGION", tid, 1);
      break;
    case OMP_EVENT_THR_END_SINGLE:
      Tau_omp_stop_timer("SINGLE_REGION", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_ORDERED:
      // for some reason, the ordered region is entered twice for some threads.
      if (Tau_collector_flags[tid].ordered_region == 0) {
        Tau_omp_start_timer("ORDERED_REGION", tid, 1);
        Tau_collector_flags[tid].ordered_region = 1;
      }
      break;
    case OMP_EVENT_THR_END_ORDERED:
      if (Tau_collector_flags[tid].ordered_region == 1) {
        Tau_omp_stop_timer("ORDERED_REGION", tid, 1);
      }
      Tau_collector_flags[tid].ordered_region = 0;
      break;
    case OMP_EVENT_THR_BEGIN_ATWT:
      Tau_omp_start_timer("ATOMIC_REGION_WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_END_ATWT:
      Tau_omp_stop_timer("ATOMIC_REGION_WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_CREATE_TASK:
      // Open64 doesn't actually create a task if there is just one thread.
      // In that case, there won't be an END_CREATE.
#if defined (TAU_OPEN64ORC)
      if (omp_get_num_threads() > 1) {
        Tau_omp_start_timer("CREATE_TASK", tid, 1);
      }
#else
      Tau_omp_start_timer("CREATE_TASK", tid, 1);
#endif
      break;
    case OMP_EVENT_THR_END_CREATE_TASK_IMM:
      Tau_omp_stop_timer("CREATE_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_END_CREATE_TASK_DEL:
      Tau_omp_stop_timer("CREATE_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_SCHD_TASK:
      Tau_omp_start_timer("SCHEDULE_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_END_SCHD_TASK:
      Tau_omp_stop_timer("SCHEDULE_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_SUSPEND_TASK:
      Tau_omp_start_timer("SUSPEND_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_END_SUSPEND_TASK:
      Tau_omp_stop_timer("SUSPEND_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_STEAL_TASK:
      Tau_omp_start_timer("STEAL_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_END_STEAL_TASK:
      Tau_omp_stop_timer("STEAL_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_FETCHED_TASK:
      break;
    case OMP_EVENT_THR_BEGIN_EXEC_TASK:
      Tau_omp_start_timer("EXECUTE_TASK", tid, 1);
      Tau_collector_flags[tid].task_exec += 1;
      break;
    case OMP_EVENT_THR_BEGIN_FINISH_TASK:
      // When we get a "finish task", there might be a task executing...
      // or there might not.
      if (Tau_collector_flags[tid].task_exec > 0) {
        Tau_omp_stop_timer("EXECUTE_TASK", tid, 1);
        Tau_collector_flags[tid].task_exec -= 1;
      }
      Tau_omp_start_timer("FINISH_TASK", tid, 1);
      break;
    case OMP_EVENT_THR_END_FINISH_TASK:
      Tau_omp_stop_timer("FINISH_TASK", tid, 1);
      break;
  }
  //TAU_VERBOSE("** Thread: %d, EVENT:%s handled. **\n", tid, OMP_EVENT_NAME[event-1]);
  //fflush(stdout);
  Tau_global_decr_insideTAU();
  return;
}

void Tau_fill_header(void *message, int sz, OMP_COLLECTORAPI_REQUEST rq, OMP_COLLECTORAPI_EC ec, int rsz, int append_zero)
{
  int *psz = (int *) message; 
  *psz = sz;
   
  OMP_COLLECTORAPI_REQUEST *rnum = (OMP_COLLECTORAPI_REQUEST *) (message+sizeof(int));
  *rnum = rq;
   
  OMP_COLLECTORAPI_EC *pec = (OMP_COLLECTORAPI_EC *)(message+(sizeof(int)*2));
  *pec = ec;

  int *prsz = (int *) (message+ sizeof(int)*3);
  *prsz = rsz;

  if(append_zero) {
    psz = (int *)(message+(sizeof(int)*4)+rsz);
    *psz =0; 
  }   
}

void Tau_fill_register(void *message, OMP_COLLECTORAPI_EVENT event, int append_func, void (*func)(OMP_COLLECTORAPI_EVENT), int append_zero) {

  // get a pointer to the head of the message
  OMP_COLLECTORAPI_EVENT *pevent = (OMP_COLLECTORAPI_EVENT *) message;
  // assign the event to the first parameter
  *pevent = event;

  // increment to the next parameter
  char *mem = (char *)(message + sizeof(OMP_COLLECTORAPI_EVENT));
  if(append_func) {
         unsigned long * lmem = (message + sizeof(OMP_COLLECTORAPI_EVENT));
         *lmem = (unsigned long)func;
  }

     if(append_zero) {
       int *psz;
       if(append_func) {
            psz = (int *)(message+sizeof(OMP_COLLECTORAPI_EVENT)+ sizeof(void *)); 
   
     } else {

          psz = (int *)(message+sizeof(OMP_COLLECTORAPI_EVENT));

     }
       *psz =0;  
     } 
}

int __attribute__ ((constructor)) Tau_initialize_collector_api(void);

int Tau_initialize_collector_api(void) {
  if (Tau_collector_api != NULL) return 0;

  char *error;
#if defined (__GNUC__) && defined (__GNUC_MINOR__) && defined (__GNUC_PATCHLEVEL__)

#ifdef __APPLE__
  char * libname = "libgomp_g_wrap.dylib";
#else /* __APPLE__ */
  char * libname = "libgomp_g_wrap.so";
#endif /* __APPLE__ */

#else /* assume we are using OpenUH */
  char * libname = "libopenmp.so";
#endif /* __GNUC__ __GNUC_MINOR__ __GNUC_PATCHLEVEL__ */

  TAU_VERBOSE("Looking for library: %s\n", libname);
  void * handle = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
#if 0
  char * err = dlerror();
  if (err) { 
  if (!handle) { 
	TAU_VERBOSE("Error loading library: %s\n", libname, err);
	/* don't quit, because it might have been preloaded... */
	//return -1;
  }
#endif

  *(void **) (&Tau_collector_api) = dlsym(RTLD_DEFAULT, "__omp_collector_api");
#if 0
  err = dlerror();
  if (err) { 
	TAU_VERBOSE("Error getting '__omp_collector_api' handle: %s\n", err);
	return -1;
  }
#endif
  if (Tau_collector_api == NULL) {
    TAU_VERBOSE("__omp_collector_api symbol not found... collector API not enabled. \n");
    return -1;
  }

  omp_collector_message req;
  void *message = (void *) malloc(4);   
  int *sz = (int *) message; 
  *sz = 0;
  int rc = 0;

  /*test: check for request start, 1 message */
  message = (void *) malloc(OMP_COLLECTORAPI_HEADERSIZE+sizeof(int));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE, OMP_REQ_START, OMP_ERRCODE_OK, 0, 1);
  rc = (Tau_collector_api)(message);
  TAU_VERBOSE("__omp_collector_api() returned %d\n", rc);
  free(message);

  /*test for request of all events*/
  int i;
  int num_req=OMP_EVENT_THR_END_FINISH_TASK; /* last event */
  int register_sz = sizeof(OMP_COLLECTORAPI_EVENT)+sizeof(void *);
  int mes_size = OMP_COLLECTORAPI_HEADERSIZE+register_sz;
  message = (void *) malloc(num_req*mes_size+sizeof(int));
  for(i=0;i<num_req;i++) {  
    Tau_fill_header(message+mes_size*i,mes_size, OMP_REQ_REGISTER, OMP_ERRCODE_OK, 0, 0);
    Tau_fill_register((message+mes_size*i)+OMP_COLLECTORAPI_HEADERSIZE,OMP_EVENT_FORK+i,1, Tau_omp_event_handler, i==(num_req-1));
  } 
  rc = (Tau_collector_api)(message);
  TAU_VERBOSE("__omp_collector_api() returned %d\n", rc);
  free(message);

  // preallocate messages, because we can't malloc when signals are
  // handled
  int state_rsz = sizeof(OMP_COLLECTOR_API_THR_STATE)+sizeof(unsigned long);
  for(i=0;i<omp_get_max_threads();i++) {  
    Tau_collector_flags[i].signal_message = malloc(OMP_COLLECTORAPI_HEADERSIZE+state_rsz);
    Tau_fill_header(Tau_collector_flags[i].signal_message, OMP_COLLECTORAPI_HEADERSIZE+state_rsz, OMP_REQ_STATE, OMP_ERRCODE_OK, state_rsz, 1);
  }

#ifdef TAU_UNWIND
  //Tau_Sampling_register_unit(); // not necessary now?
#endif

#if 0
  // now, for the collector API support, create the 12 OpenMP states.
  // preallocate State timers. If we create them now, we won't run into
  // malloc issues later when they are required during signal handling.
  Tau_create_thread_state_if_necessary("OMP UNKNOWN");
  Tau_create_thread_state_if_necessary("OMP OVERHEAD");
  Tau_create_thread_state_if_necessary("OMP WORKING");
  Tau_create_thread_state_if_necessary("OMP IMPLICIT BARRIER"); 
  Tau_create_thread_state_if_necessary("OMP EXPLICIT BARRIER");
  Tau_create_thread_state_if_necessary("OMP IDLE");
  Tau_create_thread_state_if_necessary("OMP SERIAL");
  Tau_create_thread_state_if_necessary("OMP REDUCTION");
  Tau_create_thread_state_if_necessary("OMP LOCK WAIT");
  Tau_create_thread_state_if_necessary("OMP CRITICAL WAIT");
  Tau_create_thread_state_if_necessary("OMP ORDERED WAIT");
  Tau_create_thread_state_if_necessary("OMP ATOMIC WAIT");
#endif

  return 0;
}

int __attribute__ ((destructor)) Tau_finalize_collector_api(void);

int Tau_finalize_collector_api(void) {
  return 0;
#if 0
  TAU_VERBOSE("Tau_finalize_collector_api()\n");

  omp_collector_message req;
  void *message = (void *) malloc(4);   
  int *sz = (int *) message; 
  *sz = 0;
  int rc = 0;

  /*test check for request stop, 1 message */
  message = (void *) malloc(OMP_COLLECTORAPI_HEADERSIZE+sizeof(int));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE, OMP_REQ_STOP, OMP_ERRCODE_OK, 0, 1);
  rc = (Tau_collector_api)(message);
  TAU_VERBOSE("__omp_collector_api() returned %d\n", rc);
  free(message);
#endif
}

int Tau_get_thread_omp_state(int tid) {
  // if not available, return something useful
  if (Tau_collector_api == NULL) return -1;
  //TAU_VERBOSE("Thread %d, getting state...\n", tid);

  OMP_COLLECTOR_API_THR_STATE thread_state = THR_LAST_STATE;
  // query the thread state
  (Tau_collector_api)(Tau_collector_flags[tid].signal_message);
  int * rid = Tau_collector_flags[tid].signal_message + OMP_COLLECTORAPI_HEADERSIZE;
  thread_state = *rid;
  TAU_VERBOSE("Thread %d, state : %d\n", tid, thread_state);
  // return the thread state as a string
  return (int)(thread_state);
}


