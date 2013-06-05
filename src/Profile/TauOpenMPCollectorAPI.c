#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef TAU_IBM_OMPT
#include <lomp/omp.h>
#endif /* TAU_IBM_OMPT */

#include "omp_collector_api.h"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h> 
#include <string.h> 
#include <stdbool.h> 
#include "dlfcn.h" // for dynamic loading of symbols
#ifdef MERCURIUM_EXTRA
# define RTLD_DEFAULT   ((void *) 0)
#endif
#include "Profiler.h"
#ifdef TAU_USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif
#include "TauEnv.h"

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

static omp_lock_t writelock;

int Tau_collector_enabled = 0;

extern void Tau_disable_collector_api() {
  // if we didn't initialize the lock, we will crash...
  if (!TauEnv_get_collector_api_enabled()) return;
  omp_set_lock(&writelock);
  Tau_collector_enabled = 0;
  omp_unset_lock(&writelock);
}

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

static int (*Tau_collector_api)(void*) = NULL;

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
    int depth = 5;
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
    }
    return;
}

/*__inline*/ void Tau_omp_start_timer(const char * state, int tid, int use_context) {
    if (use_context == 0) {
      Tau_pure_start_task(state, tid);
	} else {
      char * regionIDstr = NULL;
      /* turns out the master thread wasn't updating it - so unlock and continue. */
      if (Tau_collector_flags[tid].timerContext == NULL) {
          regionIDstr = malloc(32);
      } else {
          regionIDstr = malloc(strlen(Tau_collector_flags[tid].timerContext) + 32);
      }
      sprintf(regionIDstr, "%s: %s", state, Tau_collector_flags[tid].timerContext);
      // it is safe to set the active timer context now.
      if (Tau_collector_flags[tid].activeTimerContext != NULL) {
        free(Tau_collector_flags[tid].activeTimerContext);
      }
      Tau_collector_flags[tid].activeTimerContext = malloc(strlen(Tau_collector_flags[tid].timerContext)+1);
      strcpy(Tau_collector_flags[tid].activeTimerContext, Tau_collector_flags[tid].timerContext);
      Tau_pure_start_task(regionIDstr, tid);
      free(regionIDstr);
    }
}

/*__inline*/ void Tau_omp_stop_timer(const char * state, int tid, int use_context) {
#if 0
    char * regionIDstr = NULL;
    if (Tau_collector_flags[tid].activeTimerContext == NULL) {
        regionIDstr = malloc(32);
    } else {
        regionIDstr = malloc(strlen(Tau_collector_flags[tid].activeTimerContext) + 32);
    }
    if (use_context == 0) {
        sprintf(regionIDstr, "%s", state);
    } else {
        sprintf(regionIDstr, "%s: %s", state, Tau_collector_flags[tid].activeTimerContext);
    }
    //TAU_VERBOSE("\t\t\t%d stopping: %s\n", tid, regionIDstr); fflush(stdout);
    omp_set_lock(&writelock);
    if (Tau_collector_enabled) {
      Tau_pure_stop_task(regionIDstr, tid);
    }
    omp_unset_lock(&writelock);
    free(regionIDstr);
#else
    omp_set_lock(&writelock);
    if (Tau_collector_enabled) {
      Tau_stop_current_timer_task(tid);
    }
    omp_unset_lock(&writelock);
#endif
}

void Tau_omp_event_handler(OMP_COLLECTORAPI_EVENT event) {
    // THIS is here in case the very last statement in the
    // program is a parallel region - the worker threads
    // may exit AFTER thread 0 has exited, which triggered
    // the worker threads to stop all timers and dump.
    if (!Tau_collector_enabled || 
        !Tau_RtsLayer_TheEnableInstrumentation()) return;

    /* Never process anything internal to TAU */
    if (Tau_global_get_insideTAU() > 0) {
        return;
    }

    Tau_global_incr_insideTAU();

    int tid = Tau_get_tid();
    //TAU_VERBOSE("** Thread: %d, EVENT:%s **\n", tid, OMP_EVENT_NAME[event-1]); fflush(stdout); fflush(stderr);

    switch(event) {
        case OMP_EVENT_FORK:
            Tau_get_current_region_context(tid);
            Tau_omp_start_timer("OpenMP_PARALLEL_REGION", tid, 1);
            Tau_collector_flags[tid].parallel++;
            break;
        case OMP_EVENT_JOIN:
            /*
               if (Tau_collector_flags[tid].idle == 1) {
               Tau_omp_stop_timer("IDLE", tid, 0);
               Tau_collector_flags[tid].idle = 0;
               }
               */
            if (Tau_collector_flags[tid].parallel>0) {
                Tau_omp_stop_timer("OpenMP_PARALLEL_REGION", tid, 1);
                Tau_collector_flags[tid].parallel--;
            }
            break;
        case OMP_EVENT_THR_BEGIN_IDLE:
            // sometimes IDLE can be called twice in a row
            if (Tau_collector_flags[tid].idle == 1 && 
                    Tau_collector_flags[tid].busy == 0) {
                break;
            }
            if (Tau_collector_flags[tid].busy == 1) {
                Tau_omp_stop_timer("OpenMP_PARALLEL_REGION", tid, 1);
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
            Tau_omp_start_timer("OpenMP_PARALLEL_REGION", tid, 1);
            Tau_collector_flags[tid].busy = 1;
            Tau_collector_flags[tid].idle = 0;
            break;
        case OMP_EVENT_THR_BEGIN_IBAR:
            Tau_omp_start_timer("OpenMP_IMPLICIT_BARRIER", tid, 1);
            break;
        case OMP_EVENT_THR_END_IBAR:
            Tau_omp_stop_timer("OpenMP_IMPLICIT_BARRIER", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_EBAR:
            Tau_omp_start_timer("OpenMP_EXPLICIT_BARRIER", tid, 1);
            break;
        case OMP_EVENT_THR_END_EBAR:
            Tau_omp_stop_timer("OpenMP_EXPLICIT_BARRIER", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_LKWT:
            Tau_omp_start_timer("OpenMP_LOCK_WAIT", tid, 1);
            break;
        case OMP_EVENT_THR_END_LKWT:
            Tau_omp_stop_timer("OpenMP_LOCK_WAIT", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_CTWT:
            Tau_omp_start_timer("OpenMP_CRITICAL_SECTION_WAIT", tid, 1);
            break;
        case OMP_EVENT_THR_END_CTWT:
            Tau_omp_stop_timer("OpenMP_CRITICAL_SECTION_WAIT", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_ODWT:
            // for some reason, the ordered region wait is entered twice for some threads.
            if (Tau_collector_flags[tid].ordered_region_wait == 0) {
                Tau_omp_start_timer("OpenMP_ORDERED_REGION_WAIT", tid, 1);
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
            Tau_omp_start_timer("OpenMP_MASTER_REGION", tid, 1);
            break;
        case OMP_EVENT_THR_END_MASTER:
            Tau_omp_stop_timer("OpenMP_MASTER_REGION", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_SINGLE:
            Tau_omp_start_timer("OpenMP_SINGLE_REGION", tid, 1);
            break;
        case OMP_EVENT_THR_END_SINGLE:
            Tau_omp_stop_timer("OpenMP_SINGLE_REGION", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_ORDERED:
            // for some reason, the ordered region is entered twice for some threads.
            if (Tau_collector_flags[tid].ordered_region == 0) {
                Tau_omp_start_timer("OpenMP_ORDERED_REGION", tid, 1);
                Tau_collector_flags[tid].ordered_region = 1;
            }
            break;
        case OMP_EVENT_THR_END_ORDERED:
            if (Tau_collector_flags[tid].ordered_region == 1) {
                Tau_omp_stop_timer("OpenMP_ORDERED_REGION", tid, 1);
            }
            Tau_collector_flags[tid].ordered_region = 0;
            break;
        case OMP_EVENT_THR_BEGIN_ATWT:
            Tau_omp_start_timer("OpenMP_ATOMIC_REGION_WAIT", tid, 1);
            break;
        case OMP_EVENT_THR_END_ATWT:
            Tau_omp_stop_timer("OpenMP_ATOMIC_REGION_WAIT", tid, 1);
            break;
        case OMP_EVENT_THR_BEGIN_CREATE_TASK:
            // Open64 doesn't actually create a task if there is just one thread.
            // In that case, there won't be an END_CREATE.
#if defined (TAU_OPEN64ORC)
            if (omp_get_num_threads() > 1) {
                Tau_omp_start_timer("OpenMP_CREATE_TASK", tid, 0);
            }
#else
            Tau_omp_start_timer("OpenMP_CREATE_TASK", tid, 0);
#endif
            break;
        case OMP_EVENT_THR_END_CREATE_TASK_IMM:
            Tau_omp_stop_timer("OpenMP_CREATE_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_END_CREATE_TASK_DEL:
            Tau_omp_stop_timer("OpenMP_CREATE_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_BEGIN_SCHD_TASK:
            Tau_omp_start_timer("OpenMP_SCHEDULE_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_END_SCHD_TASK:
            Tau_omp_stop_timer("OpenMP_SCHEDULE_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_BEGIN_SUSPEND_TASK:
            Tau_omp_start_timer("OpenMP_SUSPEND_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_END_SUSPEND_TASK:
            Tau_omp_stop_timer("OpenMP_SUSPEND_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_BEGIN_STEAL_TASK:
            Tau_omp_start_timer("OpenMP_STEAL_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_END_STEAL_TASK:
            Tau_omp_stop_timer("OpenMP_STEAL_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_FETCHED_TASK:
            break;
        case OMP_EVENT_THR_BEGIN_EXEC_TASK:
            Tau_omp_start_timer("OpenMP_EXECUTE_TASK", tid, 0);
            Tau_collector_flags[tid].task_exec += 1;
            break;
        case OMP_EVENT_THR_BEGIN_FINISH_TASK:
            // When we get a "finish task", there might be a task executing...
            // or there might not.
            if (Tau_collector_flags[tid].task_exec > 0) {
                Tau_omp_stop_timer("OpenMP_EXECUTE_TASK", tid, 0);
                Tau_collector_flags[tid].task_exec -= 1;
            }
            Tau_omp_start_timer("OpenMP_FINISH_TASK", tid, 0);
            break;
        case OMP_EVENT_THR_END_FINISH_TASK:
            Tau_omp_stop_timer("OpenMP_FINISH_TASK", tid, 0);
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

//int __attribute__ ((constructor)) Tau_initialize_collector_api(void);

static bool initializing = false;
static bool initialized = false;

#if TAU_DISABLE_SHARED
extern int __omp_collector_api(void *);
#endif

int Tau_initialize_collector_api(void) {
    //if (Tau_collector_api != NULL || initializing) return 0;
    if (initialized || initializing) return 0;
    if (!TauEnv_get_collector_api_enabled()) return;

    initializing = true;

    omp_init_lock(&writelock);

#if TAU_DISABLE_SHARED
    *(void **) (&Tau_collector_api) = __omp_collector_api;
#else

#if defined (TAU_BGP) || defined (TAU_BGQ) || defined (TAU_CRAYCNL)
    // these special systems don't support dynamic symbol loading.
    *(void **) (&Tau_collector_api) = NULL;

#else

    char *error;

#if defined (__INTEL_COMPILER)
    char * libname = "libiomp5.so";
#elif defined (__GNUC__) && defined (__GNUC_MINOR__) && defined (__GNUC_PATCHLEVEL__)

#ifdef __APPLE__
    char * libname = "libgomp_g_wrap.dylib";
#else /* __APPLE__ */
    char * libname = "libTAU-gomp.so";
#endif /* __APPLE__ */

#else /* assume we are using OpenUH */
    char * libname = "libopenmp.so";
#endif /* __GNUC__ __GNUC_MINOR__ __GNUC_PATCHLEVEL__ */

    TAU_VERBOSE("Looking for library: %s\n", libname); fflush(stdout); fflush(stderr);
    void * handle = dlopen(libname, RTLD_NOW | RTLD_GLOBAL);
#if 0
    char * err = dlerror();
    if (err) { 
        if (!handle) { 
            TAU_VERBOSE("Error loading library: %s\n", libname, err); fflush(stdout); fflush(stderr);
            /* don't quit, because it might have been preloaded... */
            //return -1;
        }
    }
#endif

    if (handle != NULL) {
        TAU_VERBOSE("Looking for symbol in library: %s\n", libname); fflush(stdout); fflush(stderr);
        *(void **) (&Tau_collector_api) = dlsym(handle, "__omp_collector_api");
    } else {
        *(void **) (&Tau_collector_api) = dlsym(RTLD_DEFAULT, "__omp_collector_api");
    }
	// set this now, either it's there or it isn't.
    initialized = true;
#if 0
    err = dlerror();
    if (err) { 
        TAU_VERBOSE("Error getting '__omp_collector_api' handle: %s\n", err); fflush(stdout); fflush(stderr);
        initializing = false;
        return -1;
    }
#endif
#endif //if defined (BGL) || defined (BGP) || defined (BGQ) || defined (TAU_CRAYCNL)
    if (Tau_collector_api == NULL) {
        TAU_VERBOSE("__omp_collector_api symbol not found... collector API not enabled. \n"); fflush(stdout); fflush(stderr);
        initializing = false;
        return -1;
    }
#endif // TAU_DISABLE_SHARED

    omp_collector_message req;
    void *message = (void *) malloc(4);   
    int *sz = (int *) message; 
    *sz = 0;
    int rc = 0;

    /*test: check for request start, 1 message */
    message = (void *) malloc(OMP_COLLECTORAPI_HEADERSIZE+sizeof(int));
    Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE, OMP_REQ_START, OMP_ERRCODE_OK, 0, 1);
    rc = (Tau_collector_api)(message);
    //TAU_VERBOSE("__omp_collector_api() returned %d\n", rc); fflush(stdout); fflush(stderr);
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
    //TAU_VERBOSE("__omp_collector_api() returned %d\n", rc); fflush(stdout); fflush(stderr);
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

    Tau_collector_enabled = 1;
    initializing = false;
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


/********************************************************
 * The functions below are for the OMPT 4.0 interface.
 * ******************************************************/

/* 
 * This header file implements a dummy tool which will execute all
 * of the implemented callbacks in the OMPT framework. When a supported
 * callback function is executed, it will print a message with some
 * relevant information.
 */

#ifndef TAU_IBM_OMPT
#include <ompt.h>
#endif /* TAU_IBM_OMPT */

void Tau_ompt_start_timer(const char * state, ompt_parallel_id_t regionid) {
    char * regionIDstr = NULL;
    regionIDstr = malloc(32);
	if (regionid > 0)
      sprintf(regionIDstr, "%s %llx", state, regionid);
	else
      sprintf(regionIDstr, "%s", state);
    Tau_pure_start_task(regionIDstr, Tau_get_tid());
    free(regionIDstr);
}

void Tau_ompt_stop_timer(const char * state, ompt_parallel_id_t regionid) {
    char * regionIDstr = NULL;
    regionIDstr = malloc(32);
    sprintf(regionIDstr, "%s %llx", state, regionid);
    Tau_pure_stop_task(regionIDstr, Tau_get_tid());
    free(regionIDstr);
}

/* These two macros make sure we don't time TAU related events */

#define TAU_OMPT_COMMON_ENTRY \
    /* Never process anything internal to TAU */ \
    if (Tau_global_get_insideTAU() > 0) { \
        return; \
    } \
    Tau_global_incr_insideTAU(); \
	int tid = Tau_get_tid(); \
    TAU_VERBOSE("%d %d: %s\n", tid, omp_get_thread_num(), __func__); \
	fflush(stdout);

#define TAU_OMPT_COMMON_EXIT \
    Tau_global_decr_insideTAU(); \

/*
 * Mandatory Events
 * 
 * The following events are supported by all OMPT implementations.
 */

/* Entering a parallel region */
void my_parallel_region_create (
  ompt_data_t  *parent_task_data,   /* tool data for parent task   */
  ompt_frame_t *parent_task_frame,  /* frame data of parent task   */
  ompt_parallel_id_t parallel_id)   /* id of parallel region       */
{
  TAU_OMPT_COMMON_ENTRY;
  Tau_get_current_region_context(tid);
  Tau_omp_start_timer("OpenMP_PARALLEL_REGION", tid, 1);
  //Tau_ompt_start_timer("PARALLEL_REGION", parallel_id);
  Tau_collector_flags[tid].parallel++;
  TAU_OMPT_COMMON_EXIT;
}

/* Exiting a parallel region */
void my_parallel_region_exit (
  ompt_data_t  *parent_task_data,   /* tool data for parent task   */
  ompt_frame_t *parent_task_frame,  /* frame data of parent task   */
  ompt_parallel_id_t parallel_id)   /* id of parallel region       */
{
  TAU_OMPT_COMMON_ENTRY;
  if (Tau_collector_flags[tid].parallel>0) {
    Tau_omp_stop_timer("OpenMP_PARALLEL_REGION", tid, 1);
    //Tau_ompt_stop_timer("PARALLEL_REGION", parallel_id);
    Tau_collector_flags[tid].parallel--;
  }
  TAU_OMPT_COMMON_EXIT;
}

/* Task creation */
void my_task_create (ompt_data_t *task_data) {
  TAU_OMPT_COMMON_ENTRY;
  //Tau_omp_start_timer("TASK", tid, 1);
  Tau_ompt_start_timer("OpenMP_TASK", 0);
  TAU_OMPT_COMMON_EXIT;
}

/* Task exit */
void my_task_exit (ompt_data_t *task_data) {
  TAU_OMPT_COMMON_ENTRY;
  //Tau_omp_stop_timer("TASK", tid, 1);
  Tau_ompt_stop_timer("OpenMP_TASK", 0);
  TAU_OMPT_COMMON_EXIT;
}

/* Thread creation */
void my_thread_create(ompt_data_t *thread_data) {
  TAU_OMPT_COMMON_ENTRY;
  //Tau_create_top_level_timer_if_necessary();
  TAU_OMPT_COMMON_EXIT;
}

/* Thread exit */
void my_thread_exit(ompt_data_t *thread_data) {
  if (!Tau_RtsLayer_TheEnableInstrumentation()) return;
  //TAU_VERBOSE("%s\n", __func__); fflush(stdout);
  TAU_OMPT_COMMON_ENTRY;
  //Tau_stop_top_level_timer_if_necessary();
  TAU_OMPT_COMMON_EXIT;
}

/* Some control event happened */
void my_control(uint64_t command, uint64_t modifier) {
  TAU_OMPT_COMMON_ENTRY;
  TAU_VERBOSE("OpenMP Control: %llx, %llx\n", command, modifier); fflush(stdout);
  // nothing to do here?
  TAU_OMPT_COMMON_EXIT;
}

/* Shutting down the OpenMP runtime */
void my_shutdown() {
  if (!Tau_RtsLayer_TheEnableInstrumentation()) return;
  TAU_OMPT_COMMON_ENTRY;
  TAU_VERBOSE("OpenMP Shutdown.\n"); fflush(stdout);
  Tau_profile_exit_all_tasks();
  TAU_PROFILE_EXIT("exiting");
  // nothing to do here?
  TAU_OMPT_COMMON_EXIT;
}

/**********************************************************************/
/* End Mandatory Events */
/**********************************************************************/

/**********************************************************************/
/* Macros for common wait, acquire, release functionality. */
/**********************************************************************/

#define TAU_OMPT_WAIT_ACQUIRE_RELEASE(WAIT_FUNC,ACQUIRED_FUNC,RELEASE_FUNC,WAIT_NAME,REGION_NAME) \
void WAIT_FUNC (ompt_wait_id_t *waitid) { \
  TAU_OMPT_COMMON_ENTRY; \
  Tau_ompt_start_timer(WAIT_NAME, 0); \
  TAU_OMPT_COMMON_EXIT; \
} \
 \
void ACQUIRED_FUNC (ompt_wait_id_t *waitid) { \
  TAU_OMPT_COMMON_ENTRY; \
  Tau_ompt_stop_timer(WAIT_NAME, 0); \
  Tau_ompt_start_timer(REGION_NAME, 0); \
  TAU_OMPT_COMMON_EXIT; \
} \
 \
void RELEASE_FUNC (ompt_wait_id_t *waitid) { \
  TAU_OMPT_COMMON_ENTRY; \
  Tau_ompt_stop_timer(REGION_NAME, 0); \
  TAU_OMPT_COMMON_EXIT; \
} \

TAU_OMPT_WAIT_ACQUIRE_RELEASE(my_wait_atomic,my_acquired_atomic,my_release_atomic,"OpenMP_ATOMIC_REGION_WAIT","OpenMP_ATOMIC_REGION")
TAU_OMPT_WAIT_ACQUIRE_RELEASE(my_wait_ordered,my_acquired_ordered,my_release_ordered,"OpenMP_ORDERED_REGION_WAIT","OpenMP_ORDERED_REGION")
TAU_OMPT_WAIT_ACQUIRE_RELEASE(my_wait_critical,my_acquired_critical,my_release_critical,"OpenMP_CRITICAL_REGION_WAIT","OpenMP_CRITICAL_REGION")
TAU_OMPT_WAIT_ACQUIRE_RELEASE(my_wait_lock,my_acquired_lock,my_release_lock,"OpenMP_LOCK_WAIT","OpenMP_LOCK")

#undef TAU_OMPT_WAIT_ACQUIRE_RELEASE

/**********************************************************************/
/* Macros for common begin / end functionality. */
/**********************************************************************/

#define TAU_OMPT_SIMPLE_BEGIN_AND_END(BEGIN_FUNCTION,END_FUNCTION,NAME) \
void BEGIN_FUNCTION (ompt_data_t  *parent_task_data, ompt_parallel_id_t parallel_id) { \
  TAU_OMPT_COMMON_ENTRY; \
  /*Tau_ompt_start_timer(NAME, parallel_id); */ \
  Tau_omp_start_timer(NAME, tid, 0); \
  TAU_OMPT_COMMON_EXIT; \
} \
\
void END_FUNCTION (ompt_data_t  *parent_task_data, ompt_parallel_id_t parallel_id) { \
  TAU_OMPT_COMMON_ENTRY; \
  /*Tau_ompt_stop_timer(NAME, parallel_id); */ \
  Tau_omp_stop_timer(NAME, tid, 0); \
  TAU_OMPT_COMMON_EXIT; \
}

TAU_OMPT_SIMPLE_BEGIN_AND_END(my_barrier_begin,my_barrier_end,"OpenMP_BARRIER")
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_wait_barrier_begin,my_wait_barrier_end,"OpenMP_WAIT_BARRIER")
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_master_begin,my_master_end,"OpenMP_MASTER_REGION")
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_loop_begin,my_loop_end,"OpenMP_LOOP")
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_section_begin,my_section_end,"OpenMP_SECTION") 
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_single_in_block_begin,my_single_in_block_end,"OpenMP_SINGLE_IN_BLOCK") 
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_single_others_begin,my_single_others_end,"OpenMP_SINGLE_OTHERS") 
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_taskwait_begin,my_taskwait_end,"OpenMP_TASKWAIT") 
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_wait_taskwait_begin,my_wait_taskwait_end,"OpenMP_WAIT_TASKWAIT") 
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_taskgroup_begin,my_taskgroup_end,"OpenMP_TASKGROUP") 
TAU_OMPT_SIMPLE_BEGIN_AND_END(my_wait_taskgroup_begin,my_wait_taskgroup_end,"OpenMP_WAIT_TASKGROUP") 

#undef TAU_OMPT_SIMPLE_BEGIN_AND_END

/**********************************************************************/
/* Specialized begin / end functionality. */
/**********************************************************************/

/* Thread end idle */
void my_idle_end(ompt_data_t *thread_data) {
  if (!Tau_RtsLayer_TheEnableInstrumentation()) return;
  TAU_OMPT_COMMON_ENTRY;
  Tau_omp_stop_timer("IDLE", tid, 0);
  // if this thread is not the master of a team, then assume this 
  // thread is entering a new parallel region
#if 0
  if (Tau_collector_flags[tid].parallel==0) {
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
  }
#endif
  TAU_OMPT_COMMON_EXIT;
}

/* Thread begin idle */
void my_idle_begin(ompt_data_t *thread_data) {
  TAU_OMPT_COMMON_ENTRY;
  // if this thread is not the master of a team, then assume this 
  // thread is exiting a parallel region
#if 0
  if (Tau_collector_flags[tid].parallel==0) {
    if (Tau_collector_flags[tid].idle == 1 && 
        Tau_collector_flags[tid].busy == 0) {
        return;
    }
    if (Tau_collector_flags[tid].busy == 1) {
        Tau_omp_stop_timer("PARALLEL_REGION", tid, 1);
        Tau_collector_flags[tid].busy = 0;
    }
    Tau_collector_flags[tid].idle = 1;
  }
#endif
  Tau_omp_start_timer("IDLE", tid, 0);
  TAU_OMPT_COMMON_EXIT;
}

#undef TAU_OMPT_COMMON_ENTRY
#undef TAU_OMPT_COMMON_EXIT

#ifdef TAU_IBM_OMPT
#define CHECK(EVENT,FUNCTION,NAME) ompt_set_callback(EVENT, FUNCTION)
#else 
#define CHECK(EVENT,FUNCTION,NAME) \
  if (ompt_set_callback(EVENT, FUNCTION) != 0) { \
    fprintf(stderr,"Failed to register OMPT callback %s!\n",NAME); \
	fflush(stderr); \
  }
#endif /* TAU_IBM_OMPT */

int ompt_initialize() {
  /* required events */
  CHECK(ompt_event_parallel_create, my_parallel_region_create, "parallel_create");
  CHECK(ompt_event_parallel_exit, my_parallel_region_exit, "parallel_exit");
  CHECK(ompt_event_task_create, my_task_create, "task_create");
  CHECK(ompt_event_task_exit, my_task_exit, "task_exit");
//#ifndef TAU_IBM_OMPT
  CHECK(ompt_event_thread_create, my_thread_create, "thread_create");
//#endif
  CHECK(ompt_event_thread_exit, my_thread_exit, "thread_exit");
  CHECK(ompt_event_control, my_control, "event_control");
#ifndef TAU_IBM_OMPT
  CHECK(ompt_event_runtime_shutdown, my_shutdown, "runtime_shutdown");
#endif /* TAU_IBM_OMPT */

  /* optional events, "blameshifting" */
#ifndef TAU_IBM_OMPT
  CHECK(ompt_event_idle_begin, my_idle_begin, "idle_begin");
  CHECK(ompt_event_idle_end, my_idle_end, "idle_end");
#endif
  //CHECK(ompt_event_wait_barrier_begin, my_wait_barrier_begin, "wait_barrier_begin");
  //CHECK(ompt_event_wait_barrier_end, my_wait_barrier_end, "wait_barrier_end");
  //CHECK(ompt_event_wait_taskwait_begin, my_wait_taskwait_begin, "wait_taskwait_begin");
  //CHECK(ompt_event_wait_taskwait_end, my_wait_taskwait_end, "wait_taskwait_end");
  //CHECK(ompt_event_wait_taskgroup_begin, my_wait_taskgroup_begin, "wait_taskgroup_begin");
  //CHECK(ompt_event_wait_taskgroup_end, my_wait_taskgroup_end, "wait_taskgroup_end");
  //CHECK(ompt_event_release_lock, my_release_lock, "release_lock");
//ompt_event(ompt_event_release_nest_lock_last, ompt_wait_callback_t, 18, ompt_event_release_nest_lock_implemented) /* last nest lock release */
  //CHECK(ompt_event_release_critical, my_release_critical, "release_critical");
  //CHECK(ompt_event_release_atomic, my_release_atomic, "release_atomic");
  //CHECK(ompt_event_release_ordered, my_release_ordered, "release_ordered");

  /* optional events, synchronous events */
  //CHECK(ompt_event_implicit_task_create, my_task_create, "task_create");
  //CHECK(ompt_event_implicit_task_exit, my_task_exit, "task_exit");
  CHECK(ompt_event_barrier_begin, my_barrier_begin, "barrier_begin");
  CHECK(ompt_event_barrier_end, my_barrier_end, "barrier_end");
  //CHECK(ompt_event_master_begin, my_master_begin, "master_begin");
  //CHECK(ompt_event_master_end, my_master_end, "master_end");
//ompt_event(ompt_event_task_switch, ompt_task_switch_callback_t, 24, ompt_event_task_switch_implemented) /* task switch */
  //CHECK(ompt_event_loop_begin, my_loop_begin, "loop_begin");
  //CHECK(ompt_event_loop_end, my_loop_end, "loop_end");
  //CHECK(ompt_event_section_begin, my_section_begin, "section_begin");
  //CHECK(ompt_event_section_end, my_section_end, "section_end");
  //CHECK(ompt_event_single_in_block_begin, my_single_in_block_begin, "single_in_block_begin");
  //CHECK(ompt_event_single_in_block_end, my_single_in_block_end, "single_in_block_end");
  //CHECK(ompt_event_single_others_begin, my_single_others_begin, "single_others_begin");
  //CHECK(ompt_event_single_others_end, my_single_others_end, "single_others_end");
  //CHECK(ompt_event_taskwait_begin, my_taskwait_begin, "taskwait_begin");
  //CHECK(ompt_event_taskwait_end, my_taskwait_end, "taskwait_end");
  //CHECK(ompt_event_taskgroup_begin, my_taskgroup_begin, "taskgroup_begin");
  //CHECK(ompt_event_taskgroup_end, my_taskgroup_end, "taskgroup_end");

//ompt_event(ompt_event_release_nest_lock_prev, ompt_parallel_callback_t, 41, ompt_event_release_nest_lock_prev_implemented) /* prev nest lock release */

  //CHECK(ompt_event_wait_lock, my_wait_lock, "wait_lock");
//ompt_event(ompt_event_wait_nest_lock, ompt_wait_callback_t, 43, ompt_event_wait_nest_lock_implemented) /* nest lock wait */
  //CHECK(ompt_event_wait_critical, my_wait_critical, "wait_critical");
  //CHECK(ompt_event_wait_atomic, my_wait_atomic, "wait_atomic");
  //CHECK(ompt_event_wait_ordered, my_wait_ordered, "wait_ordered");

  //CHECK(ompt_event_acquired_lock, my_acquired_lock, "acquired_lock");
//ompt_event(ompt_event_acquired_nest_lock_first, ompt_wait_callback_t, 48, ompt_event_acquired_nest_lock_first_implemented) /* 1st nest lock acquired */
//ompt_event(ompt_event_acquired_nest_lock_next, ompt_parallel_callback_t, 49, ompt_event_acquired_nest_lock_next_implemented) /* next nest lock acquired*/
  //CHECK(ompt_event_acquired_critical, my_acquired_critical, "acquired_critical");
  //CHECK(ompt_event_acquired_atomic, my_acquired_atomic, "acquired_atomic");
  //CHECK(ompt_event_acquired_ordered, my_acquired_ordered, "acquired_ordered");

//ompt_event(ompt_event_init_lock, ompt_wait_callback_t, 53, ompt_event_init_lock_implemented) /* lock init */
//ompt_event(ompt_event_init_nest_lock, ompt_wait_callback_t, 54, ompt_event_init_nest_lock_implemented) /* nest lock init */
//ompt_event(ompt_event_destroy_lock, ompt_wait_callback_t, 55, ompt_event_destroy_lock_implemented) /* lock destruction */
//ompt_event(ompt_event_destroy_nest_lock, ompt_wait_callback_t, 56, ompt_event_destroy_nest_lock_implemented) /* nest lock destruction */

//ompt_event(ompt_event_flush, ompt_thread_callback_t, 57, ompt_event_flush_implemented) /* after executing flush */

  return 1;
}

/* THESE ARE OTHER WEAK IMPLEMENTATIONS, IN CASE OMPT SUPPORT IS NONEXISTENT */

/* initialization */
#ifndef TAU_USE_OMPT
extern __attribute__ (( weak ))
  int ompt_set_callback(ompt_event_t evid, ompt_callback_t cb) { return -1; };
#endif
