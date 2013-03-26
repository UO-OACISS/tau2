#include "omp_collector_api.h"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h> 
#include <string.h> 
#include <dlfcn.h> // for dynamic loading of symbols
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
  char *timerContext; // 8 bytes(?)
  char *activeTimerContext; // 8 bytes(?)
  void *signal_message; // preallocated message for signal handling, 8 bytes
  char _pad[64-((sizeof(void*))+(2*sizeof(char*))+(5*sizeof(int)))];
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
  
extern const int OMP_COLLECTORAPI_HEADERSIZE;
char OMP_EVENT_NAME[22][50]= {
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
  "OMP_EVENT_THR_END_ATWT" };

const int OMP_COLLECTORAPI_HEADERSIZE=4*sizeof(int);

static int (*Tau_collector_api)(OMP_COLLECTORAPI_EVENT);

extern char * TauInternal_CurrentCallsiteTimerName(int tid);

void Tau_get_region_id(int tid) {
/* get the region ID */
  omp_collector_message req;
  int currentid_rsz = sizeof(long);
  void * message = (void *) calloc(OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz+sizeof(int), sizeof(char));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz, OMP_REQ_CURRENT_PRID, OMP_ERRCODE_OK, currentid_rsz, 1);
  long * rid = message + OMP_COLLECTORAPI_HEADERSIZE;
  int rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
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

char * show_backtrace (void) {
  char * location = NULL;
  unw_cursor_t cursor; unw_context_t uc;
  unw_word_t ip, sp;

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  int index = 0;
  while (unw_step(&cursor) > 0) {
    // we want to pop 3 levels of the stack:
    // - Tau_get_current_region_context()
    // - Tau_omp_event_handler()
    // - fork()
    // - ?? <- the source location we want
    if (index++ == 3) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      unw_get_reg(&cursor, UNW_REG_SP, &sp);
      char * newShort;
      void * tmpInfo = (void*)Tau_sampling_resolveCallSite(ip, "OPENMP", NULL, &newShort, 0);
      //void * tmpInfo = (void*)Tau_sampling_resolveCallSite(ip, "UNWIND", NULL, &newShort, 0);
      Tau_collector_api_CallSiteInfo * myInfo = (Tau_collector_api_CallSiteInfo*)(tmpInfo);
      location = malloc(strlen(myInfo->name)+1);
      strcpy(location, myInfo->name);
      //printf ("ip = %lx, sp = %lx, name= %s\n", (long) ip, (long) sp, myInfo->name);
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
  tmpStr = show_backtrace(); // find our source location
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
  //printf("Got timer: %s\n", Tau_collector_flags[tid].timerContext);
  //printf("Forking with %d threads\n", omp_get_max_threads());
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
    printf("Thread %d Got timer: %s\n", i, Tau_collector_flags[i].timerContext);
    printf("Thread %d Got active timer: %s\n", i, Tau_collector_flags[i].activeTimerContext);
    */
  }
  return;
}

/*__inline*/ void Tau_omp_start_timer(const char * state, int tid, int use_context) {
  char * regionIDstr = NULL;
  if (Tau_collector_flags[tid].timerContext == NULL) {
    regionIDstr = malloc(32);
  } else {
    regionIDstr = malloc(strlen(Tau_collector_flags[tid].timerContext) + 32);
  }
  if (use_context == 0) {
    //sprintf(regionIDstr, "OpenMP %s", state);
    sprintf(regionIDstr, "[OPENMP] : %s", state);
  } else {
    //sprintf(regionIDstr, "%s : OpenMP %s", Tau_collector_flags[tid].timerContext, state);
    sprintf(regionIDstr, "%s : %s", Tau_collector_flags[tid].timerContext, state);
    // it is safe to set the active timer context now.
    if (Tau_collector_flags[tid].activeTimerContext != NULL) {
      free(Tau_collector_flags[tid].activeTimerContext);
    }
    Tau_collector_flags[tid].activeTimerContext = malloc(strlen(Tau_collector_flags[tid].timerContext)+1);
    strcpy(Tau_collector_flags[tid].activeTimerContext, Tau_collector_flags[tid].timerContext);
  }
  //printf("Thread %d : Starting : %s\n", tid, regionIDstr);
  TAU_STATIC_TIMER_START(regionIDstr);
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
    sprintf(regionIDstr, "[OPENMP] : %s", state);
  } else {
    //sprintf(regionIDstr, "%s : OpenMP %s", Tau_collector_flags[tid].activeTimerContext, state);
    sprintf(regionIDstr, "%s : %s", Tau_collector_flags[tid].activeTimerContext, state);
  }
  //printf("Thread %d : Stopping : %s\n", tid, regionIDstr);
#if 0 
  TAU_STATIC_TIMER_STOP(regionIDstr);
#else // this will prevent overlapping timers.
  TAU_GLOBAL_TIMER_STOP();
#endif
  free(regionIDstr);
}

void Tau_omp_event_handler(OMP_COLLECTORAPI_EVENT event) {
  /* Never process anything internal to TAU */
  if (Tau_global_get_insideTAU() > 0) {
    return;
  }

  Tau_global_incr_insideTAU();

  int tid = omp_get_thread_num();
  TAU_VERBOSE("** Thread: %d, EVENT:%s **\n", tid, OMP_EVENT_NAME[event-1]);
  fflush(stdout);

  switch(event) {
    case OMP_EVENT_FORK:
      Tau_get_current_region_context(tid);
      Tau_omp_start_timer("PARALLEL REGION", tid, 1);
      Tau_collector_flags[tid].parallel++;
      break;
    case OMP_EVENT_JOIN:
/*
      if (Tau_collector_flags[tid].idle == 1) {
        Tau_omp_stop_timer("IDLE", tid, 0);
        Tau_collector_flags[tid].idle = 0;
      }
*/
      Tau_omp_stop_timer("PARALLEL REGION", tid, 1);
      Tau_collector_flags[tid].parallel--;
      break;
    case OMP_EVENT_THR_BEGIN_IDLE:
      // sometimes IDLE can be called twice in a row
      if (Tau_collector_flags[tid].idle == 1) {
        break;
      }
      if (Tau_collector_flags[tid].busy == 1) {
        Tau_omp_stop_timer("PARALLEL REGION", tid, 1);
        Tau_collector_flags[tid].busy = 0;
      }
/*
      Tau_omp_start_timer("IDLE", tid, 0);
      Tau_collector_flags[tid].idle = 1;
*/
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
      Tau_collector_flags[tid].activeTimerContext = malloc(strlen(Tau_collector_flags[tid].timerContext)+1);
      strcpy(Tau_collector_flags[tid].activeTimerContext, Tau_collector_flags[tid].timerContext);
      Tau_omp_start_timer("PARALLEL REGION", tid, 1);
      Tau_collector_flags[tid].busy = 1;
      break;
    case OMP_EVENT_THR_BEGIN_IBAR:
      Tau_omp_start_timer("Implicit BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_END_IBAR:
      Tau_omp_stop_timer("Implicit BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_EBAR:
      Tau_omp_start_timer("Explicit BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_END_EBAR:
      Tau_omp_stop_timer("Explicit BARRIER", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_LKWT:
      Tau_omp_start_timer("Lock WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_END_LKWT:
      Tau_omp_stop_timer("Lock WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_CTWT:
      Tau_omp_start_timer("Critical Section WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_END_CTWT:
      Tau_omp_stop_timer("Critical Section WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_ODWT:
      // for some reason, the ordered region wait is entered twice for some threads.
      if (Tau_collector_flags[tid].ordered_region_wait == 0) {
        Tau_omp_start_timer("Ordered Region WAIT", tid, 1);
      }
      Tau_collector_flags[tid].ordered_region_wait = 1;
      break;
    case OMP_EVENT_THR_END_ODWT:
      if (Tau_collector_flags[tid].ordered_region_wait == 1) {
        Tau_omp_stop_timer("Ordered Region WAIT", tid, 1);
      }
      Tau_collector_flags[tid].ordered_region_wait = 0;
      break;
    case OMP_EVENT_THR_BEGIN_MASTER:
      Tau_omp_start_timer("Master Region", tid, 1);
      break;
    case OMP_EVENT_THR_END_MASTER:
      Tau_omp_stop_timer("Master Region", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_SINGLE:
      Tau_omp_start_timer("Single Region", tid, 1);
      break;
    case OMP_EVENT_THR_END_SINGLE:
      Tau_omp_stop_timer("Single Region", tid, 1);
      break;
    case OMP_EVENT_THR_BEGIN_ORDERED:
      // for some reason, the ordered region is entered twice for some threads.
      if (Tau_collector_flags[tid].ordered_region == 0) {
        Tau_omp_start_timer("Ordered Region", tid, 1);
        Tau_collector_flags[tid].ordered_region = 1;
      }
      break;
    case OMP_EVENT_THR_END_ORDERED:
      if (Tau_collector_flags[tid].ordered_region == 1) {
        Tau_omp_stop_timer("Ordered Region", tid, 1);
      }
      Tau_collector_flags[tid].ordered_region = 0;
      break;
    case OMP_EVENT_THR_BEGIN_ATWT:
      Tau_omp_start_timer("Atomic Region WAIT", tid, 1);
      break;
    case OMP_EVENT_THR_END_ATWT:
      Tau_omp_stop_timer("Atomic Region WAIT", tid, 1);
      break;
  }
  //printf("** Thread: %d, EVENT:%s handled. **\n", tid, OMP_EVENT_NAME[event-1]);
  fflush(stdout);
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

  void * handle = NULL;
  handle = dlopen("libopenmp.so", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    //dlerror();    /* Clear any existing error */
    TAU_VERBOSE("libopenmp.so not found... \n");
    handle = dlopen("liblibgomp_g_wrap.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      TAU_VERBOSE("liblibgomp_g_wrap.so not found... collector API not enabled. \n");
      return -1;
    }
  }

  //dlerror();    /* Clear any existing error */

  *(void **) (&Tau_collector_api) = dlsym(handle, "__omp_collector_api");
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
  rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
  TAU_VERBOSE("__omp_collector_api() returned %d\n", rc);
  free(message);

  /*test for request of all events*/
  int i;
  int num_req=OMP_EVENT_THR_END_ATWT; /* last event */
  int register_sz = sizeof(OMP_COLLECTORAPI_EVENT)+sizeof(void *);
  int mes_size = OMP_COLLECTORAPI_HEADERSIZE+register_sz;
  message = (void *) malloc(num_req*mes_size+sizeof(int));
  for(i=0;i<num_req;i++) {  
    Tau_fill_header(message+mes_size*i,mes_size, OMP_REQ_REGISTER, OMP_ERRCODE_OK, 0, 0);
    Tau_fill_register((message+mes_size*i)+OMP_COLLECTORAPI_HEADERSIZE,OMP_EVENT_FORK+i,1, Tau_omp_event_handler, i==(num_req-1));
  } 
  rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
  TAU_VERBOSE("__omp_collector_api() returned %d\n", rc);
  free(message);

  // preallocate messages, because we can't malloc when signals are
  // handled
  int state_rsz = sizeof(OMP_COLLECTOR_API_THR_STATE)+sizeof(unsigned long);
  for(i=0;i<omp_get_max_threads();i++) {  
    Tau_collector_flags[i].signal_message = malloc(OMP_COLLECTORAPI_HEADERSIZE+state_rsz);
    Tau_fill_header(Tau_collector_flags[i].signal_message, OMP_COLLECTORAPI_HEADERSIZE+state_rsz, OMP_REQ_STATE, OMP_ERRCODE_OK, state_rsz, 1);
  }

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
  rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
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
  (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(Tau_collector_flags[tid].signal_message));
  int * rid = Tau_collector_flags[tid].signal_message + OMP_COLLECTORAPI_HEADERSIZE;
  thread_state = *rid;
  TAU_VERBOSE("Thread %d, state : %d\n", tid, thread_state);
  // return the thread state as a string
  return (int)(thread_state);
}


