#include "omp_collector_api.h"
#include "omp.h"
#include <stdlib.h>
#include <stdio.h> 
#include <dlfcn.h> // for dynamic loading of symbols
#include <TAU.h>

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
  int implicit_barrier; // 4 bytes
  int explicit_barrier; // 4 bytes
  int ordered_region_wait; // 4 bytes
  int ordered_region; // 4 bytes
  long regionID; // 8 bytes(?)
  long activeRegionID; // 8 bytes(?)
  int padding[5];
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

void Tau_get_regionID(int tid, int forking) {
/* get the region ID */
  omp_collector_message req;
  int currentid_rsz = sizeof(long);
  void * message = (void *) calloc(OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz+sizeof(int), sizeof(char));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz, OMP_REQ_CURRENT_PRID, OMP_ERRCODE_OK, currentid_rsz, 1);
  long * rid = message + OMP_COLLECTORAPI_HEADERSIZE;
  int rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
  TAU_VERBOSE("Thread %d, region ID : %ld\n", tid, *rid);
  Tau_collector_flags[tid].regionID = *rid;
  if (forking > 0) {
    int i;
    for (i == 0 ; i < omp_get_num_threads() ; i++) {
      Tau_collector_flags[i].regionID = *rid;
    }
  }
  free(message);
  return;
}

__inline void Tau_omp_start_timer(const char * state, int tid) {
  char regionIDstr[32] = {0};
  if (tid < 0) {
    sprintf(regionIDstr, "OpenMP %s", state);
  } else {
    sprintf(regionIDstr, "OpenMP %s: %d", state, Tau_collector_flags[tid].regionID);
    Tau_collector_flags[tid].activeRegionID = Tau_collector_flags[tid].regionID;
  }
  TAU_STATIC_TIMER_START(regionIDstr);
}

__inline void Tau_omp_stop_timer(const char * state, int tid) {
  char regionIDstr[32] = {0};
  if (tid < 0) {
    sprintf(regionIDstr, "OpenMP %s", state);
  } else {
    sprintf(regionIDstr, "OpenMP %s: %d", state, Tau_collector_flags[tid].activeRegionID);
  }
  TAU_STATIC_TIMER_STOP(regionIDstr);
}

void Tau_omp_event_handler(OMP_COLLECTORAPI_EVENT event) {
  int tid = omp_get_thread_num();
  TAU_VERBOSE("** Thread: %d, EVENT:%s **\n", tid, OMP_EVENT_NAME[event-1]);

  switch(event) {
    case OMP_EVENT_FORK:
      Tau_get_regionID(tid, 1);
      Tau_omp_start_timer("PARALLEL REGION", tid);
      Tau_collector_flags[tid].parallel++;
/*
      Tau_omp_start_timer("BUSY", tid);
      Tau_collector_flags[tid].busy = 1;
*/
      break;
    case OMP_EVENT_JOIN:
      Tau_get_regionID(tid, 1);
      if (Tau_collector_flags[tid].idle == 1) {
        Tau_omp_stop_timer("IDLE", -1);
        Tau_collector_flags[tid].idle = 0;
/*
      } else if (Tau_collector_flags[tid].busy == 1) {
        Tau_omp_stop_timer("BUSY", tid);
        Tau_collector_flags[tid].busy = 0;
*/
      }
      Tau_omp_stop_timer("PARALLEL REGION", tid);
      Tau_collector_flags[tid].parallel--;
      break;
    case OMP_EVENT_THR_BEGIN_IDLE:
      // sometimes IDLE can be called twice in a row
      if (Tau_collector_flags[tid].idle == 1) {
        break;
      }
      if (Tau_collector_flags[tid].busy == 1) {
        // yes, STOP the "busy" timer
        //Tau_omp_stop_timer("BUSY", tid);
        Tau_omp_stop_timer("PARALLEL REGION", tid);
        Tau_collector_flags[tid].busy = 0;
      }
      Tau_omp_start_timer("IDLE", -1);
      Tau_collector_flags[tid].idle = 1;
      break;
    case OMP_EVENT_THR_END_IDLE:
      if (Tau_collector_flags[tid].idle == 1) {
        Tau_omp_stop_timer("IDLE", -1);
        Tau_collector_flags[tid].idle = 0;
      }
      // yes, START the "busy" timer
      //Tau_omp_start_timer("BUSY", tid);
      Tau_omp_start_timer("PARALLEL REGION", tid);
      Tau_collector_flags[tid].busy = 1;
      break;
    case OMP_EVENT_THR_BEGIN_IBAR:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Implicit BARRIER", tid);
      Tau_collector_flags[tid].implicit_barrier = 1;
      break;
    case OMP_EVENT_THR_END_IBAR:
      //Tau_get_regionID(tid, 0);
      if (Tau_collector_flags[tid].implicit_barrier == 1) {
        Tau_omp_stop_timer("Implicit BARRIER", tid);
        Tau_collector_flags[tid].implicit_barrier = 0;
/*
      } else if (Tau_collector_flags[tid].explicit_barrier == 1) {
        Tau_omp_stop_timer("Implicit BARRIER", tid);
        Tau_collector_flags[tid].explicit_barrier = 0;
*/
      }
      break;
    case OMP_EVENT_THR_BEGIN_EBAR:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Explicit BARRIER", tid);
      Tau_collector_flags[tid].explicit_barrier = 1;
      break;
    case OMP_EVENT_THR_END_EBAR:
      //Tau_get_regionID(tid, 0);
      if (Tau_collector_flags[tid].explicit_barrier == 1) {
        Tau_omp_stop_timer("Explicit BARRIER", tid);
        Tau_collector_flags[tid].explicit_barrier = 0;
/*
      } else if (Tau_collector_flags[tid].implicit_barrier == 1) {
        Tau_omp_stop_timer("Implicit BARRIER", tid);
        Tau_collector_flags[tid].implicit_barrier = 0;
*/
      }
      break;
    case OMP_EVENT_THR_BEGIN_LKWT:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Lock WAIT", tid);
      break;
    case OMP_EVENT_THR_END_LKWT:
      //Tau_get_regionID(tid, 0);
      Tau_omp_stop_timer("Lock WAIT", tid);
      break;
    case OMP_EVENT_THR_BEGIN_CTWT:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Critical Section WAIT", tid);
      break;
    case OMP_EVENT_THR_END_CTWT:
      //Tau_get_regionID(tid, 0);
      Tau_omp_stop_timer("Critical Section WAIT", tid);
      break;
    case OMP_EVENT_THR_BEGIN_ODWT:
      //Tau_get_regionID(tid, 0);
      // for some reason, the ordered region wait is entered twice for some threads.
      if (Tau_collector_flags[tid].ordered_region_wait == 0) {
        Tau_omp_start_timer("Ordered Region WAIT", tid);
      }
      Tau_collector_flags[tid].ordered_region_wait = 1;
      break;
    case OMP_EVENT_THR_END_ODWT:
      //Tau_get_regionID(tid, 0);
      if (Tau_collector_flags[tid].ordered_region_wait == 1) {
        Tau_omp_stop_timer("Ordered Region WAIT", tid);
      }
      Tau_collector_flags[tid].ordered_region_wait = 0;
      break;
    case OMP_EVENT_THR_BEGIN_MASTER:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Master Region", tid);
      break;
    case OMP_EVENT_THR_END_MASTER:
      //Tau_get_regionID(tid, 0);
      Tau_omp_stop_timer("Master Region", tid);
      break;
    case OMP_EVENT_THR_BEGIN_SINGLE:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Single Region", tid);
      break;
    case OMP_EVENT_THR_END_SINGLE:
      //Tau_get_regionID(tid, 0);
      Tau_omp_stop_timer("Single Region", tid);
      break;
    case OMP_EVENT_THR_BEGIN_ORDERED:
      //Tau_get_regionID(tid, 0);
      // for some reason, the ordered region is entered twice for some threads.
      if (Tau_collector_flags[tid].ordered_region == 0) {
        Tau_omp_start_timer("Ordered Region", tid);
        Tau_collector_flags[tid].ordered_region = 1;
      }
      break;
    case OMP_EVENT_THR_END_ORDERED:
      //Tau_get_regionID(tid, 0);
      if (Tau_collector_flags[tid].ordered_region == 1) {
        Tau_omp_stop_timer("Ordered Region", tid);
      }
      Tau_collector_flags[tid].ordered_region = 0;
      break;
    case OMP_EVENT_THR_BEGIN_ATWT:
      //Tau_get_regionID(tid, 0);
      Tau_omp_start_timer("Atomic Region WAIT", tid);
      break;
    case OMP_EVENT_THR_END_ATWT:
      //Tau_get_regionID(tid, 0);
      Tau_omp_stop_timer("Atomic Region WAIT", tid);
      break;
  }
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

  TAU_VERBOSE("Tau_initialize_collector_api()\n");

  void * handle = NULL;
  handle = dlopen("libopenmp.so", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    //TAU_VERBOSE("%s\n", dlerror());
    return -1;
  }

  //dlerror();    /* Clear any existing error */

  *(void **) (&Tau_collector_api) = dlsym(handle, "__omp_collector_api");

  //if ((error = dlerror()) != NULL)  {
  //  TAU_VERBOSE("%s\n", error);
  //  return -2;
  //}

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


