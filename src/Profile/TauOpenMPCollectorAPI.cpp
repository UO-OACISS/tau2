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
  int parallel; // 4 bytes
  int implicit_barrier; // 4 bytes
  int explicit_barrier; // 4 bytes
  int ordered_region_wait; // 4 bytes
  int ordered_region; // 4 bytes
  int padding[10];
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

static int regionID = 0;
static char regionIDstr[32] = {0};

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

static void * handle;
//static int (*Tau_collector_api)(OMP_COLLECTORAPI_EVENT);
static int (*Tau_collector_api)(void *);

void Tau_omp_event_handler(OMP_COLLECTORAPI_EVENT event)
{
/* get the region ID */
  omp_collector_message req;
  void *message = (void *) malloc(4);
  int *sz = (int *) message;
  *sz = 0;

  int currentid_rsz = sizeof(long);
  message = (void *) malloc(OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz+sizeof(int));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE+currentid_rsz, OMP_REQ_CURRENT_PRID, OMP_ERRCODE_OK, currentid_rsz, 1);
  int localRegionID = (Tau_collector_api)(message);
  free(message);

  TAU_VERBOSE("** Thread: %d, RegionID: %d, EVENT:%s **\n", omp_get_thread_num(), localRegionID, OMP_EVENT_NAME[event-1]);

  switch(event) {
    case OMP_EVENT_FORK:
      sprintf(regionIDstr, "OpenMP PARALLEL REGION: %d", regionID);
      regionID++;
      Tau_collector_flags[omp_get_thread_num()].parallel++;
      TAU_STATIC_TIMER_START(regionIDstr);
      TAU_STATIC_TIMER_START("OpenMP BUSY");
      break;
    case OMP_EVENT_JOIN:
      if (Tau_collector_flags[omp_get_thread_num()].idle == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP IDLE");
      } else {
        TAU_STATIC_TIMER_STOP("OpenMP BUSY");
      }
      Tau_collector_flags[omp_get_thread_num()].idle = 0;
      sprintf(regionIDstr, "OpenMP PARALLEL REGION: %d", regionID - Tau_collector_flags[omp_get_thread_num()].parallel);
      TAU_STATIC_TIMER_STOP(regionIDstr);
      Tau_collector_flags[omp_get_thread_num()].parallel--;
      break;
    case OMP_EVENT_THR_BEGIN_IDLE:
      // yes, STOP the "busy" timer
      TAU_STATIC_TIMER_STOP("OpenMP BUSY");
      TAU_STATIC_TIMER_START("OpenMP IDLE");
      Tau_collector_flags[omp_get_thread_num()].idle = 1;
      break;
    case OMP_EVENT_THR_END_IDLE:
      if (Tau_collector_flags[omp_get_thread_num()].idle == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP IDLE");
      }
      Tau_collector_flags[omp_get_thread_num()].idle = 0;
      // yes, START the "busy" timer
      TAU_STATIC_TIMER_START("OpenMP BUSY");
      break;
    case OMP_EVENT_THR_BEGIN_IBAR:
      TAU_STATIC_TIMER_START("OpenMP Implicit BARRIER");
      Tau_collector_flags[omp_get_thread_num()].implicit_barrier = 1;
      break;
    case OMP_EVENT_THR_END_IBAR:
      if (Tau_collector_flags[omp_get_thread_num()].implicit_barrier == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP Implicit BARRIER");
        Tau_collector_flags[omp_get_thread_num()].implicit_barrier = 0;
      } else if (Tau_collector_flags[omp_get_thread_num()].explicit_barrier == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP Explicit BARRIER");
        Tau_collector_flags[omp_get_thread_num()].explicit_barrier = 0;
      }
      break;
    case OMP_EVENT_THR_BEGIN_EBAR:
      TAU_STATIC_TIMER_START("OpenMP Explicit BARRIER");
      Tau_collector_flags[omp_get_thread_num()].explicit_barrier = 1;
      break;
    case OMP_EVENT_THR_END_EBAR:
      if (Tau_collector_flags[omp_get_thread_num()].explicit_barrier == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP Explicit BARRIER");
        Tau_collector_flags[omp_get_thread_num()].explicit_barrier = 0;
      } else if (Tau_collector_flags[omp_get_thread_num()].implicit_barrier == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP Implicit BARRIER");
        Tau_collector_flags[omp_get_thread_num()].implicit_barrier = 0;
      }
      break;
    case OMP_EVENT_THR_BEGIN_LKWT:
      TAU_STATIC_TIMER_START("OpenMP Lock WAIT");
      break;
    case OMP_EVENT_THR_END_LKWT:
      TAU_STATIC_TIMER_STOP("OpenMP Lock WAIT");
      break;
    case OMP_EVENT_THR_BEGIN_CTWT:
      TAU_STATIC_TIMER_START("OpenMP Critical Section WAIT");
      break;
    case OMP_EVENT_THR_END_CTWT:
      TAU_STATIC_TIMER_STOP("OpenMP Critical Section WAIT");
      break;
    case OMP_EVENT_THR_BEGIN_ODWT:
      // for some reason, the ordered region wait is entered twice for some threads.
      if (Tau_collector_flags[omp_get_thread_num()].ordered_region_wait == 0) {
        TAU_STATIC_TIMER_START("OpenMP Ordered Region WAIT");
      }
      Tau_collector_flags[omp_get_thread_num()].ordered_region_wait = 1;
      break;
    case OMP_EVENT_THR_END_ODWT:
      if (Tau_collector_flags[omp_get_thread_num()].ordered_region_wait == 1) {
        TAU_STATIC_TIMER_STOP("OpenMP Ordered Region WAIT");
      }
      Tau_collector_flags[omp_get_thread_num()].ordered_region_wait = 0;
      break;
    case OMP_EVENT_THR_BEGIN_MASTER:
      TAU_STATIC_TIMER_START("OpenMP Master Region");
      break;
    case OMP_EVENT_THR_END_MASTER:
      TAU_STATIC_TIMER_STOP("OpenMP Master Region");
      break;
    case OMP_EVENT_THR_BEGIN_SINGLE:
      TAU_STATIC_TIMER_START("OpenMP Single Region");
      break;
    case OMP_EVENT_THR_END_SINGLE:
      TAU_STATIC_TIMER_STOP("OpenMP Single Region");
      break;
    case OMP_EVENT_THR_BEGIN_ORDERED:
      // for some reason, the ordered region is entered twice for some threads.
      if (Tau_collector_flags[omp_get_thread_num()].ordered_region == 0) {
        TAU_STATIC_TIMER_START("OpenMP Ordered Region");
        Tau_collector_flags[omp_get_thread_num()].ordered_region = 1;
      }
      break;
    case OMP_EVENT_THR_END_ORDERED:
      TAU_STATIC_TIMER_STOP("OpenMP Ordered Region");
      Tau_collector_flags[omp_get_thread_num()].ordered_region = 0;
      break;
    case OMP_EVENT_THR_BEGIN_ATWT:
      TAU_STATIC_TIMER_START("OpenMP Atomic Region WAIT");
      break;
    case OMP_EVENT_THR_END_ATWT:
      TAU_STATIC_TIMER_STOP("OpenMP Atomic Region WAIT");
      break;
  }
  return;
}

void Tau_fill_header(void *message, int sz, OMP_COLLECTORAPI_REQUEST rq, OMP_COLLECTORAPI_EC ec, int rsz, int append_zero)
{
    int *psz = (int *) message; 
   *psz = sz;
   
   OMP_COLLECTORAPI_REQUEST *rnum = (OMP_COLLECTORAPI_REQUEST *)(message) + sizeof(int);
   *rnum = rq;
   
   OMP_COLLECTORAPI_EC *pec = (OMP_COLLECTORAPI_EC *)(message) + (sizeof(int)*2);
   *pec = ec;

   int *prsz = (int *)(message) + (sizeof(int)*3);
   *prsz = rsz;

   if(append_zero) {
    psz = (int *)(message) + ((sizeof(int)*4)+rsz);
   *psz =0; 
   }   
  
}

void Tau_fill_register(void *message, OMP_COLLECTORAPI_EVENT event, int append_func, void (*func)(OMP_COLLECTORAPI_EVENT), int append_zero) {

  // get a pointer to the head of the message
  OMP_COLLECTORAPI_EVENT *pevent = (OMP_COLLECTORAPI_EVENT *) message;
  // assign the event to the first parameter
  *pevent = event;

  // increment to the next parameter
  char *mem = (char *)(message) + sizeof(OMP_COLLECTORAPI_EVENT);
  if(append_func) {
         unsigned long * lmem = (unsigned long *)(message) + sizeof(OMP_COLLECTORAPI_EVENT);
         *lmem = (unsigned long)func;
  }

     if(append_zero) {
       int *psz;
       if(append_func) {
            psz = (int *)(message) + (sizeof(OMP_COLLECTORAPI_EVENT)+ sizeof(void *)); 
   
     } else {

          psz = (int *)(message) + (sizeof(OMP_COLLECTORAPI_EVENT));

     }
       *psz =0;  
     } 
}

int Tau_initialize_collector_api(void) {
  char *error;

  TAU_VERBOSE("Tau_initialize_collector_api()\n");

  handle = dlopen("libopenmp.so", RTLD_LAZY);
  if (!handle) {
    TAU_VERBOSE("%s\n", dlerror());
    //exit(EXIT_FAILURE);
    return 0;
  }

  dlerror();    /* Clear any existing error */

  *(void **) (&Tau_collector_api) = dlsym(handle, "__omp_collector_api");

  if ((error = dlerror()) != NULL)  {
    TAU_VERBOSE("%s\n", error);
    //exit(EXIT_FAILURE);
    return 0;
  }

  omp_collector_message req;
  void *message = (void *) malloc(4);   
  int *sz = (int *) message; 
  *sz = 0;
  int rc = 0;

  /*test: check for request start, 1 message */
  message = (void *) malloc(OMP_COLLECTORAPI_HEADERSIZE+sizeof(int));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE, OMP_REQ_START, OMP_ERRCODE_OK, 0, 1);
  //rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
  rc = (Tau_collector_api)(message);
  printf("__omp_collector_api() returned %d\n", rc);
  free(message);

  /*test for request of all events*/
  int i;
  int num_req=OMP_EVENT_THR_END_ATWT; /* last event */
  int register_sz = sizeof(OMP_COLLECTORAPI_EVENT)+sizeof(void *);
  int mes_size = OMP_COLLECTORAPI_HEADERSIZE+register_sz;
  message = (void *) malloc(num_req*mes_size+sizeof(int));
  OMP_COLLECTORAPI_EVENT event = OMP_EVENT_FORK;
  for(i=0;i<num_req;i++) {  
    event = (OMP_COLLECTORAPI_EVENT)(OMP_EVENT_FORK + i);
    int * tmpPointer = (int *)(message) + (mes_size * i);
    Tau_fill_header(tmpPointer,mes_size, OMP_REQ_REGISTER, OMP_ERRCODE_OK, 0, 0);
    Tau_fill_register((tmpPointer)+OMP_COLLECTORAPI_HEADERSIZE,event,1, Tau_omp_event_handler, i==(num_req-1));
  } 
  //rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
  rc = (Tau_collector_api)(message);
  printf("__omp_collector_api() returned %d\n", rc);
  free(message);

  return 1;
}

int Tau_finalize_collector_api(void) {
  fprintf(stderr, "finalizeCollector()\n");

  omp_collector_message req;
  void *message = (void *) malloc(4);   
  int *sz = (int *) message; 
  *sz = 0;
  int rc = 0;

  /*test check for request stop, 1 message */
  message = (void *) malloc(OMP_COLLECTORAPI_HEADERSIZE+sizeof(int));
  Tau_fill_header(message, OMP_COLLECTORAPI_HEADERSIZE, OMP_REQ_STOP, OMP_ERRCODE_OK, 0, 1);
  //rc = (Tau_collector_api)((OMP_COLLECTORAPI_EVENT)(message));
  rc = (Tau_collector_api)(message);
  printf("__omp_collector_api() returned %d\n", rc);
  free(message);
}


