#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdbool.h>
#include <Profile/Profiler.h>
#include <stdio.h>

#include "omp_collector_util.h"
#include <stdlib.h>

#include "gomp_wrapper_types.h"
#include "omp.h"

#if 0
#define DEBUGPRINT(format, args...) \
{ printf(format, ## args); fflush(stdout); }
#else
#define DEBUGPRINT(format, args...) \
{ }
#endif

#if 0
#define TAU_TIMER_START_ENABLED(NAME,TID) if (Tau_global_get_insideTAU() == 0) {Tau_pure_start_task(NAME,TID);}
#define TAU_TIMER_START_ENABLED(NAME,TID) if (Tau_global_get_insideTAU() == 0) {Tau_pure_stop_task(NAME,TID);}
//#define TAU_TIMER_START_ENABLED(NAME,TID) if (Tau_global_get_insideTAU() == 0) {fprintf(stderr,"%d starting %s\n",TID,NAME);fflush(stderr);}
//#define TAU_TIMER_STOP_ENABLED(NAME,TID)  if (Tau_global_get_insideTAU() == 0) {fprintf(stderr,"%d stopping %s\n",TID,NAME);fflush(stderr);}
#else
#define TAU_TIMER_START_ENABLED(NAME,TID)
#define TAU_TIMER_STOP_ENABLED(NAME,TID)
#endif


extern int Tau_global_get_insideTAU();

struct Tau_gomp_wrapper_status_flags {
    int depth; // 4 bytes
    int ordered[10];  // 40 bytes
    void * proxy[10]; // 80 bytes
};

/* This structure is designed to wrap the outlined functions
 * created for: 
 * GOMP_parallel_start
 * GOMP_parallel_loop_static_start
 * GOMP_parallel_loop_dynamic_start
 * GOMP_parallel_loop_guided_start
 * GOMP_parallel_loop_runtime_start
 * GOMP_parallel_sections_start
 * GOMP_task
 */
typedef struct Tau_gomp_proxy_wrapper {
    // The function pointer
    void (*a1)(void *);
    // The argument pointer
    void *a2;
    // The function name
    char * name;
    //
} TAU_GOMP_PROXY_WRAPPER;

/* This array is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
static struct Tau_gomp_wrapper_status_flags Tau_gomp_flags[TAU_MAX_THREADS] __attribute__ ((aligned(128))) = {0};
//__thread struct Tau_gomp_wrapper_status_flags Tau_gomp_flags = {0};
//__thread int _ordered[10] = {0}; // 4 bytes
//__thread int _depth = 0; // 4 bytes
//__thread void * _proxy[10] = {0}; // should be enough?

extern struct CallSiteInfo * Tau_sampling_resolveCallSite(unsigned long address,
        const char *tag, const char *childName, char **newShortName, char addAddress);

/* This function is used to wrap the outlined functions for parallel regions.
*/
void Tau_gomp_parallel_start_proxy(void * a2) {
    DEBUGPRINT("Parallel Proxy %d!\n", Tau_get_tid());
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(a2);

    __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
    __ompc_set_state(THR_WORK_STATE);
    if(proxy->name != NULL) TAU_TIMER_START_ENABLED(proxy->name, Tau_get_tid()); 
    (proxy->a1)(proxy->a2);
    if(proxy->name != NULL) TAU_TIMER_STOP_ENABLED(proxy->name, Tau_get_tid()); 
    __ompc_set_state(THR_IDLE_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IDLE);
}

/* This function is used to wrap the outlined functions for tasks.
*/
void Tau_gomp_task_proxy(void * a2) {
    DEBUGPRINT("Task Proxy %d!\n", Tau_get_tid());
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(a2);

    __ompc_event_callback(OMP_EVENT_THR_BEGIN_EXEC_TASK);
    __ompc_set_state(THR_WORK_STATE);
    //if(proxy->name != NULL) TAU_TIMER_START_ENABLED(proxy->name, Tau_get_tid()); 
    (proxy->a1)(proxy->a2);
    //if(proxy->name != NULL) TAU_TIMER_STOP_ENABLED(proxy->name, Tau_get_tid()); 
    __ompc_set_state(THR_TASK_FINISH_STATE);
    // this pair of events goes together to end the task
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_FINISH_TASK);
    __ompc_event_callback(OMP_EVENT_THR_END_FINISH_TASK);
    __ompc_set_state(THR_IDLE_STATE);
}

/**********************************************************
  omp_set_lock
 **********************************************************/

void  tau_omp_set_lock(omp_set_lock_p omp_set_lock_h, omp_lock_t *lock)  {
    DEBUGPRINT("omp_set_lock %d\n", Tau_get_tid());

    OMP_COLLECTOR_API_THR_STATE previous;
    if (Tau_global_get_insideTAU() == 0) { 
        previous = __ompc_set_state(THR_LKWT_STATE);
        __ompc_event_callback(OMP_EVENT_THR_BEGIN_LKWT);
    }

    (*omp_set_lock_h)(lock);

    if (Tau_global_get_insideTAU() == 0) { 
        __ompc_event_callback(OMP_EVENT_THR_END_LKWT);
        __ompc_set_state(previous);
    }
}


/**********************************************************
  omp_set_nest_lock
 **********************************************************/

void  tau_omp_set_nest_lock(omp_set_nest_lock_p omp_set_nest_lock_h, omp_nest_lock_t *nest_lock)  {
    DEBUGPRINT("omp_set_nest_lock %d\n", Tau_get_tid());

    OMP_COLLECTOR_API_THR_STATE previous;
    if (Tau_global_get_insideTAU() == 0) { 
        previous = __ompc_set_state(THR_LKWT_STATE);
        __ompc_event_callback(OMP_EVENT_THR_BEGIN_LKWT);
    }

    (*omp_set_nest_lock_h)(nest_lock);

    if (Tau_global_get_insideTAU() == 0) { 
        __ompc_event_callback(OMP_EVENT_THR_END_LKWT);
        __ompc_set_state(previous);
    }
}

/**********************************************************
  GOMP_barrier
 **********************************************************/

void  tau_GOMP_barrier(GOMP_barrier_p GOMP_barrier_h)  {
    DEBUGPRINT("GOMP_barrier %d\n", Tau_get_tid());

    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_EBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_EBAR);
    TAU_TIMER_START_ENABLED("GOMP_barrier", Tau_get_tid()); 

    (*GOMP_barrier_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_barrier", Tau_get_tid()); 
    __ompc_event_callback(OMP_EVENT_THR_END_EBAR);
    __ompc_set_state(previous);
}

/**********************************************************
  GOMP_critical_start
 **********************************************************/

void  tau_GOMP_critical_start(GOMP_critical_start_p GOMP_critical_start_h)  {
    DEBUGPRINT("GOMP_critical_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_critical_start", Tau_get_tid()); 

    (*GOMP_critical_start_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_critical_start", Tau_get_tid());
    //__ompc_set_state(THR_CTWT_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);
}


/**********************************************************
  GOMP_critical_end
 **********************************************************/

void  tau_GOMP_critical_end(GOMP_critical_end_p GOMP_critical_end_h)  {
    DEBUGPRINT("GOMP_critical_end %d\n", Tau_get_tid());

    //__ompc_event_callback(OMP_EVENT_THR_END_CTWT);
    //__ompc_set_state(THR_WORK_STATE);
    TAU_TIMER_START_ENABLED("GOMP_critical_end", Tau_get_tid()); 

    (*GOMP_critical_end_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_critical_end", Tau_get_tid());
}


/**********************************************************
  GOMP_critical_name_start
 **********************************************************/

void  tau_GOMP_critical_name_start(GOMP_critical_name_start_p GOMP_critical_name_start_h, void ** a1)  {
    DEBUGPRINT("GOMP_critical_name_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_critical_name_start", Tau_get_tid()); 

    (*GOMP_critical_name_start_h)( a1);

    TAU_TIMER_STOP_ENABLED("GOMP_critical_name_start", Tau_get_tid()); 
    //__ompc_set_state(THR_CTWT_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);

}


/**********************************************************
  GOMP_critical_name_end
 **********************************************************/

void  tau_GOMP_critical_name_end(GOMP_critical_name_end_p GOMP_critical_name_end_h, void ** a1)  {

    DEBUGPRINT("GOMP_critical_name_end %d\n", Tau_get_tid());

    //__ompc_event_callback(OMP_EVENT_THR_END_CTWT);
    //__ompc_set_state(THR_WORK_STATE);
    TAU_TIMER_START_ENABLED("GOMP_critical_name_end", Tau_get_tid()); 

    (*GOMP_critical_name_end_h)( a1);

    TAU_TIMER_STOP_ENABLED("GOMP_critical_name_end", Tau_get_tid());

}


/**********************************************************
  GOMP_atomic_start
 **********************************************************/

void  tau_GOMP_atomic_start(GOMP_atomic_start_p GOMP_atomic_start_h)  {

    DEBUGPRINT("GOMP_atomic_start %d\n", Tau_get_tid());

    __ompc_set_state(THR_ATWT_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ATWT);
    TAU_TIMER_START_ENABLED("GOMP_atomic_start", Tau_get_tid()); 

    (*GOMP_atomic_start_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_atomic_start", Tau_get_tid());

}


/**********************************************************
  GOMP_atomic_end
 **********************************************************/

void  tau_GOMP_atomic_end(GOMP_atomic_end_p GOMP_atomic_end_h)  {

    DEBUGPRINT("GOMP_atomic_end %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_atomic_end", Tau_get_tid()); 

    (*GOMP_atomic_end_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_atomic_end", Tau_get_tid());
    __ompc_event_callback(OMP_EVENT_THR_END_ATWT);
    __ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_loop_static_start
 **********************************************************/

bool  tau_GOMP_loop_static_start(GOMP_loop_static_start_p GOMP_loop_static_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_static_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_static_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_static_start", Tau_get_tid()); 
    return retval;

}


/**********************************************************
  GOMP_loop_dynamic_start
 **********************************************************/

bool tau_GOMP_loop_dynamic_start(GOMP_loop_dynamic_start_p GOMP_loop_dynamic_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_dynamic_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_dynamic_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_dynamic_start", Tau_get_tid()); 
    return retval;

}


/**********************************************************
  GOMP_loop_guided_start
 **********************************************************/

bool tau_GOMP_loop_guided_start(GOMP_loop_guided_start_p GOMP_loop_guided_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_guided_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_guided_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_guided_start", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_runtime_start
 **********************************************************/

bool tau_GOMP_loop_runtime_start(GOMP_loop_runtime_start_p GOMP_loop_runtime_start_h, long a1, long a2, long a3, long * a4, long * a5)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_runtime_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_runtime_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_runtime_start", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_static_start
 **********************************************************/

bool tau_GOMP_loop_ordered_static_start(GOMP_loop_ordered_static_start_p GOMP_loop_ordered_static_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_static_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_static_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_static_start", Tau_get_tid()); 
    Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth] = 1;
    //_ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_dynamic_start
 **********************************************************/

bool tau_GOMP_loop_ordered_dynamic_start(GOMP_loop_ordered_dynamic_start_p GOMP_loop_ordered_dynamic_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_dynamic_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_dynamic_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_dynamic_start", Tau_get_tid()); 
    Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth] = 1;
    //_ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);
    return retval;

}


/**********************************************************
  GOMP_loop_ordered_guided_start
 **********************************************************/

bool tau_GOMP_loop_ordered_guided_start(GOMP_loop_ordered_guided_start_p GOMP_loop_ordered_guided_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_guided_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_guided_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_guided_start", Tau_get_tid()); 
    Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth] = 1;
    //_ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_runtime_start
 **********************************************************/

bool tau_GOMP_loop_ordered_runtime_start(GOMP_loop_ordered_runtime_start_p GOMP_loop_ordered_runtime_start_h, long a1, long a2, long a3, long * a4, long * a5)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_runtime_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_runtime_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_runtime_start", Tau_get_tid()); 
    Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth] = 1;
    //_ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    return retval;

}


/**********************************************************
  GOMP_loop_static_next
 **********************************************************/

bool tau_GOMP_loop_static_next(GOMP_loop_static_next_p GOMP_loop_static_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_static_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_static_next", Tau_get_tid());
    retval  =  (*GOMP_loop_static_next_h)( a1,  a2);
    TAU_TIMER_STOP_ENABLED("GOMP_loop_static_next", Tau_get_tid());
    return retval;

}


/**********************************************************
  GOMP_loop_dynamic_next
 **********************************************************/

bool tau_GOMP_loop_dynamic_next(GOMP_loop_dynamic_next_p GOMP_loop_dynamic_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_dynamic_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_dynamic_next", Tau_get_tid()); 

    retval  =  (*GOMP_loop_dynamic_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_dynamic_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_guided_next
 **********************************************************/

bool tau_GOMP_loop_guided_next(GOMP_loop_guided_next_p GOMP_loop_guided_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_guided_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_guided_next", Tau_get_tid()); 

    retval  =  (*GOMP_loop_guided_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_guided_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_runtime_next
 **********************************************************/

bool tau_GOMP_loop_runtime_next(GOMP_loop_runtime_next_p GOMP_loop_runtime_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_runtime_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_runtime_next", Tau_get_tid()); 

    retval  =  (*GOMP_loop_runtime_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_runtime_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_static_next
 **********************************************************/

bool tau_GOMP_loop_ordered_static_next(GOMP_loop_ordered_static_next_p GOMP_loop_ordered_static_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_static_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_static_next", Tau_get_tid()); 

    __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
    retval  =  (*GOMP_loop_ordered_static_next_h)( a1,  a2);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_static_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_dynamic_next
 **********************************************************/

bool tau_GOMP_loop_ordered_dynamic_next(GOMP_loop_ordered_dynamic_next_p GOMP_loop_ordered_dynamic_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_dynamic_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_dynamic_next", Tau_get_tid()); 

    __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
    retval  =  (*GOMP_loop_ordered_dynamic_next_h)( a1,  a2);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_dynamic_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_guided_next
 **********************************************************/

bool tau_GOMP_loop_ordered_guided_next(GOMP_loop_ordered_guided_next_p GOMP_loop_ordered_guided_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_guided_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_guided_next", Tau_get_tid()); 

    __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
    retval  =  (*GOMP_loop_ordered_guided_next_h)( a1,  a2);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_guided_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_runtime_next
 **********************************************************/

bool tau_GOMP_loop_ordered_runtime_next(GOMP_loop_ordered_runtime_next_p GOMP_loop_ordered_runtime_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_runtime_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ordered_runtime_next", Tau_get_tid()); 

    __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
    retval  =  (*GOMP_loop_ordered_runtime_next_h)( a1,  a2);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ordered_runtime_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_parallel_loop_static_start
 **********************************************************/

void tau_GOMP_parallel_loop_static_start(GOMP_parallel_loop_static_start_p GOMP_parallel_loop_static_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

    DEBUGPRINT("GOMP_parallel_loop_static_start %d\n", Tau_get_tid());

    int tid = Tau_get_tid();
    TAU_TIMER_START_ENABLED("GOMP_parallel_loop_static_start", tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->name = NULL;
    //Tau_sampling_resolveCallSite((unsigned long)(a1), "OPENMP", NULL, &(proxy->name), 0);
    (*GOMP_parallel_loop_static_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    //_proxy[_depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    //_depth = _depth + 1;

    TAU_TIMER_STOP_ENABLED("GOMP_parallel_loop_static_start", tid); 

}


/**********************************************************
  GOMP_parallel_loop_dynamic_start
 **********************************************************/

void tau_GOMP_parallel_loop_dynamic_start(GOMP_parallel_loop_dynamic_start_p GOMP_parallel_loop_dynamic_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

    DEBUGPRINT("GOMP_parallel_loop_dynamic_start %d\n", Tau_get_tid());

    int tid = Tau_get_tid();
    TAU_TIMER_START_ENABLED("GOMP_parallel_loop_dynamic_start", tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->name = NULL;
    //Tau_sampling_resolveCallSite((unsigned long)(a1), "OPENMP", NULL, &(proxy->name), 0);
    (*GOMP_parallel_loop_dynamic_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    //_proxy[_depth] = proxy;
    //_depth = _depth + 1;

    TAU_TIMER_STOP_ENABLED("GOMP_parallel_loop_dynamic_start", tid); 

}


/**********************************************************
  GOMP_parallel_loop_guided_start
 **********************************************************/

void tau_GOMP_parallel_loop_guided_start(GOMP_parallel_loop_guided_start_p GOMP_parallel_loop_guided_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

    DEBUGPRINT("GOMP_parallel_loop_guided_start %d\n", Tau_get_tid());

    int tid = Tau_get_tid();
    TAU_TIMER_START_ENABLED("GOMP_parallel_loop_guided_start", tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->name = NULL;
    //Tau_sampling_resolveCallSite((unsigned long)(a1), "OPENMP", NULL, &(proxy->name), 0);
    (*GOMP_parallel_loop_guided_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    //_proxy[_depth] = proxy;
    //_depth = _depth + 1;

    TAU_TIMER_STOP_ENABLED("GOMP_parallel_loop_guided_start", tid); 

}


/**********************************************************
  GOMP_parallel_loop_runtime_start
 **********************************************************/

void tau_GOMP_parallel_loop_runtime_start(GOMP_parallel_loop_runtime_start_p GOMP_parallel_loop_runtime_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6)  {

    DEBUGPRINT("GOMP_parallel_loop_runtime_start %d\n", Tau_get_tid());

    int tid = Tau_get_tid();
    TAU_TIMER_START_ENABLED("GOMP_parallel_loop_runtime_start", tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->name = NULL;
    //Tau_sampling_resolveCallSite((unsigned long)(a1), "OPENMP", NULL, &(proxy->name), 0);
    (*GOMP_parallel_loop_runtime_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    //_proxy[_depth] = proxy;
    //_depth = _depth + 1;

    TAU_TIMER_STOP_ENABLED("GOMP_parallel_loop_runtime_start", tid); 

}


/**********************************************************
  GOMP_loop_end
 **********************************************************/

void tau_GOMP_loop_end(GOMP_loop_end_p GOMP_loop_end_h)  {

    DEBUGPRINT("GOMP_loop_end %d\n", Tau_get_tid());

    OMP_COLLECTOR_API_THR_STATE previous;
    //if (_ordered[_depth]) {
    if (Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth]) {
        //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
        __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
        Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth] = 0;
        //_ordered[_depth] = 0;
    }
    previous = __ompc_set_state(THR_IBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IBAR);
    TAU_TIMER_START_ENABLED("GOMP_loop_end", Tau_get_tid()); 

    (*GOMP_loop_end_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_loop_end", Tau_get_tid());
    __ompc_event_callback(OMP_EVENT_THR_END_IBAR);
    __ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_loop_end_nowait
 **********************************************************/

void tau_GOMP_loop_end_nowait(GOMP_loop_end_nowait_p GOMP_loop_end_nowait_h)  {

    DEBUGPRINT("GOMP_loop_end_nowait %d\n", Tau_get_tid());

    if (Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth]) {
        //if (_ordered[_depth]) 
        //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
        __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
        Tau_gomp_flags[Tau_get_tid()].ordered[Tau_gomp_flags[Tau_get_tid()].depth] = 0;
        //_ordered[_depth] = 0;
    }
    TAU_TIMER_START_ENABLED("GOMP_loop_end_nowait", Tau_get_tid()); 

    (*GOMP_loop_end_nowait_h)();

    TAU_TIMER_STOP_ENABLED("GOMP_loop_end_nowait", Tau_get_tid());
    __ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_loop_ull_static_start
 **********************************************************/

bool tau_GOMP_loop_ull_static_start(GOMP_loop_ull_static_start_p GOMP_loop_ull_static_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_static_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_static_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ull_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_static_start", Tau_get_tid()); 
    return retval;

}


/**********************************************************
  GOMP_loop_ull_dynamic_start
 **********************************************************/

bool tau_GOMP_loop_ull_dynamic_start(GOMP_loop_ull_dynamic_start_p GOMP_loop_ull_dynamic_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_dynamic_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_dynamic_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ull_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_dynamic_start", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ull_guided_start
 **********************************************************/

bool tau_GOMP_loop_ull_guided_start(GOMP_loop_ull_guided_start_p GOMP_loop_ull_guided_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_guided_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_guided_start", Tau_get_tid()); 

    retval  =  (*GOMP_loop_ull_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_guided_start", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ull_runtime_start
 **********************************************************/

bool tau_GOMP_loop_ull_runtime_start(GOMP_loop_ull_runtime_start_p GOMP_loop_ull_runtime_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_runtime_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_runtime_start", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_runtime_start", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_static_start
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_static_start(GOMP_loop_ull_ordered_static_start_p GOMP_loop_ull_ordered_static_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_static_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_static_start", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_static_start", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_dynamic_start
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_dynamic_start(GOMP_loop_ull_ordered_dynamic_start_p GOMP_loop_ull_ordered_dynamic_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_dynamic_start", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_dynamic_start", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_guided_start
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_guided_start(GOMP_loop_ull_ordered_guided_start_p GOMP_loop_ull_ordered_guided_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_guided_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_guided_start", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_guided_start", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_runtime_start
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_runtime_start(GOMP_loop_ull_ordered_runtime_start_p GOMP_loop_ull_ordered_runtime_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_runtime_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_runtime_start", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_runtime_start", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_static_next
 **********************************************************/

bool tau_GOMP_loop_ull_static_next(GOMP_loop_ull_static_next_p GOMP_loop_ull_static_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_static_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_static_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_static_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_static_next", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_dynamic_next
 **********************************************************/

bool tau_GOMP_loop_ull_dynamic_next(GOMP_loop_ull_dynamic_next_p GOMP_loop_ull_dynamic_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_dynamic_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_dynamic_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_dynamic_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_dynamic_next", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_guided_next
 **********************************************************/

bool tau_GOMP_loop_ull_guided_next(GOMP_loop_ull_guided_next_p GOMP_loop_ull_guided_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_guided_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_guided_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_guided_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_guided_next", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_runtime_next
 **********************************************************/

bool tau_GOMP_loop_ull_runtime_next(GOMP_loop_ull_runtime_next_p GOMP_loop_ull_runtime_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_runtime_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_runtime_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_runtime_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_runtime_next", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_static_next
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_static_next(GOMP_loop_ull_ordered_static_next_p GOMP_loop_ull_ordered_static_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_static_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_static_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_static_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_static_next", Tau_get_tid()); 

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_dynamic_next
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_dynamic_next(GOMP_loop_ull_ordered_dynamic_next_p GOMP_loop_ull_ordered_dynamic_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_dynamic_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_dynamic_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_dynamic_next", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_guided_next
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_guided_next(GOMP_loop_ull_ordered_guided_next_p GOMP_loop_ull_ordered_guided_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_guided_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_guided_next", Tau_get_tid());

    retval  =  (*GOMP_loop_ull_ordered_guided_next_h)( a1,  a2);

    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_guided_next", Tau_get_tid());

    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_runtime_next
 **********************************************************/

bool tau_GOMP_loop_ull_ordered_runtime_next(GOMP_loop_ull_ordered_runtime_next_p GOMP_loop_ull_ordered_runtime_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_runtime_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_loop_ull_ordered_runtime_next", Tau_get_tid());
    retval  =  (*GOMP_loop_ull_ordered_runtime_next_h)( a1,  a2);
    TAU_TIMER_STOP_ENABLED("GOMP_loop_ull_ordered_runtime_next", Tau_get_tid());
    return retval;

}


/**********************************************************
  GOMP_ordered_start
 **********************************************************/

void tau_GOMP_ordered_start(GOMP_ordered_start_p GOMP_ordered_start_h)  {

    DEBUGPRINT("GOMP_ordered_start %d\n", Tau_get_tid());

    // our turn to work in the ordered region!
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
    __ompc_set_state(THR_WORK_STATE);
    TAU_TIMER_START_ENABLED("GOMP_ordered_start", Tau_get_tid()); 
    (*GOMP_ordered_start_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_ordered_start", Tau_get_tid());

}


/**********************************************************
  GOMP_ordered_end
 **********************************************************/

void tau_GOMP_ordered_end(GOMP_ordered_end_p GOMP_ordered_end_h)  {

    DEBUGPRINT("GOMP_ordered_end %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_ordered_end", Tau_get_tid());
    (*GOMP_ordered_end_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_ordered_end", Tau_get_tid());
    // we can't do this, or we might get overlapping timers.
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);
}

/**********************************************************
  GOMP_parallel_start
 **********************************************************/

void tau_GOMP_parallel_start(GOMP_parallel_start_p GOMP_parallel_start_h, void (*a1)(void *), void * a2, unsigned int a3)  {
    int numThreads = a3 == 0 ? omp_get_max_threads() : 1;
    DEBUGPRINT("GOMP_parallel_start %d of %d\n", Tau_get_tid(), numThreads);

    __ompc_set_state(THR_OVHD_STATE);
    __ompc_event_callback(OMP_EVENT_FORK);
    int tid = Tau_get_tid();
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->name = NULL;
    //Tau_sampling_resolveCallSite((unsigned long)(a1), "OPENMP", NULL, &(proxy->name), 0);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    //_proxy[_depth] = proxy;
    //_depth = _depth + 1;

    // time the call
    TAU_TIMER_START_ENABLED("GOMP_parallel_start", tid); 
    (*GOMP_parallel_start_h)( &Tau_gomp_parallel_start_proxy, proxy,  a3);
    TAU_TIMER_STOP_ENABLED("GOMP_parallel_start", tid); 
    __ompc_set_state(THR_WORK_STATE);

    DEBUGPRINT("GOMP_parallel_start %d of %d (on exit)\n", Tau_get_tid(), omp_get_num_threads());

}


/**********************************************************
  GOMP_parallel_end
 **********************************************************/

void tau_GOMP_parallel_end(GOMP_parallel_end_p GOMP_parallel_end_h)  {

    DEBUGPRINT("GOMP_parallel_end %d of %d\n", Tau_get_tid(), omp_get_num_threads());

    __ompc_set_state(THR_IBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IBAR);
    int tid = Tau_get_tid();
    TAU_TIMER_START_ENABLED("GOMP_parallel_end", tid); 
    (*GOMP_parallel_end_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_parallel_end", tid); 
    // do this at the end, so we can join all the threads.
    __ompc_event_callback(OMP_EVENT_THR_END_IBAR);
    __ompc_event_callback(OMP_EVENT_JOIN);
    __ompc_set_state(THR_SERIAL_STATE);
    // free the proxy wrapper, and reduce the depth
    int depth = Tau_gomp_flags[tid].depth;
    if (Tau_gomp_flags[tid].proxy[depth] != NULL) {
        TAU_GOMP_PROXY_WRAPPER * proxy = Tau_gomp_flags[tid].proxy[depth];
        //int depth = _depth;
        //if (_proxy[depth] != NULL) 
        //TAU_GOMP_PROXY_WRAPPER * proxy = _proxy[depth];
        //free(proxy->name);
        free(proxy);
        Tau_gomp_flags[tid].proxy[depth] = NULL;
        Tau_gomp_flags[tid].depth = depth - 1;
        //_proxy[depth] = NULL;
        //_depth = depth - 1;
    } else {
        // assume the worst...
        Tau_gomp_flags[tid].depth = 0;
        Tau_gomp_flags[tid].proxy[0] = NULL;
        //_depth = 0;
        //_proxy[0] = NULL;
    }

}


/**********************************************************
  GOMP_task
 **********************************************************/

void tau_GOMP_task(GOMP_task_p GOMP_task_h, void (*a1)(void *), void * a2, void (*a3)(void *, void *), long a4, long a5, bool a6, unsigned int a7)  {

    DEBUGPRINT("GOMP_task %d\n", Tau_get_tid());

    int tid = Tau_get_tid();

    /* 
     * Don't do the proxy wrapper for tasks. What we need to do is allocate the
     * proxies and put them in some data structure that we can clean up when the
     * parallel region is over. However, that could produce massive overhead in
     * both time and space - every task that is created we wrap it. Instead, we will
     * trust that instrumentation and/or sampling will handle the time spent in 
     * the task.
     */

#if 1
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->name = NULL;
#ifdef TAU_UNWIND
    // this doesn't use unwind, but unwind support would include bfd, most likely
    Tau_sampling_resolveCallSite((unsigned long)(a1), "OPENMP", NULL, &(proxy->name), 0);
#endif
    if (proxy->name != NULL) {
        DEBUGPRINT("GOMP_task %s\n", proxy->name);
        TAU_TIMER_START_ENABLED(proxy->name, tid);
        __ompc_set_state(THR_TASK_CREATE_STATE);
        __ompc_event_callback(OMP_EVENT_THR_BEGIN_CREATE_TASK);
        (*GOMP_task_h)( Tau_gomp_task_proxy, proxy,  a3,  a4,  a5,  a6, a7);
        __ompc_event_callback(OMP_EVENT_THR_END_CREATE_TASK_IMM);
        __ompc_set_state(THR_TASK_SCHEDULE_STATE);
        TAU_TIMER_STOP_ENABLED(proxy->name, tid);
    } else {
        TAU_TIMER_START_ENABLED("OpenMP_TASK", tid);
        __ompc_set_state(THR_TASK_CREATE_STATE);
        __ompc_event_callback(OMP_EVENT_THR_BEGIN_CREATE_TASK);
        //(*GOMP_task_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
        (*GOMP_task_h)( Tau_gomp_task_proxy, proxy,  a3,  a4,  a5,  a6, a7);
        __ompc_event_callback(OMP_EVENT_THR_END_CREATE_TASK_IMM);
        __ompc_set_state(THR_TASK_SCHEDULE_STATE);
        TAU_TIMER_STOP_ENABLED("OpenMP_TASK", tid);
    }
#else
    /* just call the task creation, for now */
    __ompc_set_state(THR_TASK_CREATE_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_CREATE_TASK);
    //TAU_TIMER_START_ENABLED("GOMP_task", tid);
    (*GOMP_task_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    //TAU_TIMER_STOP_ENABLED("GOMP_task", tid);
    __ompc_event_callback(OMP_EVENT_THR_END_CREATE_TASK_IMM);
    __ompc_set_state(THR_TASK_SCHEDULE_STATE);
#endif

}


/**********************************************************
  GOMP_taskwait
 **********************************************************/

void tau_GOMP_taskwait(GOMP_taskwait_p GOMP_taskwait_h)  {

    DEBUGPRINT("GOMP_taskwait %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_taskwait", Tau_get_tid()); 
    (*GOMP_taskwait_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_taskwait", Tau_get_tid()); 
    __ompc_set_state(THR_IBAR_STATE);

}


/**********************************************************
  GOMP_taskyield - only exists in gcc 4.7 and greater, and only as a stub

  void tau_GOMP_taskyield(GOMP_taskyield_p GOMP_taskyield_h)  {

  DEBUGPRINT("GOMP_taskyield %d\n", Tau_get_tid());

  TAU_TIMER_START_ENABLED("GOMP_taskyield", Tau_get_tid());
  (*GOMP_taskyield_h)();
  TAU_TIMER_STOP_ENABLED("GOMP_taskyield", Tau_get_tid());

  }
 **********************************************************/


/**********************************************************
  GOMP_sections_start
 **********************************************************/

unsigned int tau_GOMP_sections_start(GOMP_sections_start_p GOMP_sections_start_h, unsigned int a1)  {

    unsigned int retval = 0;
    DEBUGPRINT("GOMP_sections_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_sections_start", Tau_get_tid());
    retval  =  (*GOMP_sections_start_h)( a1);
    TAU_TIMER_STOP_ENABLED("GOMP_sections_start", Tau_get_tid());
    return retval;

}


/**********************************************************
  GOMP_sections_next
 **********************************************************/

unsigned int tau_GOMP_sections_next(GOMP_sections_next_p GOMP_sections_next_h)  {

    unsigned int retval = 0;
    DEBUGPRINT("GOMP_sections_next %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_sections_next", Tau_get_tid());
    retval  =  (*GOMP_sections_next_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_sections_next", Tau_get_tid());
    return retval;

}


/**********************************************************
  GOMP_parallel_sections_start
 **********************************************************/

void tau_GOMP_parallel_sections_start(GOMP_parallel_sections_start_p GOMP_parallel_sections_start_h, void (*a1)(void *), void * a2, unsigned int a3, unsigned int a4)  {

    DEBUGPRINT("GOMP_parallel_sections_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_parallel_sections_start", Tau_get_tid()); 
    (*GOMP_parallel_sections_start_h)( a1,  a2,  a3,  a4);
    TAU_TIMER_STOP_ENABLED("GOMP_parallel_sections_start", Tau_get_tid()); 
    // do this AFTER the start, so we know how many threads there are.
    __ompc_event_callback(OMP_EVENT_FORK);
    __ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_sections_end
 **********************************************************/

void tau_GOMP_sections_end(GOMP_sections_end_p GOMP_sections_end_h)  {

    DEBUGPRINT("GOMP_sections_end %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_sections_end", Tau_get_tid());
    (*GOMP_sections_end_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_sections_end", Tau_get_tid());

}


/**********************************************************
  GOMP_sections_end_nowait
 **********************************************************/

void tau_GOMP_sections_end_nowait(GOMP_sections_end_nowait_p GOMP_sections_end_nowait_h)  {

    DEBUGPRINT("GOMP_sections_end_nowait %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_sections_end_nowait", Tau_get_tid());
    (*GOMP_sections_end_nowait_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_sections_end_nowait", Tau_get_tid());

}


/**********************************************************
  GOMP_single_start
 **********************************************************/

bool tau_GOMP_single_start(GOMP_single_start_p GOMP_single_start_h)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_single_start %d\n", Tau_get_tid());

    // in this case, the single region is entered and executed
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
    TAU_TIMER_START_ENABLED("GOMP_single_start", Tau_get_tid()); 
    retval  =  (*GOMP_single_start_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_single_start", Tau_get_tid()); 
    // the code is done, so fire the callback end event
    __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);

    return retval;

}


/**********************************************************
  GOMP_single_copy_start
 **********************************************************/

void * tau_GOMP_single_copy_start(GOMP_single_copy_start_p GOMP_single_copy_start_h)  {

    void * retval = 0;
    DEBUGPRINT("GOMP_single_copy_start %d\n", Tau_get_tid());

    TAU_TIMER_START_ENABLED("GOMP_single_copy_start", Tau_get_tid()); 
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
    retval  =  (*GOMP_single_copy_start_h)();
    TAU_TIMER_STOP_ENABLED("GOMP_single_copy_start", Tau_get_tid()); 
    return retval;

}


/**********************************************************
  GOMP_single_copy_end
 **********************************************************/

void tau_GOMP_single_copy_end(GOMP_single_copy_end_p GOMP_single_copy_end_h, void * a1)  {

    DEBUGPRINT("GOMP_single_copy_end %d\n", Tau_get_tid());

    __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);
    TAU_TIMER_START_ENABLED("GOMP_single_copy_end", Tau_get_tid());
    (*GOMP_single_copy_end_h)( a1);
    TAU_TIMER_STOP_ENABLED("GOMP_single_copy_end", Tau_get_tid());

}

