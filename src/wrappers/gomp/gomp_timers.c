#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdbool.h>
#include <Profile/Profiler.h>
#include <stdio.h>

#include "omp_collector_util.h"
#include <stdlib.h>
#include <stdint.h>

#include "gomp_wrapper_types.h"
#include "omp.h"

#define TAU_GOMP_INLINE inline extern

#if 0
#define DEBUGPRINT(format, args...) \
{ printf(format, ## args); fflush(stdout); }
#else
#define DEBUGPRINT(format, args...) \
{ }
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
    // copy function buffer?
    char *buf;
} TAU_GOMP_PROXY_WRAPPER;

/* This array is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
//static struct Tau_gomp_wrapper_status_flags Tau_gomp_flags[TAU_MAX_THREADS] __attribute__ ((aligned(128))) = {0};
static __thread int _depth = 0;
static __thread int _ordered[10] = {0};
static __thread TAU_GOMP_PROXY_WRAPPER * _proxy[10] = {0};
static __thread TAU_GOMP_PROXY_WRAPPER * _current_proxy = 0;

extern void * Tau_get_gomp_proxy_address() {
    if (_current_proxy != 0) {
	    return _current_proxy->a1;
	}
    if (_depth == 0) {
	    return 0;
	}
    return (_proxy[_depth-1])->a1;
};

extern struct CallSiteInfo * Tau_sampling_resolveCallSite(unsigned long address,
        const char *tag, const char *childName, char **newShortName, char addAddress);

/* This function is used to wrap the outlined functions for parallel regions.
*/
void Tau_gomp_parallel_start_proxy(void * a2) {
	Tau_global_incr_insideTAU();
    DEBUGPRINT("Parallel Proxy %d!\n", Tau_get_thread());
    TAU_GOMP_PROXY_WRAPPER * old_proxy = _current_proxy;
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(a2);
	_current_proxy = proxy;
    __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_WORK_STATE);
	Tau_global_decr_insideTAU();
    (proxy->a1)(proxy->a2);
	Tau_global_incr_insideTAU();
    __ompc_set_state(previous);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IDLE);
	_current_proxy = old_proxy;
	Tau_global_decr_insideTAU();
}

/* This function is used to wrap the outlined functions for tasks.
*/
void Tau_gomp_task_proxy(void * a2) {
	Tau_global_incr_insideTAU();
    TAU_GOMP_PROXY_WRAPPER * old_proxy = _current_proxy;
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(a2);
    DEBUGPRINT("Task Proxy %d, %p, %p\n", Tau_get_thread(), proxy->a1, proxy->a2);
	_current_proxy = proxy;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_EXEC_TASK);
    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_WORK_STATE);
	Tau_global_decr_insideTAU();
    (proxy->a1)(proxy->a2);
	Tau_global_incr_insideTAU();
	//if (proxy->a3) free(proxy->a2);
    __ompc_set_state(THR_TASK_FINISH_STATE);
    // this pair of events goes together to end the task
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_FINISH_TASK);
    __ompc_event_callback(OMP_EVENT_THR_END_FINISH_TASK);
    __ompc_set_state(previous);
	_current_proxy = old_proxy;
	Tau_global_decr_insideTAU();
}

/**********************************************************
  pthread_create
 **********************************************************/

// proxy function pointer
typedef struct tau_pthread_pack {
  start_routine_p start_routine;
  void * arg;
} TAU_PTHREAD_PACK_T;

// proxy function for starting the top level timer (and intitializing sampling)
void * tau_gomp_pthread_function(void *arg)
{
  TAU_PTHREAD_PACK_T * pack = (TAU_PTHREAD_PACK_T*)arg;
  //TAU_REGISTER_THREAD();
  Tau_create_top_level_timer_if_necessary();
  void * ret = pack->start_routine(pack->arg);
  return ret;
}

// the actual wrapper
int tau_gomp_pthread_create_wrapper(pthread_create_p pthread_create_call, pthread_t * threadp, const pthread_attr_t * attr, start_routine_p start_routine, void * arg) {
  // create the proxy wrapper
  TAU_PTHREAD_PACK_T * pack = (TAU_PTHREAD_PACK_T*)(malloc(sizeof(TAU_PTHREAD_PACK_T)));
  pack->start_routine = start_routine;
  pack->arg = arg;
  // spawn the thread
  return pthread_create_call(threadp, attr, tau_gomp_pthread_function, (void*)pack);
}

/**********************************************************
  omp_set_lock
 **********************************************************/

TAU_GOMP_INLINE void  tau_omp_set_lock(omp_set_lock_p omp_set_lock_h, omp_lock_t *lock)  {
    //DEBUGPRINT("omp_set_lock %d\n", Tau_get_thread());

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

TAU_GOMP_INLINE void  tau_omp_set_nest_lock(omp_set_nest_lock_p omp_set_nest_lock_h, omp_nest_lock_t *nest_lock)  {
    //DEBUGPRINT("omp_set_nest_lock %d\n", Tau_get_thread());

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

TAU_GOMP_INLINE void  tau_GOMP_barrier(GOMP_barrier_p GOMP_barrier_h)  {
    DEBUGPRINT("GOMP_barrier %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_EBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_EBAR);

	Tau_global_decr_insideTAU();
    (*GOMP_barrier_h)();
	Tau_global_incr_insideTAU();

    __ompc_event_callback(OMP_EVENT_THR_END_EBAR);
    __ompc_set_state(previous);
}

/**********************************************************
  GOMP_critical_start
 **********************************************************/

TAU_GOMP_INLINE void  tau_GOMP_critical_start(GOMP_critical_start_p GOMP_critical_start_h)  {
    DEBUGPRINT("GOMP_critical_start %d\n", Tau_get_thread());
	Tau_global_decr_insideTAU();
    (*GOMP_critical_start_h)();
	Tau_global_incr_insideTAU();
    //__ompc_set_state(THR_CTWT_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);
}


/**********************************************************
  GOMP_critical_end
 **********************************************************/

TAU_GOMP_INLINE void  tau_GOMP_critical_end(GOMP_critical_end_p GOMP_critical_end_h)  {
    DEBUGPRINT("GOMP_critical_end %d\n", Tau_get_thread());
    //__ompc_event_callback(OMP_EVENT_THR_END_CTWT);
    //__ompc_set_state(THR_WORK_STATE);
	Tau_global_decr_insideTAU();
    (*GOMP_critical_end_h)();
	Tau_global_incr_insideTAU();
}


/**********************************************************
  GOMP_critical_name_start
 **********************************************************/

TAU_GOMP_INLINE void  tau_GOMP_critical_name_start(GOMP_critical_name_start_p GOMP_critical_name_start_h, void ** a1)  {
    DEBUGPRINT("GOMP_critical_name_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    (*GOMP_critical_name_start_h)( a1);
	Tau_global_incr_insideTAU();
    //__ompc_set_state(THR_CTWT_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);

}


/**********************************************************
  GOMP_critical_name_end
 **********************************************************/

TAU_GOMP_INLINE void  tau_GOMP_critical_name_end(GOMP_critical_name_end_p GOMP_critical_name_end_h, void ** a1)  {

    DEBUGPRINT("GOMP_critical_name_end %d\n", Tau_get_thread());

    //__ompc_event_callback(OMP_EVENT_THR_END_CTWT);
    //__ompc_set_state(THR_WORK_STATE);
	Tau_global_decr_insideTAU();
    (*GOMP_critical_name_end_h)( a1);
	Tau_global_incr_insideTAU();
}


/**********************************************************
  GOMP_atomic_start
 **********************************************************/

TAU_GOMP_INLINE void  tau_GOMP_atomic_start(GOMP_atomic_start_p GOMP_atomic_start_h)  {

    DEBUGPRINT("GOMP_atomic_start %d\n", Tau_get_thread());

    __ompc_set_state(THR_ATWT_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ATWT);
	Tau_global_decr_insideTAU();
    (*GOMP_atomic_start_h)();
	Tau_global_incr_insideTAU();
}


/**********************************************************
  GOMP_atomic_end
 **********************************************************/

TAU_GOMP_INLINE void  tau_GOMP_atomic_end(GOMP_atomic_end_p GOMP_atomic_end_h)  {

    DEBUGPRINT("GOMP_atomic_end %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    (*GOMP_atomic_end_h)();
	Tau_global_incr_insideTAU();
    __ompc_event_callback(OMP_EVENT_THR_END_ATWT);
    __ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_loop_static_start
 **********************************************************/

TAU_GOMP_INLINE bool  tau_GOMP_loop_static_start(GOMP_loop_static_start_p GOMP_loop_static_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_static_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();
    return retval;

}


/**********************************************************
  GOMP_loop_dynamic_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_dynamic_start(GOMP_loop_dynamic_start_p GOMP_loop_dynamic_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_dynamic_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();

    return retval;

}


/**********************************************************
  GOMP_loop_guided_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_guided_start(GOMP_loop_guided_start_p GOMP_loop_guided_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_guided_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_runtime_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_runtime_start(GOMP_loop_runtime_start_p GOMP_loop_runtime_start_h, long a1, long a2, long a3, long * a4, long * a5)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_runtime_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ordered_static_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_static_start(GOMP_loop_ordered_static_start_p GOMP_loop_ordered_static_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_static_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();

    //Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth] = 1;
    _ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_dynamic_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_dynamic_start(GOMP_loop_ordered_dynamic_start_p GOMP_loop_ordered_dynamic_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_dynamic_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();

    //Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth] = 1;
    _ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);
    return retval;

}


/**********************************************************
  GOMP_loop_ordered_guided_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_guided_start(GOMP_loop_ordered_guided_start_p GOMP_loop_ordered_guided_start_h, long a1, long a2, long a3, long a4, long * a5, long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_guided_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();

    //Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth] = 1;
    _ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    return retval;

}


/**********************************************************
  GOMP_loop_ordered_runtime_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_runtime_start(GOMP_loop_ordered_runtime_start_p GOMP_loop_ordered_runtime_start_h, long a1, long a2, long a3, long * a4, long * a5)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_runtime_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5);
	Tau_global_incr_insideTAU();

    //Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth] = 1;
    _ordered[_depth] = 1;
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);

    return retval;

}


/**********************************************************
  GOMP_loop_static_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_static_next(GOMP_loop_static_next_p GOMP_loop_static_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_static_next %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_static_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();
    return retval;

}


/**********************************************************
  GOMP_loop_dynamic_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_dynamic_next(GOMP_loop_dynamic_next_p GOMP_loop_dynamic_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_dynamic_next %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_dynamic_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_guided_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_guided_next(GOMP_loop_guided_next_p GOMP_loop_guided_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_guided_next %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_guided_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_runtime_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_runtime_next(GOMP_loop_runtime_next_p GOMP_loop_runtime_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_runtime_next %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_runtime_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ordered_static_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_static_next(GOMP_loop_ordered_static_next_p GOMP_loop_ordered_static_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_static_next %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_static_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);


    return retval;

}


/**********************************************************
  GOMP_loop_ordered_dynamic_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_dynamic_next(GOMP_loop_ordered_dynamic_next_p GOMP_loop_ordered_dynamic_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_dynamic_next %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
	previous = __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_dynamic_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);


    return retval;

}


/**********************************************************
  GOMP_loop_ordered_guided_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_guided_next(GOMP_loop_ordered_guided_next_p GOMP_loop_ordered_guided_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_guided_next %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
	previous = __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_guided_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);


    return retval;

}


/**********************************************************
  GOMP_loop_ordered_runtime_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ordered_runtime_next(GOMP_loop_ordered_runtime_next_p GOMP_loop_ordered_runtime_next_h, long * a1, long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ordered_runtime_next %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_WORK_STATE);
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ordered_runtime_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);


    return retval;

}


/**********************************************************
  GOMP_parallel_loop_static_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_loop_static_start(GOMP_parallel_loop_static_start_p GOMP_parallel_loop_static_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

    DEBUGPRINT("GOMP_parallel_loop_static_start %d\n", Tau_get_thread());

    //int tid = Tau_get_thread();
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
	//Tau_global_decr_insideTAU();
    //(*GOMP_parallel_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	//Tau_global_incr_insideTAU();
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
	proxy->buf = NULL;
	Tau_global_decr_insideTAU();
    (*GOMP_parallel_loop_static_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();
    // save the pointer so we can free it later
    //Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    _proxy[_depth] = proxy;
    //Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    _depth = _depth + 1;


}


/**********************************************************
  GOMP_parallel_loop_dynamic_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_loop_dynamic_start(GOMP_parallel_loop_dynamic_start_p GOMP_parallel_loop_dynamic_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

    //DEBUGPRINT("GOMP_parallel_loop_dynamic_start %d\n", Tau_get_thread());

    //int tid = Tau_get_thread();
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
	//Tau_global_decr_insideTAU();
    //(*GOMP_parallel_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	//Tau_global_incr_insideTAU();
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
	proxy->buf = NULL;
	Tau_global_decr_insideTAU();
    (*GOMP_parallel_loop_dynamic_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();
    // save the pointer so we can free it later
    //Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    //Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    _proxy[_depth] = proxy;
    _depth = _depth + 1;


}


/**********************************************************
  GOMP_parallel_loop_guided_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_loop_guided_start(GOMP_parallel_loop_guided_start_p GOMP_parallel_loop_guided_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

    DEBUGPRINT("GOMP_parallel_loop_guided_start %d\n", Tau_get_thread());

    //int tid = Tau_get_thread();
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
	//Tau_global_decr_insideTAU();
    //(*GOMP_parallel_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	//Tau_global_incr_insideTAU();
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->buf = NULL;
	Tau_global_decr_insideTAU();
    (*GOMP_parallel_loop_guided_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();
    // save the pointer so we can free it later
    //Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    //Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    _proxy[_depth] = proxy;
    _depth = _depth + 1;


}


/**********************************************************
  GOMP_parallel_loop_runtime_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_loop_runtime_start(GOMP_parallel_loop_runtime_start_p GOMP_parallel_loop_runtime_start_h, void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6)  {

    DEBUGPRINT("GOMP_parallel_loop_runtime_start %d\n", Tau_get_thread());

    //int tid = Tau_get_thread();
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
	//Tau_global_decr_insideTAU();
    //(*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	//Tau_global_incr_insideTAU();
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->buf = NULL;
	Tau_global_decr_insideTAU();
    (*GOMP_parallel_loop_runtime_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();
    // save the pointer so we can free it later
    //Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    //Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
    _proxy[_depth] = proxy;
    _depth = _depth + 1;


}


/**********************************************************
  GOMP_loop_end
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_loop_end(GOMP_loop_end_p GOMP_loop_end_h)  {

    DEBUGPRINT("GOMP_loop_end %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
    if (_ordered[_depth]) {
    //if (Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth]) {
        //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
        __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
        //Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth] = 0;
        _ordered[_depth] = 0;
    }
    previous = __ompc_set_state(THR_IBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IBAR);

	Tau_global_decr_insideTAU();
    (*GOMP_loop_end_h)();
	Tau_global_incr_insideTAU();

    __ompc_event_callback(OMP_EVENT_THR_END_IBAR);
    __ompc_set_state(previous);

}


/**********************************************************
  GOMP_loop_end_nowait
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_loop_end_nowait(GOMP_loop_end_nowait_p GOMP_loop_end_nowait_h)  {

    DEBUGPRINT("GOMP_loop_end_nowait %d\n", Tau_get_thread());

    //if (Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth]) {
    if (_ordered[_depth])  {
        //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
        __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
        //Tau_gomp_flags[Tau_get_thread()].ordered[Tau_gomp_flags[Tau_get_thread()].depth] = 0;
        _ordered[_depth] = 0;
    }

	Tau_global_decr_insideTAU();
    (*GOMP_loop_end_nowait_h)();
	Tau_global_incr_insideTAU();

    //__ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_loop_ull_static_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_static_start(GOMP_loop_ull_static_start_p GOMP_loop_ull_static_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_static_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();

    return retval;

}


/**********************************************************
  GOMP_loop_ull_dynamic_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_dynamic_start(GOMP_loop_ull_dynamic_start_p GOMP_loop_ull_dynamic_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_dynamic_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_guided_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_guided_start(GOMP_loop_ull_guided_start_p GOMP_loop_ull_guided_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_guided_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_runtime_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_runtime_start(GOMP_loop_ull_runtime_start_p GOMP_loop_ull_runtime_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_runtime_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_static_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_static_start(GOMP_loop_ull_ordered_static_start_p GOMP_loop_ull_ordered_static_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_static_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_dynamic_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_dynamic_start(GOMP_loop_ull_ordered_dynamic_start_p GOMP_loop_ull_ordered_dynamic_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_guided_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_guided_start(GOMP_loop_ull_ordered_guided_start_p GOMP_loop_ull_ordered_guided_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_guided_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_runtime_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_runtime_start(GOMP_loop_ull_ordered_runtime_start_p GOMP_loop_ull_ordered_runtime_start_h, bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_runtime_start %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_static_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_static_next(GOMP_loop_ull_static_next_p GOMP_loop_ull_static_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_static_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_static_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_dynamic_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_dynamic_next(GOMP_loop_ull_dynamic_next_p GOMP_loop_ull_dynamic_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_dynamic_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_dynamic_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_guided_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_guided_next(GOMP_loop_ull_guided_next_p GOMP_loop_ull_guided_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_guided_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_guided_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_runtime_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_runtime_next(GOMP_loop_ull_runtime_next_p GOMP_loop_ull_runtime_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_runtime_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_runtime_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_static_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_static_next(GOMP_loop_ull_ordered_static_next_p GOMP_loop_ull_ordered_static_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_static_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_static_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_dynamic_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_dynamic_next(GOMP_loop_ull_ordered_dynamic_next_p GOMP_loop_ull_ordered_dynamic_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_dynamic_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_guided_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_guided_next(GOMP_loop_ull_ordered_guided_next_p GOMP_loop_ull_ordered_guided_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_guided_next %d\n", Tau_get_thread());


	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_guided_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();


    return retval;

}


/**********************************************************
  GOMP_loop_ull_ordered_runtime_next
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_loop_ull_ordered_runtime_next(GOMP_loop_ull_ordered_runtime_next_p GOMP_loop_ull_ordered_runtime_next_h, unsigned long long * a1, unsigned long long * a2)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_loop_ull_ordered_runtime_next %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_loop_ull_ordered_runtime_next_h)( a1,  a2);
	Tau_global_incr_insideTAU();
    return retval;

}


/**********************************************************
  GOMP_ordered_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_ordered_start(GOMP_ordered_start_p GOMP_ordered_start_h)  {

    DEBUGPRINT("GOMP_ordered_start %d\n", Tau_get_thread());

    // our turn to work in the ordered region!
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
    __ompc_set_state(THR_WORK_STATE);
	Tau_global_decr_insideTAU();
    (*GOMP_ordered_start_h)();
	Tau_global_incr_insideTAU();

}


/**********************************************************
  GOMP_ordered_end
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_ordered_end(GOMP_ordered_end_p GOMP_ordered_end_h)  {

    DEBUGPRINT("GOMP_ordered_end %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    (*GOMP_ordered_end_h)();
	Tau_global_incr_insideTAU();
    // we can't do this, or we might get overlapping timers.
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
    __ompc_set_state(THR_ODWT_STATE);
}

/**********************************************************
  GOMP_parallel_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_start(GOMP_parallel_start_p GOMP_parallel_start_h, void (*a1)(void *), void * a2, unsigned int a3)  {
    DEBUGPRINT("GOMP_parallel_start %d of %d\n", Tau_get_thread(), a3==0?omp_get_max_threads():1);

    // increment the region counter
	incr_current_region_id();
    __ompc_set_state(THR_OVHD_STATE);
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->buf = NULL;
    // save the pointer so we can free it later
    _proxy[_depth] = proxy;
    _depth = _depth + 1;
	// do this now, so we can get the _proxy address.
    __ompc_event_callback(OMP_EVENT_FORK);

    // time the call
	Tau_global_decr_insideTAU();
    (*GOMP_parallel_start_h)( &Tau_gomp_parallel_start_proxy, proxy,  a3);
	Tau_global_incr_insideTAU();
    __ompc_set_state(THR_WORK_STATE);

    DEBUGPRINT("GOMP_parallel_start %d of %d (on exit)\n", Tau_get_thread(), omp_get_num_threads());

}


/**********************************************************
  GOMP_parallel_end
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_end(GOMP_parallel_end_p GOMP_parallel_end_h)  {

    DEBUGPRINT("GOMP_parallel_end %d of %d\n", Tau_get_thread(), omp_get_num_threads());

    __ompc_set_state(THR_IBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IBAR);
	Tau_global_decr_insideTAU();
    (*GOMP_parallel_end_h)();
	Tau_global_incr_insideTAU();
    // do this at the end, so we can join all the threads.
    __ompc_event_callback(OMP_EVENT_THR_END_IBAR);
    __ompc_event_callback(OMP_EVENT_JOIN);
    __ompc_set_state(THR_SERIAL_STATE);
    // free the proxy wrapper, and reduce the depth
    int depth = _depth - 1;
    if (depth >= 0 && _proxy[depth] != NULL) {
        TAU_GOMP_PROXY_WRAPPER * proxy = _proxy[depth];
		//TAU_VERBOSE("1 Thread %d freeing proxy pointer %p\n", Tau_get_thread(), proxy); fflush(stderr);
        free(proxy);
        _proxy[depth] = NULL;
        _depth = depth;
    } else {
        // assume the worst...
        _depth = 0;
        _proxy[0] = NULL;
    }

}


/**********************************************************
  GOMP_task
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_task(GOMP_task_p GOMP_task_h, void (*a1)(void *), void * a2, void (*a3)(void *, void *), long a4, long a5, bool a6, unsigned int a7)  {

    DEBUGPRINT("GOMP_task %d\n", Tau_get_thread());

    //int tid = Tau_get_thread();

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
	//Tau_global_decr_insideTAU();
    //(*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
	//Tau_global_incr_insideTAU();
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->buf = NULL;
    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_TASK_CREATE_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_CREATE_TASK);
	if (a3 != NULL) {
      /*proxy->buf = (char*)(malloc(a4 + a5 - 1));
      proxy->a2 = (char *) (((uintptr_t) proxy->buf + a5 - 1)
                          & ~(uintptr_t) (a5 - 1));
      a3(proxy->buf, proxy->a2);
	  */
	  Tau_global_decr_insideTAU();
      (*GOMP_task_h)( a1, a2, a3, a4, a5, a6, a7);
	  Tau_global_incr_insideTAU();
	} else {
	  Tau_global_decr_insideTAU();
      (*GOMP_task_h)( Tau_gomp_task_proxy, proxy,  NULL,  a4,  a5,  a6, a7);
	  Tau_global_incr_insideTAU();
	}
    __ompc_event_callback(OMP_EVENT_THR_END_CREATE_TASK_IMM);
    __ompc_set_state(previous);
#else
    /* just call the task creation, for now */
    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_TASK_CREATE_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_CREATE_TASK);
	Tau_global_decr_insideTAU();
    (*GOMP_task_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
	Tau_global_incr_insideTAU();
    __ompc_event_callback(OMP_EVENT_THR_END_CREATE_TASK_IMM);
    //__ompc_set_state(THR_TASK_SCHEDULE_STATE);
    __ompc_set_state(previous);
#endif

}


/**********************************************************
  GOMP_taskwait
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_taskwait(GOMP_taskwait_p GOMP_taskwait_h)  {

    DEBUGPRINT("GOMP_taskwait %d\n", Tau_get_thread());

    OMP_COLLECTOR_API_THR_STATE previous;
    previous = __ompc_set_state(THR_IBAR_STATE);
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_IBAR);
	Tau_global_decr_insideTAU();
    (*GOMP_taskwait_h)();
	Tau_global_incr_insideTAU();
    __ompc_event_callback(OMP_EVENT_THR_END_IBAR);
    __ompc_set_state(previous);

}


/**********************************************************
  GOMP_taskyield - only exists in gcc 4.7 and greater, and only as a stub

  TAU_GOMP_INLINE void tau_GOMP_taskyield(GOMP_taskyield_p GOMP_taskyield_h)  {

  DEBUGPRINT("GOMP_taskyield %d\n", Tau_get_thread());

  Tau_global_decr_insideTAU();
  (*GOMP_taskyield_h)();
  Tau_global_incr_insideTAU();

  }
 **********************************************************/


/**********************************************************
  GOMP_sections_start
 **********************************************************/

TAU_GOMP_INLINE unsigned int tau_GOMP_sections_start(GOMP_sections_start_p GOMP_sections_start_h, unsigned int a1)  {

    unsigned int retval = 0;
    DEBUGPRINT("GOMP_sections_start %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_sections_start_h)( a1);
	Tau_global_incr_insideTAU();
    return retval;

}


/**********************************************************
  GOMP_sections_next
 **********************************************************/

TAU_GOMP_INLINE unsigned int tau_GOMP_sections_next(GOMP_sections_next_p GOMP_sections_next_h)  {

    unsigned int retval = 0;
    DEBUGPRINT("GOMP_sections_next %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_sections_next_h)();
	Tau_global_incr_insideTAU();
    return retval;

}


/**********************************************************
  GOMP_parallel_sections_start
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_parallel_sections_start(GOMP_parallel_sections_start_p GOMP_parallel_sections_start_h, void (*a1)(void *), void * a2, unsigned int a3, unsigned int a4)  {

    DEBUGPRINT("GOMP_parallel_sections_start %d\n", Tau_get_thread());

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    proxy->buf = NULL;
    // save the pointer so we can free it later
    _proxy[_depth] = proxy;
    _depth = _depth + 1;

	Tau_global_decr_insideTAU();
    //(*GOMP_parallel_sections_start_h)( a1,  a2,  a3,  a4);
    (*GOMP_parallel_sections_start_h)( &Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4);
	Tau_global_incr_insideTAU();
    // do this AFTER the start, so we know how many threads there are.
    __ompc_event_callback(OMP_EVENT_FORK);
    __ompc_set_state(THR_WORK_STATE);

}


/**********************************************************
  GOMP_sections_end
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_sections_end(GOMP_sections_end_p GOMP_sections_end_h)  {

    DEBUGPRINT("GOMP_sections_end %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    (*GOMP_sections_end_h)();
	Tau_global_incr_insideTAU();

    // free the proxy wrapper, and reduce the depth
    int depth = _depth - 1;
    if (depth >= 0 && _proxy[depth] != NULL) {
        TAU_GOMP_PROXY_WRAPPER * proxy = _proxy[depth];
		//TAU_VERBOSE("2 Thread %d freeing proxy pointer %p\n", Tau_get_thread(), proxy); fflush(stderr);
        free(proxy);
        _proxy[depth] = NULL;
        _depth = depth;
    } else {
        // assume the worst...
        _depth = 0;
        _proxy[0] = NULL;
    }

}


/**********************************************************
  GOMP_sections_end_nowait
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_sections_end_nowait(GOMP_sections_end_nowait_p GOMP_sections_end_nowait_h)  {

    DEBUGPRINT("GOMP_sections_end_nowait %d\n", Tau_get_thread());

	Tau_global_decr_insideTAU();
    (*GOMP_sections_end_nowait_h)();
	Tau_global_incr_insideTAU();

    // IS THIS RISKY? COULD WE FREE THE PROXY POINTER BEFORE THE OTHER THREADS ARE DONE?

    // free the proxy wrapper, and reduce the depth
    int depth = _depth - 1;
    if (depth >= 0 && _proxy[depth] != NULL) {
        TAU_GOMP_PROXY_WRAPPER * proxy = _proxy[depth];
		//TAU_VERBOSE("3 Thread %d freeing proxy pointer %p\n", Tau_get_thread(), proxy); fflush(stderr);
		//
		//Don't actually free the pointer - because this is a "no wait" situation, some
		//other thread may need this pointer. Unfortunately, this will be a tiny memory
		//leak...
		//
        //free(proxy);
        _proxy[depth] = NULL;
        _depth = depth;
    } else {
        // assume the worst...
        _depth = 0;
        _proxy[0] = NULL;
    }

}


/**********************************************************
  GOMP_single_start
 **********************************************************/

TAU_GOMP_INLINE bool tau_GOMP_single_start(GOMP_single_start_p GOMP_single_start_h)  {

    bool retval = 0;
    DEBUGPRINT("GOMP_single_start %d\n", Tau_get_thread());

    // in this case, the single region is entered and executed
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_single_start_h)();
	Tau_global_incr_insideTAU();
    // the code is done, so fire the callback end event
    __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);

    return retval;

}


/**********************************************************
  GOMP_single_copy_start
 **********************************************************/

TAU_GOMP_INLINE void * tau_GOMP_single_copy_start(GOMP_single_copy_start_p GOMP_single_copy_start_h)  {

    void * retval = 0;
    DEBUGPRINT("GOMP_single_copy_start %d\n", Tau_get_thread());

    __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
	Tau_global_decr_insideTAU();
    retval  =  (*GOMP_single_copy_start_h)();
	Tau_global_incr_insideTAU();
    return retval;

}


/**********************************************************
  GOMP_single_copy_end
 **********************************************************/

TAU_GOMP_INLINE void tau_GOMP_single_copy_end(GOMP_single_copy_end_p GOMP_single_copy_end_h, void * a1)  {

    DEBUGPRINT("GOMP_single_copy_end %d\n", Tau_get_thread());

    __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);
	Tau_global_decr_insideTAU();
    (*GOMP_single_copy_end_h)( a1);
	Tau_global_incr_insideTAU();

}

