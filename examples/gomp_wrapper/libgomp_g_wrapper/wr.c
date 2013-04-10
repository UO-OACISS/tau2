#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <libgomp_g.h>
#include <Profile/Profiler.h>
#include <stdio.h>

#include <dlfcn.h>
#include "omp_collector_util.h"
#include <stdlib.h>

#include "gomp_wrapper_types.h"

#if 0
#define DEBUGPRINT(format, args...) \
{ printf(format, ## args); fflush(stdout); }
#else
#define DEBUGPRINT(format, args...) \
{ }
#endif

#define RESET_DLERROR() dlerror()
#define CHECK_DLERROR() { \
  char const * err = dlerror(); \
  if (err) { \
    printf("Error getting %s handle: %s\n", name, err); \
    fflush(stdout); \
    exit(1); \
  } \
}

static void * get_system_function_handle(char const * name, void * caller)
{
  char const * err;
  void * handle;

  // Reset error pointer
  RESET_DLERROR();

  // Attempt to get the function handle
  handle = dlsym(RTLD_NEXT, name);

  // Detect errors
  CHECK_DLERROR();

  // Prevent recursion if more than one wrapping approach has been loaded.
  // This happens because we support wrapping pthreads three ways at once:
  // #defines in Profiler.h, -Wl,-wrap on the link line, and LD_PRELOAD.
  if (handle == caller) {
    RESET_DLERROR();
    void * syms = dlopen(NULL, RTLD_NOW);
    CHECK_DLERROR();
    do {
      RESET_DLERROR();
      handle = dlsym(syms, name);
      CHECK_DLERROR();
    } while (handle == caller);
  }

  return handle;
}


const char * tau_orig_libname = "libgomp.1.dylib";
static void *tau_handle = NULL;

extern int Tau_global_get_insideTAU();

struct Tau_gomp_wrapper_status_flags {
  int ordered; // 4 bytes
  int critical; // 4 bytes
  int single; // 4 bytes
  int depth; // 4 bytes
  void * proxy[6]; // should be enough?
  char _pad[64-(4*sizeof(int) + 6*sizeof(void*))];
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
//
} TAU_GOMP_PROXY_WRAPPER;

/* This array is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
#if defined __INTEL__COMPILER
__declspec (align(64)) static struct Tau_gomp_wrapper_status_flags Tau_gomp_flags[TAU_MAX_THREADS] = {0};
#elif defined __GNUC__
static struct Tau_gomp_wrapper_status_flags Tau_gomp_flags[TAU_MAX_THREADS] __attribute__ ((aligned(64))) = {0};
#else
static struct Tau_gomp_wrapper_status_flags Tau_gomp_flags[TAU_MAX_THREADS] = {0};
#endif

/* This function is used to wrap the outlined functions for parallel regions.
 */
void Tau_gomp_parallel_start_proxy(void * a2) {
  DEBUGPRINT("Parallel Proxy %d!\n", Tau_get_tid());
  TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(a2);
  __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
  (proxy->a1)(proxy->a2);
  __ompc_event_callback(OMP_EVENT_THR_BEGIN_IDLE);
}

/* This function is used to wrap the outlined functions for tasks.
 */
void Tau_gomp_task_proxy(void * a2) {
  DEBUGPRINT("Task Proxy %d!\n", Tau_get_tid());
  TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(a2);
  //__ompc_event_callback(OMP_EVENT_THR_END_IDLE);
  (proxy->a1)(proxy->a2);
  //__ompc_event_callback(OMP_EVENT_THR_BEGIN_IDLE);
}

/**********************************************************
   GOMP_barrier
 **********************************************************/

void  GOMP_barrier()  {

  static GOMP_barrier_p GOMP_barrier_h = NULL;
  DEBUGPRINT("GOMP_barrier %d\n", Tau_get_tid());

    if (GOMP_barrier_h == NULL) {
	  GOMP_barrier_h = (GOMP_barrier_p)get_system_function_handle("GOMP_barrier",(void*)GOMP_barrier);
	}

    if (Tau_global_get_insideTAU() == 0) { 
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_EBAR);
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    (*GOMP_barrier_h)();

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
      __ompc_event_callback(OMP_EVENT_THR_END_EBAR);
	}
}

/**********************************************************
   GOMP_critical_start
 **********************************************************/

void  GOMP_critical_start()  {

  static GOMP_critical_start_p GOMP_critical_start_h = NULL;
  DEBUGPRINT("GOMP_critical_start %d\n", Tau_get_tid());
  
    if (GOMP_critical_start_h == NULL) {
	  GOMP_critical_start_h = (GOMP_critical_start_p)get_system_function_handle("GOMP_critical_start",(void*)GOMP_critical_start);
	}

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}
	
    (*GOMP_critical_start_h)();
	
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);
	}
}


/**********************************************************
   GOMP_critical_end
 **********************************************************/

void  GOMP_critical_end()  {

  static GOMP_critical_end_p GOMP_critical_end_h = NULL;
  DEBUGPRINT("GOMP_critical_end %d\n", Tau_get_tid());

    if (GOMP_critical_end_h == NULL) {
	  GOMP_critical_end_h = (GOMP_critical_end_p)get_system_function_handle("GOMP_critical_end",(void*)GOMP_critical_end);
	}

    if (Tau_global_get_insideTAU() == 0) { 
      __ompc_event_callback(OMP_EVENT_THR_END_CTWT);
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}
	
    (*GOMP_critical_end_h)();
	
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
	}
}


/**********************************************************
   GOMP_critical_name_start
 **********************************************************/

void  GOMP_critical_name_start(void ** a1)  {

  static GOMP_critical_name_start_p GOMP_critical_name_start_h = NULL;
  DEBUGPRINT("GOMP_critical_name_start %d\n", Tau_get_tid());

    if (GOMP_critical_name_start_h == NULL) {
	  GOMP_critical_name_start_h = (GOMP_critical_name_start_p)get_system_function_handle("GOMP_critical_name_start",(void*)GOMP_critical_name_start);
    }

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	} else { DEBUGPRINT("not measuring TAU\n"); }

    (*GOMP_critical_name_start_h)( a1);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);
	}

}


/**********************************************************
   GOMP_critical_name_end
 **********************************************************/

void  GOMP_critical_name_end(void ** a1)  {

  static GOMP_critical_name_end_p GOMP_critical_name_end_h = NULL;
  DEBUGPRINT("GOMP_critical_name_end %d\n", Tau_get_tid());

    if (GOMP_critical_name_end_h == NULL) {
	  GOMP_critical_name_end_h = (GOMP_critical_name_end_p)get_system_function_handle("GOMP_critical_name_end",(void*)GOMP_critical_name_end);
    }

    if (Tau_global_get_insideTAU() == 0) { 
      __ompc_event_callback(OMP_EVENT_THR_END_CTWT);
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	} else { DEBUGPRINT("not measuring TAU\n"); }

    (*GOMP_critical_name_end_h)( a1);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
	}

}


/**********************************************************
   GOMP_atomic_start
 **********************************************************/

void  GOMP_atomic_start()  {

  static GOMP_atomic_start_p GOMP_atomic_start_h;
  DEBUGPRINT("GOMP_atomic_start %d\n", Tau_get_tid());

    if (GOMP_atomic_start_h == NULL) {
	  GOMP_atomic_start_h = (GOMP_atomic_start_p)get_system_function_handle("GOMP_atomic_start",(void*)GOMP_atomic_start);
    }

    if (Tau_global_get_insideTAU() == 0) {
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    (*GOMP_atomic_start_h)();

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ATWT);
	}

}


/**********************************************************
   GOMP_atomic_end
 **********************************************************/

void  GOMP_atomic_end()  {

  static GOMP_atomic_end_p GOMP_atomic_end_h = NULL;
  DEBUGPRINT("GOMP_atomic_end %d\n", Tau_get_tid());

    if (GOMP_atomic_end_h == NULL) {
	  GOMP_atomic_end_h = (GOMP_atomic_end_p)get_system_function_handle("GOMP_atomic_end",(void*)GOMP_atomic_end);
    }

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    (*GOMP_atomic_end_h)();

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
      __ompc_event_callback(OMP_EVENT_THR_END_ATWT);
	}

}


/**********************************************************
   GOMP_loop_static_start
 **********************************************************/

bool  GOMP_loop_static_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static GOMP_loop_static_start_p GOMP_loop_static_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_static_start %d\n", Tau_get_tid());

    if (GOMP_loop_static_start_h == NULL) {
	  GOMP_loop_static_start_h = (GOMP_loop_static_start_p)get_system_function_handle("GOMP_loop_static_start",(void*)GOMP_loop_static_start);
    }

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	}
  return retval;

}


/**********************************************************
   GOMP_loop_dynamic_start
 **********************************************************/

bool  GOMP_loop_dynamic_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static GOMP_loop_dynamic_start_p GOMP_loop_dynamic_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_dynamic_start %d\n", Tau_get_tid());

    if (GOMP_loop_dynamic_start_h == NULL) {
	  GOMP_loop_dynamic_start_h = (GOMP_loop_dynamic_start_p)get_system_function_handle("GOMP_loop_dynamic_start",(void*)GOMP_loop_dynamic_start);
    }

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	}
  return retval;

}


/**********************************************************
   GOMP_loop_guided_start
 **********************************************************/

bool  GOMP_loop_guided_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static GOMP_loop_guided_start_p GOMP_loop_guided_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_guided_start %d\n", Tau_get_tid());

    if (GOMP_loop_guided_start_h == NULL) {
	  GOMP_loop_guided_start_h = (GOMP_loop_guided_start_p)get_system_function_handle("GOMP_loop_guided_start",(void*)GOMP_loop_guided_start);
	}
    
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	}

  return retval;

}


/**********************************************************
   GOMP_loop_runtime_start
 **********************************************************/

bool  GOMP_loop_runtime_start(long a1, long a2, long a3, long * a4, long * a5)  {

  static GOMP_loop_runtime_start_p GOMP_loop_runtime_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_runtime_start %d\n", Tau_get_tid());

    if (GOMP_loop_runtime_start_h == NULL) {
	  GOMP_loop_runtime_start_h = (GOMP_loop_runtime_start_p)get_system_function_handle("GOMP_loop_runtime_start",(void*)GOMP_loop_runtime_start);
	}

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	}
  
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_static_start
 **********************************************************/

bool  GOMP_loop_ordered_static_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static GOMP_loop_ordered_static_start_p GOMP_loop_ordered_static_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_static_start %d\n", Tau_get_tid());

    if (GOMP_loop_ordered_static_start_h == NULL) {
	  GOMP_loop_ordered_static_start_h = (GOMP_loop_ordered_static_start_p)get_system_function_handle("GOMP_loop_ordered_static_start",(void*)GOMP_loop_ordered_static_start);
	}

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	  Tau_gomp_flags[Tau_get_tid()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	}

  return retval;

}


/**********************************************************
   GOMP_loop_ordered_dynamic_start
 **********************************************************/

bool  GOMP_loop_ordered_dynamic_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static GOMP_loop_ordered_dynamic_start_p GOMP_loop_ordered_dynamic_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_dynamic_start %d\n", Tau_get_tid());

    if (GOMP_loop_ordered_dynamic_start_h == NULL) {
	  GOMP_loop_ordered_dynamic_start_h = (GOMP_loop_ordered_dynamic_start_p)get_system_function_handle("GOMP_loop_ordered_dynamic_start",(void*)GOMP_loop_ordered_dynamic_start);
	}

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	  Tau_gomp_flags[Tau_get_tid()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	}
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_guided_start
 **********************************************************/

bool  GOMP_loop_ordered_guided_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static GOMP_loop_ordered_guided_start_p GOMP_loop_ordered_guided_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_guided_start %d\n", Tau_get_tid());

    if (GOMP_loop_ordered_guided_start_h == NULL) {
	  GOMP_loop_ordered_guided_start_h = (GOMP_loop_ordered_guided_start_p)get_system_function_handle("GOMP_loop_ordered_guided_start",(void*)GOMP_loop_ordered_guided_start);
	}

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}

    retval  =  (*GOMP_loop_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	  Tau_gomp_flags[Tau_get_tid()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	}
  
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_runtime_start
 **********************************************************/

bool  GOMP_loop_ordered_runtime_start(long a1, long a2, long a3, long * a4, long * a5)  {

  static GOMP_loop_ordered_runtime_start_p GOMP_loop_ordered_runtime_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_runtime_start %d\n", Tau_get_tid());

    if (GOMP_loop_ordered_runtime_start_h == NULL) {
	  GOMP_loop_ordered_runtime_start_h = (GOMP_loop_ordered_runtime_start_p)get_system_function_handle("GOMP_loop_ordered_runtime_start",(void*)GOMP_loop_ordered_runtime_start);
	}

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
	}
	
    retval  =  (*GOMP_loop_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5);

    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	  Tau_gomp_flags[Tau_get_tid()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	}
  
  return retval;

}


/**********************************************************
   GOMP_loop_static_next
 **********************************************************/

bool  GOMP_loop_static_next(long * a1, long * a2)  {

  static GOMP_loop_static_next_p GOMP_loop_static_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_static_next %d\n", Tau_get_tid());

    if (GOMP_loop_static_next_h == NULL) {
	  GOMP_loop_static_next_h = (GOMP_loop_static_next_p)get_system_function_handle("GOMP_loop_static_next",(void*)GOMP_loop_static_next);
	}

    if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
    retval  =  (*GOMP_loop_static_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }
  return retval;

}


/**********************************************************
   GOMP_loop_dynamic_next
 **********************************************************/

bool  GOMP_loop_dynamic_next(long * a1, long * a2)  {

  static GOMP_loop_dynamic_next_p GOMP_loop_dynamic_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_dynamic_next %d\n", Tau_get_tid());

  if (GOMP_loop_dynamic_next_h == NULL) {
	GOMP_loop_dynamic_next_h = (GOMP_loop_dynamic_next_p)get_system_function_handle("GOMP_loop_dynamic_next",(void*)GOMP_loop_dynamic_next);
	}

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_dynamic_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }
  
  return retval;

}


/**********************************************************
   GOMP_loop_guided_next
 **********************************************************/

bool  GOMP_loop_guided_next(long * a1, long * a2)  {

  static GOMP_loop_guided_next_p GOMP_loop_guided_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_guided_next %d\n", Tau_get_tid());

  if (GOMP_loop_guided_next_h == NULL) {
	GOMP_loop_guided_next_h = (GOMP_loop_guided_next_p)get_system_function_handle("GOMP_loop_guided_next",(void*)GOMP_loop_guided_next);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_guided_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_loop_runtime_next
 **********************************************************/

bool  GOMP_loop_runtime_next(long * a1, long * a2)  {

  static GOMP_loop_runtime_next_p GOMP_loop_runtime_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_runtime_next %d\n", Tau_get_tid());

  if (GOMP_loop_runtime_next_h == NULL) {
	GOMP_loop_runtime_next_h = (GOMP_loop_runtime_next_p)get_system_function_handle("GOMP_loop_runtime_next",(void*)GOMP_loop_runtime_next);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_runtime_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_loop_ordered_static_next
 **********************************************************/

bool  GOMP_loop_ordered_static_next(long * a1, long * a2)  {

  static GOMP_loop_ordered_static_next_p GOMP_loop_ordered_static_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_static_next %d\n", Tau_get_tid());

  if (GOMP_loop_ordered_static_next_h == NULL) {
	GOMP_loop_ordered_static_next_h = (GOMP_loop_ordered_static_next_p)get_system_function_handle("GOMP_loop_ordered_static_next",(void*)GOMP_loop_ordered_static_next);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_ordered_static_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_loop_ordered_dynamic_next
 **********************************************************/

bool  GOMP_loop_ordered_dynamic_next(long * a1, long * a2)  {

  static GOMP_loop_ordered_dynamic_next_p GOMP_loop_ordered_dynamic_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_dynamic_next %d\n", Tau_get_tid());

  if (GOMP_loop_ordered_dynamic_next_h == NULL) {
	GOMP_loop_ordered_dynamic_next_h = (GOMP_loop_ordered_dynamic_next_p)get_system_function_handle("GOMP_loop_ordered_dynamic_next",(void*)GOMP_loop_ordered_dynamic_next);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_ordered_dynamic_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_loop_ordered_guided_next
 **********************************************************/

bool  GOMP_loop_ordered_guided_next(long * a1, long * a2)  {

  static GOMP_loop_ordered_guided_next_p GOMP_loop_ordered_guided_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_guided_next %d\n", Tau_get_tid());

  if (GOMP_loop_ordered_guided_next_h == NULL) {
	GOMP_loop_ordered_guided_next_h = (GOMP_loop_ordered_guided_next_p)get_system_function_handle("GOMP_loop_ordered_guided_next",(void*)GOMP_loop_ordered_guided_next);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_ordered_guided_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_loop_ordered_runtime_next
 **********************************************************/

bool  GOMP_loop_ordered_runtime_next(long * a1, long * a2)  {

  static GOMP_loop_ordered_runtime_next_p GOMP_loop_ordered_runtime_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ordered_runtime_next %d\n", Tau_get_tid());

  if (GOMP_loop_ordered_runtime_next_h == NULL) {
	GOMP_loop_ordered_runtime_next_h = (GOMP_loop_ordered_runtime_next_p)get_system_function_handle("GOMP_loop_ordered_runtime_next",(void*)GOMP_loop_ordered_runtime_next);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_ordered_runtime_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_parallel_loop_static_start
 **********************************************************/

void  GOMP_parallel_loop_static_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

  static GOMP_parallel_loop_static_start_p GOMP_parallel_loop_static_start_h = NULL;
  DEBUGPRINT("GOMP_parallel_loop_static_start %d\n", Tau_get_tid());

  if (GOMP_parallel_loop_static_start_h == NULL) {
	GOMP_parallel_loop_static_start_h = (GOMP_parallel_loop_static_start_p)get_system_function_handle("GOMP_parallel_loop_static_start",(void*)GOMP_parallel_loop_static_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    int tid = Tau_get_tid();
    Tau_pure_start_task(__FUNCTION__, tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    (*GOMP_parallel_loop_static_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;

    Tau_pure_stop_task(__FUNCTION__, tid); 
  } else {
    (*GOMP_parallel_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  }

}


/**********************************************************
   GOMP_parallel_loop_dynamic_start
 **********************************************************/

void  GOMP_parallel_loop_dynamic_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

  static GOMP_parallel_loop_dynamic_start_p GOMP_parallel_loop_dynamic_start_h = NULL;
  DEBUGPRINT("GOMP_parallel_loop_dynamic_start %d\n", Tau_get_tid());

  if (GOMP_parallel_loop_dynamic_start_h == NULL) {
	GOMP_parallel_loop_dynamic_start_h = (GOMP_parallel_loop_dynamic_start_p)get_system_function_handle("GOMP_parallel_loop_dynamic_start",(void*)GOMP_parallel_loop_dynamic_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    int tid = Tau_get_tid();
    Tau_pure_start_task(__FUNCTION__, tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    (*GOMP_parallel_loop_dynamic_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;

    Tau_pure_stop_task(__FUNCTION__, tid); 
  } else {
    (*GOMP_parallel_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  }

}


/**********************************************************
   GOMP_parallel_loop_guided_start
 **********************************************************/

void  GOMP_parallel_loop_guided_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

  static GOMP_parallel_loop_guided_start_p GOMP_parallel_loop_guided_start_h = NULL;
  DEBUGPRINT("GOMP_parallel_loop_guided_start %d\n", Tau_get_tid());

  if (GOMP_parallel_loop_guided_start_h == NULL) {
	GOMP_parallel_loop_guided_start_h = (GOMP_parallel_loop_guided_start_p)get_system_function_handle("GOMP_parallel_loop_guided_start",(void*)GOMP_parallel_loop_guided_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    int tid = Tau_get_tid();
    Tau_pure_start_task(__FUNCTION__, tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    (*GOMP_parallel_loop_guided_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6,  a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;

    Tau_pure_stop_task(__FUNCTION__, tid); 
  } else {
    (*GOMP_parallel_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  }

}


/**********************************************************
   GOMP_parallel_loop_runtime_start
 **********************************************************/

void  GOMP_parallel_loop_runtime_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6)  {

  static GOMP_parallel_loop_runtime_start_p GOMP_parallel_loop_runtime_start_h = NULL;
  DEBUGPRINT("GOMP_parallel_loop_runtime_start %d\n", Tau_get_tid());

  if (GOMP_parallel_loop_runtime_start_h == NULL) {
	GOMP_parallel_loop_runtime_start_h = (GOMP_parallel_loop_runtime_start_p)get_system_function_handle("GOMP_parallel_loop_runtime_start",(void*)GOMP_parallel_loop_runtime_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    int tid = Tau_get_tid();
    Tau_pure_start_task(__FUNCTION__, tid); 

    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    (*GOMP_parallel_loop_runtime_start_h)( Tau_gomp_parallel_start_proxy,  proxy,  a3,  a4,  a5,  a6);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;

    Tau_pure_stop_task(__FUNCTION__, tid); 
  } else {
    (*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
  }

}


/**********************************************************
   GOMP_loop_end
 **********************************************************/

void  GOMP_loop_end()  {

  static GOMP_loop_end_p GOMP_loop_end_h = NULL;
  DEBUGPRINT("GOMP_loop_end %d\n", Tau_get_tid());

  if (GOMP_loop_end_h == NULL) {
	GOMP_loop_end_h = (GOMP_loop_end_p)get_system_function_handle("GOMP_loop_end",(void*)GOMP_loop_end);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    if (Tau_gomp_flags[Tau_get_tid()].ordered) {
      ////__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
      __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
	  Tau_gomp_flags[Tau_get_tid()].ordered = 0;
	}
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  (*GOMP_loop_end_h)();

  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
  }

}


/**********************************************************
   GOMP_loop_end_nowait
 **********************************************************/

void  GOMP_loop_end_nowait()  {

  static GOMP_loop_end_nowait_p GOMP_loop_end_nowait_h = NULL;
  DEBUGPRINT("GOMP_loop_end_nowait %d\n", Tau_get_tid());

  if (GOMP_loop_end_nowait_h == NULL) {
	GOMP_loop_end_nowait_h = (GOMP_loop_end_nowait_p)get_system_function_handle("GOMP_loop_end_nowait",(void*)GOMP_loop_end_nowait);
  }

  if (Tau_global_get_insideTAU() == 0) { 
	if (Tau_gomp_flags[Tau_get_tid()].ordered) {
      //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
      __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
	  Tau_gomp_flags[Tau_get_tid()].ordered = 0;
	}
	Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  (*GOMP_loop_end_nowait_h)();

  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
  }

}


/**********************************************************
   GOMP_loop_ull_static_start
 **********************************************************/

bool  GOMP_loop_ull_static_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static GOMP_loop_ull_static_start_p GOMP_loop_ull_static_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_static_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_static_start_h == NULL) {
    GOMP_loop_ull_static_start_h = (GOMP_loop_ull_static_start_p)get_system_function_handle("GOMP_loop_ull_static_start",(void*)GOMP_loop_ull_static_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }
  
  retval  =  (*GOMP_loop_ull_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_dynamic_start
 **********************************************************/

bool  GOMP_loop_ull_dynamic_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static GOMP_loop_ull_dynamic_start_p GOMP_loop_ull_dynamic_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_dynamic_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_dynamic_start_h == NULL) {
    GOMP_loop_ull_dynamic_start_h = (GOMP_loop_ull_dynamic_start_p)get_system_function_handle("GOMP_loop_ull_dynamic_start",(void*)GOMP_loop_ull_dynamic_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_ull_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }
  
  return retval;

}


/**********************************************************
   GOMP_loop_ull_guided_start
 **********************************************************/

bool  GOMP_loop_ull_guided_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static GOMP_loop_ull_guided_start_p GOMP_loop_ull_guided_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_guided_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_guided_start_h == NULL) {
    GOMP_loop_ull_guided_start_h = (GOMP_loop_ull_guided_start_p)get_system_function_handle("GOMP_loop_ull_guided_start",(void*)GOMP_loop_ull_guided_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }

  retval  =  (*GOMP_loop_ull_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

  if (Tau_global_get_insideTAU() == 0) { 
    Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_runtime_start
 **********************************************************/

bool  GOMP_loop_ull_runtime_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

  static GOMP_loop_ull_runtime_start_p GOMP_loop_ull_runtime_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_runtime_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_runtime_start_h == NULL) {
    GOMP_loop_ull_runtime_start_h = (GOMP_loop_ull_runtime_start_p)get_system_function_handle("GOMP_loop_ull_runtime_start",(void*)GOMP_loop_ull_runtime_start);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_static_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_static_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static GOMP_loop_ull_ordered_static_start_p GOMP_loop_ull_ordered_static_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_static_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_static_start_h == NULL) {
    GOMP_loop_ull_ordered_static_start_h = (GOMP_loop_ull_ordered_static_start_p)get_system_function_handle("GOMP_loop_ull_ordered_static_start",(void*)GOMP_loop_ull_ordered_static_start);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_dynamic_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_dynamic_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static GOMP_loop_ull_ordered_dynamic_start_p GOMP_loop_ull_ordered_dynamic_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_dynamic_start_h == NULL) {
    GOMP_loop_ull_ordered_dynamic_start_h = (GOMP_loop_ull_ordered_dynamic_start_p)get_system_function_handle("GOMP_loop_ull_ordered_dynamic_start",(void*)GOMP_loop_ull_ordered_dynamic_start);
  }

    if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

    retval  =  (*GOMP_loop_ull_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_guided_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_guided_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static GOMP_loop_ull_ordered_guided_start_p GOMP_loop_ull_ordered_guided_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_guided_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_guided_start_h == NULL) {
    GOMP_loop_ull_ordered_guided_start_h = (GOMP_loop_ull_ordered_guided_start_p)get_system_function_handle("GOMP_loop_ull_ordered_guided_start",(void*)GOMP_loop_ull_ordered_guided_start);
  }

    if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

    retval  =  (*GOMP_loop_ull_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);

    if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_runtime_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_runtime_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

  static GOMP_loop_ull_ordered_runtime_start_p GOMP_loop_ull_ordered_runtime_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_runtime_start %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_runtime_start_h == NULL) {
    GOMP_loop_ull_ordered_runtime_start_h = (GOMP_loop_ull_ordered_runtime_start_p)get_system_function_handle("GOMP_loop_ull_ordered_runtime_start",(void*)GOMP_loop_ull_ordered_runtime_start);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_static_next
 **********************************************************/

bool  GOMP_loop_ull_static_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_static_next_p GOMP_loop_ull_static_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_static_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_static_next_h == NULL) {
    GOMP_loop_ull_static_next_h = (GOMP_loop_ull_static_next_p)get_system_function_handle("GOMP_loop_ull_static_next",(void*)GOMP_loop_ull_static_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_static_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_dynamic_next
 **********************************************************/

bool  GOMP_loop_ull_dynamic_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_dynamic_next_p GOMP_loop_ull_dynamic_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_dynamic_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_dynamic_next_h == NULL) {
    GOMP_loop_ull_dynamic_next_h = (GOMP_loop_ull_dynamic_next_p)get_system_function_handle("GOMP_loop_ull_dynamic_next",(void*)GOMP_loop_ull_dynamic_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_dynamic_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_guided_next
 **********************************************************/

bool  GOMP_loop_ull_guided_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_guided_next_p GOMP_loop_ull_guided_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_guided_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_guided_next_h == NULL) {
    GOMP_loop_ull_guided_next_h = (GOMP_loop_ull_guided_next_p)get_system_function_handle("GOMP_loop_ull_guided_next",(void*)GOMP_loop_ull_guided_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_guided_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_runtime_next
 **********************************************************/

bool  GOMP_loop_ull_runtime_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_runtime_next_p GOMP_loop_ull_runtime_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_runtime_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_runtime_next_h == NULL) {
    GOMP_loop_ull_runtime_next_h = (GOMP_loop_ull_runtime_next_p)get_system_function_handle("GOMP_loop_ull_runtime_next",(void*)GOMP_loop_ull_runtime_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_runtime_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_static_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_static_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_ordered_static_next_p GOMP_loop_ull_ordered_static_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_static_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_static_next_h == NULL) {
    GOMP_loop_ull_ordered_static_next_h = (GOMP_loop_ull_ordered_static_next_p)get_system_function_handle("GOMP_loop_ull_ordered_static_next",(void*)GOMP_loop_ull_ordered_static_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_ordered_static_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_dynamic_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_dynamic_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_ordered_dynamic_next_p GOMP_loop_ull_ordered_dynamic_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_dynamic_next_h == NULL) {
    GOMP_loop_ull_ordered_dynamic_next_h = (GOMP_loop_ull_ordered_dynamic_next_p)get_system_function_handle("GOMP_loop_ull_ordered_dynamic_next",(void*)GOMP_loop_ull_ordered_dynamic_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_ordered_dynamic_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_guided_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_guided_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_ordered_guided_next_p GOMP_loop_ull_ordered_guided_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_guided_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_guided_next_h == NULL) {
    GOMP_loop_ull_ordered_guided_next_h = (GOMP_loop_ull_ordered_guided_next_p)get_system_function_handle("GOMP_loop_ull_ordered_guided_next",(void*)GOMP_loop_ull_ordered_guided_next);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }

  retval  =  (*GOMP_loop_ull_ordered_guided_next_h)( a1,  a2);

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_runtime_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_runtime_next(unsigned long long * a1, unsigned long long * a2)  {

  static GOMP_loop_ull_ordered_runtime_next_p GOMP_loop_ull_ordered_runtime_next_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_loop_ull_ordered_runtime_next %d\n", Tau_get_tid());

  if (GOMP_loop_ull_ordered_runtime_next_h == NULL) {
    GOMP_loop_ull_ordered_runtime_next_h = (GOMP_loop_ull_ordered_runtime_next_p)get_system_function_handle("GOMP_loop_ull_ordered_runtime_next",(void*)GOMP_loop_ull_ordered_runtime_next);
  }
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  retval  =  (*GOMP_loop_ull_ordered_runtime_next_h)( a1,  a2);
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }
  return retval;

}


/**********************************************************
   GOMP_ordered_start
 **********************************************************/

void  GOMP_ordered_start()  {

  static GOMP_ordered_start_p GOMP_ordered_start_h = NULL;
  DEBUGPRINT("GOMP_ordered_start %d\n", Tau_get_tid());

  if (GOMP_ordered_start_h == NULL) {
    GOMP_ordered_start_h = (GOMP_ordered_start_p)get_system_function_handle("GOMP_ordered_start",(void*)GOMP_ordered_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
	// our turn to work in the ordered region!
    //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
	Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }
  (*GOMP_ordered_start_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

}


/**********************************************************
   GOMP_ordered_end
 **********************************************************/

void  GOMP_ordered_end()  {

  static GOMP_ordered_end_p GOMP_ordered_end_h = NULL;
  DEBUGPRINT("GOMP_ordered_end %d\n", Tau_get_tid());

  if (GOMP_ordered_end_h == NULL) {
    GOMP_ordered_end_h = (GOMP_ordered_end_p)get_system_function_handle("GOMP_ordered_end",(void*)GOMP_ordered_end);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  (*GOMP_ordered_end_h)();
  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_stop_task(__FUNCTION__, Tau_get_tid());
	// wait for those after us to handle the ordered region...
    //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
  }

}

/**********************************************************
   GOMP_parallel_start
 **********************************************************/

void  GOMP_parallel_start(void (*a1)(void *), void * a2, unsigned int a3)  {
  static GOMP_parallel_start_p GOMP_parallel_start_h = NULL;
  int numThreads = a3 == 0 ? omp_get_max_threads() : 1;
  DEBUGPRINT("GOMP_parallel_start %d of %d\n", Tau_get_tid(), numThreads);

    if (!GOMP_parallel_start_h) {
      GOMP_parallel_start_h = (GOMP_parallel_start_p)get_system_function_handle("GOMP_parallel_start",(void*)GOMP_parallel_start); 
	}
    if (Tau_global_get_insideTAU() == 0) { 
      __ompc_event_callback(OMP_EVENT_FORK);
	  int tid = Tau_get_tid();
	  /* 
	   * Don't actually pass in the work for the parallel region, but a pointer
	   * to our proxy function with the data for the parallel region outlined function.
	   */
      TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
      proxy->a1 = a1;
      proxy->a2 = a2;
	  // save the pointer so we can free it later
	  Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
	  Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;

      // time the call
      Tau_pure_start_task(__FUNCTION__, tid); 
      (*GOMP_parallel_start_h)( &Tau_gomp_parallel_start_proxy, proxy,  a3);
	  Tau_pure_stop_task(__FUNCTION__, tid); 

      DEBUGPRINT("GOMP_parallel_start %d of %d (on exit)\n", Tau_get_tid(), omp_get_num_threads());
	} else {
      (*GOMP_parallel_start_h)(a1,  a2,  a3);
	}

}


/**********************************************************
   GOMP_parallel_end
 **********************************************************/

void  GOMP_parallel_end()  {

  static GOMP_parallel_end_p GOMP_parallel_end_h = NULL;
  DEBUGPRINT("GOMP_parallel_end %d of %d\n", Tau_get_tid(), omp_get_num_threads());

  if (GOMP_parallel_end_h == NULL) {
    GOMP_parallel_end_h = (GOMP_parallel_end_p)get_system_function_handle("GOMP_parallel_end",(void*)GOMP_parallel_end);
  }

  if (Tau_global_get_insideTAU() == 0) { 
    int tid = Tau_get_tid();
	Tau_pure_start_task(__FUNCTION__, tid); 
    (*GOMP_parallel_end_h)();
	Tau_pure_stop_task(__FUNCTION__, tid); 
	// do this at the end, so we can join all the threads.
    __ompc_event_callback(OMP_EVENT_JOIN);
	// free the proxy wrapper, and reduce the depth
	if (Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] != NULL) {
	  free(Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth]);
	  Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = NULL;
	  Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth - 1;
	} else {
	  // assume the worst...
	  Tau_gomp_flags[tid].depth = 0;
	  Tau_gomp_flags[tid].proxy[0] = NULL;
	}
  } else {
    (*GOMP_parallel_end_h)();
  }

}


/**********************************************************
   GOMP_task
 **********************************************************/

void  GOMP_task(void (*a1)(void *), void * a2, void (*a3)(void *, void *), long a4, long a5, bool a6, unsigned int a7)  {

  static GOMP_task_p GOMP_task_h = NULL;
  DEBUGPRINT("GOMP_task %d\n", Tau_get_tid());

  if (GOMP_task_h == NULL) {
    GOMP_task_h = (GOMP_task_p)get_system_function_handle("GOMP_task",(void*)GOMP_task);
  }

  if (Tau_global_get_insideTAU() == 0) {
    int tid = Tau_get_tid();
    Tau_pure_start_task(__FUNCTION__, tid);

#if 0
    /* 
     * Don't actually pass in the work for the parallel region, but a pointer
     * to our proxy function with the data for the parallel region outlined function.
     */
    //(*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    TAU_GOMP_PROXY_WRAPPER * proxy = (TAU_GOMP_PROXY_WRAPPER*)(malloc(sizeof(TAU_GOMP_PROXY_WRAPPER)));
    proxy->a1 = a1;
    proxy->a2 = a2;
    (*GOMP_task_h)( Tau_gomp_task_proxy,  proxy,  a3,  a4,  a5,  a6, a7);
    // save the pointer so we can free it later
    Tau_gomp_flags[tid].proxy[Tau_gomp_flags[tid].depth] = proxy;
    Tau_gomp_flags[tid].depth = Tau_gomp_flags[tid].depth + 1;
#else
    (*GOMP_task_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
#endif
    Tau_pure_stop_task(__FUNCTION__, tid);
  } else {
    (*GOMP_task_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  }

}


/**********************************************************
   GOMP_taskwait
 **********************************************************/

void  GOMP_taskwait()  {

  static GOMP_taskwait_p GOMP_taskwait_h = NULL;
  DEBUGPRINT("GOMP_taskwait %d\n", Tau_get_tid());

  if (GOMP_taskwait_h == NULL) {
    GOMP_taskwait_h = (GOMP_taskwait_p)get_system_function_handle("GOMP_taskwait",(void*)GOMP_taskwait);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  (*GOMP_taskwait_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

}


/**********************************************************
   GOMP_taskyield
 **********************************************************/

void  GOMP_taskyield()  {

  static GOMP_taskyield_p GOMP_taskyield_h = NULL;
  DEBUGPRINT("GOMP_taskyield %d\n", Tau_get_tid());

  if (GOMP_taskyield_h == NULL) {
    GOMP_taskyield_h = (GOMP_taskyield_p)get_system_function_handle("GOMP_taskyield",(void*)GOMP_taskyield);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  (*GOMP_taskyield_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

}


/**********************************************************
   GOMP_sections_start
 **********************************************************/

unsigned int  GOMP_sections_start(unsigned int a1)  {

  static GOMP_sections_start_p GOMP_sections_start_h = NULL;
  unsigned int retval = 0;
  DEBUGPRINT("GOMP_sections_start %d\n", Tau_get_tid());

  if (GOMP_sections_start_h == NULL) {
    GOMP_sections_start_h = (GOMP_sections_start_p)get_system_function_handle("GOMP_sections_start",(void*)GOMP_sections_start);
  }
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  retval  =  (*GOMP_sections_start_h)( a1);
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }
  return retval;

}


/**********************************************************
   GOMP_sections_next
 **********************************************************/

unsigned int  GOMP_sections_next()  {

  static GOMP_sections_next_p GOMP_sections_next_h = NULL;
  unsigned int retval = 0;
  DEBUGPRINT("GOMP_sections_next %d\n", Tau_get_tid());

  if (GOMP_sections_next_h == NULL) {
    GOMP_sections_next_h = (GOMP_sections_next_p)get_system_function_handle("GOMP_sections_next",(void*)GOMP_sections_next);
  }
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  retval  =  (*GOMP_sections_next_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }
  return retval;

}


/**********************************************************
   GOMP_parallel_sections_start
 **********************************************************/

void  GOMP_parallel_sections_start(void (*a1)(void *), void * a2, unsigned int a3, unsigned int a4)  {

  static GOMP_parallel_sections_start_p GOMP_parallel_sections_start_h = NULL;
  DEBUGPRINT("GOMP_parallel_sections_start %d\n", Tau_get_tid());

  if (GOMP_parallel_sections_start_h == NULL) {
    GOMP_parallel_sections_start_h = (GOMP_parallel_sections_start_p)get_system_function_handle("GOMP_parallel_sections_start",(void*)GOMP_parallel_sections_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }
  (*GOMP_parallel_sections_start_h)( a1,  a2,  a3,  a4);
  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
    // do this AFTER the start, so we know how many threads there are.
    __ompc_event_callback(OMP_EVENT_FORK);
  }

}


/**********************************************************
   GOMP_sections_end
 **********************************************************/

void  GOMP_sections_end()  {

  static GOMP_sections_end_p GOMP_sections_end_h = NULL;
  DEBUGPRINT("GOMP_sections_end %d\n", Tau_get_tid());

  if (GOMP_sections_end_h == NULL) {
    GOMP_sections_end_h = (GOMP_sections_end_p)get_system_function_handle("GOMP_sections_end",(void*)GOMP_sections_end);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  (*GOMP_sections_end_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

}


/**********************************************************
   GOMP_sections_end_nowait
 **********************************************************/

void  GOMP_sections_end_nowait()  {

  static GOMP_sections_end_nowait_p GOMP_sections_end_nowait_h = NULL;
  DEBUGPRINT("GOMP_sections_end_nowait %d\n", Tau_get_tid());

  if (GOMP_sections_end_nowait_h == NULL) {
    GOMP_sections_end_nowait_h = (GOMP_sections_end_nowait_p)get_system_function_handle("GOMP_sections_end_nowait",(void*)GOMP_sections_end_nowait);
  }

  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  (*GOMP_sections_end_nowait_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

}


/**********************************************************
   GOMP_single_start
 **********************************************************/

bool  GOMP_single_start()  {

  static GOMP_single_start_p GOMP_single_start_h = NULL;
  bool retval = 0;
  DEBUGPRINT("GOMP_single_start %d\n", Tau_get_tid());

  if (GOMP_single_start_h == NULL) {
    GOMP_single_start_h = (GOMP_single_start_p)get_system_function_handle("GOMP_single_start",(void*)GOMP_single_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
	// in this case, the single region is entered and executed
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
    Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
  }
  retval  =  (*GOMP_single_start_h)();
  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
	// the code is done, so fire the callback end event
    __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);
  }

  return retval;

}


/**********************************************************
   GOMP_single_copy_start
 **********************************************************/

void *  GOMP_single_copy_start()  {

  static GOMP_single_copy_start_p GOMP_single_copy_start_h = NULL;
  void * retval = 0;
  DEBUGPRINT("GOMP_single_copy_start %d\n", Tau_get_tid());

  if (GOMP_single_copy_start_h == NULL) {
    GOMP_single_copy_start_h = (GOMP_single_copy_start_p)get_system_function_handle("GOMP_single_copy_start",(void*)GOMP_single_copy_start);
  }

  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); 
    __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
  }
  retval  =  (*GOMP_single_copy_start_h)();
  if (Tau_global_get_insideTAU() == 0) { 
	Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); 
  }
  return retval;

}


/**********************************************************
   GOMP_single_copy_end
 **********************************************************/

void  GOMP_single_copy_end(void * a1)  {

  static GOMP_single_copy_end_p GOMP_single_copy_end_h = NULL;
  DEBUGPRINT("GOMP_single_copy_end %d\n", Tau_get_tid());

  if (GOMP_single_copy_end_h == NULL) {
    GOMP_single_copy_end_h = (GOMP_single_copy_end_p)get_system_function_handle("GOMP_single_copy_end",(void*)GOMP_single_copy_end);
  }

  __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_start_task(__FUNCTION__, Tau_get_tid()); }
  (*GOMP_single_copy_end_h)( a1);
  if (Tau_global_get_insideTAU() == 0) { Tau_pure_stop_task(__FUNCTION__, Tau_get_tid()); }

}

