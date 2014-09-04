#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdbool.h>
#include <stdio.h>
#include "omp_collector_util.h"
#include <stdlib.h>
#include "gomp_wrapper_types.h"
#include "TAU.h"
#include "tau_wrapper_declarations.h"

#ifdef TAU_PRELOAD_LIB
/**************** this section is for ld preload only ****************** */

#include <dlfcn.h>

#define RESET_DLERROR() dlerror()
#define CHECK_DLERROR() { \
    char const * err = dlerror(); \
    if (err) { \
        printf("Error getting %s handle: %s\n", name, err); \
        fflush(stdout); \
        exit(1); \
    } \
}

/* so that OpenMP and PTHREAD can both be defined, don't wrap this one
   if Pthreads is used. */
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

/**********************************************************
  pthread_create
 **********************************************************/

// this is here to capture when threads are spawned by GOMP.
// That is in case we are sampling only, and not capturing
// ORA events. This way, we can initialize sampling on all threads.
int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
    start_routine_p start_routine, void* arg)
{
  static pthread_create_p pthread_create_h = NULL;
  if (pthread_create_h == NULL) {
    pthread_create_h = (pthread_create_p)get_system_function_handle(
        "pthread_create", (void*)pthread_create);
  }
  return tau_gomp_pthread_create_wrapper(pthread_create_h, thread, attr, start_routine, arg);
}

/**********************************************************
  omp_set_lock
 **********************************************************/

void omp_set_lock(omp_lock_t *lock)  {
    static omp_set_lock_p omp_set_lock_h = NULL;
    if (omp_set_lock_h == NULL) {
        omp_set_lock_h = (omp_set_lock_p)get_system_function_handle("omp_set_lock",(void*)omp_set_lock);
    }
    tau_omp_set_lock(omp_set_lock_h, lock);
}

/**********************************************************
  omp_set_nest_lock
 **********************************************************/

void omp_set_nest_lock(omp_nest_lock_t *nest_lock) {
    static omp_set_nest_lock_p omp_set_nest_lock_h = NULL;
    if (omp_set_nest_lock_h == NULL) {
        omp_set_nest_lock_h = (omp_set_nest_lock_p)get_system_function_handle("omp_set_nest_lock",(void*)omp_set_nest_lock);
    }
    tau_omp_set_nest_lock(omp_set_nest_lock_h, nest_lock);
}

/**********************************************************
  GOMP_barrier
 **********************************************************/

void  GOMP_barrier()  {
    Tau_global_incr_insideTAU();
    static GOMP_barrier_p GOMP_barrier_h = NULL;
    if (GOMP_barrier_h == NULL) {
        GOMP_barrier_h = (GOMP_barrier_p)get_system_function_handle("GOMP_barrier",(void*)GOMP_barrier);
    }
    tau_GOMP_barrier(GOMP_barrier_h);
    Tau_global_decr_insideTAU();
}

/**********************************************************
  GOMP_critical_start
 **********************************************************/

void  GOMP_critical_start()  {
    Tau_global_incr_insideTAU();
    static GOMP_critical_start_p GOMP_critical_start_h = NULL;
    if (GOMP_critical_start_h == NULL) {
        GOMP_critical_start_h = (GOMP_critical_start_p)get_system_function_handle("GOMP_critical_start",(void*)GOMP_critical_start);
    }
    tau_GOMP_critical_start(GOMP_critical_start_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_critical_end
 **********************************************************/

void  GOMP_critical_end()  {
    Tau_global_incr_insideTAU();
    static GOMP_critical_end_p GOMP_critical_end_h = NULL;
    if (GOMP_critical_end_h == NULL) {
        GOMP_critical_end_h = (GOMP_critical_end_p)get_system_function_handle("GOMP_critical_end",(void*)GOMP_critical_end);
    }
    tau_GOMP_critical_end(GOMP_critical_end_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_critical_name_start
 **********************************************************/

void  GOMP_critical_name_start(void ** a1)  {
    Tau_global_incr_insideTAU();
    static GOMP_critical_name_start_p GOMP_critical_name_start_h = NULL;
    if (GOMP_critical_name_start_h == NULL) {
        GOMP_critical_name_start_h = (GOMP_critical_name_start_p)get_system_function_handle("GOMP_critical_name_start",(void*)GOMP_critical_name_start);
    }
    tau_GOMP_critical_name_start(GOMP_critical_name_start_h, a1);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_critical_name_end
 **********************************************************/

void  GOMP_critical_name_end(void ** a1)  {
    Tau_global_incr_insideTAU();
    static GOMP_critical_name_end_p GOMP_critical_name_end_h = NULL;
    if (GOMP_critical_name_end_h == NULL) {
        GOMP_critical_name_end_h = (GOMP_critical_name_end_p)get_system_function_handle("GOMP_critical_name_end",(void*)GOMP_critical_name_end);
    }
    tau_GOMP_critical_name_end(GOMP_critical_name_end_h, a1);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_atomic_start
 **********************************************************/

void  GOMP_atomic_start()  {
    Tau_global_incr_insideTAU();
    static GOMP_atomic_start_p GOMP_atomic_start_h;
    if (GOMP_atomic_start_h == NULL) {
        GOMP_atomic_start_h = (GOMP_atomic_start_p)get_system_function_handle("GOMP_atomic_start",(void*)GOMP_atomic_start);
    }
    tau_GOMP_atomic_start(GOMP_atomic_start_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_atomic_end
 **********************************************************/

void  GOMP_atomic_end()  {
    Tau_global_incr_insideTAU();
    static GOMP_atomic_end_p GOMP_atomic_end_h = NULL;
    if (GOMP_atomic_end_h == NULL) {
        GOMP_atomic_end_h = (GOMP_atomic_end_p)get_system_function_handle("GOMP_atomic_end",(void*)GOMP_atomic_end);
    }
    tau_GOMP_atomic_end(GOMP_atomic_end_h);
    Tau_global_decr_insideTAU();
}

#ifdef TAU_GOMP_WRAP_EVERYTHING

/**********************************************************
  GOMP_loop_static_start
 **********************************************************/

bool  GOMP_loop_static_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_static_start_p GOMP_loop_static_start_h = NULL;
    if (GOMP_loop_static_start_h == NULL) {
        GOMP_loop_static_start_h = (GOMP_loop_static_start_p)get_system_function_handle("GOMP_loop_static_start",(void*)GOMP_loop_static_start);
    }
    bool retval = tau_GOMP_loop_static_start(GOMP_loop_static_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_dynamic_start
 **********************************************************/

bool  GOMP_loop_dynamic_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_dynamic_start_p GOMP_loop_dynamic_start_h = NULL;
    if (GOMP_loop_dynamic_start_h == NULL) {
        GOMP_loop_dynamic_start_h = (GOMP_loop_dynamic_start_p)get_system_function_handle("GOMP_loop_dynamic_start",(void*)GOMP_loop_dynamic_start);
    }
    bool retval = tau_GOMP_loop_dynamic_start(GOMP_loop_dynamic_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_guided_start
 **********************************************************/

bool  GOMP_loop_guided_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_guided_start_p GOMP_loop_guided_start_h = NULL;
    if (GOMP_loop_guided_start_h == NULL) {
        GOMP_loop_guided_start_h = (GOMP_loop_guided_start_p)get_system_function_handle("GOMP_loop_guided_start",(void*)GOMP_loop_guided_start);
    }
    bool retval = tau_GOMP_loop_guided_start(GOMP_loop_guided_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_runtime_start
 **********************************************************/

bool  GOMP_loop_runtime_start(long a1, long a2, long a3, long * a4, long * a5)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_runtime_start_p GOMP_loop_runtime_start_h = NULL;
    if (GOMP_loop_runtime_start_h == NULL) {
        GOMP_loop_runtime_start_h = (GOMP_loop_runtime_start_p)get_system_function_handle("GOMP_loop_runtime_start",(void*)GOMP_loop_runtime_start);
    }
    bool retval = tau_GOMP_loop_runtime_start(GOMP_loop_runtime_start_h, a1,  a2,  a3,  a4,  a5);
    Tau_global_decr_insideTAU();
	return retval;
}

#endif

/**********************************************************
    Tau_global_incr_insideTAU();
  GOMP_loop_ordered_static_start
 **********************************************************/

bool  GOMP_loop_ordered_static_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_static_start_p GOMP_loop_ordered_static_start_h = NULL;
    if (GOMP_loop_ordered_static_start_h == NULL) {
        GOMP_loop_ordered_static_start_h = (GOMP_loop_ordered_static_start_p)get_system_function_handle("GOMP_loop_ordered_static_start",(void*)GOMP_loop_ordered_static_start);
    }
    bool retval = tau_GOMP_loop_ordered_static_start(GOMP_loop_ordered_static_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_dynamic_start
 **********************************************************/

bool  GOMP_loop_ordered_dynamic_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_dynamic_start_p GOMP_loop_ordered_dynamic_start_h = NULL;
    if (GOMP_loop_ordered_dynamic_start_h == NULL) {
        GOMP_loop_ordered_dynamic_start_h = (GOMP_loop_ordered_dynamic_start_p)get_system_function_handle("GOMP_loop_ordered_dynamic_start",(void*)GOMP_loop_ordered_dynamic_start);
    }
    bool retval = tau_GOMP_loop_ordered_dynamic_start(GOMP_loop_ordered_dynamic_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_guided_start
 **********************************************************/

bool  GOMP_loop_ordered_guided_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_guided_start_p GOMP_loop_ordered_guided_start_h = NULL;
    if (GOMP_loop_ordered_guided_start_h == NULL) {
        GOMP_loop_ordered_guided_start_h = (GOMP_loop_ordered_guided_start_p)get_system_function_handle("GOMP_loop_ordered_guided_start",(void*)GOMP_loop_ordered_guided_start);
    }
    bool retval = tau_GOMP_loop_ordered_guided_start(GOMP_loop_ordered_guided_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_runtime_start
 **********************************************************/

bool  GOMP_loop_ordered_runtime_start(long a1, long a2, long a3, long * a4, long * a5)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_runtime_start_p GOMP_loop_ordered_runtime_start_h = NULL;
    if (GOMP_loop_ordered_runtime_start_h == NULL) {
        GOMP_loop_ordered_runtime_start_h = (GOMP_loop_ordered_runtime_start_p)get_system_function_handle("GOMP_loop_ordered_runtime_start",(void*)GOMP_loop_ordered_runtime_start);
    }
    bool retval = tau_GOMP_loop_ordered_runtime_start(GOMP_loop_ordered_runtime_start_h, a1,  a2,  a3,  a4,  a5);
    Tau_global_decr_insideTAU();
	return retval;
}

#ifdef TAU_GOMP_WRAP_EVERYTHING

/**********************************************************
  GOMP_loop_static_next
 **********************************************************/

bool  GOMP_loop_static_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_static_next_p GOMP_loop_static_next_h = NULL;
    if (GOMP_loop_static_next_h == NULL) {
        GOMP_loop_static_next_h = (GOMP_loop_static_next_p)get_system_function_handle("GOMP_loop_static_next",(void*)GOMP_loop_static_next);
    }
    bool retval = tau_GOMP_loop_static_next(GOMP_loop_static_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_dynamic_next
 **********************************************************/

bool  GOMP_loop_dynamic_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_dynamic_next_p GOMP_loop_dynamic_next_h = NULL;
    if (GOMP_loop_dynamic_next_h == NULL) {
        GOMP_loop_dynamic_next_h = (GOMP_loop_dynamic_next_p)get_system_function_handle("GOMP_loop_dynamic_next",(void*)GOMP_loop_dynamic_next);
    }
    bool retval = tau_GOMP_loop_dynamic_next (GOMP_loop_dynamic_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_guided_next
 **********************************************************/

bool  GOMP_loop_guided_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_guided_next_p GOMP_loop_guided_next_h = NULL;
    if (GOMP_loop_guided_next_h == NULL) {
        GOMP_loop_guided_next_h = (GOMP_loop_guided_next_p)get_system_function_handle("GOMP_loop_guided_next",(void*)GOMP_loop_guided_next);
    }
    bool retval = tau_GOMP_loop_guided_next (GOMP_loop_guided_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_runtime_next
 **********************************************************/

bool  GOMP_loop_runtime_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_runtime_next_p GOMP_loop_runtime_next_h = NULL;
    if (GOMP_loop_runtime_next_h == NULL) {
        GOMP_loop_runtime_next_h = (GOMP_loop_runtime_next_p)get_system_function_handle("GOMP_loop_runtime_next",(void*)GOMP_loop_runtime_next);
    }
    bool retval = tau_GOMP_loop_runtime_next(GOMP_loop_runtime_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_static_next
 **********************************************************/

bool  GOMP_loop_ordered_static_next(long * a1, long * a2)  {
    static GOMP_loop_ordered_static_next_p GOMP_loop_ordered_static_next_h = NULL;
    if (GOMP_loop_ordered_static_next_h == NULL) {
        GOMP_loop_ordered_static_next_h = (GOMP_loop_ordered_static_next_p)get_system_function_handle("GOMP_loop_ordered_static_next",(void*)GOMP_loop_ordered_static_next);
    }
    bool retval = tau_GOMP_loop_ordered_static_next(GOMP_loop_ordered_static_next_h, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_dynamic_next
 **********************************************************/

bool  GOMP_loop_ordered_dynamic_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_dynamic_next_p GOMP_loop_ordered_dynamic_next_h = NULL;
    if (GOMP_loop_ordered_dynamic_next_h == NULL) {
        GOMP_loop_ordered_dynamic_next_h = (GOMP_loop_ordered_dynamic_next_p)get_system_function_handle("GOMP_loop_ordered_dynamic_next",(void*)GOMP_loop_ordered_dynamic_next);
    }
    bool retval = tau_GOMP_loop_ordered_dynamic_next(GOMP_loop_ordered_dynamic_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_guided_next
 **********************************************************/

bool  GOMP_loop_ordered_guided_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_guided_next_p GOMP_loop_ordered_guided_next_h = NULL;
    if (GOMP_loop_ordered_guided_next_h == NULL) {
        GOMP_loop_ordered_guided_next_h = (GOMP_loop_ordered_guided_next_p)get_system_function_handle("GOMP_loop_ordered_guided_next",(void*)GOMP_loop_ordered_guided_next);
    }
    bool retval = tau_GOMP_loop_ordered_guided_next (GOMP_loop_ordered_guided_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ordered_runtime_next
 **********************************************************/

bool  GOMP_loop_ordered_runtime_next(long * a1, long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ordered_runtime_next_p GOMP_loop_ordered_runtime_next_h = NULL;
    if (GOMP_loop_ordered_runtime_next_h == NULL) {
        GOMP_loop_ordered_runtime_next_h = (GOMP_loop_ordered_runtime_next_p)get_system_function_handle("GOMP_loop_ordered_runtime_next",(void*)GOMP_loop_ordered_runtime_next);
    }
    bool retval = tau_GOMP_loop_ordered_runtime_next (GOMP_loop_ordered_runtime_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}

#endif

/**********************************************************
  GOMP_parallel_loop_static_start
 **********************************************************/

void  GOMP_parallel_loop_static_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_loop_static_start_p GOMP_parallel_loop_static_start_h = NULL;
    if (GOMP_parallel_loop_static_start_h == NULL) {
        GOMP_parallel_loop_static_start_h = (GOMP_parallel_loop_static_start_p)get_system_function_handle("GOMP_parallel_loop_static_start",(void*)GOMP_parallel_loop_static_start);
    }
    tau_GOMP_parallel_loop_static_start(GOMP_parallel_loop_static_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_parallel_loop_dynamic_start
 **********************************************************/

void  GOMP_parallel_loop_dynamic_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_loop_dynamic_start_p GOMP_parallel_loop_dynamic_start_h = NULL;
    if (GOMP_parallel_loop_dynamic_start_h == NULL) {
        GOMP_parallel_loop_dynamic_start_h = (GOMP_parallel_loop_dynamic_start_p)get_system_function_handle("GOMP_parallel_loop_dynamic_start",(void*)GOMP_parallel_loop_dynamic_start);
    }
    tau_GOMP_parallel_loop_dynamic_start(GOMP_parallel_loop_dynamic_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_parallel_loop_guided_start
 **********************************************************/

void  GOMP_parallel_loop_guided_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_loop_guided_start_p GOMP_parallel_loop_guided_start_h = NULL;
    if (GOMP_parallel_loop_guided_start_h == NULL) {
        GOMP_parallel_loop_guided_start_h = (GOMP_parallel_loop_guided_start_p)get_system_function_handle("GOMP_parallel_loop_guided_start",(void*)GOMP_parallel_loop_guided_start);
    }
    tau_GOMP_parallel_loop_guided_start(GOMP_parallel_loop_guided_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_parallel_loop_runtime_start
 **********************************************************/

void  GOMP_parallel_loop_runtime_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_loop_runtime_start_p GOMP_parallel_loop_runtime_start_h = NULL;
    if (GOMP_parallel_loop_runtime_start_h == NULL) {
        GOMP_parallel_loop_runtime_start_h = (GOMP_parallel_loop_runtime_start_p)get_system_function_handle("GOMP_parallel_loop_runtime_start",(void*)GOMP_parallel_loop_runtime_start);
    }
    tau_GOMP_parallel_loop_runtime_start(GOMP_parallel_loop_runtime_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_loop_end
 **********************************************************/

void  GOMP_loop_end()  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_end_p GOMP_loop_end_h = NULL;
    if (GOMP_loop_end_h == NULL) {
        GOMP_loop_end_h = (GOMP_loop_end_p)get_system_function_handle("GOMP_loop_end",(void*)GOMP_loop_end);
    }
    tau_GOMP_loop_end(GOMP_loop_end_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_loop_end_nowait
 **********************************************************/

void  GOMP_loop_end_nowait()  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_end_nowait_p GOMP_loop_end_nowait_h = NULL;
    if (GOMP_loop_end_nowait_h == NULL) {
        GOMP_loop_end_nowait_h = (GOMP_loop_end_nowait_p)get_system_function_handle("GOMP_loop_end_nowait",(void*)GOMP_loop_end_nowait);
    }
    tau_GOMP_loop_end_nowait(GOMP_loop_end_nowait_h);
    Tau_global_decr_insideTAU();
}

#ifdef TAU_GOMP_WRAP_EVERYTHING

/**********************************************************
  GOMP_loop_ull_static_start
 **********************************************************/

bool  GOMP_loop_ull_static_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_static_start_p GOMP_loop_ull_static_start_h = NULL;
    if (GOMP_loop_ull_static_start_h == NULL) {
        GOMP_loop_ull_static_start_h = (GOMP_loop_ull_static_start_p)get_system_function_handle("GOMP_loop_ull_static_start",(void*)GOMP_loop_ull_static_start);
    }
    bool retval = tau_GOMP_loop_ull_static_start(GOMP_loop_ull_static_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_dynamic_start
 **********************************************************/

bool  GOMP_loop_ull_dynamic_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_dynamic_start_p GOMP_loop_ull_dynamic_start_h = NULL;
    if (GOMP_loop_ull_dynamic_start_h == NULL) {
        GOMP_loop_ull_dynamic_start_h = (GOMP_loop_ull_dynamic_start_p)get_system_function_handle("GOMP_loop_ull_dynamic_start",(void*)GOMP_loop_ull_dynamic_start);
    }
    bool retval = tau_GOMP_loop_ull_dynamic_start(GOMP_loop_ull_dynamic_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_guided_start
 **********************************************************/

bool  GOMP_loop_ull_guided_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_guided_start_p GOMP_loop_ull_guided_start_h = NULL;
    if (GOMP_loop_ull_guided_start_h == NULL) {
        GOMP_loop_ull_guided_start_h = (GOMP_loop_ull_guided_start_p)get_system_function_handle("GOMP_loop_ull_guided_start",(void*)GOMP_loop_ull_guided_start);
    }
    bool retval = tau_GOMP_loop_ull_guided_start(GOMP_loop_ull_guided_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_runtime_start
 **********************************************************/

bool  GOMP_loop_ull_runtime_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_runtime_start_p GOMP_loop_ull_runtime_start_h = NULL;
    if (GOMP_loop_ull_runtime_start_h == NULL) {
        GOMP_loop_ull_runtime_start_h = (GOMP_loop_ull_runtime_start_p)get_system_function_handle("GOMP_loop_ull_runtime_start",(void*)GOMP_loop_ull_runtime_start);
    }
    bool retval = tau_GOMP_loop_ull_runtime_start(GOMP_loop_ull_runtime_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_static_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_static_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_static_start_p GOMP_loop_ull_ordered_static_start_h = NULL;
    if (GOMP_loop_ull_ordered_static_start_h == NULL) {
        GOMP_loop_ull_ordered_static_start_h = (GOMP_loop_ull_ordered_static_start_p)get_system_function_handle("GOMP_loop_ull_ordered_static_start",(void*)GOMP_loop_ull_ordered_static_start);
    }
    bool retval = tau_GOMP_loop_ull_ordered_static_start(GOMP_loop_ull_ordered_static_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_dynamic_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_dynamic_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_dynamic_start_p GOMP_loop_ull_ordered_dynamic_start_h = NULL;
    if (GOMP_loop_ull_ordered_dynamic_start_h == NULL) {
        GOMP_loop_ull_ordered_dynamic_start_h = (GOMP_loop_ull_ordered_dynamic_start_p)get_system_function_handle("GOMP_loop_ull_ordered_dynamic_start",(void*)GOMP_loop_ull_ordered_dynamic_start);
    }
    bool retval = tau_GOMP_loop_ull_ordered_dynamic_start(GOMP_loop_ull_ordered_dynamic_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_guided_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_guided_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_guided_start_p GOMP_loop_ull_ordered_guided_start_h = NULL;
    if (GOMP_loop_ull_ordered_guided_start_h == NULL) {
        GOMP_loop_ull_ordered_guided_start_h = (GOMP_loop_ull_ordered_guided_start_p)get_system_function_handle("GOMP_loop_ull_ordered_guided_start",(void*)GOMP_loop_ull_ordered_guided_start);
    }
    bool retval = tau_GOMP_loop_ull_ordered_guided_start(GOMP_loop_ull_ordered_guided_start_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_runtime_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_runtime_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_runtime_start_p GOMP_loop_ull_ordered_runtime_start_h = NULL;
    if (GOMP_loop_ull_ordered_runtime_start_h == NULL) {
        GOMP_loop_ull_ordered_runtime_start_h = (GOMP_loop_ull_ordered_runtime_start_p)get_system_function_handle("GOMP_loop_ull_ordered_runtime_start",(void*)GOMP_loop_ull_ordered_runtime_start);
    }
    bool retval = tau_GOMP_loop_ull_ordered_runtime_start(GOMP_loop_ull_ordered_runtime_start_h, a1,  a2,  a3,  a4,  a5,  a6);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_static_next
 **********************************************************/

bool  GOMP_loop_ull_static_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_static_next_p GOMP_loop_ull_static_next_h = NULL;
    if (GOMP_loop_ull_static_next_h == NULL) {
        GOMP_loop_ull_static_next_h = (GOMP_loop_ull_static_next_p)get_system_function_handle("GOMP_loop_ull_static_next",(void*)GOMP_loop_ull_static_next);
    }
    bool retval = tau_GOMP_loop_ull_static_next(GOMP_loop_ull_static_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_dynamic_next
 **********************************************************/

bool  GOMP_loop_ull_dynamic_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_dynamic_next_p GOMP_loop_ull_dynamic_next_h = NULL;
    if (GOMP_loop_ull_dynamic_next_h == NULL) {
        GOMP_loop_ull_dynamic_next_h = (GOMP_loop_ull_dynamic_next_p)get_system_function_handle("GOMP_loop_ull_dynamic_next",(void*)GOMP_loop_ull_dynamic_next);
    }
    bool retval = tau_GOMP_loop_ull_dynamic_next(GOMP_loop_ull_dynamic_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_guided_next
 **********************************************************/

bool  GOMP_loop_ull_guided_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_guided_next_p GOMP_loop_ull_guided_next_h = NULL;
    if (GOMP_loop_ull_guided_next_h == NULL) {
        GOMP_loop_ull_guided_next_h = (GOMP_loop_ull_guided_next_p)get_system_function_handle("GOMP_loop_ull_guided_next",(void*)GOMP_loop_ull_guided_next);
    }
    bool retval = tau_GOMP_loop_ull_guided_next(GOMP_loop_ull_guided_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_runtime_next
 **********************************************************/

bool  GOMP_loop_ull_runtime_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_runtime_next_p GOMP_loop_ull_runtime_next_h = NULL;
    if (GOMP_loop_ull_runtime_next_h == NULL) {
        GOMP_loop_ull_runtime_next_h = (GOMP_loop_ull_runtime_next_p)get_system_function_handle("GOMP_loop_ull_runtime_next",(void*)GOMP_loop_ull_runtime_next);
    }
    bool retval = tau_GOMP_loop_ull_runtime_next(GOMP_loop_ull_runtime_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_static_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_static_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_static_next_p GOMP_loop_ull_ordered_static_next_h = NULL;
    if (GOMP_loop_ull_ordered_static_next_h == NULL) {
        GOMP_loop_ull_ordered_static_next_h = (GOMP_loop_ull_ordered_static_next_p)get_system_function_handle("GOMP_loop_ull_ordered_static_next",(void*)GOMP_loop_ull_ordered_static_next);
    }
    bool retval = tau_GOMP_loop_ull_ordered_static_next(GOMP_loop_ull_ordered_static_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_dynamic_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_dynamic_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_dynamic_next_p GOMP_loop_ull_ordered_dynamic_next_h = NULL;
    if (GOMP_loop_ull_ordered_dynamic_next_h == NULL) {
        GOMP_loop_ull_ordered_dynamic_next_h = (GOMP_loop_ull_ordered_dynamic_next_p)get_system_function_handle("GOMP_loop_ull_ordered_dynamic_next",(void*)GOMP_loop_ull_ordered_dynamic_next);
    }
    bool retval = tau_GOMP_loop_ull_ordered_dynamic_next(GOMP_loop_ull_ordered_dynamic_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_guided_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_guided_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_guided_next_p GOMP_loop_ull_ordered_guided_next_h = NULL;
    if (GOMP_loop_ull_ordered_guided_next_h == NULL) {
        GOMP_loop_ull_ordered_guided_next_h = (GOMP_loop_ull_ordered_guided_next_p)get_system_function_handle("GOMP_loop_ull_ordered_guided_next",(void*)GOMP_loop_ull_ordered_guided_next);
    }
    bool retval = tau_GOMP_loop_ull_ordered_guided_next(GOMP_loop_ull_ordered_guided_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_loop_ull_ordered_runtime_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_runtime_next(unsigned long long * a1, unsigned long long * a2)  {
    Tau_global_incr_insideTAU();
    static GOMP_loop_ull_ordered_runtime_next_p GOMP_loop_ull_ordered_runtime_next_h = NULL;
    if (GOMP_loop_ull_ordered_runtime_next_h == NULL) {
        GOMP_loop_ull_ordered_runtime_next_h = (GOMP_loop_ull_ordered_runtime_next_p)get_system_function_handle("GOMP_loop_ull_ordered_runtime_next",(void*)GOMP_loop_ull_ordered_runtime_next);
    }
    bool retval = tau_GOMP_loop_ull_ordered_runtime_next(GOMP_loop_ull_ordered_runtime_next_h, a1,  a2);
    Tau_global_decr_insideTAU();
	return retval;
}

#endif

/**********************************************************
  GOMP_ordered_start
 **********************************************************/

void  GOMP_ordered_start()  {
    Tau_global_incr_insideTAU();
    static GOMP_ordered_start_p GOMP_ordered_start_h = NULL;
    if (GOMP_ordered_start_h == NULL) {
        GOMP_ordered_start_h = (GOMP_ordered_start_p)get_system_function_handle("GOMP_ordered_start",(void*)GOMP_ordered_start);
    }
    tau_GOMP_ordered_start(GOMP_ordered_start_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_ordered_end
 **********************************************************/

void  GOMP_ordered_end()  {
    Tau_global_incr_insideTAU();
    static GOMP_ordered_end_p GOMP_ordered_end_h = NULL;
    if (GOMP_ordered_end_h == NULL) {
        GOMP_ordered_end_h = (GOMP_ordered_end_p)get_system_function_handle("GOMP_ordered_end",(void*)GOMP_ordered_end);
    }
    tau_GOMP_ordered_end(GOMP_ordered_end_h);
    Tau_global_decr_insideTAU();
}

/**********************************************************
  GOMP_parallel_start
 **********************************************************/

void  GOMP_parallel_start(void (*a1)(void *), void * a2, unsigned int a3)  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_start_p GOMP_parallel_start_h = NULL;
    if (!GOMP_parallel_start_h) {
        GOMP_parallel_start_h = (GOMP_parallel_start_p)get_system_function_handle("GOMP_parallel_start",(void*)GOMP_parallel_start); 
    }
    tau_GOMP_parallel_start(GOMP_parallel_start_h, a1,  a2,  a3);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_parallel_end
 **********************************************************/

void  GOMP_parallel_end()  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_end_p GOMP_parallel_end_h = NULL;
    if (GOMP_parallel_end_h == NULL) {
        GOMP_parallel_end_h = (GOMP_parallel_end_p)get_system_function_handle("GOMP_parallel_end",(void*)GOMP_parallel_end);
    }
    tau_GOMP_parallel_end(GOMP_parallel_end_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_task
 **********************************************************/

void  GOMP_task(void (*a1)(void *), void * a2, void (*a3)(void *, void *), long a4, long a5, bool a6, unsigned int a7)  {
    Tau_global_incr_insideTAU();
    static GOMP_task_p GOMP_task_h = NULL;
    if (GOMP_task_h == NULL) {
        GOMP_task_h = (GOMP_task_p)get_system_function_handle("GOMP_task",(void*)GOMP_task);
    }
    tau_GOMP_task(GOMP_task_h, a1,  a2,  a3,  a4,  a5,  a6,  a7);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_taskwait
 **********************************************************/

void  GOMP_taskwait()  {
    Tau_global_incr_insideTAU();
    static GOMP_taskwait_p GOMP_taskwait_h = NULL;
    if (GOMP_taskwait_h == NULL) {
        GOMP_taskwait_h = (GOMP_taskwait_p)get_system_function_handle("GOMP_taskwait",(void*)GOMP_taskwait);
    }
    tau_GOMP_taskwait(GOMP_taskwait_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_taskyield - only exists in gcc 4.7 and greater, and only as a stub

void  GOMP_taskyield()  {
    Tau_global_incr_insideTAU();
    static GOMP_taskyield_p GOMP_taskyield_h = NULL;
    if (GOMP_taskyield_h == NULL) {
        GOMP_taskyield_h = (GOMP_taskyield_p)get_system_function_handle("GOMP_taskyield",(void*)GOMP_taskyield);
    }
    tau_GOMP_taskyield(GOMP_taskyield_h);
    Tau_global_decr_insideTAU();
}
 **********************************************************/



/**********************************************************
  GOMP_sections_start
 **********************************************************/

unsigned int  GOMP_sections_start(unsigned int a1)  {
    Tau_global_incr_insideTAU();
    static GOMP_sections_start_p GOMP_sections_start_h = NULL;
    if (GOMP_sections_start_h == NULL) {
        GOMP_sections_start_h = (GOMP_sections_start_p)get_system_function_handle("GOMP_sections_start",(void*)GOMP_sections_start);
    }
    unsigned int retval = tau_GOMP_sections_start(GOMP_sections_start_h, a1);
    Tau_global_decr_insideTAU();
	return retval;
}

#ifdef TAU_TIME_GOMP_NEXT

/**********************************************************
  GOMP_sections_next
 **********************************************************/

unsigned int  GOMP_sections_next()  {
    Tau_global_incr_insideTAU();
    static GOMP_sections_next_p GOMP_sections_next_h = NULL;
    if (GOMP_sections_next_h == NULL) {
        GOMP_sections_next_h = (GOMP_sections_next_p)get_system_function_handle("GOMP_sections_next",(void*)GOMP_sections_next);
    }
    unsigned int retval = tau_GOMP_sections_next(GOMP_sections_next_h);
    Tau_global_decr_insideTAU();
	return retval;
}

#endif

/**********************************************************
  GOMP_parallel_sections_start
 **********************************************************/

void  GOMP_parallel_sections_start(void (*a1)(void *), void * a2, unsigned int a3, unsigned int a4)  {
    Tau_global_incr_insideTAU();
    static GOMP_parallel_sections_start_p GOMP_parallel_sections_start_h = NULL;
    if (GOMP_parallel_sections_start_h == NULL) {
        GOMP_parallel_sections_start_h = (GOMP_parallel_sections_start_p)get_system_function_handle("GOMP_parallel_sections_start",(void*)GOMP_parallel_sections_start);
    }
    tau_GOMP_parallel_sections_start(GOMP_parallel_sections_start_h, a1,  a2,  a3,  a4);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_sections_end
 **********************************************************/

void  GOMP_sections_end()  {
    Tau_global_incr_insideTAU();
    static GOMP_sections_end_p GOMP_sections_end_h = NULL;
    if (GOMP_sections_end_h == NULL) {
        GOMP_sections_end_h = (GOMP_sections_end_p)get_system_function_handle("GOMP_sections_end",(void*)GOMP_sections_end);
    }
    tau_GOMP_sections_end(GOMP_sections_end_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_sections_end_nowait
 **********************************************************/

void  GOMP_sections_end_nowait()  {
    Tau_global_incr_insideTAU();
    static GOMP_sections_end_nowait_p GOMP_sections_end_nowait_h = NULL;
    if (GOMP_sections_end_nowait_h == NULL) {
        GOMP_sections_end_nowait_h = (GOMP_sections_end_nowait_p)get_system_function_handle("GOMP_sections_end_nowait",(void*)GOMP_sections_end_nowait);
    }
    tau_GOMP_sections_end_nowait(GOMP_sections_end_nowait_h);
    Tau_global_decr_insideTAU();
}


/**********************************************************
  GOMP_single_start
 **********************************************************/

bool  GOMP_single_start()  {
    Tau_global_incr_insideTAU();
    static GOMP_single_start_p GOMP_single_start_h = NULL;
    if (GOMP_single_start_h == NULL) {
        GOMP_single_start_h = (GOMP_single_start_p)get_system_function_handle("GOMP_single_start",(void*)GOMP_single_start);
    }
    bool retval = tau_GOMP_single_start(GOMP_single_start_h);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_single_copy_start
 **********************************************************/

void *  GOMP_single_copy_start()  {
    Tau_global_incr_insideTAU();
    static GOMP_single_copy_start_p GOMP_single_copy_start_h = NULL;
    if (GOMP_single_copy_start_h == NULL) {
        GOMP_single_copy_start_h = (GOMP_single_copy_start_p)get_system_function_handle("GOMP_single_copy_start",(void*)GOMP_single_copy_start);
    }
    void * retval = tau_GOMP_single_copy_start(GOMP_single_copy_start_h);
    Tau_global_decr_insideTAU();
	return retval;
}


/**********************************************************
  GOMP_single_copy_end
 **********************************************************/

void  GOMP_single_copy_end(void * a1)  {
    Tau_global_incr_insideTAU();
    static GOMP_single_copy_end_p GOMP_single_copy_end_h = NULL;
    if (GOMP_single_copy_end_h == NULL) {
        GOMP_single_copy_end_h = (GOMP_single_copy_end_p)get_system_function_handle("GOMP_single_copy_end",(void*)GOMP_single_copy_end);
    }
    tau_GOMP_single_copy_end(GOMP_single_copy_end_h, a1);
    Tau_global_decr_insideTAU();
}

#else // not TAU_PRELOAD_LIB
/**************** this section is for static linking and wrapping ****************** */

// this is here to capture when threads are spawned by GOMP.
// That is in case we are sampling only, and not capturing
// ORA events. This way, we can initialize sampling on all threads.
// The sad truth is that this version of the wrapper will never be used - 
// see the man page for ld for details. Because link-time wrapping ONLY wraps 
// UNDEFINED references to the function, when GOMP calls pthread_create, 
// that is already defined. Therefore, we can't capture the call to 
// pthread_create from GOMP, only from user code. That said, the dlsym() 
// (TAU_PRELOAD_LIB) method above will still work when using tau_exec.

/* so that OpenMP and PTHREAD can both be defined, don't wrap this one
   if Pthreads is used. */
int __real_pthread_create(pthread_t *, const pthread_attr_t *, start_routine_p, void *);
int __wrap_pthread_create(pthread_t * thread, const pthread_attr_t * attr, start_routine_p start_routine, void * arg)
{
  return tau_gomp_pthread_create_wrapper(__real_pthread_create, thread, attr, start_routine, arg);
}

void __real_omp_set_lock(omp_lock_t *lock);
void __wrap_omp_set_lock(omp_lock_t *lock) {
    tau_omp_set_lock(__real_omp_set_lock, lock);
}

void __real_omp_set_nest_lock(omp_nest_lock_t *lock);
void __wrap_omp_set_nest_lock(omp_nest_lock_t *lock) {
    tau_omp_set_nest_lock(__real_omp_set_nest_lock, lock);
}

void __real_GOMP_barrier();
void __wrap_GOMP_barrier() {
    Tau_global_incr_insideTAU();
    tau_GOMP_barrier(__real_GOMP_barrier);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_critical_start ();
void __wrap_GOMP_critical_start () {
    Tau_global_incr_insideTAU();
    tau_GOMP_critical_start (__real_GOMP_critical_start);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_critical_end ();
void __wrap_GOMP_critical_end () {
    Tau_global_incr_insideTAU();
    tau_GOMP_critical_end (__real_GOMP_critical_end);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_critical_name_start (void **);
void __wrap_GOMP_critical_name_start (void ** a1) {
    Tau_global_incr_insideTAU();
    tau_GOMP_critical_name_start (__real_GOMP_critical_name_start, a1);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_critical_name_end (void **);
void __wrap_GOMP_critical_name_end (void ** a1) {
    Tau_global_incr_insideTAU();
    tau_GOMP_critical_name_end (__real_GOMP_critical_name_end, a1);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_atomic_start ();
void __wrap_GOMP_atomic_start () {
    Tau_global_incr_insideTAU();
    tau_GOMP_atomic_start (__real_GOMP_atomic_start);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_atomic_end ();
void __wrap_GOMP_atomic_end () {
    Tau_global_incr_insideTAU();
    tau_GOMP_atomic_end (__real_GOMP_atomic_end);
    Tau_global_decr_insideTAU();
}

#ifdef TAU_GOMP_WRAP_EVERYTHING

bool __real_GOMP_loop_static_start (long, long, long, long, long *, long *);
bool __wrap_GOMP_loop_static_start (long a1, long a2, long a3, long a4, long * a5, long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_static_start (__real_GOMP_loop_static_start, a1, a2, a3, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_dynamic_start (long, long, long, long, long *, long *);
bool __wrap_GOMP_loop_dynamic_start (long a1, long a2, long a3, long a4, long * a5, long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_dynamic_start (__real_GOMP_loop_dynamic_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_guided_start (long, long, long, long, long *, long *);
bool __wrap_GOMP_loop_guided_start (long a1, long a2, long a3, long a4, long * a5, long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_guided_start (__real_GOMP_loop_guided_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_runtime_start (long, long, long, long *, long *);
bool __wrap_GOMP_loop_runtime_start (long a1, long a2, long a3, long * a4, long * a5) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_runtime_start (__real_GOMP_loop_runtime_start, a1, a2, a3, a4, a5);
    Tau_global_decr_insideTAU();
	return retval;
}

#endif

bool __real_GOMP_loop_ordered_static_start (long, long, long, long, long *, long *);
bool __wrap_GOMP_loop_ordered_static_start (long a1, long a2, long a3, long a4, long * a5, long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_static_start (__real_GOMP_loop_ordered_static_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_dynamic_start (long, long, long, long, long *, long *);
bool __wrap_GOMP_loop_ordered_dynamic_start (long a1, long a2, long a3, long a4, long * a5, long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_dynamic_start (__real_GOMP_loop_ordered_dynamic_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_guided_start (long, long, long, long, long *, long *);
bool __wrap_GOMP_loop_ordered_guided_start (long a1, long a2, long a3, long a4, long * a5, long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_guided_start (__real_GOMP_loop_ordered_guided_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_runtime_start (long, long, long, long *, long *);
bool __wrap_GOMP_loop_ordered_runtime_start (long a1, long a2, long a3, long * a4, long * a5) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_runtime_start (__real_GOMP_loop_ordered_runtime_start, a1, a2, a3, a4, a5);
    Tau_global_decr_insideTAU();
	return retval;
}

#ifdef TAU_GOMP_WRAP_EVERYTHING
bool __real_GOMP_loop_static_next (long *, long *);
bool __wrap_GOMP_loop_static_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_static_next (__real_GOMP_loop_static_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_dynamic_next (long *, long *);
bool __wrap_GOMP_loop_dynamic_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_dynamic_next (__real_GOMP_loop_dynamic_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_guided_next (long *, long *);
bool __wrap_GOMP_loop_guided_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_guided_next (__real_GOMP_loop_guided_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_runtime_next (long *, long *);
bool __wrap_GOMP_loop_runtime_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_runtime_next (__real_GOMP_loop_runtime_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_static_next (long *, long *);
bool __wrap_GOMP_loop_ordered_static_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_static_next (__real_GOMP_loop_ordered_static_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_dynamic_next (long *, long *);
bool __wrap_GOMP_loop_ordered_dynamic_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_dynamic_next (__real_GOMP_loop_ordered_dynamic_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_guided_next (long *, long *);
bool __wrap_GOMP_loop_ordered_guided_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_guided_next (__real_GOMP_loop_ordered_guided_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ordered_runtime_next (long *, long *);
bool __wrap_GOMP_loop_ordered_runtime_next (long * a1, long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ordered_runtime_next (__real_GOMP_loop_ordered_runtime_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}
#endif

void __real_GOMP_parallel_loop_static_start (void(*)(void *), void *, unsigned int, long, long, long, long);
void __wrap_GOMP_parallel_loop_static_start (void(*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7) {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_loop_static_start (__real_GOMP_parallel_loop_static_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_parallel_loop_dynamic_start (void(*)(void *), void *, unsigned int, long, long, long, long);
void __wrap_GOMP_parallel_loop_dynamic_start (void(*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7) {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_loop_dynamic_start (__real_GOMP_parallel_loop_dynamic_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_parallel_loop_guided_start (void(*)(void *), void *, unsigned int, long, long, long, long);
void __wrap_GOMP_parallel_loop_guided_start (void(*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7) {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_loop_guided_start (__real_GOMP_parallel_loop_guided_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_parallel_loop_runtime_start (void(*)(void *), void *, unsigned int, long, long, long);
void __wrap_GOMP_parallel_loop_runtime_start (void(*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6) {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_loop_runtime_start (__real_GOMP_parallel_loop_runtime_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_loop_end ();
void __wrap_GOMP_loop_end () {
    Tau_global_incr_insideTAU();
    tau_GOMP_loop_end (__real_GOMP_loop_end);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_loop_end_nowait ();
void __wrap_GOMP_loop_end_nowait () {
    Tau_global_incr_insideTAU();
    tau_GOMP_loop_end_nowait (__real_GOMP_loop_end_nowait);
    Tau_global_decr_insideTAU();
}

#ifdef TAU_GOMP_WRAP_EVERYTHING
bool __real_GOMP_loop_ull_static_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_static_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_static_start (__real_GOMP_loop_ull_static_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_guided_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_guided_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_guided_start (__real_GOMP_loop_ull_guided_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_runtime_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_runtime_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_runtime_start (__real_GOMP_loop_ull_runtime_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_static_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_static_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_static_start (__real_GOMP_loop_ull_ordered_static_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_dynamic_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_dynamic_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_dynamic_start (__real_GOMP_loop_ull_dynamic_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_dynamic_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_dynamic_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_dynamic_start (__real_GOMP_loop_ull_ordered_dynamic_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_guided_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_guided_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_guided_start (__real_GOMP_loop_ull_ordered_guided_start, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_runtime_start (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_runtime_start (bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_runtime_start (__real_GOMP_loop_ull_ordered_runtime_start, a1, a2, a3, a4, a5, a6);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_static_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_static_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_static_next (__real_GOMP_loop_ull_static_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_dynamic_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_dynamic_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_dynamic_next (__real_GOMP_loop_ull_dynamic_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_guided_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_guided_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_guided_next (__real_GOMP_loop_ull_guided_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_runtime_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_runtime_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_runtime_next (__real_GOMP_loop_ull_runtime_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_static_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_static_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_static_next (__real_GOMP_loop_ull_ordered_static_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_dynamic_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_dynamic_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_dynamic_next (__real_GOMP_loop_ull_ordered_dynamic_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_guided_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_guided_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_guided_next (__real_GOMP_loop_ull_ordered_guided_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}

bool __real_GOMP_loop_ull_ordered_runtime_next (unsigned long long *, unsigned long long *);
bool __wrap_GOMP_loop_ull_ordered_runtime_next (unsigned long long * a1, unsigned long long * a2) {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_loop_ull_ordered_runtime_next (__real_GOMP_loop_ull_ordered_runtime_next, a1, a2);
    Tau_global_decr_insideTAU();
	return retval;
}
#endif

void __real_GOMP_ordered_start ();
void __wrap_GOMP_ordered_start () {
    Tau_global_incr_insideTAU();
    tau_GOMP_ordered_start (__real_GOMP_ordered_start);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_ordered_end ();
void __wrap_GOMP_ordered_end () {
    Tau_global_incr_insideTAU();
    tau_GOMP_ordered_end (__real_GOMP_ordered_end);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_parallel_start (void(*)(void *), void *, unsigned int);
void __wrap_GOMP_parallel_start (void(*a1)(void *), void * a2, unsigned int a3) {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_start (__real_GOMP_parallel_start, a1, a2, a3);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_parallel_end ();
void __wrap_GOMP_parallel_end () {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_end (__real_GOMP_parallel_end);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_task (void(*)(void *), void *, void(*)(void *, void *), long, long, bool, unsigned int);
void __wrap_GOMP_task (void(*a1)(void *), void * a2, void(*a3)(void *, void *), long a4, long a5, bool a6, unsigned int a7) {
    Tau_global_incr_insideTAU();
    tau_GOMP_task (__real_GOMP_task, a1, a2, a3, a4, a5, a6, a7);
    Tau_global_decr_insideTAU();
}

void __real_GOMP_taskwait ();
void __wrap_GOMP_taskwait () {
    Tau_global_incr_insideTAU();
    tau_GOMP_taskwait (__real_GOMP_taskwait);
    Tau_global_decr_insideTAU();
}

/* taskyield only exists in 4.7 or greater */
#ifndef GOMP_taskyield
//void GOMP_taskyield () {};
#else

void __real_GOMP_taskyield ();
void __wrap_GOMP_taskyield () {
    Tau_global_incr_insideTAU();
    tau_GOMP_taskyield (__real_GOMP_taskyield);
    Tau_global_decr_insideTAU();
}

unsigned int __real_GOMP_sections_start (unsigned int);
unsigned int __wrap_GOMP_sections_start (unsigned int a1) {
    Tau_global_incr_insideTAU();
    unsigned int retval = tau_GOMP_sections_start (__real_GOMP_sections_start);
    Tau_global_decr_insideTAU();
	return retval;
}

unsigned int __real_GOMP_sections_next ();
unsigned int __wrap_GOMP_sections_next () {
    Tau_global_incr_insideTAU();
    unsigned int retval = tau_GOMP_sections_next (__real_GOMP_sections_next);
    Tau_global_decr_insideTAU();
	return retval;
}
#endif

void __real_GOMP_parallel_sections_start (void(*)(void *), void *, unsigned int, unsigned int);
void __wrap_GOMP_parallel_sections_start (void(*a1)(void *), void * a2, unsigned int a3, unsigned int a4) {
    Tau_global_incr_insideTAU();
    tau_GOMP_parallel_sections_start (__real_GOMP_parallel_sections_start, a1, a2, a3, a4);
    Tau_global_decr_insideTAU();
}

#ifdef TAU_GOMP_WRAP_EVERYTHING

void __real_GOMP_sections_end ();
void __wrap_GOMP_sections_end () {
    Tau_global_incr_insideTAU();
    tau_GOMP_sections_end (__real_GOMP_sections_end);
    Tau_global_decr_insideTAU();
}
void __real_GOMP_sections_end_nowait ();
void __wrap_GOMP_sections_end_nowait () {
    Tau_global_incr_insideTAU();
    tau_GOMP_sections_end_nowait (__real_GOMP_sections_end_nowait);
    Tau_global_decr_insideTAU();
}

#endif

bool __real_GOMP_single_start ();
bool __wrap_GOMP_single_start () {
    Tau_global_incr_insideTAU();
    bool retval = tau_GOMP_single_start (__real_GOMP_single_start);
    Tau_global_decr_insideTAU();
	return retval;
}

void * __real_GOMP_single_copy_start ();
void * __wrap_GOMP_single_copy_start () {
    Tau_global_incr_insideTAU();
    void * retval = tau_GOMP_single_copy_start (__real_GOMP_single_copy_start);
    Tau_global_decr_insideTAU();
	return retval;
}

void __real_GOMP_single_copy_end (void *);
void __wrap_GOMP_single_copy_end (void * a1) {
    Tau_global_incr_insideTAU();
    tau_GOMP_single_copy_end (__real_GOMP_single_copy_end, a1);
    Tau_global_decr_insideTAU();
}

#endif // TAU_PRELOAD_LIB
