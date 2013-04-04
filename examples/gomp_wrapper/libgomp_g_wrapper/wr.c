#include <libgomp_g.h>
#include <Profile/Profiler.h>
#include <stdio.h>

#include <dlfcn.h>
#include "omp_collector_util.h"

#if 1
#define DEBUGPRINT(format, args...) \
{ printf(format, ## args); fflush(stdout); }
#else
#define DEBUGPRINT(format, args...) \
{ }
#endif

const char * tau_orig_libname = "libgomp.so";
static void *tau_handle = NULL;

extern int Tau_global_get_insideTAU();

struct Tau_gomp_wrapper_status_flags {
  int ordered; // 4 bytes
  int critical; // 4 bytes
  int single; // 4 bytes
  char _pad[64-(3*sizeof(int))];
};

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

extern void Tau_lock_environment();
extern void Tau_unlock_environment();

/**********************************************************
   GOMP_barrier
 **********************************************************/

void  GOMP_barrier()  {

  static void (*GOMP_barrier_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_barrier()", "", TAU_USER1);
  DEBUGPRINT("GOMP_barrier %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_barrier_h == NULL)
      GOMP_barrier_h = dlsym(tau_handle,"GOMP_barrier"); 
    if (GOMP_barrier_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
	// make the collector API callback
    if (Tau_global_get_insideTAU() == 0) { 
      Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_EBAR);
      Tau_unlock_environment();
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_barrier_h)();
    if (Tau_global_get_insideTAU() == 0) { 
      Tau_lock_environment();
	  Tau_stop_current_timer(); 
      __ompc_event_callback(OMP_EVENT_THR_END_EBAR);
      Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_critical_start
 **********************************************************/

void  GOMP_critical_start()  {

  static void (*GOMP_critical_start_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_critical_start()", "", TAU_USER1);
  DEBUGPRINT("GOMP_critical_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_critical_start_h == NULL)
      GOMP_critical_start_h = dlsym(tau_handle,"GOMP_critical_start"); 
    if (GOMP_critical_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start 
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  // ok, safe to continue
	  Tau_unlock_environment();
	}
    (*GOMP_critical_start_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer();
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);
	  // ok, safe to continue
	  Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_critical_end
 **********************************************************/

void  GOMP_critical_end()  {

  static void (*GOMP_critical_end_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_critical_end()", "", TAU_USER1);
  DEBUGPRINT("GOMP_critical_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_critical_end_h == NULL)
      GOMP_critical_end_h = dlsym(tau_handle,"GOMP_critical_end"); 
    if (GOMP_critical_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_CTWT);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_critical_end_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer();
	  Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_critical_name_start
 **********************************************************/

void  GOMP_critical_name_start(void ** a1)  {

  static void (*GOMP_critical_name_start_h) (void **) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_critical_name_start(void **)", "", TAU_USER1);
  DEBUGPRINT("GOMP_critical_name_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_critical_name_start_h == NULL)
      GOMP_critical_name_start_h = dlsym(tau_handle,"GOMP_critical_name_start"); 
    if (GOMP_critical_name_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start 
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_critical_name_start_h)( a1);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer(); 
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_CTWT);
	  Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_critical_name_end
 **********************************************************/

void  GOMP_critical_name_end(void ** a1)  {

  static void (*GOMP_critical_name_end_h) (void **) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_critical_name_end(void **)", "", TAU_USER1);
  DEBUGPRINT("GOMP_critical_name_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_critical_name_end_h == NULL)
      GOMP_critical_name_end_h = dlsym(tau_handle,"GOMP_critical_name_end"); 
    if (GOMP_critical_name_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_CTWT);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_critical_name_end_h)( a1);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer();
	  Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_atomic_start
 **********************************************************/

void  GOMP_atomic_start()  {

  static void (*GOMP_atomic_start_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_atomic_start()", "", TAU_USER1);
  DEBUGPRINT("GOMP_atomic_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_atomic_start_h == NULL)
      GOMP_atomic_start_h = dlsym(tau_handle,"GOMP_atomic_start"); 
    if (GOMP_atomic_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) {
	  // protect this from the GOMP_parallel_start 
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_atomic_start_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer();
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ATWT);
	  // ok, safe to continue
	  Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_atomic_end
 **********************************************************/

void  GOMP_atomic_end()  {

  static void (*GOMP_atomic_end_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_atomic_end()", "", TAU_USER1);
  DEBUGPRINT("GOMP_atomic_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_atomic_end_h == NULL)
      GOMP_atomic_end_h = dlsym(tau_handle,"GOMP_atomic_end"); 
    if (GOMP_atomic_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    (*GOMP_atomic_end_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer();
      __ompc_event_callback(OMP_EVENT_THR_END_ATWT);
	  Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_loop_static_start
 **********************************************************/

bool  GOMP_loop_static_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static bool (*GOMP_loop_static_start_h) (long, long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_static_start(long, long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_static_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_static_start_h == NULL)
      GOMP_loop_static_start_h = dlsym(tau_handle,"GOMP_loop_static_start"); 
    if (GOMP_loop_static_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_dynamic_start
 **********************************************************/

bool  GOMP_loop_dynamic_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static bool (*GOMP_loop_dynamic_start_h) (long, long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_dynamic_start(long, long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_dynamic_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_dynamic_start_h == NULL)
      GOMP_loop_dynamic_start_h = dlsym(tau_handle,"GOMP_loop_dynamic_start"); 
    if (GOMP_loop_dynamic_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_guided_start
 **********************************************************/

bool  GOMP_loop_guided_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static bool (*GOMP_loop_guided_start_h) (long, long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_guided_start(long, long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_guided_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_guided_start_h == NULL)
      GOMP_loop_guided_start_h = dlsym(tau_handle,"GOMP_loop_guided_start"); 
    if (GOMP_loop_guided_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_runtime_start
 **********************************************************/

bool  GOMP_loop_runtime_start(long a1, long a2, long a3, long * a4, long * a5)  {

  static bool (*GOMP_loop_runtime_start_h) (long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_runtime_start(long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_runtime_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_runtime_start_h == NULL)
      GOMP_loop_runtime_start_h = dlsym(tau_handle,"GOMP_loop_runtime_start"); 
    if (GOMP_loop_runtime_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_static_start
 **********************************************************/

bool  GOMP_loop_ordered_static_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static bool (*GOMP_loop_ordered_static_start_h) (long, long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_static_start(long, long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_static_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_static_start_h == NULL)
      GOMP_loop_ordered_static_start_h = dlsym(tau_handle,"GOMP_loop_ordered_static_start"); 
    if (GOMP_loop_ordered_static_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer(); 
	  Tau_gomp_flags[omp_get_thread_num()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	  Tau_unlock_environment();
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_dynamic_start
 **********************************************************/

bool  GOMP_loop_ordered_dynamic_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static bool (*GOMP_loop_ordered_dynamic_start_h) (long, long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_dynamic_start(long, long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_dynamic_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_dynamic_start_h == NULL)
      GOMP_loop_ordered_dynamic_start_h = dlsym(tau_handle,"GOMP_loop_ordered_dynamic_start"); 
    if (GOMP_loop_ordered_dynamic_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer(); 
	  Tau_gomp_flags[omp_get_thread_num()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	  Tau_unlock_environment();
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_guided_start
 **********************************************************/

bool  GOMP_loop_ordered_guided_start(long a1, long a2, long a3, long a4, long * a5, long * a6)  {

  static bool (*GOMP_loop_ordered_guided_start_h) (long, long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_guided_start(long, long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_guided_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_guided_start_h == NULL)
      GOMP_loop_ordered_guided_start_h = dlsym(tau_handle,"GOMP_loop_ordered_guided_start"); 
    if (GOMP_loop_ordered_guided_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer(); 
	  Tau_gomp_flags[omp_get_thread_num()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	  Tau_unlock_environment();
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_runtime_start
 **********************************************************/

bool  GOMP_loop_ordered_runtime_start(long a1, long a2, long a3, long * a4, long * a5)  {

  static bool (*GOMP_loop_ordered_runtime_start_h) (long, long, long, long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_runtime_start(long, long, long, long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_runtime_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_runtime_start_h == NULL)
      GOMP_loop_ordered_runtime_start_h = dlsym(tau_handle,"GOMP_loop_ordered_runtime_start"); 
    if (GOMP_loop_ordered_runtime_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // protect this from the GOMP_parallel_start
	  Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    retval  =  (*GOMP_loop_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
	  Tau_stop_current_timer(); 
	  Tau_gomp_flags[omp_get_thread_num()].ordered = 1;
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_ORDERED);
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	  Tau_unlock_environment();
	}
  }
  return retval;

}


/**********************************************************
   GOMP_loop_static_next
 **********************************************************/

bool  GOMP_loop_static_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_static_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_static_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_static_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_static_next_h == NULL)
      GOMP_loop_static_next_h = dlsym(tau_handle,"GOMP_loop_static_next"); 
    if (GOMP_loop_static_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_static_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_dynamic_next
 **********************************************************/

bool  GOMP_loop_dynamic_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_dynamic_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_dynamic_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_dynamic_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_dynamic_next_h == NULL)
      GOMP_loop_dynamic_next_h = dlsym(tau_handle,"GOMP_loop_dynamic_next"); 
    if (GOMP_loop_dynamic_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_dynamic_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_guided_next
 **********************************************************/

bool  GOMP_loop_guided_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_guided_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_guided_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_guided_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_guided_next_h == NULL)
      GOMP_loop_guided_next_h = dlsym(tau_handle,"GOMP_loop_guided_next"); 
    if (GOMP_loop_guided_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_guided_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_runtime_next
 **********************************************************/

bool  GOMP_loop_runtime_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_runtime_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_runtime_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_runtime_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_runtime_next_h == NULL)
      GOMP_loop_runtime_next_h = dlsym(tau_handle,"GOMP_loop_runtime_next"); 
    if (GOMP_loop_runtime_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_runtime_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_static_next
 **********************************************************/

bool  GOMP_loop_ordered_static_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_ordered_static_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_static_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_static_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_static_next_h == NULL)
      GOMP_loop_ordered_static_next_h = dlsym(tau_handle,"GOMP_loop_ordered_static_next"); 
    if (GOMP_loop_ordered_static_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ordered_static_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_dynamic_next
 **********************************************************/

bool  GOMP_loop_ordered_dynamic_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_ordered_dynamic_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_dynamic_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_dynamic_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_dynamic_next_h == NULL)
      GOMP_loop_ordered_dynamic_next_h = dlsym(tau_handle,"GOMP_loop_ordered_dynamic_next"); 
    if (GOMP_loop_ordered_dynamic_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ordered_dynamic_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_guided_next
 **********************************************************/

bool  GOMP_loop_ordered_guided_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_ordered_guided_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_guided_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_guided_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_guided_next_h == NULL)
      GOMP_loop_ordered_guided_next_h = dlsym(tau_handle,"GOMP_loop_ordered_guided_next"); 
    if (GOMP_loop_ordered_guided_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ordered_guided_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ordered_runtime_next
 **********************************************************/

bool  GOMP_loop_ordered_runtime_next(long * a1, long * a2)  {

  static bool (*GOMP_loop_ordered_runtime_next_h) (long *, long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ordered_runtime_next(long *, long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ordered_runtime_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ordered_runtime_next_h == NULL)
      GOMP_loop_ordered_runtime_next_h = dlsym(tau_handle,"GOMP_loop_ordered_runtime_next"); 
    if (GOMP_loop_ordered_runtime_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ordered_runtime_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_parallel_loop_static_start
 **********************************************************/

void  GOMP_parallel_loop_static_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

  static void (*GOMP_parallel_loop_static_start_h) (void (*)(void *), void *, unsigned int, long, long, long, long) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_loop_static_start(void (*)(void *), void *, unsigned int, long, long, long, long)", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_loop_static_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_loop_static_start_h == NULL)
      GOMP_parallel_loop_static_start_h = dlsym(tau_handle,"GOMP_parallel_loop_static_start"); 
    if (GOMP_parallel_loop_static_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    (*GOMP_parallel_loop_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_parallel_loop_dynamic_start
 **********************************************************/

void  GOMP_parallel_loop_dynamic_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

  static void (*GOMP_parallel_loop_dynamic_start_h) (void (*)(void *), void *, unsigned int, long, long, long, long) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_loop_dynamic_start(void (*)(void *), void *, unsigned int, long, long, long, long)", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_loop_dynamic_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_loop_dynamic_start_h == NULL)
      GOMP_parallel_loop_dynamic_start_h = dlsym(tau_handle,"GOMP_parallel_loop_dynamic_start"); 
    if (GOMP_parallel_loop_dynamic_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_parallel_loop_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }

}


/**********************************************************
   GOMP_parallel_loop_guided_start
 **********************************************************/

void  GOMP_parallel_loop_guided_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6, long a7)  {

  static void (*GOMP_parallel_loop_guided_start_h) (void (*)(void *), void *, unsigned int, long, long, long, long) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_loop_guided_start(void (*)(void *), void *, unsigned int, long, long, long, long)", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_loop_guided_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_loop_guided_start_h == NULL)
      GOMP_parallel_loop_guided_start_h = dlsym(tau_handle,"GOMP_parallel_loop_guided_start"); 
    if (GOMP_parallel_loop_guided_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_parallel_loop_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }

}


/**********************************************************
   GOMP_parallel_loop_runtime_start
 **********************************************************/

void  GOMP_parallel_loop_runtime_start(void (*a1)(void *), void * a2, unsigned int a3, long a4, long a5, long a6)  {

  static void (*GOMP_parallel_loop_runtime_start_h) (void (*)(void *), void *, unsigned int, long, long, long) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_loop_runtime_start(void (*)(void *), void *, unsigned int, long, long, long)", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_loop_runtime_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_loop_runtime_start_h == NULL)
      GOMP_parallel_loop_runtime_start_h = dlsym(tau_handle,"GOMP_parallel_loop_runtime_start"); 
    if (GOMP_parallel_loop_runtime_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_lock_environment();
      __ompc_event_callback(OMP_EVENT_THR_END_IDLE);
	  TAU_PROFILE_START(t); 
	  Tau_unlock_environment();
	}
    (*GOMP_parallel_loop_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }

}


/**********************************************************
   GOMP_loop_end
 **********************************************************/

void  GOMP_loop_end()  {

  static void (*GOMP_loop_end_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_loop_end()", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_loop_end_h == NULL)
      GOMP_loop_end_h = dlsym(tau_handle,"GOMP_loop_end"); 
    if (GOMP_loop_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  if (Tau_gomp_flags[omp_get_thread_num()].ordered) {
        ////__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
        __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
	    Tau_gomp_flags[omp_get_thread_num()].ordered = 0;
	  }
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_loop_end_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer();
	}
  }

}


/**********************************************************
   GOMP_loop_end_nowait
 **********************************************************/

void  GOMP_loop_end_nowait()  {

  static void (*GOMP_loop_end_nowait_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_loop_end_nowait()", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_end_nowait %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_loop_end_nowait_h == NULL)
      GOMP_loop_end_nowait_h = dlsym(tau_handle,"GOMP_loop_end_nowait"); 
    if (GOMP_loop_end_nowait_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  if (Tau_gomp_flags[omp_get_thread_num()].ordered) {
        //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
        __ompc_event_callback(OMP_EVENT_THR_END_ORDERED);
	    Tau_gomp_flags[omp_get_thread_num()].ordered = 0;
	  }
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_loop_end_nowait_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer();
	}
  }
}


/**********************************************************
   GOMP_loop_ull_static_start
 **********************************************************/

bool  GOMP_loop_ull_static_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static bool (*GOMP_loop_ull_static_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_static_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_static_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_static_start_h == NULL)
      GOMP_loop_ull_static_start_h = dlsym(tau_handle,"GOMP_loop_ull_static_start"); 
    if (GOMP_loop_ull_static_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_dynamic_start
 **********************************************************/

bool  GOMP_loop_ull_dynamic_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static bool (*GOMP_loop_ull_dynamic_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_dynamic_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_dynamic_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_dynamic_start_h == NULL)
      GOMP_loop_ull_dynamic_start_h = dlsym(tau_handle,"GOMP_loop_ull_dynamic_start"); 
    if (GOMP_loop_ull_dynamic_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_guided_start
 **********************************************************/

bool  GOMP_loop_ull_guided_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static bool (*GOMP_loop_ull_guided_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_guided_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_guided_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_guided_start_h == NULL)
      GOMP_loop_ull_guided_start_h = dlsym(tau_handle,"GOMP_loop_ull_guided_start"); 
    if (GOMP_loop_ull_guided_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_runtime_start
 **********************************************************/

bool  GOMP_loop_ull_runtime_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

  static bool (*GOMP_loop_ull_runtime_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_runtime_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_runtime_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_runtime_start_h == NULL)
      GOMP_loop_ull_runtime_start_h = dlsym(tau_handle,"GOMP_loop_ull_runtime_start"); 
    if (GOMP_loop_ull_runtime_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_static_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_static_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static bool (*GOMP_loop_ull_ordered_static_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_static_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_static_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_static_start_h == NULL)
      GOMP_loop_ull_ordered_static_start_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_static_start"); 
    if (GOMP_loop_ull_ordered_static_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_static_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_dynamic_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_dynamic_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static bool (*GOMP_loop_ull_ordered_dynamic_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_dynamic_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_dynamic_start_h == NULL)
      GOMP_loop_ull_ordered_dynamic_start_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_dynamic_start"); 
    if (GOMP_loop_ull_ordered_dynamic_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_dynamic_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_guided_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_guided_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long a5, unsigned long long * a6, unsigned long long * a7)  {

  static bool (*GOMP_loop_ull_ordered_guided_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_guided_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_guided_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_guided_start_h == NULL)
      GOMP_loop_ull_ordered_guided_start_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_guided_start"); 
    if (GOMP_loop_ull_ordered_guided_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_guided_start_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_runtime_start
 **********************************************************/

bool  GOMP_loop_ull_ordered_runtime_start(bool a1, unsigned long long a2, unsigned long long a3, unsigned long long a4, unsigned long long * a5, unsigned long long * a6)  {

  static bool (*GOMP_loop_ull_ordered_runtime_start_h) (bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_runtime_start(bool, unsigned long long, unsigned long long, unsigned long long, unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_runtime_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_runtime_start_h == NULL)
      GOMP_loop_ull_ordered_runtime_start_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_runtime_start"); 
    if (GOMP_loop_ull_ordered_runtime_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_runtime_start_h)( a1,  a2,  a3,  a4,  a5,  a6);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_static_next
 **********************************************************/

bool  GOMP_loop_ull_static_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_static_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_static_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_static_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_static_next_h == NULL)
      GOMP_loop_ull_static_next_h = dlsym(tau_handle,"GOMP_loop_ull_static_next"); 
    if (GOMP_loop_ull_static_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_static_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_dynamic_next
 **********************************************************/

bool  GOMP_loop_ull_dynamic_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_dynamic_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_dynamic_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_dynamic_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_dynamic_next_h == NULL)
      GOMP_loop_ull_dynamic_next_h = dlsym(tau_handle,"GOMP_loop_ull_dynamic_next"); 
    if (GOMP_loop_ull_dynamic_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_dynamic_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_guided_next
 **********************************************************/

bool  GOMP_loop_ull_guided_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_guided_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_guided_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_guided_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_guided_next_h == NULL)
      GOMP_loop_ull_guided_next_h = dlsym(tau_handle,"GOMP_loop_ull_guided_next"); 
    if (GOMP_loop_ull_guided_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_guided_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_runtime_next
 **********************************************************/

bool  GOMP_loop_ull_runtime_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_runtime_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_runtime_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_runtime_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_runtime_next_h == NULL)
      GOMP_loop_ull_runtime_next_h = dlsym(tau_handle,"GOMP_loop_ull_runtime_next"); 
    if (GOMP_loop_ull_runtime_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_runtime_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_static_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_static_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_ordered_static_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_static_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_static_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_static_next_h == NULL)
      GOMP_loop_ull_ordered_static_next_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_static_next"); 
    if (GOMP_loop_ull_ordered_static_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_static_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_dynamic_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_dynamic_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_ordered_dynamic_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_dynamic_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_dynamic_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_dynamic_next_h == NULL)
      GOMP_loop_ull_ordered_dynamic_next_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_dynamic_next"); 
    if (GOMP_loop_ull_ordered_dynamic_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_dynamic_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_guided_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_guided_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_ordered_guided_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_guided_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_guided_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_guided_next_h == NULL)
      GOMP_loop_ull_ordered_guided_next_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_guided_next"); 
    if (GOMP_loop_ull_ordered_guided_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_guided_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_loop_ull_ordered_runtime_next
 **********************************************************/

bool  GOMP_loop_ull_ordered_runtime_next(unsigned long long * a1, unsigned long long * a2)  {

  static bool (*GOMP_loop_ull_ordered_runtime_next_h) (unsigned long long *, unsigned long long *) = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_loop_ull_ordered_runtime_next(unsigned long long *, unsigned long long *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_loop_ull_ordered_runtime_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_loop_ull_ordered_runtime_next_h == NULL)
      GOMP_loop_ull_ordered_runtime_next_h = dlsym(tau_handle,"GOMP_loop_ull_ordered_runtime_next"); 
    if (GOMP_loop_ull_ordered_runtime_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_loop_ull_ordered_runtime_next_h)( a1,  a2);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_ordered_start
 **********************************************************/

void  GOMP_ordered_start()  {

  static void (*GOMP_ordered_start_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_ordered_start()", "", TAU_USER1);
  DEBUGPRINT("GOMP_ordered_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_ordered_start_h == NULL)
      GOMP_ordered_start_h = dlsym(tau_handle,"GOMP_ordered_start"); 
    if (GOMP_ordered_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // our turn to work in the ordered region!
      //__ompc_event_callback(OMP_EVENT_THR_END_ODWT);
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_ordered_start_h)();
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_ordered_end
 **********************************************************/

void  GOMP_ordered_end()  {

  static void (*GOMP_ordered_end_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_ordered_end()", "", TAU_USER1);
  DEBUGPRINT("GOMP_ordered_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_ordered_end_h == NULL)
      GOMP_ordered_end_h = dlsym(tau_handle,"GOMP_ordered_end"); 
    if (GOMP_ordered_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    (*GOMP_ordered_end_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer();
	  // wait for those after us to handle the ordered region...
      //__ompc_event_callback(OMP_EVENT_THR_BEGIN_ODWT);
	}
  }

}


/**********************************************************
   GOMP_parallel_start
 **********************************************************/

void  GOMP_parallel_start(void (*a1)(void *), void * a2, unsigned int a3)  {
  static void (*GOMP_parallel_start_h) (void (*)(void *), void *, unsigned int) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_start(void (*)(void *), void *, unsigned int)", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_start %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_start_h == NULL)
      GOMP_parallel_start_h = dlsym(tau_handle,"GOMP_parallel_start"); 
    if (GOMP_parallel_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
	// This is the beginning of a parallel region. We want to lock the environment,
	// because in the callback function AFTER the parallel_start call, we will be
	// starting parallel region timers for all the threads in the region. We want to
	// make sure those threads don't start timers before then, or else we will have
	// overlapping timers.
    if (Tau_global_get_insideTAU() == 0) { 
      Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_parallel_start_h)( a1,  a2,  a3);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
      DEBUGPRINT("GOMP_parallel_start %d of %d (on exit)\n", omp_get_thread_num(), omp_get_num_threads());
      // do this AFTER the start, so we know how many threads there are.
      __ompc_event_callback(OMP_EVENT_FORK);
	  // unlock the environment to let other threads progress
      Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_parallel_end
 **********************************************************/

void  GOMP_parallel_end()  {

  static void (*GOMP_parallel_end_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_end()", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_end %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_end_h == NULL)
      GOMP_parallel_end_h = dlsym(tau_handle,"GOMP_parallel_end"); 
    if (GOMP_parallel_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_parallel_end_h)();
    if (Tau_global_get_insideTAU() == 0) { 
      Tau_lock_environment();
	  Tau_stop_current_timer(); 
	  // do this at the end, so we can join all the threads.
      __ompc_event_callback(OMP_EVENT_JOIN);
      Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_task
 **********************************************************/

void  GOMP_task(void (*a1)(void *), void * a2, void (*a3)(void *, void *), long a4, long a5, bool a6, unsigned int a7)  {

  static void (*GOMP_task_h) (void (*)(void *), void *, void (*)(void *, void *), long, long, bool, unsigned int) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_task(void (*)(void *), void *, void (*)(void *, void *), long, long, bool, unsigned int)", "", TAU_USER1);
  DEBUGPRINT("GOMP_task %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_task_h == NULL)
      GOMP_task_h = dlsym(tau_handle,"GOMP_task"); 
    if (GOMP_task_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
  (*GOMP_task_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_taskwait
 **********************************************************/

void  GOMP_taskwait()  {

  static void (*GOMP_taskwait_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_taskwait()", "", TAU_USER1);
  DEBUGPRINT("GOMP_taskwait %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_taskwait_h == NULL)
      GOMP_taskwait_h = dlsym(tau_handle,"GOMP_taskwait"); 
    if (GOMP_taskwait_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
  (*GOMP_taskwait_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_taskyield
 **********************************************************/

void  GOMP_taskyield()  {

  static void (*GOMP_taskyield_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_taskyield()", "", TAU_USER1);
  DEBUGPRINT("GOMP_taskyield %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_taskyield_h == NULL)
      GOMP_taskyield_h = dlsym(tau_handle,"GOMP_taskyield"); 
    if (GOMP_taskyield_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
  (*GOMP_taskyield_h)();
  if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_sections_start
 **********************************************************/

unsigned int  GOMP_sections_start(unsigned int a1)  {

  static unsigned int (*GOMP_sections_start_h) (unsigned int) = NULL;
  unsigned int retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned int GOMP_sections_start(unsigned int)", "", TAU_USER1);
  DEBUGPRINT("GOMP_sections_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_sections_start_h == NULL)
      GOMP_sections_start_h = dlsym(tau_handle,"GOMP_sections_start"); 
    if (GOMP_sections_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_sections_start_h)( a1);
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_sections_next
 **********************************************************/

unsigned int  GOMP_sections_next()  {

  static unsigned int (*GOMP_sections_next_h) () = NULL;
  unsigned int retval = 0;
  TAU_PROFILE_TIMER(t,"unsigned int GOMP_sections_next()", "", TAU_USER1);
  DEBUGPRINT("GOMP_sections_next %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_sections_next_h == NULL)
      GOMP_sections_next_h = dlsym(tau_handle,"GOMP_sections_next"); 
    if (GOMP_sections_next_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    retval  =  (*GOMP_sections_next_h)();
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }
  return retval;

}


/**********************************************************
   GOMP_parallel_sections_start
 **********************************************************/

void  GOMP_parallel_sections_start(void (*a1)(void *), void * a2, unsigned int a3, unsigned int a4)  {

  static void (*GOMP_parallel_sections_start_h) (void (*)(void *), void *, unsigned int, unsigned int) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_parallel_sections_start(void (*)(void *), void *, unsigned int, unsigned int)", "", TAU_USER1);
  DEBUGPRINT("GOMP_parallel_sections_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_parallel_sections_start_h == NULL)
      GOMP_parallel_sections_start_h = dlsym(tau_handle,"GOMP_parallel_sections_start"); 
    if (GOMP_parallel_sections_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { 
      Tau_lock_environment();
	  TAU_PROFILE_START(t); 
	}
    (*GOMP_parallel_sections_start_h)( a1,  a2,  a3,  a4);
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
      // do this AFTER the start, so we know how many threads there are.
      __ompc_event_callback(OMP_EVENT_FORK);
      Tau_unlock_environment();
	}
  }

}


/**********************************************************
   GOMP_sections_end
 **********************************************************/

void  GOMP_sections_end()  {

  static void (*GOMP_sections_end_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_sections_end()", "", TAU_USER1);
  DEBUGPRINT("GOMP_sections_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_sections_end_h == NULL)
      GOMP_sections_end_h = dlsym(tau_handle,"GOMP_sections_end"); 
    if (GOMP_sections_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    (*GOMP_sections_end_h)();
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_sections_end_nowait
 **********************************************************/

void  GOMP_sections_end_nowait()  {

  static void (*GOMP_sections_end_nowait_h) () = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_sections_end_nowait()", "", TAU_USER1);
  DEBUGPRINT("GOMP_sections_end_nowait %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_sections_end_nowait_h == NULL)
      GOMP_sections_end_nowait_h = dlsym(tau_handle,"GOMP_sections_end_nowait"); 
    if (GOMP_sections_end_nowait_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
    if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
    (*GOMP_sections_end_nowait_h)();
    if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}


/**********************************************************
   GOMP_single_start
 **********************************************************/

bool  GOMP_single_start()  {

  static bool (*GOMP_single_start_h) () = NULL;
  bool retval = 0;
  TAU_PROFILE_TIMER(t,"bool GOMP_single_start()", "", TAU_USER1);
  DEBUGPRINT("GOMP_single_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_single_start_h == NULL)
      GOMP_single_start_h = dlsym(tau_handle,"GOMP_single_start"); 
    if (GOMP_single_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  // in this case, the single region is entered and executed
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
      TAU_PROFILE_START(t); 
	}
    retval  =  (*GOMP_single_start_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	  // the code is done, so fire the callback end event
      __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);
	}
  }
  return retval;

}


/**********************************************************
   GOMP_single_copy_start
 **********************************************************/

void *  GOMP_single_copy_start()  {

  static void * (*GOMP_single_copy_start_h) () = NULL;
  void * retval = 0;
  TAU_PROFILE_TIMER(t,"void *GOMP_single_copy_start()", "", TAU_USER1);
  DEBUGPRINT("GOMP_single_copy_start %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } else { 
    if (GOMP_single_copy_start_h == NULL)
      GOMP_single_copy_start_h = dlsym(tau_handle,"GOMP_single_copy_start"); 
    if (GOMP_single_copy_start_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
    if (Tau_global_get_insideTAU() == 0) { 
	  TAU_PROFILE_START(t); 
      __ompc_event_callback(OMP_EVENT_THR_BEGIN_SINGLE);
	}
    retval  =  (*GOMP_single_copy_start_h)();
    if (Tau_global_get_insideTAU() == 0) { 
	  Tau_stop_current_timer(); 
	}
  }
  return retval;

}


/**********************************************************
   GOMP_single_copy_end
 **********************************************************/

void  GOMP_single_copy_end(void * a1)  {

  static void (*GOMP_single_copy_end_h) (void *) = NULL;
  TAU_PROFILE_TIMER(t,"void GOMP_single_copy_end(void *)", "", TAU_USER1);
  DEBUGPRINT("GOMP_single_copy_end %d\n", omp_get_thread_num());
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } else { 
    if (GOMP_single_copy_end_h == NULL)
      GOMP_single_copy_end_h = dlsym(tau_handle,"GOMP_single_copy_end"); 
    if (GOMP_single_copy_end_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  __ompc_event_callback(OMP_EVENT_THR_END_SINGLE);
  if (Tau_global_get_insideTAU() == 0) { TAU_PROFILE_START(t); }
  (*GOMP_single_copy_end_h)( a1);
  if (Tau_global_get_insideTAU() == 0) { Tau_stop_current_timer(); }
  }

}

