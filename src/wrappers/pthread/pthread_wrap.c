/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997  						   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: pthread_wrap.c				  **
 **	Description 	: TAU Profiling Package RTS Layer definitions     **
 **			  for wrapping syscalls like exit                 **
 **	Contact		: tau-team@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <Profile/TauEnv.h>

#if !defined(__APPLE__)
#define TAU_PTHREAD_BARRIER_AVAILABLE
#endif

typedef void * (*start_routine_p)(void *);
typedef int (*pthread_create_p)(pthread_t *, const pthread_attr_t *, start_routine_p, void *arg);
typedef void (*pthread_exit_p)(void *);
typedef int (*pthread_join_p)(pthread_t, void **);
#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
typedef int (*pthread_barrier_wait_p)(pthread_barrier_t *);
#endif

extern int tau_pthread_create_wrapper(pthread_create_p pthread_create_call,
    pthread_t * threadp, const pthread_attr_t * attr, start_routine_p, void * arg);
extern int tau_pthread_join_wrapper(pthread_join_p pthread_join_call, pthread_t thread, void **retval);
extern void tau_pthread_exit_wrapper(pthread_exit_p pthread_exit_call, void * value_ptr);
#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
extern int tau_pthread_barrier_wait_wrapper(pthread_barrier_wait_p pthread_barrier_wait_call,
    pthread_barrier_t * barrier);
#endif


#ifdef TAU_PRELOAD_LIB
/********************************/
/* LD_PRELOAD wrapper functions */
/********************************/

#define RESET_DLERROR() dlerror()
#define CHECK_DLERROR() { \
  char const * err = dlerror(); \
  if (err) { \
    printf("Error getting %s handle: %s\n", name, err); \
    fflush(stdout); \
    exit(1); \
  } \
}

static
void * get_system_function_handle(char const * name, void * caller)
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

int pthread_create(pthread_t* thread, const pthread_attr_t* attr,
    start_routine_p start_routine, void* arg)
{
  static pthread_create_p _pthread_create = NULL;
  if (!_pthread_create) {
    _pthread_create = (pthread_create_p)get_system_function_handle(
        "pthread_create", (void*)pthread_create);
  }
  return tau_pthread_create_wrapper(_pthread_create, thread, attr, start_routine, arg);
}

int pthread_join(pthread_t thread, void ** retval)
{
  static pthread_join_p _pthread_join = NULL;
  if (!_pthread_join) {
    _pthread_join = (pthread_join_p)get_system_function_handle(
        "pthread_join", (void*)pthread_join);
  }
  return tau_pthread_join_wrapper(_pthread_join, thread, retval);
}

void pthread_exit(void * value_ptr)
{
  static pthread_exit_p _pthread_exit = NULL;
  if (!_pthread_exit) {
    _pthread_exit = (pthread_exit_p)get_system_function_handle(
        "pthread_exit", (void*)pthread_exit);
  }
  tau_pthread_exit_wrapper(_pthread_exit, value_ptr);
}

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
int pthread_barrier_wait(pthread_barrier_t * barrier)
{
  static pthread_barrier_wait_p _pthread_barrier_wait = NULL;
  if (!_pthread_barrier_wait) {
    _pthread_barrier_wait = (pthread_barrier_wait_p)get_system_function_handle(
        "pthread_barrier_wait", (void*)pthread_barrier_wait);
  }
  return tau_pthread_barrier_wait_wrapper(_pthread_barrier_wait, barrier);
}
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

#else // Wrap via the the link line.

#ifndef TAU_SUPPRESS_PTHREAD_CREATE_WRAPPER
int __real_pthread_create(pthread_t *, const pthread_attr_t *, start_routine_p, void *);
int __wrap_pthread_create(pthread_t * thread, const pthread_attr_t * attr, start_routine_p start_routine, void * arg)
{
  return tau_pthread_create_wrapper(__real_pthread_create, thread, attr, start_routine, arg);
}
/*
#else 
int __real___wrap_pthread_create(pthread_t *, const pthread_attr_t *, start_routine_p, void *);
int __wrap___wrap_pthread_create(pthread_t * thread, const pthread_attr_t * attr, start_routine_p start_routine, void * arg)
{
  printf("Inside __wrap___wrap_pthread_create\n");
  return tau_pthread_create_wrapper(__real___wrap_pthread_create, thread, attr, start_routine, arg);
}
*/
#endif /* TAU_WRAP_PTHREAD_CREATE */

int __real_pthread_join(pthread_t, void **);
int __wrap_pthread_join(pthread_t thread, void **retval)
{
  return tau_pthread_join_wrapper(__real_pthread_join, thread, retval);
}

void __real_pthread_exit(void *);
void __wrap_pthread_exit(void * value_ptr)
{
  tau_pthread_exit_wrapper(__real_pthread_exit, value_ptr);
}

#ifdef TAU_PTHREAD_BARRIER_AVAILABLE
int __real_pthread_barrier_wait(pthread_barrier_t *);
int __wrap_pthread_barrier_wait(pthread_barrier_t * barrier)
{
  return tau_pthread_barrier_wait_wrapper(__real_pthread_barrier_wait, barrier);
}
#endif /* TAU_PTHREAD_BARRIER_AVAILABLE */

#endif //TAU_PRELOAD_LIB
