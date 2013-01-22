/****************************************************************************
**      TAU Portable Profiling Package         **
**      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2010                     **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**  File          : memory_wrapper_dynamic.c
**  Description   : TAU Profiling Package
**  Contact       : tau-bugs@cs.uoregon.edu
**  Documentation : See http://www.cs.uoregon.edu/research/tau
**
**  Description   : TAU memory profiler and debugger
**
****************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>

#include <TAU.h>
#include <Profile/Profiler.h>
#include <Profile/TauMemory.h>
#include <memory_wrapper.h>

#ifdef strcpy
#undef strcpy
#endif

#ifdef _MSC_VER
/* define these functions as non-intrinsic */
#pragma function( memcpy, strcpy, strcat )
#endif


int Tau_memory_wrapper_init(void)
{
  static int init = 0;
  if (init) return 0;
  init = 1;

  if (Tau_init_check_dl_initialized()) {
    Tau_global_incr_insideTAU();
    Tau_init_initializeTAU();
    Tau_create_top_level_timer_if_necessary();
    Tau_memory_set_wrapper_present(1);
    Tau_global_decr_insideTAU();
    return 0;
  }
  return 1;
}

int Tau_memory_wrapper_passthrough(void)
{
#ifdef TAU_OPENMP
  static int work_around = 0;
  int retval;

  if (work_around) return work_around;
  ++work_around;

  // The order of these statements is important
  retval = !Tau_init_check_dl_initialized()
      || !Tau_init_check_initialized()
      || Tau_global_getLightsOut()
      || Tau_global_get_insideTAU();

  --work_around;
  return retval;

#else

  return !Tau_init_check_dl_initialized()
      || !Tau_init_check_initialized()
      || Tau_global_getLightsOut()
      || Tau_global_get_insideTAU();

#endif
}


void * get_system_function_handle(char const * name)
{
  char const * err;
  void * handle;

  // Reset error pointer
  dlerror();

  // Attempt to get the function handle
  handle = dlsym(RTLD_NEXT, name);

  // Detect errors
  if ((err = dlerror())) {
    // These calls are unsafe, but we're about to die anyway.
    printf("Error getting %s handle: %s\n", name, err);
    fflush(stdout);
    exit(1);
  }

  return handle;
}


/******************************************************************************
 * libc memory allocation/deallocation wrappers
 ******************************************************************************/


void * malloc(size_t size)
{
  return malloc_handle(size);
}

void * calloc(size_t count, size_t size)
{
  return calloc_handle(count, size);
}

void free(void * ptr)
{
  return free_handle(ptr);
}

#ifdef HAVE_MEMALIGN
void * memalign(size_t alignment, size_t size)
{
  return memalign_handle(alignment, size);
}
#endif

int posix_memalign(void **ptr, size_t alignment, size_t size)
{
  return posix_memalign_handle(ptr, alignment, size);
}

void * realloc(void * ptr, size_t size)
{
  return realloc_handle(ptr, size);
}

void * valloc(size_t size)
{
  return valloc_handle(size);
}

#ifdef HAVE_PVALLOC
void * pvalloc(size_t size)
{
  return pvalloc_handle(size);
}
#endif


/******************************************************************************
 *
 ******************************************************************************/


malloc_t Tau_get_system_malloc()
{
  return (malloc_t)get_system_function_handle("malloc");
}

calloc_t Tau_get_system_calloc()
{
  return (calloc_t)get_system_function_handle("calloc");
}

realloc_t Tau_get_system_realloc()
{
  return (realloc_t)get_system_function_handle("realloc");
}

memalign_t Tau_get_system_memalign()
{
  return (memalign_t)get_system_function_handle("memalign");
}

posix_memalign_t Tau_get_system_posix_memalign()
{
  return (posix_memalign_t)get_system_function_handle("posix_memalign");
}

valloc_t Tau_get_system_valloc()
{
  return (valloc_t)get_system_function_handle("valloc");
}

pvalloc_t Tau_get_system_pvalloc()
{
  return (pvalloc_t)get_system_function_handle("pvalloc");
}

free_t Tau_get_system_free()
{
  return (free_t)get_system_function_handle("free");
}


/******************************************************************************
 * pthread wrappers 
 ******************************************************************************/


int pthread_getattr_np(pthread_t thread, pthread_attr_t *attr)
{
  typedef int (*pthread_getattr_np_t)(pthread_t, pthread_attr_t*);
  static pthread_getattr_np_t pthread_getattr_np_system = NULL;

  int retval;

  Tau_memory_wrapper_disable();

  if (!pthread_getattr_np_system) {
    pthread_getattr_np_system = (pthread_getattr_np_t)get_system_function_handle("pthread_getattr_np");
  }

  retval = pthread_getattr_np_system(thread, attr);
  
  Tau_memory_wrapper_enable();

  return retval;
}

int pthread_attr_destroy(pthread_attr_t *attr)
{
  typedef int (*pthread_attr_destroy_t)(pthread_attr_t *);
  static pthread_attr_destroy_t pthread_attr_destroy_system = NULL;

  int retval;

  Tau_memory_wrapper_disable();

  if (!pthread_attr_destroy_system) {
    pthread_attr_destroy_system = (pthread_attr_destroy_t)get_system_function_handle("pthread_attr_destroy");
  }

  retval = pthread_attr_destroy_system(attr);

  Tau_memory_wrapper_enable();

  return retval;
}

int pthread_attr_init(pthread_attr_t *attr)
{
  typedef int (*pthread_attr_init_t)(pthread_attr_t *);
  static pthread_attr_init_t pthread_attr_init_system = NULL;

  int retval;

  Tau_memory_wrapper_disable();

  if (!pthread_attr_init_system) {
    pthread_attr_init_system = (pthread_attr_init_t)get_system_function_handle("pthread_attr_init");
  }

  retval = pthread_attr_init_system(attr);

  Tau_memory_wrapper_enable();

  return retval;
}


/*********************************************************************
 * EOF
 ********************************************************************/
