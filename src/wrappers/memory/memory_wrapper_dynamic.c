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

// Must be defined before dlfcn.h to get RTLD_NEXT
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


#if 1
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
#endif


/******************************************************************************
 * libc memory allocation/deallocation wrappers
 ******************************************************************************/

#ifdef HAVE_MALLOC
void * malloc(size_t size)
{
  return malloc_wrapper(size);
}
#endif

#ifdef HAVE_CALLOC
void * calloc(size_t count, size_t size)
{
  return calloc_wrapper(count, size);
}
#endif

#ifdef HAVE_REALLOC
void * realloc(void * ptr, size_t size)
{
  return realloc_wrapper(ptr, size);
}
#endif

#ifdef HAVE_FREE
void free(void * ptr)
{
  free_wrapper(ptr);
}
#endif

#ifdef HAVE_MEMALIGN
void * memalign(size_t alignment, size_t size)
{
  return memalign_wrapper(alignment, size);
}
#endif

#ifdef HAVE_POSIX_MEMALIGN
int posix_memalign(void **ptr, size_t alignment, size_t size)
{
  return posix_memalign_wrapper(ptr, alignment, size);
}
#endif

#ifdef HAVE_VALLOC
void * valloc(size_t size)
{
  return valloc_wrapper(size);
}
#endif

#ifdef HAVE_PVALLOC
void * pvalloc(size_t size)
{
  return pvalloc_wrapper(size);
}
#endif

#ifdef HAVE_PUTS
int puts(const char *s)
{
  return puts_wrapper(s);
}
#endif


/******************************************************************************
 *
 ******************************************************************************/

malloc_t get_system_malloc()
{
#ifdef HAVE_MALLOC
  return (malloc_t)get_system_function_handle("malloc");
#else
  return NULL;
#endif
}

calloc_t get_system_calloc()
{
#ifdef HAVE_CALLOC
  return (calloc_t)get_system_function_handle("calloc");
#else
  return NULL;
#endif
}

realloc_t get_system_realloc()
{
#ifdef HAVE_REALLOC
  return (realloc_t)get_system_function_handle("realloc");
#else
  return NULL;
#endif
}

memalign_t get_system_memalign()
{
#ifdef HAVE_MEMALIGN
  return (memalign_t)get_system_function_handle("memalign");
#else
  return NULL;
#endif
}

posix_memalign_t get_system_posix_memalign()
{
#ifdef HAVE_POSIX_MEMALIGN
  return (posix_memalign_t)get_system_function_handle("posix_memalign");
#else
  return NULL;
#endif
}

valloc_t get_system_valloc()
{
#ifdef HAVE_VALLOC
  return (valloc_t)get_system_function_handle("valloc");
#else
  return NULL;
#endif
}

pvalloc_t get_system_pvalloc()
{
#ifdef HAVE_PVALLOC
  return (pvalloc_t)get_system_function_handle("pvalloc");
#else
  return NULL;
#endif
}

puts_t get_system_puts()
{
#ifdef HAVE_PUTS
  return (puts_t)get_system_function_handle("puts");
#else
  return NULL;
#endif
}

free_t get_system_free()
{
#ifdef HAVE_FREE
  return (free_t)get_system_function_handle("free");
#else
  return NULL;
#endif
}


/******************************************************************************
 * pthread wrappers
 ******************************************************************************/

#if 0
int pthread_getattr_np(pthread_t thread, pthread_attr_t *attr)
{
  typedef int (*pthread_getattr_np_t)(pthread_t, pthread_attr_t*);
  static pthread_getattr_np_t pthread_getattr_np_system = NULL;

  int retval;

  memory_wrapper_disable();

  if (!pthread_getattr_np_system) {
    pthread_getattr_np_system = (pthread_getattr_np_t)get_system_function_handle("pthread_getattr_np");
  }

  retval = pthread_getattr_np_system(thread, attr);

  memory_wrapper_enable();

  return retval;
}

int pthread_attr_destroy(pthread_attr_t *attr)
{
  typedef int (*pthread_attr_destroy_t)(pthread_attr_t *);
  static pthread_attr_destroy_t pthread_attr_destroy_system = NULL;

  int retval;

  memory_wrapper_disable();

  if (!pthread_attr_destroy_system) {
    pthread_attr_destroy_system = (pthread_attr_destroy_t)get_system_function_handle("pthread_attr_destroy");
  }

  retval = pthread_attr_destroy_system(attr);

  memory_wrapper_enable();

  return retval;
}

int pthread_attr_init(pthread_attr_t *attr)
{
  typedef int (*pthread_attr_init_t)(pthread_attr_t *);
  static pthread_attr_init_t pthread_attr_init_system = NULL;

  int retval;

  memory_wrapper_disable();

  if (!pthread_attr_init_system) {
    pthread_attr_init_system = (pthread_attr_init_t)get_system_function_handle("pthread_attr_init");
  }

  retval = pthread_attr_init_system(attr);

  memory_wrapper_enable();

  return retval;
}
#endif

/*********************************************************************
 * EOF
 ********************************************************************/
