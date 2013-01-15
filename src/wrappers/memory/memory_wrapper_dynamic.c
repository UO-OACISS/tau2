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


int Tau_memory_wrapper_init(void)
{
  static int init = 0;
  if (init) return 0;
  init = 1;

  if (Tau_init_check_dl_initialized()) {
    Tau_global_incr_insideTAU();
    Tau_init_initializeTAU();
    Tau_create_top_level_timer_if_necessary();
    Tau_set_memory_wrapper_present(1);
    Tau_global_decr_insideTAU();
    return 0;
  }
  return 1;
}

int Tau_memory_wrapper_passthrough(void)
{
  return Tau_global_get_insideTAU()
      || !Tau_init_check_initialized()
      || !Tau_init_check_dl_initialized()
      || Tau_global_getLightsOut();
}


static inline
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





/*********************************************************************
 * EOF
 ********************************************************************/
