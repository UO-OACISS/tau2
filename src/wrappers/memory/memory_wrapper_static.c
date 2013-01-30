/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		      : memory_wrapper_static.c
**	Description 	: TAU Profiling Package
**	Contact		    : tau-bugs@cs.uoregon.edu
**	Documentation	: See http://www.cs.uoregon.edu/research/tau
**
**  Description   : TAU memory profiler and debugger
**
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <TAU.h>
#include <Profile/Profiler.h>
#include <Profile/TauMemory.h>
#include <memory_wrapper.h>

#ifdef HAVE_MALLOC
extern void * __real_malloc(size_t size);
#endif
#ifdef HAVE_CALLOC
extern void * __real_calloc(size_t count, size_t size);
#endif
#ifdef HAVE_REALLOC
extern void * __real_realloc(void * ptr, size_t size);
#endif
#ifdef HAVE_FREE
extern void __real_free(void * ptr);
#endif
#ifdef HAVE_MEMALIGN
extern void * __real_memalign(size_t alignment, size_t size);
#endif
#ifdef HAVE_POSIX_MEMALIGN
extern int __real_posix_memalign(void **ptr, size_t alignment, size_t size);
#endif
#ifdef HAVE_VALLOC
extern void * __real_valloc(size_t size);
#endif
#ifdef HAVE_PVALLOC
extern void * __real_pvalloc(size_t size);
#endif


int Tau_memory_wrapper_init(void)
{
  static int init = 0;
  if (init) return 0;
  init = 1;

  Tau_global_incr_insideTAU();
  Tau_init_initializeTAU();
  Tau_create_top_level_timer_if_necessary();
  Tau_memory_set_wrapper_present(1);
  Tau_global_decr_insideTAU();
  return 0;
}

int Tau_memory_wrapper_passthrough(void)
{
  // The order of these statements is important
  return !Tau_init_check_initialized()
      || Tau_global_getLightsOut()
      || Tau_global_get_insideTAU();
}

malloc_t Tau_get_system_malloc()
{
#ifdef HAVE_MALLOC
  return __real_malloc;
#else
  return NULL;
#endif
}

calloc_t Tau_get_system_calloc()
{
#ifdef HAVE_CALLOC
  return __real_calloc;
#else
  return NULL;
#endif
}

realloc_t Tau_get_system_realloc()
{
#ifdef HAVE_REALLOC
  return __real_realloc;
#else
  return NULL;
#endif
}

memalign_t Tau_get_system_memalign()
{
#ifdef HAVE_MEMALIGN
  return __real_memalign;
#else
  return NULL;
#endif
}

posix_memalign_t Tau_get_system_posix_memalign()
{
#ifdef HAVE_POSIX_MEMALIGN
  return __real_posix_memalign;
#else
  return NULL;
#endif
}

valloc_t Tau_get_system_valloc()
{
#ifdef HAVE_VALLOC
  return __real_valloc;
#else
  return NULL;
#endif
}

pvalloc_t Tau_get_system_pvalloc()
{
#ifdef HAVE_PVALLOC
  return __real_pvalloc;
#else
  return NULL;
#endif
}

free_t Tau_get_system_free()
{
#ifdef HAVE_FREE
  return __real_free;
#else
  return NULL;
#endif
}

void * __wrap_malloc(size_t size)
{
  return malloc_handle(size);
}

void * __wrap_calloc(size_t count, size_t size)
{
  return calloc_handle(count, size);
}

void __wrap_free(void * ptr)
{
  return free_handle(ptr);
}

void * __wrap_memalign(size_t alignment, size_t size)
{
  return memalign_handle(alignment, size);
}

int __wrap_posix_memalign(void **ptr, size_t alignment, size_t size)
{
  return posix_memalign_handle(ptr, alignment, size);
}

void * __wrap_realloc(void * ptr, size_t size)
{
  return realloc_handle(ptr, size);
}

void * __wrap_valloc(size_t size)
{
  return valloc_handle(size);
}

void * __wrap_pvalloc(size_t size)
{
  return pvalloc_handle(size);
}

/*********************************************************************
 * EOF
 ********************************************************************/
