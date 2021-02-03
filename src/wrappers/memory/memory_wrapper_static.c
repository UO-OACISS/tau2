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
#ifdef HAVE_PUTS
extern int __real_puts(const char *s);
#endif



malloc_t get_system_malloc()
{
#ifdef HAVE_MALLOC
  return __real_malloc;
#else
  return NULL;
#endif
}

calloc_t get_system_calloc()
{
#ifdef HAVE_CALLOC
  return __real_calloc;
#else
  return NULL;
#endif
}

realloc_t get_system_realloc()
{
#ifdef HAVE_REALLOC
  return __real_realloc;
#else
  return NULL;
#endif
}

memalign_t get_system_memalign()
{
#ifdef HAVE_MEMALIGN
  return __real_memalign;
#else
  return NULL;
#endif
}

posix_memalign_t get_system_posix_memalign()
{
#ifdef HAVE_POSIX_MEMALIGN
  return __real_posix_memalign;
#else
  return NULL;
#endif
}

valloc_t get_system_valloc()
{
#ifdef HAVE_VALLOC
  return __real_valloc;
#else
  return NULL;
#endif
}

pvalloc_t get_system_pvalloc()
{
#ifdef HAVE_PVALLOC
  return __real_pvalloc;
#else
  return NULL;
#endif
}

puts_t get_system_puts()
{
#ifdef HAVE_PUTS
  return __real_puts;
#else
  return NULL;
#endif
}

free_t get_system_free()
{
#ifdef HAVE_FREE
  return __real_free;
#else
  return NULL;
#endif
}

void * __wrap_malloc(size_t size)
{
  return malloc_wrapper(size);
}

void * __wrap_calloc(size_t count, size_t size)
{
  return calloc_wrapper(count, size);
}

void __wrap_free(void * ptr)
{
  free_wrapper(ptr);
}

void * __wrap_memalign(size_t alignment, size_t size)
{
  return memalign_wrapper(alignment, size);
}

int __wrap_posix_memalign(void **ptr, size_t alignment, size_t size)
{
  return posix_memalign_wrapper(ptr, alignment, size);
}

void * __wrap_realloc(void * ptr, size_t size)
{
  return realloc_wrapper(ptr, size);
}

void * __wrap_valloc(size_t size)
{
  return valloc_wrapper(size);
}

void * __wrap_pvalloc(size_t size)
{
  return pvalloc_wrapper(size);
}

int __wrap_puts(const char *s)
{
  return puts_wrapper(s);
}

/*********************************************************************
 * EOF
 ********************************************************************/
