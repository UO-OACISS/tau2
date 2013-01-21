/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		      : memory_wrapper.cpp
**	Description 	: TAU Profiling Package
**	Contact		    : tau-bugs@cs.uoregon.edu
**	Documentation	: See http://www.cs.uoregon.edu/research/tau
**
**  Description   : TAU memory profiler and debugger
**
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
  
#include <memory.h>
#if (defined(__APPLE_CC__) || defined(TAU_APPLE_XLC) || defined(TAU_APPLE_PGI))
#include <malloc/malloc.h>
#elif defined(TAU_FREEBSD)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <TAU.h>
#include <Profile/Profiler.h>
#include <Profile/TauMemory.h>
#include "memory_wrapper.h"

// Assume 4K pages unless we know otherwise.
// We cannot determine this at runtime because it must be known during
// the bootstrap process and it would be unsafe to make any system calls there.
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

// Size of heap memory for library wrapper bootstrapping
#define BOOTSTRAP_HEAP_SIZE (3*PAGE_SIZE)

// Bootstrap routines
void * malloc_bootstrap(size_t size);
void * calloc_bootstrap(size_t count, size_t size);
void   free_bootstrap(void * ptr);
void * memalign_bootstrap(size_t alignment, size_t size);
int    posix_memalign_bootstrap(void **ptr, size_t alignment, size_t size);
void * realloc_bootstrap(void * ptr, size_t size);
void * valloc_bootstrap(size_t size);
void * pvalloc_bootstrap(size_t size);

#if 0
int strcmp_bootstrap(const char *s1, const char *s2);
#endif

// Handles to function implementations that are called
// when the wrapped function is invoked.
// Everybody starts in the bootstrap state.
malloc_t malloc_handle = malloc_bootstrap;
calloc_t calloc_handle = calloc_bootstrap;
free_t free_handle = free_bootstrap;
memalign_t memalign_handle = memalign_bootstrap;
posix_memalign_t posix_memalign_handle = posix_memalign_bootstrap;
realloc_t realloc_handle = realloc_bootstrap;
valloc_t valloc_handle = valloc_bootstrap;
pvalloc_t pvalloc_handle = pvalloc_bootstrap;

#if 0
strcmp_t strcmp_handle = strcmp_bootstrap;
#endif

// Handles to the system implementation of the function.
// These are initialized during bootstrap.
malloc_t malloc_system = NULL;
calloc_t calloc_system = NULL;
free_t free_system = NULL;
memalign_t memalign_system = NULL;
posix_memalign_t posix_memalign_system = NULL;
realloc_t realloc_system = NULL;
valloc_t valloc_system = NULL;
pvalloc_t pvalloc_system = NULL;

#if 0
strcmp_t strcmp_system = NULL;
#endif

// Memory for bootstrapping.  Must not be static.
char bootstrap_heap[BOOTSTRAP_HEAP_SIZE];
char * bootstrap_base = bootstrap_heap;


static inline
int is_bootstrap(void * ptr)
{
  char const * const p = (char*)ptr;
  return (p < bootstrap_heap + BOOTSTRAP_HEAP_SIZE) && (bootstrap_heap < p);
}

static inline
void * bootstrap_alloc(size_t align, size_t size)
{
  char * ptr;

  // Check alignment.  Default alignment is sizeof(long)
  if(!align) {
    align = sizeof(long);

    if (size < align) {
      // Align to the next lower power of two
      align = size;
      while (align & (align-1)) {
        align &= align-1;
      }
    }
  }

  // Calculate address
  ptr = (char*)(((size_t)bootstrap_base + (align-1)) & ~(align-1));
  bootstrap_base = ptr + size;

  // Check for overflow
  if ((ptr + size) >= (bootstrap_heap + BOOTSTRAP_HEAP_SIZE)) {
    // These calls are unsafe, but we're about to die anyway.
    printf("TAU bootstreap heap exceeded.  Increase BOOTSTRAP_HEAP_SIZE in " __FILE__ " and try again.\n");
    fflush(stdout);
    exit(1);
  }

  return (void*)ptr;
}

static inline
void bootstrap_free(void * ptr)
{
//  if (is_bootstrap(ptr)) {
//    // Do nothing: bootstrap memory is deallocated on program exit
//  }
}

/*********************************************************************
 * malloc
 ********************************************************************/

void * malloc_init(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    malloc_system = Tau_get_system_malloc();
  }

  if (!malloc_system) {
    return bootstrap_alloc(0, size);
  }

  return NULL; // Indicates success
}

void * malloc_enabled(size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return malloc_system(size);
  }
  return Tau_malloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * malloc_disabled(size_t size)
{
  void * ptr = malloc_init(size);
  if (ptr) return ptr;

  return malloc_system(size);
}

void * malloc_bootstrap(size_t size)
{
  void * ptr = malloc_init(size);
  if (ptr) return ptr;

  if (Tau_memory_wrapper_init()) {
    return malloc_system(size);
  }

  malloc_handle = malloc_enabled;
  return malloc_enabled(size);
}

/*********************************************************************
 * calloc
 ********************************************************************/

void * calloc_init(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    calloc_system = Tau_get_system_calloc();
  }

  if (!calloc_system) {
    char * ptr = (char*)bootstrap_alloc(0, size);
    char const * const end = ptr + size;
    char * p = ptr;
    while (p < end) {
      *p = (char)0;
      ++p;
    }
    return (void *)ptr;
  }

  return NULL; // Indicates success
}

void * calloc_enabled(size_t count, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return calloc_system(count, size);
  }
  return Tau_calloc(count, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * calloc_disabled(size_t count, size_t size)
{
  void * ptr = calloc_init(count*size);
  if (ptr) return ptr;

  return calloc_system(count, size);
}

void * calloc_bootstrap(size_t count, size_t size)
{
  void * ptr = calloc_init(count*size);
  if (ptr) return ptr;

  if (Tau_memory_wrapper_init()) {
    return calloc_system(count, size);
  }

  calloc_handle = calloc_enabled;
  return calloc_enabled(count, size);
}

/*********************************************************************
 * realloc
 ********************************************************************/

void * realloc_init(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    realloc_system = Tau_get_system_realloc();
  }

  if (!realloc_system) {
    return bootstrap_alloc(0, size);
  }

  return NULL; // Indicates success
}

void * realloc_enabled(void * ptr, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return realloc_system(ptr, size);
  }
  return Tau_realloc(ptr, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * realloc_disabled(void * ptr, size_t size)
{
  void * iptr = realloc_init(size);
  if (iptr) return iptr;

  return realloc_system(ptr, size);
}

void * realloc_bootstrap(void * ptr, size_t size)
{
  void * iptr = realloc_init(size);
  if (iptr) return iptr;

  if (Tau_memory_wrapper_init()) {
    return realloc_system(ptr, size);
  }

  realloc_handle = realloc_enabled;
  return realloc_enabled(ptr, size);
}

/*********************************************************************
 * memalign
 ********************************************************************/

void * memalign_init(size_t alignment, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    memalign_system = Tau_get_system_memalign();
  }

  if (!memalign_system) {
    return bootstrap_alloc(alignment, size);
  }

  return NULL; // Indicates success
}

void * memalign_enabled(size_t alignment, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return memalign_system(alignment, size);
  }
  return Tau_memalign(alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * memalign_disabled(size_t alignment, size_t size)
{
  void * iptr = memalign_init(alignment, size);
  if (iptr) return iptr;

  return memalign_system(alignment, size);
}

void * memalign_bootstrap(size_t alignment, size_t size)
{
  void * iptr = memalign_init(alignment, size);
  if (iptr) return iptr;

  if (Tau_memory_wrapper_init()) {
    return memalign_system(alignment, size);
  }

  memalign_handle = memalign_enabled;
  return memalign_enabled(alignment, size);
}

/*********************************************************************
 * posix_memalign
 ********************************************************************/

void * posix_memalign_init(size_t alignment, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    posix_memalign_system = Tau_get_system_posix_memalign();
  }

  if (!posix_memalign_system) {
    return bootstrap_alloc(alignment, size);
  }

  return NULL; // Indicates success
}

int posix_memalign_enabled(void ** ptr, size_t alignment, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return posix_memalign_system(ptr, alignment, size);
  }
  return Tau_posix_memalign(ptr, alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

int posix_memalign_disabled(void ** ptr, size_t alignment, size_t size)
{
  void * iptr = posix_memalign_init(alignment, size);
  if (iptr) {
    *ptr = iptr;
    return 0;
  }

  return posix_memalign_system(ptr, alignment, size);
}

int posix_memalign_bootstrap(void ** ptr, size_t alignment, size_t size)
{
  void * iptr = posix_memalign_init(alignment, size);
  if (iptr) {
    *ptr = iptr;
    return 0;
  }

  if (Tau_memory_wrapper_init()) {
    return posix_memalign_system(ptr, alignment, size);
  }

  posix_memalign_handle = posix_memalign_enabled;
  return posix_memalign_enabled(ptr, alignment, size);
}

/*********************************************************************
 * valloc
 ********************************************************************/

void * valloc_init(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    valloc_system = Tau_get_system_valloc();
  }

  if (!valloc_system) {
    return bootstrap_alloc(PAGE_SIZE, size);
  }

  return NULL; // Indicates success
}

void * valloc_enabled(size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return valloc_system(size);
  }
  return Tau_valloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * valloc_disabled(size_t size)
{
  void * ptr = valloc_init(size);
  if (ptr) return ptr;

  return valloc_system(size);
}

void * valloc_bootstrap(size_t size)
{
  void * ptr = valloc_init(size);
  if (ptr) return ptr;

  if (Tau_memory_wrapper_init()) {
    return valloc_system(size);
  }

  valloc_handle = valloc_enabled;
  return valloc_enabled(size);
}

/*********************************************************************
 * pvalloc
 ********************************************************************/

void * pvalloc_init(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    pvalloc_system = Tau_get_system_pvalloc();
  }

  if (!pvalloc_system) {
    size = (size + PAGE_SIZE-1) & ~(PAGE_SIZE-1);
    return bootstrap_alloc(PAGE_SIZE, size);
  }

  return NULL; // Indicates success
}

void * pvalloc_enabled(size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return pvalloc_system(size);
  }
  return Tau_pvalloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * pvalloc_disabled(size_t size)
{
  void * ptr = pvalloc_init(size);
  if (ptr) return ptr;

  return pvalloc_system(size);
}

void * pvalloc_bootstrap(size_t size)
{
  void * ptr = pvalloc_init(size);
  if (ptr) return ptr;

  if (Tau_memory_wrapper_init()) {
    return pvalloc_system(size);
  }

  pvalloc_handle = pvalloc_enabled;
  return pvalloc_enabled(size);
}

/*********************************************************************
 * free
 ********************************************************************/

int free_init(void * ptr)
{
  static int initializing = 0;
  if (!initializing) {
   initializing = 1;
   free_system = Tau_get_system_free();
  }

  if (!free_system) {
    bootstrap_free(ptr);
    return 1;
  }

  return 0; // Indicates success
}

void free_enabled(void * ptr)
{
#if 1
  if (!Tau_global_getLightsOut()) {
    if (Tau_memory_is_tau_allocation(ptr)) {
      Tau_free(ptr, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
    } else if (is_bootstrap(ptr)) {
      bootstrap_free(ptr);
    } else {
      free_system(ptr);
    }
  }

#else

  if (!is_bootstrap(ptr)) {
    if (Tau_memory_wrapper_passthrough()) {
      if (!Tau_global_getLightsOut()) {
        free_system(ptr);
      } else {
        // TODO
      }
    } else {
      Tau_free(ptr, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
    }
  } else {
    bootstrap_free(ptr);
  }
#endif
}

void free_disabled(void * ptr)
{
  if (free_init(ptr)) return;
  return free_system(ptr);
}

void free_bootstrap(void * ptr)
{
  if (free_init(ptr)) return;

  if (Tau_memory_wrapper_init()) {
    return free_system(ptr);
  }

  free_handle = free_enabled;
  return free_enabled(ptr);
}

/*********************************************************************
 * strcmp
 ********************************************************************/

#if 0
int strcmp_init(char const * s1, char const * s2, int * retval)
{
  static int initializing = 0;
  if (!initializing) {
   initializing = 1;
   strcmp_system = Tau_get_system_strcmp();
  }

  if (!strcmp_system) {
    *retval = __tau_strcmp(s1, s2);
    return 1;
  }

  return 0;
}

int strcmp_enabled(char const * s1, char const * s2)
{
  if (Tau_memory_wrapper_passthrough()) {
    return strcmp_system(s1, s2);
  }
  return Tau_strcmp(s1, s2, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

int strcmp_disabled(char const * s1, char const * s2)
{
  int ret;
  if (strcmp_init(s1, s2, &ret)) return ret;
  return strcmp_system(s1, s2);
}

int strcmp_bootstrap(char const * s1, char const * s2)
{
  int ret;
  if (strcmp_init(s1, s2, &ret)) return ret;

  if (Tau_memory_wrapper_init()) {
    return strcmp_system(s1, s2);
  }

  strcmp_handle = strcmp_enabled;
  return strcmp_enabled(s1, s2);
}
#endif

/*********************************************************************
 * Wrapper enable/disable
 ********************************************************************/

// Enables for all threads (i.e. not thread safe)
void Tau_memory_wrapper_enable(void)
{
  if (malloc_handle == malloc_disabled) {
    malloc_handle = malloc_bootstrap;
    calloc_handle = calloc_bootstrap;
    realloc_handle = realloc_bootstrap;
    memalign_handle = memalign_bootstrap;
    posix_memalign_handle = posix_memalign_bootstrap;
    valloc_handle = valloc_bootstrap;
    pvalloc_handle = pvalloc_bootstrap;
    free_handle = free_bootstrap;

#if 0
    strcmp_handle = strcmp_bootstrap;
#endif
  }
}

// Disables for all threads (i.e. not thread safe)
void Tau_memory_wrapper_disable(void)
{
  if (malloc_handle != malloc_disabled) {
    malloc_handle = malloc_disabled;
    calloc_handle = calloc_disabled;
    free_handle = free_disabled;
    memalign_handle = memalign_disabled;
    posix_memalign_handle = posix_memalign_disabled;
    realloc_handle = realloc_disabled;
    valloc_handle = valloc_disabled;
    pvalloc_handle = pvalloc_disabled;

#if 0
    strcmp_handle = strcmp_disabled;
#endif
  }
}


/*********************************************************************
 * EOF
 ********************************************************************/
