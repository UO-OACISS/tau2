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
#include <memory_wrapper.h>

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

// malloc handles.  These must not be static.
malloc_t malloc_handle = malloc_bootstrap;
malloc_t malloc_system = NULL;

// calloc handles.  These must not be static.
calloc_t calloc_handle = calloc_bootstrap;
calloc_t calloc_system = NULL;

// free handles.  These must not be static.
free_t free_handle = free_bootstrap;
free_t free_system = NULL;

// memalign handles.  These must not be static.
memalign_t memalign_handle = memalign_bootstrap;
memalign_t memalign_system = NULL;

// posix_memalign handles.  These must not be static.
posix_memalign_t posix_memalign_handle = posix_memalign_bootstrap;
posix_memalign_t posix_memalign_system = NULL;

// realloc handles.  These must not be static.
realloc_t realloc_handle = realloc_bootstrap;
realloc_t realloc_system = NULL;

// valloc handles.  These must not be static.
valloc_t valloc_handle = valloc_bootstrap;
valloc_t valloc_system = NULL;

// pvalloc handles.  These must not be static.
pvalloc_t pvalloc_handle = pvalloc_bootstrap;
pvalloc_t pvalloc_system = NULL;

// Memory for bootstrapping.  Must not be static.
char bootstrap_heap[BOOTSTRAP_HEAP_SIZE];
char * bootstrap_base = bootstrap_heap;


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
  // Bootstrap memory is deallocated on program exit
}

int Tau_is_bootstrap(void * ptr)
{
  char const * const p = (char*)ptr;
  return (p < bootstrap_heap + BOOTSTRAP_HEAP_SIZE) && (bootstrap_heap < p);
}


/*********************************************************************
 * malloc
 ********************************************************************/

void * malloc_active(size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return malloc_system(size);
  }
  return Tau_malloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * malloc_bootstrap(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    malloc_system = Tau_get_system_malloc();
  }

  if (!malloc_system) {
    return bootstrap_alloc(0, size);
  }

  if (Tau_memory_wrapper_init()) {
    return malloc_system(size);
  }

  malloc_handle = malloc_active;
  return malloc_active(size);
}

/*********************************************************************
 * calloc
 ********************************************************************/

void * calloc_active(size_t count, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return calloc_system(count, size);
  }
  return Tau_calloc(count, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * calloc_bootstrap(size_t count, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    calloc_system = Tau_get_system_calloc();
  }

  if (!calloc_system) {
    char * ptr = (char*)bootstrap_alloc(0, size*count);
    char const * const end = ptr + size*count;
    char * p = ptr;
    while (p < end) {
      *p = (char)0;
      ++p;
    }
    return (void *)ptr;
  }

  if (Tau_memory_wrapper_init()) {
    return calloc_system(count, size);
  }

  calloc_handle = calloc_active;
  return calloc_active(count, size);
}

/*********************************************************************
 * free
 ********************************************************************/

void free_active(void * ptr)
{
  if (Tau_memory_wrapper_passthrough()) {
    return free_system(ptr);
  }
  return Tau_free(ptr, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void free_bootstrap(void * ptr)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    free_system = Tau_get_system_free();
  }

  if (!free_system) {
    bootstrap_free(ptr);
  }

  if (Tau_memory_wrapper_init()) {
    return free_system(ptr);
  }

  free_handle = free_active;
  return free_active(ptr);
}

/*********************************************************************
 * memalign
 ********************************************************************/

void * memalign_active(size_t alignment, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return memalign_system(alignment, size);
  }
  return Tau_memalign(alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * memalign_bootstrap(size_t alignment, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    memalign_system = Tau_get_system_memalign();
  }

  if (!memalign_system) {
    return bootstrap_alloc(alignment, size);
  }

  if (Tau_memory_wrapper_init()) {
    return memalign_system(alignment, size);
  }

  memalign_handle = memalign_active;
  return memalign_active(alignment, size);
}

/*********************************************************************
 * posix_memalign
 ********************************************************************/

int posix_memalign_active(void **ptr, size_t alignment, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return posix_memalign_system(ptr, alignment, size);
  }
  return Tau_posix_memalign(ptr, alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

int posix_memalign_bootstrap(void **ptr, size_t alignment, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    posix_memalign_system = Tau_get_system_posix_memalign();
  }

  if (!posix_memalign_system) {
    *ptr = bootstrap_alloc(alignment, size);
    return 0;
  }

  if (Tau_memory_wrapper_init()) {
    return posix_memalign_system(ptr, alignment, size);
  }

  posix_memalign_handle = posix_memalign_active;
  return posix_memalign_active(ptr, alignment, size);
}

/*********************************************************************
 * realloc
 ********************************************************************/

void * realloc_active(void * ptr, size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return realloc_system(ptr, size);
  }
  return Tau_realloc(ptr, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * realloc_bootstrap(void * ptr, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    realloc_system = Tau_get_system_realloc();
  }

  if (!realloc_system) {
    return bootstrap_alloc(0, size);
  }

  if (Tau_memory_wrapper_init()) {
    return realloc_system(ptr, size);
  }

  realloc_handle = realloc_active;
  return realloc_active(ptr, size);
}

/*********************************************************************
 * valloc
 ********************************************************************/

void * valloc_active(size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return valloc_system(size);
  }
  return Tau_valloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * valloc_bootstrap(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    valloc_system = Tau_get_system_valloc();
  }

  if (!valloc_system) {
    return bootstrap_alloc(PAGE_SIZE, size);
  }

  if (Tau_memory_wrapper_init()) {
    return valloc_system(size);
  }

  valloc_handle = valloc_active;
  return valloc_active(size);
}

/*********************************************************************
 * pvalloc
 ********************************************************************/

void * pvalloc_active(size_t size)
{
  if (Tau_memory_wrapper_passthrough()) {
    return pvalloc_system(size);
  }
  return Tau_pvalloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * pvalloc_bootstrap(size_t size)
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

  if (Tau_memory_wrapper_init()) {
    return pvalloc_system(size);
  }

  pvalloc_handle = pvalloc_active;
  return pvalloc_active(size);
}

/*********************************************************************
 * EOF
 ********************************************************************/
