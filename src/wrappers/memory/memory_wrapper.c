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
#include "Profile/TauInit.h"

#ifndef TAU_MULTITHREAD
#if defined(TAU_OPENMP) || defined(PTHREADS)
#define TAU_MULTITHREAD
#endif
#endif

#ifdef TAU_MULTITHREAD
#include <pthread.h>

// Thread-specific function handles
pthread_once_t multithread_init_once = PTHREAD_ONCE_INIT;
pthread_key_t flag_key;
pthread_mutex_t flag_mutex;
#endif

// Handles to the system implementation of the function
malloc_t malloc_system;
calloc_t calloc_system;
realloc_t realloc_system;
free_t free_system;
memalign_t memalign_system;
posix_memalign_t posix_memalign_system;
valloc_t valloc_system;
pvalloc_t pvalloc_system;
puts_t puts_system;

// Memory for bootstrapping.  Must not be static.
char bootstrap_heap[BOOTSTRAP_HEAP_SIZE];
char * bootstrap_base = bootstrap_heap;


static inline
int is_bootstrap(void * ptr)
{
  char const * const p = (char*)ptr;
  return (p < bootstrap_heap + BOOTSTRAP_HEAP_SIZE) && (bootstrap_heap < p);
}

static
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
  if (bootstrap_base >= (bootstrap_heap + BOOTSTRAP_HEAP_SIZE)) {
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
 * Wrapper enable/disable
 ********************************************************************/

#ifdef TAU_MULTITHREAD
void multithread_init(void)
{
  pthread_key_create(&flag_key, NULL);
  pthread_mutex_init(&flag_mutex, NULL);
}

int * memory_wrapper_disabled_flag(void)
{
  int * flag;

  pthread_once(&multithread_init_once, multithread_init);

  flag = (int*)pthread_getspecific(flag_key);
  if (!flag) {
    pthread_mutex_lock(&flag_mutex);
    flag = (int*)bootstrap_alloc(64, sizeof(int));
    pthread_mutex_unlock(&flag_mutex);
    // Start disabled. TauInit will enable the wrapper when TAU initializes
    *flag = 1;
    // Update thread specific data
    pthread_setspecific(flag_key, (void*)flag);
  }

  return flag;
}
#else

int memory_wrapper_process_flag = 1;
int * memory_wrapper_disabled_flag(void)
{
  return &memory_wrapper_process_flag;
}
#endif

void memory_wrapper_enable(void)
{
  *memory_wrapper_disabled_flag() = 0;
}

void memory_wrapper_disable(void)
{
  *memory_wrapper_disabled_flag() = 1;
}

int memory_wrapper_init(void)
{
  static int init = 0;
  if (init) return 0;

#ifdef TAU_MULTITHREAD
  pthread_once(&multithread_init_once, multithread_init);
#endif

#ifdef TAU_MEMORY_WRAPPER_DYNAMIC
  if (Tau_init_check_dl_initialized()) {
    Tau_memory_wrapper_register(memory_wrapper_enable, memory_wrapper_disable);
    init = 1;
    return 0;
  }
  return 1;
#else
  Tau_memory_wrapper_register(memory_wrapper_enable, memory_wrapper_disable);
  init = 1;
  return 0;
#endif
}

/*********************************************************************
 * malloc
 ********************************************************************/

void * malloc_wrapper(size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      malloc_system = get_system_malloc();
    }

    if (!malloc_system) {
      return bootstrap_alloc(0, size);
    }

    if (memory_wrapper_init()) {
      return malloc_system(size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return malloc_system(size);
  } else {
    return Tau_malloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}


/*********************************************************************
 * calloc
 ********************************************************************/

void * calloc_wrapper(size_t count, size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      calloc_system = get_system_calloc();
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

    if (memory_wrapper_init()) {
      return calloc_system(count, size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return calloc_system(count, size);
  } else {
    return Tau_calloc(count, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}

/*********************************************************************
 * realloc
 ********************************************************************/

void * realloc_wrapper(void * ptr, size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      realloc_system = get_system_realloc();
    }

    if (!realloc_system) {
      return bootstrap_alloc(0, size);
    }

    if (memory_wrapper_init()) {
      return realloc_system(ptr, size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return realloc_system(ptr, size);
  } else {
    return Tau_realloc(ptr, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}

/*********************************************************************
 * free
 ********************************************************************/

void free_wrapper(void * ptr)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
     initializing = 1;
     free_system = get_system_free();
    }

    if (!free_system) {
      bootstrap_free(ptr);
      return;
    }

    if (memory_wrapper_init()) {
      if (is_bootstrap(ptr)) {
        bootstrap_free(ptr);
      } else {
        free_system(ptr);
      }
      return;
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    if (is_bootstrap(ptr)) {
      bootstrap_free(ptr);
    } else if (!Tau_global_getLightsOut()) {
      free_system(ptr);
    }
  } else {
    if (Tau_memory_is_tau_allocation(ptr)) {
      Tau_free(ptr, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
    } else if (is_bootstrap(ptr)) {
      bootstrap_free(ptr);
    } else {
      free_system(ptr);
    }
  }
}


/*********************************************************************
 * memalign
 ********************************************************************/

void * memalign_wrapper(size_t alignment, size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      memalign_system = get_system_memalign();
    }

    if (!memalign_system) {
      return bootstrap_alloc(0, size);
    }

    if (memory_wrapper_init()) {
      return memalign_system(alignment, size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return memalign_system(alignment, size);
  } else {
    return Tau_memalign(alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}

/*********************************************************************
 * posix_memalign
 ********************************************************************/

int posix_memalign_wrapper(void ** ptr, size_t alignment, size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      posix_memalign_system = get_system_posix_memalign();
    }

    if (!posix_memalign_system) {
      *ptr = bootstrap_alloc(0, size);
      return 0;
    }

    if (memory_wrapper_init()) {
      return posix_memalign_system(ptr, alignment, size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return posix_memalign_system(ptr, alignment, size);
  } else {
    return Tau_posix_memalign(ptr, alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}

/*********************************************************************
 * valloc
 ********************************************************************/

void * valloc_wrapper(size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      valloc_system = get_system_valloc();
    }

    if (!valloc_system) {
      return bootstrap_alloc(PAGE_SIZE, size);
    }

    if (memory_wrapper_init()) {
      return valloc_system(size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return valloc_system(size);
  } else {
    return Tau_valloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}

/*********************************************************************
 * pvalloc
 ********************************************************************/

void * pvalloc_wrapper(size_t size)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      pvalloc_system = get_system_pvalloc();
    }

    if (!pvalloc_system) {
      size = (size + PAGE_SIZE-1) & ~(PAGE_SIZE-1);
      return bootstrap_alloc(PAGE_SIZE, size);
    }

    if (memory_wrapper_init()) {
      return pvalloc_system(size);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return pvalloc_system(size);
  } else {
    return Tau_pvalloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
  }
}

/*********************************************************************
 * puts
 * We need to wrap puts simply so we don't generate a false positive
 * on printf() statememts.  puts will allocate some memory to buffer
 * output, and TAU will report it as a leak otherwise.
 ********************************************************************/

int puts_wrapper(const char *s)
{
  static int initializing = 0;
  static int bootstrapped = 0;

  if (!bootstrapped) {
    if (!initializing) {
      initializing = 1;
      puts_system = get_system_puts();
    }

    if (!puts_system) {
      return 0;
    }

    if (memory_wrapper_init()) {
      return puts_system(s);
    }

    bootstrapped = 1;
  }

  if (*memory_wrapper_disabled_flag()) {
    return puts_system(s);
  } else {
    memory_wrapper_disable();
    return puts_system(s);
    memory_wrapper_enable();
  }
}

/*********************************************************************
 * EOF
 ********************************************************************/
