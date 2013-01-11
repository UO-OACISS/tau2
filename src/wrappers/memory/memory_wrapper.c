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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600 /* see: man posix_memalign */
#endif

#include <dlfcn.h>

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

// Assume 4K pages unless we know otherwise.
// We cannot determine this at runtime because it must be known during
// the bootstrap process and it would be unsafe to make any system calls there.
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

// Size of heap memory for library wrapper bootstrapping
#define BOOTSTRAP_HEAP_SIZE (3*PAGE_SIZE)

// Types of function pointers for wrapped functions
typedef void * (*malloc_t)(size_t);
typedef void * (*calloc_t)(size_t, size_t);
typedef void   (*free_t)(void *);
typedef void * (*memalign_t)(size_t, size_t);
typedef int    (*posix_memalign_t)(void **, size_t, size_t);
typedef void * (*realloc_t)(void *, size_t);
typedef void * (*valloc_t)(size_t);
typedef void * (*pvalloc_t)(size_t);

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

static inline
int is_bootstrap(void * ptr)
{
  char const * const p = (char*)ptr;
  return (bootstrap_heap < p) && (p < bootstrap_heap + BOOTSTRAP_HEAP_SIZE);
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
  if (ptr && !is_bootstrap(ptr)) {
    return free_handle(ptr);
  }
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
 * malloc
 ********************************************************************/

void * malloc_active(size_t size)
{
  if (Tau_memory_passthrough()) {
    return malloc_system(size);
  }
  return Tau_malloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * malloc_bootstrap(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    malloc_system = (malloc_t)get_system_function_handle("malloc");
  }

  if (!malloc_system) {
    return bootstrap_alloc(0, size);
  }

  malloc_handle = malloc_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return malloc_active(size);
  }
  return malloc_system(size);
}

/*********************************************************************
 * calloc
 ********************************************************************/

void * calloc_active(size_t count, size_t size)
{
  if (Tau_memory_passthrough()) {
    return calloc_system(count, size);
  }
  return Tau_calloc(count, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * calloc_bootstrap(size_t count, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    calloc_system = (calloc_t)get_system_function_handle("calloc");
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

  calloc_handle = calloc_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return calloc_active(count, size);
  }
  return calloc_system(count, size);
}

/*********************************************************************
 * free
 ********************************************************************/

void free_active(void * ptr)
{
  if (Tau_memory_passthrough()) {
    return free_system(ptr);
  }
  return Tau_free(ptr, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void free_bootstrap(void * ptr)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    free_system = (free_t)get_system_function_handle("free");
  }

  if (!free_system) {
    bootstrap_free(ptr);
  }

  free_handle = free_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return free_active(ptr);
  }
  return free_system(ptr);
}

/*********************************************************************
 * memalign
 ********************************************************************/

void * memalign_active(size_t alignment, size_t size)
{
  if (Tau_memory_passthrough()) {
    return memalign_system(alignment, size);
  }
  return Tau_memalign(alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * memalign_bootstrap(size_t alignment, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    memalign_system = (memalign_t)get_system_function_handle("memalign");
  }

  if (!memalign_system) {
    return bootstrap_alloc(alignment, size);
  }

  memalign_handle = memalign_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return memalign_active(alignment, size);
  }
  return memalign_system(alignment, size);
}

/*********************************************************************
 * posix_memalign
 ********************************************************************/

int posix_memalign_active(void **ptr, size_t alignment, size_t size)
{
  if (Tau_memory_passthrough()) {
    return posix_memalign_system(ptr, alignment, size);
  }
  return Tau_posix_memalign(ptr, alignment, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

int posix_memalign_bootstrap(void **ptr, size_t alignment, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    posix_memalign_system = (posix_memalign_t)get_system_function_handle("posix_memalign");
  }

  if (!posix_memalign_system) {
    *ptr = bootstrap_alloc(alignment, size);
    return 0;
  }

  posix_memalign_handle = posix_memalign_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return posix_memalign_active(ptr, alignment, size);
  }
  return posix_memalign_system(ptr, alignment, size);
}

/*********************************************************************
 * realloc
 ********************************************************************/

void * realloc_active(void * ptr, size_t size)
{
  if (Tau_memory_passthrough()) {
    return realloc_system(ptr, size);
  }
  return Tau_realloc(ptr, size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * realloc_bootstrap(void * ptr, size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    realloc_system = (realloc_t)get_system_function_handle("realloc");
  }

  if (!realloc_system) {
    return bootstrap_alloc(0, size);
  }

  realloc_handle = realloc_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return realloc_active(ptr, size);
  }
  return realloc_system(ptr, size);
}

/*********************************************************************
 * valloc
 ********************************************************************/

void * valloc_active(size_t size)
{
  if (Tau_memory_passthrough()) {
    return valloc_system(size);
  }
  return Tau_valloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * valloc_bootstrap(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    valloc_system = (valloc_t)get_system_function_handle("valloc");
  }

  if (!valloc_system) {
    return bootstrap_alloc(PAGE_SIZE, size);
  }

  valloc_handle = valloc_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return valloc_active(size);
  }
  return valloc_system(size);
}

/*********************************************************************
 * pvalloc
 ********************************************************************/

void * pvalloc_active(size_t size)
{
  if (Tau_memory_passthrough()) {
    return pvalloc_system(size);
  }
  return Tau_pvalloc(size, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_LINE);
}

void * pvalloc_bootstrap(size_t size)
{
  static int initializing = 0;
  if (!initializing) {
    initializing = 1;
    pvalloc_system = (pvalloc_t)get_system_function_handle("pvalloc");
  }

  if (!pvalloc_system) {
    size = (size + PAGE_SIZE-1) & ~(PAGE_SIZE-1);
    return bootstrap_alloc(PAGE_SIZE, size);
  }

  pvalloc_handle = pvalloc_active;

  if (Tau_init_check_dl_initialized()) {
    Tau_memory_init();
    return pvalloc_active(size);
  }
  return pvalloc_system(size);
}


#if 0
/* Wrapping C++ operator new() and operator delete() is just too much trouble.
 * There are many complications, and it's pointless because these operators will
 * ultimately call some form of libc allocate (malloc, etc.) so wrapping those
 * functions is enough.
 *
 * This implementation is left here in case the above turns out to be untrue.
 */

#ifdef TAU_WINDOWS
int Tau_operator_new_handler(size_t size)  { return 0; }
#else
void Tau_operator_new_handler() { }
#endif

struct TauDeleteFlags
{
public:

  TauDeleteFlags() :
    filename(NULL), lineno(0)
  { }

  char const * filename;
  int lineno;
};

static TauDeleteFlags & Tau_operator_delete_flags()
{
  static TauDeleteFlags ** delete_flags = NULL;
  if (!delete_flags) {
    delete_flags = (TauDeleteFlags**)calloc(TAU_MAX_THREADS, sizeof(TauDeleteFlags*));
  }

  int tid = Tau_get_tid();
  if (!delete_flags[tid]) {
    delete_flags[tid] = new TauDeleteFlags;
  }

  return *delete_flags[tid];
}

typedef void * (*operator_new_wrapper_t)(size_t, bool, bool, const char *, int);
static operator_new_wrapper_t & TheTauOperatorNewWrapper() {
  static operator_new_wrapper_t wrapper = Tau_operator_new;
  return wrapper;
}

typedef void (*operator_delete_wrapper_t)(void *, bool, bool);
static operator_delete_wrapper_t & TheTauOperatorDeleteWrapper() {
  static operator_delete_wrapper_t wrapper = Tau_operator_delete;
  return wrapper;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * Tau_system_operator_new(size_t size, bool array, bool nothrow, const char * filename, int lineno)
{
  return malloc(size);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void Tau_system_operator_delete(void * ptr, bool array, bool nothrow)
{
  free(ptr);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * Tau_operator_new(size_t size, bool array, bool nothrow, const char * filename, int lineno)
{
  void * ptr = NULL;

  if (Tau_memdbg_passthrough()) {
    ptr = Tau_system_operator_new(size, array, nothrow, filename, lineno);
    return ptr;
  }

  // Change wrapper pointer to prevent recursion while processing this operator invocation
  TheTauOperatorNewWrapper() = Tau_system_operator_new;

  // Use system new with placement to avoid recursion
  // The effect is: TauAllocation * alloc = new TauAllocation
  TauAllocation * alloc = (TauAllocation*)Tau_system_operator_new(
      sizeof(TauAllocation), false, nothrow, __FILE__, __LINE__);
  new (alloc) TauAllocation();

  if (TauEnv_get_memdbg()) {
    do {
      ptr = alloc->Allocate(0, size, filename, lineno);

      if (!ptr) {
#ifdef TAU_WINDOWS
        _PNH h = _set_new_handler(Tau_operator_new_handler);
        _set_new_handler(h);
#else
        std::new_handler h = std::set_new_handler(Tau_operator_new_handler);
        std::set_new_handler(h);
#endif
        if (h) {
          try {
#ifdef TAU_WINDOWS
            int retval = h(size);
#else
            h();
#endif
          } catch (std::bad_alloc&) {
            if (nothrow) return NULL;
            else throw;
          }
        } else {
          if (nothrow) return NULL;
          else throw std::bad_alloc();
        }
      }
    } while (!ptr);
  } else {
    ptr = Tau_system_operator_new(size, array, nothrow, filename, lineno);
    alloc->TrackAllocation(ptr, size, filename, lineno);
  }

  // Reset wrapper pointer
  TheTauOperatorNewWrapper() = Tau_operator_new;

  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void Tau_operator_delete(void * ptr, bool array, bool nothrow)
{
  // delete NULL should do nothing by definition
  if (ptr) {

    if (Tau_memdbg_passthrough()) {
      Tau_system_operator_delete(ptr, array, nothrow);
      return;
    }

    // Change wrapper pointer to prevent recursion while processing this operator invocation
    TheTauOperatorDeleteWrapper() = Tau_system_operator_delete;

    // The flags us the filename:lno of the delete invocation since
    // these can't be passed in the arguments
    TauDeleteFlags & flags = Tau_operator_delete_flags();

    char const * filename;
    int lineno;

    if (flags.filename) {
      filename = flags.filename;
      lineno = flags.lineno;
    } else {
      filename = "Unknown";
      lineno = 0;
    }

    addr_t addr = (addr_t)ptr;
    TauAllocation * alloc = TauAllocation::Find(addr);

    if (alloc) {
      if (TauEnv_get_memdbg()) {
        alloc->Deallocate(filename, lineno);
      } else {
        alloc->TrackDeallocation(filename, lineno);
        Tau_system_operator_delete(ptr, array, nothrow);
      }
      Tau_system_operator_delete(alloc, false, nothrow);
    } else {
      TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found\n", addr);
      Tau_system_operator_delete(ptr, array, nothrow);
    }

    // Clear flags
    flags.filename = NULL;
    flags.lineno = 0;

    // Reset wrapper pointer
    TheTauOperatorDeleteWrapper() = Tau_operator_delete;
  } // if (ptr)
}

void Tau_set_operator_delete_flags(const char * filename, int lineno)
{
  TauDeleteFlags & flags = Tau_operator_delete_flags();

  if(flags.filename) {
    TAU_VERBOSE("TAU: ERROR - delete flags were already set!\n");
  }
  flags.filename = filename;
  flags.lineno = lineno;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new(size_t size) throw(std::bad_alloc)
{
  return TheTauOperatorNewWrapper()(size, false, false, "Unknown", 0);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new[](size_t size) throw(std::bad_alloc)
{
  return TheTauOperatorNewWrapper()(size, true, false, "Unknown", 0);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void operator delete(void * ptr) throw()
{
  TheTauOperatorDeleteWrapper()(ptr, false, false);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void operator delete[](void * ptr) throw()
{
  TheTauOperatorDeleteWrapper()(ptr, true, false);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new(size_t size, const std::nothrow_t & nothrow) throw()
{
  return TheTauOperatorNewWrapper()(size, false, true, "Unknown", 0);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new[](size_t size, const std::nothrow_t & nothrow) throw()
{
  return TheTauOperatorNewWrapper()(size, true, true, "Unknown", 0);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void operator delete(void * ptr, const std::nothrow_t & nothrow) throw()
{
  TheTauOperatorDeleteWrapper()(ptr, false, true);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void operator delete[](void * ptr, const std::nothrow_t & nothrow) throw()
{
  TheTauOperatorDeleteWrapper()(ptr, true, true);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new(size_t size, char const * filename, int lineno) throw(std::bad_alloc)
{
  return TheTauOperatorNewWrapper()(size, false, false, filename, lineno);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new[](size_t size, char const * filename, int lineno) throw(std::bad_alloc)
{
  return TheTauOperatorNewWrapper()(size, true, false, filename, lineno);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new(size_t size, const std::nothrow_t & nothrow, char const * filename, int lineno) throw()
{
  return TheTauOperatorNewWrapper()(size, false, true, filename, lineno);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * operator new[](size_t size, const std::nothrow_t & nothrow, char const * filename, int lineno) throw()
{
  return TheTauOperatorNewWrapper()(size, true, true, filename, lineno);
}
#endif

/*********************************************************************
 * EOF
 ********************************************************************/
