/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://tau.uoregon.edu                             **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**      File            : TauMemory.h                                      **
**      Contact         : tau-bugs@cs.uoregon.edu                          **
**      Documentation   : See http://tau.uoregon.edu                       **
**      Description     : Support for memory tracking                      **
**                                                                         **
****************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_MEMORY_H_
#define _TAU_MEMORY_H_

#if defined(__APPLE__) || defined(TAU_XLC) || defined(TAU_WINDOWS)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef TAU_WINDOWS
#define ENOMEM 0
#endif

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600 /* see: man posix_memalign */
#endif
#endif

#ifdef _MSC_VER
/* define these functions as non-intrinsic */
#pragma function( memcpy, strcpy, strcat )
#endif

#include <stdlib.h>
#include <tau_internal.h>

// PAGESIZE can be a synonym for PAGE_SIZE
#if !defined(PAGE_SIZE) && defined(PAGESIZE)
#define PAGE_SIZE PAGESIZE
#endif

// Constants indicating unknown source location
#define TAU_MEMORY_UNKNOWN_LINE 0
#define TAU_MEMORY_UNKNOWN_FILE "Unknown"
#define TAU_MEMORY_UNKNOWN_FILE_STRLEN 7


// Which platforms have malloc?
#ifndef HAVE_MALLOC
// Assume all platforms have malloc
#define HAVE_MALLOC 1
#endif

// Which platforms have calloc?
#ifndef HAVE_CALLOC
// Assume all platforms have calloc
#define HAVE_CALLOC 1
#endif

// Which platforms have realloc?
#ifndef HAVE_REALLOC
// Assume all platforms have realloc
#define HAVE_REALLOC 1
#endif

// Which platforms have free?
#ifndef HAVE_FREE
// Assume all platforms have free
#define HAVE_FREE 1
#endif

// Which platforms have puts?
#ifndef HAVE_PUTS
// Assume all platforms have puts
#define HAVE_PUTS 1
#endif

// Which platforms have memalign?
#ifndef HAVE_MEMALIGN
#if defined(__APPLE__) || defined(TAU_XLC) || defined(TAU_WINDOWS)
#undef HAVE_MEMALIGN
#else
#define HAVE_MEMALIGN 1
#endif
#endif

// Which platforms have posix_memalign?
#ifndef HAVE_POSIX_MEMALIGN
#if defined(TAU_WINDOWS) || (defined(__APPLE__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ +0 < 1060))
#undef HAVE_POSIX_MEMALIGN
#ifndef ENOMEM
#define ENOMEM 12
#endif /* ENOMEM defined */
#elif (_POSIX_C_SOURCE >= 200112L) || (_XOPEN_SOURCE >= 600)
#define HAVE_POSIX_MEMALIGN 1
#endif
#endif

// Which platforms have valloc?
#ifndef HAVE_VALLOC
#if (defined(__APPLE__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ +0 < 1060)) \
    || defined(TAU_XLC) || defined(TAU_WINDOWS)
#undef HAVE_VALLOC
#elif defined(_BSD_SOURCE) || (_XOPEN_SOURCE >= 500 || _XOPEN_SOURCE && _XOPEN_SOURCE_EXTENDED) \
      && !(_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600)
#define HAVE_VALLOC 1
#endif
#endif

// Which platforms have pvalloc?
#ifndef HAVE_PVALLOC
#if defined(__APPLE__) || defined(TAU_XLC) || defined(TAU_WINDOWS) || defined(TAU_ANDROID) || defined(TAU_NEC_SX)
#undef HAVE_PVALLOC
#else
#define HAVE_PVALLOC 1
#endif
#endif

// Which platforms have reallocf?
#ifndef HAVE_REALLOCF
#if !defined(__APPLE__)
#undef HAVE_REALLOCF
#else
#define HAVE_REALLOCF
#endif
#endif

// Which platforms have aligned_alloc?
#ifndef HAVE_ALIGNED_ALLOC
#if defined(_ISOC11_SOURCE)
#define HAVE_ALIGNED_ALLOC 1
#else
#undef HAVE_ALIGNED_ALLOC
#endif
#endif

// Which platforms have mallinfo?
#ifndef HAVE_MALLINFO
#if !( defined(TAU_CRAYXMT) || defined(TAU_CATAMOUNT) || defined(TAU_NEC_SX) ) && \
     ( defined (__linux__) || defined (_AIX) || defined(sgi) || \
       defined (__alpha) || defined (CRAYCC) || defined(__blrts__))
#define HAVE_MALLINFO 1
#endif
#endif


#ifdef __cplusplus
#include <mutex>
//=============================================================================
// Allocation record for heap allocation made by TAU or by the system.
// Manages guarded (de)allocation and memory profiling events.
//
class TauAllocation
{

// ----------------------------------------------------------------------------
// Public class members
//
public:

  typedef unsigned char * addr_t;

  struct allocation_map_t : public TAU_HASH_MAP<addr_t, class TauAllocation*> {
    allocation_map_t() {
      Tau_init_initializeTAU();
    }
    virtual ~allocation_map_t() {
      Tau_destructor_trigger();
    }
  };

  struct event_map_t : public TAU_HASH_MAP<unsigned long, tau::TauContextUserEvent*> {
    event_map_t() {
      Tau_init_initializeTAU();
    }
    virtual ~event_map_t() {
      Tau_destructor_trigger();
    }
  };

  typedef TAU_HASH_MAP<tau::TauUserEvent*, tau::TauUserEvent*> leak_event_map_t;

  // Database of allocation records (read-only outside this class)
  static allocation_map_t const & AllocationMap() {
    return __allocation_map();
  }

  // Total bytes allocated
  static size_t BytesAllocated() {
    return __bytes_allocated();
  }

  // Total bytes deallocated
  static size_t BytesDeallocated() {
    return __bytes_deallocated();
  }

  // Bytes of memory protection overhead
  static size_t BytesOverhead() {
    return __bytes_overhead();
  }

  // Returns true if the given allocation size should be protected
  static bool AllocationShouldBeProtected(size_t size);

  // Trigger memory leak detection
  static void DetectLeaks(void);

  // Search for an allocation record by address
  static TauAllocation * Find(allocation_map_t::key_type const & key);
  static TauAllocation * Find(void * ptr) { return Find((addr_t)ptr); }
  static TauAllocation * FindContaining(void * ptr);

  // Records memory used at the current time
  static void TriggerHeapMemoryUsageEvent(const char * prefix = nullptr);

  // Records memory remaining at the current time
  static void TriggerMemoryHeadroomEvent(void);

  // Records memory overhead consumed by memory debugger
  static void TriggerMemDbgOverheadEvent(void);

  // Uses the MMU to mark an address range as protected.
  // Touching memory in that range will trigger a segfault or bus error
  static int Protect(addr_t addr, size_t size);

  // Uses the MMU to mark an address range as unprotected (i.e. normal memory).
  static int Unprotect(addr_t addr, size_t size);


// ----------------------------------------------------------------------------
// Private class members
//
private:

  // Read/write allocation map
  static allocation_map_t & __allocation_map();

  // Read/write leak event map
  static leak_event_map_t & __leak_event_map();

  // Read/write bytes allocated
  static size_t & __bytes_allocated();

  // Read/write bytes deallocated
  static size_t & __bytes_deallocated();

  // Bytes of memory protection overhead
  static size_t & __bytes_overhead();

  // internal lock for memory management
  static std::mutex mtx;

// ----------------------------------------------------------------------------
// Public instance members
//
public:

  TauAllocation() :
    alloc_addr(NULL), alloc_size(0),
    user_addr(NULL), user_size(0),
    lguard_addr(NULL), lguard_size(0),
    uguard_addr(NULL), uguard_size(0),
    lgap_addr(NULL), lgap_size(0),
    ugap_addr(NULL), ugap_size(0),
    tracked(false), allocated(false)
  {
    // Initialize leak event map early on since Intel 12.x compilers
    // can't construct tr1::hash_map from exit()
    static leak_event_map_t & leak_event_map = __leak_event_map();
	// use the static object, so the compiler doesn't complain
	leak_event_map.size();
  }

  // True if ptr is in the range tracked by this allocation
  bool Contains(void * ptr) const {
    addr_t addr = (addr_t)ptr;
    return (alloc_addr <= addr) && (addr < (alloc_addr+alloc_size));
  }

  // True if this is not a TAU allocation but a tracked allocation
  bool IsTracked() const {
    return tracked;
  }

  // True if this allocation has not been deallocated
  bool IsAllocated() const {
    return allocated;
  }

  // Creates and tracks a new guarded allocation with specified alignment
  void * Allocate(size_t const size, size_t align, size_t min_align, const char * filename, int lineno);

  // Deallocates the tracked allocation if it was made with Allocate()
  void Deallocate(const char * filename, int lineno);

  // Changes the size of the tracked guarded allocation
  void * Reallocate(size_t const size, size_t align, size_t min_align, const char * filename, int lineno);

  // Tracks an allocation made by the system and records the allocation event
  void TrackAllocation(void * ptr, size_t size, const char * filename, int lineno);

  // Stops tracking an allocation made by the system and records the deallocation event
  void TrackDeallocation(const char * filename, int lineno);

  // Tracks a reallocation made by the system and records the appropriate allocation or deallocation event(s)
  void TrackReallocation(void * ptr, size_t size, const char * filename, int lineno);

// ----------------------------------------------------------------------------
// Private instance members
//
private:

  tau::TauUserEvent * alloc_event;  ///< Allocation event (for leak detection)

  addr_t alloc_addr;    ///< Unadjusted address
  size_t alloc_size;    ///< Unadjusted size
  addr_t user_addr;     ///< Address presented to user
  size_t user_size;     ///< Size requested by user
  addr_t lguard_addr;   ///< Protected lower range address
  size_t lguard_size;   ///< Protected lower range size
  addr_t uguard_addr;   ///< Protected upper range address
  size_t uguard_size;   ///< Protected upper range size
  addr_t lgap_addr;     ///< Unprotected lower gap address
  size_t lgap_size;     ///< Unprotected lower gap size
  addr_t ugap_addr;     ///< Unprotected upper gap address
  size_t ugap_size;     ///< Unprotected upper gap size

  bool tracked;         ///< True if this is not a TAU allocation but a tracked allocation
  bool allocated;       ///< True if this allocation has not been deallocated

  // Quickly translates a filename and line number to a unique hash
  unsigned long LocationHash(unsigned long hash, char const * data);

  // Triggers the allocation event associated with a filename and line number
  void TriggerAllocationEvent(size_t size, char const * filename, int lineno);

  // Triggers the deallocation event associated with a filename and line number
  void TriggerDeallocationEvent(size_t size, char const * filename, int lineno);

  // Triggers the memory error event associated with a filename and line number
  void TriggerErrorEvent(char const * descript, char const * filename, int lineno);

};
// END class TauAllocation
//=============================================================================
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void (*wrapper_enable_handle_t)(void);
typedef void (*wrapper_disable_handle_t)(void);

void Tau_memory_initialize(void);
int Tau_memory_is_tau_allocation(void * ptr);

void Tau_memory_wrapper_register(wrapper_enable_handle_t, wrapper_disable_handle_t);
void Tau_memory_wrapper_enable(void);
void Tau_memory_wrapper_disable(void);
int Tau_memory_wrapper_is_registered(void);

size_t Tau_page_size(void);
double Tau_max_RSS(void);
int Tau_estimate_free_memory(void);
void Tau_detect_memory_leaks(void);

void Tau_track_memory_allocation(void *, size_t, char const *, int);
void Tau_track_memory_deallocation(void *, char const *, int);
void Tau_track_memory_reallocation(void *, void *, size_t, char const *, int);

void * Tau_malloc(size_t, char const *, int);
void * Tau_calloc(size_t, size_t, char const *, int);
void   Tau_free(void *, char const *, int);
void * Tau_memalign(size_t, size_t, char const *, int);
int    Tau_posix_memalign(void **, size_t, size_t, char const *, int);
void * Tau_realloc(void *, size_t, char const *, int);
void * Tau_valloc(size_t, char const *, int);
void * Tau_pvalloc(size_t, char const *, int);

#if 0
int __tau_strcmp(char const *, char const *);
int Tau_strcmp(char const *, char const *, char const *, int);

char * Tau_strdup(char const *, char const *, int);
char * Tau_strcpy(char *, char const *, char const *, int);
char * Tau_strcat(char *, char const *, char const *, int);

char * Tau_strncpy(char *, char const *, size_t, char const *, int);
char * Tau_strncat(char *, char const *, size_t, char const *, int);

void * Tau_memcpy(void *, void const *, size_t, char const *, int);
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_MEMORY_H_ */

/***************************************************************************
 * $RCSfile: TauMemory.h,v $   $Author: amorris $
 * $Revision: 1.4 $   $Date: 2010/02/03 06:09:44 $
 * TAU_VERSION_ID: $Id: TauMemory.h,v 1.4 2010/02/03 06:09:44 amorris Exp $
 ***************************************************************************/
