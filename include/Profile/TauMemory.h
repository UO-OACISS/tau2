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

#include <tau_internal.h>

#if defined(__darwin__) || defined(__APPLE__) || defined(TAU_XLC)
#undef HAVE_MEMALIGN
#undef HAVE_PVALLOC
#else
#define HAVE_MEMALIGN 1
#define HAVE_PVALLOC 1
#endif

#define TAU_MEMORY_UNKNOWN_LINE 0
#define TAU_MEMORY_UNKNOWN_FILE "Unknown"
#define TAU_MEMORY_UNKNOWN_FILE_STRLEN 7

#ifdef __cplusplus

class TauAllocation
{

public:

  typedef unsigned char * addr_t;
  typedef TauContextUserEvent user_event_t;

  typedef TAU_HASH_MAP<addr_t, class TauAllocation*> allocation_map_t;
  static allocation_map_t & AllocationMap() {
    Tau_global_incr_insideTAU();
    allocation_map_t & allocMap = __allocationMap();
    Tau_global_decr_insideTAU();
    return allocMap;
  }
  static allocation_map_t & __allocationMap() {
    static allocation_map_t alloc_map;
    return alloc_map;
  }

  static TauAllocation * Find(allocation_map_t::key_type const & key) {
    static allocation_map_t const & alloc_map = AllocationMap();
    allocation_map_t::const_iterator it = alloc_map.find(key);
    if (it != alloc_map.end())
      return it->second;
    return NULL;
  }
  static TauAllocation * Find(void * ptr) {
    return Find((addr_t)ptr);
  }

  static size_t & BytesAllocated() {
    // Not thread safe!
    static size_t bytes = 0;
    return bytes;
  }

  static void DetectLeaks(void);

  static TauAllocation * FindContaining(void * ptr);

public:

  TauAllocation() :
    alloc_addr(NULL), alloc_size(0),
    user_addr(NULL), user_size(0),
    lguard_addr(NULL), lguard_size(0),
    uguard_addr(NULL), uguard_size(0),
    lgap_addr(NULL), lgap_size(0),
    ugap_addr(NULL), ugap_size(0),
    alloc_event(NULL)
  { }

  bool Contains(void * ptr) const {
    addr_t addr = (addr_t)ptr;
    return (alloc_addr <= ptr) && (ptr < (alloc_addr+alloc_size));
  }

  bool InUpperGuard(void * ptr) const {
    addr_t addr = (addr_t)ptr;
    return uguard_addr && (uguard_addr <= ptr) && (ptr < (uguard_addr+uguard_size));
  }

  bool InLowerGuard(void * ptr) const {
    addr_t addr = (addr_t)ptr;
    return lguard_addr && (lguard_addr <= ptr) && (ptr < (lguard_addr+lguard_size));
  }

  user_event_t * GetAllocationEvent() const {
    return alloc_event;
  }

  void * Allocate(size_t const size, size_t align, size_t min_align, const char * filename, int lineno);
  void Deallocate(const char * filename, int lineno);

  void TrackAllocation(void * ptr, size_t size, const char * filename, int lineno);
  void TrackDeallocation(const char * filename, int lineno);

  void EnableUpperGuard();
  void DisableUpperGuard();
  void EnableLowerGuard();
  void DisableLowerGuard();

private:

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

  user_event_t * alloc_event; ///< Allocation event (for leak detection)

  void ProtectPages(addr_t addr, size_t size);
  void UnprotectPages(addr_t addr, size_t size);

  unsigned long LocationHash(unsigned long hash, char const * data);
  void TriggerAllocationEvent(char const * filename, int lineno);
  void TriggerDeallocationEvent(char const * filename, int lineno);
  void TriggerErrorEvent(char const * descript, char const * filename, int lineno);
};
#endif

// libc bindings

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void Tau_memory_initialize(void);

int Tau_memory_is_wrapper_present(void);
void Tau_memory_set_wrapper_present(int);
int Tau_memory_is_tau_allocation(void * ptr);

size_t Tau_page_size(void);

void Tau_detect_memory_leaks(void);
size_t Tau_get_bytes_allocated(void);

void Tau_track_memory_allocation(void *, size_t, char const *, int);
void Tau_track_memory_deallocation(void *, char const *, int);

void * Tau_allocate_unprotected(size_t);

void * Tau_malloc(size_t, char const *, int);
void * Tau_calloc(size_t, size_t, char const *, int);
void   Tau_free(void *, char const *, int);
#ifdef HAVE_MEMALIGN
void * Tau_memalign(size_t, size_t, char const *, int);
#endif
int    Tau_posix_memalign(void **, size_t, size_t, char const *, int);
void * Tau_realloc(void *, size_t, char const *, int);
void * Tau_valloc(size_t, char const *, int);
#ifdef HAVE_PVALLOC
void * Tau_pvalloc(size_t, char const *, int);
#endif

int __tau_strcmp(char const *, char const *);
int Tau_strcmp(char const *, char const *, char const *, int);

#if 0
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
