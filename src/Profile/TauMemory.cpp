/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2004  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauMemory.cpp 				  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <Profile/Profiler.h>
#include <tau_internal.h>
#if (defined(__APPLE_CC__) || defined(TAU_APPLE_XLC) || defined(TAU_APPLE_PGI))
#include <malloc/malloc.h>
#elif defined(TAU_FREEBSD)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#include <map>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <stdlib.h>

#ifdef TAU_BGP
#include <kernel_interface.h>
#endif

#if defined(TAU_WINDOWS)
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define  MAP_ANONYMOUS MAP_ANON
#endif

typedef unsigned char * addr_t;
typedef TauContextUserEvent user_event_t;

class TauAllocation
{

public:

  TauAllocation() :
    alloc_addr(NULL), alloc_size(0),
    user_addr(NULL), user_size(0),
    prot_addr(NULL), prot_size(0),
    gap_addr(NULL), gap_size(0)
  { }

  void * Allocate(size_t align, size_t size, const char * filename, int lineno);
  void Deallocate(const char * filename, int lineno);

  void TrackAllocation(void * ptr, size_t size, const char * filename, int lineno);
  void TrackDeallocation(const char * filename, int lineno);

  addr_t alloc_addr;    ///< Unadjusted address
  size_t alloc_size;    ///< Unadjusted size
  addr_t user_addr;     ///< Address presented to user
  size_t user_size;     ///< Size requested by user
  addr_t prot_addr;     ///< Protected upper range address
  size_t prot_size;     ///< Protected upper range size
  addr_t gap_addr;      ///< Unprotected gap address
  size_t gap_size;      ///< Unprotected gap size

  user_event_t * event; ///< Allocation event (for leak detection)

private:

  void ProtectPages(addr_t addr, size_t size);
  void UnprotectPages(addr_t addr, size_t size);

  void TriggerAllocationEvent(char const * filename, int lineno);
  void TriggerDeallocationEvent(char const * filename, int lineno);
  unsigned long LocationHash(unsigned long hash, char const * data);

};



typedef TAU_HASH_MAP<unsigned long, user_event_t*> event_map_t;
static event_map_t & TheTauAllocationEventMap() {
  static event_map_t event_map;
  return event_map;
}

typedef TAU_HASH_MAP<addr_t, class TauAllocation*> allocation_map_t;
static allocation_map_t & TheTauAllocationMap() {
  static allocation_map_t alloc_map;
  return alloc_map;
}

static TauAllocation * TheTauAllocationMap(allocation_map_t::key_type const & key) {
  allocation_map_t const & alloc_map = TheTauAllocationMap();
  allocation_map_t::const_iterator it = alloc_map.find(key);
  if (it != alloc_map.end())
    return it->second;
  return NULL;
}


// Incremental string hashing function.
// Uses Paul Hsieh's SuperFastHash, the same as in Google Chrome.
unsigned long TauAllocation::LocationHash(unsigned long hash, char const * data)
{
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )

  uint32_t tmp;
  int len = strlen(data);
  int rem;

  rem = len & 3;
  len >>= 2;

  for (; len > 0; len--) {
    hash += get16bits(data);
    tmp = (get16bits(data + 2) << 11) ^ hash;
    hash = (hash << 16) ^ tmp;
    data += 2 * sizeof(uint16_t);
    hash += hash >> 11;
  }

  switch (rem) {
  case 3:
    hash += get16bits(data);
    hash ^= hash << 16;
    hash ^= ((signed char)data[sizeof(uint16_t)]) << 18;
    hash += hash >> 11;
    break;
  case 2:
    hash += get16bits(data);
    hash ^= hash << 11;
    hash += hash >> 17;
    break;
  case 1:
    hash += (signed char)*data;
    hash ^= hash << 10;
    hash += hash >> 1;
    break;
  }

  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;

  return hash;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerAllocationEvent(char const * filename, int lineno)
{
  unsigned long file_hash = LocationHash(lineno, filename);

  event_map_t & event_map = TheTauAllocationEventMap();

  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    char * s = (char*)malloc(strlen(filename)+128);
    sprintf(s, "heap allocate <file=%s, line=%d>", filename, lineno);
    event = new user_event_t(s);
    event_map[file_hash] = event;
    free((void*)s);
  } else {
    event = it->second;
  }
  event->TriggerEvent(user_size);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerDeallocationEvent(char const * filename, int lineno)
{
  user_event_t * e;
  unsigned long file_hash = LocationHash(lineno, filename);

  event_map_t & event_map = TheTauAllocationEventMap();

  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    char * s = (char*)malloc(strlen(filename)+128);
    sprintf(s, "heap free <file=%s, line=%d>", filename, lineno);
    e = new user_event_t(s);
    event_map[file_hash] = e;
    free((void*)s);
  } else {
    e = it->second;
  }
  e->TriggerEvent(user_size);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::ProtectPages(addr_t addr, size_t size)
{
#if defined(TAU_WINDOWS)

  SIZE_T OldProtect, retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  size_t tail_size;
  BOOL ret;

  while(size > 0) {
    retQuery = VirtualQuery((void*)addr, &MemInfo, sizeof(MemInfo));
    if (retQuery < sizeof(MemInfo)) {
      TAU_VERBOSE("TAU: ERROR - VirtualQuery() failed\n");
    }
    tail_size = (size > MemInfo.RegionSize) ? MemInfo.RegionSize : size;
    ret = VirtualProtect((LPVOID)addr, (DWORD)tail_size, (DWORD)PAGE_READWRITE, (PDWORD) &OldProtect);
    if (!ret) {
      TAU_VERBOSE("TAU: ERROR - VirtualProtect(%p, %ld) failed\n", addr, tail_size);
    }

    addr += tail_size;
    size -= tail_size;
  }

#else

  if (mprotect((void*)addr, size, PROT_NONE) < 0) {
    TAU_VERBOSE("TAU: ERROR - mprotect(%p, %ld, PROT_NONE) failed\n", addr, size);
  }

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::UnprotectPages(addr_t addr, size_t size)
{
#if defined(TAU_WINDOWS)

  SIZE_T OldProtect, retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  size_t tail_size;
  BOOL ret;

  while (size > 0) {
    retQuery = VirtualQuery((void*)addr, &MemInfo, sizeof(MemInfo));
    if (retQuery < sizeof(MemInfo)) {
      TAU_VERBOSE("TAU: ERROR - VirtualQuery() failed\n");
    }
    tail_size = (size > MemInfo.RegionSize) ? MemInfo.RegionSize : size;
    ret = VirtualProtect((LPVOID)addr, (DWORD)tail_size, (DWORD)PAGE_NOACCESS, (PDWORD)&OldProtect);
    if (!ret) {
      TAU_VERBOSE("TAU: ERROR - VirtualProtecct(%p, %ld) failed\n", addr, tail_size);
    }

    addr += tail_size;
    size -= tail_size;
  }

#else

  if (mprotect((void*)addr, size, PROT_READ|PROT_WRITE) < 0) {
    TAU_VERBOSE("TAU: ERROR - mprotect(%p, %ld, PROT_READ|PROT_WRITE) failed\n", addr, size);
  }

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * TauAllocation::Allocate(size_t align, size_t size, const char * filename, int lineno)
{
#ifndef PAGE_SIZE
  static size_t const PAGE_SIZE = Tau_page_size();
#endif
  static bool const PROTECT_ABOVE = TauEnv_get_memdbg_protect_above();
  static bool const PROTECT_BELOW = TauEnv_get_memdbg_protect_below();

  // Check size
  if (!size) {
    // TODO: Zero-size malloc
  }

  // Check alignment
  if(!align) {
    align = TauEnv_get_memdbg_alignment();
  }
  if (size < align) {
    // Align to the next lower power of two
    align = size;
    while (align & (align-1)) {
      align &= align-1;
    }
  }

  // Round up to the next page boundary
  alloc_size = ((size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
  // Include space for protection pages
  if (PROTECT_ABOVE)
    alloc_size += PAGE_SIZE;
  if (PROTECT_BELOW)
    alloc_size += PAGE_SIZE;
  // Round to next alignment boundary
  if (align > PAGE_SIZE)
    alloc_size += align - PAGE_SIZE;

#if defined(TAU_WINDOWS)

  alloc_addr = (addr_t)VirtualAlloc(NULL, (DWORD)alloc_size, (DWORD)MEM_COMMIT, (DWORD)PAGE_READWRITE);
  if (!alloc_addr) {
    TAU_VERBOSE("TAU: ERROR - VirtualAlloc(%ld) failed\n", alloc_size);
    return NULL;
  }

#else

  static addr_t suggest_start = NULL;

#if defined(MAP_ANONYMOUS)
  alloc_addr = (addr_t)mmap((void*)suggest_start, alloc_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
#else
  static int fd = -1;
  if (fd == -1) {
    if ((fd = open("/dev/zero", O_RDWR)) < 0) {
      TAU_VERBOSE("TAU: ERROR - open() on /dev/zero failed\n");
      return NULL;
    }
  }
  alloc_addr = (addr_t)mmap((void*)suggest_start, alloc_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
#endif

  if (alloc_addr == MAP_FAILED) {
    TAU_VERBOSE("TAU: ERROR - mmap(%ld) failed\n", alloc_size);
    return NULL;
  }

  // Suggest the next allocation begin after this one
  suggest_start = alloc_addr + alloc_size;

#endif

  if (PROTECT_BELOW) {
    if (PROTECT_ABOVE) {
      // TODO
    } else {
      // TODO
    }
  } else if (PROTECT_ABOVE) {
    // Address range with requested alignment and adjacent to guard page
    user_addr = (addr_t)((size_t)(alloc_addr + alloc_size - PAGE_SIZE - size) & ~(align-1));
    user_size = size;
    // Guard page address and size
    prot_addr = (addr_t)((size_t)(user_addr + user_size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
    prot_size = (size_t)(alloc_addr + alloc_size - prot_addr);
    // Gap address and size
    gap_addr = user_addr + user_size;
    gap_size = prot_addr - gap_addr;

    // Permit access to all except the guard page
    UnprotectPages(alloc_addr, (size_t)(prot_addr - alloc_addr));
    // Deny access to the guard page
    ProtectPages(prot_addr, prot_size);
  }

  TheTauAllocationMap()[user_addr] = this;
  TriggerAllocationEvent(filename, lineno);

  // All done with bookkeeping, get back to the user
  return user_addr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::Deallocate(const char * filename, int lineno)
{
#if defined(TAU_WINDOWS)

  addr_t tmp_addr = alloc->addr;
  size_t tmp_size = alloc->size;
  SIZE_T retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  BOOL ret;

  /* release physical memory commited to virtual address space */
  while (tmp_size > 0) {
    retQuery = VirtualQuery((void*)tmp_addr, &MemInfo, sizeof(MemInfo));

    if (retQuery < sizeof(MemInfo)) {
      TAU_VERBOSE("TAU: ERROR - VirtualQuery() failed\n");
    }

    if (MemInfo.State == MEM_COMMIT) {
      ret = VirtualFree((LPVOID)MemInfo.BaseAddress, (DWORD)MemInfo.RegionSize, (DWORD) MEM_DECOMMIT);
      if (!ret) {
        TAU_VERBOSE("TAU: ERROR - VirtualFree(%p,%ld,MEM_DECOMMIT) failed\n", tmp_addr, tmp_size);
      }
    }

    tmp_addr += MemInfo.RegionSize;
    tmp_size -= MemInfo.RegionSize;
  }

  /* release virtual address space */
  ret = VirtualFree((LPVOID)alloc->addr, (DWORD)0, (DWORD)MEM_RELEASE);
  if (!ret) {
    TAU_VERBOSE("TAU: ERROR - VirtualFree(%p, %ld, MEM_RELEASE) failed\n", alloc->addr, alloc->size);
  }

#else

  if (munmap(alloc_addr, alloc_size) < 0) {
    TAU_VERBOSE("TAU: ERROR - munmap(%p, %ld) failed\n", alloc_addr, alloc_size);
  }

  TriggerDeallocationEvent(filename, lineno);
  TheTauAllocationMap().erase(user_addr);

  alloc_addr = NULL;
  alloc_size = 0;
  user_addr = NULL;
  user_size = 0;

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackAllocation(void * ptr, size_t size, const char * filename, int lineno)
{
  addr_t addr = (addr_t)ptr;

  alloc_addr = addr;
  alloc_size = size;
  user_addr = addr;
  user_size = size;
  prot_addr = NULL;
  prot_size = 0;
  gap_addr = NULL;
  gap_size = 0;

  TheTauAllocationMap()[user_addr] = this;
  TriggerAllocationEvent(filename, lineno);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackDeallocation(const char * filename, int lineno)
{
  TriggerDeallocationEvent(filename, lineno);
  TheTauAllocationMap().erase(user_addr);

  alloc_addr = NULL;
  alloc_size = 0;
  user_addr = NULL;
  user_size = 0;
}



//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
size_t Tau_page_size(void)
{
#ifdef PAGE_SIZE
  return (size_t)PAGE_SIZE;
#else
  static size_t page_size = 0;

  if (!page_size) {
#if defined(TAU_WINDOWS)
    SYSTEM_INFO SystemInfo;
    GetSystemInfo(&SystemInfo);
    page_size = (size_t)SystemInfo.dwPageSize;
#elif defined(_SC_PAGESIZE)
    page_size = (size_t)sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
    page_size = (size_t)sysconf(_SC_PAGE_SIZE);
#else
    page_size = getpagesize();
#endif
  }

  return page_size;
#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_detect_memory_leaks(void)
{
  typedef TAU_HASH_MAP<user_event_t*, TauUserEvent*> leak_event_map_t;
  static leak_event_map_t leak_map;

  if (!TauEnv_get_track_memory_leaks()) return;

  allocation_map_t & alloc_map = TheTauAllocationMap();
  if (alloc_map.empty()) return;

  for(allocation_map_t::iterator it=alloc_map.begin(); it != alloc_map.end(); it++) {
    TauAllocation * alloc = it->second;
    size_t size = alloc->user_size;
    user_event_t * event = alloc->event;

    leak_event_map_t::iterator jt = leak_map.find(event);
    if (jt == leak_map.end()) {
      char * s = (char*)malloc(strlen(event->GetEventName())+32);
      sprintf(s, "MEMORY LEAK! %s", event->GetEventName());
      TauUserEvent * leak_event = new TauUserEvent(s);
      leak_map[event] = leak_event;
      leak_event->TriggerEvent(size);
      free((void*)s);
    } else {
      jt->second->TriggerEvent(size);
    }
  }
}



//////////////////////////////////////////////////////////////////////
// Tau_track_memory_allocation does everything that Tau_malloc does except
// allocate memory
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_allocation(void * ptr, size_t size, char const * filename, int lineno)
{
  addr_t addr = (addr_t)ptr;
  TauAllocation * alloc = TheTauAllocationMap(addr);
  if (!alloc) {
    alloc = new TauAllocation;
  } else {
    TAU_VERBOSE("TAU: ERROR - Allocation record for %p already exists\n", addr);
  }
  alloc->TrackAllocation(ptr, size, filename, lineno);
}

//////////////////////////////////////////////////////////////////////
// Tau_track_memory_deallocation does everything that Tau_free does except
// de-allocate memory
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_deallocation(void * ptr, char const * filename, int lineno)
{
  addr_t addr = (addr_t)ptr;
  TauAllocation * alloc = TheTauAllocationMap(addr);
  if (alloc) {
    alloc->TrackDeallocation(filename, lineno);
    delete alloc;
  } else {
    TAU_VERBOSE("TAU: WARNING - No allocation record found for %p\n", addr);
  }
}

//////////////////////////////////////////////////////////////////////
// Tau_new returns the expression (new[] foo) and  does everything that
// Tau_track_memory_allocation does
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_new(void * ptr, size_t size, char const * filename, int lineno)
{
  /* the memory is already allocated by the time we see this ptr */
  Tau_track_memory_allocation(ptr, size, filename, lineno);
  return ptr;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_malloc(size_t size, const char * filename, int lineno)
{
  void * ptr;

  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(0, size, filename, lineno);
  } else {
    ptr = malloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }

  return ptr;
}


//////////////////////////////////////////////////////////////////////
// Tau_calloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_calloc(size_t count, size_t size, const char * filename, int lineno)
{
  void * ptr;

  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(0, count*size, filename, lineno);
  } else {
    ptr = calloc(count, size);
    Tau_track_memory_allocation(ptr, count*size, filename, lineno);
  }

  return ptr;
}


//////////////////////////////////////////////////////////////////////
// Tau_free calls Tau_free_before and free's the memory allocated
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_free(void * ptr, char const * filename, int lineno)
{
  addr_t addr = (addr_t)ptr;
  TauAllocation * alloc = TheTauAllocationMap(addr);

  if (alloc) {
    if (TauEnv_get_memdbg()) {
      alloc->Deallocate(filename, lineno);
    } else {
      alloc->TrackDeallocation(filename, lineno);
      free(ptr);
    }
    delete alloc;
  } else {
    TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found\n", addr);
    free(ptr);
  }
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
#if HAVE_MEMALIGN
extern "C"
void * Tau_memalign(size_t alignment, size_t size, const char * filename, int lineno)
{
  void * ptr;

  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(alignment, size, filename, lineno);
  } else {
    ptr = memalign(alignment, size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }

  return ptr;
}
#endif


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_posix_memalign(void **ptr, size_t alignment, size_t size,
    const char * filename, int lineno)
{
  int retval;

  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    *ptr = alloc->Allocate(alignment, size, filename, lineno);
  } else {
    retval = posix_memalign(ptr, alignment, size);
    Tau_track_memory_allocation(*ptr, size, filename, lineno);
  }

  return retval;
}


//////////////////////////////////////////////////////////////////////
// Tau_realloc calls free_before, realloc and memory allocation tracking routine
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_realloc(void * ptr, size_t size, const char * filename, int lineno)
{
  // realloc can be called with NULL in some implementations

  if (TauEnv_get_memdbg()) {
    if (ptr) {
      Tau_free(ptr, filename, lineno);
    }
    ptr = Tau_malloc(size, filename, lineno);
  } else {
    if (ptr) {
      Tau_track_memory_deallocation(ptr, filename, lineno);
    }
    ptr = realloc(ptr, size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_valloc(size_t size, const char * filename, int lineno)
{
  void * ptr;

  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(Tau_page_size(), size, filename, lineno);
  } else {
    ptr = valloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }

  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
#if HAVE_PVALLOC
extern "C"
void * Tau_pvalloc(size_t size, const char * filename, int lineno)
{
#ifndef PAGE_SIZE
  static size_t const PAGE_SIZE = Tau_page_size();
#endif

  void * ptr;

  // pvalloc allocates the smallest set of complete pages
  // that can hold the requested number of bytes
  size = (size + PAGE_SIZE-1) & ~(PAGE_SIZE-1);

  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(Tau_page_size(), size, filename, lineno);
  } else {
    ptr = pvalloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }

  return ptr;
}
#endif

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
char * Tau_strdup(const char *str, const char * filename, int lineno)
{
  char * ptr;

  size_t size = strlen(str) + 1;
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = (char*)alloc->Allocate(0, size, filename, lineno);
    memcpy((void*)ptr, (void*)str, size);
  } else {
    ptr = strdup(str);
    Tau_track_memory_allocation((void*)ptr, size, filename, lineno);
  }

  return ptr;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_memcpy(void *dst, const void *src, size_t size, const char * filename, int lineno)
{
  return memcpy(dst, src, size);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
char * Tau_strcpy(char *dst, const char *src, const char * filename, int lineno)
{
  return strcpy(dst, src);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
char * Tau_strncpy(char *dst, const char *src, size_t size, const char * filename, int lineno)
{
  return strncpy(dst, src, size);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
char * Tau_strcat(char *dst, const char *src, const char * filename, int lineno)
{
  return strcat(dst, src);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
char * Tau_strncat(char *dst, const char *src, size_t size, const char * filename, int lineno)
{
  return strncat(dst, src, size);
}


/***************************************************************************
 * $RCSfile: TauMemory.cpp,v $   $Author: amorris $
 * $Revision: 1.33 $   $Date: 2010/01/27 00:47:51 $
 * TAU_VERSION_ID: $Id: TauMemory.cpp,v 1.33 2010/01/27 00:47:51 amorris Exp $ 
 ***************************************************************************/
