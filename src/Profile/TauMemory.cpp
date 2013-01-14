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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <Profile/Profiler.h>
#include <Profile/TauMemory.h>
#include <Profile/TauInit.h>
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
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#include <map.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

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

using namespace std;

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define  MAP_ANONYMOUS MAP_ANON
#endif


static int & TheTauMemoryWrapperPresent(void)
{
  static int flag = 0;
  return flag;
}

extern "C"
int Tau_memory_wrapper_present(void)
{
  return TheTauMemoryWrapperPresent();
}

extern "C"
void Tau_set_memory_wrapper_present(int value)
{
  TheTauMemoryWrapperPresent() = value;
}



typedef unsigned char * addr_t;
typedef TauContextUserEvent user_event_t;

class TauAllocation
{

public:

  typedef TAU_HASH_MAP<addr_t, class TauAllocation*> allocation_map_t;
  static allocation_map_t & AllocationMap() {
    static allocation_map_t alloc_map;
    return alloc_map;
  }

  static TauAllocation * Find(allocation_map_t::key_type const & key) {
    allocation_map_t const & alloc_map = AllocationMap();
    allocation_map_t::const_iterator it = alloc_map.find(key);
    if (it != alloc_map.end())
      return it->second;
    return NULL;
  }

  static size_t & BytesAllocated() {
    // Not thread safe!
    static size_t bytes = 0;
    return bytes;
  }

  static void DetectLeaks(void);

public:

  TauAllocation() :
    alloc_addr(NULL), alloc_size(0),
    user_addr(NULL), user_size(0),
    lguard_addr(NULL), lguard_size(0),
    uguard_addr(NULL), uguard_size(0),
    lgap_addr(NULL), lgap_size(0),
    ugap_addr(NULL), ugap_size(0)
  { }

  void * Allocate(size_t const size, size_t align, size_t min_align, const char * filename, int lineno);
  void Deallocate(const char * filename, int lineno);

  void TrackAllocation(void * ptr, size_t size, const char * filename, int lineno);
  void TrackDeallocation(const char * filename, int lineno);

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

  user_event_t * event; ///< Allocation event (for leak detection)

  void ProtectPages(addr_t addr, size_t size);
  void UnprotectPages(addr_t addr, size_t size);

  unsigned long LocationHash(unsigned long hash, char const * data);
  void TriggerAllocationEvent(char const * filename, int lineno);
  void TriggerDeallocationEvent(char const * filename, int lineno);
  void TriggerErrorEvent(char const * descript, char const * filename, int lineno);
};



// Incremental string hashing function.
// Uses Paul Hsieh's SuperFastHash, the same as in Google Chrome.
unsigned long TauAllocation::LocationHash(unsigned long hash, char const * data)
{
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )

  uint32_t tmp;
  int len;
  int rem;

  // Optimize for the common case
  if (hash == TAU_MEMORY_UNKNOWN_LINE) {
    // If we suspect the common case, compare data to TAU_MEMORY_UNKNOWN_FILE
    // while simultaneously getting strlen(data).
    char const * pd = data;
    char const * pu = TAU_MEMORY_UNKNOWN_FILE;
    len = 0;
    while (*pu && *pd) {
      ++pu;
      ++pd;
      ++len;
    }
    if (len == TAU_MEMORY_UNKNOWN_FILE_STRLEN && *pu == *pd) {
      // Return the pre-computed hash.
      return 7558261977980762395UL;
    } else {
      // Finish getting the length
      while (*pd) {
        ++pd;
        ++len;
      }
    }
  } else {
    len = strlen(data);
  }

  // Loop unrolling
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
  Tau_global_incr_insideTAU();

  typedef TAU_HASH_MAP<unsigned long, user_event_t*> event_map_t;
  static event_map_t event_map;

  unsigned long file_hash = LocationHash(lineno, filename);

  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    char * s;
    if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
        !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
    {
      event = new user_event_t("heap allocate");
    } else {
      char * s = (char*)malloc(strlen(filename)+128);
      sprintf(s, "heap allocate <file=%s, line=%d>", filename, lineno);
      event = new user_event_t(s);
      free((void*)s);
    }
    event_map[file_hash] = event;
  } else {
    event = it->second;
  }
  event->TriggerEvent(user_size);

  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerDeallocationEvent(char const * filename, int lineno)
{
  Tau_global_incr_insideTAU();

  typedef TAU_HASH_MAP<unsigned long, user_event_t*> event_map_t;
  static event_map_t event_map;

  unsigned long file_hash = LocationHash(lineno, filename);

  user_event_t * e;

  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
        !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
    {
      e = new user_event_t("heap free");
    } else {
      char * s = (char*)malloc(strlen(filename)+128);
      sprintf(s, "heap free <file=%s, line=%d>", filename, lineno);
      e = new user_event_t(s);
      free((void*)s);
    }
    event_map[file_hash] = e;
  } else {
    e = it->second;
  }
  e->TriggerEvent(user_size);

  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerErrorEvent(char const * descript, char const * filename, int lineno)
{
  Tau_global_incr_insideTAU();

  typedef TAU_HASH_MAP<unsigned long, user_event_t*> event_map_t;
  static event_map_t event_map;

  unsigned long file_hash = LocationHash(lineno, filename);

  user_event_t * e;

  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    char * s;
    if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
        !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
    {
      s = (char*)malloc(strlen(descript)+128);
      sprintf(s, "MEMORY ERROR! %s", descript);
    } else {
      s = (char*)malloc(strlen(descript)+strlen(filename)+128);
      sprintf(s, "MEMORY ERROR! %s <file=%s, line=%d>", descript, filename, lineno);
    }
    e = new user_event_t(s);
    event_map[file_hash] = e;
    free((void*)s);
  } else {
    e = it->second;
  }
  e->TriggerEvent(user_size);

  Tau_global_decr_insideTAU();
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::DetectLeaks(void)
{
  Tau_global_incr_insideTAU();

  typedef TAU_HASH_MAP<user_event_t*, TauUserEvent*> leak_event_map_t;
  static leak_event_map_t leak_map;

  allocation_map_t & alloc_map = AllocationMap();
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
  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::ProtectPages(addr_t addr, size_t size)
{
  Tau_global_incr_insideTAU();
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
  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::UnprotectPages(addr_t addr, size_t size)
{
  Tau_global_incr_insideTAU();
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
  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * TauAllocation::Allocate(size_t size, size_t align, size_t min_align,
    const char * filename, int lineno)
{
#ifndef PAGE_SIZE
  static size_t const PAGE_SIZE = Tau_page_size();
#endif
  static bool const PROTECT_ABOVE = TauEnv_get_memdbg_protect_above();
  static bool const PROTECT_BELOW = TauEnv_get_memdbg_protect_below();

  Tau_global_incr_insideTAU();

  // Check size
  if (!size && !TauEnv_get_memdbg_zero_malloc()) {
    TriggerErrorEvent("Allocation of zero bytes.", filename, lineno);
    Tau_global_decr_insideTAU();
    return NULL;
  }

  // Check alignment
  if(!align) {
    align = TauEnv_get_memdbg_alignment();

    if (size < align) {
      // Align to the next lower power of two
      align = size;
      while (align & (align-1)) {
        align &= align-1;
      }
    }
  }

  // Alignment must be a power of two
  if ((int)align != ((int)align & -(int)align)) {
    TriggerErrorEvent("Alignment is not a power of two", filename, lineno);
    Tau_global_decr_insideTAU();
    return NULL;
  }

  // Alignment must be a multiple of the minimum alignment (a power of two)
  if (min_align && ((align < min_align) || (align & (min_align-1)))) {
    char s[256];
    sprintf(s, "Alignment is not a multiple of %d", min_align);
    TriggerErrorEvent(s, filename, lineno);
    Tau_global_decr_insideTAU();
    return NULL;
  }

  // Round up to the next page boundary
  alloc_size = ((size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
  // Include space for protection pages
  if (PROTECT_ABOVE)
    alloc_size += PAGE_SIZE;
  if (PROTECT_BELOW)
    alloc_size += PAGE_SIZE;
  if (align > PAGE_SIZE)
    alloc_size += align - PAGE_SIZE;

#if defined(TAU_WINDOWS)

  alloc_addr = (addr_t)VirtualAlloc(NULL, (DWORD)alloc_size, (DWORD)MEM_COMMIT, (DWORD)PAGE_READWRITE);
  if (!alloc_addr) {
    TAU_VERBOSE("TAU: ERROR - VirtualAlloc(%ld) failed\n", alloc_size);
    Tau_global_decr_insideTAU();
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
      Tau_global_decr_insideTAU();
      return NULL;
    }
  }
  alloc_addr = (addr_t)mmap((void*)suggest_start, alloc_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
#endif

  if (alloc_addr == MAP_FAILED) {
    TAU_VERBOSE("TAU: ERROR - mmap(%ld) failed\n", alloc_size);
    Tau_global_decr_insideTAU();
    return NULL;
  }

  // Suggest the next allocation begin after this one
  suggest_start = alloc_addr + alloc_size;

#endif

  if (PROTECT_BELOW) {
    // Address range with requested alignment and adjacent to guard page
    user_addr = (addr_t)(((size_t)alloc_addr + PAGE_SIZE + align-1) & ~(align-1));
    user_size = size;
    // Lower guard page address and size
    lguard_addr = alloc_addr;
    lguard_size = (size_t)(user_addr - alloc_addr) & ~(PAGE_SIZE-1);
    // Front gap address and size
    lgap_addr = (addr_t)((size_t)user_addr & ~(PAGE_SIZE-1));
    lgap_size = (size_t)(user_addr - lgap_addr);

    if (PROTECT_ABOVE) {
      // Upper guard page address and size
      uguard_addr = (addr_t)((size_t)(user_addr + user_size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
      uguard_size = (size_t)(alloc_addr + alloc_size - uguard_addr);
      // Back gap address and size
      ugap_addr = user_addr + user_size;
      ugap_size = (size_t)(uguard_addr - ugap_addr);

      // Permit access to all except the upper and lower guard pages
      UnprotectPages(lgap_addr, (size_t)(uguard_addr - lgap_addr));
      // Deny access to the lower guard page
      ProtectPages(lguard_addr, lguard_size);
      // Deny access to the upper guard page
      ProtectPages(uguard_addr, uguard_size);

    } else {
      // Upper guard page address and size
      uguard_addr = NULL;
      uguard_size = 0;
      // Back gap address and size
      ugap_addr = user_addr + user_size;
      ugap_size = (size_t)(alloc_addr + alloc_size - ugap_addr);

      // Permit access to all except the lower guard page
      UnprotectPages(lgap_addr, (size_t)(alloc_addr + alloc_size - lgap_addr));
      // Deny access to the lower guard page
      ProtectPages(lguard_addr, lguard_size);
    }
  } else if (PROTECT_ABOVE) {
    // Address range with requested alignment and adjacent to guard page
    user_addr = (addr_t)(((size_t)alloc_addr + alloc_size - PAGE_SIZE - size) & ~(align-1));
    user_size = size;
    // Lower guard page address and size
    lguard_addr = NULL;
    lguard_size = 0;
    // Upper guard page address and size
    uguard_addr = (addr_t)((size_t)(user_addr + user_size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
    uguard_size = (size_t)(alloc_addr + alloc_size - uguard_addr);
    // Front gap address and size
    lgap_addr = alloc_addr;
    lgap_size = (size_t)(user_addr - alloc_addr);
    // Back gap address and size
    ugap_addr = user_addr + user_size;
    ugap_size = (size_t)(uguard_addr - ugap_addr);

    // Permit access to all except the upper guard page
    UnprotectPages(alloc_addr, (size_t)(uguard_addr - alloc_addr));
    // Deny access to the upper guard page
    ProtectPages(uguard_addr, uguard_size);
  }

  BytesAllocated() += user_size;
  AllocationMap()[user_addr] = this;
  TriggerAllocationEvent(filename, lineno);

//  printf("%s:%d :: %p, %ld, %p, %ld, %p, %ld, %p, %ld, %p, %ld\n",
//      filename, lineno, alloc_addr, alloc_size, user_addr, user_size,
//      prot_addr, prot_size, lgap_addr, lgap_size, ugap_addr, ugap_size);
//  fflush(stdout);

  // All done with bookkeeping, get back to the user
  Tau_global_decr_insideTAU();
  return user_addr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::Deallocate(const char * filename, int lineno)
{
  Tau_global_incr_insideTAU();
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
  BytesAllocated() -= user_size;
  AllocationMap().erase(user_addr);

  alloc_addr = NULL;
  alloc_size = 0;
  user_addr = NULL;
  user_size = 0;

#endif
  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackAllocation(void * ptr, size_t size, const char * filename, int lineno)
{
  Tau_global_incr_insideTAU();
  addr_t addr = (addr_t)ptr;

  if (!alloc_addr) {
    alloc_addr = addr;
    alloc_size = size;
    user_addr = addr;
    user_size = size;
  }

  AllocationMap()[user_addr] = this;
  TriggerAllocationEvent(filename, lineno);
  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackDeallocation(const char * filename, int lineno)
{
  Tau_global_incr_insideTAU();
  TriggerDeallocationEvent(filename, lineno);
  AllocationMap().erase(user_addr);
  Tau_global_decr_insideTAU();
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

  Tau_global_incr_insideTAU();
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
  Tau_global_decr_insideTAU();

  return page_size;
#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_detect_memory_leaks(void)
{
  Tau_global_incr_insideTAU();
  if (TauEnv_get_track_memory_leaks()) {
    TauAllocation::DetectLeaks();
  }
  Tau_global_decr_insideTAU();
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
size_t Tau_get_bytes_allocated(void)
{
  return TauAllocation::BytesAllocated();
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_allocation(void * ptr, size_t size, char const * filename, int lineno)
{
  //printf("%s\n", __PRETTY_FUNCTION__); fflush(stdout);

  Tau_global_incr_insideTAU();
  addr_t addr = (addr_t)ptr;
  TauAllocation * alloc = TauAllocation::Find(addr);
  if (!alloc) {
    //printf("%s: new TauAllocation for %p\n", __PRETTY_FUNCTION__, (char*)ptr); fflush(stdout);
    alloc = new TauAllocation;
    alloc->TrackAllocation(ptr, size, filename, lineno);
  } else {
    TAU_VERBOSE("TAU: WARNING - Allocation record for %p already exists\n", addr);
  }
  Tau_global_decr_insideTAU();
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_deallocation(void * ptr, char const * filename, int lineno)
{
  Tau_global_incr_insideTAU();
  addr_t addr = (addr_t)ptr;
  TauAllocation * alloc = TauAllocation::Find(addr);
  if (alloc) {
    alloc->TrackDeallocation(filename, lineno);
    // TrackDeallocation triggers an event, so deleting alloc is not crazy
    delete alloc;
  } else {
    TAU_VERBOSE("TAU: WARNING - No allocation record found for %p\n", addr);
  }
  Tau_global_decr_insideTAU();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_malloc(size_t size, const char * filename, int lineno)
{
  //printf("%s\n", __PRETTY_FUNCTION__); fflush(stdout);
  void * ptr;

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    //printf("%s: alloc->Allocate\n", __PRETTY_FUNCTION__); fflush(stdout);
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, 0, 0, filename, lineno);
  } else {
    //printf("%s: malloc\n", __PRETTY_FUNCTION__); fflush(stdout);
    ptr = malloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

  return ptr;
}


//////////////////////////////////////////////////////////////////////
// Tau_calloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_calloc(size_t count, size_t size, const char * filename, int lineno)
{
  void * ptr;

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(count*size, 0, 0, filename, lineno);
  } else {
    ptr = calloc(count, size);
    Tau_track_memory_allocation(ptr, count*size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

  return ptr;
}


//////////////////////////////////////////////////////////////////////
// Tau_free calls Tau_free_before and free's the memory allocated
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_free(void * ptr, char const * filename, int lineno)
{
  //printf("%s\n", __PRETTY_FUNCTION__); fflush(stdout);

  if (ptr) {
    addr_t addr = (addr_t)ptr;
    TauAllocation * alloc = TauAllocation::Find(addr);

    Tau_global_incr_insideTAU();
    if (alloc) {
      //printf("%s: alloc found\n", __PRETTY_FUNCTION__); fflush(stdout);
      if (TauEnv_get_memdbg()) {
        //printf("%s: alloc->Deallocate\n", __PRETTY_FUNCTION__); fflush(stdout);
        alloc->Deallocate(filename, lineno);
      } else {
        //printf("%s: free\n", __PRETTY_FUNCTION__); fflush(stdout);
        alloc->TrackDeallocation(filename, lineno);
        free(ptr);
      }
      delete alloc;
    } else {
      TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", addr);
      free(ptr);
    }
    Tau_global_decr_insideTAU();
  }
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
#ifdef HAVE_MEMALIGN
extern "C"
void * Tau_memalign(size_t alignment, size_t size, const char * filename, int lineno)
{
  void * ptr;

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, alignment, 0, filename, lineno);
  } else {
    ptr = memalign(alignment, size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

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

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    *ptr = alloc->Allocate(size, alignment, sizeof(void*), filename, lineno);
    retval = (ptr != NULL);
  } else {
    retval = posix_memalign(ptr, alignment, size);
    Tau_track_memory_allocation(*ptr, size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

  return retval;
}


//////////////////////////////////////////////////////////////////////
// Tau_realloc calls free_before, realloc and memory allocation tracking routine
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_realloc(void * ptr, size_t size, const char * filename, int lineno)
{
  Tau_global_incr_insideTAU();
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
  Tau_global_decr_insideTAU();

  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_valloc(size_t size, const char * filename, int lineno)
{
  void * ptr;

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, Tau_page_size(), 0, filename, lineno);
  } else {
    ptr = valloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
#ifdef HAVE_PVALLOC
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

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, Tau_page_size(), 0, filename, lineno);
  } else {
    ptr = pvalloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

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

  Tau_global_incr_insideTAU();
  if (TauEnv_get_memdbg()) {
    TauAllocation * alloc = new TauAllocation;
    ptr = (char*)alloc->Allocate(size, 0, 0, filename, lineno);
    memcpy((void*)ptr, (void*)str, size);
  } else {
    ptr = strdup(str);
    Tau_track_memory_allocation((void*)ptr, size, filename, lineno);
  }
  Tau_global_decr_insideTAU();

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
