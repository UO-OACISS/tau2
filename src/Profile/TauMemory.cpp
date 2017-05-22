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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tau_types.h>
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
#include <errno.h>
#ifdef __PIN__
void *pvalloc(size_t size) { 
   return malloc(size); 
}
#endif /* __PIN__ */

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
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#if !defined(TAU_CATAMOUNT) && (defined(__QK_USER__) || defined(__LIBCATAMOUNT__ ))
#define TAU_CATAMOUNT
#endif /* __QK_USER__ || __LIBCATAMOUNT__ */
#ifdef TAU_CATAMOUNT
#include <catamount/catmalloc.h>
#endif /* TAU_CATAMOUNT */

#ifdef TAU_BEACON
#include <Profile/TauBeacon.h>
#endif /* TAU_BEACON */

using namespace std;
using namespace tau;

// MAP_ANON is a synonym for MAP_ANONYMOUS
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define  MAP_ANONYMOUS MAP_ANON
#endif

#include <Profile/TauPin.h>

bool wrapper_registered = false;
wrapper_enable_handle_t wrapper_enable_handle = NULL;
wrapper_disable_handle_t wrapper_disable_handle = NULL;

extern "C" int Tau_trigger_memory_rss_hwm(void);

// Returns true if the given allocation size should be protected
static inline bool AllocationShouldBeProtected(size_t size)
{
  return (TauEnv_get_memdbg() && !(
      (TauEnv_get_memdbg_overhead() && (TauEnv_get_memdbg_overhead_value() < TauAllocation::BytesOverhead())) ||
      (TauEnv_get_memdbg_alloc_min() && (size < TauEnv_get_memdbg_alloc_min_value())) ||
      (TauEnv_get_memdbg_alloc_max() && (size > TauEnv_get_memdbg_alloc_max_value()))));
}

static inline void BuildTimerName(char * buff, char const * funcname, char const * filename, int lineno)
{
  if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
      !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
  {
    sprintf(buff, "%s", funcname);
  } else {
    sprintf(buff, "%s [{%s} {%d,1}-{%d,1}]", funcname, filename, lineno, lineno);
  }
}

//////////////////////////////////////////////////////////////////////
// Triggers leak detection
//////////////////////////////////////////////////////////////////////
void TauAllocation::DetectLeaks(void)
{

  allocation_map_t const & alloc_map = AllocationMap();
  if (alloc_map.empty()) {
    TAU_VERBOSE("TAU: No memory leaks detected");
    return;
  }

  leak_event_map_t & leak_map = __leak_event_map();
  TAU_VERBOSE("TAU: There are %d memory leaks", leak_map.size());

  for(allocation_map_t::const_iterator it=alloc_map.begin(); it != alloc_map.end(); it++) {
    TauAllocation * alloc = it->second;
    size_t size = alloc->user_size;
    TauUserEvent * event = alloc->alloc_event;

    leak_event_map_t::iterator jt = leak_map.find(event);
    if (jt == leak_map.end()) {
      TauSafeString tmp("MEMORY LEAK! " + event->GetName());
      TauUserEvent * leak_event = new TauUserEvent(tmp.c_str());
      leak_map[event] = leak_event;
      leak_event->TriggerEvent(size);
    } else {
      jt->second->TriggerEvent(size);
    }
  }
}

//////////////////////////////////////////////////////////////////////
// Read/write allocation map
//////////////////////////////////////////////////////////////////////
TauAllocation::allocation_map_t & TauAllocation::__allocation_map()
{
  static allocation_map_t alloc_map;
  return alloc_map;
}

//////////////////////////////////////////////////////////////////////
// Read/write leak event map
//////////////////////////////////////////////////////////////////////
TauAllocation::leak_event_map_t & TauAllocation::__leak_event_map()
{
  static leak_event_map_t leak_event_map;
  return leak_event_map;
}

//////////////////////////////////////////////////////////////////////
// Total bytes allocated
//////////////////////////////////////////////////////////////////////
size_t & TauAllocation::__bytes_allocated()
{
  static size_t bytes = 0;
  return bytes;
}

//////////////////////////////////////////////////////////////////////
// Total bytes deallocated
//////////////////////////////////////////////////////////////////////
size_t & TauAllocation::__bytes_deallocated()
{
  static size_t bytes = 0;
  return bytes;
}

//////////////////////////////////////////////////////////////////////
// Bytes of memory protection overhead
//////////////////////////////////////////////////////////////////////
size_t & TauAllocation::__bytes_overhead()
{
  static size_t bytes = 0;
  return bytes;
}

//////////////////////////////////////////////////////////////////////
// Search for an allocation record by base address
//////////////////////////////////////////////////////////////////////
TauAllocation * TauAllocation::Find(allocation_map_t::key_type const & key)
{
  TauAllocation * found = NULL;
  if (key) {
    RtsLayer::LockDB();
    allocation_map_t const & alloc_map = AllocationMap();
    allocation_map_t::const_iterator it = alloc_map.find(key);
    if (it != alloc_map.end()) {
      found = it->second;
    }
    RtsLayer::UnLockDB();
  }
  return found;
}

//////////////////////////////////////////////////////////////////////
// Find the allocation record that contains the given address
//////////////////////////////////////////////////////////////////////
TauAllocation * TauAllocation::FindContaining(void * ptr)
{
  TauAllocation * found = NULL;
  if (ptr) {
    RtsLayer::LockDB();
    allocation_map_t const & allocMap = AllocationMap();
    allocation_map_t::const_iterator it;
    for(it = allocMap.begin(); it != allocMap.end(); it++) {
      TauAllocation * const alloc = it->second;
      if (alloc->Contains(ptr)) {
        found = alloc;
        break;
      }
    }
    RtsLayer::UnLockDB();
  }
  return found;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * TauAllocation::Allocate(size_t size, size_t align, size_t min_align,
    const char * filename, int lineno)
{
#ifndef PAGE_SIZE
  size_t const PAGE_SIZE = Tau_page_size();
#endif
  bool const protect_above = TauEnv_get_memdbg_protect_above();
  bool const protect_below = TauEnv_get_memdbg_protect_below();
  bool const fill_gap = TauEnv_get_memdbg_fill_gap();

  // Fail or not, this is a TAU allocation not a tracked system allocation
  tracked = false;

  // Check size
  if (!size && !TauEnv_get_memdbg_zero_malloc()) {
    TriggerErrorEvent("Allocation of zero bytes", filename, lineno);
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
    return NULL;
  }

  // Alignment must be a multiple of the minimum alignment (a power of two)
  if (min_align && ((align < min_align) || (align & (min_align-1)))) {
    char s[256];
    sprintf(s, "Alignment is not a multiple of %ld", min_align);
    TriggerErrorEvent(s, filename, lineno);
    return NULL;
  }

  // Round up to the next page boundary
  alloc_size = ((size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
  // Include space for protection pages
  if (protect_above)
    alloc_size += PAGE_SIZE;
  if (protect_below)
    alloc_size += PAGE_SIZE;
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
  alloc_addr = (addr_t)mmap((void*)suggest_start, alloc_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
#else
  static int fd = -1;
  if (fd == -1) {
    if ((fd = open("/dev/zero", O_RDWR)) < 0) {
      TAU_VERBOSE("TAU: ERROR - open() on /dev/zero failed\n");
      return NULL;
    }
  }
  alloc_addr = (addr_t)mmap((void*)suggest_start, alloc_size, PROT_NONE, MAP_PRIVATE, fd, 0);
#endif

  if (alloc_addr == MAP_FAILED) {
    char * errstr = strerror(errno);
    TAU_VERBOSE("TAU: ERROR - mmap(%ld) failed: %s\n", alloc_size, errstr);
    return NULL;
  }

  // Suggest the next allocation begin after this one
  suggest_start = alloc_addr + alloc_size;

#endif

  if (protect_below) {
    // Address range with requested alignment and adjacent to guard page
    user_addr = (addr_t)(((size_t)alloc_addr + PAGE_SIZE + align-1) & ~(align-1));
    user_size = size;
    // Lower guard page address and size
    lguard_addr = alloc_addr;
    lguard_size = (size_t)(user_addr - alloc_addr) & ~(PAGE_SIZE-1);
    // Front gap address and size
    lgap_addr = (addr_t)((size_t)user_addr & ~(PAGE_SIZE-1));
    lgap_size = (size_t)(user_addr - lgap_addr);

    if (protect_above) {
      // Upper guard page address and size
      uguard_addr = (addr_t)((size_t)(user_addr + user_size + PAGE_SIZE-1) & ~(PAGE_SIZE-1));
      uguard_size = (size_t)(alloc_addr + alloc_size - uguard_addr);
      // Back gap address and size
      ugap_addr = user_addr + user_size;
      ugap_size = (size_t)(uguard_addr - ugap_addr);

      // Permit access to all except the upper and lower guard pages
      Unprotect(lgap_addr, (size_t)(uguard_addr - lgap_addr));
      // Deny access to the lower guard page
      Protect(lguard_addr, lguard_size);
      // Deny access to the upper guard page
      Protect(uguard_addr, uguard_size);
    } else {
      // Upper guard page address and size
      uguard_addr = NULL;
      uguard_size = 0;
      // Back gap address and size
      ugap_addr = user_addr + user_size;
      ugap_size = (size_t)(alloc_addr + alloc_size - ugap_addr);

      // Permit access to all except the lower guard page
      Unprotect(lgap_addr, (size_t)(alloc_addr + alloc_size - lgap_addr));
      // Deny access to the lower guard page
      Protect(lguard_addr, lguard_size);
    }
  } else if (protect_above) {
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
    Unprotect(alloc_addr, (size_t)(uguard_addr - alloc_addr));
    // Deny access to the upper guard page
    Protect(uguard_addr, uguard_size);
  }

  if (fill_gap) {
    unsigned char const fill_pattern = TauEnv_get_memdbg_fill_gap_value();
    if(lgap_size) memset(lgap_addr, fill_pattern, lgap_size);
    if(ugap_size) memset(ugap_addr, fill_pattern, ugap_size);
  }

  RtsLayer::LockDB();
  __bytes_allocated() += user_size;
  __bytes_overhead() += alloc_size - user_size;
  __allocation_map()[user_addr] = this;
  RtsLayer::UnLockDB();

  allocated = true;
  TriggerAllocationEvent(user_size, filename, lineno);
  TriggerMemDbgOverheadEvent();
  TriggerHeapMemoryUsageEvent();

//  printf("%s:%d :: alloc=(%p, %ld), user=(%p, %ld), "
//      "lguard=(%p, %ld), uguard=(%p, %ld), lgap=(%p, %ld), ugap=(%p, %ld)\n",
//      filename, lineno, alloc_addr, alloc_size, user_addr, user_size,
//      lguard_addr, lguard_size, uguard_addr, uguard_size,
//      lgap_addr, lgap_size, ugap_addr, ugap_size);
//  fflush(stdout);

  // All done with bookkeeping, get back to the user
  return user_addr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::Deallocate(const char * filename, int lineno)
{
  bool const protect_free = TauEnv_get_memdbg_protect_free();

  // Fail or not, this is a TAU allocation not a tracked system allocation
  tracked = false;

  // Error if this allocation has already been deallocated
  if (!allocated) {
    TriggerErrorEvent("Deallocation of unallocated memory", filename, lineno);
    return;
  }

  // Mark record as deallocated
  allocated = false;

  if (!protect_free) {
#if defined(TAU_WINDOWS)

  addr_t tmp_addr = alloc_addr;
  size_t tmp_size = alloc_size;
  SIZE_T retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  BOOL ret;

  // release physical memory committed to virtual address space
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

  // release virtual address space
  ret = VirtualFree((LPVOID)alloc_addr, (DWORD)0, (DWORD)MEM_RELEASE);
  if (!ret) {
    TAU_VERBOSE("TAU: ERROR - VirtualFree(%p, %ld, MEM_RELEASE) failed\n", alloc_addr, alloc_size);
  }
#else
    if (munmap(alloc_addr, alloc_size) < 0) {
      char * errstr = strerror(errno);
      TAU_VERBOSE("TAU: ERROR - munmap(%p, %ld) failed: %s\n", alloc_addr, alloc_size, errstr);
    }
#endif
  } else {
    // TAU_MEMDBG_PROTECT_FREE is set, so don't deallocate just protect
    Protect(alloc_addr, alloc_size);
  }

  RtsLayer::LockDB();
  __bytes_deallocated() += user_size;
  if (protect_free) {
    __bytes_overhead() += user_size;
  } else {
    __bytes_overhead() -= alloc_size - user_size;
    __allocation_map().erase(user_addr);
  }
  RtsLayer::UnLockDB();

  TriggerDeallocationEvent(user_size, filename, lineno);
  TriggerMemDbgOverheadEvent();
  TriggerHeapMemoryUsageEvent();

  if (!protect_free) {
    // If free memory isn't protected then we don't need this record anymore
    delete this;
  }
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * TauAllocation::Reallocate(size_t size, size_t align, size_t min_align,
    const char * filename, int lineno)
{
  TauAllocation * resized = new TauAllocation(*this);
  size_t copy_size = (size > user_size) ? user_size : size;

  void * ptr = resized->Allocate(size, align, min_align, filename, lineno);
  if (ptr) {
    memcpy(ptr, (void*)user_addr, copy_size);
    Deallocate(filename, lineno);
  } else {
    delete resized;
  }

  TriggerHeapMemoryUsageEvent();
  return ptr;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackAllocation(void * ptr, size_t size, const char * filename, int lineno)
{
  tracked = true;
  allocated = true;

  if (size) {
    if (!alloc_addr) {
      addr_t addr = (addr_t)ptr;
      alloc_addr = addr;
      alloc_size = size;
      user_addr = addr;
      user_size = size;
    }
    RtsLayer::LockDB();
    __bytes_allocated() += user_size;
    __allocation_map()[user_addr] = this;
    RtsLayer::UnLockDB();

    TriggerAllocationEvent(user_size, filename, lineno);
    TriggerHeapMemoryUsageEvent();
  } else if (!TauEnv_get_memdbg_zero_malloc()) {
    TriggerErrorEvent("Allocation of zero bytes", filename, lineno);
  }
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackDeallocation(const char * filename, int lineno)
{
  tracked = true;
  allocated = false;

  RtsLayer::LockDB();
  __bytes_deallocated() += user_size;
  __allocation_map().erase(user_addr);
  RtsLayer::UnLockDB();

  TriggerDeallocationEvent(user_size, filename, lineno);
  TriggerHeapMemoryUsageEvent();

  // We don't need this record anymore
  delete this;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TrackReallocation(void * ptr, size_t size, const char * filename, int lineno)
{
  addr_t addr = (addr_t)ptr;

  // Don't do anything if nothing changed
  if (user_addr == addr && user_size == size) return;

  if (user_addr) {
    if (size) {
      if (user_addr == addr) {
        if (user_size > size) {
          TriggerDeallocationEvent(user_size - size, filename, lineno);
        } else {
          TriggerAllocationEvent(size - user_size, filename, lineno);
        }
        tracked = true;
        allocated = true;
        user_size = size;
        alloc_size = size;
      } else {
        // Track deallocation of old memory without destroying this object
        RtsLayer::LockDB();
        __bytes_deallocated() += user_size;
        __allocation_map().erase(user_addr);
        RtsLayer::UnLockDB();
        TriggerDeallocationEvent(user_size, filename, lineno);

        // Reuse this object to track the new resized allocation
        TrackAllocation(ptr, size, filename, lineno);
      }
    } else {
      TrackDeallocation(filename, lineno);
    }
  } else {
    TrackAllocation(ptr, size, filename, lineno);
  }
  TriggerHeapMemoryUsageEvent();
}


// Incremental string hashing function.
// Uses Paul Hsieh's SuperFastHash, the same as in Google Chrome.
unsigned long TauAllocation::LocationHash(unsigned long hash, char const * data)
{
#define get16bits(d) ((((x_uint32)(((const x_uint8 *)(d))[1])) << 8)\
                       +(x_uint32)(((const x_uint8 *)(d))[0]) )

  x_uint32 tmp;
  int len;
  int rem;

  TAU_ASSERT((data != NULL), "Null string passed to TauAllocation::LocationHash");

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
      // Return a portable special case hash, which is actually the hash of the
      // null string with a zero seed, instead of the arch-dependent hash.
      return 0;
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
    data += 2 * sizeof(x_uint16);
    hash += hash >> 11;
  }

  switch (rem) {
  case 3:
    hash += get16bits(data);
    hash ^= hash << 16;
    hash ^= ((signed char)data[sizeof(x_uint16)]) << 18;
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

#undef get16bits
}


//////////////////////////////////////////////////////////////////////
// Records memory used at the current time
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerHeapMemoryUsageEvent() {
  TAU_REGISTER_EVENT(evt, "Heap Memory Used (KB)");
  TAU_EVENT(evt, Tau_max_RSS());
}

//////////////////////////////////////////////////////////////////////
// Records memory remaining at the current time
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerMemoryHeadroomEvent(void) {
  TAU_REGISTER_CONTEXT_EVENT(evt, "Memory Headroom Left (MB)");
  TAU_CONTEXT_EVENT(evt, Tau_estimate_free_memory());
}

//////////////////////////////////////////////////////////////////////
// Records memory overhead consumed by memory debugger
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerMemDbgOverheadEvent() {
  TAU_REGISTER_EVENT(evt, "Memory Debugger Overhead (KB)");
  TAU_EVENT(evt, BytesOverhead() >> 10);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerAllocationEvent(size_t size, char const * filename, int lineno)
{
  static event_map_t event_map;
  TauContextUserEvent * event;

  unsigned long file_hash = LocationHash(lineno, filename);

  RtsLayer::LockDB();
  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
        !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
    {
      event = new TauContextUserEvent("Heap Allocate");
    } else {
      char * name = new char[strlen(filename)+128];
      sprintf(name, "Heap Allocate <file=%s, line=%d>", filename, lineno);
      event = new TauContextUserEvent(name);
      delete[] name;
    }
    event_map[file_hash] = event;
  } else {
    event = it->second;
  }
  RtsLayer::UnLockDB();

  event->TriggerEvent(size);
  alloc_event = event->getContextUserEvent();
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerDeallocationEvent(size_t size, char const * filename, int lineno)
{
  static event_map_t event_map;

  unsigned long file_hash = LocationHash(lineno, filename);
  TauContextUserEvent * e;

  RtsLayer::LockDB();
  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
        !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
    {
      e = new TauContextUserEvent("Heap Free");
    } else {
      char * name = new char[strlen(filename)+128];
      sprintf(name, "Heap Free <file=%s, line=%d>", filename, lineno);
      e = new TauContextUserEvent(name);
      delete[] name;
    }
    event_map[file_hash] = e;
  } else {
    e = it->second;
  }
  RtsLayer::UnLockDB();

  e->TriggerEvent(size);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::TriggerErrorEvent(char const * descript, char const * filename, int lineno)
{
  static event_map_t event_map;

  unsigned long file_hash = LocationHash(lineno, filename);
  TauContextUserEvent * e;

  RtsLayer::LockDB();
  event_map_t::iterator it = event_map.find(file_hash);
  if (it == event_map.end()) {
    char * name;
    if ((lineno == TAU_MEMORY_UNKNOWN_LINE) &&
        !(strncmp(filename, TAU_MEMORY_UNKNOWN_FILE, TAU_MEMORY_UNKNOWN_FILE_STRLEN)))
    {
      name = new char[strlen(descript)+128];
      sprintf(name, "Memory Error! %s", descript);
    } else {
      name = new char[strlen(descript)+strlen(filename)+128];
      sprintf(name, "Memory Error! %s <file=%s, line=%d>", descript, filename, lineno);
    }
    e = new TauContextUserEvent(name);
    event_map[file_hash] = e;
    delete[] name;
  } else {
    e = it->second;
  }
  RtsLayer::UnLockDB();

  e->TriggerEvent(1);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
int TauAllocation::Protect(addr_t addr, size_t size)
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
      return ret;
    }

    addr += tail_size;
    size -= tail_size;
  }
  return ret;

#else

  int ret = 0;
  if ((ret = mprotect((void*)addr, size, PROT_NONE))) {
    char * errstr = strerror(errno);
    TAU_VERBOSE("TAU: ERROR - mprotect(%p, %ld, PROT_NONE) failed: %s\n", addr, size, errstr);
  }
  return ret;

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
int TauAllocation::Unprotect(addr_t addr, size_t size)
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
      return ret;
    }

    addr += tail_size;
    size -= tail_size;
  }
  return ret;

#else

  int ret = 0;
  if ((ret = mprotect((void*)addr, size, PROT_READ|PROT_WRITE))) {
    TAU_VERBOSE("TAU: ERROR - mprotect(%p, %ld, PROT_READ|PROT_WRITE) failed\n", addr, size);
  }
  return ret;

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_memory_initialize(void)
{
  TauInternalFunctionGuard protects_this_function;

  // Trigger the map's constructor
  static TauAllocation::allocation_map_t const & alloc = TauAllocation::AllocationMap();
  // use the map to prevent compiler warnings about unused variables
  alloc.size();

  atexit(Tau_memory_wrapper_disable);
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_memory_is_tau_allocation(void * ptr)
{
  Tau_global_incr_insideTAU();
  TauAllocation * alloc = TauAllocation::Find(ptr);
  Tau_global_decr_insideTAU();
  return alloc != NULL;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_memory_wrapper_register(wrapper_enable_handle_t enable_handle, wrapper_disable_handle_t disable_handle)
{
  wrapper_enable_handle = enable_handle;
  wrapper_disable_handle = disable_handle;
  wrapper_registered = true;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_memory_wrapper_enable(void)
{
  if (wrapper_enable_handle) {
    wrapper_enable_handle();
  }
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_memory_wrapper_disable(void)
{
  if (wrapper_disable_handle) {
    wrapper_disable_handle();
  }
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_memory_wrapper_is_registered(void)
{
  return wrapper_registered;
}


//////////////////////////////////////////////////////////////////////
// Get the system page size
//////////////////////////////////////////////////////////////////////
extern "C"
size_t Tau_page_size(void)
{
#ifdef PAGE_SIZE
  return (size_t)PAGE_SIZE;
#else
  static size_t page_size = 0;
  if (!page_size) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;
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
// Get memory size (max resident set size) in KB
//////////////////////////////////////////////////////////////////////
extern "C"
double Tau_max_RSS(void)
{
  if (TauAllocation::BytesAllocated()) {
    size_t bytes = TauAllocation::BytesAllocated() - TauAllocation::BytesDeallocated();
    return (double)bytes / 1024.0;
  } else {
#if defined(HAVE_MALLINFO)
    struct mallinfo minfo = mallinfo();
    double used = minfo.hblkhd + minfo.usmblks + minfo.uordblks;
    return used / 1024.0;
#elif (defined(TAU_BGP) || defined(TAU_BGQ))
    uint32_t heap_size;
    Kernel_GetMemorySize( KERNEL_MEMSIZE_HEAP, &heap_size );
    return (double)heap_size / 1024.0;
#elif defined(TAU_CATAMOUNT)
    size_t fragments;
    unsigned long total_free, largest_free, total_used;
    if (!heap_info(&fragments, &total_free, &largest_free, &total_used)) {
      return (double)total_used / 1024.0;
    } else {
      return 0.0;
    }
#elif (! (defined (TAU_WINDOWS) || defined (CRAYCC)) )
    struct rusage res;
    getrusage(RUSAGE_SELF, &res);
    return (double) res.ru_maxrss;
#else
    TAU_VERBOSE("TAU: WARNING - Couldn't determine RSS\n");
    return 0;
#endif
  }
}

//////////////////////////////////////////////////////////////////////
// The amount of memory available for use (in MB)
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_estimate_free_memory(void)
{
  /* Catamount has a heap_info call that returns the available memory headroom */
#if defined(TAU_CATAMOUNT)
  size_t fragments;
  unsigned long total_free, largest_free, total_used;
  if (heap_info(&fragments, &total_free, &largest_free, &total_used) == 0) {
    // return free memory in MB
    return (int)(total_free/(1024*1024));
  }
  return 0; /* if it didn't work */
#elif defined(TAU_BGP)
  uint32_t available_heap;
  Kernel_GetMemorySize( KERNEL_MEMSIZE_ESTHEAPAVAIL, &available_heap );
  return available_heap / (1024*1024);
#else
  #define TAU_BLOCK_COUNT 1024
  char * blocks[TAU_BLOCK_COUNT];
  char * ptr;
  int freemem = 0;
  int factor = 1;

  int i = 0; /* initialize it */
  while (1) {
    ptr = (char*)malloc(factor * 1024 * 1024); /* 1MB chunk */
    if (ptr && i < TAU_BLOCK_COUNT) {
      blocks[i++] = ptr;
      freemem += factor; /* assign the MB allocated */
      factor *= 2; /* try with twice as much the next time */
    } else {
      if (factor == 1) break;
      factor = 1; /* try with a smaller chunk size */
    }
  }

  for (int j = 0; j < i; j++)
    free(blocks[j]);

  return freemem;

#endif /* TAU_CATAMOUNT */
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_detect_memory_leaks(void)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  if (TauEnv_get_track_memory_leaks()) {
    TauAllocation::DetectLeaks();
  }
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_allocation(void * ptr, size_t size, char const * filename, int lineno)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  TauAllocation * alloc = TauAllocation::Find(ptr);
  if (!alloc) {
    alloc = new TauAllocation;
    alloc->TrackAllocation(ptr, size, filename, lineno);
  } else {
    //TAU_VERBOSE("TAU: WARNING - Allocation record for %p already exists\n", ptr);
  }
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_deallocation(void * ptr, char const * filename, int lineno)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  TauAllocation * alloc = TauAllocation::Find(ptr);
  if (alloc) {
    alloc->TrackDeallocation(filename, lineno);
  } else {
    TAU_VERBOSE("TAU: WARNING - No allocation record found for %p\n", ptr);
  }
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_reallocation(void * newPtr, void * ptr, size_t size,
    char const * filename, int lineno)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  TauAllocation * alloc = TauAllocation::Find(ptr);
  if (!alloc) {
    alloc = new TauAllocation;
  }
  alloc->TrackReallocation(newPtr, size, filename, lineno);
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_malloc(size_t size, const char * filename, int lineno)
{
#ifdef HAVE_MALLOC
  void * ptr;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * malloc(size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, 0, 0, filename, lineno);
  } else {
    ptr = malloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  TAU_PROFILE_STOP(t);
}
else{
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, 0, 0, filename, lineno);
  } else {
    ptr = malloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
}

  return ptr;
#else
  return NULL;
#endif
}


//////////////////////////////////////////////////////////////////////
// Tau_calloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_calloc(size_t count, size_t size, const char * filename, int lineno)
{
#ifdef HAVE_CALLOC
  void * ptr;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * calloc(size_t, size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(count*size, 0, 0, filename, lineno);
    if (ptr) memset(ptr, 0, size);
  } else {
    ptr = calloc(count, size);
    Tau_track_memory_allocation(ptr, count*size, filename, lineno);
  }
  TAU_PROFILE_STOP(t);
}
else{
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(count*size, 0, 0, filename, lineno);
    if (ptr) memset(ptr, 0, size);
  } else {
    ptr = calloc(count, size);
    Tau_track_memory_allocation(ptr, count*size, filename, lineno);
  }
}

  return ptr;
#else
  return NULL;
#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_realloc(void * ptr, size_t size, const char * filename, int lineno)
{
#ifdef HAVE_REALLOC
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * realloc(void*, size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = NULL;
    if (ptr) {
      if (size) {
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          ptr = alloc->Reallocate(size, 0, 0, filename, lineno);
        } else {
          // Trying to resize an allocation made outside TAU
          // Use system's realloc so we know the allocation size then copy
          // to a guarded allocation
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          void * tmpPtr = realloc(ptr, size);
          if (tmpPtr) {
            alloc = new TauAllocation;
            void * newPtr = alloc->Allocate(size, 0, 0, filename, lineno);
            memcpy(newPtr, tmpPtr, size);
            free(tmpPtr);
            ptr = newPtr;
          } else {
            // If the system realloc failed then we aren't going to succeed either
            ptr = NULL;
          }
        }
      } else {
        // Calling realloc with size == 0 is the same as calling free(ptr)
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          alloc->Deallocate(filename, lineno);
        } else {
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          free(ptr);
        }
        ptr = NULL;
      }
    } else {
      alloc = new TauAllocation;
      ptr = alloc->Allocate(size, 0, 0, filename, lineno);
    }
  } else {
    void * newPtr = realloc(ptr, size);
    if (newPtr) {
      Tau_track_memory_reallocation(newPtr, ptr, size, filename, lineno);
    }
    ptr = newPtr;
  }
  TAU_PROFILE_STOP(t);
}
else{
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = NULL;
    if (ptr) {
      if (size) {
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          ptr = alloc->Reallocate(size, 0, 0, filename, lineno);
        } else {
          // Trying to resize an allocation made outside TAU
          // Use system's realloc so we know the allocation size then copy
          // to a guarded allocation
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          void * tmpPtr = realloc(ptr, size);
          if (tmpPtr) {
            alloc = new TauAllocation;
            void * newPtr = alloc->Allocate(size, 0, 0, filename, lineno);
            memcpy(newPtr, tmpPtr, size);
            free(tmpPtr);
            ptr = newPtr;
          } else {
            // If the system realloc failed then we aren't going to succeed either
            ptr = NULL;
          }
        }
      } else {
        // Calling realloc with size == 0 is the same as calling free(ptr)
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          alloc->Deallocate(filename, lineno);
        } else {
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          free(ptr);
        }
        ptr = NULL;
      }
    } else {
      alloc = new TauAllocation;
      ptr = alloc->Allocate(size, 0, 0, filename, lineno);
    }
  } else {
    void * newPtr = realloc(ptr, size);
    if (newPtr) {
      Tau_track_memory_reallocation(newPtr, ptr, size, filename, lineno);
    }
    ptr = newPtr;
  }
}
  return ptr;
#else
  return NULL;
#endif
}


//////////////////////////////////////////////////////////////////////
// Tau_free calls Tau_free_before and free's the memory allocated
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_free(void * ptr, char const * filename, int lineno)
{
#ifdef HAVE_FREE
  if (ptr) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;
    TauAllocation * alloc = TauAllocation::Find(ptr);

if(TauEnv_get_show_memory_functions()) {
    char name[1024];
    BuildTimerName(name, "void free(void*) C", filename, lineno);
    TAU_PROFILE_TIMER(t, name, "", TAU_USER);
    TAU_PROFILE_START(t);
    if (alloc) {
      if (alloc->IsTracked()) {
        alloc->TrackDeallocation(filename, lineno);
        free(ptr);
      } else {
        alloc->Deallocate(filename, lineno);
      }
    } else {
      TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
      free(ptr);
    }
    TAU_PROFILE_STOP(t);
}
else{
    if (alloc) {
      if (alloc->IsTracked()) {
        alloc->TrackDeallocation(filename, lineno);
        free(ptr);
      } else {
        alloc->Deallocate(filename, lineno);
      }
    } else {
      TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
      free(ptr);
    }
}
  }
#endif
}



//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_memalign(size_t alignment, size_t size, const char * filename, int lineno)
{
#ifdef HAVE_MEMALIGN
  void * ptr;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * memalign(size_t, size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, alignment, 0, filename, lineno);
  } else {
    ptr = memalign(alignment, size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  TAU_PROFILE_STOP(t);
}
else {
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, alignment, 0, filename, lineno);
  } else {
    ptr = memalign(alignment, size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
}
  return ptr;
#else
  return NULL;
#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_posix_memalign(void **ptr, size_t alignment, size_t size,
    const char * filename, int lineno)
{
#ifdef HAVE_POSIX_MEMALIGN
  int retval;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "int posix_memalign(void**, size_t, size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    *ptr = alloc->Allocate(size, alignment, sizeof(void*), filename, lineno);
    retval = (ptr != NULL);
  } else {
    retval = posix_memalign(ptr, alignment, size);
    Tau_track_memory_allocation(*ptr, size, filename, lineno);
  }
  TAU_PROFILE_STOP(t);
}
else{
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    *ptr = alloc->Allocate(size, alignment, sizeof(void*), filename, lineno);
    retval = (ptr != NULL);
  } else {
    retval = posix_memalign(ptr, alignment, size);
    Tau_track_memory_allocation(*ptr, size, filename, lineno);
  }
}
  return retval;
#else
  *ptr = NULL;
  return ENOMEM;
#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_valloc(size_t size, const char * filename, int lineno)
{
#ifdef HAVE_VALLOC
  void * ptr;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * valloc(size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, Tau_page_size(), 0, filename, lineno);
  } else {
    ptr = valloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  TAU_PROFILE_STOP(t);
}
else {
  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, Tau_page_size(), 0, filename, lineno);
  } else {
    ptr = valloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
}
  return ptr;
#else
  return NULL;
#endif
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_pvalloc(size_t size, const char * filename, int lineno)
{
#ifdef HAVE_PVALLOC
#ifndef PAGE_SIZE
  size_t const PAGE_SIZE = Tau_page_size();
#endif

  void * ptr;

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * pvalloc(size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);

  // pvalloc allocates the smallest set of complete pages
  // that can hold the requested number of bytes
  size = (size + PAGE_SIZE-1) & ~(PAGE_SIZE-1);

  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, PAGE_SIZE, 0, filename, lineno);
  } else {
    ptr = pvalloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
  TAU_PROFILE_STOP(t);
}
else {

  // pvalloc allocates the smallest set of complete pages
  // that can hold the requested number of bytes
  size = (size + PAGE_SIZE-1) & ~(PAGE_SIZE-1);

  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = new TauAllocation;
    ptr = alloc->Allocate(size, PAGE_SIZE, 0, filename, lineno);
  } else {
    ptr = pvalloc(size);
    Tau_track_memory_allocation(ptr, size, filename, lineno);
  }
}
  return ptr;
#else
  return NULL;
#endif
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_reallocf(void * ptr, size_t size, const char * filename, int lineno)
{
#ifdef HAVE_REALLOCF
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

if(TauEnv_get_show_memory_functions()) {
  char name[1024];
  BuildTimerName(name, "void * reallocf(void*, size_t) C", filename, lineno);
  TAU_PROFILE_TIMER(t, name, "", TAU_USER);
  TAU_PROFILE_START(t);

  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = NULL;
    if (ptr) {
      if (size) {
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          ptr = alloc->Reallocate(size, 0, 0, filename, lineno);
          if (!ptr) {
            alloc->Deallocate(filename, lineno);
          }
        } else {
          // Trying to resize an allocation made outside TAU
          // Use system's realloc so we know the allocation size then copy
          // to a guarded allocation
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          void * tmpPtr = reallocf(ptr, size);
          if (tmpPtr) {
            alloc = new TauAllocation;
            void * newPtr = alloc->Allocate(size, 0, 0, filename, lineno);
            memcpy(newPtr, tmpPtr, size);
            free(tmpPtr);
            ptr = newPtr;
          } else {
            // If the system reallocf failed then we aren't going to succeed either
            ptr = NULL;
          }
        }
      } else {
        // Calling reallocf with size == 0 is the same as calling free(ptr)
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          alloc->Deallocate(filename, lineno);
        } else {
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          free(ptr);
        }
        ptr = NULL;
      }
    } else {
      alloc = new TauAllocation;
      ptr = alloc->Allocate(size, 0, 0, filename, lineno);
    }
  } else {
    void * newPtr = reallocf(ptr, size);
    if (newPtr) {
      Tau_track_memory_reallocation(newPtr, ptr, size, filename, lineno);
    }
    ptr = newPtr;
  }
  TAU_PROFILE_STOP(t);
}
else {

  if (AllocationShouldBeProtected(size)) {
    TauAllocation * alloc = NULL;
    if (ptr) {
      if (size) {
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          ptr = alloc->Reallocate(size, 0, 0, filename, lineno);
          if (!ptr) {
            alloc->Deallocate(filename, lineno);
          }
        } else {
          // Trying to resize an allocation made outside TAU
          // Use system's realloc so we know the allocation size then copy
          // to a guarded allocation
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          void * tmpPtr = reallocf(ptr, size);
          if (tmpPtr) {
            alloc = new TauAllocation;
            void * newPtr = alloc->Allocate(size, 0, 0, filename, lineno);
            memcpy(newPtr, tmpPtr, size);
            free(tmpPtr);
            ptr = newPtr;
          } else {
            // If the system reallocf failed then we aren't going to succeed either
            ptr = NULL;
          }
        }
      } else {
        // Calling reallocf with size == 0 is the same as calling free(ptr)
        alloc = TauAllocation::Find(ptr);
        if (alloc) {
          alloc->Deallocate(filename, lineno);
        } else {
          TAU_VERBOSE("TAU: WARNING - Allocation record for %p not found.\n", ptr);
          free(ptr);
        }
        ptr = NULL;
      }
    } else {
      alloc = new TauAllocation;
      ptr = alloc->Allocate(size, 0, 0, filename, lineno);
    }
  } else {
    void * newPtr = reallocf(ptr, size);
    if (newPtr) {
      Tau_track_memory_reallocation(newPtr, ptr, size, filename, lineno);
    }
    ptr = newPtr;
  }
}
  return ptr;
#else
  return NULL;
#endif
}

//////////////////////////////////////////////////////////////////////
// Tau_open_status returns the file descriptor of /proc/self/status
//////////////////////////////////////////////////////////////////////
extern "C" int Tau_open_status(void) {

#ifndef TAU_WINDOWS 
  int fd = open ("/proc/self/status", O_RDONLY);
#else
  int fd = -1;
#endif /* TAU_WINDOWS */

  if (fd == -1) {
    perror("Couldn't open /proc/self/status for tracking memory");
  }

  return fd;
}

//////////////////////////////////////////////////////////////////////
// Tau_read_status returns the VmRSS and VmHWM (high water mark of 
// RSS or resident set size) to give an accurate idea of the memory
// footprint of the program. It gets this from parsing /proc/self/status
//////////////////////////////////////////////////////////////////////
extern "C" int Tau_read_status(int fd, long long * rss, long long * hwm) {
  char buf[2048];
  int ret, i, j, bytesread;
  memset(buf, 0, 2048);

  ret = lseek(fd, 0, SEEK_SET);
  if (ret == -1) {
    perror("lseek failure on /proc/self/status");
    return ret;
  }

  bytesread = read(fd, buf, 2048);
  if (bytesread == -1) {
    perror("Error reading from /proc/self/status");
    return bytesread;
  }
  *hwm = 0LL;
  *rss = 0LL;
  for(i=0; i < bytesread; i++) {
   /* Search for VmHWM for high water mark of memory from /proc/self/status */
    if (buf[i] == '\n' && buf[i+1] == 'V' && buf[i+2] == 'm' && buf[i+3] == 'H' && buf[i+4] == 'W' && buf[i+5] == 'M' && buf[i+6] == ':') {
        for (j = 7 ; j+i < bytesread ; j++) {
            if (buf[i+j] != ' ') {
                sscanf(&buf[i+j], "%lld", hwm);
                //printf("VmHWM: %lld\n", *hwm);
                break;
            }
        }
    }
   /* Search for VmRSS for resident set size of memory from /proc/self/status */
    if (buf[i] == '\n' && buf[i+1] == 'V' && buf[i+2] == 'm' && buf[i+3] == 'R' && buf[i+4] == 'S' && buf[i+5] == 'S' && buf[i+6] == ':') {
        for (j = 7 ; j+i < bytesread ; j++) {
            if (buf[i+j] != ' ') {
                sscanf(&buf[i+j], "%lld", rss);
                //printf("VmRSS: %lld\n", *rss);
                break;
            }
        }
      break;
    }
  }
  return ret;
}

//////////////////////////////////////////////////////////////////////
// Tau_close_status closes the file descriptor associated with /proc/self/status
//////////////////////////////////////////////////////////////////////
extern "C" int Tau_close_status(int fd) {
  int ret=close(fd);

  if (ret == -1) {
    perror("close failed on /proc/self/status");
  }
  return ret;
}

//////////////////////////////////////////////////////////////////////
// Tau_trigger_memory_rss_hwm triggers resident memory size and high water 
// mark events
//////////////////////////////////////////////////////////////////////
extern "C" int Tau_trigger_memory_rss_hwm(void) {
  static int fd=Tau_open_status();

  long long vmrss, vmhwm; 
  TAU_REGISTER_EVENT(proc_rss, "Memory Footprint (VmRSS) (KB)");
  TAU_REGISTER_CONTEXT_EVENT(proc_vmhwm, "Peak Memory Usage Resident Set Size (VmHWM) (KB)");

  Tau_read_status(fd, &vmrss, &vmhwm);

  if (vmrss > 0)
    TAU_EVENT(proc_rss, (double) vmrss);
  if (vmhwm > 0)
    TAU_CONTEXT_EVENT(proc_vmhwm, (double) vmhwm);

#ifdef TAU_BEACON
  TauBeaconPublish((double) vmrss, "KB", "MEMORY", "Memory Footprint (VmRSS - Resident Set Size)");
  TauBeaconPublish((double) vmhwm, "KB", "MEMORY", "Peak Memory Usage (VmHWM - High Water Mark)");
#endif /* TAU_BEACON */
  // TAU_VERBOSE("Tau_trigger_memory_rss_hwm: rss = %lld, hwm = %lld in KB\n", vmrss, vmhwm);

  return 1; // SUCCESS
}



#if 0

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"

#undef memchr
#undef memcmp

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_memcpy(void *dst, const void *src, size_t size, const char * filename, int lineno)
{
  return memcpy(dst, src, size);
}

#undef memmove
#undef memset

int strcasecmp(const char *s1, const char *s2);

int strncasecmp(const char *s1, const char *s2, size_t n);

char *index(const char *s, int c);

char *rindex(const char *s, int c);

char *stpcpy(char *dest, const char *src);

//char *strcat(char *dest, const char *src);
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

char *strchr(const char *s, int c);


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////

int __tau_strcmp(char const * s1, char const * s2)
{
  while (*s1 || *s2) {
    if (*s1 != *s2) return *s1 - *s2;
    ++s1, ++s2;
  }
  return 0;
}
int Tau_strcmp(char const * s1, char const * s2, const char * filename, int lineno)
{
  // Maybe do something with filename/lineno...
  Tau_global_incr_insideTAU();
  int retval = __tau_strcmp(s1, s2);
  Tau_global_decr_insideTAU();
  return retval;
}


int strcoll(const char *s1, const char *s2);

//char *strcpy(char *dest, const char *src);
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



size_t strcspn(const char *s, const char *reject);

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

char *strfry(char *string);

size_t strlen(const char *s);

char *strncat(char *dest, const char *src, size_t n);

int strncmp(const char *s1, const char *s2, size_t n);

char *strncpy(char *dest, const char *src, size_t n);

char *strpbrk(const char *s, const char *accept);

char *strrchr(const char *s, int c);

char *strsep(char **stringp, const char *delim);

size_t strspn(const char *s, const char *accept);

char *strstr(const char *haystack, const char *needle);

char *strtok(char *s, const char *delim);

size_t strxfrm(char *dest, const char *src, size_t n);

#endif

/***************************************************************************
 * $RCSfile: TauMemory.cpp,v $   $Author: amorris $
 * $Revision: 1.33 $   $Date: 2010/01/27 00:47:51 $
 * TAU_VERSION_ID: $Id: TauMemory.cpp,v 1.33 2010/01/27 00:47:51 amorris Exp $ 
 ***************************************************************************/
