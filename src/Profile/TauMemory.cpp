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

#define MAX_STRING_LEN 1024
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )

typedef unsigned char byte_t;
typedef unsigned long iaddr_t;
typedef unsigned long hash_t;
typedef TauContextUserEvent user_event_t;
typedef TAU_HASH_MAP<hash_t, user_event_t*> malloc_map_t;
typedef std::pair<size_t, user_event_t*> pointer_size_t;
typedef TAU_MULTIMAP<iaddr_t, pointer_size_t> pointer_size_map_t;
typedef TAU_HASH_MAP<iaddr_t, TauUserEvent*> leak_map_t;


class TauAllocation
{
public:

  TauAllocation() :
    _addr(NULL), _size(0),
    _user_addr(NULL), _user_size(0),
    _prot_addr(NULL), _prot_size(0),
    _gap_addr(NULL), _gap_size(0)
  { }

  void * Allocate(size_t size, size_t alignment, bool prot_above, bool prot_below);

  void Deallocate();

private:
  void _Protect(void * addr, size_t size);
  void _Unprotect(void * addr, size_t size);

  byte_t * _addr;
  size_t _size;
  byte_t * _user_addr;
  size_t _user_size;
  byte_t * _prot_addr;
  size_t _prot_size;
  byte_t * _gap_addr;
  size_t _gap_size;

  static size_t _page_size;
};

size_t TauAllocation::_page_size = Tau_page_size();


//////////////////////////////////////////////////////////////////////
static malloc_map_t & TheTauMallocMap(void)
{
  static malloc_map_t mallocmap;
  return mallocmap;
}

//////////////////////////////////////////////////////////////////////
// We store the leak detected events here 
//////////////////////////////////////////////////////////////////////
static leak_map_t & TheTauMemoryLeakMap(void)
{
  static leak_map_t leakmap;
  return leakmap;
}

//////////////////////////////////////////////////////////////////////
// This map stores the memory allocated and its associations
//////////////////////////////////////////////////////////////////////
static pointer_size_map_t & TheTauPointerSizeMap(void)
{
  static pointer_size_map_t sizemap;
  return sizemap;
}


// Incremental string hashing function.
// Uses Paul Hsieh's SuperFastHash, the same as in Google Chrome.
static hash_t Tau_hash(hash_t hash, char const * data)
{
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
size_t Tau_page_size(void)
{
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
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void * TauAllocation::Allocate(size_t size, size_t alignment, bool prot_above, bool prot_below)
{
  // Check size
  if (!size) {
    // TODO: Zero-size malloc
  }

  // Alignment=0 ==> default alignment
  // Alignment=1 ==> byte alignment (potentially sub-word)
  if (!alignment) {
    alignment = TauEnv_get_memdbg_alignment();
  }

  // Sub-alignment allocation size
  if (size < alignment) {
    // Align to the next lower power of two
    alignment = size;
    while (alignment & (alignment-1)) {
      alignment &= alignment-1;
    }
  }

  // Alignment must be a power of two
  if ((int)alignment != ((int)alignment & -(int)alignment)) {
    TAU_VERBOSE("TAU: ERROR - Alignment %ld is not a power of two", alignment);
    _addr = NULL;
    _size = 0;
    return NULL;
  }

  // Round up to the next page boundary
  _size = ((size + _page_size-1) & ~(_page_size-1));
  // Include space for protection pages
  if (prot_above)
    _size += _page_size;
  if (prot_below)
    _size += _page_size;
  // Round to next alignment boundary
  if (alignment > _page_size)
    _size += alignment - _page_size;

#if defined(TAU_WINDOWS)

  _addr = VirtualAlloc(NULL, (DWORD)_size, (DWORD)MEM_COMMIT, (DWORD)PAGE_READWRITE);
  if (!_addr) {
    TAU_VERBOSE("TAU: ERROR - VirtualAlloc(%ld) failed\n", _size);
    _addr = NULL;
    _size = 0;
    return NULL;
  }

#else

  static void * _suggest_start;

#if defined(MAP_ANONYMOUS)
  _addr = (byte_t*)mmap(_suggest_start, _size,
      PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
#else
  static int fd = -1;
  if (fd == -1) {
    if ((fd = open("/dev/zero", O_RDWR)) < 0) {
      TAU_VERBOSE("TAU: ERROR - open() on /dev/zero failed\n");
      _addr = NULL;
      _size = 0;
      return NULL;
    }
  }
  _addr = (byte_t*)mmap(_suggest_start, _size,
      PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
#endif

  if (_addr == MAP_FAILED) {
    TAU_VERBOSE("TAU: ERROR - mmap(%ld) failed\n", _size);
    _addr = NULL;
    _size = 0;
    return NULL;
  }

  // Suggest the next allocation begin after this one
  _suggest_start = (void*)(_addr + _size);

#endif

  if (prot_below) {
    if (prot_above) {
      // TODO
    } else {
      // TODO
    }
  } else if (prot_above) {
    // The address to return to the user
    _user_size = size;
    _user_addr = (byte_t*)((size_t)(_addr + _size - _page_size - _user_size) & ~(alignment-1));
    // The protected page address
    _prot_addr = (byte_t*)((size_t)(_user_addr + _user_size + _page_size-1) & ~(_page_size-1));
    _prot_size = (size_t)(_addr + _size - _prot_addr);
    // The gap address
    _gap_addr = _user_addr + _user_size;
    _gap_size = _prot_addr - _gap_addr;

    // Permit access to the user address range
    _Unprotect(_addr, (size_t)(_prot_addr - _addr));
    // Deny access to the guard page
    _Protect(_prot_addr, _prot_size);
  }

  // All done with bookkeeping, get back to the user
  return _user_addr;
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::Deallocate()
{
  if(!_addr) return;

#if defined(TAU_WINDOWS)

  void * tmp_addr = _addr;
  size_t tmp_size = _size;
  SIZE_T retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  BOOL ret;

  /* release physical memory commited to virtual address space */
  while (tmp_size > 0) {
    retQuery = VirtualQuery(tmp_addr, &MemInfo, sizeof(MemInfo));

    if (retQuery < sizeof(MemInfo)) {
      TAU_VERBOSE("TAU: ERROR - VirtualQuery() failed\n");
    }

    if (MemInfo.State == MEM_COMMIT) {
      ret = VirtualFree((LPVOID)MemInfo.BaseAddress, (DWORD)MemInfo.RegionSize, (DWORD) MEM_DECOMMIT);
      if (!ret) {
        TAU_VERBOSE("TAU: ERROR - VirtualFree(0x%lx,%ld,MEM_DECOMMIT) failed\n", (iaddr_t)tmp_addr, tmp_size);
      }
    }

    tmp_addr = ((char*)tmp_addr) + MemInfo.RegionSize;
    tmp_size -= MemInfo.RegionSize;
  }

  /* release virtual address space */
  ret = VirtualFree((LPVOID)_addr, (DWORD)0, (DWORD)MEM_RELEASE);
  if (!ret) {
    TAU_VERBOSE("TAU: ERROR - VirtualFree(0x%lx,%ld,MEM_RELEASE) failed\n", (iaddr_t)_addr, _size);
  }

#else

  if (munmap(_addr, _size) < 0) {
    TAU_VERBOSE("TAU: ERROR - munmap(0x%lx, %ld) failed\n", (iaddr_t)_addr, _size);
  }

  _addr = NULL;
  _size = 0;

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::_Protect(void * addr, size_t size)
{
#if defined(TAU_WINDOWS)

  SIZE_T OldProtect, retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  size_t tail_size;
  BOOL ret;

  while(size > 0) {
    retQuery = VirtualQuery(addr, &MemInfo, sizeof(MemInfo));
    if (retQuery < sizeof(MemInfo)) {
      TAU_VERBOSE("TAU: ERROR - VirtualQuery() failed\n");
    }
    tail_size = (size > MemInfo.RegionSize) ? MemInfo.RegionSize : size;
    ret = VirtualProtect((LPVOID)addr, (DWORD)tail_size, (DWORD)PAGE_READWRITE, (PDWORD) &OldProtect);
    if (!ret) {
      TAU_VERBOSE("TAU: ERROR - VirtualProtecct(0x%lx, %ld) failed\n", (iaddr_t)addr, tail_size);
    }

    addr = ((char*)addr) + tail_size;
    size -= tail_size;
  }

#else

  if (mprotect(addr, size, PROT_NONE) < 0) {
    TAU_VERBOSE("TAU: ERROR - mprotect(0x%lx, %ld, PROT_NONE) failed\n", (iaddr_t)addr, size);
  }

#endif
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
void TauAllocation::_Unprotect(void * addr, size_t size)
{
#if defined(TAU_WINDOWS)

  SIZE_T OldProtect, retQuery;
  MEMORY_BASIC_INFORMATION MemInfo;
  size_t tail_size;
  BOOL ret;

  while (size > 0) {
    retQuery = VirtualQuery(addr, &MemInfo, sizeof(MemInfo));
    if (retQuery < sizeof(MemInfo)) {
      TAU_VERBOSE("TAU: ERROR - VirtualQuery() failed\n");
    }
    tail_size = (size > MemInfo.RegionSize) ? MemInfo.RegionSize : size;
    ret = VirtualProtect((LPVOID)addr, (DWORD)tail_size, (DWORD)PAGE_NOACCESS, (PDWORD)&OldProtect);
    if (!ret) {
      TAU_VERBOSE("TAU: ERROR - VirtualProtecct(0x%lx, %ld) failed\n", (iaddr_t)addr, tail_size);
    }

    addr = ((char*)addr) + tail_size;
    size -= tail_size;
  }

#else

  if (mprotect(addr, size, PROT_READ|PROT_WRITE) < 0) {
    TAU_VERBOSE("TAU: ERROR - mprotect(0x%lx, %ld, PROT_READ|PROT_WRITE) failed\n", (iaddr_t)addr, size);
  }

#endif
}



//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
static void * Tau_allocate(size_t size, const char * filename, int lineno)
{
  TauAllocation alloc;

  size_t alignment = TauEnv_get_memdbg_alignment();
  bool prot_above = TauEnv_get_memdbg_protect_above();
  bool prot_below = TauEnv_get_memdbg_protect_below();

  void * ptr = alloc.Allocate(size, alignment, prot_above, prot_below);

  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
static void Tau_deallocate(void * baseAddr, const char * filename, int lineno)
{
  // TODO
}

//////////////////////////////////////////////////////////////////////
// Creates/accesses the event associated with tracking memory allocation
//////////////////////////////////////////////////////////////////////
static user_event_t * Tau_before_system_allocate(char const * filename, int lineno)
{
  hash_t file_hash = Tau_hash(lineno, filename);

  malloc_map_t & mallocmap = TheTauMallocMap();
  malloc_map_t::iterator it = mallocmap.find(file_hash);
  user_event_t * e;

  if (it == mallocmap.end()) {
    char * s = (char*)malloc(strlen(filename)+128);
    sprintf(s, "malloc size <file=%s, line=%d>", filename, lineno);
#ifdef DEBUGPROF
    printf("C++: Tau_malloc: creating new user event %s\n", s);
#endif /* DEBUGPROF */
    e = new user_event_t(s);
    mallocmap[file_hash] = e;
    free((void*)s);
  } else { /* found it */
#ifdef DEBUGPROF
    printf("Found it! Name = %s\n", it->second->GetEventName());
#endif /* DEBUGPROF */
    e = it->second;
  }
#ifdef DEBUGPROF
  printf("C++: Tau_malloc: %s:%d:%d\n", filename, lineno, size);
#endif /* DEBUGPROF */

  return e; /* the event that is created in this routine */
}

//////////////////////////////////////////////////////////////////////
// Triggers the event associated with the address allocated
//////////////////////////////////////////////////////////////////////
static void Tau_after_system_allocate(void * ptr, size_t size, user_event_t * e)
{
  // We can't trigger the event until after the allocation because we don't know
  // the actual size of the allocation until it is made
  e->TriggerEvent(size);

  iaddr_t addr = Tau_convert_ptr_to_unsigned_long(ptr);
  TheTauPointerSizeMap().insert(pair<iaddr_t, pointer_size_t>(addr, pointer_size_t(size, e)));
}


//////////////////////////////////////////////////////////////////////
// Does everything prior to free'ing the memory
//////////////////////////////////////////////////////////////////////
static void Tau_before_system_deallocate(char const * file, int line, void * ptr)
{
#ifdef DEBUGPROF
  printf("C++: Tau_free_before: file = %s, ptr=%lx,  long file = %uld\n", file, file, file_hash);
#endif /* DEBUGPROF */

  pointer_size_map_t & sizemap = TheTauPointerSizeMap();
  malloc_map_t & mallocmap = TheTauMallocMap();

  hash_t file_hash = Tau_hash(line, file);
  iaddr_t addr = Tau_convert_ptr_to_unsigned_long(ptr);

  size_t sz;
  pointer_size_map_t::iterator size_it = sizemap.find(addr);
  if (size_it != sizemap.end()) {
    // Size was found, but sometimes a single address corresponds to multiple
    // allocations, i.e. Intel compilers can do this when there's a leak.
#ifdef DEBUGPROF
    if (sizemap.count(addr) > 1) {
      printf("Found more than one occurrence of addr in Tau_free_before\n");
    }
#endif /* DEBUG */
    sz = size_it->second.first;
    // Remove the record
    sizemap.erase(size_it);
  } else {
    sz = 0;
  }

  malloc_map_t::iterator it = mallocmap.find(file_hash);
  if (it == mallocmap.end()) {
    char * s = (char*)malloc(strlen(file)+64);
    sprintf(s, "free size <file=%s, line=%d>", file, line);

#ifdef DEBUGPROF
    printf("C++: Tau_free: creating new user event %s\n", s);
#endif /* DEBUGPROF */

    user_event_t * e = new user_event_t(s);
    e->TriggerEvent(sz);
    mallocmap[file_hash] = e;
    free((void*)s);
  } else {
#ifdef DEBUGPROF
    printf("Found it! Name = %s\n", it->second->GetEventName());
#endif /* DEBUGPROF */
    it->second->TriggerEvent(sz);
  }
#ifdef DEBUGPROF
  printf("C++: Tau_free: %s:%d\n", file, line);  
#endif /* DEBUGPROF */
}


//////////////////////////////////////////////////////////////////////
// TauDetectMemoryLeaks iterates over the list of pointers and checks
// which blocks have not been freed. This is called at the very end of
// the program from Profiler::StoreData
//////////////////////////////////////////////////////////////////////
extern "C"
void TauDetectMemoryLeaks(void)
{
  pointer_size_map_t & sizemap = TheTauPointerSizeMap();
  leak_map_t & leakmap = TheTauMemoryLeakMap();
  if (sizemap.empty()) return;

  for (pointer_size_map_t::iterator size_it=sizemap.begin();
       size_it != sizemap.end(); size_it++)
  {
    size_t sz = size_it->second.first;
    user_event_t * e = size_it->second.second;

#ifdef DEBUGPROF
    printf("Found leak for block of memory of size %d from memory allocated at:%s\n",
        sz, e->GetEventName());
#endif /* DEBUGPROF */

    iaddr_t leak_key = Tau_convert_ptr_to_unsigned_long(e);
    leak_map_t::iterator leak_it = leakmap.find(leak_key);
    if (leak_it == leakmap.end()) {
      char * s = (char*)malloc(strlen(e->GetEventName())+32);
      sprintf(s, "MEMORY LEAK! %s", e->GetEventName());
      TauUserEvent * leakevent = new TauUserEvent(s);
      leakmap[leak_key] = leakevent;
      leakevent->TriggerEvent(sz);
      free((void*)s);
    } else {
      leak_it->second->TriggerEvent(sz);
    }
  }
}



//////////////////////////////////////////////////////////////////////
// Tau_track_memory_allocation does everything that Tau_malloc does except
// allocate memory
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_allocation(char const * file, int line, size_t size, void * ptr)
{
  Tau_after_system_allocate(ptr, size, Tau_before_system_allocate(file, line));
}

//////////////////////////////////////////////////////////////////////
// Tau_track_memory_deallocation does everything that Tau_free does except
// de-allocate memory
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_track_memory_deallocation(const char *file, int line, void * ptr)
{
  Tau_before_system_deallocate(file, line, ptr);
}

//////////////////////////////////////////////////////////////////////
// Tau_new returns the expression (new[] foo) and  does everything that
// Tau_track_memory_allocation does
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_new(char const * file, int line, size_t size, void * ptr)
{
  /* the memory is already allocated by the time we see this ptr */
  Tau_track_memory_allocation(file, line, size, ptr);
  return ptr;
}


//////////////////////////////////////////////////////////////////////
// Tau_malloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_malloc(size_t size, const char * filename, int lineno)
{
  void * ptr;

  if (TauEnv_get_memdbg()) {
    ptr = Tau_allocate(size, filename, lineno);
  } else {
    user_event_t * e = Tau_before_system_allocate(filename, lineno);
    ptr = malloc(size);
    Tau_after_system_allocate(ptr, size, e);
  }

  return ptr;
}


//////////////////////////////////////////////////////////////////////
// Tau_calloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_calloc(size_t elemCount, size_t elemSize, const char * filename, int lineno)
{
  /* Get the event that is created */
  user_event_t * e = Tau_before_system_allocate(filename, lineno);

  void * ptr = calloc(elemCount, elemSize);

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_after_system_allocate(ptr, elemCount * elemSize, e);

  return ptr;  /* what was allocated */
}


//////////////////////////////////////////////////////////////////////
// Tau_free calls Tau_free_before and free's the memory allocated
//////////////////////////////////////////////////////////////////////
extern "C"
void Tau_free(void * baseAddr, char const * filename, int lineno)
{
  if (TauEnv_get_memdbg()) {
    Tau_deallocate(baseAddr, filename, lineno);
  } else {
    Tau_before_system_deallocate(filename, lineno, baseAddr);
    free(baseAddr);
  }
}


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
#if HAVE_MEMALIGN
extern "C"
void * Tau_memalign(size_t alignment, size_t size, const char * filename, int lineno)
{
  /* Get the event that is created */
  user_event_t * e = Tau_before_system_allocate(filename, lineno);

  void * ptr = memalign(alignment, size);

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_after_system_allocate(ptr, size, e);

  return ptr;
}
#endif


//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
int Tau_posix_memalign(void **memptr, size_t alignment, size_t size,
    const char * filename, int lineno)
{
  /* Get the event that is created */
  user_event_t * e = Tau_before_system_allocate(filename, lineno);

  int retval = posix_memalign(memptr, alignment, size);

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_after_system_allocate(*memptr, size, e);

  return retval;
}


//////////////////////////////////////////////////////////////////////
// Tau_realloc calls free_before, realloc and memory allocation tracking routine
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_realloc(void * baseAdr, size_t newSize, const char * filename, int lineno)
{
  Tau_before_system_deallocate(filename, lineno, baseAdr);
  void *retval = realloc(baseAdr, newSize);
  Tau_track_memory_allocation(filename, lineno, newSize, retval);
  return retval;
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
void * Tau_valloc(size_t size, const char * filename, int lineno)
{
  /* Get the event that is created */
  user_event_t * e = Tau_before_system_allocate(filename, lineno);

  void * ptr = valloc(size);

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_after_system_allocate(ptr, size, e);

  return ptr;  /* what was allocated */
}

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
#if HAVE_PVALLOC
extern "C"
void * Tau_pvalloc(size_t size, const char * filename, int lineno)
{
  /* Get the event that is created */
  user_event_t * e = Tau_before_system_allocate(filename, lineno);

  void * ptr = pvalloc(size);

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_after_system_allocate(ptr, size, e);

  return ptr;  /* what was allocated */
}
#endif

//////////////////////////////////////////////////////////////////////
// TODO: Docs
//////////////////////////////////////////////////////////////////////
extern "C"
char * Tau_strdup(const char *str, const char * filename, int lineno)
{
  size_t size = strlen(str);

  /* Get the event that is created */
  user_event_t * e = Tau_before_system_allocate(filename, lineno);

  char * ptr = strdup(str);

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_after_system_allocate(ptr, size, e);

  return ptr;  /* what was allocated */
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
