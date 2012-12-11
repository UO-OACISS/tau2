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
#include <Profile/Profiler.h>

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

#if (defined(__QK_USER__) || defined(__LIBCATAMOUNT__ ))
#define TAU_CATAMOUNT
#endif /* __QK_USER__ || __LIBCATAMOUNT__ */
#ifdef TAU_CATAMOUNT
#include <catamount/catmalloc.h>
#endif /* TAU_CATAMOUNT */

#include <stdlib.h>

#ifdef TAU_BGP
#include <kernel_interface.h>
#endif

#define MAX_STRING_LEN 1024
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )


typedef unsigned long hash_t;
typedef TauContextUserEvent user_event_t;
typedef std::map<hash_t, user_event_t*> malloc_map_t;
typedef std::pair<size_t, user_event_t*> pointer_size_t;
typedef std::multimap<long, pointer_size_t> pointer_size_map_t;
typedef std::map<long, TauUserEvent*> leak_map_t;


//////////////////////////////////////////////////////////////////////
malloc_map_t & TheTauMallocMap(void)
{
  static malloc_map_t mallocmap;
  return mallocmap;
}

//////////////////////////////////////////////////////////////////////
// We store the leak detected events here 
//////////////////////////////////////////////////////////////////////
leak_map_t & TheTauMemoryLeakMap(void)
{
  static leak_map_t leakmap;
  return leakmap;
}

//////////////////////////////////////////////////////////////////////
// This map stores the memory allocated and its associations
//////////////////////////////////////////////////////////////////////
pointer_size_map_t & TheTauPointerSizeMap(void)
{
  static pointer_size_map_t sizemap;
  return sizemap;
}


// Incremental string hashing function.
// Uses Paul Hsieh's SuperFastHash, the same as in Google Chrome.
hash_t Tau_hash(hash_t hash, char const * data)
{
  uint32_t tmp;
  int len = strnlen(data, MAX_STRING_LEN);
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
// Tau_malloc_before creates/access the event associated with tracking
// memory allocation for the specified line and file. 
//////////////////////////////////////////////////////////////////////
user_event_t * Tau_malloc_before(const char *file, int line, size_t size)
{
#ifdef DEBUGPROF
  printf("C++: Tau_malloc_before: file = %s, ptr=%lx,  long file = %uld\n", file, file, file_hash);
#endif /* DEBUGPROF */

  hash_t file_hash = Tau_hash(line, file);

  malloc_map_t & mallocmap = TheTauMallocMap();
  malloc_map_t::iterator it = mallocmap.find(file_hash);
  user_event_t * e;

  if (it == mallocmap.end()) {
    char * s = (char*)malloc(strnlen(file, MAX_STRING_LEN)+128);
    sprintf(s, "malloc size <file=%s, line=%d>", file, line);
#ifdef DEBUGPROF
    printf("C++: Tau_malloc: creating new user event %s\n", s);
#endif /* DEBUGPROF */
    e = new user_event_t(s);
    e->TriggerEvent(size);
    mallocmap[file_hash] = e;
    free((void*)s);
  } else { /* found it */
#ifdef DEBUGPROF
    printf("Found it! Name = %s\n", it->second->GetEventName());
#endif /* DEBUGPROF */
    e = it->second;
    e->TriggerEvent(size);
  }
#ifdef DEBUGPROF
  printf("C++: Tau_malloc: %s:%d:%d\n", file, line, size);
#endif /* DEBUGPROF */

  return e; /* the event that is created in this routine */
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc_after associates the event and size with the address allocated
//////////////////////////////////////////////////////////////////////
void Tau_malloc_after(void * ptr, size_t size, user_event_t * e)
{
  TheTauPointerSizeMap().insert(pair<long, pointer_size_t>(Tau_convert_ptr_to_long(ptr), pointer_size_t(size, e)));
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
void * Tau_malloc(const char *file, int line, size_t size)
{
  /* Get the event that is created */
   user_event_t * e = Tau_malloc_before(file, line, size);

  void * ptr = malloc(size);

#ifdef DEBUGPROF
  printf("TAU_MALLOC<%d>: %s:%d ptr = %p size = %d\n", RtsLayer::myNode(), file, line, ptr, size);
#endif /* DEBUGPROF */

  /* associate the event generated and its size with the address of memory
   * allocated by malloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_malloc_after(ptr, size, e);

  return ptr;  /* what was allocated */
}

//////////////////////////////////////////////////////////////////////
// Tau_calloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
void * Tau_calloc(const char *file, int line, size_t nmemb, size_t size)
{
  /* Get the event that is created */
  user_event_t * e = Tau_malloc_before(file, line, nmemb * size);

  void * ptr = calloc(nmemb, size);

#ifdef DEBUGPROF
  printf("TAU_CALLOC<%d>: %s:%d ptr = %p size = %d\n", RtsLayer::myNode(), file, line, ptr, size);
#endif /* DEBUGPROF */

  /* associate the event generated and its size with the address of memory
   * allocated by calloc. This is used later for memory leak detection and
   * to evaluate the size of the memory freed in the Tau_free(ptr) routine. */
  Tau_malloc_after(ptr, nmemb * size, e);

  return ptr;  /* what was allocated */
}

//////////////////////////////////////////////////////////////////////
// Tau_track_memory_allocation does everything that Tau_malloc does except
// allocate memory
//////////////////////////////////////////////////////////////////////
void Tau_track_memory_allocation(const char *file, int line, size_t size, void * ptr)
{
#ifdef DEBUGPROF
  printf("allocation: %d, ptr = %lx\n", line, ptr);
#endif /* DEBUGPROF */
  Tau_malloc_after(ptr, size, Tau_malloc_before(file, line, size));
}

//////////////////////////////////////////////////////////////////////
// Tau_new returns the expression (new[] foo) and  does everything that 
// Tau_track_memory_allocation does
//////////////////////////////////////////////////////////////////////
void * Tau_new(const char *file, int line, size_t size, void * ptr)
{
  /* the memory is already allocated by the time we see this ptr */
  Tau_track_memory_allocation(file, line, size, ptr);
  return ptr;
}

//////////////////////////////////////////////////////////////////////
// Tau_free_before does everything prior to free'ing the memory
//////////////////////////////////////////////////////////////////////
void Tau_free_before(const char *file, int line, void * ptr)
{
#ifdef DEBUGPROF
  printf("C++: Tau_free_before: file = %s, ptr=%lx,  long file = %uld\n", file, file, file_hash);
#endif /* DEBUGPROF */

  typedef pointer_size_map_t::iterator size_iter_t;
  typedef malloc_map_t::iterator malloc_iter_t;

  pointer_size_map_t & sizemap = TheTauPointerSizeMap();
  malloc_map_t & mallocmap = TheTauMallocMap();

  hash_t file_hash = Tau_hash(line, file);
  long iptr = Tau_convert_ptr_to_long(ptr);

  size_t sz = 0;
  size_iter_t size_it = sizemap.find(iptr);
  if (size_it != sizemap.end()) {
    // Size was found, but sometimes a single address corresponds multiple allocations,
    // i.e. Intel compilers can do this when there's a leak.
    // See how many allocations correspond to the address ptr points to
    pair<size_iter_t, size_iter_t> range = sizemap.equal_range(iptr);
    if (range.first != range.second) {
  #ifdef DEBUG
      printf("Found more than one occurrence of ptr in TauGetMemoryAllocatedSize\n");
  #endif /* DEBUG */
      // Just pick one since we can't know which allocation size is correct.
      size_it = range.second;
    }
    sz = size_it->second.first;
    // Remove the record
    sizemap.erase(size_it);
  }

  malloc_iter_t it = mallocmap.find(file_hash);
  if (it == mallocmap.end()) {
    char * s = (char*)malloc(strnlen(file, MAX_STRING_LEN)+32);
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
    printf("Found it! Name = %s\n", it->second->contextevent->GetEventName());
#endif /* DEBUGPROF */
    it->second->TriggerEvent(sz);
  }
#ifdef DEBUGPROF
  printf("C++: Tau_free: %s:%d\n", file, line);  
#endif /* DEBUGPROF */
}

//////////////////////////////////////////////////////////////////////
// Tau_free calls Tau_free_before and free's the memory allocated 
//////////////////////////////////////////////////////////////////////
void Tau_free(const char *file, int line, void * p)
{
  Tau_free_before(file, line, p);

#ifdef DEBUGPROF
  printf("TAU_FREE  <%d>: %s:%d ptr = %p\n", RtsLayer::myNode(), file, line, p);
#endif /* DEBUGPROF */

  /* and actually free the memory */
  free(p);
}

//////////////////////////////////////////////////////////////////////
// Tau_realloc calls free_before, realloc and memory allocation tracking routine
//////////////////////////////////////////////////////////////////////
void* Tau_realloc(const char *file, int line, void * p, size_t size)
{
  Tau_free_before(file, line, p); 
  void *retval = realloc(p, size);
  Tau_track_memory_allocation(file, line, size, retval);
  return retval;
}

//////////////////////////////////////////////////////////////////////
// Tau_track_memory_deallocation does everything that Tau_free does except
// de-allocate memory
//////////////////////////////////////////////////////////////////////
void Tau_track_memory_deallocation(const char *file, int line, void * ptr)
{
  //printf("DEallocation: %s:%d, ptr = %lx\n", file, line, ptr);
#ifdef DEBUGPROF
  printf("DEallocation: %d, ptr = %lx\n", line, ptr);
#endif /* DEBUGPROF */
  Tau_free_before(file, line, ptr);
}

//////////////////////////////////////////////////////////////////////
// TauDetectMemoryLeaks iterates over the list of pointers and checks
// which blocks have not been freed. This is called at the very end of
// the program from Profiler::StoreData
//////////////////////////////////////////////////////////////////////
void TauDetectMemoryLeaks(void)
{
  typedef pointer_size_map_t::iterator size_iter_t;
  typedef leak_map_t::iterator leak_iter_t;

  pointer_size_map_t & sizemap = TheTauPointerSizeMap();
  leak_map_t & leakmap = TheTauMemoryLeakMap();
  if (sizemap.empty()) return;

  for (size_iter_t size_it = sizemap.begin(); size_it != sizemap.end(); size_it++) {
    size_t sz = size_it->second.first;
    user_event_t * e = size_it->second.second;

#ifdef DEBUGPROF
    printf("Found leak for block of memory of size %d from memory allocated at:%s\n",
        sz, e->GetEventName());
#endif /* DEBUGPROF */

    long leak_key = Tau_convert_ptr_to_long(e);
    leak_iter_t leak_it = leakmap.find(leak_key);
    if (leak_it == leakmap.end()) {
      char * s = (char*)malloc(strnlen(e->GetEventName(), MAX_STRING_LEN)+32);
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
// The amount of memory available for use (in MB) 
//////////////////////////////////////////////////////////////////////
int TauGetFreeMemory(void)
{

  /* Catamount has a heap_info call that returns the available memory headroom */
#if defined(TAU_CATAMOUNT)
  size_t fragments;
  unsigned long total_free, largest_free, total_used;
  if (heap_info(&fragments, &total_free, &largest_free, &total_used) == 0)
  {  /* return free memory in MB */
    return  (int) (total_free/(1024*1024));
  }
  return 0; /* if it didn't work */
#elif defined(TAU_BGP)
  uint32_t available_heap;
  Kernel_GetMemorySize( KERNEL_MEMSIZE_ESTHEAPAVAIL, &available_heap );
  return available_heap / (1024 * 1024);
#else
#define TAU_BLOCK_COUNT 1024

  char* blocks[TAU_BLOCK_COUNT];
  char* ptr;
  int i,j;
  int freemem = 0;
  int factor = 1;

  i = 0; /* initialize it */
  while (1)
  {
    ptr = (char *) malloc(factor*1024*1024); /* 1MB chunk */
    if (ptr && i < TAU_BLOCK_COUNT)
    { /* so we don't go over the size of the blocks */
      blocks[i] = ptr;
      i++; /* increment the no. of elements in the blocks array */
      freemem += factor; /* assign the MB allocated */
      factor *= 2;  /* try with twice as much the next time */
    }
    else
    {
      if (factor == 1) break; /* out of the loop */
      factor = 1; /* try with a smaller chunk size */
    }
  }

  for (j=0; j < i; j++)
    free(blocks[j]);

  return freemem;

#endif /* TAU_CATAMOUNT */
}

/***************************************************************************
 * $RCSfile: TauMemory.cpp,v $   $Author: amorris $
 * $Revision: 1.33 $   $Date: 2010/01/27 00:47:51 $
 * TAU_VERSION_ID: $Id: TauMemory.cpp,v 1.33 2010/01/27 00:47:51 amorris Exp $ 
 ***************************************************************************/
