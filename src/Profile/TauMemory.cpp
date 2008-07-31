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

//#define DEBUGPROF 1
//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <Profile/Profiler.h>
#if (defined(__APPLE_CC__) || defined(TAU_APPLE_XLC))
#include <malloc/malloc.h>
#else
#ifdef TAU_FREEBSD
#include <stdlib.h> 
#else /* TAU_FREEBSD */
#include <malloc.h> 
#endif /* TAU_FREEBSD */
#endif /* apple */

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


//////////////////////////////////////////////////////////////////////
// Class for building the map
//////////////////////////////////////////////////////////////////////
struct Tault2Longs
{
  bool operator() (const long *l1, const long *l2) const
 { /* each element has two longs, char * and line no. */

   if (l1[0] != l2[0]) return (l1[0] < l2[0]);
   return l1[1] < l2[1]; 
 }
};

struct TaultLong
{
  bool operator() (const long l1, const long l2) const
 { 
   return l1 < l2; 
 }
};

#define TAU_USER_EVENT_TYPE TauContextUserEvent
//#define TAU_USER_EVENT_TYPE TauUserEvent
//#define TAU_MALLOC_MAP_TYPE long*, TAU_USER_EVENT_TYPE *, Tault2Longs
#define TAU_MALLOC_MAP_TYPE pair<long,unsigned long>, TauUserEvent *, less<pair<long,unsigned long> >
#define TAU_MEMORY_LEAK_MAP_TYPE long, TauUserEvent *, TaultLong

//////////////////////////////////////////////////////////////////////
map<TAU_MALLOC_MAP_TYPE >& TheTauMallocMap(void)
{
  static map<TAU_MALLOC_MAP_TYPE > mallocmap;
  return mallocmap;
}

//////////////////////////////////////////////////////////////////////
// We store the leak detected events here 
//////////////////////////////////////////////////////////////////////
map<TAU_MEMORY_LEAK_MAP_TYPE >& TheTauMemoryLeakMap(void)
{
  static map<TAU_MEMORY_LEAK_MAP_TYPE > leakmap;
  return leakmap;
}


 

//////////////////////////////////////////////////////////////////////
// This routine
//////////////////////////////////////////////////////////////////////

#ifdef TAU_USE_SDBM_HASH
unsigned long
Tau_hash(unsigned char *str)
{
  unsigned long hash = 0;
  int c;

  while (c = *str++)
    hash = c + (hash << 6) + (hash << 16) - hash;

  return hash;
}
#else
#ifdef TAU_USE_MAP_BASED_HASH
struct TauLtStr{
  bool operator()(unsigned char* s1, unsigned char* s2) const{
    printf("comparing s1 %s:%p and s2 %s:%p\n", s1,s1, s2, s2);
    return strcmp((const char *)s1, (const char *) s2) < 0;
  }//operator()
};


//////////////////////////////////////////////////////////////////////
unsigned long Tau_hash(const char *str)
{
  static map<string, unsigned long, less<string> > fileDB;
  static unsigned long counter = 0;
  unsigned long myid;
  map<string, unsigned long, less<string> >::iterator it;

  if ((it = fileDB.find(string(str))) != fileDB.end()) {
    myid = (*it).second;

#ifdef DEBUG
    printf("Tau_hash: found name %s: %p: %ld\n", str, str, myid);
#endif /* DEBUG */
 
  }
  else {
    RtsLayer::LockDB();
    myid = ++counter; /* increment the id */
    RtsLayer::UnLockDB();
#ifdef DEBUG
    printf("Tau_hash: Not found name %s: %p: %ld\n", str, str, myid);
#endif /* DEBUG */

    fileDB[string(str)] = myid;
  }
  return myid;

}
#else
//////////////////////////////////////////////////////////////////////
unsigned long Tau_hash(unsigned char *str)
{
  unsigned long hash = 5381;
  int c;
   
  while (c = *str++)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
  return hash;
}
#endif /* map hash */
#endif





//////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////
#define TAU_POINTER_SIZE_MAP_TYPE long, pair<size_t, long>, TaultLong

//////////////////////////////////////////////////////////////////////
// This map stores the memory allocated and its associations
//////////////////////////////////////////////////////////////////////
multimap<TAU_POINTER_SIZE_MAP_TYPE >& TheTauPointerSizeMap(void)
{
  static multimap<TAU_POINTER_SIZE_MAP_TYPE > pointermap;
  return pointermap;
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc_before creates/access the event associated with tracking
// memory allocation for the specified line and file. 
//////////////////////////////////////////////////////////////////////
TAU_USER_EVENT_TYPE* Tau_malloc_before(const char *file, int line, size_t size)
{
/* we use pair<long,long> (line, file) as the key to index the mallocmap */
#ifdef TAU_USE_MAP_BASED_HASH
  unsigned long file_hash = Tau_hash(file);
#else
  unsigned long file_hash = Tau_hash((unsigned char *)file);
#endif 

#ifdef DEBUGPROF
  printf("C++: Tau_malloc_before: file = %s, ptr=%lx,  long file = %uld\n", file, file, file_hash);
#endif /* DEBUGPROF */
  map<TAU_MALLOC_MAP_TYPE >::iterator it = TheTauMallocMap().find(pair<long, unsigned long>(line,file_hash));
  TAU_USER_EVENT_TYPE *e ;

  if (it == TheTauMallocMap().end())
  {
    /* Couldn't find it */
    char *s = new char [strlen(file)+32];  
    sprintf(s, "malloc size <file=%s, line=%d>",file, line);
#ifdef DEBUGPROF
    printf("C++: Tau_malloc: creating new user event %s\n", s);
#endif /* DEBUGPROF */
    e = new TAU_USER_EVENT_TYPE(s);
    e->TriggerEvent(size);
    TheTauMallocMap()[pair<long,unsigned long>(line, file_hash)] = e->contextevent;
    /* do not store the TauContextUserEvent, but the UserEvent that represents
    the full name at the point of execution  of trigger. The former is just
    a vessel for storing the last stored value of the context event. The latter
    is the actual context event (A=>B=>foo) that it triggered */
    delete[] (s);
  }
  else
  { /* found it */
    TauUserEvent *foundevt;
#ifdef DEBUGPROF
    printf("Found it! Name = %s\n", (*it).second->GetEventName());
#endif /* DEBUGPROF */
    foundevt = (*it).second;
    foundevt->ctxevt->TriggerEvent(size);
    e = foundevt->ctxevt;
  }
#ifdef DEBUGPROF
  printf("C++: Tau_malloc: %s:%d:%d\n", file, line, size);
#endif /* DEBUGPROF */

  return e; /* the event that is created in this routine */
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc_after associates the event and size with the address allocated
//////////////////////////////////////////////////////////////////////
void Tau_malloc_after(TauVoidPointer ptr, size_t size, TAU_USER_EVENT_TYPE *e)
{
#ifdef TAU_WINDOWS
  char *p1 = (char*) (void*)ptr;
#else
  char *p1 = ptr;
#endif
  /* store the size of memory allocated with the address of the pointer */
  //TheTauPointerSizeMap()[(long)p1] = pair<size_t, long>(size, (long) e); 
  TheTauPointerSizeMap().insert(pair<const long, pair<size_t, long> >((long)p1, pair<size_t, long>(size, (long) e->contextevent))); 
  return;
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc calls the before and after routines and allocates memory
//////////////////////////////////////////////////////////////////////
TauVoidPointer Tau_malloc(const char *file, int line, size_t size)
{
  TAU_USER_EVENT_TYPE *e; 

  /* We get the event that is created */
  e = Tau_malloc_before(file, line, size);

  TauVoidPointer ptr = malloc(size);

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
// Tau_track_memory_allocation does everything that Tau_malloc does except
// allocate memory
//////////////////////////////////////////////////////////////////////
void Tau_track_memory_allocation(const char *file, int line, size_t size, TauVoidPointer ptr)
{
  //printf("allocation: %s:%d, ptr = %lx\n", file, line, ptr);
#ifdef DEBUGPROF
  printf("allocation: %d, ptr = %lx\n", line, ptr);
#endif /* DEBUGPROF */
  Tau_malloc_after(ptr, size, Tau_malloc_before(file, line, size));
}

//////////////////////////////////////////////////////////////////////
// Tau_new returns the expression (new[] foo) and  does everything that 
// Tau_track_memory_allocation does
//////////////////////////////////////////////////////////////////////
TauVoidPointer Tau_new(const char *file, int line, size_t size, TauVoidPointer ptr)
{ /* the memory is already allocated by the time we see this ptr */
  Tau_track_memory_allocation(file, line, size, ptr);
  return ptr;
}
		  

//////////////////////////////////////////////////////////////////////
// TauGetMemoryAllocatedSize returns the size of the pointer p
//////////////////////////////////////////////////////////////////////
size_t TauGetMemoryAllocatedSize(TauVoidPointer p)
{
  pair<size_t, long> result; 
#ifdef TAU_WINDOWS
  char *p1 = (char*) (void*)p;
#else
  char *p1 = p;
#endif

  typedef multimap<TAU_POINTER_SIZE_MAP_TYPE >::iterator I;

  I it = TheTauPointerSizeMap().find((long)p1);
  I it2, found_it;
  TAU_USER_EVENT_TYPE *e;

  if (it == TheTauPointerSizeMap().end())
    return 0; // don't know the size 
  else
  {
    found_it = it;  /* initialize */
    if (TheTauPointerSizeMap().count((long)p1) > 1)
    { /* Intel compiler reuses addresses that have leaks */
#ifdef DEBUG
      printf("Found more than one occurence of p1 in TauGetMemoryAllocatedSize\n");
#endif /* DEBUG */

      pair<I,I> range = TheTauPointerSizeMap().equal_range((long)p1);

      for (I it = range.first; it != range.second; ++it) {
	found_it = it;
#ifdef DEBUG
	printf("TAUGETMEMORYALLOCATEDSIZE: found <%d,%lx> \n", (*it2).second.first, (*it2).second.second);
#endif /* DEBUG */
      }
    } // if count > 1
    //TheTauPointerSizeMap().erase(it);
// Erase the last pointer you find in the list. Sometimes the same address
// can appear twice. 
    TheTauPointerSizeMap().erase(found_it);
    result = (*found_it).second;
    return result.first; /* or size_t, the first entry of the pair */
  }
}

//////////////////////////////////////////////////////////////////////
// Tau_free_before does everything prior to free'ing the memory
//////////////////////////////////////////////////////////////////////
void Tau_free_before(const char *file, int line, TauVoidPointer p)
{
  /* We've set the key */
#ifdef TAU_USE_MAP_BASED_HASH
  unsigned long file_hash = Tau_hash(file);
#else
  unsigned long file_hash = Tau_hash((unsigned char *)file);
#endif /* TAU_USE_MAP_BASED_HASH */

#ifdef DEBUGPROF
  printf("C++: Tau_free_before: file = %s, ptr=%lx,  long file = %uld\n", file, file, file_hash);
#endif /* DEBUGPROF */
  map<TAU_MALLOC_MAP_TYPE >::iterator it = TheTauMallocMap().find(pair<long,unsigned long>(line,file_hash));
  TAU_USER_EVENT_TYPE *e;

  size_t sz = TauGetMemoryAllocatedSize(p);
  if (it == TheTauMallocMap().end())
  {
    /* Couldn't find it */
    char *s = new char [strlen(file)+32];  
    sprintf(s, "free size <file=%s, line=%d>",file, line);
#ifdef DEBUGPROF
    printf("C++: Tau_free: creating new user event %s\n", s);
#endif /* DEBUGPROF */
    e = new TAU_USER_EVENT_TYPE(s);
    e->TriggerEvent(sz);
    //mallocmap.insert(map<TAU_MALLOC_MAP_TYPE >::value_type(pair<long,unsigned long>(line,file_hash), e));
    TheTauMallocMap()[pair<long,unsigned long>(line, file_hash)] = e->contextevent;
    delete[] (s); 
  }
  else
  { /* found it */
    TauUserEvent *foundevt;
#ifdef DEBUGPROF
    printf("Found it! Name = %s\n", (*it).second->GetEventName());
#endif /* DEBUGPROF */
    foundevt = (*it).second; 
    foundevt->ctxevt->TriggerEvent(sz);
    e = foundevt->ctxevt;
  }
#ifdef DEBUGPROF
  printf("C++: Tau_free: %s:%d\n", file, line);  
#endif /* DEBUGPROF */
}

//////////////////////////////////////////////////////////////////////
// Tau_free calls Tau_free_before and free's the memory allocated 
//////////////////////////////////////////////////////////////////////
void Tau_free(const char *file, int line, TauVoidPointer p)
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
void* Tau_realloc(const char *file, int line, TauVoidPointer p, size_t size)
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
void Tau_track_memory_deallocation(const char *file, int line, TauVoidPointer ptr)
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
int TauDetectMemoryLeaks(void)
{
  if (TheTauPointerSizeMap().empty()) return 0; /* do nothing */
  multimap<TAU_POINTER_SIZE_MAP_TYPE >::iterator it;

  for( it = TheTauPointerSizeMap().begin(); it != TheTauPointerSizeMap().end();
	it++)
  {
    pair<size_t, long> leak = (*it).second;
    size_t sz = leak.first; 
    TauUserEvent *e = (TauUserEvent *) leak.second;
#ifdef DEBUGPROF
    printf("Found leak for block of memory of size %d from memory allocated at:%s\n", 
    sz, e->GetEventName());
#endif /* DEBUGPROF */
    /* Have we seen e before? */
    map<TAU_MEMORY_LEAK_MAP_TYPE >::iterator it = TheTauMemoryLeakMap().find((long) e);
    if (it == TheTauMemoryLeakMap().end())
    { /* didn't find it! */
      string s (string("MEMORY LEAK! ")+e->GetEventName());
      TauUserEvent *leakevent = new TauUserEvent(s.c_str());

      TheTauMemoryLeakMap()[(long)e] = leakevent; 
      leakevent->TriggerEvent(sz);
    }
    else
    {
      (*it).second->TriggerEvent(sz);
    }
    /* Instead of making a new leakevent each time, we should use another
     * map that maps the event e with the newevent and triggers it multiple times */
  }
  return 1;
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc for C++ has file and line information
//////////////////////////////////////////////////////////////////////
extern "C" void *Tau_malloc_C( const char *file, int line, size_t size)
{
#ifdef DEBUGPROF
  printf("C: Tau_malloc: %s:%d:%d\n", file, line, size);
#endif /* DEBUGPROF */
  return (void *) Tau_malloc(file, line, size);
}

//////////////////////////////////////////////////////////////////////
// Tau_free for C++ has file and line information
//////////////////////////////////////////////////////////////////////
extern "C" void Tau_free_C(const char *file, int line, void *p)
{
#ifdef DEBUGPROF
  printf("C: Tau_free: %s:%d\n", file, line);
#endif /* DEBUGPROF */
  Tau_free(file, line, p);
}

//////////////////////////////////////////////////////////////////////
// Tau_realloc for C++ has file and line information
//////////////////////////////////////////////////////////////////////
extern "C" void * Tau_realloc_C(const char *file, int line, void *p, size_t size)
{
#ifdef DEBUGPROF
  printf("C: Tau_realloc: %s:%d\n", file, line);
#endif /* DEBUGPROF */
  /* realloc acts like a free and malloc */ 
  return (void *) Tau_realloc(file, line, p, size);

}
//////////////////////////////////////////////////////////////////////
// The amount of memory available for use (in MB) 
//////////////////////////////////////////////////////////////////////

#define TAU_BLOCK_COUNT 1024

/* Catamount has a heap_info call that returns the available memory headroom */
#ifdef TAU_CATAMOUNT
int TauGetFreeMemory(void)
{
  size_t fragments;
  unsigned long total_free, largest_free, total_used;
  if (heap_info(&fragments, &total_free, &largest_free, &total_used) == 0)
  {  /* return free memory in MB */
    return  (int) (total_free/(1024*1024));
  }
  return 0; /* if it didn't work */
}
#else /* TAU_CATAMOUNT */
int TauGetFreeMemory(void)
{
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
}
#endif /* TAU_CATAMOUNT */

/***************************************************************************
 * $RCSfile: TauMemory.cpp,v $   $Author: amorris $
 * $Revision: 1.30 $   $Date: 2008/07/31 23:49:53 $
 * TAU_VERSION_ID: $Id: TauMemory.cpp,v 1.30 2008/07/31 23:49:53 amorris Exp $ 
 ***************************************************************************/
