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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <Profile/Profiler.h>
#if (defined(__APPLE_CC__) || defined(TAU_APPLE_XLC))
#include <malloc/malloc.h>
#else
#include <malloc.h> 
#endif /* apple */

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

//////////////////////////////////////////////////////////////////////
// Class for building the map
//////////////////////////////////////////////////////////////////////
struct Tault2Longs
{
  bool operator() (const long *l1, const long *l2) const
 { /* each element has two longs, char * and line no. */
   /* first check 0th index (size) */
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
#define TAU_MALLOC_MAP_TYPE long*, TauUserEvent *, Tault2Longs

map<TAU_MALLOC_MAP_TYPE >& TheTauMallocMap(void)
{
  static map<TAU_MALLOC_MAP_TYPE > mallocmap;
  return mallocmap;
}
 

//////////////////////////////////////////////////////////////////////
// This routine
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// This class allows us to convert void * to the desired type in malloc
//////////////////////////////////////////////////////////////////////

class TauVoidPointer {
  void *p;
  public:
    TauVoidPointer (void *pp) : p (pp) { }
    template <class T> operator T *() { return (T *) p; }
};

//////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////
#define TAU_POINTER_SIZE_MAP_TYPE long, size_t, TaultLong

//////////////////////////////////////////////////////////////////////
// 
//////////////////////////////////////////////////////////////////////
map<TAU_POINTER_SIZE_MAP_TYPE >& TheTauPointerSizeMap(void)
{
  static map<TAU_POINTER_SIZE_MAP_TYPE > pointermap;
  return pointermap;
}

//////////////////////////////////////////////////////////////////////
// Tau_malloc for C++ has file and line information
//////////////////////////////////////////////////////////////////////
TauVoidPointer Tau_malloc(const char *file, int line, size_t size)
{
  long *key = new long[2];
  key[0] = (long) file;
  key[1] = (long) line;
 
  /* We've set the key */
  map<TAU_MALLOC_MAP_TYPE >::iterator it = TheTauMallocMap().find(key);

  if (it == TheTauMallocMap().end())
  {
    /* Couldn't find it */
    char *s = new char [strlen(file)+32];  
    sprintf(s, "malloc size <file=%s, line=%d>",file, line);
    TauUserEvent *e = new TauUserEvent(s);
    e->TriggerEvent(size);
    TheTauMallocMap().insert(map<TAU_MALLOC_MAP_TYPE >::value_type(key, e));
  }
  else
  { /* found it */
    (*it).second->TriggerEvent(size);
  }
#ifdef DEBUGPROF
  printf("C++: Tau_malloc: %s:%d:%d\n", file, line, size);
#endif /* DEBUGPROF */

  /* Add the size to the map */
  TauVoidPointer ptr = malloc(size);
  char *p1 = ptr;
  TheTauPointerSizeMap()[(long)p1] = size; 
  return ptr;
}

//////////////////////////////////////////////////////////////////////
// TauGetMemoryAllocatedSize returns the size of the pointer p
//////////////////////////////////////////////////////////////////////
size_t TauGetMemoryAllocatedSize(TauVoidPointer p)
{
  char *p1 = p;
  map<TAU_POINTER_SIZE_MAP_TYPE >::iterator it = TheTauPointerSizeMap().find((long)p1);
  if (it == TheTauPointerSizeMap().end())
    return 0; // don't know the size 
  else
    return (*it).second;  
}
//////////////////////////////////////////////////////////////////////
// Tau_free for C++ has file and line information
//////////////////////////////////////////////////////////////////////
void Tau_free(const char *file, int line, TauVoidPointer p)
{
  long *key = new long[2];
  key[0] = (long) file;
  key[1] = (long) line;
 
  /* We've set the key */
  map<TAU_MALLOC_MAP_TYPE >::iterator it = TheTauMallocMap().find(key);

  size_t sz = TauGetMemoryAllocatedSize(p);
  if (it == TheTauMallocMap().end())
  {
    /* Couldn't find it */
    char *s = new char [strlen(file)+32];  
    sprintf(s, "free size <file=%s, line=%d>",file, line);
    TauUserEvent *e = new TauUserEvent(s);
    e->TriggerEvent(sz);
    TheTauMallocMap().insert(map<TAU_MALLOC_MAP_TYPE >::value_type(key, e));
  }
  else
  { /* found it */
    (*it).second->TriggerEvent(sz);
  }
#ifdef DEBUGPROF
  printf("C++: Tau_free: %s:%d\n", file, line);  
#endif /* DEBUGPROF */
  free(p);
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

/***************************************************************************
 * $RCSfile: TauMemory.cpp,v $   $Author: sameer $
 * $Revision: 1.6 $   $Date: 2004/03/17 02:14:35 $
 * TAU_VERSION_ID: $Id: TauMemory.cpp,v 1.6 2004/03/17 02:14:35 sameer Exp $ 
 ***************************************************************************/
