/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2004                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**      File            : TauHandler.cpp                                  **
**      Description     : TAU Profiling Package                           **
**      Author          : Sameer Shende                                   **
**      Contact         : sameer@cs.uoregon.edu sameer@acl.lanl.gov       **
**      Documentation   : See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/


#ifndef _TAU_MALLOC_H_
#define _TAU_MALLOC_H_
#define _MALLOC_H 1

#include <stdlib.h>
#include <Profile/TauMemory.h>

/* needed for Linux stdlib.h */
#define __malloc_and_calloc_defined 
#define __need_malloc_and_calloc

// libc bindings

#define malloc(SIZE)                Tau_malloc(SIZE, __FILE__, __LINE__)
#define calloc(ELEMCOUNT, ELEMSIZE) Tau_calloc(ELEMCOUNT, ELEMSIZE, __FILE__, __LINE__)
#define free(BASEADR)               Tau_free(BASEADR, __FILE__, __LINE__)
#ifdef HAVE_MEMALIGN
#define memalign(ALIGNMENT, SIZE)   Tau_memalign(ALIGNMENT, SIZE, __FILE__, __LINE__)
#endif
#define posix_memalign(MEMPTR, ALIGNMENT, SIZE)  Tau_posix_memalign(MEMPTR, ALIGNMENT, SIZE, __FILE__, __LINE__)
#define realloc(BASEADR, NEWSIZE)   Tau_realloc(BASEADR, NEWSIZE, __FILE__, __LINE__)
#define valloc(SIZE)                Tau_valloc(SIZE, __FILE__, __LINE__)
#ifdef HAVE_PVALLOC
#define pvalloc(SIZE)               Tau_pvalloc(SIZE, __FILE__, __LINE__)
#endif

#endif /* _TAU_MALLOC_H_ */
