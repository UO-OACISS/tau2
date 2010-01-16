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
#include <sys/types.h>
/* needed for Linux stdlib.h */
#define __malloc_and_calloc_defined 
#define __need_malloc_and_calloc
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void *malloc (size_t size);
void *Tau_malloc_C( const char *file, int line, size_t size);
void Tau_free_C(const char *file, int line, void *p);
void *Tau_realloc_C(const char *file, int line, void *p, size_t size);
void free(void *p);

void *calloc(size_t nmemb, size_t size);
void *Tau_calloc_C(const char *file, int line, size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);


#ifdef __cplusplus
}
#endif /* __cplusplus */
/*
//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////
*/

/********************************************************************/
/* For further details see David Mazieres (NYU) article:
 * http://www.scs.cs.nyu.edu/~dm/c++-new.html 
 * The above article describes the core design ideas on which the 
 * TAU memory allocator is based */
/********************************************************************/

#include <sys/types.h>
#ifndef TAU_USE_CXX_MALLOC_API
/* For C */ 

/* void *Tau_malloc_C( const char *file, int line, size_t size);
*/
#define malloc(size) Tau_malloc_C(__FILE__, __LINE__, size)

/* void free (void *);
  void Tau_free_C(const char *file, int line, void *p);
*/
#define free(p) Tau_free_C(__FILE__, __LINE__, p)

#define realloc(p, s) Tau_realloc_C(__FILE__, __LINE__, p, s)

#define calloc(n, s) Tau_calloc_C(__FILE__, __LINE__, n, s)

#else /* TAU_USE_CXX_MALLOC_API */
/* For C++ */

class TauVoidPointer {
  void *p;
  public:
    TauVoidPointer (void *pp) : p (pp) { }
    template <class T> operator T *() { return (T *) p; }
};

TauVoidPointer Tau_malloc(const char *file, int line, size_t size);
void Tau_free(const char *file, int line, TauVoidPointer p);

#define malloc(size) Tau_malloc(__FILE__, __LINE__, size)
#define calloc(nmemb, size) Tau_calloc(__FILE__, __LINE__, nmemb, size)
#define free(p) Tau_free(__FILE__, __LINE__, p)

#endif /* TAU_USE_CXX_MALLOC_API */



#endif /* _TAU_MALLOC_H_ */
