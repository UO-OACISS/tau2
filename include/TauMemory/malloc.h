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

void * malloc (size_t size);
void * realloc(void *ptr, size_t size);
void * calloc(size_t nmemb, size_t size);
void free(void *p);

void * Tau_malloc( const char *file, int line, size_t size);
void * Tau_realloc(const char *file, int line, void *p, size_t size);
void * Tau_calloc(const char *file, int line, size_t nmemb, size_t size);
void Tau_free(const char *file, int line, void *p);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#define malloc(size)  Tau_malloc(__FILE__, __LINE__, size)
#define realloc(p, s) Tau_realloc(__FILE__, __LINE__, p, s)
#define calloc(n, s)  Tau_calloc(__FILE__, __LINE__, n, s)
#define free(p)       Tau_free(__FILE__, __LINE__, p)

#endif /* _TAU_MALLOC_H_ */
