/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://tau.uoregon.edu                             **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**      File            : TauMemory.h                                      **
**      Contact         : tau-bugs@cs.uoregon.edu                          **
**      Documentation   : See http://tau.uoregon.edu                       **
**      Description     : Support for memory tracking                      **
**                                                                         **
****************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#ifndef _TAU_MEMORY_H_
#define _TAU_MEMORY_H_

#include <tau_internal.h>

#ifdef __cplusplus
#ifdef TAU_DOT_H_LESS_HEADERS
#include <new>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <new.h>
#endif /* TAU_DOT_H_LESS_HEADERS */
#endif

#if defined(__darwin__) || defined(__APPLE__) || defined(TAU_XLC)
#undef HAVE_MEMALIGN
#undef HAVE_PVALLOC
#else
#define HAVE_MEMALIGN 1
#define HAVE_PVALLOC 1
#endif

// libc bindings

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

size_t Tau_page_size(void);

void Tau_detect_memory_leaks(void);

void Tau_track_memory_allocation(void *, size_t, char const *, int);
void Tau_track_memory_deallocation(void *, char const *, int);

void * Tau_malloc(size_t, char const *, int);
void * Tau_calloc(size_t, size_t, char const *, int);
void   Tau_free(void *, char const *, int);
#ifdef HAVE_MEMALIGN
void * Tau_memalign(size_t, size_t, char const *, int);
#endif
int    Tau_posix_memalign(void **, size_t, size_t, char const *, int);
void * Tau_realloc(void *, size_t, char const *, int);
void * Tau_valloc(size_t, char const *, int);
#ifdef HAVE_PVALLOC
void * Tau_pvalloc(size_t, char const *, int);
#endif

char * Tau_strdup(char const *, char const *, int);
char * Tau_strcpy(char *, char const *, char const *, int);
char * Tau_strcat(char *, char const *, char const *, int);

char * Tau_strncpy(char *, char const *, size_t, char const *, int);
char * Tau_strncat(char *, char const *, size_t, char const *, int);

void * Tau_memcpy(void *, void const *, size_t, char const *, int);

#ifdef __cplusplus
}
#endif /* __cplusplus */


// C++ bindings

#ifdef __cplusplus

#if 0

void * Tau_operator_new(size_t, bool, const char *, int);
void Tau_operator_delete(void *, bool, const char *, int);
int Tau_operator_delete_init(const char *, int);

void * operator new(std::size_t) throw(std::bad_alloc);
void * operator new[](std::size_t) throw(std::bad_alloc);

void   operator delete(void *) throw();
void   operator delete[](void *) throw();

#if 0
void * operator new(std::size_t, const std::nothrow_t &) throw();
void * operator new[](std::size_t, const std::nothrow_t &) throw();
#endif

void   operator delete(void *, const std::nothrow_t &) throw();
void   operator delete[](void *, const std::nothrow_t &) throw();

#if 0
void * operator new(std::size_t, void *) throw(std::bad_alloc);
void * operator new[](std::size_t, void *) throw(std::bad_alloc);
void   operator delete(void *, void *) throw();
void   operator delete[](void *, void *) throw();
#endif

void * operator new(std::size_t, char const *, int) throw(std::bad_alloc);
void * operator new[](std::size_t, char const *, int) throw(std::bad_alloc);
void   operator delete(void *, char const *, int) throw();
void   operator delete[](void *, char const *, int) throw();

void * operator new(std::size_t, const std::nothrow_t &, char const *, int) throw();
void * operator new[](std::size_t, const std::nothrow_t &, char const *, int) throw();
void   operator delete(void *, const std::nothrow_t &, char const *, int) throw();
void   operator delete[](void *, const std::nothrow_t &, char const *, int) throw();

void * operator new(std::size_t, void *, char const *, int) throw(std::bad_alloc);
void * operator new[](std::size_t, void *, char const *, int) throw(std::bad_alloc);
void   operator delete(void *, void *, char const *, int) throw();
void   operator delete[](void *, void *, char const *, int) throw();
#endif

#endif


#endif /* _TAU_MEMORY_H_ */

/***************************************************************************
 * $RCSfile: TauMemory.h,v $   $Author: amorris $
 * $Revision: 1.4 $   $Date: 2010/02/03 06:09:44 $
 * TAU_VERSION_ID: $Id: TauMemory.h,v 1.4 2010/02/03 06:09:44 amorris Exp $ 
 ***************************************************************************/
