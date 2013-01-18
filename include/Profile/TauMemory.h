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

#if defined(__darwin__) || defined(__APPLE__) || defined(TAU_XLC)
#undef HAVE_MEMALIGN
#undef HAVE_PVALLOC
#else
#define HAVE_MEMALIGN 1
#define HAVE_PVALLOC 1
#endif

#define TAU_MEMORY_UNKNOWN_LINE 0
#define TAU_MEMORY_UNKNOWN_FILE "Unknown"
#define TAU_MEMORY_UNKNOWN_FILE_STRLEN 7

// libc bindings

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int Tau_memory_wrapper_present(void);
void Tau_set_memory_wrapper_present(int);

void Tau_memory_initialize(void);

size_t Tau_page_size(void);

void Tau_detect_memory_leaks(void);
size_t Tau_get_bytes_allocated(void);

void Tau_track_memory_allocation(void *, size_t, char const *, int);
void Tau_track_memory_deallocation(void *, char const *, int);

void * Tau_allocate_unprotected(size_t);

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

int __tau_strcmp(char const *, char const *);
int Tau_strcmp(char const *, char const *, char const *, int);

#if 0
char * Tau_strdup(char const *, char const *, int);
char * Tau_strcpy(char *, char const *, char const *, int);
char * Tau_strcat(char *, char const *, char const *, int);

char * Tau_strncpy(char *, char const *, size_t, char const *, int);
char * Tau_strncat(char *, char const *, size_t, char const *, int);

void * Tau_memcpy(void *, void const *, size_t, char const *, int);
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_MEMORY_H_ */

/***************************************************************************
 * $RCSfile: TauMemory.h,v $   $Author: amorris $
 * $Revision: 1.4 $   $Date: 2010/02/03 06:09:44 $
 * TAU_VERSION_ID: $Id: TauMemory.h,v 1.4 2010/02/03 06:09:44 amorris Exp $ 
 ***************************************************************************/
