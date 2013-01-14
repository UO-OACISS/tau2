/****************************************************************************
**      TAU Portable Profiling Package         **
**      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2010                     **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**  File          : memory_wrapper.h
**  Description   : TAU Profiling Package
**  Contact       : tau-bugs@cs.uoregon.edu
**  Documentation : See http://www.cs.uoregon.edu/research/tau
**
**  Description   : TAU memory profiler and debugger
**
****************************************************************************/

#ifndef MEMORY_WRAPPER_H_
#define MEMORY_WRAPPER_H_

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600 /* see: man posix_memalign */
#endif

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Types of function pointers for wrapped functions
typedef void * (*malloc_t)(size_t);
typedef void * (*calloc_t)(size_t, size_t);
typedef void * (*realloc_t)(void *, size_t);
typedef void * (*memalign_t)(size_t, size_t);
typedef int    (*posix_memalign_t)(void **, size_t, size_t);
typedef void * (*valloc_t)(size_t);
typedef void * (*pvalloc_t)(size_t);
typedef void   (*free_t)(void *);

// Handles to an implementation of the call
extern malloc_t malloc_handle;
extern calloc_t calloc_handle;
extern free_t free_handle;
extern memalign_t memalign_handle;
extern posix_memalign_t posix_memalign_handle;
extern realloc_t realloc_handle;
extern valloc_t valloc_handle;
extern pvalloc_t pvalloc_handle;

// Returns a handle to the system's implementation of the routine
malloc_t Tau_get_system_malloc();
calloc_t Tau_get_system_calloc();
realloc_t Tau_get_system_realloc();
memalign_t Tau_get_system_memalign();
posix_memalign_t Tau_get_system_posix_memalign();
valloc_t Tau_get_system_valloc();
pvalloc_t Tau_get_system_pvalloc();
free_t Tau_get_system_free();

int Tau_memory_wrapper_init(void);
int Tau_memory_wrapper_passthrough(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MEMORY_WRAPPER_H_ */
