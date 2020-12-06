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

#include <stdlib.h>
#include "memory_wrapper_strings.h"

#ifdef __cplusplus
extern "C" {
#endif

// Assume 4K pages unless we know otherwise.
// We cannot determine this at runtime because it must be known during
// the bootstrap process and it would be unsafe to make any system calls there.
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif

// Size of heap memory for library wrapper bootstrapping
#ifdef __APPLE__
// Starting on macOS 11, PAGE_SIZE is not constant on macOS
// Apple recommends using PAGE_MAX_SIZE instead.
// see https://developer.apple.com/videos/play/wwdc2020/10214/?time=549
#define BOOTSTRAP_HEAP_SIZE (3*PAGE_MAX_SIZE)
#else
#define BOOTSTRAP_HEAP_SIZE (3*PAGE_SIZE)
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


// Returns a handle to the system's implementation of the routine
malloc_t get_system_malloc();
calloc_t get_system_calloc();
realloc_t get_system_realloc();
memalign_t get_system_memalign();
posix_memalign_t get_system_posix_memalign();
valloc_t get_system_valloc();
pvalloc_t get_system_pvalloc();
free_t get_system_free();

void * malloc_wrapper(size_t size);
void * calloc_wrapper(size_t count, size_t size);
void * realloc_wrapper(void * ptr, size_t size);
void free_wrapper(void * ptr);
void * memalign_wrapper(size_t alignment, size_t size);
int posix_memalign_wrapper(void ** ptr, size_t alignment, size_t size);
void * valloc_wrapper(size_t size);
void * pvalloc_wrapper(size_t size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MEMORY_WRAPPER_H_ */
