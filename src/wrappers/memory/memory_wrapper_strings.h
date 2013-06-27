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

#ifndef MEMORY_WRAPPER_STRINGS_H_
#define MEMORY_WRAPPER_STRINGS_H_

// Types of function pointers for wrapped functions
typedef int (*strcmp_t)(char const *, char const *);

extern strcmp_t strcmp_handle;

strcmp_t Tau_get_system_strcmp();

#endif /* MEMORY_WRAPPER_STRINGS_H_ */
