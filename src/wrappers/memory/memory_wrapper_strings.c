/****************************************************************************
**      TAU Portable Profiling Package         **
**      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2010                     **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**  File          : memory_wrapper_dynamic.c
**  Description   : TAU Profiling Package
**  Contact       : tau-bugs@cs.uoregon.edu
**  Documentation : See http://www.cs.uoregon.edu/research/tau
**
**  Description   : TAU memory profiler and debugger
**
****************************************************************************/

#include "memory_wrapper_strings.h"

#ifdef _MSC_VER
/* define these functions as non-intrinsic */
#pragma function( memcpy, strcpy, strcat )
#endif

/******************************************************************************
 * libc string function wrappers
 ******************************************************************************/

#if 0
strcmp_t Tau_get_system_strcmp()
{
  return (strcmp_t)get_system_function_handle("strcmp");
}

int strcmp(const char *s1, const char *s2)
{
  return strcmp_handle(s1, s2);
}
#endif


/*********************************************************************
 * EOF
 ********************************************************************/
