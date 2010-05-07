/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauMemoryWrap.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : LD_PRELOAD memory wrapper                        **
**                                                                         **
****************************************************************************/

#define _GNU_SOURCE
#include <dlfcn.h>

#define _XOPEN_SOURCE 600 /* see: man posix_memalign */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
  
#include <stdarg.h>

  
#include <TAU.h>
#include <Profile/Profiler.h>
#include <Profile/TauInit.h>
#include <Profile/TauMemoryWrap.h>


/*********************************************************************
 * malloc
 ********************************************************************/
void *malloc (size_t size) {
  Tau_memorywrap_checkInit();
  static void* (*_malloc)(size_t size) = NULL;

  if (_malloc == NULL) {
    _malloc = ( void* (*)(size_t size)) dlsym(RTLD_NEXT, "malloc");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _malloc(size);
  }

  Tau_global_incr_insideTAU();

  void *ptr = _malloc(size);
  Tau_memorywrap_add_ptr(ptr, size);
  Tau_global_decr_insideTAU();
  return ptr;
}

/*********************************************************************
 * calloc
 ********************************************************************/
// void *calloc (size_t nmemb, size_t size) {
//   Tau_memorywrap_checkInit();
//   static void* (*_calloc)(size_t nmemb, size_t size) = NULL;

//   if (_calloc == NULL) {
//     _calloc = ( void* (*)(size_t nmemb, size_t size)) dlsym(RTLD_NEXT, "calloc");
//   }

//   if (Tau_memorywrap_checkPassThrough()) {
//     return _calloc(nmemb, size);
//   }

//   Tau_global_incr_insideTAU();

//   void *ptr = _calloc(nmemb, size);
//   Tau_memorywrap_add_ptr(ptr, nmemb * size);
//   TAU_CONTEXT_EVENT(global().heapMemoryUserEvent, global().bytesAllocated);
//   Tau_global_decr_insideTAU();
//   return ptr;
// }


/*********************************************************************
 * realloc
 ********************************************************************/
void *realloc (void *ptr, size_t size) {
  Tau_memorywrap_checkInit();
  static void* (*_realloc)(void *ptr, size_t size) = NULL;

  if (_realloc == NULL) {
    _realloc = ( void* (*)(void *ptr, size_t size)) dlsym(RTLD_NEXT, "realloc");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _realloc(ptr, size);
  }

  Tau_global_incr_insideTAU();

  void *ret_ptr = _realloc(ptr, size);

  Tau_memorywrap_remove_ptr(ptr);
  Tau_memorywrap_add_ptr(ret_ptr, size);

  Tau_global_decr_insideTAU();
  return ret_ptr;
}


/*********************************************************************
 * posix_memalign
 ********************************************************************/
int posix_memalign (void **memptr, size_t alignment, size_t size) {
  Tau_memorywrap_checkInit();
  static int (*_posix_memalign)(void **memptr, size_t alignment, size_t size) = NULL;

  if (_posix_memalign == NULL) {
    _posix_memalign = ( int (*)(void **memptr, size_t alignment, size_t size)) dlsym(RTLD_NEXT, "posix_memalign");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _posix_memalign(memptr, alignment, size);
  }

  Tau_global_incr_insideTAU();

  int ret = _posix_memalign(memptr, alignment, size);
  if (ret == 0) {
    Tau_memorywrap_add_ptr(*memptr, size);
  }

  Tau_global_decr_insideTAU();
  return ret;
}



/*********************************************************************
 * free
 ********************************************************************/
void free (void *ptr) {
  Tau_memorywrap_checkInit();
  static void (*_free)(void *ptr) = NULL;

  if (_free == NULL) {
    _free = ( void (*)(void *ptr)) dlsym(RTLD_NEXT, "free");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    _free(ptr);
    return;
  }

  Tau_global_incr_insideTAU();
  _free(ptr);
  Tau_memorywrap_remove_ptr(ptr);
  Tau_global_decr_insideTAU();
}

/*********************************************************************
 * EOF
 ********************************************************************/
