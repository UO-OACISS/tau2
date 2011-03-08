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
#include <memory.h>

  
#include <TAU.h>
#include <Profile/Profiler.h>
#include <Profile/TauMemoryWrap.h>


/*********************************************************************
 * malloc
 ********************************************************************/
void *malloc (size_t size) {
  static void* (*_malloc)(size_t size) = NULL;

  if (_malloc == NULL) {
    _malloc = ( void* (*)(size_t size)) dlsym(RTLD_NEXT, "malloc");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _malloc(size);
  }

  Tau_memorywrap_checkInit();
  Tau_global_incr_insideTAU();
  void *ptr = _malloc(size);
  Tau_memorywrap_add_ptr(ptr, size);
  Tau_global_decr_insideTAU();
  return ptr;
}

#ifndef TAU_VALLOC_AVAILABLE 
#ifndef __APPLE__
#define TAU_VALLOC_AVAILABLE 
#endif /* APPLE */
#endif /* TAU_VALLOC_AVAILABLE */

#ifdef TAU_VALLOC_AVAILABLE
#include <malloc.h>
/*********************************************************************
 * valloc
 ********************************************************************/
void *valloc (size_t size) {
  static void* (*_valloc)(size_t size) = NULL;

  if (_valloc == NULL) {
    _valloc = ( void* (*)(size_t size)) dlsym(RTLD_NEXT, "valloc");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _valloc(size);
  }

  Tau_memorywrap_checkInit();
  Tau_global_incr_insideTAU();
  void *ptr = _valloc(size);
  Tau_memorywrap_add_ptr(ptr, size);
  Tau_global_decr_insideTAU();
  return ptr;
}

/*********************************************************************
 * memalign
 ********************************************************************/
void * memalign (size_t alignment, size_t size) {
  static void * (*_memalign)(size_t alignment, size_t size) = NULL;
  static void *ret; 

  if (_memalign == NULL) {
    _memalign = ( void * (*)(size_t alignment, size_t size)) dlsym(RTLD_NEXT, "memalign");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _memalign(alignment, size);
  }

  Tau_memorywrap_checkInit();
  Tau_global_incr_insideTAU();

  ret = _memalign(alignment, size);
  if (ret != NULL) {
    Tau_memorywrap_add_ptr(ret, size);
  }

  Tau_global_decr_insideTAU();
  return ret;
}

#endif /* TAU_VALLOC_AVAILABLE */


/*********************************************************************
 * calloc
 ********************************************************************/
#define TAU_EXTRA_MEM_SIZE 2048
static char tau_calloc_mem[TAU_EXTRA_MEM_SIZE]; 
static int tau_calloc_mem_size = 0;
static int tau_calloc_used = 0;
static int tau_calloc_freed = 0;

void *calloc (size_t nmemb, size_t size) {
   static void* (*_calloc)(size_t nmemb, size_t size) = NULL;

   static int checkinit = 0;
   static int numcalls = 0;
   numcalls++;
   if (checkinit == 0) {
     checkinit = 1;

    Tau_global_incr_insideTAU();

     _calloc = ( void* (*)(size_t nmemb, size_t size)) dlsym(RTLD_NEXT, "calloc");

    Tau_global_decr_insideTAU();
   }


   if (_calloc == NULL && tau_calloc_used == 0  && size < TAU_EXTRA_MEM_SIZE) {
     /* if (size > ) { */
     /*   printf("TAU: Error: Static array exceeds initial allocation request in calloc: size = %d\n", (int) size); */
     /*   exit(1); */
     /* } */
     Tau_global_incr_insideTAU();
     tau_calloc_used = 1;
     memset (tau_calloc_mem, 0, size); 
     tau_calloc_mem_size = nmemb * size;

     /* Tau_memorywrap_add_ptr(tau_calloc_mem, nmemb * size); */
     Tau_global_decr_insideTAU();
     return (void *) tau_calloc_mem;  
   }
   
   if (Tau_memorywrap_checkPassThrough()) {
     return _calloc(nmemb, size);
   }

   Tau_memorywrap_checkInit();


   Tau_global_incr_insideTAU();

   void *ptr = _calloc(nmemb, size);
   
   if (!(numcalls == 3 && size == 1040)) {
     Tau_memorywrap_add_ptr(ptr, nmemb * size);
   }
   Tau_global_decr_insideTAU();
   return ptr;
}


/*********************************************************************
 * realloc
 ********************************************************************/
void *realloc (void *ptr, size_t size) {
  static void* (*_realloc)(void *ptr, size_t size) = NULL;

  if (_realloc == NULL) {
    _realloc = ( void* (*)(void *ptr, size_t size)) dlsym(RTLD_NEXT, "realloc");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _realloc(ptr, size);
  }

  Tau_memorywrap_checkInit();
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
  static int (*_posix_memalign)(void **memptr, size_t alignment, size_t size) = NULL;

  if (_posix_memalign == NULL) {
    _posix_memalign = ( int (*)(void **memptr, size_t alignment, size_t size)) dlsym(RTLD_NEXT, "posix_memalign");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    return _posix_memalign(memptr, alignment, size);
  }

  Tau_memorywrap_checkInit();
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
  static void (*_free)(void *ptr) = NULL;

  if (ptr == tau_calloc_mem) {
    /* Tau_memorywrap_remove_ptr(ptr); */
    return;
  }

  if (_free == NULL) {
    _free = ( void (*)(void *ptr)) dlsym(RTLD_NEXT, "free");
  }

  if (Tau_memorywrap_checkPassThrough()) {
    _free(ptr);
    return;
  }

  Tau_memorywrap_checkInit();
  Tau_global_incr_insideTAU();
  _free(ptr);
  Tau_memorywrap_remove_ptr(ptr);
  Tau_global_decr_insideTAU();
}

/*********************************************************************
 * EOF
 ********************************************************************/
