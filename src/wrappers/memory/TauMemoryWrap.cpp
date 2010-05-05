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

#include <Profile/TauInit.h>
    
#define dprintf TAU_VERBOSE

#include <vector>
using namespace std;





/*********************************************************************
 * set of global data
 ********************************************************************/
static int lightsOut = 0;
class MemoryWrapGlobal {
public:
  int bytesAllocated;
  map<void*,long> pointerMap;
  void *heapMemoryUserEvent;

  MemoryWrapGlobal() {
    lightsOut = 0;
    bytesAllocated = 0;
    heapMemoryUserEvent = 0;
    Tau_get_context_userevent(&heapMemoryUserEvent, "Heap Memory Allocated");
  }
  ~MemoryWrapGlobal() {
    lightsOut = 1;
  }
};


/*********************************************************************
 * access to global data
 ********************************************************************/
static MemoryWrapGlobal& global() {
  static MemoryWrapGlobal memoryWrapGlobal;
  return memoryWrapGlobal;
}



/*********************************************************************
 * return whether we should pass through and not track the IO
 ********************************************************************/
static int Tau_iowrap_checkPassThrough() {
  if (Tau_global_get_insideTAU() > 0 || lightsOut) {
    return 1;
  } else {
    return 0;
  }
}


/*********************************************************************
 * initializer
 ********************************************************************/
void Tau_memorywrap_checkInit() {
  static int init = 0;
  if (init == 1) {
    return;
  }
  init = 1;

  Tau_global_incr_insideTAU();
  Tau_init_initializeTAU();
  Tau_create_top_level_timer_if_necessary();
  Tau_global_decr_insideTAU();
}


void Tau_memorywrap_add_ptr (void *ptr, size_t size) {
  if (ptr != NULL) {
    global().pointerMap[ptr] = size;
    global().bytesAllocated += size;
  }
}

void Tau_memorywrap_remove_ptr (void *ptr) {
  if (ptr != NULL) {
    map<void*,long>::iterator it = global().pointerMap.find(ptr);
    if (it != global().pointerMap.end()) {
      global().bytesAllocated -= global().pointerMap[ptr];
      global().pointerMap.erase(ptr);
    }
  }
}


/*********************************************************************
 * malloc
 ********************************************************************/
void *malloc (size_t size) {
  Tau_memorywrap_checkInit();
  static void* (*_malloc)(size_t size) = NULL;

  if (_malloc == NULL) {
    _malloc = ( void* (*)(size_t size)) dlsym(RTLD_NEXT, "malloc");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _malloc(size);
  }

  Tau_global_incr_insideTAU();
  TAU_PROFILE_TIMER(t, "malloc()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  void *ptr = _malloc(size);
  Tau_memorywrap_add_ptr(ptr, size);
  TAU_CONTEXT_EVENT(global().heapMemoryUserEvent, global().bytesAllocated);
  TAU_PROFILE_STOP(t); 
  Tau_global_decr_insideTAU();
  return ptr;
}

/*********************************************************************
 * calloc
 ********************************************************************/
void *calloc (size_t nmemb, size_t size) {
  Tau_memorywrap_checkInit();
  static void* (*_calloc)(size_t nmemb, size_t size) = NULL;

  if (_calloc == NULL) {
    _calloc = ( void* (*)(size_t nmemb, size_t size)) dlsym(RTLD_NEXT, "calloc");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _calloc(nmemb, size);
  }

  Tau_global_incr_insideTAU();
  TAU_PROFILE_TIMER(t, "calloc()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  void *ptr = _calloc(nmemb, size);
  Tau_memorywrap_add_ptr(ptr, nmemb * size);
  TAU_CONTEXT_EVENT(global().heapMemoryUserEvent, global().bytesAllocated);
  TAU_PROFILE_STOP(t); 
  Tau_global_decr_insideTAU();
  return ptr;
}


/*********************************************************************
 * realloc
 ********************************************************************/
void *realloc (void *ptr, size_t size) {
  Tau_memorywrap_checkInit();
  static void* (*_realloc)(void *ptr, size_t size) = NULL;

  if (_realloc == NULL) {
    _realloc = ( void* (*)(void *ptr, size_t size)) dlsym(RTLD_NEXT, "realloc");
  }

  if (Tau_iowrap_checkPassThrough()) {
    return _realloc(ptr, size);
  }

  Tau_global_incr_insideTAU();
  TAU_PROFILE_TIMER(t, "realloc()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  void *ret_ptr = _realloc(ptr, size);

  Tau_memorywrap_remove_ptr(ptr);
  Tau_memorywrap_add_ptr(ret_ptr, size);

  TAU_CONTEXT_EVENT(global().heapMemoryUserEvent, global().bytesAllocated);
  TAU_PROFILE_STOP(t); 
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

  if (Tau_iowrap_checkPassThrough()) {
    return _posix_memalign(memptr, alignment, size);
  }

  Tau_global_incr_insideTAU();
  TAU_PROFILE_TIMER(t, "posix_memalign()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  int ret = _posix_memalign(memptr, alignment, size);
  if (ret == 0) {
    Tau_memorywrap_add_ptr(*memptr, size);
  }

  TAU_CONTEXT_EVENT(global().heapMemoryUserEvent, global().bytesAllocated);
  TAU_PROFILE_STOP(t); 
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

  if (Tau_iowrap_checkPassThrough()) {
    _free(ptr);
    return;
  }

  Tau_global_incr_insideTAU();
  TAU_PROFILE_TIMER(t, "free()", " ", TAU_IO);
  TAU_PROFILE_START(t);

  _free(ptr);

  Tau_memorywrap_remove_ptr(ptr);

  TAU_CONTEXT_EVENT(global().heapMemoryUserEvent, global().bytesAllocated);
  TAU_PROFILE_STOP(t); 
  Tau_global_decr_insideTAU();
}





/*********************************************************************
 * EOF
 ********************************************************************/
