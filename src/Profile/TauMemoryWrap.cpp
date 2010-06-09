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
**      Description     : memory wrapper                                   **
**                                                                         **
****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
  
#include <TAU.h>
#include <Profile/Profiler.h>
#include <Profile/TauInit.h>


/*********************************************************************
 * This object represents a memory allocation, it consists of 
 * a location (TAU context) and a number of bytes
 ********************************************************************/
class MemoryAllocation {
public:
  size_t numBytes;
  string location;
  MemoryAllocation() {
    numBytes = 0;
    location = "";
  }
  MemoryAllocation(size_t nBytes, string loc) : numBytes(nBytes), location(loc) {
  }
  MemoryAllocation(size_t nBytes) : numBytes(nBytes), location("") {
  }
};


/*********************************************************************
 * set of global data
 ********************************************************************/
class MemoryWrapGlobal {
public:
  x_uint64 bytesAllocated;
  TAU_HASH_MAP<void*,MemoryAllocation> pointerMap;
  void *heapMemoryUserEvent;
  void *mallocUserEvent;
  void *freeUserEvent;
  int wrapperActive;

  MemoryWrapGlobal() {
    bytesAllocated = 0;
    heapMemoryUserEvent = 0;
    mallocUserEvent = 0;
    freeUserEvent = 0;

    if (getenv("TAU_MEMORY_WRAPPER_ENABLED")) {
      wrapperActive = 1;
    } else {
      wrapperActive = 0;
    }

    Tau_get_context_userevent(&heapMemoryUserEvent, "Heap Memory Allocated");
    Tau_get_context_userevent(&mallocUserEvent, "malloc size (bytes)");
    Tau_get_context_userevent(&freeUserEvent, "free size (bytes)");
  }
  ~MemoryWrapGlobal() {
    Tau_destructor_trigger();
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
 * return the number of bytes currently tracked
 ********************************************************************/
extern "C" x_uint64 Tau_memorywrap_getBytesAllocated() {
  return global().bytesAllocated;
}

/*********************************************************************
 * return whether the wrapper has been activated or not
 ********************************************************************/
extern "C" int Tau_memorywrap_getWrapperActive() {
  return global().wrapperActive;
}

/*********************************************************************
 * return whether we should pass through and not track the IO
 ********************************************************************/
extern "C" int Tau_memorywrap_checkPassThrough() {
  if (Tau_global_get_insideTAU() > 0 || Tau_global_getLightsOut() || Tau_init_check_initialized() == 0) {
    return 1;
  } else {
    return 0;
  }
}


/*********************************************************************
 * hook registered to be called at profile write time, we trigger the leaks here
 ********************************************************************/
void Tau_memorywrap_writeHook() {
  if (!TauEnv_get_track_memory_leaks()) {
    return;
  }
  RtsLayer::LockDB();
  
  map<string, TauUserEvent*> userEventMap; // map location to user event

  TAU_HASH_MAP<void*,MemoryAllocation>::const_iterator it;
  for (it=global().pointerMap.begin(); it != global().pointerMap.end(); ++it) { // iterate over still-allocated objects
    
    map<string, TauUserEvent*>::const_iterator search = userEventMap.find(it->second.location);
    if (search == userEventMap.end()) { // not found, create a user event for it
      string s (string("MEMORY LEAK! : ")+it->second.location);
      TauUserEvent *leakEvent = new TauUserEvent(s.c_str());
      userEventMap[it->second.location] = leakEvent;
    }

    // trigger the event
    userEventMap[it->second.location]->TriggerEvent(it->second.numBytes);

    //fprintf (stderr, "[%p] leak of %d bytes, allocated at %s\n", it->first, it->second.numBytes, it->second.location.c_str());
  }
  RtsLayer::UnLockDB();
}

/*********************************************************************
 * initializer
 ********************************************************************/
extern "C" void Tau_memorywrap_checkInit() {
  static int init = 0;
  if (init == 1) {
    return;
  }
  init = 1;

  Tau_global_incr_insideTAU();
  Tau_init_initializeTAU();
  Tau_create_top_level_timer_if_necessary();
  // register write hook
  Tau_global_addWriteHook(Tau_memorywrap_writeHook);
  Tau_global_decr_insideTAU();
}

/*********************************************************************
 * generate context string
 ********************************************************************/
static string Tau_memorywrap_getContextString() {
  int tid = RtsLayer::myThread();
  Profiler *current = TauInternal_CurrentProfiler(tid);
  Profiler *p = current;
  int depth = TauEnv_get_callpath_depth();
  string delimiter(" => ");
  string name("");

  while (current != NULL && depth != 0) {
    if (current != p) {
      name = current->ThisFunction->GetName() + string(" ") +
	current->ThisFunction->GetType() + delimiter + name;
    } else {
      name = current->ThisFunction->GetName() + string (" ") +
	current->ThisFunction->GetType();
    }
    current = current->ParentProfiler;
    depth--;
  }
  return name;
}

/*********************************************************************
 * add a pointer to the collection
 ********************************************************************/
extern "C" void Tau_memorywrap_add_ptr (void *ptr, size_t size) {
  global().wrapperActive = 1;
  if (ptr != NULL) {
    RtsLayer::LockDB();
    if (TauEnv_get_track_memory_leaks()) {
      global().pointerMap[ptr] = MemoryAllocation(size, Tau_memorywrap_getContextString());
    } else {
      global().pointerMap[ptr] = MemoryAllocation(size);
    }
    global().bytesAllocated += size;

    TAU_CONTEXT_EVENT(global().mallocUserEvent, size);
    TAU_TRACK_MEMORY_HERE();
    RtsLayer::UnLockDB();
  }
}

/*********************************************************************
 * remove a pointer from the collection
 ********************************************************************/
extern "C" void Tau_memorywrap_remove_ptr (void *ptr) {
  global().wrapperActive = 1;
  if (ptr != NULL) {
    RtsLayer::LockDB();
    TAU_HASH_MAP<void*,MemoryAllocation>::const_iterator it = global().pointerMap.find(ptr);
    if (it != global().pointerMap.end()) {
      int size = global().pointerMap[ptr].numBytes;
      global().bytesAllocated -= size;
      global().pointerMap.erase(ptr);
      TAU_CONTEXT_EVENT(global().freeUserEvent, size);
      TAU_TRACK_MEMORY_HERE();
    }
    RtsLayer::UnLockDB();
  }
}

