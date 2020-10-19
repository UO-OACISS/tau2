/**
 * VampirTrace
 * http://www.tu-dresden.de/zih/vampirtrace
 *
 * Copyright (c) 2005-2008, ZIH, TU Dresden, Federal Republic of Germany
 *
 * Copyright (c) 1998-2005, Forschungszentrum Juelich GmbH, Federal
 * Republic of Germany
 *
 * See the file COPYRIGHT in the package base directory for details
 **/

/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 2008  						   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/****************************************************************************
 **	File 		: Comp_xl.cpp    				   **
 **	Description 	: TAU Profiling Package				   **
 **	Contact		: tau-bugs@cs.uoregon.edu               	   **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
 **                                                                         **
 **      Description     : This file contains the hooks for IBM based       **
 **                        compiler instrumentation                         **
 **                                                                         **
 ****************************************************************************/

//
// Notes:
// Compiler-based instrumentation for IBM XL compilers is not robust.
// __func_trace_enter and __func_trace_exit have no way to uniquely
// identify the routine that is being entered or exited.  The routine
// name, filename, and line number are passed, but they can be NULL
// or nonprintable strings if optimization is used (especially -O3).
// Nor are the char pointers for name and fname unique to the routine: 
// different routines in different files may have the same pointer 
// values (perhaps because the name and file name are somehow 
// determined dynamically?).  Even with optimization disabled, the 
// filename string value may be different for matching enter and 
// exit calls.  This is especially true for static initializers,
// constructors, and destructors in C++.
//
// According to the XL documentation, a void ** const will also be passed
// that _is_ unique to the calling routine.  Sadly, this is not the case
// on (at least) BlueGene/P systems.  For void ** const user_data, 
// user_data == 0x80808080 for many calls, suggesting that the parameter 
// is not being passed.  This is true for any optimization level.
//
// To make this work, we must read the actual values of the routine
// name and filename string (guarding against NULL or invalid strings)
// and calculate a hash to identify the routine.  This is expensive,
// and it must be done on each and every enter or exit.
//
// Application performance is going to suffer badly, especially
// for C++ codes with lots of templates.  Use std::vector<> in any
// simple example and you'll quickly see what I mean.
//
#include <TAU.h>
#include <Profile/TauInit.h>

#include <stdint.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>

#define MAX_STRING_LEN 1024
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )

static bool finished = false;

struct HashNode
{
  HashNode() : fi(NULL)
  { }

  FunctionInfo * fi;
};

// Function information is keyed to a hash composed from
// the function name and file name.
// This is slow, but without any other unique identifier
// we don't have much choice.
// Here are some of the reasons we don't have a fast unique id:
//  - void ** user_info: isn't passed on BGP, 
//                       may be NULL or nonsense under -O3
//  - char * name: may be the same for multiple calls, 
//                 may be NULL or nonsense under -O3
//  - char * fname: may be the same for multiple calls, 
//                  may be NULL or nonsense under -O3
typedef uint32_t key_type;
struct HashTable : public TAU_HASH_MAP<key_type, HashNode*>
{
  HashTable() {
    Tau_init_initializeTAU();
  }
  virtual ~HashTable() {
    Tau_destructor_trigger();
  }
};

// These static functions cause the initializer to be called
// before we start working with the data structure

#if 0
// Prefered source of FunctionInfo pointers
static HashTable & get_name_file_hashtable()
{
  static HashTable htab;
  return htab;
}

// Secondary source of FunctionInfo pointers.
// Used when primary lookup fails 
// (e.g. different fnames, but it really IS the same function)
static HashTable & get_name_only_hashtable()
{
  static HashTable htab;
  return htab;
}
#endif

// Incremental string hashing function.
// Uses Paul Hsieh's SuperFastHash, the same as in Google Chrome.
uint32_t get_hash(uint32_t hash, char const * data, int len)
{
  uint32_t tmp;
  int rem;

  rem = len & 3;
  len >>= 2;

  for (; len > 0; len--) {
    hash += get16bits (data);
    tmp = (get16bits (data+2) << 11) ^ hash;
    hash = (hash << 16) ^ tmp;
    data += 2 * sizeof(uint16_t);
    hash += hash >> 11;
  }

  switch (rem) {
  case 3:
    hash += get16bits (data);
    hash ^= hash << 16;
    hash ^= ((signed char)data[sizeof(uint16_t)]) << 18;
    hash += hash >> 11;
    break;
  case 2:
    hash += get16bits (data);
    hash ^= hash << 11;
    hash += hash >> 17;
    break;
  case 1:
    hash += (signed char)*data;
    hash ^= hash << 10;
    hash += hash >> 1;
    break;
  }

  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;

  return hash;
}

// String processing function for the routine name.
// Checks for excluded names, calculates string length, and
// finally calculates string hash if needed.
uint32_t get_name_hash(uint32_t hash, char ** pdata, size_t * plen, bool * pexclude)
{
  char const * data = *pdata;
  int len;
  bool exclude = false;

  if (data) {
    for (len = 0; len < MAX_STRING_LEN; ++len) {
      char c = data[len];
      if (c == 0) {
        break;
      } else if (c == '@' || c == '$') {
        // Exclude IBM OpenMP runtime functions
        //exclude = true;
      } else if (c < 32 || c > 126) {
        exclude = false;
        data = "(optimized out)";
        len = 15;
        break;
      }
    }
  } else {
    data = "(optimized out)";
    len = 15;
  }
  *pdata = (char*)data;
  *plen = len;
  *pexclude = exclude;

  if (!exclude) {
    return get_hash(hash, data, len);
  } else {
    return 0;
  }
}

// String processing function for the file name.
// Checks for excluded files, calculates string length, and
// finally calculates string hash if needed.
uint32_t get_filename_hash(uint32_t hash, char ** pdata, size_t * plen, bool * pexclude)
{
  char const * data = *pdata;
  int len;
  bool exclude = false;

  if (data) {
    for (len = 0; len < MAX_STRING_LEN; ++len) {
      char c = data[len];
      if (c == 0) {
        break;
      } else if (c < 32 || c > 126) {
        exclude = false;
        data = "(optimized out)";
        len = 15;
        break;
      }
    }
  } else {
    data = "(optimized out)";
    len = 15;
  }
  *pdata = (char*)data;
  *plen = len;
  *pexclude = exclude;

  if (!exclude) {
    return get_hash(hash, data, len);
  } else {
    return 0;
  }
}

extern "C" void Tau_profile_exit_all_threads(void);

// Cleanup function
extern "C" void runOnExit()
{
  finished = true;
  Tau_profile_exit_all_threads();
  Tau_destructor_trigger();
}

extern "C" void __func_trace_enter(char * name, char * fname, int lno, void ** const user_data)
{
  static bool need_init = true;
  //printf("Enter: %s [{%s} {%d,0}]\n", name, fname, lno);

  // Don't profile if we're done executing.
  if (finished) return;

  // Don't profile if we're still initializing.
  if (Tau_init_initializingTAU()) return;

  // Don't profile TAU internals
  if (Tau_global_get_insideTAU() > 0) return;

  // Protect TAU from itself.  This MUST occur here before we query the TID or
  // use the hash table.  Any later and TAU's memory wrapper will profile TAU
  // and crash or deadlock.
  // Note that this also prevents reentrency into this routine.
  {
    TauInternalFunctionGuard protects_this_function;

    if (need_init) {
      need_init = false;

      // Initialize TAU
      Tau_init_initializeTAU();
      Tau_create_top_level_timer_if_necessary();
      TheUsingCompInst() = 1;
      if (Tau_get_node() == -1) {
        TAU_PROFILE_SET_NODE(0);
      }

      // Register callback
      //atexit(runOnExit);

      TAU_VERBOSE("XL compiler-based instrumentation initialized\n");
    }

#if 0
    // Guard against re-entry (does this actually happen?)
    int tid = Tau_get_thread();

    // Build the hashtable keys while checking for exclusion
    key_type name_key, key;
    size_t nlen, flen;
    bool excluded = false;
    name_key = get_name_hash(0, &name, &nlen, &excluded);
    if (excluded) {
      return;
    }
    key = get_filename_hash(name_key, &fname, &flen, &excluded);
    if (excluded) {
      return;
    }

    // Get the function info
    HashNode * node = get_name_file_hashtable()[key];
    if (!node) {
      RtsLayer::LockDB();
      node = get_name_file_hashtable()[key];
      if (!node) {
        node = new HashNode;
        get_name_file_hashtable()[key] = node;
      }
      RtsLayer::UnLockDB();
    }
    HashNode & hn = *node;

    // Create new function info if it doesn't already exist
    if (!hn.fi) {
      RtsLayer::LockDB();
      // Check again: another thread may have created fi while we were locking
      if (!hn.fi) {
        // Build the routine name
        size_t size = nlen + flen + 32;
        char * buff = (char*)malloc(size);
        snprintf(buff, size, "%s [{%s} {%d,0}]", name, fname, lno);

        //TAU_VERBOSE("Routine %d: %s\n", get_name_file_hashtable().size(), buff);

        // Create function info
        void * handle = NULL;
        TAU_PROFILER_CREATE(handle, buff, "", TAU_DEFAULT);
        hn.fi = (FunctionInfo*)handle;

        // Also track this function by name since sometimes
        // XL compilers incorrectly report different file names
        // on function entry/exit
        HashNode * node = get_name_only_hashtable()[name_key];
        if (!node) {
          RtsLayer::LockDB();
          node = get_name_only_hashtable()[name_key];
          if (!node) {
            node = new HashNode;
            get_name_only_hashtable()[name_key] = node;
          }
          RtsLayer::UnLockDB();
        }
        HashNode & name_hn = *node;

        if (!name_hn.fi) {
          // Note: collision is still possible, but we're out
          // of options at this point so just cross your fingers
          name_hn.fi = hn.fi;
        }

        // Cleanup
        free((void*)buff);
      }
      RtsLayer::UnLockDB();
    }

    // Start the timer
    //TAU_VERBOSE("Starting: %s\n", fi->GetName());
    Tau_start_timer(hn.fi, 0, tid);
#else
  // Build the routine name
    size_t nlen, flen;
    bool excluded = false;
    key_type name_key, key;
    name_key = get_name_hash(0, &name, &nlen, &excluded);
    key = get_filename_hash(name_key, &fname, &flen, &excluded);
    if (key == 0) {
      TAU_VERBOSE("Warning: Filename hash is zero: %s\n", fname);
    }
    size_t size = nlen + flen + 32;
    char * buff = (char*)malloc(size);
    snprintf(buff, size, "%s [{%s} {%d,0}]", name, fname, lno);
    Tau_pure_start(buff);
#endif
  }    // END inside TAU
}

extern "C" void __func_trace_exit(char * name, char * fname, int lno, void ** const user_data)
{
  //printf("Exit: %s [{%s} {%d,0}]\n", name, fname, lno);

  // Don't profile if we're done executing.
  if (finished) return;

  // Don't profile if we're still initializing.
  if (Tau_init_initializingTAU()) return;

  // Don't profile TAU internals
  if (Tau_global_get_insideTAU() > 0) return;

  // Protect TAU from itself.  This MUST occur here before we query the TID or
  // use the hash table.  Any later and TAU's memory wrapper will profile TAU
  // and crash or deadlock.
  // Note that this also prevents reentrency into this routine.
  {
    TauInternalFunctionGuard protects_this_function;

#if 0
    int tid = Tau_get_thread();

    // Build the hashtable key while checking for exclusion
    key_type name_key, key;
    size_t nlen, flen;
    bool excluded = false;
    name_key = get_name_hash(0, &name, &nlen, &excluded);
    if (excluded) {
      return;
    }
    key = get_filename_hash(name_key, &fname, &flen, &excluded);
    if (excluded) {
      return;
    }

    // Get the function info and stop the timer
    HashNode * hn = get_name_file_hashtable()[key];
    if (hn && hn->fi) {
      //TAU_VERBOSE("Stopping: %s\n", fi->GetName());
      Tau_stop_timer(hn->fi, tid);
    } else {
      // Uh oh, XL compiler has given us a __func_trace_exit that doesn't
      // match a known __func_trace_enter.  This can happen when fname is
      // (incorectly) different on enter and exit. Sadly, this actually happens.
      // Fall back to name-only resolution.
      HashNode * name_hn = get_name_only_hashtable()[name_key];
      if (name_hn && name_hn->fi) {
        //TAU_VERBOSE("Warning: name-only lookup on %s [{%s} {%d,0}]\n", name, fname, lno);
        Tau_stop_timer(name_hn->fi, tid);
      } else {
        // If execution reaches this point, you'll probably segfault before much longer.
        TAU_VERBOSE("Warning: unmached __func_trace_exit: %s [{%s} {%d,0}]\n", name, fname, lno);
      }
    }
#else
    Tau_stop_current_timer();
  } // END inside TAU
#endif
}

extern "C" void __func_trace_catch(char * name, char * fname, int lno, void ** const user_data)
{
  // Catch ignored for now
}

