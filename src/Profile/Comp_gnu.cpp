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

/*****************************************************************************
 **			TAU Portable Profiling Package			    **
 **			http://www.cs.uoregon.edu/research/tau	            **
 *****************************************************************************
 **    Copyright 2008  						   	    **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/*****************************************************************************
 **	File 		: Comp_gnu.cpp  				    **
 **	Description 	: TAU Profiling Package				    **
 **	Contact		: tau-bugs@cs.uoregon.edu               	    **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau        **
 **                                                                         **
 **      Description     : This file contains the hooks for GNU based       **
 **                        compiler instrumentation                         **
 **                                                                         **
 *****************************************************************************/

#ifndef TAU_XLC

#include <TAU.h>
#include <Profile/TauInit.h>
#include <Profile/TauBfd.h>

#include <vector>
#include <mutex>

#include <tau_internal.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
// #include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#ifdef TAU_OPENMP
#  include <omp.h>
#endif /* TAU_OPENMP */

#include <Profile/TauBfd.h>
#include <Profile/TauInit.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <dlfcn.h>
#ifdef TAU_HAVE_CORESYMBOLICATION
#include "CoreSymbolication.h"
#endif
#endif /* __APPLE__ */

using namespace std;

/*
 *-----------------------------------------------------------------------------
 * Simple hash table to map function addresses to region names/identifier
 *-----------------------------------------------------------------------------
 */

struct HashNode
{
  HashNode() : fi(NULL), excluded(false)
  { }

  TauBfdInfo info;		///< Filename, line number, etc.
  FunctionInfo * fi;		///< Function profile information
  bool excluded;			///< Is function excluded from profiling?
};

struct HashTable : public TAU_HASH_MAP<unsigned long, HashNode*>
{
  HashTable() {
    Tau_init_initializeTAU();
  }
  virtual ~HashTable() {
    Tau_destructor_trigger();
  }
};

static std::mutex & theMutex() {
  static std::mutex mtx;
  return mtx;
}

static HashTable & TheHashTable()
{
  static HashTable htab;
  return htab;
}

static TAU_HASH_MAP<unsigned long, HashNode*>& TheLocalHashTable(){
  static thread_local TAU_HASH_MAP<unsigned long, HashNode*> lhtab;
  return lhtab;
}

static tau_bfd_handle_t & TheBfdUnitHandle()
{
  static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    RtsLayer::LockEnv();
    if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
      bfdUnitHandle = Tau_bfd_registerUnit();
    }
    RtsLayer::UnLockEnv();
  }
  return bfdUnitHandle;
}

/*
 * Get symbol table by using BFD
 */
static void issueBfdWarningIfNecessary()
{
#ifndef TAU_BFD
  static bool warningIssued = false;
  if (!warningIssued) {
#ifndef __APPLE__
    fprintf(stderr,"TAU Warning: Comp_gnu - "
        "BFD is not available during TAU build. Symbols may not be resolved!\n");
    fflush(stderr);
#endif
    warningIssued = true;
  }
#endif
}

bool isExcluded(char const * funcname)
{
  return funcname && (
      // Intel compiler static initializer
      (strcmp(funcname, "__sti__$E") == 0)
      // Tau Profile wrappers
      || strstr(funcname, "Tau_Profile_Wrapper"));
}

void updateHashTable(unsigned long addr, const char *funcname)
{
  HashNode * hn = TheLocalHashTable()[addr];
  if (!hn) {
    std::lock_guard<std::mutex> lck (theMutex());
    hn = TheHashTable()[addr];
    if (!hn) {
      hn = new HashNode;
      TheHashTable()[addr] = hn;
    }
    TheLocalHashTable()[addr] = hn;
  }
  hn->info.funcname = funcname;
  hn->excluded = isExcluded(funcname);
}

extern "C" void Tau_profile_exit_all_threads(void);

static int executionFinished = 0;
void runOnExit()
{
  executionFinished = 1;
  Tau_profile_exit_all_threads();

  // clear the hash map to eliminate memory leaks
  HashTable & mytab = TheHashTable();
  for ( TAU_HASH_MAP<unsigned long, HashNode*>::iterator it = mytab.begin(); it != mytab.end(); ++it ) {
  	HashNode * node = it->second;
    if (node != NULL && node->fi) {
#ifndef TAU_TBB_SUPPORT
// At the end of a TBB program, it crashes here.
		//delete node->fi;
#endif /* TAU_TBB_SUPPORT */
	}
    delete node;
  }
  mytab.clear();

#ifdef TAU_BFD
  Tau_delete_bfd_units();
#endif
  Tau_destructor_trigger();
}

//
// Instrumentation callback functions
//
extern "C"
{

// Prevent accidental instrumentation of the instrumentation functions
// It's highly unlikely because you'd have to compile TAU with
// -finstrument-functions, but better safe than sorry.

#ifndef MERCURIUM_EXTRA
__attribute__((no_instrument_function))
void __cyg_profile_func_enter(void*, void*);

__attribute__((no_instrument_function))
void _cyg_profile_func_enter(void*, void*);

__attribute__((no_instrument_function))
void __pat_tp_func_entry(const void *, const void *);

__attribute__((no_instrument_function))
void ___cyg_profile_func_enter(void*, void*);

__attribute__((no_instrument_function))
void __cyg_profile_func_exit(void*, void*);

__attribute__((no_instrument_function))
void _cyg_profile_func_exit(void*, void*);

__attribute__((no_instrument_function))
void ___cyg_profile_func_exit(void*, void*);

__attribute__((no_instrument_function))
void __pat_tp_func_return(const void *ea, const void *ra);

__attribute__((no_instrument_function))
void profile_func_enter(void*, void*);

__attribute__((no_instrument_function))
void profile_func_exit(void*, void*);
#endif


#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_enter
#endif /* SICORTEX || TAU_SCOREP */


void __cyg_profile_func_enter(void* func, void* callsite)
{
  static bool gnu_init = true;
  HashNode * node;
  /* This is the entry point into TAU from PDT-instrumented C++ codes, so
   * make sure that TAU is ready to go before doing anything else! */

  // Don't profile if we're done executing or still initializing
  if (executionFinished || Tau_init_initializingTAU() || Tau_get_inside_initialize()) return;

  static int do_this_once = Tau_init_initializeTAU();

  // Don't profile TAU internals. This also prevents reentrancy.
  if (Tau_global_get_insideTAU() > 0) return;


  // Convert void * to integer
  void * funcptr = func;
#ifdef __ia64__
  funcptr = *( void ** )func;
#endif
  unsigned long addr = Tau_convert_ptr_to_unsigned_long(funcptr);

  // Quickly get the hash node and discover if this is an excluded function.
  // Sampling and the memory wrapper require us to protect this region,
  // but otherwise we don't pay that overhead. (Sampling because it can
  // interrupt the application anywhere and memory because the hash table
  // lookup allocates memory).
  {
    TauInternalFunctionGuard protects_this_region(
        TauEnv_get_ebs_enabled() || Tau_memory_wrapper_is_registered());

    // Get the hash node

    node = TheLocalHashTable()[addr];
    if (!node) {
      // We must be inside TAU before we lock the database
      TauInternalFunctionGuard protects_this_region;

      std::lock_guard<std::mutex> lck (theMutex());
      node = TheHashTable()[addr];
      if (!node) {
        node = new HashNode;
        // This is a work around for an elusive bug observed at LLNL.
        // Sometimes the new node was not initialized when -optShared was used so
        // TAU_PROFILER_CREATE would not get called and TAU was crashing inside a
        // fi->GetProfileGroup() call because fi was not a valid address.
        // We explicitly initialize the node to work around this.
        node->fi = NULL;
        node->excluded = false;
        TheHashTable()[addr] = node;

      }
      TheLocalHashTable()[addr] =  node;
    }
    // Skip excluded functions
    if (node->excluded) return;
  } // END protected region


  // Construct and start the function timer.  This region needs to be protected
  // in all situations.
  {
    TauInternalFunctionGuard protects_this_region;

    // Get BFD handle
    tau_bfd_handle_t & bfdUnitHandle = TheBfdUnitHandle();

    if (gnu_init) {
      gnu_init = false;

      Tau_init_initializeTAU();

      issueBfdWarningIfNecessary();

      // Create hashtable entries for all symbols in the executable
      // via a fast scan of the executable's symbol table.
      // It makes sense to load the entire symbol table because all
      // symbols in the executable are likely to be encountered
      // during the run
      Tau_bfd_processBfdExecInfo(bfdUnitHandle, updateHashTable);

      TheUsingCompInst() = 1;

      // For UPC: Initialize the node
      if (RtsLayer::myNode() == -1) {
        TAU_PROFILE_SET_NODE(0);
      }

      // we register this here at the end so that it is called
      // before the VT objects are destroyed.  Objects are destroyed and atexit targets are
      // called in the opposite order in which they are created and registered.
      // Note: This doesn't work work VT with MPI, they re-register their atexit routine
      //       During MPI_Init.
      atexit(runOnExit);
    }

    // Start the timer if it's not an excluded function
    if (!node->fi) {
      std::lock_guard<std::mutex> lck (theMutex());
      if (!node->fi) {
        // Resolve function info if it hasn't already been retrieved
        if (!node->info.probeAddr) {
#if defined(__APPLE__)
#if defined(TAU_HAVE_CORESYMBOLICATION)
         static CSSymbolicatorRef symbolicator = CSSymbolicatorCreateWithPid(getpid());
         CSSourceInfoRef source_info = CSSymbolicatorGetSourceInfoWithAddressAtTime(symbolicator, (vm_address_t)addr, kCSNow);
         if(!CSIsNull(source_info)) {
             CSSymbolRef symbol = CSSourceInfoGetSymbol(source_info);
             node->info.probeAddr = addr;
             node->info.filename = strdup(CSSourceInfoGetPath(source_info));
             node->info.funcname = strdup(CSSymbolGetName(symbol));
             node->info.lineno = CSSourceInfoGetLineNumber(source_info);
         }
#else
          Dl_info info;
          int rc = dladdr((const void *)addr, &info);
          if (rc != 0) {
            node->info.probeAddr = addr;
            node->info.filename = strdup(info.dli_fname);
            node->info.funcname = strdup(info.dli_sname);
            node->info.lineno = 0; // Apple doesn't give us line numbers.
          }
#endif
#else
          Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, node->info);
#endif
        }

        //Do not profile this routine, causes crashes with the intel compilers.
        node->excluded = isExcluded(node->info.funcname);

	// TBB: sometimes we get a null filename and function name. In that case
	// exclude that routine.
	if(!node->info.filename || !node->info.funcname) {
		node->excluded = 1;
		return;
	}

        // Build routine name for TAU function info
        unsigned int size = strlen(node->info.funcname) + strlen(node->info.filename) + 128;
        char * routine = (char*)malloc(size);
        if (TauEnv_get_bfd_lookup()) {
          char *dem_name = Tau_demangle_name(node->info.funcname);
          //sprintf(routine, "%s [{%s} {%d,0}]", node->info.funcname, node->info.filename, node->info.lineno);
#ifdef DEBUG_PROF
          printf("name = %s, dem_name = %s\n", node->info.funcname, dem_name);
#endif /* DEBUG_PROF */
          snprintf(routine, size,  "%s [{%s} {%d,0}]", dem_name, node->info.filename, node->info.lineno);
          free(dem_name);
        } else {
          snprintf(routine, size,  "[%s] UNRESOLVED %s ADDR %lx", node->info.funcname, node->info.filename, addr);
        }

        // Create function info
        void * handle = NULL;
        TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
        node->fi = (FunctionInfo*)handle;

        // Cleanup
        free((void*)routine);
      }
    }

    if (!node->excluded) {
      //GNU has some internal routines that occur before main in entered. To
      //ensure that a single top-level timer is present start the dummy '.TAU
      //application' timer. -SB
      Tau_create_top_level_timer_if_necessary();
      Tau_start_timer(node->fi, 0, RtsLayer::myThread());
#ifdef TAU_UNWIND
        if(TauEnv_get_region_addresses()) {
          node->fi->StartAddr = addr;
        }
#endif
    }

    if (!(node->fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
      //printf("COMP_GNU >>>>>>>>>> Excluding: %s, addr: %d, throttled.\n", node->fi->GetName(), addr);
      node->excluded = true;
    }
  } // END protected region
}

void _cyg_profile_func_enter(void* func, void* callsite)
{
  __cyg_profile_func_enter(func, callsite);
}

void __pat_tp_func_entry(const void *ea, const void *ra)
{
  __cyg_profile_func_enter((void *)ea, (void *)ra);

}

void profile_func_enter(void* func, void* callsite)
{
  __cyg_profile_func_enter(func, callsite);
}

void ___cyg_profile_func_enter(void* func, void* callsite)
{
  __cyg_profile_func_enter(func, callsite);
}

#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_exit
#endif /* SICORTEX || TAU_SCOREP */
void __cyg_profile_func_exit(void* func, void* callsite)
{
  // These checks must be done before anything else.

  // Don't profile if we're done executing.
  if (executionFinished) return;

  // Don't profile if we're still initializing.
  if (Tau_init_initializingTAU()) return;

  // Don't profile if we're done initializing but have yet to return from the init function
  if (Tau_get_inside_initialize()) return;

  // Don't profile TAU internals. This also prevents reentrancy.
  if (Tau_global_get_insideTAU() > 0) return;

  HashNode * hn;
  unsigned long addr;

  // Quickly get the hash node and discover if this is an excluded function.
  // Sampling and the memory wrapper require us to protect this region,
  // but otherwise we don't pay that overhead. (Sampling because it can
  // interrupt the application anywhere and memory because the hash table
  // lookup allocates memory).
  {
    TauInternalFunctionGuard protects_this_region(
        TauEnv_get_ebs_enabled() || Tau_memory_wrapper_is_registered());

    void * funcptr = func;
#ifdef __ia64__
    funcptr = *( void ** )func;
#endif
    addr = Tau_convert_ptr_to_unsigned_long(funcptr);

    // Get the hash node
    hn = TheLocalHashTable()[addr];
    if(!hn){
        std::lock_guard<std::mutex> lck (theMutex());
        hn = TheHashTable()[addr];
    }
    // Skip excluded functions or functions we didn't enter
    if (!hn || hn->excluded || !hn->fi) return;
  } // END protected region



  // Stop the timer.  This routine is protected so we don't need another guard.
  Tau_stop_timer(hn->fi, RtsLayer::myThread());
#ifdef TAU_UNWIND
  if(TauEnv_get_region_addresses()) {
    hn->fi->StopAddr = addr;
  }
#endif
}


void _cyg_profile_func_exit(void* func, void* callsite)
{
  __cyg_profile_func_exit(func, callsite);
}

void ___cyg_profile_func_exit(void* func, void* callsite)
{
  __cyg_profile_func_exit(func, callsite);
}

void profile_func_exit(void* func, void* callsite)
{
  __cyg_profile_func_exit(func, callsite);
}

void __pat_tp_func_return(const void *ea, const void *ra)
{
  __cyg_profile_func_exit((void *)ea, (void *)ra);
}

#ifdef TAU_FX_AARCH64
int __cxa_thread_atexit(void (*func)(), void *obj,
                                   void *dso_symbol) {
  int __cxa_thread_atexit_impl(void (*)(), void *, void *);
  return __cxa_thread_atexit_impl(func, obj, dso_symbol);
}
#endif /* TAU_FX_AARCH64 */
}    // extern "C"

#endif /* TAU_XLC */
