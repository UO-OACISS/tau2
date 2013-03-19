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

#include <vector>
#include <set>
using namespace std;
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
#endif /* __APPLE__ */


static int compInstDisabled[TAU_MAX_THREADS];

static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

/*
 *-----------------------------------------------------------------------------
 * Simple hash table to map function addresses to region names/identifier
 *-----------------------------------------------------------------------------
 */

struct HashNode
{
	HashNode() : fi(NULL), excluded(false)
	{ }

	HashNode(const HashNode &h) : fi(NULL), excluded(false)
	{ }

	TauBfdInfo info;		///< Filename, line number, etc.
	FunctionInfo * fi;		///< Function profile information
	bool excluded;			///< Is function excluded from profiling?
};

typedef TAU_HASH_MAP<unsigned long, HashNode> HashTable;

typedef std::set<unsigned long> ExcludeList;

HashTable& TheHashTable()
{
	static HashTable htab;
	return htab;
}

/*
 * Get symbol table by using BFD
 */
#ifndef TAU_BFD
static void issueBfdWarningIfNecessary() {
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr,"TAU Warning: Comp_gnu - "
    		"BFD is not available during TAU build. Symbols may not be resolved!\n");
    warningIssued = true;
  }
}
#endif

void updateHashTable(unsigned long addr, const char *funcname)
{
	HashNode & hn = TheHashTable()[addr];
	hn.info.funcname = funcname;
	hn.excluded = funcname && (
			// Intel compiler static initializer
			(strcmp(funcname, "__sti__$E") == 0)
			// Tau Profile wrappers
			|| strstr(funcname, "Tau_Profile_Wrapper")
			);
}


static int executionFinished = 0;
void runOnExit() {
	executionFinished = 1;
	Tau_destructor_trigger();
}

//
// Instrumentation callback functions
//
extern "C" {

// Prevent accidental instrumentation of the instrumentation functions
// It's highly unlikely because you'd have to compile TAU with
// -finstrument-functions, but better safe than sorry.

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

#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_enter
#endif /* SICORTEX || TAU_SCOREP */
void __cyg_profile_func_enter(void* func, void* callsite)
{
  static bool gnu_init = true;

#ifndef TAU_BFD
	issueBfdWarningIfNecessary();
#endif /* TAU_BFD */

	void * funcptr = func;
#ifdef __ia64__
	funcptr = *( void ** )func;
#endif

  if (executionFinished) return;

	unsigned long addr = Tau_convert_ptr_to_unsigned_long(funcptr);

  // Get previously hashed info, or efficiently create
  // a new hash node if it didn't already exist
	HashNode & hn = TheHashTable()[addr];

	// Skip excluded functions
	if (hn.excluded) return;

  if (Tau_global_get_insideTAU() > 0) return;
  TauInternalFunctionGuard protects_this_function;

	//prevent entry into cyg_profile functions while still initializing TAU
	if (Tau_init_initializingTAU()) return;

	int tid = Tau_get_tid();

	if (gnu_init) {
		gnu_init = false;

		// initialize array of flags that prevent re-entry
		for (int i = 0; i < TAU_MAX_THREADS; i++) {
			compInstDisabled[i] = 0;
		}

		Tau_init_initializeTAU();

		if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
			bfdUnitHandle = Tau_bfd_registerUnit();
		}

		// Create hashtable entries for all symbols in the executable
		// via a fast scan of the executable's symbol table.
		// It makes sense to load the entire symbol table because all
		// symbols in the executable are likely to be encountered
		// during the run
		Tau_bfd_processBfdExecInfo(bfdUnitHandle, updateHashTable);

		TheUsingCompInst() = 1;
		// *CWL* - CompGnu's interactions with UPC originally blew away UPC's
		//         settings. Unfortunately, it cannot also be removed. The
		//         compromise is to check that the value is -1 (uninitialized)
		//         and set it to 0 if so.
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
	
	// prevent re-entry of this routine on a per thread basis
	if (compInstDisabled[tid]) {
		return;
	}
	compInstDisabled[tid] = 1;

	// Start the timer if it's not an excluded function
	if(hn.fi == NULL) {
		RtsLayer::LockDB(); // lock, then check again
		// *CWL* - why? Because another thread could be creating this now.
		//         Lock-and-check-again is more efficient than Lock-first-check-later.
		if (hn.fi == NULL) {
			// Resolve function info if it hasn't already been retrieved
			if(hn.info.probeAddr == 0) {
				Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, hn.info);
			}

      //We can remove these functions from the profile as well if we choose.
      //strncmp(hn.info.funcname, "std::", 5)  == 0||
      //strncmp(hn.info.funcname, "__gnu_cxx::", 11) == 0)
      //
      //Do not profile this routine, causes crashes with the intel compilers.
      if (strcmp(hn.info.funcname, "__sti__$E") == 0)
      {
        hn.excluded = true;
      }

			// Tau_bfd_resolveBfdInfo should have made all fields non-NULL,
			// but we're going to be extra safe in case something changes
			if(hn.info.funcname == NULL) {
				TAU_VERBOSE("Unexpected NULL pointer!\n");
				hn.info.funcname = "(unknown)";
			}
			if(hn.info.filename == NULL) {
				TAU_VERBOSE("Unexpected NULL pointer!\n");
				hn.info.filename = "(unknown)";
			}

			// Build routine name for TAU function info
			unsigned int size = strlen(hn.info.funcname) +
					strlen(hn.info.filename) + 128;
			char * routine = (char*)malloc(size);
			sprintf(routine, "%s [{%s} {%d,0}]", hn.info.funcname,
					hn.info.filename, hn.info.lineno);

			// Create function info
			void *handle = NULL;
			TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
			hn.fi = (FunctionInfo*)handle;

			// Cleanup
			free((void*)routine);
		}
		RtsLayer::UnLockDB();
	}
  if (hn.excluded == false)
  {
		//GNU has some internal routines that occur before main in entered. To
		//ensure that a single top-level timer is present start the dummy '.TAU
		//application' timer. -SB
		Tau_create_top_level_timer_if_necessary();
	  Tau_start_timer(hn.fi, 0, tid);
	}	
	if (!(hn.fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
		//printf("COMP_GNU >>>>>>>>>> Excluding: %s, addr: %d, throttled.\n", hn.fi->GetName(), addr);
		hn.excluded = true;
	}

	// finished in this routine, allow entry
	compInstDisabled[tid] = 0;
}

void _cyg_profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}

void __pat_tp_func_entry(const void *ea, const void *ra) {
  __cyg_profile_func_enter((void *)ea, (void *)ra);
  
}

void profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}

void ___cyg_profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}


#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_exit
#endif /* SICORTEX || TAU_SCOREP */
void __cyg_profile_func_exit(void* func, void* callsite)
{
#ifndef TAU_BFD
	issueBfdWarningIfNecessary();
#endif /* TAU_BFD */

	void * funcptr = func;
#ifdef __ia64__
	funcptr = *( void ** )func;
#endif
	unsigned long addr = Tau_convert_ptr_to_unsigned_long(funcptr);

	HashNode & hn = TheHashTable()[addr];
	if (!hn.excluded) {
    TauInternalFunctionGuard protects_this_function;

	  int tid = Tau_get_tid();

    // prevent entry into cyg_profile functions while inside entry
	  //prevent entry into cyg_profile functions while still initializing TAU
    if (compInstDisabled[tid] || executionFinished || Tau_init_initializingTAU()) {
      return;
    }

    if (hn.fi) {
      Tau_stop_timer(hn.fi, tid);
    }
	}
}

void _cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

void ___cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

void profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

void __pat_tp_func_return(const void *ea, const void *ra) {
  __cyg_profile_func_exit((void *)ea, (void *)ra);
}

} // extern "C"

#endif /* TAU_XLC */
