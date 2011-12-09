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
#include <map>
#ifdef __GNUC__
#include <cxxabi.h>
#endif /* __GNUC__ */
using namespace std;


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

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif /* __APPLE__ */


/* Initialization flag */
static bool gnu_init = true;

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

	TauBfdInfo info;		///< Filename, line number, etc.
	FunctionInfo * fi;		///< Function profile information
	bool excluded;			///< Is function excluded from profiling?
};

typedef std::map<unsigned long, HashNode> HashTable;
HashTable htab;

/*
 * Get symbol table by using BFD
 */

static const char *tau_filename;
static const char *tau_funcname;
static unsigned int tau_line_no;
static int tau_symbol_found; 

extern "C" int Tau_get_backtrace_off_by_one_correction(void);

static void issueBfdWarningIfNecessary() {
  static bool warningIssued = false;
  if (!warningIssued) {
    fprintf(stderr,"TAU Warning: Comp_gnu - "
    		"BFD is not available during TAU build. Symbols may not be resolved!\n");
    warningIssued = true;
  }
}

void updateHashTable(unsigned long addr, const char *funcname)
{
	HashNode & hn = htab[addr];
	hn.info.funcname = funcname;
	hn.excluded = funcname && (
			// Intel compiler static initializer
			(strcmp(funcname, "__sti__$E") == 0)
			// Tau Profile wrappers
			|| strstr(funcname, "Tau_Profile_Wrapper")
			);
}


static TauBfdAddrMap * getAddressMap(unsigned long addr)
{
	if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
		bfdUnitHandle = Tau_bfd_registerUnit();
	}

	// if Tau_bfd_registerUnit has been called, maps have been loaded
	vector<TauBfdAddrMap*> const & addressMap =
			Tau_bfd_getAddressMaps(bfdUnitHandle);
	for (size_t i = 0; i < addressMap.size(); i++) {
		if (addr >= addressMap[i]->start && addr <= addressMap[i]->end) {
			return addressMap[i];
		}
	}

	// Wasn't found in any ranges, try updating the maps.
	// NOTE: *CWL* - This simplified means of detecting epoch changes will
	//       suffer from pathological cases where a function's address in
	//       one dynamically loaded module can coincide with another
	//       function's address in another dynamically loaded module.
	//
	//       Sampling CANNOT take this approach to epoch changes. It must
	//       rely on traps to dlopen calls.

	Tau_bfd_updateAddressMaps(bfdUnitHandle);

	for (size_t i = 0; i < addressMap.size(); i++) {
		if (addr >= addressMap[i]->start && addr <= addressMap[i]->end) {
			return addressMap[i];
		}
	}

	TAU_VERBOSE("Comp_gnu: getAddressMap - "
			"failed to find address [%p] after 2 tries\n", addr);
	// Still not found?  Give up
	return NULL;
}


bool tauGetFilenameAndLineNo(unsigned long addr)
{
	TAU_VERBOSE("tauGetFilenameAndLineNo: addr=%p\n", addr);

	if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
		bfdUnitHandle = Tau_bfd_registerUnit();
	}

	// Use BFD to resolve address info
	TauBfdInfo info;
	tau_symbol_found = Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, info);

	if (tau_symbol_found) {
		tau_line_no = info.lineno;
		if (info.funcname) {
			// TODO: Is this leaking memory?
			tau_funcname = strdup(info.funcname);
		} else {
			tau_funcname = NULL;
		}
		if (info.filename) {
			// TODO: Is this leaking memory?
			tau_filename = strdup(info.filename);
		} else {
			tau_filename = NULL;
		}
	} else {
		tau_line_no = 0;
		tau_funcname = NULL;
		tau_filename = NULL;
	}
	return tau_symbol_found;
}

int tauPrintAddr(int i, char *token, unsigned long addr) {
  static int flag = 0;
  if (flag == 0) {
    flag = 1;
  }
  char field[2048];
  char metadata[256];
  char *dem_name = NULL;
  char demangled_name[2048], line_info[2048];
  char cmd[2048];
  FILE *pipe_fp;
  TauBfdAddrMap *map = getAddressMap(addr);
  line_info[0]=0;

  /* Do we have a demangled name? */
  if (dem_name == (char *) NULL)  {
    char *subtoken=token;
    int i = 0;
    while (*subtoken!= '(' && i  < strlen(token)) {
      subtoken++; i++;
    }
    subtoken--; /* move the pointer to before the ( so we can use strtok */
    TAU_VERBOSE("Subtoken=%s\n", subtoken);
    char *subs=strtok(subtoken,"(+");
    subs = strtok(NULL,"+");
    if (subs == (char *) NULL) subs = token;
#ifndef __GNUC__
    sprintf(cmd, "c++filt %s", subs);
    TAU_VERBOSE("popen %s\n", cmd);
    pipe_fp = popen(cmd, "r");
    //fscanf(pipe_fp,"%s", demangled_name);
    int ret = fread(demangled_name, 1, 1024, pipe_fp);
    TAU_VERBOSE("name = %s, Demangled name = %s, ret = %d\n", token, demangled_name, ret);
    pclose(pipe_fp);
    dem_name = demangled_name;
#else /* __GNUC__ */
    std::size_t len=1024;
    int stat;
    char *out_buf= (char *) malloc (len);
    char *name = abi::__cxa_demangle(subs, out_buf, &len, &stat);
    if (stat == 0) dem_name = out_buf;
    else dem_name = subs;
    TAU_VERBOSE("DEM_NAME subs= %s dem_name= %s, name = %s, len = %d, stat=%d\n", subs, dem_name, name, len, stat);
#endif /* __GNUC__ */

  }
  if (dem_name == (char *) NULL) dem_name = token;
  TAU_VERBOSE("tauPrintAddr: final demangled name [%s]\n", dem_name);

#ifdef TAU_EXE
  if (map != NULL) {
    sprintf(cmd, "addr2line -e %s 0x%lx", map->name, addr);
    TAU_VERBOSE("popen %s\n", cmd);
    pipe_fp = popen(cmd, "r");
    fscanf(pipe_fp,"%s", line_info);
    TAU_VERBOSE("cmd = %s, line number = %s\n", cmd, line_info);
    pclose(pipe_fp);
    sprintf(field, "[%s] [%s] [%s]", dem_name, line_info, map->name);
  }
#endif /* TAU_EXE */
  /* The reason the TAU_BFD tag is still here is to allow for alternatives */
#ifdef TAU_BFD
  tauGetFilenameAndLineNo(addr);
  if (tau_symbol_found) {
    TAU_VERBOSE("tauPrintAddr: Symbol found for [addr=%p]\n", addr);
    sprintf(field, "[%s] [%s:%d] [%s]", tau_funcname, tau_filename, tau_line_no, map->name);
  } else {
    TAU_VERBOSE("tauPrintAddr: Symbol for [addr=%p] not found\n", addr);
    if (dem_name != NULL && map != NULL) {
      // Get address from gdb if possible
      TAU_VERBOSE("tauPrintAddr: Getting information from GDB instead\n");
      sprintf(field, "[%s] [Addr=%p] [%s]", dem_name,
	      addr+Tau_get_backtrace_off_by_one_correction(), map->name);
    } else {
      TAU_VERBOSE("tauPrintAddr: No Information Available\n");
      sprintf(field, "[%s] [addr=%p]", dem_name,
	      addr+Tau_get_backtrace_off_by_one_correction());
    }
  }
#else
  issueBfdWarningIfNecessary();
#endif /* TAU_BFD */
  sprintf(metadata, "BACKTRACE %3d", i-1);
  TAU_METADATA(metadata, field);
  return 0;
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


#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_enter
#endif /* SICORTEX || TAU_SCOREP */
void __cyg_profile_func_enter(void* func, void* callsite)
{
#ifndef TAU_BFD
	issueBfdWarningIfNecessary();
#endif /* TAU_BFD */

	if (executionFinished) {
		return;
	}

	void * funcptr = func;
#ifdef __ia64__
	funcptr = *( void ** )func;
#endif
	unsigned long addr = Tau_convert_ptr_to_unsigned_long(funcptr);

	int tid = Tau_get_tid();

	if (gnu_init) {
		gnu_init = false;

		// initialize array of flags that prevent re-entry
		for (int i = 0; i < TAU_MAX_THREADS; i++) {
			compInstDisabled[i] = 0;
		}

		Tau_init_initializeTAU();
		Tau_global_incr_insideTAU_tid(tid);

		//GNU has some internal routines that occur before main in entered. To
		//ensure that a single top-level timer is present start the dummy '.TAU
		//application' timer. -SB
		Tau_create_top_level_timer_if_necessary();
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
		//TAU_PROFILE_SET_NODE(0);
		Tau_global_decr_insideTAU_tid(tid);

		// we register this here at the end so that it is called
		// before the VT objects are destroyed.  Objects are destroyed and atexit targets are
		// called in the opposite order in which they are created and registered.

		// Note: This doesn't work work VT with MPI, they re-register their atexit routine
		//       During MPI_Init.
		atexit(runOnExit);
	}

	// prevent re-entry of this routine on a per thread basis
	Tau_global_incr_insideTAU_tid(tid);
	if (compInstDisabled[tid]) {
		Tau_global_decr_insideTAU_tid(tid);
		return;
	}
	compInstDisabled[tid] = 1;

	// Get previously hashed info, or efficiently create
	// a new hash node if it didn't already exist
	HashNode & hn = htab[addr];

	// Start the timer if it's not an excluded function
	if (!hn.excluded) {
		if(hn.fi == NULL) {
			RtsLayer::LockDB(); // lock, then check again
			// *CWL* - why? Because another thread could be creating this now.
			//         Lock-and-check-again is more efficient than Lock-first-check-later.
			if (hn.fi == NULL) {
				// Resolve function info if it hasn't already been retrieved
				if(hn.info.probeAddr == 0) {
					Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, hn.info);
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
		Tau_start_timer(hn.fi, 0, tid);
	}

	// finished in this routine, allow entry
	compInstDisabled[tid] = 0;
	Tau_global_decr_insideTAU_tid(tid);
}

void _cyg_profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}

void __pat_tp_func_entry(const void *ea, const void *ra) {
  __cyg_profile_func_enter((void *)ea, (void *)ra);
  
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

	int tid = Tau_get_tid();
	Tau_global_incr_insideTAU_tid(tid);

	// prevent entry into cyg_profile functions while inside entry
	if (compInstDisabled[tid]) {
		return;
	}

	if (executionFinished) {
		return;
	}

	void * funcptr = func;
#ifdef __ia64__
	funcptr = *( void ** )func;
#endif

	HashNode & hn = htab[Tau_convert_ptr_to_unsigned_long(funcptr)];
	if (!hn.excluded && hn.fi) {
		Tau_stop_timer(hn.fi, tid);
	}
	Tau_global_decr_insideTAU_tid(tid);
}

void _cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

void ___cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

void __pat_tp_func_return(const void *ea, const void *ra) {
  __cyg_profile_func_exit((void *)ea, (void *)ra);
}

} // extern "C"

#endif /* TAU_XLC */
