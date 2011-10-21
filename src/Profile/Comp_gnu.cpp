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
	HashNode() :
		addr(0), name(NULL), fname(NULL),
		lno(0), fi(NULL), excluded(false)
	{ }

	HashNode(size_t _addr, char const * _name, char const * _fname,
			 size_t _lno, FunctionInfo * _fi, bool _excluded) :
		addr(_addr), name(_name), fname(_fname),
		lno(_lno), fi(_fi), excluded(_excluded)
	{ }

	unsigned long addr;		///< Function address
	char const * name;		///< Function name
	char const * fname;		///< Filename
	unsigned int lno;		///< Line number
	FunctionInfo * fi;		///< Function profile information
	bool excluded;			///< Is function excluded from profiling?
};

class AddressNameHash
{

public:

	typedef std::map<unsigned long, HashNode*> map_type;

	HashNode * put(unsigned long addr, char const * n, char const * fn,
			int lno, bool excluded = false)
	{
		// Don't add if already found
		map_type::iterator it = map.find(addr);
		if(it != map.end()) return it->second;
		return (map[addr] = new HashNode(addr, n, fn, lno, NULL, excluded));
	}

	HashNode * get(unsigned long addr) const
	{
		map_type::const_iterator it = map.find(addr);
		if(it != map.end()) return it->second;
		return NULL;
	}

	HashNode * get(void * addr) const
	{
		return get(Tau_convert_ptr_to_unsigned_long(addr));
	}

	size_t size() const {
		return map.size();
	}

private:

	map_type map;

};

static AddressNameHash htab;

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

HashNode * updateHashTable(unsigned long addr, const char *funcname,
		const char *filename, int lineno)
{
	// Hash symbol information while excluding:
	bool exclude = (
			// Intel compiler static initializer
			(strcmp(funcname, "__sti__$E") == 0) ||
			// Tau Profile wrappers
			strstr(funcname, "Tau_Profile_Wrapper") ||
			// GNU libstdc++ symbols
			(strstr(funcname, "__gnu_cxx") == funcname) ||
			// MinGW symbols
			(strstr(funcname, "__mingw_") == funcname) ||
			(strstr(funcname, "__w64-mingw") == funcname) ||
			(strstr(funcname, "___w64-mingw") == funcname)
			);

	return htab.put(addr, funcname, filename, lineno, exclude);
}

/*
 * Get symbol table via BFD
 */
static void get_symtab(void)
{
	if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
		bfdUnitHandle = Tau_bfd_registerUnit();
	}

	TAU_VERBOSE("Comp_gnu: get_symtab loading executable symbol table\n");

	// Fast scan of the executable's symbol table
	Tau_bfd_processBfdExecInfo(bfdUnitHandle, updateHashTable);
}

static TauBfdAddrMap * getAddressMap(unsigned long addr)
{
	if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
		bfdUnitHandle = Tau_bfd_registerUnit();
	}

	// if Tau_bfd_registerUnit has been called, maps have been previously loaded
	vector<TauBfdAddrMap*> & addressMap = Tau_bfd_getAddressMaps(bfdUnitHandle);
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

	if (!tau_symbol_found) {
		// Try again with the executable module.
		tau_symbol_found = Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, addr, info);
	}

	if (tau_symbol_found) {
		tau_line_no = info.lno;
		if (info.funcname) {
			tau_funcname = strdup(info.funcname);
		} else {
			tau_funcname = NULL;
		}
		if (info.filename) {
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

static FunctionInfo * createFunctionInfo(HashNode * hn, unsigned long addr)
{
	// Don't do work when we don't have to
	if(hn->fi) return hn->fi;

	if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
		bfdUnitHandle = Tau_bfd_registerUnit();
	}

	RtsLayer::LockDB(); // lock, then check again
	// *CWL* - why? Because another thread could be creating this now.
	//         Lock-and-check-again is more efficient than Lock-first-check-later.
	if (hn->fi == NULL) {

		char const * filename = hn->fname;

		// Resolve function info if it hasn't already been retrieved
		if (!filename) {
			filename = "(unknown)";

			TauBfdInfo info;
			bool resolved = Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, addr, info);

			if (resolved) {
				if (info.filename) {
					filename = info.filename;
				}
				if (info.funcname) {
					hn->name = info.funcname;
				}
				hn->lno = info.lno;
			}
		}

		char routine[strlen(hn->name) + strlen(filename) + 128];
		sprintf(routine, "%s [{%s} {%d,0}]", hn->name, filename, hn->lno);

		void *handle = NULL;
		TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
		hn->fi = (FunctionInfo*)handle;
	}
	RtsLayer::UnLockDB();

	return hn->fi;
}

static HashNode *createHashNode(unsigned long addr)
{
	// Note: It is assumed that no hash node for addr exists

//	TAU_VERBOSE("createHashNode: addr = [%p]\n", addr);
	if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
		bfdUnitHandle = Tau_bfd_registerUnit();
	}

	// Get info from BFD about this address
	TauBfdInfo info;
	bool resolved = Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr, info);

	HashNode * hn = NULL;
	if (resolved) {
		hn = updateHashTable(addr, info.funcname, info.filename, info.lno);

		RtsLayer::LockDB(); // lock, then check again
		// *CWL* - why? Because another thread could be creating this now.
		//         Lock-and-check-again is more efficient than Lock-first-check-later.
		if (hn->fi == NULL) {
			char routine[strlen(hn->name) + strlen(hn->fname) + 128];
			sprintf(routine, "%s [{%s} {%d,0}]", hn->name, hn->fname, hn->lno);
			// Create function info for the new node
			void *handle = NULL;
			TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
			hn->fi = (FunctionInfo*)handle;
		}
		RtsLayer::UnLockDB();
	} else {
		hn = htab.put(addr, "(unknown)", "(unknown)", 0);

		RtsLayer::LockDB(); // lock, then check again
		// *CWL* - why? Because another thread could be creating this now.
		//         Lock-and-check-again is more efficient than Lock-first-check-later.
		if (hn->fi == NULL) {
			// Couldn't map the address to the function name.
			// Give up and make a node named by address.
			char routine[2048];
			sprintf(routine, "addr=<%p>", (void*)(uintptr_t)addr);
			// Create function info for the new node
			void * handle = NULL;
			TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
			hn->fi = (FunctionInfo*)handle;
		}
		RtsLayer::UnLockDB();
	}
	return hn;

#if 0
	// Load the module symbol table
	tau_bfd_module_handle_t moduleHandle =
			Tau_bfd_getModuleHandle(bfdUnitHandle, (uintptr_t)addr);
	if (moduleHandle != TAU_BFD_NULL_MODULE_HANDLE) {
		int result = Tau_bfd_processBfdModuleInfo(bfdUnitHandle,
				moduleHandle, updateHashTable);
		if (result == TAU_BFD_SYMTAB_LOAD_FAILED) {
			TAU_VERBOSE("createHashNode: Failed to load symtab for Module %d.");
			// Note: don't fall back to executable because we scanned the exe
			// in initialization. Repeating the scan won't find anything new.
		}
	}

	// Get the newly created hash node
	HashNode * hn = htab.get(addr);

	if(hn) {
		// Create function info if needed
		createFunctionInfo(hn, addr);
	} else {
		// Couldn't map the address to the function name.
		// Give up and make a node named by address.
		char routine[2048];
		sprintf(routine, "addr=<%p>", (void*)(uintptr_t)addr);
		hn = htab.put(addr, "(unknown)", "(unknown)", 0);

		// Create function info for the new node
		void * handle = NULL;
		TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
		hn->fi = (FunctionInfo*)handle;
	}

	return hn;
#endif
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
	int i;
	int tid;

#ifndef TAU_BFD
	issueBfdWarningIfNecessary();
#endif /* TAU_BFD */

	if (executionFinished) {
		return;
	}

	HashNode *hn;
	void * funcptr = func;
#ifdef __ia64__
	funcptr = *( void ** )func;
#endif

	tid = Tau_get_tid();

	if (gnu_init) {
		gnu_init = false;

		// initialize array of flags that prevent re-entry
		for (i = 0; i < TAU_MAX_THREADS; i++) {
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
		get_symtab();

		TheUsingCompInst() = 1;
		TAU_PROFILE_SET_NODE(0);
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

	// Check for unhashed address.
	// Since the executable symbol table has already been processed,
	// the address is either resolvable in another module, or unresolvable.
	if ((hn = htab.get(funcptr)) == NULL) {
//		TAU_VERBOSE("Previously unhashed funcptr [%p]\n",
//					Tau_convert_ptr_to_unsigned_long(funcptr));
		hn = createHashNode(Tau_convert_ptr_to_unsigned_long(funcptr));
	}

	// Start the timer if it's not an excluded function
	if (!hn->excluded) {
		createFunctionInfo(hn, Tau_convert_ptr_to_unsigned_long(funcptr));
		Tau_start_timer(hn->fi, 0, tid);
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
void __cyg_profile_func_exit(void* func, void* callsite) {
  int tid;

#ifndef TAU_BFD
  issueBfdWarningIfNecessary();
#endif /* TAU_BFD */

  tid = Tau_get_tid();
  Tau_global_incr_insideTAU_tid(tid);

  // prevent entry into cyg_profile functions while inside entry
  tid = Tau_get_tid();
  if (compInstDisabled[tid]) {
    return;
  }

  if (executionFinished) {
    return;
  }
  HashNode *hn;
  void * funcptr = func;
#ifdef __ia64__
  funcptr = *( void ** )func;
#endif

  if ( (hn = htab.get(funcptr)) ) {
    if (hn->excluded) {
      Tau_global_decr_insideTAU_tid(tid);
      return;
    }

    Tau_stop_timer(hn->fi, tid);
  } else {
    //printf ("NOT FOUND! : ");
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
