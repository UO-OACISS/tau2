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
#ifdef __GNUC__
#include <cxxabi.h>
#endif /* __GNUC__ */
using namespace std;


#include <stdio.h>

#include <stdlib.h>
#include <string.h>
// #include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef TAU_OPENMP
#  include <omp.h>
#endif /* TAU_OPENMP */

#ifndef TAU_MAX_SYMBOLS_TO_BE_RESOLVED
#define TAU_MAX_SYMBOLS_TO_BE_RESOLVED 3000
#endif /* TAU_MAX_SYMBOLS_TO_BE_RESOLVED */

#include <Profile/TauBfd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif /* __APPLE__ */


static int gnu_init = 1;       /* is initialization needed? */

static int compInstDisabled[TAU_MAX_THREADS];

static tau_bfd_handle_t bfdUnitHandle = TAU_BFD_NULL_HANDLE;

/*
 *-----------------------------------------------------------------------------
 * Simple hash table to map function addresses to region names/identifier
 *-----------------------------------------------------------------------------
 */

typedef struct HN {
  unsigned long id;            /* hash code (address of function */
  const char* name;   /* associated function name       */
  const char* fname;  /*            file name           */
  int lno;            /*            line number         */
  FunctionInfo *fi;
  int excluded;
  struct HN* next;
} HashNode;

#define HASH_MAX 1021

static HashNode* htab[HASH_MAX];

/*
 * Stores function name `n' under hash code `h'
 */

static HashNode* hash_get(unsigned long h);

// This version is used for when the FI cannot be immediately resolved.
static void hash_put(unsigned long h, const char* n, const char* fn, int lno,
		     int excluded = 0) {
    // already found, do not add.
    if (hash_get(h) != NULL) {
	return;
    }
  long id = h % HASH_MAX;
  HashNode *add = (HashNode*)malloc(sizeof(HashNode));
  add->id = h;
  add->name = n ? (const char*)strdup(n) : n;
  add->fname = fn ? (const char*)strdup(fn) : fn;
  add->lno   = lno;
  add->fi = NULL;
  add->excluded = excluded;
  add->next = htab[id];
  htab[id] = add;
}

static void hash_put(unsigned long h, HashNode *add) {
  long id = h % HASH_MAX;
  add->next = htab[id];
  htab[id] = add;
}

/*
 * Lookup hash code `h'
 * Returns hash table entry if already stored, otherwise NULL
 */

static HashNode* hash_get(unsigned long h) {
  long id = h % HASH_MAX;
  HashNode *curr = htab[id];
  while ( curr ) {
    if ( curr->id == h ) {
      return curr;
    }
    curr = curr->next;
  }
  return NULL;
}


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
    fprintf(stderr,"TAU Warning: Comp_gnu - BFD is not available. Symbols may not be resolved!\n");
    warningIssued = true;
  }
}

void updateHashTable(unsigned long addr,
		     const char *funcname, const char *filename,
		     int lineno) {
  // Simply hash encountered symbol information while excluding
  //    certain symbols.
  if ((strstr(funcname, "Tau_Profile_Wrapper")) ||
      (strcmp(funcname, "__sti__$E") == 0)) {
    /* exclude Tau Profile wrappers */ 
    /* exclude intel compiler static initializer */
    hash_put(addr, funcname, filename, lineno, 1);
  } else {
    hash_put(addr, funcname, filename, lineno);
  }
}

/*
 * Get symbol table either by using BFD or by parsing nm-file
 */
static void get_symtab(void) {
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_REUSE_GLOBALS);
  }
  /*  char *path = Tau_bfd_getExecutablePath(bfdUnitHandle); */
  /* Open the executable path */
  /*  get_symtab_bfd(path, 0); */
  TAU_VERBOSE("Comp_gnu: get_symtab loading executable symbol table\n");
  // Pre-process each of the executable's symbols up to a limit
  Tau_bfd_processBfdExecInfo(bfdUnitHandle,
			     TAU_MAX_SYMBOLS_TO_BE_RESOLVED,     
			     updateHashTable);
}

vector<TauBfdAddrMap> *addressMap = NULL;

static TauBfdAddrMap *getAddressMap(unsigned long addr) {
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_REUSE_GLOBALS);
  }
  // if Tau_bfd_registerUnit has been called, maps have been previously loaded
  addressMap = Tau_bfd_getAddressMaps(bfdUnitHandle);
  for (unsigned int i=0;i<addressMap->size();i++) {
    if (addr >= (*addressMap)[i].start && addr <= (*addressMap)[i].end) {
      return &((*addressMap)[i]);
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
  addressMap = Tau_bfd_getAddressMaps(bfdUnitHandle);

  for (unsigned int i=0;i<addressMap->size();i++) {
    if (addr >= (*addressMap)[i].start && addr <= (*addressMap)[i].end) {
      return &((*addressMap)[i]);
    }
  }

  TAU_VERBOSE("Comp_gnu: getAddressMap - failed to find address [%p] after 2 tries\n", addr);
  // Still not found?  Give up
  return NULL;
}


bool tauGetFilenameAndLineNo(unsigned long addr) {
  bool success = false;
  TAU_VERBOSE("tauGetFilenameAndLineNo: addr = %x, addr=%p\n", addr, addr);
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_REUSE_GLOBALS);
  }
  TauBfdInfo *resolvedInfo = 
    Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr);
  if (resolvedInfo == NULL) {
      // Try again with the executable module.
      resolvedInfo = Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, addr);
  }
  if (resolvedInfo != NULL) {
    success = true;
    tau_symbol_found = true; 
    tau_line_no = resolvedInfo->lineno;
    if (resolvedInfo->funcname != NULL) {
      tau_funcname = strdup(resolvedInfo->funcname);
    } else {
      tau_funcname = NULL;
    }
    if (resolvedInfo->filename != NULL) {
      tau_filename = strdup(resolvedInfo->filename);
    } else {
      tau_filename = NULL;
    }
  } else {
    tau_symbol_found = false;
    tau_line_no = 0;
    tau_funcname = NULL;
    tau_filename = NULL;
  }
  return success;
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
  if (map != NULL) { 

#ifdef TAU_EXE 
    sprintf(cmd, "addr2line -e %s 0x%lx", map->name, addr);
    TAU_VERBOSE("popen %s\n", cmd);
    pipe_fp = popen(cmd, "r");
    fscanf(pipe_fp,"%s", line_info);
    TAU_VERBOSE("cmd = %s, line number = %s\n", cmd, line_info);
    pclose(pipe_fp);
    sprintf(field, "[%s] [%s] [%s]", dem_name, line_info, map->name);
#endif /* TAU_EXE */
    /* The reason the TAU_BFD tag is still here is to allow for alternatives */
#ifdef TAU_BFD
    tauGetFilenameAndLineNo(addr);
    if (tau_symbol_found) 
    {
      if (tau_line_no && dem_name && map && map->name) { 
        TAU_VERBOSE("dem_name=%s\n", dem_name);
        TAU_VERBOSE("tau_filename=%s\n", tau_filename);
        TAU_VERBOSE("tau_line_no=%d\n", tau_line_no);
        TAU_VERBOSE("map->name=%s\n", map->name);
        sprintf(field, "[%s] [%s:%d] [%s]", dem_name, tau_filename, tau_line_no, map->name);
      } else { 
        // Get address from gdb if possible
        sprintf(field, "[%s] [Addr=%p] [%s]", dem_name, 
	  addr+Tau_get_backtrace_off_by_one_correction(), map->name);
      } 
      TAU_VERBOSE("AFTER FIELD: %s\n", field);
    }
    else
    {
      sprintf(field, "[%s] [addr=%p] [%s]", dem_name, 
	addr+Tau_get_backtrace_off_by_one_correction(), map->name);
    }
#else
    issueBfdWarningIfNecessary();
#endif /* TAU_BFD */
  } else {
    sprintf(field, "[%s] [addr=%p]", dem_name, 
	addr+Tau_get_backtrace_off_by_one_correction());
  }
  sprintf(metadata, "BACKTRACE %3d", i-1);
  TAU_METADATA(metadata, field);
  return 0;
}

static HashNode *createHashNode(unsigned long addr) {
  // Pre-condition: hn will ALWAYS be NULL on initial entry.
  //      This function is only called if addr cannot be found in
  //        some HashNode.
  TAU_VERBOSE("createHashNode: addr = [%p]\n", addr);
  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_REUSE_GLOBALS);
  }
  tau_bfd_module_handle_t moduleHandle = 
    Tau_bfd_getModuleHandle(bfdUnitHandle, addr);
  if (moduleHandle != TAU_BFD_NULL_MODULE_HANDLE) {
      TAU_VERBOSE("createHashNode: Trying Module %d for address [%p]\n",
		  moduleHandle, addr);
    int result;
    result = Tau_bfd_processBfdModuleInfo(bfdUnitHandle, moduleHandle,
					  TAU_MAX_SYMBOLS_TO_BE_RESOLVED,
					  updateHashTable);
    if (result == TAU_BFD_SYMTAB_LOAD_FAILED) {
	// try again with the executable.
	TAU_VERBOSE("createHashNode: Failed to load symtab, trying again with executable.\n");
	Tau_bfd_processBfdExecInfo(bfdUnitHandle,
				   TAU_MAX_SYMBOLS_TO_BE_RESOLVED,
				   updateHashTable);
    }
    HashNode *hn = hash_get(addr);
    if (hn != NULL) {
      if (hn->fi == NULL) {
	// remove the path
	const char *filename = hn->fname;
	if (filename) {
	  while (strchr(filename,'/') != NULL) {
	    filename = strchr(filename,'/')+1;
	  }
	} else {
	  filename = "(unknown)";
	  // No name! Attempt to resolve the address
	  TauBfdInfo *info = Tau_bfd_resolveBfdInfo(bfdUnitHandle, addr);
	  // If the general solution fails, we try the executable-only
	  if (info == NULL) {
	      info = Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, addr);
	  }
	  if (info->filename != NULL) {
	    filename = strdup(info->filename);
	  }
	  if ((info->funcname != NULL) && (hn->name == NULL)) {
	    hn->name = strdup(info->funcname);
	  }
	  hn->lno = info->lineno;
	}
	char routine[2048];
	sprintf (routine, "%s [{%s} {%d,0}]", hn->name, filename, hn->lno);
	void *handle=NULL;
	TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
	hn->fi = (FunctionInfo*) handle;
      }
      return hn;
    }
  }
  // Unmapped or subsequently unresolved Address - 
  //    Unknown name for FI based on addr
  char routine[2048];
  sprintf (routine, "addr=<%p>", (void*)addr);
  void *handle=NULL;
  TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
  
  HashNode *add = (HashNode*)malloc(sizeof(HashNode));
  add->id = addr;
  add->name  = "UNKNOWN";
  add->fname = "UNKNOWN";
  add->lno   = -1;
  add->fi = (FunctionInfo*) handle;
  add->excluded = 0;
  hash_put(addr, add);
  return add;
}


static int executionFinished = 0;
void runOnExit() {
  executionFinished = 1;
  Tau_destructor_trigger();
}

#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_enter
#endif /* SICORTEX || TAU_SCOREP */
extern "C" void __cyg_profile_func_enter(void* func, void* callsite) {
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
    gnu_init = 0;

    // initialize array of flags that prevent re-entry
    for (i=0; i<TAU_MAX_THREADS; i++) {
      compInstDisabled[i] = 0;
    }

    Tau_init_initializeTAU();
    Tau_global_incr_insideTAU_tid(tid);
    if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
      bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_REUSE_GLOBALS);
    }
    // If we have registered, we should have a map.
    addressMap = Tau_bfd_getAddressMaps(bfdUnitHandle);
    get_symtab();
    TheUsingCompInst() = 1;
    TAU_PROFILE_SET_NODE(0);
    Tau_global_decr_insideTAU_tid(tid);
  }

  // prevent re-entry of this routine on a per thread basis
  Tau_global_incr_insideTAU_tid(tid);
  if (compInstDisabled[tid]) {
    Tau_global_decr_insideTAU_tid(tid);
    return;
  }
  compInstDisabled[tid] = 1;

  if ((hn = hash_get((unsigned long)funcptr))) {
    if (hn->excluded) {
      // finished in this routine, allow entry
      compInstDisabled[tid] = 0;
      Tau_global_decr_insideTAU_tid(tid);
      return;
    }
    if (hn->fi == NULL) {

      RtsLayer::LockDB(); // lock, then check again
      // *CWL* - why? Because another thread could be creating this now.
      //         Lock-and-check-again is more efficient than 
      //            Lock-first-check-later.
      if ( hn->fi == NULL) {
	// remove the path
	const char *filename = hn->fname;
	if (filename) {
	  while (strchr(filename,'/') != NULL) {
	    filename = strchr(filename,'/')+1;
	  }
	} else {
	  // *CWL* - filename can be NULL for a hashed address when it
	  //    is considered too expensive to resolve ALL symbols in an
	  //    executable or module when creating the hashtable for caching
	  //    purposes. In this case, we fall back to resolving the symbol
	  //    as and when the address is encountered.
	  filename = "(unknown)"; 
	  if (bfdUnitHandle == TAU_BFD_NULL_HANDLE) {
	    bfdUnitHandle = Tau_bfd_registerUnit(TAU_BFD_REUSE_GLOBALS);
	  }
	  TauBfdInfo *info = Tau_bfd_resolveBfdInfo(bfdUnitHandle, 
						    (unsigned long)funcptr);
	  if (info == NULL) {
	      // Try again with executable
	      info = Tau_bfd_resolveBfdExecInfo(bfdUnitHandle, 
						(unsigned long)funcptr);
	  }
	  if (info != NULL) {
	      if (info->filename != NULL) {
		  filename = strdup(info->filename);
	      }
	      if (info->funcname != NULL) {
		  hn->name = strdup(info->funcname);
	      }
	      hn->lno = info->lineno;
	      free(info);
	  }
	  /*
          for(i=0; i<nr_all_syms-1; i++) {
            if (syms && syms[i] && ((void *)( syms[i]->section->vma+syms[i]->value) == funcptr)) { 
              unsigned int linenumber;
              bfd_find_nearest_line(BfdImage, bfd_get_section(syms[i]), syms,
	        syms[i]->value, &filename, &hn->name, &linenumber);
	      hn->lno = linenumber;
	      break;
            }
          }
	  */
	  
	}
	
	char *routine;
        if (filename == NULL) filename=strdup("unknown");
	routine = (char*) malloc (strlen(hn->name)+strlen(filename)+1024);
	sprintf (routine, "%s [{%s} {%d,0}]", hn->name, filename, hn->lno);
	void *handle=NULL;
	TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
	free(routine);
	hn->fi = (FunctionInfo*) handle;
      } 
      RtsLayer::UnLockDB();
    }
    Tau_start_timer(hn->fi,0, tid);
  } else {

    RtsLayer::LockDB(); // lock, then check again
    
    if ( (hn = hash_get((unsigned long)funcptr))) {
      Tau_start_timer(hn->fi, 0, tid);
    } else {
      TAU_VERBOSE("Previously unhashed funcptr [%p]\n",(unsigned long)funcptr);
      HashNode *node = createHashNode((unsigned long)funcptr);
      Tau_start_timer(node->fi, 0, tid);
    }
    
    RtsLayer::UnLockDB();

  }

  if ( gnu_init ) {
    // we register this here at the end so that it is called 
    // before the VT objects are destroyed.  Objects are destroyed and atexit targets are 
    // called in the opposite order in which they are created and registered.

    // Note: This doesn't work work VT with MPI, they re-register their atexit routine
    //       During MPI_Init.
    atexit(runOnExit);
  }

  // finished in this routine, allow entry
  compInstDisabled[tid] = 0;
  Tau_global_decr_insideTAU_tid(tid);
}

extern "C" void _cyg_profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}

extern "C" void __pat_tp_func_entry(const void *ea, const void *ra) {
  __cyg_profile_func_enter((void *)ea, (void *)ra);
  
}

extern "C" void __pat_tp_func_return(const void *ea, const void *ra) {
  __cyg_profile_func_enter((void *)ea, (void *)ra);
}

extern "C" void ___cyg_profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}


#if (defined(TAU_SICORTEX) || defined(TAU_SCOREP))
#pragma weak __cyg_profile_func_exit
#endif /* SICORTEX || TAU_SCOREP */
extern "C" void __cyg_profile_func_exit(void* func, void* callsite) {
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

  if ( (hn = hash_get((unsigned long)funcptr)) ) {
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

extern "C" void _cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

extern "C" void ___cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

#endif /* TAU_XLC */
