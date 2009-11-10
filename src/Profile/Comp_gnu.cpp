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
using namespace std;


#include <stdio.h>
#ifdef TAU_BFD
#  include "bfd.h"
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include "demangle.h"
#  endif /* HAVE_GNU_DEMANGLE */
#endif /* TAU_BFD */


#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef TAU_OPENMP
#  include <omp.h>
#endif


#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif


static int gnu_init = 1;       /* is initialization needed? */

static int compInstDisabled[TAU_MAX_THREADS];

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

static void hash_put(unsigned long h, const char* n, const char* fn, int lno) {
//   printf ("put with %s\n", n);
  long id = h % HASH_MAX;
  HashNode *add = (HashNode*)malloc(sizeof(HashNode));
  add->id = h;
  add->name = n ? (const char*)strdup(n) : n;
  add->fname = fn ? (const char*)strdup(fn) : fn;
  add->lno   = lno;
  add->fi = NULL;
  add->excluded = 0;
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

static void get_symtab_bfd(const char *module, unsigned long offset) {
#ifdef TAU_BFD
  bfd * BfdImage = 0;
  int nr_all_syms;
  int i; 
  size_t size;
  asymbol **syms;
  int do_getsrc = 1;
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
  int do_demangle = 1;
#endif /* HAVE_GNU_DEMANGLE */

  /* initialize BFD */
  bfd_init();

  /* get executable image */
  BfdImage = bfd_openr(module, 0 );
  if ( ! BfdImage ) {
    fprintf (stderr,"TAU: BFD: bfd_openr(%s): failed\n", module);
    return;
  }

  /* check image format */
  if ( ! bfd_check_format(BfdImage, bfd_object) ) { 
    fprintf(stderr,"TAU: BFD: bfd_check_format(%s): failed\n", module);
    return;
  }
  /* return if file has no symbols at all */
  if ( ! ( bfd_get_file_flags(BfdImage) & HAS_SYMS ) ) {
    fprintf(stderr,"TAU: BFD: bfd_get_file_flags(%s): no symbols found\n", module);
    return;
  }
   
  /* get the upper bound number of symbols */
  size = bfd_get_symtab_upper_bound(BfdImage);
   
  /* HAS_SYMS can be set even with no symbols in the file! */
  if ( size < 1 ) {
    fprintf(stderr,"TAU: BFD: bfd_get_symtab_upper_bound(): < 1\n");
  }
   
  /* read canonicalized symbols */
  syms = (asymbol **)malloc(size);
  nr_all_syms = bfd_canonicalize_symtab(BfdImage, syms);
  if ( nr_all_syms < 1 ) {
    fprintf(stderr,"TAU: BFD: No symbols found in '%s' (did you compile with -g?) : bfd_canonicalize_symtab(): < 1\n", module);
    return;
  }
   
  for (i=0; i<nr_all_syms; ++i) {
    char* dem_name = 0;
    unsigned long addr;
    const char* filename;
    const char* funcname;
    unsigned int lno;
      
    //       /* ignore system functions */
    //       if ( strncmp(syms[i]->name, "__", 2) == 0 ||
    // 	   strncmp(syms[i]->name, "bfd_", 4) == 0 ||
    // 	   strstr(syms[i]->name, "@@") != NULL ) continue;


    /* get filename and linenumber from debug info */
    /* needs -g */
    filename = NULL;
    lno = 0;
    if ( do_getsrc ) {
      bfd_find_nearest_line(BfdImage, bfd_get_section(syms[i]), syms,
			    syms[i]->value, &filename, &funcname, &lno);
    }

    /* calculate function address */
    addr = syms[i]->section->vma+syms[i]->value;

    /* use demangled name if possible */
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
    if ( do_demangle ) {
      dem_name = cplus_demangle(syms[i]->name,
				DMGL_PARAMS | DMGL_ANSI 
				| DMGL_VERBOSE | DMGL_TYPES);
    }
#endif /* HAVE_GNU_DEMANGLE */


    const char *name = syms[i]->name;
    if (dem_name) {
      name = dem_name;
    }
    hash_put(offset+addr, name, filename, lno);

    if (strstr(name, "Tau_Profile_Wrapper")) {
      HashNode *hn = hash_get(offset+addr);
      if (hn) {
	hn->excluded = 1;
      }
    } else if (strcmp(name, "__sti__$E") == 0) {
      /* exclude intel compiler static initializer */
      HashNode *hn = hash_get(offset+addr);
      if (hn) {
	hn->excluded = 1;
      }
    }
  }

  free(syms);
  bfd_close(BfdImage);
#endif
  return;
}


int getBGPJobID(const char *path, char *name) {
  DIR *pdir = NULL;
  pdir = opendir(path);
  if (pdir == NULL) {
    return -1;
  }

  struct dirent *pent = NULL;
  int i;
  for (i=0; i < 3; i++) {
    pent = readdir(pdir);
    if (pent == NULL) {
      return -1;
    }
  }

  strcpy(name, pent->d_name);
  closedir(pdir);
  return 0;
}

int getBGPExePath(char *path) {
  int rc;
  char jobid[256];
  rc = getBGPJobID("/jobs", jobid);
  if (rc != 0) {
    return -1;
  }

  sprintf (path, "/jobs/%s/exe", jobid);
  return 0;
}


/*
 * Get symbol table either by using BFD or by parsing nm-file
 */
static void get_symtab(void) {
  char path[4096];
  uint32_t size = sizeof(path);
  int rc;

#ifndef TAU_BFD
  fprintf(stderr, "TAU: Warning! BFD not found, symbols will not be resolved\n");
  return;
#endif

  /* System dependent methods to find the executable */

  /* Default: Linux systems */
  sprintf (path, "/proc/self/exe");
  
#ifdef TAU_AIX
  sprintf (path, "/proc/%d/object/a.out", getpid());
#endif
  
#ifdef TAU_BGP
  rc = getBGPExePath(path);
  if (rc != 0) {
    fprintf(stderr, "TAU: Warning! BFD not found, symbols will not be resolved\n");
    return;
  }
#endif
  
#ifdef __APPLE__
  _NSGetExecutablePath(path, &size);
#endif
  
  /* Open the executable path */
  get_symtab_bfd(path, 0);
}

typedef struct addrmap_t {
  unsigned long start, end, offset;
  int loaded;
  char name[256];
} addrmap;


vector<addrmap> *addressMap = NULL;

static int updateMaps() {

  if (addressMap == NULL) {
    addressMap = new vector<addrmap>();
  }
  addressMap->clear();

  FILE *mapsfile = fopen ("/proc/self/maps", "r");
  if (mapsfile == NULL) {
    return -1;
  }
  
  char line[4096];
  while (!feof(mapsfile)) {
    fgets(line, 4096, mapsfile);
    //printf ("=> %s", line);
    unsigned long start, end, offset;
    char module[4096];
    char perms[5];
    module[0] = 0;
    sscanf(line, "%lx-%lx %s %lx %*s %*u %[^\n]", &start, &end, perms, &offset, module);

    if (*module && ((strcmp(perms, "r-xp") == 0) || (strcmp(perms, "rwxp") == 0))) {
      //printf ("got %s, %p-%p (%p)\n", module, start, end, offset);
      addrmap entry;
      entry.start = start;
      entry.end = end;
      entry.offset = offset;
      entry.loaded = 0;
      sprintf (entry.name, module);
      addressMap->push_back(entry);
    }
  }
  return 0;
}



static addrmap *getAddressMap(unsigned long addr) {
  for (unsigned int i=0;i<addressMap->size();i++) {
    if (addr >= (*addressMap)[i].start && addr <= (*addressMap)[i].end) {
      return &((*addressMap)[i]);
    }
  }

  // Wasn't found in any ranges, try updating the maps
  updateMaps();

  for (unsigned int i=0;i<addressMap->size();i++) {
    if (addr >= (*addressMap)[i].start && addr <= (*addressMap)[i].end) {
      return &((*addressMap)[i]);
    }
  }

  // Still not found?  Give up
  return NULL;
}


static HashNode *createHashNode(long addr) {
  addrmap *map = getAddressMap(addr);

  if (map && map->loaded == 0) {
    get_symtab_bfd(map->name, map->start);
    map->loaded = true;
    
    HashNode *hn = hash_get(addr);
    if (hn) {
      if ( hn->fi == NULL) {

	// remove the path
	const char *filename = hn->fname;
        if (filename) {
	  while (strchr(filename,'/') != NULL) {
	    filename = strchr(filename,'/')+1;
	  }
        } else {
          filename = "(unknown)";
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


  // Unknown
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


#ifdef TAU_SICORTEX
#pragma weak __cyg_profile_func_enter
#endif
extern "C" void __cyg_profile_func_enter(void* func, void* callsite) {
  int i;
  int tid;

  if (executionFinished) {
    return;
  }

  HashNode *hn;
  void * funcptr = func;
#ifdef __ia64__
  funcptr = *( void ** )func;
#endif

  if (gnu_init) {
    gnu_init = 0;

    // initialize array of flags that prevent re-entry
    for (i=0; i<TAU_MAX_THREADS; i++) {
      compInstDisabled[i] = 0;
    }

    get_symtab();
    InitializeTAU();
    TheUsingCompInst() = 1;
    TAU_PROFILE_SET_NODE(0);
    updateMaps();
  }

  // prevent re-entry of this routine on a per thread basis
  tid = Tau_get_tid();
  if (compInstDisabled[tid]) {
    return;
  }
  compInstDisabled[tid] = 1;

  if ((hn = hash_get((long)funcptr))) {
    if (hn->excluded) {
      // finished in this routine, allow entry
      compInstDisabled[tid] = 0;
      return;
    }
    if (hn->fi == NULL) {

      RtsLayer::LockDB(); // lock, then check again
      
      if ( hn->fi == NULL) {
	// remove the path
	const char *filename = hn->fname;
	if (filename) {
	  while (strchr(filename,'/') != NULL) {
	    filename = strchr(filename,'/')+1;
	  }
	} else {
	  filename = "(unknown)";
	}
	
	char *routine;
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
    
    if ( (hn = hash_get((long)funcptr))) {
      Tau_start_timer(hn->fi, 0, tid);
    } else {
      HashNode *node = createHashNode((long)funcptr);
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
}

extern "C" void _cyg_profile_func_enter(void* func, void* callsite) {
  __cyg_profile_func_enter(func, callsite);
}


#ifdef TAU_SICORTEX
#pragma weak __cyg_profile_func_exit
#endif
extern "C" void __cyg_profile_func_exit(void* func, void* callsite) {
  int tid;

  tid = Tau_get_tid();

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

  if ( (hn = hash_get((long)funcptr)) ) {
    if (hn->excluded) {
      return;
    }

    Tau_stop_timer(hn->fi, tid);
  } else {
    //printf ("NOT FOUND! : ");
  }
}

extern "C" void _cyg_profile_func_exit(void* func, void* callsite) {
  __cyg_profile_func_exit(func, callsite);
}

#endif /* TAU_XLC */
