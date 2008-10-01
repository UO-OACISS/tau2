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

#include <TAU.h>


typedef struct HN {
  long id;            /* hash code (address of function */
  FunctionInfo *fi;
  struct HN* next;
} HashNode;

#define HASH_MAX 1021
static HashNode* htab[HASH_MAX];


static HashNode *hash_put(long h, FunctionInfo *fi) {
  long id = h % HASH_MAX;
  HashNode *add = (HashNode*)malloc(sizeof(HashNode));
  add->id = h;
  add->fi = fi;
  add->next = htab[id];
  htab[id] = add;
  return add;
}

static HashNode *hash_get(long h) {
  long id = h % HASH_MAX;
  HashNode *curr = htab[id];
  while ( curr ) {
    if ( curr->id == h ) {
      return curr;
    }
    curr = curr->next;
  }
  return 0;
}

static HashNode *register_region(char *func, char *file, int lno) {
  uint32_t rid;
  uint32_t fid;
  HashNode* nhn;

  char routine[2048];
  sprintf (routine, "%s [{%s} {%d,0}]", func, file, lno);

  void *handle=NULL;
  TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
  FunctionInfo *fi = (FunctionInfo*)handle;
  nhn = hash_put((long) func, fi);
  return nhn;
}


extern "C" void __func_trace_enter(char* name, char* fname, int lno) {
  static int initialized = 0;
  HashNode *hn;

  if (initialized == 0) {
    initialized = 1;
    TheUsingCompInst() = 1;
    TAU_PROFILE_SET_NODE(0);
  }

  // ignore IBM OMP runtime functions
  if (strchr(name, '@') != NULL ) return;

  // look up in the hash table
  if ((hn = hash_get((long) name)) == 0 ) {
    // not found, register the region
#   ifdef TAU_OPENMP
    if (omp_in_parallel()) {
#     pragma omp critical (tau_comp_xl_1)
      {
        if ( (hn = hash_get((long) name)) == 0 ) {
          hn = register_region(name, fname, lno);
        }
      }
    } else {
      hn = register_region(name, fname, lno);
    }
#   else
    hn = register_region(name, fname, lno);
#   endif
  }

  Tau_start_timer(hn->fi, 0);
  //TAU_START(name);
}

extern "C" void __func_trace_exit(char* name, char *fname, int lno) {
  HashNode *hn;

  // ignore IBM OMP runtime functions
  if ( strchr(name, '@') != NULL ) return;

  hn = hash_get((long) name);
  Tau_stop_timer(hn->fi);

  //TAU_STOP(name);
}

