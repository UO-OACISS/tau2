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
**	File 		: Comp_intel9.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains the hooks for Intel 9 based   **
**                        compiler instrumentation                         **
**                                                                         **
****************************************************************************/

#include <TAU.h>

//#define USE_MAP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef TAU_OPENMP
#  include <omp.h>
#endif

#ifdef __ia64__
#define COMP_INST_KEY "_2$ITC$0"
#else
#define COMP_INST_KEY "_2.ITC.0"
#endif


static int intel_init = 1;

typedef struct HN {
  long id;
  FunctionInfo *fi;
  struct HN* next;
} HashNode;

#define HASH_MAX 1021

static HashNode* htab[HASH_MAX];

static HashNode *hash_put(long h) {
  long id = h % HASH_MAX;
  HashNode *add = (HashNode*)malloc(sizeof(HashNode));
  add->id = h;
  add->fi = NULL;
  add->next = htab[id];
  htab[id] = add;
  return add;
}

static HashNode* hash_get(long h) {
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


static map<int*, FunctionInfo*> theMap;

extern "C" void __VT_IntelEntry(char* str, int* id, int* id2) {
  HashNode *hn;
  long frame;

  if (intel_init) { // initialization
    intel_init = 0;
    TheUsingCompInst() = 1;
    TAU_PROFILE_SET_NODE(0);
  }

#ifdef USE_MAP
  FunctionInfo *fi = 0;
  map<int*, FunctionInfo*>::iterator it = theMap.find(id2);
  if (it == theMap.end()) {
    void *handle=NULL;
    TAU_PROFILER_CREATE(handle, str, "", TAU_DEFAULT);
    fi = (FunctionInfo*)handle;
    theMap[id2] = fi;
  } else {
    fi = (*it).second;
  }
  Tau_start_timer(fi,0,Tau_get_tid());

#else

//   if ((hn = hash_get((long)id2))) {
//     Tau_start_timer(hn->fi,0);
//   } else {
//     hn = hash_put((long)id2);
//     printf ("Registered %s for %p:%p\n", str, id, id2);
//     void *handle=NULL;
//     TAU_PROFILER_CREATE(handle, str, "", TAU_DEFAULT);
//     hn->fi = (FunctionInfo*) handle;
//     Tau_start_timer(hn->fi,0,Tau_get_tid());
//   }

  if (*id != 0) {
    Tau_start_timer(TheFunctionDB()[*id],0,Tau_get_tid());
  } else {
    //printf ("Registered %s for %p:%p\n", str, id, id2);
    void *handle=NULL;
    TAU_PROFILER_CREATE(handle, str, "", TAU_DEFAULT);
    FunctionInfo *fi = (FunctionInfo*)handle;
    Tau_start_timer(fi,0,Tau_get_tid());
    *id = TheFunctionDB().size()-1;
  }
  *id2 = *id;
#endif

  //printf ("VT Entry: %s, %p(%d), %p(%d)\n", str, id, *id, id2, *id2);
}

extern "C" void VT_IntelEntry(char* str, int* id, int* id2) {
  __VT_IntelEntry(str, id, id2);
}


extern "C" void __VT_IntelExit(int* id2) {
#ifdef USE_MAP
//   map<int*, FunctionInfo*>::iterator it = theMap.find(id2);
//   if (it != theMap.end()) {
//     FunctionInfo *fi = (*it).second;
//     Tau_stop_timer(fi);
//   }  
#else
//   HashNode *hn;
//   if ((hn = hash_get((long)id2))) {
//     Tau_stop_timer(hn->fi);
//   }
  Tau_stop_timer(TheFunctionDB()[*id2], Tau_get_tid());
#endif


  //printf ("VT Exit: %p\n", id2);
}

extern "C" void VT_IntelExit(int* id2) {
  __VT_IntelExit(id2);
}


extern "C" void __VT_IntelCatch(int* id2) {
}
extern "C" void VT_IntelCatch(int* id2) {
  __VT_IntelCatch(id2);
}




