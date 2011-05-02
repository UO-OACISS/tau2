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
**	File 		: Comp_pgi.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains the hooks for PGI based       **
**                        compiler instrumentation                         **
**                                                                         **
****************************************************************************/

#include <TAU.h>
#include <Profile/TauInit.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef TAU_OPENMP
#  include <omp.h>
#endif


struct s1 {
  long l1;
  long l2;
  double d1;
  double d2;
  long isseen;
  char *c;
  void *p1;
  long lineno;
  void *p2;
  struct s1 *p3;
  int fid;
  int rid;
  char *file;
  char *rout;
};

extern "C"  int Tau_profile_exit_all_threads(void); 
#define dprintf TAU_VERBOSE
// called during termination
#pragma save_all_regs
extern "C" void __rouexit() {
  Tau_destructor_trigger();
}

// called during program initialization
#pragma save_all_regs
extern "C" void __rouinit() {
  Tau_init_initializeTAU();
  TheUsingCompInst() = 1;
  TAU_PROFILE_SET_NODE(0);
  atexit(__rouexit);
}

int Tau_get_function_index_in_DB(FunctionInfo *fi) {
  int i = TheFunctionDB().size();
  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().end();
      it != TheFunctionDB().begin(); it--, i--) {
    if (*it == fi)
    {
      dprintf("MATCH! i=%d, TheFunctionDB()[%d]->GetName() = %s, fi->GetName() = %s\n", i, i, TheFunctionDB()[i]->GetName(), fi->GetName());
      return i;
    }
  }
  return -1;
}

int Tau_ignore_count[TAU_MAX_THREADS]={0};
// called at the beginning of each profiled routine
#pragma save_all_regs
extern "C" void ___rouent2(struct s1 *p) {
  char routine[2048];
  int isseen_local = p->isseen;

  int tid = Tau_get_tid();
  if (p->isseen == -1) {
    Tau_ignore_count[tid] ++;  // the rouent2 shouldn't call stop
    return;
  }
  if ((!Tau_init_check_initialized()) || (Tau_global_get_insideTAU() > 0 )) { 
    //dprintf("TAU not initialized /inside TAU in __rouent2. Going to ignore this one!name = p->rout %s\n", p->rout);
    Tau_ignore_count[tid] ++;  // the rouent2 shouldn't call stop
    return;
  }

  /* Some routines like length__Q2_3std20char_traits__tm__2_cSFPCc are called
     before main and get called repeatedly when <iostream> and cout are used
     in a C++ application. We need to create a top level timer if necessary */
  p->isseen = -1; 
  Tau_create_top_level_timer_if_necessary(); 
  p->isseen = isseen_local; 

  if (!p->isseen) {
    sprintf (routine, "%s [{%s} {%d,0}]", p->rout, p->file, p->lineno);
    char* modpos;
    
    /* fix opari output file names */
    if ( (modpos = strstr(p->file, ".mod.")) != NULL ) {
      strcpy(modpos, modpos+4);
    }
      
#ifdef TAU_OPENMP
    
    if (omp_in_parallel()) {
      int returnFromBlock = 0;
#pragma omp critical (tau_comp_pgi_1)
      {
	if (!p->isseen) {	
	  void *handle=NULL;
	  p->isseen ++;
	  TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
	  FunctionInfo *fi = (FunctionInfo*)handle;
          if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
	    Tau_ignore_count[tid]++; // the rouent2 shouldn't call stop
	    returnFromBlock = 1;
            /* return; Not allowed inside an omp critical */
          } else {
	    Tau_start_timer(fi,0, tid);
	    p->rid = Tau_get_function_index_in_DB(fi);
	  }
	}
      }
      if (returnFromBlock == 1) {
	return;
      }
    } else {
      void *handle=NULL;
      p->isseen = -1;
      TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
      FunctionInfo *fi = (FunctionInfo*)handle;
      if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
	Tau_ignore_count[tid]++; // the rouent2 shouldn't call stop
	return;
      }
      Tau_start_timer(fi,0, tid);
      p->rid = Tau_get_function_index_in_DB(fi);
      p->isseen = isseen_local+1;
    }
#else
    void *handle=NULL;
    p->isseen = -1; // hold on, this is in the middle of creating a profiler
    // if we re-enter this routine, just return so we can continue in the right
    // place. 
    TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
    FunctionInfo *fi = (FunctionInfo*)handle;
    if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
      Tau_ignore_count[tid]++; // the rouent2 shouldn't call stop
      return;
    }
    Tau_start_timer(fi,0, tid);
    p->rid = Tau_get_function_index_in_DB(fi);
    p->isseen = isseen_local+1;
#endif
  } else {
    FunctionInfo *fi = (FunctionInfo*)(TheFunctionDB()[p->rid]);
    if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
      Tau_ignore_count[tid]++; // the rouent2 shouldn't call stop
      return;
    }
    Tau_start_timer(fi, 0, tid);
  }
}

// called at the end of each profiled routine
#pragma save_all_regs
extern "C" void ___rouret2(void) {
  int tid = Tau_get_tid();
  if (Tau_ignore_count[tid] == 0) { 
    TAU_MAPPING_PROFILE_STOP(0);
  }
  else {
    Tau_ignore_count[tid]--;
  }
}

#pragma save_used_gp_regs
extern "C" void ___linent2(void *l) {
}
