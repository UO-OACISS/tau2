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


// called during termination
extern "C" void __rouexit() {
  Tau_destructor_trigger();
}

// called during program initialization
extern "C" void __rouinit() {
  Tau_init_initializeTAU();
  TheUsingCompInst() = 1;
  TAU_PROFILE_SET_NODE(0);
  atexit(__rouexit);
}

// called at the beginning of each profiled routine
#pragma save_all_gp_regs
extern "C" void ___rouent2(struct s1 *p) {
  char routine[2048];

  if (!p->isseen) {
    sprintf (routine, "%s [{%s} {%d,0}]", p->rout, p->file, p->lineno);
    char* modpos;
    
    /* fix opari output file names */
    if ( (modpos = strstr(p->file, ".mod.")) != NULL ) {
      strcpy(modpos, modpos+4);
    }
      
#ifdef TAU_OPENMP
    
    if (omp_in_parallel()) {
#pragma omp critical (tau_comp_pgi_1)
      {
	if (!p->isseen) {	
	  void *handle=NULL;
	  TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
	  FunctionInfo *fi = (FunctionInfo*)handle;
	  Tau_start_timer(fi,0, Tau_get_tid());
	  p->rid = TheFunctionDB().size()-1;
	  p->isseen = 1;
	}
      }
    } else {
      void *handle=NULL;
      TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
      FunctionInfo *fi = (FunctionInfo*)handle;
      Tau_start_timer(fi,0, Tau_get_tid());
      p->rid = TheFunctionDB().size()-1;
      p->isseen = 1;
    }
#else
    void *handle=NULL;
    TAU_PROFILER_CREATE(handle, routine, "", TAU_DEFAULT);
    FunctionInfo *fi = (FunctionInfo*)handle;
    Tau_start_timer(fi,0, Tau_get_tid());
    p->rid = TheFunctionDB().size()-1;
    p->isseen = 1;
#endif
  } else {
    Tau_start_timer(TheFunctionDB()[p->rid],0, Tau_get_tid());
  }
}

// called at the end of each profiled routine
extern "C" void ___rouret2(void) {
  TAU_MAPPING_PROFILE_STOP(0);
}

extern "C" void ___linent2(void *l) {}
