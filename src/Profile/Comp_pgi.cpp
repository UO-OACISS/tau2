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
#include <vector>
#ifdef TAU_OPENMP
#  include <omp.h>
#endif

// adding this check prevents compiler warnings for unknown pragmas.
#ifdef __PGI

enum RoutineState { ROUTINE_CREATED = 1, ROUTINE_THROTTLED = 2 };

using namespace std;


struct s1 {
  long l1;
  long l2;
  double d1;
  double d2;
  long isseen; //status of this routine.
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

extern "C" void Tau_profile_exit_all_threads(void);

#define dprintf TAU_VERBOSE
// called during termination
#pragma save_all_regs
extern "C" void __rouexit(void) {
  Tau_profile_exit_all_threads();
  Tau_destructor_trigger();
}

// called during program initialization
#pragma save_all_regs
extern "C" void __rouinit(void) {
  Tau_init_initializeTAU();
  TheUsingCompInst() = 1;
#if (defined (TAU_MPI) && defined(TAU_CRAYCNL))
  // If I am the master process spawned by aprun on the Cray,
  //  don't set it to 0.
  if (RtsLayer::myNode() != -1) {
    TAU_PROFILE_SET_NODE(0);
  }
#else
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */
  atexit(__rouexit);
}

int Tau_get_function_index_in_DB(FunctionInfo *fi) {
	RtsLayer::LockDB();
  int i = TheFunctionDB().size();
	int v = -1;
  for (vector<FunctionInfo*>::iterator it = TheFunctionDB().end();
      it != TheFunctionDB().begin(); it--, i--) {
    if (*it == fi)
    {
      dprintf("MATCH! i=%d, TheFunctionDB()[%d]->GetName() = %s, fi->GetName() = %s\n", i, i, TheFunctionDB()[i]->GetName(), fi->GetName());
      v = i;
    }
  }
	RtsLayer::UnLockDB();
  return v;
}

struct Tau_thread_ignore_flag {
	int count; // 4 bytes
	int padding[15]; // remaining 60 bytes.
};

/*
#ifdef __INTEL__COMPILER
__declspec (align(64)) static struct Tau_thread_ignore_flag Tau_ignore[TAU_MAX_THREADS] = {0};
#else
#ifdef __GNUC__
static struct Tau_thread_ignore_flag Tau_ignore[TAU_MAX_THREADS] __attribute__ ((aligned(64))) = {0};
#else
static struct Tau_thread_ignore_flag Tau_ignore[TAU_MAX_THREADS] = {0};
#endif
#endif
*/


struct Tau_ignore_list : vector<Tau_thread_ignore_flag*>{
    Tau_ignore_list(){
      }
     virtual ~Tau_ignore_list(){
         Tau_destructor_trigger();
     }
   };

static Tau_ignore_list & Tau_ignore_vector(){
    static Tau_ignore_list TIInstance;
    return TIInstance;
}
inline void checkTau_ignore_vector(int tid){
        while(Tau_ignore_vector().size()<=tid){
        RtsLayer::LockDB();
                Tau_ignore_vector().push_back(new Tau_thread_ignore_flag());
        RtsLayer::UnLockDB();
        }
}

/*static inline void Tau_ignore_inc(int tid){
	checkTau_ignore_vector(tid);
        (Tau_ignore_vector()[tid])->count++;
}*/

static inline int& Tau_ignore_count(int tid){
	checkTau_ignore_vector(tid);
	return Tau_ignore_vector()[tid]->count;
}


// called at the beginning of each profiled routine
#pragma save_all_regs
extern "C" void ___rouent2(struct s1 *p) {

  int tid = Tau_get_local_tid();
  
  if (!p->isseen) {
		RtsLayer::LockEnv();
		/* Some routines like length__Q2_3std20char_traits__tm__2_cSFPCc are called
			 before main and get called repeatedly when <iostream> and cout are used
			 in a C++ application. We need to create a top level timer if necessary */
  	char routine[2048];
    snprintf (routine, sizeof(routine),  "%s [{%s} {%ld,0}]", p->rout, p->file, p->lineno);
    char* modpos;
    
    /* fix opari output file names */
    if ( (modpos = strstr(p->file, ".mod.")) != NULL ) {
      strcpy(modpos, modpos+4);
    }
		void *vp;
		if (!p->isseen) {
			Tau_create_top_level_timer_if_necessary_task(tid); 
			TAU_PROFILER_CREATE(vp, routine, "", TAU_DEFAULT);
			FunctionInfo *fi = (FunctionInfo *)vp;
			p->rid = Tau_get_function_index_in_DB(fi);
			Tau_start_timer(fi, 0, Tau_get_thread());
			if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
				Tau_ignore_count(tid)++; // the rouent2 shouldn't call stop
				p->isseen = ROUTINE_THROTTLED;	
			}
			else {
				p->isseen = ROUTINE_CREATED;
			}
		}
		RtsLayer::UnLockEnv();
	} // seen this routine before, no need to create another FunctionInfo object.
	else 
	{
    if (p->isseen == ROUTINE_THROTTLED) {
      Tau_ignore_count(tid)++; // the rouent2 shouldn't call stop
      return;
    }
		RtsLayer::LockDB();
    FunctionInfo *fi = (FunctionInfo*)(TheFunctionDB()[p->rid]);
		RtsLayer::UnLockDB();
	  Tau_start_timer(fi, 0, Tau_get_thread());
		if (!(fi->GetProfileGroup() & RtsLayer::TheProfileMask())) {
			Tau_ignore_count(tid)++; // the rouent2 shouldn't call stop
			p->isseen = ROUTINE_THROTTLED;	
		}
	}
}

#pragma save_all_gp_regs
extern "C" void ___instent64 (void* a1, void* a2, void* a3, void* a4, struct s1 *p) {
  ___rouent2(p);
}

#pragma save_all_gp_regs
extern "C" void ___rouent64 (void* a1, void* a2, void* a3, void* a4, struct s1 *p) {
  ___rouent2(p);
}

#pragma save_all_gp_regs
extern "C" void ___instentavx (void* a1, void* a2, void* a3, void* a4, struct s1 *p) {
  ___rouent2(p);
}


// called at the end of each profiled routine
#pragma save_all_regs
extern "C" void ___rouret2(void) {
  
	int tid = Tau_get_local_tid();
	
	if (Tau_ignore_count(tid) == 0)
	{
  	TAU_MAPPING_PROFILE_STOP(0);
	} else {
    Tau_ignore_count(tid)--;
	}
}

#pragma save_all_gp_regs
extern "C" void ___instret64(void* a1, void* a2, void* a3, void* a4, struct s1 *p) {
  	___rouret2();
}

#pragma save_all_gp_regs
extern "C" void ___rouret(void) {
  	___rouret2();
}

#pragma save_all_gp_regs
extern "C" void ___rouret64(void) {
  	___rouret2();
}

#pragma save_all_gp_regs
extern "C" void ___instretavx(void* a1, void* a2, void* a3, void* a4, struct s1 *p) {
  	___rouret2();
}

#pragma save_used_gp_regs
extern "C" void ___linent2(void *l) {
}

#endif //ifdef __PGI
