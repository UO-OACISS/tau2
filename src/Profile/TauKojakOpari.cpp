/*************************************************************************/
/* TAU OPARI Layer							 */
/* Copyright (C) 2001                                                    */
/* University of Oregon, Los Alamos National Laboratory, and 		 */
/* Forschungszentrum Juelich, Zentralinstitut fuer Angewandte Mathematik */
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>

//#define DEBUG_PROF 1
#include <Profile/Profiler.h>
#ifdef TAU_OPENMP
#ifndef _OPENMP
#define _OPENMP
#endif /* _OPENMP */
#endif /* TAU_OPENMP */
#include "pomp_lib.h"

/* These two defines specify if we want region based views or construct based
views or both */
#ifdef TAU_OPARI_REGION
#define TAU_OPENMP_REGION_VIEW
#elif TAU_OPARI_CONSTRUCT
#define TAU_AGGREGATE_OPENMP_TIMINGS
#else
#define TAU_AGGREGATE_OPENMP_TIMINGS
#define TAU_OPENMP_REGION_VIEW
#endif /* in the default mode, define both! */

#define OpenMP TAU_USER
#define TAU_EMBEDDED_MAPPING 1

omp_lock_t tau_ompregdescr_lock; 

TAU_GLOBAL_TIMER(tatomic, "atomic enter/exit", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tbarrier, "barrier enter/exit", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tcriticalb, "critical begin/end", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tcriticale, "critical enter/exit", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tfor, "for enter/exit", "[OpenMP]", OpenMP);
TAU_GLOBAL_TIMER(tmaster, "master begin/end", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tparallelb, "parallel begin/end", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tparallelf, "parallel fork/join", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tsectionb, "section begin/end", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tsectione, "sections enter/exit", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tsingleb, "single begin/end", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tsinglee, "single enter/exit", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tworkshare, "workshare enter/exit", "[OpenMP]", OpenMP); 
TAU_GLOBAL_TIMER(tregion, "inst region begin/end", "[OpenMP]", OpenMP); 


#define NUM_OMP_TYPES 15

static char* omp_names[15] = {"atomic enter/exit", "barrier enter/exit", "critical begin/end", 
			     "critical enter/exit", "for enter/exit", "master begin/end",
			     "parallel begin/end", "parallel fork/join", "section begin/end",
			     "sections enter/exit", "single begin/end", "single enter/exit",
			      "workshare enter/exit", "inst region begin/end", "flush enter/exit" };


#define TAU_OMP_ATOMIC      0
#define TAU_OMP_BARRIER     1
#define TAU_OMP_CRITICAL_BE 2
#define TAU_OMP_CRITICAL_EE 3
#define TAU_OMP_FOR_EE      4
#define TAU_OMP_MASTER_BE   5
#define TAU_OMP_PAR_BE      6
#define TAU_OMP_PAR_FJ      7
#define TAU_OMP_SECTION_BE  8
#define TAU_OMP_SECTION_EE  9
#define TAU_OMP_SINGLE_BE  10
#define TAU_OMP_SINGLE_EE  11
#define TAU_OMP_WORK_EE    12
#define TAU_OMP_INST_BE    13
#define TAU_OMP_FLUSH_EE   14

static int omp_tracing    = 1;
static int omp_fin_called = 0;

/*
 * Fortran wrappers calling the C versions
 */

extern "C" {
/****************************/
void pomp_finalize() {
  POMP_Finalize();
}

void pomp_finalize_() {
  POMP_Finalize();
}

void pomp_finalize__() {
  POMP_Finalize();
}

void POMP_FINALIZE() {
  POMP_Finalize();
}

/****************************/

void pomp_init() {
  POMP_Init();
}

void pomp_init_() {
  POMP_Init();
}

void pomp_init__() {
  POMP_Init();
}

void POMP_INIT() {
  POMP_Init();
}

/****************************/

void pomp_off() {
  omp_tracing = 0;
}

void pomp_off_() {
  omp_tracing = 0;
}

void pomp_off__() {
  omp_tracing = 0;
}

void POMP_OFF() {
  omp_tracing = 0;
}

/****************************/

void pomp_on() {
  omp_tracing = 1;
}

void pomp_on_() {
  omp_tracing = 1;
}

void pomp_on__() {
  omp_tracing = 1;
}

void POMP_ON() {
  omp_tracing = 1;
}

/****************************/

void pomp_atomic_enter(int* id) {
  if ( omp_tracing ) POMP_Atomic_enter(pomp_rd_table[*id]);
}

void pomp_atomic_enter_(int* id) {
  if ( omp_tracing ) POMP_Atomic_enter(pomp_rd_table[*id]);
}

void pomp_atomic_enter__(int* id) {
  if ( omp_tracing ) POMP_Atomic_enter(pomp_rd_table[*id]);
}
void POMP_ATOMIC_ENTER(int* id) {
  if ( omp_tracing ) POMP_Atomic_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_atomic_exit(int* id) {
  if ( omp_tracing ) POMP_Atomic_exit(pomp_rd_table[*id]);
}

void pomp_atomic_exit_(int* id) {
  if ( omp_tracing ) POMP_Atomic_exit(pomp_rd_table[*id]);
}

void pomp_atomic_exit__(int* id) {
  if ( omp_tracing ) POMP_Atomic_exit(pomp_rd_table[*id]);
}

void POMP_ATOMIC_EXIT(int* id) {
  if ( omp_tracing ) POMP_Atomic_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_barrier_enter(int* id) {
  if ( omp_tracing ) POMP_Barrier_enter(pomp_rd_table[*id]);
}

void pomp_barrier_enter_(int* id) {
  if ( omp_tracing ) POMP_Barrier_enter(pomp_rd_table[*id]);
}

void pomp_barrier_enter__(int* id) {
  if ( omp_tracing ) POMP_Barrier_enter(pomp_rd_table[*id]);
}

void POMP_BARRIER_ENTER(int* id) {
  if ( omp_tracing ) POMP_Barrier_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_barrier_exit(int* id) {
  if ( omp_tracing ) POMP_Barrier_exit(pomp_rd_table[*id]);
}

void pomp_barrier_exit_(int* id) {
  if ( omp_tracing ) POMP_Barrier_exit(pomp_rd_table[*id]);
}

void pomp_barrier_exit__(int* id) {
  if ( omp_tracing ) POMP_Barrier_exit(pomp_rd_table[*id]);
}

void POMP_BARRIER_EXIT(int* id) {
  if ( omp_tracing ) POMP_Barrier_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_begin(int* id) {
  if ( omp_tracing ) POMP_Critical_begin(pomp_rd_table[*id]);
}

void pomp_critical_begin_(int* id) {
  if ( omp_tracing ) POMP_Critical_begin(pomp_rd_table[*id]);
}

void pomp_critical_begin__(int* id) {
  if ( omp_tracing ) POMP_Critical_begin(pomp_rd_table[*id]);
}

void POMP_CRITICAL_BEGIN(int* id) {
  if ( omp_tracing ) POMP_Critical_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_end(int* id) {
  if ( omp_tracing ) POMP_Critical_end(pomp_rd_table[*id]);
}

void pomp_critical_end_(int* id) {
  if ( omp_tracing ) POMP_Critical_end(pomp_rd_table[*id]);
}

void pomp_critical_end__(int* id) {
  if ( omp_tracing ) POMP_Critical_end(pomp_rd_table[*id]);
}

void POMP_CRITICAL_END(int* id) {
  if ( omp_tracing ) POMP_Critical_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_enter(int* id) {
  if ( omp_tracing ) POMP_Critical_enter(pomp_rd_table[*id]);
}

void pomp_critical_enter_(int* id) {
  if ( omp_tracing ) POMP_Critical_enter(pomp_rd_table[*id]);
}

void pomp_critical_enter__(int* id) {
  if ( omp_tracing ) POMP_Critical_enter(pomp_rd_table[*id]);
}

void POMP_CRITICAL_ENTER(int* id) {
  if ( omp_tracing ) POMP_Critical_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_exit(int* id) {
  if ( omp_tracing ) POMP_Critical_exit(pomp_rd_table[*id]);
}

void pomp_critical_exit_(int* id) {
  if ( omp_tracing ) POMP_Critical_exit(pomp_rd_table[*id]);
}

void pomp_critical_exit__(int* id) {
  if ( omp_tracing ) POMP_Critical_exit(pomp_rd_table[*id]);
}

void POMP_CRITICAL_EXIT(int* id) {
  if ( omp_tracing ) POMP_Critical_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_do_enter(int* id) {
  if ( omp_tracing ) POMP_For_enter(pomp_rd_table[*id]);
}

void pomp_do_enter_(int* id) {
  if ( omp_tracing ) POMP_For_enter(pomp_rd_table[*id]);
}

void pomp_do_enter__(int* id) {
  if ( omp_tracing ) POMP_For_enter(pomp_rd_table[*id]);
}

void POMP_DO_ENTER(int* id) {
  if ( omp_tracing ) POMP_For_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_do_exit(int* id) {
  if ( omp_tracing ) POMP_For_exit(pomp_rd_table[*id]);
}

void pomp_do_exit_(int* id) {
  if ( omp_tracing ) POMP_For_exit(pomp_rd_table[*id]);
}

void pomp_do_exit__(int* id) {
  if ( omp_tracing ) POMP_For_exit(pomp_rd_table[*id]);
}

void POMP_DO_EXIT(int* id) {
  if ( omp_tracing ) POMP_For_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_master_begin(int* id) {
  if ( omp_tracing ) POMP_Master_begin(pomp_rd_table[*id]);
}

void pomp_master_begin_(int* id) {
  if ( omp_tracing ) POMP_Master_begin(pomp_rd_table[*id]);
}

void pomp_master_begin__(int* id) {
  if ( omp_tracing ) POMP_Master_begin(pomp_rd_table[*id]);
}

void POMP_MASTER_BEGIN(int* id) {
  if ( omp_tracing ) POMP_Master_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_master_end(int* id) {
  if ( omp_tracing ) POMP_Master_end(pomp_rd_table[*id]);
}

void pomp_master_end_(int* id) {
  if ( omp_tracing ) POMP_Master_end(pomp_rd_table[*id]);
}

void pomp_master_end__(int* id) {
  if ( omp_tracing ) POMP_Master_end(pomp_rd_table[*id]);
}

void POMP_MASTER_END(int* id) {
  if ( omp_tracing ) POMP_Master_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_begin(int* id) {
  if ( omp_tracing ) POMP_Parallel_begin(pomp_rd_table[*id]);
}

void pomp_parallel_begin_(int* id) {
  if ( omp_tracing ) POMP_Parallel_begin(pomp_rd_table[*id]);
}

void pomp_parallel_begin__(int* id) {
  if ( omp_tracing ) POMP_Parallel_begin(pomp_rd_table[*id]);
}

void POMP_PARALLEL_BEGIN(int* id) {
  if ( omp_tracing ) POMP_Parallel_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_end(int* id) {
  if ( omp_tracing ) POMP_Parallel_end(pomp_rd_table[*id]);
}

void pomp_parallel_end_(int* id) {
  if ( omp_tracing ) POMP_Parallel_end(pomp_rd_table[*id]);
}

void pomp_parallel_end__(int* id) {
  if ( omp_tracing ) POMP_Parallel_end(pomp_rd_table[*id]);
}

void POMP_PARALLEL_END(int* id) {
  if ( omp_tracing ) POMP_Parallel_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_fork(int* id) {
  if ( omp_tracing ) POMP_Parallel_fork(pomp_rd_table[*id]);
}

void pomp_parallel_fork_(int* id) {
  if ( omp_tracing ) POMP_Parallel_fork(pomp_rd_table[*id]);
}

void pomp_parallel_fork__(int* id) {
  if ( omp_tracing ) POMP_Parallel_fork(pomp_rd_table[*id]);
}

void POMP_PARALLEL_FORK(int* id) {
  if ( omp_tracing ) POMP_Parallel_fork(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_join(int* id) {
  if ( omp_tracing ) POMP_Parallel_join(pomp_rd_table[*id]);
}

void pomp_parallel_join_(int* id) {
  if ( omp_tracing ) POMP_Parallel_join(pomp_rd_table[*id]);
}

void pomp_parallel_join__(int* id) {
  if ( omp_tracing ) POMP_Parallel_join(pomp_rd_table[*id]);
}

void POMP_PARALLEL_JOIN(int* id) {
  if ( omp_tracing ) POMP_Parallel_join(pomp_rd_table[*id]);
}

/****************************/

void pomp_section_begin(int* id) {
  if ( omp_tracing ) POMP_Section_begin(pomp_rd_table[*id]);
}

void pomp_section_begin_(int* id) {
  if ( omp_tracing ) POMP_Section_begin(pomp_rd_table[*id]);
}

void pomp_section_begin__(int* id) {
  if ( omp_tracing ) POMP_Section_begin(pomp_rd_table[*id]);
}

void POMP_SECTION_BEGIN(int* id) {
  if ( omp_tracing ) POMP_Section_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_section_end(int* id) {
  if ( omp_tracing ) POMP_Section_end(pomp_rd_table[*id]);
}

void pomp_section_end_(int* id) {
  if ( omp_tracing ) POMP_Section_end(pomp_rd_table[*id]);
}

void pomp_section_end__(int* id) {
  if ( omp_tracing ) POMP_Section_end(pomp_rd_table[*id]);
}

void POMP_SECTION_END(int* id) {
  if ( omp_tracing ) POMP_Section_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_sections_enter(int* id) {
  if ( omp_tracing ) POMP_Sections_enter(pomp_rd_table[*id]);
}

void pomp_sections_enter_(int* id) {
  if ( omp_tracing ) POMP_Sections_enter(pomp_rd_table[*id]);
}

void pomp_sections_enter__(int* id) {
  if ( omp_tracing ) POMP_Sections_enter(pomp_rd_table[*id]);
}

void POMP_SECTIONS_ENTER(int* id) {
  if ( omp_tracing ) POMP_Sections_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_sections_exit(int* id) {
  if ( omp_tracing ) POMP_Sections_exit(pomp_rd_table[*id]);
}

void pomp_sections_exit_(int* id) {
  if ( omp_tracing ) POMP_Sections_exit(pomp_rd_table[*id]);
}

void pomp_sections_exit__(int* id) {
  if ( omp_tracing ) POMP_Sections_exit(pomp_rd_table[*id]);
}

void POMP_SECTIONS_EXIT(int* id) {
  if ( omp_tracing ) POMP_Sections_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_begin(int* id) {
  if ( omp_tracing ) POMP_Single_begin(pomp_rd_table[*id]);
}

void pomp_single_begin_(int* id) {
  if ( omp_tracing ) POMP_Single_begin(pomp_rd_table[*id]);
}

void pomp_single_begin__(int* id) {
  if ( omp_tracing ) POMP_Single_begin(pomp_rd_table[*id]);
}

void POMP_SINGLE_BEGIN(int* id) {
  if ( omp_tracing ) POMP_Single_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_end(int* id) {
  if ( omp_tracing ) POMP_Single_end(pomp_rd_table[*id]);
}

void pomp_single_end_(int* id) {
  if ( omp_tracing ) POMP_Single_end(pomp_rd_table[*id]);
}

void pomp_single_end__(int* id) {
  if ( omp_tracing ) POMP_Single_end(pomp_rd_table[*id]);
}

void POMP_SINGLE_END(int* id) {
  if ( omp_tracing ) POMP_Single_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_enter(int* id) {
  if ( omp_tracing ) POMP_Single_enter(pomp_rd_table[*id]);
}

void pomp_single_enter_(int* id) {
  if ( omp_tracing ) POMP_Single_enter(pomp_rd_table[*id]);
}

void pomp_single_enter__(int* id) {
  if ( omp_tracing ) POMP_Single_enter(pomp_rd_table[*id]);
}

void POMP_SINGLE_ENTER(int* id) {
  if ( omp_tracing ) POMP_Single_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_exit(int* id) {
  if ( omp_tracing ) POMP_Single_exit(pomp_rd_table[*id]);
}

void pomp_single_exit_(int* id) {
  if ( omp_tracing ) POMP_Single_exit(pomp_rd_table[*id]);
}

void pomp_single_exit__(int* id) {
  if ( omp_tracing ) POMP_Single_exit(pomp_rd_table[*id]);
}

void POMP_SINGLE_EXIT(int* id) {
  if ( omp_tracing ) POMP_Single_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_workshare_enter(int* id) {
  if ( omp_tracing ) POMP_Workshare_enter(pomp_rd_table[*id]);
}

void pomp_workshare_enter_(int* id) {
  if ( omp_tracing ) POMP_Workshare_enter(pomp_rd_table[*id]);
}

void pomp_workshare_enter__(int* id) {
  if ( omp_tracing ) POMP_Workshare_enter(pomp_rd_table[*id]);
}

void POMP_WORKSHARE_ENTER(int* id) {
  if ( omp_tracing ) POMP_Workshare_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_workshare_exit(int* id) {
  if ( omp_tracing ) POMP_Workshare_exit(pomp_rd_table[*id]);
}

void pomp_workshare_exit_(int* id) {
  if ( omp_tracing ) POMP_Workshare_exit(pomp_rd_table[*id]);
}

void pomp_workshare_exit__(int* id) {
  if ( omp_tracing ) POMP_Workshare_exit(pomp_rd_table[*id]);
}

void POMP_WORKSHARE_EXIT(int* id) {
  if ( omp_tracing ) POMP_Workshare_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_begin(int* id) {
  if ( omp_tracing ) POMP_Begin(pomp_rd_table[*id]);
}

void pomp_begin_(int* id) {
  if ( omp_tracing ) POMP_Begin(pomp_rd_table[*id]);
}

void pomp_begin__(int* id) {
  if ( omp_tracing ) POMP_Begin(pomp_rd_table[*id]);
}

void POMP_BEGIN(int* id) {
  if ( omp_tracing ) POMP_Begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_end(int* id) {
  if ( omp_tracing ) POMP_End(pomp_rd_table[*id]);
}

void pomp_end_(int* id) {
  if ( omp_tracing ) POMP_End(pomp_rd_table[*id]);
}

void pomp_end__(int* id) {
  if ( omp_tracing ) POMP_End(pomp_rd_table[*id]);
}

void POMP_END(int* id) {
  if ( omp_tracing ) POMP_End(pomp_rd_table[*id]);
}

/****************************/

void pomp_flush_enter(int* id) {
  if ( omp_tracing ) POMP_Flush_enter(pomp_rd_table[*id]);
}

void pomp_flush_enter_(int* id) {
  if ( omp_tracing ) POMP_Flush_enter(pomp_rd_table[*id]);
}

void pomp_flush_enter__(int* id) {
  if ( omp_tracing ) POMP_Flush_enter(pomp_rd_table[*id]);
}

void POMP_FLUSH_ENTER(int* id) {
  if ( omp_tracing ) POMP_Flush_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_flush_exit(int* id) {
  if ( omp_tracing ) POMP_Flush_exit(pomp_rd_table[*id]);
}

void pomp_flush_exit_(int* id) {
  if ( omp_tracing ) POMP_Flush_exit(pomp_rd_table[*id]);
}

void pomp_flush_exit__(int* id) {
  if ( omp_tracing ) POMP_Flush_exit(pomp_rd_table[*id]);
}

void POMP_FLUSH_EXIT(int* id) {
  if ( omp_tracing ) POMP_Flush_exit(pomp_rd_table[*id]);
}

/****************************/
} /* extern "C" */

/*
 * C pomp function library
 */

/* TAU specific calls */
int tau_openmp_init(void)
{
  omp_init_lock(&tau_ompregdescr_lock);
  return 0;
}


void TauStartOpenMPRegionTimer(struct ompregdescr *r, int index)
{
  static int tau_openmp_initialized = tau_openmp_init();
/* For any region, create a mapping between a region r and timer t and
   start the timer. */

  omp_set_lock(&tau_ompregdescr_lock);

  if (!r->data) {
#ifdef TAU_OPENMP_PARTITION_REGION
    FunctionInfo **flist = new FunctionInfo*[NUM_OMP_TYPES];
    for (int i=0; i < NUM_OMP_TYPES; i++) {
      char rname[1024], rtype[1024];
      sprintf(rname, "%s %s (%s)", r->name, r->sub_name, omp_names[i]);
      sprintf(rtype, "[OpenMP location: file:%s <%d, %d>]",
	      r->file_name, r->begin_first_line, r->end_last_line);
      
      FunctionInfo *f = new FunctionInfo(rname, rtype, OpenMP, "OpenMP");
      flist[i] = f;
    }
    r->data = (void*)flist;
#else
    char rname[1024], rtype[1024];
    sprintf(rname, "%s %s", r->name, r->sub_name);
    sprintf(rtype, "[OpenMP location: file:%s <%d, %d>]",
	    r->file_name, r->begin_first_line, r->end_last_line);
    
    FunctionInfo *f = new FunctionInfo(rname, rtype, OpenMP, "OpenMP");
    r->data = (void*)f;
#endif
  }
  
#ifdef TAU_OPENMP_PARTITION_REGION
  FunctionInfo *f = ((FunctionInfo **)r->data)[index];
#else 
  FunctionInfo *f = (FunctionInfo *)r->data;
#endif
  Tau_start_timer(f, 0);
  
  omp_unset_lock(&tau_ompregdescr_lock);
}


void TauStopOpenMPRegionTimer(struct ompregdescr *r, int index)
{

#ifdef TAU_OPENMP_PARTITION_REGION
    FunctionInfo *f = ((FunctionInfo **)r->data)[index];
#else
    FunctionInfo *f = (FunctionInfo *)r->data;
#endif
    TauGroup_t gr = f->GetProfileGroup();

    int tid = RtsLayer::myThread(); 
    Profiler *p =TauInternal_CurrentProfiler(tid); 
    if (p->ThisFunction == f) {
      Tau_stop_timer(f);
    } else {
      // nothing, it must have been disabled/throttled
    }
}





/* pomp library calls */

void POMP_Finalize() {
  if ( ! omp_fin_called ) {
    omp_fin_called = 1;
  }
#ifdef DEBUG_PROF
  fprintf(stderr, "  0: finalize\n");
#endif /* DEBUG_PROF */
}

void POMP_Init() {
  int i;

  atexit(POMP_Finalize);

#ifdef DEBUG_PROF
  fprintf(stderr, "  0: init\n");
#endif /* DEBUG_PROF */

  for(i=0; i<POMP_MAX_ID; ++i) {
    if ( pomp_rd_table[i] ) {
      pomp_rd_table[i]->data = 0; /* allocate space for performance data here */
    }
  }
}

void POMP_Off() {
  TAU_DISABLE_INSTRUMENTATION();
  omp_tracing = 0;
}

void POMP_On() {
  TAU_ENABLE_INSTRUMENTATION();
  omp_tracing = 1;
}

void POMP_Atomic_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tatomic);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r,TAU_OMP_ATOMIC); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter atomic\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Atomic_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_ATOMIC);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  atomic\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Barrier_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tbarrier);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_BARRIER); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    if ( r->name[0] == 'b' )
      fprintf(stderr, "%3d: enter barrier\n", omp_get_thread_num());
    else
      fprintf(stderr, "%3d: enter implicit barrier of %s\n",
	      omp_get_thread_num(), r->name);
  }
#endif /* DEBUG_PROF */
}

void POMP_Barrier_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_BARRIER);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    if ( r->name[0] == 'b' )
      fprintf(stderr, "%3d: exit  barrier\n", omp_get_thread_num());
    else
      fprintf(stderr, "%3d: exit  implicit barrier of %s\n",
	      omp_get_thread_num(), r->name);
  }
#endif /* DEBUG_PROF */
}

void POMP_Critical_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tcriticalb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_CRITICAL_BE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void POMP_Critical_end(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_CRITICAL_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void POMP_Critical_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tcriticale);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_CRITICAL_EE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void POMP_Critical_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_CRITICAL_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

	
void POMP_For_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tfor); 
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_FOR_EE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter for\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_For_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_FOR_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  // as in a stack. lifo
  

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  for\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Master_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tmaster);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_MASTER_BE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Master_end(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_MASTER_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Parallel_begin(struct ompregdescr* r) {
/* if there is no top level timer, create it */
  Tau_create_top_level_timer_if_necessary();

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tparallelb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_PAR_BE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Parallel_end(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_PAR_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Parallel_fork(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tparallelf);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_PAR_FJ); 
#endif /* TAU_OPENMP_REGION_VIEW */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: fork  parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Parallel_join(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_PAR_FJ);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: join  parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Section_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsectionb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_SECTION_BE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin section\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Section_end(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_SECTION_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */


#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   section\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Sections_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsectione);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_SECTION_EE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter sections\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Sections_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_SECTION_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  sections\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Single_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsingleb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_SINGLE_BE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Single_end(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_SINGLE_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Single_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsinglee);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_SINGLE_EE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Single_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_SINGLE_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Workshare_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tworkshare);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_WORK_EE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter workshare\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Workshare_exit(struct ompregdescr* r) {

  
#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_WORK_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  workshare\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void POMP_Begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tregion);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_INST_BE); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void POMP_End(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_INST_BE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}


void POMP_Flush_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tregion);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r, TAU_OMP_FLUSH_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: flush enter region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void POMP_Flush_exit(struct ompregdescr* r) {

#ifdef TAU_OPENMP_REGION_VIEW
  TauStopOpenMPRegionTimer(r, TAU_OMP_FLUSH_EE);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
 

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: flush exit   region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}


void POMP_Init_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_init_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: init lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_lock(s);
}

void POMP_Destroy_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_destroy_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: destroy lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_lock(s);
}

void POMP_Set_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_set_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: set lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_lock(s);
}

void POMP_Unset_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_unset_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: unset lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_lock(s);
}

int  POMP_Test_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_test_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: test lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_lock(s);
}

void POMP_Init_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_init_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: init nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_nest_lock(s);
}

void POMP_Destroy_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_destroy_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: destroy nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_nest_lock(s);
}

void POMP_Set_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_set_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: set nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_nest_lock(s);
}

void POMP_Unset_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_unset_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: unset nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_nest_lock(s);
}

int  POMP_Test_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_test_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: test nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_nest_lock(s);
}


/***************************************************************************
 * $RCSfile: TauKojakOpari.cpp,v $   $Author: amorris $
 * $Revision: 1.8 $   $Date: 2009/02/24 21:30:23 $
 * POOMA_VERSION_ID: $Id: TauKojakOpari.cpp,v 1.8 2009/02/24 21:30:23 amorris Exp $
 ***************************************************************************/


