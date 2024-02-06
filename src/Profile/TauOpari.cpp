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


static int omp_tracing    = 1;
static int omp_fin_called = 0;

/*
 * Fortran wrappers calling the C versions
 */

extern "C" {
/****************************/
void pomp_finalize_() {
  pomp_finalize();
}

void pomp_finalize__() {
  pomp_finalize();
}

void POMP_FINALIZE() {
  pomp_finalize();
}

/****************************/

void pomp_init_() {
  pomp_init();
}

void pomp_init__() {
  pomp_init();
}

void POMP_INIT() {
  pomp_init();
}

/****************************/

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

void pomp_atomic_enter_(int* id) {
  if ( omp_tracing ) pomp_atomic_enter(pomp_rd_table[*id]);
}

void pomp_atomic_enter__(int* id) {
  if ( omp_tracing ) pomp_atomic_enter(pomp_rd_table[*id]);
}
void POMP_ATOMIC_ENTER(int* id) {
  if ( omp_tracing ) pomp_atomic_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_atomic_exit_(int* id) {
  if ( omp_tracing ) pomp_atomic_exit(pomp_rd_table[*id]);
}

void pomp_atomic_exit__(int* id) {
  if ( omp_tracing ) pomp_atomic_exit(pomp_rd_table[*id]);
}

void POMP_ATOMIC_EXIT(int* id) {
  if ( omp_tracing ) pomp_atomic_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_barrier_enter_(int* id) {
  if ( omp_tracing ) pomp_barrier_enter(pomp_rd_table[*id]);
}

void pomp_barrier_enter__(int* id) {
  if ( omp_tracing ) pomp_barrier_enter(pomp_rd_table[*id]);
}

void POMP_BARRIER_ENTER(int* id) {
  if ( omp_tracing ) pomp_barrier_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_barrier_exit_(int* id) {
  if ( omp_tracing ) pomp_barrier_exit(pomp_rd_table[*id]);
}

void pomp_barrier_exit__(int* id) {
  if ( omp_tracing ) pomp_barrier_exit(pomp_rd_table[*id]);
}

void POMP_BARRIER_EXIT(int* id) {
  if ( omp_tracing ) pomp_barrier_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_begin_(int* id) {
  if ( omp_tracing ) pomp_critical_begin(pomp_rd_table[*id]);
}

void pomp_critical_begin__(int* id) {
  if ( omp_tracing ) pomp_critical_begin(pomp_rd_table[*id]);
}

void POMP_CRITICAL_BEGIN(int* id) {
  if ( omp_tracing ) pomp_critical_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_end_(int* id) {
  if ( omp_tracing ) pomp_critical_end(pomp_rd_table[*id]);
}

void pomp_critical_end__(int* id) {
  if ( omp_tracing ) pomp_critical_end(pomp_rd_table[*id]);
}

void POMP_CRITICAL_END(int* id) {
  if ( omp_tracing ) pomp_critical_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_enter_(int* id) {
  if ( omp_tracing ) pomp_critical_enter(pomp_rd_table[*id]);
}

void pomp_critical_enter__(int* id) {
  if ( omp_tracing ) pomp_critical_enter(pomp_rd_table[*id]);
}

void POMP_CRITICAL_ENTER(int* id) {
  if ( omp_tracing ) pomp_critical_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_critical_exit_(int* id) {
  if ( omp_tracing ) pomp_critical_exit(pomp_rd_table[*id]);
}

void pomp_critical_exit__(int* id) {
  if ( omp_tracing ) pomp_critical_exit(pomp_rd_table[*id]);
}

void POMP_CRITICAL_EXIT(int* id) {
  if ( omp_tracing ) pomp_critical_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_do_enter_(int* id) {
  if ( omp_tracing ) pomp_for_enter(pomp_rd_table[*id]);
}

void pomp_do_enter__(int* id) {
  if ( omp_tracing ) pomp_for_enter(pomp_rd_table[*id]);
}

void POMP_DO_ENTER(int* id) {
  if ( omp_tracing ) pomp_for_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_do_exit_(int* id) {
  if ( omp_tracing ) pomp_for_exit(pomp_rd_table[*id]);
}

void pomp_do_exit__(int* id) {
  if ( omp_tracing ) pomp_for_exit(pomp_rd_table[*id]);
}

void POMP_DO_EXIT(int* id) {
  if ( omp_tracing ) pomp_for_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_master_begin_(int* id) {
  if ( omp_tracing ) pomp_master_begin(pomp_rd_table[*id]);
}

void pomp_master_begin__(int* id) {
  if ( omp_tracing ) pomp_master_begin(pomp_rd_table[*id]);
}

void POMP_MASTER_BEGIN(int* id) {
  if ( omp_tracing ) pomp_master_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_master_end_(int* id) {
  if ( omp_tracing ) pomp_master_end(pomp_rd_table[*id]);
}

void pomp_master_end__(int* id) {
  if ( omp_tracing ) pomp_master_end(pomp_rd_table[*id]);
}

void POMP_MASTER_END(int* id) {
  if ( omp_tracing ) pomp_master_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_begin_(int* id) {
  if ( omp_tracing ) pomp_parallel_begin(pomp_rd_table[*id]);
}

void pomp_parallel_begin__(int* id) {
  if ( omp_tracing ) pomp_parallel_begin(pomp_rd_table[*id]);
}

void POMP_PARALLEL_BEGIN(int* id) {
  if ( omp_tracing ) pomp_parallel_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_end_(int* id) {
  if ( omp_tracing ) pomp_parallel_end(pomp_rd_table[*id]);
}

void pomp_parallel_end__(int* id) {
  if ( omp_tracing ) pomp_parallel_end(pomp_rd_table[*id]);
}

void POMP_PARALLEL_END(int* id) {
  if ( omp_tracing ) pomp_parallel_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_fork_(int* id) {
  if ( omp_tracing ) pomp_parallel_fork(pomp_rd_table[*id]);
}

void pomp_parallel_fork__(int* id) {
  if ( omp_tracing ) pomp_parallel_fork(pomp_rd_table[*id]);
}

void POMP_PARALLEL_FORK(int* id) {
  if ( omp_tracing ) pomp_parallel_fork(pomp_rd_table[*id]);
}

/****************************/

void pomp_parallel_join_(int* id) {
  if ( omp_tracing ) pomp_parallel_join(pomp_rd_table[*id]);
}

void pomp_parallel_join__(int* id) {
  if ( omp_tracing ) pomp_parallel_join(pomp_rd_table[*id]);
}

void POMP_PARALLEL_JOIN(int* id) {
  if ( omp_tracing ) pomp_parallel_join(pomp_rd_table[*id]);
}

/****************************/

void pomp_section_begin_(int* id) {
  if ( omp_tracing ) pomp_section_begin(pomp_rd_table[*id]);
}

void pomp_section_begin__(int* id) {
  if ( omp_tracing ) pomp_section_begin(pomp_rd_table[*id]);
}

void POMP_SECTION_BEGIN(int* id) {
  if ( omp_tracing ) pomp_section_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_section_end_(int* id) {
  if ( omp_tracing ) pomp_section_end(pomp_rd_table[*id]);
}

void pomp_section_end__(int* id) {
  if ( omp_tracing ) pomp_section_end(pomp_rd_table[*id]);
}

void POMP_SECTION_END(int* id) {
  if ( omp_tracing ) pomp_section_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_sections_enter_(int* id) {
  if ( omp_tracing ) pomp_sections_enter(pomp_rd_table[*id]);
}

void pomp_sections_enter__(int* id) {
  if ( omp_tracing ) pomp_sections_enter(pomp_rd_table[*id]);
}

void POMP_SECTIONS_ENTER(int* id) {
  if ( omp_tracing ) pomp_sections_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_sections_exit_(int* id) {
  if ( omp_tracing ) pomp_sections_exit(pomp_rd_table[*id]);
}

void pomp_sections_exit__(int* id) {
  if ( omp_tracing ) pomp_sections_exit(pomp_rd_table[*id]);
}

void POMP_SECTIONS_EXIT(int* id) {
  if ( omp_tracing ) pomp_sections_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_begin_(int* id) {
  if ( omp_tracing ) pomp_single_begin(pomp_rd_table[*id]);
}

void pomp_single_begin__(int* id) {
  if ( omp_tracing ) pomp_single_begin(pomp_rd_table[*id]);
}

void POMP_SINGLE_BEGIN(int* id) {
  if ( omp_tracing ) pomp_single_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_end_(int* id) {
  if ( omp_tracing ) pomp_single_end(pomp_rd_table[*id]);
}

void pomp_single_end__(int* id) {
  if ( omp_tracing ) pomp_single_end(pomp_rd_table[*id]);
}

void POMP_SINGLE_END(int* id) {
  if ( omp_tracing ) pomp_single_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_enter_(int* id) {
  if ( omp_tracing ) pomp_single_enter(pomp_rd_table[*id]);
}

void pomp_single_enter__(int* id) {
  if ( omp_tracing ) pomp_single_enter(pomp_rd_table[*id]);
}

void POMP_SINGLE_ENTER(int* id) {
  if ( omp_tracing ) pomp_single_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_single_exit_(int* id) {
  if ( omp_tracing ) pomp_single_exit(pomp_rd_table[*id]);
}

void pomp_single_exit__(int* id) {
  if ( omp_tracing ) pomp_single_exit(pomp_rd_table[*id]);
}

void POMP_SINGLE_EXIT(int* id) {
  if ( omp_tracing ) pomp_single_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_workshare_enter_(int* id) {
  if ( omp_tracing ) pomp_workshare_enter(pomp_rd_table[*id]);
}

void pomp_workshare_enter__(int* id) {
  if ( omp_tracing ) pomp_workshare_enter(pomp_rd_table[*id]);
}

void POMP_WORKSHARE_ENTER(int* id) {
  if ( omp_tracing ) pomp_workshare_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_workshare_exit_(int* id) {
  if ( omp_tracing ) pomp_workshare_exit(pomp_rd_table[*id]);
}

void pomp_workshare_exit__(int* id) {
  if ( omp_tracing ) pomp_workshare_exit(pomp_rd_table[*id]);
}

void POMP_WORKSHARE_EXIT(int* id) {
  if ( omp_tracing ) pomp_workshare_exit(pomp_rd_table[*id]);
}

/****************************/

void pomp_begin_(int* id) {
  if ( omp_tracing ) pomp_begin(pomp_rd_table[*id]);
}

void pomp_begin__(int* id) {
  if ( omp_tracing ) pomp_begin(pomp_rd_table[*id]);
}

void POMP_BEGIN(int* id) {
  if ( omp_tracing ) pomp_begin(pomp_rd_table[*id]);
}

/****************************/

void pomp_end_(int* id) {
  if ( omp_tracing ) pomp_end(pomp_rd_table[*id]);
}

void pomp_end__(int* id) {
  if ( omp_tracing ) pomp_end(pomp_rd_table[*id]);
}

void POMP_END(int* id) {
  if ( omp_tracing ) pomp_end(pomp_rd_table[*id]);
}

/****************************/

void pomp_flush_enter_(int* id) {
  if ( omp_tracing ) pomp_flush_enter(pomp_rd_table[*id]);
}

void pomp_flush_enter__(int* id) {
  if ( omp_tracing ) pomp_flush_enter(pomp_rd_table[*id]);
}

void POMP_FLUSH_ENTER(int* id) {
  if ( omp_tracing ) pomp_flush_enter(pomp_rd_table[*id]);
}

/****************************/

void pomp_flush_exit_(int* id) {
  if ( omp_tracing ) pomp_flush_exit(pomp_rd_table[*id]);
}

void pomp_flush_exit__(int* id) {
  if ( omp_tracing ) pomp_flush_exit(pomp_rd_table[*id]);
}

void POMP_FLUSH_EXIT(int* id) {
  if ( omp_tracing ) pomp_flush_exit(pomp_rd_table[*id]);
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


void TauStartOpenMPRegionTimer(struct ompregdescr *r)
{
  static int tau_openmp_initialized = tau_openmp_init();
/* For any region, create a mapping between a region r and timer t and
   start the timer. */

  omp_set_lock(&tau_ompregdescr_lock);
  if (r->data)
  {
    Profiler *p = new Profiler((FunctionInfo *) r->data, OpenMP, true, RtsLayer::myThread());
    p->Start();
  }
  else
  {
    char rname[256], rtype[1024];
    snprintf(rname, sizeof(rname),  "%s %s", r->name, r->sub_name);
    snprintf(rtype, sizeof(rtype),  "[OpenMP location: file:%s <%d, %d>]",
        r->file_name, r->begin_first_line, r->end_last_line);

    FunctionInfo *f = new FunctionInfo(rname, rtype, OpenMP, "OpenMP");
    r->data = (void *) f;
    Profiler *p = new Profiler (f, OpenMP, true, RtsLayer::myThread());
    p->Start();
  }
  omp_unset_lock(&tau_ompregdescr_lock);

}

/* pomp library calls */

void pomp_finalize() {
  if ( ! omp_fin_called ) {
    omp_fin_called = 1;
  }
#ifdef DEBUG_PROF
  TAU_VERBOSE( "  0: finalize\n");
#endif /* DEBUG_PROF */
}

void pomp_init() {
  int i;

  atexit(pomp_finalize);

#ifdef DEBUG_PROF
  TAU_VERBOSE( "  0: init\n");
#endif /* DEBUG_PROF */

  for(i=0; i<POMP_MAX_ID; ++i) {
    if ( pomp_rd_table[i] ) {
      pomp_rd_table[i]->data = 0; /* allocate space for performance data here */
    }
  }
}

void pomp_off() {
  TAU_DISABLE_INSTRUMENTATION();
  omp_tracing = 0;
}

void pomp_on() {
  TAU_ENABLE_INSTRUMENTATION();
  omp_tracing = 1;
}

void pomp_atomic_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tatomic);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: enter atomic\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_atomic_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: exit  atomic\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_barrier_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tbarrier);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    if ( r->name[0] == 'b' )
      TAU_VERBOSE( "%3d: enter barrier\n", omp_get_thread_num());
    else
      TAU_VERBOSE( "%3d: enter implicit barrier of %s\n",
	      omp_get_thread_num(), r->name);
  }
#endif /* DEBUG_PROF */
}

void pomp_barrier_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    if ( r->name[0] == 'b' )
      TAU_VERBOSE( "%3d: exit  barrier\n", omp_get_thread_num());
    else
      TAU_VERBOSE( "%3d: exit  implicit barrier of %s\n",
	      omp_get_thread_num(), r->name);
  }
#endif /* DEBUG_PROF */
}

void pomp_critical_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tcriticalb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: begin critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void pomp_critical_end(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: end   critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void pomp_critical_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tcriticale);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: enter critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void pomp_critical_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: exit  critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

	
void pomp_for_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tfor); 
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: enter for\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_for_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  // as in a stack. lifo
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: exit  for\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_master_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tmaster);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: begin master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_master_end(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: end   master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_parallel_begin(struct ompregdescr* r) {

/* if there is no top level timer, create it */
  Tau_create_top_level_timer_if_necessary();
#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tparallelb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: begin parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_parallel_end(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: end   parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_parallel_fork(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tparallelf);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: fork  parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_parallel_join(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: join  parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_section_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsectionb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: begin section\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_section_end(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: end   section\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_sections_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsectione);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: enter sections\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_sections_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: exit  sections\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_single_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsingleb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: begin single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_single_end(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: end   single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_single_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tsinglee);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: enter single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_single_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: exit  single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_workshare_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tworkshare);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: enter workshare\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_workshare_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */
  
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: exit  workshare\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tregion);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: begin region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void pomp_end(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
  
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: end   region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}


void pomp_flush_enter(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tregion);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r);
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: flush enter region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void pomp_flush_exit(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_STOP(); /* global timer stop */
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */
 
#ifdef TAU_OPENMP_REGION_VIEW
  TAU_GLOBAL_TIMER_STOP(); /* region timer stop */
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: flush exit   region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}


void pomp_init_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_init_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: init lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_lock(s);
}

void pomp_destroy_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_destroy_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: destroy lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_lock(s);
}

void pomp_set_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_set_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: set lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_lock(s);
}

void pomp_unset_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_unset_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: unset lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_lock(s);
}

int  pomp_test_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_test_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: test lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_lock(s);
}

void pomp_init_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_init_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: init nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_nest_lock(s);
}

void pomp_destroy_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_destroy_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: destroy nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_nest_lock(s);
}

void pomp_set_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_set_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: set nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_nest_lock(s);
}

void pomp_unset_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_unset_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: unset nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_nest_lock(s);
}

int  pomp_test_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_test_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    TAU_VERBOSE( "%3d: test nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_nest_lock(s);
}


/***************************************************************************
 * $RCSfile: TauOpari.cpp,v $   $Author: sameer $
 * $Revision: 1.9 $   $Date: 2005/10/31 23:45:00 $
 * POOMA_VERSION_ID: $Id: TauOpari.cpp,v 1.9 2005/10/31 23:45:00 sameer Exp $
 ***************************************************************************/


