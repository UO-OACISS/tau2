/*************************************************************************/
/* TAU OPARI Layer							 */
/* Copyright (C) 2001                                                    */
/* University of Oregon, Los Alamos National Laboratory, and 		 */
/* Forschungszentrum Juelich, Zentralinstitut fuer Angewandte Mathematik */
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>

//#define DEBUG_PROF 1
#include "pomp_lib.h"
#include <Profile/Profiler.h>

/* These two defines specify if we want region based views or construct based
views or both */

#define TAU_AGGREGATE_OPENMP_TIMINGS
#define TAU_OPENMP_REGION_VIEW

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

#define FSUB(name) name##_

static int omp_tracing    = 1;
static int omp_fin_called = 0;

/*
 * Fortran wrappers calling the C versions
 */

void FSUB(pomp_finalize)() {
  pomp_finalize();
}

void FSUB(pomp_init)() {
  pomp_init();
}

void FSUB(pomp_off)() {
  omp_tracing = 0;
}

void FSUB(pomp_on)() {
  omp_tracing = 1;
}

void FSUB(pomp_atomic_enter)(int* id) {
  if ( omp_tracing ) pomp_atomic_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_atomic_exit)(int* id) {
  if ( omp_tracing ) pomp_atomic_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_barrier_enter)(int* id) {
  if ( omp_tracing ) pomp_barrier_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_barrier_exit)(int* id) {
  if ( omp_tracing ) pomp_barrier_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_critical_begin)(int* id) {
  if ( omp_tracing ) pomp_critical_begin(pomp_rd_table[*id]);
}

void FSUB(pomp_critical_end)(int* id) {
  if ( omp_tracing ) pomp_critical_end(pomp_rd_table[*id]);
}

void FSUB(pomp_critical_enter)(int* id) {
  if ( omp_tracing ) pomp_critical_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_critical_exit)(int* id) {
  if ( omp_tracing ) pomp_critical_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_do_enter)(int* id) {
  if ( omp_tracing ) pomp_for_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_do_exit)(int* id) {
  if ( omp_tracing ) pomp_for_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_master_begin)(int* id) {
  if ( omp_tracing ) pomp_master_begin(pomp_rd_table[*id]);
}

void FSUB(pomp_master_end)(int* id) {
  if ( omp_tracing ) pomp_master_end(pomp_rd_table[*id]);
}

void FSUB(pomp_parallel_begin)(int* id) {
  if ( omp_tracing ) pomp_parallel_begin(pomp_rd_table[*id]);
}

void FSUB(pomp_parallel_end)(int* id) {
  if ( omp_tracing ) pomp_parallel_end(pomp_rd_table[*id]);
}

void FSUB(pomp_parallel_fork)(int* id) {
  if ( omp_tracing ) pomp_parallel_fork(pomp_rd_table[*id]);
}

void FSUB(pomp_parallel_join)(int* id) {
  if ( omp_tracing ) pomp_parallel_join(pomp_rd_table[*id]);
}

void FSUB(pomp_section_begin)(int* id) {
  if ( omp_tracing ) pomp_section_begin(pomp_rd_table[*id]);
}

void FSUB(pomp_section_end)(int* id) {
  if ( omp_tracing ) pomp_section_end(pomp_rd_table[*id]);
}

void FSUB(pomp_sections_enter)(int* id) {
  if ( omp_tracing ) pomp_sections_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_sections_exit)(int* id) {
  if ( omp_tracing ) pomp_sections_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_single_begin)(int* id) {
  if ( omp_tracing ) pomp_single_begin(pomp_rd_table[*id]);
}

void FSUB(pomp_single_end)(int* id) {
  if ( omp_tracing ) pomp_single_end(pomp_rd_table[*id]);
}

void FSUB(pomp_single_enter)(int* id) {
  if ( omp_tracing ) pomp_single_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_single_exit)(int* id) {
  if ( omp_tracing ) pomp_single_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_workshare_enter)(int* id) {
  if ( omp_tracing ) pomp_workshare_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_workshare_exit)(int* id) {
  if ( omp_tracing ) pomp_workshare_exit(pomp_rd_table[*id]);
}

void FSUB(pomp_begin)(int* id) {
  if ( omp_tracing ) pomp_begin(pomp_rd_table[*id]);
}

void FSUB(pomp_end)(int* id) {
  if ( omp_tracing ) pomp_end(pomp_rd_table[*id]);
}

void FSUB(pomp_flush_enter)(int* id) {
  if ( omp_tracing ) pomp_flush_enter(pomp_rd_table[*id]);
}

void FSUB(pomp_flush_exit)(int* id) {
  if ( omp_tracing ) pomp_flush_exit(pomp_rd_table[*id]);
}


/*
 * C pomp function library
 */

/* TAU specific calls */
int tau_openmp_init(void)
{
  omp_init_lock(&tau_ompregdescr_lock);
  return 0;
}

static int tau_openmp_initialized = tau_openmp_init();

void TauStartOpenMPRegionTimer(struct ompregdescr *r)
{
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
    sprintf(rname, "%s %s", r->name, r->sub_name);
    sprintf(rtype, "[OpenMP location: file:%s <%d, %d>]",
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
  fprintf(stderr, "  0: finalize\n");
#endif /* DEBUG_PROF */
}

void pomp_init() {
  int i;

  atexit(pomp_finalize);

#ifdef DEBUG_PROF
  fprintf(stderr, "  0: init\n");
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
    fprintf(stderr, "%3d: enter atomic\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: exit  atomic\n", omp_get_thread_num());
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
      fprintf(stderr, "%3d: enter barrier\n", omp_get_thread_num());
    else
      fprintf(stderr, "%3d: enter implicit barrier of %s\n",
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
      fprintf(stderr, "%3d: exit  barrier\n", omp_get_thread_num());
    else
      fprintf(stderr, "%3d: exit  implicit barrier of %s\n",
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
    fprintf(stderr, "%3d: begin critical %s\n",
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
    fprintf(stderr, "%3d: end   critical %s\n",
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
    fprintf(stderr, "%3d: enter critical %s\n",
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
    fprintf(stderr, "%3d: exit  critical %s\n",
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
    fprintf(stderr, "%3d: enter for\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: exit  for\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: begin master\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: end   master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void pomp_parallel_begin(struct ompregdescr* r) {

#ifdef TAU_AGGREGATE_OPENMP_TIMINGS
  TAU_GLOBAL_TIMER_START(tparallelb);
#endif /* TAU_AGGREGATE_OPENMP_TIMINGS */

#ifdef TAU_OPENMP_REGION_VIEW
  TauStartOpenMPRegionTimer(r); 
#endif /* TAU_OPENMP_REGION_VIEW */

#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin parallel\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: end   parallel\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: fork  parallel\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: join  parallel\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: begin section\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: end   section\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: enter sections\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: exit  sections\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: begin single\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: end   single\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: enter single\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: exit  single\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: enter workshare\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: exit  workshare\n", omp_get_thread_num());
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
    fprintf(stderr, "%3d: begin region %s\n",
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
    fprintf(stderr, "%3d: end   region %s\n",
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
    fprintf(stderr, "%3d: flush enter region %s\n",
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
    fprintf(stderr, "%3d: flush exit   region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}


void pomp_init_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_init_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: init lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_lock(s);
}

void pomp_destroy_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_destroy_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: destroy lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_lock(s);
}

void pomp_set_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_set_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: set lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_lock(s);
}

void pomp_unset_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_unset_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: unset lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_lock(s);
}

int  pomp_test_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_test_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: test lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_lock(s);
}

void pomp_init_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_init_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: init nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_nest_lock(s);
}

void pomp_destroy_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_destroy_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: destroy nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_nest_lock(s);
}

void pomp_set_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_set_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: set nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_nest_lock(s);
}

void pomp_unset_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_unset_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: unset nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_nest_lock(s);
}

int  pomp_test_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_test_nest_lock", "[OpenMP]", OpenMP);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: test nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_nest_lock(s);
}


/***************************************************************************
 * $RCSfile: TauOpari.cpp,v $   $Author: sameer $
 * $Revision: 1.6 $   $Date: 2001/10/22 17:57:54 $
 * POOMA_VERSION_ID: $Id: TauOpari.cpp,v 1.6 2001/10/22 17:57:54 sameer Exp $
 ***************************************************************************/


