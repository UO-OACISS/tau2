/*************************************************************************/
/* TAU OPARI Layer							 */
/* Copyright (C) 2001                                                    */
/* University of Oregon, Los Alamos National Laboratory, and 		 */
/* Forschungszentrum Juelich, Zentralinstitut fuer Angewandte Mathematik */
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>

//#define DEBUG_PROF 1
#include "omperf_lib.h"
#include <Profile/Profiler.h>


TAU_GLOBAL_TIMER(tatomic, "atomic enter/exit", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tbarrier, "barrier enter/exit", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tcriticalb, "critical begin/end", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tcriticale, "critical enter/exit", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tfor, "for enter/exit", "[OpenMP]", TAU_DEFAULT);
TAU_GLOBAL_TIMER(tmaster, "master begin/end", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tparallelb, "parallel begin/end", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tparallelf, "parallel fork/join", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tsectionb, "section begin/end", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tsectione, "sections enter/exit", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tsingleb, "single begin/end", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tsinglee, "single enter/exit", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tworkshare, "workshare enter/exit", "[OpenMP]", TAU_DEFAULT); 
TAU_GLOBAL_TIMER(tregion, "region begin/end", "[OpenMP]", TAU_DEFAULT); 

#define FSUB(name) name##_

static int omp_tracing    = 1;
static int omp_fin_called = 0;

/*
 * Fortran wrappers calling the C versions
 */

void FSUB(omperf_finalize)() {
  omperf_finalize();
}

void FSUB(omperf_init)() {
  omperf_init();
}

void FSUB(omperf_off)() {
  omp_tracing = 0;
}

void FSUB(omperf_on)() {
  omp_tracing = 1;
}

void FSUB(omperf_atomic_enter)(int* id) {
  if ( omp_tracing ) omperf_atomic_enter(omp_rd_table[*id]);
}

void FSUB(omperf_atomic_exit)(int* id) {
  if ( omp_tracing ) omperf_atomic_exit(omp_rd_table[*id]);
}

void FSUB(omperf_barrier_enter)(int* id) {
  if ( omp_tracing ) omperf_barrier_enter(omp_rd_table[*id]);
}

void FSUB(omperf_barrier_exit)(int* id) {
  if ( omp_tracing ) omperf_barrier_exit(omp_rd_table[*id]);
}

void FSUB(omperf_critical_begin)(int* id) {
  if ( omp_tracing ) omperf_critical_begin(omp_rd_table[*id]);
}

void FSUB(omperf_critical_end)(int* id) {
  if ( omp_tracing ) omperf_critical_end(omp_rd_table[*id]);
}

void FSUB(omperf_critical_enter)(int* id) {
  if ( omp_tracing ) omperf_critical_enter(omp_rd_table[*id]);
}

void FSUB(omperf_critical_exit)(int* id) {
  if ( omp_tracing ) omperf_critical_exit(omp_rd_table[*id]);
}

void FSUB(omperf_do_enter)(int* id) {
  if ( omp_tracing ) omperf_for_enter(omp_rd_table[*id]);
}

void FSUB(omperf_do_exit)(int* id) {
  if ( omp_tracing ) omperf_for_exit(omp_rd_table[*id]);
}

void FSUB(omperf_master_begin)(int* id) {
  if ( omp_tracing ) omperf_master_begin(omp_rd_table[*id]);
}

void FSUB(omperf_master_end)(int* id) {
  if ( omp_tracing ) omperf_master_end(omp_rd_table[*id]);
}

void FSUB(omperf_parallel_begin)(int* id) {
  if ( omp_tracing ) omperf_parallel_begin(omp_rd_table[*id]);
}

void FSUB(omperf_parallel_end)(int* id) {
  if ( omp_tracing ) omperf_parallel_end(omp_rd_table[*id]);
}

void FSUB(omperf_parallel_fork)(int* id) {
  if ( omp_tracing ) omperf_parallel_fork(omp_rd_table[*id]);
}

void FSUB(omperf_parallel_join)(int* id) {
  if ( omp_tracing ) omperf_parallel_join(omp_rd_table[*id]);
}

void FSUB(omperf_section_begin)(int* id) {
  if ( omp_tracing ) omperf_section_begin(omp_rd_table[*id]);
}

void FSUB(omperf_section_end)(int* id) {
  if ( omp_tracing ) omperf_section_end(omp_rd_table[*id]);
}

void FSUB(omperf_sections_enter)(int* id) {
  if ( omp_tracing ) omperf_sections_enter(omp_rd_table[*id]);
}

void FSUB(omperf_sections_exit)(int* id) {
  if ( omp_tracing ) omperf_sections_exit(omp_rd_table[*id]);
}

void FSUB(omperf_single_begin)(int* id) {
  if ( omp_tracing ) omperf_single_begin(omp_rd_table[*id]);
}

void FSUB(omperf_single_end)(int* id) {
  if ( omp_tracing ) omperf_single_end(omp_rd_table[*id]);
}

void FSUB(omperf_single_enter)(int* id) {
  if ( omp_tracing ) omperf_single_enter(omp_rd_table[*id]);
}

void FSUB(omperf_single_exit)(int* id) {
  if ( omp_tracing ) omperf_single_exit(omp_rd_table[*id]);
}

void FSUB(omperf_workshare_enter)(int* id) {
  if ( omp_tracing ) omperf_workshare_enter(omp_rd_table[*id]);
}

void FSUB(omperf_workshare_exit)(int* id) {
  if ( omp_tracing ) omperf_workshare_exit(omp_rd_table[*id]);
}

void FSUB(omperf_begin)(int* id) {
  if ( omp_tracing ) omperf_begin(omp_rd_table[*id]);
}

void FSUB(omperf_end)(int* id) {
  if ( omp_tracing ) omperf_end(omp_rd_table[*id]);
}

/*
 * C omperf function library
 */

void omperf_finalize() {
  if ( ! omp_fin_called ) {
    omp_fin_called = 1;
  }
  fprintf(stderr, "  0: finalize\n");
}

void omperf_init() {
  int i;

  atexit(omperf_finalize);
  fprintf(stderr, "  0: init\n");

  for(i=0; i<OMPERF_MAX_ID; ++i) {
    if ( omp_rd_table[i] ) {
      omp_rd_table[i]->data = 0; /* allocate space for performance data here */
    }
  }
}

void omperf_off() {
  TAU_DISABLE_INSTRUMENTATION();
  omp_tracing = 0;
}

void omperf_on() {
  TAU_ENABLE_INSTRUMENTATION();
  omp_tracing = 1;
}

void omperf_atomic_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tatomic);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter atomic\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_atomic_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  atomic\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_barrier_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tbarrier);
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

void omperf_barrier_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
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

void omperf_critical_begin(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tcriticalb);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void omperf_critical_end(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void omperf_critical_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tcriticale);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void omperf_critical_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  critical %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

	

void omperf_for_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tfor); 
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter for\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_for_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP(); 
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  for\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_master_begin(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tmaster);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_master_end(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   master\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_parallel_begin(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tparallelb);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_parallel_end(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_parallel_fork(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tparallelf);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: fork  parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_parallel_join(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: join  parallel\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_section_begin(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tsectionb);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin section\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_section_end(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   section\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_sections_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tsectione);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter sections\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_sections_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  sections\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_single_begin(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tsingleb);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_single_end(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_single_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tsinglee);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_single_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  single\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_workshare_enter(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tworkshare);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: enter workshare\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_workshare_exit(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: exit  workshare\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
}

void omperf_begin(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_START(tregion);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: begin region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void omperf_end(struct ompregdescr* r) {
  TAU_GLOBAL_TIMER_STOP();
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: end   region %s\n",
            omp_get_thread_num(), r->sub_name);
  }
#endif /* DEBUG_PROF */
}

void omperf_init_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_init_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: init lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_lock(s);
}

void omperf_destroy_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_destroy_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: destroy lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_lock(s);
}

void omperf_set_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_set_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: set lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_lock(s);
}

void omperf_unset_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_unset_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: unset lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_lock(s);
}

int  omperf_test_lock(omp_lock_t *s) {
  TAU_PROFILE("omp_test_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: test lock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_lock(s);
}

void omperf_init_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_init_nest_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: init nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_init_nest_lock(s);
}

void omperf_destroy_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_destroy_nest_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: destroy nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_destroy_nest_lock(s);
}

void omperf_set_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_set_nest_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: set nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_set_nest_lock(s);
}

void omperf_unset_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_unset_nest_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: unset nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  omp_unset_nest_lock(s);
}

int  omperf_test_nest_lock(omp_nest_lock_t *s) {
  TAU_PROFILE("omp_test_nest_lock", "[OpenMP]", TAU_DEFAULT);
#ifdef DEBUG_PROF
  if ( omp_tracing ) {
    fprintf(stderr, "%3d: test nestlock\n", omp_get_thread_num());
  }
#endif /* DEBUG_PROF */
  return omp_test_nest_lock(s);
}

