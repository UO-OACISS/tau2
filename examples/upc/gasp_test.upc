/* Copyright University of Florida, UC Berkeley upc group */

#include <stdio.h>
#include <upc.h>
#ifdef __UPC_COLLECTIVE__
#include <upc_collective.h>
#endif
#ifndef __UPC_PUPC__
#error __UPC_PUPC__ not defined!
#endif
#include <pupc.h>

#include <stdio.h>
#include <stdlib.h>

void bar();

void foo() {
  upc_barrier 2000;
}

int dummy0;
shared int si;
shared [10] double sba[100*THREADS];

#pragma pupc off
int dummy1;
void bar2() {
  dummy0++;
  upc_barrier 66666;
}

void bar() {
  dummy1++;
  upc_barrier 6666;
  bar2();
}
#pragma pupc on

void collectives() {
  #define SZ 100
  static shared int A[THREADS];
  static shared int B[THREADS];
  static shared [] int *C;
  static shared [SZ] int D[SZ*THREADS];
  static shared void *E;
  static shared void *F;
  static shared [SZ] int G[SZ*THREADS];
  C = upc_all_alloc(1, THREADS*SZ*sizeof(int));
  E = upc_all_alloc(THREADS*THREADS, SZ*sizeof(int));
  F = upc_all_alloc(THREADS*THREADS, SZ*sizeof(int));
  upc_barrier;
#ifdef __UPC_COLLECTIVE__
  upc_all_broadcast( B, &A[1], sizeof(int), UPC_IN_ALLSYNC|UPC_OUT_ALLSYNC);
  upc_barrier;
  upc_all_scatter( D, C, sizeof(int)*SZ, UPC_IN_MYSYNC|UPC_OUT_MYSYNC);
  upc_barrier;
  upc_all_gather( C, D, sizeof(int)*SZ, UPC_IN_NOSYNC|UPC_OUT_NOSYNC);
  upc_barrier;
  upc_all_gather_all( E, D, sizeof(int)*SZ, 0);
  upc_barrier;
  upc_all_exchange( E, F, sizeof(int)*SZ, UPC_IN_MYSYNC|UPC_OUT_ALLSYNC);
  upc_barrier;
  A[MYTHREAD] = (MYTHREAD+1)%THREADS;
  upc_all_permute( D, G, A, sizeof(int)*SZ, UPC_IN_ALLSYNC|UPC_OUT_MYSYNC);
  upc_barrier;
  upc_all_reduceI( A, D, UPC_ADD, SZ*THREADS, SZ, NULL, UPC_IN_ALLSYNC|UPC_OUT_NOSYNC);
  upc_barrier;
  upc_all_prefix_reduceI( A, D, UPC_MULT, SZ*THREADS, SZ, NULL, UPC_IN_NOSYNC|UPC_OUT_MYSYNC);
  upc_barrier;
#endif
}

int junk = 0;

int main() {
  int v = 0;
  int me = MYTHREAD;
  double d = 1.34;
  int x = 4;
  unsigned int evt = pupc_create_event("USER1", "dval=%f ival=%i");
  unsigned int evt2 = pupc_create_event("USER2", "phase %s, iter %i");
  upc_barrier 100;
  printf("Hello from %i/%i\n",MYTHREAD,THREADS);
  upc_barrier 200;
  shared [10] int *p = upc_all_alloc(10,10*sizeof(int));
  shared int *p2 = upc_global_alloc(20,sizeof(int));
  shared int *p3 = upc_alloc(sizeof(int));
  upc_memput(p,(void*)p3,sizeof(int));
  upc_memget((void*)p3,p,sizeof(int));
  upc_memcpy(p3,p2,sizeof(int));
  upc_free(p2);
  upc_free(p3);
  void *lp = malloc(10);
  free(lp);
  void *lp2 = calloc(1,10);
  lp2 = realloc(lp2, 100);
  free(lp2);
  shared [10] strict int *sp = p+1;
  shared relaxed int *rp;
  rp = (shared relaxed int *)p;
  junk++;
  *rp = 2;
  junk++;
  *sp = 1;
  junk++;
  upc_fence;
  junk++;
  v += *rp;
  junk++;
  v += *sp;
  junk++;
  *sp = junk+v;
#ifdef __BERKELEY_UPC__
  bupc_handle_t h1 = bupc_memput_async(p,(void*)p3,sizeof(int));
  bupc_handle_t h2 = bupc_memget_async((void*)p3,p,sizeof(int));
  bupc_handle_t h3 = bupc_memcpy_async(p3,p,sizeof(int));
  bupc_waitsync(h1);
  while (!bupc_trysync(h2));
  bupc_waitsync(h3);
#endif
  upc_lock_t *l1 = upc_global_lock_alloc();
  upc_lock_t *l2 = upc_all_lock_alloc();
  int val = upc_lock_attempt(l1);
  upc_lock(l2);
  upc_unlock(l2);
  upc_lock_free(l1);
  for (int i=0;i<5;i++) {
    if (i == 2) pupc_control(0);
    pupc_event_atomic(evt2,"doit",i);
    pupc_event_start(evt,d,x);
    d *= 1.293;
    x *= 2;
    pupc_event_end(evt,d,x);
    upc_barrier i;
    if (i == 2) pupc_control(1);
  }
  upc_barrier 1000;
#pragma pupc off
  upc_barrier 666;
  foo();
#pragma pupc on
  upc_barrier 3000;
  upc_notify 4000;
  upc_wait 4000;
  collectives();
  upc_barrier;
  if (!MYTHREAD) printf("done.\n");
  /*upc_global_exit(10);*/
  /*exit(0); */
  return 0;
}
#pragma isolated_call foo

