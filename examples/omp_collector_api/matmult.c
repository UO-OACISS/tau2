/******************************************************************************
*   OpenMp Example - Matrix Multiply - C Version
*   Demonstrates a matrix multiply using OpenMP. 
*
*   Modified from here:
*   https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
*
*   For  PAPI_FP_INS, the exclusive count for the event: 
*   for (null) [OpenMP location: file:matmult.c ]
*   should be  2E+06 / Number of Threads 
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "matmult_initialize.h"

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1000
#endif

#define NRA MATRIX_SIZE                 /* number of rows in matrix A */
#define NCA MATRIX_SIZE                 /* number of columns in matrix A */
#define NCB MATRIX_SIZE                 /* number of columns in matrix B */

double** allocateMatrix(int rows, int cols) {
  int i;
  double **matrix = (double**)malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = (double*)malloc((sizeof(double)) * cols);
  }
  return matrix;
}

__inline double multiply(double a, double b) {
	return a * b;
}

#ifdef OMP_NESTED
// cols_a and rows_b are the same value
void compute_nested(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i) shared(a,b,c) num_threads(2)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait
    for (i=0; i<rows_a; i++) {
#pragma omp parallel private(i,j,k) shared(a,b,c) num_threads(2)
      {
#pragma omp for nowait
      for(j=0; j<cols_b; j++) {
        for (k=0; k<cols_a; k++) {
          c[i][j] += multiply(a[i][k], b[k][j]);
        }
      }
      }
    }
  }   /*** End of parallel region ***/
}
#endif

// cols_a and rows_b are the same value
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait
    for (i=0; i<rows_a; i++) {
      for(j=0; j<cols_b; j++) {
        for (k=0; k<cols_a; k++) {
          c[i][j] += multiply(a[i][k], b[k][j]);
        }
      }
    }
  }   /*** End of parallel region ***/
}

// cols_a and rows_b are the same value
void compute_triangular(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait
    for (i=0; i<rows_a; i++) {
      for(j=i; j<cols_b; j++) {
        for (k=j; k<cols_a; k++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }   /*** End of parallel region ***/
}

void compute_interchange(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait
    for (i=0; i<rows_a; i++) {
      for (k=0; k<cols_a; k++) {
        for(j=0; j<cols_b; j++) {
          c[i][j] += multiply(a[i][k], b[k][j]);
        }
      }
    }
  }   /*** End of parallel region ***/
}

double do_work(void) {
  double **a,           /* matrix A to be multiplied */
  **b,           /* matrix B to be multiplied */
  **c;           /* result matrix C */
  a = allocateMatrix(NRA, NCA);
  b = allocateMatrix(NCA, NCB);
  c = allocateMatrix(NRA, NCB);  

/*** Spawn a parallel region explicitly scoping all variables ***/

  initialize(a, NRA, NCA);
  initialize(b, NCA, NCB);
  initialize(c, NRA, NCB);

  printf("Compute...\n");
  compute(a, b, c, NRA, NCA, NCB);
#ifdef OMP_NESTED
  if (omp_get_nested()) {
    compute_nested(a, b, c, NRA, NCA, NCB);
  }
#endif
  printf("Compute with interchange...\n");
  compute_interchange(a, b, c, NRA, NCA, NCB);
  printf("Sleep 1 second...\n");
  sleep(1);
  printf("Compute triangular...\n");
  compute_triangular(a, b, c, NRA, NCA, NCB);

  return c[0][1]; 
}

void * threaded_func(void *data)
{
  do_work();
  return NULL;
}

// also does master test
void do_barrier_test(void) {
  printf("\n*** PERFORMING BARRIER TEST ***\n\n");
  int a[5], i;

#pragma omp parallel
  {
    // Perform some computation.
#pragma omp for
    for (i = 0; i < 5; i++) {
      a[i] = i * i;
    }
    
    // Print intermediate results.
#pragma omp master
    for (i = 0; i < 5; i++) {
      printf("a[%d] = %d\n", i, a[i]);
    }
    
    // Wait.
#pragma omp barrier
    
    // Continue with the computation.
#pragma omp for
    for (i = 0; i < 5; i++) {
      a[i] += i;
    }
  }
}

void * do_lock_test(void) {
  printf("\n*** PERFORMING LOCK TEST ***\n\n");
  omp_lock_t my_lock;
  omp_init_lock(&my_lock);

#pragma omp parallel 
  {
    int tid = omp_get_thread_num( );
    int i, j;

    for (i = 0; i < 5; ++i) {
      omp_set_lock(&my_lock);
      printf("Thread %d - starting locked region\n", tid);
      printf("Thread %d - ending locked region\n", tid);
      omp_unset_lock(&my_lock);
    }
  }
  omp_destroy_lock(&my_lock);
}

void * do_critical_test(void) {
  printf("\n*** PERFORMING CRITICAL TEST ***\n\n");
  int i;
  int max;
  int a[MATRIX_SIZE];

  for (i = 0; i < MATRIX_SIZE; i++) 
  {
    a[i] = rand();
    //printf("%d\n", a[i]);
  }

  max = a[0];
#pragma omp parallel for
  for (i = 1; i < MATRIX_SIZE; i++) 
  {
    if (a[i] > max)
    {
#pragma omp critical
      {
// compare a[i] and max again because max 
// could have been changed by another thread after 
// the comparison outside the critical section
        if (a[i] > max)
          max = a[i];
      }
    }
  }
   
  printf("max = %d\n", max);
}

void do_sections_test(void) {
  printf("\n*** PERFORMING SECTIONS TEST ***\n\n");
#pragma omp parallel sections
  {
    printf("Hello from thread %d\n", omp_get_thread_num());
#pragma omp section
    printf("Hello from thread %d\n", omp_get_thread_num());
  }
}

void do_ordered_test(void) {
  printf("\n*** PERFORMING ORDERED TEST ***\n\n");
  int a[MATRIX_SIZE], i;
#pragma omp parallel
  {
#pragma omp ordered
  {
    // Perform some computation.
#pragma omp for
    for (i = 0; i < omp_get_num_threads(); i++) {
      a[i] = i * i;
    }
  }
  }
}

void * do_single_test(void) {
  printf("\n*** PERFORMING SINGLE TEST ***\n\n");
  int a[5], i;

#pragma omp parallel
  {
    // Perform some computation.
#pragma omp for
    for (i = 0; i < 5; i++) {
      a[i] = i * i;
    }
    
    // Print intermediate results.
#pragma omp single
    for (i = 0; i < 5; i++) {
      printf("a[%d] = %d\n", i, a[i]);
    }
    
    // Wait.
#pragma omp barrier
    
    // Continue with the computation.
#pragma omp for
    for (i = 0; i < 5; i++) {
      a[i] += i;
    }
  }
  return (a);
}

void * do_atomic_test(void) {
  printf("\n*** PERFORMING ATOMIC TEST ***\n\n");
  int count = 0;
#pragma omp parallel
  {
#pragma omp atomic
    count++;
  }
  printf("Number of threads: %d\n", count);
}


int main (int argc, char *argv[]) 
{
  //int i;
  //for (i = 0 ; i < 3 ; i++) {
    do_work();
  //}

  do_barrier_test();
  do_lock_test();
  do_critical_test();
  do_ordered_test();
  do_single_test();
  do_atomic_test();
  do_sections_test();

  printf ("Done.\n");

  return 0;
}

