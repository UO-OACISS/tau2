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

#include "matmult_initialize.h"

#ifdef TAU_MPI
#include <mpi.h>
/* NOTE: MPI is just used to spawn multiple copies of the kernel to different ranks.
This is not a parallel implementation */
#endif /* TAU_MPI */

#ifdef PTHREADS
#include <pthread.h>
#endif /* PTHREADS */

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 500
#endif

#define NRA MATRIX_SIZE                 /* number of rows in matrix A */
#define NCA MATRIX_SIZE                 /* number of columns in matrix A */
#define NCB MATRIX_SIZE                 /* number of columns in matrix B */

double** allocateMatrix(int rows, int cols) {
  int i;
  double **matrix = malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = malloc((sizeof(double)) * cols);
  }
  return matrix;
}

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
          c[i][j] += a[i][k] * b[k][j];
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

  compute(a, b, c, NRA, NCA, NCB);
  compute_interchange(a, b, c, NRA, NCA, NCB);

  return c[0][1]; 
}

void * threaded_func(void *data)
{
  do_work();
}

int main (int argc, char *argv[]) 
{

#ifdef PTHREADS
  int ret;
  pthread_attr_t  attr;
  pthread_t       tid1, tid2, tid3;
#endif /* PTHREADS */


#ifdef TAU_MPI
  MPI_Init(&argc, &argv); 
#endif /* TAU_MPI */

#ifdef PTHREADS
  if (ret = pthread_create(&tid1, NULL, threaded_func, NULL) )
  {
    printf("Error: pthread_create (1) fails ret = %d\n", ret);
    exit(1);
  }   

  if (ret = pthread_create(&tid2, NULL, threaded_func, NULL) )
  {
    printf("Error: pthread_create (2) fails ret = %d\n", ret);
    exit(1);
  }   

  if (ret = pthread_create(&tid3, NULL, threaded_func, NULL) )
  {
    printf("Error: pthread_create (3) fails ret = %d\n", ret);
    exit(1);
  }   

#endif /* PTHREADS */

/* On thread 0: */
  do_work();

#ifdef PTHREADS 
  if (ret = pthread_join(tid1, NULL) )
  {
    printf("Error: pthread_join (1) fails ret = %d\n", ret);
    exit(1);
  }   

  if (ret = pthread_join(tid2, NULL) )
  {
    printf("Error: pthread_join (2) fails ret = %d\n", ret);
    exit(1);
  }   

  if (ret = pthread_join(tid3, NULL) )
  {
    printf("Error: pthread_join (3) fails ret = %d\n", ret);
    exit(1);
  }   

#endif /* PTHREADS */

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */
  printf ("Done.\n");

  return 0;
}

