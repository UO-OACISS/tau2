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
int provided;
#include <mpi.h>
/* NOTE: MPI is just used to spawn multiple copies of the kernel to different ranks.
This is not a parallel implementation */
#endif /* TAU_MPI */

#define ITERATIONS 250

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 512
#endif

double** allocateMatrix(int rows, int cols) {
  int i;
  double **matrix = (double**)malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = (double*)malloc((sizeof(double)) * cols);
  }
  return matrix;
}

void freeMatrix(double** matrix, int rows, int cols) {
  int i;
  for (i=0; i<rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

#ifdef APP_USE_INLINE_MULTIPLY
__inline double multiply(double a, double b) {
	return a * b;
}
#endif /* APP_USE_INLINE_MULTIPLY */

#if 0
// cols_a and rows_b are the same value
void compute_nested(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  double tmp = 0.0;
//num_threads(2)
#pragma omp parallel private(i) shared(a,b,c) 
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait schedule(dynamic,1)
    for (i=0; i<rows_a; i++) {
//num_threads(2)
#pragma omp parallel private(i,j,k) shared(a,b,c) 
      {
#pragma omp for nowait schedule(dynamic,1)
        for (k=0; k<cols_a; k++) {
          for(j=0; j<cols_b; j++) {
#ifdef APP_USE_INLINE_MULTIPLY
              c[i][j] += multiply(a[i][k], b[k][j]);
#else 
              tmp = a[i][k];
			  tmp = tmp * b[k][j];
              c[i][j] += tmp;
#endif 
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
#pragma omp for schedule(dynamic) nowait
    for (i=0; i<rows_a; i++) {
      for(j=0; j<cols_b; j++) {
        for (k=0; k<cols_a; k++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else /* APP_USE_INLINE_MULTIPLY */
          c[i][j] += a[i][k] * b[k][j];
#endif /* APP_USE_INLINE_MULTIPLY */
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
#pragma omp for schedule(dynamic) nowait
    for (i=0; i<rows_a; i++) {
      for (k=0; k<cols_a; k++) {
        for(j=0; j<cols_b; j++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else /* APP_USE_INLINE_MULTIPLY */
          c[i][j] += a[i][k] * b[k][j];
#endif /* APP_USE_INLINE_MULTIPLY */
        }
      }
    }
  }   /*** End of parallel region ***/
}

double do_work(int matrix_size) {
  double **a,           /* matrix A to be multiplied */
  **b,           /* matrix B to be multiplied */
  **c;           /* result matrix C */
  a = allocateMatrix(matrix_size, matrix_size);
  b = allocateMatrix(matrix_size, matrix_size);
  c = allocateMatrix(matrix_size, matrix_size);  

/*** Spawn a parallel region explicitly scoping all variables ***/

  initialize(a, matrix_size, matrix_size);
  initialize(b, matrix_size, matrix_size);
  initialize(c, matrix_size, matrix_size);

#ifdef TAU_MPI
  //MPI_Barrier(MPI_COMM_WORLD);
#endif /* TAU_MPI */

  compute(a, b, c, matrix_size, matrix_size, matrix_size);

#ifdef TAU_MPI
  //MPI_Barrier(MPI_COMM_WORLD);
#endif /* TAU_MPI */

  compute_interchange(a, b, c, matrix_size, matrix_size, matrix_size);

  double result = c[0][0];

  freeMatrix(a, matrix_size, matrix_size);
  freeMatrix(b, matrix_size, matrix_size);
  freeMatrix(c, matrix_size, matrix_size);

#ifdef TAU_MPI
  //MPI_Barrier(MPI_COMM_WORLD);
#endif /* TAU_MPI */

  return result;
}

int main (int argc, char *argv[]) 
{

  int rank = 0;
#ifdef TAU_MPI
  int rc = MPI_SUCCESS;
  int comm_size = 0;
  rc = MPI_Init(&argc, &argv); 
  if (rc != MPI_SUCCESS) {
    char *errorstring;
    int length = 0;
    MPI_Error_string(rc, errorstring, &length);
    printf("Error: MPI_Init failed, rc = %d\n%s\n", rc, errorstring);
    exit(1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* TAU_MPI */

/* On thread 0: */
  int i;
  for (i = 0 ; i < ITERATIONS ; i++) {
    if(rank == 0) { printf("Iteration %d\n", i); }
    double ratio = ((double)rand() / (double)RAND_MAX);
    int matrix_size = (int)(ratio * (double)MATRIX_SIZE);
    // make sure we have at least 1 cell
    matrix_size = matrix_size > 32 ? matrix_size : 32;
    //printf("Matrix size = %d\n", matrix_size);
    do_work(matrix_size);
  }

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */
  if(rank == 0) { printf ("Done.\n"); }

  return 0;
}

