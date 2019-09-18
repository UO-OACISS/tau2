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

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 1024
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

void freeMatrix(double** matrix, int rows, int cols) {
  int i;
  for (i=0; i<rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

/////////////////////////////////////////////////////////////////////
// compute multiplies a and b and returns the result in c using ijk. 
// cols_a and rows_b are the same value
/////////////////////////////////////////////////////////////////////
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  printf("%s\n", __func__);
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

///////////////////////////////////////////////////////////////////////
// compute_interchange multiplies a and b and returns the result in c 
// using ikj loop.  cols_a and rows_b are the same value
///////////////////////////////////////////////////////////////////////
void compute_interchange(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  printf("%s\n", __func__);
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

///////////////////////////////////////////////////////////////////////
// compute_interchange multiplies a and b and returns the result in c 
// using ikj loop.  cols_a and rows_b are the same value
///////////////////////////////////////////////////////////////////////
void compute_target(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  printf("%s\n", __func__);
#pragma omp target parallel map(to: a, b) map(tofrom: c)
  {
    int i,j,k;
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for
    for (i=0; i<rows_a; i++) {
      for (k=0; k<cols_a; k++) {
        for(j=0; j<cols_b; j++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }   /*** End of parallel region ***/
#if 0
  printf("%s target teams distribute \n", __func__);
  {
    int i,j,k;
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp target teams distribute parallel for map(to: a, b) map(tofrom: c) \
                                                    thread_limit(16)
    for (i=0; i<rows_a; i++) {
      for (k=0; k<cols_a; k++) {
        for(j=0; j<cols_b; j++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }   /*** End of parallel region ***/
#endif
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
  compute_target(a, b, c, NRA, NCA, NCB);

  double result = c[0][1];

  freeMatrix(a, NRA, NCA);
  freeMatrix(b, NCA, NCB);
  freeMatrix(c, NCA, NCB);

  return result;
}

int main (int argc, char *argv[]) 
{
  int i;
  for (i = 0 ; i < 1 ; i++) {
    do_work();
  }

  printf ("Done.\n");

  return 0;
}

