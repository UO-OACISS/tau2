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

#ifdef TAU_MPI
#include <mpi.h>
#endif /* TAU_MPI */

#define ITERATIONS 1000
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

double do_work(void) {
  double **a,           /* matrix A to be multiplied */
         **b,           /* matrix B to be multiplied */
         **c;           /* result matrix C */
  a = allocateMatrix(NRA, NCA);
  b = allocateMatrix(NCA, NCB);
  c = allocateMatrix(NRA, NCB);  

  freeMatrix(a, NRA, NCA);
  freeMatrix(b, NCA, NCB);
  freeMatrix(c, NCA, NCB);

  return 1.0;
}

int main (int argc, char *argv[]) 
{
#ifdef TAU_MPI
  int rc = MPI_Init(&argc, &argv); 
  if (rc != MPI_SUCCESS) {
    char *errorstring;
    int length = 0;
    MPI_Error_string(rc, errorstring, &length);
    printf("Error: MPI_Init failed, rc = %d\n%s\n", rc, errorstring);
    exit(1);
  }
#endif

  int i;
  for (i = 0 ; i < ITERATIONS ; i++) {
    if(i % 100 == 0) { printf("Iteration %d\n", i); }
    do_work();
  }

  printf ("Done.\n");

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */
  return 0;
}

