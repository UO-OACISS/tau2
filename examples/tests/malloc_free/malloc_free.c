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

#define ITERATIONS 10000
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

void Tau_track_memory_here(void);
void Tau_track_power_here(void);
void Tau_track_load_here(void);
void Tau_track_memory_rss_and_hwm_here(void);
void Tau_track_memory(void);
void Tau_track_power(void);
void Tau_track_load(void);
void Tau_track_memory_rss_and_hwm(void);

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

  /* records the heap, with no context, even though it says "here". */
  //Tau_track_memory_here();
  /* records the power, with context. */
  //Tau_track_power_here();
  /* records the load, with context. */
  //Tau_track_load_here();
  /* records the rss/hwm, with context. */
  //Tau_track_memory_rss_and_hwm_here();

  /* does nothing - just enables heap tracking  */
  //Tau_track_memory();
  /* records the load, without context */
  //Tau_track_load();
  /* records the power, without context */
  //Tau_track_power();
  /* records the rss/hwm, without context. */
  //Tau_track_memory_rss_and_hwm();

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
    if(i % 1000 == 0) { printf("Iteration %d\n", i); }
    do_work();
  }

  printf ("Done.\n");

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */
  return 0;
}

