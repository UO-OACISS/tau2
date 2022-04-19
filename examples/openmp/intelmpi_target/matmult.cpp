// matmul.cpp: Matrix Multiplication Example using OpenMP Offloading 
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef TAU_MPI
int provided;
#include <mpi.h>
/* NOTE: MPI is just used to spawn multiple copies of the kernel to different ranks.
This is not a parallel implementation */
#endif /* TAU_MPI */


#define MAX 128
int A[MAX][MAX], B[MAX][MAX], C[MAX][MAX], C_SERIAL[MAX][MAX];

typedef int BOOL;
typedef int TYPE;

BOOL check_result(TYPE *actual, TYPE *expected, unsigned n) {
    for (unsigned i = 0; i < n; i++) {
        if(actual[i] != expected[i]) {
            printf("Value mismatch at index = %d. Expected: %d"
                   ", Actual: %d.\n", i, expected[i], actual[i]);
            return 0;
        }
    }
    return 1;
}

void __attribute__ ((noinline)) Compute()
{
  #pragma omp target teams distribute parallel for map(to: A, B) map(tofrom: C) \
                                                   thread_limit(128)
  {
    for (int i = 0; i < MAX; i++)
    for (int j = 0; j < MAX; j++)
    for (int k = 0; k < MAX; k++)
         C[i][j] += A[i][k] * B[k][j];
  }
}

int main(int argc, char **argv) {


#ifdef TAU_MPI
  int rc; 
  rc = MPI_Init(&argc, &argv);
    
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
      if (provided == MPI_THREAD_MULTIPLE) {
        printf("provided is MPI_THREAD_MULTIPLE\n");
      } else if (provided == MPI_THREAD_FUNNELED) {
        printf("provided is MPI_THREAD_FUNNELED\n");
      }
  }
#endif /* TAU_MPI */


  for (int i = 0; i < MAX; i++)
    for (int j = 0; j < MAX; j++) {
      A[i][j] = i + j - 1;
      B[i][j] = i - j + 1;
    }

  for (int i = 0; i < MAX; i++)
  for (int j = 0; j < MAX; j++)
  for (int k = 0; k < MAX; k++)
       C_SERIAL[i][j] += A[i][k] * B[k][j];

  Compute();

  if (!check_result((int*) &C[0][0], (int*) &C_SERIAL[0][0], MAX * MAX)) {
    printf("FAILED\n");
    return 1;
  }

  printf("PASSED\n");

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */

  return 0;
}
