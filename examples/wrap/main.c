#include <stdio.h>
#include <mpi.h>


int main(int argc, char **argv) {
  int x; 
  MPI_Init(&argc, &argv);

  x = foo(42);
   /* Call dgemm with different sizes */
  x = dgemm(100);
  x = dgemm(10000);
  x = dgemm(100000);
  printf("Inside main, x = %d\n", x);
  MPI_Finalize();
  return x;
}
