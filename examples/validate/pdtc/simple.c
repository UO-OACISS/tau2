
#include <stdio.h>

#ifdef TAU_MPI
/* NOTE: MPI is just used to spawn multiple copies of the kernel to different ranks.
This is not a parallel implementation */
#include <mpi.h>
#endif /* TAU_MPI */

int main (int argc, char **argv) {

#ifdef TAU_MPI
  int provided;
  int rc = MPI_SUCCESS;
#if defined(PTHREADS)
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#elif defined(TAU_OPENMP)
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#else
  rc = MPI_Init(&argc, &argv); 
#endif /* THREADS */
  if (rc != MPI_SUCCESS) {
    char *errorstring;
    int length = 0;
    MPI_Error_string(rc, errorstring, &length);
    printf("Error: MPI_Init failed, rc = %d\n%s\n", rc, errorstring);
    exit(1);
  }
#endif /* TAU_MPI */

  printf ("Hello, world\n");

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */
  return 0;
}
