#include <mpi.h>

extern int nv_main(int argc, char *argv[]);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  nv_main(argc, argv);
  MPI_Finalize();
  return 0;
}
