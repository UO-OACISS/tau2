#include <mpi.h>
extern mpi_init_(void*,void*);
extern mpi_init__(void*,void*);
static void unused() {
  MPI_Init(0,0);
  mpi_init_(0,0);
  mpi_init__(0,0);
}
