#include <mpi.h>
extern int mpi_init_(void*,void*);
extern int mpi_init__(void*,void*);
void tau_unused_routine() {
  MPI_Init(0,0);
  mpi_init_(0,0);
  mpi_init__(0,0);
}
