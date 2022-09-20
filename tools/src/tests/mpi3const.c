#include <mpi.h>

#ifndef TAU_MPICH3_CONST
#define TAU_MPICH3_CONST
#endif

int  MPI_Send(TAU_MPICH3_CONST void * buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm)
{
  return PMPI_Send(buf, count, datatype, dest, tag, comm);
}

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
