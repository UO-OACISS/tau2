#include <mpi.h>

#ifndef TAU_MPICH3_CONST
#define TAU_MPICH3_CONST
#endif

int  MPI_Get_address( TAU_MPICH3_CONST void * location, MPI_Aint * address)
{
  return PMPI_Get_address( location, address );
}

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
