#include <mpi.h>

#ifndef MPI_VERSION
#define MPI_VERSION 2
#endif

#if MPI_VERSION < 3

int  MPI_Type_hindexed( int count, const int * blocklens, const MPI_Aint * indices, MPI_Datatype old_type, MPI_Datatype * newtype)
{
return PMPI_Type_hindexed( count, blocklens, indices, old_type, newtype );
}

#endif

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}