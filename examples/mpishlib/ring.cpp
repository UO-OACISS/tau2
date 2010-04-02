#include <Profile/Profiler.h>
#ifndef TAU_GROUP_RING
#define TAU_GROUP_RING TAU_GET_PROFILE_GROUP("RING")
#endif /* TAU_GROUP_RING */ 
#include <stdio.h>
#include <mpi.h>
#include "ring.h"

static const int anz = 512;
void C::method() {
  TAU_PROFILE("void C::method()", TAU_CT(*this), TAU_GROUP_RING);

    int i;
    int field[anz];
    MPI_Status status;

    for (i=0; i<anz; i++)
      field[i] = i;

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (me==0) {
      MPI_Send(&field, anz, MPI_INT, 1, 4711, MPI_COMM_WORLD);
      MPI_Recv(&field, anz, MPI_INT, proc-1, 4711, MPI_COMM_WORLD, &status);
    }
    else {
      MPI_Recv(&field, anz, MPI_INT, me-1, 4711, MPI_COMM_WORLD, &status);
      if (me == proc-1)
        MPI_Send(&field, anz, MPI_INT, 0, 4711, MPI_COMM_WORLD);
      else
        MPI_Send(&field, anz, MPI_INT, me+1, 4711, MPI_COMM_WORLD);
    }
    printf("%d done.\n", me);
  }
