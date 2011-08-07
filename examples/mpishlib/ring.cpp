#include <Profile/Profiler.h>
#ifndef TAU_GROUP_RING
#define TAU_GROUP_RING TAU_GET_PROFILE_GROUP("RING")
#endif /* TAU_GROUP_RING */ 
#include <stdio.h>
#include <mpi.h>
#include "ring.h"

double A[1024][1024], B[1024][1024], Cary[1024][1024];
static const int anz = 512;
void C::method() {
  TAU_PROFILE("void C::method()", TAU_CT(*this), TAU_GROUP_RING);
   

    int i;
    int n1, n2, n3;
    int field[anz];
    MPI_Status status;

    for (i=0; i<anz; i++)
      field[i] = i;

    MPI_Barrier(MPI_COMM_WORLD);
    for(n1=0; n1 < 1000; n1++) 
      for(n2=0; n2 < 1000; n2++)  {
        A[n1][n2] = B[n1][n2] = n1+n2*1.0002;
	Cary[n1][n2] = 0;
      }
 
    
    for (n1=0; n1 < 1000; n1++) {
      for (n2=0; n2 < 1000; n2++) {
        for (n3=0; n3 < 1000; n3++) {
          Cary[n1][n2] += A[n1][n3]*B[n3][n2]; 
        }
      }
    }
    if (me==0) {
      MPI_Send(&field, anz, MPI_INT, 1, 4711, MPI_COMM_WORLD);
      Cary[n1][n2] = n1/(n2-n2);
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
