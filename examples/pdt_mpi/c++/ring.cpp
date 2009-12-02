#include <stdio.h>

#ifndef MPICH_IGNORE_CXX_SEEK
#define MPICH_IGNORE_CXX_SEEK
#endif

#include <mpi.h>

static const int anz = 512;
class C {
public:
  C(int m, int p) : me(m), proc(p) {}
  void method() {
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

private:
  int proc, me;
};

int main(int argc, char **argv) 
{
  int proc, me;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &proc);
  MPI_Comm_rank (MPI_COMM_WORLD, &me);

  C c(me, proc);
  c.method();

  MPI_Finalize ();
}
