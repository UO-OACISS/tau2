#include <Profile/Profiler.h>
#ifndef TAU_GROUP_RING
#define TAU_GROUP_RING TAU_GET_PROFILE_GROUP("RING")
#endif /* TAU_GROUP_RING */ 
#include <stdio.h>
#include <mpi.h>

static const int anz = 512;
class C {
public:
  C(int m, int p) : me(m), proc(p) {
    TAU_PROFILE("C &C::C(int, int)", TAU_CT(*this), TAU_GROUP_RING);
  }
  void method() {
    TAU_PROFILE("void C::method()", TAU_CT(*this), TAU_GROUP_RING);
    
    int i;
    int field[anz];
    MPI_Status status;

    for (i=0; i<anz; i++)
      field[i] = i;

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (me==0) {
      MPI_Send(&field, anz, MPI_INT, 1, 35, MPI_COMM_WORLD);
      MPI_Recv(&field, anz, MPI_INT, proc-1, 35, MPI_COMM_WORLD, &status);
    }
    else {
      MPI_Recv(&field, anz, MPI_INT, me-1, 35, MPI_COMM_WORLD, &status);
      if (me == proc-1)
        MPI_Send(&field, anz, MPI_INT, 0, 35, MPI_COMM_WORLD);
      else
        MPI_Send(&field, anz, MPI_INT, me+1, 35, MPI_COMM_WORLD);
    }
    printf("%d done.\n", me);
  }

private:
  int proc, me;
};

int foo() {
  TAU_PROFILE("foo", "", TAU_USER);
  printf ("foo: This function should not show up in the trace if -MPITRACE is configured\n");
}

int main(int argc, char **argv) 
{
  TAU_PROFILE("int main(int, char **)", " ", TAU_DEFAULT);
  TAU_INIT(&argc, &argv); 
#ifndef TAU_MPI
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI */

  int proc, me;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &proc);
  MPI_Comm_rank (MPI_COMM_WORLD, &me);

  foo();
  C c(me, proc);
  c.method();

  MPI_Finalize ();
}
