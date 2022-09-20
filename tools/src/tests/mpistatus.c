
  #include <mpi.h>
  #include <memory.h>
  int foo(MPI_Fint *status)
  {
    MPI_Status s;
    return MPI_Status_f2c(status, &s);
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
