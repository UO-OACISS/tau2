  #include <mpi.h>

  //extern MPI_Fint * MPI_F_STATUSES_IGNORE;
  MPI_Fint * foo(void) {
    return MPI_F_STATUSES_IGNORE;
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
