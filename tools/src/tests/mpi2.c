  #include <mpi.h>

  int foo(MPI_Win win)
  {
    int ret;
    ret = MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
    return ret;
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
