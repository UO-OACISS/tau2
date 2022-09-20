    #include <mpi.h>
    int foo (MPI_File fh, void *buf, int count,
             MPI_Datatype datatype, MPIO_Request *request)
    {
      return MPI_File_iread_shared (fh, buf, count, datatype, request);
    }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
