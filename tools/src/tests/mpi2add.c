  #include <mpi.h>

  int f1( int * errorclass)
  {
    return MPI_Add_error_class( errorclass) ;
  }

  int f2( char * datarep, int incount, MPI_Datatype datatype, MPI_Aint * size)
  {
    return MPI_Pack_external_size( datarep, incount, datatype, size) ;
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
