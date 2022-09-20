  #include <mpi.h>

  int f1( void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype
  , MPI_Op op, MPI_Comm comm)
  {
    return MPI_Exscan( sendbuf, recvbuf, count, datatype, op, comm) ;
  }

  int f2( MPI_Datatype type, MPI_Datatype * newtype)
  {
    return MPI_Type_dup( type, newtype) ;
  }

  int f3( int p, int r, MPI_Datatype * newtype)
  {
    return MPI_Type_create_f90_real( p, r, newtype) ;
  }

  int f4( int p, int r, MPI_Datatype * newtype)
  {
    return MPI_Type_create_f90_complex( p, r, newtype) ;
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
