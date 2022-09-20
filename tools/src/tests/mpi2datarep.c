  #include <mpi.h>

  int foo( char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, void * extra_state)
  {
    return MPI_Register_datarep( datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state);

  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
