  #include <mpi.h>

  int foo(MPI_Win_copy_attr_function *f1, MPI_Type_copy_attr_function *f2,
        MPI_Comm_copy_attr_function *f3)
  {
    return 0;
  }

  int MPI_Type_get_attr(MPI_Datatype type, int type_keyval, void *attribute, int *flag)
  {
    return PMPI_Type_get_attr(type, type_keyval, attribute, flag);
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
