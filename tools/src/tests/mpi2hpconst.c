  #include <mpi.h>
  /* HP MPI uses const char * instead of char * in the
     prototypes of the following routines */
  int foo(MPI_Info Info, const char *key, const char *value)
  {
    return PMPI_Info_set(Info, key, value);
  }
  int f1(MPI_Info a, const char * b, int c, char * d, int *e)
  {
    return PMPI_Info_get(a,b,c,d,e);
  }
  int f2(MPI_Info a, const char *b)
  {
    return PMPI_Info_delete(a,b);
  }
  int MPI_Info_get_valuelen(MPI_Info a, const char * b, int * c, int * d)
  { /* above decl triggers an error on T3E */
    return PMPI_Info_get_valuelen(a,b,c,d);
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
