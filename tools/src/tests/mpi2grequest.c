  #include <mpi.h>

  int foo(MPI_Grequest_query_function *grequest_query_fn,
              MPI_Grequest_free_function *grequest_free_fn,
              MPI_Grequest_cancel_function *grequest_cancel_fn,
              void *extra_state, MPI_Request *request)
  {
    return MPI_Grequest_start(grequest_query_fn,
              grequest_free_fn,
              grequest_cancel_fn,
              extra_state, request);

  }

  int bar(MPI_Request request)
  {
    return MPI_Grequest_complete(request);
  }

int main (int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Finalize();
}
