#include <mpi.h>

void check_args(int rank, int size) {
   if ((size < 2 || size % 2 != 0) && rank == 0) {
       fprintf(stderr, "\n\n**************************************************************\n");
       fprintf(stderr, "ERROR! Please use an even number of MPI ranks, greater than 1.\n");
       fprintf(stderr, "**************************************************************\n\n");
       MPI_Abort(MPI_COMM_WORLD, 0);
   }
}

#define ELEMENTS_TO_SEND 100

void exchange_data(int rank, int size) {
    int tag = 0;
    if (rank % 2 == 0) {
        // even ranks
        int buf[ELEMENTS_TO_SEND] = {0};
        MPI_Send(buf, ELEMENTS_TO_SEND, MPI_INT, rank+1, tag, MPI_COMM_WORLD); 
    } else {
        // odd ranks
        int buf[ELEMENTS_TO_SEND] = {0};
        int source = 0;
        MPI_Status status;
        MPI_Recv(buf, ELEMENTS_TO_SEND, MPI_INT, rank-1, tag, MPI_COMM_WORLD, &status); 
    }
}
