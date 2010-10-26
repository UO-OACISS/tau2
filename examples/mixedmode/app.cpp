#include <stdio.h> 
#include "mpi.h"
#include <pthread.h>

void * threaded_func(void *data);

pthread_barrier_t bar;

int main (int argc, char **argv)
{
MPI_Status status;
int rank, ret, size; 
int data_to_send, data_to_recv, message_tag;
int count, destination_tid;

  ret = MPI_Init(&argc, &argv);
  ret = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 2) {
    printf("Please use only 2 MPI processes (mpirun -np 2 app)\n");
    MPI_Finalize();
    return(1);
  }
  printf("After Initialization - my rank is %d out of %d procs\n", rank, size);
  /* send data from rank 0 to 1 */

  pthread_attr_t  attr;
  pthread_t       tid;

  pthread_attr_init(&attr);


  pthread_barrier_init(&bar, NULL, 2);

  if (ret = pthread_create(&tid, NULL, threaded_func, &rank) )
  {
    perror("pthread_create fails");
    return 1;
  }

  pthread_barrier_wait(&bar);

  if (rank == 0)
    { /* send data */
      data_to_send = 5767;
      count = 1;
      destination_tid = 1;
      message_tag = 34;
      ret = MPI_Send(&data_to_send, count, MPI_INT, destination_tid, message_tag, MPI_COMM_WORLD);
    }
  else
    {
      /* recv data */
      count = 1;
      ret = MPI_Recv(&data_to_recv, count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      printf("rank = %d:Received data = %d, tag = %d, source = %d\n", rank, data_to_recv,status.MPI_TAG, status.MPI_SOURCE);
    }
      
  ret = MPI_Finalize();
  if (ret = pthread_join(tid, NULL) )
  {
    perror("pthread_join failed");
    return ret;
  }
  pthread_barrier_destroy(&bar);
}
 




