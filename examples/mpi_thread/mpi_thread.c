// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Ping pong example with MPI_Send and MPI_Recv. Two processes ping pong a
// number back and forth, incrementing it until it reaches a given value.

// Modified to demonstrate initializing MPI on another thread

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

void wait(int seconds) {
    struct timespec delay;
    delay.tv_sec = seconds;
    delay.tv_nsec = 0;
    while(nanosleep(&delay, &delay)) { };
}

void * init_mpi(void * unused) {
  (void)unused;
  // Initialize the MPI environment
  int provided = 0;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  if(provided != MPI_THREAD_MULTIPLE) {
    fprintf(stderr, "Requested but not provided with MPI_THREAD_MULTIPLE\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  printf("MPI initialized, init thread waiting 1 sec");
  wait(1);
  printf("MPI init thread done");
  pthread_exit(NULL);
}

void * wait_fn(void * unused) {
    (void)unused;
    printf("Hello from Wait Thread\n");
    wait(3);
    printf("Goodbye from Wait Thread\n");
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
  const int PING_PONG_LIMIT = 10;

  pthread_t wait_thread;
  pthread_create(&wait_thread, NULL, wait_fn, NULL);

  pthread_t init_thread;
  pthread_create(&init_thread, NULL, init_mpi, NULL);
  pthread_join(init_thread, NULL);

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We are assuming 2 processes for this task
  if (world_size != 2) {
    fprintf(stderr, "World size must be two for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int ping_pong_count = 0;
  int partner_rank = (world_rank + 1) % 2;
  while (ping_pong_count < PING_PONG_LIMIT) {
    if (world_rank == ping_pong_count % 2) {
      // Increment the ping pong count before you send it
      ping_pong_count++;
      MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
      printf("%d sent and incremented ping_pong_count %d to %d\n",
             world_rank, ping_pong_count, partner_rank);
    } else {
      MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printf("%d received ping_pong_count %d from %d\n",
             world_rank, ping_pong_count, partner_rank);
    }
  }

  pthread_join(wait_thread, NULL);

  MPI_Finalize();
}
