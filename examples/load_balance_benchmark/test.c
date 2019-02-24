#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <TAU.h>

struct data_t {
  int work;
  int should_rebalance;
};

typedef struct data_t data_t;

int WORK_ITER=5000;

#define T 1000

void do_work() {
  usleep(1000);
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    #ifdef TAU_MPI
    MPI_Init(NULL, NULL);
    data_t data;
    size_t load_balance_module = TAU_CREATE_TRIGGER("load balance module");
    int x;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    for(int t=0; t < T; t++) {

      for(int w=0; w < WORK_ITER; w++) {
        do_work();
      }

      data.work = WORK_ITER;
      data.should_rebalance = 0;

      TAU_TRIGGER(load_balance_module, (void*)&data);

      MPI_Barrier(MPI_COMM_WORLD);

      if(data.should_rebalance) {
        fprintf(stderr, "Rebalancing...\n");
        WORK_ITER = 5000;
      } else {
        srand((unsigned) world_rank);
        x = (rand() % 100)*((-1*(world_rank)%2)^1);
        WORK_ITER = WORK_ITER + x;
        fprintf(stderr, "Work for iteration %d\n", WORK_ITER);
      }
    } 

    // Finalize the MPI environment.
    MPI_Finalize();
    #endif
    return 0;
}

