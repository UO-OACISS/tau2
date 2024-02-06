#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <TAU.h>
#include <Profile/TauPluginTypes.h>

int WORK_ITER=5000;

#define T 20

void do_work() {
  usleep(1000);
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    #ifdef TAU_MPI
    MPI_Init(&argc, &argv);
    int data, i, t, w;
    char filename[100];
    FILE * ptr;
    size_t load_balance_module = TAU_CREATE_TRIGGER("load balance module");
    int x;
   
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    snprintf(filename, sizeof(filename),  "%d.txt", world_rank);
    ptr = fopen(filename, "w");

    Tau_enable_plugin_for_trigger_event(TAU_PLUGIN_EVENT_TRIGGER, load_balance_module, 0);
    TAU_PROFILE_TIMER(timer, "trigger_timer", "", TAU_DEFAULT);
    srand((unsigned) world_rank);

    for(t=0; t < T; t++) {
      for(w=0; w < WORK_ITER; w++) {
        do_work();
      }

      data = WORK_ITER;

      TAU_PROFILE_START(timer);
      TAU_TRIGGER(load_balance_module, (void *)&data); 
      TAU_PROFILE_STOP(timer);
      MPI_Barrier(MPI_COMM_WORLD);
     
      if(data) {
        fprintf(stderr, "Rebalancing...\n");
        WORK_ITER = 5000;
        fprintf(ptr, "%d,%d,%d\n", t, world_rank, WORK_ITER);
      } else {
        x = (rand() % 100)*((-1*(world_rank)%2)^1);
        WORK_ITER = WORK_ITER + x;
        fprintf(ptr, "%d,%d,%d\n", t, world_rank, WORK_ITER);
        fprintf(stderr, "Work for iteration %d\n", WORK_ITER);
      }

      fflush(ptr);
    } 

    // Finalize the MPI environment.
    MPI_Finalize();
    #endif
    return 0;
}

