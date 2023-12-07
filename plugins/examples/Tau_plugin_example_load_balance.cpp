/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>

#ifdef TAU_MPI
  #include <mpi.h>

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  return 0;
}

int Tau_plugin_event_trigger(Tau_plugin_event_trigger_data_t* data) {

  #ifdef TAU_MPI
  int rank; int size;
  int global_min, global_max;
  int global_sum; float sum_, avg_, min_, max_;

  int local = *((int*)(data->data));

  PMPI_Reduce(&local, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&local, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&local, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  rank = RtsLayer::myNode();

  if(rank == 0) {
    sum_ = global_sum;
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    fprintf(stderr, "Avg, min, max are %f %d %d \n", (sum_/size), global_min, global_max);
    avg_ = (sum_ / size);
    min_ = global_min;
    max_ = global_max;

    if((max_ - min_) > 0.10 * avg_) {
      fprintf(stderr, "Should rebalance...\n");
      local = 1;
    } else {
      local = 0;
    }
  }

  PMPI_Bcast(&local, 1, MPI_INT, 0, MPI_COMM_WORLD);

  *((int*)(data->data)) = local;

  #endif

  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->Trigger = Tau_plugin_event_trigger;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

#endif
