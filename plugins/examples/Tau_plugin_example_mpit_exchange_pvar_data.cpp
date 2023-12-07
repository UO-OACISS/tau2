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

#include <Profile/TauTrace.h>

#ifdef TAU_MPI
#include <mpi.h>

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  fprintf(stderr, "TAU PLUGIN: Inside end of execution.\n");
  return 0;
}

int Tau_plugin_event_mpit(Tau_plugin_event_mpit_data_t* data) {

  int rank; long long int local_val; long long int global_val;

  rank = RtsLayer::myNode();

  //fprintf(stderr, "PVAR Name %s and value %lld from rank %d\n", data->pvar_name, data->pvar_value, rank);

  local_val = data->pvar_value;

  #ifdef TAU_MPI
  PMPI_Allreduce(&local_val, &global_val, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  #endif

  //fprintf(stderr, "Max value: %lld\n", global_val);
  if(global_val>0)
  {
    fprintf(stderr, "PVAR Name %s and value %lld from rank %d\n Max value: %lld\n", data->pvar_name, data->pvar_value, rank, global_val);
  }

  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->Mpit = Tau_plugin_event_mpit;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

#endif
