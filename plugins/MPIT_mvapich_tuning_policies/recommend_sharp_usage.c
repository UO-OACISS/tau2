#include <mpi.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMpiTTypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauAPI.h>

#define dprintf TAU_VERBOSE
#define TAU_NAME_LENGTH 1024

int Tau_mpi_t_recommend_sharp_usage(int argc, void** argv) {
  double *exclusiveTimeApp = (double*)malloc(sizeof(double));
  void* ptr = NULL;
  ptr = (void *)Tau_pure_search_for_function("MPI_Wait()");
  Tau_get_exclusive_values(ptr, exclusiveTimeApp, Tau_get_thread());

  printf("Total time exclusive for application is %f\n", exclusiveTimeApp);
  return 0;
}

int Tau_plugin_init_func(PluginManager* plugin_manager) {
  printf("Hi there! I recommend the user to enable SHArP or not! My init func has been called\n");
  Tau_util_plugin_manager_register_role_hook(plugin_manager, "MPIT_Recommend", Tau_mpi_t_recommend_sharp_usage);
  return 0;
}

