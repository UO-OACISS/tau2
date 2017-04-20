#include <Profile/Profiler.h>
#include <Profile/UserEvent.h>
#include <Profile/TauMetrics.h>

#include <Profile/TauEnv.h>
#include <Profile/TauMpiTTypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauAPI.h>

#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>

#define dprintf TAU_VERBOSE
#define TAU_NAME_LENGTH 1024

extern "C" int TauProfiler_updateAllIntermediateStatistics(void);

extern "C" int Tau_mpi_t_recommend_sharp_usage(int argc, void** argv) {
  double exclusiveTimeAllReduce, inclusiveTimeApp = 0.0;
  double meanAllReduceMessageSize = 0;

  const char **counterNames;
  int numCounters;
  std::vector<FunctionInfo*>::iterator it;

  //Get the most up-to-date profile information
  TauProfiler_updateAllIntermediateStatistics();
  TauMetrics_getCounterList(&counterNames, &numCounters);
  RtsLayer::LockDB();


  //Get inclusive value for entire application
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    if(strcmp(fi->GetName(), ".TAU application") == 0) {
      for (int m = 0; m < numCounters; m++) {
        if(strcmp(counterNames[m], "TIME") == 0) {
          inclusiveTimeApp = fi->getDumpInclusiveValues(Tau_get_thread())[m];
        }
      }
    }
  }
  
  //Get exclusive value for MPI_Allreduce
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    if(strcmp(fi->GetName(), "MPI_Allreduce()") == 0) {
      for (int m = 0; m < numCounters; m++) {
        if(strcmp(counterNames[m], "TIME") == 0) {
          exclusiveTimeAllReduce = fi->getDumpExclusiveValues(Tau_get_thread())[m];
        }
      }
    }
  }

  //Get the mean MPI_Allreduce message size  
  std::vector<tau::TauUserEvent*, TauSignalSafeAllocator<tau::TauUserEvent*>>::iterator it2;
  int numEvents;
  std::stringstream tmp_str;
  std::stringstream all_reduce_event_name("Message size for all-reduce");

  for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
    tau::TauUserEvent *ue = (*it2);
    if(ue && ue->GetNumEvents(Tau_get_thread()) == 0) continue; 
    tmp_str << ue->GetName(); 

    if(tmp_str.str() == all_reduce_event_name.str()) {
      meanAllReduceMessageSize = ue->GetMean(Tau_get_thread());
      break;
    }
    tmp_str.str(std::string());
  }

  std::cout << "Total percentage of MPI_Allreduce() and mean message size are " << (exclusiveTimeAllReduce/inclusiveTimeApp) << "  " << meanAllReduceMessageSize << std::endl;

  RtsLayer::UnLockDB();

  //Generate recommendation for the user

  TAU_METADATA("TAU_MPI_T_RECOMMEND_SHARP_USAGE", "You could see potential improvement in performance by configuring MVAPICH with --enable-sharp and enabling MV2_ENABLE_SHARP in MVAPICH version 2.3a and above");
  
  return 0;
}

extern "C" int Tau_plugin_init_func(PluginManager* plugin_manager) {
  printf("Hi there! I recommend the user to enable SHArP or not! My init func has been called\n");
  Tau_util_plugin_manager_register_role_hook(plugin_manager, "MPIT_Recommend", Tau_mpi_t_recommend_sharp_usage);
  return 0;
}

