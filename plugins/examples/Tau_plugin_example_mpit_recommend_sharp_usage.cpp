#ifdef TAU_MPI
#ifdef TAU_MPI_T

#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>

#include <Profile/Profiler.h>
#include <Profile/UserEvent.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauPlugin.h>

#include <Profile/TauEnv.h>
#include <Profile/TauMpiTTypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <Profile/TauAPI.h>

#define TAU_NAME_LENGTH 1024

extern "C" int TauProfiler_updateAllIntermediateStatistics(void);

/*This function is invoked at the end of execution, and gathers exclusive time spent in MPI_Allreduce
 * versus the total application execution time, along with the message size for MPI_Allreduce.
 * If the application spends a signigicant time in MPI_Allreduce, and if the message size involved in MPI_Allreduce
 * is low, then it recommends the user to enable SHArP through a message on TAU_METADATA*/
extern "C" int Tau_plugin_example_mpit_recommend_sharp_usage(Tau_plugin_event_end_of_execution_data_t* data) {
  double exclusiveTimeAllReduce, inclusiveTimeApp = 0.0;
  double meanAllReduceMessageSize = 0;

  const char **counterNames;
  int numCounters;
  std::vector<FunctionInfo*>::iterator it;

  TAU_VERBOSE("TAU PLUGIN: Sharp recommendation generated for tid: %d\n", data->tid);

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
  std::vector<tau::TauUserEvent*, TauSignalSafeAllocator<tau::TauUserEvent*> >::iterator it2;
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

  //std::cout << "Total percentage of MPI_Allreduce() and mean message size are " << (exclusiveTimeAllReduce/inclusiveTimeApp) << "  " << meanAllReduceMessageSize << std::endl;

  RtsLayer::UnLockDB();

  //Generate recommendation for the user if appropriate conditions are met
  if(((exclusiveTimeAllReduce/inclusiveTimeApp) > .30 ) && meanAllReduceMessageSize < 32.0) {
    TAU_METADATA("TAU_RECOMMENDATION", "MPI_T_RECOMMEND_SHARP_USAGE: You could see potential improvement in performance by configuring MVAPICH with --enable-sharp and enabling MV2_ENABLE_SHARP in MVAPICH version 2.3a and above");
  }

  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 *  * Every plugin MUST implement this function to register callbacks for various events 
 *   * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv) {
  Tau_plugin_callbacks_t * cb = (Tau_plugin_callbacks_t*)malloc(sizeof(Tau_plugin_callbacks_t));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->EndOfExecution = Tau_plugin_example_mpit_recommend_sharp_usage;
  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb);

  return 0;
}
#endif /* TAU_MPI_T */
#endif /* TAU_MPI */
