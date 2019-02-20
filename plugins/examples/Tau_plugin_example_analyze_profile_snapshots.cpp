/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include <Profile/TauEnv.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauCollate.h>
#include <Profile/TauUtil.h>
#include <Profile/TauXML.h>

#include <mpi.h>
#include <Profile/TauPlugin.h>

#include <Profile/TauTrace.h>


typedef struct snapshot_buffer {
  double ***gExcl, ***gIncl;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;
  double **gAtomicMin, **gAtomicMax;
  double **gAtomicCalls, **gAtomicMean;
  double **gAtomicSumSqr;
  double **sAtomicMin, **sAtomicMax;
  double **sAtomicCalls, **sAtomicMean;
  double **sAtomicSumSqr;
  Tau_unify_object_t *functionUnifier;
  Tau_unify_object_t *atomicUnifier;
  int *numEventThreads;
  int *globalEventMap;
  int *numAtomicEventThreads;
  int *globalAtomicEventMap;
} snapshot_buffer_t;

#define N_SNAPSHOTS 5
snapshot_buffer_t s_buffer[5]; //Store upto N_SNAPSHOTS snapshots

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  return 0;
}

int Tau_plugin_event_trigger(Tau_plugin_event_trigger_data_t* data) {
 
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;
  static int current_s_buffer_index = 0;

  FILE *f;
#ifdef TAU_MPI
  MPI_Status status;
#endif 
  x_uint64 start, end;

  Tau_unify_unifyDefinitions_MPI();

  int rank = 0;
  int size = 1;

#ifdef TAU_MPI

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

#endif


  int numEvents = 0;
  int globalNumThreads;

  int numAtomicEvents = 0;
  
  if (TauEnv_get_stat_precompute() == 1) {
    // Unification must already be called.
    s_buffer[current_s_buffer_index].functionUnifier = Tau_unify_getFunctionUnifier();
    numEvents = s_buffer[current_s_buffer_index].functionUnifier->globalNumItems;
    s_buffer[current_s_buffer_index].numEventThreads = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    s_buffer[current_s_buffer_index].globalEventMap = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<s_buffer[current_s_buffer_index].functionUnifier->globalNumItems; i++) { 
      // -1 indicates that the event did not occur for this rank
      s_buffer[current_s_buffer_index].globalEventMap[i] = -1; 
    }
    for (int i=0; i<s_buffer[current_s_buffer_index].functionUnifier->localNumItems; i++) {
      s_buffer[current_s_buffer_index].globalEventMap[s_buffer[current_s_buffer_index].functionUnifier->mapping[i]] = i; // set reverse mapping
    }
    Tau_collate_get_total_threads_MPI(s_buffer[current_s_buffer_index].functionUnifier, &globalNumThreads, &(s_buffer[current_s_buffer_index].numEventThreads),
				  numEvents, s_buffer[current_s_buffer_index].globalEventMap,false);

    Tau_collate_allocateFunctionBuffers(&(s_buffer[current_s_buffer_index].gExcl), &(s_buffer[current_s_buffer_index].gIncl),
					&(s_buffer[current_s_buffer_index].gNumCalls), &(s_buffer[current_s_buffer_index].gNumSubr),
					numEvents,
					Tau_Global_numCounters,
					COLLATE_OP_BASIC);
    if (rank == 0) {
      Tau_collate_allocateFunctionBuffers(&(s_buffer[current_s_buffer_index].sExcl), &(s_buffer[current_s_buffer_index].sIncl),
					  &(s_buffer[current_s_buffer_index].sNumCalls), &(s_buffer[current_s_buffer_index].sNumSubr),
					  numEvents,
					  Tau_Global_numCounters,
					  COLLATE_OP_DERIVED);
    }
    Tau_collate_compute_statistics_MPI(s_buffer[current_s_buffer_index].functionUnifier, s_buffer[current_s_buffer_index].globalEventMap, 
				   numEvents, 
				   globalNumThreads, s_buffer[current_s_buffer_index].numEventThreads,
				   &(s_buffer[current_s_buffer_index].gExcl), &(s_buffer[current_s_buffer_index].gIncl), &(s_buffer[current_s_buffer_index].gNumCalls), &(s_buffer[current_s_buffer_index].gNumSubr),
				   &(s_buffer[current_s_buffer_index].sExcl), &(s_buffer[current_s_buffer_index].sIncl), &(s_buffer[current_s_buffer_index].sNumCalls), &(s_buffer[current_s_buffer_index].sNumSubr));

    s_buffer[current_s_buffer_index].atomicUnifier = Tau_unify_getAtomicUnifier();
    numAtomicEvents = s_buffer[current_s_buffer_index].atomicUnifier->globalNumItems;
    s_buffer[current_s_buffer_index].numAtomicEventThreads = 
      (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    s_buffer[current_s_buffer_index].globalAtomicEventMap = (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<numAtomicEvents; i++) { 
      // -1 indicates that the event did not occur for this rank
      s_buffer[current_s_buffer_index].globalAtomicEventMap[i] = -1; 
    }
    for (int i=0; i<s_buffer[current_s_buffer_index].atomicUnifier->localNumItems; i++) {
      // set reverse mapping
      s_buffer[current_s_buffer_index].globalAtomicEventMap[s_buffer[current_s_buffer_index].atomicUnifier->mapping[i]] = i;
    }

    Tau_collate_get_total_threads_MPI(s_buffer[current_s_buffer_index].atomicUnifier, &globalNumThreads, &(s_buffer[current_s_buffer_index].numAtomicEventThreads),
				  numAtomicEvents, s_buffer[current_s_buffer_index].globalAtomicEventMap,true);
    
    Tau_collate_allocateAtomicBuffers(&(s_buffer[current_s_buffer_index].gAtomicMin), &(s_buffer[current_s_buffer_index].gAtomicMax),
				      &(s_buffer[current_s_buffer_index].gAtomicCalls), &(s_buffer[current_s_buffer_index].gAtomicMean),
				      &(s_buffer[current_s_buffer_index].gAtomicSumSqr),
				      numAtomicEvents,
				      COLLATE_OP_BASIC);
    if (rank == 0) {
      Tau_collate_allocateAtomicBuffers(&(s_buffer[current_s_buffer_index].sAtomicMin), &(s_buffer[current_s_buffer_index].sAtomicMax),
					&(s_buffer[current_s_buffer_index].sAtomicCalls), &(s_buffer[current_s_buffer_index].sAtomicMean),
					&(s_buffer[current_s_buffer_index].sAtomicSumSqr),
					numAtomicEvents,
					COLLATE_OP_DERIVED);
    }
    Tau_collate_compute_atomicStatistics_MPI(s_buffer[current_s_buffer_index].atomicUnifier, s_buffer[current_s_buffer_index].globalAtomicEventMap, 
					 numAtomicEvents, 
					 globalNumThreads, 
					 s_buffer[current_s_buffer_index].numAtomicEventThreads,
					 &(s_buffer[current_s_buffer_index].gAtomicMin), &(s_buffer[current_s_buffer_index].gAtomicMax), 
					 &(s_buffer[current_s_buffer_index].gAtomicCalls), &(s_buffer[current_s_buffer_index].gAtomicMean),
					 &(s_buffer[current_s_buffer_index].gAtomicSumSqr),
					 &(s_buffer[current_s_buffer_index].sAtomicMin), &(s_buffer[current_s_buffer_index].sAtomicMax), 
					 &(s_buffer[current_s_buffer_index].sAtomicCalls), &(s_buffer[current_s_buffer_index].sAtomicMean),
					 &(s_buffer[current_s_buffer_index].sAtomicSumSqr));

  }
  current_s_buffer_index++;
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

