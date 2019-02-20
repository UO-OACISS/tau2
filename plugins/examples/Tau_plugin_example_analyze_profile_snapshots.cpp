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

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  return 0;
}

int Tau_plugin_event_trigger(Tau_plugin_event_trigger_data_t* data) {
 
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

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


  Tau_unify_object_t *functionUnifier;
  int numEvents = 0;
  int globalNumThreads;
  int *numEventThreads;
  int *globalEventMap = 0;

  double ***gExcl, ***gIncl;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;

  Tau_unify_object_t *atomicUnifier;
  int numAtomicEvents = 0;
  int *numAtomicEventThreads;
  int *globalAtomicEventMap = 0;
  
  double **gAtomicMin, **gAtomicMax;
  double **gAtomicCalls, **gAtomicMean;
  double **gAtomicSumSqr;
  double **sAtomicMin, **sAtomicMax;
  double **sAtomicCalls, **sAtomicMean;
  double **sAtomicSumSqr;

  if (TauEnv_get_stat_precompute() == 1) {
    // Unification must already be called.
    functionUnifier = Tau_unify_getFunctionUnifier();
    numEvents = functionUnifier->globalNumItems;
    numEventThreads = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    globalEventMap = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<functionUnifier->globalNumItems; i++) { 
      // -1 indicates that the event did not occur for this rank
      globalEventMap[i] = -1; 
    }
    for (int i=0; i<functionUnifier->localNumItems; i++) {
      globalEventMap[functionUnifier->mapping[i]] = i; // set reverse mapping
    }
    Tau_collate_get_total_threads_MPI(functionUnifier, &globalNumThreads, &numEventThreads,
				  numEvents, globalEventMap,false);

    Tau_collate_allocateFunctionBuffers(&gExcl, &gIncl,
					&gNumCalls, &gNumSubr,
					numEvents,
					Tau_Global_numCounters,
					COLLATE_OP_BASIC);
    if (rank == 0) {
      Tau_collate_allocateFunctionBuffers(&sExcl, &sIncl,
					  &sNumCalls, &sNumSubr,
					  numEvents,
					  Tau_Global_numCounters,
					  COLLATE_OP_DERIVED);
    }
    Tau_collate_compute_statistics_MPI(functionUnifier, globalEventMap, 
				   numEvents, 
				   globalNumThreads, numEventThreads,
				   &gExcl, &gIncl, &gNumCalls, &gNumSubr,
				   &sExcl, &sIncl, &sNumCalls, &sNumSubr);

    atomicUnifier = Tau_unify_getAtomicUnifier();
    numAtomicEvents = atomicUnifier->globalNumItems;
    numAtomicEventThreads = 
      (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    globalAtomicEventMap = (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<numAtomicEvents; i++) { 
      // -1 indicates that the event did not occur for this rank
      globalAtomicEventMap[i] = -1; 
    }
    for (int i=0; i<atomicUnifier->localNumItems; i++) {
      // set reverse mapping
      globalAtomicEventMap[atomicUnifier->mapping[i]] = i;
    }
    Tau_collate_get_total_threads_MPI(atomicUnifier, &globalNumThreads, &numAtomicEventThreads,
				  numAtomicEvents, globalAtomicEventMap,true);
    
    Tau_collate_allocateAtomicBuffers(&gAtomicMin, &gAtomicMax,
				      &gAtomicCalls, &gAtomicMean,
				      &gAtomicSumSqr,
				      numAtomicEvents,
				      COLLATE_OP_BASIC);
    if (rank == 0) {
      Tau_collate_allocateAtomicBuffers(&sAtomicMin, &sAtomicMax,
					&sAtomicCalls, &sAtomicMean,
					&sAtomicSumSqr,
					numAtomicEvents,
					COLLATE_OP_DERIVED);
    }
    Tau_collate_compute_atomicStatistics_MPI(atomicUnifier, globalAtomicEventMap, 
					 numAtomicEvents, 
					 globalNumThreads, 
					 numAtomicEventThreads,
					 &gAtomicMin, &gAtomicMax, 
					 &gAtomicCalls, &gAtomicMean,
					 &gAtomicSumSqr,
					 &sAtomicMin, &sAtomicMax, 
					 &sAtomicCalls, &sAtomicMean,
					 &sAtomicSumSqr);

  }
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

