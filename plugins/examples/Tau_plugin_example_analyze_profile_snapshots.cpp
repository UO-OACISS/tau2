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
#include <utility>

#include <Profile/TauEnv.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauCollate.h>
#include <Profile/TauUtil.h>
#include <Profile/TauXML.h>

#ifdef TAU_MPI
#include <mpi.h>

#include <list>
#include <vector>

#include <Profile/TauPlugin.h>

#include <Profile/TauTrace.h>


typedef struct snapshot_buffer {
  double ***gExcl, ***gIncl;
  double_int **gExcl_min, **gIncl_min;
  double_int **gExcl_max, **gIncl_max;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;
  double **gAtomicMin, **gAtomicMax;
  double_int *gAtomicMin_min, *gAtomicMax_max;
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
  std::vector <int> top_5_excl_time_mean;
} snapshot_buffer_t;

#define N_SNAPSHOTS 2000
snapshot_buffer_t s_buffer[N_SNAPSHOTS]; //Store upto N_SNAPSHOTS snapshots

int is_instrumentation_enabled = 1;

int counter = 0;

bool sort_func(const std::pair<double, int>& first, const std::pair<double, int>& second)
{
  return (first.first < second.first);
}

void disable_instrumentation_if_necessary(int index, int rank) {

   int should_i_disable_instrumentation = 1;

   if(rank == 0) {
     for(int i = 0 ; i < 5; i++) {
       if(s_buffer[index].top_5_excl_time_mean[i] != s_buffer[index-1].top_5_excl_time_mean[i]) {
         should_i_disable_instrumentation = 0; break;
       }
     }
   }

   MPI_Bcast(&should_i_disable_instrumentation, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if(should_i_disable_instrumentation) {
     fprintf(stderr, "Disabling instrumentation at index %d\n", index);
     TAU_DISABLE_INSTRUMENTATION();
     is_instrumentation_enabled = 0;
   }
}


int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  return 0;
}

int Tau_plugin_event_trigger(Tau_plugin_event_trigger_data_t* data) {

  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

#ifdef TAU_ANALYTICS_INSTRUMENTATION_TOGGLE
  if(!is_instrumentation_enabled && counter < 5) {
    counter++;
    return 0;
  }

  if(counter == 5) {
    fprintf(stderr, "Enabling instrumentation again...\n");
    TAU_ENABLE_INSTRUMENTATION();
    is_instrumentation_enabled = 1;
    counter = 0;
  }
#endif

  //Update the profile!
  TauProfiler_updateAllIntermediateStatistics();
  static int index = 0;

  Tau_unify_unifyDefinitions_MPI();

  int rank = 0;
  int size = 1;

#ifdef TAU_MPI

  rank = RtsLayer::myNode();
  size = tau_totalnodes(0,1);

#endif


  int numEvents = 0;
  int globalNumThreads;

  int numAtomicEvents = 0;

  if (TauEnv_get_stat_precompute() == 1) {
    // Unification must already be called.
    s_buffer[index].functionUnifier = Tau_unify_getFunctionUnifier();
    numEvents = s_buffer[index].functionUnifier->globalNumItems;
    s_buffer[index].numEventThreads = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    s_buffer[index].globalEventMap = (int*)TAU_UTIL_MALLOC(numEvents*sizeof(int));
    // initialize all to -1
    for (int i=0; i<s_buffer[index].functionUnifier->globalNumItems; i++) {
      // -1 indicates that the event did not occur for this rank
      s_buffer[index].globalEventMap[i] = -1;
    }
    for (int i=0; i<s_buffer[index].functionUnifier->localNumItems; i++) {
      s_buffer[index].globalEventMap[s_buffer[index].functionUnifier->mapping[i]] = i; // set reverse mapping
    }
    Tau_collate_get_total_threads_MPI(s_buffer[index].functionUnifier, &globalNumThreads, &(s_buffer[index].numEventThreads),
				  numEvents, s_buffer[index].globalEventMap,false);

    Tau_collate_allocateFunctionBuffers(&(s_buffer[index].gExcl), &(s_buffer[index].gIncl),
					&(s_buffer[index].gNumCalls), &(s_buffer[index].gNumSubr),
					numEvents,
					Tau_Global_numCounters,
					COLLATE_OP_BASIC);

    s_buffer[index].gExcl_min = (double_int **)TAU_UTIL_MALLOC(sizeof(double_int *)*Tau_Global_numCounters);
    s_buffer[index].gIncl_min = (double_int **)TAU_UTIL_MALLOC(sizeof(double_int *)*Tau_Global_numCounters);
    s_buffer[index].gExcl_max = (double_int **)TAU_UTIL_MALLOC(sizeof(double_int *)*Tau_Global_numCounters);
    s_buffer[index].gIncl_max = (double_int **)TAU_UTIL_MALLOC(sizeof(double_int *)*Tau_Global_numCounters);

    // Please note the use of Calloc
    for (int m=0; m<Tau_Global_numCounters; m++) {
      s_buffer[index].gExcl_min[m] = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numEvents);
      s_buffer[index].gIncl_min[m] = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numEvents);
      s_buffer[index].gExcl_max[m] = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numEvents);
      s_buffer[index].gIncl_max[m] = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numEvents);

    }

    if (rank == 0) {
      Tau_collate_allocateFunctionBuffers(&(s_buffer[index].sExcl), &(s_buffer[index].sIncl),
					  &(s_buffer[index].sNumCalls), &(s_buffer[index].sNumSubr),
					  numEvents,
					  Tau_Global_numCounters,
					  COLLATE_OP_DERIVED);
    }

    Tau_collate_compute_statistics_MPI_with_minmaxloc(s_buffer[index].functionUnifier, s_buffer[index].globalEventMap,
				   numEvents,
				   globalNumThreads, s_buffer[index].numEventThreads,
				   &(s_buffer[index].gExcl), &(s_buffer[index].gIncl),
				   &(s_buffer[index].gExcl_min), &(s_buffer[index].gIncl_min),
				   &(s_buffer[index].gExcl_max), &(s_buffer[index].gIncl_max),
                                   &(s_buffer[index].gNumCalls), &(s_buffer[index].gNumSubr),
				   &(s_buffer[index].sExcl), &(s_buffer[index].sIncl),
                                   &(s_buffer[index].sNumCalls), &(s_buffer[index].sNumSubr), MPI_COMM_WORLD);

    if(rank == 0) {
      for (int m=0; m<Tau_Global_numCounters; m++)  {
        for(int n=0; n<numEvents; n++) {
          fprintf(stderr, "Counter %d: The min exclusive, max exclusive, min inclusive, max inclusive values for event %d are located on processes %d, %d, %d and %d with values %f, %f, %f, %f\n", m, n, s_buffer[index].gExcl_min[m][n].index, s_buffer[index].gExcl_max[m][n].index, s_buffer[index].gIncl_min[m][n].index, s_buffer[index].gIncl_max[m][n].index, s_buffer[index].gExcl_min[m][n].value, s_buffer[index].gExcl_max[m][n].value, s_buffer[index].gIncl_min[m][n].value, s_buffer[index].gIncl_max[m][n].value);
        }
      }
    }

    /* End  interval event calculations */
    /* Start atomic statistic calculations */

    s_buffer[index].atomicUnifier = Tau_unify_getAtomicUnifier();
    numAtomicEvents = s_buffer[index].atomicUnifier->globalNumItems;

    s_buffer[index].numAtomicEventThreads =
      (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));
    s_buffer[index].globalAtomicEventMap = (int*)TAU_UTIL_MALLOC(numAtomicEvents*sizeof(int));

    // initialize all to -1
    for (int i=0; i<numAtomicEvents; i++) {
      // -1 indicates that the event did not occur for this rank
      s_buffer[index].globalAtomicEventMap[i] = -1;
    }
    for (int i=0; i<s_buffer[index].atomicUnifier->localNumItems; i++) {
      // set reverse mapping
      s_buffer[index].globalAtomicEventMap[s_buffer[index].atomicUnifier->mapping[i]] = i;
    }

    Tau_collate_get_total_threads_MPI(s_buffer[index].atomicUnifier, &globalNumThreads, &(s_buffer[index].numAtomicEventThreads),
				  numAtomicEvents, s_buffer[index].globalAtomicEventMap,true);

    Tau_collate_allocateAtomicBuffers(&(s_buffer[index].gAtomicMin), &(s_buffer[index].gAtomicMax),
				      &(s_buffer[index].gAtomicCalls), &(s_buffer[index].gAtomicMean),
				      &(s_buffer[index].gAtomicSumSqr),
				      numAtomicEvents,
				      COLLATE_OP_BASIC);
   s_buffer[index].gAtomicMin_min = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numAtomicEvents);
   s_buffer[index].gAtomicMax_max = (double_int *)TAU_UTIL_CALLOC(sizeof(double_int)*numAtomicEvents);

    if (rank == 0) {
      Tau_collate_allocateAtomicBuffers(&(s_buffer[index].sAtomicMin), &(s_buffer[index].sAtomicMax),
					&(s_buffer[index].sAtomicCalls), &(s_buffer[index].sAtomicMean),
					&(s_buffer[index].sAtomicSumSqr),
					numAtomicEvents,
					COLLATE_OP_DERIVED);
    }

    Tau_collate_compute_atomicStatistics_MPI_with_minmaxloc(s_buffer[index].atomicUnifier, s_buffer[index].globalAtomicEventMap,
					 numAtomicEvents,
					 globalNumThreads,
					 s_buffer[index].numAtomicEventThreads,
					 &(s_buffer[index].gAtomicMin), &(s_buffer[index].gAtomicMax),
					 &(s_buffer[index].gAtomicMin_min), &(s_buffer[index].gAtomicMax_max),
					 &(s_buffer[index].gAtomicCalls), &(s_buffer[index].gAtomicMean),
					 &(s_buffer[index].gAtomicSumSqr),
					 &(s_buffer[index].sAtomicMin), &(s_buffer[index].sAtomicMax),
					 &(s_buffer[index].sAtomicCalls), &(s_buffer[index].sAtomicMean),
					 &(s_buffer[index].sAtomicSumSqr), MPI_COMM_WORLD);


#ifdef TAU_ANALYTICS_INSTRUMENTATION_TOGGLE

   if(rank == 0) {
     std::list<std::pair<double, int> > sorted_list;

     for(int i = 0; i < numEvents; i++) {
       sorted_list.push_back(std::make_pair(s_buffer[index].sExcl[stat_mean_all][0][i], i));
     }

     sorted_list.sort(sort_func);
     std::list<std::pair<double, int> >::iterator it=sorted_list.begin();

     for(int i = 0; i < 5; i++, it++) {
       s_buffer[index].top_5_excl_time_mean.push_back(it->second);
     }
   }

   if(index)
     disable_instrumentation_if_necessary(index, rank);
#endif

   if(rank == 0) {
      for(int i=0; i<numAtomicEvents; i++)
        fprintf(stderr, "The min and max for atomic event %d lies with processes %d and %d with values %f and %f\n", i, s_buffer[index].gAtomicMin_min[i].index, s_buffer[index].gAtomicMax_max[i].index, s_buffer[index].gAtomicMin_min[i].value, s_buffer[index].gAtomicMax_max[i].value);
    }
  }

  index++;
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

