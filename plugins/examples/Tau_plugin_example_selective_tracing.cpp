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

int stop_tracing = 0;

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  RtsLayer::LockDB();

  for (int tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
    TauTraceClose(tid);
  }

  RtsLayer::UnLockDB();

  return 0;
}

int Tau_plugin_event_function_entry(Tau_plugin_event_function_entry_data_t* data) {

  if(stop_tracing)
    return 0;

  //fprintf(stderr, "TAU PLUGIN: Function %s with id %d has entered at timestamp %lu on tid: %d\n", data->timer_name, data->func_id, data->timestamp, data->tid);
  
  TauTraceEvent(data->func_id, 1 /* entry */, data->tid, data->timestamp, 1 /* use supplied timestamp */, TAU_TRACE_EVENT_KIND_FUNC);

  return 0;
}

int Tau_plugin_event_post_init(Tau_plugin_event_post_init_data_t *data) {

    RtsLayer::LockDB();

    for (int tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
      TauTraceInit(tid);
    }
    RtsLayer::UnLockDB();

    return 0;
}

int Tau_plugin_event_function_exit(Tau_plugin_event_function_exit_data_t* data) {

  if(stop_tracing)
    return 0;

  #ifdef TAU_MPI
  /* Initialized OTF2 */
/*  if(!strcmp(data->timer_name, "MPI_Init()")) {
  
    RtsLayer::LockDB();

    for (int tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
      TauTraceInit(tid);
    }

    RtsLayer::UnLockDB();
  }*/
  #endif

//  fprintf(stderr, "TAU PLUGIN: Function %s with id %d has exited at timestamp %lu on tid: %d\n", data->timer_name, data->func_id, data->timestamp, data->tid);
  
  TauTraceEvent(data->func_id, -1 /* entry */, data->tid, data->timestamp, 1 /* use supplied timestamp */, TAU_TRACE_EVENT_KIND_FUNC);
  
  #ifdef TAU_MPI
  if(!strcmp(data->timer_name, "MPI_Finalize()"))
    stop_tracing = 1;
  #endif
  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->PostInit =  Tau_plugin_event_post_init;
  cb->FunctionEntry = Tau_plugin_event_function_entry;
  cb->FunctionExit = Tau_plugin_event_function_exit;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

