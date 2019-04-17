/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for atomic event trigger event
 * *
 * *********************************************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>

static int counter[1000];

int Tau_plugin_test_event_atomic_event_trigger(Tau_plugin_event_atomic_event_trigger_data_t* data) {
  
  counter[data->tid]++;

  return 0;
}

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t * data) {


if(data->tid == 0) { 
  for(int i = 0 ; counter[i] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d events\n", i, counter[i] + 1);
}

  return 0;
}

/* This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {

  for(int i = 0 ; i < 1000; i++)
    counter[i] = -1;

  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->AtomicEventTrigger = Tau_plugin_test_event_atomic_event_trigger;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

