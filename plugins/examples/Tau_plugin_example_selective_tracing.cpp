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

int Tau_plugin_event_function_entry(Tau_plugin_event_function_entry_data_t* data) {
  fprintf(stderr, "TAU PLUGIN: Function %s has entered at timestamp: %d\n", data->timer_name, data->timestamp);
  
  return 0;
}

int Tau_plugin_event_function_exit(Tau_plugin_event_function_exit_data_t* data) {
  fprintf(stderr, "TAU PLUGIN: Function %s has exited at timestamp: %d\n", data->timer_name, data->timestamp);
  
  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->FunctionEntry = Tau_plugin_event_function_entry;
  cb->FunctionExit = Tau_plugin_event_function_exit;
  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);
  //TAU_ADD_REGEX("(compute)(.*)");
  //TAU_DISABLE_PLUGIN_FOR_SPECIFIC_EVENT(TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION, "(compute)(.*)", id);

  return 0;
}

