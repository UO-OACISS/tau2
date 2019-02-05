/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for end of execution event
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


int Tau_plugin_test_event_phase_entry(Tau_plugin_event_phase_entry_data_t* data) {
  
  std::cout << "TAU PLUGIN: Phase entry for phase: " << data->phase_name << std::endl;

  return 0;
}

int Tau_plugin_test_event_phase_exit(Tau_plugin_event_phase_exit_data_t* data) {
  
  std::cout << "TAU PLUGIN: Phase exit for phase: " << data->phase_name << std::endl;

  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->PhaseEntry = Tau_plugin_test_event_phase_entry;
  cb->PhaseExit = Tau_plugin_test_event_phase_exit;
  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

