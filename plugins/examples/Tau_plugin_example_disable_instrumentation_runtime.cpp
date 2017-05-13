/************************************************************************************************
 * *   Plugin Example - CPP Example
 * *   Demonstrates disabling of instrumentation at runtime using the function registration event
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

//externs
extern void printExcludeList(void);
extern int processInstrumentationRequests(char *fname);
extern bool instrumentEntity(const std::string& function_name);

/*This function gets invoked at function registration.
 * It checks if the function getting registrtion is present in the exclude list specified by the instrumentation file
 * and is so, sets the function group to TAU_DISABLE, effectively disabling function from getting instrumented*/
int Tau_plugin_example_check_and_set_disable_group(Tau_plugin_event_function_registration_data data) {
  
  const char * pch = strchr(((FunctionInfo *)data.function_info_ptr)->GetName(), ')');
  processInstrumentationRequests("select.tau");
  int position = (pch - ((FunctionInfo *)data.function_info_ptr)->GetName()) + 1;

  /*Check if function is .TAU application. If not, proceed to check if function needs to be instrumented*/
  if(strcmp(((FunctionInfo *)data.function_info_ptr)->GetName(), ".TAU application") != 0) {
    /*If function should not instrumented, set profile group to TAU_DISABLE*/
    if(!instrumentEntity(std::string(((FunctionInfo *)data.function_info_ptr)->GetName(), position))) {
      RtsLayer::LockDB();
      Tau_profile_set_group(data.function_info_ptr, TAU_DISABLE);
      ((FunctionInfo *)data.function_info_ptr)->SetPrimaryGroupName("TAU_DISABLE");
      RtsLayer::UnLockDB();
    }
  }

  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(PluginManager* plugin_manager) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  Tau_util_init_tau_plugin_callbacks(cb);
  cb->FunctionRegistrationComplete = Tau_plugin_example_check_and_set_disable_group;
  Tau_util_plugin_register_callbacks(cb);

  return 0;
}

