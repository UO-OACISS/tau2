#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>


void printExcludeList(void);
int processInstrumentationRequests(char *fname);
bool instrumentEntity(const std::string& function_name);

int Tau_plugin_example_check_and_set_disable_group(Tau_plugin_event_function_registration_data data) {
  
  const char * pch = strchr(((FunctionInfo *)data.function_info_ptr)->GetName(), ')');
  processInstrumentationRequests("select.tau");
  int position = (pch - ((FunctionInfo *)data.function_info_ptr)->GetName()) + 1;

  if(strcmp(((FunctionInfo *)data.function_info_ptr)->GetName(), ".TAU application") != 0) {
    if(!instrumentEntity(std::string(((FunctionInfo *)data.function_info_ptr)->GetName(), position))) {
      RtsLayer::LockDB();
      Tau_profile_set_group(data.function_info_ptr, TAU_DISABLE);
      ((FunctionInfo *)data.function_info_ptr)->SetPrimaryGroupName("TAU_DISABLE");
      RtsLayer::UnLockDB();
    }
  }

  return 0;
}

extern "C" int Tau_plugin_init_func(PluginManager* plugin_manager) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  Tau_util_init_tau_plugin_callbacks(cb);
  cb->FunctionRegistrationComplete = Tau_plugin_example_check_and_set_disable_group;
  Tau_util_plugin_register_callbacks(cb);

  return 0;
}

