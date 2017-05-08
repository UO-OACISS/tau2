#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <Profile/TauPlugin.h>

int Tau_plugin_test_event_function_registration_complete(Tau_plugin_event_function_registration_data data) {
  printf("I get called on function registration complete. Function %s has been registered for tid: %d\n", data.function_name, data.tid);
  return 0;
}

extern "C" int Tau_plugin_init_func(PluginManager* plugin_manager) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  Tau_util_init_tau_plugin_callbacks(cb);
  cb->FunctionRegistrationComplete = Tau_plugin_test_event_function_registration_complete;
  Tau_util_plugin_register_callbacks(cb);

  return 0;
}

