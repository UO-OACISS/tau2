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

static int counter[1000][7];

int Tau_plugin_test_event_atomic_event_trigger(Tau_plugin_event_atomic_event_trigger_data_t* data) {
  
  counter[data->tid][0]++;

  return 0;
}

int Tau_plugin_event_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t * data) {


if(data->tid == 0) { 
  for(int i = 0 ; counter[i][0] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d atomic events triggered\n", i, counter[i][0] + 1);

  for(int i = 0 ; counter[i][1] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d function entries \n", i, counter[i][1] + 1);

  for(int i = 0 ; counter[i][2] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d function exits \n", i, counter[i][2] + 1);

  for(int i = 0 ; counter[i][3] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d function registrations \n", i, counter[i][3] + 1);

  for(int i = 0 ; counter[i][4] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d interrupts \n", i, counter[i][4] + 1);

  for(int i = 0 ; counter[i][5] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d atomic event registration complete \n", i, counter[i][5] + 1);

  for(int i = 0 ; counter[i][6] != -1 && i < 1000; i++)
    fprintf(stderr, "Thread %d had %d post init \n", i, counter[i][6] + 1);
}

  return 0;
}

int Tau_plugin_test_event_function_entry(Tau_plugin_event_function_entry_data_t * data) {
  counter[data->tid][1]++;

  return 0;
}

int Tau_plugin_test_event_function_exit(Tau_plugin_event_function_exit_data_t * data) {
  counter[data->tid][2]++;

  return 0;
}

int Tau_plugin_test_event_function_registration(Tau_plugin_event_function_registration_data_t * data) {
  counter[data->tid][3]++;

  return 0;
}

int Tau_plugin_test_event_interrupt(Tau_plugin_event_interrupt_trigger_data_t * data) {
  counter[data->tid][4]++;

  return 0;
}

int Tau_plugin_test_event_atomic_event_registration(Tau_plugin_event_atomic_event_registration_data_t * data) {
  counter[data->tid][5]++;

  return 0;
}

int Tau_plugin_test_event_post_init(Tau_plugin_event_post_init_data_t * data) {
  counter[data->tid][6]++;

  return 0;
}

/* This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {

  for(int i = 0 ; i < 1000; i++) {
    for(int j = 0 ; j < 7; j++) 
    counter[i][j] = -1;
  }

  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->PostInit = Tau_plugin_test_event_post_init;
  cb->AtomicEventRegistrationComplete = Tau_plugin_test_event_atomic_event_registration;
  cb->InterruptTrigger = Tau_plugin_test_event_interrupt;
  cb->FunctionRegistrationComplete = Tau_plugin_test_event_function_registration;
  cb->FunctionEntry = Tau_plugin_test_event_function_entry;
  cb->FunctionExit = Tau_plugin_test_event_function_exit;
  cb->AtomicEventTrigger = Tau_plugin_test_event_atomic_event_trigger;
  cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

