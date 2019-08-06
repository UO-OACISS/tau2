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

#ifdef TAU_MPI
#include <mpi.h>

int Tau_plugin_event_end_of_execution_null(Tau_plugin_event_end_of_execution_data_t *data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_trigger_null(Tau_plugin_event_trigger_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_dump_null(Tau_plugin_event_dump_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_metadata_registration_complete_null(Tau_plugin_event_metadata_registration_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_null(Tau_plugin_event_post_init_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}  

int Tau_plugin_event_send_null(Tau_plugin_event_send_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_recv_null(Tau_plugin_event_recv_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_entry_null(Tau_plugin_event_function_entry_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_exit_null(Tau_plugin_event_function_exit_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_atomic_trigger_null(Tau_plugin_event_atomic_event_trigger_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    /* Required event support */
    cb->Trigger = Tau_plugin_event_trigger_null;
    cb->Dump = Tau_plugin_event_dump_null;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_null;
    cb->PostInit = Tau_plugin_event_post_init_null;
    cb->EndOfExecution = Tau_plugin_event_end_of_execution_null;

    /* Trace events */
    cb->Send = Tau_plugin_event_send_null;
    cb->Recv = Tau_plugin_event_recv_null;
    cb->FunctionEntry = Tau_plugin_event_function_entry_null;
    cb->FunctionExit = Tau_plugin_event_function_exit_null;
    cb->AtomicEventTrigger = Tau_plugin_event_atomic_trigger_null;

    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

    return 0;
}

#endif
