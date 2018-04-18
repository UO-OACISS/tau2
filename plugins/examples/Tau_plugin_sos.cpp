/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#if defined(TAU_SOS)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <TauSOS.h>

/* Only dump data to SOS if we aren't doing periodic dumps */
int Tau_plugin_sos_dump(Tau_plugin_event_dump_data data) {
    //printf("TAU PLUGIN SOS: dump\n");
    if (thePluginOptions().env_sos_periodic != 1) { 
        TAU_SOS_send_data();
    }
    return 0;
}

int Tau+plugin_sos_mpit(Tau_plugin_mpit_data data) {

    Tau_SOS_send_data();

    return 0;
}

/* This is a weird event, not sure what for */
int Tau_plugin_finalize(Tau_plugin_event_function_finalize_data data) {
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_sos_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data data) {
    //fprintf(stdout, "TAU PLUGIN SOS Pre-Finalize\n"); fflush(stdout);
    // OK to do it from any thread, because it came from MPI_Finalize
    TAU_SOS_send_data();
    /* We used to finalize now, but we no longer use MPI in the finalization
     * so it's ok to wait until all timers are done */
    /*
    if (data.tid == 0) {
        TAU_SOS_finalize();
    }
    */
    return 0;
}

/* This happens from Profiler.cpp, when data is written out. */
int Tau_plugin_sos_end_of_execution(Tau_plugin_event_end_of_execution_data data) {
    //fprintf(stdout, "TAU PLUGIN SOS Finalize\n"); fflush(stdout);
    if (data.tid == 0) {
        TAU_SOS_finalize();
    }
    return 0;
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_sos_post_init(Tau_plugin_event_post_init_data data) {
    //fprintf(stdout, "TAU PLUGIN SOS Post Init\n"); fflush(stdout);
    TAU_SOS_send_data();
    return 0;
}

/* This happens on Tau_start() */
int Tau_plugin_sos_function_entry(Tau_plugin_event_function_entry_data data) {
    /* First, check to see if we are including/excluding this timer */
    /*
    if (thePluginOptions().env_sos_use_selection == 1) {
        if (Tau_SOS_contains(thePluginOptions().excluded_timers, data.timer_name, false) ||
            !Tau_SOS_contains(thePluginOptions().included_timers, data.timer_name, true)) {
            return 0;
        }
    }
    */
    /* todo: filter on group, timer name */
    std::stringstream ss;
    ss << "TAU_EVENT_ENTRY:" << data.tid << ":" << data.timer_name;
    //std::cout << ss.str() << std::endl;
    Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
    return 0;
}

/* This happens on Tau_stop() */
int Tau_plugin_sos_function_exit(Tau_plugin_event_function_exit_data data) {
    /* First, check to see if we are including/excluding this timer */
    /*
    if (thePluginOptions().env_sos_use_selection == 1) {
        if (Tau_SOS_contains(thePluginOptions().excluded_timers, data.timer_name, false) ||
            !Tau_SOS_contains(thePluginOptions().included_timers, data.timer_name, true)) {
            return 0;
        }
    }
    */
    /* todo: filter on group, timer name */
    std::stringstream ss;
    ss << "TAU_EVENT_EXIT:" << data.tid << ":" << data.timer_name;
    //std::cout << ss.str() << std::endl;
    Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
    return 0;
}

/* This happens on Tau_userevent() */
int Tau_plugin_sos_atomic_trigger(Tau_plugin_event_atomic_event_trigger_data data) {
    /* First, check to see if we are including/excluding this counter */
    /*
    if (thePluginOptions().env_sos_use_selection == 1) {
        if (Tau_SOS_contains(thePluginOptions().excluded_counters, data.counter_name, false) ||
            !Tau_SOS_contains(thePluginOptions().included_counters, data.counter_name, true)) {
            return 0;
        }
    }
    */
    std::stringstream ss;
    ss << "TAU_EVENT_COUNTER:" << data.tid << ":" << data.counter_name;
    //std::cout << ss.str() << " = " << data.value << std::endl;
    Tau_SOS_pack_long(ss.str().c_str(), data.value);
    return 0;
}

/* This happens for special events from ADIOS, MPI */
int Tau_plugin_sos_current_timer_exit(Tau_plugin_event_current_timer_exit_data data) {
    Tau_SOS_pack_current_timer(data.name_prefix);
    return 0;
}

/* This happens on MPI_Send events (and similar) */
int Tau_plugin_sos_send(Tau_plugin_event_send_data data) {
    /* todo: filter on group, timer name */
    std::stringstream ss;
    ss << "TAU_EVENT_SEND:" << data.tid 
        << ":" << data.message_tag 
        << ":" << data.destination 
        << ":" << data.bytes_sent;
    //std::cout << ss.str() << std::endl;
    Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
    return 0;
}

/* This happens on MPI_Recv events (and similar) */
int Tau_plugin_sos_recv(Tau_plugin_event_recv_data data) {
    /* todo: filter on group, timer name */
    std::stringstream ss;
    ss << "TAU_EVENT_RECV:" << data.tid 
        << ":" << data.message_tag 
        << ":" << data.source 
        << ":" << data.bytes_received;
    //std::cout << ss.str() << std::endl;
    Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
    return 0;
}

/* This happens when a Metadata field is saved. */
int Tau_plugin_metadata_registration_complete_func(Tau_plugin_event_metadata_registration_data data) {
    //fprintf(stdout, "TAU Metadata registration\n"); fflush(stdout);
    std::stringstream ss;
    ss << "TAU_Metadata:" << 0 << ":" << data.name;
    switch(data.value->type) {
        case TAU_METADATA_TYPE_STRING:
            Tau_SOS_pack_string(ss.str().c_str(), data.value->data.cval);
            break;
        case TAU_METADATA_TYPE_INTEGER:
            Tau_SOS_pack_integer(ss.str().c_str(), data.value->data.ival);
            break;
        case TAU_METADATA_TYPE_DOUBLE:
            Tau_SOS_pack_double(ss.str().c_str(), data.value->data.dval);
            break;
        case TAU_METADATA_TYPE_TRUE:
            Tau_SOS_pack_string(ss.str().c_str(), const_cast<char*>("true"));
            break;
        case TAU_METADATA_TYPE_FALSE:
            Tau_SOS_pack_string(ss.str().c_str(), const_cast<char*>("false"));
            break;
        case TAU_METADATA_TYPE_NULL:
            Tau_SOS_pack_string(ss.str().c_str(), const_cast<char*>("(null)"));
            break;
        default:
            break;
    }
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    //fprintf(stdout, "TAU PLUGIN SOS Init\n"); fflush(stdout);
    // Parse our settings
    TAU_SOS_parse_environment_variables();
    // Check the value of TAU_SOS
    if (!thePluginOptions().env_sos_enabled) 
    { 
        printf("*** SOS NOT ENABLED! ***\n"); 
        return 0; 
    }
    TAU_SOS_init();
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
    /* Required event support */
    cb->Dump = Tau_plugin_sos_dump;
    cb->Mpit = Tau_plugin_sos_mpit;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_func;
    cb->PostInit = Tau_plugin_sos_post_init;
    cb->PreEndOfExecution = Tau_plugin_sos_pre_end_of_execution;
    cb->EndOfExecution = Tau_plugin_sos_end_of_execution;
    /* Event tracing support */
    if (thePluginOptions().env_sos_tracing) {
        cb->Send = Tau_plugin_sos_send;
        cb->Recv = Tau_plugin_sos_recv;
        cb->FunctionEntry = Tau_plugin_sos_function_entry;
        cb->FunctionExit = Tau_plugin_sos_function_exit;
        cb->AtomicEventTrigger = Tau_plugin_sos_atomic_trigger;
    }
    /* Specialized support for ADIOS, MPI events (ADIOS Skel/Pooky support) */
    if (thePluginOptions().env_sos_trace_adios) {
        cb->CurrentTimerExit = Tau_plugin_sos_current_timer_exit;
    }
    /* Not sure what this thing does */
    //cb->FunctionFinalize = Tau_plugin_finalize;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb);
    return 0;
}



#endif // TAU_SOS
