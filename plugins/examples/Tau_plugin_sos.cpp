/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


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

int Tau_plugin_sos_dump(Tau_plugin_event_dump_data data) {
  //printf("TAU PLUGIN SOS: dump\n");
 
  TAU_SOS_send_data();
 
  return 0;
}

int Tau_plugin_finalize(Tau_plugin_event_function_finalize_data data) {

  fprintf(stdout, "TAU PLUGIN SOS Finalize\n");

  TAU_SOS_finalize();

  return 0;
}

int Tau_plugin_sos_post_init(Tau_plugin_event_post_init_data data) {

  fprintf(stdout, "TAU PLUGIN SOS Post Init\n");

  TAU_SOS_send_data();

  return 0;
}

int Tau_plugin_sos_function_entry(Tau_plugin_event_function_entry_data data) {
  /* todo: filter on group, timer name */
  std::stringstream ss;
  ss << "TAU_EVENT_ENTRY:" << data.tid << ":" << data.timer_name;
  Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
  return 0;
}

int Tau_plugin_sos_function_exit(Tau_plugin_event_function_exit_data data) {
  /* todo: filter on group, timer name */
  std::stringstream ss;
  ss << "TAU_EVENT_EXIT:" << data.tid << ":" << data.timer_name;
  Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
  return 0;
}

int Tau_plugin_sos_send(Tau_plugin_event_send_data data) {
  /* todo: filter on group, timer name */
  std::stringstream ss;
  ss << "TAU_EVENT_SEND:" << data.tid 
      << ":" << data.message_tag 
      << ":" << data.destination 
      << ":" << data.bytes_sent;
  Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
  return 0;
}

int Tau_plugin_sos_recv(Tau_plugin_event_recv_data data) {
  /* todo: filter on group, timer name */
  std::stringstream ss;
  ss << "TAU_EVENT_RECV:" << data.tid 
      << ":" << data.message_tag 
      << ":" << data.source 
      << ":" << data.bytes_received;
  Tau_SOS_pack_long(ss.str().c_str(), data.timestamp);
  return 0;
}

int Tau_plugin_metadata_registration_complete_func(Tau_plugin_event_metadata_registration_data data) {
  // fprintf(stdout, "TAU Metadata registration\n");
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

  fprintf(stdout, "TAU PLUGIN SOS Init\n");
  TAU_SOS_init();
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->Dump = Tau_plugin_sos_dump;
  cb->PostInit = Tau_plugin_sos_post_init;
  cb->FunctionEntry = Tau_plugin_sos_function_entry;
  cb->FunctionExit = Tau_plugin_sos_function_exit;
  cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_func;
  cb->FunctionFinalize = Tau_plugin_finalize;
  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb);

  return 0;
}


