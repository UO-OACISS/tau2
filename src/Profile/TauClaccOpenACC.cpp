#ifndef TAU_PGI_OPENACC
#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <acc_prof.h>
#include <sstream>
#include <iostream>

#define TAU_ACC_NAME_LEN 4096
#define VERSION 0.1



#define TAU_SET_EVENT_NAME(event_name, str) event_name.append(str); break

/* cf Tau_ompt_finalized in TauOMPT.cpp */
static bool Tau_clacc_finalized(bool changeValue = false) {
    static bool _finalized = false;
    if (changeValue) {
        _finalized = true;
    }
    return _finalized;
}

////////////////////////////////////////////////////////////////////////////
    /*extern "C" static*/ void
Tau_openacc_callback( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
    std::string event_name;
    std::string user_event_name;
    std::string srcinfo{" "};
    std::string lineinfo{" {"};

    //acc_event_t *event_type_info = NULL;
    acc_data_event_info*   data_event_info = NULL;
    acc_launch_event_info* launch_event_info = NULL;
    //acc_other_event_info*  other_event_info = NULL;
    TAU_VERBOSE("event_type: %d\n", prof_info->event_type);


    switch (prof_info->event_type) {
        case acc_ev_device_init_start: {
            Tau_create_top_level_timer_if_necessary();
            TAU_SET_EVENT_NAME(event_name, ">openacc_init");
        }
        case acc_ev_device_init_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_init");
        }
        case acc_ev_device_shutdown_start: {
            TAU_SET_EVENT_NAME(event_name, ">openacc_shutdown");
        }
        case acc_ev_device_shutdown_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_shutdown");
        }
        case acc_ev_runtime_shutdown: {
            //TAU_SET_EVENT_NAME(event_name, "openacc_runtime_shutdown");
            return;
        }
        /*
        case acc_ev_done:
            TAU_SET_EVENT_NAME(event_name, "openacc_done");
        }
        case acc_ev_wait_start:
            TAU_SET_EVENT_NAME(event_name, ">openacc_wait");
        }
        case acc_ev_wait_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_wait");
        }
        case acc_ev_update_start: {
            TAU_SET_EVENT_NAME(event_name, ">openacc_update");
        }
        case acc_ev_update_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_update");
        }
        */
        case acc_ev_enter_data_start: {
            TAU_SET_EVENT_NAME(event_name, ">openacc_enter_data");
        }
        case acc_ev_enter_data_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_enter_data");
        }
        case acc_ev_exit_data_start: {
            TAU_SET_EVENT_NAME(event_name, ">openacc_exit_data");
        }
        case acc_ev_exit_data_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_exit_data");
        }

        case acc_ev_enqueue_launch_start: {
            /* kernel_name is always set to NULL (permitted by OpenACC), num_gangs, num_workers, and vector_length are omitted */
            if (event_info) {
                launch_event_info = &(event_info->launch_event);
                event_name.append(">openacc_enqueue_launch");
            }
            break;
        }
        case acc_ev_enqueue_launch_end: {
            if (event_info) { /* same */
                launch_event_info = &(event_info->launch_event);
                event_name.append("<openacc_enqueue_launch");
            }
            break;
        }
        case acc_ev_enqueue_upload_start: {
            if (event_info) {
                data_event_info = &(event_info->data_event);
                TAU_VERBOSE("UPLOAD start: Var_name = %s, bytes=%d \n",
                    data_event_info->var_name, event_info->data_event.bytes);
                if (data_event_info->var_name) {
                    user_event_name.append("Data transfer from host to device <variable=");
                    user_event_name.append(data_event_info->var_name);
                    user_event_name.append(">");
                } else {
                    user_event_name.append("Data transfer from host to device <other>");
                }
                TAU_TRIGGER_EVENT(user_event_name.c_str(), event_info->data_event.bytes);
            }
            TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_upload");
        }
        case acc_ev_enqueue_upload_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_upload");
        }
        case acc_ev_enqueue_download_start: {
            if (event_info) {
                data_event_info = &(event_info->data_event);
                TAU_VERBOSE("DOWNLOAD start: Var_name = %s, bytes=%d \n", data_event_info->var_name,
                        event_info->data_event.bytes);
                if (data_event_info->var_name) {
                    user_event_name.append("Data transfer from device to host <variable=");
                    user_event_name.append(data_event_info->var_name);
                    user_event_name.append(">");
                } else {
                    user_event_name.append("Data transfer from device to host <other>");
                }
                TAU_TRIGGER_EVENT(user_event_name.c_str(), event_info->data_event.bytes);
            }
            TAU_SET_EVENT_NAME(event_name, ">openacc_enqueue_download");
        }
        case acc_ev_enqueue_download_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_enqueue_download");
        }
        case acc_ev_compute_construct_start: {
            TAU_SET_EVENT_NAME(event_name, ">openacc_compute_construct");
        }
        case acc_ev_compute_construct_end: {
            TAU_SET_EVENT_NAME(event_name, "<openacc_compute_construct");
        }
        case acc_ev_create: {
            TAU_SET_EVENT_NAME(event_name, "openacc_create");
        }
        case acc_ev_delete: {
            TAU_SET_EVENT_NAME(event_name, "openacc_delete");
        }
        case acc_ev_alloc: {
            TAU_SET_EVENT_NAME(event_name, "openacc_alloc");
        }
        case acc_ev_free: {
            TAU_SET_EVENT_NAME(event_name, "openacc_free");
        }
        default: {
            TAU_SET_EVENT_NAME(event_name, "default");
        }
    }

    if (prof_info) {
        TAU_VERBOSE("Device=%d ", prof_info->device_number);
        TAU_VERBOSE("Thread=%d ", prof_info->thread_id); /* Seems to be always set to 0 */
        srcinfo.append(prof_info->func_name);
        srcinfo.append(" [{");
        srcinfo.append(prof_info->src_file);
        srcinfo.append("}");
        TAU_VERBOSE( "src info: %s\n", srcinfo.c_str() );
        TAU_VERBOSE( "Line nb %d\n", prof_info->line_no );
        lineinfo.append(std::to_string(prof_info->line_no));
        if (prof_info->end_line_no) {
            lineinfo.append("-");
            lineinfo.append(std::to_string(prof_info->end_line_no));
        }
        lineinfo.append("}");
        srcinfo.append(lineinfo);
        srcinfo.append("]");
        event_name.append(srcinfo);
    }
    const char* name = event_name.c_str();
    TAU_VERBOSE("%s\n", name+1 );
    if ( name[0] == '>') {
        TAU_VERBOSE("START>>%s\n", name+1 );
        TAU_START( name+1 );
    }  else if ( name[0] == '<' ) {
        TAU_VERBOSE("STOP <<%s\n", name+1 );
        Tau_global_stop();
    } else {
        TAU_VERBOSE("event_name = %s\n", name);
    }
}

/* Register the actions */

void acc_register_library( acc_prof_reg reg, acc_prof_reg unreg,
        acc_prof_lookup lookup ) {
    TAU_VERBOSE("Inside acc_register_library\n");
    static bool called{false};
    if (called) return;
    called = true;
    reg( acc_ev_device_init_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_device_init_end, &Tau_openacc_callback, acc_reg );
    /*
    reg( acc_ev_device_shutdown_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_device_shutdown_end, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_runtime_shutdown, &Tau_openacc_callback, acc_reg );
    */
    /*  reg( acc_ev_update_start, &Tau_openacc_callback, acc_reg );
        reg( acc_ev_update_end, &Tau_openacc_callback, acc_reg );
        reg( acc_ev_wait_start, &Tau_openacc_callback, acc_reg );
        reg( acc_ev_wait_end, &Tau_openacc_callback, acc_reg );*/
    reg( acc_ev_create, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_delete, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_alloc, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_free, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enter_data_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enter_data_end, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_exit_data_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_exit_data_end, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_compute_construct_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_compute_construct_end, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_launch_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_launch_end, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_upload_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_upload_end, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_download_start, &Tau_openacc_callback, acc_reg );
    reg( acc_ev_enqueue_download_end, &Tau_openacc_callback, acc_reg );
}


#endif // ndef TAU_PGI_OPENACC
