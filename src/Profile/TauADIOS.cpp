/* 
 * ADIOS is freely available under the terms of the BSD license described
 * in the COPYING file in the top level directory of this source distribution.
 *
 * Copyright (c) 2008 - 2009.  UT-BATTELLE, LLC. All rights reserved.
 */

/* ADIOS Event Callback API - TAU Tool implementation
 *
 * This source file is a TAU implementation of the ADIOS callback API
 * for tools, or ADIOST.
 */

#define ADIOST_WEAK

#include "Profile/TauSOS.h"
#include "Profile/Profiler.h"
#include "Profile/UserEvent.h"
#include "Profile/TauMetrics.h"

#include "adiost_callback_api.h"
#include <stdint.h>
#define ADIOST_EXTERN extern "C"

ADIOST_EXTERN void my_open ( int64_t file_descriptor, adiost_event_type_t type,
    const char * group_name, const char * file_name, const char * mode) {
    if (type == adiost_event_enter) {
        //Tau_pure_start_task("ADIOS open to close", Tau_get_thread());
        Tau_pure_start_task("ADIOS open", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS open", Tau_get_thread());
    }
}

ADIOST_EXTERN void my_close(int64_t file_descriptor, adiost_event_type_t type) {
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS close", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS close", Tau_get_thread());
        //Tau_pure_stop_task("ADIOS open to close", Tau_get_thread());
    }
}

ADIOST_EXTERN void my_write( int64_t file_descriptor, adiost_event_type_t type) {
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS write", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS write", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_read( int64_t file_descriptor, adiost_event_type_t type) {
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS read", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS read", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_advance_step( int64_t file_descriptor,
    adiost_event_type_t type) {
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS advance step", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS advance step", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_group_size(int64_t file_descriptor, 
    adiost_event_type_t type, uint64_t data_size, uint64_t total_size) {
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS group size", Tau_get_thread());
    } else {
        Tau_trigger_context_event("ADIOS data size", (double)data_size);
        Tau_trigger_context_event("ADIOS total size", (double)total_size);
        Tau_pure_stop_task("ADIOS group size", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_transform( int64_t file_descriptor,
        adiost_event_type_t type) {
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS transform", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS transform", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_send_open_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath send open msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath send open msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_send_close_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath send close msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath send close msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_send_finalize_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath send finalize msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath send finalize msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_send_flush_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath send flush msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath send flush msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_send_var_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath send var msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath send var msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_process_open_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath process open msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath process open msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_process_close_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath process close msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath process close msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_process_finalize_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath process finalize msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath process finalize msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_process_flush_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath process flush msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath process flush msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_process_var_msg(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath process var msg", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath process var msg", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_fp_copy_buffer(int64_t file_descriptor,
        adiost_event_type_t type) { 
    if (type == adiost_event_enter) {
        Tau_pure_start_task("ADIOS flexpath copy buffer", Tau_get_thread());
    } else {
        Tau_pure_stop_task("ADIOS flexpath copy buffer", Tau_get_thread());
    }
} 

ADIOST_EXTERN void my_finalize(void) {
}

// This function is for checking that the function registration worked.
#define CHECK(EVENT,FUNCTION,NAME) \
    printf("TAU: Registering ADIOST callback %s...",NAME); \
    fflush(stderr); \
    if (adiost_fn_set_callback(EVENT, (adiost_callback_t)(FUNCTION)) != \
                    adiost_set_result_registration_success) { \
        printf("\n\tFailed to register ADIOST callback %s!\n",NAME); \
        fflush(stderr); \
    } else { \
        printf("success.\n"); \
    } \

ADIOST_EXTERN void TAU_adiost_initialize (adiost_function_lookup_t adiost_fn_lookup,
    const char *runtime_version, unsigned int adiost_version) {

    adiost_set_callback_t adiost_fn_set_callback = 
        (adiost_set_callback_t)adiost_fn_lookup("adiost_set_callback");

    fprintf(stderr,"Registering ADIOS tool events...\n"); fflush(stderr);
    CHECK(adiost_event_open,         my_open,          "adios_open");
    CHECK(adiost_event_close,        my_close,         "adios_close");
    CHECK(adiost_event_write,        my_write,         "adios_write");
    CHECK(adiost_event_read,         my_read,          "adios_read");
    CHECK(adiost_event_advance_step, my_advance_step,  "adios_advance_step");
    CHECK(adiost_event_group_size,   my_group_size,    "adios_group_size");
    CHECK(adiost_event_transform,    my_transform,     "adios_transform");
    CHECK(adiost_event_fp_send_open_msg, 
        my_fp_send_open_msg, "adios_fp_send_open_msg");
    CHECK(adiost_event_fp_send_close_msg, 
        my_fp_send_close_msg, "adios_fp_send_close_msg");
    CHECK(adiost_event_fp_send_finalize_msg, 
        my_fp_send_finalize_msg, "adios_fp_send_finalize_msg");
    CHECK(adiost_event_fp_send_flush_msg, 
        my_fp_send_flush_msg, "adios_fp_send_flush_msg");
    CHECK(adiost_event_fp_send_var_msg, 
        my_fp_send_var_msg, "adios_fp_send_var_msg");
    CHECK(adiost_event_fp_process_open_msg, 
        my_fp_process_open_msg, "adios_fp_process_open_msg");
    CHECK(adiost_event_fp_process_close_msg, 
        my_fp_process_close_msg, "adios_fp_process_close_msg");
    CHECK(adiost_event_fp_process_finalize_msg, 
        my_fp_process_finalize_msg, "adios_fp_process_finalize_msg");
    CHECK(adiost_event_fp_process_flush_msg, 
        my_fp_process_flush_msg, "adios_fp_process_flush_msg");
    CHECK(adiost_event_fp_process_var_msg, 
        my_fp_process_var_msg, "adios_fp_process_var_msg");
    CHECK(adiost_event_fp_copy_buffer, 
        my_fp_copy_buffer, "adios_fp_copy_buffer");
    CHECK(adiost_event_library_shutdown, my_finalize, "adios_finalize");
}

ADIOST_EXTERN adiost_initialize_t adiost_tool() { return TAU_adiost_initialize; }

