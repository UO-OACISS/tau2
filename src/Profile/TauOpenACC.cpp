/****************************************************************************
 **            TAU Portable Profiling Package               **
 **            http://www.cs.uoregon.edu/research/tau               **
 *****************************************************************************
 **    Copyright 1997-2015                                     **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **    File         : TauOpenACC.cpp                  **
 **    Description     : TAU Profiling Package                  **
 **    Contact        : tau-bugs@cs.uoregon.edu                **
 **    Documentation    : See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////



#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <openacc.h>
#include "pgi_acc_prof.h"

#ifdef CUPTI
#include <cupti.h>
#include <cuda.h>
#endif

#include <sstream>

#include <Profile/Profiler.h>
#include <Profile/TauOpenACC.h>
#include <Profile/TauGpuAdapterOpenACC.h>
#include <Profile/TauGpu.h>
#include <Profile/TauBfd.h>

#define TAU_SET_EVENT_NAME(event_name, str) strcpy(event_name, str)
////////////////////////////////////////////////////////////////////////////

// strings for OpenACC parent constructs; based on enum acc_construct_t
const char* acc_constructs[] = {
    "parallel",   // 0
    "kernels",   // 1
    "loop",   // 2
    "data",   // 3
    "enter data",   // 4
    "exit data",   // 5
    "host data",   // 6
    "atomic",   // 7
    "declare",   // 8
    "init",   // 9
    "shutdown",   // 10
    "set",   // 11
    "update",   // 12
    "routine",   // 13
    "wait",   // 14
    "runtime api",   // 15
    "serial",   // 16
};

/* We need to validate the construct that PGI sets, because they give us
 * garbage like:
 * 269822688 or -887713968 */
const char* get_acc_construct(const acc_construct_t index) {
    static const char* unknown = "unknown";
    if (index < 0 || index > 16) {
        return unknown;
    }
    return acc_constructs[index];
}

extern "C" static void
Tau_openacc_launch_callback(acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info)
{
    acc_launch_event_info* launch_event = &(event_info->launch_event);
    int start = -1; // 0 = stop timer, 1 = start timer, -1 = something else; trigger event
    std::stringstream ss;

    switch(prof_info->event_type) {
        // note: these don't correspond to when kernels are actually run, just when they're put in the
        // execution queue on the device
        case acc_ev_enqueue_launch_start:
            start = 1;
            ss << "OpenACC enqueue launch: ";
            break;
        case acc_ev_enqueue_launch_end:
            start = 0;
            break;
        default:
            start = -1;
            ss << "UNKNOWN OPENACC LAUNCH EVENT";
            fprintf(stderr, "ERROR: Unknown launch event passed to OpenACC launch event callback.");
    }

    // if this is an end event, short circuit by grabbing the FunctionInfo pointer out of tool_info
    // and stopping that timer; if the pointer is NULL something bad happened, print warning and kill
    // whatever timer is on top of the stack
    if (start == 0) {
        if (launch_event->tool_info == NULL) {
            fprintf(stderr, "WARNING: OpenACC launch end event has bad matching start event.");
            Tau_global_stop();
        }
        else {
            Tau_lite_stop_timer(launch_event->tool_info);
        }
        return;
    }

    char * demangled = Tau_demangle_name(launch_event->kernel_name);
    if (demangled) {
        ss << demangled;
    }
    free(demangled);

    if (launch_event->implicit) {
        ss << " (implicit)";
    }
    if (launch_event->parent_construct < 9999) {
        ss << " "
           << get_acc_construct(launch_event->parent_construct);
    }

    ss << " [{"
        << prof_info->src_file
        << "} {"
        << prof_info->line_no
        << ","
        << prof_info->end_line_no
        << "}]";

    // if this is a start event, get the FunctionInfo and put it in tool_info so the end event will
    // get it to stop the timer
    if (start == 1) {
        void* func_info = Tau_get_function_info(ss.str().c_str(), "", TAU_USER, "TAU_OPENACC");
        launch_event->tool_info = func_info;
        Tau_lite_start_timer(func_info, 0);
        TAU_TRIGGER_CONTEXT_EVENT("OpenACC Gangs", launch_event->num_gangs);
        TAU_TRIGGER_CONTEXT_EVENT("OpenACC Workers", launch_event->num_workers);
        TAU_TRIGGER_CONTEXT_EVENT("OpenACC Vector Lanes", launch_event->vector_length);
    }
    else {
        TAU_TRIGGER_EVENT(ss.str().c_str(), 0);
    }
}

std::map<int, void*>& get_safe_profiler_map(void) {
    static std::map<int, void*> theMap;
    return theMap;
}

void populate_safe_profiler_map(void) {
    std::map<int, void*>& theMap = get_safe_profiler_map();
    theMap[acc_ev_enqueue_upload_start] =
        Tau_pure_search_for_function("OpenACC enqueue data transfer (HtoD)", 1);
    theMap[acc_ev_enqueue_download_start] =
        Tau_pure_search_for_function("OpenACC enqueue data transfer (DtoH)", 1);
}

    extern "C" static void
Tau_openacc_data_callback_signal_safe( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
    std::map<int, void*>& theMap = get_safe_profiler_map();
    switch(prof_info->event_type) {
        case acc_ev_enqueue_upload_start:
            Tau_start_timer(theMap[acc_ev_enqueue_upload_start], 0, Tau_get_thread());
            break;
        case acc_ev_enqueue_download_start:
            Tau_start_timer(theMap[acc_ev_enqueue_download_start], 0, Tau_get_thread());
            break;
        case acc_ev_enqueue_upload_end:
            Tau_stop_timer(theMap[acc_ev_enqueue_upload_start], Tau_get_thread());
            break;
        case acc_ev_enqueue_download_end:
            Tau_stop_timer(theMap[acc_ev_enqueue_download_start], Tau_get_thread());
            break;
        default:
            break;
    }
}

#if 0
    extern "C" static void
Tau_openacc_data_callback( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
    acc_data_event_info* data_event = &(event_info->data_event);
    char file_name[256];
    char event_name[256];
    char event_data[256];
    int start = -1;

    switch(prof_info->event_type) {
        case acc_ev_enqueue_upload_start:
            start = 1;
            snprintf(event_name, sizeof(event_name),  "OpenACC enqueue data transfer (HtoD)");
            break;
        case acc_ev_enqueue_upload_end:
            start = 0;
            break;
        case acc_ev_enqueue_download_start:
            start = 1;
            snprintf(event_name, sizeof(event_name),  "OpenACC enqueue data transfer (DtoH)");
            break;
        case acc_ev_enqueue_download_end:
            start = 0;
            break;
        case acc_ev_create:
            start = -1;
            snprintf(event_name, sizeof(event_name),  "OpenACC device data create");
            break;
        case acc_ev_delete:
            start = -1;
            snprintf(event_name, sizeof(event_name),  "OpenACC device data delete");
            break;
        case acc_ev_alloc:
            start = -1;
            snprintf(event_name, sizeof(event_name),  "OpenACC device alloc");
            break;
        case acc_ev_free:
            start = -1;
            snprintf(event_name, sizeof(event_name),  "OpenACC device free");
            break;
        default:
            start = -1;
            snprintf(event_name, sizeof(event_name),  "UNKNOWN OPENACC DATA EVENT");
            fprintf(stderr, "ERROR: Unknown data event passed to OpenACC data event callback.");
    }

    // if this is an end event, short circuit by grabbing the FunctionInfo pointer out of tool_info
    // and stopping that timer; if the pointer is NULL something bad happened, print warning and kill
    // whatever timer is on top of the stack
    if (start == 0) {
        if (data_event->tool_info == NULL) {
            fprintf(stderr, "WARNING: OpenACC launch end event has bad matching start event.");
            Tau_global_stop();
        }
        else {
            Tau_lite_stop_timer(data_event->tool_info);
        }
        return;
    }

    snprintf(file_name, sizeof(file_name),  "%s:%s-%s",
            prof_info->src_file,
            (prof_info->line_no > 0) ? std::to_string(prof_info->line_no).c_str() : "?",
            (prof_info->end_line_no > 0) ? std::to_string(prof_info->end_line_no).c_str() : "?");

    snprintf(event_data, sizeof(event_data),  " ; variable name = %s %s; parent construct = %s (%s)",
            //                               name ^  ^ (implicit move?)         ^ file and line no.
            (data_event->var_name) ? data_event->var_name : "unknown variable",
            (data_event->implicit) ? "(implicit move)" : "",
            (data_event->parent_construct < 9999) ? acc_constructs[data_event->parent_construct] : "unknown construct",
            file_name);

    strcat(event_name, event_data);

    // if this is a start event, get the FunctionInfo and put it in tool_info so the end event will
    // get it to stop the timer
    if (start == 1) {
        void* func_info = Tau_get_function_info(event_name, "", TAU_USER, "TAU_OPENACC");
        data_event->tool_info = func_info;
        Tau_lite_start_timer(func_info, 0);
        TAU_TRIGGER_EVENT(&event_name[0], data_event->bytes);
    }
    else {
        TAU_TRIGGER_EVENT(&event_name[0], data_event->bytes);
    }
}

#endif

    extern "C" static void
Tau_openacc_other_callback( acc_prof_info* prof_info, acc_event_info* event_info, acc_api_info* api_info )
{
    acc_other_event_info* other_event = &(event_info->other_event);
    int start = -1;
    std::stringstream ss;

    switch(prof_info->event_type) {
        case acc_ev_device_init_start:
            start = 1;
            ss << "OpenACC device init";
            break;
        case acc_ev_device_init_end:
            start = 0;
            break;
        case acc_ev_device_shutdown_start:
            start = 1;
            ss << "OpenACC device shutdown";
            break;
        case acc_ev_device_shutdown_end:
            start = 0;
            break;
        case acc_ev_runtime_shutdown:
            start = -1;
            ss << "OpenACC runtime shutdown";
            break;
        case acc_ev_enter_data_start:
            start = 1;
            ss << "OpenACC enter data";
            break;
        case acc_ev_enter_data_end:
            start = 0;
            break;
        case acc_ev_exit_data_start:
            start = 1;
            ss << "OpenACC exit data";
            break;
        case acc_ev_exit_data_end:
            start = 0;
            break;
        case acc_ev_update_start:
            start = 1;
            ss << "OpenACC update";
            break;
        case acc_ev_update_end:
            start = 0;
            break;
        case acc_ev_compute_construct_start:
            start = 1;
            ss << "OpenACC compute construct";
            break;
        case acc_ev_compute_construct_end:
            start = 0;
            break;
        case acc_ev_wait_start:
            start = 1;
            ss << "OpenACC wait";
            break;
        case acc_ev_wait_end:
            start = 0;
            break;
        default:
            start = -1;
            ss << "UNKNOWN OPENACC OTHER EVENT";
            fprintf(stderr, "ERROR: Unknown other event passed to OpenACC other event callback.");
            return;
    }

    // if this is an end event, short circuit by grabbing the FunctionInfo pointer out of tool_info
    // and stopping that timer; if the pointer is NULL something bad happened, print warning and kill
    // whatever timer is on top of the stack
    if (start == 0) {
        if (other_event->tool_info == NULL) {
            fprintf(stderr, "WARNING: OpenACC launch end event has bad matching start event.");
            Tau_global_stop();
        }
        else {
            Tau_lite_stop_timer(other_event->tool_info);
        }
        return;
    }

    if (other_event->implicit) {
        ss << " (implicit)";
    }
    if (other_event->parent_construct < 9999) {
        ss << " "
           << get_acc_construct(other_event->parent_construct);
    }

    if (prof_info->src_file != nullptr) {
        ss << " [{"
            << prof_info->src_file
            << "} {"
            << prof_info->line_no
            << ","
            << prof_info->end_line_no
            << "}]";
    }

#ifdef DEBUG_OPENACC
    printf("%s\n", ss.str().c_str());
#endif

    // if this is a start event, get the FunctionInfo and put it in tool_info so the end event will
    // get it to stop the timer
    if (start == 1) {
        void* func_info = Tau_get_function_info(ss.str().c_str(), "", TAU_USER, "TAU_OPENACC");
        other_event->tool_info = func_info;
        Tau_lite_start_timer(func_info, 0);
    }
    else {
        TAU_TRIGGER_EVENT(ss.str().c_str(), 0);
    }
}



#define CUPTI_CALL(call)                                                \
    do {                                                                  \
        CUptiResult _status = call;                                         \
        if (_status != CUPTI_SUCCESS) {                                     \
            const char *errstr;                                               \
            cuptiGetResultString(_status, &errstr);                           \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                       \
            exit(-1);                                                     \
        }                                                                   \
    } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))


//array enumerating CUpti_OpenAccEventKind strings
const char* openacc_event_names[] = {
    "OpenACC invalid event", // "CUPTI_OPENACC_EVENT_KIND_INVALD",
    "OpenACC device init", // "CUPTI_OPENACC_EVENT_KIND_DEVICE_INIT",
    "OpenACC device shutdown", // "CUPTI_OPENACC_EVENT_KIND_DEVICE_SHUTDOWN",
    "OpenACC runtime shutdown", // "CUPTI_OPENACC_EVENT_KIND_RUNTIME_SHUTDOWN",
    "OpenACC enqueue launch", // "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_LAUNCH",
    "OpenACC enqueue upload", // "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_UPLOAD",
    "OpenACC enqueue download", // "CUPTI_OPENACC_EVENT_KIND_ENQUEUE_DOWNLOAD",
    "OpenACC wait", // "CUPTI_OPENACC_EVENT_KIND_WAIT",
    "OpenACC implicit wait", // "CUPTI_OPENACC_EVENT_KIND_IMPLICIT_WAIT",
    "OpenACC compute construct", // "CUPTI_OPENACC_EVENT_KIND_COMPUTE_CONSTRUCT",
    "OpenACC update", // "CUPTI_OPENACC_EVENT_KIND_UPDATE",
    "OpenACC enter data", // "CUPTI_OPENACC_EVENT_KIND_ENTER_DATA",
    "OpenACC exit data", // "CUPTI_OPENACC_EVENT_KIND_EXIT_DATA",
    "OpenACC create", // "CUPTI_OPENACC_EVENT_KIND_CREATE",
    "OpenACC delete", // "CUPTI_OPENACC_EVENT_KIND_DELETE",
    "OpenACC alloc", // "CUPTI_OPENACC_EVENT_KIND_ALLOC",
    "OpenACC free" // "CUPTI_OPENACC_EVENT_KIND_FREE"
};

static size_t openacc_records = 0;
#ifdef CUPTI
    void
Tau_openacc_process_cupti_activity(CUpti_Activity *record)
{
    GpuEventAttributes* map = nullptr;
    int map_size = 0;
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_OPENACC_DATA:
            {
                CUpti_ActivityOpenAccData *oacc_data = (CUpti_ActivityOpenAccData*) record;
                if (oacc_data->deviceType != acc_device_nvidia) {
                    printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc_data->deviceType, acc_device_nvidia);
                    exit(-1);
                }

                map_size = 2;
                map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);

                static TauContextUserEvent* bytes;
                Tau_get_context_userevent((void**) &bytes, "Bytes transfered");
                map[0].userEvent = bytes;
                map[0].data = oacc_data->bytes;
                break;
            }
        case CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH:
            {
                CUpti_ActivityOpenAccLaunch *oacc_launch = (CUpti_ActivityOpenAccLaunch*) record;
                if (oacc_launch->deviceType != acc_device_nvidia) {
                    printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc_launch->deviceType, acc_device_nvidia);
                    exit(-1);
                }

                map_size = 4;
                map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);

                static TauContextUserEvent* gangs;
                Tau_get_context_userevent((void**) &gangs, "Num gangs");
                map[0].userEvent = gangs;
                map[0].data = oacc_launch->numGangs;

                static TauContextUserEvent* workers;
                Tau_get_context_userevent((void**) &workers, "Num workers");
                map[1].userEvent = workers;
                map[1].data = oacc_launch->numWorkers;

                static TauContextUserEvent* vector;
                Tau_get_context_userevent((void**) &vector, "Vector lanes");
                map[2].userEvent = vector;
                map[2].data = oacc_launch->vectorLength;

                break;
            }
        case CUPTI_ACTIVITY_KIND_OPENACC_OTHER:
            {
                CUpti_ActivityOpenAccData *oacc_other = (CUpti_ActivityOpenAccData*) record;
                if (oacc_other->deviceType != acc_device_nvidia) {
                    printf("Error: OpenACC device type is %u, not %u (acc_device_nvidia)\n", oacc_other->deviceType, acc_device_nvidia);
                    exit(-1);
                }

                map_size = 1;
                map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
                break;
            }

        default:
            {
                return;
            }
    }

    CUpti_ActivityOpenAcc* oacc = (CUpti_ActivityOpenAcc*) record;
    // TODO: are we guaranteed to only get openacc events? I don't know. Guess we'll find out.
    // always add duration at the end
    uint32_t context = oacc->cuContextId;
    uint32_t device = oacc->cuDeviceId;
    uint32_t stream = oacc->cuStreamId;
    uint32_t corr_id = oacc->externalId; // pretty sure this is right
    uint64_t start = oacc->start;
    uint64_t end = oacc->end;

    int task = oacc->cuThreadId;

    static TauContextUserEvent* duration;
    Tau_get_context_userevent((void**) &duration, "Duration");
    map[map_size-1].userEvent = duration;
    map[map_size-1].data = (double)(end - start) / (double)1e6;

    //TODO: could do better but this'll do for now
    const char* name = openacc_event_names[oacc->eventKind];

    Tau_openacc_register_gpu_event(name, device, stream, context, task, corr_id, map, map_size, start/1e3, end/1e3);

    openacc_records++;
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        printf("Error: out of memory\n");
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                Tau_openacc_process_cupti_activity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int) dropped);
        }
    }

    free(buffer);
}
#endif
void finalize()
{
#ifdef CUPTI
    cuptiActivityFlushAll(0);
#endif
    printf("Found %llu OpenACC records\n", (long long unsigned) openacc_records);
}

#ifdef CUPTI
//void Tau_cupti_onload(void);
#endif

////////////////////////////////////////////////////////////////////////////
    extern "C" void
acc_register_library(acc_prof_reg reg, acc_prof_reg unreg, acc_prof_lookup lookup)
{
    TAU_VERBOSE("Inside acc_register_library\n");

    // Launch events
    reg( acc_ev_enqueue_launch_start,      Tau_openacc_launch_callback, acc_reg );
    reg( acc_ev_enqueue_launch_end,        Tau_openacc_launch_callback, acc_reg );

    /* The data events aren't signal safe, likely because the memory transfers
     * happen as a page fault, which is a signal.  So, that means we can't
     * allocate any memory, which limits what we can do.  For that reason, we only
     * handle some events, and with static, preallocated timers. */
    // Data events
    populate_safe_profiler_map();
    reg( acc_ev_enqueue_upload_start,      Tau_openacc_data_callback_signal_safe, acc_reg );
    reg( acc_ev_enqueue_upload_end,        Tau_openacc_data_callback_signal_safe, acc_reg );
    reg( acc_ev_enqueue_download_start,    Tau_openacc_data_callback_signal_safe, acc_reg );
    reg( acc_ev_enqueue_download_end,      Tau_openacc_data_callback_signal_safe, acc_reg );
#if 0
    reg( acc_ev_create,                    Tau_openacc_data_callback, acc_reg );
    reg( acc_ev_delete,                    Tau_openacc_data_callback, acc_reg );
    reg( acc_ev_alloc,                     Tau_openacc_data_callback, acc_reg );
    reg( acc_ev_free,                      Tau_openacc_data_callback, acc_reg );
#endif
    // Other events
    reg( acc_ev_device_init_start,         Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_device_init_end,           Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_device_shutdown_start,     Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_device_shutdown_end,       Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_runtime_shutdown,          Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_enter_data_start,          Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_enter_data_end,            Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_exit_data_start,           Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_exit_data_end,             Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_update_start,              Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_update_end,                Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_compute_construct_start,   Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_compute_construct_end,     Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_wait_start,                Tau_openacc_other_callback, acc_reg );
    reg( acc_ev_wait_end,                  Tau_openacc_other_callback, acc_reg );


#ifdef CUPTI
    if (cuptiOpenACCInitialize((void*)reg, (void*)unreg, (void*)lookup) != CUPTI_SUCCESS) {
        printf("ERROR: failed to initialize CUPTI OpenACC support\n");
    }

    /* Initialize CUPTI handling */
    //Tau_cupti_onload(); // assume this has been done with the tau_exec -cupti flag!

    TAU_VERBOSE("Initialized CUPTI for OpenACC\n");

    CUptiResult cupti_err = CUPTI_SUCCESS;

    cupti_err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_DATA);
    cupti_err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_LAUNCH);
    cupti_err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OPENACC_OTHER);

    if (cupti_err != CUPTI_SUCCESS) {
        printf("ERROR: unable to enable some OpenACC CUPTI measurements\n");
    }

    /* No! Don't do this here, this call is in CuptiActivity.cpp */
    /*
       cupti_err = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);

       if (cupti_err != CUPTI_SUCCESS) {
       printf("ERROR: unable to register buffers with CUPTI\n");
       }
       */
#endif
    atexit(finalize);

} // acc_register_library

