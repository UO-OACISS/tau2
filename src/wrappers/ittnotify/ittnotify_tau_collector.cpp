/****************************************************************************
**			TAU Portable Profiling Package                                 **
**			http://www.cs.uoregon.edu/research/tau                         **
*****************************************************************************
**    Copyright 2025                         						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich, ParaTools, Inc.                           **
****************************************************************************/
/****************************************************************************
**	File 		  : ittnotify_wrap_dynamic.cpp 			        	       **
**	Description   : Intel ITTNotify Collector 			                   **
**  Author        : Nicholas Chaimov                                       **
**	Contact		  : tau-bugs@cs.uoregon.edu               	               **
**	Documentation : See http://www.cs.uoregon.edu/research/tau             **
**                                                                         **
**      Description     : Collects task markers from ITTNotify             **
**                                                                         **
****************************************************************************/

/* Includes code from the Intel ITTNotify Collector Reference Implementation
   located at https://github.com/intel/ittapi/
   under the BSD 3-clause license:

   Copyright (c) 2019 Intel Corporation. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* This implements a basic ITTNotify collector for TAU.
   Intel ITTNotify is an API for applications to report information for use by Intel
   performance tools, such as VTune, among others.
   It consists of two parts:

     - The static part, a static library which the application links against.
       It provides an implementation of the ITTNotify API which will search
       for a collector and call corresponding functions in the collector if
       one exists.

     - The dynamic part, or the collector, a dynamic library which is called
       by the static part. Intel's tools provide a collector. This file implements
       a collector for TAU.

   A collector is set with the env var $INTEL_LIBITTNOTIFY64.

   A complication is that once a collector is registered, the static part will delegate all
   work to the dynamic part instead of performing the work itself. Thus, the collector
   must implement itself all API functions that are necessary for its functioning.
   This is why this collector must implement __itt_string_handle_create and __itt_domain_create.
   See https://github.com/intel/ittapi/issues/178 for more information.

   Useful resources:

     - Implementation of static part: 
         https://github.com/intel/ittapi/tree/master/src/ittnotify
   
     - Implementation of dynamic part:
         https://github.com/intel/ittapi/tree/master/src/ittnotify_refcol

     - API documentation:
         https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2025-0/instrumentation-tracing-technology-api-reference.html
*/


//#define TAU_DEBUG_ITTNOTIFY

#include <stdio.h>

#include <cstdint>
#include <cinttypes>
#include <string>
#include <sstream>
#include <stack>
#include <map>

#define INTEL_NO_MACRO_BODY
#define INTEL_ITTNOTIFY_API_PRIVATE
#include "ittnotify.h"
#include "ittnotify_config.h"
#include "jitprofiling.h"

#include "Profile/Profiler.h"

// Lock the mutex for access to ITTNotify shared linked lists.
// Adapted from https://github.com/intel/ittapi/blob/master/src/ittnotify/ittnotify_static.c
#define ITT_MUTEX_INIT_AND_LOCK(p) {                                 \
    if (PTHREAD_SYMBOLS)                                             \
    {                                                                \
        if (!p->mutex_initialized)                                    \
        {                                                            \
            if (__itt_interlocked_compare_exchange(&p->atomic_counter, 1, 0) == 0) \
            {                                                        \
                __itt_mutex_init(&p->mutex);                          \
                p->mutex_initialized = 1;                             \
            }                                                        \
            else                                                     \
                while (!p->mutex_initialized)                         \
                    __itt_thread_yield();                            \
        }                                                            \
        __itt_mutex_lock(&p->mutex);                                  \
    }                                                                \
}


// Maintain a stack for each domain. 
// __itt_task_end stops the most recently started task for a given domain.
using itt_stack_t = std::map<std::string, std::stack<std::string>>;

static itt_stack_t & itt_stack() {
    static thread_local itt_stack_t theStack;
    return theStack;
}

// The __itt_global holds pointers to global data, such as the 
// domain list and string handle list.
static __itt_global * itt_global_ptr = NULL;


// Fill in function pointers for the ITTNotify functions implemented
// by this collector.
static void fill_func_ptr_per_lib(__itt_global* p) {
    __itt_api_info* api_list = (__itt_api_info*)p->api_list_ptr;

    for (int i = 0; api_list[i].name != NULL; i++) {
#ifdef TAU_DEBUG_ITTNOTIFY
        fprintf(stderr, "filling func ptr for %s\n", api_list[i].name);
#endif
        *(api_list[i].func_ptr) = (void*)__itt_get_proc(p->lib, api_list[i].name);
        if (*(api_list[i].func_ptr) == NULL) {
            *(api_list[i].func_ptr) = api_list[i].null_func;
        }
    }
}

// Fill in function pointers and save pointer to global state.
ITT_EXTERN_C void ITTAPI __itt_api_init(__itt_global* p, __itt_group_id init_groups) {
    if (p != NULL) {
        (void)init_groups;
        fill_func_ptr_per_lib(p);
		itt_global_ptr = p;
#ifdef TAU_DEBUG_ITTNOTIFY
        fprintf(stderr, "__itt_api_init\n");
#endif
    }
    else {
        fprintf(stderr, "TAU ITTNotify Collector: Failed to initialize dynamic library\n");
    }
}

// Used by ITT_MUTEX_INIT_AND_LOCK
// Adapted from https://github.com/intel/ittapi/blob/master/src/ittnotify/ittnotify_static.c
static void __itt_report_error(int code, ...) {
    va_list args;
    va_start(args, code);
    if (itt_global_ptr->error_handler != NULL)
    {
        __itt_error_handler_t* handler = (__itt_error_handler_t*)(size_t)itt_global_ptr->error_handler;
        handler((__itt_error_code)code, args);
    }
    va_end(args);
}

// Create string handle.
// Stores in same linked list used by the static part.
// Adapted from https://github.com/intel/ittapi/blob/master/src/ittnotify/ittnotify_static.c
ITT_EXTERN_C __itt_string_handle * ITTAPI __itt_string_handle_create(const char *name) {
    if(name == NULL) {
        return NULL;
    }

    __itt_string_handle * head = itt_global_ptr->string_list;
    __itt_string_handle * tail = NULL;
    __itt_string_handle * result = NULL;
    
    if(PTHREAD_SYMBOLS) {
        ITT_MUTEX_INIT_AND_LOCK(itt_global_ptr);
    }

    // Search for existing
    
    for(__itt_string_handle * cur = head; cur != NULL; cur = cur->next) {
        tail = cur;
        if(cur->strA != NULL && !__itt_fstrcmp(cur->strA, name)) {
            result = cur;
#ifdef TAU_DEBUG_ITTNOTIFY
            fprintf(stderr, "Found string handle %s\n", result->strA);
#endif
            break;
        }
    }

    // Create if not existing

    if(result == NULL) {
        NEW_STRING_HANDLE_A(itt_global_ptr, result, tail, name);
#ifdef TAU_DEBUG_ITTNOTIFY
        fprintf(stderr, "Placed new string handle %s after %s\n", name, tail->strA);
#endif
    }

    if(PTHREAD_SYMBOLS) {
        __itt_mutex_unlock(&itt_global_ptr->mutex);
    }

#ifdef TAU_DEBUG_ITTNOTIFY
    fprintf(stderr, "Returning string handle %p for %s\n", result, name);
#endif
    return result;
}

// Create domain.
// Stores in same linked list used by the static part.
// Adapted from https://github.com/intel/ittapi/blob/master/src/ittnotify/ittnotify_static.c
ITT_EXTERN_C __itt_domain * ITTAPI __itt_domain_create(const char * name) {
    if(name == NULL) {
        return NULL;
    }

    __itt_domain * head = itt_global_ptr->domain_list;
    __itt_domain * tail = NULL;
    __itt_domain * result = NULL;

    if(PTHREAD_SYMBOLS) {
        ITT_MUTEX_INIT_AND_LOCK(itt_global_ptr);
    }

    // Search for existing
    
    for(__itt_domain * cur = head; cur != NULL; cur = cur->next) {
        tail = cur;
        if(cur->nameA != NULL && !__itt_fstrcmp(cur->nameA, name)) {
            result = cur;
#ifdef TAU_DEBUG_ITTNOTIFY
            fprintf(stderr, "Found domain %s\n", result->nameA);
#endif
            break;
        }
    }

    // Create if not existing

    if(result == NULL) {
        NEW_DOMAIN_A(itt_global_ptr, result, tail, name);
#ifdef TAU_DEBUG_ITTNOTIFY
        fprintf(stderr, "Placed new domain %s after %s\n", name, tail->nameA);
#endif
    }

    if(PTHREAD_SYMBOLS) {
        __itt_mutex_unlock(&itt_global_ptr->mutex);
    }

#ifdef TAU_DEBUG_ITTNOTIFY
    fprintf(stderr, "Returning domain %p for %s\n", result, name);
#endif
    return result;
}


// Start TAU timer for task and store the name in the stack for this domain.
ITT_EXTERN_C void ITTAPI __itt_task_begin(
    const __itt_domain *domain, __itt_id taskid, __itt_id parentid, __itt_string_handle *name) {
    if(domain == NULL) {
        fprintf(stderr, "TAU ITTNotify Collector: __itt_task_begin: domain should not be NULL\n");
        return;
    }
    const std::string domainName{domain->nameA};
    const std::string taskName{name->strA};
    std::stringstream ss;
    ss << domainName << "::" << taskName;
    const std::string timerName{ss.str()};
    itt_stack()[domainName].push(timerName);
    TAU_START(timerName.c_str());
#ifdef TAU_DEBUG_ITTNOTIFY
    fprintf(stderr, "task begin:  %s\n", timerName.c_str());
#endif
}

// Stop the TAU timer for the most-recently-started task for this domain.
ITT_EXTERN_C void ITTAPI __itt_task_end(const __itt_domain *domain)
{
    if(domain == NULL) {
        fprintf(stderr, "TAU ITTNotify Collector: __itt_task_end: domain should not be NULL\n");
        return;
    }
    const std::string domainName{domain->nameA};
    const std::string timerName{itt_stack()[domainName].top()};
    itt_stack()[domainName].pop();
    TAU_STOP(timerName.c_str());
#ifdef TAU_DEBUG_ITTNOTIFY
    fprintf(stderr, "task end: %s\n", timerName.c_str());
#endif
}

// Initialize JIT profiler
ITT_EXTERN_C iJIT_IsProfilingActiveFlags JITAPI Initialize(void) {
#ifdef TAU_DEBUG_ITTNOTIFY
    fprintf(stderr, "Initializing TAU ITT JIT Event Collector\n");
#endif
    return iJIT_SAMPLING_ON;
}

#ifdef TAU_DEBUG_ITTNOTIFY
static const char * jit_event_to_string(iJIT_JVM_EVENT event_type) {
    switch(event_type) {
        case iJVM_EVENT_TYPE_SHUTDOWN: return "iJVM_EVENT_TYPE_SHUTDOWN";
        case iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED: return "iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED";
        case iJVM_EVENT_TYPE_METHOD_UNLOAD_START: return "iJVM_EVENT_TYPE_METHOD_UNLOAD_START";
        case iJVM_EVENT_TYPE_METHOD_UPDATE: return "iJVM_EVENT_TYPE_METHOD_UPDATE";
        case iJVM_EVENT_TYPE_METHOD_INLINE_LOAD_FINISHED: return "iJVM_EVENT_TYPE_METHOD_INLINE_LOAD_FINISHED";
        case iJVM_EVENT_TYPE_METHOD_UPDATE_V2: return "iJVM_EVENT_TYPE_METHOD_UPDATE_V2";
        case iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2: return "iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2";
        case iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V3: return "iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V3";
        default: return "<UNKNOWN iJIT_JVM_EVENT>";
    }
}

static void jit_debug_print(iJIT_JVM_EVENT event_type, void *EventSpecificData) {
    fprintf(stderr, "JIT NotifyEvent: event_type=%s ", jit_event_to_string(event_type));
    
    if(event_type == iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED) {
        piJIT_Method_Load method_load_data = (piJIT_Method_Load)EventSpecificData;
        fprintf(stderr, "method_id=%u ", method_load_data->method_id);
        fprintf(stderr, "method_name=%s ", method_load_data->method_name);
        fprintf(stderr, "method_load_address=%p ", method_load_data->method_load_address);
        fprintf(stderr, "method_size=%u ", method_load_data->method_size);
        fprintf(stderr, "line_number_size=%u n", method_load_data->line_number_size);
        fprintf(stderr, "class_file_name=%s ", method_load_data->class_file_name);
        fprintf(stderr, "source_file_name=%s\n", method_load_data->source_file_name);
    }
}
#endif

// Register JIT address with TAU Sampling
ITT_EXTERN_C int JITAPI NotifyEvent(iJIT_JVM_EVENT event_type, void *EventSpecificData) {
#ifdef TAU_DEBUG_ITTNOTIFY
    jit_debug_print(event_type, EventSpecificData);
#endif
    if(event_type == iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED) {
        piJIT_Method_Load method_load_data = (piJIT_Method_Load)EventSpecificData;
        if(method_load_data->method_name == NULL) {
            fprintf(stderr, "TAU ITTNotify Collector: JIT method_name should not be NULL\n");
            return 0;
        }
        uintptr_t start = (uintptr_t) method_load_data->method_load_address;
        uintptr_t end = start + ((uintptr_t) method_load_data->method_size);
        char * name = strdup(method_load_data->method_name);
#ifdef TAU_DEBUG_ITTNOTIFY
        fprintf(stderr, "Registering range 0x%" PRIxPTR " to 0x%" PRIxPTR " with name %s\n", start, end, name);
#endif
        Tau_sampling_register_external_range(start, end, name);
    }

    return 0;
}

