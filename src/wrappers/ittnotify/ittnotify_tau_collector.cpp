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

//#define TAU_DEBUG_ITTNOTIFY

#include <stdio.h>

#include <string>
#include <sstream>
#include <stack>
#include <map>

#define INTEL_NO_MACRO_BODY
#define INTEL_ITTNOTIFY_API_PRIVATE
#include "ittnotify.h"
#include "ittnotify_config.h"

#include "Profile/Profiler.h"

using itt_stack_t = std::map<std::string, std::stack<std::string>>;

static itt_stack_t & itt_stack() {
    static thread_local itt_stack_t theStack;
    return theStack;
}

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

ITT_EXTERN_C void ITTAPI __itt_api_init(__itt_global* p, __itt_group_id init_groups) {
    if (p != NULL) {
        (void)init_groups;
        fill_func_ptr_per_lib(p);
#ifdef TAU_DEBUG_ITTNOTIFY
        fprintf(stderr, "__itt_api_init\n");
#endif
    }
    else {
        fprintf(stderr, "TAU ITTNotify Collector: Failed to initialize dynamic library\n");
    }
}

ITT_EXTERN_C void ITTAPI __itt_task_begin(
    const __itt_domain *domain, __itt_id taskid, __itt_id parentid, __itt_string_handle *name) {
    if(domain == NULL) {
        fprintf(stderr, "TAU ITTNotify Collector: __itt_task_begin: domain should not be NULL");
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

ITT_EXTERN_C void ITTAPI __itt_task_end(const __itt_domain *domain)
{
    if(domain == NULL) {
        fprintf(stderr, "TAU ITTNotify Collector: __itt_task_end: domain should not be NULL");
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


