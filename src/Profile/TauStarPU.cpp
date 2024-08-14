#include <TAU.h>
#include <Profile/TauBfd.h>

#include <starpu_profiling_tool.h>
#include <starpu.h>

#include <sstream>
#include <stack> // for stack
#include <map> // for map
#include <iostream>
#include <dlfcn.h> // link with -ldl -rdynamic a

#include <stdio.h>

//#warning "Compiling StarPU support"

std::map<int,std::string> device_types;
std::map<int,std::string> event_types;

extern "C" void Tau_profile_exit_all_threads(void);

void runOnExitStarPU() {
  Tau_destructor_trigger();
}

extern "C" void Tau_set_thread_fake(int tid);
extern "C" void Tau_set_fake_thread_use_cpu_metric(int tid);
extern "C" char* Tau_ompt_resolve_callsite_eagerly(unsigned long addr, char * resolved_address);

/* All the callbacks are handled by this function */

void myfunction_cb( struct starpu_prof_tool_info* prof_info,  union starpu_prof_tool_event_info* event_info, struct starpu_prof_tool_api_info* api_info ){

    std::string event_name {event_types[prof_info->event_type]};
    std::string device_name {device_types[prof_info->driver_type]};
    std::stringstream info;
    int tag = 0;

    bool enter = true;
    switch(  prof_info->event_type ) {
    case starpu_prof_tool_event_init:
    case starpu_prof_tool_event_init_begin:
    case starpu_prof_tool_event_driver_init:
        break;
    case starpu_prof_tool_event_terminate:
    case starpu_prof_tool_event_init_end:
    case starpu_prof_tool_event_driver_deinit:
    case starpu_prof_tool_event_driver_init_end:
    case starpu_prof_tool_event_end_cpu_exec:
    case starpu_prof_tool_event_end_gpu_exec:
    case starpu_prof_tool_event_end_transfer:
        enter = false;
        break;
    case starpu_prof_tool_event_driver_init_start:
        info << " : " << device_name.c_str(); // << ":" << prof_info->device_number  << "}]";
        event_name = event_name + info.str();
        break;
    case starpu_prof_tool_event_start_cpu_exec:
    case starpu_prof_tool_event_start_gpu_exec:
        if(TauEnv_get_ompt_resolve_address_eagerly()) {
            char resolved_address[4096] = {0};
            Tau_ompt_resolve_callsite_eagerly((unsigned long)prof_info->fun_ptr, resolved_address);
            info << resolved_address;
        } else {
            info << " : ADDR <" << std::hex << prof_info->fun_ptr << ">";
        }
        info << " [{task " << prof_info->task_name <<  " model " << prof_info->model_name << "}]";
        event_name = event_name + info.str();
        break;
    case starpu_prof_tool_event_start_transfer:
        TAU_TRACE_SENDMSG( tag, prof_info->memnode, prof_info->bytes_transfered );
        info << " [{ memnode " << prof_info->memnode << " }]";
        //        info << " task " << prof_info->task_name <<  " model " << prof_info->model_name;
        event_name = event_name + info.str();
        break;
    default:
        std::cout <<  "Unknown callback " <<  prof_info->event_type << std::endl;
        break;
    }

    void * handle = nullptr;
    static thread_local std::stack<void*> myts;
    int tid = RtsLayer::myThread();
    if (enter) {
        handle = Tau_get_function_info(event_name.c_str(), "", TAU_DEFAULT, "TAU_STARPU");
        Tau_start_timer(handle,0,tid);
        myts.push(handle);
    }  else {
        if (myts.size() == 0) {
            TAU_VERBOSE("TAU tid: %u, Pthread tid: %u, Event: %s %s; Timer stack is empty, bug in StarPU support!\n",
                tid, RtsLayer::getTid(), event_name.c_str(), enter?"entry":"exit"); fflush(stderr);
            return;
        }
        handle = myts.top();
        Tau_stop_timer(handle, tid);
        myts.pop();
    }
}

/* Library initialization: callback registration */
extern "C" {
 void starpu_prof_tool_library_register( starpu_prof_tool_entry_register_func reg, starpu_prof_tool_entry_register_func unreg){

    device_types[starpu_prof_tool_driver_cpu] = "CPU";
    device_types[starpu_prof_tool_driver_gpu] = "GPU";
    device_types[starpu_prof_tool_driver_hip] = "HIP";
    device_types[starpu_prof_tool_driver_ocl] = "OpenCL";

    event_types[starpu_prof_tool_event_none] = "StarPU None";
    event_types[starpu_prof_tool_event_init] = "StarPU";
    event_types[starpu_prof_tool_event_terminate] = "StarPU";
    event_types[starpu_prof_tool_event_init_begin] = "StarPU init";
    event_types[starpu_prof_tool_event_init_end] = "StarPU init";
    event_types[starpu_prof_tool_event_driver_init] = "StarPU driver ";
    event_types[starpu_prof_tool_event_driver_deinit] = "StarPU driver ";
    event_types[starpu_prof_tool_event_driver_init_start] = "StarPU driver init ";
    event_types[starpu_prof_tool_event_driver_init_end] = "StarPU driver init ";
    event_types[starpu_prof_tool_event_start_cpu_exec] = "StarPU exec ";
    event_types[starpu_prof_tool_event_end_cpu_exec] = "StarPU exec ";
    event_types[starpu_prof_tool_event_start_gpu_exec] = "StarPU exec ";
    event_types[starpu_prof_tool_event_end_gpu_exec] = "StarPU exec ";
    event_types[starpu_prof_tool_event_start_transfer] = "StarPU transfer ";
    event_types[starpu_prof_tool_event_end_transfer] = "StarPU transfer ";
    event_types[starpu_prof_tool_event_user_start] = "StarPU user event ";
    event_types[starpu_prof_tool_event_user_end] = "StarPU user event ";

    enum  starpu_prof_tool_command info = starpu_prof_tool_command_reg;
    reg( starpu_prof_tool_event_init_begin, &myfunction_cb, info );
    reg( starpu_prof_tool_event_init_end, &myfunction_cb, info );
    reg( starpu_prof_tool_event_init, &myfunction_cb, info );
    reg( starpu_prof_tool_event_terminate, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_init, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_deinit, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_init_start, &myfunction_cb, info );
    reg( starpu_prof_tool_event_driver_init_end, &myfunction_cb, info );
    reg( starpu_prof_tool_event_start_cpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_end_cpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_start_gpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_end_gpu_exec, &myfunction_cb, info );
    reg( starpu_prof_tool_event_start_transfer, &myfunction_cb, info );
    reg( starpu_prof_tool_event_end_transfer, &myfunction_cb, info );

    atexit( runOnExitStarPU );
}
}


