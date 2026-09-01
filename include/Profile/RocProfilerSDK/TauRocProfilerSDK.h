//TauRocProfilerSDK.h
#ifndef _TAU_ROCMSDK_H_
#define _TAU_ROCMSDK_H_

#include <Profile/TauBfd.h>  // for name demangling
#include "Profile/Profiler.h"

//Need to check, are they all needed? 
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>


//Need to check, are they all needed? 
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <unistd.h>

// Size after which the kernel map is processed
#define DISPATCH_MAP_MAX_SIZE 128

//Map to identify kernels and some of their information, GPU side
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
kernel_symbol_map_t           client_kernels   = {};

// Map to sort the kernel dispatches map[<time,stream>]=dispath_info
// We also need a mutex for the map
static std::map<std::pair<uint64_t, uint64_t>, rocprofiler_callback_tracing_kernel_dispatch_data_t> kernel_time_stream_map;
static std::mutex kernel_dispatch_map_mutex;


//Map to identify kernels and some of their information, CPU side
//using host_function_data_t =
//    rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t;
//using host_functions_map_t = std::unordered_map<uint64_t, host_function_data_t>;
//kernel_symbol_map_t           cpu_client_kernels   = {};

//May move to include/Profile/nccl/types.h
// along with https://github.com/UO-OACISS/tau2/blob/7409bb076e38c065f4266a10005e43590e9ec135/src/Profile/TauNCCL.cpp#L356
const char* ncclDataTypeName(ncclDataType_t datatype)
{
    switch (datatype)
    {
        case ncclInt8:     return "ncclInt8";
        case ncclUint8:    return "ncclUint8";
        case ncclInt32:   return "ncclInt32";
        case ncclUint32:  return "ncclUint32";
        case ncclInt64:   return "ncclInt64";
        case ncclUint64:  return "ncclUint64";
        case ncclFloat16: return "ncclFloat16";
        case ncclFloat32: return "ncclFloat32";
        case ncclFloat64: return "ncclFloat64";
        case ncclBfloat16:return "ncclBfloat16";
        default:          return "Unknown";
    }
}

static uint8_t ncclDataTypeToSize(ncclDataType_t dt)
{
    switch (dt)
    {
        case ncclInt8:
        case ncclUint8:
            return 1;

        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;

        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;

        case ncclFloat16:
        case ncclBfloat16:
            return 2;

        default:
            return 0;
    }
}


// The delta timestamp is in microseconds between CPU and GPU.
static double deltaTimestamp_ms = 0;

extern "C" x_uint64 TauTraceGetTimeStamp();
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" void Tau_metadata_task(const char *name, const char* value, int tid);
extern int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc);
extern void sdk_pc_sampling_flush();
extern std::map<rocprofiler_dispatch_id_t, std::pair<int, double>> dispatch_kernel_time;


//We want to use the same IDs for the GPUs between the sampling and tracing.
// the device handle gives a number, but this map starts from 0
std::map< uint64_t, uint32_t> TAU_rocsdk_id_device_map;


//Enum to enable or disable metric profiling
typedef enum profile_metrics {
	NO_METRICS = 1,
	WRONG_NAME = 2,
	PROFILE_METRICS = 3
} profile_metrics;

#ifndef ROCPROFILER_CALL
#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg << " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }
#endif


//Function
rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t agents_ver,
                       const void** agents_arr,
                       size_t       num_agents,
                       void*        udata)
{
    if(agents_ver == ROCPROFILER_AGENT_INFO_VERSION_NONE)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
    auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
    for(size_t i = 0; i < num_agents; ++i)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
        agents_v->emplace_back(*agent);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}


// Check if there are any GPU agents
//We also can get the GPU properties, maybe should add as metadata
std::vector<rocprofiler_agent_v0_t> get_gpu_device_agents()
{
    std::vector<rocprofiler_agent_v0_t> agents;
    std::vector<rocprofiler_agent_v0_t> gpu_agents;
    // Query the agents, only a single callback is made that contains a vector
    // of all agents.
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                            &query_available_agents,
                                            sizeof(rocprofiler_agent_t),
                                            const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");
    for(const auto& agent : agents)
    {
        //int a_type = 1;
        if(agent.type == ROCPROFILER_AGENT_TYPE_NONE )
        {
            printf("!! FOUND AGENT WITH TYPE NONE\n");
            continue;
        }
        if(agent.type == ROCPROFILER_AGENT_TYPE_GPU) 
        {
            gpu_agents.push_back(agent);
            //a_type = 0;
        }
        
        //printf("Agent %lu node %u CPU(0)GPU(1) %d\n", agent.id.handle, agent.node_id, a_type);
        TAU_rocsdk_id_device_map[agent.id.handle] = agent.node_id;
    }
    return gpu_agents;
}

//Function to initizalize deltaTimestamps_ns
// we want this function to only be called once
bool run_once() {
    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    double startTimestampCPU = TauTraceGetTimeStamp(); // TAU is in microseconds!
    uint64_t startTimestampGPU;
    rocprofiler_get_timestamp(&startTimestampGPU); //rocprofiler is in nanoseconds!
    startTimestampCPU += (TauTraceGetTimeStamp()); // TAU is in microseconds!
    startTimestampCPU = startTimestampCPU / 2;

    // assume CPU timestamp is greater than GPU
    TAU_VERBOSE("HIP timestamp: %lf\n", startTimestampGPU);
    TAU_VERBOSE("CPU timestamp: %lf\n", startTimestampCPU);
    deltaTimestamp_ms = startTimestampCPU - ((double)startTimestampGPU/1e3);
    TAU_VERBOSE("HIP delta timestamp: %lf\n", deltaTimestamp_ms);
    return true;
}

double Tau_rocprofsdk_synchronized_gpu_timestamp(int tid, double value){
  double sync_ts = deltaTimestamp_ms+value;
  metric_set_gpu_timestamp(tid, sync_ts);
  return sync_ts;
}

void Tau_add_metadata_for_task(const char *key, int value, int taskid) {
  char buf[1024];
  snprintf(buf, sizeof(buf),  "%d", value);
  Tau_metadata_task(key, buf, taskid);
  TAU_VERBOSE("Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
}

std::string demangle_kernel_rocprofsdk(std::string k_name, int add_filename)
{
    std::string task_name;
    //__omp_offloading_36_523fe22f_compute_target_l105.kd
    static std::string omp_off_string = "__omp_offloading";
    //Each GPU implementation shows the name in a similar way,
    // but some are demangled and anothers demangled,
    // in the case of AMD, they seem to be demangled
    if( strncmp(k_name.c_str(), omp_off_string.c_str(), omp_off_string.length())==0)
    {
        int pos_key=omp_off_string.length();
        for(int i =0; i<2; i++)
        {
            pos_key = k_name.find_first_of('_', pos_key + 1);
        }
        int pos_ll = k_name.find_last_of("l");
        task_name = "OMP OFFLOADING ";
        task_name = task_name  + Tau_demangle_name(k_name.substr(pos_key+1,pos_ll-pos_key-2).c_str());
        if(add_filename == 0)
            return task_name;
        std::string s_omp_line = k_name.substr(pos_ll+1,k_name.find_last_of(".")-pos_ll-1);
        task_name = task_name + " [{UNRESOLVED} {";
        task_name = task_name + s_omp_line;
        task_name = task_name + ",0}]";
    }
    else
    {
        task_name = Tau_demangle_name(k_name.c_str());
    }
    return task_name;
}

#endif // _TAU_ROCMSDK_H_