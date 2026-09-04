//TauRocProfilerSDK_hc.h

#ifndef  PROFILE_SDKCOUNTERS_H
#define PROFILE_SDKCOUNTERS_H

#include <Profile/TauBfd.h>  // for name demangling
#include "Profile/Profiler.h"
#include <Profile/TauEnv.h>


//Enum to enable or disable metric profiling
typedef enum profile_metrics {
	NO_METRICS = 1,
	WRONG_NAME = 2,
	PROFILE_METRICS = 3
} profile_metrics;

#include <vector>
#include <cstdint>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <map>
#include <iostream>
#include <string>
#include <string_view>


#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

//Map to identify kernels and some of their information
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

//The callback for the hardware counters does not have timers, we want to tie the callback
// to the ending timer of the kernel dispatch and also know the taskid we need to use
//std::map<rocprofiler_dispatch_id_t, std::pair<int, double>> dispatch_kernel_time;

extern kernel_symbol_map_t           client_kernels;
extern std::string demangle_kernel_rocprofsdk(std::string k_name, int add_filename);

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

//Compatible Hardware Counter Profiling is only available at Rocprofiler 1 and newer versions
#if (ROCPROFILER_VERSION_MAJOR >= 1)
    #define PROFILE_SDKCOUNTERS
#else
    #warning "This rocprofiler-sdk version is unable to profile hardware counters, minimum 1.0"
#endif //(ROCPROFILER_VERSION_MAJOR >= 1)


#ifdef PROFILE_SDKCOUNTERS
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/callback_tracing.h>

extern int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer);
extern void register_kernel_dispatch(rocprofiler_kernel_dispatch_info_t dispatch_info, rocprofiler_timestamp_t end_timestamp);
void get_rocsdk_counters(rocprofiler_dispatch_id_t dispatch_id, int taskid, double curr_ts);

#else // No PROFILE_SDKCOUNTERS
typedef rocprofiler_profile_config_id_t rocprofiler_counter_config_id_t;


int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer)
{ 
  const char* rocm_metrics=std::getenv("ROCM_METRICS");
  if( rocm_metrics )
    printf("[TAU] ROCM Metrics not available for this rocprofiler-sdk version.\n");
  else{
    const char* rocsdk_metrics = TauEnv_get_rocsdk_metrics();
    if (rocsdk_metrics[0] != '\0')
      printf("[TAU] ROCM Metrics not available for this rocprofiler-sdk version.\n");
  }
  return NO_METRICS;
}

void register_kernel_dispatch(rocprofiler_counter_config_id_t dispatch_info, rocprofiler_timestamp_t end_timestamp)
{}

void get_rocsdk_counters(rocprofiler_dispatch_id_t dispatch_id, int taskid, double curr_ts)
{}

#endif //PROFILE_SDKCOUNTERS



#endif //PROFILE_SDKCOUNTERS_H