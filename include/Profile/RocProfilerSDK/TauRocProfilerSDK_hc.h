//TauRocProfilerSDK_hc.h

#ifndef  PROFILE_SDKCOUNTERS_H
#define PROFILE_SDKCOUNTERS_H

#include <Profile/TauBfd.h>  // for name demangling

#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/callback_tracing.h>

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


//Map to identify kernels and some of their information
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;


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

//Enum to enable or disable metric profiling
typedef enum profile_metrics {
	NO_METRICS = 1,
	WRONG_NAME = 2,
	PROFILE_METRICS = 3
};


//Compatible Hardware Counter Profiling is only available at Rocprofiler 0.5 and newer versions
#if (ROCPROFILER_VERSION_MINOR > 4) && (ROCPROFILER_VERSION_MAJOR == 0)
#define PROFILE_SDKCOUNTERS
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>

extern std::string read_hc_record(void* payload, uint32_t kind, kernel_symbol_map_t client_kernels, uint64_t* agentid, double* counter_value);
extern int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer);

#else

std::string read_hc_record(void* payload, uint32_t kind, kernel_symbol_map_t client_kernels, uint64_t* agentid, double* counter_value)
{
  return std::string();
}
int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer)
{ 
  return NO_METRICS;
}
#endif //PROFILE_SDKCOUNTERS



#endif //PROFILE_SDKCOUNTERS_H