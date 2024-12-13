//TauRocProfilerSDK_pc.h

#ifndef SAMPLING_SDKPC_H
#define SAMPLING_SDKPC_H

#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>

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
#include <iomanip>

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

  
//Due to some bugs, PC Sampling is available, but does not work in older versions
// do not compile for 4.0 and older
//Also, the implementation is not fully done, in future releases, we may
// be able to get stall reasons
#if (ROCPROFILER_VERSION_MINOR > 4) && (ROCPROFILER_VERSION_MAJOR == 0) && defined(TAU_ENABLE_ROCPROFILERSDK_PC)
#define SAMPLING_SDKPC
#include "Profile/RocProfilerSDK/TauRocProfilerSDK_add_tr.hpp"

constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

using avail_configs_vec_t         = std::vector<rocprofiler_pc_sampling_configuration_t>;

struct tool_agent_info
{
    rocprofiler_agent_id_t               agent_id;
    std::unique_ptr<avail_configs_vec_t> avail_configs;
    const rocprofiler_agent_t*           agent;
};
using tool_agent_info_vec_t       = std::vector<std::unique_ptr<tool_agent_info>>;
using pc_sampling_buffer_id_vec_t = std::vector<rocprofiler_buffer_id_t>;

int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc);
void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record);
void show_results_pc();
#else
int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc)
{
    return 0;
}
void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record)
{
    return ;
}
void show_results_pc()
{
    return;
}
#endif //VERSION OR ENABLED
    
#endif //SAMPLING_SDKPC_H
