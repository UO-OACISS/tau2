//TauRocProfilerSDK_pc.h

#ifndef SAMPLING_SDKPC_H
#define SAMPLING_SDKPC_H

#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/rocprofiler.h>

//Due to some bugs, PC Sampling is available, but does not work in older versions
// do not compile for 4.0 and older
//Also, the implementation is not fully done, in future releases, we may
// be able to get stall reasons
#ifdef TAU_ENABLE_ROCPROFILERSDK_PC
    #if (ROCPROFILER_VERSION_MINOR > 4) && (ROCPROFILER_VERSION_MAJOR == 0)
        #define SAMPLING_SDKPC
    #elif (ROCPROFILER_VERSION_MAJOR >= 1)
        #define SAMPLING_SDKPC
        
    #else
        #warning "This rocprofiler-sdk version is unable to use PC Sampling"
    #endif
#endif

#ifdef SAMPLING_SDKPC
#include "Profile/RocProfilerSDK/TauRocProfilerSDK_add_tr.hpp"
#include <Profile/TauBfd.h>  // for name demangling
#include "Profile/Profiler.h"


#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/version.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>

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
#endif // ROCPROFILER_CALL

  


constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);
constexpr int    MAX_FAILURES      = 10;

using avail_configs_vec_t         = std::vector<rocprofiler_pc_sampling_configuration_t>;

struct tool_agent_info
{
    rocprofiler_agent_id_t               agent_id;
    std::unique_ptr<avail_configs_vec_t> avail_configs;
    const rocprofiler_agent_t*           agent;
};
using tool_agent_info_vec_t       = std::vector<std::unique_ptr<tool_agent_info>>;
using pc_sampling_buffer_id_vec_t = std::vector<rocprofiler_buffer_id_t>;



struct rocsdk_instruction
{
    rocsdk_instruction() = default;
    rocsdk_instruction(std::string _inst, std::string _kernel_name, std::string _comment, uint64_t _ld_addr)
    : inst(_inst)
    , kernel_name(_kernel_name)
    , comment(_comment)
    , ld_addr(_ld_addr)
    {}
    std::string inst{};
    std::string kernel_name{};
    std::string comment{};
    uint64_t    ld_addr{0};     // Instruction load address, if from loaded codeobj
};

struct TauSDKSampleEvent {

    rocprofiler_timestamp_t entry;
    rocprofiler_timestamp_t exit;
    std::string name;
    int taskid;

    TauSDKSampleEvent(): taskid(0) {}
    TauSDKSampleEvent(string event_name, rocprofiler_timestamp_t begin, rocprofiler_timestamp_t end, int t) : name(event_name), taskid(t)
    {
        entry = begin;
        exit  = end;
    }
    void printEvent() {
        std::cout <<name<<" Task: "<<taskid<<", \t\tEntry: "<<entry<<" , Exit = "<<exit;
    }
    bool appearsBefore(struct TauSDKSampleEvent other_event) {
        if ((taskid == other_event.taskid) &&
            (entry < other_event.entry) &&
            (exit < other_event.entry))  {
            // both entry and exit of my event is before the entry of the other event.
            return true;
        } else
            return false;
    }

    bool operator < (struct TauSDKSampleEvent two) {
        if (entry < two.entry) 
            return true;
        else 
            return false;
    }
  
};

extern int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc);
extern void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record);
extern void sdk_pc_sampling_flush();
#else
extern int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc)
{
    #ifdef TAU_ENABLE_ROCPROFILERSDK_PC
        printf("[TAU] PC Sampling not available for this rocprofiler-sdk version.\n");
    #else
        printf("[TAU] PC Sampling not available for rocprofiler-sdk [compile with -elfutils=download]\n");
    #endif
    return 0;
}
void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record)
{
    return ;
}

void sdk_pc_sampling_flush()
{
    return ;
}

#endif //VERSION OR ENABLED
    
#endif //SAMPLING_SDKPC_H
