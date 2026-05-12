// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include <iostream>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/callback_tracing.h>

#include <unistd.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <vector>

/**
 * Tests the collection of all counters on the agent the test is run on.
 */

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
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }


rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t agents_ver,
                       const void** agents_arr,
                       size_t       num_agents,
                       void*        udata)
{
    if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
}


std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents()
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
      if(agent.type == ROCPROFILER_AGENT_TYPE_GPU) 
      {
        gpu_agents.push_back(agent);
      }
    }
    return gpu_agents;
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    
      //Check if there are any ROCm GPUs available
    std::vector<rocprofiler_agent_v0_t> agents = get_gpu_device_agents();
    if(agents.empty())
    {
        std::cerr << "No ROCm GPUs found" << std::endl;
        return 1;
    }

    const char* previous_gpu = nullptr;

    for(const auto& agent : agents)
	{
        if(previous_gpu != nullptr && std::strcmp(agent.name, previous_gpu) == 0)
        {
            continue;
        }
		//build_profile_for_agent(agent.id);
        std::vector<rocprofiler_counter_id_t> gpu_counters;
        // Iterate all the counters on the agent and store them in gpu_counters.
        ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                       agent.id,
                       [](rocprofiler_agent_id_t,
                          rocprofiler_counter_id_t* counters,
                          size_t                    num_counters,
                          void*                     user_data) {
                           std::vector<rocprofiler_counter_id_t>* vec =
                               static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                           for(size_t i = 0; i < num_counters; i++)
                           {
                               vec->push_back(counters[i]);
                           }
                           return ROCPROFILER_STATUS_SUCCESS;
                       },
                       static_cast<void*>(&gpu_counters)),
                   "Could not fetch supported counters");
        std::cout << "=========================================================" << std::endl;
        std::cout << "Counters available with: " << agent.name << std::endl;
        for(auto& counter : gpu_counters)
        {
            rocprofiler_counter_info_v0_t info;
            ROCPROFILER_CALL( rocprofiler_query_counter_info(
                    counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
                    "Could not query info for counter");
            
            if(info.is_constant)
            {
                std::cout << "Name: " << info.name << " [CONSTANT VALUE] description: " << info.description << std::endl;
            }
            else if(info.is_derived)
            {
                std::cout << "Name: " << info.name << " description: " << info.description << " expression: " << info. expression << std::endl;
            }
            else if(!info.is_derived)
            {
                std::cout << "Name: " << info.name << " description: " << info.description << std::endl;
            }
        }
        previous_gpu = agent.name;
	}
    
    return 0;
}

void
tool_fini(void*)
{

}



extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure_(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "CounterClientSample";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::cout << info.str() << std::endl;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(nullptr)};

    // return pointer to configure data
    return &cfg;
}


int main()
{
    ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure_), "force configuration");
    return 0;
}