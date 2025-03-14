//TauRocProfilerSDK.cpp
// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//This file is a modified version of
// https://github.com/ROCm/rocprofiler-sdk/blob/ccd1e54293768a756fb95c21bff51d95d5f6b20c/tests/pc_sampling/address_translation.cpp
#include "Profile/RocProfilerSDK/TauRocProfilerSDK.h"


//Map to identify kernels and some of their information
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

extern int init_pc_sampling(rocprofiler_context_id_t client_ctx, int enabled_hc);
extern void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record);
extern void show_results_pc();

extern std::string read_hc_record(void* payload, uint32_t kind, kernel_symbol_map_t client_kernels, uint64_t* agentid, double* counter_value, rocprofiler_timestamp_t* c_timestamp);
extern int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer);


//Flag to check if TAU called the flush function
//we want to avoid flushing after TAU has written the profile files
int volatile flushed = 0;

//Disable functions called by TAU if rocm profiling not initialized
int volatile rocprofsdk_initialized = 0;

//Buffer for rocprofiler data
rocprofiler_buffer_id_t       client_buffer    = {};

//Flag to enable/disable profiling
int volatile hc_profiling = 0;
int volatile pc_sampling = 0;

//Buffer to identify names of ROCm calls
using buffer_kind_names_t = std::map<rocprofiler_buffer_tracing_kind_t, const char*>;
using buffer_kind_operation_names_t = std::map<rocprofiler_buffer_tracing_kind_t, 
                                                std::map<rocprofiler_tracing_operation_t, const char*>>;
                                                
struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};
};
buffer_name_info              client_name_info = {};

//Map to identify kernels and some of their information
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
kernel_symbol_map_t           client_kernels   = {};

rocprofiler_context_id_t      client_ctx       = {0};

//Map for ROCTX Start and Stop correlation
std::map<roctx_range_id_t, const char*> roctx_start_stop = {};

//Vector for ROCTX Push and Pop
std::vector<const char*> roctx_push_pop = {};

//Map to idenfity agent_id(GPU) with our own identifier
std::map<uint64_t , int> tau_rocm_agent_id = {};

//List of events, used  to sort events by timestamp
std::list<struct TauSDKEvent> TauRocmSDKList;
std::mutex SDKList_mtx;

std::map<int, rocprofiler_timestamp_t> tau_last_timestamp_published;
std::mutex last_mtx;
//static rocprofiler_timestamp_t tau_last_timestamp_published = 0;

//------------------------------------------------------------------------------------------------
//Check if -rocm is set with env variable TAU_USE_ROCPROFILERSDK
//As TAU has not initialized, needs to read the variable here
int use_rocprofilersdk()
{
   const char* use_rocprofiler =  std::getenv("TAU_USE_ROCPROFILERSDK");
   if( use_rocprofiler )
   {
     if ( atoi(use_rocprofiler) == 1)
     {
       return 1;
     }
   }
   /*
   const char* t_metrics =  std::getenv("TAU_METRICS");
   if( t_metrics )
   {
    if ( std::strstr(t_metrics,"TAUGPU_TIME") !=NULL )
    {
       return 1;
    }
    else
    {
      std:cerr << "TAU_METRICS does not contain TAUGPU_TIME, timers will not be valid" << std::endl;
      return 0;
    }
   }
   else
   {
    std:cerr << "TAU_METRICS does not contain TAUGPU_TIME, timers will not be valid" << std::endl;
    return 0;
   }
   */
   return 0;
}

//When rocprofiler-sdk stops being experimental
//this may be removed or hidden
void rocsdk_version_check(uint32_t                 version,
                      const char*              runtime_version)
{
  // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << "TAU is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl; 
  
}

//------------------------------------------------------------------------------------------------

//https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/source/include/rocprofiler-sdk/fwd.h#L181
//Not all kinds are supported, look at the definitions in new rocprofiler-sdk versions and implement if supported
static const auto supported_kinds = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
    ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
    //ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
    ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
    ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API,
    ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
    ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
    ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API,
    ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API,
    ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
    ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
    //ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION,
    //ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY
};

//This is a bit different from the ROCm samples,
//in this case we use maps, which changes the tool
//a bit when using the callbacks
inline buffer_name_info
get_buffer_tracing_names()
{


    auto cb_name_info = buffer_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_buffer_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t   operation,
                                               void*                             data_v) {
        auto* name_info_v = static_cast<buffer_name_info*>(data_v);

        if(supported_kinds.count(kindv) > 0)
        {
            const char* name = nullptr;
            ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                 kindv, operation, &name, nullptr),
                             "query buffer tracing kind operation name");
            //EXPECT_TRUE(name != nullptr) << "kind=" << kindv << ", operation=" << operation;
            if(name) name_info_v->operation_names[kindv][operation] = name;
        }
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_buffer_tracing_kind_t kind, void* data) {
        //  store the buffer kind name
        auto*       name_info_v = static_cast<buffer_name_info*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_name(kind, &name, nullptr),
                         "query buffer tracing kind operation name");
        //EXPECT_TRUE(name != nullptr) << "kind=" << kind;
        if(name) name_info_v->kind_names[kind] = name;

        if(supported_kinds.count(kind) > 0)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "iterating buffer tracing kind operations");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_buffer_tracing_kinds(tracing_kind_cb,
                                                              static_cast<void*>(&cb_name_info)),
                     "iterating buffer tracing kinds");

    return cb_name_info;
}



/**
 * Returns all GPU agents visible to rocprofiler on the system
 */
std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents()
{
    std::vector<rocprofiler_agent_v0_t> agents;

    // Callback used by rocprofiler_query_available_agents to return
    // agents on the device. This can include CPU agents as well. We
    // select GPU agents only (i.e. type == ROCPROFILER_AGENT_TYPE_GPU)
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // Query the agents, only a single callback is made that contains a vector
    // of all agents.
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");
    return agents;
}


std::string get_copy_direction(int direction, int* queueid, uint64_t source, uint64_t destination)
{
  std::string mem_cpy_kind;

  switch(direction)
  {
    case ROCPROFILER_MEMORY_COPY_NONE:
      mem_cpy_kind = "MEMORY_COPY_NONE";
      break;
    case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:
      mem_cpy_kind = "MEMORY_COPY_HOST_TO_HOST";
      break;
    case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:
    {
      mem_cpy_kind = "MEMORY_COPY_HOST_TO_DEVICE";
      auto agent_id_elem = tau_rocm_agent_id.find(destination);
      if(agent_id_elem == tau_rocm_agent_id.end())
      {
        tau_rocm_agent_id[destination] = tau_rocm_agent_id.size();
      }   
      *queueid = 3 + tau_rocm_agent_id[destination];
      break;
    }
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:
    {
      mem_cpy_kind = "MEMORY_COPY_DEVICE_TO_HOST";
      auto agent_id_elem = tau_rocm_agent_id.find(source);
      if(agent_id_elem == tau_rocm_agent_id.end())
      {
        tau_rocm_agent_id[source] = tau_rocm_agent_id.size();
      }  
      *queueid = 3 + tau_rocm_agent_id[source];
      break;
    }
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE:
    {
      auto agent_id_elem = tau_rocm_agent_id.find(source);
      if(agent_id_elem == tau_rocm_agent_id.end())
      {
        tau_rocm_agent_id[source] = tau_rocm_agent_id.size();
      }
      auto agent_id_elemd = tau_rocm_agent_id.find(destination);
      if(agent_id_elemd == tau_rocm_agent_id.end())
      {
        tau_rocm_agent_id[destination] = tau_rocm_agent_id.size();
      }  
      mem_cpy_kind = "MEMORY_COPY_DEVICE_TO_DEVICE";
      mem_cpy_kind += " destination id ";
      mem_cpy_kind += std::to_string(tau_rocm_agent_id[destination]);
      *queueid = 3 + tau_rocm_agent_id[source];
      break;
    }
    case ROCPROFILER_MEMORY_COPY_LAST:
      mem_cpy_kind = "MEMORY_COPY_LAST";
      break;
    default:
      mem_cpy_kind = "MEMORY_COPY Unknown";
      break;
  }
  return mem_cpy_kind;
}

//Publish event to TAU
void TAU_publish_sdk_event(TauSDKEvent sdk_event)
{
  TAU_VERBOSE("TAU_publish_sdk_event \n");
  last_mtx.lock();
  rocprofiler_timestamp_t last_timestamp;
  
  std::map<int, rocprofiler_timestamp_t>::iterator it = tau_last_timestamp_published.find(sdk_event.taskid);
  if(it == tau_last_timestamp_published.end())
  {
    tau_last_timestamp_published[sdk_event.taskid] = 0;
    last_timestamp = 0;
  }
  else
  {
    last_timestamp = it->second;
  }

  if( sdk_event.entry < last_timestamp )
  {
    TAU_VERBOSE("ERROR: new event's timestamp is older than previous event timestamp, current look ahead window is %d\n", TAU_ROCMSDK_LOOK_AHEAD);
    TAU_VERBOSE("ERROR: modify TAU_ROCMSDK_LOOK_AHEAD with -useropt=-DTAU_ROCMSDK_LOOK_AHEAD=%d or bigger\n", TAU_ROCMSDK_LOOK_AHEAD*2);
    TAU_VERBOSE("ERROR: if this is a hsa_* task [task: %s], some may overlap and this error should be ignored\n", sdk_event.name.c_str());
    //TAU_VERBOSE("- Entry: %u Exit: %u %s task: %d\n", sdk_event.entry, sdk_event.exit, sdk_event.name.c_str(), sdk_event.taskid);
    last_mtx.unlock();
    return;
  }
  //TAU_VERBOSE("Add Entry: %u Exit: %u %s task: %d\n", sdk_event.entry, sdk_event.exit, sdk_event.name.c_str(), sdk_event.taskid);

  tau_last_timestamp_published[sdk_event.taskid] = sdk_event.exit;

  last_mtx.unlock();
  /*
          ss.str("");
          ss << "Private segment size : " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.private_segment_size);
          Tau_userevent_thread(ue, value, taskid);
          
  */
  if(!sdk_event.name.empty())
  {
    double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(sdk_event.taskid, ((double)sdk_event.entry/1e3)); // convert to microseconds
    metric_set_gpu_timestamp(sdk_event.taskid, timestamp_entry);
    TAU_START_TASK(sdk_event.name.c_str(), sdk_event.taskid);


    double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(sdk_event.taskid, ((double)sdk_event.exit/1e3)); // convert to microseconds
    metric_set_gpu_timestamp(sdk_event.taskid, timestamp_exit);
    TAU_STOP_TASK(sdk_event.name.c_str(), sdk_event.taskid);
    Tau_set_last_timestamp_ns(timestamp_exit);
    for(auto& sdk_u_event : sdk_event.userevents)
    {
      void* ue = nullptr;
      Tau_get_context_userevent(&ue, sdk_u_event.ev_name.c_str());
      TAU_CONTEXT_EVENT_THREAD_TS(ue, sdk_u_event.value, sdk_event.taskid, timestamp_exit);
    }    
  }
  else
  {
    for(auto& sdk_u_event : sdk_event.userevents)
    {
      void* ue = nullptr;
      double timestamp = Tau_metric_set_synchronized_gpu_timestamp(sdk_event.taskid, ((double)sdk_event.exit/1e3)); // convert to microseconds
      Tau_get_context_userevent(&ue, sdk_u_event.ev_name.c_str());
      TAU_CONTEXT_EVENT_THREAD_TS(ue, sdk_u_event.value, sdk_event.taskid, timestamp);
    }  
  }
  
  
}

//Process event into the list, add to TAU if needed
void TAU_process_sdk_event(TauSDKEvent sdk_event)
{
  TAU_VERBOSE("TAU_process_sdk_event\n");
  //TauRocmSDKList // TauSDKEvent
  SDKList_mtx.lock();
  TauRocmSDKList.push_back(sdk_event);
  TauRocmSDKList.sort();

  if(TauRocmSDKList.size() < TAU_ROCMSDK_LOOK_AHEAD)
  {
    SDKList_mtx.unlock();
    return;
  }
  else
  {
    TAU_publish_sdk_event(TauRocmSDKList.front());
    TauRocmSDKList.pop_front();
  }
  SDKList_mtx.unlock();
}


//Buffered tracing callback
void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
  //If we have already flushed all the events, tau has ended, do not profile
  if(flushed)
    return;
    
  assert(user_data != nullptr);
  assert(drop_count == 0 && "drop count should be zero for lossless policy");
  
  if(num_headers == 0)
    throw std::runtime_error{
         "rocprofiler invoked a buffer callback with no headers. this should never happen"};
  else if(headers == nullptr)
    throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                             "array of headers. this should never happen"};
  
  static unsigned long long last_timestamp = Tau_get_last_timestamp_ns();
  
  //TAU_VERBOSE("tool_tracing_callback\n");

  static int disable_user_events = TauEnv_get_tracing();

  for(size_t i = 0; i < num_headers; ++i)
  {
    auto* header = headers[i];
    if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
    {
      if(header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
         header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
         header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
         header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API)
      {
        auto* record = static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);
        if(record->start_timestamp > record->end_timestamp)
        {
          auto msg = std::stringstream{};
          msg << "hsa api: start > end (" << record->start_timestamp << " > "
              << record->end_timestamp
              << "). diff = " << (record->start_timestamp - record->end_timestamp);
          std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
          // throw std::runtime_error{msg.str()};
        }

        //Different types of events will appear as different threads in the profile
        //This is to differenciate kernels, API calls and other events
        int queueid = 0;
        unsigned long long timestamp = 0L;
        int taskid = Tau_get_initialized_queues(queueid);
        if (taskid == -1) { // not initialized
          TAU_CREATE_TASK(taskid);
          Tau_set_initialized_queues(queueid, taskid);
          timestamp = record->start_timestamp;
          Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE0", taskid);
          last_timestamp = timestamp;
          // Set the timestamp for TAUGPU_TIME:
          Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                                    ((double)timestamp / 1e3));
          Tau_create_top_level_timer_if_necessary_task(taskid);
          Tau_add_metadata_for_task("HSA_API_ID", taskid, taskid);

        }
        
        std::string task_name;
        task_name = client_name_info.operation_names[record->kind][record->operation];
        std::vector<TauSDKUserEvent> record_events;
        struct TauSDKEvent e(task_name, record->start_timestamp, record->end_timestamp, taskid, record_events);
        //TAU_VERBOSE("taskid: %d start_ts: %lf end_ts: %lf\n", e.taskid, (double)e.entry, (double)e.exit);
        TAU_process_sdk_event(e);
        /*

        TAU_VERBOSE("taskid: %d start_ts: %lf end_ts: %lf\n", taskid, (double)record->start_timestamp/1e3, (double)record->end_timestamp/1e3);

        double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_entry);
        TAU_START_TASK(task_name.c_str(), taskid);

        double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_exit);
        TAU_STOP_TASK(task_name.c_str(), taskid);
        Tau_set_last_timestamp_ns(timestamp_exit);*/
          
      }
      else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
      {
        auto* record = static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);
        if(record->start_timestamp > record->end_timestamp)
        {
          auto msg = std::stringstream{};
          msg << "hip api: start > end (" << record->start_timestamp << " > "
              << record->end_timestamp
              << "). diff = " << (record->start_timestamp - record->end_timestamp);
          std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
          // throw std::runtime_error{msg.str()};
        }

        int queueid = 1;
        unsigned long long timestamp = 0L;
        int taskid = Tau_get_initialized_queues(queueid);
        if (taskid == -1) { // not initialized
          TAU_CREATE_TASK(taskid);
          Tau_set_initialized_queues(queueid, taskid);
          timestamp = record->start_timestamp;
          Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE1", taskid);
          last_timestamp = timestamp;
          // Set the timestamp for TAUGPU_TIME:
          Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                                    ((double)timestamp / 1e3));
          Tau_create_top_level_timer_if_necessary_task(taskid);
          Tau_add_metadata_for_task("HIP_RUNTIME_API_ID", taskid, taskid);

        }
        
        std::string task_name;
        task_name = client_name_info.operation_names[record->kind][record->operation];
        std::vector<TauSDKUserEvent> record_events;
        struct TauSDKEvent e(task_name, record->start_timestamp, record->end_timestamp, taskid, record_events);
        //TAU_VERBOSE("taskid: %d start_ts: %lf end_ts: %lf\n", e.taskid, (double)e.entry, (double)e.exit);
        TAU_process_sdk_event(e);
        /*
        double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_entry);
        TAU_START_TASK(task_name.c_str(), taskid);
        
        double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_exit);
        TAU_STOP_TASK(task_name.c_str(), taskid);
        Tau_set_last_timestamp_ns(timestamp_exit);*/
        
      }
      else if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
      {
        //printf("ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH\n");
        auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
        if(record->start_timestamp > record->end_timestamp)
          throw std::runtime_error("kernel dispatch: start > end");

        //This should be related to the GPU id(agent_id.handle which is uint64_t)
        //int queueid = 1 + (int)record->dispatch_info.agent_id.handle;
        auto agent_id = record->dispatch_info.agent_id.handle;
        auto agent_id_elem = tau_rocm_agent_id.find(agent_id);
        if(agent_id_elem == tau_rocm_agent_id.end())
        {
          tau_rocm_agent_id[agent_id] = tau_rocm_agent_id.size();
        }
   

        int queueid = 3 + tau_rocm_agent_id[agent_id];
        unsigned long long timestamp = 0L;
        int taskid = Tau_get_initialized_queues(queueid);
        if (taskid == -1) { // not initialized
          TAU_CREATE_TASK(taskid);
          Tau_set_initialized_queues(queueid, taskid);
          timestamp = record->start_timestamp;
          Tau_check_timestamps(last_timestamp, timestamp, "NEW GPU QUEUE", taskid);
          last_timestamp = timestamp;
          // Set the timestamp for TAUGPU_TIME:
          Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                                    ((double)timestamp / 1e3));
          Tau_create_top_level_timer_if_necessary_task(taskid);
          Tau_add_metadata_for_task("HIP_RUNTIME_API_ID", taskid, taskid);

        }
        
        std::string task_name;
        task_name = Tau_demangle_name(client_kernels.at(record->dispatch_info.kernel_id).kernel_name);
        std::vector<TauSDKUserEvent> record_events;

        std::string event_name = "Private segment size : ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent pr_seg_si( (double)record->dispatch_info.private_segment_size
                                          , event_name);
        record_events.push_back(pr_seg_si);

        event_name = "Group segment size : ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent gr_seg_si( (double)record->dispatch_info.group_segment_size
                                          , event_name);
        record_events.push_back(gr_seg_si);

        event_name = "Workgroup size X: ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent workx( (double)record->dispatch_info.workgroup_size.x
                                          , event_name);
        record_events.push_back(workx);

        event_name = "Workgroup size Y: ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent worky( (double)record->dispatch_info.workgroup_size.y
                                          , event_name);
        record_events.push_back(worky);
        event_name = "Workgroup size Z: ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent workz( (double)record->dispatch_info.workgroup_size.z
                                          , event_name);
        record_events.push_back(workz);

        event_name = "Grid size X: ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent gridx( (double)record->dispatch_info.grid_size.x
                                          , event_name);
        record_events.push_back(gridx);

        event_name = "Grid size Y: ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent gridy( (double)record->dispatch_info.grid_size.y
                                          , event_name);
        record_events.push_back(gridy);

        event_name = "Grid size Z: ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent gridz( (double)record->dispatch_info.grid_size.z
                                          , event_name);
        record_events.push_back(gridz);

        /*
        if(!disable_user_events)
        {
          std::stringstream ss;
          std::string tmp;
          void* ue = nullptr;
          double value;
          
          ss.str("");
          ss << "Private segment size : " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.private_segment_size);
          Tau_userevent_thread(ue, value, taskid);
          
          ss.str("");
          ss << "Group segment size : " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.group_segment_size);
          Tau_userevent_thread(ue, value, taskid);
                    
          ss.str("");
          ss << "Workgroup size X: " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.workgroup_size.x);
          Tau_userevent_thread(ue, value, taskid);
          
          ss.str("");
          ss << "Workgroup size Y: " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.workgroup_size.y);
          Tau_userevent_thread(ue, value, taskid);
          
          ss.str("");
          ss << "Workgroup size Z: " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.workgroup_size.z);
          Tau_userevent_thread(ue, value, taskid);
          
          ss.str("");
          ss << "Grid size X: " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.grid_size.x);
          Tau_userevent_thread(ue, value, taskid);
          
          ss.str("");
          ss << "Grid size Y: " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.grid_size.y);
          Tau_userevent_thread(ue, value, taskid);
          
          ss.str("");
          ss << "Grid size Z: " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->dispatch_info.grid_size.z);
          Tau_userevent_thread(ue, value, taskid);
        }*/
        
        
        std::string kernel_name = "[ROCm Kernel] ";
        kernel_name += task_name;
        struct TauSDKEvent e(kernel_name, record->start_timestamp, record->end_timestamp, taskid, record_events);
        //TAU_VERBOSE("taskid: %d start_ts: %lf end_ts: %lf\n", e.taskid, (double)e.entry, (double)e.exit);

        TAU_process_sdk_event(e);

        /*
        double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_entry);
        TAU_START_TASK(kernel_name.c_str(), taskid);
        
  
        double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_exit);
        TAU_STOP_TASK(kernel_name.c_str(), taskid);
        Tau_set_last_timestamp_ns(timestamp_exit);*/
        
      }
      else if(header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
      {
        auto* record = static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);
        if(record->start_timestamp > record->end_timestamp)
          throw std::runtime_error("memory copy: start > end");

        //We want to tie, if possible, the copy to a GPU
        //Looking at the type of copy and the destination
        //and source, we may be able to. If not, set it to
        //a queue with no GPUs
        std::string task_name;
        int queueid = 2;
        task_name = get_copy_direction(record->operation, &queueid, record->src_agent_id.handle, record->dst_agent_id.handle);        
                          
        unsigned long long timestamp = 0L;
        int taskid = Tau_get_initialized_queues(queueid);
        if (taskid == -1) { // not initialized
          TAU_CREATE_TASK(taskid);
          Tau_set_initialized_queues(queueid, taskid);
          timestamp = record->start_timestamp;
          Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE3", taskid);
          last_timestamp = timestamp;
          // Set the timestamp for TAUGPU_TIME:
          Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                                    ((double)timestamp / 1e3));
          Tau_create_top_level_timer_if_necessary_task(taskid);
          Tau_add_metadata_for_task("ROCM_MEMORY_ID", taskid, taskid);

        }
        
        std::stringstream ss;
  			std::string tmp;
  			void* ue = nullptr;
  			double value;
        std::vector<TauSDKUserEvent> record_events;

        /*
  			if(!disable_user_events)
        {
          ss.str("");
          ss << "bytes copied : " << task_name;
          tmp = ss.str();
          ue = Tau_get_userevent(tmp.c_str());
          value = (double)(record->bytes);
          Tau_userevent_thread(ue, value, taskid);
        }*/

        std::string event_name = "bytes copied : ";
        event_name = event_name + task_name;
        struct TauSDKUserEvent bcopy( (double)record->bytes
                                          , event_name);
        record_events.push_back(bcopy);

        struct TauSDKEvent e(task_name, record->start_timestamp, record->end_timestamp, taskid, record_events);
        //("taskid: %d start_ts: %lf end_ts: %lf\n", e.taskid, (double)e.entry, (double)e.exit);
        TAU_process_sdk_event(e);

        /*
        double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_entry);
        TAU_START_TASK(task_name.c_str(), taskid);
        
        double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
        metric_set_gpu_timestamp(taskid, timestamp_exit);
        TAU_STOP_TASK(task_name.c_str(), taskid);
        Tau_set_last_timestamp_ns(timestamp_exit); */
      }
      //https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/samples/api_buffered_tracing/client.cpp#L284
      /*else if(header->kind == ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION)
      {
        printf("ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION\n");
        auto* record =
                static_cast<rocprofiler_buffer_tracing_page_migration_record_t*>(header->payload);

        auto info = std::stringstream{};

        info << "kind=" << record->kind << ", operation=" << record->operation
             << ", pid=" << record->pid << ", start=" << record->start_timestamp
             << ", stop=" << record->end_timestamp
             << ", name=" << client_name_info.operation_names[record->kind][record->operation];

        switch(record->operation)
        {
            case ROCPROFILER_PAGE_MIGRATION_PAGE_MIGRATE:
            {
                info << ", page_fault=(" << record->page_fault.read_fault << ", "
                     << record->page_fault.migrated << ", " << record->page_fault.node_id
                     << ", " << std::hex << "0x" << record->page_fault.address << ")";
                break;
            }
            case ROCPROFILER_PAGE_MIGRATION_PAGE_FAULT:
            {
                info << ", page_migrate=(" << std::hex << "0x"
                     << record->page_migrate.start_addr << ", 0x"
                     << record->page_migrate.end_addr << ", " << std::dec
                     << record->page_migrate.from_node << ", " << record->page_migrate.to_node
                     << ", " << record->page_migrate.prefetch_node << ", "
                     << record->page_migrate.preferred_node << ", "
                     << record->page_migrate.trigger << ")";
                break;
            }
            case ROCPROFILER_PAGE_MIGRATION_QUEUE_SUSPEND:
            {
                info << ", queue_suspend=(" << record->queue_suspend.rescheduled << ", "
                     << record->queue_suspend.node_id << ", " << record->queue_suspend.trigger
                     << ")";
                break;
            }
            case ROCPROFILER_PAGE_MIGRATION_UNMAP_FROM_GPU:
            {
                info << ", unmap_from_gpu=(" << record->unmap_from_gpu.node_id << std::hex
                     << ", 0x" << record->unmap_from_gpu.start_addr << ", 0x"
                     << record->unmap_from_gpu.end_addr << ", " << std::dec
                     << record->unmap_from_gpu.trigger << ")";
                break;
            }
            case ROCPROFILER_PAGE_MIGRATION_NONE:
            case ROCPROFILER_PAGE_MIGRATION_LAST:
            {
                throw std::runtime_error{"unexpected page migration value"};
                break;
            }
        }

        if(record->start_timestamp > record->end_timestamp)
            throw std::runtime_error("page migration: start > end");
            
        std::cout << info.str() << std::endl;

      }*/
      /*else if(header->kind == ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY)
      {
        printf("ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY\n");
        auto* record =
                static_cast<rocprofiler_buffer_tracing_scratch_memory_record_t*>(header->payload);

        auto info = std::stringstream{};

        auto _elapsed =
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
                std::chrono::nanoseconds{record->end_timestamp - record->start_timestamp})
                .count();

        info << "tid=" << record->thread_id << ", context=" << context.handle
             << ", buffer_id=" << buffer_id.handle
             << ", cid=" << record->correlation_id.internal
             << ", extern_cid=" << record->correlation_id.external.value
             << ", kind=" << record->kind << ", operation=" << record->operation
             << ", agent_id=" << record->agent_id.handle
             << ", queue_id=" << record->queue_id.handle << ", thread_id=" << record->thread_id
             << ", elapsed=" << std::setprecision(3) << std::fixed << _elapsed
             << " usec, flags=" << record->flags
             << ", name=" << client_name_info.operation_names[record->kind][record->operation];
        std::cout << info.str() << std::endl;
      }*/
      //Need to look into them      
      /*else if(header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API)
      {
        printf("ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API\n");
      }
      else if(header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API)
      {
        printf("ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API\n");
      }
      else if(header->kind == ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API)
      {
        printf("ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API\n");
      }*/
      //ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API //what does this do?
      
    }
    else
    {
      /*if(header->category == ROCPROFILER_BUFFER_CATEGORY_NONE)
        printf("ROCPROFILER_BUFFER_CATEGORY_NONE events should not be obtained in tool_tracing_callback\n");
      if(header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        printf("ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING events should not be obtained in tool_tracing_callback\n");*/
      //Hardware Counter Profiling
      if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS )
      {
        //printf("ROCPROFILER_BUFFER_CATEGORY_COUNTERS\n");
        uint64_t agent_id = 0;
        double counter_value;
        rocprofiler_timestamp_t c_timestamp;
        std::string msg = read_hc_record(header->payload, header->kind, client_kernels, &agent_id, &counter_value, &c_timestamp);
        if(agent_id == 0)
          continue;        
  			//This should be related to the GPU id(agent_id.handle which is uint64_t)
        //int queueid = 1 + (int)record->dispatch_info.agent_id.handle;
        auto agent_id_elem = tau_rocm_agent_id.find(agent_id);
        if(agent_id_elem == tau_rocm_agent_id.end())
        {
          tau_rocm_agent_id[agent_id] = tau_rocm_agent_id.size();
        }        
        int queueid = 3 + tau_rocm_agent_id[agent_id];
        int taskid = Tau_get_initialized_queues(queueid);
        //void* ue = nullptr;
  			//ue = Tau_get_userevent(msg.c_str());
  			double value = (double)(counter_value);
  			//Tau_userevent_thread(ue, value, taskid);
        

        
        std::vector<TauSDKUserEvent> record_events;
        struct TauSDKUserEvent bcopy( value, msg);
        record_events.push_back(bcopy);

        std::string task_name;
        struct TauSDKEvent e(task_name, c_timestamp, c_timestamp, taskid, record_events);
        //("taskid: %d start_ts: %lf end_ts: %lf\n", e.taskid, (double)e.entry, (double)e.exit);
        TAU_process_sdk_event(e);


      }
      /*if(header->category ==ROCPROFILER_BUFFER_CATEGORY_LAST)
        printf("ROCPROFILER_BUFFER_CATEGORY_LAST events should not be obtained in tool_tracing_callback\n");*/
    }
  }
  //TAU_VERBOSE("tool_tracing_callback-end\n");
}


//https://github.com/ROCm/rocprofiler-sdk/blob/ad48201912995e1db4f6e65266bce2792056b3c6/source/lib/rocprofiler-sdk-tool/tool.cpp#L410C1-L559C2
//Callback for ROCTX marker events Push, Pop, Start and Stop
void
tool_roctx_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 data)
{
  if(flushed)
    return;
  static thread_local auto stacked_range =
      std::vector<rocprofiler_buffer_tracing_marker_api_record_t>{};

  if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API)
  {
    auto* marker_data =
        static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
    
    //In the case of Push and Pop, the relation between both does not exist
    //use the order in which both are called to relate them to each other
    if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA)
    {
      if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
      {
        if(marker_data->args.roctxRangePushA.message)
        {
          //std::cout << "TAU! roctxRangePush message: " << marker_data->args.roctxRangePushA.message << std::endl;
          roctx_push_pop.emplace_back(marker_data->args.roctxRangePushA.message);
          std::string event_name = "[roctx] ";
          event_name += marker_data->args.roctxRangePushA.message;
          TAU_START(event_name.c_str());
        }
      }
    }
    else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
          if(roctx_push_pop.empty())
          {
            std::cerr << "roctxRangePop was invoked more times than roctxRangePush" << std::endl;
            return;
          }
          auto push_name = roctx_push_pop.back();
          roctx_push_pop.pop_back();
          //std::cout << "TAU! roctxRangePop message:" << push_name << std::endl;
          std::string event_name = "[roctx] ";
          event_name += push_name;
          TAU_STOP(event_name.c_str());
        }
    }
    //Start and Stop can be identified comparing ids, in the case of Start 
    //marker_data->retval.roctx_range_id_t_retval
    //in the case of Stop , the matching id is found in
    //marker_data->args.roctxRangeStop.id
    else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
           marker_data->args.roctxRangeStartA.message)
        {
          roctx_start_stop[marker_data->retval.roctx_range_id_t_retval] = marker_data->args.roctxRangeStartA.message;
          std::string event_name = "[roctx] ";
          event_name += marker_data->args.roctxRangeStartA.message;
          TAU_START(event_name.c_str());
        }
    }
    else if(record.operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop)
    {
      if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
      {
        auto req_id = marker_data->args.roctxRangeStop.id;
        auto start_name = roctx_start_stop.find(req_id);
        if(start_name == roctx_start_stop.end())
        {
          std::cerr << "Failed to find RangeStart with requested id\n" << std::endl;
          return;
        }
        std::string event_name = "[roctx] ";
        event_name += start_name->second;
        TAU_STOP(event_name.c_str());
        roctx_start_stop.erase(req_id);
      }
    }
  }
}



//Callback used to register kernels and obtain their information for buffered tracing
//also flushes tracing information
void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
  if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
     record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
  {
    //printf("ROCPROFILER_CODE_OBJECT_LOAD\n");
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
    {
      // flush the buffer to ensure that any lookups for the client kernel names for the code
      // object are completed
      auto flush_status = rocprofiler_flush_buffer(client_buffer);
      if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
        ROCPROFILER_CALL(flush_status, "buffer flush");
    }
    //Only execute if PC Sampling enabled
    if(pc_sampling == 1)
    {
      //printf("codeobj_tracing_callback\n");
      codeobj_tracing_callback(record);
    }
    else
    {

      //printf("No codeobj_tracing_callback\n");
    }
  }
  else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
          record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
  {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
      client_kernels.emplace(data->kernel_id, *data);
    }
    /*else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
    {
      client_kernels.erase(data->kernel_id);
    }*/
  }
}



//Initialization of rocprofiler-sdk
int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
  //Check that tool_init was called correctly
  assert(tool_data != nullptr);
  
  //Check if there are any ROCm GPUs available
  std::vector<rocprofiler_agent_v0_t> agents = get_gpu_device_agents();
	if(agents.empty())
  {
    std::cerr << "No ROCm GPUs found" << std::endl;
    rocprofsdk_initialized = 0;
    return 1;
  }
  
  client_name_info = get_buffer_tracing_names();
  
  //Configure service to obtain callback names
  ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");
  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};
  ROCPROFILER_CALL(
                    rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_code_object_callback,
                                                       nullptr),
                                                       "code object tracing service configure");                                                   
                                                       
  //Configure service to obtain ROCTX information
  ROCPROFILER_CALL(
                    rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
                                                       nullptr,
                                                       0,
                                                       tool_roctx_callback,
                                                       nullptr),
                                                       "roctx marker tracing service configure");  
  
                                                       
  //Create buffer for buffered tracing
  constexpr auto buffer_size_bytes      = 4096;
  constexpr auto buffer_watermark_bytes = buffer_size_bytes - (buffer_size_bytes / 8);
  ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               buffer_size_bytes,
                                               buffer_watermark_bytes,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                                               "buffer creation");
  
   //Configure rocprofiler-sdk to trace the services in supported_kinds
   for(const auto& kind_id : supported_kinds)
  {
    std::string msg = "configuring buffer tracing for kind id: "+std::to_string(kind_id);

    ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                     client_ctx, kind_id, nullptr, 0, client_buffer),
                     msg.c_str());
  }
  
  //Buffered tracing (tool_tracing_callback) uses its own thread
  //tool_code_object_callback uses the original thread
  auto client_thread = rocprofiler_callback_thread_t{};
  ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                   "creating callback thread");

  ROCPROFILER_CALL(rocprofiler_assign_callback_thread(client_buffer, client_thread),
                   "assignment of thread for buffer");

  int valid_ctx = 0;
  ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                   "context validity check");
  if(valid_ctx == 0)
  {
      // notify rocprofiler that initialization failed
      // and all the contexts, buffers, etc. created
      // should be ignored
      std::cerr<< "ROCm context not valid" << std::endl;
      return -1;
  }
  
  hc_profiling = init_hc_profiling(agents, client_ctx, client_buffer);
  if(hc_profiling == PROFILE_METRICS)
    pc_sampling = init_pc_sampling(client_ctx, 1);
  else
    pc_sampling = init_pc_sampling(client_ctx, 0);
  
  if( (hc_profiling == PROFILE_METRICS) && pc_sampling)
  {
    std::cerr << "[TAU] rocprofiler-sdk is unable to profile hardware counter and perform pc sampling at the same time \n Select only one" << std::endl;
    pc_sampling = 0;
    hc_profiling = NO_METRICS;
    return -1;
  }

  ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start");
  rocprofsdk_initialized = 1;
  return 0;
}


//End of rocprofiler-sdk
void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);
    if(pc_sampling)
      show_results_pc();

}


//Configure rocprofiler-sdk when executing the application
//executed before TAU starts
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
  if(use_rocprofilersdk() == 0)
    return nullptr;

  rocsdk_version_check(version, runtime_version);
  
  char* client_tool_data = "";
  // create configure data
  static auto cfg =
      rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                          &tool_init,
                                          &tool_fini,
                                          static_cast<void*>(client_tool_data)};
  
  // return pointer to configure data
  return &cfg;
}


//Flush ROCm buffer/s before TAU ends
void Tau_rocprofsdk_flush(){
  if(rocprofsdk_initialized==0)
  {
    TAU_VERBOSE("Flag -rocm not set, rocm is not profiled\n");
    return;
  }
  if(flushed==1)
    return;
  TAU_VERBOSE("Tau_rocprofsdk_flush\n");
  ROCPROFILER_CALL(rocprofiler_flush_buffer(client_buffer), "buffer flush");
  
  
  TauRocmSDKList.sort();
  while(!TauRocmSDKList.empty())
  {
    TAU_publish_sdk_event(TauRocmSDKList.front());
    TauRocmSDKList.pop_front();
  }
  flushed = 1;
  ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop");
}
