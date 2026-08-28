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

//

//This file is a modified version of
// https://github.com/ROCm/rocprofiler-sdk/blob/ccd1e54293768a756fb95c21bff51d95d5f6b20c/tests/pc_sampling/address_translation.cpp
#include "Profile/RocProfilerSDK/TauRocProfilerSDK.h"
#include "Profile/TauMetrics.h"
#include <sys/syscall.h>



//Initialization flag, only to check if Tau_rocm_initialize_v3 was called
static int initialized_v3 = 0;
//Configuration flag, checks if it was configured or failed to configure
static int configured_v3 = 0;

static int pc_sampling = 0;
static int hc_profiling = 0;

//Initialization mutex
std::mutex SDK_init_lock;
//Flushing mutex
std::mutex SDKFlush_mtx;

//Flag to check if TAU called the flush function
//we want to avoid flushing after TAU has written the profile files
static int flushed = 0;

//To identify buffer names of ROCm calls
using buffer_kind_names_t = std::map<rocprofiler_buffer_tracing_kind_t, const char*>;
using buffer_kind_operation_names_t = std::map<rocprofiler_buffer_tracing_kind_t, 
                                                std::map<rocprofiler_tracing_operation_t, const char*>>;
                                                
struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};
};
buffer_name_info              b_client_name_info = {};

//To identify callback names of ROCm calls
using callback_kind_names_t = std::map<rocprofiler_callback_tracing_kind_t, const char*>;
using callback_kind_operation_names_t = std::map<rocprofiler_callback_tracing_kind_t, 
                                                std::map<rocprofiler_tracing_operation_t, const char*>>;
                                                
struct callback_name_info
{
    callback_kind_names_t           kind_names      = {};
    callback_kind_operation_names_t operation_names = {};
};
callback_name_info              c_client_name_info = {};




//Buffer for rocprofiler data
rocprofiler_buffer_id_t       client_buffer    = {};
extern void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record);
extern int init_hc_profiling(std::vector<rocprofiler_agent_v0_t> agents, rocprofiler_context_id_t client_ctx, rocprofiler_buffer_id_t client_buffer);


//Map to identify the queue id given a stream id and device id
static std::map<std::pair<uint64_t, uint64_t>, uint64_t> streamid_queueid_map;

//Use a mutex to avoid accessing the map by more than one thread
std::mutex stream_queue_mtx;


//------------------------------------------------------------------------------------------------
//Check if -rocm is set with env variable TAU_USE_ROCPROFILERSDK
//As TAU has not initialized, needs to read the variable here
int use_rocprofilersdk()
{
  //int check_metric = TauMetrics_getMetricIndexFromName("TAUGPU_TIME");
  int check_enable = TauEnv_get_rocsdk_enable();

  //if((check_enable == 1) && (check_metric == -1))
  //{
  //  std::cout << "[TAU] TAUGPU_TIME was not set, to avoid incorrect timers, rocprofiler-sdk is disabled" << std::endl;
  //  return 0;
  //}
  return check_enable;
}

//Check rocsdk version, useful to debug when using non-system SDK
void rocsdk_version_check(uint32_t                 version,
                      const char*              runtime_version)
{
  // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;
    TAU_VERBOSE("TAU is using rocprofiler-sdk v%u.%u.%u (%s)\n", major, minor, patch, runtime_version); 
}


//Callbacks supported by the general callback, if disabled, may need individual callback,
// or still not supported
static const auto c_supported_kinds = std::unordered_set<rocprofiler_callback_tracing_kind_t>{
    //ROCPROFILER_CALLBACK_TRACING_NONE = 0,
    ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,       ///< @see ::rocprofiler_hsa_core_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,    ///< @see ::rocprofiler_hsa_amd_ext_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,  ///< @see ::rocprofiler_hsa_image_ext_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,  ///< @see ::rocprofiler_hsa_finalize_ext_api_id_t
    ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,       ///< @see ::rocprofiler_hip_runtime_api_id_t
    #if( ROCPROFILER_VERSION_MAJOR > 0)
    ROCPROFILER_CALLBACK_TRACING_HIP_STREAM,  ///< @see ::rocprofiler_hip_stream_operation_t
    #endif // #if( ROCPROFILER_VERSION_MAJOR > 0)
    ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,    ///< Callbacks for kernel dispatches
    ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,     ///< @see ::rocprofiler_marker_core_api_id_t

    ROCPROFILER_CALLBACK_TRACING_RCCL_API,           ///< RCCL tracing
    
    //ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY,  ///< @see ::rocprofiler_scratch_memory_operation_t
    //ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API,     ///< @see ::rocprofiler_marker_name_api_id_t
    //ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API,  ///< @see
    //ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,    ///< @see ::rocprofiler_hip_compiler_api_id_t
    //ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,        ///< @see ::rocprofiler_memory_copy_operation_t
    //ROCPROFILER_CALLBACK_TRACING_OMPT,               ///< @see ::rocprofiler_ompt_operation_t
    //ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,  ///< @see ::rocprofiler_memory_allocation_operation_t
    //ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION,  ///< Callback notifying that a runtime
                                                          ///< library has been initialized
    //ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API,           ///< rocDecode API Tracing
    //ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API,             ///< rocJPEG API Tracing

    //ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,  ///< @see ::rocprofiler_marker_control_api_id_t                                                     ///< ::rocprofiler_marker_core_range_api_id_t
    //ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,     ///< @see ::rocprofiler_hip_graph_operation_t
    //ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API,  ///< rocSHMEM API tracing
    //ROCPROFILER_CALLBACK_TRACING_HIPFILE_API,   ///< hipFILE API Tracing
    //ROCPROFILER_CALLBACK_TRACING_LAST,
    //Has its own callback
    //ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,     ///< @see ::rocprofiler_code_object_operation_t
};


rocprofiler_context_id_t      client_ctx       = {0};

inline callback_name_info
get_callback_tracing_names()
{
    auto cb_name_info = callback_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb = [](rocprofiler_callback_tracing_kind_t kindv,
                                               rocprofiler_tracing_operation_t     operation,
                                               void*                               data_v) {
        auto* name_info_v = static_cast<callback_name_info*>(data_v);

        const char* name   = nullptr;
        auto        status = rocprofiler_query_callback_tracing_kind_operation_name(
            kindv, operation, &name, nullptr);
        if(status == ROCPROFILER_STATUS_SUCCESS && name) name_info_v->operation_names[kindv][operation] = name;
        return 0;
    };

    //
    //  callback for each buffer kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<callback_name_info*>(data);
        const char* name        = nullptr;
        auto        status = rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr);
        if(status == ROCPROFILER_STATUS_SUCCESS && name) name_info_v->kind_names[kind] = name;

        rocprofiler_iterate_callback_tracing_kind_operations(kind, tracing_kind_operation_cb, data);
        return 0;
    };

    rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb, &cb_name_info);

    return cb_name_info;
}

std::vector<uint64_t>*
get_stream_stack()
{
    static thread_local std::vector<uint64_t> stream_stack;
    return &stream_stack;
}

bool stream_stack_not_null()
{
    return get_stream_stack() != nullptr;
}

void push_stream_id(uint64_t id)
{
    get_stream_stack()->emplace_back(id);
}

void pop_stream_id()
{
    return get_stream_stack()->pop_back();
}

uint64_t get_stream_id()
{
    auto* stack = get_stream_stack();

    return stack->empty() ? 0 : stack->back();
}

bool stream_stack_empty()
{
    return get_stream_stack()->empty();
}

void tau_rocsdk_kernel_dispatch(rocprofiler_callback_tracing_record_t record)
{
  auto* kernel_dispatch = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
  std::string task_name = demangle_kernel_rocprofsdk(
                            client_kernels.at(kernel_dispatch->dispatch_info.kernel_id).kernel_name, 1);


  #ifdef ROCSDK_DEBUG
  std::cout << c_client_name_info.kind_names[record.kind] << " "
          << c_client_name_info.operation_names[record.kind][record.operation]
          << " phase: " << record.phase
          << " tid: " << record.thread_id
          << " cid_i: " << record.correlation_id.internal
          << " cid_e: " << record.correlation_id.external.value
          << " kernel_id: " << kernel_dispatch->dispatch_info.kernel_id
          << " kernel_name: " << task_name
          << " start: " << kernel_dispatch->start_timestamp 
          << " end: " << kernel_dispatch->end_timestamp
          << " agent: " << kernel_dispatch->dispatch_info.agent_id.handle
          << " d_id: " << kernel_dispatch->dispatch_info.dispatch_id
          << " grid_size x: " << kernel_dispatch->dispatch_info.grid_size.x
          << " grid_size y: " << kernel_dispatch->dispatch_info.grid_size.y
          << " grid_size z: " << kernel_dispatch->dispatch_info.grid_size.z
          << " group_segment_size: " << kernel_dispatch->dispatch_info.group_segment_size
          << " private_segment_size: " << kernel_dispatch->dispatch_info.private_segment_size
          << " queue_id: " << kernel_dispatch->dispatch_info.queue_id.handle
          << " workgroup_size x: " << kernel_dispatch->dispatch_info.workgroup_size.x
          << " workgroup_size y: " << kernel_dispatch->dispatch_info.workgroup_size.y
          << " workgroup_size z: " << kernel_dispatch->dispatch_info.workgroup_size.z
          //<< " kernel: " << kernel_dispatch.dispatch_info.queue_id
          //<< " stream: " << kernel_dispatch.dispatch_info.queue_id
          << std::endl;
  #endif //ROCSDK_DEBUG

  uint64_t cur_stream = record.correlation_id.external.value;
  uint64_t cur_agent = kernel_dispatch->dispatch_info.agent_id.handle;
  double start_ts = ((double)kernel_dispatch->start_timestamp)/1e3;
  double end_ts = ((double)kernel_dispatch->end_timestamp)/1e3;
  //printf("start_ts %lf end_ts %lf total %lf\n", start_ts, end_ts, end_ts-start_ts);

  //We lock the access to the stream task map, and also locks the task START/STOP
  stream_queue_mtx.lock();

  //We want to obtain the taskid as we can't use TAU_START/STOP
  int taskid;
  auto it = streamid_queueid_map.find({cur_stream, cur_agent});
  if (it != streamid_queueid_map.end()) {
    taskid = it->second;
  }
  else
  {
    
    TAU_CREATE_TASK(taskid);
    streamid_queueid_map[{cur_stream, cur_agent}] = taskid;
    Tau_rocprofsdk_synchronized_gpu_timestamp(taskid, start_ts);
    //printf("CREATE %lf\n", Tau_rocprofsdk_synchronized_gpu_timestamp(taskid, start_ts));
    Tau_add_metadata_for_task("TAU_TASK_ID", taskid, taskid);
    Tau_add_metadata_for_task("ROCM_GPU_ID", TAU_rocsdk_id_device_map[cur_agent], taskid);
    Tau_add_metadata_for_task("ROCM_THREAD_ID", record.thread_id, taskid);
    Tau_add_metadata_for_task("ROCM_STREAM_ID", cur_stream, taskid);
    Tau_create_top_level_timer_if_necessary_task(taskid);
  }
  

  Tau_rocprofsdk_synchronized_gpu_timestamp(taskid, start_ts);
  //printf("KERNEL_s taskid %d %lf\n", taskid, Tau_rocprofsdk_synchronized_gpu_timestamp(taskid, start_ts));
  TAU_START_TASK( task_name.c_str(), taskid);

  double end_sync_ts = Tau_rocprofsdk_synchronized_gpu_timestamp(taskid, end_ts);
  //printf("KERNEL_e taskid %d %lf\n", taskid, Tau_rocprofsdk_synchronized_gpu_timestamp(taskid, end_ts));
  TAU_STOP_TASK( task_name.c_str(), taskid);

  dispatch_kernel_time[kernel_dispatch->dispatch_info.dispatch_id] = {taskid, end_sync_ts};

  void* ue = nullptr;
  std::string event_name;

  event_name = "Private segment size : " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.private_segment_size, taskid, end_sync_ts);

  event_name = "Group segment size : " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.group_segment_size, taskid, end_sync_ts);

  event_name = "Workgroup size X: " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.workgroup_size.x, taskid, end_sync_ts);

  event_name = "Workgroup size Y: " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.workgroup_size.y, taskid, end_sync_ts);

  event_name = "Workgroup size Z: " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.workgroup_size.z, taskid, end_sync_ts);

  event_name = "Grid size X: " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.grid_size.x, taskid, end_sync_ts);

  event_name = "Grid size Y: " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.grid_size.y, taskid, end_sync_ts);

  event_name = "Grid size Z: " + task_name;
  Tau_get_context_userevent(&ue, event_name.c_str());
  TAU_CONTEXT_EVENT_THREAD_TS(ue, kernel_dispatch->dispatch_info.grid_size.z, taskid, end_sync_ts);

  stream_queue_mtx.unlock();
    
}

//To use with rocprofiler_iterate_callback_tracing_kind_operation_args
//Returns a string with all the information, only use for debug
auto info_data_cb_string( rocprofiler_callback_tracing_kind_t,
                            rocprofiler_tracing_operation_t,
                            uint32_t          arg_num,
                            const void* const arg_value_addr,
                            int32_t           indirection_count,
                            const char*       arg_type,
                            const char*       arg_name,
                            const char*       arg_value_str,
                            int32_t           dereference_count,
                            void*             cb_data)
{
  auto& dss = *static_cast<std::stringstream*>(cb_data);
  dss << ((arg_num == 0) ? "(" : ", ");
  dss << arg_num << ": " << arg_name << "=" << arg_value_str;
  return 0;

}

//To use with rocprofiler_iterate_callback_tracing_kind_operation_args
//Returns a vector, where each position is a std::pair of arg_name and value, 
// useful if we want to extract values from hipMemcpy* or other functions
auto info_data_cb_vec( rocprofiler_callback_tracing_kind_t,
                            rocprofiler_tracing_operation_t,
                            uint32_t          arg_num,
                            const void* const arg_value_addr,
                            int32_t           indirection_count,
                            const char*       arg_type,
                            const char*       arg_name,
                            const char*       arg_value_str,
                            int32_t           dereference_count,
                            void*             cb_data)
{
  
  
  auto& dss = *static_cast<std::vector<std::pair<std::string, std::string>>*>(cb_data);
  dss.emplace_back(arg_name, arg_value_str);
  (void) arg_value_addr;
  (void) arg_type;
  (void) indirection_count;
  (void) dereference_count;
  return 0;
  
}

void tau_roctx_process( rocprofiler_callback_tracing_marker_api_data_t* marker_data,  
                        rocprofiler_tracing_operation_t operation , 
                        rocprofiler_callback_phase_t phase )
{
  static thread_local std::vector<const char*> roctx_push_pop = {};
  static std::map<roctx_range_id_t, const char*>  roctx_start_stop = {};

  //Correlation ID is only valid to find the start and end of the same type of call,
  // cannot be used to detect the push/pop pairs
  // as an alternative, we use a stack for push/pop as pop has no parameters 
  // and a map for start/stop as they use an id
  if(operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA)
  {
    if(phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
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
  else if (operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop)
  {
    if(phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
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
  //START AND STOP CAN BE CALLED BY DIFFERENT THREADS
  //Use TAU_START/STOP at this moment, will need to be modified
  else if (operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA)
  {
    if(phase == ROCPROFILER_CALLBACK_PHASE_EXIT && marker_data->args.roctxRangeStartA.message)
    {
      roctx_start_stop[marker_data->retval.roctx_range_id_t_retval] = marker_data->args.roctxRangeStartA.message;
      std::string event_name = "[roctx] ";
      event_name += marker_data->args.roctxRangeStartA.message;
      TAU_START(event_name.c_str());
    }
  }
  else if (operation == ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop)
  {
    if(phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
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
  else if (operation == ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA)
  {
    if(phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
      std::string mark_str = "roctxMark ";
      if(marker_data->args.roctxMarkA.message)
      {
        mark_str += marker_data->args.roctxMarkA.message;
      }
      void* ue = nullptr;
      Tau_get_context_userevent(&ue, mark_str.c_str());
      TAU_CONTEXT_EVENT(ue, 1);
    }
  }

}

int
print_args(rocprofiler_callback_tracing_kind_t domain_idx,
           rocprofiler_tracing_operation_t     op_idx,
           uint32_t                            arg_num,
           const void* const                   arg_value_addr,
           int32_t                             arg_indirection_count,
           const char*                         arg_type,
           const char*                         arg_name,
           const char*                         arg_value_str,
           int32_t                             arg_dereference_count,
           void*                               data)
{

    if(arg_num == 0)
    {
        const char* _kind      = nullptr;
        const char* _operation = nullptr;

        rocprofiler_query_callback_tracing_kind_name(domain_idx, &_kind, nullptr);
        rocprofiler_query_callback_tracing_kind_operation_name(
            domain_idx, op_idx, &_operation, nullptr);

        printf("\n!! [%s] %s\n", _kind, _operation);
    }

    printf("!!   %u: %-16s = %s\n", arg_num, arg_name, arg_value_str);

    // unused in example
    (void) arg_value_addr;
    (void) arg_indirection_count;
    (void) arg_dereference_count;
    (void) data;

    return 0;
}

void tau_rccl_process(rocprofiler_callback_tracing_record_t record)
{

  static std::unordered_map<ncclComm_t, int> comm_nranks;

  if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
  {
    #if ROCSDK_DEBUG
    std::cout << "ENTER "
          << c_client_name_info.kind_names[record.kind] << " "
          << c_client_name_info.operation_names[record.kind][record.operation]
          << " tid: " << record.thread_id
          << " cid: " << record.correlation_id.internal
          << std::endl;
    #endif //ROCSDK_DEBUG
    TAU_START(c_client_name_info.operation_names[record.kind][record.operation]);
    if(record.operation == ROCPROFILER_RCCL_API_ID_ncclCommInitRank)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      comm_nranks[*rccl_data->args.ncclCommInitRank.newcomm] = rccl_data->args.ncclCommInitRank.nranks;
    }
  }
  else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
  {
    std::string rccl_call_name = c_client_name_info.operation_names[record.kind][record.operation];
    #if ROCSDK_DEBUG
    std::cout << "EXIT "
          << c_client_name_info.kind_names[record.kind] << " "
          << rccl_call_name
          << " tid: " << record.thread_id
          << " cid: " << record.correlation_id.internal
          << std::endl;
    #endif //ROCSDK_DEBUG
    TAU_STOP(rccl_call_name.c_str());
    if(record.operation == ROCPROFILER_RCCL_API_ID_ncclAllReduce)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclAllReduce.count*ncclDataTypeToSize(rccl_data->args.ncclAllReduce.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclAllGather)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclAllGather.sendcount*ncclDataTypeToSize(rccl_data->args.ncclAllGather.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclAllToAll)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclAllToAll.count*ncclDataTypeToSize(rccl_data->args.ncclAllToAll.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclBroadcast)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclBroadcast.count*ncclDataTypeToSize(rccl_data->args.ncclBroadcast.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclGather)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclGather.sendcount*ncclDataTypeToSize(rccl_data->args.ncclGather.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclReduce)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclReduce.count*ncclDataTypeToSize(rccl_data->args.ncclReduce.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclReduceScatter)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclReduceScatter.recvcount*ncclDataTypeToSize(rccl_data->args.ncclReduceScatter.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclScatter)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclScatter.recvcount*ncclDataTypeToSize(rccl_data->args.ncclScatter.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclSend)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclSend.count*ncclDataTypeToSize(rccl_data->args.ncclSend.datatype));
    }
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclRecv)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      TAU_CONTEXT_EVENT(ue, rccl_data->args.ncclRecv.count*ncclDataTypeToSize(rccl_data->args.ncclRecv.datatype));
    }
    //Special case with multiple different sizes
    else if(record.operation == ROCPROFILER_RCCL_API_ID_ncclAllToAllv)
    {
      auto rccl_data = static_cast<rocprofiler_callback_tracing_rccl_api_data_t*>(record.payload);
      void* ue = nullptr;
      rccl_call_name += " Message size ";
      Tau_get_context_userevent(&ue, rccl_call_name.c_str());
      int nranks = comm_nranks[rccl_data->args.ncclAllToAllv.comm];
      size_t total_size = 0;
      for (int i = 0; i < nranks; ++i)
      {
          total_size += rccl_data->args.ncclAllToAllv.sendcounts[i];
          total_size += rccl_data->args.ncclAllToAllv.recvcounts[i];
      }
      TAU_CONTEXT_EVENT(ue, total_size*ncclDataTypeToSize(rccl_data->args.ncclAllToAllv.datatype));
    }
  }
}

void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 callback_data)
{
    assert(callback_data != nullptr);
    //Invalid trace
    if(record.kind < ROCPROFILER_CALLBACK_TRACING_NONE || record.kind >= ROCPROFILER_CALLBACK_TRACING_LAST)
      return;
    
    /*
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER  && record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT )
    {
      if(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE)
        std::cout << "!! Callback ROCPROFILER_CALLBACK_PHASE_NONE " 
                  << c_client_name_info.kind_names[record.kind] 
                  << " " << c_client_name_info.operation_names[record.kind][record.operation] << std::endl;
      else if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD )
        std::cout << "!! Callback ROCPROFILER_CALLBACK_PHASE_LOAD " 
                  << c_client_name_info.kind_names[record.kind] 
                  << " " << c_client_name_info.operation_names[record.kind][record.operation] << std::endl;
      else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        std::cout << "!! Callback ROCPROFILER_CALLBACK_PHASE_UNLOAD " 
                  << c_client_name_info.kind_names[record.kind] 
                  << " " << c_client_name_info.operation_names[record.kind][record.operation] << std::endl;
    }
    */

    //Check if supported, as we may not support OMPT or others with this profiler, as we already have
    // and OMPT profiler
    //We can use context id to check if we received the start or only the end phase
    // or the user data with a flag
    /*
    pid_t tid1 = syscall(SYS_gettid);
    if (tid1 != record.thread_id)
    {
         
      auto     now = std::chrono::steady_clock::now().time_since_epoch().count();
      uint64_t dt  = 0;
      if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
          user_data->value = now;
      else
          dt = (now - user_data->value);
      auto info = std::stringstream{};
      info << std::left << "tid=" << record.thread_id << ", cid=" << std::setw(3)
          << record.correlation_id.internal << ", kind=" << c_client_name_info.kind_names[record.kind]
          //   record.kind
          << ", operation=" << c_client_name_info.operation_names[record.kind][record.operation]
          //record.operation 
          << ", phase=" << record.phase
          << ", dt_nsec=" << dt;
      
      auto    info_data = std::stringstream{};
      //ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
      //                    record, info_data_cb_string, record.phase, static_cast<void*>(&info_data)),
      //                "Failure iterating trace operation args");
      auto info_data_str = info_data.str();
      if(!info_data_str.empty()) info << " " << info_data_str << ")";
        std::cout << "!! cur_tid " << tid1 << " " << info.str()  << std::endl;
    }
    */
    
    
    switch(record.kind)
    {
      //Invalid callbacks
      case ROCPROFILER_CALLBACK_TRACING_NONE:
      case ROCPROFILER_CALLBACK_TRACING_LAST:
      {
        break;
      }
      case ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API:
      case ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API:
      case ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API:
      case ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API:
      {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
          TAU_START(c_client_name_info.operation_names[record.kind][record.operation]);
        }

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
          TAU_STOP(c_client_name_info.operation_names[record.kind][record.operation]);
        }
        #ifdef ROCSDK_DEBUG
        /*std::cout << (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER ? "ENTER ":"EXIT ")
                  << c_client_name_info.kind_names[record.kind] << " "
                  << c_client_name_info.operation_names[record.kind][record.operation]
                  << " tid: " << record.thread_id
                  << " cid: " << record.correlation_id.internal
                  << std::endl;*/
        #endif //ROCSDK_DEBUG
        break;
      }
      /*case ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API:
      {
        printf("HIP_COMPILER_API TODO !\n");
        #ifdef ROCSDK_DEBUG
        std::cout << (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER ? "ENTER ":"EXIT ")
                  << c_client_name_info.kind_names[record.kind] << " "
                  << c_client_name_info.operation_names[record.kind][record.operation]
                  << " tid: " << record.thread_id
                  << " cid: " << record.correlation_id.internal
                  << std::endl;
        #endif //ROCSDK_DEBUG
        break;
      }*/
      case ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API:
      {
        #ifdef ROCSDK_DEBUG
        auto    info_data = std::stringstream{};
        ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                          record, info_data_cb_string, record.phase, static_cast<void*>(&info_data)),
                      "Failure iterating trace operation args");
        std::cout << (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER ? "ENTER ":"EXIT ")
                  << c_client_name_info.kind_names[record.kind] << " "
                  << c_client_name_info.operation_names[record.kind][record.operation]
                  << " tid: " << record.thread_id
                  << " cid: " << record.correlation_id.internal
                   << " " << info_data.str() <<std::endl;
        #endif //ROCSDK_DEBUG
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
          TAU_START(c_client_name_info.operation_names[record.kind][record.operation]);
        }
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
          std::string name_hipcall = c_client_name_info.operation_names[record.kind][record.operation];
          TAU_STOP(name_hipcall.c_str());
          if (name_hipcall.compare(0, 9, "hipMemcpy") == 0)
          {
            std::vector<std::pair<std::string, std::string>> info_data_v;
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operation_args(
                          record, info_data_cb_vec, record.phase, static_cast<void*>(&info_data_v)),
                          "Failure iterating trace operation args");
            size_t sizeBytes;
            std::string kind = "Memory copy ";
            for (const auto& [name, value] : info_data_v) {
                if (name == "sizeBytes") {
                    sizeBytes =std::stoull(value);
                } else if (name == "kind") {
                    kind += value;
                }
            }
            void* ue = nullptr;
            Tau_get_context_userevent(&ue, kind.c_str());
            TAU_CONTEXT_EVENT(ue, (double) sizeBytes);
            //std::cout << sizeBytes << " " << kind << std::endl;
          }
        }
        
        break;
      }
      #if( ROCPROFILER_VERSION_MAJOR > 0)
      case ROCPROFILER_CALLBACK_TRACING_HIP_STREAM:
      {
        //Do not profile, only use to get the streams
        if(record.operation == ROCPROFILER_HIP_STREAM_SET)
        {
          auto* stream_data = static_cast<rocprofiler_callback_tracing_hip_stream_data_t*>(record.payload);
          if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
          {
            push_stream_id(stream_data->stream_id.handle);
          }
          if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
          {
            pop_stream_id();
          }
          #ifdef ROCSDK_DEBUG          
          std::cout << "!! "
                    << record.phase << " "
                    << c_client_name_info.kind_names[record.kind] << " "
                    << c_client_name_info.operation_names[record.kind][record.operation]
                    << " tid: " << record.thread_id
                    << " cid_i: " << record.correlation_id.internal
                    << " stream id: " << stream_data->stream_id.handle
                    << std::endl;
          #endif //ROCSDK_DEBUG
        }
        break;
      }
      #endif //( ROCPROFILER_VERSION_MAJOR > 0)
      case ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH:
      {
        //There are multiple operations and phases for this callback, 
        // we only want when the dispatch is complete and its phase is none.
        //operation=KERNEL_DISPATCH_COMPLETE
          if(record.operation == ROCPROFILER_KERNEL_DISPATCH_COMPLETE)
          {
            tau_rocsdk_kernel_dispatch(record);
          }
          else
          {
            #ifdef ROCSDK_DEBUG
              auto* kernel_dispatch = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
              std::string task_name = demangle_kernel_rocprofsdk(
                                        client_kernels.at(kernel_dispatch->dispatch_info.kernel_id).kernel_name, 1);

              std::cout << c_client_name_info.kind_names[record.kind] << " "
                      << c_client_name_info.operation_names[record.kind][record.operation]
                      << " phase: " << record.phase
                      << " tid: " << record.thread_id
                      << " cid_i: " << record.correlation_id.internal
                      << " cid_e: " << record.correlation_id.external.value
                      << " kernel_id: " << kernel_dispatch->dispatch_info.kernel_id
                      << " kernel_name: " << task_name
                      << " start: " << kernel_dispatch->start_timestamp 
                      << " end: " << kernel_dispatch->end_timestamp
                      << " agent: " << kernel_dispatch->dispatch_info.agent_id.handle
                      << " d_id: " << kernel_dispatch->dispatch_info.dispatch_id
                      << " grid_size x: " << kernel_dispatch->dispatch_info.grid_size.x
                      << " grid_size y: " << kernel_dispatch->dispatch_info.grid_size.y
                      << " grid_size z: " << kernel_dispatch->dispatch_info.grid_size.z
                      << " group_segment_size: " << kernel_dispatch->dispatch_info.group_segment_size
                      << " private_segment_size: " << kernel_dispatch->dispatch_info.private_segment_size
                      << " queue_id: " << kernel_dispatch->dispatch_info.queue_id.handle
                      << " workgroup_size x: " << kernel_dispatch->dispatch_info.workgroup_size.x
                      << " workgroup_size y: " << kernel_dispatch->dispatch_info.workgroup_size.y
                      << " workgroup_size z: " << kernel_dispatch->dispatch_info.workgroup_size.z
                      //<< " kernel: " << kernel_dispatch.dispatch_info.queue_id
                      //<< " stream: " << kernel_dispatch.dispatch_info.queue_id
                      << std::endl;
              #endif //ROCSDK_DEBUG
          }
        break;
      }
      //The memory copies have some issues with timers, they look async, but are reported
      // by the same thread, so overelaps occur or we discard them, as an alternative,
      // Memcpy related hip calls can provide the size of the copies and if they are HostToDevice,
      // HostToHost, etc.
      /*case ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY:
      {
        //Only the end has the timers
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
          printf("MEMCPY TODO!!\n");
          auto* mcpy_data = static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(record.payload);
          //#ifdef ROCSDK_DEBUG
          std::cout << record.phase << " "
                    << c_client_name_info.kind_names[record.kind] << " "
                    << c_client_name_info.operation_names[record.kind][record.operation]
                    << " tid: " << record.thread_id
                    << " cid: " << record.correlation_id.internal
                    << " bytes: " << mcpy_data->bytes
                    << " src_address: " << mcpy_data->src_address.ptr
                    << " dst_address: " << mcpy_data->dst_address.ptr
                    << " src_agent_id: " << mcpy_data->src_agent_id.handle
                    << " src_agent_id: " << TAU_rocsdk_id_device_map[mcpy_data->src_agent_id.handle]
                    << " dst_agent_id: " << mcpy_data->dst_agent_id.handle
                    << " src_agent_id: " << TAU_rocsdk_id_device_map[mcpy_data->dst_agent_id.handle]
                    << " start_timestamp: " << mcpy_data->start_timestamp
                    << " end_timestamp: " << mcpy_data->end_timestamp
                    << std::endl;
          //#endif //ROCSDK_DEBUG
        }
        break;


      }*/
      /*
      case ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION:
      {
        printf("MEMALLOC TODO!!\n");
        break;
      }
      */
      //case ROCPROFILER_CALLBACK_TRACING_ROCSHMEM_API:
      //{
      //    break;
      //}
      //case ROCPROFILER_CALLBACK_TRACING_HIPFILE_API:
      //{
      //    break;
      //}
      //Will need to update this part, with 7.2.4 roctxThreadRangeA appears, which is not documented.
      //case ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_RANGE_API:
      case ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API:
      {
        auto* marker_data = static_cast<rocprofiler_callback_tracing_marker_api_data_t*>(record.payload);
        #ifdef ROCSDK_DEBUG
        std::cout << (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER ? "ENTER ":"EXIT ")
                  << c_client_name_info.kind_names[record.kind] << " "
                  << c_client_name_info.operation_names[record.kind][record.operation]
                  << " tid: " << record.thread_id
                  << " cid: " << record.correlation_id.internal
                  << std::endl;
        #endif //ROCSDK_DEBUG         
        tau_roctx_process(marker_data, record.operation, record.phase);
        break;
      }
      //I don't think we support this
      //https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.2.1/api-reference/rocprofiler-sdk-roctx_api/roctx_modules/naming-utilities.html
      /*
      case ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API:
      {
        printf("MARKER NAME TODO!!\n");
        std::cout << record.phase << " "
                  << c_client_name_info.kind_names[record.kind] << " "
                  << c_client_name_info.operation_names[record.kind][record.operation]
                  << " tid: " << record.thread_id
                  << " cid: " << record.correlation_id.internal
                  << std::endl;
        break;
      }*/
      case ROCPROFILER_CALLBACK_TRACING_RCCL_API:
      {
        tau_rccl_process(record);
        break;
      }

      //case ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY:
      //case ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION:
      //case ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API:
      //case ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH:
      //{
      //  printf("OTHERS TODO!!\n");
        /*std::cout << (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER ? "ENTER ":"EXIT ")
                  << c_client_name_info.kind_names[record.kind] << " "
                  << c_client_name_info.operation_names[record.kind][record.operation]
                  << " tid: " << record.thread_id
                  << " cid: " << record.correlation_id.internal
                  << std::endl;*/
      //  break;
      //}
      default:
        break;
    }

    return;


    


}

void
tool_tracing_ctrl_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t*,
                           void* client_data)
{
    auto* ctx = static_cast<rocprofiler_context_id_t*>(client_data);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(*ctx), "pausing client context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(*ctx), "resuming client context");
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
    //Only execute if PC Sampling enabled
    if(pc_sampling == 1)
    {
      //printf("codeobj_tracing_callback\n");
      codeobj_tracing_callback(record);
    }
  }
  else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
          record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
  {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
      client_kernels.emplace(data->kernel_id, *data);
      //Only enable if needed for DEBUG
      //std::cout << data->kernel_id << " "<< data->kernel_name << std::endl;
    }

  }
}

void
tool_control_init(rocprofiler_context_id_t& primary_ctx)
{
    // Create a specialized (throw-away) context for handling ROCTx profiler pause and resume.
    // A separate context is used because if the context that is associated with roctxProfilerPause
    // disabled that same context, a call to roctxProfilerResume would be ignored because the
    // context that enables the callback for that API call is disabled.
    auto cntrl_ctx = rocprofiler_context_id_t{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&cntrl_ctx), "control context creation failed");

    // enable callback marker tracing with only the pause/resume operations
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         cntrl_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         tool_tracing_ctrl_callback,
                         &primary_ctx),
                     "callback tracing service failed to configure");

    // start the context so that it is always active
    ROCPROFILER_CALL(rocprofiler_start_context(cntrl_ctx), "start of control context");
}

//In the tool it is called set_kernel_rename_and_stream_correlation_id
// and also sets an struct with the pointer that externval correlation provides
// as we only need one value, we use the value field
int set_stream_correlation_id(rocprofiler_thread_id_t  thr_id,
                                            rocprofiler_context_id_t ctx_id,
                                            rocprofiler_external_correlation_id_request_kind_t kind,
                                            rocprofiler_tracing_operation_t                    op,
                                            uint64_t                 internal_corr_id,
                                            rocprofiler_user_data_t* external_corr_id,
                                            void*                    user_data)
{

    // Set the external correlation id service to point to struct
    external_corr_id->value = get_stream_id();

    return 0;
}

int tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
  assert(tool_data != nullptr);

  static std::mutex init_mutex;

  init_mutex.lock();

  //Check if there are any available agents, if not, do not initialize
  std::vector<rocprofiler_agent_v0_t> agents = get_gpu_device_agents();
  if(agents.empty())
  {
    printf("NO ROCm AGENTS FOUND\n");
    init_mutex.unlock();
    return 1;
  }

  static bool once = run_once();

  c_client_name_info = get_callback_tracing_names();

  ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation failed");

  // enable the control
  tool_control_init(client_ctx);

  for(auto itr : c_supported_kinds)
  {
      ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                        client_ctx, itr, nullptr, 0, tool_tracing_callback, tool_data),
                        "callback tracing service failed to configure");
  }

  ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                        client_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                        nullptr, 0, tool_code_object_callback, nullptr),
                        "code object tracing service configure"); 


  auto external_corr_id_request_kinds =  
              std::array<rocprofiler_external_correlation_id_request_kind_t, 1>{
              ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH};//,
              //ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,
              //ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,
              //ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API*/};
  ROCPROFILER_CALL(rocprofiler_configure_external_correlation_id_request_service(
                            client_ctx,
                            external_corr_id_request_kinds.data(),
                            external_corr_id_request_kinds.size(),
                            set_stream_correlation_id,
                            nullptr),
                        "Could not configure external correlation id request service");

  int valid_ctx = 0;
  ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                    "failure checking context validity");
  if(valid_ctx == 0)
  {
      // notify rocprofiler that initialization failed
      // and all the contexts, buffers, etc. created
      // should be ignored
      init_mutex.unlock();
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


  ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");
  configured_v3 = 1;
  init_mutex.unlock();
  // no errors
  return 0;
}

//End of rocprofiler-sdk
void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);
}


//Configure rocprofiler-sdk when executing the application
//executed before TAU starts
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure_(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
  //Removed for rocprofiler_force_configure, as TAU calls this part
  // previously, this function was called before TAU.
  //Changed due to issues with ROCm aware MPI, where it deadlocked.
  //Another alternative, would be to make the RocProfilerSDK part a library,
  // similar to Cupti, and use LD_PRELOAD, if errors appear with other frameworks.
  /*Tau_init_initializeTAU();

  #if (!(defined (TAU_MPI) || (TAU_SHMEM)))
  if (Tau_get_node() == -1) {
      TAU_PROFILE_SET_NODE(0);
  }
  #endif // TAU_MPI || TAU_SHMEM 
  */
  TAU_VERBOSE("Inside rocprofiler_configure\n");
  // Check in case the tool is launched but we don't need it
  if(use_rocprofilersdk() == 0)
  {
    TAU_VERBOSE("Do not use rocprofiler-sdk\n");
    return nullptr;
  }
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
  SDKFlush_mtx.lock();
  if(configured_v3==0)
  {
    TAU_VERBOSE("Flag -rocm not set or failed to configure, rocm is not profiled\n");
    SDKFlush_mtx.unlock();
    return;
  }
  if(flushed==1)
  {
    SDKFlush_mtx.unlock();
    return;
  }  
  
  if(pc_sampling == 1)
  {
    sdk_pc_sampling_flush();
  }
  flushed = 1;
  ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "rocprofiler context stop");
  SDKFlush_mtx.unlock();
}




void
Tau_rocm_initialize_v3()
{
    if(use_rocprofilersdk() == 0)
     return;
    int status = 0;

    SDK_init_lock.lock();

    if(initialized_v3 == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure_),
                         "force configuration");
        initialized_v3 = 1;
    }
    SDK_init_lock.unlock();

}
