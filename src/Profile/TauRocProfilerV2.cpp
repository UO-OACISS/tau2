/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

//#include "Profile/hsa_rsrc_factory.h"
#include <hsa/hsa.h>
#include <Profile/TauRocm.h>
#include <Profile/TauBfd.h>  // for name demangling
#include <rocprofiler/v2/rocprofiler.h>


#include <dlfcn.h>
#include <string.h>
#include <unistd.h>

#include <cstdio>
#include <cstdarg>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <amd_comgr/amd_comgr.h>

#include <atomic>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>


// Dispatch callbacks and context handlers synchronization
pthread_mutex_t rocm_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Tool is unloaded
volatile bool is_loaded = false;

/* I know it's bad form to have this map just hanging out here,
 * but when I wrapped it with a getter function, it failed to work.
 * A regular map was always empty, and an unordered map would crash
 * at the find method.  Whatever.  Maybe it's a hipcc problem.
 * So, it's possible we can access this map after the program has
 * destroyed it, but that's a risk I am willing to take. */
using mapType = std::unordered_map<uint64_t, std::string>;
static mapType themap;
std::mutex mapLock;

// the user event for correlation IDs
//static void* TraceCorrelationID;

// Launch a kernel
void Tau_roctracer_register_activity(uint64_t id, std::string function_name_in, std::string kernel_name_in) {
    std::string name = "";
    if (( function_name_in.size() >1 ) && ( kernel_name_in.size() >1 ))
    {
      name = function_name_in + " " + Tau_demangle_name(kernel_name_in.c_str());
    }
    else if( function_name_in.size() >1 )
    {
      name = function_name_in;
    }
    else if (kernel_name_in.size() >1)
    {
      name = Tau_demangle_name(kernel_name_in.c_str());
    }
    else{
      printf("ERROR!! No function nor kernel names!!\n");
    }
    //printf("Register: (%s) (%s) as %s\n",function_name_in.c_str(), kernel_name_in.c_str(), name.c_str());
    mapLock.lock();
    themap.insert(std::pair<uint64_t, std::string>(id, name));
    mapLock.unlock();
    return;
}

// resolve a kernel
std::string Tau_roctracer_lookup_activity(uint64_t id) {
    std::string name("");
    mapLock.lock();
    auto i = themap.find(id);
    if (i != themap.end()) {
        name = i->second;
        themap.erase(id);
    } else {
        TAU_VERBOSE("WARNING! Kernel name not found for correlation %lu\n", id);
    }
    mapLock.unlock();
    return name;
}


// Macro to check ROCProfiler calls status
#define CHECK_ROCPROFILER(call)                                                \
  do {                                                                         \
    if ((call) != ROCPROFILER_STATUS_SUCCESS)                                  \
      printf("%s\n", "Error: ROCProfiler API Call Error!");                    \
  } while (false)

rocprofiler_session_id_t session_id;
rocprofiler_buffer_id_t counter_buffer_id;
rocprofiler_buffer_id_t trace_buffer_id;
static std::vector<rocprofiler_filter_id_t> filter_ids;

  const char* GetDomainName(rocprofiler_tracer_activity_domain_t domain) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX:
        return "ROCTX_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HIP_API:
        return "HIP_API_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HIP_OPS:
        return "HIP_OPS_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_API:
        return "HSA_API_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_OPS:
        return "HSA_OPS_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_EVT:
        return "HSA_EVT_DOMAIN";
        break;
      default:
        return "";
    }
  }


/*
https://github.com/UO-OACISS/tau2/blob/a921b0a6a3b1b69979e1801bc1aff8fe21dd9fcc/src/Profile/TauRocTracer.cpp
https://github.com/ROCm-Developer-Tools/rocprofiler/blob/f914c8a819690d581ca9a2a9b463e676fdb47d81/plugin/file/file.cpp#L229
https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/src/tools/rocprofv2/rocprofv2.cpp
https://github.com/ROCm-Developer-Tools/rocprofiler/blob/amd-master/src/tools/tool.cpp
https://github.com/ROCm-Developer-Tools/rocprofiler/blob/f914c8a819690d581ca9a2a9b463e676fdb47d81/samples/common/helper.cpp
https://github.com/UO-OACISS/tau2/blob/a921b0a6a3b1b69979e1801bc1aff8fe21dd9fcc/src/Profile/TauRocTracer.cpp
TauEnv_get_thread_per_gpu_stream()
*/

void FlushTracerRecord(rocprofiler_record_tracer_t tracer_record, rocprofiler_session_id_t session_id)
{

  int taskid, queueid;
  unsigned long long timestamp = 0L;
  static unsigned long long last_timestamp = Tau_get_last_timestamp_ns();
  queueid = tracer_record.queue_id.handle;
  taskid = Tau_get_initialized_queues(queueid);
  static std::map<uint64_t, std::string> timer_map;
  static std::mutex map_lock;
  //printf("tracer_record taskid %d\n", taskid);
  if (taskid == -1) { // not initialized
    TAU_CREATE_TASK(taskid);
    TAU_VERBOSE(
        "Tau_rocm_dump_context_entry: associating queueid %d with taskid %d\n",
        queueid, taskid);
    Tau_set_initialized_queues(queueid, taskid);
    timestamp = tracer_record.timestamps.begin.value;
    Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE", taskid);
    last_timestamp = timestamp;
    // Set the timestamp for TAUGPU_TIME:
    Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                              ((double)timestamp / 1e3));
    Tau_create_top_level_timer_if_necessary_task(taskid);
    Tau_add_metadata_for_task("TAU_TASK_ID", taskid, taskid);
    Tau_add_metadata_for_task("ROCM_GPU_ID", tracer_record.agent_id.handle,
                              taskid);
    Tau_add_metadata_for_task("ROCM_QUEUE_ID", tracer_record.queue_id.handle,
                              taskid);
  }

    std::string kernel_name;
    std::string function_name;
    std::string roctx_message;
    uint64_t roctx_id = 0;
    bool roctx_used = false;
        if ((tracer_record.operation_id.id == 0 && tracer_record.domain == ACTIVITY_DOMAIN_HIP_OPS)) {
      if (tracer_record.api_data_handle.handle &&
          strlen(reinterpret_cast<const char*>(tracer_record.api_data_handle.handle)) > 1)
        kernel_name = Tau_demangle_name(
            reinterpret_cast<const char*>(tracer_record.api_data_handle.handle));
    }
    if (tracer_record.domain == ACTIVITY_DOMAIN_HSA_API) {
      size_t function_name_size = 0;
      CHECK_ROCPROFILER(rocprofiler_query_hsa_tracer_api_data_info_size(
          session_id, ROCPROFILER_HSA_FUNCTION_NAME, tracer_record.api_data_handle,
          tracer_record.operation_id, &function_name_size));
      if (function_name_size > 1) {
        char* function_name_c = (char*)malloc(function_name_size);
        CHECK_ROCPROFILER(rocprofiler_query_hsa_tracer_api_data_info(
            session_id, ROCPROFILER_HSA_FUNCTION_NAME, tracer_record.api_data_handle,
            tracer_record.operation_id, &function_name_c));
        if (function_name_c) function_name = std::string(function_name_c);
      }

      Tau_roctracer_register_activity(tracer_record.correlation_id.value,
              function_name, kernel_name);
    }
    if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_API) {
      size_t function_name_size = 0;
      CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info_size(
          session_id, ROCPROFILER_HIP_FUNCTION_NAME, tracer_record.api_data_handle,
          tracer_record.operation_id, &function_name_size));
      if (function_name_size > 1) {
        char* function_name_c = (char*)malloc(function_name_size);
        CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info(
            session_id, ROCPROFILER_HIP_FUNCTION_NAME, tracer_record.api_data_handle,
            tracer_record.operation_id, &function_name_c));
        if (function_name_c) function_name = std::string(function_name_c);
      }
      size_t kernel_name_size = 0;
      CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info_size(
          session_id, ROCPROFILER_HIP_KERNEL_NAME, tracer_record.api_data_handle,
          tracer_record.operation_id, &kernel_name_size));
      if (kernel_name_size > 1) {
        char* kernel_name_str = (char*)malloc(kernel_name_size * sizeof(char));
        CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info(
            session_id, ROCPROFILER_HIP_KERNEL_NAME, tracer_record.api_data_handle,
            tracer_record.operation_id, &kernel_name_str));
        if (kernel_name_str) kernel_name = Tau_demangle_name(kernel_name_str);
      }

      Tau_roctracer_register_activity(tracer_record.correlation_id.value,
              function_name, kernel_name);
    }
    if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX) {
      size_t roctx_message_size = 0;
      CHECK_ROCPROFILER(rocprofiler_query_roctx_tracer_api_data_info_size(
          session_id, ROCPROFILER_ROCTX_MESSAGE, tracer_record.api_data_handle,
          tracer_record.operation_id, &roctx_message_size));
      if (roctx_message_size > 1) {
        [[maybe_unused]] char* roctx_message_str =
            static_cast<char*>(malloc(roctx_message_size * sizeof(char)));
        CHECK_ROCPROFILER(rocprofiler_query_roctx_tracer_api_data_info(
            session_id, ROCPROFILER_ROCTX_MESSAGE, tracer_record.api_data_handle,
            tracer_record.operation_id, &roctx_message_str));
        if (roctx_message_str)
          roctx_message = Tau_demangle_name(strdup(roctx_message_str));
      }
      size_t roctx_id_size = 0;
      CHECK_ROCPROFILER(rocprofiler_query_roctx_tracer_api_data_info_size(
          session_id, ROCPROFILER_ROCTX_ID, tracer_record.api_data_handle, tracer_record.operation_id,
          &roctx_id_size));
      if (roctx_id_size > 1) {
        [[maybe_unused]] char* roctx_id_str =
            static_cast<char*>(malloc(roctx_id_size * sizeof(char)));
        CHECK_ROCPROFILER(rocprofiler_query_roctx_tracer_api_data_info(
            session_id, ROCPROFILER_ROCTX_ID, tracer_record.api_data_handle,
            tracer_record.operation_id, &roctx_id_str));
        if (roctx_id_str) {
          roctx_used = true;
          roctx_id = std::stoll(std::string(strdup(roctx_id_str)));
          free(roctx_id_str);
        }
	if(roctx_id > 0){
		if(roctx_message_size>1){
	                const std::lock_guard<std::mutex> guard(map_lock);
			TAU_START(roctx_message.c_str());
			timer_map.insert( std::pair<uint64_t, std::string>(roctx_id, roctx_message));
		}
		else{
	                const std::lock_guard<std::mutex> guard(map_lock);
			 auto p = timer_map.find(roctx_id);
                	if (p != timer_map.end()) {
                    		TAU_STOP(p->second.c_str());
                    		timer_map.erase(roctx_id);
                	}
		}
	}
      }
    }
    if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_OPS) {
      function_name = std::string(Tau_roctracer_lookup_activity(tracer_record.correlation_id.value));
      TAU_VERBOSE("ACTIVITY_DOMAIN_HIP_OPS %lu %s\n", tracer_record.correlation_id.value, function_name.c_str());
    }
    if (tracer_record.domain == ACTIVITY_DOMAIN_HSA_OPS) {
      function_name = std::string(Tau_roctracer_lookup_activity(tracer_record.correlation_id.value));
      TAU_VERBOSE("ACTIVITY_DOMAIN_HSA_OPS %lu %s\n", tracer_record.correlation_id.value, function_name.c_str());
    }
    std::string task_name;
    TAU_VERBOSE("Record [%lu]", tracer_record.header.id.handle);
    TAU_VERBOSE(", Domain(%s)", GetDomainName(tracer_record.domain));
    TAU_VERBOSE(", Begin(%lu)", tracer_record.timestamps.begin.value);
    TAU_VERBOSE(", End(%lu)", tracer_record.timestamps.end.value);
    TAU_VERBOSE(", Correlation ID(%lu)", tracer_record.correlation_id.value);

    if (roctx_used){
      TAU_VERBOSE(", ROCTX ID( %lu )", roctx_id);
    }
    if (roctx_message.size() > 1){
      TAU_VERBOSE(", ROCTX Message( %s )", roctx_message.c_str());
     task_name = roctx_message;
    }
    if (function_name.size() > 1){
      TAU_VERBOSE(", Function( %s )", function_name.c_str());
     task_name = function_name;
    }
    if (kernel_name.size() > 1){
      TAU_VERBOSE(", Kernel name( %s )", kernel_name.c_str());
     task_name = kernel_name;
    }
    if((function_name.size() > 1) && (kernel_name.size() > 1)){
      task_name = function_name+ " " + kernel_name;
    }
    TAU_VERBOSE("\n");



  if (tracer_record.domain != ACTIVITY_DOMAIN_ROCTX) {
    metric_set_gpu_timestamp(taskid, ((double)(tracer_record.timestamps.begin.value)));
    TAU_START_TASK(task_name.c_str(), taskid);

    metric_set_gpu_timestamp(taskid, ((double)(tracer_record.timestamps.end.value)));
    TAU_STOP_TASK(task_name.c_str(), taskid);
    //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", task_name, taskid, tracer_record.timestamps.end.value);
    Tau_set_last_timestamp_ns(tracer_record.timestamps.end.value);
  }
}

void FlushProfilerRecord(const rocprofiler_record_profiler_t *profiler_record,
                         rocprofiler_session_id_t session_id) {
  int taskid, queueid;
  unsigned long long timestamp = 0L;
  static unsigned long long last_timestamp = Tau_get_last_timestamp_ns();

  size_t name_length = 0;
  CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(
      ROCPROFILER_KERNEL_NAME, profiler_record->kernel_id, &name_length));
  // Taken from rocprofiler: The size hasn't changed in  recent past
  // static const uint32_t lds_block_size = 128 * 4;
  const char *kernel_name_c = "";
  std::string kernel_name_dem = "";
  if (name_length > 1) {
    kernel_name_c =
        static_cast<const char *>(malloc(name_length * sizeof(char)));
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info(
        ROCPROFILER_KERNEL_NAME, profiler_record->kernel_id, &kernel_name_c));
    kernel_name_dem = Tau_demangle_name(kernel_name_c);
    //kernel_name_dem = kernel_name_c;
  }
  queueid = profiler_record->queue_id.handle;
  taskid = Tau_get_initialized_queues(queueid);

  if (taskid == -1) { // not initialized
    TAU_CREATE_TASK(taskid);
    TAU_VERBOSE(
        "Tau_rocm_dump_context_entry: associating queueid %d with taskid %d\n",
        queueid, taskid);
    Tau_set_initialized_queues(queueid, taskid);
    timestamp = profiler_record->timestamps.begin.value;
    Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE", taskid);
    last_timestamp = timestamp;
    // Set the timestamp for TAUGPU_TIME:
    Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                              ((double)timestamp / 1e3));
    Tau_create_top_level_timer_if_necessary_task(taskid);
    Tau_add_metadata_for_task("TAU_TASK_ID", taskid, taskid);
    Tau_add_metadata_for_task("ROCM_GPU_ID", profiler_record->gpu_id.handle,
                              taskid);
    Tau_add_metadata_for_task("ROCM_QUEUE_ID", profiler_record->queue_id.handle,
                              taskid);
    Tau_add_metadata_for_task("ROCM_THREAD_ID",
                              profiler_record->thread_id.value, taskid);
  }
  
  TAU_VERBOSE(" --> NEW EVENT --> \n");
  struct TauRocmEvent e(kernel_name_dem.c_str(), profiler_record->timestamps.begin.value,
                        profiler_record->timestamps.end.value, taskid);
  TAU_VERBOSE("KERNEL: name: %s entry: %lu exit: %lu ...\n", kernel_name_dem.c_str(),
              profiler_record->timestamps.begin.value,
              profiler_record->timestamps.end.value);
  /* Capture these with counters! */
  if (!TauEnv_get_thread_per_gpu_stream()) {
    std::stringstream ss;
    void *ue = nullptr;
    double value;
    std::string tmp;
    ss << "Grid Size : " << kernel_name_dem.c_str();
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(profiler_record->kernel_properties.grid_size);
    TAU_VERBOSE("Grid Size :%ld\n",profiler_record->kernel_properties.grid_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Work Group Size : " << kernel_name_dem.c_str();
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(profiler_record->kernel_properties.workgroup_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");

    ss << "LDS Memory Size : " << kernel_name_dem;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    static const uint32_t lds_block_size = 128 * 4;
    value = (double)((profiler_record->kernel_properties.lds_size +
                      (lds_block_size - 1)) &
                     ~(lds_block_size - 1));
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Scratch Memory Size : " << kernel_name_dem;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(profiler_record->kernel_properties.scratch_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Vector Register Size (VGPR) : " << kernel_name_dem;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    // value = (double)((entry->kernel_properties.vgpr_count + 1) *
    // agent_info->vgpr_block_size);
    value = (double)(profiler_record->kernel_properties.accum_vgpr_count);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Scalar Register Size (SGPR) : " << kernel_name_dem;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());

    value = (double)(profiler_record->kernel_properties.sgpr_count);
    Tau_userevent_thread(ue, value, taskid);


  }
  Tau_process_rocm_events(e);
  timestamp = profiler_record->timestamps.begin.value;

  last_timestamp = timestamp;
  timestamp = profiler_record->timestamps.end.value;
  last_timestamp = timestamp;
  Tau_set_last_timestamp_ns(profiler_record->timestamps.end.value);

  /*metric_set_gpu_timestamp(taskid, ((double)(profiler_record->timestamps.begin.value)));
  TAU_START_TASK(kernel_name_dem.c_str(), taskid);

  metric_set_gpu_timestamp(taskid, ((double)(profiler_record->timestamps.end.value)));
  TAU_STOP_TASK(kernel_name_dem.c_str(), taskid);
  //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", task_name, taskid, tracer_record.timestamps.end.value);
  Tau_set_last_timestamp_ns(profiler_record->timestamps.end.value);*/

}

int WriteBufferRecords(const rocprofiler_record_header_t *begin,
                       const rocprofiler_record_header_t *end,
                       rocprofiler_session_id_t session_id,
                       rocprofiler_buffer_id_t buffer_id) {
  if (pthread_mutex_lock(&rocm_mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

    if(is_loaded)
    {
            TAU_VERBOSE("WriteBufferRecords\n");
            
            fflush(stdout);

            while (begin < end) {
              if (!begin)
                return 0;
              switch (begin->kind) {
              case ROCPROFILER_PROFILER_RECORD: {
                const rocprofiler_record_profiler_t *profiler_record =
                    reinterpret_cast<const rocprofiler_record_profiler_t *>(
                        begin);
                FlushProfilerRecord(profiler_record, session_id);
                break;
              }
            case ROCPROFILER_TRACER_RECORD:{
              rocprofiler_record_tracer_t* tracer_record = const_cast<rocprofiler_record_tracer_t*>(
              reinterpret_cast<const rocprofiler_record_tracer_t*>(begin));
              FlushTracerRecord(*tracer_record, session_id);
              break;
            }
              default: {
                break;
              }
              }
              rocprofiler_next_record(begin, &begin, session_id, buffer_id);
            }

    }
  if (pthread_mutex_unlock(&rocm_mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return 0;
}


extern void Tau_roc_trace_sync_call_v2(rocprofiler_record_tracer_t tracer_record,
                                  rocprofiler_session_id_t session_id){
  TAU_VERBOSE("Tau_roc_trace_sync_call_v2\n");
    if (pthread_mutex_lock(&rocm_mutex) != 0) {
      perror("pthread_mutex_lock");
      abort();
    }

      if(is_loaded)
      {
        FlushTracerRecord(tracer_record, session_id);
      }
    if (pthread_mutex_unlock(&rocm_mutex) != 0) {
      perror("pthread_mutex_unlock");
      abort();
    }


}



ROCPROFILER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 1025;

void Tau_rocm_initialize_v2() {

  if (pthread_mutex_lock(&rocm_mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

    if (!is_loaded){
            is_loaded = true;
            TAU_VERBOSE("Inside Tau_rocm_initialize_v2\n");
             if (rocprofiler_version_major() != ROCPROFILER_VERSION_MAJOR ||
                rocprofiler_version_minor() < ROCPROFILER_VERSION_MINOR) {
                printf("!!!!the ROCProfiler API version is not compatible with this tool!!\n");
                return ;
            }

            // Initialize the tools
            CHECK_ROCPROFILER(rocprofiler_initialize());

            CHECK_ROCPROFILER(rocprofiler_create_session(
                ROCPROFILER_KERNEL_REPLAY_MODE, &session_id));


            std::vector<rocprofiler_tracer_activity_domain_t> apis_requested;

            apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_API);
            apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_OPS);
            apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_API);
            apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_OPS);
            apis_requested.emplace_back(ACTIVITY_DOMAIN_ROCTX);


            // Creating Output Buffer for the data

            CHECK_ROCPROFILER(rocprofiler_create_buffer(
                session_id,
                [](const rocprofiler_record_header_t *record,
                   const rocprofiler_record_header_t *end_record,
                   rocprofiler_session_id_t session_id,
                   rocprofiler_buffer_id_t counter_buffer_id) {
                  WriteBufferRecords(record, end_record, session_id, counter_buffer_id);
                },
                1<<20, &counter_buffer_id));

            CHECK_ROCPROFILER(rocprofiler_create_buffer(
                session_id,
                [](const rocprofiler_record_header_t *record,
                   const rocprofiler_record_header_t *end_record,
                   rocprofiler_session_id_t session_id,
                   rocprofiler_buffer_id_t trace_buffer_id) {
                  WriteBufferRecords(record, end_record, session_id, trace_buffer_id);
                },
                1<<20, &trace_buffer_id));           

            rocprofiler_filter_id_t filter_id;
            [[maybe_unused]] rocprofiler_filter_property_t property = {};
            CHECK_ROCPROFILER(rocprofiler_create_filter(
                session_id, ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION,
                 rocprofiler_filter_data_t{},
                0, &filter_id, property));

            CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(
                session_id, filter_id, counter_buffer_id));
            
            filter_ids.emplace_back(filter_id);

            rocprofiler_filter_id_t filter_id1;
            [[maybe_unused]] rocprofiler_filter_property_t property1 = {};
            CHECK_ROCPROFILER(rocprofiler_create_filter(
                session_id, ROCPROFILER_API_TRACE,
                rocprofiler_filter_data_t{&apis_requested[0]},
                apis_requested.size(), &filter_id1, property1));

            CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(
                session_id, filter_id1, trace_buffer_id));
            CHECK_ROCPROFILER(rocprofiler_set_api_trace_sync_callback( session_id,
              filter_id1, Tau_roc_trace_sync_call_v2));

            filter_ids.emplace_back(filter_id1);

            CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
    }
  if (pthread_mutex_unlock(&rocm_mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

}


void Tau_rocprofiler_terminate_session()
{
      if (pthread_mutex_lock(&rocm_mutex) != 0) {
        perror("pthread_mutex_lock");
        abort();
      }
        if (is_loaded){
          is_loaded = false;
          TAU_VERBOSE("Inside rocprofiler_terminate_session\n");
                CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));

        }
        if (pthread_mutex_unlock(&rocm_mutex) != 0) {
          perror("pthread_mutex_unlock");
          abort();
        }
}

void Tau_rocprofiler_pool_flush() {
			TAU_VERBOSE("Inside Tau_rocprofiler_pool_flush_v2\n");



          CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, counter_buffer_id));
          CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, trace_buffer_id));

          Tau_rocprofiler_terminate_session();

}

// Stop tracing routine
extern void Tau_rocprofv2_stop() {

      // Destroy all profiling related objects(User buffer, sessions,
      // filters, etc..)
    Tau_rocprofiler_terminate_session();
                      // Destroy sessions
    TAU_VERBOSE("rocprofiler_destroy_session\n");
                CHECK_ROCPROFILER(rocprofiler_destroy_session(session_id));
    TAU_VERBOSE("rocprofiler_finalize\n");
            CHECK_ROCPROFILER(rocprofiler_finalize());
}

