#include "hip/hip_runtime.h"
#include <Profile/TauRocm.h>
#include <Profile/TauBfd.h>  // for name demangling

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cassert>
#include <chrono>
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

int volatile flushed = 0;


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




using buffer_kind_names_t = std::map<rocprofiler_buffer_tracing_kind_t, const char*>;
using buffer_kind_operation_names_t =
    std::map<rocprofiler_buffer_tracing_kind_t, std::map<uint32_t, const char*>>;

struct buffer_name_info
{
    buffer_kind_names_t           kind_names      = {};
    buffer_kind_operation_names_t operation_names = {};
};


rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {};
rocprofiler_buffer_id_t       client_buffer    = {};
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
kernel_symbol_map_t           client_kernels   = {};
buffer_name_info              client_name_info = {};



inline buffer_name_info
get_buffer_tracing_names()
{
    static const auto supported_kinds = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
        ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
        ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
        ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API,
        ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
    };

    auto cb_name_info = buffer_name_info{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_buffer_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<buffer_name_info*>(data_v);

            if(supported_kinds.count(kindv) > 0)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_buffer_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query buffer tracing kind operation name");
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











void
tool_code_object_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*              user_data,
                          void*                                 callback_data)
{
    if(flushed)
      return;
    //printf("----------- %s\n", __FUNCTION__);
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            //printf("----------- flush %s\n", __FUNCTION__);
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(client_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CALL(flush_status, "buffer flush");
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation ==
                ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            //printf("----------- emplace %s\n", __FUNCTION__);
            client_kernels.emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            //printf("----------- erase %s\n", __FUNCTION__);   
            client_kernels.erase(data->kernel_id);
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API)
    {
        if(record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
        {
            rocprofiler_stop_context(client_ctx);
        }
        else
        {
            rocprofiler_start_context(client_ctx);
        }
    }

    (void) user_data;
    (void) callback_data;
}




std::string get_copy_direction(int direction)
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
      mem_cpy_kind = "MEMORY_COPY_HOST_TO_DEVICE";
      break;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:
      mem_cpy_kind = "MEMORY_COPY_DEVICE_TO_HOST";
      break;
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE:
      mem_cpy_kind = "MEMORY_COPY_DEVICE_TO_DEVICE";
      break;
    case ROCPROFILER_MEMORY_COPY_LAST:
      mem_cpy_kind = "MEMORY_COPY_LAST";
      break;
    default:
      mem_cpy_kind = "MEMORY_COPY Unknown";
      break;
  }
  return mem_cpy_kind;
  //ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_NONE = 0,          ///< Unknown memory copy direction
  //  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_HOST_TO_HOST,      ///< Memory copy from host to host
  //  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_HOST_TO_DEVICE,    ///< Memory copy from host to device
  //  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_DEVICE_TO_HOST,    ///< Memory copy from device to host
  //  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_DEVICE_TO_DEVICE,  ///< Memory copy from device to device
  //  ROCPROFILER_BUFFER_TRACING_MEMORY_COPY_LAST,
}





void
tool_tracing_callback(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         user_data,
                      uint64_t                      drop_count)
{
  if(flushed)
    return;
  //printf("----------- %s\n", __FUNCTION__);
      assert(user_data != nullptr);
    assert(drop_count == 0 && "drop count should be zero for lossless policy");

    if(num_headers == 0)
        throw std::runtime_error{
            "rocprofiler invoked a buffer callback with no headers. this should never happen"};
    else if(headers == nullptr)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                                 "array of headers. this should never happen"};



    static unsigned long long last_timestamp = Tau_get_last_timestamp_ns();
    static std::map<uint64_t, std::string> timer_map;

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                (header->kind == ROCPROFILER_BUFFER_TRACING_HSA_CORE_API ||
                 header->kind == ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API ||
                 header->kind == ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API ||
                 header->kind == ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API))
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload);
            /*auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.operation_names[record->kind][record->operation];*/

            if(record->start_timestamp > record->end_timestamp)
            {
                auto msg = std::stringstream{};
                msg << "hsa api: start > end (" << record->start_timestamp << " > "
                    << record->end_timestamp
                    << "). diff = " << (record->start_timestamp - record->end_timestamp);
                std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
                // throw std::runtime_error{msg.str()};
            }
            //std::cout << "-" << info.str() << std::endl;
            //static_cast<common::call_stack_t*>(user_data)->emplace_back(
            //   common::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

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

            std::string kernel_name;
            std::string function_name;
            std::string task_name;


            function_name = client_name_info.operation_names[record->kind][record->operation];
            task_name = function_name;


            double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_entry);
            TAU_START_TASK(task_name.c_str(), taskid);
            

            double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_exit);
            TAU_STOP_TASK(task_name.c_str(), taskid);
            //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", task_name, taskid, tracer_record.timestamps.end.value);
            Tau_set_last_timestamp_ns(timestamp_exit);





        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);
            /*auto info = std::stringstream{};
            info << "tid=" << record->thread_id << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.operation_names[record->kind][record->operation];*/

            if(record->start_timestamp > record->end_timestamp)
            {
                auto msg = std::stringstream{};
                msg << "hip api: start > end (" << record->start_timestamp << " > "
                    << record->end_timestamp
                    << "). diff = " << (record->start_timestamp - record->end_timestamp);
                std::cerr << "threw an exception " << msg.str() << "\n" << std::flush;
                // throw std::runtime_error{msg.str()};
            }
             //std::cout << "+" << info.str() << std::endl;
            //static_cast<common::call_stack_t*>(user_data)->emplace_back(
            //    common::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});


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

            std::string kernel_name;
            std::string function_name;
            std::string task_name;


            function_name = client_name_info.operation_names[record->kind][record->operation];
            task_name = function_name;

            double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_entry);
            TAU_START_TASK(task_name.c_str(), taskid);
            

            double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_exit);
            TAU_STOP_TASK(task_name.c_str(), taskid);
            //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", task_name, taskid, tracer_record.timestamps.end.value);
            Tau_set_last_timestamp_ns(timestamp_exit);















        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

            /*auto info = std::stringstream{};
            info << "agent_id=" << record->agent_id.handle
                 << ", queue_id=" << record->queue_id.handle << ", kernel_id=" << record->kernel_id
                 << ", kernel=" << client_kernels.at(record->dispatch_info.kernel_id).kernel_name
                 << ", context=" << context.handle << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp
                 << ", private_segment_size=" << record->private_segment_size
                 << ", group_segment_size=" << record->group_segment_size << ", workgroup_size=("
                 << record->workgroup_size.x << "," << record->workgroup_size.y << ","
                 << record->workgroup_size.z << "), grid_size=(" << record->grid_size.x << ","
                 << record->grid_size.y << "," << record->grid_size.z << ")";*/

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("kernel dispatch: start > end");
            //std::cout << "*" << info.str() << std::endl;
            //static_cast<common::call_stack_t*>(user_data)->emplace_back(
             //  common::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

            int queueid = 2;
            unsigned long long timestamp = 0L;

            int taskid = Tau_get_initialized_queues(queueid);

            if (taskid == -1) { // not initialized
              TAU_CREATE_TASK(taskid);
              Tau_set_initialized_queues(queueid, taskid);
              timestamp = record->start_timestamp;
              Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE2", taskid);
              last_timestamp = timestamp;
              // Set the timestamp for TAUGPU_TIME:
              Tau_metric_set_synchronized_gpu_timestamp(taskid,
                                                        ((double)timestamp / 1e3));
              Tau_create_top_level_timer_if_necessary_task(taskid);
              Tau_add_metadata_for_task("ROCM_KERNEL_ID", taskid, taskid);

            }


            std::string kernel_name;
            std::string function_name;
            std::string task_name;


            function_name = Tau_demangle_name(client_kernels.at(record->dispatch_info.kernel_id).kernel_name);
            task_name = function_name;

            //Sizes seem to be bogus, implement later
            /*if (!TauEnv_get_thread_per_gpu_stream()) {
              std::stringstream ss;
              void *ue = nullptr;
              double value;
              std::string tmp;
              ss << "Grid Size : " << kernel_name_dem.c_str();
              tmp = ss.str();
              ue = Tau_get_userevent(tmp.c_str());
              //value = (double)(profiler_record->kernel_properties.grid_size);
              //TAU_VERBOSE("Grid Size :%ld\n",profiler_record->kernel_properties.grid_size);
              Tau_userevent_thread(ue, value, taskid);
            }*/








            double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_entry);
            TAU_START_TASK(task_name.c_str(), taskid);
            

            double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_exit);
            TAU_STOP_TASK(task_name.c_str(), taskid);
            //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", task_name, taskid, tracer_record.timestamps.end.value);
            Tau_set_last_timestamp_ns(timestamp_exit);




        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(header->payload);

            /*auto info = std::stringstream{};

            info << "src_agent_id=" << record->src_agent_id.handle
                 << ", dst_agent_id=" << record->dst_agent_id.handle
                 << ", direction=" << record->operation
                 << ", direction=" << get_copy_direction(record->operation) << ", context=" << context.handle
                 << ", buffer_id=" << buffer_id.handle
                 << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", start=" << record->start_timestamp
                 << ", stop=" << record->end_timestamp;*/

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("memory copy: start > end");
            //   std::cout << "/" << info.str() << std::endl;
            //static_cast<common::call_stack_t*>(user_data)->emplace_back(
            //    common::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});


            int queueid = 3;
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

            std::string kernel_name;
            std::string function_name;
            std::string task_name;


            function_name = get_copy_direction(record->operation);
            task_name = function_name;
			


            double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->start_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_entry);
            TAU_START_TASK(task_name.c_str(), taskid);
            

            double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)record->end_timestamp/1e3)); // convert to microseconds
            metric_set_gpu_timestamp(taskid, timestamp_exit);
            TAU_STOP_TASK(task_name.c_str(), taskid);
            //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", task_name, taskid, tracer_record.timestamps.end.value);
            Tau_set_last_timestamp_ns(timestamp_exit);


        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                (header->kind == ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API ||
                header->kind == ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API ||
                header->kind == ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API ))
        {
			  /*
            int queueid = 4;
            unsigned long long timestamp = 0L;

            taskid = Tau_get_initialized_queues(queueid);
        */
		  
          //printf("MARKER, not implemented, missing id and/or marker message to identify regions \n");
		  /*
			auto* record =
                static_cast<rocprofiler_buffer_tracing_marker_api_record_t*>(header->payload);
				
			auto info = std::stringstream{};

            info << ", cid=" << record->correlation_id.internal
                 << ", extern_cid=" << record->correlation_id.external.value
                 << ", kind=" << record->kind << ", operation=" << record->operation
                 << ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
                 << ", name=" << client_name_info.operation_names[record->kind][record->operation];
				//<< ", correlation_id= "<< record->correlation_id			
			

            if(record->start_timestamp > record->end_timestamp)
                throw std::runtime_error("memory copy: start > end");
            std::cout << ": " << info.str() << std::endl;
			*/
		
				
        }
			

















        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
            return;
        }
    }

    //printf("----------- %s     !!!END\n", __FUNCTION__);

}






void thread_precreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
  //printf("----------- %s\n", __FUNCTION__);
  return;
}


void thread_postcreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
  //printf("----------- %s\n", __FUNCTION__);
  return;
}


int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    //printf("----------- %s\n", __FUNCTION__);
    assert(tool_data != nullptr);

    client_name_info = get_buffer_tracing_names();

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
                                               4096,
                                               2048,
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               tool_tracing_callback,
                                               tool_data,
                                               &client_buffer),
                     "buffer creation");

    for(auto itr : {ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API})
    {
        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                             client_ctx, itr, nullptr, 0, client_buffer),
                         "buffer tracing service configure");
    }

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, client_buffer),
        "buffer tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer),
        "buffer tracing service for kernel dispatch configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, client_buffer),
        "buffer tracing service for memory copy configure");
		
		
	ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API, nullptr, 0, client_buffer),
        "buffer tracing service for memory copy configure");
	
	{
        auto throwaway_ctx = rocprofiler_context_id_t{};
        rocprofiler_create_context(&throwaway_ctx);
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             throwaway_ctx, ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                             nullptr, 0, tool_code_object_callback, &client_ctx),
                         "buffer tracing service for memory copy configure");
    }

	ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API, nullptr, 0, client_buffer),
        "buffer tracing service for memory copy configure");		
	

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
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start");

    // no errors
    return 0;
}


void
tool_fini(void* tool_data)
{
    //printf("----------- %s\n", __FUNCTION__);
    assert(tool_data != nullptr);

    //auto* _call_stack = static_cast<common::call_stack_t*>(tool_data);
    //_call_stack->emplace_back(common::source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    //print_call_stack(*_call_stack);

    //delete _call_stack;
}



void
setup()
{
    //printf("----------- %s\n", __FUNCTION__);
    int status = 0;
    if(rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure),
                         "force configuration");
    }
}

void
shutdown()
{
    //printf("----------- %s\n", __FUNCTION__);
    if(client_id)
    {
        ROCPROFILER_CALL(rocprofiler_flush_buffer(client_buffer), "buffer flush");
        client_fini_func(*client_id);
    }
}

void
start()
{
    //printf("----------- %s\n", __FUNCTION__);
    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "context start");
}

/*void
identify(uint64_t val)
{
    //printf("----------- %s\n", __FUNCTION__);
    auto _tid = rocprofiler_thread_id_t{};
    rocprofiler_get_thread_id(&_tid);
    rocprofiler_user_data_t user_data = {};
    user_data.value                   = val;
    rocprofiler_push_external_correlation_id(client_ctx, _tid, user_data);
}*/

void
stop()
{
    //printf("----------- %s\n", __FUNCTION__);
    ROCPROFILER_CALL(rocprofiler_stop_context(client_ctx), "context stop");
}


extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    //printf("----------- %s\n", __FUNCTION__);
        
    // only activate if main tool
    //if(priority > 0) return nullptr;

    // set the client name
    id->name = "TAU";

    // store client info
    client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    //auto* client_tool_data = new std::vector<source_location>{};
    char* client_tool_data = "";

    /*client_tool_data->emplace_back(
        common::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});*/

    //Is this necessary?
    ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(
                         thread_precreate,
                         thread_postcreate,
                         ROCPROFILER_LIBRARY | ROCPROFILER_HSA_LIBRARY | ROCPROFILER_HIP_LIBRARY |
                             ROCPROFILER_MARKER_LIBRARY,
                         static_cast<void*>(client_tool_data)),
                     "registration for thread creation notifications");

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    
    return &cfg;
}

       #include <unistd.h>

void Tau_rocprofv3_flush(){
   //printf("-----------!! %s\n", __FUNCTION__);
   //printf("%d\n", getpid());
   ROCPROFILER_CALL(rocprofiler_flush_buffer(client_buffer), "buffer flush");
   
   for (int i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    if (Tau_get_initialized_queues(i) != -1) {
      RtsLayer::LockDB();
      if (Tau_get_initialized_queues(i) != -1) {  // contention. Is it still -1?
        TAU_VERBOSE("Closing thread id: %d last timestamp = %llu\n", Tau_get_initialized_queues(i), Tau_get_last_timestamp_ns());
        Tau_metric_set_synchronized_gpu_timestamp(i, ((double)Tau_get_last_timestamp_ns()/1e3)); // convert to microseconds
        Tau_stop_top_level_timer_if_necessary_task(Tau_get_initialized_queues(i));
        Tau_set_initialized_queues(i, -1);
      }
      RtsLayer::UnLockDB();
    }
  }
  
  if(flushed)
    return;
  else if (flushed==0)
    flushed = -1;
  else if(flushed==-1)
    flushed = -1;
  //ROCPROFILER_CALL(rocprofiler_flush_buffer(client_buffer), "buffer flush");
}
