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
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>

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
#include <set>
#include <unistd.h>


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

using avail_configs_vec_t         = std::vector<rocprofiler_pc_sampling_configuration_t>;
struct tool_agent_info
{
    rocprofiler_agent_id_t               agent_id;
    std::unique_ptr<avail_configs_vec_t> avail_configs;
    const rocprofiler_agent_t*           agent;
};

using pc_sampling_buffer_id_vec_t = std::vector<rocprofiler_buffer_id_t>;
using tool_agent_info_vec_t       = std::vector<std::unique_ptr<tool_agent_info>>;
pc_sampling_buffer_id_vec_t* pc_buffer_ids = nullptr;
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);



using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
kernel_symbol_map_t           client_kernels   = {};
buffer_name_info              client_name_info = {};

//Enum to enable or disable metric profiling
typedef enum profile_metrics {
	NO_METRICS = 1,
	WRONG_NAME,
	PROFILE_METRICS
};

//Map to identify counters
std::map<uint64_t, const char*> used_counter_id_map ;

std::map<rocprofiler_dispatch_id_t, rocprofiler_kernel_id_t> dispatch_id_kernel_map;




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


/**
 * For a given counter, query the dimensions that it has. Typically you will
 * want to call this function once to get the dimensions and cache them.
 */
std::vector<rocprofiler_record_dimension_info_t>
counter_dimensions(rocprofiler_counter_id_t counter)
{
    std::vector<rocprofiler_record_dimension_info_t> dims;
    rocprofiler_available_dimensions_cb_t            cb =
        [](rocprofiler_counter_id_t,
           const rocprofiler_record_dimension_info_t* dim_info,
           size_t                                     num_dims,
           void*                                      user_data) {
            std::vector<rocprofiler_record_dimension_info_t>* vec =
                static_cast<std::vector<rocprofiler_record_dimension_info_t>*>(user_data);
            for(size_t i = 0; i < num_dims; i++)
            {
                vec->push_back(dim_info[i]);
            }
            return ROCPROFILER_STATUS_SUCCESS;
        };
    ROCPROFILER_CALL(rocprofiler_iterate_counter_dimensions(counter, cb, &dims),
                     "Could not iterate counter dimensions");
    return dims;
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
			
			
			
			/*std::cout << "tid=" << record->thread_id << ", context=" << context.handle
				<< ", buffer_id=" << buffer_id.handle
				<< ", cid=" << record->correlation_id.internal
				<< ", extern_cid=" << record->correlation_id.external.value
				<< ", kind=" << record->kind << ", operation=" << record->operation
				<< ", agent_id=" << record->dispatch_info.agent_id.handle
				<< ", queue_id=" << record->dispatch_info.queue_id.handle
				<< ", kernel_id=" << record->dispatch_info.kernel_id
				<< ", kernel=" << client_kernels.at(record->dispatch_info.kernel_id).kernel_name
				<< ", start=" << record->start_timestamp << ", stop=" << record->end_timestamp
				<< ", private_segment_size=" << record->dispatch_info.private_segment_size
				<< ", group_segment_size=" << record->dispatch_info.group_segment_size
				<< ", workgroup_size=(" << record->dispatch_info.workgroup_size.x << ","
				<< record->dispatch_info.workgroup_size.y << ","
				<< record->dispatch_info.workgroup_size.z << "), grid_size=("
				<< record->dispatch_info.grid_size.x << "," << record->dispatch_info.grid_size.y
				<< "," << record->dispatch_info.grid_size.z << ")" << std::endl;*/
			
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
			



		//Use the headers to idenfity the kernels
		else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
		{
				//We will use this information to identify correlate dispatch ids and kernel ids
				//with the kernel ids, we can identify the kernel's name.
				//Not all dispatches ids may appear in the same call to this function, we need to store them.
				
			auto* record =
				static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);
			/*std::cout << "[Dispatch_Id: " << record->dispatch_info.dispatch_id
				 << " , Kernel_ID: " << record->dispatch_info.kernel_id
				 << " , Kernel name: " << Tau_demangle_name(client_kernels.at(record->dispatch_info.kernel_id).kernel_name)
				 << " , Corr_Id: " << record->correlation_id.internal << ")]\n";*/
			 
					//std::map<rocprofiler_dispatch_id_t, rocprofiler_kernel_id_t> dispatch_id_name_map;
			dispatch_id_kernel_map.emplace(record->dispatch_info.dispatch_id, record->dispatch_info.kernel_id);
				
		}//Read the values and store the events
		else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
				header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
		{
			
			// Print the returned counter data.
			auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
			rocprofiler_counter_id_t counter_id = {.handle = 0};

			rocprofiler_query_record_counter_id(record->id, &counter_id);
			
			
			std::stringstream ss;
			std::string tmp;
			void* ue = nullptr;
			double value;
			
			

			/*std::cout << "  (Dispatch_Id: " << record->dispatch_id << " Kernel_Id: " << dispatch_id_kernel_map[record->dispatch_id] << " Kernel_name: "  <<
			  Tau_demangle_name(client_kernels.at(dispatch_id_kernel_map[record->dispatch_id]).kernel_name) << " Counter_Id: " << counter_id.handle
			  << " Counter name: " << used_counter_id_map[counter_id.handle] << " Record_Id: " << record->id << " Dimensions: [";*/
			
			ss << used_counter_id_map[counter_id.handle] << " ";
			ss << Tau_demangle_name(client_kernels.at(dispatch_id_kernel_map[record->dispatch_id]).kernel_name);
			ss << " [";
			

			for(auto& dim : counter_dimensions(counter_id))
			{
				size_t pos = 0;
				rocprofiler_query_record_dimension_position(record->id, dim.id, &pos);
				//std::cout << "{" << dim.name << ": " << pos << "},";
				ss << " " << dim.name << ": " << pos ;
			}
			//std::cout << "] Value [D]: " << record->counter_value << ")";
			//std::cout << std::endl;
			
			ss << "]";			
			
			int queueid = 2;

            int taskid = Tau_get_initialized_queues(queueid);			
			
			tmp = ss.str();
			ue = Tau_get_userevent(tmp.c_str());
			value = (double)(record->counter_value);
			Tau_userevent_thread(ue, value, taskid);
      
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

/**
 * Cache to store the profile configs for each agent. This is used to prevent
 * constructing the same profile config multiple times. Used by dispatch_callback
 * to select the profile config (and in turn counters) to use when a kernel dispatch
 * is received.
 */
std::unordered_map<uint64_t, rocprofiler_profile_config_id_t>&
get_profile_cache()
{
    static std::unordered_map<uint64_t, rocprofiler_profile_config_id_t> profile_cache;
    return profile_cache;
}

/**
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA queue.
 * rocprofiler_profile_config_id_t* is a return to specify what counters to collect
 * for this dispatch (dispatch_packet). This example function creates a profile
 * to collect the counter SQ_WAVES for all kernel dispatch packets.
 */
void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_profile_config_id_t*             config,
                  rocprofiler_user_data_t* /*user_data*/,
                  void* /*callback_data_args*/)
{
    /**
     * This simple example uses the same profile counter set for all agents.
     * We store this in a cache to prevent constructing many identical profile counter
     * sets.
     */
    auto search_cache = [&]() {
        if(auto pos = get_profile_cache().find(dispatch_data.dispatch_info.agent_id.handle);
           pos != get_profile_cache().end())
        {
            *config = pos->second;
            return true;
        }
        return false;
    };

    if(!search_cache())
    {
        std::cerr << "No profile for agent found in cache\n";
        exit(-1);
    }
}

/**
 * Construct a profile config for an agent. This function takes an agent (obtained from
 * get_gpu_device_agents()) and a set of counter names to collect. It returns a profile
 * that can be used when a dispatch is received for the agent to collect the specified
 * counters. Note: while you can dynamically create these profiles, it is more efficient
 * to consturct them once in advance (i.e. in tool_init()) since there are non-trivial
 * costs associated with constructing the profile.
 */
rocprofiler_profile_config_id_t
build_profile_for_agent(rocprofiler_agent_id_t       agent,
                        const std::set<std::string>& counters_to_collect)
{
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate all the counters on the agent and store them in gpu_counters.
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
                         agent,
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

    // Find the counters we actually want to collect (i.e. those in counters_to_collect)
    std::vector<rocprofiler_counter_id_t> collect_counters;
    for(auto& counter : gpu_counters)
    {
        rocprofiler_counter_info_v0_t version;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&version)),
            "Could not query info for counter");
        if(counters_to_collect.count(std::string(version.name)) > 0)
        {
            //std::clog << "Counter: " << counter.handle << " " << version.name << "\n";
            collect_counters.push_back(counter);
            used_counter_id_map[counter.handle]=version.name;
        }
    }

    // Create and return the profile
    rocprofiler_profile_config_id_t profile;
    ROCPROFILER_CALL(rocprofiler_create_profile_config(
                         agent, collect_counters.data(), collect_counters.size(), &profile),
                     "Could not construct profile cfg");

    return profile;
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


// PC sampling

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    std::stringstream ss;
    ss << "The number of delivered samples is: " << num_headers << ", "
       << "while the number of dropped samples is: " << drop_count << std::endl;
/*
    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(cur_header->hash !=
                rocprofiler_record_header_compute_hash(cur_header->category, cur_header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);

                /*ss << "(code_obj_id, offset): (" << pc_sample->pc.loaded_code_object_id << ", 0x"
                   << std::hex << pc_sample->pc.loaded_code_object_offset << "), "
                   << "timestamp: " << std::dec << pc_sample->timestamp << ", "
                   << "exec: " << std::hex << std::setw(16) << pc_sample->exec_mask << ", "
                   << "workgroup_id_(x=" << std::dec << std::setw(5) << pc_sample->workgroup_id.x
                   << ", "
                   << "y=" << std::setw(5) << pc_sample->workgroup_id.y << ", "
                   << "z=" << std::setw(5) << pc_sample->workgroup_id.z << "), "
                   << "wave_id: " << std::setw(2) << static_cast<unsigned int>(pc_sample->wave_id)
                   << ", "
                   << "chiplet: " << std::setw(2) << static_cast<unsigned int>(pc_sample->chiplet)
                   << ", "
                   << "cu_id: " << pc_sample->hw_id << ", "
                   << "correlation: {internal=" << std::setw(7)
                   << pc_sample->correlation_id.internal << ", "
                   << "external=" << std::setw(5) << pc_sample->correlation_id.external.value << "}"
                   << std::endl;
				*/  
				/*ss << "pc: " << pc_sample->pc << ", "
                   << "timestamp: " << std::dec << pc_sample->timestamp << ", "
                   << "exec: " << std::hex << std::setw(16) << pc_sample->exec_mask << ", "
                   << "workgroup_id_(x=" << std::dec << std::setw(5) << pc_sample->workgroup_id.x
                   << ", "
                   << "y=" << std::setw(5) << pc_sample->workgroup_id.y << ", "
                   << "z=" << std::setw(5) << pc_sample->workgroup_id.z << "), "
                   << "wave_id: " << std::setw(2) << static_cast<unsigned int>(pc_sample->wave_id)
                   << ", "
                   << "chiplet: " << std::setw(2) << static_cast<unsigned int>(pc_sample->chiplet)
                   << ", "
                   << "cu_id: " << pc_sample->hw_id << ", "
                   << "correlation: {internal=" << std::setw(7)
                   << pc_sample->correlation_id.internal << ", "
                   << "external=" << std::setw(5) << pc_sample->correlation_id.external.value << "}"
                   << std::endl;
            }
            else
            {
                if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_NONE)
					std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_NONE" <<std::endl;
				if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_LAST)
					std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_LAST" <<std::endl;
				if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_CODE_OBJECT_LOAD_MARKER)
					std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_CODE_OBJECT_LOAD_MARKER" <<std::endl;
				if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_CODE_OBJECT_UNLOAD_MARKER)
					std::cout << "ROCPROFILER_PC_SAMPLING_RECORD_CODE_OBJECT_UNLOAD_MARKER" <<std::endl;
				
            }
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }

    std::cout << ss.str() << std::endl;*/
}

/**
 * @brief The function queries available PC sampling configurations.
 * If there is at least one available configuration, it returns true.
 * Otherwise, this function returns false to indicate the agent does
 * not support PC sampling.
 */
int query_avail_configs_for_agent(tool_agent_info* agent_info)
{
    // Clear the available configurations vector
    agent_info->avail_configs->clear();

    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs = static_cast<avail_configs_vec_t*>(user_data);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto status = rocprofiler_query_pc_sampling_agent_configurations(
        agent_info->agent_id, cb, agent_info->avail_configs.get());

    std::stringstream ss;

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        // The query operation failed, so consider the PC sampling is unsupported at the agent.
        // This can happen if the PC sampling service is invoked within the ROCgdb.
        ss << "Querying PC sampling capabilities failed with status=" << status
           << " :: " << rocprofiler_get_status_string(status) << std::endl;
        std::cout  << ss.str() << std::endl;
        return false;
    }
    else if(agent_info->avail_configs->empty())
    {
        // No available configuration at the moment, so mark the PC sampling as unsupported.
        return false;
    }

    ss << "The agent with the id: " << agent_info->agent_id.handle << " supports the "
       << agent_info->avail_configs->size() << " configurations: " << std::endl;
    size_t ind = 0;
    for(auto& cfg : *agent_info->avail_configs)
    {
        ss << "(" << ++ind << ".) "
           << "method: " << cfg.method << ", "
           << "unit: " << cfg.unit << ", "
           << "min_interval: " << cfg.min_interval << ", "
           << "max_interval: " << cfg.max_interval << ", "
           << "flags: " << std::hex << cfg.flags << std::dec << std::endl;
    }

    std::cout << ss.str() << std::flush;

    return true;
}

void
configure_pc_sampling_prefer_stochastic(tool_agent_info*         agent_info,
                                        rocprofiler_context_id_t context_id,
                                        rocprofiler_buffer_id_t  buffer_id)
{
    int    failures = 10;
    size_t interval = 0;
    do
    {
        // Update the list of available configurations
        auto success = query_avail_configs_for_agent(agent_info);
        if(!success)
        {
            // An error occured while querying PC sampling capabilities,
            // so avoid trying configuring PC sampling service.
            // Instead return false to indicated a failure.
            ROCPROFILER_CALL(ROCPROFILER_STATUS_ERROR, "could not configure pc sampling");
        }

        const rocprofiler_pc_sampling_configuration_t* first_host_trap_config  = nullptr;
        const rocprofiler_pc_sampling_configuration_t* first_stochastic_config = nullptr;
        // Search until encountering on the stochastic configuration, if any.
        // Otherwise, use the host trap config
        for(auto const& cfg : *agent_info->avail_configs)
        {
            if(cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
            {
                first_stochastic_config = &cfg;
                break;
            }
            else if(!first_host_trap_config &&
                    cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
            {
                first_host_trap_config = &cfg;
            }
        }

        // Check if the stochastic config is found. Use host trap config otherwise.
        const rocprofiler_pc_sampling_configuration_t* picked_cfg =
            (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;

        if(picked_cfg->min_interval == picked_cfg->max_interval)
        {
            // Another process already configured PC sampling, so use the intreval it set up.
            interval = picked_cfg->min_interval;
        }
        else
        {
            interval = 10000;
        }

        auto status = rocprofiler_configure_pc_sampling_service(context_id,
                                                                agent_info->agent_id,
                                                                picked_cfg->method,
                                                                picked_cfg->unit,
                                                                interval,
                                                                buffer_id);
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            std::cout
                << ">>> We chose PC sampling interval: " << interval
                << " on the agent: " << agent_info->agent->id.handle << std::endl;
            return;
        }
        else if(status != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        {
            ROCPROFILER_CALL(status, " pc sampling not available, may be in use");
        }
        // status ==  ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE
        // means another process P2 already configured PC sampling.
        // Query available configurations again and receive the configurations picked by P2.
        // However, if P2 destroys PC sampling service after query function finished,
        // but before the `rocprofiler_configure_pc_sampling_service` is called,
        // then the `rocprofiler_configure_pc_sampling_service` will fail again.
        // The process P1 executing this loop can spin wait (starve) if it is unlucky enough
        // to always be interuppted by some other process P2 that creates/destroys
        // PC sampling service on the same device while P1 is executing the code
        // after the `query_avail_configs_for_agent` and
        // before the `rocprofiler_configure_pc_sampling_service`.
        // This should happen very rarely, but just to be sure, we introduce a counter `failures`
        // that will allow certain amount of failures to process P1.
    } while(--failures);

    // The process failed too many times configuring PC sampling,
    // report this to user;
    ROCPROFILER_CALL(ROCPROFILER_STATUS_ERROR, "failed too many times configuring PC sampling");
}


rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(rocprofiler_agent_version_t version,
                                                const void**                agents,
                                                size_t                      num_agents,
                                                void*                       user_data)
{
    assert(version == ROCPROFILER_AGENT_INFO_VERSION_0);
    // user_data represent the pointer to the array where gpu_agent will be stored
    if(!user_data) return ROCPROFILER_STATUS_ERROR;

    std::stringstream ss;

    auto* _out_agents = static_cast<tool_agent_info_vec_t*>(user_data);
    auto* _agents     = reinterpret_cast<const rocprofiler_agent_t**>(agents);
    for(size_t i = 0; i < num_agents; i++)
    {
        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            // Instantiate the tool_agent_info.
            // Store pointer to the rocprofiler_agent_t and instatiate a vector of
            // available configurations.
            // Move the ownership to the _out_agents
            auto tool_gpu_agent           = std::make_unique<tool_agent_info>();
            tool_gpu_agent->agent_id      = _agents[i]->id;
            tool_gpu_agent->avail_configs = std::make_unique<avail_configs_vec_t>();
            tool_gpu_agent->agent         = _agents[i];
            // Check if the GPU agent supports PC sampling. If so, add it to the
            // output list `_out_agents`.
            if(query_avail_configs_for_agent(tool_gpu_agent.get()))
                _out_agents->push_back(std::move(tool_gpu_agent));
        }

        ss << "[" << __FUNCTION__ << "] " << _agents[i]->name << " :: "
           << "id=" << _agents[i]->id.handle << ", "
           << "type=" << _agents[i]->type << "\n";
    }

    std::cout << ss.str() << std::endl;

    return ROCPROFILER_STATUS_SUCCESS;
}

int enable_pc_sampling ()
{
	const char* env_pc_sampling=std::getenv("ROCPROFILER_PC_SAMPLING_BETA_ENABLED");

	if(env_pc_sampling)
	{
		int len = strlen(env_pc_sampling);
		if (len == 0)
			return 1;
		if(atoi(env_pc_sampling)==1)
			return 1;
	}
	return 0;
}

//PC Sampling...end

//Modified from TauMetrics.cpp
//This is executed before TAU is initialized
template <typename T> profile_metrics get_set_metrics(const char* rocm_metrics, T agents)
{
  
	const char *token;
	char *ptr, *ptr2;
	int len = strlen(rocm_metrics);
	int i;
	bool alt_delimiter_found = false;
	std::set<std::string> counter_set;
  
	if (len == 0)
		return NO_METRICS;
  

	char *metrics = strdup(rocm_metrics);
	for (i = 0; i < len; i++) {
	  if ((rocm_metrics[i] == ',') || (rocm_metrics[i] == '|')) {
		  alt_delimiter_found = true;
		  //printf("ALT delimiter found: rocm_metrics[%d] = %c\n", i, rocm_metrics[i]);
		  break;
	  }
	}
	for (ptr = metrics; *ptr; ptr++) {
	  if (*ptr == '\\') {
		  /* escaped, skip over */
		  for (ptr2 = ptr; *(ptr2); ptr2++) {
			  *ptr2 = *(ptr2 + 1);
		  }
		  ptr++;
	  } else {
		  if (alt_delimiter_found) {
			  //printf("Alt_delimiter = %d, ptr = %c\n", alt_delimiter_found, *ptr);
			  if ((*ptr == '|') || (*ptr == ',')) {
				  // printf("Checking for | or , in %s\n", metrics);
				  *ptr = '^';
			  }
		  } else {
			  // printf("Alt_delimiter = %d, ptr = %c\n", alt_delimiter_found, *ptr);
			  if (*ptr == ':') {
				  // printf("Checking for : in %s\n", metrics);
				  *ptr = '^';
			  }
		  }
	  }
	}

	token = strtok(metrics, "^");
	while (token) {
	//std::cout << token << std::endl;
	counter_set.insert(token);
	token = strtok(NULL, "^");

	}

	
	for(const auto& agent : agents)
	{
		  // get_profile_cache() is a map that can be accessed by dispatch_callback
		  // below to select the profile config to use when a kernel dispatch is
		  // recieved.
		  get_profile_cache().emplace(
		  agent.id.handle, build_profile_for_agent(agent.id, counter_set));
	}

	//std::cout << "Map size " << used_counter_id_map.size() << " Counter_set size " << counter_set.size() << std::endl;
	if(used_counter_id_map.size() != counter_set.size())
		return WRONG_NAME;



	return PROFILE_METRICS;

}


template <typename T> profile_metrics set_rocm_metrics(T agents)
{
  std::string delimiter = ":";
  profile_metrics return_value=NO_METRICS;
  const char* rocm_metrics=std::getenv("ROCM_METRICS");
  if( rocm_metrics )
  {
    //std::cout << "ROCM_METRICS=" << rocm_metrics << std::endl;
    return_value=get_set_metrics(rocm_metrics, agents);
  }
  
  return return_value;
}


int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
	std::cout << "tool_init" << std::endl;
    //printf("----------- %s\n", __FUNCTION__);
    assert(tool_data != nullptr);
	
	//Check if there are any ROCm GPUs available
    auto agents = get_gpu_device_agents();
	if(agents.empty())
    {
        std::cerr << "No ROCm GPUs found" << std::endl;
        return 1;
    }
	
	//Try to set the profiling for the metrics in ROCM_METRICS environmental variable
	int flag_metrics_set = set_rocm_metrics(agents); 
	if(flag_metrics_set == WRONG_NAME)
	{
		std::cout << "ERROR!!: THE NUMBER OF REQUESTED COUNTERS DOES NOT MATCH THE PROFILED COUNTERS \n" 
			<< " CHECK THAT THE NAME OF THE HARDWARE COUNTERS IS CORRECT OR IS AVAILABLE"<<std::endl;
		std::cout << " METRIC PROFILING DISABLED TO AVOID PROFILING ERRORS" << std::endl;
	
	}
	
	

    client_name_info = get_buffer_tracing_names();

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");
	
	auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       code_object_ops.data(),
                                                       code_object_ops.size(),
                                                       tool_code_object_callback,
                                                       nullptr),
        "code object tracing service configure");

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

	//HSA events
    for(auto itr : {ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                    ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API})
    {
        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                             client_ctx, itr, nullptr, 0, client_buffer),
                         "buffer tracing service configure");
    }

	//HIP events
    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, client_buffer),
        "buffer tracing service configure");

	//Kernel dispatch events
    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer),
        "buffer tracing service for kernel dispatch configure");

	//Memory events
    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr, 0, client_buffer),
        "buffer tracing service for memory copy configure");
		
	// May have incompatible kernel so only emit a warning here
	//NOT IMPLEMENTED
	/*
    ROCPROFILER_WARN(rocprofiler_configure_buffer_tracing_service(
        client_ctx, ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION, nullptr, 0, client_buffer));

	//NOT IMPLEMENTED
    ROCPROFILER_CALL(
        rocprofiler_configure_buffer_tracing_service(
            client_ctx, ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY, nullptr, 0, client_buffer),
        "buffer tracing service for page migration configure");*/
		
	//ROCTX events	
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
	//END ROCTX events

    auto client_thread = rocprofiler_callback_thread_t{};
    ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_thread),
                     "creating callback thread");

    ROCPROFILER_CALL(rocprofiler_assign_callback_thread(client_buffer, client_thread),
                     "assignment of thread for buffer");
					 
					 
					 
					 
					 
					 
	//Setting PC sampling
	
	if(enable_pc_sampling())
	{
		std::cout << "Enabling PC sampling..." << std::endl;
		pc_buffer_ids = new pc_sampling_buffer_id_vec_t();
		
		tool_agent_info_vec_t pc_gpu_agents;
		
		ROCPROFILER_CALL(
			rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
											   &find_all_gpu_agents_supporting_pc_sampling_impl,
											   sizeof(rocprofiler_agent_t),
											   static_cast<void*>(&pc_gpu_agents)),
							"query available gpus with pc sampling");
		
		if(pc_gpu_agents.empty())
		{
			std::cout << "No availabe gpu agents supporting PC sampling" << std::endl;
		}
		else
		{
			for(auto& gpu_agent : pc_gpu_agents)
			{
				// creating a buffer that will hold pc sampling information
				rocprofiler_buffer_policy_t drop_buffer_action = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
				auto                        buffer_id          = rocprofiler_buffer_id_t{};
				ROCPROFILER_CALL(rocprofiler_create_buffer(client_ctx,
															BUFFER_SIZE_BYTES,
															WATERMARK,
															drop_buffer_action,
															rocprofiler_pc_sampling_callback,
															nullptr,
															&buffer_id),
									"buffer for agent in pc sampling");

				configure_pc_sampling_prefer_stochastic(
					gpu_agent.get(), client_ctx, buffer_id);

				// One helper thread per GPU agent's buffer.
				auto client_agent_thread = rocprofiler_callback_thread_t{};
				ROCPROFILER_CALL(rocprofiler_create_callback_thread(&client_agent_thread),
									"create callback thread for pc sampling");

				ROCPROFILER_CALL(rocprofiler_assign_callback_thread(buffer_id, client_agent_thread),
									"assign callback thread for pc sampling");

				pc_buffer_ids->emplace_back(buffer_id);
			}
		}
		
	}
	
	
	// End of setting PC sampling		 

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
	
	
	
	
	if( flag_metrics_set == PROFILE_METRICS)
	{
		 // Setup the dispatch profile counting service. This service will trigger the dispatch_callback
		// when a kernel dispatch is enqueued into the HSA queue. The callback will specify what
		// counters to collect by returning a profile config id. In this example, we create the profile
		// configs above and store them in the map get_profile_cache() so we can look them up at
		// dispatch.
		ROCPROFILER_CALL(rocprofiler_configure_buffered_dispatch_counting_service(
							 client_ctx, client_buffer, dispatch_callback, nullptr),
						 "Could not setup buffered service");
	}
	
	
	

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start");
		
	std::cout << "END tool_init " << flag_metrics_set <<  std::endl;
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
