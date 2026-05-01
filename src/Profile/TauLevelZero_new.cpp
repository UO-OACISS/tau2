#include <assert.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <sys/wait.h>

#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>

#include "Profile/Profiler.h"
#include "Profile/TauBfd.h"

#include "Profile/L0_new/level_zero/ze_api.h"
#include "Profile/L0_new/level_zero/zes_api.h"
#include "Profile/L0_new/level_zero/zet_api.h"

#include "Profile/L0_new/level_zero/layers/zel_tracing_api.h"
#include "Profile/L0_new/level_zero/layers/zel_tracing_register_cb.h"
#include "Profile/L0_new/common_header.h"
#include "Profile/L0_new/unimemory.h"

//#define PTI_ASSERT(X) assert(X)



//#define L0_TAU_DEBUG
#undef L0_TAU_DEBUG

#ifdef L0_TAU_DEBUG
#define L0_TAU_DEBUG_MSG(d_msg) std::cout << "\n[!!] " <<  d_msg << std::endl
#else
#define L0_TAU_DEBUG_MSG(d_msg) 
#endif

extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" x_uint64 TauTraceGetTimeStamp(int tid);
extern "C" void Tau_metadata_task(const char *name, const char *value, int tid);

//Disabled options, except Metric_Query and Stall_sampling,
// can help debug if changed to true, do not remove their code
// from ze_collector.h
struct CollectorOptions {
  bool device_timing = true;
  bool device_timeline = false;
  bool kernel_submission = true;
  bool host_timing = true;
  bool kernel_tracing = false;
  bool api_tracing = false;
  bool call_logging = false;
  bool need_tid = true;
  bool need_pid = true;
  bool verbose = true;
  bool demangle = true;
  bool kernels_per_tile = true;
  bool metric_query = false;
  bool metric_stream = false;
  bool stall_sampling = false;
};

//std::string my_log ;
//Logger *logger_ = nullptr;


#include "Profile/L0_new/ze_collector.h"
#include "Profile/L0_new/ze_metrics.h"

static double L0_TAU_init_timestamp;
static uint64_t L0_Driver_init_timestamp;
static uint64_t L0_Driver_init_timestamp1;
static int initialized = 0;
static int disabled = 0;
static CollectorOptions L0_collector_options;

//Map a device and tile to a task
static std::map<tuple<uintptr_t, int, size_t>, int> map_thread_queue;

static std::map<ze_command_queue_handle_t, int> command_queue_map;
static int comm_queue = 0;

ZeCollector* ze_collector_ = nullptr;
static ZeMetricProfiler* metric_profiler = nullptr;

OnZeKernelFinishCallback L0_k_callback = nullptr;
OnZeFunctionFinishCallback L0_a_callback = nullptr;

CollectorOptions init_collector_options()
{
    CollectorOptions init_options;
    TAU_VERBOSE("Initializing collector options\n");
    #ifdef L0METRICS
    if(TauEnv_get_l0_metrics_enable() && !TauEnv_get_l0_stall_sampling_enable())
    {
        if(strcmp("EuStallSampling", utils::GetEnv("L0_METRICGROUP").c_str())==0)
        {
            printf("Error: EuStallSampling cannot be used as a metric\n");
            return init_options;
        }
        TAU_VERBOSE("L0 metrics enabled\n");
        init_options.metric_query = true;
    }
    else if(TauEnv_get_l0_metrics_enable() && TauEnv_get_l0_stall_sampling_enable())//EuStallSampling
    {
        if(strcmp("EuStallSampling", utils::GetEnv("L0_METRICGROUP").c_str())==0)
        {
            //return init_options;
            init_options.stall_sampling = true;
            init_options.metric_stream = true;
            //Check if metrics requested, if requested, throw error, as not compatible 
            init_options.metric_query = false;

        }
        else
        {
            printf("Error: L0 cannot enable both Metric Profiling and Sampling\n");
            return init_options;
        }
    }
    else
    {
        TAU_VERBOSE("No L0 metrics requested\n");
    }

    #endif
    
    
    return init_options;
}

void Tau_add_metadata_for_task(const char *key, int value, int taskid) {
  char buf[1024];
  snprintf(buf, sizeof(buf),  "%d", value);
  Tau_metadata_task(key, buf, taskid);
  TAU_VERBOSE("Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
}

//Only call inside functions that use lock_guard, not implemented lock inside to prevent deadlocks.
//Check implementation for metrics
int Tau_get_initialized_queues(tuple<uintptr_t, int, size_t> dev_tile, ze_command_queue_handle_t comm_queue_id, double first_ts)
{
  int queue_id;
  auto it = map_thread_queue.find(dev_tile);
  if(it !=map_thread_queue.end())
  {
    queue_id = it->second;
  }
  else
  {

    std::string cur_name;
    uint32_t cur_devid;
    //ze_device_properties_t props = { ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES };
    //zeDeviceGetProperties(reinterpret_cast<ze_device_handle_t>(std::get<0>(dev_tile)), &props);
    auto it = devices_->find(reinterpret_cast<ze_device_handle_t>(std::get<0>(dev_tile)));
    if (it != devices_->end())
    {
        ZeDevice device_info = it->second;
        cur_name = device_info.device_name_;
        cur_devid = device_info.id_;
    }
    //Should not execute this part, GPUs should be registered.
    else
    {
        printf("[TAU] failed to find GPU for L0 sample.\n");
        return 0;
    }

    //printf("Running on: %s , deviceId %d uuid %d\n", props.name, props.deviceId, props.uuid);

    TAU_CREATE_TASK(queue_id);
    // losing resolution from nanoseconds to microseconds.
    metric_set_gpu_timestamp(queue_id, first_ts);
    Tau_add_metadata_for_task("TAU_TASK_ID", queue_id, queue_id);
    Tau_add_metadata_for_task("L0_GPU_ID", cur_devid, queue_id);
    //Similar to streams
    //Tau_add_metadata_for_task("L0_GPU_TILE", std::get<1>(dev_tile), queue_id);
    Tau_metadata_task("L0_GPU_NAME", cur_name.c_str(), queue_id);
    
    auto it_comm_q = command_queue_map.find(comm_queue_id);
    int curr_comm_queue = comm_queue;
    if(it_comm_q !=command_queue_map.end())
    {
        curr_comm_queue = it_comm_q->second;
    }
    else
    {
        command_queue_map[comm_queue_id] = comm_queue;
    } 
    Tau_add_metadata_for_task("L0_QUEUE_ID", curr_comm_queue, queue_id);
    Tau_add_metadata_for_task("L0_VQUEUE_ID", std::get<2>(dev_tile), queue_id);
    comm_queue++;
    Tau_create_top_level_timer_if_necessary_task(queue_id);
    //std::cout << " NEW TASK: " << queue_id << std::endl;
    map_thread_queue[dev_tile] = queue_id;
  }
  return queue_id;
}


static std::mutex queue_mutex;
/* This code is to somehow link the the kernel from the CPU to the GPU callback.
  Intel doesn't seem to provide this info. So, when a kernel is pushed onto the
  command queue, we'll push a unique id onto a local queue. When we are notified
  that the kernel finished, we'll pop it. This dangerously assumes there is only
  one command queue. */

std::queue<uint64_t>& getKernelQueue() {
    static std::queue<uint64_t> theQueue;
    return theQueue;
}

uint64_t pushKernel() {
    static uint64_t id{0};
    std::lock_guard<std::mutex> lck(queue_mutex);
    id = id + 1;
    getKernelQueue().push(id);
    //printf("Pushed %lu\n", id);
    return id;
}

uint64_t popKernel() {
    uint64_t id{0};
    auto& theQueue = getKernelQueue();
    std::lock_guard<std::mutex> lck(queue_mutex);
    if (theQueue.size() > 0) {
        id = theQueue.front();
        theQueue.pop();
    }
    //printf("Popped %lu\n", id);
    return id;
}


//Only call inside functions that use lock_guard, not implemented lock inside to prevent deadlocks.
void Tau_remove_initialized_queues(uint64_t cpu_end_ts)
{
  std::map<tuple<uintptr_t, int, size_t>, int>::iterator it;
  for(it = map_thread_queue.begin(); it != map_thread_queue.end(); it++)
  {
    int taskid = it->second;
    metric_set_gpu_timestamp(taskid, cpu_end_ts);
    Tau_stop_top_level_timer_if_necessary_task(taskid);    
  }
  map_thread_queue.clear();
}

//Should not need tasks, as the thread executing the call, executes this callback.
//Need to test with threads
//API calls
void TAU_L0_enter_event(const char* nameAPIcall)
{
    //This function is called by profilers, we do not want to profile it,
    // or we will have overlaps when using stall sampling
    //may disable the callback
    if(strcmp(nameAPIcall, "zeEventQueryStatus") == 0)
        return;
    L0_TAU_DEBUG_MSG("TAU_L0_enter_event!!\n");
    if(!initialized || disabled)
    {
        L0_TAU_DEBUG_MSG("TAU_L0_enter_event !initialized || disabled !!\n");
        return;
    }
    int current_thread = RtsLayer::myThread();
    uint64_t current_timestamp = TauTraceGetTimeStamp(0);
    L0_TAU_DEBUG_MSG("TAU_L0_enter_event " << nameAPIcall << " thread " << current_thread << " ts " <<  current_timestamp);
    TAU_START(nameAPIcall);

    //Only Correlate when kernels are involved.
    static std::string launch_name = "zeCommandListAppendLaunchKernel";
    static std::string launch_name1 = "zeCommandListAppendLaunchCooperativeKernel";
    static std::string launch_name2 = "zeCommandListAppendLaunchKernelIndirect";
    
    //This is a bit dumb, but I plan to change this part. However, the change required a lot of
    // changes. Will do them after the TAU release as I have to do a Python script to update the
    // callbacks for newer L0 versions and change hundreds of callbacks.
    if((launch_name.compare(nameAPIcall) == 0) || (launch_name1.compare(nameAPIcall) == 0)  || (launch_name2.compare(nameAPIcall) == 0) ) 
    {
        static void* TraceCorrelationID;
        Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
        current_timestamp = TauTraceGetTimeStamp(0);
        TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, pushKernel(), current_thread, current_timestamp);
    }


    // May need to be added in the future ? They don't seem to be used at this moment
    // zeCommandListAppendLaunchKernelWithParameters
    // zeCommandListAppendLaunchKernelWithArguments

    //Will be used in the future update, will also work for memory operations
    /*
    if (corr_id_ != 0) {
        //printf("!! Launch\n");
        // the user event for correlation IDs
        static void* TraceCorrelationID;
        //printf("!! TraceCorrelationID\n");
        Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
        //printf("!! Tau_get_context_userevent\n");
        TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, corr_id_, current_thread, current_timestamp);
        //printf("!! TAU_CONTEXT_EVENT_THREAD\n");
    }*/
}

//Should not need tasks, as the thread executing the call, executes this callback.
//Need to test with threads
//API calls
void TAU_L0_exit_event(const char* nameAPIcall)
{
    //This function is called by profilers, we do not want to profile it,
    // or we will have overlaps when using stall sampling
    if(strcmp(nameAPIcall, "zeEventQueryStatus") == 0)
        return;
    L0_TAU_DEBUG_MSG("TAU_L0_exit_event!!\n");
    if(!initialized || disabled)
    {
        L0_TAU_DEBUG_MSG("TAU_L0_kernel_event !initialized || disabled !!\n");
        return;
    }
    int current_thread = RtsLayer::myThread();
    uint64_t current_timestamp = TauTraceGetTimeStamp(0);
    L0_TAU_DEBUG_MSG("TAU_L0_exit_event " << nameAPIcall << " thread " << current_thread << " ts " <<  current_timestamp);
    TAU_STOP(nameAPIcall);
}

zet_metric_group_handle_t TAU_L0_get_metric_group(ze_device_handle_t curr_device_handle)
{
    auto it2 = devices_->find(curr_device_handle);
    if (it2 == devices_->end()) 
    {
        // should never get here
        return NULL;
    }
    return it2->second.metric_group_;
}

std::vector<std::string> TAU_L0_get_metric_names(zet_metric_group_handle_t metric_group)
{
    std::vector<std::string> metric_names = ze_collector_->reportMetricNames(metric_group);
    assert(metric_names.size() != 0);

    /*for(auto it_metric_names = metric_names.begin(); it_metric_names != metric_names.end(); it_metric_names++)
    {
        std::cout << "[!!] " << *it_metric_names << std::endl;
    }*/
    return metric_names;
}

double TAU_LO_translate_metric_value(const zet_typed_value_t& typed_value)
{
    double translated_value = 0;
    switch (typed_value.type) {
      case ZET_VALUE_TYPE_UINT32:
        return static_cast<double>(typed_value.value.ui32);
      case ZET_VALUE_TYPE_UINT64:
        return static_cast<double>(typed_value.value.ui64);
      case ZET_VALUE_TYPE_FLOAT32:
        return static_cast<double>(typed_value.value.fp32);
      case ZET_VALUE_TYPE_FLOAT64:
        return static_cast<double>(typed_value.value.fp64);
      case ZET_VALUE_TYPE_BOOL8:
        return static_cast<double>(typed_value.value.b8);
      default:
        PTI_ASSERT(0);
        break;
    }
    return translated_value;
}


//There are overlaps when using examples such as ze_gemm due to how
// command lists work. A list of command to execute is created and command
// that are inside a region before a barrier can be executed concurrently,
// which happens with ze_gemm and the memory copies. As we cannot detect the barrier
// with the same logic as the command list, we check for overlaps, and use a virtual
// queue id identifier to simulate that we have multiple queues
size_t check_overlap(uintptr_t curr_device, int curr_tile, uint64_t kernel_start, uint64_t kernel_end)
{
    //Map keys are current device and current tile
    //Inside each position, we want to have the start and end timestamps
    static std::map<tuple<uintptr_t, int>, vector<tuple<uint64_t, uint64_t>>> map_overlaps;
    auto key = make_tuple(curr_device, curr_tile);
    auto& vec = map_overlaps[key]; // creates empty vector if key does not exist

    for (size_t i = 0; i < vec.size(); ++i) {
        auto& t = vec[i];
        uint64_t prev_start = get<0>(t);
        uint64_t prev_end   = get<1>(t);

        if (kernel_start >= prev_end) {
            t = make_tuple(kernel_start, kernel_end);
            return i;
        }
    }

    // All tuples overlap → create a new tuple (new virtual queue)
    vec.push_back(make_tuple(kernel_start, kernel_end));
    return vec.size() - 1; // last index = new queue

    return 0;
}

std::string TAU_L0_demangle(std::string original_name)
{
    //OMP Offloaded function names appear as a string with
    // multiple information fields, parse them
    //May change if the string changes in the future
    static std::string omp_off_string = "__omp_offloading";
    std::string event_name = "[L0] GPU: ";

    if( strncmp(original_name.c_str(), omp_off_string.c_str(), omp_off_string.length())==0)
    {
    /*
        __omp_offloading_3d_2c4a55__Z14compute_target_l105
        __omp_offloading      :  standard prefex
        3d                   : DeviceID
        2c4a55               : FileID
        _Z14compute_target   : Mangled function name.  Use C++filt
        L105                 :  line number in the file.  Line-105
    */

        int pos_key=omp_off_string.length();
        for(int i =0; i<3; i++)
        {
            pos_key = original_name.find_first_of('_', pos_key + 1);
        }
        event_name = event_name + "OMP OFFLOADING ";
        event_name = event_name + Tau_demangle_name(original_name.substr(pos_key,original_name.find_last_of("l")-pos_key-1).c_str());
        event_name = event_name + " [{UNRESOLVED} {";
        event_name = event_name + original_name.substr(original_name.find_last_of("l")+1);
        event_name = event_name + ",0}]";
    }
    else
    {
        event_name = event_name + Tau_demangle_name(original_name.c_str());
    }
    return event_name ;
}

std::string TAU_L0_demangle_sampling(std::string original_name, std::string file_name, uint64_t inst_line, std::string inst_text)
{
    //Use the line and file provided by tau_map_l0_source_info and tau_map_l0_inst_info
    //OMP Offloaded function names appear as a string with
    // multiple information fields, parse them
    //May change if the string changes in the future
    static std::string omp_off_string = "__omp_offloading";
    std::string event_name = "[L0] GPU: ";

    if( strncmp(original_name.c_str(), omp_off_string.c_str(), omp_off_string.length())==0)
    {
    /*
        __omp_offloading_3d_2c4a55__Z14compute_target_l105
        __omp_offloading      :  standard prefex
        3d                   : DeviceID
        2c4a55               : FileID
        _Z14compute_target   : Mangled function name.  Use C++filt
        L105                 :  line number in the file.  Line-105
    */

        int pos_key=omp_off_string.length();
        for(int i =0; i<3; i++)
        {
            pos_key = original_name.find_first_of('_', pos_key + 1);
        }
        event_name = event_name + "OMP OFFLOADING ";
        event_name = event_name + Tau_demangle_name(original_name.substr(pos_key,original_name.find_last_of("l")-pos_key-1).c_str());
        event_name = event_name + " [{";
        event_name = event_name + file_name;
        event_name = event_name + "} {";
        event_name += (inst_line == -1)? original_name.substr(original_name.find_last_of("l")+1) : std::to_string(inst_line);
        event_name += ",0}] ";
        event_name += "{" + inst_text + "} ";
    }
    else
    {
        event_name = event_name + Tau_demangle_name(original_name.c_str());
        event_name = event_name + " [{";
        event_name = event_name + file_name;
        event_name = event_name + "} {";
        event_name += (inst_line == -1)? std::to_string(0) : std::to_string(inst_line);
        event_name += ",0}] ";
        event_name += "{" + inst_text + "} ";
    }
    return event_name ;
}

//GPU events, needs a task per device and tile(concurrent kernels)
// need a map with device and tile to task_id
//Also, there are some things detected as Kernel which really aren't
// check that it is really a kernel, or try to solve it inside ze_collector.h
// checking when events that are not kernels are added as kernels
// as they may be executed concurrently. According to Intel, those events
// can be ignore and will dissapear with newer drivers, so we will ignore them.
// Another way to solve it without ignoring them is adding them to another thread.
// Will check both options, but to avoid issues, they will be ignored at this moment, 
// as they also appear as API calls, that are already measured. device_command_names
void TAU_L0_kernel_event(const ZeCommand *command, uint64_t kernel_start, uint64_t kernel_end, int tile)
{
    L0_TAU_DEBUG_MSG("TAU_L0_kernel_event!!\n");
    if(!initialized || disabled)
    {
        L0_TAU_DEBUG_MSG("TAU_L0_kernel_event !initialized || disabled !!\n");
        return;
    }

    //Information about kernels are inside kernel_command_properties_, which needs a lock
    // which already exists inside ze_collector.h
    //Check if the kernel exists, if not, ignore.
    kernel_command_properties_mutex_.lock_shared();
    auto it = kernel_command_properties_->find(command->kernel_command_id_);
    if (it != kernel_command_properties_->end()) {

        //There are issues with some callbacks from commands(such as barriers)
        // which are shown concurrently, according to Intel, this issue should not appear 
        // with newer drivers. Therefore, we ignore them as they are already shown in the host thread
        // and do not provide additional information.
        //printf("%d\n", it->second.type_);
        if((it->second.type_ == KERNEL_COMMAND_TYPE_COMMAND) || (it->second.type_ == KERNEL_COMMAND_TYPE_INVALID))
        {
            #ifdef L0_TAU_DEBUG
            std::string event_name = TAU_L0_demangle(it->second.name_.c_str());
            L0_TAU_DEBUG_MSG("DISCARDED -- \n");
            std::cout << "Thread: " << command->tid_ << std::endl;
            std::cout << "Device: " << reinterpret_cast<uintptr_t>(command->device_) << std::endl;
            std::cout << "CommandName: " << event_name << std::endl;
            std::cout << "command->append_time_ " << command->append_time_ << std::endl;
            std::cout << "command->submit_time_ " << command->submit_time_ << std::endl;
            std::cout << "kernel_start " << kernel_start << std::endl;
            std::cout << "kernel_end " << kernel_end << std::endl;
            uint64_t kernel_dutarion = (kernel_end - kernel_start);
            std::cout << "kernel_duration " << kernel_dutarion << std::endl;
            std::cout << "Instance ID " << command->instance_id_  << std::endl;
            std::cout << "tile " << tile  << std::endl;
            std::cout << "queue_ " << command->queue_  << std::endl;
            std::cout << "engine_ordinal_ " << command->engine_ordinal_  << std::endl;
            std::cout << "engine_index_ " << command->engine_index_  << std::endl;
            std::cout << "command_list_ " << command->command_list_  << std::endl;
            #endif
            kernel_command_properties_mutex_.unlock_shared();
            return;
        }

        std::string event_name = TAU_L0_demangle(it->second.name_.c_str());
        

        #ifdef L0_TAU_DEBUG  
            L0_TAU_DEBUG_MSG("TAU_L0_kernel_event -- \n");
            std::cout << "Thread: " << command->tid_ << std::endl;
            std::cout << "Device: " << reinterpret_cast<uintptr_t>(command->device_) << std::endl;
            std::cout << "CommandName: " << event_name << std::endl;
            std::cout << "command->append_time_ " << command->append_time_ << std::endl;
            std::cout << "command->submit_time_ " << command->submit_time_ << std::endl;
            std::cout << "kernel_start " << kernel_start << std::endl;
            std::cout << "kernel_end " << kernel_end << std::endl;
            uint64_t kernel_dutarion = (kernel_end - kernel_start);
            std::cout << "kernel_duration " << kernel_dutarion << std::endl;
            std::cout << "Instance ID " << command->instance_id_  << std::endl;
            std::cout << "tile " << tile  << std::endl;
            std::cout << "queue_ " << command->queue_  << std::endl;
            std::cout << "engine_ordinal_ " << command->engine_ordinal_  << std::endl;
            std::cout << "engine_index_ " << command->engine_index_  << std::endl;
            std::cout << "command_list_ " << command->command_list_  << std::endl;
            //std::cout << "v_comm_list " << command->v_comm_list_  << std::endl;
        #endif

        //This is already inside a mutex
        static int init_first_timer = 0;
        if(!init_first_timer)
        {
            uint64_t device_timestamp;	// in ticks
            ze_result_t status = ZE_FUNC(zeDeviceGetGlobalTimestamps)(command->device_, &L0_Driver_init_timestamp, &device_timestamp);
            L0_TAU_init_timestamp = TauTraceGetTimeStamp(0);
            status = ZE_FUNC(zeDeviceGetGlobalTimestamps)(command->device_, &L0_Driver_init_timestamp1, &device_timestamp);
            PTI_ASSERT(status == ZE_RESULT_SUCCESS);
            init_first_timer = 1;
            L0_Driver_init_timestamp = (L0_Driver_init_timestamp1+L0_Driver_init_timestamp)/2;
        }


        int curr_tile = tile<0? 0:tile;
        uintptr_t curr_device = reinterpret_cast<uintptr_t>(command->device_);

        size_t v_queue_id = check_overlap(curr_device, curr_tile, kernel_start, kernel_end);

        tuple<uintptr_t, int, size_t> dev_tile(curr_device, curr_tile, v_queue_id);

        static double time_shift = L0_TAU_init_timestamp - (L0_Driver_init_timestamp/1e3);
        double translated_start = time_shift + (kernel_start/1e3);
        double translated_end = time_shift + (kernel_end/1e3);

        int task_id = -1;

        if(it->second.type_ == KERNEL_COMMAND_TYPE_COMPUTE)
        {
            L0_TAU_DEBUG_MSG("KERNEL_COMMAND_TYPE_COMPUTE!!\n");
            //std::cout  << "L0_TAU_init_timestamp   " << setprecision(numeric_limits<double>::max_digits10) << L0_TAU_init_timestamp << std::endl;
            //std::cout << "L0_Driver_init_timestamp " << L0_Driver_init_timestamp << std::endl;
            //std::cout << "kernel_start " << kernel_start << std::endl;
            //std::cout << "kernel_end " << kernel_end << std::endl;
            //std::cout << "time_shift " << setprecision(numeric_limits<double>::max_digits10) << time_shift << std::endl;
            //std::cout << "translated_start         " << setprecision(numeric_limits<double>::max_digits10)<< translated_start << std::endl;
            //std::cout << "translated_end           " << setprecision(numeric_limits<double>::max_digits10)<< translated_end << std::endl;
            //std::cout << "k_diff " << kernel_end - kernel_start << std::endl;
            //std::cout << "t_diff " << setprecision(numeric_limits<double>::max_digits10)<<  translated_end - translated_start << std::endl;
        
            task_id = Tau_get_initialized_queues(dev_tile, command->queue_, translated_start);
            metric_set_gpu_timestamp(task_id, translated_start);
            TAU_START_TASK(event_name.c_str(), task_id);
            void* TraceCorrelationID;
            Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
            //TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, command->corr_id_, task_id, translated_start);
            TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, popKernel(), task_id, translated_start);
            metric_set_gpu_timestamp(task_id, translated_end);
            TAU_STOP_TASK(event_name.c_str(), task_id);


            #ifdef L0_TAU_DEBUG  
            std::cout << "group_count.groupCountX " << command->group_count_.groupCountX << std::endl;
            std::cout << "group_count.groupCountY " << command->group_count_.groupCountY << std::endl;
            std::cout << "group_count.groupCountZ " << command->group_count_.groupCountZ << std::endl;
            std::cout << "it->second.group_size_.x " << it->second.group_size_.x << std::endl;
            std::cout << "it->second.group_size_.y " << it->second.group_size_.y << std::endl;
            std::cout << "it->second.group_size_.z " << it->second.group_size_.z << std::endl;
            std::cout << "corr_id_: " << command->corr_id_ << std::endl;
            #endif

            void* ue = nullptr;
            Tau_get_context_userevent(&ue, "Group Count X");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, command->group_count_.groupCountX, task_id, translated_end);
            Tau_get_context_userevent(&ue, "Group Count Y");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, command->group_count_.groupCountY, task_id, translated_end);
            Tau_get_context_userevent(&ue, "Group Count Z");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, command->group_count_.groupCountZ, task_id, translated_end);
            Tau_get_context_userevent(&ue, "Group Size X");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, it->second.group_size_.x, task_id, translated_end);
            Tau_get_context_userevent(&ue, "Group Size Y");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, it->second.group_size_.y, task_id, translated_end);
            Tau_get_context_userevent(&ue, "Group Size Z");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, it->second.group_size_.z, task_id, translated_end);
            
        }
        else if((it->second.type_ == KERNEL_COMMAND_TYPE_MEMORY) && (command->mem_size_ > 0))
        {
            L0_TAU_DEBUG_MSG("KERNEL_COMMAND_TYPE_MEMORY!!\n");
            task_id = Tau_get_initialized_queues(dev_tile, command->queue_, translated_start);
            metric_set_gpu_timestamp(task_id, translated_start);
            TAU_START_TASK(event_name.c_str(), task_id);
            metric_set_gpu_timestamp(task_id, translated_end);
            TAU_STOP_TASK(event_name.c_str(), task_id);
            //For future update
            //void* TraceCorrelationID;
            //Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
            //TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, command->corr_id_, task_id, translated_start);
            #ifdef L0_TAU_DEBUG 
            std::cout << "Memory: " << command->mem_size_ << std::endl;
            std::cout << "corr_id_: " << command->corr_id_ << std::endl;
            #endif // L0_TAU_DEBUG 
            

            void* ue = nullptr;
            std::size_t pos =  it->second.name_.find("zeCommandListAppend");
            if(pos == string::npos)
                pos = 0;
            else
                pos = 19;
            std::string event_name = "Memory Copy Size [" + it->second.name_.substr(pos) + "]";
            Tau_get_context_userevent(&ue, event_name.c_str());
            TAU_CONTEXT_EVENT_THREAD_TS(ue, command->mem_size_, task_id, translated_end);
        }

        if(task_id == -1)
        {
            printf("Error with task_id, skipping %s\n", event_name.c_str());
            kernel_command_properties_mutex_.unlock_shared();
            return;
        }

        #ifdef L0_TAU_DEBUG 
        std::cout << "time_shift " << time_shift << std::endl;
        std::cout << "translated_start " << translated_start << std::endl;
        std::cout << "translated_end " << translated_end << std::endl;
        #endif

        if(L0_collector_options.metric_query==true)
        {
            for (auto it = local_device_submissions_.metric_queries_submitted_.begin(); it != local_device_submissions_.metric_queries_submitted_.end();it++) 
            {
                ZeCommandMetricQuery* curr_mq = *it;
                //printf("[!!] Metric query id %lu\n", curr_mq->instance_id_);
                if(command->instance_id_ == curr_mq->instance_id_)
                {
                    //printf("[!!] Processing metrics\n");
                    ze_result_t mq_status;
                    ZeCommandMetricQuery *command_metric_query = *it;
                    mq_status = ZE_FUNC(zeEventQueryStatus)(command_metric_query->metric_query_event_);
                    if (mq_status == ZE_RESULT_SUCCESS) 
                    {
                        size_t size = 0;
                        mq_status = ZE_FUNC(zetMetricQueryGetData)(command_metric_query->metric_query_, &size, nullptr);
                        if ((mq_status == ZE_RESULT_SUCCESS) && (size > 0))
                        {
                            std::vector<uint8_t> *kmetrics = new std::vector<uint8_t>(size);
                            size_t size2 = size;
                            mq_status = ZE_FUNC(zetMetricQueryGetData)(command_metric_query->metric_query_, &size2, kmetrics->data());
                            if(size != size2)
                                break;
                            
                            static zet_metric_group_handle_t metric_group = TAU_L0_get_metric_group(command->device_);
                            assert(metric_group != NULL);
                            static std::vector<std::string> metric_names = TAU_L0_get_metric_names(metric_group);
                            
                            uint32_t value_count = 0;
                            mq_status = ZE_FUNC(zetMetricGroupCalculateMetricValues)(
                                                metric_group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
                                                kmetrics->size(), kmetrics->data(), &value_count, nullptr);
                            
                            if( mq_status != ZE_RESULT_SUCCESS)
                                printf("Error getting metric information\n");
                            
                            std::vector<zet_typed_value_t> reported_metrics(value_count);
                            mq_status = ZE_FUNC(zetMetricGroupCalculateMetricValues)(
                                                metric_group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
                                                kmetrics->size(), kmetrics->data(), 
                                                &value_count, reported_metrics.data());
                            
                            for(uint32_t i=0; i<value_count; i++)
                            {

                                double curr_metric = TAU_LO_translate_metric_value(reported_metrics[i]);
                                //std::cout << "[!!-] " << metric_names[i] << " " << curr_metric << std::endl;
                                void* ue = nullptr;
                                Tau_get_context_userevent(&ue, metric_names[i].c_str());
                                TAU_CONTEXT_EVENT_THREAD_TS(ue, curr_metric, task_id, translated_end);
                            }
                                            

                            
                        }

                    }


                    break;
                }
            }
        }

    }
    kernel_command_properties_mutex_.unlock_shared();
}



std::string GetStallKernelName(uint64_t addr)
{
    std::string stall_kernel_name = "Unknown";
    kernel_command_properties_mutex_.lock();

    auto it = map_sampling_kernels.upper_bound(addr);

    if (it != map_sampling_kernels.begin())
    {
        --it;

        uint64_t start = it->first;
        uint64_t end = start + it->second.size_;

        if (addr >= start && addr < end)
        {
            stall_kernel_name = it->second.name_;
        }
    }
    //There are times when the address may not belong to a kernel, discard them
    // Enable this for debugging
    /*
    else
    {
        for(auto& kernel_elem : map_sampling_kernels)
        {
            printf("+[%lu] %s %lu %lu\n", addr, kernel_elem.second.name_.c_str(), kernel_elem.first, kernel_elem.second.size_);
        }
    }*/
    /*
    std::cout << "+[" << "0x" << std::setw(5) << std::setfill('0') << std::hex << std::uppercase
              << addr << "] " << it->second.name_.c_str() 
              << " " << it->first << " " << it->second.size_ << std::endl;
    printf("+[%lu] %s %lu %lu\n", addr, it->second.name_.c_str(), it->first, it->second.size_);
    */
    std::string file_name = "UNRESOLVED";
    uint64_t inst_line = -1;
    std::string inst_text = "";
    auto it2 = tau_map_l0_inst_info.find(addr);
    //std::cout << "! ? Address: 0x" << std::hex << addr << std::endl;
    if(it2 != tau_map_l0_inst_info.end())
    {
        //std::cout << "!! file_id: " << it2->second.file_id << std::endl;
        file_name = tau_map_l0_source_info[it2->second.file_id];
        inst_line = it2->second.line_number;
        inst_text = it2->second.text;
    }
    /*
    for (const auto& [key, value] : tau_map_l0_source_info) {
        std::cout << "!! file_id: " << key
              << " -> " << value
              << std::endl;
    }
    for (const auto& [addr, info] : tau_map_l0_inst_info) {
        std::cout << "! Address: 0x" << std::hex << addr << std::dec
              << " | Text: " << info.text
              << " | Line: " << info.line_number
              << " | Original addr: 0x" << std::hex << info.original_address << std::dec
              << " | File ID: " << info.file_id
              << std::endl;
    }*/
    //std::cout << " !! " << file_name << std::endl;
    kernel_command_properties_mutex_.unlock();
    return TAU_L0_demangle_sampling(stall_kernel_name, file_name, inst_line, inst_text);
}

void TauStallSamplingEvents( uint64_t address, const char *event_name, uint64_t event_value, ze_device_handle_t curr_device)
{    
    tuple<uintptr_t, int, size_t> dev_tile(reinterpret_cast<uintptr_t>(curr_device), 0, 0);
    int taskid = Tau_get_initialized_queues(dev_tile, 0, 0);
    std::string kernel_name = GetStallKernelName(address);
    //With some codes, we get some sampling that are trash
    std::string l0_unkn_string = "[L0] GPU: Unknown [{UNRESOLVED} {0,0}]";
    if(kernel_name.size() >= l0_unkn_string.size() &&
        kernel_name.compare(0, l0_unkn_string.size(), l0_unkn_string) == 0)
    {
        return;
    }

    std::stringstream ss;
    ss << kernel_name ;
    //ss << " [Address: ";
    //ss << std::hex << address << "] ";
    ss << event_name;
    std::string tmp = ss.str();
    void* ue = Tau_get_userevent(tmp.c_str());
    Tau_userevent_thread(ue, (double)event_value, taskid);
}

void TauL0EnableProfiling()
{
    //TAU_VERBOSE("To enable metrics: L0_METRICGROUP=ComputeBasic\n");

    if(!TauEnv_get_l0_enable())
    {
        TAU_VERBOSE("TAU: Disabling Level Zero support as ZE_ENABLE_TRACING_LAYER was not set from tau_exec -l0\n");
        return;
    }
    else
    {
        L0_TAU_DEBUG_MSG("TauL0EnableProfiling");
    }

    L0_collector_options = init_collector_options();
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    #ifdef L0METRICS
    if(status != ZE_RESULT_SUCCESS)
    {
        printf("L0 failed to initialize\n");
        if(TauEnv_get_l0_metrics_enable())
        {
            printf("Ensure that either /proc/sys/dev/i915/perf_stream_paranoid or /proc/sys/dev/xe/observation_paranoid is set to 0\n");
            if(status == ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE)
                printf("System is missing metrics-discovery [https://github.com/intel/metrics-discovery]\n");
        }
    }
    #else
    if(status != ZE_RESULT_SUCCESS)
    {
        if(TauEnv_get_l0_metrics_enable())
        {
            printf("Was metrics_discovery_api.h found while configuring?\n");

        }
        printf("L0 failed to initialize\n");
    }
    #endif
    assert(status == ZE_RESULT_SUCCESS);
    ze_collector_ = ZeCollector::Create(L0_collector_options);

    if(L0_collector_options.stall_sampling)
    {
        metric_profiler = ZeMetricProfiler::Create();;
    }

    initialized = 1;
    TAU_VERBOSE("Initialized L0 Collector\n");
}

void TauL0DisableProfiling()
{

    if(disabled || !initialized)
        return;
    
    // wait for child process to complete
    while (wait(nullptr) > 0);

    ze_collector_->DisableTracing();
    disabled = 1;
    if(metric_profiler != nullptr)
        delete metric_profiler;
    ze_collector_->Finalize();
    //ze_collector_->flush_initialized_queues(); //--TODO
    uint64_t cpu_end_ts = TauTraceGetTimeStamp(0);
    Tau_remove_initialized_queues(cpu_end_ts);
}

void Tau_L0new_flush()
{
    TauL0DisableProfiling();
}

#if 0
//Try do do the wrapper for zeInit
typedef ze_result_t (*real_zeInit_t)(ze_init_flags_t);

ZE_APIEXPORT ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags)
{
    static real_zeInit_t real_zeInit = NULL;

    if (real_zeInit == NULL) {
        real_zeInit = (real_zeInit_t)dlsym(RTLD_NEXT, "zeInit");
        if (!real_zeInit) {
            fprintf(stderr, "Failed to find real zeInit\n");
            return ZE_RESULT_ERROR_UNKNOWN;
        }
    }

    if(!Tau_init_check_initialized())
        Tau_init_initializeTAU();

    return real_zeInit(flags);
}

#endif