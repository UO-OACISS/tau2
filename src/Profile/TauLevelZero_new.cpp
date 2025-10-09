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
#include "Profile/L0_new/common_header.gen"
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

//Disabled options, except Metric_Query and Stall_sampling
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
  bool metric_query = true;
  bool metric_stream = false;
  bool stall_sampling = false;
};

std::string my_log ;
Logger *logger_ = nullptr;


#include "Profile/L0_new/ze_collector.h"

static double L0_init_timestamp;
static int initialized = 0;
static int disabled = 0;
static CollectorOptions L0_collector_options;

//Map a device and tile to a task
static std::map<tuple<uintptr_t, int>, int> map_thread_queue;

ZeCollector* ze_collector_ = nullptr;

OnZeKernelFinishCallback L0_k_callback = nullptr;
OnZeFunctionFinishCallback L0_a_callback = nullptr;

CollectorOptions init_collector_options()
{
    CollectorOptions init_options;
    /*
    if()
        init_options.metric_query = true;
    */
    /*
    if()
    {
        init_options.stall_sampling = true;
        //Check if metrics requested, if requested, throw error, as not compatible 
        init_options.metric_query = false;
    }
    */
    
    return init_options;
}

//Only call inside functions that use lock_guard, not implemented lock inside to prevent deadlocks.
//Check implementation for metrics
int Tau_get_initialized_queues(tuple<uintptr_t, int> dev_tile)
{
  int queue_id;
  auto it = map_thread_queue.find(dev_tile);
  if(it !=map_thread_queue.end())
  {
    queue_id = it->second;
  }
  else
  {
    TAU_CREATE_TASK(queue_id);
    // losing resolution from nanoseconds to microseconds.
    metric_set_gpu_timestamp(queue_id, L0_init_timestamp);
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
  std::map<tuple<uintptr_t, int>, int>::iterator it;
  for(it = map_thread_queue.begin(); it != map_thread_queue.end(); it++)
  {
    int taskid = it->second;
    metric_set_gpu_timestamp(taskid, cpu_end_ts);
    Tau_stop_top_level_timer_if_necessary_task(taskid);    
  }
  map_thread_queue.clear();
}


void TAU_L0_metric_callback()
{
    TAU_VERBOSE("L0_metric_callback\n");
    if(!initialized || disabled)
    {
        L0_TAU_DEBUG_MSG("L0_metric_callback !initialized || disabled !!\n");
        return;
    }
    //TODO

}

//Should not need tasks, as the thread executing the call, executes this callback.
//Need to test with threads
//API calls
void TAU_L0_enter_event(const char* nameAPIcall)
{
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
    static std::string launch_name = "zeCommandListAppendLaunchKernel";
    if (launch_name.compare(nameAPIcall) == 0) {
        //printf("!! Launch\n");
        // the user event for correlation IDs
        static void* TraceCorrelationID;
        //printf("!! TraceCorrelationID\n");
        Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
        //printf("!! Tau_get_context_userevent\n");
        TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, pushKernel(), current_thread, current_timestamp);
        //printf("!! TAU_CONTEXT_EVENT_THREAD\n");
    }

}

//Should not need tasks, as the thread executing the call, executes this callback.
//Need to test with threads
//API calls
void TAU_L0_exit_event(const char* nameAPIcall)
{
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


int not_kernel(const char* k_name )
{
    int found = 0;
    for (uint32_t i = 0; i <= uint32_t(ZeDeviceCommandHandle::LastCommand); i++) 
    {
        if(strcmp(k_name, device_command_names[i])==0)
        {
            found = 1;
            break;
        }
    }

    if( found )
    {
        //printf("Not a kernel : %s\n", k_name);
        return 1;
    }
    else
    {
        return 0;
    }
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

    for(auto it_metric_names = metric_names.begin(); it_metric_names != metric_names.end(); it_metric_names++)
    {
        std::cout << "[!!] " << *it_metric_names << std::endl;
    }
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

        //OMP Offloaded function names appear as a string with
        // multiple information fields, parse them
        //May change if the string changes in the future
        std::string name = it->second.name_.c_str();
        static std::string omp_off_string = "__omp_offloading";
        std::string event_name = "[L0] GPU: ";

        if( strncmp(name.c_str(), omp_off_string.c_str(), omp_off_string.length())==0)
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
            pos_key = name.find_first_of('_', pos_key + 1);
        }
            event_name = event_name + "OMP OFFLOADING ";
            event_name = event_name + Tau_demangle_name(name.substr(pos_key,name.find_last_of("l")-pos_key-1).c_str());
            event_name = event_name +" [{UNRESOLVED} {";
            event_name = event_name + name.substr(name.find_last_of("l")+1);
            event_name = event_name +" ,0}]";
        }
        else
        {
        event_name = event_name + Tau_demangle_name(name.c_str());
        }
        
        //if(not_kernel(name.c_str()))
        if(it->second.type_ != KERNEL_COMMAND_TYPE_COMPUTE)
        {
            #ifdef L0_TAU_DEBUG  
                std::string output_msg = "Not a kernel "+ event_name;
                L0_TAU_DEBUG_MSG(output_msg.c_str());
            #endif
            kernel_command_properties_mutex_.unlock_shared();
            return;
        }

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
        #endif
        printf("[!!] Instace ID %lu\n", command->instance_id_);

 


        int curr_tile = tile<0? 0:tile;
        uintptr_t curr_device = reinterpret_cast<uintptr_t>(command->device_);
        tuple<uintptr_t, int> dev_tile(curr_device, curr_tile);
        //tau2-intel --> ze_collector.h 792 TAU_L0_kernel_event

        int task_id = Tau_get_initialized_queues(tuple(curr_device,command->tid_));
        static uint64_t firstTauTraceGetTimeStamp = TauTraceGetTimeStamp(0);
        static uint64_t firstkernel_end = kernel_end;
        static double time_shift = TauTraceGetTimeStamp(0) - (kernel_end/1e3);
        double translated_start = time_shift + (kernel_start/1e3);
        double translated_end = time_shift + (kernel_end/1e3);

        metric_set_gpu_timestamp(task_id, translated_start);
        TAU_START_TASK(event_name.c_str(), task_id);
        metric_set_gpu_timestamp(task_id, translated_end);
        TAU_STOP_TASK(event_name.c_str(), task_id);
        void* TraceCorrelationID;
        Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
        TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, popKernel(), task_id, translated_start);

        if(it->second.type_ == KERNEL_COMMAND_TYPE_COMPUTE)
        {

            #ifdef L0_TAU_DEBUG  
            std::cout << "group_count.groupCountX " << command->group_count_.groupCountX << std::endl;
            std::cout << "group_count.groupCountY " << command->group_count_.groupCountY << std::endl;
            std::cout << "group_count.groupCountZ " << command->group_count_.groupCountZ << std::endl;
            std::cout << "it->second.group_size_.x " << it->second.group_size_.x << std::endl;
            std::cout << "it->second.group_size_.y " << it->second.group_size_.y << std::endl;
            std::cout << "it->second.group_size_.z " << it->second.group_size_.z << std::endl;
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

            #ifdef L0_TAU_DEBUG 
            std::cout << "Memory: " << command->mem_size_ << std::endl;
            #endif // L0_TAU_DEBUG 

            void* ue = nullptr;
            Tau_get_context_userevent(&ue, "Memory Copy Size");
            TAU_CONTEXT_EVENT_THREAD_TS(ue, command->mem_size_, task_id, translated_end);
        }

        #ifdef L0_TAU_DEBUG 
        std::cout << "firstTauTraceGetTimeStamp " << firstTauTraceGetTimeStamp << std::endl;
        std::cout << "firstkernel_end " << firstkernel_end << std::endl;
        std::cout << "time_shift " << time_shift << std::endl;
        std::cout << "translated_start " << translated_start << std::endl;
        std::cout << "translated_end " << translated_end << std::endl;
        #endif

        if(L0_collector_options.metric_query==true)
        {
            for (auto it = local_device_submissions_.metric_queries_submitted_.begin(); it != local_device_submissions_.metric_queries_submitted_.end();it++) 
            {
                ZeCommandMetricQuery* curr_mq = *it;
                printf("[!!] Metric query id %lu\n", curr_mq->instance_id_);
                if(command->instance_id_ == curr_mq->instance_id_)
                {
                    printf("[!!] Processing metrics\n");
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

                            //https://github.com/intel/pti-gpu/blob/0be8d49d441bbcfc8ab801534afa9e4a5a0b93f7/tools/unitrace/src/levelzero/ze_collector.h#L2269
                            
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



void TauL0EnableProfiling()
{
    printf("To enable metrics: ZET_ENABLE_METRICS=1 UNITRACE_MetricGroup=ComputeBasic\n");

    if (getenv("ZE_ENABLE_TRACING_LAYER") == NULL) 
    {
        // tau_exec -level_zero was not called. Perhaps it is using -opencl
        TAU_VERBOSE("TAU: Disabling Level Zero support as ZE_ENABLE_TRACING_LAYER was not set from tau_exec -l0\n");
        return;
    }
    else
    {
        L0_TAU_DEBUG_MSG("TauL0EnableProfiling");
    }

    L0_collector_options = init_collector_options();

    L0_init_timestamp = TauTraceGetTimeStamp(0);
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    assert(status == ZE_RESULT_SUCCESS);
    logger_ = new Logger(my_log);
    ze_collector_ = ZeCollector::Create(logger_, L0_collector_options);
    initialized = 1;
    TAU_VERBOSE("Initialized L0 Collector\n");
}

void TauL0DisableProfiling()
{

    if(disabled)
        return;
    L0_TAU_DEBUG_MSG("Disabling Tau L0");

    ze_collector_->DisableTracing();
    ze_collector_->Finalize();
    //ze_collector_->flush_initialized_queues(); //--TODO
    uint64_t cpu_end_ts = TauTraceGetTimeStamp(0);
    Tau_remove_initialized_queues(cpu_end_ts);
    disabled = 1;
}
