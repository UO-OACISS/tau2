//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

//#include "tool.h"

#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <Profile/TauLevelZeroTracingAPI.h>
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>
#include <Profile/L0/utils.h>
#include <Profile/L0/ze_kernel_collector.h>
#include <Profile/L0/ze_api_collector.h>

#ifdef L0METRICS
#include <Profile/L0/ze_metric_collector.h>
#endif

#include "Profile/Profiler.h"
#include "Profile/TauBfd.h"
using namespace std;

extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" x_uint64 TauTraceGetTimeStamp(int tid);


static ZeApiCollector* api_collector = nullptr;
static ZeKernelCollector* kernel_collector = nullptr;
#ifdef L0METRICS
static ZeMetricCollector* metric_collector = nullptr;
#endif
static std::chrono::steady_clock::time_point start;
static int gpu_task_id = -1;
static int host_api_task_id = -1;
static uint64_t first_cpu_timestamp = 0L;
static uint64_t first_gpu_kernel_timestamp = 0L;
static uint64_t first_gpu_api_timestamp = 0L;
static uint64_t last_gpu_kernel_timestamp = 0L;
static uint64_t last_gpu_api_timestamp = 0L;
static uint64_t gpu_offset = 0L;
static std::mutex gpu_mutex; // TODO investigate whether it makes more sense to use a task per thread
                             // instead of a single task for all threads, rather than locking
static std::mutex queue_mutex;
extern "C" void metric_set_gpu_timestamp(int tid, double value);

static std::map<int, int> map_thread_queue;

///////////////////////////////////////////////////////////////////////////////
void Tau_metric_set_synchronized_gpu_timestamp(int tid, double value) {
  /* TAU_VERBOSE("state->offset_timestamp = %ld, value (entering) = %ld ", state->offset_timestamp, (uint64_t)value);
  metric_set_gpu_timestamp(tid, state->offset_timestamp+(uint64_t)value);
  TAU_VERBOSE("value (exiting) = %ld\n", state->offset_timestamp+(uint64_t)value);
  if (state->offset_timestamp == 0) {
    state->offset_timestamp = value;
    printf(" Setting state->offset_timestamp = %ld\n", state->offset_timestamp);
    printf("value = %ld, offset+value=%ld\n", (uint64_t) value, state->offset_timestamp + value);
  }
  */
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////


// External Tool Interface ////////////////////////////////////////////////////

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
void Usage() {
  std::cout <<
    "Usage: ./ze_hot_kernels[.exe] <application> <args>" <<
    std::endl;
}

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
int ParseArgs(int argc, char* argv[]) {
  return 1;
}

extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
void SetToolEnv() {
  utils::SetEnv("ZET_ENABLE_API_TRACING_EXP","1");
}

// Internal Tool Functionality ////////////////////////////////////////////////

static void PrintResults() {
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<uint64_t, std::nano> time = end - start;

  PTI_ASSERT(kernel_collector != nullptr);
  const ZeKernelInfoMap& kernel_info_map = kernel_collector->GetKernelInfoMap();
  if (kernel_info_map.size() == 0) {
    return;
  }

  uint64_t total_duration = 0;
  for (auto& value : kernel_info_map) {
    total_duration += value.second.total_time;
  }

  std::cerr << std::endl;
  std::cerr << "=== Device Timing Results: ===" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Total Execution Time (ns): " << time.count() << std::endl;
  std::cerr << "Total Device Time (ns): " << total_duration << std::endl;
  std::cerr << std::endl;

  if (total_duration > 0) {
    ZeKernelCollector::PrintKernelsTable(kernel_info_map);
  }

  std::cerr << std::endl;
}

// Internal Tool Functionality ////////////////////////////////////////////////

static void APIPrintResults() {
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<uint64_t, std::nano> time = end - start;

  PTI_ASSERT(api_collector != nullptr);
  const ZeFunctionInfoMap& function_info_map = api_collector->GetFunctionInfoMap();
  if (function_info_map.size() == 0) {
    return;
  }

  uint64_t total_duration = 0;
  for (auto& value : function_info_map) {
    total_duration += value.second.total_time;
  }

  std::cerr << std::endl;
  std::cerr << "=== API Timing Results: ===" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Total Execution Time (ns): " << time.count() << std::endl;
  std::cerr << "Total API Time (ns): " << total_duration << std::endl;
  std::cerr << std::endl;

  if (total_duration > 0) {
    ZeApiCollector::PrintFunctionsTable(function_info_map);
  }

  std::cerr << std::endl;
}

#ifdef L0METRICS

struct Kernel {
  uint64_t total_time;
  uint64_t call_count;
  float eu_active;
  float eu_stall;

  bool operator>(const Kernel& r) const {
    if (total_time != r.total_time) {
      return total_time > r.total_time;
    }
    return call_count > r.call_count;
  }

  bool operator!=(const Kernel& r) const {
    if (total_time == r.total_time) {
      return call_count != r.call_count;
    }
    return true;
  }
};

double get_metric_value(zet_typed_value_t metric)
{
   switch (metric.type) {
     case ZET_VALUE_TYPE_UINT32:{
       return (float) metric.value.ui32;
     }
     case ZET_VALUE_TYPE_UINT64:{
       return (float) metric.value.ui64;
     }
     case ZET_VALUE_TYPE_FLOAT32:{
       return (float) metric.value.fp32;
     }
     case ZET_VALUE_TYPE_FLOAT64:{
       return (float) metric.value.fp64;
     }
     case ZET_VALUE_TYPE_BOOL8:{
       return (float) metric.value.b8;
     }
     default:{
       return -1;
       break;
     }
   }
   
   return -1;
}

static void MetricPrintResults() {

  const KernelReportMap& kernel_report_map = metric_collector->GetKernelReportMap();
  std::vector<std::string> metriclist = metric_collector->GetMetricList();
  if (kernel_report_map.size() == 0) {
    return;
  }
  
  std::cerr << "=== Metric Results: ===" << std::endl;
  for (auto& kernel : kernel_report_map) {
  
    std::cerr << "Results Kernel: "<< kernel.first.c_str() << std::endl;
    std::vector<MetricReport> kernel_reports = kernel.second; 
    int entry = 0;
    for (auto& report_entry : kernel_reports) {
      std::cerr << "Entry "<< entry << " : " << std::endl ;
      entry++;
      assert(report_entry.size() == metriclist.size());
      int i;  
      for ( i = 0; i < metriclist.size(); i++ ){
        
        double metric_value =get_metric_value(report_entry[i]);
        std::cerr << "\t"<< metriclist[i].c_str() << "Value: " << metric_value << std::endl ;
      }
      std:cerr << std::endl;
          
          
    }
  }
   std::cerr << "======" << std::endl;
}
#endif

double TAUTranslateGPUtoCPUAPITimestamp(int tid, uint64_t gpu_ts) {
  // gpu_ts is in nanoseconds. We need the CPU timestamp result in microseconds.
  double cpu_ts = first_cpu_timestamp + ((gpu_ts - first_gpu_api_timestamp)/1e3);
  // losing resolution from nanoseconds to microseconds.
  metric_set_gpu_timestamp(tid, cpu_ts);
  Tau_create_top_level_timer_if_necessary_task(tid);

  return cpu_ts;
}

double TAUTranslateGPUtoCPUKernelTimestamp(int tid, uint64_t gpu_ts) {
  // gpu_ts is in nanoseconds. We need the CPU timestamp result in microseconds.
  double cpu_ts = first_cpu_timestamp + ((gpu_ts - first_gpu_kernel_timestamp)/1e3);
  // losing resolution from nanoseconds to microseconds.
  metric_set_gpu_timestamp(tid, cpu_ts);
  Tau_create_top_level_timer_if_necessary_task(tid);

  return cpu_ts;
}

bool TAUSetFirstGPUAPITimestamp(uint64_t gpu_ts) {
  if (first_gpu_api_timestamp == 0L) {
  TAU_VERBOSE("TAU: First GPU API Timestamp = %ld\n", gpu_ts);
    first_gpu_api_timestamp = gpu_ts;
  }
  return true;
}

bool TAUSetFirstGPUKernelTimestamp(uint64_t gpu_ts) {
  if (first_gpu_kernel_timestamp == 0L) {
    TAU_VERBOSE("TAU: First GPU Kernel Timestamp = %ld\n", gpu_ts);
    first_gpu_kernel_timestamp = gpu_ts;
  }
  return true;
}


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

void TAUOnAPIFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {
  std::lock_guard<std::mutex> guard(gpu_mutex); // Lock before we start touching task-specific state
  static bool first_ts = TAUSetFirstGPUAPITimestamp(started); 

  int taskid = host_api_task_id;
  if(taskid == -1)
  {
    TAU_CREATE_TASK(taskid);
    host_api_task_id = taskid;
  }  

#ifdef TAU_DEBUG_L0
  fprintf(stderr, "pid=%d, tid=%d, myThread=%d, task=%d, func=%s\n", RtsLayer::getPid(), RtsLayer::getTid(), RtsLayer::myThread(), taskid, name.c_str());
#endif
  double started_translated = TAUTranslateGPUtoCPUAPITimestamp(taskid, started);
  double ended_translated = TAUTranslateGPUtoCPUAPITimestamp(taskid, ended);
  TAU_VERBOSE("TAU: OnAPIFinishCallback: (raw) name: %s started: %g ended: %g task id=%d\n",
		  name.c_str(), started, ended, taskid);
  TAU_VERBOSE("TAU: OnAPIFinishCallback: (translated) name: %s started: %g ended: %g task id=%d\n",
		  name.c_str(), started_translated, ended_translated, taskid);
  // We now need to start a timer on a task at the started_translated time and end at ended_translated

  metric_set_gpu_timestamp(taskid, started_translated);
  TAU_START_TASK(name.c_str(), taskid);
  if (name.compare("zeCommandListAppendLaunchKernel") == 0) {
    // the user event for correlation IDs
    static void* TraceCorrelationID;
    Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
    TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, pushKernel(), taskid, started_translated);
  }

  metric_set_gpu_timestamp(taskid, ended_translated);
  TAU_STOP_TASK(name.c_str(), taskid);
}

//Only call inside functions that use lock_guard, not implemented lock inside to prevent deadlocks.
int Tau_get_initialized_queues(int thread)
{
  int queue_id;
  auto it = map_thread_queue.find(thread);
  //Metrics will only be shown in one thread for all L0 threads
  static int set_queue_metrics_once = 0;
  if(it !=map_thread_queue.end())
  {
    queue_id = it->second;
  }
  else
  {
    TAU_CREATE_TASK(queue_id);
    // losing resolution from nanoseconds to microseconds.
    metric_set_gpu_timestamp(queue_id, first_cpu_timestamp);
    Tau_create_top_level_timer_if_necessary_task(queue_id);
    //std::cout << " NEW TASK: " << queue_id << std::endl;
    map_thread_queue[thread] = queue_id;
    if(set_queue_metrics_once==0)
    {
      gpu_task_id = queue_id;
      set_queue_metrics_once=1;
    }
  }
  return queue_id;
}

//Only call inside functions that use lock_guard, not implemented lock inside to prevent deadlocks.
void Tau_remove_initialized_queues(uint64_t cpu_end_ts)
{
  std::map<int, int>::iterator it;
  for(it = map_thread_queue.begin(); it != map_thread_queue.end(); it++)
  {
    int taskid = it->second;
    metric_set_gpu_timestamp(taskid, cpu_end_ts);
    Tau_stop_top_level_timer_if_necessary_task(taskid);    
  }
  map_thread_queue.clear();
}


void TAUOnKernelFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {
  std::lock_guard<std::mutex> guard(gpu_mutex); // Lock before we start touching task-specific state

  static bool first_call = TAUSetFirstGPUKernelTimestamp(started);
  int current_thread = RtsLayer::myThread();
  int taskid = Tau_get_initialized_queues(current_thread);

  const char *kernel_name = name.c_str();
  double started_translated = TAUTranslateGPUtoCPUKernelTimestamp(taskid, started);
  double ended_translated = TAUTranslateGPUtoCPUKernelTimestamp(taskid, ended);

  static std::string omp_off_string = "__omp_offloading";
  std::string event = "[L0] GPU: ";

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
      event = event + "OMP OFFLOADING ";
      event = event + Tau_demangle_name(name.substr(pos_key,name.find_last_of("l")-pos_key-1).c_str());
      event = event +" [{UNRESOLVED} {";
      event = event + name.substr(name.find_last_of("l")+1);
      event = event +" ,0}]";
  }
  else
  {
    event = event + Tau_demangle_name(kernel_name);
  }

  const char *demangled_name = event.c_str();

  TAU_VERBOSE("TAU: <kernel>: (raw) name: %s  started: %ld ended: %ld task id=%d\n",
		  name.c_str(), started, ended, taskid);
  TAU_VERBOSE("TAU: <kernel>: (raw) name: %s started: %g ended: %g task id=%d\n",
    name.c_str(),  started_translated, ended_translated, taskid);

  last_gpu_kernel_timestamp = ended;
  metric_set_gpu_timestamp(taskid, started_translated);
  TAU_START_TASK(demangled_name, taskid);
  // the user event for correlation IDs
  static void* TraceCorrelationID;
  Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
  TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, popKernel(), taskid, started_translated);

  metric_set_gpu_timestamp(taskid, ended_translated);
  TAU_STOP_TASK(demangled_name, taskid);

  return;
}

#ifdef L0METRICS
void TAUOnMetricFinishCallback(void *data, const std::string& kernel_name, MetricReport report, std::vector<std::string> metriclist)
{

  int taskid = gpu_task_id;

  if(taskid == -1)
  {
    TAU_CREATE_TASK(taskid);
    gpu_task_id = taskid;
  }

  assert(report.size() == metriclist.size());
  int i;  
  for ( i = 0; i < metriclist.size(); i++ ){
    std::stringstream ss;
    void* ue = nullptr;
    std::string tmp;
    ss << metriclist[i] <<" Kernel:{" << kernel_name << "}";
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    double metric_value =get_metric_value(report[i]);
    TAU_VERBOSE("TAU Metric: <kernel>: %s Event %s Value %lf\n", kernel_name.c_str(), metriclist[i].c_str(), metric_value);
    Tau_userevent_thread(ue, metric_value, taskid);

  }

}
#endif


// Internal Tool Interface ////////////////////////////////////////////////////

void TauL0EnableProfiling() {
    //printf("%s\n", __func__);
  if (getenv("ZE_ENABLE_TRACING_LAYER") == NULL) {
    // tau_exec -level_zero was not called. Perhaps it is using -opencl
    TAU_VERBOSE("TAU: Disabling Level Zero support as ZE_ENABLE_TRACING_LAYER was not set from tau_exec -l0\n");
    return;
  }
  ze_result_t status = ZE_RESULT_SUCCESS;
  status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);

  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  driver =  utils::ze::GetGpuDriver();
  device =  utils::ze::GetGpuDevice();

  if (device == nullptr || driver == nullptr) {
    std::cout << "[WARNING] Unable to find target device" << std::endl;
    return;
  }


  kernel_collector = ZeKernelCollector::Create(driver,
                  TAUOnKernelFinishCallback, nullptr);
  /*
  //uint64_t gpu_ts = utils::i915::GetGpuTimestamp() & 0x0FFFFFFFF;
  uint64_t gpu_ts = utils::i915::GetGpuTimestamp() ;
  std::cout <<"TAU: Earliest GPU timestamp "<<gpu_ts<<std::endl;
  */
  first_cpu_timestamp = TauTraceGetTimeStamp(0);
  TAU_VERBOSE("TAU: Earliest CPU timestamp= %ld \n",first_cpu_timestamp);

  // For API calls, we create a new task and trigger the start/stop based on its
  // timestamps.

  api_collector = ZeApiCollector::Create(driver, TAUOnAPIFinishCallback, nullptr);
 
   #ifdef L0METRICS
  
  std::string value;
  //EBS Metric Groups  
  std::string metric_group("ComputeBasic");
  value = utils::GetEnv("L0_MetricGroup");
  if (!value.empty()) {
    metric_group = value;
  }      
  metric_collector = ZeMetricCollector::Create( driver, device, metric_group.c_str(), TAUOnMetricFinishCallback, nullptr);

  if (metric_collector == nullptr) {
    std::cout <<
      "[WARNING] Unable to create metric collector" << std::endl;
  }
  #endif

  start = std::chrono::steady_clock::now();
}

void TauL0DisableProfiling() {
    //printf("%s\n", __func__);
  if (kernel_collector != nullptr) {
    kernel_collector->DisableTracing();
    if (TauEnv_get_verbose())
      PrintResults();
    delete kernel_collector;
  }
  if (api_collector != nullptr) {
    api_collector->DisableTracing();
    if (TauEnv_get_verbose())
      APIPrintResults();
    delete api_collector;
  }
  
  #ifdef L0METRICS  
  if (metric_collector != nullptr) {
    metric_collector->DisableTracing();
    if (TauEnv_get_verbose())
      MetricPrintResults();
    delete metric_collector;
  }
  #endif
  
  
}

