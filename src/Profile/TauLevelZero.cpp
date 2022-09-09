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
#include <Profile/L0/utils.h>
#include <Profile/L0/ze_kernel_collector.h>
#include <Profile/L0/ze_api_collector.h>


#include "Profile/Profiler.h"
#include "Profile/TauBfd.h"
using namespace std;

extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" x_uint64 TauTraceGetTimeStamp(int tid);


static ZeApiCollector* api_collector = nullptr;
static ZeKernelCollector* kernel_collector = nullptr;
static std::chrono::steady_clock::time_point start;
static int gpu_task_id = 0;
static int host_api_task_id = 0;
static uint64_t first_cpu_timestamp = 0L;
static uint64_t first_gpu_timestamp = 0L;
static uint64_t last_gpu_timestamp = 0L;
static uint64_t gpu_offset = 0L;
extern "C" void metric_set_gpu_timestamp(int tid, double value);


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




bool TAUSetFirstGPUTimestamp(uint64_t gpu_ts) {
  TAU_VERBOSE("TAU: First GPU Timestamp = %ld\n", gpu_ts);
  if (first_gpu_timestamp == 0L) {
    first_gpu_timestamp = gpu_ts;

  }
  return true;
}

double TAUTranslateGPUtoCPUTimestamp(int tid, uint64_t gpu_ts) {
  // gpu_ts is in nanoseconds. We need the CPU timestamp result in microseconds.

  double cpu_ts = first_cpu_timestamp + ((gpu_ts - first_gpu_timestamp)/1e3);
  // losing resolution from nanoseconds to microseconds.
  metric_set_gpu_timestamp(tid, cpu_ts);
  Tau_create_top_level_timer_if_necessary_task(tid);

  return cpu_ts;
}

void TAUOnAPIFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {
  int taskid;
  static bool first_ts = TAUSetFirstGPUTimestamp(started);

  taskid = *((int *) data);
  double started_translated = TAUTranslateGPUtoCPUTimestamp(taskid, started);
  double ended_translated = TAUTranslateGPUtoCPUTimestamp(taskid, ended);
  TAU_VERBOSE("TAU: OnAPIFinishCallback: (raw) name: %s started: %g ended: %g task id=%d\n",
		  name.c_str(), started, ended, taskid);
  TAU_VERBOSE("TAU: OnAPIFinishCallback: (translated) name: %s started: %g ended: %g task id=%d\n",
		  name.c_str(), started_translated, ended_translated, taskid);
  // We now need to start a timer on a task at the started_translated time and end at ended_translated

  metric_set_gpu_timestamp(taskid, started_translated);
  TAU_START_TASK(name.c_str(), taskid);

  metric_set_gpu_timestamp(taskid, ended_translated);
  TAU_STOP_TASK(name.c_str(), taskid);
}

void TAUOnKernelFinishCallback(void *data, const std::string& name, uint64_t started, uint64_t ended) {

  static bool first_call = TAUSetFirstGPUTimestamp(started);
  int taskid;
  taskid = *((int *) data);
  const char *kernel_name = name.c_str();
  double started_translated = TAUTranslateGPUtoCPUTimestamp(taskid, started);
  double ended_translated = TAUTranslateGPUtoCPUTimestamp(taskid, ended);
  char *demangled_name = Tau_demangle_name(kernel_name);
  TAU_VERBOSE("TAU: <kernel>: (raw) name: %s  started: %ld ended: %ld task id=%d\n",
		  name.c_str(), started, ended, taskid);
  TAU_VERBOSE("TAU: <kernel>: (raw) name: %s started: %g ended: %g task id=%d\n",
    name.c_str(),  started_translated, ended_translated, taskid);

  last_gpu_timestamp = ended;
  metric_set_gpu_timestamp(taskid, started_translated);
  TAU_START_TASK(demangled_name, taskid);


  metric_set_gpu_timestamp(taskid, ended_translated);
  TAU_STOP_TASK(demangled_name, taskid);
  free(demangled_name);
  return;
}


// Internal Tool Interface ////////////////////////////////////////////////////

void TauL0EnableProfiling() {
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

  int *kernel_taskid = new int;
  TAU_CREATE_TASK(*kernel_taskid);
  void *pk = (void *) kernel_taskid;
  gpu_task_id = *kernel_taskid;
  int *api_taskid  = new int;
  //*host_taskid = RtsLayer::myThread();
  TAU_CREATE_TASK(*api_taskid);
  host_api_task_id = *api_taskid;
  kernel_collector = ZeKernelCollector::Create(driver,
                  TAUOnKernelFinishCallback, pk);
  /*
  //uint64_t gpu_ts = utils::i915::GetGpuTimestamp() & 0x0FFFFFFFF;
  uint64_t gpu_ts = utils::i915::GetGpuTimestamp() ;
  std::cout <<"TAU: Earliest GPU timestamp "<<gpu_ts<<std::endl;
  */
  first_cpu_timestamp = TauTraceGetTimeStamp(0);
  TAU_VERBOSE("TAU: Earliest CPU timestamp= %ld \n",first_cpu_timestamp);

  // For API calls, we create a new task and trigger the start/stop based on its
  // timestamps.

  void *ph = (void *) api_taskid;
  api_collector = ZeApiCollector::Create(driver, TAUOnAPIFinishCallback, ph);

  metric_set_gpu_timestamp(host_api_task_id, first_cpu_timestamp);
  Tau_create_top_level_timer_if_necessary_task(host_api_task_id);


  start = std::chrono::steady_clock::now();
}

void TauL0DisableProfiling() {
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
  //uint64_t gpu_end_ts = utils::i915::GetGpuTimestamp() & 0x0FFFFFFFF;
  /*
  uint64_t gpu_end_ts = utils::i915::GetGpuTimestamp();
  std::cout <<"TAU: Latest GPU timestamp "<<gpu_end_ts<<std::endl;
  */
  int taskid = gpu_task_id;  // GPU task id is 1;
  uint64_t last_gpu_translated = TAUTranslateGPUtoCPUTimestamp(1, last_gpu_timestamp);
  TAU_VERBOSE("TAU: Latest GPU timestamp (raw) =%ld\n", last_gpu_timestamp);
  TAU_VERBOSE("TAU: Latest GPU timestamp (translated) =%ld\n",last_gpu_translated);
  uint64_t cpu_end_ts = TauTraceGetTimeStamp(0);
  metric_set_gpu_timestamp(taskid, last_gpu_translated);
  Tau_stop_top_level_timer_if_necessary_task(taskid);

  metric_set_gpu_timestamp(host_api_task_id, cpu_end_ts);
  Tau_create_top_level_timer_if_necessary_task(host_api_task_id);

  TAU_VERBOSE("TAU: Latest CPU timestamp =%ld\n", cpu_end_ts);
  std::chrono::steady_clock::time_point chrono_end = std::chrono::steady_clock::now();
  std::chrono::duration<uint64_t, std::nano> chrono_dt = chrono_end - start;
  TAU_VERBOSE("TAU: Diff (chrono) =%ld \n", chrono_dt.count());
}



