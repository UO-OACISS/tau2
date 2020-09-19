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
#include "Profile/Profiler.h"

extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);



#include <assert.h>

//#include "ze_tracer.h"
//#include "ze_utils.h"
//#include "utils.h"
//
#define NSEC_IN_USEC 1000
#define NSEC_IN_MSEC 1000000
#define NSEC_IN_SEC  1000000000


#define ZE_FUNCTION_COUNT      (ze_tracing::ZE_FUNCTION_COUNT)
#define ZE_CALLBACK_SITE_ENTER (ze_tracing::ZE_CALLBACK_SITE_ENTER)
#define ZE_CALLBACK_SITE_EXIT  (ze_tracing::ZE_CALLBACK_SITE_EXIT)


extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" x_uint64 TauTraceGetTimeStamp(int tid);


using callback_data_t = ze_tracing::callback_data_t;
using function_id_t = ze_tracing::function_id_t;
using tracing_callback_t = ze_tracing::tracing_callback_t;

class ZeTracer {
 public:
  ZeTracer(ze_driver_handle_t driver,
           tracing_callback_t callback,
           void* user_data) {
    assert(driver != nullptr);

    data_.callback = callback;
    data_.user_data = user_data;

    ze_result_t status = ZE_RESULT_SUCCESS;
    zet_tracer_desc_t tracer_desc = {};
    tracer_desc.version = ZET_TRACER_DESC_VERSION_CURRENT;
    tracer_desc.pUserData = &data_;

    status = zetTracerCreate(driver, &tracer_desc, &handle_);
    assert(status == ZE_RESULT_SUCCESS);
  }

  ~ZeTracer() {
    if (handle_ != nullptr) {
      ze_result_t status = ZE_RESULT_SUCCESS;
      status = zetTracerDestroy(handle_);
      assert(status == ZE_RESULT_SUCCESS);
    }
  }

  bool SetTracingFunction(function_id_t function) {
    if (!IsValid()) {
      return false;
    }

    if (function >= 0 && function < ZE_FUNCTION_COUNT) {
      functions_.insert(function);
      return true;
    }

    return false;
  }

  bool Enable() {
    if (!IsValid()) {
      return false;
    }

    ze_tracing::SetTracingFunctions(handle_, functions_);

    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zetTracerSetEnabled(handle_, true);
    if (status != ZE_RESULT_SUCCESS) {
      return false;
    }

    return true;
  }

  bool Disable() {
    if (!IsValid()) {
      return false;
    }

    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zetTracerSetEnabled(handle_, false);
    if (status != ZE_RESULT_SUCCESS) {
      return false;
    }

    return true;
  }

  bool IsValid() const {
    return (handle_ != nullptr);
  }

 private:
  zet_tracer_handle_t handle_ = nullptr;
  std::set<function_id_t> functions_;
  ze_tracing::global_data_t data_;
};

namespace utils {
namespace ze {

inline void GetIntelDeviceAndDriver(ze_device_type_t type,
                                    ze_device_handle_t& device,
                                    ze_driver_handle_t& driver) {
  ze_result_t status = ZE_RESULT_SUCCESS;

  uint32_t driver_count = 0;
  status = zeDriverGet(&driver_count, nullptr);
  if (status != ZE_RESULT_SUCCESS || driver_count == 0) {
    return;
  }

  std::vector<ze_driver_handle_t> driver_list(driver_count, nullptr);
  status = zeDriverGet(&driver_count, driver_list.data());
  assert(status == ZE_RESULT_SUCCESS);

  for (uint32_t i = 0; i < driver_count; ++i) {
    uint32_t device_count = 0;
    status = zeDeviceGet(driver_list[i], &device_count, nullptr);
    if (status != ZE_RESULT_SUCCESS || device_count == 0) {
        continue;
    }

    std::vector<ze_device_handle_t> device_list(device_count, nullptr);
    status = zeDeviceGet(driver_list[i], &device_count, device_list.data());
    assert(status == ZE_RESULT_SUCCESS);

    for (uint32_t j = 0; j < device_count; ++j) {
      ze_device_properties_t props;
      props.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
      status = zeDeviceGetProperties(device_list[j], &props);
      assert(status == ZE_RESULT_SUCCESS);

      if (props.type == type && strstr(props.name, "Intel") != nullptr) {
        device = device_list[j];
        driver = driver_list[i];
        break;
      }
    }
  }

  return;
}

inline std::string GetDeviceName(ze_device_handle_t device) {
  assert(device != nullptr);
  ze_result_t status = ZE_RESULT_SUCCESS;
  ze_device_properties_t props;
  props.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
  status = zeDeviceGetProperties(device, &props);
  assert(status == ZE_RESULT_SUCCESS);
  return props.name;
}

static int GetMetricId(zet_metric_group_handle_t group, std::string name) {
  assert(group != nullptr);

  ze_result_t status = ZE_RESULT_SUCCESS;
  uint32_t metric_count = 0;
  status = zetMetricGet(group, &metric_count, nullptr);
  assert(status == ZE_RESULT_SUCCESS);

  if (metric_count == 0) {
    return -1;
  }

  std::vector<zet_metric_handle_t> metric_list(metric_count, nullptr);
  status = zetMetricGet(group, &metric_count, metric_list.data());
  assert(status == ZE_RESULT_SUCCESS);

  int target = -1;
  for (uint32_t i = 0; i < metric_count; ++i) {
    zet_metric_properties_t metric_props = {};
    metric_props.version = ZET_METRIC_PROPERTIES_VERSION_CURRENT;
    status = zetMetricGetProperties(metric_list[i], &metric_props);
    assert(status == ZE_RESULT_SUCCESS);

    if (name == metric_props.name) {
      target = i;
      break;
    }
  }

  return target;
}

static zet_metric_group_handle_t FindMetricGroup(
    ze_device_handle_t device, std::string name,
    zet_metric_group_sampling_type_t type) {
  assert(device != nullptr);
  
  ze_result_t status = ZE_RESULT_SUCCESS;
  uint32_t group_count = 0;
  status = zetMetricGroupGet(device, &group_count, nullptr);
  assert(status == ZE_RESULT_SUCCESS);
  if (group_count == 0) {
    return nullptr;
  }

  std::vector<zet_metric_group_handle_t> group_list(group_count, nullptr);
  status = zetMetricGroupGet(device, &group_count, group_list.data());
  assert(status == ZE_RESULT_SUCCESS);

  zet_metric_group_handle_t target = nullptr;
  for (uint32_t i = 0; i < group_count; ++i) {
    zet_metric_group_properties_t group_props = {};
    group_props.version = ZET_METRIC_GROUP_PROPERTIES_VERSION_CURRENT;
    status = zetMetricGroupGetProperties(group_list[i], &group_props);
    assert(status == ZE_RESULT_SUCCESS);

    if (name == group_props.name && type == group_props.samplingType) {
      target = group_list[i];
      break;
    }
  }

  return target;
}

} // namespace ze


// --- 
struct Comparator {
  template<typename T>
  bool operator()(const T& left, const T& right) {
    if (left.second != right.second) {
      return left.second > right.second;
    }
    return left.first > right.first;
  }
};
} // namespace utils


enum EventType {
  EVENT_TYPE_USER = 0,
  EVENT_TYPE_TOOL = 1
};

struct ActivityEventInfo {
  std::string name;
  ze_event_pool_handle_t event_pool;
  ze_event_handle_t event;
  EventType event_type;
};

struct GlobalToolState {
  ze_driver_handle_t driver;
  ze_device_handle_t device;
  zet_tracer_handle_t tracer;
  std::mutex klock;
  std::mutex flock;
  std::map<ze_kernel_handle_t, std::string> kernel_name_map;
  std::vector<ActivityEventInfo> activity_event_list;
  std::map< std::string, std::pair<uint64_t, int> > activity_time_map;
  ZeTracer* gpu_tracer;
  std::map< std::string, std::pair<double, int> > function_time_map;
  int tau_task_id;
  uint64_t last_timestamp; 
  uint64_t earliest_timestamp; 
  uint64_t offset_timestamp; 

};

const char* kLine =
  "+----------------------------------------"
  "----------------------------------------+";
const char* kHeader =
  "| Kernel                                 "
  "          | Call Count | Total Time, ms |";

static GlobalToolState* state = nullptr;

void Tau_metric_set_synchronized_gpu_timestamp(int tid, double value) {
  TAU_VERBOSE("state->offset_timestamp = %ld, value (entering) = %ld ", state->offset_timestamp, (uint64_t)value);
  metric_set_gpu_timestamp(tid, state->offset_timestamp+(uint64_t)value);
  TAU_VERBOSE("value (exiting) = %ld\n", state->offset_timestamp+(uint64_t)value);
  if (state->offset_timestamp == 0) {
    state->offset_timestamp = value;
/*
    printf(" Setting state->offset_timestamp = %ld\n", state->offset_timestamp);
    printf("value = %ld, offset+value=%ld\n", (uint64_t) value, state->offset_timestamp + value);
*/
  }
}

static void Callback(function_id_t function, callback_data_t* callback_data,
                     void* user_data) {
  assert(state != nullptr);
  if (callback_data->site == ZE_CALLBACK_SITE_ENTER) {
    std::chrono::time_point<std::chrono::steady_clock>* correlation_data =
      reinterpret_cast<std::chrono::time_point<std::chrono::steady_clock>*>(
          callback_data->correlation_data);
    *correlation_data = std::chrono::steady_clock::now();
    if (TauEnv_get_verbose()) {
      //std::cout <<"TAU => "<<callback_data->function_name<<std::endl;
    }
    TAU_START_TASK(callback_data->function_name, 0);
  } else {
    if (TauEnv_get_verbose()) {
      //std::cout <<"TAU <= "<<callback_data->function_name<<std::endl;
    }
    TAU_STOP_TASK(callback_data->function_name, 0);
    std::chrono::time_point<std::chrono::steady_clock>* correlation_data =
      reinterpret_cast<std::chrono::time_point<std::chrono::steady_clock>*>(
          callback_data->correlation_data);
    std::chrono::duration<double> time = std::chrono::steady_clock::now() -
      *correlation_data;

    state->flock.lock();
    size_t function_count = state->function_time_map.count(
        std::string(callback_data->function_name));
    if (function_count == 0) {
      state->function_time_map[std::string(callback_data->function_name)] =
        std::make_pair(time.count(), 1);
    } else {
      std::pair<double, int>& pair =
        state->function_time_map[std::string(callback_data->function_name)];
      pair.first += time.count();
      pair.second += 1;
    }
    state->flock.unlock();
  }
}

static void OnExitKernelCreate(
    ze_kernel_create_params_t *params, ze_result_t result,
    void *global_data, void **instance_data) {
  assert(state != nullptr);
  if (result == ZE_RESULT_SUCCESS) {
    state->klock.lock();
    ze_kernel_handle_t kernel = **(params->pphKernel);
    state->kernel_name_map[kernel] = (*(params->pdesc))->pKernelName;
    state->klock.unlock();
  }
}

static void OnExitKernelDestroy(ze_kernel_destroy_params_t *params,
                                ze_result_t result,
                                void *global_data,
                                void **instance_data) {
  if (result == ZE_RESULT_SUCCESS) {
    assert(state != nullptr);

    state->klock.lock();
    ze_kernel_handle_t kernel = *(params->phKernel);
    assert(state->kernel_name_map.count(kernel) == 1);
    state->kernel_name_map.erase(kernel);
    state->klock.unlock();
  }
}

static void OnEnterEventPoolCreate(ze_event_pool_create_params_t *params,
                                   ze_result_t result,
                                   void *global_data,
                                   void **instance_data) {
  const ze_event_pool_desc_t* desc = *(params->pdesc);
  if (desc == nullptr) {
    return;
  }

  ze_event_pool_desc_t* profiling_desc = new ze_event_pool_desc_t;
  assert(profiling_desc != nullptr);
  profiling_desc->version = desc->version;
  profiling_desc->flags = desc->flags;
  profiling_desc->count = desc->count;

  int flags = profiling_desc->flags | ZE_EVENT_POOL_FLAG_TIMESTAMP;
  profiling_desc->flags = static_cast<ze_event_pool_flag_t>(flags);

  *(params->pdesc) = profiling_desc;
  *instance_data = profiling_desc;
}

static void OnExitEventPoolCreate(ze_event_pool_create_params_t *params,
                                  ze_result_t result,
                                  void *global_data,
                                  void **instance_data) {
  ze_event_pool_desc_t* desc =
    static_cast<ze_event_pool_desc_t*>(*instance_data);
  if (desc != nullptr) {
    delete desc;
  }
}

static void ProcessEvent(size_t id) {
  assert(state != nullptr);
  assert(id >= 0 && id < state->activity_event_list.size());

  ActivityEventInfo info = state->activity_event_list[id];

/*
  std::cout <<"ProcessEvent: processing id = "<< id <<" info.name = "<<info.name ;
  std::cout <<" activity_event_list.size() = "<< state->activity_event_list.size()<<std::endl;
*/
  state->activity_event_list[id] =
    state->activity_event_list[state->activity_event_list.size() - 1];
  state->activity_event_list.pop_back();
 // std::cout <<"After pop_back: activity_event_list.size() = "<< state->activity_event_list.size()<<std::endl;

  ze_result_t status = ZE_RESULT_SUCCESS;
  ze_device_properties_t props{};
  props.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
  status = zeDeviceGetProperties(state->device, &props);
  assert(status == ZE_RESULT_SUCCESS);

  status = zeEventQueryStatus(info.event);
  if (status == ZE_RESULT_SUCCESS) {
    uint64_t start = 0;
    status = zeEventGetTimestamp(
        info.event, ZE_EVENT_TIMESTAMP_GLOBAL_START, &start);
    assert(status == ZE_RESULT_SUCCESS);
    if (TauEnv_get_verbose()) {
      std::cout <<"TAU Kernel start: "<<info.name<<" timestamp: "<< start * props.timerResolution<< " timerResolution = "<<props.timerResolution<< " task id = "<< state->tau_task_id<<std::endl;
    }

    uint64_t current_timestamp = start * props.timerResolution; 
    if (state->last_timestamp == 0) {
      // We need to calculate the offset from the start. 
      uint64_t current = TauTraceGetTimeStamp(state->tau_task_id);
      uint64_t offset =  current - state->earliest_timestamp;    
      if (TauEnv_get_verbose()) {
        std::cout <<"crrent_timestamp     (C++)= "<<current_timestamp<<std::endl; 
        std::cout <<"current         (RtsLayer)= "<<(uint64_t)current<<std::endl; 
        std::cout <<"state->earliest_timestamp = "<<state->earliest_timestamp<<std::endl; 
        std::cout <<"Offset                    = "<<offset<<std::endl; 
      }
      // Now, to use this offset, we need to subtract the current_timestamp/1e3 to get it 
      // to align with the host clock. 
      state->offset_timestamp -= (uint64_t) (current_timestamp/1e3); 
    }
    Tau_metric_set_synchronized_gpu_timestamp(state->tau_task_id, current_timestamp/1e3);
    TAU_START_TASK(info.name.c_str(), state->tau_task_id);

    uint64_t end = 0;
    status = zeEventGetTimestamp(
        info.event, ZE_EVENT_TIMESTAMP_GLOBAL_END, &end);
    assert(status == ZE_RESULT_SUCCESS);
    current_timestamp = end * props.timerResolution;
    if (TauEnv_get_verbose()) {
      std::cout <<"TAU Kernel stop : "<<info.name<<" timestamp: "<< current_timestamp << " task id = "<<state->tau_task_id<<std::endl;
    }
    Tau_metric_set_synchronized_gpu_timestamp(state->tau_task_id, current_timestamp/1e3);
    TAU_STOP_TASK(info.name.c_str(), state->tau_task_id);
    if (current_timestamp > state->last_timestamp) {
      state->last_timestamp = current_timestamp; 
    } // max timestamp seen so far. 

    auto& item = state->activity_time_map[info.name];
    item.first += (end - start) * props.timerResolution;
    item.second += 1;
  }

  if (info.event_type == EVENT_TYPE_TOOL) {
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zeEventDestroy(info.event);
    assert(status == ZE_RESULT_SUCCESS);
    status = zeEventPoolDestroy(info.event_pool);
    assert(status == ZE_RESULT_SUCCESS);
  }
}

static void OnEnterEventDestroy(ze_event_destroy_params_t *params,
                                ze_result_t result,
                                void *global_data,
                                void **instance_data) {
  assert(state != nullptr);
  state->klock.lock();
  for (size_t i = 0; i < state->activity_event_list.size(); ++i) {
    if (state->activity_event_list[i].event == *(params->phEvent)) {
      assert(state->activity_event_list[i].event_type == EVENT_TYPE_USER);
      ProcessEvent(i);
      break;
    }
  }
  state->klock.unlock();
}

static void OnEnterEventHostReset(ze_event_host_reset_params_t *params,
                                  ze_result_t result,
                                  void *global_data,
                                  void **instance_data) {
  assert(state != nullptr);
  state->klock.lock();
  for (size_t i = 0; i < state->activity_event_list.size(); ++i) {
    if (state->activity_event_list[i].event == *(params->phEvent)) {
      assert(state->activity_event_list[i].event_type == EVENT_TYPE_USER);
      ProcessEvent(i);
      break;
    }
  }
  state->klock.unlock();
}

static ze_event_handle_t OnEnterActivitySubmit(
    std::string name, ze_event_handle_t event, void **instance_data) {
  ze_result_t status = ZE_RESULT_SUCCESS;
  ActivityEventInfo* info = new ActivityEventInfo;
  assert(info != nullptr);
  info->name = name;

  if (event == nullptr) {
    ze_event_pool_desc_t event_pool_desc = {
      ZE_EVENT_POOL_DESC_VERSION_CURRENT,
      ZE_EVENT_POOL_FLAG_TIMESTAMP,
      1 };
    ze_event_pool_handle_t event_pool = nullptr;
    status = zeEventPoolCreate(state->driver, &event_pool_desc,
                               1, &(state->device), &event_pool);
    assert(status == ZE_RESULT_SUCCESS);

    ze_event_desc_t event_desc = {
      ZE_EVENT_DESC_VERSION_CURRENT,
      0,
      ZE_EVENT_SCOPE_FLAG_HOST,
      ZE_EVENT_SCOPE_FLAG_HOST };
    zeEventCreate(event_pool, &event_desc, &event);
    assert(status == ZE_RESULT_SUCCESS);

    info->event_pool = event_pool;
    info->event_type = EVENT_TYPE_TOOL;
  } else {
    info->event_pool = nullptr;
    info->event_type = EVENT_TYPE_USER;
  }

  info->event = event;
  *instance_data = info;
  return event;
}

static void OnEnterCommandListAppendLaunchKernel(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result, void* global_data, void** instance_data) {
  assert(state != nullptr);
  
  if (*(params->phKernel) ==  nullptr) {
    return;
  }

  state->klock.lock();
  assert(state->kernel_name_map.count(*(params->phKernel)) == 1);
  std::string kernel_name = state->kernel_name_map[*(params->phKernel)];
  state->klock.unlock();

  *(params->phSignalEvent) = OnEnterActivitySubmit(
      kernel_name, *(params->phSignalEvent), instance_data);
}

static void OnEnterCommandListAppendMemoryCopy(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result, void* global_data, void** instance_data) {
  *(params->phEvent) = OnEnterActivitySubmit(
      "<MemoryCopy>", *(params->phEvent), instance_data);
}

static void OnEnterCommandListAppendBarrier(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result, void* global_data, void** instance_data) {
  *(params->phSignalEvent) = OnEnterActivitySubmit(
      "<Barrier>", *(params->phSignalEvent), instance_data);
}

static void OnExitActivitySubmit(void **instance_data, ze_result_t result) {
  assert(state != nullptr);

  ActivityEventInfo* info = static_cast<ActivityEventInfo*>(*instance_data);
  if (info == nullptr) {
    return;
  }

  if (result != ZE_RESULT_SUCCESS && info != nullptr) {
    if (info->event_type == EVENT_TYPE_TOOL) {
      ze_result_t status = ZE_RESULT_SUCCESS;
      status = zeEventDestroy(info->event);
      assert(status == ZE_RESULT_SUCCESS);
      status = zeEventPoolDestroy(info->event_pool);
      assert(status == ZE_RESULT_SUCCESS);
    }
  } else {
    state->klock.lock();
    state->activity_event_list.push_back(*info);
    state->klock.unlock();
  }

  delete info;
}

static void OnExitCommandListAppendLaunchKernel(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result, void* global_data, void** instance_data) {
  assert(*(params->phSignalEvent) != nullptr);
  OnExitActivitySubmit(instance_data, result);
}

static void OnExitCommandListAppendMemoryCopy(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result, void* global_data, void** instance_data) {
  OnExitActivitySubmit(instance_data, result);
}

static void OnExitCommandListAppendBarrier(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result, void* global_data, void** instance_data) {
  OnExitActivitySubmit(instance_data, result);
}

static void EnableTracing(ze_driver_handle_t driver,
                          zet_tracer_handle_t& tracer) {
  assert(driver != nullptr);
  
  ze_result_t status = ZE_RESULT_SUCCESS;
  zet_tracer_desc_t tracer_desc = {};
  tracer_desc.version = ZET_TRACER_DESC_VERSION_CURRENT;
  tracer_desc.pUserData = nullptr;

  status = zetTracerCreate(driver, &tracer_desc, &tracer);
  if (status != ZE_RESULT_SUCCESS) {
    std::cout <<
      "[WARNING] Unable to create Level Zero tracer for target driver" <<
      std::endl;
    return;
  }

  zet_core_callbacks_t prologue_callbacks = {};
  prologue_callbacks.Event.pfnDestroyCb = OnEnterEventDestroy;
  prologue_callbacks.Event.pfnHostResetCb = OnEnterEventHostReset;
  prologue_callbacks.EventPool.pfnCreateCb = OnEnterEventPoolCreate;
  prologue_callbacks.CommandList.pfnAppendLaunchKernelCb =
    OnEnterCommandListAppendLaunchKernel;
  prologue_callbacks.CommandList.pfnAppendMemoryCopyCb =
    OnEnterCommandListAppendMemoryCopy;
  prologue_callbacks.CommandList.pfnAppendBarrierCb =
    OnEnterCommandListAppendBarrier;

  zet_core_callbacks_t epilogue_callbacks = {};
  epilogue_callbacks.Kernel.pfnCreateCb = OnExitKernelCreate;
  epilogue_callbacks.Kernel.pfnDestroyCb = OnExitKernelDestroy;
  epilogue_callbacks.EventPool.pfnCreateCb = OnExitEventPoolCreate;
  epilogue_callbacks.CommandList.pfnAppendLaunchKernelCb =
    OnExitCommandListAppendLaunchKernel;
  epilogue_callbacks.CommandList.pfnAppendMemoryCopyCb =
    OnExitCommandListAppendMemoryCopy;
  epilogue_callbacks.CommandList.pfnAppendBarrierCb =
    OnExitCommandListAppendBarrier;

  status = zetTracerSetPrologues(tracer, &prologue_callbacks);
  assert(status == ZE_RESULT_SUCCESS);
  status = zetTracerSetEpilogues(tracer, &epilogue_callbacks);
  assert(status == ZE_RESULT_SUCCESS);
  status = zetTracerSetEnabled(tracer, true);
  assert(status == ZE_RESULT_SUCCESS);
}

static void DisableTracing(zet_tracer_handle_t tracer) {
  if (tracer != nullptr) {
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zetTracerSetEnabled(tracer, false);
    assert(status == ZE_RESULT_SUCCESS);
    status = zetTracerDestroy(tracer);
    assert(status == ZE_RESULT_SUCCESS);
  }
}

static void ProcessResults() {
  assert(state != nullptr);

  state->klock.lock();
/* std::cout <<"state->activity_event_list.size() = "<< state->activity_event_list.size()<<std::endl;
  for(int i = 0; i < state->activity_event_list.size(); i++) {
    ActivityEventInfo info = state->activity_event_list[i];
    ze_device_properties_t props{}; 
    props.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
    zeDeviceGetProperties(state->device, &props);
    zeEventQueryStatus(info.event);
    uint64_t start = 0;
    zeEventGetTimestamp(
        info.event, ZE_EVENT_TIMESTAMP_CONTEXT_START, &start);
    uint64_t end = 0;
    zeEventGetTimestamp(
        info.event, ZE_EVENT_TIMESTAMP_CONTEXT_END, &end);
    auto& item = state->activity_time_map[info.name];
    //std::cout <<"Info name = "<<info.name<< " start = "<< start << " end = " << end<< std::endl; 
  }
  std::cout <<"state->activity_event_list.size() = "<< state->activity_event_list.size()<<std::endl;
*/


  while (state->activity_event_list.size() > 0) {
    ProcessEvent(state->activity_event_list.size() - 1);
  }
/* std::cout <<"state->activity_event_list.size() = "<< state->activity_event_list.size()<<std::endl;

  std::set<std::pair< std::string, std::pair<uint64_t, int> >,
           utils::Comparator> kset(
    state->activity_time_map.begin(), state->activity_time_map.end());
//  for (auto& pair : kset) {
  std::set<std::pair< std::string, std::pair<uint64_t, int> >,
           utils::Comparator>::iterator p;
  for (p = kset.begin(); p != kset.end(); p++) {
    std::cout << "| " << std::left << std::setw(48) << p->first << " | " <<
      std::right << std::setw(10) << p->second.second << " | " <<
      std::setw(14) << std::setprecision(2) <<
      std::fixed << p->second.first / static_cast<float>(NSEC_IN_MSEC) <<
      " |" << std::endl;
  }
*/
  state->klock.unlock();
}

static void PrintResults() {
  assert(state != nullptr);
  if (!TauEnv_get_verbose()) return;

  if (state->activity_time_map.size() == 0) {
    return;
  }
  if (state->function_time_map.size() == 0) {
    return;
  }

  
  std::cout << kLine << std::endl;
  std::cout << kHeader << std::endl;
  std::cout << kLine << std::endl;

  std::set<std::pair< std::string, std::pair<uint64_t, int> >,
           utils::Comparator> kset(
    state->activity_time_map.begin(), state->activity_time_map.end());
  for (auto& pair : kset) {
    std::cout << "| " << std::left << std::setw(48) << pair.first << " | " <<
      std::right << std::setw(10) << pair.second.second << " | " <<
      std::setw(14) << std::setprecision(2) <<
      std::fixed << pair.second.first / static_cast<float>(NSEC_IN_MSEC) <<
      " |" << std::endl;
  }

  std::set<std::pair< std::string, std::pair<double, int> >,
           utils::Comparator> fset(
      state->function_time_map.begin(), state->function_time_map.end());
  for (auto& pair : fset) {
    std::cout << "| " << std::left << std::setw(48) << pair.first << " | " <<
      std::right << std::setw(10) << pair.second.second << " | " <<
      std::setw(14) << std::setprecision(2) <<
      std::fixed << 1000 * pair.second.first << " |" << std::endl;
  }



  std::cout << kLine << std::endl;
  std::cout << "[INFO] Job is successfully completed" << std::endl;
}

static ZeTracer* CreateTracer(ze_device_type_t type) {
  ze_device_handle_t device = nullptr;
  ze_driver_handle_t driver = nullptr;

  utils::ze::GetIntelDeviceAndDriver(type, device, driver);
  if (device == nullptr || driver == nullptr) {
    std::cout << "[WARNING] Unable to find target" <<
      " device for tracing" << std::endl;
    return nullptr;
  }

  ZeTracer* tracer = new ZeTracer(driver, Callback, nullptr);
  if (tracer == nullptr || !tracer->IsValid()) {
    std::cout << "[WARNING] Unable to create Level Zero tracer for" <<
      " target driver" << std::endl;
    if (tracer != nullptr) {
      delete tracer;
      tracer = nullptr;
    }
    return nullptr;
  }

  for (int i = 0; i < ZE_FUNCTION_COUNT; ++i) {
    bool set = tracer->SetTracingFunction(static_cast<function_id_t>(i));
    assert(set);
  }

  bool enabled = tracer->Enable();
  assert(enabled);

  return tracer;
}

static void DestroyTracer(ZeTracer* tracer) {
  if (tracer != nullptr) {
    bool disabled = tracer->Disable();
    assert(disabled);
    delete tracer;
  }
}


void EnableProfiling() {
  TAU_VERBOSE("TAU L0: EnableProfiling\n");
  if (!TauEnv_get_level_zero_enable_api_tracing()) { 
  // ZE_ENABLE_API_TRACING not set
    if (RtsLayer::myNode() == 0) {
      printf("*****************************************************************************************************************\n");
      printf("TAU: WARNING: Please use tau_exec -oneapi to launch the application to generate Intel OneAPI Level Zero TAU data.\n");
      printf("*****************************************************************************************************************\n");
    }
    return ; 
  }
  assert(state == nullptr);
  state = new GlobalToolState;
  assert(state != nullptr);

  ze_result_t status = ZE_RESULT_SUCCESS;
  status = zeInit(ZE_INIT_FLAG_NONE);
  assert(status == ZE_RESULT_SUCCESS);
  status = zetInit(ZE_INIT_FLAG_NONE);
  assert(status == ZE_RESULT_SUCCESS);

  utils::ze::GetIntelDeviceAndDriver(
      ZE_DEVICE_TYPE_GPU, state->device, state->driver);
  if (state->device == nullptr || state->driver == nullptr) {
    std::cout << "[WARNING] Unable to find target device" << std::endl;
    return;
  }

  EnableTracing(state->driver, state->tracer);

  state->gpu_tracer = CreateTracer(ZE_DEVICE_TYPE_GPU);
  TAU_CREATE_TASK(state->tau_task_id);
  state->last_timestamp = 0; 
  state->offset_timestamp = 0; 

  state->earliest_timestamp = TauTraceGetTimeStamp(state->tau_task_id);
  //std::cout <<"START TIME = "<<state->earliest_timestamp<<std::endl; 
  Tau_metric_set_synchronized_gpu_timestamp(state->tau_task_id, state->earliest_timestamp);
  Tau_create_top_level_timer_if_necessary_task(state->tau_task_id);

}

void DisableProfiling() {
  TAU_VERBOSE("TAU L0: DisableProfiling\n");
  if (!TauEnv_get_level_zero_enable_api_tracing()) { 
    return; 
  }
  assert(state != nullptr);
  if (state->tracer != nullptr) {
    DisableTracing(state->tracer);
    ProcessResults();
    DestroyTracer(state->gpu_tracer);
    PrintResults();
  }
  uint64_t timestamp = state->last_timestamp;
  Tau_metric_set_synchronized_gpu_timestamp(state->tau_task_id, timestamp/1e3);
  TAU_STOP_TASK(".TAU application", state->tau_task_id);
  delete state;
}

// preload.cc 
#if defined(__gnu_linux__)

#include <dlfcn.h>

typedef void (*Exit)(int status) __attribute__ ((noreturn));
typedef int (*Main)(int argc, char** argv, char** envp);
typedef int (*Fini)(void);
typedef int (*LibcStartMain)(Main main, int argc, char** argv, Main init,
                             Fini fini, Fini rtld_fini, void *stack_end);

// Pointer to original application main() function
Main original_main = nullptr;

extern "C" int HookedMain(int argc, char **argv, char **envp) {
  EnableProfiling();
  int return_code = original_main(argc, argv, envp);
  DisableProfiling();
  return return_code;
}

extern "C" int __libc_start_main(Main main,
                                 int argc,
                                 char** argv,
                                 Main init,
                                 Fini fini,
                                 Fini rtld_fini,
                                 void* stack_end) {
  original_main = main;
  LibcStartMain original =
    (LibcStartMain)dlsym(RTLD_NEXT, "__libc_start_main");
  return original(HookedMain, argc, argv, init, fini, rtld_fini, stack_end);
}

extern "C" void exit(int status) {
  Exit original = (Exit)dlsym(RTLD_NEXT, "exit");
  DisableProfiling();
  original(status);
}

#else
#error not supported
#endif

