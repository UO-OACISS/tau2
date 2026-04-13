//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef PTI_TOOLS_UNITRACE_LEVEL_ZERO_METRICS_H_
#define PTI_TOOLS_UNITRACE_LEVEL_ZERO_METRICS_H_


#include <string.h>

#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <level_zero/layers/zel_tracing_api.h>
#include "ze_loader.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <atomic>
#include <sstream>
#include <thread>

#include "unimemory.h"
#include "utils.h"
#include "utils_ze.h"
#include "pti_assert.h"
#include <inttypes.h>

constexpr static uint32_t max_metric_samples = 32768;

#define MAX_METRIC_BUFFER  (8ULL * 1024ULL * 1024ULL)

inline std::string GetMetricUnits(const char* units) {
  PTI_ASSERT(units != nullptr);

  std::string result = units;
  if (result.find("null") != std::string::npos) {
    result = "";
  } else if (result.find("percent") != std::string::npos) {
    result = "%";
  }

  return result;
}

inline uint32_t GetMetricId(const std::vector<std::string>& metric_list, const std::string& metric_name) {
  PTI_ASSERT(!metric_list.empty());
  PTI_ASSERT(!metric_name.empty());

  for (size_t i = 0; i < metric_list.size(); ++i) {
    if (metric_list[i].find(metric_name) == 0) {
      return i;
    }
  }

  return metric_list.size();
}

enum ZeProfilerState {
  PROFILER_DISABLED = 0,
  PROFILER_ENABLED = 1
};

struct ZeDeviceDescriptor {
  ze_device_handle_t device_;
  ze_device_handle_t parent_device_;
  uint64_t device_timer_frequency_;
  uint64_t device_timer_mask_;
  uint64_t metric_timer_frequency_;
  uint64_t metric_timer_mask_;
  ze_driver_handle_t driver_;
  ze_context_handle_t context_;
  int32_t device_id_;
  int32_t subdevice_id_;
  int32_t num_sub_devices_;
  zet_metric_group_handle_t metric_group_;
  ze_pci_ext_properties_t pci_properties_;
  std::thread *profiling_thread_;
  std::atomic<ZeProfilerState> profiling_state_;
  std::string metric_file_name_;
  std::ofstream metric_file_stream_;
  bool stall_sampling_;
};

class ZeMetricProfiler {
 public:
  static ZeMetricProfiler* Create() {
    ZeMetricProfiler* profiler = new ZeMetricProfiler();
    UniMemory::ExitIfOutOfMemory((void *)(profiler));

    profiler->StartProfilingMetrics();

    return profiler;
  }

  ~ZeMetricProfiler() {
    printf("\n !! StopProfilingMetrics \n");
    StopProfilingMetrics();
  }

  ZeMetricProfiler(const ZeMetricProfiler& that) = delete;
  ZeMetricProfiler& operator=(const ZeMetricProfiler& that) = delete;

 private:
  std::set<int> devices_to_sample_;

  ZeMetricProfiler() {

    EnumerateDevices();
  }

  void EnumerateDevices() {

    std::string metric_group = "EuStallSampling";

    int32_t global_dev_cnt = -1;
    auto drivers = GetDriverList();
    for (auto driver : drivers) {
      ze_context_handle_t context = nullptr;
      ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};

      auto status = ZE_FUNC(zeContextCreate)(driver, &cdesc, &context);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      metric_contexts_.push_back(context);

      auto devices = GetDeviceList(driver);
      for (auto device : devices) {
        global_dev_cnt++;

        // Skip devices not found in the list, whenever the list is given
        if (!devices_to_sample_.empty()) {
          if (devices_to_sample_.find(global_dev_cnt) == devices_to_sample_.end()) {
            continue;
          }
        }

        auto sub_devices = GetSubDeviceList(device);

        ZeDeviceDescriptor *desc = new ZeDeviceDescriptor;
        UniMemory::ExitIfOutOfMemory((void *)(desc));

        desc->stall_sampling_ = true;

        desc->device_ = device;
        desc->device_id_ = global_dev_cnt;
        desc->parent_device_ = nullptr;
        desc->subdevice_id_ = -1;     // not a subdevice
        desc->num_sub_devices_ = sub_devices.size();

        ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, };
        ze_result_t status = ZE_FUNC(zeDeviceGetProperties)(device, &props);
        PTI_ASSERT(status == ZE_RESULT_SUCCESS);
        PTI_ASSERT(props.timerResolution != 0);
        PTI_ASSERT(props.kernelTimestampValidBits != 0);

        desc->device_timer_frequency_ = props.timerResolution;
        desc->device_timer_mask_ = (props.kernelTimestampValidBits == 64) ? (std::numeric_limits<uint64_t>::max)() : ((1ull << props.kernelTimestampValidBits) - 1ull);

        ze_pci_ext_properties_t pci_device_properties;
        status = ZE_FUNC(zeDevicePciGetPropertiesExt)(device, &pci_device_properties);
        PTI_ASSERT(status == ZE_RESULT_SUCCESS);
        desc->pci_properties_ = pci_device_properties;

        zet_metric_group_handle_t group = FindMetricGroup (device, metric_group, ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED);
        if (group == nullptr) {
          std::cerr << "[ERROR] Invalid metric group " << metric_group << std::endl;
          exit(-1);
        }

        zet_metric_group_properties_t metric_group_prop;
        metric_group_prop.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
        metric_group_prop.pNext = nullptr;

        zet_metric_global_timestamps_resolution_exp_t metrics_ts_prop;
        metrics_ts_prop.stype = ZET_STRUCTURE_TYPE_METRIC_GLOBAL_TIMESTAMPS_RESOLUTION_EXP;
        metrics_ts_prop.pNext = nullptr;
        metric_group_prop.pNext = &metrics_ts_prop;
        status = ZE_FUNC(zetMetricGroupGetProperties)(group, &metric_group_prop);
        PTI_ASSERT(status == ZE_RESULT_SUCCESS);
        PTI_ASSERT(metrics_ts_prop.timerResolution != 0);
        PTI_ASSERT(metrics_ts_prop.timestampValidBits != 0);

        desc->metric_timer_frequency_ = metrics_ts_prop.timerResolution;
        // specially handling certain older devices
        uint32_t device_gen = (props.deviceId & 0xFF00);
        if (device_gen == 0x0B00) {
          if (desc->metric_timer_frequency_ > desc->device_timer_frequency_) {
            desc->metric_timer_frequency_ = desc->device_timer_frequency_;
          }

          if (metrics_ts_prop.timestampValidBits == props.kernelTimestampValidBits) {
            metrics_ts_prop.timestampValidBits = metrics_ts_prop.timestampValidBits - 1;
          }
        }
        desc->metric_timer_mask_ = (metrics_ts_prop.timestampValidBits == 64) ? (std::numeric_limits<uint64_t>::max)() : ((1ull << metrics_ts_prop.timestampValidBits) - 1ull);

        desc->driver_ = driver;
        desc->context_ = context;
        desc->metric_group_ = group;

        desc->profiling_thread_ = nullptr;
        desc->profiling_state_.store(PROFILER_DISABLED, std::memory_order_release);

        device_descriptors_.insert({device, desc});

        for (size_t j = 0; j < sub_devices.size(); j++) {
          ZeDeviceDescriptor *sub_desc = new ZeDeviceDescriptor;
          UniMemory::ExitIfOutOfMemory((void *)(sub_desc));

          sub_desc->stall_sampling_ = true;

          sub_desc->device_ = sub_devices[j];
          sub_desc->device_id_ = global_dev_cnt;           // subdevice
          sub_desc->parent_device_ = device;
          sub_desc->subdevice_id_ = j;                     // a subdevice
          sub_desc->num_sub_devices_ = 0;

          sub_desc->driver_ = driver;
          sub_desc->context_ = context;
          sub_desc->metric_group_ = group;

          sub_desc->device_timer_frequency_ = desc->device_timer_frequency_;
          sub_desc->device_timer_mask_ = desc->device_timer_mask_;

          sub_desc->metric_timer_frequency_ = desc->metric_timer_frequency_;
          sub_desc->metric_timer_mask_ = desc->metric_timer_mask_;

          ze_pci_ext_properties_t pci_device_properties;
          ze_result_t status = ZE_FUNC(zeDevicePciGetPropertiesExt)(sub_devices[j], &pci_device_properties);
          PTI_ASSERT(status == ZE_RESULT_SUCCESS);

          sub_desc->pci_properties_ = pci_device_properties;

          sub_desc->driver_ = driver;
          sub_desc->context_ = context;
          sub_desc->metric_group_ = group;

          sub_desc->profiling_thread_ = nullptr;
          sub_desc->profiling_state_.store(PROFILER_DISABLED, std::memory_order_release);

          device_descriptors_.insert({sub_devices[j], sub_desc});
        } // subdevices list
      } // devices list
    } // drivers list
  }

  int GetDeviceId(ze_device_handle_t sub_device) const {
    if (auto it = device_descriptors_.find(sub_device); it != device_descriptors_.end()) {
      return it->second->device_id_;
    }
    return -1;
  }

  int GetSubDeviceId(ze_device_handle_t sub_device) const {
    if (auto it = device_descriptors_.find(sub_device); it != device_descriptors_.end()) {
      return it->second->subdevice_id_;
    }
    return -1;
  }

  ze_device_handle_t GetParentDevice(ze_device_handle_t sub_device) const {
    if (auto it = device_descriptors_.find(sub_device); it != device_descriptors_.end()) {
      return it->second->parent_device_;
    }
    return nullptr;
  }

  void StartProfilingMetrics(void) {

    for (auto [handle, device] : device_descriptors_) {
      if (device->parent_device_ != nullptr) {
        // subdevice
        continue;
      }
      device->profiling_thread_ = new std::thread(MetricProfilingThread, device);
      while (device->profiling_state_.load(std::memory_order_acquire) != PROFILER_ENABLED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void StopProfilingMetrics() {
    for (auto [handle, device] : device_descriptors_) {
      if (device->parent_device_ != nullptr) {
        // subdevice
        continue;
      }
      PTI_ASSERT(device->profiling_thread_ != nullptr);
      PTI_ASSERT(device->profiling_state_ == PROFILER_ENABLED);
      device->profiling_state_.store(PROFILER_DISABLED, std::memory_order_release);
      device->profiling_thread_->join();
      delete device->profiling_thread_;
      device->profiling_thread_ = nullptr;
      device->metric_file_stream_.close();
    }
  }

  struct ZeKernelInfo {
    int32_t subdevice_id;
    uint64_t global_instance_id;
    uint64_t metric_start;
    uint64_t metric_end;
    std::string kernel_name;
  };

  static bool CompareInterval(ZeKernelInfo& iv1, ZeKernelInfo& iv2) {
    return (iv1.metric_start < iv2.metric_start);
  }

  static std::vector<std::string> GetMetricNames(zet_metric_group_handle_t group) {
    PTI_ASSERT(group != nullptr);

    uint32_t metric_count = GetMetricCount(group);
    PTI_ASSERT(metric_count > 0);

    std::vector<zet_metric_handle_t> metrics(metric_count);
    ze_result_t status = ZE_FUNC(zetMetricGet)(group, &metric_count, metrics.data());
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(metric_count == metrics.size());

    std::vector<std::string> names;
    for (auto metric : metrics) {
      zet_metric_properties_t metric_props{
          ZET_STRUCTURE_TYPE_METRIC_PROPERTIES, };
      status = ZE_FUNC(zetMetricGetProperties)(metric, &metric_props);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);

      std::string units = GetMetricUnits(metric_props.resultUnits);
      std::string name = metric_props.name;
      if (!units.empty()) {
        name += "[" + units + "]";
      }
      names.push_back(std::move(name));
    }

    return names;
  }

 private:

    static std::string PrintTypedValue(const zet_typed_value_t& typed_value) {
    switch (typed_value.type) {
      case ZET_VALUE_TYPE_UINT32:
        return std::to_string(typed_value.value.ui32);
      case ZET_VALUE_TYPE_UINT64:
        return std::to_string(typed_value.value.ui64);
      case ZET_VALUE_TYPE_FLOAT32:
        return std::to_string(typed_value.value.fp32);
      case ZET_VALUE_TYPE_FLOAT64:
        return std::to_string(typed_value.value.fp64);
      case ZET_VALUE_TYPE_BOOL8:
        return std::to_string(static_cast<uint32_t>(typed_value.value.b8));
      default:
        PTI_ASSERT(0);
        break;
    }
    return "";  // in case of error returns empty string.
  }

  inline static std::string GetMetricUnits(const char* units) {
    PTI_ASSERT(units != nullptr);

    std::string result = units;
    if (result.find("null") != std::string::npos) {
      result = "";
    } else if (result.find("percent") != std::string::npos) {
      result = "%";
    }

    return result;
  }

  static uint32_t GetMetricCount(zet_metric_group_handle_t group) {
    PTI_ASSERT(group != nullptr);

    zet_metric_group_properties_t group_props{};
    group_props.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
    ze_result_t status = ZE_FUNC(zetMetricGroupGetProperties)(group, &group_props);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    return group_props.metricCount;
  }

  static std::vector<std::string> GetMetricList(zet_metric_group_handle_t group) {
    PTI_ASSERT(group != nullptr);

    uint32_t metric_count = GetMetricCount(group);
    PTI_ASSERT(metric_count > 0);

    std::vector<zet_metric_handle_t> metric_list(metric_count);
    ze_result_t status = ZE_FUNC(zetMetricGet)(group, &metric_count, metric_list.data());
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(metric_count == metric_list.size());

    std::vector<std::string> name_list;
    for (auto metric : metric_list) {
      zet_metric_properties_t metric_props{
          ZET_STRUCTURE_TYPE_METRIC_PROPERTIES, };
      status = ZE_FUNC(zetMetricGetProperties)(metric, &metric_props);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);

      std::string units = GetMetricUnits(metric_props.resultUnits);
      std::string name = metric_props.name;
      if (!units.empty()) {
        name += "[" + units + "]";
      }
      name_list.push_back(std::move(name));
    }

    return name_list;
  }

  static uint64_t ReadMetrics(zet_metric_streamer_handle_t streamer, uint8_t* storage, size_t ssize) {
      ze_result_t status = ZE_RESULT_SUCCESS;
      size_t data_size = ssize;
      status = ZE_FUNC(zetMetricStreamerReadData)(streamer, UINT32_MAX, &data_size, storage);
      if (status == ZE_RESULT_WARNING_DROPPED_DATA) {
          std::cerr << "[WARNING] Metric samples dropped." << std::endl;
      }
      else if (status != ZE_RESULT_SUCCESS) {
          std::cerr << "[ERROR] zetMetricStreamerReadData failed with error code: "
              << static_cast<std::size_t>(status) << std::endl;
          PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      }
      return data_size;
  }

  static uint64_t EventBasedReadMetrics(ze_event_handle_t event, zet_metric_streamer_handle_t  streamer, uint8_t *storage, size_t ssize) {
    ze_result_t status = ZE_RESULT_SUCCESS;

    //status = ZE_FUNC(zeEventHostSynchronize)(event, 0);
    status = ZE_FUNC(zeEventQueryStatus)(event);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS || status == ZE_RESULT_NOT_READY);
    if (status == ZE_RESULT_SUCCESS) {
      status = ZE_FUNC(zeEventHostReset)(event);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    else {
      // if (status == ZE_RESULT_NOT_READY)
      return 0;
    }

    return ReadMetrics(streamer, storage, ssize);
  }

  static void MetricProfilingThread(ZeDeviceDescriptor *desc) {

    ze_result_t status = ZE_RESULT_SUCCESS;

    ze_context_handle_t context = desc->context_;
    ze_device_handle_t device = desc->device_;
    zet_metric_group_handle_t group = desc->metric_group_;

    status = ZE_FUNC(zetContextActivateMetricGroups)(context, device, 1, &group);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    ze_event_pool_handle_t event_pool = nullptr;
    ze_event_pool_desc_t event_pool_desc = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr, ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
    status = ZE_FUNC(zeEventPoolCreate)(context, &event_pool_desc, 1, &device, &event_pool);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    ze_event_handle_t event = nullptr;
    ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    status = ZE_FUNC(zeEventCreate)(event_pool, &event_desc, &event);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    zet_metric_streamer_handle_t streamer = nullptr;
    uint32_t interval = std::stoi(utils::GetEnv("UNITRACE_SamplingInterval")) * 1000;  // convert us to ns

    zet_metric_streamer_desc_t streamer_desc = {ZET_STRUCTURE_TYPE_METRIC_STREAMER_DESC, nullptr, max_metric_samples, interval};
    status = ZE_FUNC(zetMetricStreamerOpen)(context, device, group, &streamer_desc, event, &streamer);
    if (status != ZE_RESULT_SUCCESS) {
        std::cerr << "[WARNING] Unable to open metric streamer for sampling (status = 0x" << std::hex << status << std::dec << "). The sampling interval might be too small or another sampling instance is active." << std::endl;
    #ifndef _WIN32
        std::cerr << "[INFO] Please also make sure /proc/sys/dev/i915/perf_stream_paranoid or /proc/sys/dev/xe/observation_paranoid is set to 0." << std::endl;
    #endif /* _WIN32 */

      status = ZE_FUNC(zeEventDestroy)(event);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);

      status = ZE_FUNC(zeEventPoolDestroy)(event_pool);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);

      // set state to enabled to let the parent thread continue
      desc->profiling_state_.store(PROFILER_ENABLED, std::memory_order_release);
      return;
    }

    std::vector<std::string> metrics_list;
    metrics_list = GetMetricList(group);
    PTI_ASSERT(!metrics_list.empty());
    printf("!! metrics_list ");
    for(auto metric : metrics_list )
      printf(" %s", metric.c_str());
    printf("\n");

    auto *raw_metrics = static_cast<uint8_t*>(malloc(sizeof(uint8_t)*MAX_METRIC_BUFFER));
    UniMemory::ExitIfOutOfMemory((void *)raw_metrics);

    auto dump_metrics = [](uint8_t *buffer, uint64_t size, std::ofstream *f, zet_metric_group_handle_t group) -> bool {

      std::vector<std::string> metric_list;
      metric_list = GetMetricList(group);
      PTI_ASSERT(!metric_list.empty());

      uint32_t ip_idx = GetMetricId(metric_list, "IP");
      if (ip_idx >= metric_list.size()) {
        // no IP metric
        return false;
      }

      constexpr uint32_t max_num_of_stall_types = 16;
      uint32_t num_samples = 0;
      uint32_t num_metrics = 0;
      auto status = ZE_FUNC(zetMetricGroupCalculateMultipleMetricValuesExp)(
              group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
              size, buffer, &num_samples, &num_metrics,
              nullptr, nullptr);
      if ((status != ZE_RESULT_SUCCESS) || (num_samples == 0) || (num_metrics == 0)) {
        std::cerr << "[WARNING size] Unable to calculate metrics (status = 0x" << std::hex << status << std::dec << ") num_samples = " << num_samples << " num_metrics = " << num_metrics << std::endl;
        return false;
      }

      std::vector<uint32_t> samples(num_samples);
      std::vector<zet_typed_value_t> metrics(num_metrics);

      status = ZE_FUNC(zetMetricGroupCalculateMultipleMetricValuesExp)(
        group, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
        size, buffer, &num_samples, &num_metrics,
        samples.data(), metrics.data());

      if ((status != ZE_RESULT_SUCCESS) && (status != ZE_RESULT_WARNING_DROPPED_DATA)) {
        std::cerr << "[WARNING samples] Unable to calculate metrics (status = 0x" << std::hex << status << std::dec << ") num_samples = " << num_samples << " num_metrics = " << num_metrics << std::endl;
        return false;
      }
      
      const zet_typed_value_t *value = metrics.data();
      for (uint32_t i = 0; i < num_samples; ++i) {
        std::string str;

        uint32_t size = samples[i];
        for (uint32_t j = 0; j < size; j += metric_list.size()) {
          uint64_t ip;
          ip = (value[j + 0].value.ui64 << 3);
          if (ip == 0) {
            continue;
          }

          std::array<uint64_t, max_num_of_stall_types> stall;
          printf("\n !! %lu, ", ip);
          // IP address is already processed. (metric_list.size() - 1) is the number of types of stall
          for (uint32_t k = 0; k <  (metric_list.size() - 1); k++) {
            //stall[k] = value[j + k + 1].value.ui64;
            printf(" %lu, ", value[j + k + 1].value.ui64);
          }
          printf("\n");
        }
        value += samples[i];
      }
      return true;
    };

    desc->profiling_state_.store(PROFILER_ENABLED, std::memory_order_release);
    while (desc->profiling_state_.load(std::memory_order_acquire) != PROFILER_DISABLED) {
      auto size = EventBasedReadMetrics(event, streamer, raw_metrics, MAX_METRIC_BUFFER);
      if (size > 0) {
        printf("\n !! Read data of size %lu\n", size);
        // If we have data, dump it to the intermediate file
        if (!dump_metrics (raw_metrics, size, &desc->metric_file_stream_, group)) {
          std::cerr << "[ERROR] Failed to write to sampling metrics file " << desc->metric_file_name_ << std::endl;
          break;
        }
      }
    }

    // Flush the remaining metrics after the profiler has stopped
    auto size = ReadMetrics(streamer, raw_metrics, MAX_METRIC_BUFFER);
    while (size > 0) {
      printf("\n !! At the end: Read data of size %lu\n", size);
      if (!dump_metrics (raw_metrics, size, &desc->metric_file_stream_, group)) {
        std::cerr << "[ERROR] Failed to write to sampling metrics file " << desc->metric_file_name_ << std::endl;
        break;
      }
      if (size < MAX_METRIC_BUFFER)
        break;
      size = ReadMetrics(streamer, raw_metrics, MAX_METRIC_BUFFER);
    }
    free (raw_metrics);

    printf("\n !! zetMetricStreamerClose \n");
    status = ZE_FUNC(zetMetricStreamerClose)(streamer);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    printf("\n !! zeEventDestroy \n");
    status = ZE_FUNC(zeEventDestroy)(event);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    printf("\n !! zeEventPoolDestroy \n");
    status = ZE_FUNC(zeEventPoolDestroy)(event_pool);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    printf("\n !! zetContextActivateMetricGroups \n");
    status = ZE_FUNC(zetContextActivateMetricGroups)(context, device, 0, &group);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  }

 private: // Data

  std::vector<ze_context_handle_t> metric_contexts_;
  std::map<ze_device_handle_t, ZeDeviceDescriptor *> device_descriptors_;
};

#endif // PTI_TOOLS_UNITRACE_LEVEL_ZERO_METRICS_H_

