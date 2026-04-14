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
static std::vector<std::string> stall_names_list;
#define MAX_METRIC_BUFFER  (8ULL * 1024ULL * 1024ULL)

void TauStallSamplingEvents( uint64_t address, const char *event_name, uint64_t event_value, ze_device_handle_t device);

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
  ze_driver_handle_t driver_;
  ze_context_handle_t context_;
  int32_t device_id_;
  zet_metric_group_handle_t metric_group_;
  std::thread *profiling_thread_;
  std::atomic<ZeProfilerState> profiling_state_;
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

        desc->device_ = device;
        desc->device_id_ = global_dev_cnt;

        printf("devices %lu subdevices %lu\n", devices.size(), sub_devices.size());

        ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2, };
        ze_result_t status = ZE_FUNC(zeDeviceGetProperties)(device, &props);
        PTI_ASSERT(status == ZE_RESULT_SUCCESS);
        PTI_ASSERT(props.timerResolution != 0);
        PTI_ASSERT(props.kernelTimestampValidBits != 0);
        printf("!! Device %p %s\n", desc->device_, props.name);

        ze_pci_ext_properties_t pci_device_properties;
        status = ZE_FUNC(zeDevicePciGetPropertiesExt)(device, &pci_device_properties);
        PTI_ASSERT(status == ZE_RESULT_SUCCESS);

        zet_metric_group_handle_t group = FindMetricGroup (device, metric_group, ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_TIME_BASED);
        if (group == nullptr) {
          std::cerr << "[ERROR] Stall sampling metrics not found, is EuStallSampling available? " << metric_group << std::endl;
          exit(-1);
        }

        if(stall_names_list.size()==0)
        {
          stall_names_list = GetMetricList(group);
          PTI_ASSERT(!stall_names_list.empty());
          printf("!! metrics_list ");
          for(auto metric : stall_names_list )
            printf(", %s", metric.c_str());
          printf("\n");
        }
        //The metric named IP should always exist for stall sampling, if not,
        // it used the wrong metric, or the name was changed and we need to update
        // the names/references of this field
        uint32_t ip_idx = GetMetricId(stall_names_list, "IP");
        if (ip_idx >= stall_names_list.size()) {
          printf("\n[TAU] Error, IP metric not found when enabling L0 stall sampling. If you see this error, contact TAU developers\n");
          exit(-1);
        }

        desc->driver_ = driver;
        desc->context_ = context;
        desc->metric_group_ = group;

        desc->profiling_thread_ = nullptr;
        desc->profiling_state_.store(PROFILER_DISABLED, std::memory_order_release);

        device_descriptors_.insert({device, desc});

  
      } // devices list
    } // drivers list
  }

  void StartProfilingMetrics(void) {

    for (auto [handle, device] : device_descriptors_) {
      device->profiling_thread_ = new std::thread(MetricProfilingThread, device);
      while (device->profiling_state_.load(std::memory_order_acquire) != PROFILER_ENABLED) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void StopProfilingMetrics() {
    for (auto [handle, device] : device_descriptors_) {
      PTI_ASSERT(device->profiling_thread_ != nullptr);
      PTI_ASSERT(device->profiling_state_ == PROFILER_ENABLED);
      device->profiling_state_.store(PROFILER_DISABLED, std::memory_order_release);
      device->profiling_thread_->join();
      delete device->profiling_thread_;
      device->profiling_thread_ = nullptr;
    }
  }

 private:


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

  static bool dump_metrics (uint8_t *buffer, uint64_t size, zet_metric_group_handle_t group, ze_device_handle_t device)
  {
      PTI_ASSERT(!stall_names_list.empty());
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
        for (uint32_t j = 0; j < size; j += stall_names_list.size()) {
          uint64_t ip;
          ip = (value[j].value.ui64 << 3);
          if (ip == 0) {
            continue;
          }

          /*char offset[32];
          uint64_t off = ip;
          snprintf(offset, sizeof(offset), "0x%" PRIx64, off);
          printf("\n !! %s, ", offset);
          std::cout << "\t\t["
                  << "0x" << std::setw(5) << std::setfill('0') << std::hex << std::uppercase
                  << ip << "] " << std::endl;
          */
          printf("\n !! %lu, ", ip);
          //kernel_command_properties_ ZeKernelCommandProperties
          // IP address is already processed. (metric_list.size() - 1) is the number of types of stall
          for (uint32_t k = 0; k <  (stall_names_list.size() - 1); k++) {
            printf(" %lu, ", value[j + k + 1].value.ui64);
            uint64_t event_value = value[j + k + 1].value.ui64;
            if(event_value != 0)
              TauStallSamplingEvents(ip, stall_names_list[k+1].c_str(), event_value, device);
          }
          printf("\n");
          

        }
        value += samples[i];
      }
      return true;
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

    auto *raw_metrics = static_cast<uint8_t*>(malloc(sizeof(uint8_t)*MAX_METRIC_BUFFER));
    UniMemory::ExitIfOutOfMemory((void *)raw_metrics);

    desc->profiling_state_.store(PROFILER_ENABLED, std::memory_order_release);
    while (desc->profiling_state_.load(std::memory_order_acquire) != PROFILER_DISABLED) {
      auto size = EventBasedReadMetrics(event, streamer, raw_metrics, MAX_METRIC_BUFFER);
      if (size > 0) {
        printf("\n !! Read data of size %lu\n", size);
        // If we have data, dump it to the intermediate file
        if (!dump_metrics (raw_metrics, size, group, device)) {
          std::cerr << "[ERROR] Failed to write to sampling metrics file " << std::endl;
          break;
        }
      }
    }

    // Flush the remaining metrics after the profiler has stopped
    auto size = ReadMetrics(streamer, raw_metrics, MAX_METRIC_BUFFER);
    while (size > 0) {
      printf("\n !! At the end: Read data of size %lu\n", size);
      if (!dump_metrics (raw_metrics, size, group, device)) {
        std::cerr << "[ERROR] Failed to write to sampling metrics file " << std::endl;
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

