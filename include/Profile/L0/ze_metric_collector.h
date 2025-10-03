//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef PTI_SAMPLES_ZE_METRIC_QUERY_ZE_METRIC_COLLECTOR_H_
#define PTI_SAMPLES_ZE_METRIC_QUERY_ZE_METRIC_COLLECTOR_H_

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <level_zero/layers/zel_tracing_api.h>

#include "metric_utils.h"
#include "ze_utils.h"

#define MAX_KERNEL_COUNT 16384



static uint32_t GetMetricCount(zet_metric_group_handle_t group) {
    PTI_ASSERT(group != nullptr);

    zet_metric_group_properties_t group_props{};
    group_props.stype = ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
    ze_result_t status = zetMetricGroupGetProperties(group, &group_props);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    return group_props.metricCount;
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

static std::vector<std::string> GroupToMetricList(zet_metric_group_handle_t group) {
    PTI_ASSERT(group != nullptr);

    uint32_t metric_count = GetMetricCount(group);
    PTI_ASSERT(metric_count > 0);

    std::vector<zet_metric_handle_t> metric_list(metric_count);
    ze_result_t status = zetMetricGet(group, &metric_count, metric_list.data());
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(metric_count == metric_list.size());

    std::vector<std::string> name_list;
    for (auto metric : metric_list) {
      zet_metric_properties_t metric_props{
          ZET_STRUCTURE_TYPE_METRIC_PROPERTIES, };
      status = zetMetricGetProperties(metric, &metric_props);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);

      std::string units = GetMetricUnits(metric_props.resultUnits);
      std::string name = metric_props.name;
      if (!units.empty()) {
        name += "[" + units + "]";
      }
      name_list.push_back(name);
    }

    return name_list;
  }


struct InstanceData {
  uint32_t kernel_id;
  zet_metric_query_handle_t metric_query;
};

struct QueryData {
  std::string kernel_name;
  zet_metric_query_handle_t query;
  ze_event_handle_t event;
};

using KernelNameMap = std::map<ze_kernel_handle_t, std::string>;
using QueryList = std::vector<QueryData>;

using MetricReport = std::vector<zet_typed_value_t>;
using KernelReportMap = std::map<std::string, std::vector<MetricReport> >;


typedef void (*OnMetricFinishCallback)(
    void* data, const std::string& name,
    MetricReport report, std::vector<std::string>);


class ZeMetricCollector {
 public: // Interface
  static ZeMetricCollector* Create(
      ze_driver_handle_t driver,
      ze_device_handle_t device,
      const char* group_name,
      OnMetricFinishCallback callback = nullptr,
      void* callback_data = nullptr,
      uint32_t max_kernel_count = MAX_KERNEL_COUNT) {
    PTI_ASSERT(driver != nullptr);
    PTI_ASSERT(device != nullptr);
    PTI_ASSERT(group_name != nullptr);
    PTI_ASSERT(callback != nullptr);
    PTI_ASSERT(max_kernel_count > 0);

    if(!utils::metrics::SufficientPrivilegesForMetrics()) {
      std::cerr << "[WARNING] Insufficent or indeterminate privileges for perf "
        << "metrics" << std::endl;
    }

    zet_metric_group_handle_t group = utils::ze::FindMetricGroup(
        device, group_name, ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED);
    if (group == nullptr) {
      std::cerr << "[WARNING] Unable to find target metric group: " <<
        group_name << std::endl;
      return nullptr;
    }
    
    std::vector<std::string> metriclist = GroupToMetricList(group);

    ze_context_handle_t context = utils::ze::GetContext(driver);
    PTI_ASSERT(context != nullptr);

    ZeMetricCollector* collector = new ZeMetricCollector(
        device, context, max_kernel_count, group, callback, callback_data, metriclist);
    PTI_ASSERT(collector != nullptr);

    ze_result_t status = ZE_RESULT_SUCCESS;
    zel_tracer_desc_t tracer_desc = {
        ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, nullptr, collector};
    zel_tracer_handle_t tracer = nullptr;
    status = zelTracerCreate(&tracer_desc, &tracer);
    if (status != ZE_RESULT_SUCCESS) {
      std::cerr << "[WARNING] Unable to create Level Zero tracer" << std::endl;
      delete collector;
      return nullptr;
    }

    collector->EnableMetrics(group);
    collector->EnableTracing(tracer);
    return collector;
  }

  ~ZeMetricCollector() {
    ze_result_t status = ZE_RESULT_SUCCESS;
    PTI_ASSERT(query_list_.size() == 0);

    if (tracer_ != nullptr) {
#if !defined(_WIN32)
      status = zelTracerDestroy(tracer_);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
#endif
    }

    if (metric_query_pool_ != nullptr || event_pool_ != nullptr) {
      DisableMetrics();
    }

    PTI_ASSERT(context_ != nullptr);
    status = zeContextDestroy(context_);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  }

  void DisableTracing() {
    PTI_ASSERT(tracer_ != nullptr);
#if !defined(_WIN32)
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zelTracerSetEnabled(tracer_, false);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
#endif
  }

  const KernelReportMap& GetKernelReportMap() const {
    return kernel_report_map_;
  }
  
  const std::vector<std::string>& GetMetricList() const {
    return metriclist_;
  }

 private: // Implementation
  ZeMetricCollector(
      ze_device_handle_t device, ze_context_handle_t context,
      uint32_t max_kernel_count, zet_metric_group_handle_t group,
      OnMetricFinishCallback callback = nullptr,
      void* callback_data = nullptr,
       std::vector<std::string> metriclist = {})
      : device_(device), context_(context),
        max_kernel_count_(max_kernel_count),
        callback_(callback), callback_data_(callback_data),
        metriclist_(metriclist) {
    PTI_ASSERT(device_ != nullptr);
    PTI_ASSERT(context_ != nullptr);
    PTI_ASSERT(max_kernel_count_ > 0);        
  }

  void EnableTracing(zel_tracer_handle_t tracer) {
    PTI_ASSERT(tracer != nullptr);
    tracer_ = tracer;

    zet_core_callbacks_t prologue_callbacks{};
    zet_core_callbacks_t epilogue_callbacks{};

    prologue_callbacks.CommandList.pfnAppendLaunchKernelCb =
      OnEnterCommandListAppendLaunchKernel;
    epilogue_callbacks.CommandList.pfnAppendLaunchKernelCb =
      OnExitCommandListAppendLaunchKernel;

    epilogue_callbacks.Kernel.pfnCreateCb = OnExitKernelCreate;
    epilogue_callbacks.Kernel.pfnDestroyCb = OnExitKernelDestroy;

    epilogue_callbacks.CommandQueue.pfnDestroyCb =
      OnExitCommandQueueDestroy;
    epilogue_callbacks.CommandQueue.pfnSynchronizeCb =
      OnExitCommandQueueSynchronize;

    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zelTracerSetPrologues(tracer_, &prologue_callbacks);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    status = zelTracerSetEpilogues(tracer_, &epilogue_callbacks);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    status = zelTracerSetEnabled(tracer_, true);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  }

  void EnableMetrics(zet_metric_group_handle_t group) {
    PTI_ASSERT(group != nullptr);

    ze_result_t status = ZE_RESULT_SUCCESS;
    metric_group_ = group;

    PTI_ASSERT(device_ != nullptr);
    PTI_ASSERT(context_ != nullptr);
    status = zetContextActivateMetricGroups(
        context_, device_, 1, &metric_group_);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    zet_metric_query_pool_desc_t metric_query_pool_desc = {
        ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC, nullptr,
        ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE, max_kernel_count_};
    status = zetMetricQueryPoolCreate(context_, device_, metric_group_,
                                      &metric_query_pool_desc,
                                      &metric_query_pool_);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(metric_query_pool_ != nullptr);

    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE, max_kernel_count_};
    status = zeEventPoolCreate(context_, &event_pool_desc,
                               0, nullptr, &event_pool_);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(event_pool_ != nullptr);
  }

  void DisableMetrics() {
    ze_result_t status = ZE_RESULT_SUCCESS;

    PTI_ASSERT(event_pool_ != nullptr);
    status = zeEventPoolDestroy(event_pool_);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    PTI_ASSERT(metric_query_pool_ != nullptr);
    status = zetMetricQueryPoolDestroy(metric_query_pool_);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    PTI_ASSERT(device_ != nullptr);
    PTI_ASSERT(context_ != nullptr);
    status = zetContextActivateMetricGroups(context_, device_, 0, nullptr);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
  }

  void AddKernelName(ze_kernel_handle_t kernel, std::string name) {
    PTI_ASSERT(kernel != nullptr);
    PTI_ASSERT(!name.empty());

    const std::lock_guard<std::mutex> lock(lock_);
    PTI_ASSERT(kernel_name_map_.count(kernel) == 0);
    kernel_name_map_[kernel] = name;
  }

  void RemoveKernelName(ze_kernel_handle_t kernel) {
    PTI_ASSERT(kernel != nullptr);

    const std::lock_guard<std::mutex> lock(lock_);
    PTI_ASSERT(kernel_name_map_.count(kernel) == 1);
    kernel_name_map_.erase(kernel);
  }

  uint32_t GetKernelId() {
    return kernel_id_.fetch_add(1, std::memory_order_acq_rel);
  }

  zet_metric_query_handle_t StartMetricQuery(
      ze_command_list_handle_t command_list, uint32_t kernel_id) {
    PTI_ASSERT(command_list != nullptr);
    if (kernel_id >= max_kernel_count_) {
      return nullptr;
    }

    ze_result_t status = ZE_RESULT_SUCCESS;
    zet_metric_query_handle_t metric_query = nullptr;

    PTI_ASSERT(metric_query_pool_ != nullptr);
    status = zetMetricQueryCreate(metric_query_pool_,
                                  kernel_id, &metric_query);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    status = zetCommandListAppendMetricQueryBegin(command_list, metric_query);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    return metric_query;
  }

  ze_event_handle_t EndMetricQuery(ze_command_list_handle_t command_list,
                                   zet_metric_query_handle_t metric_query,
                                   uint32_t kernel_id) {
    PTI_ASSERT(command_list != nullptr);
    PTI_ASSERT(metric_query != nullptr);
    PTI_ASSERT(kernel_id < max_kernel_count_);

    ze_result_t status = ZE_RESULT_SUCCESS;
    ze_event_desc_t event_desc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, kernel_id,
        ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t event = nullptr;
    PTI_ASSERT(event_pool_ != nullptr);
    status = zeEventCreate(event_pool_, &event_desc, &event);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    status = zetCommandListAppendMetricQueryEnd(
        command_list, metric_query, event, 0, nullptr);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    return event;
  }

  void AddQuery(ze_kernel_handle_t kernel,
                zet_metric_query_handle_t query,
                ze_event_handle_t event) {
    PTI_ASSERT(kernel != nullptr);
    PTI_ASSERT(query != nullptr);
    PTI_ASSERT(event != nullptr);

    const std::lock_guard<std::mutex> lock(lock_);
    PTI_ASSERT(kernel_name_map_.count(kernel) == 1);
    const std::string& kernel_name = kernel_name_map_[kernel];
    PTI_ASSERT(!kernel_name.empty());

    QueryData data{kernel_name, query, event};
    query_list_.push_back(data);
  }

  MetricReport Calculate(const std::vector<uint8_t> data) {
    ze_result_t status = ZE_RESULT_SUCCESS;
    PTI_ASSERT(data.size() > 0);

    uint32_t value_count = 0;
    PTI_ASSERT(metric_group_ != nullptr);
    status = zetMetricGroupCalculateMetricValues(
        metric_group_, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
        data.size(), data.data(), &value_count, nullptr);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(value_count > 0);

    MetricReport report(value_count);
    status = zetMetricGroupCalculateMetricValues(
        metric_group_, ZET_METRIC_GROUP_CALCULATION_TYPE_METRIC_VALUES,
        data.size(), data.data(), &value_count, report.data());
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    uint32_t metric_count = 0;
    status = zetMetricGet(metric_group_, &metric_count, nullptr);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    uint32_t report_count = value_count / metric_count;
    PTI_ASSERT(report_count * metric_count == value_count);
    PTI_ASSERT(report_count == 1);

    return report;
  }

  void ProcessQuery(const QueryData& query) {
    ze_result_t status = ZE_RESULT_SUCCESS;
    status = zeEventHostSynchronize(query.event, UINT32_MAX);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    status = zeEventDestroy(query.event);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    size_t raw_size = 0;
    status = zetMetricQueryGetData(query.query, &raw_size, nullptr);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    PTI_ASSERT(raw_size > 0);

    std::vector<uint8_t> raw_data(raw_size);
    status = zetMetricQueryGetData(
        query.query, &raw_size, raw_data.data());
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    status = zetMetricQueryDestroy(query.query);
    PTI_ASSERT(status == ZE_RESULT_SUCCESS);

    MetricReport report = Calculate(raw_data);
    PTI_ASSERT(report.size() > 0);


    std::vector<MetricReport>& report_list = kernel_report_map_[query.kernel_name];
    report_list.push_back(report);
    
    if(callback_ != nullptr){
    	callback_(callback_data_, query.kernel_name, report, metriclist_);
    }

    
  }

  void ProcessResults() {

    const std::lock_guard<std::mutex> lock(lock_);
    for (const QueryData& query : query_list_) {
      ProcessQuery(query);
    }
    query_list_.clear();
  }

 private: // Callbacks
  static void OnExitKernelCreate(
      ze_kernel_create_params_t* params, ze_result_t result,
      void* global_data, void** instance_data) {
    if (result == ZE_RESULT_SUCCESS) {
      ZeMetricCollector* collector =
        reinterpret_cast<ZeMetricCollector*>(global_data);
      PTI_ASSERT(collector != nullptr);
      collector->AddKernelName(
          **(params->pphKernel), (*(params->pdesc))->pKernelName);
    }
  }

  static void OnExitKernelDestroy(
      ze_kernel_destroy_params_t* params, ze_result_t result,
      void* global_data, void** instance_data) {
    if (result == ZE_RESULT_SUCCESS) {
      ZeMetricCollector* collector =
        reinterpret_cast<ZeMetricCollector*>(global_data);
      PTI_ASSERT(collector != nullptr);
      collector->RemoveKernelName(*(params->phKernel));
    }
  }

  static void OnEnterCommandListAppendLaunchKernel(
      ze_command_list_append_launch_kernel_params_t* params,
      ze_result_t result, void* global_data, void** instance_data) {
    ZeMetricCollector* collector =
      reinterpret_cast<ZeMetricCollector*>(global_data);
    PTI_ASSERT(collector != nullptr);

    uint32_t kernel_id = collector->GetKernelId();
    zet_metric_query_handle_t query = nullptr;

    ze_command_list_handle_t command_list = *(params->phCommandList);
    PTI_ASSERT(command_list != nullptr);

    query = collector->StartMetricQuery(command_list, kernel_id);
    if (query != nullptr) {
      InstanceData* data = new InstanceData;
      PTI_ASSERT(data != nullptr);
      data->kernel_id = kernel_id;
      data->metric_query = query;
      *instance_data = reinterpret_cast<void*>(data);
    } else {
      *instance_data = nullptr;
    }
  }

  static void OnExitCommandListAppendLaunchKernel(
      ze_command_list_append_launch_kernel_params_t* params,
      ze_result_t result, void* global_data, void** instance_data) {
    InstanceData* data = reinterpret_cast<InstanceData*>(*instance_data);
    if (data != nullptr) {
      ZeMetricCollector* collector =
        reinterpret_cast<ZeMetricCollector*>(global_data);
      PTI_ASSERT(collector != nullptr);

      ze_command_list_handle_t command_list = *(params->phCommandList);
      PTI_ASSERT(command_list != nullptr);

      PTI_ASSERT(data->metric_query != nullptr);
      ze_event_handle_t event = collector->EndMetricQuery(
          command_list, data->metric_query, data->kernel_id);
      PTI_ASSERT(event != nullptr);

      if (result == ZE_RESULT_SUCCESS) {
        ze_kernel_handle_t kernel = *(params->phKernel);
        PTI_ASSERT(kernel != nullptr);

        collector->AddQuery(kernel, data->metric_query, event);
      }

      delete data;
    }
  }

  static void OnExitCommandQueueSynchronize(
      ze_command_queue_synchronize_params_t* params,
      ze_result_t result, void* global_data, void** instance_data) {
    if (result == ZE_RESULT_SUCCESS) {
      ZeMetricCollector* collector =
        reinterpret_cast<ZeMetricCollector*>(global_data);
      PTI_ASSERT(collector != nullptr);
      collector->ProcessResults();
    }
  }

  static void OnExitCommandQueueDestroy(
      ze_command_queue_destroy_params_t* params,
      ze_result_t result, void* global_data, void** instance_data) {
    if (result == ZE_RESULT_SUCCESS) {
      ZeMetricCollector* collector =
        reinterpret_cast<ZeMetricCollector*>(global_data);
      PTI_ASSERT(collector != nullptr);
      collector->ProcessResults();
    }
  }

 private: // Data
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  zel_tracer_handle_t tracer_ = nullptr;
  
  OnMetricFinishCallback callback_ = nullptr;
  void* callback_data_ = nullptr;

  zet_metric_group_handle_t metric_group_ = nullptr;
  zet_metric_query_pool_handle_t metric_query_pool_ = nullptr;
  ze_event_pool_handle_t event_pool_ = nullptr;
  
  std::vector<std::string> metriclist_;

  std::atomic<uint32_t> kernel_id_{0};
  uint32_t max_kernel_count_ = 0;

  std::mutex lock_;
  KernelNameMap kernel_name_map_;
  QueryList query_list_;
  KernelReportMap kernel_report_map_;
};

#endif // PTI_SAMPLES_ZE_METRIC_QUERY_ZE_METRIC_COLLECTOR_H_
