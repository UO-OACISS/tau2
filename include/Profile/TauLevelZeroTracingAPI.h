//==============================================================
// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================

/* This file is auto-generated based on current version of Level Zero */

#pragma once

#include <assert.h>

#include <level_zero/zet_api.h>

namespace ze_tracing {

typedef enum _callback_site_t {
  ZE_CALLBACK_SITE_ENTER,
  ZE_CALLBACK_SITE_EXIT
} callback_site_t;

typedef struct _callback_data_t {
  callback_site_t site;
  uint64_t *correlation_data;
  const char *function_name;
  const void *function_params;
  ze_result_t function_return_value;
} callback_data_t;

typedef enum _function_id_t {
  ZE_FUNCTION_zeInit = 0,
  ZE_FUNCTION_zeDriverGet = 1,
  ZE_FUNCTION_zeDriverGetApiVersion = 2,
  ZE_FUNCTION_zeDriverGetProperties = 3,
  ZE_FUNCTION_zeDriverGetIPCProperties = 4,
  ZE_FUNCTION_zeDriverGetExtensionFunctionAddress = 5,
  ZE_FUNCTION_zeDriverAllocSharedMem = 6,
  ZE_FUNCTION_zeDriverAllocDeviceMem = 7,
  ZE_FUNCTION_zeDriverAllocHostMem = 8,
  ZE_FUNCTION_zeDriverFreeMem = 9,
  ZE_FUNCTION_zeDriverGetMemAllocProperties = 10,
  ZE_FUNCTION_zeDriverGetMemAddressRange = 11,
  ZE_FUNCTION_zeDriverGetMemIpcHandle = 12,
  ZE_FUNCTION_zeDriverOpenMemIpcHandle = 13,
  ZE_FUNCTION_zeDriverCloseMemIpcHandle = 14,
  ZE_FUNCTION_zeDeviceGet = 15,
  ZE_FUNCTION_zeDeviceGetSubDevices = 16,
  ZE_FUNCTION_zeDeviceGetProperties = 17,
  ZE_FUNCTION_zeDeviceGetComputeProperties = 18,
  ZE_FUNCTION_zeDeviceGetKernelProperties = 19,
  ZE_FUNCTION_zeDeviceGetMemoryProperties = 20,
  ZE_FUNCTION_zeDeviceGetMemoryAccessProperties = 21,
  ZE_FUNCTION_zeDeviceGetCacheProperties = 22,
  ZE_FUNCTION_zeDeviceGetImageProperties = 23,
  ZE_FUNCTION_zeDeviceGetP2PProperties = 24,
  ZE_FUNCTION_zeDeviceCanAccessPeer = 25,
  ZE_FUNCTION_zeDeviceSetLastLevelCacheConfig = 26,
  ZE_FUNCTION_zeDeviceSystemBarrier = 27,
  ZE_FUNCTION_zeDeviceMakeMemoryResident = 28,
  ZE_FUNCTION_zeDeviceEvictMemory = 29,
  ZE_FUNCTION_zeDeviceMakeImageResident = 30,
  ZE_FUNCTION_zeDeviceEvictImage = 31,
#if ZE_ENABLE_OCL_INTEROP
  ZE_FUNCTION_zeDeviceRegisterCLMemory = 32,
#endif //ZE_ENABLE_OCL_INTEROP
#if ZE_ENABLE_OCL_INTEROP
  ZE_FUNCTION_zeDeviceRegisterCLProgram = 33,
#endif //ZE_ENABLE_OCL_INTEROP
#if ZE_ENABLE_OCL_INTEROP
  ZE_FUNCTION_zeDeviceRegisterCLCommandQueue = 34,
#endif //ZE_ENABLE_OCL_INTEROP
  ZE_FUNCTION_zeCommandQueueCreate = 35,
  ZE_FUNCTION_zeCommandQueueDestroy = 36,
  ZE_FUNCTION_zeCommandQueueExecuteCommandLists = 37,
  ZE_FUNCTION_zeCommandQueueSynchronize = 38,
  ZE_FUNCTION_zeCommandListCreate = 39,
  ZE_FUNCTION_zeCommandListCreateImmediate = 40,
  ZE_FUNCTION_zeCommandListDestroy = 41,
  ZE_FUNCTION_zeCommandListClose = 42,
  ZE_FUNCTION_zeCommandListReset = 43,
  ZE_FUNCTION_zeCommandListAppendBarrier = 44,
  ZE_FUNCTION_zeCommandListAppendMemoryRangesBarrier = 45,
  ZE_FUNCTION_zeCommandListAppendLaunchKernel = 46,
  ZE_FUNCTION_zeCommandListAppendLaunchCooperativeKernel = 47,
  ZE_FUNCTION_zeCommandListAppendLaunchKernelIndirect = 48,
  ZE_FUNCTION_zeCommandListAppendLaunchMultipleKernelsIndirect = 49,
  ZE_FUNCTION_zeCommandListAppendSignalEvent = 50,
  ZE_FUNCTION_zeCommandListAppendWaitOnEvents = 51,
  ZE_FUNCTION_zeCommandListAppendEventReset = 52,
  ZE_FUNCTION_zeCommandListAppendMemoryCopy = 53,
  ZE_FUNCTION_zeCommandListAppendMemoryFill = 54,
  ZE_FUNCTION_zeCommandListAppendMemoryCopyRegion = 55,
  ZE_FUNCTION_zeCommandListAppendImageCopy = 56,
  ZE_FUNCTION_zeCommandListAppendImageCopyRegion = 57,
  ZE_FUNCTION_zeCommandListAppendImageCopyToMemory = 58,
  ZE_FUNCTION_zeCommandListAppendImageCopyFromMemory = 59,
  ZE_FUNCTION_zeCommandListAppendMemoryPrefetch = 60,
  ZE_FUNCTION_zeCommandListAppendMemAdvise = 61,
  ZE_FUNCTION_zeImageGetProperties = 62,
  ZE_FUNCTION_zeImageCreate = 63,
  ZE_FUNCTION_zeImageDestroy = 64,
  ZE_FUNCTION_zeModuleCreate = 65,
  ZE_FUNCTION_zeModuleDestroy = 66,
  ZE_FUNCTION_zeModuleGetNativeBinary = 67,
  ZE_FUNCTION_zeModuleGetGlobalPointer = 68,
  ZE_FUNCTION_zeModuleGetKernelNames = 69,
  ZE_FUNCTION_zeModuleGetFunctionPointer = 70,
  ZE_FUNCTION_zeModuleBuildLogDestroy = 71,
  ZE_FUNCTION_zeModuleBuildLogGetString = 72,
  ZE_FUNCTION_zeKernelCreate = 73,
  ZE_FUNCTION_zeKernelDestroy = 74,
  ZE_FUNCTION_zeKernelSetIntermediateCacheConfig = 75,
  ZE_FUNCTION_zeKernelSetGroupSize = 76,
  ZE_FUNCTION_zeKernelSuggestGroupSize = 77,
  ZE_FUNCTION_zeKernelSuggestMaxCooperativeGroupCount = 78,
  ZE_FUNCTION_zeKernelSetArgumentValue = 79,
  ZE_FUNCTION_zeKernelSetAttribute = 80,
  ZE_FUNCTION_zeKernelGetAttribute = 81,
  ZE_FUNCTION_zeKernelGetProperties = 82,
  ZE_FUNCTION_zeEventPoolCreate = 83,
  ZE_FUNCTION_zeEventPoolDestroy = 84,
  ZE_FUNCTION_zeEventPoolGetIpcHandle = 85,
  ZE_FUNCTION_zeEventPoolOpenIpcHandle = 86,
  ZE_FUNCTION_zeEventPoolCloseIpcHandle = 87,
  ZE_FUNCTION_zeEventCreate = 88,
  ZE_FUNCTION_zeEventDestroy = 89,
  ZE_FUNCTION_zeEventHostSignal = 90,
  ZE_FUNCTION_zeEventHostSynchronize = 91,
  ZE_FUNCTION_zeEventQueryStatus = 92,
  ZE_FUNCTION_zeEventHostReset = 93,
  ZE_FUNCTION_zeEventGetTimestamp = 94,
  ZE_FUNCTION_zeFenceCreate = 95,
  ZE_FUNCTION_zeFenceDestroy = 96,
  ZE_FUNCTION_zeFenceHostSynchronize = 97,
  ZE_FUNCTION_zeFenceQueryStatus = 98,
  ZE_FUNCTION_zeFenceReset = 99,
  ZE_FUNCTION_zeSamplerCreate = 100,
  ZE_FUNCTION_zeSamplerDestroy = 101,
  ZE_FUNCTION_COUNT = 102
} function_id_t;

typedef void (*tracing_callback_t)(
  function_id_t fid, callback_data_t* callback_data, void* user_data);

typedef struct _global_data_t{
  tracing_callback_t callback = nullptr;
  void* user_data = nullptr;
} global_data_t;

void zeInitOnEnter(
    ze_init_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeInit",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeInit,
                &callback_data, data->user_data);
}

void zeInitOnExit(
    ze_init_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeInit",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeInit,
                &callback_data, data->user_data);
}

void zeDriverGetOnEnter(
    ze_driver_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGet",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGet,
                &callback_data, data->user_data);
}

void zeDriverGetOnExit(
    ze_driver_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGet",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGet,
                &callback_data, data->user_data);
}

void zeDriverGetApiVersionOnEnter(
    ze_driver_get_api_version_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetApiVersion",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetApiVersion,
                &callback_data, data->user_data);
}

void zeDriverGetApiVersionOnExit(
    ze_driver_get_api_version_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetApiVersion",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetApiVersion,
                &callback_data, data->user_data);
}

void zeDriverGetPropertiesOnEnter(
    ze_driver_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetProperties,
                &callback_data, data->user_data);
}

void zeDriverGetPropertiesOnExit(
    ze_driver_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetProperties,
                &callback_data, data->user_data);
}

void zeDriverGetIPCPropertiesOnEnter(
    ze_driver_get_ipc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetIPCProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetIPCProperties,
                &callback_data, data->user_data);
}

void zeDriverGetIPCPropertiesOnExit(
    ze_driver_get_ipc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetIPCProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetIPCProperties,
                &callback_data, data->user_data);
}

void zeDriverGetExtensionFunctionAddressOnEnter(
    ze_driver_get_extension_function_address_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetExtensionFunctionAddress",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetExtensionFunctionAddress,
                &callback_data, data->user_data);
}

void zeDriverGetExtensionFunctionAddressOnExit(
    ze_driver_get_extension_function_address_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetExtensionFunctionAddress",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetExtensionFunctionAddress,
                &callback_data, data->user_data);
}

void zeDriverAllocSharedMemOnEnter(
    ze_driver_alloc_shared_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverAllocSharedMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverAllocSharedMem,
                &callback_data, data->user_data);
}

void zeDriverAllocSharedMemOnExit(
    ze_driver_alloc_shared_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverAllocSharedMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverAllocSharedMem,
                &callback_data, data->user_data);
}

void zeDriverAllocDeviceMemOnEnter(
    ze_driver_alloc_device_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverAllocDeviceMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverAllocDeviceMem,
                &callback_data, data->user_data);
}

void zeDriverAllocDeviceMemOnExit(
    ze_driver_alloc_device_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverAllocDeviceMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverAllocDeviceMem,
                &callback_data, data->user_data);
}

void zeDriverAllocHostMemOnEnter(
    ze_driver_alloc_host_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverAllocHostMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverAllocHostMem,
                &callback_data, data->user_data);
}

void zeDriverAllocHostMemOnExit(
    ze_driver_alloc_host_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverAllocHostMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverAllocHostMem,
                &callback_data, data->user_data);
}

void zeDriverFreeMemOnEnter(
    ze_driver_free_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverFreeMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverFreeMem,
                &callback_data, data->user_data);
}

void zeDriverFreeMemOnExit(
    ze_driver_free_mem_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverFreeMem",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverFreeMem,
                &callback_data, data->user_data);
}

void zeDriverGetMemAllocPropertiesOnEnter(
    ze_driver_get_mem_alloc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetMemAllocProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetMemAllocProperties,
                &callback_data, data->user_data);
}

void zeDriverGetMemAllocPropertiesOnExit(
    ze_driver_get_mem_alloc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetMemAllocProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetMemAllocProperties,
                &callback_data, data->user_data);
}

void zeDriverGetMemAddressRangeOnEnter(
    ze_driver_get_mem_address_range_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetMemAddressRange",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetMemAddressRange,
                &callback_data, data->user_data);
}

void zeDriverGetMemAddressRangeOnExit(
    ze_driver_get_mem_address_range_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetMemAddressRange",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetMemAddressRange,
                &callback_data, data->user_data);
}

void zeDriverGetMemIpcHandleOnEnter(
    ze_driver_get_mem_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetMemIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetMemIpcHandle,
                &callback_data, data->user_data);
}

void zeDriverGetMemIpcHandleOnExit(
    ze_driver_get_mem_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverGetMemIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverGetMemIpcHandle,
                &callback_data, data->user_data);
}

void zeDriverOpenMemIpcHandleOnEnter(
    ze_driver_open_mem_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverOpenMemIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverOpenMemIpcHandle,
                &callback_data, data->user_data);
}

void zeDriverOpenMemIpcHandleOnExit(
    ze_driver_open_mem_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverOpenMemIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverOpenMemIpcHandle,
                &callback_data, data->user_data);
}

void zeDriverCloseMemIpcHandleOnEnter(
    ze_driver_close_mem_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverCloseMemIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverCloseMemIpcHandle,
                &callback_data, data->user_data);
}

void zeDriverCloseMemIpcHandleOnExit(
    ze_driver_close_mem_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDriverCloseMemIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDriverCloseMemIpcHandle,
                &callback_data, data->user_data);
}

void zeDeviceGetOnEnter(
    ze_device_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGet",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGet,
                &callback_data, data->user_data);
}

void zeDeviceGetOnExit(
    ze_device_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGet",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGet,
                &callback_data, data->user_data);
}

void zeDeviceGetSubDevicesOnEnter(
    ze_device_get_sub_devices_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetSubDevices",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetSubDevices,
                &callback_data, data->user_data);
}

void zeDeviceGetSubDevicesOnExit(
    ze_device_get_sub_devices_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetSubDevices",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetSubDevices,
                &callback_data, data->user_data);
}

void zeDeviceGetPropertiesOnEnter(
    ze_device_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetPropertiesOnExit(
    ze_device_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetComputePropertiesOnEnter(
    ze_device_get_compute_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetComputeProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetComputeProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetComputePropertiesOnExit(
    ze_device_get_compute_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetComputeProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetComputeProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetKernelPropertiesOnEnter(
    ze_device_get_kernel_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetKernelProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetKernelProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetKernelPropertiesOnExit(
    ze_device_get_kernel_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetKernelProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetKernelProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetMemoryPropertiesOnEnter(
    ze_device_get_memory_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetMemoryProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetMemoryProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetMemoryPropertiesOnExit(
    ze_device_get_memory_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetMemoryProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetMemoryProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetMemoryAccessPropertiesOnEnter(
    ze_device_get_memory_access_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetMemoryAccessProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetMemoryAccessProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetMemoryAccessPropertiesOnExit(
    ze_device_get_memory_access_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetMemoryAccessProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetMemoryAccessProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetCachePropertiesOnEnter(
    ze_device_get_cache_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetCacheProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetCacheProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetCachePropertiesOnExit(
    ze_device_get_cache_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetCacheProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetCacheProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetImagePropertiesOnEnter(
    ze_device_get_image_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetImageProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetImageProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetImagePropertiesOnExit(
    ze_device_get_image_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetImageProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetImageProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetP2PPropertiesOnEnter(
    ze_device_get_p2_p_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetP2PProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetP2PProperties,
                &callback_data, data->user_data);
}

void zeDeviceGetP2PPropertiesOnExit(
    ze_device_get_p2_p_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceGetP2PProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceGetP2PProperties,
                &callback_data, data->user_data);
}

void zeDeviceCanAccessPeerOnEnter(
    ze_device_can_access_peer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceCanAccessPeer",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceCanAccessPeer,
                &callback_data, data->user_data);
}

void zeDeviceCanAccessPeerOnExit(
    ze_device_can_access_peer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceCanAccessPeer",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceCanAccessPeer,
                &callback_data, data->user_data);
}

void zeDeviceSetLastLevelCacheConfigOnEnter(
    ze_device_set_last_level_cache_config_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceSetLastLevelCacheConfig",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceSetLastLevelCacheConfig,
                &callback_data, data->user_data);
}

void zeDeviceSetLastLevelCacheConfigOnExit(
    ze_device_set_last_level_cache_config_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceSetLastLevelCacheConfig",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceSetLastLevelCacheConfig,
                &callback_data, data->user_data);
}

void zeDeviceSystemBarrierOnEnter(
    ze_device_system_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceSystemBarrier",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceSystemBarrier,
                &callback_data, data->user_data);
}

void zeDeviceSystemBarrierOnExit(
    ze_device_system_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceSystemBarrier",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceSystemBarrier,
                &callback_data, data->user_data);
}

void zeDeviceMakeMemoryResidentOnEnter(
    ze_device_make_memory_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceMakeMemoryResident",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceMakeMemoryResident,
                &callback_data, data->user_data);
}

void zeDeviceMakeMemoryResidentOnExit(
    ze_device_make_memory_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceMakeMemoryResident",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceMakeMemoryResident,
                &callback_data, data->user_data);
}

void zeDeviceEvictMemoryOnEnter(
    ze_device_evict_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceEvictMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceEvictMemory,
                &callback_data, data->user_data);
}

void zeDeviceEvictMemoryOnExit(
    ze_device_evict_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceEvictMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceEvictMemory,
                &callback_data, data->user_data);
}

void zeDeviceMakeImageResidentOnEnter(
    ze_device_make_image_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceMakeImageResident",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceMakeImageResident,
                &callback_data, data->user_data);
}

void zeDeviceMakeImageResidentOnExit(
    ze_device_make_image_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceMakeImageResident",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceMakeImageResident,
                &callback_data, data->user_data);
}

void zeDeviceEvictImageOnEnter(
    ze_device_evict_image_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceEvictImage",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceEvictImage,
                &callback_data, data->user_data);
}

void zeDeviceEvictImageOnExit(
    ze_device_evict_image_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceEvictImage",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceEvictImage,
                &callback_data, data->user_data);
}

#if ZE_ENABLE_OCL_INTEROP
void zeDeviceRegisterCLMemoryOnEnter(
    ze_device_register_cl_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceRegisterCLMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceRegisterCLMemory,
                &callback_data, data->user_data);
}

void zeDeviceRegisterCLMemoryOnExit(
    ze_device_register_cl_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceRegisterCLMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceRegisterCLMemory,
                &callback_data, data->user_data);
}
#endif //ZE_ENABLE_OCL_INTEROP

#if ZE_ENABLE_OCL_INTEROP
void zeDeviceRegisterCLProgramOnEnter(
    ze_device_register_cl_program_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceRegisterCLProgram",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceRegisterCLProgram,
                &callback_data, data->user_data);
}

void zeDeviceRegisterCLProgramOnExit(
    ze_device_register_cl_program_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceRegisterCLProgram",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceRegisterCLProgram,
                &callback_data, data->user_data);
}
#endif //ZE_ENABLE_OCL_INTEROP

#if ZE_ENABLE_OCL_INTEROP
void zeDeviceRegisterCLCommandQueueOnEnter(
    ze_device_register_cl_command_queue_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceRegisterCLCommandQueue",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceRegisterCLCommandQueue,
                &callback_data, data->user_data);
}

void zeDeviceRegisterCLCommandQueueOnExit(
    ze_device_register_cl_command_queue_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeDeviceRegisterCLCommandQueue",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeDeviceRegisterCLCommandQueue,
                &callback_data, data->user_data);
}
#endif //ZE_ENABLE_OCL_INTEROP

void zeCommandQueueCreateOnEnter(
    ze_command_queue_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueCreate,
                &callback_data, data->user_data);
}

void zeCommandQueueCreateOnExit(
    ze_command_queue_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueCreate,
                &callback_data, data->user_data);
}

void zeCommandQueueDestroyOnEnter(
    ze_command_queue_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueDestroy,
                &callback_data, data->user_data);
}

void zeCommandQueueDestroyOnExit(
    ze_command_queue_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueDestroy,
                &callback_data, data->user_data);
}

void zeCommandQueueExecuteCommandListsOnEnter(
    ze_command_queue_execute_command_lists_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueExecuteCommandLists",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueExecuteCommandLists,
                &callback_data, data->user_data);
}

void zeCommandQueueExecuteCommandListsOnExit(
    ze_command_queue_execute_command_lists_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueExecuteCommandLists",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueExecuteCommandLists,
                &callback_data, data->user_data);
}

void zeCommandQueueSynchronizeOnEnter(
    ze_command_queue_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueSynchronize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueSynchronize,
                &callback_data, data->user_data);
}

void zeCommandQueueSynchronizeOnExit(
    ze_command_queue_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandQueueSynchronize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandQueueSynchronize,
                &callback_data, data->user_data);
}

void zeCommandListCreateOnEnter(
    ze_command_list_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListCreate,
                &callback_data, data->user_data);
}

void zeCommandListCreateOnExit(
    ze_command_list_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListCreate,
                &callback_data, data->user_data);
}

void zeCommandListCreateImmediateOnEnter(
    ze_command_list_create_immediate_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListCreateImmediate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListCreateImmediate,
                &callback_data, data->user_data);
}

void zeCommandListCreateImmediateOnExit(
    ze_command_list_create_immediate_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListCreateImmediate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListCreateImmediate,
                &callback_data, data->user_data);
}

void zeCommandListDestroyOnEnter(
    ze_command_list_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListDestroy,
                &callback_data, data->user_data);
}

void zeCommandListDestroyOnExit(
    ze_command_list_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListDestroy,
                &callback_data, data->user_data);
}

void zeCommandListCloseOnEnter(
    ze_command_list_close_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListClose",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListClose,
                &callback_data, data->user_data);
}

void zeCommandListCloseOnExit(
    ze_command_list_close_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListClose",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListClose,
                &callback_data, data->user_data);
}

void zeCommandListResetOnEnter(
    ze_command_list_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListReset,
                &callback_data, data->user_data);
}

void zeCommandListResetOnExit(
    ze_command_list_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListReset,
                &callback_data, data->user_data);
}

void zeCommandListAppendBarrierOnEnter(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendBarrier",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendBarrier,
                &callback_data, data->user_data);
}

void zeCommandListAppendBarrierOnExit(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendBarrier",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendBarrier,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryRangesBarrierOnEnter(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryRangesBarrier",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryRangesBarrier,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryRangesBarrierOnExit(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryRangesBarrier",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryRangesBarrier,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchKernelOnEnter(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchKernel",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchKernel,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchKernelOnExit(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchKernel",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchKernel,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchCooperativeKernelOnEnter(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchCooperativeKernel",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchCooperativeKernel,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchCooperativeKernelOnExit(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchCooperativeKernel",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchCooperativeKernel,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchKernelIndirectOnEnter(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchKernelIndirect",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchKernelIndirect,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchKernelIndirectOnExit(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchKernelIndirect",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchKernelIndirect,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchMultipleKernelsIndirectOnEnter(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchMultipleKernelsIndirect",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchMultipleKernelsIndirect,
                &callback_data, data->user_data);
}

void zeCommandListAppendLaunchMultipleKernelsIndirectOnExit(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendLaunchMultipleKernelsIndirect",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendLaunchMultipleKernelsIndirect,
                &callback_data, data->user_data);
}

void zeCommandListAppendSignalEventOnEnter(
    ze_command_list_append_signal_event_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendSignalEvent",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendSignalEvent,
                &callback_data, data->user_data);
}

void zeCommandListAppendSignalEventOnExit(
    ze_command_list_append_signal_event_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendSignalEvent",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendSignalEvent,
                &callback_data, data->user_data);
}

void zeCommandListAppendWaitOnEventsOnEnter(
    ze_command_list_append_wait_on_events_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendWaitOnEvents",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendWaitOnEvents,
                &callback_data, data->user_data);
}

void zeCommandListAppendWaitOnEventsOnExit(
    ze_command_list_append_wait_on_events_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendWaitOnEvents",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendWaitOnEvents,
                &callback_data, data->user_data);
}

void zeCommandListAppendEventResetOnEnter(
    ze_command_list_append_event_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendEventReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendEventReset,
                &callback_data, data->user_data);
}

void zeCommandListAppendEventResetOnExit(
    ze_command_list_append_event_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendEventReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendEventReset,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryCopyOnEnter(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryCopy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryCopy,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryCopyOnExit(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryCopy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryCopy,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryFillOnEnter(
    ze_command_list_append_memory_fill_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryFill",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryFill,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryFillOnExit(
    ze_command_list_append_memory_fill_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryFill",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryFill,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryCopyRegionOnEnter(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryCopyRegion",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryCopyRegion,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryCopyRegionOnExit(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryCopyRegion",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryCopyRegion,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyOnEnter(
    ze_command_list_append_image_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopy,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyOnExit(
    ze_command_list_append_image_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopy,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyRegionOnEnter(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopyRegion",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopyRegion,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyRegionOnExit(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopyRegion",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopyRegion,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyToMemoryOnEnter(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopyToMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopyToMemory,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyToMemoryOnExit(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopyToMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopyToMemory,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyFromMemoryOnEnter(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopyFromMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopyFromMemory,
                &callback_data, data->user_data);
}

void zeCommandListAppendImageCopyFromMemoryOnExit(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendImageCopyFromMemory",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendImageCopyFromMemory,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryPrefetchOnEnter(
    ze_command_list_append_memory_prefetch_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryPrefetch",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryPrefetch,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemoryPrefetchOnExit(
    ze_command_list_append_memory_prefetch_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemoryPrefetch",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemoryPrefetch,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemAdviseOnEnter(
    ze_command_list_append_mem_advise_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemAdvise",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemAdvise,
                &callback_data, data->user_data);
}

void zeCommandListAppendMemAdviseOnExit(
    ze_command_list_append_mem_advise_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeCommandListAppendMemAdvise",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeCommandListAppendMemAdvise,
                &callback_data, data->user_data);
}

void zeImageGetPropertiesOnEnter(
    ze_image_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeImageGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeImageGetProperties,
                &callback_data, data->user_data);
}

void zeImageGetPropertiesOnExit(
    ze_image_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeImageGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeImageGetProperties,
                &callback_data, data->user_data);
}

void zeImageCreateOnEnter(
    ze_image_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeImageCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeImageCreate,
                &callback_data, data->user_data);
}

void zeImageCreateOnExit(
    ze_image_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeImageCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeImageCreate,
                &callback_data, data->user_data);
}

void zeImageDestroyOnEnter(
    ze_image_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeImageDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeImageDestroy,
                &callback_data, data->user_data);
}

void zeImageDestroyOnExit(
    ze_image_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeImageDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeImageDestroy,
                &callback_data, data->user_data);
}

void zeModuleCreateOnEnter(
    ze_module_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleCreate,
                &callback_data, data->user_data);
}

void zeModuleCreateOnExit(
    ze_module_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleCreate,
                &callback_data, data->user_data);
}

void zeModuleDestroyOnEnter(
    ze_module_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleDestroy,
                &callback_data, data->user_data);
}

void zeModuleDestroyOnExit(
    ze_module_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleDestroy,
                &callback_data, data->user_data);
}

void zeModuleGetNativeBinaryOnEnter(
    ze_module_get_native_binary_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetNativeBinary",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetNativeBinary,
                &callback_data, data->user_data);
}

void zeModuleGetNativeBinaryOnExit(
    ze_module_get_native_binary_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetNativeBinary",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetNativeBinary,
                &callback_data, data->user_data);
}

void zeModuleGetGlobalPointerOnEnter(
    ze_module_get_global_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetGlobalPointer",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetGlobalPointer,
                &callback_data, data->user_data);
}

void zeModuleGetGlobalPointerOnExit(
    ze_module_get_global_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetGlobalPointer",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetGlobalPointer,
                &callback_data, data->user_data);
}

void zeModuleGetKernelNamesOnEnter(
    ze_module_get_kernel_names_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetKernelNames",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetKernelNames,
                &callback_data, data->user_data);
}

void zeModuleGetKernelNamesOnExit(
    ze_module_get_kernel_names_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetKernelNames",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetKernelNames,
                &callback_data, data->user_data);
}

void zeModuleGetFunctionPointerOnEnter(
    ze_module_get_function_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetFunctionPointer",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetFunctionPointer,
                &callback_data, data->user_data);
}

void zeModuleGetFunctionPointerOnExit(
    ze_module_get_function_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleGetFunctionPointer",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleGetFunctionPointer,
                &callback_data, data->user_data);
}

void zeModuleBuildLogDestroyOnEnter(
    ze_module_build_log_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleBuildLogDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleBuildLogDestroy,
                &callback_data, data->user_data);
}

void zeModuleBuildLogDestroyOnExit(
    ze_module_build_log_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleBuildLogDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleBuildLogDestroy,
                &callback_data, data->user_data);
}

void zeModuleBuildLogGetStringOnEnter(
    ze_module_build_log_get_string_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleBuildLogGetString",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleBuildLogGetString,
                &callback_data, data->user_data);
}

void zeModuleBuildLogGetStringOnExit(
    ze_module_build_log_get_string_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeModuleBuildLogGetString",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeModuleBuildLogGetString,
                &callback_data, data->user_data);
}

void zeKernelCreateOnEnter(
    ze_kernel_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelCreate,
                &callback_data, data->user_data);
}

void zeKernelCreateOnExit(
    ze_kernel_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelCreate,
                &callback_data, data->user_data);
}

void zeKernelDestroyOnEnter(
    ze_kernel_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelDestroy,
                &callback_data, data->user_data);
}

void zeKernelDestroyOnExit(
    ze_kernel_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelDestroy,
                &callback_data, data->user_data);
}

void zeKernelSetIntermediateCacheConfigOnEnter(
    ze_kernel_set_intermediate_cache_config_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetIntermediateCacheConfig",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetIntermediateCacheConfig,
                &callback_data, data->user_data);
}

void zeKernelSetIntermediateCacheConfigOnExit(
    ze_kernel_set_intermediate_cache_config_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetIntermediateCacheConfig",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetIntermediateCacheConfig,
                &callback_data, data->user_data);
}

void zeKernelSetGroupSizeOnEnter(
    ze_kernel_set_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetGroupSize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetGroupSize,
                &callback_data, data->user_data);
}

void zeKernelSetGroupSizeOnExit(
    ze_kernel_set_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetGroupSize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetGroupSize,
                &callback_data, data->user_data);
}

void zeKernelSuggestGroupSizeOnEnter(
    ze_kernel_suggest_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSuggestGroupSize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSuggestGroupSize,
                &callback_data, data->user_data);
}

void zeKernelSuggestGroupSizeOnExit(
    ze_kernel_suggest_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSuggestGroupSize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSuggestGroupSize,
                &callback_data, data->user_data);
}

void zeKernelSuggestMaxCooperativeGroupCountOnEnter(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSuggestMaxCooperativeGroupCount",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSuggestMaxCooperativeGroupCount,
                &callback_data, data->user_data);
}

void zeKernelSuggestMaxCooperativeGroupCountOnExit(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSuggestMaxCooperativeGroupCount",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSuggestMaxCooperativeGroupCount,
                &callback_data, data->user_data);
}

void zeKernelSetArgumentValueOnEnter(
    ze_kernel_set_argument_value_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetArgumentValue",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetArgumentValue,
                &callback_data, data->user_data);
}

void zeKernelSetArgumentValueOnExit(
    ze_kernel_set_argument_value_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetArgumentValue",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetArgumentValue,
                &callback_data, data->user_data);
}

void zeKernelSetAttributeOnEnter(
    ze_kernel_set_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetAttribute",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetAttribute,
                &callback_data, data->user_data);
}

void zeKernelSetAttributeOnExit(
    ze_kernel_set_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelSetAttribute",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelSetAttribute,
                &callback_data, data->user_data);
}

void zeKernelGetAttributeOnEnter(
    ze_kernel_get_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelGetAttribute",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelGetAttribute,
                &callback_data, data->user_data);
}

void zeKernelGetAttributeOnExit(
    ze_kernel_get_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelGetAttribute",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelGetAttribute,
                &callback_data, data->user_data);
}

void zeKernelGetPropertiesOnEnter(
    ze_kernel_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelGetProperties,
                &callback_data, data->user_data);
}

void zeKernelGetPropertiesOnExit(
    ze_kernel_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeKernelGetProperties",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeKernelGetProperties,
                &callback_data, data->user_data);
}

void zeEventPoolCreateOnEnter(
    ze_event_pool_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolCreate,
                &callback_data, data->user_data);
}

void zeEventPoolCreateOnExit(
    ze_event_pool_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolCreate,
                &callback_data, data->user_data);
}

void zeEventPoolDestroyOnEnter(
    ze_event_pool_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolDestroy,
                &callback_data, data->user_data);
}

void zeEventPoolDestroyOnExit(
    ze_event_pool_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolDestroy,
                &callback_data, data->user_data);
}

void zeEventPoolGetIpcHandleOnEnter(
    ze_event_pool_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolGetIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolGetIpcHandle,
                &callback_data, data->user_data);
}

void zeEventPoolGetIpcHandleOnExit(
    ze_event_pool_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolGetIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolGetIpcHandle,
                &callback_data, data->user_data);
}

void zeEventPoolOpenIpcHandleOnEnter(
    ze_event_pool_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolOpenIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolOpenIpcHandle,
                &callback_data, data->user_data);
}

void zeEventPoolOpenIpcHandleOnExit(
    ze_event_pool_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolOpenIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolOpenIpcHandle,
                &callback_data, data->user_data);
}

void zeEventPoolCloseIpcHandleOnEnter(
    ze_event_pool_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolCloseIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolCloseIpcHandle,
                &callback_data, data->user_data);
}

void zeEventPoolCloseIpcHandleOnExit(
    ze_event_pool_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventPoolCloseIpcHandle",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventPoolCloseIpcHandle,
                &callback_data, data->user_data);
}

void zeEventCreateOnEnter(
    ze_event_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventCreate,
                &callback_data, data->user_data);
}

void zeEventCreateOnExit(
    ze_event_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventCreate,
                &callback_data, data->user_data);
}

void zeEventDestroyOnEnter(
    ze_event_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventDestroy,
                &callback_data, data->user_data);
}

void zeEventDestroyOnExit(
    ze_event_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventDestroy,
                &callback_data, data->user_data);
}

void zeEventHostSignalOnEnter(
    ze_event_host_signal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventHostSignal",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventHostSignal,
                &callback_data, data->user_data);
}

void zeEventHostSignalOnExit(
    ze_event_host_signal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventHostSignal",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventHostSignal,
                &callback_data, data->user_data);
}

void zeEventHostSynchronizeOnEnter(
    ze_event_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventHostSynchronize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventHostSynchronize,
                &callback_data, data->user_data);
}

void zeEventHostSynchronizeOnExit(
    ze_event_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventHostSynchronize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventHostSynchronize,
                &callback_data, data->user_data);
}

void zeEventQueryStatusOnEnter(
    ze_event_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventQueryStatus",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventQueryStatus,
                &callback_data, data->user_data);
}

void zeEventQueryStatusOnExit(
    ze_event_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventQueryStatus",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventQueryStatus,
                &callback_data, data->user_data);
}

void zeEventHostResetOnEnter(
    ze_event_host_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventHostReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventHostReset,
                &callback_data, data->user_data);
}

void zeEventHostResetOnExit(
    ze_event_host_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventHostReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventHostReset,
                &callback_data, data->user_data);
}

void zeEventGetTimestampOnEnter(
    ze_event_get_timestamp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventGetTimestamp",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventGetTimestamp,
                &callback_data, data->user_data);
}

void zeEventGetTimestampOnExit(
    ze_event_get_timestamp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeEventGetTimestamp",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeEventGetTimestamp,
                &callback_data, data->user_data);
}

void zeFenceCreateOnEnter(
    ze_fence_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceCreate,
                &callback_data, data->user_data);
}

void zeFenceCreateOnExit(
    ze_fence_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceCreate,
                &callback_data, data->user_data);
}

void zeFenceDestroyOnEnter(
    ze_fence_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceDestroy,
                &callback_data, data->user_data);
}

void zeFenceDestroyOnExit(
    ze_fence_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceDestroy,
                &callback_data, data->user_data);
}

void zeFenceHostSynchronizeOnEnter(
    ze_fence_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceHostSynchronize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceHostSynchronize,
                &callback_data, data->user_data);
}

void zeFenceHostSynchronizeOnExit(
    ze_fence_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceHostSynchronize",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceHostSynchronize,
                &callback_data, data->user_data);
}

void zeFenceQueryStatusOnEnter(
    ze_fence_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceQueryStatus",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceQueryStatus,
                &callback_data, data->user_data);
}

void zeFenceQueryStatusOnExit(
    ze_fence_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceQueryStatus",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceQueryStatus,
                &callback_data, data->user_data);
}

void zeFenceResetOnEnter(
    ze_fence_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceReset,
                &callback_data, data->user_data);
}

void zeFenceResetOnExit(
    ze_fence_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeFenceReset",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeFenceReset,
                &callback_data, data->user_data);
}

void zeSamplerCreateOnEnter(
    ze_sampler_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeSamplerCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeSamplerCreate,
                &callback_data, data->user_data);
}

void zeSamplerCreateOnExit(
    ze_sampler_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeSamplerCreate",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeSamplerCreate,
                &callback_data, data->user_data);
}

void zeSamplerDestroyOnEnter(
    ze_sampler_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_ENTER,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeSamplerDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeSamplerDestroy,
                &callback_data, data->user_data);
}

void zeSamplerDestroyOnExit(
    ze_sampler_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) {
  global_data_t* data =
    reinterpret_cast<global_data_t*>(global_user_data);
  assert(data != nullptr);

  callback_data_t callback_data = {
    ZE_CALLBACK_SITE_EXIT,
    reinterpret_cast<uint64_t*>(instance_user_data),
    "zeSamplerDestroy",
    reinterpret_cast<void*>(params),
    result };

  data->callback(ZE_FUNCTION_zeSamplerDestroy,
                &callback_data, data->user_data);
}

void SetTracingFunctions(zet_tracer_handle_t tracer,
                         std::set<function_id_t> functions) {
  zet_core_callbacks_t prologue = {};
  zet_core_callbacks_t epilogue = {};

  for (auto fid : functions) {
    switch (fid) {
      case ZE_FUNCTION_zeInit:
        prologue.Global.pfnInitCb = zeInitOnEnter;
        epilogue.Global.pfnInitCb = zeInitOnExit;
      case ZE_FUNCTION_zeDriverGet:
        prologue.Driver.pfnGetCb = zeDriverGetOnEnter;
        epilogue.Driver.pfnGetCb = zeDriverGetOnExit;
      case ZE_FUNCTION_zeDriverGetApiVersion:
        prologue.Driver.pfnGetApiVersionCb = zeDriverGetApiVersionOnEnter;
        epilogue.Driver.pfnGetApiVersionCb = zeDriverGetApiVersionOnExit;
      case ZE_FUNCTION_zeDriverGetProperties:
        prologue.Driver.pfnGetPropertiesCb = zeDriverGetPropertiesOnEnter;
        epilogue.Driver.pfnGetPropertiesCb = zeDriverGetPropertiesOnExit;
      case ZE_FUNCTION_zeDriverGetIPCProperties:
        prologue.Driver.pfnGetIPCPropertiesCb = zeDriverGetIPCPropertiesOnEnter;
        epilogue.Driver.pfnGetIPCPropertiesCb = zeDriverGetIPCPropertiesOnExit;
      case ZE_FUNCTION_zeDriverGetExtensionFunctionAddress:
        prologue.Driver.pfnGetExtensionFunctionAddressCb = zeDriverGetExtensionFunctionAddressOnEnter;
        epilogue.Driver.pfnGetExtensionFunctionAddressCb = zeDriverGetExtensionFunctionAddressOnExit;
      case ZE_FUNCTION_zeDriverAllocSharedMem:
        prologue.Driver.pfnAllocSharedMemCb = zeDriverAllocSharedMemOnEnter;
        epilogue.Driver.pfnAllocSharedMemCb = zeDriverAllocSharedMemOnExit;
      case ZE_FUNCTION_zeDriverAllocDeviceMem:
        prologue.Driver.pfnAllocDeviceMemCb = zeDriverAllocDeviceMemOnEnter;
        epilogue.Driver.pfnAllocDeviceMemCb = zeDriverAllocDeviceMemOnExit;
      case ZE_FUNCTION_zeDriverAllocHostMem:
        prologue.Driver.pfnAllocHostMemCb = zeDriverAllocHostMemOnEnter;
        epilogue.Driver.pfnAllocHostMemCb = zeDriverAllocHostMemOnExit;
      case ZE_FUNCTION_zeDriverFreeMem:
        prologue.Driver.pfnFreeMemCb = zeDriverFreeMemOnEnter;
        epilogue.Driver.pfnFreeMemCb = zeDriverFreeMemOnExit;
      case ZE_FUNCTION_zeDriverGetMemAllocProperties:
        prologue.Driver.pfnGetMemAllocPropertiesCb = zeDriverGetMemAllocPropertiesOnEnter;
        epilogue.Driver.pfnGetMemAllocPropertiesCb = zeDriverGetMemAllocPropertiesOnExit;
      case ZE_FUNCTION_zeDriverGetMemAddressRange:
        prologue.Driver.pfnGetMemAddressRangeCb = zeDriverGetMemAddressRangeOnEnter;
        epilogue.Driver.pfnGetMemAddressRangeCb = zeDriverGetMemAddressRangeOnExit;
      case ZE_FUNCTION_zeDriverGetMemIpcHandle:
        prologue.Driver.pfnGetMemIpcHandleCb = zeDriverGetMemIpcHandleOnEnter;
        epilogue.Driver.pfnGetMemIpcHandleCb = zeDriverGetMemIpcHandleOnExit;
      case ZE_FUNCTION_zeDriverOpenMemIpcHandle:
        prologue.Driver.pfnOpenMemIpcHandleCb = zeDriverOpenMemIpcHandleOnEnter;
        epilogue.Driver.pfnOpenMemIpcHandleCb = zeDriverOpenMemIpcHandleOnExit;
      case ZE_FUNCTION_zeDriverCloseMemIpcHandle:
        prologue.Driver.pfnCloseMemIpcHandleCb = zeDriverCloseMemIpcHandleOnEnter;
        epilogue.Driver.pfnCloseMemIpcHandleCb = zeDriverCloseMemIpcHandleOnExit;
      case ZE_FUNCTION_zeDeviceGet:
        prologue.Device.pfnGetCb = zeDeviceGetOnEnter;
        epilogue.Device.pfnGetCb = zeDeviceGetOnExit;
      case ZE_FUNCTION_zeDeviceGetSubDevices:
        prologue.Device.pfnGetSubDevicesCb = zeDeviceGetSubDevicesOnEnter;
        epilogue.Device.pfnGetSubDevicesCb = zeDeviceGetSubDevicesOnExit;
      case ZE_FUNCTION_zeDeviceGetProperties:
        prologue.Device.pfnGetPropertiesCb = zeDeviceGetPropertiesOnEnter;
        epilogue.Device.pfnGetPropertiesCb = zeDeviceGetPropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetComputeProperties:
        prologue.Device.pfnGetComputePropertiesCb = zeDeviceGetComputePropertiesOnEnter;
        epilogue.Device.pfnGetComputePropertiesCb = zeDeviceGetComputePropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetKernelProperties:
        prologue.Device.pfnGetKernelPropertiesCb = zeDeviceGetKernelPropertiesOnEnter;
        epilogue.Device.pfnGetKernelPropertiesCb = zeDeviceGetKernelPropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetMemoryProperties:
        prologue.Device.pfnGetMemoryPropertiesCb = zeDeviceGetMemoryPropertiesOnEnter;
        epilogue.Device.pfnGetMemoryPropertiesCb = zeDeviceGetMemoryPropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetMemoryAccessProperties:
        prologue.Device.pfnGetMemoryAccessPropertiesCb = zeDeviceGetMemoryAccessPropertiesOnEnter;
        epilogue.Device.pfnGetMemoryAccessPropertiesCb = zeDeviceGetMemoryAccessPropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetCacheProperties:
        prologue.Device.pfnGetCachePropertiesCb = zeDeviceGetCachePropertiesOnEnter;
        epilogue.Device.pfnGetCachePropertiesCb = zeDeviceGetCachePropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetImageProperties:
        prologue.Device.pfnGetImagePropertiesCb = zeDeviceGetImagePropertiesOnEnter;
        epilogue.Device.pfnGetImagePropertiesCb = zeDeviceGetImagePropertiesOnExit;
      case ZE_FUNCTION_zeDeviceGetP2PProperties:
        prologue.Device.pfnGetP2PPropertiesCb = zeDeviceGetP2PPropertiesOnEnter;
        epilogue.Device.pfnGetP2PPropertiesCb = zeDeviceGetP2PPropertiesOnExit;
      case ZE_FUNCTION_zeDeviceCanAccessPeer:
        prologue.Device.pfnCanAccessPeerCb = zeDeviceCanAccessPeerOnEnter;
        epilogue.Device.pfnCanAccessPeerCb = zeDeviceCanAccessPeerOnExit;
      case ZE_FUNCTION_zeDeviceSetLastLevelCacheConfig:
        prologue.Device.pfnSetLastLevelCacheConfigCb = zeDeviceSetLastLevelCacheConfigOnEnter;
        epilogue.Device.pfnSetLastLevelCacheConfigCb = zeDeviceSetLastLevelCacheConfigOnExit;
      case ZE_FUNCTION_zeDeviceSystemBarrier:
        prologue.Device.pfnSystemBarrierCb = zeDeviceSystemBarrierOnEnter;
        epilogue.Device.pfnSystemBarrierCb = zeDeviceSystemBarrierOnExit;
      case ZE_FUNCTION_zeDeviceMakeMemoryResident:
        prologue.Device.pfnMakeMemoryResidentCb = zeDeviceMakeMemoryResidentOnEnter;
        epilogue.Device.pfnMakeMemoryResidentCb = zeDeviceMakeMemoryResidentOnExit;
      case ZE_FUNCTION_zeDeviceEvictMemory:
        prologue.Device.pfnEvictMemoryCb = zeDeviceEvictMemoryOnEnter;
        epilogue.Device.pfnEvictMemoryCb = zeDeviceEvictMemoryOnExit;
      case ZE_FUNCTION_zeDeviceMakeImageResident:
        prologue.Device.pfnMakeImageResidentCb = zeDeviceMakeImageResidentOnEnter;
        epilogue.Device.pfnMakeImageResidentCb = zeDeviceMakeImageResidentOnExit;
      case ZE_FUNCTION_zeDeviceEvictImage:
        prologue.Device.pfnEvictImageCb = zeDeviceEvictImageOnEnter;
        epilogue.Device.pfnEvictImageCb = zeDeviceEvictImageOnExit;
#if ZE_ENABLE_OCL_INTEROP
      case ZE_FUNCTION_zeDeviceRegisterCLMemory:
        prologue.Device.pfnRegisterCLMemoryCb = zeDeviceRegisterCLMemoryOnEnter;
        epilogue.Device.pfnRegisterCLMemoryCb = zeDeviceRegisterCLMemoryOnExit;
#endif //ZE_ENABLE_OCL_INTEROP
#if ZE_ENABLE_OCL_INTEROP
      case ZE_FUNCTION_zeDeviceRegisterCLProgram:
        prologue.Device.pfnRegisterCLProgramCb = zeDeviceRegisterCLProgramOnEnter;
        epilogue.Device.pfnRegisterCLProgramCb = zeDeviceRegisterCLProgramOnExit;
#endif //ZE_ENABLE_OCL_INTEROP
#if ZE_ENABLE_OCL_INTEROP
      case ZE_FUNCTION_zeDeviceRegisterCLCommandQueue:
        prologue.Device.pfnRegisterCLCommandQueueCb = zeDeviceRegisterCLCommandQueueOnEnter;
        epilogue.Device.pfnRegisterCLCommandQueueCb = zeDeviceRegisterCLCommandQueueOnExit;
#endif //ZE_ENABLE_OCL_INTEROP
      case ZE_FUNCTION_zeCommandQueueCreate:
        prologue.CommandQueue.pfnCreateCb = zeCommandQueueCreateOnEnter;
        epilogue.CommandQueue.pfnCreateCb = zeCommandQueueCreateOnExit;
      case ZE_FUNCTION_zeCommandQueueDestroy:
        prologue.CommandQueue.pfnDestroyCb = zeCommandQueueDestroyOnEnter;
        epilogue.CommandQueue.pfnDestroyCb = zeCommandQueueDestroyOnExit;
      case ZE_FUNCTION_zeCommandQueueExecuteCommandLists:
        prologue.CommandQueue.pfnExecuteCommandListsCb = zeCommandQueueExecuteCommandListsOnEnter;
        epilogue.CommandQueue.pfnExecuteCommandListsCb = zeCommandQueueExecuteCommandListsOnExit;
      case ZE_FUNCTION_zeCommandQueueSynchronize:
        prologue.CommandQueue.pfnSynchronizeCb = zeCommandQueueSynchronizeOnEnter;
        epilogue.CommandQueue.pfnSynchronizeCb = zeCommandQueueSynchronizeOnExit;
      case ZE_FUNCTION_zeCommandListCreate:
        prologue.CommandList.pfnCreateCb = zeCommandListCreateOnEnter;
        epilogue.CommandList.pfnCreateCb = zeCommandListCreateOnExit;
      case ZE_FUNCTION_zeCommandListCreateImmediate:
        prologue.CommandList.pfnCreateImmediateCb = zeCommandListCreateImmediateOnEnter;
        epilogue.CommandList.pfnCreateImmediateCb = zeCommandListCreateImmediateOnExit;
      case ZE_FUNCTION_zeCommandListDestroy:
        prologue.CommandList.pfnDestroyCb = zeCommandListDestroyOnEnter;
        epilogue.CommandList.pfnDestroyCb = zeCommandListDestroyOnExit;
      case ZE_FUNCTION_zeCommandListClose:
        prologue.CommandList.pfnCloseCb = zeCommandListCloseOnEnter;
        epilogue.CommandList.pfnCloseCb = zeCommandListCloseOnExit;
      case ZE_FUNCTION_zeCommandListReset:
        prologue.CommandList.pfnResetCb = zeCommandListResetOnEnter;
        epilogue.CommandList.pfnResetCb = zeCommandListResetOnExit;
      case ZE_FUNCTION_zeCommandListAppendBarrier:
        prologue.CommandList.pfnAppendBarrierCb = zeCommandListAppendBarrierOnEnter;
        epilogue.CommandList.pfnAppendBarrierCb = zeCommandListAppendBarrierOnExit;
      case ZE_FUNCTION_zeCommandListAppendMemoryRangesBarrier:
        prologue.CommandList.pfnAppendMemoryRangesBarrierCb = zeCommandListAppendMemoryRangesBarrierOnEnter;
        epilogue.CommandList.pfnAppendMemoryRangesBarrierCb = zeCommandListAppendMemoryRangesBarrierOnExit;
      case ZE_FUNCTION_zeCommandListAppendLaunchKernel:
        prologue.CommandList.pfnAppendLaunchKernelCb = zeCommandListAppendLaunchKernelOnEnter;
        epilogue.CommandList.pfnAppendLaunchKernelCb = zeCommandListAppendLaunchKernelOnExit;
      case ZE_FUNCTION_zeCommandListAppendLaunchCooperativeKernel:
        prologue.CommandList.pfnAppendLaunchCooperativeKernelCb = zeCommandListAppendLaunchCooperativeKernelOnEnter;
        epilogue.CommandList.pfnAppendLaunchCooperativeKernelCb = zeCommandListAppendLaunchCooperativeKernelOnExit;
      case ZE_FUNCTION_zeCommandListAppendLaunchKernelIndirect:
        prologue.CommandList.pfnAppendLaunchKernelIndirectCb = zeCommandListAppendLaunchKernelIndirectOnEnter;
        epilogue.CommandList.pfnAppendLaunchKernelIndirectCb = zeCommandListAppendLaunchKernelIndirectOnExit;
      case ZE_FUNCTION_zeCommandListAppendLaunchMultipleKernelsIndirect:
        prologue.CommandList.pfnAppendLaunchMultipleKernelsIndirectCb = zeCommandListAppendLaunchMultipleKernelsIndirectOnEnter;
        epilogue.CommandList.pfnAppendLaunchMultipleKernelsIndirectCb = zeCommandListAppendLaunchMultipleKernelsIndirectOnExit;
      case ZE_FUNCTION_zeCommandListAppendSignalEvent:
        prologue.CommandList.pfnAppendSignalEventCb = zeCommandListAppendSignalEventOnEnter;
        epilogue.CommandList.pfnAppendSignalEventCb = zeCommandListAppendSignalEventOnExit;
      case ZE_FUNCTION_zeCommandListAppendWaitOnEvents:
        prologue.CommandList.pfnAppendWaitOnEventsCb = zeCommandListAppendWaitOnEventsOnEnter;
        epilogue.CommandList.pfnAppendWaitOnEventsCb = zeCommandListAppendWaitOnEventsOnExit;
      case ZE_FUNCTION_zeCommandListAppendEventReset:
        prologue.CommandList.pfnAppendEventResetCb = zeCommandListAppendEventResetOnEnter;
        epilogue.CommandList.pfnAppendEventResetCb = zeCommandListAppendEventResetOnExit;
      case ZE_FUNCTION_zeCommandListAppendMemoryCopy:
        prologue.CommandList.pfnAppendMemoryCopyCb = zeCommandListAppendMemoryCopyOnEnter;
        epilogue.CommandList.pfnAppendMemoryCopyCb = zeCommandListAppendMemoryCopyOnExit;
      case ZE_FUNCTION_zeCommandListAppendMemoryFill:
        prologue.CommandList.pfnAppendMemoryFillCb = zeCommandListAppendMemoryFillOnEnter;
        epilogue.CommandList.pfnAppendMemoryFillCb = zeCommandListAppendMemoryFillOnExit;
      case ZE_FUNCTION_zeCommandListAppendMemoryCopyRegion:
        prologue.CommandList.pfnAppendMemoryCopyRegionCb = zeCommandListAppendMemoryCopyRegionOnEnter;
        epilogue.CommandList.pfnAppendMemoryCopyRegionCb = zeCommandListAppendMemoryCopyRegionOnExit;
      case ZE_FUNCTION_zeCommandListAppendImageCopy:
        prologue.CommandList.pfnAppendImageCopyCb = zeCommandListAppendImageCopyOnEnter;
        epilogue.CommandList.pfnAppendImageCopyCb = zeCommandListAppendImageCopyOnExit;
      case ZE_FUNCTION_zeCommandListAppendImageCopyRegion:
        prologue.CommandList.pfnAppendImageCopyRegionCb = zeCommandListAppendImageCopyRegionOnEnter;
        epilogue.CommandList.pfnAppendImageCopyRegionCb = zeCommandListAppendImageCopyRegionOnExit;
      case ZE_FUNCTION_zeCommandListAppendImageCopyToMemory:
        prologue.CommandList.pfnAppendImageCopyToMemoryCb = zeCommandListAppendImageCopyToMemoryOnEnter;
        epilogue.CommandList.pfnAppendImageCopyToMemoryCb = zeCommandListAppendImageCopyToMemoryOnExit;
      case ZE_FUNCTION_zeCommandListAppendImageCopyFromMemory:
        prologue.CommandList.pfnAppendImageCopyFromMemoryCb = zeCommandListAppendImageCopyFromMemoryOnEnter;
        epilogue.CommandList.pfnAppendImageCopyFromMemoryCb = zeCommandListAppendImageCopyFromMemoryOnExit;
      case ZE_FUNCTION_zeCommandListAppendMemoryPrefetch:
        prologue.CommandList.pfnAppendMemoryPrefetchCb = zeCommandListAppendMemoryPrefetchOnEnter;
        epilogue.CommandList.pfnAppendMemoryPrefetchCb = zeCommandListAppendMemoryPrefetchOnExit;
      case ZE_FUNCTION_zeCommandListAppendMemAdvise:
        prologue.CommandList.pfnAppendMemAdviseCb = zeCommandListAppendMemAdviseOnEnter;
        epilogue.CommandList.pfnAppendMemAdviseCb = zeCommandListAppendMemAdviseOnExit;
      case ZE_FUNCTION_zeImageGetProperties:
        prologue.Image.pfnGetPropertiesCb = zeImageGetPropertiesOnEnter;
        epilogue.Image.pfnGetPropertiesCb = zeImageGetPropertiesOnExit;
      case ZE_FUNCTION_zeImageCreate:
        prologue.Image.pfnCreateCb = zeImageCreateOnEnter;
        epilogue.Image.pfnCreateCb = zeImageCreateOnExit;
      case ZE_FUNCTION_zeImageDestroy:
        prologue.Image.pfnDestroyCb = zeImageDestroyOnEnter;
        epilogue.Image.pfnDestroyCb = zeImageDestroyOnExit;
      case ZE_FUNCTION_zeModuleCreate:
        prologue.Module.pfnCreateCb = zeModuleCreateOnEnter;
        epilogue.Module.pfnCreateCb = zeModuleCreateOnExit;
      case ZE_FUNCTION_zeModuleDestroy:
        prologue.Module.pfnDestroyCb = zeModuleDestroyOnEnter;
        epilogue.Module.pfnDestroyCb = zeModuleDestroyOnExit;
      case ZE_FUNCTION_zeModuleGetNativeBinary:
        prologue.Module.pfnGetNativeBinaryCb = zeModuleGetNativeBinaryOnEnter;
        epilogue.Module.pfnGetNativeBinaryCb = zeModuleGetNativeBinaryOnExit;
      case ZE_FUNCTION_zeModuleGetGlobalPointer:
        prologue.Module.pfnGetGlobalPointerCb = zeModuleGetGlobalPointerOnEnter;
        epilogue.Module.pfnGetGlobalPointerCb = zeModuleGetGlobalPointerOnExit;
      case ZE_FUNCTION_zeModuleGetKernelNames:
        prologue.Module.pfnGetKernelNamesCb = zeModuleGetKernelNamesOnEnter;
        epilogue.Module.pfnGetKernelNamesCb = zeModuleGetKernelNamesOnExit;
      case ZE_FUNCTION_zeModuleGetFunctionPointer:
        prologue.Module.pfnGetFunctionPointerCb = zeModuleGetFunctionPointerOnEnter;
        epilogue.Module.pfnGetFunctionPointerCb = zeModuleGetFunctionPointerOnExit;
      case ZE_FUNCTION_zeModuleBuildLogDestroy:
        prologue.ModuleBuildLog.pfnDestroyCb = zeModuleBuildLogDestroyOnEnter;
        epilogue.ModuleBuildLog.pfnDestroyCb = zeModuleBuildLogDestroyOnExit;
      case ZE_FUNCTION_zeModuleBuildLogGetString:
        prologue.ModuleBuildLog.pfnGetStringCb = zeModuleBuildLogGetStringOnEnter;
        epilogue.ModuleBuildLog.pfnGetStringCb = zeModuleBuildLogGetStringOnExit;
      case ZE_FUNCTION_zeKernelCreate:
        prologue.Kernel.pfnCreateCb = zeKernelCreateOnEnter;
        epilogue.Kernel.pfnCreateCb = zeKernelCreateOnExit;
      case ZE_FUNCTION_zeKernelDestroy:
        prologue.Kernel.pfnDestroyCb = zeKernelDestroyOnEnter;
        epilogue.Kernel.pfnDestroyCb = zeKernelDestroyOnExit;
      case ZE_FUNCTION_zeKernelSetIntermediateCacheConfig:
        prologue.Kernel.pfnSetIntermediateCacheConfigCb = zeKernelSetIntermediateCacheConfigOnEnter;
        epilogue.Kernel.pfnSetIntermediateCacheConfigCb = zeKernelSetIntermediateCacheConfigOnExit;
      case ZE_FUNCTION_zeKernelSetGroupSize:
        prologue.Kernel.pfnSetGroupSizeCb = zeKernelSetGroupSizeOnEnter;
        epilogue.Kernel.pfnSetGroupSizeCb = zeKernelSetGroupSizeOnExit;
      case ZE_FUNCTION_zeKernelSuggestGroupSize:
        prologue.Kernel.pfnSuggestGroupSizeCb = zeKernelSuggestGroupSizeOnEnter;
        epilogue.Kernel.pfnSuggestGroupSizeCb = zeKernelSuggestGroupSizeOnExit;
      case ZE_FUNCTION_zeKernelSuggestMaxCooperativeGroupCount:
        prologue.Kernel.pfnSuggestMaxCooperativeGroupCountCb = zeKernelSuggestMaxCooperativeGroupCountOnEnter;
        epilogue.Kernel.pfnSuggestMaxCooperativeGroupCountCb = zeKernelSuggestMaxCooperativeGroupCountOnExit;
      case ZE_FUNCTION_zeKernelSetArgumentValue:
        prologue.Kernel.pfnSetArgumentValueCb = zeKernelSetArgumentValueOnEnter;
        epilogue.Kernel.pfnSetArgumentValueCb = zeKernelSetArgumentValueOnExit;
      case ZE_FUNCTION_zeKernelSetAttribute:
        prologue.Kernel.pfnSetAttributeCb = zeKernelSetAttributeOnEnter;
        epilogue.Kernel.pfnSetAttributeCb = zeKernelSetAttributeOnExit;
      case ZE_FUNCTION_zeKernelGetAttribute:
        prologue.Kernel.pfnGetAttributeCb = zeKernelGetAttributeOnEnter;
        epilogue.Kernel.pfnGetAttributeCb = zeKernelGetAttributeOnExit;
      case ZE_FUNCTION_zeKernelGetProperties:
        prologue.Kernel.pfnGetPropertiesCb = zeKernelGetPropertiesOnEnter;
        epilogue.Kernel.pfnGetPropertiesCb = zeKernelGetPropertiesOnExit;
      case ZE_FUNCTION_zeEventPoolCreate:
        prologue.EventPool.pfnCreateCb = zeEventPoolCreateOnEnter;
        epilogue.EventPool.pfnCreateCb = zeEventPoolCreateOnExit;
      case ZE_FUNCTION_zeEventPoolDestroy:
        prologue.EventPool.pfnDestroyCb = zeEventPoolDestroyOnEnter;
        epilogue.EventPool.pfnDestroyCb = zeEventPoolDestroyOnExit;
      case ZE_FUNCTION_zeEventPoolGetIpcHandle:
        prologue.EventPool.pfnGetIpcHandleCb = zeEventPoolGetIpcHandleOnEnter;
        epilogue.EventPool.pfnGetIpcHandleCb = zeEventPoolGetIpcHandleOnExit;
      case ZE_FUNCTION_zeEventPoolOpenIpcHandle:
        prologue.EventPool.pfnOpenIpcHandleCb = zeEventPoolOpenIpcHandleOnEnter;
        epilogue.EventPool.pfnOpenIpcHandleCb = zeEventPoolOpenIpcHandleOnExit;
      case ZE_FUNCTION_zeEventPoolCloseIpcHandle:
        prologue.EventPool.pfnCloseIpcHandleCb = zeEventPoolCloseIpcHandleOnEnter;
        epilogue.EventPool.pfnCloseIpcHandleCb = zeEventPoolCloseIpcHandleOnExit;
      case ZE_FUNCTION_zeEventCreate:
        prologue.Event.pfnCreateCb = zeEventCreateOnEnter;
        epilogue.Event.pfnCreateCb = zeEventCreateOnExit;
      case ZE_FUNCTION_zeEventDestroy:
        prologue.Event.pfnDestroyCb = zeEventDestroyOnEnter;
        epilogue.Event.pfnDestroyCb = zeEventDestroyOnExit;
      case ZE_FUNCTION_zeEventHostSignal:
        prologue.Event.pfnHostSignalCb = zeEventHostSignalOnEnter;
        epilogue.Event.pfnHostSignalCb = zeEventHostSignalOnExit;
      case ZE_FUNCTION_zeEventHostSynchronize:
        prologue.Event.pfnHostSynchronizeCb = zeEventHostSynchronizeOnEnter;
        epilogue.Event.pfnHostSynchronizeCb = zeEventHostSynchronizeOnExit;
      case ZE_FUNCTION_zeEventQueryStatus:
        prologue.Event.pfnQueryStatusCb = zeEventQueryStatusOnEnter;
        epilogue.Event.pfnQueryStatusCb = zeEventQueryStatusOnExit;
      case ZE_FUNCTION_zeEventHostReset:
        prologue.Event.pfnHostResetCb = zeEventHostResetOnEnter;
        epilogue.Event.pfnHostResetCb = zeEventHostResetOnExit;
      case ZE_FUNCTION_zeEventGetTimestamp:
        prologue.Event.pfnGetTimestampCb = zeEventGetTimestampOnEnter;
        epilogue.Event.pfnGetTimestampCb = zeEventGetTimestampOnExit;
      case ZE_FUNCTION_zeFenceCreate:
        prologue.Fence.pfnCreateCb = zeFenceCreateOnEnter;
        epilogue.Fence.pfnCreateCb = zeFenceCreateOnExit;
      case ZE_FUNCTION_zeFenceDestroy:
        prologue.Fence.pfnDestroyCb = zeFenceDestroyOnEnter;
        epilogue.Fence.pfnDestroyCb = zeFenceDestroyOnExit;
      case ZE_FUNCTION_zeFenceHostSynchronize:
        prologue.Fence.pfnHostSynchronizeCb = zeFenceHostSynchronizeOnEnter;
        epilogue.Fence.pfnHostSynchronizeCb = zeFenceHostSynchronizeOnExit;
      case ZE_FUNCTION_zeFenceQueryStatus:
        prologue.Fence.pfnQueryStatusCb = zeFenceQueryStatusOnEnter;
        epilogue.Fence.pfnQueryStatusCb = zeFenceQueryStatusOnExit;
      case ZE_FUNCTION_zeFenceReset:
        prologue.Fence.pfnResetCb = zeFenceResetOnEnter;
        epilogue.Fence.pfnResetCb = zeFenceResetOnExit;
      case ZE_FUNCTION_zeSamplerCreate:
        prologue.Sampler.pfnCreateCb = zeSamplerCreateOnEnter;
        epilogue.Sampler.pfnCreateCb = zeSamplerCreateOnExit;
      case ZE_FUNCTION_zeSamplerDestroy:
        prologue.Sampler.pfnDestroyCb = zeSamplerDestroyOnEnter;
        epilogue.Sampler.pfnDestroyCb = zeSamplerDestroyOnExit;
      default:
        break;
    }
  }

  ze_result_t status = ZE_RESULT_SUCCESS;
  status = zetTracerSetPrologues(tracer, &prologue);
  assert(status == ZE_RESULT_SUCCESS);
  status = zetTracerSetEpilogues(tracer, &epilogue);
  assert(status == ZE_RESULT_SUCCESS);
}

} // namespace ze_tracing