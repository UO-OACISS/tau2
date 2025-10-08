#include <string>
#include <cstdio>
#include <cstdint>
#define TO_HEX_STRING(str, val) \
    {char buffer[32]; \
    std::sprintf(buffer, "0x%lx", (uintptr_t)(val)); \
    str += std::string(buffer); \
    }

static const char* GetResultString(unsigned result) {
  switch (result) {
    case ZE_RESULT_SUCCESS:
      return "ZE_RESULT_SUCCESS";
    case ZE_RESULT_NOT_READY:
      return "ZE_RESULT_NOT_READY";
    case ZE_RESULT_ERROR_DEVICE_LOST:
      return "ZE_RESULT_ERROR_DEVICE_LOST";
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
      return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
      return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
      return "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
      return "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
      return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
      return "ZE_RESULT_ERROR_NOT_AVAILABLE";
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
      return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
    case ZE_RESULT_WARNING_DROPPED_DATA:
      return "ZE_RESULT_WARNING_DROPPED_DATA";
    case ZE_RESULT_ERROR_UNINITIALIZED:
      return "ZE_RESULT_ERROR_UNINITIALIZED";
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
      return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
    case ZE_RESULT_ERROR_INVALID_SIZE:
      return "ZE_RESULT_ERROR_INVALID_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
      return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
      return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
      return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
      return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
      return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
      return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
      return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
      return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
      return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
      return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
      return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
      return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
      return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
    case ZE_RESULT_WARNING_ACTION_REQUIRED:
      return "ZE_RESULT_WARNING_ACTION_REQUIRED";
    case ZE_RESULT_ERROR_INVALID_KERNEL_HANDLE:
      return "ZE_RESULT_ERROR_INVALID_KERNEL_HANDLE";
    case ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX:
      return "ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX";
    case ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE:
      return "ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE";
    case ZE_RESULT_EXP_ERROR_REMOTE_DEVICE:
      return "ZE_RESULT_EXP_ERROR_REMOTE_DEVICE";
    case ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE:
      return "ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE";
    case ZE_RESULT_EXP_RTAS_BUILD_RETRY:
      return "ZE_RESULT_EXP_RTAS_BUILD_RETRY";
    case ZE_RESULT_EXP_RTAS_BUILD_DEFERRED:
      return "ZE_RESULT_EXP_RTAS_BUILD_DEFERRED";
    case ZE_RESULT_ERROR_UNKNOWN:
      return "ZE_RESULT_ERROR_UNKNOWN";
    case ZE_RESULT_FORCE_UINT32:
      return "ZE_RESULT_FORCE_UINT32";
    default:
      break;
  }
  return "UNKNOWN";
}

static const char* GetStructureTypeString(unsigned structure_type) {
  switch (structure_type) {
    case ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES";
    case ZE_STRUCTURE_TYPE_CONTEXT_DESC:
      return "ZE_STRUCTURE_TYPE_CONTEXT_DESC";
    case ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC:
      return "ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC";
    case ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC:
      return "ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC";
    case ZE_STRUCTURE_TYPE_EVENT_POOL_DESC:
      return "ZE_STRUCTURE_TYPE_EVENT_POOL_DESC";
    case ZE_STRUCTURE_TYPE_EVENT_DESC:
      return "ZE_STRUCTURE_TYPE_EVENT_DESC";
    case ZE_STRUCTURE_TYPE_FENCE_DESC:
      return "ZE_STRUCTURE_TYPE_FENCE_DESC";
    case ZE_STRUCTURE_TYPE_IMAGE_DESC:
      return "ZE_STRUCTURE_TYPE_IMAGE_DESC";
    case ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC";
    case ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC:
      return "ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC";
    case ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD";
    case ZE_STRUCTURE_TYPE_MODULE_DESC:
      return "ZE_STRUCTURE_TYPE_MODULE_DESC";
    case ZE_STRUCTURE_TYPE_MODULE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_MODULE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_KERNEL_DESC:
      return "ZE_STRUCTURE_TYPE_KERNEL_DESC";
    case ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES";
    case ZE_STRUCTURE_TYPE_SAMPLER_DESC:
      return "ZE_STRUCTURE_TYPE_SAMPLER_DESC";
    case ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC:
      return "ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC";
    case ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32";
    case ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32";
    case ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_RAYTRACING_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC";
    case ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_FLOAT_ATOMIC_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_CACHE_RESERVATION_EXT_DESC";
    case ZE_STRUCTURE_TYPE_EU_COUNT_EXT:
      return "ZE_STRUCTURE_TYPE_EU_COUNT_EXT";
    case ZE_STRUCTURE_TYPE_SRGB_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_SRGB_EXT_DESC";
    case ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_LINKAGE_INSPECTION_EXT_DESC";
    case ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DRIVER_MEMORY_FREE_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC";
    case ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_MEMORY_COMPRESSION_HINTS_EXT_DESC";
    case ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_IMAGE_ALLOCATION_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT:
      return "ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT";
    case ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXT_DESC";
    case ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC";
    case ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_SCHEDULING_HINT_EXP_DESC";
    case ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_IMAGE_VIEW_PLANAR_EXP_DESC";
    case ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2:
      return "ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2";
    case ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_IMAGE_MEMORY_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_POWER_SAVING_HINT_EXP_DESC";
    case ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_COPY_BANDWIDTH_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_FABRIC_VERTEX_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_FABRIC_EDGE_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_DESC";
    case ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXP_DESC";
    case ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_RTAS_PARALLEL_OPERATION_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS:
      return "ZE_STRUCTURE_TYPE_RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS";
    case ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_COUNTER_BASED_EVENT_POOL_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_GROUP_SIZE_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_GROUP_SIZE_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_GLOBAL_OFFSET_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_GLOBAL_OFFSET_EXP_DESC";
    case ZE_STRUCTURE_TYPE_PITCHED_ALLOC_DEVICE_EXP_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_PITCHED_ALLOC_DEVICE_EXP_PROPERTIES";
    case ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_BINDLESS_IMAGE_EXP_DESC";
    case ZE_STRUCTURE_TYPE_PITCHED_IMAGE_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_PITCHED_IMAGE_EXP_DESC";
    case ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC:
      return "ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC";
    case ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC:
      return "ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC";
    case ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_EXT_DESC";
    case ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WIN32_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WIN32_EXT_DESC";
    case ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_FD_EXT_DESC:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_FD_EXT_DESC";
    case ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXT:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_EXT";
    case ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXT:
      return "ZE_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_WAIT_PARAMS_EXT";
    case ZE_STRUCTURE_TYPE_DRIVER_DDI_HANDLES_EXT_PROPERTIES:
      return "ZE_STRUCTURE_TYPE_DRIVER_DDI_HANDLES_EXT_PROPERTIES";
    case ZE_STRUCTURE_TYPE_FORCE_UINT32:
      return "ZE_STRUCTURE_TYPE_FORCE_UINT32";
    default:
      break;
  }
  return "UNKNOWN";
}

static void zeInitOnEnter(
    ze_init_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeInit" ;
  TAU_L0_enter_event( func_name );
}

static void zeInitOnExit(
    ze_init_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeInit" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetOnEnter(
    ze_driver_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGet" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetOnExit(
    ze_driver_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGet" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetApiVersionOnEnter(
    ze_driver_get_api_version_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGetApiVersion" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetApiVersionOnExit(
    ze_driver_get_api_version_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGetApiVersion" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetPropertiesOnEnter(
    ze_driver_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGetProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetPropertiesOnExit(
    ze_driver_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGetProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetIpcPropertiesOnEnter(
    ze_driver_get_ipc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGetIpcProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetIpcPropertiesOnExit(
    ze_driver_get_ipc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGetIpcProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetExtensionPropertiesOnEnter(
    ze_driver_get_extension_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGetExtensionProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetExtensionPropertiesOnExit(
    ze_driver_get_extension_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGetExtensionProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetOnEnter(
    ze_device_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGet" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetOnExit(
    ze_device_get_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGet" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetSubDevicesOnEnter(
    ze_device_get_sub_devices_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetSubDevices" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetSubDevicesOnExit(
    ze_device_get_sub_devices_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetSubDevices" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetPropertiesOnEnter(
    ze_device_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetPropertiesOnExit(
    ze_device_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetComputePropertiesOnEnter(
    ze_device_get_compute_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetComputeProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetComputePropertiesOnExit(
    ze_device_get_compute_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetComputeProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetModulePropertiesOnEnter(
    ze_device_get_module_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetModuleProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetModulePropertiesOnExit(
    ze_device_get_module_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetModuleProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetCommandQueueGroupPropertiesOnEnter(
    ze_device_get_command_queue_group_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetCommandQueueGroupProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetCommandQueueGroupPropertiesOnExit(
    ze_device_get_command_queue_group_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetCommandQueueGroupProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetMemoryPropertiesOnEnter(
    ze_device_get_memory_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetMemoryProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetMemoryPropertiesOnExit(
    ze_device_get_memory_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetMemoryProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetMemoryAccessPropertiesOnEnter(
    ze_device_get_memory_access_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetMemoryAccessProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetMemoryAccessPropertiesOnExit(
    ze_device_get_memory_access_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetMemoryAccessProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetCachePropertiesOnEnter(
    ze_device_get_cache_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetCacheProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetCachePropertiesOnExit(
    ze_device_get_cache_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetCacheProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetImagePropertiesOnEnter(
    ze_device_get_image_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetImageProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetImagePropertiesOnExit(
    ze_device_get_image_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetImageProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetExternalMemoryPropertiesOnEnter(
    ze_device_get_external_memory_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetExternalMemoryProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetExternalMemoryPropertiesOnExit(
    ze_device_get_external_memory_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetExternalMemoryProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetP2PPropertiesOnEnter(
    ze_device_get_p2_p_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetP2PProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetP2PPropertiesOnExit(
    ze_device_get_p2_p_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetP2PProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceCanAccessPeerOnEnter(
    ze_device_can_access_peer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceCanAccessPeer" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceCanAccessPeerOnExit(
    ze_device_can_access_peer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceCanAccessPeer" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetStatusOnEnter(
    ze_device_get_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetStatus" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetStatusOnExit(
    ze_device_get_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetStatus" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextCreateOnEnter(
    ze_context_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextCreateOnExit(
    ze_context_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextDestroyOnEnter(
    ze_context_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextDestroyOnExit(
    ze_context_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitContextDestroy(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeContextDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextGetStatusOnEnter(
    ze_context_get_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextGetStatus" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextGetStatusOnExit(
    ze_context_get_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextGetStatus" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextSystemBarrierOnEnter(
    ze_context_system_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextSystemBarrier" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextSystemBarrierOnExit(
    ze_context_system_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextSystemBarrier" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextMakeMemoryResidentOnEnter(
    ze_context_make_memory_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextMakeMemoryResident" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextMakeMemoryResidentOnExit(
    ze_context_make_memory_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextMakeMemoryResident" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextEvictMemoryOnEnter(
    ze_context_evict_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextEvictMemory" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextEvictMemoryOnExit(
    ze_context_evict_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextEvictMemory" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextMakeImageResidentOnEnter(
    ze_context_make_image_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextMakeImageResident" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextMakeImageResidentOnExit(
    ze_context_make_image_resident_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextMakeImageResident" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextEvictImageOnEnter(
    ze_context_evict_image_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextEvictImage" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextEvictImageOnExit(
    ze_context_evict_image_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextEvictImage" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandQueueCreateOnEnter(
    ze_command_queue_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandQueueCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandQueueCreateOnExit(
    ze_command_queue_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitCommandQueueCreate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeCommandQueueCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandQueueDestroyOnEnter(
    ze_command_queue_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandQueueDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandQueueDestroyOnExit(
    ze_command_queue_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitCommandQueueDestroy(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeCommandQueueDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandQueueExecuteCommandListsOnEnter(
    ze_command_queue_execute_command_lists_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandQueueExecuteCommandLists(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandQueueExecuteCommandLists" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandQueueExecuteCommandListsOnExit(
    ze_command_queue_execute_command_lists_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandQueueExecuteCommandLists(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandQueueExecuteCommandLists" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandQueueSynchronizeOnEnter(
    ze_command_queue_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandQueueSynchronize" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandQueueSynchronizeOnExit(
    ze_command_queue_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandQueueSynchronize(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandQueueSynchronize" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListCreateOnEnter(
    ze_command_list_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListCreateOnExit(
    ze_command_list_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitCommandListCreate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListCreateImmediateOnEnter(
    ze_command_list_create_immediate_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListCreateImmediate" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListCreateImmediateOnExit(
    ze_command_list_create_immediate_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitCommandListCreateImmediate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListCreateImmediate" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListDestroyOnEnter(
    ze_command_list_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListDestroyOnExit(
    ze_command_list_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitCommandListDestroy(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListCloseOnEnter(
    ze_command_list_close_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListClose(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListClose" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListCloseOnExit(
    ze_command_list_close_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListClose" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListResetOnEnter(
    ze_command_list_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListReset" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListResetOnExit(
    ze_command_list_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitCommandListReset(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListReset" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendWriteGlobalTimestampOnEnter(
    ze_command_list_append_write_global_timestamp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendWriteGlobalTimestamp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendWriteGlobalTimestampOnExit(
    ze_command_list_append_write_global_timestamp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendWriteGlobalTimestamp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendBarrierOnEnter(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendBarrier(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendBarrier" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendBarrierOnExit(
    ze_command_list_append_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendBarrier(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendBarrier" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemoryRangesBarrierOnEnter(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendMemoryRangesBarrier(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendMemoryRangesBarrier" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemoryRangesBarrierOnExit(
    ze_command_list_append_memory_ranges_barrier_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendMemoryRangesBarrier(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendMemoryRangesBarrier" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemoryCopyOnEnter(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendMemoryCopy(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendMemoryCopy" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemoryCopyOnExit(
    ze_command_list_append_memory_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendMemoryCopy(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendMemoryCopy" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemoryFillOnEnter(
    ze_command_list_append_memory_fill_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendMemoryFill(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendMemoryFill" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemoryFillOnExit(
    ze_command_list_append_memory_fill_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendMemoryFill(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendMemoryFill" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemoryCopyRegionOnEnter(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendMemoryCopyRegion(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendMemoryCopyRegion" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemoryCopyRegionOnExit(
    ze_command_list_append_memory_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendMemoryCopyRegion(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendMemoryCopyRegion" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemoryCopyFromContextOnEnter(
    ze_command_list_append_memory_copy_from_context_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendMemoryCopyFromContext(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendMemoryCopyFromContext" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemoryCopyFromContextOnExit(
    ze_command_list_append_memory_copy_from_context_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendMemoryCopyFromContext(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendMemoryCopyFromContext" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendImageCopyOnEnter(
    ze_command_list_append_image_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendImageCopy(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendImageCopy" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendImageCopyOnExit(
    ze_command_list_append_image_copy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendImageCopy(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendImageCopy" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendImageCopyRegionOnEnter(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendImageCopyRegion(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendImageCopyRegion" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendImageCopyRegionOnExit(
    ze_command_list_append_image_copy_region_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendImageCopyRegion(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendImageCopyRegion" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendImageCopyToMemoryOnEnter(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendImageCopyToMemory(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendImageCopyToMemory" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendImageCopyToMemoryOnExit(
    ze_command_list_append_image_copy_to_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendImageCopyToMemory(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendImageCopyToMemory" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendImageCopyFromMemoryOnEnter(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendImageCopyFromMemory(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendImageCopyFromMemory" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendImageCopyFromMemoryOnExit(
    ze_command_list_append_image_copy_from_memory_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendImageCopyFromMemory(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendImageCopyFromMemory" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemoryPrefetchOnEnter(
    ze_command_list_append_memory_prefetch_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendMemoryPrefetch" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemoryPrefetchOnExit(
    ze_command_list_append_memory_prefetch_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendMemoryPrefetch" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendMemAdviseOnEnter(
    ze_command_list_append_mem_advise_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendMemAdvise" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendMemAdviseOnExit(
    ze_command_list_append_mem_advise_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendMemAdvise" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendSignalEventOnEnter(
    ze_command_list_append_signal_event_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendSignalEvent" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendSignalEventOnExit(
    ze_command_list_append_signal_event_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendSignalEvent" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendWaitOnEventsOnEnter(
    ze_command_list_append_wait_on_events_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendWaitOnEvents" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendWaitOnEventsOnExit(
    ze_command_list_append_wait_on_events_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendWaitOnEvents" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendEventResetOnEnter(
    ze_command_list_append_event_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendEventReset(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendEventReset" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendEventResetOnExit(
    ze_command_list_append_event_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendEventReset(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendEventReset" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendQueryKernelTimestampsOnEnter(
    ze_command_list_append_query_kernel_timestamps_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendQueryKernelTimestamps" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendQueryKernelTimestampsOnExit(
    ze_command_list_append_query_kernel_timestamps_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendQueryKernelTimestamps" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendLaunchKernelOnEnter(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendLaunchKernel(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendLaunchKernel" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendLaunchKernelOnExit(
    ze_command_list_append_launch_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendLaunchKernel(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendLaunchKernel" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendLaunchCooperativeKernelOnEnter(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendLaunchCooperativeKernel(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendLaunchCooperativeKernel" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendLaunchCooperativeKernelOnExit(
    ze_command_list_append_launch_cooperative_kernel_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendLaunchCooperativeKernel(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendLaunchCooperativeKernel" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendLaunchKernelIndirectOnEnter(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListAppendLaunchKernelIndirect(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListAppendLaunchKernelIndirect" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendLaunchKernelIndirectOnExit(
    ze_command_list_append_launch_kernel_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListAppendLaunchKernelIndirect(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListAppendLaunchKernelIndirect" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendLaunchMultipleKernelsIndirectOnEnter(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendLaunchMultipleKernelsIndirect" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendLaunchMultipleKernelsIndirectOnExit(
    ze_command_list_append_launch_multiple_kernels_indirect_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendLaunchMultipleKernelsIndirect" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageGetPropertiesOnEnter(
    ze_image_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageGetProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageGetPropertiesOnExit(
    ze_image_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeImageGetProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageCreateOnEnter(
    ze_image_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageCreateOnExit(
    ze_image_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitImageCreate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeImageCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageDestroyOnEnter(
    ze_image_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageDestroyOnExit(
    ze_image_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitImageDestroy(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeImageDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemAllocSharedOnEnter(
    ze_mem_alloc_shared_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemAllocShared" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemAllocSharedOnExit(
    ze_mem_alloc_shared_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemAllocShared" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemAllocDeviceOnEnter(
    ze_mem_alloc_device_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemAllocDevice" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemAllocDeviceOnExit(
    ze_mem_alloc_device_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemAllocDevice" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemAllocHostOnEnter(
    ze_mem_alloc_host_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemAllocHost" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemAllocHostOnExit(
    ze_mem_alloc_host_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemAllocHost" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemFreeOnEnter(
    ze_mem_free_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemFree" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemFreeOnExit(
    ze_mem_free_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemFree" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetAllocPropertiesOnEnter(
    ze_mem_get_alloc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetAllocProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetAllocPropertiesOnExit(
    ze_mem_get_alloc_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetAllocProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetAddressRangeOnEnter(
    ze_mem_get_address_range_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetAddressRange" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetAddressRangeOnExit(
    ze_mem_get_address_range_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetAddressRange" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetIpcHandleOnEnter(
    ze_mem_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetIpcHandleOnExit(
    ze_mem_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemOpenIpcHandleOnEnter(
    ze_mem_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemOpenIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemOpenIpcHandleOnExit(
    ze_mem_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemOpenIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemCloseIpcHandleOnEnter(
    ze_mem_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemCloseIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemCloseIpcHandleOnExit(
    ze_mem_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemCloseIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeFenceCreateOnEnter(
    ze_fence_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFenceCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeFenceCreateOnExit(
    ze_fence_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFenceCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeFenceDestroyOnEnter(
    ze_fence_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFenceDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeFenceDestroyOnExit(
    ze_fence_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFenceDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeFenceHostSynchronizeOnEnter(
    ze_fence_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFenceHostSynchronize" ;
  TAU_L0_enter_event( func_name );
}

static void zeFenceHostSynchronizeOnExit(
    ze_fence_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitFenceHostSynchronize(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeFenceHostSynchronize" ;
  TAU_L0_exit_event( func_name );
}

static void zeFenceQueryStatusOnEnter(
    ze_fence_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFenceQueryStatus" ;
  TAU_L0_enter_event( func_name );
}

static void zeFenceQueryStatusOnExit(
    ze_fence_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFenceQueryStatus" ;
  TAU_L0_exit_event( func_name );
}

static void zeFenceResetOnEnter(
    ze_fence_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFenceReset" ;
  TAU_L0_enter_event( func_name );
}

static void zeFenceResetOnExit(
    ze_fence_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFenceReset" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolCreateOnEnter(
    ze_event_pool_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterEventPoolCreate(params, global_user_data, instance_user_data);



  const char* func_name = "zeEventPoolCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolCreateOnExit(
    ze_event_pool_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitEventPoolCreate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeEventPoolCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolDestroyOnEnter(
    ze_event_pool_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolDestroyOnExit(
    ze_event_pool_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolGetIpcHandleOnEnter(
    ze_event_pool_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolGetIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolGetIpcHandleOnExit(
    ze_event_pool_get_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolGetIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolOpenIpcHandleOnEnter(
    ze_event_pool_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolOpenIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolOpenIpcHandleOnExit(
    ze_event_pool_open_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolOpenIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolCloseIpcHandleOnEnter(
    ze_event_pool_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolCloseIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolCloseIpcHandleOnExit(
    ze_event_pool_close_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolCloseIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventCreateOnEnter(
    ze_event_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventCreateOnExit(
    ze_event_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventDestroyOnEnter(
    ze_event_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;

  //Inserted call
    OnEnterEventDestroy(params, global_user_data, instance_user_data, &kids);
    if (kids.size() != 0) {
        ze_instance_data.kid = kids[0];
    }
    else {
        ze_instance_data.kid = (uint64_t)(-1);
    }



  const char* func_name = "zeEventDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventDestroyOnExit(
    ze_event_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  if (ze_instance_data.kid != (uint64_t)(-1)) {
      kids.push_back(ze_instance_data.kid);
  }


  const char* func_name = "zeEventDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventHostSignalOnEnter(
    ze_event_host_signal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventHostSignal" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventHostSignalOnExit(
    ze_event_host_signal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventHostSignal" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventHostSynchronizeOnEnter(
    ze_event_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventHostSynchronize" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventHostSynchronizeOnExit(
    ze_event_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitEventHostSynchronize(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeEventHostSynchronize" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventQueryStatusOnEnter(
    ze_event_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventQueryStatus" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventQueryStatusOnExit(
    ze_event_query_status_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitEventQueryStatus(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeEventQueryStatus" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventHostResetOnEnter(
    ze_event_host_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;

  //Inserted call
    OnEnterEventHostReset(params, global_user_data, instance_user_data, &kids);
    if (kids.size() != 0) {
        ze_instance_data.kid = kids[0];
    }
    else {
        ze_instance_data.kid = (uint64_t)(-1);
    }



  const char* func_name = "zeEventHostReset" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventHostResetOnExit(
    ze_event_host_reset_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  if (ze_instance_data.kid != (uint64_t)(-1)) {
      kids.push_back(ze_instance_data.kid);
  }


  const char* func_name = "zeEventHostReset" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventQueryKernelTimestampOnEnter(
    ze_event_query_kernel_timestamp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventQueryKernelTimestamp" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventQueryKernelTimestampOnExit(
    ze_event_query_kernel_timestamp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventQueryKernelTimestamp" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleCreateOnEnter(
    ze_module_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleCreateOnExit(
    ze_module_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitModuleCreate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeModuleCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleDestroyOnEnter(
    ze_module_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterModuleDestroy(params, global_user_data, instance_user_data);



  const char* func_name = "zeModuleDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleDestroyOnExit(
    ze_module_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleDynamicLinkOnEnter(
    ze_module_dynamic_link_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleDynamicLink" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleDynamicLinkOnExit(
    ze_module_dynamic_link_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleDynamicLink" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleGetNativeBinaryOnEnter(
    ze_module_get_native_binary_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleGetNativeBinary" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleGetNativeBinaryOnExit(
    ze_module_get_native_binary_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleGetNativeBinary" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleGetGlobalPointerOnEnter(
    ze_module_get_global_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleGetGlobalPointer" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleGetGlobalPointerOnExit(
    ze_module_get_global_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleGetGlobalPointer" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleGetKernelNamesOnEnter(
    ze_module_get_kernel_names_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleGetKernelNames" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleGetKernelNamesOnExit(
    ze_module_get_kernel_names_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleGetKernelNames" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleGetPropertiesOnEnter(
    ze_module_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleGetProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleGetPropertiesOnExit(
    ze_module_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleGetProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleGetFunctionPointerOnEnter(
    ze_module_get_function_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleGetFunctionPointer" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleGetFunctionPointerOnExit(
    ze_module_get_function_pointer_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleGetFunctionPointer" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleBuildLogDestroyOnEnter(
    ze_module_build_log_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleBuildLogDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleBuildLogDestroyOnExit(
    ze_module_build_log_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleBuildLogDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleBuildLogGetStringOnEnter(
    ze_module_build_log_get_string_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleBuildLogGetString" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleBuildLogGetStringOnExit(
    ze_module_build_log_get_string_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleBuildLogGetString" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelCreateOnEnter(
    ze_kernel_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelCreateOnExit(
    ze_kernel_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitKernelCreate(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeKernelCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelDestroyOnEnter(
    ze_kernel_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelDestroyOnExit(
    ze_kernel_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitKernelDestroy(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeKernelDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSetCacheConfigOnEnter(
    ze_kernel_set_cache_config_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSetCacheConfig" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSetCacheConfigOnExit(
    ze_kernel_set_cache_config_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSetCacheConfig" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSetGroupSizeOnEnter(
    ze_kernel_set_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSetGroupSize" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSetGroupSizeOnExit(
    ze_kernel_set_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  //Inserted call
    OnExitKernelSetGroupSize(params, result, global_user_data, instance_user_data);



  const char* func_name = "zeKernelSetGroupSize" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSuggestGroupSizeOnEnter(
    ze_kernel_suggest_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSuggestGroupSize" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSuggestGroupSizeOnExit(
    ze_kernel_suggest_group_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSuggestGroupSize" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSuggestMaxCooperativeGroupCountOnEnter(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSuggestMaxCooperativeGroupCount" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSuggestMaxCooperativeGroupCountOnExit(
    ze_kernel_suggest_max_cooperative_group_count_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSuggestMaxCooperativeGroupCount" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSetArgumentValueOnEnter(
    ze_kernel_set_argument_value_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSetArgumentValue" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSetArgumentValueOnExit(
    ze_kernel_set_argument_value_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSetArgumentValue" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSetIndirectAccessOnEnter(
    ze_kernel_set_indirect_access_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSetIndirectAccess" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSetIndirectAccessOnExit(
    ze_kernel_set_indirect_access_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSetIndirectAccess" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelGetIndirectAccessOnEnter(
    ze_kernel_get_indirect_access_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelGetIndirectAccess" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelGetIndirectAccessOnExit(
    ze_kernel_get_indirect_access_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelGetIndirectAccess" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelGetSourceAttributesOnEnter(
    ze_kernel_get_source_attributes_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelGetSourceAttributes" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelGetSourceAttributesOnExit(
    ze_kernel_get_source_attributes_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelGetSourceAttributes" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelGetPropertiesOnEnter(
    ze_kernel_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelGetProperties" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelGetPropertiesOnExit(
    ze_kernel_get_properties_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelGetProperties" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelGetNameOnEnter(
    ze_kernel_get_name_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelGetName" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelGetNameOnExit(
    ze_kernel_get_name_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelGetName" ;
  TAU_L0_exit_event( func_name );
}

static void zeSamplerCreateOnEnter(
    ze_sampler_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeSamplerCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zeSamplerCreateOnExit(
    ze_sampler_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeSamplerCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zeSamplerDestroyOnEnter(
    ze_sampler_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeSamplerDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zeSamplerDestroyOnExit(
    ze_sampler_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeSamplerDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zePhysicalMemCreateOnEnter(
    ze_physical_mem_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zePhysicalMemCreate" ;
  TAU_L0_enter_event( func_name );
}

static void zePhysicalMemCreateOnExit(
    ze_physical_mem_create_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zePhysicalMemCreate" ;
  TAU_L0_exit_event( func_name );
}

static void zePhysicalMemDestroyOnEnter(
    ze_physical_mem_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zePhysicalMemDestroy" ;
  TAU_L0_enter_event( func_name );
}

static void zePhysicalMemDestroyOnExit(
    ze_physical_mem_destroy_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zePhysicalMemDestroy" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemReserveOnEnter(
    ze_virtual_mem_reserve_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemReserve" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemReserveOnExit(
    ze_virtual_mem_reserve_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemReserve" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemFreeOnEnter(
    ze_virtual_mem_free_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemFree" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemFreeOnExit(
    ze_virtual_mem_free_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemFree" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemQueryPageSizeOnEnter(
    ze_virtual_mem_query_page_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemQueryPageSize" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemQueryPageSizeOnExit(
    ze_virtual_mem_query_page_size_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemQueryPageSize" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemMapOnEnter(
    ze_virtual_mem_map_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemMap" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemMapOnExit(
    ze_virtual_mem_map_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemMap" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemUnmapOnEnter(
    ze_virtual_mem_unmap_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemUnmap" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemUnmapOnExit(
    ze_virtual_mem_unmap_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemUnmap" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemSetAccessAttributeOnEnter(
    ze_virtual_mem_set_access_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemSetAccessAttribute" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemSetAccessAttributeOnExit(
    ze_virtual_mem_set_access_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemSetAccessAttribute" ;
  TAU_L0_exit_event( func_name );
}

static void zeVirtualMemGetAccessAttributeOnEnter(
    ze_virtual_mem_get_access_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeVirtualMemGetAccessAttribute" ;
  TAU_L0_enter_event( func_name );
}

static void zeVirtualMemGetAccessAttributeOnExit(
    ze_virtual_mem_get_access_attribute_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeVirtualMemGetAccessAttribute" ;
  TAU_L0_exit_event( func_name );
}

static void zeInitDriversOnEnter(
    ze_init_drivers_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeInitDrivers" ;
  TAU_L0_enter_event( func_name );
}

static void zeInitDriversOnExit(
    ze_init_drivers_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeInitDrivers" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASBuilderCreateExpOnEnter(
    ze_rtas_builder_create_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASBuilderCreateExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASBuilderCreateExpOnExit(
    ze_rtas_builder_create_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASBuilderCreateExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASBuilderGetBuildPropertiesExpOnEnter(
    ze_rtas_builder_get_build_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASBuilderGetBuildPropertiesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASBuilderGetBuildPropertiesExpOnExit(
    ze_rtas_builder_get_build_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASBuilderGetBuildPropertiesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASBuilderBuildExpOnEnter(
    ze_rtas_builder_build_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASBuilderBuildExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASBuilderBuildExpOnExit(
    ze_rtas_builder_build_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASBuilderBuildExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASBuilderDestroyExpOnEnter(
    ze_rtas_builder_destroy_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASBuilderDestroyExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASBuilderDestroyExpOnExit(
    ze_rtas_builder_destroy_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASBuilderDestroyExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASParallelOperationCreateExpOnEnter(
    ze_rtas_parallel_operation_create_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASParallelOperationCreateExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASParallelOperationCreateExpOnExit(
    ze_rtas_parallel_operation_create_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASParallelOperationCreateExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASParallelOperationGetPropertiesExpOnEnter(
    ze_rtas_parallel_operation_get_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASParallelOperationGetPropertiesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASParallelOperationGetPropertiesExpOnExit(
    ze_rtas_parallel_operation_get_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASParallelOperationGetPropertiesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASParallelOperationJoinExpOnEnter(
    ze_rtas_parallel_operation_join_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASParallelOperationJoinExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASParallelOperationJoinExpOnExit(
    ze_rtas_parallel_operation_join_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASParallelOperationJoinExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeRTASParallelOperationDestroyExpOnEnter(
    ze_rtas_parallel_operation_destroy_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeRTASParallelOperationDestroyExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeRTASParallelOperationDestroyExpOnExit(
    ze_rtas_parallel_operation_destroy_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeRTASParallelOperationDestroyExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetExtensionFunctionAddressOnEnter(
    ze_driver_get_extension_function_address_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGetExtensionFunctionAddress" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetExtensionFunctionAddressOnExit(
    ze_driver_get_extension_function_address_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGetExtensionFunctionAddress" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverGetLastErrorDescriptionOnEnter(
    ze_driver_get_last_error_description_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverGetLastErrorDescription" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverGetLastErrorDescriptionOnExit(
    ze_driver_get_last_error_description_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverGetLastErrorDescription" ;
  TAU_L0_exit_event( func_name );
}

static void zeDriverRTASFormatCompatibilityCheckExpOnEnter(
    ze_driver_rtas_format_compatibility_check_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDriverRTASFormatCompatibilityCheckExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeDriverRTASFormatCompatibilityCheckExpOnExit(
    ze_driver_rtas_format_compatibility_check_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDriverRTASFormatCompatibilityCheckExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetGlobalTimestampsOnEnter(
    ze_device_get_global_timestamps_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetGlobalTimestamps" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetGlobalTimestampsOnExit(
    ze_device_get_global_timestamps_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetGlobalTimestamps" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceImportExternalSemaphoreExtOnEnter(
    ze_device_import_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceImportExternalSemaphoreExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceImportExternalSemaphoreExtOnExit(
    ze_device_import_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceImportExternalSemaphoreExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceReleaseExternalSemaphoreExtOnEnter(
    ze_device_release_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceReleaseExternalSemaphoreExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceReleaseExternalSemaphoreExtOnExit(
    ze_device_release_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceReleaseExternalSemaphoreExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceReserveCacheExtOnEnter(
    ze_device_reserve_cache_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceReserveCacheExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceReserveCacheExtOnExit(
    ze_device_reserve_cache_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceReserveCacheExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceSetCacheAdviceExtOnEnter(
    ze_device_set_cache_advice_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceSetCacheAdviceExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceSetCacheAdviceExtOnExit(
    ze_device_set_cache_advice_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceSetCacheAdviceExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeDevicePciGetPropertiesExtOnEnter(
    ze_device_pci_get_properties_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDevicePciGetPropertiesExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeDevicePciGetPropertiesExtOnExit(
    ze_device_pci_get_properties_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDevicePciGetPropertiesExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetFabricVertexExpOnEnter(
    ze_device_get_fabric_vertex_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetFabricVertexExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetFabricVertexExpOnExit(
    ze_device_get_fabric_vertex_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetFabricVertexExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeDeviceGetRootDeviceOnEnter(
    ze_device_get_root_device_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeDeviceGetRootDevice" ;
  TAU_L0_enter_event( func_name );
}

static void zeDeviceGetRootDeviceOnExit(
    ze_device_get_root_device_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeDeviceGetRootDevice" ;
  TAU_L0_exit_event( func_name );
}

static void zeContextCreateExOnEnter(
    ze_context_create_ex_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeContextCreateEx" ;
  TAU_L0_enter_event( func_name );
}

static void zeContextCreateExOnExit(
    ze_context_create_ex_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeContextCreateEx" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandQueueGetOrdinalOnEnter(
    ze_command_queue_get_ordinal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandQueueGetOrdinal" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandQueueGetOrdinalOnExit(
    ze_command_queue_get_ordinal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandQueueGetOrdinal" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandQueueGetIndexOnEnter(
    ze_command_queue_get_index_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandQueueGetIndex" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandQueueGetIndexOnExit(
    ze_command_queue_get_index_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandQueueGetIndex" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListGetNextCommandIdWithKernelsExpOnEnter(
    ze_command_list_get_next_command_id_with_kernels_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListGetNextCommandIdWithKernelsExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListGetNextCommandIdWithKernelsExpOnExit(
    ze_command_list_get_next_command_id_with_kernels_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListGetNextCommandIdWithKernelsExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListUpdateMutableCommandKernelsExpOnEnter(
    ze_command_list_update_mutable_command_kernels_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListUpdateMutableCommandKernelsExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListUpdateMutableCommandKernelsExpOnExit(
    ze_command_list_update_mutable_command_kernels_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListUpdateMutableCommandKernelsExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendSignalExternalSemaphoreExtOnEnter(
    ze_command_list_append_signal_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendSignalExternalSemaphoreExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendSignalExternalSemaphoreExtOnExit(
    ze_command_list_append_signal_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendSignalExternalSemaphoreExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendWaitExternalSemaphoreExtOnEnter(
    ze_command_list_append_wait_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendWaitExternalSemaphoreExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendWaitExternalSemaphoreExtOnExit(
    ze_command_list_append_wait_external_semaphore_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendWaitExternalSemaphoreExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendImageCopyToMemoryExtOnEnter(
    ze_command_list_append_image_copy_to_memory_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendImageCopyToMemoryExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendImageCopyToMemoryExtOnExit(
    ze_command_list_append_image_copy_to_memory_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendImageCopyToMemoryExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListAppendImageCopyFromMemoryExtOnEnter(
    ze_command_list_append_image_copy_from_memory_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListAppendImageCopyFromMemoryExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListAppendImageCopyFromMemoryExtOnExit(
    ze_command_list_append_image_copy_from_memory_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListAppendImageCopyFromMemoryExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListHostSynchronizeOnEnter(
    ze_command_list_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListHostSynchronize" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListHostSynchronizeOnExit(
    ze_command_list_host_synchronize_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListHostSynchronize(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListHostSynchronize" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListCreateCloneExpOnEnter(
    ze_command_list_create_clone_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListCreateCloneExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListCreateCloneExpOnExit(
    ze_command_list_create_clone_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListCreateCloneExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListGetDeviceHandleOnEnter(
    ze_command_list_get_device_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListGetDeviceHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListGetDeviceHandleOnExit(
    ze_command_list_get_device_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListGetDeviceHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListGetContextHandleOnEnter(
    ze_command_list_get_context_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListGetContextHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListGetContextHandleOnExit(
    ze_command_list_get_context_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListGetContextHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListGetOrdinalOnEnter(
    ze_command_list_get_ordinal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListGetOrdinal" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListGetOrdinalOnExit(
    ze_command_list_get_ordinal_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListGetOrdinal" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListImmediateGetIndexOnEnter(
    ze_command_list_immediate_get_index_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListImmediateGetIndex" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListImmediateGetIndexOnExit(
    ze_command_list_immediate_get_index_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListImmediateGetIndex" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListIsImmediateOnEnter(
    ze_command_list_is_immediate_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListIsImmediate" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListIsImmediateOnExit(
    ze_command_list_is_immediate_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListIsImmediate" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListImmediateAppendCommandListsExpOnEnter(
    ze_command_list_immediate_append_command_lists_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);

  //Inserted call
    OnEnterCommandListImmediateAppendCommandListsExp(params, global_user_data, instance_user_data);



  const char* func_name = "zeCommandListImmediateAppendCommandListsExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListImmediateAppendCommandListsExpOnExit(
    ze_command_list_immediate_append_command_lists_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);
  std::vector<uint64_t> kids;
  //Inserted call
    OnExitCommandListImmediateAppendCommandListsExp(params, result, global_user_data, instance_user_data, &kids);



  const char* func_name = "zeCommandListImmediateAppendCommandListsExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListGetNextCommandIdExpOnEnter(
    ze_command_list_get_next_command_id_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListGetNextCommandIdExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListGetNextCommandIdExpOnExit(
    ze_command_list_get_next_command_id_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListGetNextCommandIdExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListUpdateMutableCommandsExpOnEnter(
    ze_command_list_update_mutable_commands_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListUpdateMutableCommandsExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListUpdateMutableCommandsExpOnExit(
    ze_command_list_update_mutable_commands_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListUpdateMutableCommandsExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListUpdateMutableCommandSignalEventExpOnEnter(
    ze_command_list_update_mutable_command_signal_event_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListUpdateMutableCommandSignalEventExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListUpdateMutableCommandSignalEventExpOnExit(
    ze_command_list_update_mutable_command_signal_event_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListUpdateMutableCommandSignalEventExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeCommandListUpdateMutableCommandWaitEventsExpOnEnter(
    ze_command_list_update_mutable_command_wait_events_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeCommandListUpdateMutableCommandWaitEventsExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeCommandListUpdateMutableCommandWaitEventsExpOnExit(
    ze_command_list_update_mutable_command_wait_events_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeCommandListUpdateMutableCommandWaitEventsExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventQueryTimestampsExpOnEnter(
    ze_event_query_timestamps_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventQueryTimestampsExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventQueryTimestampsExpOnExit(
    ze_event_query_timestamps_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventQueryTimestampsExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventQueryKernelTimestampsExtOnEnter(
    ze_event_query_kernel_timestamps_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventQueryKernelTimestampsExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventQueryKernelTimestampsExtOnExit(
    ze_event_query_kernel_timestamps_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventQueryKernelTimestampsExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventGetEventPoolOnEnter(
    ze_event_get_event_pool_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventGetEventPool" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventGetEventPoolOnExit(
    ze_event_get_event_pool_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventGetEventPool" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventGetSignalScopeOnEnter(
    ze_event_get_signal_scope_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventGetSignalScope" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventGetSignalScopeOnExit(
    ze_event_get_signal_scope_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventGetSignalScope" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventGetWaitScopeOnEnter(
    ze_event_get_wait_scope_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventGetWaitScope" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventGetWaitScopeOnExit(
    ze_event_get_wait_scope_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventGetWaitScope" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolPutIpcHandleOnEnter(
    ze_event_pool_put_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolPutIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolPutIpcHandleOnExit(
    ze_event_pool_put_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolPutIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolGetContextHandleOnEnter(
    ze_event_pool_get_context_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolGetContextHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolGetContextHandleOnExit(
    ze_event_pool_get_context_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolGetContextHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeEventPoolGetFlagsOnEnter(
    ze_event_pool_get_flags_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeEventPoolGetFlags" ;
  TAU_L0_enter_event( func_name );
}

static void zeEventPoolGetFlagsOnExit(
    ze_event_pool_get_flags_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeEventPoolGetFlags" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageGetMemoryPropertiesExpOnEnter(
    ze_image_get_memory_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageGetMemoryPropertiesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageGetMemoryPropertiesExpOnExit(
    ze_image_get_memory_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeImageGetMemoryPropertiesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageViewCreateExpOnEnter(
    ze_image_view_create_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageViewCreateExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageViewCreateExpOnExit(
    ze_image_view_create_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeImageViewCreateExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageGetAllocPropertiesExtOnEnter(
    ze_image_get_alloc_properties_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageGetAllocPropertiesExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageGetAllocPropertiesExtOnExit(
    ze_image_get_alloc_properties_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeImageGetAllocPropertiesExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageViewCreateExtOnEnter(
    ze_image_view_create_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageViewCreateExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageViewCreateExtOnExit(
    ze_image_view_create_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeImageViewCreateExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeImageGetDeviceOffsetExpOnEnter(
    ze_image_get_device_offset_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeImageGetDeviceOffsetExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeImageGetDeviceOffsetExpOnExit(
    ze_image_get_device_offset_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeImageGetDeviceOffsetExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSetGlobalOffsetExpOnEnter(
    ze_kernel_set_global_offset_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSetGlobalOffsetExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSetGlobalOffsetExpOnExit(
    ze_kernel_set_global_offset_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSetGlobalOffsetExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelGetBinaryExpOnEnter(
    ze_kernel_get_binary_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelGetBinaryExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelGetBinaryExpOnExit(
    ze_kernel_get_binary_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelGetBinaryExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeKernelSchedulingHintExpOnEnter(
    ze_kernel_scheduling_hint_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeKernelSchedulingHintExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeKernelSchedulingHintExpOnExit(
    ze_kernel_scheduling_hint_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeKernelSchedulingHintExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemFreeExtOnEnter(
    ze_mem_free_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemFreeExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemFreeExtOnExit(
    ze_mem_free_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemFreeExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetIpcHandleFromFileDescriptorExpOnEnter(
    ze_mem_get_ipc_handle_from_file_descriptor_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetIpcHandleFromFileDescriptorExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetIpcHandleFromFileDescriptorExpOnExit(
    ze_mem_get_ipc_handle_from_file_descriptor_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetIpcHandleFromFileDescriptorExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetFileDescriptorFromIpcHandleExpOnEnter(
    ze_mem_get_file_descriptor_from_ipc_handle_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetFileDescriptorFromIpcHandleExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetFileDescriptorFromIpcHandleExpOnExit(
    ze_mem_get_file_descriptor_from_ipc_handle_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetFileDescriptorFromIpcHandleExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemPutIpcHandleOnEnter(
    ze_mem_put_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemPutIpcHandle" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemPutIpcHandleOnExit(
    ze_mem_put_ipc_handle_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemPutIpcHandle" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemSetAtomicAccessAttributeExpOnEnter(
    ze_mem_set_atomic_access_attribute_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemSetAtomicAccessAttributeExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemSetAtomicAccessAttributeExpOnExit(
    ze_mem_set_atomic_access_attribute_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemSetAtomicAccessAttributeExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetAtomicAccessAttributeExpOnEnter(
    ze_mem_get_atomic_access_attribute_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetAtomicAccessAttributeExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetAtomicAccessAttributeExpOnExit(
    ze_mem_get_atomic_access_attribute_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetAtomicAccessAttributeExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeMemGetPitchFor2dImageOnEnter(
    ze_mem_get_pitch_for2d_image_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeMemGetPitchFor2dImage" ;
  TAU_L0_enter_event( func_name );
}

static void zeMemGetPitchFor2dImageOnExit(
    ze_mem_get_pitch_for2d_image_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeMemGetPitchFor2dImage" ;
  TAU_L0_exit_event( func_name );
}

static void zeModuleInspectLinkageExtOnEnter(
    ze_module_inspect_linkage_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeModuleInspectLinkageExt" ;
  TAU_L0_enter_event( func_name );
}

static void zeModuleInspectLinkageExtOnExit(
    ze_module_inspect_linkage_ext_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeModuleInspectLinkageExt" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricEdgeGetExpOnEnter(
    ze_fabric_edge_get_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricEdgeGetExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricEdgeGetExpOnExit(
    ze_fabric_edge_get_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricEdgeGetExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricEdgeGetVerticesExpOnEnter(
    ze_fabric_edge_get_vertices_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricEdgeGetVerticesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricEdgeGetVerticesExpOnExit(
    ze_fabric_edge_get_vertices_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricEdgeGetVerticesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricEdgeGetPropertiesExpOnEnter(
    ze_fabric_edge_get_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricEdgeGetPropertiesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricEdgeGetPropertiesExpOnExit(
    ze_fabric_edge_get_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricEdgeGetPropertiesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricVertexGetExpOnEnter(
    ze_fabric_vertex_get_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricVertexGetExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricVertexGetExpOnExit(
    ze_fabric_vertex_get_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricVertexGetExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricVertexGetSubVerticesExpOnEnter(
    ze_fabric_vertex_get_sub_vertices_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricVertexGetSubVerticesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricVertexGetSubVerticesExpOnExit(
    ze_fabric_vertex_get_sub_vertices_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricVertexGetSubVerticesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricVertexGetPropertiesExpOnEnter(
    ze_fabric_vertex_get_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricVertexGetPropertiesExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricVertexGetPropertiesExpOnExit(
    ze_fabric_vertex_get_properties_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricVertexGetPropertiesExp" ;
  TAU_L0_exit_event( func_name );
}

static void zeFabricVertexGetDeviceExpOnEnter(
    ze_fabric_vertex_get_device_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);



  const char* func_name = "zeFabricVertexGetDeviceExp" ;
  TAU_L0_enter_event( func_name );
}

static void zeFabricVertexGetDeviceExpOnExit(
    ze_fabric_vertex_get_device_exp_params_t* params,
    ze_result_t result,
    void* global_user_data,
    void** instance_user_data) 
{
  ZeCollector* collector =
    reinterpret_cast<ZeCollector*>(global_user_data);


  const char* func_name = "zeFabricVertexGetDeviceExp" ;
  TAU_L0_exit_event( func_name );
}

void EnableTracing(zel_tracer_handle_t tracer) {
  ze_result_t status = ZE_RESULT_SUCCESS;
    if (ZeLoader::get().zelTracerKernelSetIndirectAccessRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSetIndirectAccessRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSetIndirectAccessOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSetIndirectAccessRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSetIndirectAccessOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendImageCopyFromMemoryExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyFromMemoryExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendImageCopyFromMemoryExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyFromMemoryExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendImageCopyFromMemoryExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextSystemBarrierRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextSystemBarrierRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextSystemBarrierOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextSystemBarrierRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextSystemBarrierOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventHostSignalRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventHostSignalRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventHostSignalOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventHostSignalRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventHostSignalOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelGetSourceAttributesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelGetSourceAttributesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelGetSourceAttributesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelGetSourceAttributesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelGetSourceAttributesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemAllocHostRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemAllocHostRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemAllocHostOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemAllocHostRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemAllocHostOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendSignalExternalSemaphoreExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendSignalExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendSignalExternalSemaphoreExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendSignalExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendSignalExternalSemaphoreExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASBuilderDestroyExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASBuilderDestroyExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASBuilderDestroyExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASBuilderDestroyExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASBuilderDestroyExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASBuilderBuildExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASBuilderBuildExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASBuilderBuildExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASBuilderBuildExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASBuilderBuildExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetIpcHandleFromFileDescriptorExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetIpcHandleFromFileDescriptorExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetIpcHandleFromFileDescriptorExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetIpcHandleFromFileDescriptorExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetIpcHandleFromFileDescriptorExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextEvictMemoryRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextEvictMemoryRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextEvictMemoryOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextEvictMemoryRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextEvictMemoryOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSetGlobalOffsetExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSetGlobalOffsetExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSetGlobalOffsetExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSetGlobalOffsetExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSetGlobalOffsetExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleGetFunctionPointerRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleGetFunctionPointerRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleGetFunctionPointerOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleGetFunctionPointerRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleGetFunctionPointerOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemAllocDeviceRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemAllocDeviceRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemAllocDeviceOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemAllocDeviceRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemAllocDeviceOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemCloseIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemCloseIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemCloseIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemCloseIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemCloseIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleGetGlobalPointerRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleGetGlobalPointerRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleGetGlobalPointerOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleGetGlobalPointerRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleGetGlobalPointerOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolPutIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolPutIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolPutIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolPutIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolPutIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageViewCreateExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageViewCreateExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageViewCreateExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageViewCreateExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageViewCreateExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendWriteGlobalTimestampRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendWriteGlobalTimestampRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendWriteGlobalTimestampOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendWriteGlobalTimestampRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendWriteGlobalTimestampOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextCreateExRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextCreateExRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextCreateExOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextCreateExRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextCreateExOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetExtensionFunctionAddressRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetExtensionFunctionAddressRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetExtensionFunctionAddressOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetExtensionFunctionAddressRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetExtensionFunctionAddressOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceCanAccessPeerRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceCanAccessPeerRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceCanAccessPeerOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceCanAccessPeerRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceCanAccessPeerOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelGetBinaryExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelGetBinaryExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelGetBinaryExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelGetBinaryExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelGetBinaryExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventQueryTimestampsExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventQueryTimestampsExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventQueryTimestampsExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventQueryTimestampsExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventQueryTimestampsExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendLaunchMultipleKernelsIndirectRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendLaunchMultipleKernelsIndirectRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendLaunchMultipleKernelsIndirectOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendLaunchMultipleKernelsIndirectRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendLaunchMultipleKernelsIndirectOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricVertexGetPropertiesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricVertexGetPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricVertexGetPropertiesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricVertexGetPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricVertexGetPropertiesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageGetPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageGetPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageGetPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetFileDescriptorFromIpcHandleExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetFileDescriptorFromIpcHandleExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetFileDescriptorFromIpcHandleExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetFileDescriptorFromIpcHandleExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetFileDescriptorFromIpcHandleExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleDynamicLinkRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleDynamicLinkRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleDynamicLinkOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleDynamicLinkRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleDynamicLinkOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleGetNativeBinaryRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleGetNativeBinaryRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleGetNativeBinaryOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleGetNativeBinaryRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleGetNativeBinaryOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemQueryPageSizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemQueryPageSizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemQueryPageSizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemQueryPageSizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemQueryPageSizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleGetPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleGetPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleGetPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemUnmapRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemUnmapRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemUnmapOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemUnmapRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemUnmapOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandQueueGetIndexRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandQueueGetIndexRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandQueueGetIndexOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandQueueGetIndexRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandQueueGetIndexOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListUpdateMutableCommandsExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandsExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListUpdateMutableCommandsExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandsExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListUpdateMutableCommandsExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListIsImmediateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListIsImmediateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListIsImmediateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListIsImmediateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListIsImmediateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListCreateCloneExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListCreateCloneExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListCreateCloneExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListCreateCloneExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListCreateCloneExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetPitchFor2dImageRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetPitchFor2dImageRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetPitchFor2dImageOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetPitchFor2dImageRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetPitchFor2dImageOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleGetKernelNamesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleGetKernelNamesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleGetKernelNamesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleGetKernelNamesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleGetKernelNamesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageGetDeviceOffsetExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageGetDeviceOffsetExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageGetDeviceOffsetExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageGetDeviceOffsetExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageGetDeviceOffsetExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFenceQueryStatusRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFenceQueryStatusRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFenceQueryStatusOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFenceQueryStatusRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFenceQueryStatusOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolOpenIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolOpenIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolOpenIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolOpenIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolOpenIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetImagePropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetImagePropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetImagePropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetImagePropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetImagePropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListUpdateMutableCommandWaitEventsExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandWaitEventsExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListUpdateMutableCommandWaitEventsExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandWaitEventsExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListUpdateMutableCommandWaitEventsExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextGetStatusRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextGetStatusRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextGetStatusOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextGetStatusRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextGetStatusOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetGlobalTimestampsRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetGlobalTimestampsRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetGlobalTimestampsOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetGlobalTimestampsRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetGlobalTimestampsOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetP2PPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetP2PPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetP2PPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetP2PPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetP2PPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventGetSignalScopeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventGetSignalScopeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventGetSignalScopeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventGetSignalScopeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventGetSignalScopeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolGetContextHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolGetContextHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolGetContextHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolGetContextHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolGetContextHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricVertexGetDeviceExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricVertexGetDeviceExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricVertexGetDeviceExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricVertexGetDeviceExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricVertexGetDeviceExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASBuilderCreateExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASBuilderCreateExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASBuilderCreateExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASBuilderCreateExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASBuilderCreateExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListGetDeviceHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListGetDeviceHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListGetDeviceHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListGetDeviceHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListGetDeviceHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageViewCreateExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageViewCreateExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageViewCreateExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageViewCreateExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageViewCreateExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSchedulingHintExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSchedulingHintExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSchedulingHintExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSchedulingHintExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSchedulingHintExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricVertexGetExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricVertexGetExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricVertexGetExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricVertexGetExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricVertexGetExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricEdgeGetPropertiesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricEdgeGetPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricEdgeGetPropertiesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricEdgeGetPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricEdgeGetPropertiesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelGetNameRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelGetNameRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelGetNameOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelGetNameRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelGetNameOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASParallelOperationDestroyExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASParallelOperationDestroyExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASParallelOperationDestroyExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASParallelOperationDestroyExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASParallelOperationDestroyExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceReleaseExternalSemaphoreExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceReleaseExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceReleaseExternalSemaphoreExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceReleaseExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceReleaseExternalSemaphoreExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemPutIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemPutIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemPutIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemPutIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemPutIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendImageCopyToMemoryExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyToMemoryExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendImageCopyToMemoryExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyToMemoryExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendImageCopyToMemoryExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetMemoryAccessPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetMemoryAccessPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetMemoryAccessPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetMemoryAccessPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetMemoryAccessPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSetCacheConfigRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSetCacheConfigRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSetCacheConfigOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSetCacheConfigRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSetCacheConfigOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetAtomicAccessAttributeExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetAtomicAccessAttributeExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetAtomicAccessAttributeExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetAtomicAccessAttributeExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetAtomicAccessAttributeExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelGetPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelGetPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelGetPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextMakeImageResidentRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextMakeImageResidentRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextMakeImageResidentOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextMakeImageResidentRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextMakeImageResidentOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextMakeMemoryResidentRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextMakeMemoryResidentRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextMakeMemoryResidentOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextMakeMemoryResidentRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextMakeMemoryResidentOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemFreeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemFreeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemFreeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemFreeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemFreeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextEvictImageRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextEvictImageRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextEvictImageOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextEvictImageRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextEvictImageOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetCachePropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetCachePropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetCachePropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetCachePropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetCachePropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDevicePciGetPropertiesExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDevicePciGetPropertiesExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDevicePciGetPropertiesExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDevicePciGetPropertiesExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDevicePciGetPropertiesExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageGetMemoryPropertiesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageGetMemoryPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageGetMemoryPropertiesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageGetMemoryPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageGetMemoryPropertiesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerSamplerCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerSamplerCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeSamplerCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerSamplerCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeSamplerCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListGetNextCommandIdExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListGetNextCommandIdExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListGetNextCommandIdExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListGetNextCommandIdExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListGetNextCommandIdExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendQueryKernelTimestampsRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendQueryKernelTimestampsRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendQueryKernelTimestampsOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendQueryKernelTimestampsRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendQueryKernelTimestampsOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerSamplerDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerSamplerDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeSamplerDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerSamplerDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeSamplerDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventQueryKernelTimestampRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventQueryKernelTimestampRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventQueryKernelTimestampOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventQueryKernelTimestampRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventQueryKernelTimestampOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemFreeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemFreeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemFreeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemFreeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemFreeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListGetOrdinalRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListGetOrdinalRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListGetOrdinalOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListGetOrdinalRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListGetOrdinalOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolGetIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolGetIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolGetIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolGetIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolGetIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandQueueGetOrdinalRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandQueueGetOrdinalRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandQueueGetOrdinalOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandQueueGetOrdinalRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandQueueGetOrdinalOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerPhysicalMemDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerPhysicalMemDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zePhysicalMemDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerPhysicalMemDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zePhysicalMemDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageGetAllocPropertiesExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageGetAllocPropertiesExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageGetAllocPropertiesExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageGetAllocPropertiesExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageGetAllocPropertiesExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetExtensionPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetExtensionPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetExtensionPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetExtensionPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetExtensionPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetSubDevicesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetSubDevicesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetSubDevicesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetSubDevicesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetSubDevicesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListGetNextCommandIdWithKernelsExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListGetNextCommandIdWithKernelsExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListGetNextCommandIdWithKernelsExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListGetNextCommandIdWithKernelsExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListGetNextCommandIdWithKernelsExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricEdgeGetVerticesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricEdgeGetVerticesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricEdgeGetVerticesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricEdgeGetVerticesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricEdgeGetVerticesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolCloseIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolCloseIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolCloseIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolCloseIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolCloseIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemSetAccessAttributeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemSetAccessAttributeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemSetAccessAttributeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemSetAccessAttributeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemSetAccessAttributeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceImportExternalSemaphoreExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceImportExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceImportExternalSemaphoreExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceImportExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceImportExternalSemaphoreExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASParallelOperationGetPropertiesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASParallelOperationGetPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASParallelOperationGetPropertiesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASParallelOperationGetPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASParallelOperationGetPropertiesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolGetFlagsRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolGetFlagsRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolGetFlagsOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolGetFlagsRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolGetFlagsOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricVertexGetSubVerticesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricVertexGetSubVerticesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricVertexGetSubVerticesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricVertexGetSubVerticesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricVertexGetSubVerticesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemAllocSharedRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemAllocSharedRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemAllocSharedOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemAllocSharedRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemAllocSharedOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelGetIndirectAccessRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelGetIndirectAccessRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelGetIndirectAccessOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelGetIndirectAccessRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelGetIndirectAccessOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListUpdateMutableCommandKernelsExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandKernelsExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListUpdateMutableCommandKernelsExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandKernelsExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListUpdateMutableCommandKernelsExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASParallelOperationJoinExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASParallelOperationJoinExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASParallelOperationJoinExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASParallelOperationJoinExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASParallelOperationJoinExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventGetEventPoolRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventGetEventPoolRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventGetEventPoolOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventGetEventPoolRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventGetEventPoolOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSetArgumentValueRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSetArgumentValueRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSetArgumentValueOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSetArgumentValueRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSetArgumentValueOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemMapRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemMapRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemMapOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemMapRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemMapOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventGetWaitScopeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventGetWaitScopeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventGetWaitScopeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventGetWaitScopeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventGetWaitScopeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemOpenIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemOpenIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemOpenIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemOpenIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemOpenIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceReserveCacheExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceReserveCacheExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceReserveCacheExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceReserveCacheExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceReserveCacheExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListImmediateGetIndexRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListImmediateGetIndexRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListImmediateGetIndexOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListImmediateGetIndexRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListImmediateGetIndexOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetApiVersionRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetApiVersionRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetApiVersionOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetApiVersionRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetApiVersionOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleBuildLogGetStringRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleBuildLogGetStringRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleBuildLogGetStringOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleBuildLogGetStringRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleBuildLogGetStringOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSuggestMaxCooperativeGroupCountRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSuggestMaxCooperativeGroupCountRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSuggestMaxCooperativeGroupCountOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSuggestMaxCooperativeGroupCountRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSuggestMaxCooperativeGroupCountOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetExternalMemoryPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetExternalMemoryPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetExternalMemoryPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetExternalMemoryPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetExternalMemoryPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetRootDeviceRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetRootDeviceRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetRootDeviceOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetRootDeviceRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetRootDeviceOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemAdviseRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemAdviseRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemAdviseOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemAdviseRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemAdviseOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendSignalEventRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendSignalEventRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendSignalEventOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendSignalEventRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendSignalEventOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleBuildLogDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleBuildLogDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleBuildLogDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleBuildLogDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleBuildLogDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerPhysicalMemCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerPhysicalMemCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zePhysicalMemCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerPhysicalMemCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zePhysicalMemCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceSetCacheAdviceExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceSetCacheAdviceExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceSetCacheAdviceExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceSetCacheAdviceExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceSetCacheAdviceExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerInitRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerInitRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeInitOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerInitRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeInitOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetIpcPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetIpcPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetIpcPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetIpcPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetIpcPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetCommandQueueGroupPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetCommandQueueGroupPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetCommandQueueGroupPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetCommandQueueGroupPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetCommandQueueGroupPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemReserveRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemReserveRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemReserveOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemReserveRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemReserveOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFenceDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFenceDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFenceDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFenceDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFenceDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASParallelOperationCreateExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASParallelOperationCreateExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASParallelOperationCreateExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASParallelOperationCreateExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASParallelOperationCreateExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetMemoryPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetMemoryPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetMemoryPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetMemoryPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetMemoryPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerVirtualMemGetAccessAttributeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerVirtualMemGetAccessAttributeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeVirtualMemGetAccessAttributeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerVirtualMemGetAccessAttributeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeVirtualMemGetAccessAttributeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetAddressRangeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetAddressRangeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetAddressRangeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetAddressRangeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetAddressRangeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFenceResetRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFenceResetRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFenceResetOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFenceResetRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFenceResetOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerInitDriversRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerInitDriversRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeInitDriversOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerInitDriversRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeInitDriversOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerRTASBuilderGetBuildPropertiesExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerRTASBuilderGetBuildPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeRTASBuilderGetBuildPropertiesExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerRTASBuilderGetBuildPropertiesExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeRTASBuilderGetBuildPropertiesExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetComputePropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetComputePropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetComputePropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetComputePropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetComputePropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetStatusRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetStatusRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetStatusOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetStatusRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetStatusOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventQueryKernelTimestampsExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventQueryKernelTimestampsExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventQueryKernelTimestampsExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventQueryKernelTimestampsExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventQueryKernelTimestampsExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListGetContextHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListGetContextHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListGetContextHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListGetContextHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListGetContextHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemSetAtomicAccessAttributeExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemSetAtomicAccessAttributeExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemSetAtomicAccessAttributeExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemSetAtomicAccessAttributeExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemSetAtomicAccessAttributeExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemoryPrefetchRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemoryPrefetchRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemoryPrefetchOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemoryPrefetchRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemoryPrefetchOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetAllocPropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetAllocPropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetAllocPropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetAllocPropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetAllocPropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemFreeExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemFreeExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemFreeExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemFreeExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemFreeExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetFabricVertexExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetFabricVertexExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetFabricVertexExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetFabricVertexExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetFabricVertexExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleInspectLinkageExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleInspectLinkageExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleInspectLinkageExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleInspectLinkageExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleInspectLinkageExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFenceCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFenceCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFenceCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFenceCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFenceCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendWaitOnEventsRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendWaitOnEventsRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendWaitOnEventsOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendWaitOnEventsRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendWaitOnEventsOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerMemGetIpcHandleRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerMemGetIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeMemGetIpcHandleOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerMemGetIpcHandleRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeMemGetIpcHandleOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSuggestGroupSizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSuggestGroupSizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSuggestGroupSizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSuggestGroupSizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSuggestGroupSizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendWaitExternalSemaphoreExtRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendWaitExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendWaitExternalSemaphoreExtOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendWaitExternalSemaphoreExtRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendWaitExternalSemaphoreExtOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverRTASFormatCompatibilityCheckExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverRTASFormatCompatibilityCheckExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverRTASFormatCompatibilityCheckExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverRTASFormatCompatibilityCheckExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverRTASFormatCompatibilityCheckExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDeviceGetModulePropertiesRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDeviceGetModulePropertiesRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDeviceGetModulePropertiesOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDeviceGetModulePropertiesRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDeviceGetModulePropertiesOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerDriverGetLastErrorDescriptionRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerDriverGetLastErrorDescriptionRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeDriverGetLastErrorDescriptionOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerDriverGetLastErrorDescriptionRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeDriverGetLastErrorDescriptionOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListUpdateMutableCommandSignalEventExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandSignalEventExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListUpdateMutableCommandSignalEventExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListUpdateMutableCommandSignalEventExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListUpdateMutableCommandSignalEventExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFabricEdgeGetExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFabricEdgeGetExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFabricEdgeGetExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFabricEdgeGetExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFabricEdgeGetExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventHostResetRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventHostResetRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventHostResetOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventHostResetRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventHostResetOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventPoolCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventPoolCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventPoolCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventPoolCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventPoolCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendEventResetRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendEventResetRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendEventResetOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendEventResetRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendEventResetOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendLaunchKernelRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendLaunchKernelRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendLaunchKernelOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendLaunchKernelRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendLaunchKernelOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendLaunchCooperativeKernelRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendLaunchCooperativeKernelRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendLaunchCooperativeKernelOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendLaunchCooperativeKernelRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendLaunchCooperativeKernelOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendLaunchKernelIndirectRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendLaunchKernelIndirectRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendLaunchKernelIndirectOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendLaunchKernelIndirectRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendLaunchKernelIndirectOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemoryCopyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemoryCopyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemoryCopyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemoryCopyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemoryCopyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemoryFillRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemoryFillRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemoryFillOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemoryFillRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemoryFillOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendBarrierRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendBarrierRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendBarrierOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendBarrierRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendBarrierOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemoryRangesBarrierRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemoryRangesBarrierRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemoryRangesBarrierOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemoryRangesBarrierRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemoryRangesBarrierOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemoryCopyRegionRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemoryCopyRegionRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemoryCopyRegionOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemoryCopyRegionRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemoryCopyRegionOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendMemoryCopyFromContextRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendMemoryCopyFromContextRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendMemoryCopyFromContextOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendMemoryCopyFromContextRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendMemoryCopyFromContextOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendImageCopyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendImageCopyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendImageCopyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendImageCopyRegionRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyRegionRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendImageCopyRegionOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyRegionRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendImageCopyRegionOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendImageCopyToMemoryRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyToMemoryRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendImageCopyToMemoryOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyToMemoryRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendImageCopyToMemoryOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListAppendImageCopyFromMemoryRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyFromMemoryRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListAppendImageCopyFromMemoryOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListAppendImageCopyFromMemoryRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListAppendImageCopyFromMemoryOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandQueueExecuteCommandListsRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandQueueExecuteCommandListsRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandQueueExecuteCommandListsOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandQueueExecuteCommandListsRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandQueueExecuteCommandListsOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListCloseRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListCloseRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListCloseOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListCloseRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListCloseOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListCreateImmediateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListCreateImmediateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListCreateImmediateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListCreateImmediateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListCreateImmediateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListResetRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListResetRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListResetOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListResetRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListResetOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandQueueCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandQueueCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandQueueCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandQueueCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandQueueCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandQueueSynchronizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandQueueSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandQueueSynchronizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandQueueSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandQueueSynchronizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandQueueDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandQueueDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandQueueDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandQueueDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandQueueDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerImageDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerImageDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeImageDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerImageDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeImageDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerModuleDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerModuleDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeModuleDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerModuleDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeModuleDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelCreateRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelCreateRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelCreateOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelCreateRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelCreateOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelSetGroupSizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelSetGroupSizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelSetGroupSizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelSetGroupSizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelSetGroupSizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerKernelDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerKernelDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeKernelDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerKernelDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeKernelDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventHostSynchronizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventHostSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventHostSynchronizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventHostSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventHostSynchronizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListHostSynchronizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListHostSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListHostSynchronizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListHostSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListHostSynchronizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerEventQueryStatusRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerEventQueryStatusRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeEventQueryStatusOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerEventQueryStatusRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeEventQueryStatusOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerFenceHostSynchronizeRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerFenceHostSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeFenceHostSynchronizeOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerFenceHostSynchronizeRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeFenceHostSynchronizeOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerContextDestroyRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerContextDestroyRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeContextDestroyOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerContextDestroyRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeContextDestroyOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }
    if (ZeLoader::get().zelTracerCommandListImmediateAppendCommandListsExpRegisterCallback_ != nullptr) {
      status = ZeLoader::get().zelTracerCommandListImmediateAppendCommandListsExpRegisterCallback_(tracer, ZEL_REGISTER_PROLOGUE, zeCommandListImmediateAppendCommandListsExpOnEnter);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
      status = ZeLoader::get().zelTracerCommandListImmediateAppendCommandListsExpRegisterCallback_(tracer, ZEL_REGISTER_EPILOGUE, zeCommandListImmediateAppendCommandListsExpOnExit);
      PTI_ASSERT(status == ZE_RESULT_SUCCESS);
    }

  status = ZE_FUNC(zelTracerSetEnabled)(tracer, true);
  PTI_ASSERT(status == ZE_RESULT_SUCCESS);
}