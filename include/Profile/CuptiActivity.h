#ifndef CUPTI_ACTIVITY_H
#define CUPTI_ACTIVITY_H

#include <Profile/TauGpu.h>
#include <Profile/CudaSass.h>
#include <cuda.h>
#include <cupti.h>
#include <math.h>
#include <iostream>
#include <limits.h>

//#define TAU_CUPTI_DEBUG_COUNTERS
//#define TAU_DEBUG_CUPTI

#if CUPTI_API_VERSION >= 2

#define CUDA_CHECK_ERROR(err, str) \
	if (err != CUDA_SUCCESS) \
  { \
		const char* err_name; \
		const char* err_str; \
		cuGetErrorName(err, &err_name); \
		cuGetErrorString(err, &err_str); \
		fprintf(stderr, "CUDA driver error %s: %s\n", err_name, err_str); \
		fprintf(stderr, str); \
	} \

#define CUPTI_CHECK_ERROR(err, str) \
	if (err != CUPTI_SUCCESS) \
  { \
	    const char * errStr; \
		cuptiGetResultString(err, &errStr); \
		fprintf(stderr, "TAU: CUPTI error in %s: %s\n", str, errStr); \
	} \

#define ACTIVITY_BUFFER_SIZE (4096 * 1024)
#define ACTIVITY_ENTRY_LIMIT 1024
// #define ACTIVITY_BUFFER_SIZE (8192*1024)
/* Some API calls deprecated in 5.5
 */

#if CUDA_VERSION >= 9000

#define CUpti_ActivityKernel CUpti_ActivityKernel4
#define CUpti_ActivityNvLink CUpti_ActivityNvLink2
// >= 8000 carryover
#define CUpti_ActivityPCSampling CUpti_ActivityPCSampling2
// >= 7000 carryover
#define CUpti_ActivityDevice CUpti_ActivityDevice2
#define CUpti_ActivityUnifiedMemoryCounter CUpti_ActivityUnifiedMemoryCounter2
// >= 6050 carryover
#define CUpti_ActivityBranch CUpti_ActivityBranch2
#define CUpti_ActivityGlobalAccess CUpti_ActivityGlobalAccess2
// >= 5050 carryover
#define runtimeCorrelationId correlationId

#elif CUDA_VERSION >= 8000

#define CUpti_ActivityPCSampling CUpti_ActivityPCSampling2
// >= 7000 carryover
#define CUpti_ActivityKernel CUpti_ActivityKernel3
#define CUpti_ActivityDevice CUpti_ActivityDevice2
#define CUpti_ActivityUnifiedMemoryCounter CUpti_ActivityUnifiedMemoryCounter2
// >= 6050 carryover
#define CUpti_ActivityBranch CUpti_ActivityBranch2
#define CUpti_ActivityGlobalAccess CUpti_ActivityGlobalAccess2
// >= 5050 carryover
#define runtimeCorrelationId correlationId

#elif CUDA_VERSION >= 7000

#define CUpti_ActivityKernel CUpti_ActivityKernel3
#define CUpti_ActivityDevice CUpti_ActivityDevice2
#define CUpti_ActivityUnifiedMemoryCounter CUpti_ActivityUnifiedMemoryCounter2
// >= 6050 carryover
#define CUpti_ActivityBranch CUpti_ActivityBranch2
#define CUpti_ActivityGlobalAccess CUpti_ActivityGlobalAccess2
// >= 5050 carryover
#define runtimeCorrelationId correlationId

#elif CUDA_VERSION >= 6050

#define CUpti_ActivityBranch CUpti_ActivityBranch2
#define CUpti_ActivityGlobalAccess CUpti_ActivityGlobalAccess2
// >= 5050 carryover
#define CUpti_ActivityKernel CUpti_ActivityKernel2
#define runtimeCorrelationId correlationId

#elif CUDA_VERSION >= 5050 && CUDA_VERSION <= 6050

#define CUpti_ActivityKernel CUpti_ActivityKernel2
#define runtimeCorrelationId correlationId

#endif


#if CUPTI_API_VERSION >= 4
#define TAU_ASYNC_ACTIVITY_API
#endif

class CudaEnvironment
{
 public:
  std::vector<double> timestamp;
  uint32_t deviceId;
  std::vector<uint32_t> smClock;
  std::vector<uint32_t> memoryClock;
  std::vector<uint32_t> gpuTemperature;
  std::vector<uint32_t> power;
  uint32_t powerLimit;
  std::vector<uint32_t> fanSpeed;
};

extern "C" void Tau_cupti_set_offset(double cpu_gpu_offset);

// unified memory
extern "C" void Tau_cupti_configure_unified_memory(void);

extern "C" void Tau_set_context_event_name(void *ue, const char *name);
extern "C" void Tau_write_user_event_as_metric(void *ue);
extern "C" void * Tau_return_context_userevent(const char *name);

extern "C" void metric_set_gpu_timestamp(int tid, double value);

extern "C" void Tau_cupti_find_context_event(
						TauContextUserEvent** u,
						const char *name,
            bool context);

extern "C" void Tau_cupti_register_metadata(
						uint32_t deviceId,
						GpuMetadata *metadata,
						int metadata_size);

extern "C" void Tau_cupti_register_host_calling_site(
						uint32_t correlationId,
						const char *name);

extern "C" void Tau_cupti_register_device_calling_site(
						int64_t correlationId,
						const char *name);

extern "C" void Tau_cupti_enter_memcpy_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						int bytes_copied,
						int memcpy_type,
						int taskId);

extern "C" void Tau_cupti_exit_memcpy_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						int bytes_copied,
						int memcpy_type,
						int taskId);

extern "C" void Tau_cupti_register_memcpy_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						double start,
						double stop,
						int bytes_copied,
						int memcpy_type,
						int direction,
						int taskId);

extern "C" void Tau_cupti_register_unifmem_event(
						 const char *name,
						 uint32_t deviceId,
						 uint32_t streamId,
						 uint32_t processId,
						 uint64_t start,
						 uint64_t end,
						 uint64_t value,
						 int unifmem_type,
						 int direction,
						 int taskId);

extern "C" void Tau_cupti_register_gpu_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						int64_t parentGridId,
						bool cdp,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes,
						double start,
						double stop,
						int taskId);

extern "C" void Tau_cupti_register_gpu_sync_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						double start,
						double stop,
						int taskId);

extern "C" void Tau_cupti_register_gpu_atomic_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes,
						int taskId);

extern "C" void Tau_cupti_gpu_enter_event_from_cpu(const char* name, int tid);
extern "C" void Tau_cupti_gpu_exit_event_from_cpu(const char* name, int tid);

/* extern "C" void Tau_cupti_register_func_event( */
/*                                               const char *name, */
/*                                               uint32_t deviceId, */
/*                                               uint32_t streamId, */
/*                                               uint32_t contextId, */
/*                                               uint32_t functionIndex, */
/*                                               double timestamp, */
/*                                               uint32_t id, */
/*                                               uint32_t moduleId, */
/* 					      const char *kname, */
/*                                               const char *demangled); */

/* extern "C" void Tau_cupti_register_instruction_event( */
/* 						     const char *name, */
/* 						     uint32_t deviceId, */
/* 						     uint32_t streamId, */
/* 						     uint32_t contextId, */
/* 						     uint32_t correlationId, */
/* 						     double start, */
/* 						     double stop, double delta_tstamp, */
/* 						     uint32_t sourceLocatorId, */
/* 						     uint32_t functionId, */
/* 						     uint32_t pcOffset, */
/* 						     uint32_t executed, */
/* 						     uint32_t threadsExecuted); */

/* extern "C" void Tau_cupti_register_source_event( */
/*                                                 const char *name, */
/*                                                 uint32_t deviceId, */
/*                                                 uint32_t streamId, */
/*                                                 uint32_t contextId, */
/*                                                 uint32_t sourceId, */
/*                                                 double timestamp, */
/*                                                 const char *fileName, */
/*                                                 uint32_t lineNumber); */

extern "C" x_uint64 TauTraceGetTimeStamp();

extern "C" void handle_nvtx_callback(CUpti_CallbackId id, const void *cbdata);


void Tau_cupti_process_buffer(CUcontext c, uint32_t stream, uint8_t* buffer, size_t size, size_t validSize);

void Tau_cupti_activity_flush_all();

void Tau_cupti_register_buffer_creation(uint8_t** buffer, size_t* size, size_t* maxNumRecords);

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params);

void Tau_cupti_record_activity(CUpti_Activity *record);

void __attribute__ ((constructor)) Tau_cupti_onload(void);

void Tau_cupti_subscribe(void);

CUpti_EventGroup* Tau_cupti_get_eventgroup(void);

void Tau_cupti_init(void);

void __attribute__ ((destructor)) Tau_cupti_onunload(void);

void get_values_from_memcpy(const CUpti_CallbackData *info, CUpti_CallbackId id, CUpti_CallbackDomain domain, int &kind, int &count);

int getMemcpyType(int kind);
int getUnifmemType(int kind);

int getParentFunction(uint32_t id);

bool function_is_sync(CUpti_CallbackId id);
bool function_is_memcpy(CUpti_CallbackId id, CUpti_CallbackDomain domain);
bool function_is_launch(CUpti_CallbackId id);
bool function_is_exit(CUpti_CallbackId id);

bool cupti_api_runtime();
bool cupti_api_driver();

typedef std::map<TauContextUserEvent *, TAU_EVENT_DATATYPE> eventMap_t;
static std::set<uint32_t> set_gpuThread;

int gpu_occupancy_available(int deviceId);

void record_gpu_occupancy(int32_t blockX,
                          int32_t blockY,
                          int32_t blockZ,
			  uint16_t registersPerThread,
			  int32_t staticSharedMemory,
                          uint32_t deviceId,
                          const char *name,
                          eventMap_t *map);

void record_gpu_launch(int cId, const char *name);
void record_gpu_counters(int device_id, const char *name, uint32_t id, eventMap_t *m);
int record_imix_counters(const char* name, uint32_t taskId, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end);
void transport_imix_counters(uint32_t vec, Instrmix imixT, const char* name, uint32_t taskId, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end, TauContextUserEvent * tc);
ImixStats write_runtime_imix(uint32_t corrId, uint32_t taskId, std::map<std::pair<int, int>, CudaOps> map_disassem, std::string kernel);
void write_sass_counters();
void write_sass_output();
void record_environment_counters(const char *name, uint32_t taskId, uint32_t device, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end);
void transport_environment_counters(std::vector<uint32_t> vec, EnvType envT, const char* name, uint32_t taskId, uint32_t deviceId, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end, TauContextUserEvent* tc);

int get_device_count();
int get_device_id();

#if CUPTI_API_VERSION >= 3
void form_context_event_name(CUpti_ActivityKernel *kernel, CUpti_ActivitySourceLocator *source, const char *event, std::string *name);


#endif // CUPTI_API_VERSION >= 3

#ifndef TAU_MAX_GPU_DEVICES
#define TAU_MAX_GPU_DEVICES 16
#endif


#if CUDA_VERSION >= 6000
static const char * getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind);
static const char * getUvmCounterScopeString(CUpti_ActivityUnifiedMemoryCounterScope scope);
static const char * getComputeApiKindString(CUpti_ActivityComputeApiKind kind);
#endif

typedef struct cupti_eventData_st {
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
} cupti_eventData;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  cupti_eventData *eventData;
  uint64_t eventVal;
} RuntimeApiTrace_t;

typedef std::map<uint32_t, CUpti_ActivityDevice> device_map_t;
//static std::map<uint32_t, CUpti_ActivityDevice> deviceMap;

void record_gpu_counters_at_launch(int device, int task);
void record_gpu_counters_at_sync();
void record_gpu_counters_at_sync(int device, int task);

void clear_counters(int device);

int get_task_from_id(int id, int task);

#define CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(name, id, info, kind, count) \
	if ((id) == CUPTI_RUNTIME_TRACE_CBID_##name##_v3020) \
	{ \
		kind = ((name##_v3020_params *) info->functionParams)->kind; \
		count = ((name##_v3020_params *) info->functionParams)->count; \
	}

#define S(x) #x
#define SX(x) S(x)
#define RECORD_DEVICE_METADATA(n, device) \
  std::ostringstream str_##n; \
	str_##n << device->n; \
	int string_length_##n = strlen(str_##n.str().c_str()) + 1; \
	char *stored_name_##n = (char*) malloc(sizeof(char)*string_length_##n); \
	strncpy(stored_name_##n,  str_##n.str().c_str(), sizeof(char)*string_length_##n);  \
	metadata[id].name = (char*)("GPU " SX(n)); \
	metadata[id].value = stored_name_##n; \
	id++
#endif

	//Tau_metadata("GPU " SX(name), str_##name.str().c_str());

#endif //CUPTI_ACTIVITY_H
