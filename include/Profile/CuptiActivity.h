#include <Profile/TauGpu.h>
#include <Profile/CuptiLayer.h>
#include <cuda.h>
#include <cupti.h>
#include <math.h>
#include <iostream>
#include <limits.h>

#if CUPTI_API_VERSION >= 2

#ifdef TAU_BFD
#define HAVE_DECL_BASENAME 1
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include <demangle.h>
#  endif /* HAVE_GNU_DEMANGLE */
#  include <bfd.h>
#endif /* TAU_BFD */

#define CUDA_CHECK_ERROR(err, str) \
	if (err != CUDA_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

#define CUPTI_CHECK_ERROR(err, str) \
	if (err != CUPTI_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

#define ACTIVITY_BUFFER_SIZE (4096 * 1024)

/* Some API calls deprecated in 5.5
 */
#if CUDA_VERSION >= 5050

#define runtimeCorrelationId correlationId
#define CUpti_ActivityKernel CUpti_ActivityKernel2

#endif

extern "C" void Tau_cupti_set_offset(
            uint64_t timestamp
            );

extern "C" void Tau_set_context_event_name(void *ue, const char *name);
extern "C" void Tau_write_user_event_as_metric(void *ue);
extern "C" void * Tau_return_context_userevent(const char *name);

extern "C" void metric_set_gpu_timestamp(int tid, int idx, double value);

extern "C" void Tau_cupti_find_context_event(
						TauContextUserEvent** u, 
						const char *name,
            bool context);

extern "C" void Tau_cupti_register_metadata(
						uint32_t deviceId,
						GpuMetadata *metadata,
						int metadata_size);

extern "C" void Tau_cupti_register_host_calling_site(
						uint32_t correlationId);

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
						int memcpy_type);

extern "C" void Tau_cupti_exit_memcpy_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						int bytes_copied,
						int memcpy_type);

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
            int direction);

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
						double stop);

extern "C" void Tau_cupti_register_gpu_atomic_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes);

extern "C" x_uint64 TauTraceGetTimeStamp();

uint8_t *activityBuffer;
CUpti_SubscriberHandle subscriber;

int number_of_streams;
std::vector<int> streamIds;

std::vector<TauContextUserEvent *> counterEvents;

void Tau_cupti_register_sync_event(CUcontext c, uint32_t stream);

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params);

void Tau_cupti_record_activity(CUpti_Activity *record);

void __attribute__ ((constructor)) Tau_cupti_onload(void);

void Tau_cupti_subscribe(void);

void __attribute__ ((destructor)) Tau_cupti_onunload(void);

void get_values_from_memcpy(const CUpti_CallbackData *info, CUpti_CallbackId id, CUpti_CallbackDomain domain, int &kind, int &count);

int getMemcpyType(int kind);
const char* demangleName(const char *n);

int getParentFunction(uint32_t id);

bool function_is_sync(CUpti_CallbackId id);
bool function_is_memcpy(CUpti_CallbackId id, CUpti_CallbackDomain domain);
bool function_is_launch(CUpti_CallbackId id);
bool function_is_exit(CUpti_CallbackId id);

bool registered_sync = false;

bool cupti_api_runtime();
bool cupti_api_driver();

typedef std::map<TauContextUserEvent *, TAU_EVENT_DATATYPE> eventMap_t;
eventMap_t eventMap; 

int gpu_occupancy_available(int deviceId);

void record_gpu_occupancy(int32_t blockX, 
                          int32_t blockY,
                          int32_t blockZ,
			                    uint16_t registersPerThread,
		                      int32_t staticSharedMemory,
                          uint32_t deviceId,
                          const char *name, 
                          eventMap_t *map);

void record_gpu_launch(int cId);
void record_gpu_counters(int device_id, const char *name, uint32_t id, eventMap_t *m);

int get_device_count();

#if CUPTI_API_VERSION >= 3
void form_context_event_name(CUpti_ActivityKernel *kernel, CUpti_ActivitySourceLocator *source, const char *event, std::string *name);

std::map<uint32_t, CUpti_ActivitySourceLocator> sourceLocatorMap;
#endif // CUPTI_API_VERSION >= 3

std::map<uint32_t, CUpti_ActivityDevice> deviceMap;
//std::map<uint32_t, CUpti_ActivityGlobalAccess> globalAccessMap;
std::map<uint32_t, CUpti_ActivityKernel> kernelMap;

#define TAU_MAX_GPU_DEVICES 16


/* CUPTI API callbacks are called from CUPTI's signal handlers and thus cannot
 * allocate/deallocate memory. So all the counters values need to be allocated
 * on the Stack. */

uint64_t counters_at_last_launch[TAU_MAX_GPU_DEVICES][TAU_MAX_COUNTERS] = {ULONG_MAX};
uint64_t current_counters[TAU_MAX_GPU_DEVICES][TAU_MAX_COUNTERS] = {0};

int kernels_encountered[TAU_MAX_GPU_DEVICES] = {0};
int kernels_recorded[TAU_MAX_GPU_DEVICES] = {0};

bool counters_averaged_warning_issued[TAU_MAX_GPU_DEVICES] = {false};
bool counters_bounded_warning_issued[TAU_MAX_GPU_DEVICES] = {false};

void record_gpu_counters_at_launch(int device)
{ 
  kernels_encountered[device]++;
  if (Tau_CuptiLayer_get_num_events() > 0 &&
      !counters_averaged_warning_issued[device] && 
      kernels_encountered[device] > 1) {
    TAU_VERBOSE("TAU Warning: CUPTI events will be avereged, multiple kernel deteched between synchronization points.\n");
    counters_averaged_warning_issued[device] = true;
    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++) {
      Tau_CuptiLayer_set_event_name(n, TAU_CUPTI_COUNTER_AVERAGED); 
    }
  }
  int n_counters = Tau_CuptiLayer_get_num_events();
  if (n_counters > 0 && counters_at_last_launch[device][0] == ULONG_MAX) {
    Tau_CuptiLayer_read_counters(device, counters_at_last_launch[device]);
  }
#ifdef TAU_CUPTI_DEBUG_COUNTERS
  std::cout << "at launch ====> " << std::endl;
  std::cout << "\tlast launch:      " << counters_at_last_launch[device][0] << std::endl;
  std::cout << "\tcurrent counters: " << current_counters[device][0] << std::endl;
#endif
}
  
void record_gpu_counters_at_sync(int device)
{
  if (kernels_encountered[device] == 0) {
   return;
  }
  Tau_CuptiLayer_read_counters(device, current_counters[device]);
#ifdef TAU_CUPTI_DEBUG_COUNTERS
  std::cout << "at sync   ====> " << std::endl;
  std::cout << "\tlast launch:      " << counters_at_last_launch[device][0] << std::endl;
  std::cout << "\tcurrent counters: " << current_counters[device][0] << std::endl;
#endif
}

void clear_counters(int device)
{
  for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++)
  {
    counters_at_last_launch[device][n] = ULONG_MAX;
  }
  kernels_encountered[device] = 0;
  kernels_recorded[device] = 0;

}

const char *last_recorded_kernel_name;

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
	strcpy(stored_name_##n, str_##n.str().c_str()); \
	metadata[id].name = "GPU " SX(n); \
	metadata[id].value = stored_name_##n; \
	id++
#endif

	//Tau_metadata("GPU " SX(name), str_##name.str().c_str()); 

