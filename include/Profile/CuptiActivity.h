#include <Profile/TauGpu.h>
#include <Profile/CuptiLayer.h>
#include <cuda.h>
#include <cupti.h>
#include <math.h>
#include <iostream>

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

extern "C" void Tau_set_context_event_name(void *ue, char *name);
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

extern "C" void Tau_cupti_register_calling_site(
						uint32_t correlationId,
						FunctionInfo *current_function);

extern "C" void Tau_cupti_register_sync_site(
						uint32_t correlationId, 
            uint64_t *counters,
            int number_of_counters);

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
						int memcpy_type);

extern "C" void Tau_cupti_register_gpu_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes,
						double *start,
						double *stop,
            int number_of_metrics);

extern "C" void Tau_cupti_register_gpu_atomic_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes);

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
void record_gpu_occupancy(CUpti_ActivityKernel *k, const char *name, eventMap_t *m);
void record_gpu_launch(int cId, FunctionInfo *f);
void record_gpu_counters(int device_id, const char *name, uint32_t id, double *s_metrics, double* e_metrics);

#if CUPTI_API_VERSION >= 3
void form_context_event_name(CUpti_ActivityKernel *kernel, CUpti_ActivitySourceLocator *source, const char *event, std::string *name);

std::map<uint32_t, CUpti_ActivitySourceLocator> sourceLocatorMap;
#endif // CUPTI_API_VERSION >= 3

std::map<uint32_t, CUpti_ActivityDevice> deviceMap;
//std::map<uint32_t, CUpti_ActivityGlobalAccess> globalAccessMap;
std::map<uint32_t, CUpti_ActivityKernel> kernelMap;


struct GpuState {

  int kernels_encountered;
  int kernels_recorded;
  //structure: counter 1,2,3...
  uint64_t *counters_at_last_launch;
  uint64_t *current_counters;
  static int device_count;
  static int n_counters;
  bool counters_averaged_warning_issued;
  bool counters_bounded_warning_issued;

public:
  int device_num;
  GpuState() {
    n_counters = Tau_CuptiLayer_get_num_events();
    cuDeviceGetCount(&device_count);
    cuCtxGetDevice(&device_num);
    kernels_encountered = 0;
    kernels_recorded = 0;
    counters_at_last_launch = (uint64_t *) calloc(n_counters, sizeof(uint64_t));
    current_counters = (uint64_t *) calloc(n_counters, sizeof(uint64_t));
    counters_averaged_warning_issued = false;
    counters_bounded_warning_issued = false;
    clear();
  }
  GpuState(int n) { 
    n_counters = Tau_CuptiLayer_get_num_events();
    cuDeviceGetCount(&device_count);
    device_num = n;
    kernels_encountered = 0;
    kernels_recorded = 0;
    counters_at_last_launch = (uint64_t *) calloc(n_counters, sizeof(uint64_t));
    current_counters = (uint64_t *) calloc(n_counters, sizeof(uint64_t));
    counters_averaged_warning_issued = false;
    counters_bounded_warning_issued = false;
    clear();
  }
  uint64_t *start_counters()
  {
    return counters_at_last_launch;
  }

  uint64_t *end_counters()
  {
    return current_counters;
  }

  void clear() {
    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++)
    {
      counters_at_last_launch[n] = -1;
      kernels_encountered = 0;
      kernels_recorded = 0;
    }
  }

  void record_gpu_counters_at_launch()
  {
    kernels_encountered++;
    //std::cout << "kernel encountered." << std::endl;
    if (Tau_CuptiLayer_get_num_events() > 0 &&
        !counters_averaged_warning_issued && 
        kernels_encountered > 1) {
      TAU_VERBOSE("Warning: CUPTI events will be avereged, multiple kernel deteched between synchronization points.\n");
      counters_averaged_warning_issued = true;
    }
    n_counters = Tau_CuptiLayer_get_num_events();
    if (n_counters > 0 && counters_at_last_launch[0] == -1) {
    //kernelInfoMap[correlationId].counters = (uint64_t **) malloc(n_counters*device_count*sizeof(uint64_t));
    //for (int i=0; i<device_count; i++)
    //{
      Tau_CuptiLayer_read_counters(device_num, counters_at_last_launch);
      //printf("[at launch] device 0, counter 0: %llu.\n", counters_at_last_launch[0]);

    }
  }
  void record_gpu_counters_at_sync()
  {
    //std::cout << "recording counters at sync." << std::endl;
    if (kernels_encountered == 0) {
     return;
    }
    /*if (counters_at_last_launch[0] == 0) {
      return;
    }*/
    Tau_CuptiLayer_read_counters(device_num, current_counters);

    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++)
    {
      //printf("counter %d: start: %llu end: %llu diff: %llu num: %d.\n", n, counters_at_last_launch[n], current_counters[n], current_counters[n] - counters_at_last_launch[n], kernels_encountered);
      //current_counters[n] = (current_counters[n] - counters_at_last_launch[n]) / kernels_encountered; 
      current_counters[n] = current_counters[n] / kernels_encountered; 
    }
  }
  //take the end counts to get a difference.
  /*
  void difference(K o)
  {
    //print difference
    for (int d = 0; d < device_count; d++)
    {
      uint64_t *o_counters;
      o_counters = o.counters(d);
      uint64_t *my_counters;
      my_counters = counters(d);
      for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++)
      {
        printf("counter %d: start: %llu end: %llu diff: %llu.\n", n, my_counters[n], o_counters[n], o_counters[n] - my_counters[n]);
        my_counters[n] = o_counters[n] - my_counters[n]; 
      }
    }    
    */
};
int GpuState::device_count = 0;
int GpuState::n_counters = 0;

//std::vector<int> CurrentState::kernels_encountered;
std::map<uint32_t, GpuState> CurrentGpuState;

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

