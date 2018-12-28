#include <Profile/CuptiActivity.h>
#include <iostream>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
using namespace std;

#if CUPTI_API_VERSION >= 2
#include <dlfcn.h>

static int subscribed = 0;
static unsigned int parent_tid = 0;
static uint64_t timestamp;

// From CuptiActivity.h
uint8_t *activityBuffer;
CUpti_SubscriberHandle subscriber;

int number_of_streams[TAU_MAX_THREADS] = {0};
std::vector<int> streamIds[TAU_MAX_THREADS];

std::vector<TauContextUserEvent *> counterEvents[TAU_MAX_THREADS];

bool registered_sync = false;
eventMap_t eventMap[TAU_MAX_THREADS]; 
#if CUPTI_API_VERSION >= 3
std::map<uint32_t, CUpti_ActivitySourceLocator> sourceLocatorMap;
static std::map<uint32_t, CUpti_ActivitySourceLocator> srcLocMap;
#endif // CUPTI_API_VERSION >= 3

device_map_t & __deviceMap()
{
  static device_map_t deviceMap;
  return deviceMap;
}
std::map<uint32_t, CUpti_ActivityKernel> kernelMap[TAU_MAX_THREADS];

static std::map<uint32_t, CUpti_ActivityFunction> functionMap;
static std::map<uint32_t, std::list<CUpti_ActivityInstructionExecution> > instructionMap; // indexing by functionId 
static std::map<std::pair<int, int>, CudaOps> map_disassem;
static std::map<std::string, ImixStats> map_imix_static;

// sass output
FILE *fp_source[TAU_MAX_THREADS];
FILE *fp_instr[TAU_MAX_THREADS];
FILE *fp_func[TAU_MAX_THREADS];
FILE *cubin;


static int device_count_total = 1;
static double recentTimestamp = 0;

static uint32_t buffers_queued = 0;

/* CUPTI API callbacks are called from CUPTI's signal handlers and thus cannot
 * allocate/deallocate memory. So all the counters values need to be allocated
 * on the Stack. */

uint64_t counters_at_last_launch[TAU_MAX_THREADS][TAU_MAX_COUNTERS] = {ULONG_MAX};
uint64_t current_counters[TAU_MAX_THREADS][TAU_MAX_COUNTERS] = {0};

int kernels_encountered[TAU_MAX_THREADS] = {0};
int kernels_recorded[TAU_MAX_THREADS] = {0};

bool counters_averaged_warning_issued[TAU_MAX_THREADS] = {false};
bool counters_bounded_warning_issued[TAU_MAX_THREADS] = {false};
const char *last_recorded_kernel_name;

//#define TAU_DEBUG_CUPTI 1
//#define TAU_DEBUG_CUPTI_SASS 1
//#define TAU_DEBUG_SASS_PROF 1
//#define TAU_DEBUG_CUPTI_COUNTERS 1
//#define TAU_CUPTI_DEBUG_COUNTERS 1

/* BEGIN: unified memory */
#define CUPTI_CALL(call)                                                    \
do {                                                                        \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(-1);                                                             \
    }                                                                       \
} while (0)
/* END: Unified Memory */

// // CUDA Thread
// map<uint32_t, CudaThread> & CudaThreadMap()
// {
//   static map<uint32_t, CudaThread> map_cudaThread;
//   return map_cudaThread;
// }

/* BEGIN:  Dump cubin (sass) */
// static std::map<std::string, ImixStats> map_imixStats;

#if CUDA_VERSION >= 5500
void CUPTIAPI dumpCudaModule(CUpti_CallbackId cbid, void *resourceDescriptor)
{

  if(TauEnv_get_cuda_track_sass()) {
    const char *pCubin;
    size_t cubinSize;
    std::string border = "======================================================================";
    // dump the cubin at MODULE_LOADED_STARTING
    CUpti_ModuleResourceData *moduleResourceData = (CUpti_ModuleResourceData *)resourceDescriptor; 
    // #endif
    // assume cubin will always be dumped, check if OpenACC

    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      //#if DUMP_CUBIN
      // if(TauEnv_get_cuda_track_sass()){
      // You can use nvdisasm to dump the SASS from the cubin. 
      // Try nvdisasm -b -fun <function_id> sass_to_source.cubin

      pCubin = moduleResourceData->pCubin;
      cubinSize = moduleResourceData->cubinSize;
      int i = get_device_id();
      // BEGIN: CUBIN Dump
      char str_source[500];
      char str_int[5];
      strcpy (str_source,TauEnv_get_profiledir());
      strcat (str_source,"/");
      strcat (str_source,"sass_source_map_loaded_");
      sprintf (str_int, "%d", (get_device_id() + 1));
      strcat (str_source, str_int);
      strcat (str_source, ".cubin");

      cubin = fopen(str_source, "wb");
      
      if (cubin == NULL) {
	printf("sass_source_map.cubin failed\n");
      }
      
      fwrite(pCubin, sizeof(uint8_t), cubinSize, cubin);
      fclose(cubin);
      // END:  CUBIN Dump
            
#ifdef TAU_DEBUG_CUPTI_SASS
      cout << "get_device_id(): " << get_device_id() << endl;
#endif
      map_disassem = parse_cubin(str_source, get_device_id());
      map_imix_static = print_instruction_mixes();

    }
    // else if (cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    //   // You can dump the cubin either at MODULE_LOADED or MODULE_UNLOAD_STARTING

    //   pCubin = moduleResourceData->pCubin;
    //   cubinSize = moduleResourceData->cubinSize;

    //   char str_source[500];
    //   strcpy (str_source,TauEnv_get_profiledir());
    //   strcat (str_source,"/");
    //   strcat (str_source,"sass_source_map_unload_start.cubin");
      
    //   cubin = fopen(str_source, "wb");
      
    //   if (cubin == NULL) {
    //   	printf("sass_source_map.cubin failed\n");
    //   }
      
    //   fwrite(pCubin, sizeof(uint8_t), cubinSize, cubin);
    //   fclose(cubin);

    // }
  }
}

static void
handleResource(CUpti_CallbackId cbid, const CUpti_ResourceData *resourceData)
{

  if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    dumpCudaModule(cbid, resourceData->resourceDescriptor);
  }
  // else if (cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
  //   dumpCudaModule(cbid, resourceData->resourceDescriptor);
  // }
  
}
#endif
/* END:  Dump cubin (sass) */

CUresult cuInit(unsigned int a1)
{
#ifdef TAU_DEBUG_CUPTI
    printf("in cuInit\n");
#endif
    if (parent_tid == 0) {
      parent_tid = pthread_self();
      // parent_tid = RtsLayer::getTid();
      //printf("[CuptiActivity]:  Set parent_tid as: %u\n", parent_tid);
    }
    typedef CUresult (*cuInit_p_h)(unsigned int);
    static void *libcuda_handle = (void *)dlopen("libcuda.so", RTLD_NOW);
    if (!libcuda_handle) {
        perror("Error opening libcuda.so in dlopen call");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    static cuInit_p_h cuInit_h = (cuInit_p_h)dlsym(libcuda_handle, "cuInit");
    if (!cuInit_h) {
        perror("Error obtaining cuInit symbol info from dlopen'ed lib");
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    Tau_cupti_subscribe();
    return cuInit_h(a1);
}

void Tau_cupti_subscribe()
{
	if(subscribed) return;
#ifdef TAU_DEBUG_CUPTI
    printf("in Tau_cupti_subscribe\n");
#endif
	CUptiResult err = CUPTI_SUCCESS;
	CUresult err2 = CUDA_SUCCESS;

	TAU_VERBOSE("TAU: Subscribing to CUPTI.\n");
	err = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)Tau_cupti_callback_dispatch, NULL);
    CUPTI_CHECK_ERROR(err, "cuptiSubscribe");
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE);
    CUPTI_CHECK_ERROR(err, "cuptiActivityEnable");
	
	//setup global activity queue.
    size_t size;
    size_t maxRecords;

  // With the ASYNC ACTIVITY API CUPTI will call 
  // Tau_cupti_register_buffer_creation() when it needs a new activity buffer
  // and Tau_cupti_register_sync_event() when a buffer is completed so all we
  // need to do here is to register these callback functions.
#ifdef TAU_ASYNC_ACTIVITY_API
    err = cuptiActivityRegisterCallbacks(Tau_cupti_register_buffer_creation, Tau_cupti_register_sync_event);
    CUPTI_CHECK_ERROR(err, "cuptiActivityRegisterCallbacks");
#else

    Tau_cupti_register_buffer_creation(&activityBuffer, &size, &maxRecords);
	err = cuptiActivityEnqueueBuffer(NULL, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);
	CUPTI_CHECK_ERROR(err, "cuptiActivityEnqueueBuffer");
#endif
	subscribed = 1;
}

void Tau_cupti_onload()
{
	if (!subscribed) {
		Tau_cupti_subscribe();
	}
	TAU_VERBOSE("TAU: Enabling CUPTI callbacks.\n");

	CUptiResult err = CUPTI_SUCCESS;
	CUresult err2 = CUDA_SUCCESS;
  
	if (cupti_api_runtime())
	{
#ifdef TAU_DEBUG_CUPTI
		printf("TAU: Subscribing to RUNTIME API.\n");
#endif
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
	    CUPTI_CHECK_ERROR(err, "cuptiEnableDomain (CUPTI_CB_DOMAIN_RUNTIME_API)");
		//runtime_enabled = true;
	}
	if (cupti_api_driver())
	{
#ifdef TAU_DEBUG_CUPTI
		printf("TAU: Subscribing to DRIVER API.\n");
#endif
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
	    CUPTI_CHECK_ERROR(err, "cuptiEnableDomain (CUPTI_CB_DOMAIN_DRIVER_API)");
		//driver_enabled = true;
	}
  
    	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE); 
    CUPTI_CHECK_ERROR(err, "cuptiEnableDomain (CUPTI_CB_DOMAIN_SYNCHRONIZE)");
    	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE); 
    CUPTI_CHECK_ERROR(err, "cuptiEnableDomain (CUPTI_CB_DOMAIN_RESOURCE)");	
    	CUDA_CHECK_ERROR(err2, "Cannot set Domain, check if the CUDA toolkit version is supported by the install CUDA driver.\n");
	/* BEGIN source line info */
	/* Need to check if device is pre-Fermi */
#if CUDA_VERSION >= 5500
  if(TauEnv_get_cuda_track_sass()) {
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION);
	CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION)");
	// err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE);
	// CUPTI_CHECK_ERROR(err, "cuptiEnableDomain (CUPTI_CB_DOMAIN_RESOURCE)");
  }
#endif
 	/* END source line info */
    // 	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT);
    // CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_CONTEXT)");
    // 	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    // CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_DRIVER)");
    // 	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    // CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_RUNTIME)");
if(!TauEnv_get_cuda_track_sass()) {
    	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_MEMCPY)");
 }	
#if CUDA_VERSION >= 5050
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2);
    CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_MEMCPY2)");
#endif
/*  SASS incompatible with KIND_CONCURRENT_KERNEL  */
if(!TauEnv_get_cuda_track_sass()) {
#if CUDA_VERSION >= 5000
    	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)");
#else
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_KERNEL)");
#endif
}
 else {
   err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
   CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_KERNEL)");
 }

#if CUPTI_API_VERSION >= 3
  if (strcasecmp(TauEnv_get_cuda_instructions(), "GLOBAL_ACCESS") == 0)
  {
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
      CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR)");
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS);
      CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS)");
  } else if (strcasecmp(TauEnv_get_cuda_instructions(), "BRANCH") == 0)
  {
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
      CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR)");
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_BRANCH);
      CUPTI_CHECK_ERROR(err, "cuptiActivityEnable (CUPTI_ACTIVITY_KIND_BRANCH)");
  }
#else
  if (strcasecmp(TauEnv_get_cuda_instructions(), "GLOBAL_ACCESS") == 0 ||
      strcasecmp(TauEnv_get_cuda_instructions(), "BRANCH") == 0)
  {
		printf("TAU WARNING: DISABLING CUDA %s tracking. Please use CUDA 5.0 or greater.\n", TauEnv_get_cuda_instructions());
  }
#endif //CUPTI_API_VERSIOn >= 3

  //cout << "Tau_cupti_onload():  get_device_id(): " << get_device_id() << endl;

  CUpti_ActivityDevice device = __deviceMap()[get_device_id()];

	if ((device.computeCapabilityMajor > 3) ||
		device.computeCapabilityMajor == 3 &&
		device.computeCapabilityMinor >= 0)
	{
	  
  if(TauEnv_get_cuda_track_unified_memory()) {
#if CUDA_VERSION >= 7000
    CUptiResult res = CUPTI_SUCCESS;
	CUresult err2 = CUDA_SUCCESS;
    CUpti_ActivityUnifiedMemoryCounterConfig config[2];
    CUresult er;
    cuInit(0);

    // configure unified memory counters
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    res = cuptiActivityConfigureUnifiedMemoryCounter(config, 2);
    if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED) {
      printf("Test is waived, unified memory is not supported on the underlying platform.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
      printf("Test is waived, unified memory is not supported on the device.\n");
    }
    else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) {
      printf("Test is waived, unified memory is not supported on the non-P2P multi-gpu setup.\n");
    }
    else {
      CUPTI_CALL(res);
    }

    // enable unified memory counter activity
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

#elif CUDA_VERSION >= 6000 && CUDA_VERSION <= 6050
    CUptiResult res = CUPTI_SUCCESS;
	CUresult err2 = CUDA_SUCCESS;
    CUpti_ActivityUnifiedMemoryCounterConfig config[3];

    cuInit(0);

    // configure unified memory counters
    config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    config[0].deviceId = 0;
    config[0].enable = 1;

    config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    config[1].deviceId = 0;
    config[1].enable = 1;

    config[2].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    config[2].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT;
    config[2].deviceId = 0;
    config[2].enable = 1;

    res = cuptiActivityConfigureUnifiedMemoryCounter(config, 3);
    if (res == CUPTI_ERROR_NOT_SUPPORTED) {
      printf("Test is waived, unified memory is not supported on the underlying platform.\n");
    }
    else {
      CUPTI_CALL(res);
    }

    // enable unified memory counter activity
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

#else
    printf("Unified memory supported only in CUDA 6.0 and over.\n");
#endif

  }
	}
	else {
	  CUDA_CHECK_ERROR(err2, "CUDA Compute Capability 3.0 or higher required!\n");
	}  
  CUDA_CHECK_ERROR(err2, "Cannot enqueue buffer.\n");
  
  //uint64_t timestamp;
  err = cuptiGetTimestamp(&timestamp);
  CUDA_CHECK_ERROR(err2, "Cannot get timestamp.\n");
  Tau_cupti_set_offset(TauTraceGetTimeStamp() - ((double)timestamp / 1e3));
  //Tau_cupti_set_offset((-1) * timestamp / 1e3);
  //cerr << "begining timestamp: " << TauTraceGetTimeStamp() - ((double)timestamp/1e3) << "ms.\n" << endl;
  //Tau_cupti_set_offset(0);

  Tau_gpu_init();
}

void Tau_cupti_onunload() {
#if CUDA_VERSION >= 6000
  if(TauEnv_get_cuda_track_unified_memory()) {
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
  }
#endif

}

/* This callback handles synchronous things */

// Extra bool param that tells whether to run code
void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
#ifdef TAU_DEBUG_CUPTI
	printf("in Tau_cupti_callback_dispatch\n");
#endif
#if defined(PTHREADS)
	if (!TauEnv_get_tauCuptiAvail()) {
	  unsigned int cur_tid = pthread_self(); // needed for IBM P8
	  const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *) params;
	  unsigned int corrid = cbInfo->correlationId;
	  if (cur_tid != parent_tid) {
	    // Register GPU thread count here
	    unsigned int systid = cur_tid;
	    RtsLayer::LockEnv();
	    if (map_cudaThread.find(corrid) == map_cudaThread.end()) {
	      unsigned int parenttid = parent_tid;
	      int tauvtid = Tau_get_thread();
	      unsigned int contextid = cbInfo->contextUid;
	      const char* funcname = cbInfo->functionName;
	      register_cuda_thread(systid, parenttid, tauvtid, corrid, contextid, funcname);
	    }
	    // track unique threads seen
	    if (set_gpuThread.find(cur_tid) == set_gpuThread.end()) {
          // reserve a thread ID from TAU
	      int threadid = Tau_create_task();
          // Start a top level timer on that thread.
          Tau_create_top_level_timer_if_necessary_task(threadid);
          //printf("VIRTUAL THREAD: %d\n", threadid);
	      set_gpuThread.insert(cur_tid);
	      TauEnv_set_cudaTotalThreads(TauEnv_get_cudaTotalThreads() + 1);
	      map_cuptiThread[Tau_get_thread()] = threadid;
	    }
	    RtsLayer::UnLockEnv();
	  }
    }
#endif
	//Just in case we encounter a callback before TAU is intialized or finished.
  if (!Tau_init_check_initialized() || Tau_global_getLightsOut()) { 
#ifdef TAU_DEBUG_CUPTI
      printf("TAU: [WARNING] Got CUPTI callback but TAU is either not yet initialized or has finished!\n");
#endif
	  return;
  }
#if CUDA_VERSION >= 5500
	if (domain == CUPTI_CB_DOMAIN_RESOURCE)
	{
	  // if we want runtime cubin dump
	  if(TauEnv_get_cuda_track_sass()) {
	    handleResource(id, (CUpti_ResourceData *)params);
	  }
	}
#endif
#ifndef TAU_ASYNC_ACTIVITY_API
	if (domain == CUPTI_CB_DOMAIN_RESOURCE)
	{
		//A resource was created, let us enqueue a buffer in order to capture events
		//that happen on that resource.
		if (id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
		{
			CUptiResult err = CUPTI_SUCCESS;
	        CUresult err2 = CUDA_SUCCESS;
			CUpti_ResourceData* resource = (CUpti_ResourceData*) params;
#ifdef TAU_DEBUG_CUPTI
			printf("TAU: Resource created: Enqueuing Buffer with context=%p stream=%d.\n", resource->context, 0);
#endif
			activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);

			err = cuptiActivityEnqueueBuffer(resource->context, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);
			CUDA_CHECK_ERROR(err2, "Cannot enqueue buffer in context.\n");
		}
		else if (id == CUPTI_CBID_RESOURCE_STREAM_CREATED)
		{
			CUptiResult err = CUPTI_SUCCESS;
	        CUresult err2 = CUDA_SUCCESS;
			CUpti_ResourceData* resource = (CUpti_ResourceData*) params;
    		uint32_t stream;
			err = cuptiGetStreamId(resource->context, resource->resourceHandle.stream, &stream);
			CUPTI_CHECK_ERROR(err, "cuptiGetStreamId");
#ifdef TAU_DEBUG_CUPTI
			printf("TAU: Stream created: Enqueuing Buffer with context=%p stream=%d.\n", resource->context, stream);
#endif

			activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
			err = cuptiActivityEnqueueBuffer(resource->context, stream, activityBuffer, ACTIVITY_BUFFER_SIZE);
			CUDA_CHECK_ERROR(err2, "Cannot enqueue buffer in stream.\n");
			int taskId = 0;
#if defined(PTHREADS)
			if (map_cudaThread.find(corrid) != map_cudaThread.end()) {
			  int local_vtid = map_cudaThread[corrid].tau_vtid;
			  taskId = map_cuptiThread[local_vtid];
			}
#endif
			streamIds[taskId].push_back(stream);
			number_of_streams[taskId]++;
		}

	}
	else if (domain == CUPTI_CB_DOMAIN_SYNCHRONIZE)
	{
#ifdef TAU_DEBUG_CUPTI
		printf("register sync from callback.\n");
#endif
		CUpti_SynchronizeData *sync = (CUpti_SynchronizeData *) params;
		uint32_t stream;
		CUptiResult err = CUPTI_SUCCESS;
	    CUresult err2 = CUDA_SUCCESS;
		//Global Buffer
#if defined(PTHREADS)
    int count_iter = TauEnv_get_cudaTotalThreads();
#else
    int count_iter = get_device_count();
#endif
    for (int i=0; i<count_iter; i++) {
      record_gpu_counters_at_sync(i);
    }
		Tau_cupti_register_sync_event(NULL, 0, NULL, 0, 0);
    
		err = cuptiGetStreamId(sync->context, sync->stream, &stream);
		CUPTI_CHECK_ERROR(err, " cuptiGetStreamId");
		Tau_cupti_register_sync_event(sync->context, stream, NULL, 0, 0);
		int taskId = 0;
#if defined(PTHREADS)
		if (map_cudaThread.find(corrid) != map_cudaThread.end()) {
		  int local_vtid = map_cudaThread[corrid].tau_vtid;
		  taskId = map_cuptiThread[local_vtid];
		}
#endif
		for (int s=0; s<number_of_streams[taskId]; s++)
		{
			Tau_cupti_register_sync_event(sync->context, streamIds[taskId].at(s), NULL, 0, 0);
		}
	}
	else if (domain == CUPTI_CB_DOMAIN_DRIVER_API ||
					 domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
#else
	if (domain == CUPTI_CB_DOMAIN_DRIVER_API ||
					 domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
#endif //TAU_ASYNC_ACTIVITY_API
		const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *) params;

        // BEGIN handling memcpy
		if (function_is_memcpy(id, domain))
		{
#ifdef TAU_DEBUG_CUPTI
			printf("TAU: CUPTI callback for memcpy\n");
#endif
			int kind;
			int count;
			get_values_from_memcpy(cbInfo, id, domain, kind, count);
			if (cbInfo->callbackSite == CUPTI_API_ENTER)
			{
			  int taskId = 0;  // gpu no?
#if defined(PTHREADS)
			  if (map_cudaThread.find(cbInfo->correlationId) != map_cudaThread.end()) {
			    int local_vtid = map_cudaThread[cbInfo->correlationId].tau_vtid;
			    taskId = map_cuptiThread[local_vtid];
			  }
#endif
				Tau_cupti_enter_memcpy_event(
					cbInfo->functionName, -1, 0, cbInfo->contextUid, cbInfo->correlationId, 
					count, getMemcpyType(kind), taskId
				);
				Tau_cupti_register_host_calling_site(cbInfo->correlationId, cbInfo->functionName);
				/*
				CuptiGpuEvent new_id = CuptiGpuEvent(TAU_GPU_USE_DEFAULT_NAME, (uint32_t)0, cbInfo->contextUid, cbInfo->correlationId, NULL, 0);
				Tau_gpu_enter_memcpy_event(
					cbInfo->functionName,
					&new_id,
					count,
					getMemcpyType(kind)
				);
				*/
			}
			else
			{
#ifdef TAU_DEBUG_CUPTI
				cerr << "callback for " << cbInfo->functionName << ", exit." << endl;
#endif
				int taskId = 0;
#if defined(PTHREADS)
				if (map_cudaThread.find(cbInfo->correlationId) != map_cudaThread.end()) {
				  int local_vtid = map_cudaThread[cbInfo->correlationId].tau_vtid;
				  taskId = map_cuptiThread[local_vtid];
				}
#endif
				Tau_cupti_exit_memcpy_event(
					cbInfo->functionName, -1, 0, cbInfo->contextUid, cbInfo->correlationId, 
					count, getMemcpyType(kind), taskId
				);
				/*
				CuptiGpuEvent new_id = CuptiGpuEvent(TAU_GPU_USE_DEFAULT_NAME, (uint32_t)0, cbInfo->contextUid, cbInfo->correlationId, NULL, 0);
				Tau_gpu_exit_memcpy_event(
					cbInfo->functionName,
					&new_id,
					getMemcpyType(kind)
				);
				*/
				if (function_is_sync(id))
				{ 
          
#ifdef TAU_DEBUG_CUPTI
					cerr << "sync function name: " << cbInfo->functionName << endl;
#endif
					//Disable counter tracking during the sync.
					//Tau_CuptiLayer_disable();
					//cuCtxSynchronize();
					cudaDeviceSynchronize();
					//Tau_CuptiLayer_enable();
#if defined(PTHREADS)
	  int count_iter = TauEnv_get_cudaTotalThreads();
#else
          int count_iter = get_device_count();
#endif
          for (int i=0; i<count_iter; i++) {
            record_gpu_counters_at_sync(i);
          }

#ifdef TAU_ASYNC_ACTIVITY_API
          Tau_cupti_activity_flush_all();
          //cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
          //cuptiActivityFlush(cbInfo->context, 0, CUPTI_ACTIVITY_FLAG_NONE);
#else
					Tau_cupti_register_sync_event(cbInfo->context, 0, NULL, 0, 0);
#endif
          
				}
			}
		} // END handling memcpy
		else // This is something other than memcpy
		{
			if (cbInfo->callbackSite == CUPTI_API_ENTER)
			{
				if (function_is_exit(id))
				{
          //Do one last flush since this is our last opportunity.
#ifdef TAU_ASYNC_ACTIVITY_API
          cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
#endif
					//Stop collecting cupti counters.
					Tau_CuptiLayer_finalize();
				}
                if(strcmp(cbInfo->functionName, "cudaDeviceReset") == 0) {
                    fprintf(stderr, "TAU: WARNING! cudaDeviceReset was called. CUPTI counters will not be measured from now on.\n");
                }
				Tau_gpu_enter_event(cbInfo->functionName);
				if (function_is_launch(id))
				{ // ENTRY to a launch function
          Tau_CuptiLayer_init();

#ifdef TAU_DEBUG_CUPTI
          printf("[at call (enter), %d] name: %s.\n", cbInfo->correlationId, cbInfo->functionName);
#endif
				  record_gpu_launch(cbInfo->correlationId, cbInfo->functionName);
					CUdevice device;
					cuCtxGetDevice(&device);
                                        Tau_cuda_Event_Synchonize();
					int taskId = 0;
#if defined(PTHREADS)
					if (map_cudaThread.find(cbInfo->correlationId) != map_cudaThread.end()) {
					  int local_vtid = map_cudaThread[cbInfo->correlationId].tau_vtid;
					  taskId = map_cuptiThread[local_vtid];
					}
#endif
					record_gpu_counters_at_launch(taskId);
				}
#ifdef TAU_DEBUG_CUPTI
				cerr << "callback for " << cbInfo->functionName << ", enter." << endl;
#endif
			}
			else if (cbInfo->callbackSite == CUPTI_API_EXIT)
			{
				if (function_is_launch(id)) // EXIT FROM a launch function
				{
				  record_gpu_launch(cbInfo->correlationId, cbInfo->functionName);
				}
#ifdef TAU_DEBUG_CUPTI_FORCE_SYNC
      //for testing only. 
				if (function_is_launch(id))
				{
                    printf("synthetic sync point.\n");
                    cuCtxSynchronize();
                    FunctionInfo *p = TauInternal_CurrentProfiler(RtsLayer::myThread())->CallPathFunction;
                }
#endif
      
			
#ifdef TAU_DEBUG_CUPTI
				cerr << "callback for " << cbInfo->functionName << ", exit." << endl;
				printf("[at call (exit), %d] name: %s.\n", cbInfo->correlationId, cbInfo->functionName);
#endif
				Tau_gpu_exit_event(cbInfo->functionName);
				if (function_is_sync(id))
				{
#ifdef TAU_DEBUG_CUPTI
					cerr << "sync function name: " << cbInfo->functionName << endl;
#endif
					//Tau_CuptiLayer_disable();
					//cuCtxSynchronize();
					cudaDeviceSynchronize();
					//Tau_CuptiLayer_enable();
#if defined(PTHREADS)
	  int count_iter = TauEnv_get_cudaTotalThreads();
#else
          int count_iter = get_device_count();
#endif
          for (int i=0; i<count_iter; i++) {
            record_gpu_counters_at_sync(i);
          }

#ifdef TAU_ASYNC_ACTIVITY_API
          Tau_cupti_activity_flush_all();
          //cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
          //cuptiActivityFlush(cbInfo->context, 0, CUPTI_ACTIVITY_FLAG_NONE);
#else
					Tau_cupti_register_sync_event(cbInfo->context, 0, NULL, 0, 0);
#endif
					//Tau_CuptiLayer_enable();
				}
			}
		}
	}
	//invaild or nvtx, do nothing
	else {
		return;
	}

}

void CUPTIAPI Tau_cupti_activity_flush_all() {      
    if((Tau_CuptiLayer_get_num_events() > 0) || (buffers_queued++ > ACTIVITY_ENTRY_LIMIT)) {
        buffers_queued = 0;
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_NONE);
    }
}

void CUPTIAPI Tau_cupti_register_sync_event(CUcontext context, uint32_t stream, uint8_t *activityBuffer, size_t size, size_t bufferSize)
{
//   if(TauEnv_get_cuda_track_sass()) {
//     if(TauEnv_get_cuda_csv_output()) {
// #ifdef TAU_DEBUG_CUPTI_SASS
//       printf("[CuptiActivity]:  About to call createFilePointerSass, device_count: %i\n", device_count);
// #endif
//       createFilePointerSass(device_count);
//     }
//   }
  //Since we do not control the synchronization points this is only place where
  //we can record the gpu counters.
#ifdef TAU_ASYNC_ACTIVITY_API
#if defined(PTHREADS)
  int count_iter = TauEnv_get_cudaTotalThreads();
#else
  int count_iter = get_device_count();
#endif
  for (int i=0; i<count_iter; i++) {
    record_gpu_counters_at_sync(i);
  }
#endif
  //TAU_PROFILE("Tau_cupti_register_sync_event", "", TAU_DEFAULT);
	//printf("in sync: context=%p stream=%d.\n", context, stream);
	registered_sync = true;
  CUptiResult err, status;
  CUresult err2 = CUDA_SUCCESS;
  CUpti_Activity *record = NULL;
	//size_t bufferSize = 0;
  
  //start

#if defined(PTHREADS)
  if (count_iter > TAU_MAX_THREADS) {
    printf("TAU ERROR: Maximum number of threads (%d) exceeded. Please reconfigure TAU with -useropt=-DTAU_MAX_THREADS=3200 or some higher number\n", TAU_MAX_THREADS);
    exit(1);
  }
#else
  if (count_iter > TAU_MAX_GPU_DEVICES) {
    printf("TAU ERROR: Maximum number of devices (%d) exceeded. Please reconfigure TAU with -useropt=-DTAU_MAX_GPU_DEVICES=32 or some higher number\n", TAU_MAX_GPU_DEVICES);
    exit(1);
  }
#endif

// for the ASYNC ACTIVITY API assume that the activityBuffer is vaild
#ifdef TAU_ASYNC_ACTIVITY_API
  err = CUPTI_SUCCESS;
#else
	err = cuptiActivityDequeueBuffer(context, stream, &activityBuffer, &bufferSize);
#endif
	//printf("err: %d.\n", err);

    uint64_t num_buffers = 0;
	if (err == CUPTI_SUCCESS)
	{
		//printf("succesfully dequeue'd buffer.\n");
    //TAU_START("next record loop");
    //TAU_PROFILE_TIMER(g, "getNextRecord", "", TAU_DEFAULT);
    //TAU_PROFILE_TIMER(r, "record_activity", "", TAU_DEFAULT);
		do {
      //TAU_PROFILE_START(g);
			status = cuptiActivityGetNextRecord(activityBuffer, bufferSize, &record);
      //TAU_PROFILE_STOP(g);
			if (status == CUPTI_SUCCESS) {
        //TAU_PROFILE_START(r);
				Tau_cupti_record_activity(record);
                ++num_buffers;
        //TAU_PROFILE_STOP(r);
			}
			else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
				//const char *str;
				//cuptiGetResultString(status, &str);
				//printf("TAU ERROR: buffer limit encountered: %s.\n", str);
				break;
			}
			else {
				const char *str;
				cuptiGetResultString(status, &str);
				printf("TAU ERROR: cannot retrieve record from buffer: %s.\n", str);
				break;
			}
		} while (status != CUPTI_ERROR_MAX_LIMIT_REACHED);
    //TAU_STOP("next record loop");
			
		size_t number_dropped;
		err = cuptiActivityGetNumDroppedRecords(NULL, 0, &number_dropped);

		if (number_dropped > 0)
			printf("TAU WARNING: %d CUDA records dropped, consider increasing the CUPTI_BUFFER size.", number_dropped);

    // With the ASYNC ACTIVITY API CUPTI will take care of calling
    // Tau_cupti_register_buffer_creation() when it needs a new activity buffer so
    // we are free to deallocate it here.
#ifdef TAU_ASYNC_ACTIVITY_API
    free(activityBuffer);
#else

		//Need to requeue buffer by context, stream.
		err = cuptiActivityEnqueueBuffer(context, stream, activityBuffer, ACTIVITY_BUFFER_SIZE);
#endif
		CUDA_CHECK_ERROR(err2, "Cannot requeue buffer.\n");
		

    for (int i=0; i < count_iter; i++) {
#ifdef TAU_DEBUG_CUPTI_COUNTERS
      printf("Kernels encountered/recorded: %d/%d.\n", kernels_encountered[i], kernels_recorded[i]);
#endif

      if (kernels_recorded[i] == kernels_encountered[i])
      {
        clear_counters(i);
        last_recorded_kernel_name = NULL;
      } else if (kernels_recorded[i] > kernels_encountered[i]) {
        //printf("TAU: Recorded more kernels than were launched, exiting.\n");
        //abort();
        //exit(1);
      }
    }
//     for (int i=0; i < device_count; i++) {
// #ifdef TAU_DEBUG_CUPTI_COUNTERS
//       printf("Kernels encountered/recorded: %d/%d.\n", kernels_encountered[i], kernels_recorded[i]);
// #endif

//       if (kernels_recorded[i] == kernels_encountered[i])
//       {
//         clear_counters(i);
//         last_recorded_kernel_name = NULL;
//       } else if (kernels_recorded[i] > kernels_encountered[i]) {
//         //printf("TAU: Recorded more kernels than were launched, exiting.\n");
//         //abort();
//         //exit(1);
//       }
//     }
  } else if (err != CUPTI_ERROR_QUEUE_EMPTY) {
#ifdef TAU_DEBUG_CUPTI
		printf("TAU: CUPTI Activity queue is empty.\n");
		//CUDA_CHECK_ERROR(err2, "Cannot dequeue buffer.\n");
#endif
	} else if (err != CUPTI_ERROR_INVALID_PARAMETER) {
#ifdef TAU_DEBUG_CUPTI
        printf("TAU: CUPTI Invalid buffer");
		//CUDA_CHECK_ERROR(err2, "Cannot dequeue buffer, invalid buffer.\n");
#endif
	} else {
		printf("TAU: CUPTI Unknown error cannot read from buffer.\n");
	}

}

void CUPTIAPI Tau_cupti_register_buffer_creation(uint8_t **activityBuffer, size_t *size, size_t *maxNumRecords)
{
	uint8_t* bf = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
  if (bf == NULL) {
    printf("sufficient memory available to allocate activity buffer of size %d.", ACTIVITY_BUFFER_SIZE);
    exit(1);
  }
  *activityBuffer = bf;
  *size = ACTIVITY_BUFFER_SIZE;
  *maxNumRecords = 0;
}

bool register_cuda_thread(unsigned int sys_tid, unsigned int parent_tid, int tau_vtid, unsigned int corr_id, unsigned int context_id, const char* func_name) {
  
  CudaThread ct;
  ct.sys_tid = sys_tid;
  ct.parent_tid = parent_tid;
  ct.tau_vtid = tau_vtid;
  ct.correlation_id = corr_id;
  ct.context_id = context_id;
  ct.function_name = func_name;
  map_cudaThread[corr_id] = ct;
  return true;
}

/* This callback handles asynchronous activity */

void Tau_cupti_record_activity(CUpti_Activity *record)
{
  // can't handle out-of-order events
  // if (TauEnv_get_tracing()) { return; }
  // currentTimestamp
  uint64_t currentTimestamp;
  double d_currentTimestamp;
  CUptiResult err = CUPTI_SUCCESS;
  CUresult err2 = CUDA_SUCCESS;
  err = cuptiGetTimestamp(&currentTimestamp); // nanosec
  ///////
  // Within python,
  //   seconds = (int)(cumsum / 1000) % 60
  //   minutes = (int)(cumsum / (1000*60)) % 60
  ///////
  d_currentTimestamp = (double)currentTimestamp/1e3; // orig
  // d_currentTimestamp = (double)currentTimestamp/1e6; 


  CUDA_CHECK_ERROR(err2, "Cannot get timestamp.\n");
  
  switch (record->kind) {
  // case CUPTI_ACTIVITY_KIND_DRIVER:
  //   {
  //     CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
  //     printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
  //            api->cbid,
  // 	     (unsigned long long) (api->start - timestamp),
  // 	     (unsigned long long) (api->end - timestamp),
  // 	     api->processId, api->threadId, api->correlationId);
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_RUNTIME:
  //   {
  //     CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
  //     printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
  //            api->cbid,
  // 	     (unsigned long long) (api->start - timestamp),
  // 	     (unsigned long long) (api->end - timestamp),
  // 	     api->processId, api->threadId, api->correlationId);
  //     if (map_cudaThread.find(api->correlationId) != map_cudaThread.end()) {
  // 	printf("  map_cudaThread[%u]: %i\n", 
  // 	       api->correlationId, map_cudaThread[api->correlationId].tau_vtid);
  //     }
  //     else {
  // 	printf("  map_cudaThread[%u] does not exist\n", api->correlationId);
  //     }
  //     break;
  //   }
  // case CUPTI_ACTIVITY_KIND_CONTEXT:
  //   {
  //     CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
  //     // printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
  //     //        context->contextId, context->deviceId,
  //     //        getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
  //     //        (int) context->nullStreamId);
  //     contextMap[context->contextId] = *context;
  //     break;
  //   }
  	case CUPTI_ACTIVITY_KIND_MEMCPY:
#if CUDA_VERSION >= 5050
	  case CUPTI_ACTIVITY_KIND_MEMCPY2:
#endif
		{	
      uint32_t deviceId;
      uint32_t streamId;
      uint32_t contextId;
      uint64_t start;
      uint64_t end;
      uint64_t bytes;
      uint8_t copyKind;
			int id;
      int direction = MESSAGE_UNKNOWN;

#if CUDA_VERSION >= 5050
      if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY2) {
        CUpti_ActivityMemcpy2 *memcpy = (CUpti_ActivityMemcpy2 *)record;
        deviceId = memcpy->deviceId;
        streamId = memcpy->streamId;
        contextId = memcpy->contextId;
        start = memcpy->start;
        end = memcpy->end;
        bytes = memcpy->bytes;
        copyKind = memcpy->copyKind;
        id = memcpy->correlationId;
#ifdef TAU_DEBUG_CUPTI
        cerr << "recording memcpy (device, stream, context, correlation): " << memcpy->deviceId << ", " << memcpy->streamId << ", " << memcpy->contextId << ", " << memcpy->correlationId << ", " << memcpy->start << "-" << memcpy->end << "ns.\n" << endl;
		cerr << "recording memcpy src: " << memcpy->srcDeviceId << "/" << memcpy->srcContextId << endl;
		cerr << "recording memcpy dst: " << memcpy->dstDeviceId << "/" << memcpy->dstContextId << endl;
#endif
	// get Correlationid
	int taskId = 0;
#if defined(PTHREADS)
	if (map_cudaThread.find(id) != map_cudaThread.end()) {
	  int local_vtid = map_cudaThread[id].tau_vtid;
	  taskId = map_cuptiThread[local_vtid];
	}
#endif
        Tau_cupti_register_memcpy_event(
          TAU_GPU_USE_DEFAULT_NAME,
          memcpy->srcDeviceId,
          streamId,
          memcpy->srcContextId,
          id,
          start / 1e3,
          end / 1e3,
          bytes,
          getMemcpyType(copyKind),
          MESSAGE_RECIPROCAL_SEND, taskId
        );
        Tau_cupti_register_memcpy_event(
          TAU_GPU_USE_DEFAULT_NAME,
          memcpy->dstDeviceId,
          streamId,
          memcpy->dstContextId,
          id,
          start / 1e3,
          end / 1e3,
          bytes,
          getMemcpyType(copyKind),
          MESSAGE_RECIPROCAL_RECV, taskId
        );
} else {
#endif
        CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
        deviceId = memcpy->deviceId;
        streamId = memcpy->streamId;
        contextId = memcpy->contextId;
        start = memcpy->start;
        end = memcpy->end;
        bytes = memcpy->bytes;
        copyKind = memcpy->copyKind;
        if (cupti_api_runtime())
        {
          id = memcpy->runtimeCorrelationId;
        }
        else
        {
          id = memcpy->correlationId;
        }
        if (getMemcpyType(copyKind) == MemcpyHtoD) {
          direction = MESSAGE_RECV;
        } else if (getMemcpyType(copyKind) == MemcpyDtoH) {
          direction = MESSAGE_SEND;
        }
#ifdef TAU_DEBUG_CUPTI
			cerr << "recording memcpy: " << end - start << "ns.\n" << endl;
		    cerr << "recording memcpy on device: " << deviceId << endl;
		    cerr << "recording memcpy kind: " << getMemcpyType(copyKind) << endl;
#endif 
			//We do not always know on the corresponding host event on
			//the CPU what type of copy we have so we need to register 
			//the bytes copied here. Be careful we only want to record 
			//the bytes copied once.
		    int taskId = 0;
#if defined(PTHREADS)
		    if (map_cudaThread.find(id) != map_cudaThread.end()) {
		      int local_vtid = map_cudaThread[id].tau_vtid;
		      taskId = map_cuptiThread[local_vtid];
		    }
#endif
			Tau_cupti_register_memcpy_event(
				TAU_GPU_USE_DEFAULT_NAME,
				deviceId,
				streamId,
				contextId,
				id,
				start / 1e3,
				end / 1e3,
				bytes,
				getMemcpyType(copyKind),
				direction, taskId
			);
			/*
			CuptiGpuEvent gId = CuptiGpuEvent(TAU_GPU_USE_DEFAULT_NAME, memcpy->streamId, memcpy->contextId, id, NULL, 0);
			//cuptiGpuEvent cuRec = cuptiGpuEvent(TAU_GPU_USE_DEFAULT_NAME, &gId, NULL); 
			Tau_gpu_register_memcpy_event(
				&gId,
				memcpy->start / 1e3, 
				memcpy->end / 1e3, 
				TAU_GPU_UNKNOW_TRANSFER_SIZE, 
				getMemcpyType(memcpy->copyKind));
			*/	
#if CUDA_VERSION >= 5050
  }
#endif
      
				break;
      }

    if(TauEnv_get_cuda_track_unified_memory()) {
#if CUDA_VERSION >= 6000
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
    {	
      CUpti_ActivityUnifiedMemoryCounterKind counterKind;
      uint32_t deviceId;
      uint32_t streamId;
      uint32_t processId;
      CUpti_ActivityUnifiedMemoryCounterScope scope;
      uint64_t start;
      uint64_t end;
      uint64_t value;
      int direction = MESSAGE_UNKNOWN;
      CUpti_ActivityUnifiedMemoryCounter *umemcpy = (CUpti_ActivityUnifiedMemoryCounter *)record;
      
#ifdef TAU_DEBUG_CUPTI
#if CUDA_VERSION >= 7000
      printf("UNIFIED_MEMORY_COUNTER [ %llu %llu ] kind=%s value=%llu src %u dst %u, streamId=%u\n",
	     (unsigned long long)(umemcpy->start),
	     (unsigned long long)(umemcpy->end),
	     getUvmCounterKindString(umemcpy->counterKind),
	     (unsigned long long)umemcpy->value,
	     umemcpy->srcId,
	     umemcpy->dstId,
	     umemcpy->streamId);
#else
      printf("UNIFIED_MEMORY_COUNTER [ %llu ], current stamp: %llu, scope=%d kind=%s value=%llu device %u\n",
	     (unsigned long long)(umemcpy->timestamp), TauTraceGetTimeStamp(), 
	     umemcpy->scope,
	     getUvmCounterKindString(umemcpy->counterKind),
	     (unsigned long long)umemcpy->value,
	     umemcpy->deviceId);
#endif
#endif
      counterKind = umemcpy->counterKind;
#if CUDA_VERSION < 7000
      streamId = -1;
      start = umemcpy->timestamp;
      end = umemcpy->timestamp;
      deviceId = umemcpy->deviceId;
#else
      streamId = umemcpy->streamId;
      start = umemcpy->start;
      end=umemcpy->end;
      deviceId = umemcpy->dstId;
#endif
      processId = umemcpy->processId;
      value = umemcpy->value;
      
      if (getUnifmemType(counterKind) == BytesHtoD) {
	direction = MESSAGE_RECV;
      } else if (getUnifmemType(counterKind) == BytesDtoH) {
	direction = MESSAGE_SEND;
      }
      
      //We do not always know on the corresponding host event on
      //the CPU what type of copy we have so we need to register 
      //the bytes copied here. Be careful we only want to record 
      //the bytes copied once.
      int taskId = 1; // need to get correlation id from CUpti_ActivityStream
      Tau_cupti_register_unifmem_event(
				       TAU_GPU_USE_DEFAULT_NAME,
				       deviceId,
				       streamId,
				       processId,
				       start,
				       end,
				       value,
				       getUnifmemType(counterKind),
				       direction,
				       taskId
				       );
      
      break;
    }
#endif
    }

  	case CUPTI_ACTIVITY_KIND_KERNEL:
#if CUDA_VERSION >= 5000
  	case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
#endif
#if CUDA_VERSION >= 5050
    case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
#endif
    {
      const char *name;
      uint32_t deviceId;
      uint32_t streamId;
      uint32_t contextId;
      uint32_t correlationId;
#if CUDA_VERSION < 5050
      uint32_t runtimeCorrelationId;
#endif
      uint64_t start;
      uint64_t end;
      int64_t gridId;
      int64_t parentGridId;
      uint32_t blockX;
      uint32_t blockY; 
      uint32_t blockZ;
      uint32_t dynamicSharedMemory;
      uint32_t staticSharedMemory;
      uint32_t localMemoryPerThread;
      uint32_t registersPerThread;

#if CUDA_VERSION >= 5050
      if (record->kind == CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
	printf(" inside cdp_kernel\n");
        CUpti_ActivityCdpKernel *kernel = (CUpti_ActivityCdpKernel *)record;
        name = kernel->name;
        deviceId = kernel->deviceId;
        streamId = kernel->streamId;
        contextId = kernel->contextId;
        correlationId = kernel->correlationId;
        gridId = kernel->gridId;
        parentGridId = kernel->parentGridId;
        start = kernel->start;
        end = kernel->end;
        blockX = kernel->blockX;
        blockY = kernel->blockY; 
        blockZ = kernel->blockZ;
        dynamicSharedMemory = kernel->dynamicSharedMemory;
        staticSharedMemory = kernel->staticSharedMemory;
        localMemoryPerThread = kernel->localMemoryPerThread;
        registersPerThread = kernel->registersPerThread;
      }
      else {
#endif
        CUpti_ActivityKernel *kernel = (CUpti_ActivityKernel *)record;
        name = kernel->name;
        deviceId = kernel->deviceId;
        streamId = kernel->streamId;
        contextId = kernel->contextId;
        correlationId = kernel->correlationId;
#if CUDA_VERSION >= 7000
        runtimeCorrelationId = correlationId;
        gridId = kernel->gridId;
#elif CUDA_VERSION >= 5050 && CUDA_VERSION <= 6500
        gridId = kernel->gridId;
        runtimeCorrelationId = kernel->runtimeCorrelationId;
#else
        gridId = 0;
        runtimeCorrelationId = correlationId;
#endif
        start = kernel->start;
        end = kernel->end;
        blockX = kernel->blockX;
        blockY = kernel->blockY; 
        blockZ = kernel->blockZ;
        dynamicSharedMemory = kernel->dynamicSharedMemory;
        staticSharedMemory = kernel->staticSharedMemory;
        localMemoryPerThread = kernel->localMemoryPerThread;
        registersPerThread = kernel->registersPerThread;
        //find FunctionInfo object from FunctionInfoMap
        // kernelMap[kernel->correlationId] = *kernel;
	int taskId = 0;
#if defined(PTHREADS)
	if (map_cudaThread.find(kernel->correlationId) != map_cudaThread.end()) {
	  int local_vtid = map_cudaThread[kernel->correlationId].tau_vtid;
	  taskId = map_cuptiThread[local_vtid];
	}
#endif
        kernelMap[taskId][kernel->correlationId] = *kernel;
#if CUDA_VERSION >= 5050
      }
#endif
#ifdef TAU_DEBUG_CUPTI
      cerr << "recording kernel (device, stream, context, correlation, grid, name): " << deviceId << ", " << streamId << ", " << contextId << ", " << correlationId << ", " << gridId << ", " << name << ", "<< start << "-" << end << "ns.\n" << endl;
      /*if (record->kind == CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
        cerr << "CDP kernel, parent is: " << parentGridId << endl;
      }*/
	  // cerr << "recording kernel (id): "  << kernel->correlationId << ", " << kernel->name << ", "<< kernel->end - kernel->start << "ns.\n" << endl;
#endif
      uint32_t id;
      if (cupti_api_runtime())
      {
	id = runtimeCorrelationId;
      }
      else
      {
	id = correlationId;
      }
      int taskId = 0;
#if defined(PTHREADS)
      if (map_cudaThread.find(id) != map_cudaThread.end()) {
	int local_vtid = map_cudaThread[id].tau_vtid;
	taskId = map_cuptiThread[local_vtid];
      }
#endif
      // At this point store source locator and function maps accumulated, then clear maps
      for (std::map<uint32_t, CUpti_ActivitySourceLocator>::iterator it = sourceLocatorMap.begin();
	   it != sourceLocatorMap.end();
	   it++) {
	uint32_t srclocid = it->first;
	CUpti_ActivitySourceLocator source = it->second;
	cout << "[CuptiActivity] testing iter for source locator (id): " << source.id << ", " << source.fileName << ", " << source.lineNumber << ".\n" << endl;
      }
      eventMap[taskId].erase(eventMap[taskId].begin(), eventMap[taskId].end());
      const char* name_og = name;
      name = demangleName(name);

      int number_of_metrics = Tau_CuptiLayer_get_num_events() + 1;
      double metrics_start[number_of_metrics];
      double metrics_end[number_of_metrics];
#if CUDA_VERSION >= 5050
      if (record->kind != CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
	int taskId = 0;
#if defined(PTHREADS)
	if (map_cudaThread.find(id) != map_cudaThread.end()) {
	  int local_vtid = map_cudaThread[id].tau_vtid;
	  taskId = map_cuptiThread[local_vtid];
	}
#endif
	record_gpu_counters(taskId, name, id, &eventMap[taskId]);
      }
#else
      int taskId = 0;
#if defined(PTHREADS)
      if (map_cudaThread.find(id) != map_cudaThread.end()) {
	int local_vtid = map_cudaThread[id].tau_vtid;
	taskId = map_cuptiThread[local_vtid];
      }
#endif
      record_gpu_counters(taskId, name, id, &eventMap[taskId]);
#endif
      if (TauEnv_get_cuda_track_sass()) {
	if (!functionMap.empty() && !instructionMap.empty()) {
	  // TODO:  Add source and function to maps here?
	  printf("[CuptiActivity]:  sass detected, taskId %i\n", taskId);

	  // TAU_VERBOSE("About to record imix counters\n");
	  // record_imix_counters(name, taskId, streamId, contextId, id, end);
	  // // if csv, dump here
	  // if(TauEnv_get_cuda_csv_output()){
	  //   dump_sass_to_csv();
	  // }	  
	}
	else {
	  TAU_VERBOSE("Instruction execution data not available\n");
	}
      }
			if (gpu_occupancy_available(deviceId))
			{
			  int taskId = 0;
#if defined(PTHREADS)
			  if (map_cudaThread.find(id) != map_cudaThread.end()) {
			    int local_vtid = map_cudaThread[id].tau_vtid;
			    taskId = map_cuptiThread[local_vtid];
			  }
#endif
        record_gpu_occupancy(blockX, 
                            blockY,
                            blockZ,
                            registersPerThread,
                            staticSharedMemory,
                            deviceId,
                            name, 
                            &eventMap[taskId]);
        static TauContextUserEvent* bs;
        static TauContextUserEvent* dm;
        static TauContextUserEvent* sm;
        static TauContextUserEvent* lm;
        static TauContextUserEvent* lr;
        Tau_get_context_userevent((void **) &bs, "Block Size");
        Tau_get_context_userevent((void **) &dm, "Shared Dynamic Memory (bytes)");
        Tau_get_context_userevent((void **) &sm, "Shared Static Memory (bytes)");
        Tau_get_context_userevent((void **) &lm, "Local Memory (bytes per thread)");
        Tau_get_context_userevent((void **) &lr, "Local Registers (per thread)");

        eventMap[taskId][bs] = blockX * blockY * blockZ;
        eventMap[taskId][dm] = dynamicSharedMemory;
        eventMap[taskId][sm] = staticSharedMemory;
        eventMap[taskId][lm] = localMemoryPerThread;
        eventMap[taskId][lr] = registersPerThread;
			}
      
      GpuEventAttributes *map;
			int map_size = eventMap[taskId].size();
			map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
      int i = 0;
      for (eventMap_t::iterator it = eventMap[taskId].begin(); it != eventMap[taskId].end(); it++)
      {
        /*if(it->first == NULL) {
            std::cerr << "Event was null!" << std::endl;
        } else {
            std::cerr << "Event was not null: " << it->first << std::endl;
        }*/
        //std::cerr << "event name: " << it->first->GetUserEventName() << std::endl;
        map[i].userEvent = it->first;
        map[i].data = it->second;
        i++;
      }
			
#if CUDA_VERSION >= 5050
      if (record->kind == CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
        if (TauEnv_get_cuda_track_cdp()) {
	  int taskId = 0;
#if defined(PTHREADS)
	  if (map_cudaThread.find(id) != map_cudaThread.end()) {
	    int local_vtid = map_cudaThread[id].tau_vtid;
	    taskId = map_cuptiThread[local_vtid];
	  }
#endif
	Tau_cupti_register_gpu_event(name, deviceId,
				       streamId, contextId, id, parentGridId, 
				       true, map, map_size,
				     start / 1e3, end / 1e3, taskId);
        }
      } else {
#endif
	int taskId = 0;
#if defined(PTHREADS)
	if (map_cudaThread.find(id) != map_cudaThread.end()) {
	  int local_vtid = map_cudaThread[id].tau_vtid;
	  taskId = map_cuptiThread[local_vtid];
	}
#endif
        Tau_cupti_register_gpu_event(name, deviceId,
				     streamId, contextId, id, 0, false, map, map_size,
				     start / 1e3, end / 1e3, taskId);
#if CUDA_VERSION >= 5050
      }
#endif
        Tau_cupti_register_device_calling_site(gridId, name);
			/*
			CuptiGpuEvent gId = CuptiGpuEvent(name, kernel->streamId, kernel->contextId, id, map, map_size);
			//cuptiGpuEvent cuRec = cuptiGpuEvent(name, &gId, &map);
			Tau_gpu_register_gpu_event(
				&gId, 
				kernel->start / 1e3,
				kernel->end / 1e3);
			*/	

			break;
    }

  	case CUPTI_ACTIVITY_KIND_DEVICE:
		{
			CUpti_ActivityDevice *device = (CUpti_ActivityDevice *)record;

			int nMeta = 17;
			
			GpuMetadata *metadata = (GpuMetadata *) malloc(sizeof(GpuMetadata) * nMeta);
			int id = 0;
			//first the name.
			metadata[id].name = (char*)("GPU Name");
			metadata[id].value = device->name;
			id++;

			//the rest.
			RECORD_DEVICE_METADATA(computeCapabilityMajor, device);
			RECORD_DEVICE_METADATA(computeCapabilityMinor, device);
			RECORD_DEVICE_METADATA(constantMemorySize, device);
			RECORD_DEVICE_METADATA(coreClockRate, device);
			RECORD_DEVICE_METADATA(globalMemoryBandwidth, device);
			RECORD_DEVICE_METADATA(globalMemorySize, device);
			RECORD_DEVICE_METADATA(l2CacheSize, device);
			RECORD_DEVICE_METADATA(maxIPC, device);
			RECORD_DEVICE_METADATA(maxRegistersPerBlock, device);
			RECORD_DEVICE_METADATA(maxSharedMemoryPerBlock, device);
			RECORD_DEVICE_METADATA(maxThreadsPerBlock, device);
			RECORD_DEVICE_METADATA(maxWarpsPerMultiprocessor, device);
			RECORD_DEVICE_METADATA(maxBlocksPerMultiprocessor, device);
			RECORD_DEVICE_METADATA(numMemcpyEngines, device);
			RECORD_DEVICE_METADATA(numMultiprocessors, device);
			RECORD_DEVICE_METADATA(numThreadsPerWarp, device);
	
			//cerr << "recording metadata (device): " << device->id << endl;
			__deviceMap()[device->id] = *device;
#if CUDA_VERSION < 5000
      if (__deviceMap().size() > 1 && Tau_CuptiLayer_get_num_events() > 0)
      {
        TAU_VERBOSE("TAU Warning: CUDA 5.0 or greater is needed to record counters on more that one GPU device at the same time.\n");
      }
#endif
			Tau_cupti_register_metadata(device->id, metadata, nMeta);
			break;
		}
#if CUPTI_API_VERSION >= 3
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
    {
      CUpti_ActivitySourceLocator *source = (CUpti_ActivitySourceLocator *)record;
      sourceLocatorMap[source->id] = *source;
      // double tstamp;
      // uint32_t sourceId;
      // const char *fileName;
      // uint32_t lineNumber;
#ifdef TAU_DEBUG_CUPTI
      cerr << "source locator (id): " << source->id << ", " << source->fileName << ", " << source->lineNumber << ".\n" << endl;
#endif
      srcLocMap[source->id] = *source;

#if CUDA_VERSION >= 5500
      if(TauEnv_get_cuda_track_sass()) {

#ifdef TAU_DEBUG_CUPTI_SASS
	  printf("SOURCE_LOCATOR SrcLctrId: %d, File: %s, Line: %d, Kind: %u\n", 
	  	 source->id, source->fileName, source->lineNumber, source->kind);
#endif
    // if(TauEnv_get_cuda_csv_output()){
    //   // TAU stores time in microsec (1.0e-6), nanosec->microsec 1->0.001 ns/1000
    //   // Source Locator same for all GPUs
    //   int taskId = CudaThreadMap()[cbInfo->correlationId].tau_vtid;
    //   // create file pointer here
    //   FILE* fp_sour = createFileSourceSass(taskId - 1); // want 0,...,N-1, not 1,...,N!
      
    //   fprintf(fp_sour, "%f,%d,%s,%d,%u\n",
    // 	      d_currentTimestamp,source->id, source->fileName, source->lineNumber, source->kind);

    // }
      // char name[] = "SOURCE_LOCATOR";
      // Tau_cupti_register_source_event(name, 0, 0, 0, sourceId, d_currentTimestamp, fileName, lineNumber);

      }
#endif

    }

#if CUDA_VERSION >= 5500
	case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION: {
    if(TauEnv_get_cuda_track_sass()) {
	  CUpti_ActivityInstructionExecution *instrRecord = (CUpti_ActivityInstructionExecution *)record;

	  // uint32_t correlationId;
	  // uint32_t executed;
	  // uint32_t functionId;
	  // uint32_t pcOffset;
	  // uint32_t sourceLocatorId;
	  // uint32_t threadsExecuted;
	  // CUpti_ActivityContext cResult = contextMap.find(current_context_id)->second;
	  
#ifdef TAU_DEBUG_CUPTI_SASS

	  printf("INSTRUCTION_EXECUTION corr: %u, executed: %u, flags: %u, functionId: %u, kind: %u, notPredOffThreadsExecuted: %u, pcOffset: %u, sourceLocatorId: %u, threadsExecuted: %u\n",
	  	 instrRecord->correlationId, instrRecord->executed, 
	  	 instrRecord->flags, instrRecord->functionId, 
	  	 instrRecord->kind, instrRecord->notPredOffThreadsExecuted,
	  	 instrRecord->pcOffset, instrRecord->sourceLocatorId, 
	  	 instrRecord->threadsExecuted);
#endif
  // if(TauEnv_get_cuda_csv_output()){
  //   int taskId = CudaThreadMap()[sourceRecord->correlationId].tau_vtid;
  //   // create file pointer here
  //   FILE* fp_inst = createFileInstrSass(taskId - 1); // want 0,...,N-1, not 1,...,N!
  //   fprintf(fp_inst, "%f,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
  // 	    d_currentTimestamp,
  // 	    sourceRecord->correlationId, sourceRecord->executed, 
  // 	    sourceRecord->flags, sourceRecord->functionId, 
  // 	    sourceRecord->kind, sourceRecord->notPredOffThreadsExecuted,
  // 	    sourceRecord->pcOffset, sourceRecord->sourceLocatorId, 
  // 	    sourceRecord->threadsExecuted);
  // }
  // InstrSampling is;
  // is.correlationId = sourceRecord->correlationId;
  // is.executed = sourceRecord->executed;
  // is.functionId = sourceRecord->functionId;
  // is.pcOffset = sourceRecord->pcOffset;
  // is.sourceLocatorId = sourceRecord->sourceLocatorId;
  // is.threadsExecuted = sourceRecord->threadsExecuted;
  // is.timestamp_delta = d_currentTimestamp-recentTimestamp;
  // is.timestamp_current = d_currentTimestamp;
  // instructionMap[is.functionId].push_back(is);
	  instructionMap[instrRecord->functionId].push_back(*instrRecord);
	  // TODO:  store sourceid -> correlationId pair

	  // // printf("d_currentTImestamp: %f, recentTimestamp: %f, tstamp_delta: %f\n", 
	  // // 	 d_currentTimestamp, recentTimestamp, tstamp_delta);
	  // Tau_cupti_register_instruction_event(name,cResult.deviceId,
	  // 				       (int)cResult.nullStreamId,
	  // 				       cResult.contextId,correlationId,recentTimestamp,
	  // 				       d_currentTimestamp,tstamp_delta,
	  // 				       sourceLocatorId,functionId,
	  // 				       pcOffset,executed,
	  // 				       threadsExecuted);
    }
    break;
	}
#endif
#if CUDA_VERSION >= 5500
	case CUPTI_ACTIVITY_KIND_FUNCTION: {
	  if(TauEnv_get_cuda_track_sass()) {
	    CUpti_ActivityFunction *fResult = (CUpti_ActivityFunction *)record;
	    
	    // uint32_t contextId;
	    // uint32_t functionIndex;
	    // uint32_t id;
	    // uint32_t moduleId;
	    // const char *kname;
	    
#ifdef TAU_DEBUG_CUPTI_SASS
	    printf("FUCTION contextId: %u, functionIndex: %u, id %u, kind: %u, moduleId %u, name %s, device: %i\n",
		   fResult->contextId,
		   fResult->functionIndex,
		   fResult->id,
		   fResult->kind,
		   fResult->moduleId,
		   fResult->name);
#endif
	    // char str_demangled[100];
	    // strcpy (str_demangled, demangleName(fResult->name));
// 	    CUpti_ActivityContext cResult = contextMap.find(fResult->contextId)->second;
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	    printf("context->contextId: %u, device: %u\n", cResult.contextId, cResult.deviceId);
// #endif
// 	    // current_device_id = cResult.deviceId;
// 	    current_context_id = cResult.contextId;
	    
	    // FuncSampling fs;
	    // fs.fid = fResult->id;
	    // fs.contextId = fResult->contextId;
	    // fs.functionIndex = fResult->functionIndex;
	    // fs.moduleId = fResult->moduleId;
	    // fs.name = fResult->name;
	    // fs.demangled = str_demangled;
	    // fs.timestamp = d_currentTimestamp;
	    // fs.deviceId = cResult.deviceId;
	    
	    // functionMap[fs.fid] = fs;
	    functionMap[fResult->id] = *fResult;
// 	    if(TauEnv_get_cuda_csv_output()){
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	      printf("[CuptiActivity]:  About to write out to csv:\n  %f, %u, %u, %u, %u, %u, %s, %s\n",
// 		     d_currentTimestamp, fResult->contextId,
// 		     fResult->functionIndex,
// 		     fResult->id,
// 		     fResult->kind,
// 		     fResult->moduleId,
// 		     fResult->name, demangleName(fResult->name));
// #endif
// 	      int taskId = CudaThreadMap()[cbInfo->correlationId].tau_vtid;
// 	      FILE* fp_funct = createFileFuncSass(taskId-1);
// 	      fprintf(fp_funct, "%f;%u;%u;%u;%u;%u;%s;%s\n",
// 		      d_currentTimestamp,
// 		      fResult->contextId,
// 		      fResult->functionIndex,
// 		      fResult->id,
// 		      fResult->kind,
// 		      fResult->moduleId,
// 		      fResult->name, demangleName(fResult->name));
// 	    }
	  // char name[] = "FUNCTION_ACTIVITY";
	  // char str_demangled[100];
	  // strcpy (str_demangled, demangleName(fResult->name));
	  // contextId = fResult->contextId;
	  // functionIndex = fResult->functionIndex;
	  // id = fResult->id;
	  // moduleId = fResult->moduleId;
	  // kname = fResult->name;

	  // Tau_cupti_register_func_event(name, cResult.deviceId,
	  // 				(int)cResult.nullStreamId, contextId, functionIndex,
	  // 				d_currentTimestamp, id, moduleId,
	  // 				kname, str_demangled);
	  }
                                                                                                               
	  break;
	}
#endif

    case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS:
    {
			CUpti_ActivityGlobalAccess *global_access = (CUpti_ActivityGlobalAccess *)record;
#ifdef TAU_DEBUG_CUPTI
			cerr << "global access (cor. id) (source id): " << global_access->correlationId << ", " << global_access->sourceLocatorId << ", " << global_access->threadsExecuted << ".\n" << endl;
#endif
      // CUpti_ActivityKernel *kernel = &kernelMap[global_access->correlationId];
      int taskId = 0;
#if defined(PTHREADS)
      if (map_cudaThread.find(global_access->correlationId) != map_cudaThread.end()) {
	int local_vtid = map_cudaThread[global_access->correlationId].tau_vtid;
	taskId = map_cuptiThread[local_vtid];
      }
#endif
      CUpti_ActivityKernel *kernel = &kernelMap[taskId][global_access->correlationId];
      CUpti_ActivitySourceLocator *source = &sourceLocatorMap[global_access->sourceLocatorId];

      if (kernel->kind != CUPTI_ACTIVITY_KIND_INVALID)
      {
	int taskId = 0;
#if defined(PTHREADS)
	if (map_cudaThread.find(global_access->correlationId) != map_cudaThread.end()) {
	  int local_vtid = map_cudaThread[global_access->correlationId].tau_vtid;
	  taskId = map_cuptiThread[local_vtid];
	}
#endif
        eventMap[taskId].erase(eventMap[taskId].begin(), eventMap[taskId].end());

        std::string name;
        form_context_event_name(kernel, source, "Accesses to Global Memory", &name);
        TauContextUserEvent* ga;
        Tau_cupti_find_context_event(&ga, name.c_str(), false);
        eventMap[taskId][ga] = global_access->executed;
        int map_size = eventMap[taskId].size();
        GpuEventAttributes *map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
        int i = 0;
        for (eventMap_t::iterator it = eventMap[taskId].begin(); it != eventMap[taskId].end(); it++)
        {
          map[i].userEvent = it->first;
          map[i].data = it->second;
          i++;
        }
        uint32_t id;
        if (cupti_api_runtime())
        {
	  #if CUDA_VERSION >= 6000 && CUDA_VERSION <= 6500
          id = kernel->runtimeCorrelationId;
	  #else
	  id = kernel->correlationId;
	  #endif
        }
        else
        {
          id = kernel->correlationId;
        }
        Tau_cupti_register_gpu_atomic_event(demangleName(kernel->name), kernel->deviceId,
					    kernel->streamId, kernel->contextId, id, map, map_size, taskId);
      }
    }
    case CUPTI_ACTIVITY_KIND_BRANCH:
    {
			CUpti_ActivityBranch *branch = (CUpti_ActivityBranch *)record;
#ifdef TAU_DEBUG_CUPTI
			cerr << "branch (cor. id) (source id): " << branch->correlationId << ", " << branch->sourceLocatorId << ", " << branch->threadsExecuted << ".\n" << endl;
#endif
     
      // CUpti_ActivityKernel *kernel = &kernelMap[branch->correlationId];
      int taskId = 0;
#if defined(PTHREADS)
      if (map_cudaThread.find(branch->correlationId) != map_cudaThread.end()) {
	int local_vtid = map_cudaThread[branch->correlationId].tau_vtid;
	taskId = map_cuptiThread[local_vtid];
      }
#endif
      CUpti_ActivityKernel *kernel = &kernelMap[taskId][branch->correlationId];
      CUpti_ActivitySourceLocator *source = &sourceLocatorMap[branch->sourceLocatorId];

      if (kernel->kind != CUPTI_ACTIVITY_KIND_INVALID)
      {
	int taskId = 0;
#if defined(PTHREADS)
	if (map_cudaThread.find(branch->correlationId) != map_cudaThread.end()) {
	  int local_vtid = map_cudaThread[branch->correlationId].tau_vtid;
	  taskId = map_cuptiThread[local_vtid];
	}
#endif
        eventMap[taskId].erase(eventMap[taskId].begin(), eventMap[taskId].end());
        
        std::string name;
        form_context_event_name(kernel, source, "Branches Executed", &name);
        TauContextUserEvent* be;
        Tau_cupti_find_context_event(&be, name.c_str(), false);
        eventMap[taskId][be] = branch->executed;
        
        form_context_event_name(kernel, source, "Branches Diverged", &name);
        TauContextUserEvent* de;
        Tau_cupti_find_context_event(&de, name.c_str(), false);
        eventMap[taskId][de] = branch->diverged;

        GpuEventAttributes *map;
        int map_size = eventMap[taskId].size();
        map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
        int i = 0;
        for (eventMap_t::iterator it = eventMap[taskId].begin(); it != eventMap[taskId].end(); it++)
        {
          map[i].userEvent = it->first;
          map[i].data = it->second;
          i++;
        }
        uint32_t id;
        if (cupti_api_runtime())
        {
          id = kernel->runtimeCorrelationId;
        }
        else
        {
          id = kernel->correlationId;
        }
        Tau_cupti_register_gpu_atomic_event(demangleName(kernel->name), kernel->deviceId,
					    kernel->streamId, kernel->contextId, id, map, map_size, taskId);
      }
    }
#endif //CUPTI_API_VERSION >= 3
	}
  recentTimestamp = d_currentTimestamp;

}

//Helper function givens ceiling with given significance.
int ceil(float value, int significance)
{
	return ceil(value/significance)*significance;
}

int gpu_occupancy_available(int deviceId)
{ 
	//device callback not called.
	if (__deviceMap().empty())
	{
		return 0;
	}

	CUpti_ActivityDevice device = __deviceMap()[deviceId];

	if ((device.computeCapabilityMajor > 7) ||
		device.computeCapabilityMajor == 7 &&
		device.computeCapabilityMinor > 1)
	{
		TAU_VERBOSE("TAU Warning: GPU occupancy calculator is not implemented for devices of compute capability > 7.1.");
		return 0;
	}
	//gpu occupancy available.
	return 1;	
}
int gpu_source_locations_available()
{
  //always available. 
  return 1;
}

// void dump_sass_to_csv(int task_id) {
//   // instr
  
// //   // create file pointer here
// //   fopen(fp_instr = createFileInstrSass(task_id - 1); // want 0,...,N-1, not 1,...,N!
	
// //   char str_int[5];
// //   sprintf (str_int, "%d", (task_id + 1));
// //   if ( fp_instr[task_id] == NULL ) {
// // #ifdef TAU_DEBUG_CUPTI_SASS
// //     printf("About to create file pointer for instr csv: %i\n", task_id);
// // #endif
// //     char str_instr[500];
// //     strcpy (str_instr, TauEnv_get_profiledir());
// //     strcat (str_instr,"/");
// //     strcat (str_instr,"sass_instr_");
// //     strcat (str_instr, str_int);
// //     strcat (str_instr, ".csv");
      
// //     fp_instr[task_id] = fopen(str_instr, "w");
// //     fprintf(fp_instr[task_id], "timestamp,correlationId,executed,flags,functionId,kind,\
// // notPredOffThreadsExecuted,pcOffset,sourceLocatorId,threadsExecuted\n");
// //     if (fp_instr[task_id] == NULL) {
// // #ifdef TAU_DEBUG_CUPTI_SASS
// //       printf("fp_instr[%i] failed\n", task_id);
// // #endif
// //     }
// //     else {
// // #ifdef TAU_DEBUG_CUPTI_SASS
// //       printf("fp_instr[%i] created successfully\n", task_id);
// // #endif
// //     }
// //   }
// //   else {
// // #ifdef TAU_DEBUG_CUPTI_SASS
// //     printf("fp_instr[%i] already exists!\n", task_id);
// // #endif
// //   }

//   FILE* fp_inst = createFileInstrSass(task_id);
//   for (std::map<uint32_t, InstrSampling>::iterator iter = instructionMap.begin(); 
//        iter != instructionMap.end(); 
//        iter++) {
//     InstrSampling is = iter->second;
//     // iterate map, clear
//     fprintf(fp_inst, "%f,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
// 	    is.timestamp_current,
// 	    is.correlationId, is.executed, 
// 	    sourceRecord->flags, is.functionId, 
// 	    sourceRecord->kind, sourceRecord->notPredOffThreadsExecuted,
// 	    is.pcOffset, is.sourceLocatorId, 
// 	    is.threadsExecuted);
//   }
//   fclose(fp_inst);
//   // source
  
//   // TAU stores time in microsec (1.0e-6), nanosec->microsec 1->0.001 ns/1000
//   // create file pointer here
//   FILE* fp_sour = createFileSourceSass(task_id - 1); // want 0,...,N-1, not 1,...,N!
  
//   fprintf(fp_sour, "%f,%d,%s,%d,%u\n",
// 	  d_currentTimestamp,source->id, source->fileName, source->lineNumber, source->kind);
  
  
//   // func
//   FILE* fp_funct = createFileFuncSass(task_id - 1);
//   for (std::map<uint32_t, FuncSampling>::iterator iter = functionMap.begin(); iter != functionMap.end(); iter++) {
//     fprintf(fp_funct, "%f;%u;%u;%u;%u;%u;%s;%s\n",
// 	    d_currentTimestamp,
// 	    fResult->contextId,
// 	    fResult->functionIndex,
// 	    fResult->id,
// 	    fResult->kind,
// 	    fResult->moduleId,
// 	    fResult->name, demangleName(fResult->name));
//   }  
//   // Each time imix counters recorded, erase instructionMap.
//   std::map<uint32_t, std::list<InstrSampling> >::iterator it_temp = instructionMap.find(fid);
//   instructionMap.erase(it_temp);
  
// }

void transport_imix_counters(uint32_t vec, Instrmix imixT, const char* name, uint32_t deviceId, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end, TauContextUserEvent * tc)
 {
   int taskId = 0;
#if defined(PTHREADS)
   if (map_cudaThread.find(id) != map_cudaThread.end()) {
     int local_vtid = map_cudaThread[id].tau_vtid;
     taskId = map_cuptiThread[local_vtid];
   }
#endif
   eventMap[taskId][tc] = vec;
   
   GpuEventAttributes *map;
   int map_size = eventMap[taskId].size();
   map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
   int i = 0;
   
   for (eventMap_t::iterator it = eventMap[taskId].begin(); it != eventMap[taskId].end(); it++) {
     map[i].userEvent = it->first;
     map[i].data = it->second;
     i++;
   }
   // transport
   Tau_cupti_register_gpu_event(name, deviceId,
				streamId, contextId, id, 0, false, map, map_size,
				end / 1e3, end / 1e3, taskId);
 }

void record_imix_counters(const char* name, uint32_t deviceId, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end) {
   // check if data available
  bool update = false;
  int taskId = 0;
#if (PTHREADS)
  if (map_cudaThread.find(id) != map_cudaThread.end()) {
    int local_vtid = map_cudaThread[id].tau_vtid;
    taskId = map_cuptiThread[local_vtid];
  }
#endif
  for (std::map<uint32_t, CUpti_ActivityFunction>::iterator iter = functionMap.begin(); 
       iter != functionMap.end(); 
       iter++) {
    CUpti_ActivityFunction fResult = iter->second;
    uint32_t fid = fResult.id;
    const char* name2 = demangleName(fResult.name);

    if (strcmp(name, name2) == 0) {
      // check if fid exists
      if (instructionMap.find(fid) == instructionMap.end()) {
	TAU_VERBOSE("[CuptiActivity] warning:  Instruction mix counters not recorded\n");
      }
      else {
	std::list<CUpti_ActivityInstructionExecution> instrSamp_list = instructionMap.find(fid)->second;

	ImixStats is_runtime = write_runtime_imix(fid, map_disassem, name);
#ifdef TAU_DEBUG_CUPTI
	cout << "[CuptiActivity]:  Name: " << name << 
	  ", FLOPS_raw: " << is_runtime.flops_raw << ", MEMOPS_raw: " << is_runtime.memops_raw <<
	  ", CTRLOPS_raw: " << is_runtime.ctrlops_raw << ", TOTOPS_raw: " << is_runtime.totops_raw << ".\n";
      #endif
	update = true;
	static TauContextUserEvent* fp_ops;
	static TauContextUserEvent* mem_ops;
	static TauContextUserEvent* ctrl_ops;
	
	Tau_get_context_userevent((void **) &fp_ops, "Floating Point Operations");
	Tau_get_context_userevent((void **) &mem_ops, "Memory Operations");
	Tau_get_context_userevent((void **) &ctrl_ops, "Control Operations");
	
	uint32_t  v_flops = is_runtime.flops_raw;
	uint32_t v_memops = is_runtime.memops_raw;
	uint32_t v_ctrlops = is_runtime.totops_raw;
	
	
	transport_imix_counters(v_flops, FlPtOps, name, deviceId, streamId, contextId, id, end, fp_ops);
	transport_imix_counters(v_memops, MemOps, name, deviceId, streamId, contextId, id, end, mem_ops);
	transport_imix_counters(v_ctrlops, CtrlOps, name, deviceId, streamId, contextId, id, end, ctrl_ops);
	
	// Each time imix counters recorded, erase instructionMap.
	std::map<uint32_t, std::list<CUpti_ActivityInstructionExecution> >::iterator it_temp = instructionMap.find(fid);
	instructionMap.erase(it_temp);
	eventMap[taskId].erase(eventMap[taskId].begin(), eventMap[taskId].end());
      }
      
    }
  }
  if(!update) {
    TAU_VERBOSE("TAU Warning:  Did not record instruction operations.\n");
  }

}

ImixStats write_runtime_imix(uint32_t functionId, std::map<std::pair<int, int>, CudaOps> map_disassem, std::string kernel)
{

#ifdef TAU_DEBUG_SASS
  cout << "[CudaSass]: write_runtime_imix begin\n";
#endif

  // look up from map_imix_static
  ImixStats imix_stats;
  string current_kernel = "";
  int flops_raw = 0;
  int ctrlops_raw = 0;
  int memops_raw = 0;
  int totops_raw = 0;
  double flops_pct = 0;
  double ctrlops_pct = 0;
  double memops_pct = 0;
  std::list<CUpti_ActivityInstructionExecution> instrSamp_list = instructionMap.find(functionId)->second;
  // check if entries exist
  if (!instrSamp_list.empty()) {
    // cout << "[CuptiActivity]:  instrSamp_list not empty\n";
    for (std::list<CUpti_ActivityInstructionExecution>::iterator iter=instrSamp_list.begin();
	 iter != instrSamp_list.end(); 
	 iter++) {
      CUpti_ActivityInstructionExecution is = *iter;
      
      // TODO:  Get line info here...
      int sid = is.sourceLocatorId;
      // cout << "[CuptiActivity]:  is.sourceLocatorId: " << is.sourceLocatorId << endl;
      int lineno = -1;
      if ( srcLocMap.find(sid) != srcLocMap.end() ) {
	lineno = srcLocMap.find(sid)->second.lineNumber;
	// cout << "[CuptiActivity]:  lineno: " << lineno << endl;
	std::pair<int, int> p1 = std::make_pair(lineno, (unsigned int) is.pcOffset);

	for (std::map<std::pair<int, int>,CudaOps>::iterator iter= map_disassem.begin();
	     iter != map_disassem.end(); iter++) { 
	  CudaOps cuops = iter->second;
	  // cout << "cuops pair(" << cuops.lineno << ", " << cuops.pcoffset << ")\n";
	  if (map_disassem.find(p1) != map_disassem.end()) {
	    CudaOps cuops = map_disassem.find(p1)->second;
	    // cout << "[CuptiActivity]:  cuops.instruction: " << cuops.instruction << endl;
	    // map to disassem
	    int instr_type = get_instruction_mix_category(cuops.instruction);
	    switch(instr_type) {
	      // Might be non-existing ops, don't count those!
	      case FloatingPoint: case Integer:
	      case SIMD: case Conversion: {
		flops_raw++;
		totops_raw++;
		break;
	      }
	      case LoadStore: case Texture:
	      case Surface: {
		memops_raw++;
		totops_raw++;
		break;
	      }
	      case Control: case Move:
	      case Predicate: {
		ctrlops_raw++;
		totops_raw++;
		break;
	      }
	      case Misc: {
		totops_raw++;
		break;
	      }
	    }
	  }
	  else {
#if TAU_DEBUG_DISASM
	    cout << "[CuptiActivity]:  map_disassem does not exist for pair(" 
	    	 << lineno << "," << is->pcOffset << ")\n";
#endif
	  }
	}
      }
      else {
#if TAU_DEBUG_DISASM
	cout << "[CuptiActivity]:  srcLocMap does not exist for sid: " << sid << endl;
#endif
      }
    }
  }
  else {
    cout << "[CuptiActivity]: instrSamp_list empty!\n";
  }
  
  string kernel_iter = kernel;

  flops_pct = ((float)flops_raw/totops_raw) * 100;
  memops_pct = ((float)memops_raw/totops_raw) * 100;
  ctrlops_pct = ((float)ctrlops_raw/totops_raw) * 100;
  // push onto map
  imix_stats.flops_raw = flops_raw;
  imix_stats.ctrlops_raw = ctrlops_raw;
  imix_stats.memops_raw = memops_raw;
  imix_stats.totops_raw = totops_raw;
  imix_stats.flops_pct = flops_pct;
  imix_stats.ctrlops_pct = ctrlops_pct;
  imix_stats.memops_pct = memops_pct;
  imix_stats.kernel = kernel_iter;

#ifdef TAU_DEBUG_DISASM
  cout << "[CudaDisassembly]:  current_kernel: " << kernel_iter << endl;
  cout << "  FLOPS: " << flops_raw << ", MEMOPS: " << memops_raw 
       << ", CTRLOPS: " << ctrlops_raw << ", TOTOPS: " << totops_raw << "\n";
  cout << setprecision(2) << "  FLOPS_pct: " << flops_pct << "%, MEMOPS_pct: " 
       << memops_pct << "%, CTRLOPS_pct: " << ctrlops_pct << "%\n";
#endif

  return imix_stats;
}


//  void record_imix_counters(const char* name, uint32_t deviceId, uint32_t streamId, uint32_t contextId, uint32_t id, uint64_t end) {
//    // check if data available
//   bool update = false;

//   for (std::map<uint32_t, FuncSampling>::iterator iter = functionMap.begin(); iter != functionMap.end(); iter++) {
//     uint32_t fid = iter->second.fid;
//     const char* name2 = demangleName(iter->second.name);

//     if (strcmp(name, name2) == 0) {
//       // check if fid exists
//       if (instructionMap.find(fid) == instructionMap.end()) {
// 	TAU_VERBOSE("[CuptiActivity] warning:  Instruction mix counters not recorded\n");
//       }
//       else {
// 	std::list<InstrSampling> instrSamp_list = instructionMap.find(fid)->second;

// 	ImixStats is_runtime = write_runtime_imix(fid, instrSamp_list, map_disassem, srcLocMap, name);
// #ifdef TAU_DEBUG_CUPTI
// 	cout << "[CuptiActivity]:  Name: " << name << 
// 	  ", FLOPS_raw: " << is_runtime.flops_raw << ", MEMOPS_raw: " << is_runtime.memops_raw <<
// 	  ", CTRLOPS_raw: " << is_runtime.ctrlops_raw << ", TOTOPS_raw: " << is_runtime.totops_raw << ".\n";
//       #endif
// 	update = true;
// 	static TauContextUserEvent* fp_ops;
// 	static TauContextUserEvent* mem_ops;
// 	static TauContextUserEvent* ctrl_ops;
	
// 	Tau_get_context_userevent((void **) &fp_ops, "Floating Point Operations");
// 	Tau_get_context_userevent((void **) &mem_ops, "Memory Operations");
// 	Tau_get_context_userevent((void **) &ctrl_ops, "Control Operations");
	
// 	uint32_t  v_flops = is_runtime.flops_raw;
// 	uint32_t v_memops = is_runtime.memops_raw;
// 	uint32_t v_ctrlops = is_runtime.totops_raw;
	
	
// 	transport_imix_counters(v_flops, FlPtOps, name, deviceId, streamId, contextId, id, end, fp_ops);
// 	transport_imix_counters(v_memops, MemOps, name, deviceId, streamId, contextId, id, end, mem_ops);
// 	transport_imix_counters(v_ctrlops, CtrlOps, name, deviceId, streamId, contextId, id, end, ctrl_ops);
	
// 	// Each time imix counters recorded, erase instructionMap.
// 	std::map<uint32_t, std::list<InstrSampling> >::iterator it_temp = instructionMap.find(fid);
// 	instructionMap.erase(it_temp);
// 	eventMap.erase(eventMap.begin(), eventMap.end());
//       }
      
//     }
//   }
//   if(!update) {
//     TAU_VERBOSE("TAU Warning:  Did not record instruction operations.\n");
//   }

// }

  
void record_gpu_launch(int correlationId, const char *name)
{
#ifdef TAU_DEBUG_CUPTI
  printf("TAU: CUPTI recording GPU launch: %s\n", name);
#endif
  Tau_cupti_register_host_calling_site(correlationId, name);	
}

void record_gpu_counters(int device_id, const char *name, uint32_t correlationId, eventMap_t *m)
{
  int taskId = 0;
#if defined(PTHREADS)
  if (map_cudaThread.find(correlationId) != map_cudaThread.end()) {
    int local_vtid = map_cudaThread[correlationId].tau_vtid;
    taskId = map_cuptiThread[local_vtid];
  }
#endif
  if (Tau_CuptiLayer_get_num_events() > 0 &&
      !counters_bounded_warning_issued[device_id] && 
      last_recorded_kernel_name != NULL && 
      strcmp(last_recorded_kernel_name, name) != 0) 
  {
    TAU_VERBOSE("TAU Warning: CUPTI events will be bounded, multiple different kernel deteched between synchronization points.\n");
    counters_bounded_warning_issued[device_id] = true;
    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++) {
      Tau_CuptiLayer_set_event_name(n, TAU_CUPTI_COUNTER_BOUNDED); 
    }
  }
  last_recorded_kernel_name = name;
  {
    //increment kernel count.
    
    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++) {
#ifdef TAU_DEBUG_CUPTI_COUNTERS
      std::cout << "at record: "<< name << " ====> " << std::endl;
      std::cout << "\tstart: " << counters_at_last_launch[device_id][n] << std::endl;
      std::cout << "\t stop: " << current_counters[device_id][n] << std::endl;
#endif
      TauContextUserEvent* c;
      const char *name = Tau_CuptiLayer_get_event_name(n);
      if (n >= counterEvents[device_id].size()) {
        c = (TauContextUserEvent *) Tau_return_context_userevent(name);
        counterEvents[device_id].push_back(c);
      } else {
        c = counterEvents[device_id][n];
      }
      Tau_set_context_event_name(c, name);
      if (counters_averaged_warning_issued[device_id] == true)
      {
        eventMap[taskId][c] = (current_counters[device_id][n] - counters_at_last_launch[device_id][n]);
      }
      else {
        eventMap[taskId][c] = (current_counters[device_id][n] - counters_at_last_launch[device_id][n]) * kernels_encountered[device_id];
      }
      
    }
    kernels_recorded[device_id]++;
  }
}
void record_gpu_occupancy(int32_t blockX, 
                          int32_t blockY,
                          int32_t blockZ,
			                    uint16_t registersPerThread,
		                      int32_t staticSharedMemory,
                          uint32_t deviceId,
                          const char *name, 
                          eventMap_t *eventMap)
{
	CUpti_ActivityDevice device = __deviceMap()[deviceId];


	int myWarpsPerBlock = (int)ceil(
				(double)(blockX * blockY * blockZ)/
				(double)(device.numThreadsPerWarp)
			); 

	int allocatable_warps = min(
		(int)device.maxBlocksPerMultiprocessor, 
		(int)floor(
			(float) device.maxWarpsPerMultiprocessor/
			myWarpsPerBlock	
		)
	);

	static TauContextUserEvent* alW;
	Tau_get_context_userevent((void **) &alW, "Allocatable Blocks per SM given Thread count (Blocks)");
	(*eventMap)[alW] = allocatable_warps;
  //eventMap[5].userEvent = alW;
	//eventMap[5].data = allocatable_warps;

	int myRegistersPerBlock = device.computeCapabilityMajor < 2 ?
		ceil(
			ceil(
				(float)myWarpsPerBlock, 2	
			)*
			registersPerThread*
			device.numThreadsPerWarp,
			device.computeCapabilityMinor < 2 ? 256 : 512
		) :
		ceil(
			registersPerThread*
			device.numThreadsPerWarp,
			device.computeCapabilityMajor < 3 ? 128 : 256
		)*
		ceil(
			myWarpsPerBlock, device.computeCapabilityMajor < 3 ? 2 : 4
		);

	int allocatable_registers = (int)floor(
		device.maxRegistersPerBlock/
		max(
			myRegistersPerBlock, 1
			)
		);
	
	if (allocatable_registers == 0)
		allocatable_registers = device.maxBlocksPerMultiprocessor;
	

	static TauContextUserEvent* alR;
	Tau_get_context_userevent((void **) &alR, "Allocatable Blocks Per SM given Registers used (Blocks)");
  (*eventMap)[alR] = allocatable_registers;

	int sharedMemoryUnit;
	switch(device.computeCapabilityMajor)
	{
		case 1: sharedMemoryUnit = 512; break;
		case 2: sharedMemoryUnit = 128; break;
	case 3: case 5: case 6: case 7: sharedMemoryUnit = 256; break;
	}
	int mySharedMemoryPerBlock = ceil(
		staticSharedMemory,
		sharedMemoryUnit
	);

	int allocatable_shared_memory = mySharedMemoryPerBlock > 0 ?
		floor(
			device.maxSharedMemoryPerBlock/
			mySharedMemoryPerBlock
		) :
		device.maxThreadsPerBlock
		;
	
	static TauContextUserEvent* alS;
	Tau_get_context_userevent((void **) &alS, "Allocatable Blocks Per SM given Shared Memory usage (Blocks)");
  (*eventMap)[alS] = allocatable_shared_memory;

	int allocatable_blocks = min(allocatable_warps, min(allocatable_registers, allocatable_shared_memory));

	int occupancy = myWarpsPerBlock * allocatable_blocks;

// #define RESULTS_TO_STDOUT 1
#ifdef RESULTS_TO_STDOUT
	printf("[%s] occupancy calculator:\n", name);

	printf("myWarpsPerBlock            = %d.\n", myWarpsPerBlock);
	printf("allocatable warps          = %d.\n", allocatable_warps);
	printf("myRegistersPerBlock        = %d.\n", myRegistersPerBlock);
	printf("allocatable registers      = %d.\n", allocatable_registers);
	printf("mySharedMemoryPerBlock     = %d.\n", mySharedMemoryPerBlock);
	printf("allocatable shared memory  = %d.\n", allocatable_shared_memory);

	printf("              >> occupancy = %d (%2.0f%% of %d).\n", 
		occupancy, ((float)occupancy/device.maxWarpsPerMultiprocessor)*100, device.maxWarpsPerMultiprocessor);
#endif

	static TauContextUserEvent* occ;
	Tau_get_context_userevent((void **) &occ, "GPU Occupancy (Warps)");
  (*eventMap)[occ] = occupancy;

}

#if CUPTI_API_VERSION >= 3
void form_context_event_name(CUpti_ActivityKernel *kernel, CUpti_ActivitySourceLocator *source, const char *event_name, std::string *name)
{         

  stringstream file_and_line("");
  file_and_line << event_name << " : ";
  file_and_line << demangleName(kernel->name);
  if (source->kind != CUPTI_ACTIVITY_KIND_INVALID)
  {
    file_and_line << " => [{" << source->fileName   << "}";
    file_and_line <<  " {" << source->lineNumber << "}]";
  }

   *name = file_and_line.str();

  //cout << "file and line: " << file_and_line.str() << endl;

}
#endif // CUPTI_API_VERSION >= 3


bool function_is_sync(CUpti_CallbackId id)
{
	return (	
		//unstable results otherwise(
		//runtimeAPI
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3021 ||
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020 ||
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020 || 
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020 ||
		id == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
		id == CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020 ||
		//id == CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020 ||
		//driverAPI
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpy_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2 ||
		id == CUPTI_DRIVER_TRACE_CBID_cuEventSynchronize //||
		//id == CUPTI_DRIVER_TRACE_CBID_cuEventQuery

				 );
}
bool function_is_exit(CUpti_CallbackId id)
{
	
	return (
		id == CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020 || 
		id == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020
		//driverAPI
				 );
	
}
bool function_is_launch(CUpti_CallbackId id) { 
	return id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020
		     || id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel
#if CUDA_VERSION >= 7000
             || id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
             || id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000 
             || id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000
#endif
             ;

}

bool function_is_memcpy(CUpti_CallbackId id, CUpti_CallbackDomain domain) {
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
	return (
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArray_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArray_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyArrayToArray_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToArrayAsync_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromArrayAsync_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbolAsync_v3020 ||
		id ==     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbolAsync_v3020
	);
	}
	else if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
	{
		return (
		id ==     CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
		id ==     CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
    id ==     CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 ||
    id ==     CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2
		);
	}
	else
	{
		return false;
	}
}

void get_values_from_memcpy(const CUpti_CallbackData *info, CUpti_CallbackId id, CUpti_CallbackDomain domain, int &kind, int &count)
{
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyToArray, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyFromArray, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyArrayToArray, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyToSymbol, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyFromSymbol, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyToArrayAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyFromArrayAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyToSymbolAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpyFromSymbolAsync, id, info, kind, count)
	}
	//driver API
	else
	{
		if (id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2)
		{
			kind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
			count = ((cuMemcpyHtoD_v2_params *) info->functionParams)->ByteCount;
		}
    else if (id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2)
		{
			kind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
			count = ((cuMemcpyHtoDAsync_v2_params *) info->functionParams)->ByteCount;
		}
		else if (id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2)
		{
			kind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH;
			count = ((cuMemcpyDtoH_v2_params *) info->functionParams)->ByteCount;
		}
    else if (id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2)
		{
			kind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH;
			count = ((cuMemcpyDtoHAsync_v2_params *) info->functionParams)->ByteCount;
		}
		else
		{
			//cannot find byte count
			kind = -1;
			count = 0;
		}

	}
}
int getMemcpyType(int kind)
{
	switch(kind)
	{
		case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
			return MemcpyHtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
			return MemcpyDtoH;
		/*
		case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
			return MemcpyHtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
			return MemcpyDtoH;
		case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
			return MemcpyDtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
			return MemcpyDtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
			return MemcpyDtoD;
		*/
		case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
			return MemcpyDtoD;
		default:
			return MemcpyUnknown;
	}
}

#if CUDA_VERSION >= 6000
int getUnifmemType(int kind)
{
  switch(kind)
    {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
      return BytesHtoD;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
      return BytesDtoH;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
      return CPUPageFault;
    default:
      return UnifmemUnknown;
    }
}
static const char *
getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
{
    switch (kind) 
    {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
        return "BYTES_TRANSFER_HTOD";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
        return "BYTES_TRANSFER_DTOH";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
        return "CPU_PAGE_FAULT_COUNT";
    default:
        break;
    }
    return "<unknown>";
}

static const char *
getUvmCounterScopeString(CUpti_ActivityUnifiedMemoryCounterScope scope)
{
    switch (scope) 
    {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE:
        return "PROCESS_SINGLE_DEVICE";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES:
        return "PROCESS_ALL_DEVICES";
    default:
        break;
    }
    return "<unknown>";
}
#endif

const char *demangleName(const char* name)
{
	const char *dem_name = 0;
	//printf("demangling: %s.\n", name);
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
	//printf("demangling name....\n");
	dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE |
	DMGL_TYPES);
	//check to see if demangling failed (name was not mangled).
	if (dem_name == NULL)
	{
		dem_name = name;
	}
#else
	dem_name = name;
#endif /* HAVE_GPU_DEMANGLE */
	//printf("demanged: %s.\n", dem_name);
	return dem_name;
}


bool cupti_api_runtime()
{
	return (0 == strcasecmp(TauEnv_get_cupti_api(), "runtime") || 
			0 == strcasecmp(TauEnv_get_cupti_api(), "both"));
}
bool cupti_api_driver()
{
	return (0 == strcasecmp(TauEnv_get_cupti_api(), "driver") || 
			0 == strcasecmp(TauEnv_get_cupti_api(), "both")); 
}

int get_device_count()
{
#if CUDA_VERSION >= 5000
  int device_count;
  cuDeviceGetCount(&device_count);
  return device_count;
#else
  return 1;
#endif

}
/*  BEGIN:  SASS added  */
int get_device_id() 
{
  int deviceId;
  cudaGetDevice(&deviceId);
  return deviceId;
}

FILE* createFileSourceSass(int task_id) 
{
  char str_int[5];
  sprintf (str_int, "%d", (task_id));
  if ( fp_source[task_id] == NULL ) {
#ifdef TAU_DEBUG_CUPTI_SASS
    printf("About to create file pointer for source csv: %i\n", task_id);
#endif
    char str_source[500];
    strcpy (str_source,TauEnv_get_profiledir());
    strcat (str_source,"/");
    strcat (str_source,"sass_source_");
    strcat (str_source, str_int);
    strcat (str_source, ".csv");
      
    fp_source[task_id] = fopen(str_source, "a");
    fprintf(fp_source[task_id], "timestamp,id,fileName,lineNumber,kind\n");
    if (fp_source[task_id] == NULL) {
#ifdef TAU_DEBUG_CUPTI_SASS
      printf("fp_source[%i] failed\n", task_id);
#endif
    }
    else {
#ifdef TAU_DEBUG_CUPTI_SASS
      printf("fp_source[%i] created successfully\n", task_id);
#endif
    }
  }
  else {
#ifdef TAU_DEBUG_CUPTI_SASS
    printf("fp_source[%i] already exists!\n", task_id);
#endif
  }
  return fp_source[task_id];
}

FILE* createFileInstrSass(int task_id) 
{
  char str_int[5];
  sprintf (str_int, "%d", (task_id));
  if ( fp_instr[task_id] == NULL ) {
#ifdef TAU_DEBUG_CUPTI_SASS
    printf("About to create file pointer for instr csv: %i\n", task_id);
#endif
    char str_instr[500];
    strcpy (str_instr, TauEnv_get_profiledir());
    strcat (str_instr,"/");
    strcat (str_instr,"sass_instr_");
    strcat (str_instr, str_int);
    strcat (str_instr, ".csv");
      
    fp_instr[task_id] = fopen(str_instr, "a");
    fprintf(fp_instr[task_id], "timestamp,correlationId,executed,flags,functionId,kind,\
notPredOffThreadsExecuted,pcOffset,sourceLocatorId,threadsExecuted\n");
    if (fp_instr[task_id] == NULL) {
#ifdef TAU_DEBUG_CUPTI_SASS
      printf("fp_instr[%i] failed\n", task_id);
#endif
    }
    else {
#ifdef TAU_DEBUG_CUPTI_SASS
      printf("fp_instr[%i] created successfully\n", task_id);
#endif
    }
  }
  else {
#ifdef TAU_DEBUG_CUPTI_SASS
    printf("fp_instr[%i] already exists!\n", task_id);
#endif
  }
  return fp_instr[task_id];
}

FILE* createFileFuncSass(int task_id) 
{
  char str_int[5];
  sprintf (str_int, "%d", (task_id));
  if ( fp_func[task_id] == NULL ) {
#ifdef TAU_DEBUG_CUPTI_SASS
    printf("About to create file pointer for func csv: %i\n", task_id);
#endif
    char str_func[500];
    strcpy (str_func, TauEnv_get_profiledir());
    strcat (str_func,"/");
    strcat (str_func,"sass_func_");
    strcat (str_func, str_int);
    strcat (str_func, ".csv");
      
    fp_func[task_id] = fopen(str_func, "a");
    fprintf(fp_func[task_id], "timestamp;contextId;functionIndex;id;kind;moduleId;name;demangled\n");
    if (fp_func[task_id] == NULL) {
#ifdef TAU_DEBUG_CUPTI_SASS
      printf("fp_func[%i] failed\n", task_id);
#endif
    }
    else {
#ifdef TAU_DEBUG_CUPTI_SASS
      printf("fp_func[%i] created successfully\n", task_id);
#endif
    }
  }
  else {
#ifdef TAU_DEBUG_CUPTI_SASS
    printf("fp_func[%i] already exists!\n", task_id);
#endif
  }
  return fp_func[task_id];
}

// void createFilePointerSass(int device_count) 
// {
// #ifdef TAU_DEBUG_CUPTI_SASS
//   printf ("Inside sass/csv, about to create fp\n");
//   printf("device_count: %i\n", device_count);
// #endif
//   if (device_count < 0) {
//     printf("Couldn't detect device inside fp creation, FAIL\n");
//   }

//   for (int i = 0; i < device_count; i++) {
//     char str_int[5];
//     sprintf (str_int, "%d", (i+1));
//     if ( fp_source[i] == NULL ) {
// #ifdef TAU_DEBUG_CUPTI_SASS
//       printf("About to create file pointer csv: %i\n", i);
// #endif
//       char str_source[500];
//       strcpy (str_source,TauEnv_get_profiledir());
//       strcat (str_source,"/");
//       strcat (str_source,"sass_source_");
//       strcat (str_source, str_int);
//       strcat (str_source, ".csv");
      
//       fp_source[i] = fopen(str_source, "w");
//       fprintf(fp_source[i], "timestamp,id,fileName,lineNumber,kind\n");
//       if (fp_source[i] == NULL) {
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	printf("fp_source[%i] failed\n", i);
// #endif
//       }
//       else {
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	printf("fp_source[%i] created successfully\n", i);
// #endif
//       }
//     }
//     else {
// #ifdef TAU_DEBUG_CUPTI_SASS
//       printf("fp_source[%i] already exists!\n", i);
// #endif
//     }
//     if (fp_instr[i] == NULL) {

//       char str_instr[500];
//       strcpy (str_instr,TauEnv_get_profiledir());
//       strcat (str_instr,"/");
//       strcat (str_instr,"sass_instr_");
//       strcat (str_instr, str_int);
//       strcat (str_instr, ".csv");
      
//       fp_instr[i] = fopen(str_instr, "w");
//       fprintf(fp_instr[i], "timestamp,correlationId,executed,flags,functionId,kind,\
// notPredOffThreadsExecuted,pcOffset,sourceLocatorId,threadsExecuted\n");
//       if (fp_instr[i] == NULL) {
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	printf("fp_instr[%i] failed\n", i);
// #endif
//       }
//       else {
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	printf("fp_instr[%i] created successfully\n", i);
// #endif
//       }
//     }
//     else {
// #ifdef TAU_DEBUG_CUPTI_SASS
//       printf("fp_instr[%i] already exists!\n", i);
// #endif
//     }
//     if(fp_func[i] == NULL) {
//       char str_func[500];
//       strcpy (str_func,TauEnv_get_profiledir());
//       strcat (str_func,"/");
//       strcat (str_func,"sass_func_");
//       strcat (str_func, str_int);
//       strcat (str_func, ".csv");
      
//       fp_func[i] = fopen(str_func, "w");
//       fprintf(fp_func[i], "timestamp;contextId;functionIndex;id;kind;moduleId;name;demangled\n");
//       if (fp_func[i] == NULL) {
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	printf("fp_func[%i] failed\n", i);
// #endif
//       }
//       else {
// #ifdef TAU_DEBUG_CUPTI_SASS
// 	printf("fp_func[%i] created successfully\n", i);
// #endif
//       }
//     }
//     else {
// #ifdef TAU_DEBUG_CUPTI_SASS
//       printf("fp_func[%i] already exists!\n", i);
// #endif
//     }

//   } // deviceCount
// }

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
  std::cout << "at launch (" << device << ") ====> " << std::endl;
    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++) {
      std::cout << "\tlast launch:      " << counters_at_last_launch[device][n] << std::endl;
      std::cout << "\tcurrent counters: " << current_counters[device][n] << std::endl;
    }
#endif
}
  
void record_gpu_counters_at_sync(int device)
{
  if (kernels_encountered[device] == 0) {
   return;
  }
  Tau_CuptiLayer_read_counters(device, current_counters[device]);
#ifdef TAU_CUPTI_DEBUG_COUNTERS
  std::cout << "at sync (" << device << ") ====> " << std::endl;
    for (int n = 0; n < Tau_CuptiLayer_get_num_events(); n++) {
      std::cout << "\tlast launch:      " << counters_at_last_launch[device][n] << std::endl;
      std::cout << "\tcurrent counters: " << current_counters[device][n] << std::endl;
    }
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
/*  END:  SASS added  */

// #if CUDA_VERSION >= 6000
// static const char *
// getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
// {
//     switch (kind) 
//     {
//     case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
//         return "BYTES_TRANSFER_HTOD";
//     case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
//         return "BYTES_TRANSFER_DTOH";
//     case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
//         return "CPU_PAGE_FAULT_COUNT";
//     default:
//         break;
//     }
//     return "<unknown>";
// }

#endif //CUPTI API VERSION >= 2
