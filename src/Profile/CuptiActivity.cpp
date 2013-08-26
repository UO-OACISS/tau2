#include <Profile/CuptiActivity.h>
#include <iostream>
using namespace std;

#if CUPTI_API_VERSION >= 2
#include <dlfcn.h>

const char * tau_orig_libname = "libcuda.so";
static void *tau_handle = NULL;

static int subscribed = 0;

CUresult cuInit(unsigned int a1) {

  typedef CUresult (*cuInit_p_h) (unsigned int);
  static cuInit_p_h cuInit_h = NULL;
  CUresult retval;
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuInit_h == NULL)
	cuInit_h = (cuInit_p_h) dlsym(tau_handle,"cuInit"); 
    if (cuInit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
	Tau_cupti_subscribe();
	subscribed = 1;
  retval  =  (*cuInit_h)( a1);
  }
  return retval;
}

void Tau_cupti_subscribe()
{
	//cerr << "in subscribe." << endl;
	CUptiResult err;
	TAU_VERBOSE("TAU: Subcribing to CUPTI.\n");
	err = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)Tau_cupti_callback_dispatch, NULL);
	//to collect device info 
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE);
	
	//setup global activity queue.
	activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
	err = cuptiActivityEnqueueBuffer(NULL, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);

}
void Tau_cupti_onload()
{
	if (!subscribed) {
		Tau_cupti_subscribe();
	}
	TAU_VERBOSE("TAU: Enabling CUPTI callbacks.\n");

	CUptiResult err;
  
	if (cupti_api_runtime())
	{
		//printf("TAU: Subscribing to RUNTIME API.\n");
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
		//runtime_enabled = true;
	}
	if (cupti_api_driver())
	{
		//printf("TAU: Subscribing to DRIVER API.\n");
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
		//driver_enabled = true;
	}
  
	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE); 
	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE); 

	CUDA_CHECK_ERROR(err, "Cannot set Domain, check if the CUDA toolkit version is supported by the install CUDA driver.\n");
	

 	
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT);
	
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
#if CUDA_VERSION >= 5050
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2);
#endif
#if CUDA_VERSION >= 5000
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
#else
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
#endif

#if CUPTI_API_VERSION >= 3
  if (strcasecmp(TauEnv_get_cuda_instructions(), "GLOBAL_ACCESS") == 0)
  {
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS);
  } else if (strcasecmp(TauEnv_get_cuda_instructions(), "BRANCH") == 0)
  {
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR);
	  err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_BRANCH);
  }
#else
  if (strcasecmp(TauEnv_get_cuda_instructions(), "GLOBAL_ACCESS") == 0 ||
      strcasecmp(TauEnv_get_cuda_instructions(), "BRANCH") == 0)
  {
		printf("TAU WARNING: DISABLING CUDA %s tracking. Please use CUDA 5.0 or greater.\n", TauEnv_get_cuda_instructions());
  }
#endif //CUPTI_API_VERSIOn >= 3
	
  CUDA_CHECK_ERROR(err, "Cannot enqueue buffer.\n");

  uint64_t timestamp;
  err = cuptiGetTimestamp(&timestamp);
	CUDA_CHECK_ERROR(err, "Cannot get timestamp.\n");
  Tau_cupti_set_offset(TauTraceGetTimeStamp() - ((double)timestamp / 1e3));
  //Tau_cupti_set_offset((-1) * timestamp / 1e3);
	//cerr << "begining timestamp: " << TauTraceGetTimeStamp() - ((double)timestamp/1e3) << "ms.\n" << endl;
  //Tau_cupti_set_offset(0);

  Tau_gpu_init();
}

void Tau_cupti_onunload() {}

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
	//Just in case we encounter a callback before TAU is intialized or finished.
  if (!Tau_init_check_initialized() || Tau_global_getLightsOut()) { return; }

	if (domain == CUPTI_CB_DOMAIN_RESOURCE)
	{
		//A resource was created, let us enqueue a buffer in order to capture events
		//that happen on that resource.
		if (id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
		{
			CUptiResult err;
			CUpti_ResourceData* resource = (CUpti_ResourceData*) params;
			//printf("Enqueuing Buffer with context=%p stream=%d.\n", resource->context, 0);
			activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
			err = cuptiActivityEnqueueBuffer(resource->context, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);
			CUDA_CHECK_ERROR(err, "Cannot enqueue buffer in context.\n");
		}
		else if (id == CUPTI_CBID_RESOURCE_STREAM_CREATED)
		{
			CUptiResult err;
			CUpti_ResourceData* resource = (CUpti_ResourceData*) params;
    	uint32_t stream;
			err = cuptiGetStreamId(resource->context, resource->resourceHandle.stream, &stream);
			//printf("Enqueuing Buffer with context=%p stream=%d.\n", resource->context, stream);
			activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
			err = cuptiActivityEnqueueBuffer(resource->context, stream, activityBuffer, ACTIVITY_BUFFER_SIZE);
			CUDA_CHECK_ERROR(err, "Cannot enqueue buffer in stream.\n");
			streamIds.push_back(stream);
			number_of_streams++;
		}

	}
	else if (domain == CUPTI_CB_DOMAIN_SYNCHRONIZE)
	{
		//printf("register sync from callback.\n");
		CUpti_SynchronizeData *sync = (CUpti_SynchronizeData *) params;
		uint32_t stream;
		CUptiResult err;
		//Global Buffer
    int device_count = get_device_count();
    for (int i=0; i<device_count; i++) {
      record_gpu_counters_at_sync(i);
    }
		Tau_cupti_register_sync_event(NULL, 0);
    
		err = cuptiGetStreamId(sync->context, sync->stream, &stream);
		Tau_cupti_register_sync_event(sync->context, stream);
		for (int s=0; s<number_of_streams; s++)
		{
			Tau_cupti_register_sync_event(sync->context, streamIds.at(s));
		}
	}
	else if (domain == CUPTI_CB_DOMAIN_DRIVER_API ||
					 domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
		const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *) params;
		if (function_is_memcpy(id, domain))
		{
			int kind;
			int count;
			get_values_from_memcpy(cbInfo, id, domain, kind, count);
			if (cbInfo->callbackSite == CUPTI_API_ENTER)
			{
				Tau_cupti_enter_memcpy_event(
					cbInfo->functionName, -1, 0, cbInfo->contextUid, cbInfo->correlationId, 
					count, getMemcpyType(kind)
				);
				Tau_cupti_register_host_calling_site(cbInfo->correlationId);
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
				//cerr << "callback for " << cbInfo->functionName << ", exit." << endl;
				Tau_cupti_exit_memcpy_event(
					cbInfo->functionName, -1, 0, cbInfo->contextUid, cbInfo->correlationId, 
					count, getMemcpyType(kind)
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
          
					//cerr << "sync function name: " << cbInfo->functionName << endl;
					//Disable counter tracking during the sync.
					//Tau_CuptiLayer_disable();
					//cuCtxSynchronize();
					cudaDeviceSynchronize();
					//Tau_CuptiLayer_enable();
          int device_count = get_device_count();
          for (int i=0; i<device_count; i++) {
            record_gpu_counters_at_sync(i);
          }
					Tau_cupti_register_sync_event(cbInfo->context, 0);
          
				}
			}
		}
		else
		{
			if (cbInfo->callbackSite == CUPTI_API_ENTER)
			{
				if (function_is_exit(id))
				{
					//Stop collecting cupti counters.
					Tau_CuptiLayer_finalize();
				}
				Tau_gpu_enter_event(cbInfo->functionName);
				if (function_is_launch(id))
				{
          Tau_CuptiLayer_init();

          //printf("[at call (enter), %d] name: %s.\n", cbInfo->correlationId, cbInfo->functionName);
				  record_gpu_launch(cbInfo->correlationId);
				}
				//cerr << "callback for " << cbInfo->functionName << ", enter." << endl;
			}
			else if (cbInfo->callbackSite == CUPTI_API_EXIT)
			{
				if (function_is_launch(id))
				{
				  record_gpu_launch(cbInfo->correlationId);
				}
      /* for testing only. 
				if (function_is_launch(id))
				{
          printf("synthetic sync point.\n");
          cuCtxSynchronize();
					FunctionInfo *p = TauInternal_CurrentProfiler(Tau_RtsLayer_getTid())->CallPathFunction;
        }
      */
				//cerr << "callback for " << cbInfo->functionName << ", exit." << endl;
        //printf("[at call (exit), %d] name: %s.\n", cbInfo->correlationId, cbInfo->functionName);
				Tau_gpu_exit_event(cbInfo->functionName);
				if (function_is_sync(id))
				{
					//cerr << "sync function name: " << cbInfo->functionName << endl;
					//Tau_CuptiLayer_disable();
					cuCtxSynchronize();
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

void Tau_cupti_register_sync_event(CUcontext context, uint32_t stream)
{
	//printf("in sync: context=%p stream=%d.\n", context, stream);
	registered_sync = true;
  CUptiResult err, status;
  CUpti_Activity *record = NULL;
	size_t bufferSize = 0;
  
  int device_count = get_device_count();
  //start
  if (device_count > TAU_MAX_GPU_DEVICES) {
    printf("TAU ERROR: Maximum number of devices (%d) exceeded. Please reconfigure TAU with -useropt=-DTAU_MAX_GPU_DEVICES=32 or some higher number\n", TAU_MAX_GPU_DEVICES);
    exit(1);
  }

	err = cuptiActivityDequeueBuffer(context, stream, &activityBuffer, &bufferSize);
	//printf("err: %d.\n", err);

	if (err == CUPTI_SUCCESS)
	{
		//printf("succesfully dequeue'd buffer.\n");
		do {
			status = cuptiActivityGetNextRecord(activityBuffer, bufferSize, &record);
			if (status == CUPTI_SUCCESS) {
				Tau_cupti_record_activity(record);
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
			
		size_t number_dropped;
		err = cuptiActivityGetNumDroppedRecords(NULL, 0, &number_dropped);

		if (number_dropped > 0)
			printf("TAU WARNING: %d CUDA records dropped, consider increasing the CUPTI_BUFFER size.", number_dropped);

		//Need to requeue buffer by context, stream.
		err = cuptiActivityEnqueueBuffer(context, stream, activityBuffer, ACTIVITY_BUFFER_SIZE);
		CUDA_CHECK_ERROR(err, "Cannot requeue buffer.\n");
   
    for (int i=0; i < device_count; i++) {
      //printf("Kernels encountered/recorded: %d/%d.\n", CurrentGpuState[i].kernels_encountered, CurrentGpuState[0].kernels_recorded);
      if (kernels_recorded[i] == kernels_encountered[i])
      {
        clear_counters(i);
        last_recorded_kernel_name = NULL;
      } else if (kernels_recorded[i] > kernels_encountered[i]) {
        printf("TAU: Recorded more kernels than were launched, exiting.\n");
        abort();
        exit(1);
      }
    }
  } else if (err != CUPTI_ERROR_QUEUE_EMPTY) {
		//printf("TAU: Activity queue is empty.\n");
		//CUDA_CHECK_ERROR(err, "Cannot dequeue buffer.\n");
	} else if (err != CUPTI_ERROR_INVALID_PARAMETER) {
		//CUDA_CHECK_ERROR(err, "Cannot dequeue buffer, invalid buffer.\n");
	} else {
		printf("TAU: Unknown error cannot read from buffer.\n");
	}


}

void Tau_cupti_record_activity(CUpti_Activity *record)
{

  
	//printf("in record activity.\n");
  switch (record->kind) {
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
        cerr << "recording memcpy (device, stream, context, correlation): " << memcpy->deviceId << ", " << memcpy->streamId << ", " << memcpy->contextId << ", " << memcpy->correlationId << ", " << memcpy->start << "-" << memcpy->end << "ns.\n" << endl;
		    cerr << "recording memcpy src: " << memcpy->srcDeviceId << "/" << memcpy->srcContextId << endl;
		    cerr << "recording memcpy dst: " << memcpy->dstDeviceId << "/" << memcpy->dstContextId << endl;
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
          MESSAGE_RECIPROCAL_SEND
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
          MESSAGE_RECIPROCAL_RECV
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
			//cerr << "recording memcpy: " << end - start << "ns.\n" << endl;
		  //cerr << "recording memcpy on device: " << id << endl;
		  //cerr << "recording memcpy kind: " << copyKind << endl;
			//We do not always know on the corresponding host event on
			//the CPU what type of copy we have so we need to register 
			//the bytes copied here. Be careful we only want to record 
			//the bytes copied once.
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
        direction
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
        runtimeCorrelationId = kernel->runtimeCorrelationId;
#if CUDA_VERSION >= 5050
        gridId = kernel->gridId;
#else
        gridId = 0;
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
        kernelMap[kernel->correlationId] = *kernel;
#if CUDA_VERSION >= 5050
      }
#endif
      //cerr << "recording kernel (device, stream, context, correlation, grid, name): " << deviceId << ", " << streamId << ", " << contextId << ", " << correlationId << ", " << gridId << ", " << name << ", "<< start << "-" << end << "ns.\n" << endl;
      /*if (record->kind == CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
        cerr << "CDP kernel, parent is: " << parentGridId << endl;
      }*/
			//cerr << "recording kernel (id): "  << kernel->correlationId << ", " << kernel->name << ", "<< kernel->end - kernel->start << "ns.\n" << endl;
      

			name = demangleName(name);

      eventMap.erase(eventMap.begin(), eventMap.end());
			if (gpu_occupancy_available(deviceId))
			{
        record_gpu_occupancy(blockX, 
                            blockY,
                            blockZ,
                            registersPerThread,
                            staticSharedMemory,
                            deviceId,
                            name, 
                            &eventMap);
			}

			uint32_t id;
			if (cupti_api_runtime())
			{
				id = runtimeCorrelationId;
			}
			else
			{
				id = correlationId;
			}
      int number_of_metrics = Tau_CuptiLayer_get_num_events() + 1;
      double metrics_start[number_of_metrics];
      double metrics_end[number_of_metrics];
#if CUDA_VERSION >= 5050
      if (record->kind != CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
        record_gpu_counters(deviceId, name, id, &eventMap);
      }
#else
      record_gpu_counters(deviceId, name, id, &eventMap);
#endif
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

      eventMap[bs] = blockX * blockY * blockZ;
      eventMap[dm] = dynamicSharedMemory;
      eventMap[sm] = staticSharedMemory;
      eventMap[lm] = localMemoryPerThread;
      eventMap[lr] = registersPerThread;
      
      GpuEventAttributes *map;
			int map_size = eventMap.size();
			map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
      int i = 0;
      for (eventMap_t::iterator it = eventMap.begin(); it != eventMap.end(); it++)
      {
        map[i].userEvent = it->first;
        map[i].data = it->second;
        i++;
      }
			
#if CUDA_VERSION >= 5050
      if (record->kind == CUPTI_ACTIVITY_KIND_CDP_KERNEL) {
        if (TauEnv_get_cuda_track_cdp()) {
          Tau_cupti_register_gpu_event(name, deviceId,
            streamId, contextId, id, parentGridId, true, map, map_size,
            start / 1e3, end / 1e3);
        }
      } else {
#endif
        Tau_cupti_register_gpu_event(name, deviceId,
          streamId, contextId, id, 0, false, map, map_size,
          start / 1e3, end / 1e3);
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
			metadata[id].name = "GPU Name";
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
			deviceMap[device->id] = *device;
#if CUDA_VERSION < 5000
      if (deviceMap.size() > 1 && Tau_CuptiLayer_get_num_events() > 0)
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
			//cerr << "source locator (id): " << source->id << ", " << source->fileName << ", " << source->lineNumber << ".\n" << endl;
      sourceLocatorMap[source->id] = *source;
    }
    case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS:
    {
			CUpti_ActivityGlobalAccess *global_access = (CUpti_ActivityGlobalAccess *)record;
			//cerr << "global access (cor. id) (source id): " << global_access->correlationId << ", " << global_access->sourceLocatorId << ", " << global_access->threadsExecuted << ".\n" << endl;
      //globalAccessMap[global_access->correlationId] = *global_access;
     
      CUpti_ActivityKernel *kernel = &kernelMap[global_access->correlationId];
      CUpti_ActivitySourceLocator *source = &sourceLocatorMap[global_access->sourceLocatorId];

      if (kernel->kind != CUPTI_ACTIVITY_KIND_INVALID)
      {
        eventMap.erase(eventMap.begin(), eventMap.end());

        std::string name;
        form_context_event_name(kernel, source, "Accesses to Global Memory", &name);
        TauContextUserEvent* ga;
        Tau_cupti_find_context_event(&ga, name.c_str(), false);
        eventMap[ga] = global_access->executed;
        int map_size = eventMap.size();
        GpuEventAttributes *map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
        int i = 0;
        for (eventMap_t::iterator it = eventMap.begin(); it != eventMap.end(); it++)
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
          kernel->streamId, kernel->contextId, id, map, map_size);
      }
    }
    case CUPTI_ACTIVITY_KIND_BRANCH:
    {
			CUpti_ActivityBranch *branch = (CUpti_ActivityBranch *)record;
			//cerr << "branch (cor. id) (source id): " << branch->correlationId << ", " << branch->sourceLocatorId << ", " << branch->threadsExecuted << ".\n" << endl;
     
      CUpti_ActivityKernel *kernel = &kernelMap[branch->correlationId];
      CUpti_ActivitySourceLocator *source = &sourceLocatorMap[branch->sourceLocatorId];

      if (kernel->kind != CUPTI_ACTIVITY_KIND_INVALID)
      {
        eventMap.erase(eventMap.begin(), eventMap.end());
        
        std::string name;
        form_context_event_name(kernel, source, "Branches Executed", &name);
        TauContextUserEvent* be;
        Tau_cupti_find_context_event(&be, name.c_str(), false);
        eventMap[be] = branch->executed;
        
        form_context_event_name(kernel, source, "Branches Diverged", &name);
        TauContextUserEvent* de;
        Tau_cupti_find_context_event(&de, name.c_str(), false);
        eventMap[de] = branch->diverged;

        GpuEventAttributes *map;
        int map_size = eventMap.size();
        map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
        int i = 0;
        for (eventMap_t::iterator it = eventMap.begin(); it != eventMap.end(); it++)
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
          kernel->streamId, kernel->contextId, id, map, map_size);
      }
    }
#endif //CUPTI_API_VERSION >= 3
	}
}

//Helper function givens ceiling with given significance.
int ceil(float value, int significance)
{
	return ceil(value/significance)*significance;
}

int gpu_occupancy_available(int deviceId)
{
	//device callback not called.
	if (deviceMap.empty())
	{
		return 0;
	}

	CUpti_ActivityDevice device = deviceMap[deviceId];

	if ((device.computeCapabilityMajor > 3) ||
		device.computeCapabilityMajor == 3 &&
		device.computeCapabilityMinor > 5)
	{
		TAU_VERBOSE("TAU Warning: GPU occupancy calculator is not implemented for devices of compute capability > 3.5.");
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

void record_gpu_launch(int correlationId)
{
  Tau_cupti_register_host_calling_site(correlationId);	

  CUdevice device;
  cuCtxGetDevice(&device);

  record_gpu_counters_at_launch(device);
}
void record_gpu_counters(int device_id, const char *name, uint32_t correlationId, eventMap_t *m)
{
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
      //std::cout << "at record: "<< name << " ====> " << std::endl;
      //std::cout << "\tstart: " << counters_at_last_launch[device_id][n] << std::endl;
      //std::cout << "\t stop: " << current_counters[device_id][n] << std::endl;

      TauContextUserEvent* c;
      const char *name = Tau_CuptiLayer_get_event_name(n);
      if (n >= counterEvents.size()) {
        c = (TauContextUserEvent *) Tau_return_context_userevent(name);
        counterEvents.push_back(c);
      } else {
        c = counterEvents[n];
      }
      Tau_set_context_event_name(c, name);
      eventMap[c] = (current_counters[device_id][n] - counters_at_last_launch[device_id][n]) * kernels_encountered[device_id];

      
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
                          eventMap_t *map)
{
	CUpti_ActivityDevice device = deviceMap[deviceId];


	int myWarpsPerBlock = ceil(
				(blockX * blockY * blockZ)/
				device.numThreadsPerWarp
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
	(*map)[alW] = allocatable_warps;
  //map[5].userEvent = alW;
	//map[5].data = allocatable_warps;

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
  (*map)[alR] = allocatable_registers;

	int sharedMemoryUnit;
	switch(device.computeCapabilityMajor)
	{
		case 1: sharedMemoryUnit = 512; break;
		case 2: sharedMemoryUnit = 128; break;
		case 3: sharedMemoryUnit = 256; break;
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
  (*map)[alS] = allocatable_shared_memory;

	int allocatable_blocks = min(allocatable_warps, min(allocatable_registers, allocatable_shared_memory));

	int occupancy = myWarpsPerBlock * allocatable_blocks;

//#define RESULTS_TO_STDOUT 1
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
  (*map)[occ] = occupancy;

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
	return id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
		     id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel;
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

#endif //CUPTI API VERSION >= 2
