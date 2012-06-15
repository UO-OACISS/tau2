#include <Profile/CuptiActivity.h>
#include <iostream>
using namespace std;

#if CUPTI_API_VERSION >= 2

void Tau_cupti_onload()
{
	//printf("in onload.\n");
	CUptiResult err;
	err = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)Tau_cupti_callback_dispatch, NULL);
  
	if (cupti_api_runtime())
	{
		//printf("TAU: Subscribing to RUNTIME API.\n");
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
		//runtime_enabled = true;
	}
	if (cupti_api_driver())
	{
		printf("TAU: Subscribing to DRIVER API.\n");
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
		//driver_enabled = true;
	}

	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE); 
	err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE); 

	CUDA_CHECK_ERROR(err, "Cannot set Domain.\n");

	//setup global activity queue.
	activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
	err = cuptiActivityEnqueueBuffer(NULL, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);
 	
	//to collect device info 
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE);
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT);
	
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
	CUDA_CHECK_ERROR(err, "Cannot enqueue buffer.\n");

	Tau_gpu_init();
}

void Tau_cupti_onunload() {}

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
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
		//Tau_cupti_register_sync_event(NULL, 0);
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
				FunctionInfo *p = TauInternal_CurrentProfiler(Tau_RtsLayer_getTid())->ThisFunction;
				Tau_cupti_register_calling_site(cbInfo->correlationId, p);
				//functionInfoMap[cbInfo->correlationId] = p;	
				Tau_cupti_enter_memcpy_event(
					TAU_GPU_USE_DEFAULT_NAME, 0, cbInfo->contextUid, cbInfo->correlationId, 
					count, getMemcpyType(kind)
				);
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
				Tau_cupti_exit_memcpy_event(
					TAU_GPU_USE_DEFAULT_NAME, 0, cbInfo->contextUid, cbInfo->correlationId, 
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
					Tau_CuptiLayer_disable();
					cuCtxSynchronize();
					Tau_CuptiLayer_enable();
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
				else if (function_is_launch(id))
				{
					FunctionInfo *p = TauInternal_CurrentProfiler(Tau_RtsLayer_getTid())->ThisFunction;
					Tau_cupti_register_calling_site(cbInfo->correlationId, p);
					//functionInfoMap[cbInfo->correlationId] = p;	
					//printf("at launch id: %d.\n", cbInfo->correlationId);
					Tau_CuptiLayer_init();
				}
				Tau_gpu_enter_event(cbInfo->functionName);
			}
			else if (cbInfo->callbackSite == CUPTI_API_EXIT)
			{
				Tau_gpu_exit_event(cbInfo->functionName);
				if (function_is_sync(id))
				{
					//cerr << "sync function name: " << cbInfo->functionName << endl;
					Tau_CuptiLayer_disable();
					cuCtxSynchronize();
					Tau_CuptiLayer_enable();
					Tau_cupti_register_sync_event(cbInfo->context, 0);
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
	
	} else if (err != CUPTI_ERROR_QUEUE_EMPTY) {
		//printf("TAU: Activity queue is empty.\n");
		//CUDA_CHECK_ERROR(err, "Cannot dequeue buffer.\n");
	} else if (err != CUPTI_ERROR_INVALID_PARAMETER) {
		CUDA_CHECK_ERROR(err, "Cannot dequeue buffer, invalid buffer.\n");
	} else {
		printf("TAU: Unknown error cannot read from buffer.\n");
	}
		
}

void Tau_cupti_record_activity(CUpti_Activity *record)
{
	//printf("in record activity.\n");
  switch (record->kind) {
  	case CUPTI_ACTIVITY_KIND_MEMCPY:
		{	
      CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
			//cerr << "recording memcpy: " << memcpy->end - memcpy->start << "ns.\n" << endl;
		  //cerr << "recording memcpy on device: " << memcpy->streamId << "/" << memcpy->runtimeCorrelationId << endl;
			int id;
			if (cupti_api_runtime())
			{
				id = memcpy->runtimeCorrelationId;
			}
			else
			{
				id = memcpy->correlationId;
			}
			Tau_cupti_register_memcpy_event(
				TAU_GPU_USE_DEFAULT_NAME,
				memcpy->streamId,
				memcpy->contextId,
				id,
				memcpy->start / 1e3,
				memcpy->end / 1e3,
				TAU_GPU_UNKNOWN_TRANSFER_SIZE,
				getMemcpyType(memcpy->copyKind)
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
				break;
		}
  	case CUPTI_ACTIVITY_KIND_KERNEL:
		{
			//find FunctionInfo object from FunctionInfoMap
      CUpti_ActivityKernel *kernel = (CUpti_ActivityKernel *)record;
			//cerr << "recording kernel: " << kernel->name << ", " << kernel->end - kernel->start << "ns.\n" << endl;

			GpuEventAttributes *map;
			int map_size = 5;
			map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * map_size);
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
			map[0].userEvent = bs;
			map[0].data = kernel->blockX * kernel->blockY * kernel->blockZ;
			map[1].userEvent = dm;
			map[1].data = kernel->dynamicSharedMemory;
			map[2].userEvent = sm;
			map[2].data= kernel->staticSharedMemory;
			map[3].userEvent = lm;
			map[3].data = kernel->localMemoryPerThread;
			map[4].userEvent = lr;
			map[4].data = kernel->registersPerThread;

			const char* name;
			uint32_t id;
			if (cupti_api_runtime())
			{
				id = kernel->runtimeCorrelationId;
			}
			else
			{
				id = kernel->correlationId;
				//printf("correlationid: %d.\n", id);
			}
			name = demangleName(kernel->name);
		  //cerr << "recording kernel on device: " << kernel->streamId << "/" << id << endl;
			Tau_cupti_register_gpu_event(name,
				kernel->streamId, kernel->contextId, id, map, map_size,
				kernel->start / 1e3, kernel->end / 1e3);
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

			static bool recorded_metadata = false;
			if (!recorded_metadata)
			{

				CUpti_ActivityDevice *device = (CUpti_ActivityDevice *)record;
				
				//first the name.
				Tau_metadata("GPU Name", device->name);

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
				RECORD_DEVICE_METADATA(numMemcpyEngines, device);
				RECORD_DEVICE_METADATA(numMultiprocessors, device);
				RECORD_DEVICE_METADATA(numThreadsPerWarp, device);
			
				recorded_metadata = true;
			}
			break;
		}
	}
}

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
		id ==     CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2
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
		else if (id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2)
		{
			kind = CUPTI_ACTIVITY_MEMCPY_KIND_DTOH;
			count = ((cuMemcpyDtoH_v2_params *) info->functionParams)->ByteCount;
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
		case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
			return MemcpyHtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
			return MemcpyDtoH;
		case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
			return MemcpyDtoH;
		case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
			return MemcpyDtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
			return MemcpyDtoD;
		case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
			return MemcpyDtoD;
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

#endif
