#include <Profile/TauGpuAdapterCupti.h>

void Tau_cupti_onload()
{
	printf("in onload.\n");
	CUptiResult err;
	err = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)Tau_cupti_callback_dispatch, NULL);
  
	if (0 == strcasecmp(TauEnv_get_cupti_api(), "runtime") || 
			0 == strcasecmp(TauEnv_get_cupti_api(), "both"))
	{
		//printf("TAU: Subscribing to RUNTIME API.\n");
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
		//runtime_enabled = true;
	}
  if (0 == strcasecmp(TauEnv_get_cupti_api(), "driver") || 
			0 == strcasecmp(TauEnv_get_cupti_api(), "both")) 
	{
		//printf("TAU: Subscribing to DRIVER API.\n");
		err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
		//driver_enabled = true;
	}

	//err = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE); 

	CUDA_CHECK_ERROR(err, "Cannot set Domain.\n");

	//setup activity queue.
	activityBuffer = (uint8_t *)malloc(ACTIVITY_BUFFER_SIZE);
	err = cuptiActivityEnqueueBuffer(NULL, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
	err = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
	CUDA_CHECK_ERROR(err, "Cannot enqueue buffer.\n");

	Tau_gpu_init();
}

void Tau_cupti_onunload()
{
	printf("in onunload.\n");
	Tau_cupti_register_sync_event();
  CUptiResult err;
  err = cuptiUnsubscribe(subscriber);
}

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
	const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *) params;
	if (cbInfo->callbackSite == CUPTI_API_ENTER)
	{
		Tau_gpu_enter_event(cbInfo->functionName);
	}
	else
	{
		Tau_gpu_exit_event(cbInfo->functionName);
	}
}

void Tau_cupti_register_sync_event()
{
	printf("in sync.\n");
  CUptiResult err, status;
  CUpti_Activity *record = NULL;
	size_t bufferSize = 0;

	err = cuptiActivityDequeueBuffer(NULL, 0, &activityBuffer, &bufferSize);
	printf("activity buffer size: %d.\n", bufferSize);
	CUDA_CHECK_ERROR(err, "Cannot dequeue buffer.\n");

	do {
		status = cuptiActivityGetNextRecord(activityBuffer, bufferSize, &record);
		if (status == CUPTI_SUCCESS) {
			Tau_cupti_record_activity(record);
		}
		else if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
	    CUDA_CHECK_ERROR(err, "Cannot get next record.\n");
			break;
		}	
	} while (status != CUPTI_ERROR_MAX_LIMIT_REACHED);
	
	size_t number_dropped;
	err = cuptiActivityGetNumDroppedRecords(NULL, 0, &number_dropped);

	if (number_dropped > 0)
		printf("TAU WARNING: %d CUDA records dropped, consider increasing the CUPTI_BUFFER size.", number_dropped);

	//requeue buffer
	err = cuptiActivityEnqueueBuffer(NULL, 0, activityBuffer, ACTIVITY_BUFFER_SIZE);
	CUDA_CHECK_ERROR(err, "Cannot requeue buffer.\n");
}

void Tau_cupti_record_activity(CUpti_Activity *record)
{
	//printf("in record activity");
  switch (record->kind) {
  	case CUPTI_ACTIVITY_KIND_MEMCPY:
		{
      CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
			Tau_gpu_register_memcpy_event(
				cuptiRecord(TAU_GPU_USE_DEFAULT_NAME, memcpy->streamId), 
				memcpy->start / 1e3, 
				memcpy->end / 1e3, 
				memcpy->bytes, 
				getMemcpyType(memcpy->copyKind));
				break;
		}
  	case CUPTI_ACTIVITY_KIND_KERNEL:
		{
      CUpti_ActivityKernel *kernel = (CUpti_ActivityKernel *)record;
			Tau_gpu_register_gpu_event(
				cuptiRecord(demangleName(kernel->name), kernel->streamId), 
				kernel->start / 1e3,
				kernel->end / 1e3);
				break;
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
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
	//printf("demangling name....\n");
	dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE |
	DMGL_TYPES);
#else
	dem_name = name;
#endif /* HAVE_GPU_DEMANGLE */
	return dem_name;
}
