#include <Profile/TauGpuAdapterCupti.h>
#include <stdio.h>

class cudaGpuId : public gpuId {

	int contextId;
	int deviceId;

public:
	cudaGpuId(const int cId, const int dId) :
	contextId(cId), deviceId(dId) {} 
  
	char* printId();
	x_uint64 id_p1() { return contextId; }
	x_uint64 id_p2() { return deviceId; }
};

char* cudaGpuId::printId() 
{
		char *r;
		sprintf(r, "%d:%d", contextId, deviceId);
		return r;
}

class cudaEventId : public eventId
{
	int id;
	public:
	cudaEventId(const int a) :
		id(a) {}
	
	// for use in STL Maps	
	bool operator<(const cudaEventId& A) const
	{ 
			return id<A.id; 
	}
};


void __attribute__ ((constructor)) Tau_cuda_onload(void);
void __attribute__ ((destructor)) Tau_cuda_onunload(void);


void Tau_cuda_timestamp_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
  const CUpti_RuntimeTraceApi *cbInfo = (CUpti_RuntimeTraceApi*)params;
  RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;

	CUptiResult err;
	if (cbInfo->callbacksite == CUPTI_API_ENTER)
	{
		printf("Enter: %s:%d.\n", cbInfo->functionName, cbInfo->functionId);
		if (cbInfo->functionId == 31)
		{
			cudaMemcpy_params params;
			memcpy(&params, (cudaMemcpy_params *) cbInfo->params,
			sizeof(cudaMemcpy_params));
			printf("Enter Memcpy: dest: %d, src: %d, count: %llu, kind: %d.\n",
			params.dst, params.src,
			params.count, params.kind);

      err = cuptiDeviceGetTimestamp(cbInfo->ctx, &traceData->startTimestamp);
			CUDA_CHECK_ERROR(err, "Cannot Access Timestamp.\n");
		
			printf("GPU Timestamp: %llu.\n", traceData->startTimestamp);

			//TODO: sort out GPU ids
			//TODO: memory copies from device to device.

			if (params.kind == 1)
			{
				Tau_gpu_enter_memcpy_event(
				&cudaEventId(cbInfo->functionId), &cudaGpuId(0,0), MemcpyHtoD);
			}
			else if (params.kind == 2)
			{
				Tau_gpu_enter_memcpy_event(
				&cudaEventId(cbInfo->functionId), &cudaGpuId(0,0), MemcpyDtoH);
			}
		}
		else 
		{
			if (cbInfo->functionId == 13)
			{
				err = cuptiDeviceGetTimestamp(cbInfo->ctx, &traceData->startTimestamp);
				CUDA_CHECK_ERROR(err, "Cannot Access Timestamp.\n");
			  printf("GPU Timestamp: %llu.\n", traceData->startTimestamp);
			}
			Tau_gpu_enter_event(cbInfo->functionName, &cudaEventId(cbInfo->functionId));
		}
	}
	else if (cbInfo->callbacksite == CUPTI_API_EXIT)
	{
		if (cbInfo->functionId == 31)
		{
      
			err = cuptiDeviceGetTimestamp(cbInfo->ctx, &traceData->endTimestamp);
			CUDA_CHECK_ERROR(err, "Cannot Access Timestamp.\n");
			
			printf("GPU Timestamp: %llu.\n", traceData->endTimestamp);
			
			cudaMemcpy_params params;
			memcpy(&params, (cudaMemcpy_params *) cbInfo->params,
			sizeof(cudaMemcpy_params));
			if (params.kind == 1)
			{
				Tau_gpu_exit_memcpy_event(
				&cudaEventId(cbInfo->functionId), &cudaGpuId(0,0), MemcpyHtoD);
				Tau_gpu_register_memcpy_event(&cudaEventId(cbInfo->functionId), &cudaGpuId(0,0),
				(double) traceData->startTimestamp/1000, (double)
				traceData->endTimestamp/1000, params.count,
				MemcpyHtoD);
			}
			else if (params.kind == 2)
			{
				Tau_gpu_exit_memcpy_event(
				&cudaEventId(cbInfo->functionId), &cudaGpuId(0,0), MemcpyDtoH);
				Tau_gpu_register_memcpy_event(&cudaEventId(cbInfo->functionId), &cudaGpuId(0,0),
				(double) traceData->startTimestamp/1000, (double)
				traceData->endTimestamp/1000, params.count,
				MemcpyDtoH);
			}
			
		}
		else
		{
			if (cbInfo->functionId == 13)
			{
				err = cuptiDeviceGetTimestamp(cbInfo->ctx, &traceData->endTimestamp);
				CUDA_CHECK_ERROR(err, "Cannot Access Timestamp.\n");
			  printf("GPU Timestamp: %llu.\n", traceData->endTimestamp);
				Tau_gpu_register_gpu_event(cbInfo->functionName,
				&cudaEventId(cbInfo->functionId), (double) traceData->startTimestamp/1000,
				(double)traceData->endTimestamp/1000);
			}
			Tau_gpu_exit_event(cbInfo->functionName, &cudaEventId(cbInfo->functionId));
			//	Shutdown at Thread Exit
			if (cbInfo->functionId == 123)
			{
				Tau_gpu_exit();
				return;
			}
		}
		printf("Exit: %s:%d.\n", cbInfo->functionName, cbInfo->functionId);
	}
}

CUpti_SubscriberHandle subscriber;

void Tau_cuda_onload(void)
{
	printf("in Tau_cuda_onload.\n");
	RuntimeApiTrace_t trace[LAUNCH_LAST];
  CUdevice device = 0;
	int computeCapabilityMajor=0;
	int computeCapabilityMinor=0;
  CUresult err;

	/* check removed, cannot call cuDeviceComputeCapability until some CUDA
	 * initialization is completed.
  err = cuDeviceComputeCapability( &computeCapabilityMajor, 
                               &computeCapabilityMinor, 
                               device);
	CUDA_CHECK_ERROR(err, "Cannot check Computer Capability.\n");
	if (computeCapabilityMajor > 1) {
			printf("cupti supported only for Tesla\n");
			exit(1);
	}*/
	err = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)Tau_cuda_timestamp_callback , &trace);
	CUDA_CHECK_ERROR(err, "Cannot Subscribe.\n");

	//err = cuptiEnableDomain(1, subscriber,CUPTI_CB_DOMAIN_DRIVER_API_TRACE);
	err = cuptiEnableDomain(1, subscriber,CUPTI_CB_DOMAIN_RUNTIME_API_TRACE);
	CUDA_CHECK_ERROR(err, "Cannot set Domain.\n");

	Tau_gpu_init();
}

void Tau_cuda_onunload(void)
{
	printf("in Tau_cuda_onunload.\n");
  CUresult err;
  err = cuptiUnsubscribe(subscriber);
  CUDA_CHECK_ERROR(err, "Cannot unsubscribe.\n");
	
}
