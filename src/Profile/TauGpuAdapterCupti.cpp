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


bool functionIsMemcpy(int id)
{
	return		(//runtime API
						id == 31  || id == 41 ||
						id == 39  || id == 40 ||
						id == 32  || id == 33 ||
						//driver API
						id == 276 || id == 279 ||
						id == 278 || id == 290
					  );
}

template<class APItype> int callbacksite(const APItype &info)
{
	return info->callbacksite;
}
template<class APItype> const char* functionName(const APItype &info,
																								 CUpti_CallbackDomain domain)
{
	const char *orig_name = info->functionName;
	char *name = new char[strlen(orig_name) + 13];
	sprintf(name, "%s (%s)",orig_name, (domain ==
	CUPTI_CB_DOMAIN_RUNTIME_API_TRACE ? "RuntimeAPI" : "DriverAPI"));
	return name;
}
template<class APItype> int functionId(const APItype &info)
{
	return info->functionId;
}

#define CAST_TO_MEMCPY_TYPE_AND_CALL(name, id, info, params, member) \
	if ((id) == CUPTI_RUNTIME_TRACE_CBID_##name##_v3020) \
	{ \
		printf("id match: %d,\n", id ); \
		member = ((name##_params *) info->params)->member; \
	}
			

template<class APItype> int kind(const APItype &info, CUpti_CallbackId id)
{
	//if (id == CUPTI_RUNTIME_TRACE_cudaMemcpy_v3020)
//		return ((cudaMemcpy_params *) info->params)->kind;
	//return info->kind;
	void *params;
	int kind = -1;
	CAST_TO_MEMCPY_TYPE_AND_CALL(cudaMemcpy, id, info, params, kind)
	CAST_TO_MEMCPY_TYPE_AND_CALL(cudaMemcpyToArray, id, info, params, kind)

	printf("kind is %d.\n", kind);
	return kind;

}
template<class APItype> int count(const APItype &info, CUpti_CallbackId id)
{
	return ((cudaMemcpy_params *) info->params)->count;
	//return info->count;
}
template<class APItype> bool isMemcpy(const APItype &info)
{
  return functionIsMemcpy(functionId(info));
}

/*
class CBInfo
{
public:
	int callbacksite;
	const char *functionName;
	int functionId;
	int kind;
	int count;

	CBInfo()
	{
		callbacksite = 0; 
		functionName = "";
		functionId = 0;
		kind = 0;
		count = 0;
	}
};

class CBInfoRuntime
{

  CUpti_RuntimeTraceApi *cbInfo;
	CUpti_CallbackDomain domain;

public:

	CBInfoRuntime(const void *params)
	{
		cbInfo = (CUpti_RuntimeTraceApi*)params;
	}

	int callbacksite()
	{
		return cbInfo->callbacksite;
	}

	bool isMemcpy()
	{
		return functionIsMemcpy(cbInfo->functionId);
	}

};*/




void Tau_cuda_timestamp_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
	//const CBInfo *cbInfo = new CBInfo();

	const CUpti_RuntimeTraceApi *cbInfo = (CUpti_RuntimeTraceApi *) params;

	const char *name = functionName(cbInfo, domain);
	int site = callbacksite(cbInfo);
	bool memcpy = isMemcpy(cbInfo);

	/*if (domain == CUPTI_CB_DOMAIN_RUNTIME_API_TRACE)
	{
  	cbInfo = static_cast<const CBInfo*>(params);
  	RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;
	}
	else
	{
		const CBInfoDriver *cbInfo = new CBInfoRuntime(params);
	}*/

	CUptiResult err;
	//const char *name = new char[strlen(cbInfo->functionName) + 13];
	//sprintf(name, "%s (%s)",cbInfo->functionName, (domain ==
	//CUPTI_CB_DOMAIN_RUNTIME_API_TRACE ? "RuntimeAPI" : "DriverAPI"));

	//if (cbInfo->site == CUPTI_API_ENTER)
	if (site == CUPTI_API_ENTER)
	{
		printf("Enter: %s.\n", name);
		//if (functionmemcpy(cbInfo->functionId))
		if (memcpy)
		{
			int memcpyKind = kind(cbInfo, id);
			int memcpyCount = count(cbInfo, id);

			printf("Is memcpy.\n");
			//cudaMemcpy_params params;
			//memcpy(&params, (cudaMemcpy_params *) cbInfo->params,
			//sizeof(cudaMemcpy_params));
			//printf("Enter Memcpy: dest: %d, src: %d, count: %llu, kind: %d.\n",
			//params.dst, params.src,
			//params.count, params.kind);

			//TODO: sort out GPU ids
			//TODO: memory copies from device to device.
			//printf("cuda D2D is: %d.\n", cudaMemcpyDeviceToDevice);
			if (memcpyKind == cudaMemcpyHostToDevice)
			{
				Tau_gpu_enter_memcpy_event(name, 
				&cudaEventId(functionId(cbInfo)), &cudaGpuId(0,0), memcpyCount, MemcpyHtoD);
			}
			else if (memcpyKind == cudaMemcpyDeviceToHost)
			{
				Tau_gpu_enter_memcpy_event(name,
				&cudaEventId(functionId(cbInfo)), &cudaGpuId(0,0), memcpyCount, MemcpyDtoH);
			}
			else if (memcpyKind == cudaMemcpyDeviceToDevice)
			{
				printf("TODO: track DeviceToDevice MemCpys.\n");
			}
		}
		else 
		{
			Tau_gpu_enter_event(name, &cudaEventId(functionId(cbInfo)));
		}
	}
	else if (site == CUPTI_API_EXIT)
	{
		if (memcpy)
		{
			int memcpyKind = kind(cbInfo, id);
			int memcpyCount = count(cbInfo, id);
			/*
			cudaMemcpy_params params;
			memcpy(&params, (cudaMemcpy_params *) cbInfo->params,
			sizeof(cudaMemcpy_params));*/
			if (memcpyKind == cudaMemcpyHostToDevice)
			{
				Tau_gpu_exit_memcpy_event(name,
				&cudaEventId(functionId(cbInfo)), &cudaGpuId(0,0), MemcpyHtoD);
			}
			else if (memcpyKind == cudaMemcpyDeviceToHost)
			{
				Tau_gpu_exit_memcpy_event(name,
				&cudaEventId(functionId(cbInfo)), &cudaGpuId(0,0), MemcpyDtoH);
			}
			else if (memcpyKind == cudaMemcpyDeviceToDevice)
			{
				printf("TODO: track DeviceToDevice MemCpys.\n");
			}
		}
		else
		{
			Tau_gpu_exit_event(name, &cudaEventId(functionId(cbInfo)));
			//	Shutdown at Thread Exit
			if (functionId(cbInfo) == 123)
			{
				Tau_gpu_exit();
				return;
			}
		}
		//printf("Exit: %s:%d.\n", cbInfo->functionName, cbInfo->functionId);
	}
}

CUpti_SubscriberHandle rtSubscriber;
CUpti_SubscriberHandle drSubscriber;

void Tau_cuda_onload(void)
{
	//printf("in Tau_cuda_onload.\n");
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
	err = cuptiSubscribe(&rtSubscriber, (CUpti_CallbackFunc)Tau_cuda_timestamp_callback , &trace);
	err = cuptiSubscribe(&drSubscriber, (CUpti_CallbackFunc)Tau_cuda_timestamp_callback , &trace);
	CUDA_CHECK_ERROR(err, "Cannot Subscribe.\n");

	err = cuptiEnableDomain(1, rtSubscriber,CUPTI_CB_DOMAIN_RUNTIME_API_TRACE);
	err = cuptiEnableDomain(1, drSubscriber,CUPTI_CB_DOMAIN_DRIVER_API_TRACE);
	CUDA_CHECK_ERROR(err, "Cannot set Domain.\n");

	Tau_gpu_init();
}

void Tau_cuda_onunload(void)
{
	//printf("in Tau_cuda_onunload.\n");
  CUresult err;
  err = cuptiUnsubscribe(rtSubscriber);
  err = cuptiUnsubscribe(drSubscriber);
  CUDA_CHECK_ERROR(err, "Cannot unsubscribe.\n");
	
}
