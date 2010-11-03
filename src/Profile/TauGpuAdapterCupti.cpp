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

/*template<class MemcpyParam> void get_kind(const MemcpyParam &param, int &kind)
{
	kind = params->kind;

}*/

template<class APItype> void get_value_from_memcpy(const APItype &info,
																									CUpti_CallbackId id,
																									CUpti_CallbackDomain domain,
																									int &kind,
																									int &count)
{
	
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API_TRACE)
	{
		//CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(info->functionName, id, info, kind, count)
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
		/* these calls do not have count member.
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2D, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2DToArray, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2DFromArray, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2DArrayToArray, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2DAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2DToArrayAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy2DFromArrayAsync, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy3D, id, info, kind, count)
    CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(cudaMemcpy3DAsync, id, info, kind, count)
		*/
	}
	else
	{
		//CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(info->functionName, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoD_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoDAsync_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoH_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoHAsync_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoD_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoDAsync_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoH_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoHAsync_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoD_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoA_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoA_v2, id, info, kind, count)
		//These structors do not have ByteCount.
		//CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy2D_v2, id, info, kind, count)
		//CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy2DUnaligned_v2, id, info, kind, count)
		//CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy2DAsync_v2, id, info, kind, count)
		//CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy3D_v2, id, info, kind, count)
		//CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy3DAsync_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoA_v2, id, info, kind, count)
		CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoAAsync_v2, id, info, kind, count)

    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoD, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoH, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoD, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoA, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoD, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoA, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoH, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoA, id, info, kind, count)
		//These structors do not have ByteCount.
    //CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy2D, id, info, kind, count)
    //CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy2DUnaligned, id, info, kind, count)
    //CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy3D, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoDAsync, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoHAsync, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyDtoDAsync, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyHtoAAsync, id, info, kind, count)
    CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpyAtoHAsync, id, info, kind, count)
		//These structors do not have ByteCount.
    //CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy2DAsync, id, info, kind, count)
    //CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy3DAsync, id, info, kind, count)
		//No struct.
    //CAST_TO_DRIVER_MEMCPY_TYPE_AND_CALL(cuMemcpy_v2, id, info, kind, count)


	}
	//printf("[1] kind is %d.\n", kind);
}

template<class APItype> int kind(const APItype &info, CUpti_CallbackId id, 
																 CUpti_CallbackDomain domain) 
{
	//if (id == CUPTI_RUNTIME_TRACE_cudaMemcpy_v3020)
//		return ((cudaMemcpy_params *) info->params)->kind;
	//return info->kind;

	int kind = -1;
	int count = 0;

	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API_TRACE)
	{
		get_value_from_memcpy(info, id, domain, kind, count);
	}
	else
	{
		//TODO: parse name
		kind = 1;	
	}

	//printf("[2] kind is %d.\n", kind);
	return kind;

}
template<class APItype> int count(const APItype &info, CUpti_CallbackId id,
																  CUpti_CallbackDomain domain)
{
	int kind = -1;
	int count = 0;
	get_value_from_memcpy(info, id, domain, kind, count);
	//printf("[2] kind is %d.\n", kind);
	return count;
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
	const char *name;
	int site;
	bool memcpy;
	int memcpyKind;
	int memcpyCount;
	int funcId;

	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API_TRACE)
	{
		const CUpti_RuntimeTraceApi *cbInfo = (CUpti_RuntimeTraceApi *) params;

		funcId = functionId(cbInfo);
		name = functionName(cbInfo, domain);
		site = callbacksite(cbInfo);
		memcpy = isMemcpy(cbInfo);
		if (memcpy)
		{
			memcpyKind = kind(cbInfo, id, domain);
			memcpyCount = count(cbInfo, id, domain);
		}
	}
	else
	{
		const CUpti_DriverTraceApi *cbInfo = (CUpti_DriverTraceApi *) params;

		funcId = functionId(cbInfo);
		name = functionName(cbInfo, domain);
		site = callbacksite(cbInfo);
		memcpy = isMemcpy(cbInfo);
		if (memcpy)
		{
			memcpyKind = kind(cbInfo, id, domain);
			memcpyCount = count(cbInfo, id, domain);
		}
	}

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
		//printf("Enter: %s.\n", name);
		//if (functionmemcpy(cbInfo->functionId))
		if (memcpy)
		{
			//printf("Is memcpy: %d.\n", memcpyKind);
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
				&cudaEventId(funcId), &cudaGpuId(0,0), memcpyCount, MemcpyHtoD);
			}
			else if (memcpyKind == cudaMemcpyDeviceToHost)
			{
				Tau_gpu_enter_memcpy_event(name,
				&cudaEventId(funcId), &cudaGpuId(0,0), memcpyCount, MemcpyDtoH);
			}
			else if (memcpyKind == cudaMemcpyDeviceToDevice)
			{
				printf("TODO: track DeviceToDevice MemCpys.\n");
			}
		}
		else 
		{
			Tau_gpu_enter_event(name, &cudaEventId(funcId));
		}
	}
	else if (site == CUPTI_API_EXIT)
	{
		if (memcpy)
		{
			/*
			cudaMemcpy_params params;
			memcpy(&params, (cudaMemcpy_params *) cbInfo->params,
			sizeof(cudaMemcpy_params));*/
			if (memcpyKind == cudaMemcpyHostToDevice)
			{
				Tau_gpu_exit_memcpy_event(name,
				&cudaEventId(funcId), &cudaGpuId(0,0), MemcpyHtoD);
			}
			else if (memcpyKind == cudaMemcpyDeviceToHost)
			{
				Tau_gpu_exit_memcpy_event(name,
				&cudaEventId(funcId), &cudaGpuId(0,0), MemcpyDtoH);
			}
			else if (memcpyKind == cudaMemcpyDeviceToDevice)
			{
				printf("TODO: track DeviceToDevice MemCpys.\n");
			}
		}
		else
		{
			Tau_gpu_exit_event(name, &cudaEventId(funcId));
			//	Shutdown at Thread Exit
			if (funcId == 123)
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

	//Get env variables
	char *runtime_api, *driver_api;
	runtime_api = getenv("TAU_CUPTI_RUNTIME");
	driver_api = getenv("TAU_CUPTI_DRIVER");
	//printf("ENV: %s.\n", runtime_api);
	//printf("ENV: %s.\n", driver_api);
	if (runtime_api != NULL)
	{
		printf("TAU: Subscribing to RUNTIME API.\n");
		err = cuptiEnableDomain(1, rtSubscriber,CUPTI_CB_DOMAIN_RUNTIME_API_TRACE);
	}
	if (driver_api != NULL)
	{
		printf("TAU: Subscribing to DRIVER API.\n");
		err = cuptiEnableDomain(1, drSubscriber,CUPTI_CB_DOMAIN_DRIVER_API_TRACE);
	}
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
