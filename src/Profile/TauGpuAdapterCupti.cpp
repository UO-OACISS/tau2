#include <Profile/TauMetrics.h>
#include <Profile/TauGpuAdapterCupti.h>
#include <stdio.h>
#include <string>
using namespace std;


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

template<class APItype> void context(const APItype &info,
																					CUpti_CallbackId id,
																				  CUcontext *ctx)
{
	printf("in context template.\n");
	ctx = ((cuCtxCreate_v2_params *) info->params)->pctx;
  //CAST_TO_DRIVER_CONTEXT_TYPE_AND_CALL(cuCtxCreate, id, info, ctx)
  //CAST_TO_DRIVER_CONTEXT_TYPE_AND_CALL(cuCtxCreate_v2, id, info, ctx)
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

bool cupti_metrics_initialized = false;
bool track_instructions = false;

CUpti_EventGroup eventGroup;
CUpti_EventID eventId = 20;

void cupti_metrics_init(CUcontext ctx)
{
	printf("2 initalizing metrics, context %d.\n", ctx);

	printf("Event Group (before init): %d\n", eventGroup);
	CUptiResult cuptiErr;
	cuptiErr = cuptiEventGroupCreate(ctx, &eventGroup, 0);
	CUPTI_CHECK_ERROR(cuptiErr, "cuptiEventGroupCreate\n");

	printf("Event Group (1): %d\n", eventGroup);
	cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId);
	CUPTI_CHECK_ERROR(cuptiErr, "cuptiEventGroupAddEvent\n");

	printf("Event Group (2): %d\n", eventGroup);
	cuptiErr = cuptiEventGroupEnable(eventGroup);
	printf("CUPTI_ERR: %d.\n", cuptiErr);
	CUPTI_CHECK_ERROR(cuptiErr, "cuptiEventGroupEnable\n");
}

void metric_read_cupti_ins(int tid, int idx, double values[])
{
	if (!cupti_metrics_initialized)
	{
		values[idx] = 0;
	}
	else
	{
		
		uint64_t eventVal;
		size_t bytesRead = sizeof (uint64_t);
    CUptiResult cuptiErr;
		cuptiErr = cuptiEventGroupReadEvent(eventGroup, 
																					CUPTI_EVENT_READ_FLAG_NONE, 
																					eventId, &bytesRead, &eventVal);
    CUPTI_CHECK_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
    //printf("CUDA INS value: %llu\n", (unsigned long long int) eventVal);
    //printf("CUDA INS value: %f\n", (double) eventVal);
		values[idx] = (double) eventVal;
	  
		//printf("read cupti metric value.\n");
	}
}





void Tau_cuda_timestamp_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params)
{
	//const CBInfo *cbInfo = new CBInfo();
	//printf("in callback.\n");
	const char *name;
	int site;
	bool memcpy;
	int memcpyKind;
	int memcpyCount;
	int funcId;
	CUcontext ctx;

	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API_TRACE)
	{
		//printf("getting data form runtime API.\n");
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
		if (id == CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020 && track_instructions
		&& !cupti_metrics_initialized)
		{
			const CUpti_RuntimeTraceApi *cbInfo = (CUpti_RuntimeTraceApi *) params;
			//cuCtxSynchronize();
			//context(cbInfo, id, &ctx);
			//printf("creating context, %s.\n", name);
			CUcontext ctx;
			CUdevice device;
			int *dId = ((cudaGetDevice_v3020_params *) cbInfo->params)->device;
			//printf("device id: %d", *dId);
			cuDeviceGet(&device, *dId);
			cuCtxCreate(&ctx, 0, device);
			cupti_metrics_init(ctx);
			cupti_metrics_initialized = true;
			//printf("1 initalizing metrics, context %d.\n", ctx);
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
		//printf("id: %d. ctxCreate: %d.\n", id,
		//CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2);
		// if cuCtxCreate()
		//if ((id == CUPTI_DRIVER_TRACE_CBID_cuCtxCreate || 
		//id == CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2) && track_instructions)
		//printf("Enter: %s.\n", name);
	}

	CUptiResult err;
	if (site == CUPTI_API_ENTER)
	{
		if (memcpy)
		{
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
			if ((id == CUPTI_DRIVER_TRACE_CBID_cuCtxCreate || 
			id == CUPTI_DRIVER_TRACE_CBID_cuCtxCreate_v2) && track_instructions)
			{
				const CUpti_DriverTraceApi *cbInfo = (CUpti_DriverTraceApi *) params;
				//CUcontext ctx;
				//context(cbInfo, id, &ctx); 
				//cupti_metrics_init(ctx);
				//cupti_metrics_initialized = true;
			  //context(cbInfo, id, &ctx); 
			  //printf("2 initalizing metrics, context %d.\n", ctx);
			}
		}
		//printf("Exit: %s:%d.\n", cbInfo->functionName, cbInfo->functionId);
	}
}


CUpti_SubscriberHandle rtSubscriber;
bool runtime_enabled = false;
CUpti_SubscriberHandle drSubscriber;
bool driver_enabled = false;

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
		runtime_enabled= true;
	}
	if (driver_api != NULL)
	{
		printf("TAU: Subscribing to DRIVER API.\n");
		err = cuptiEnableDomain(1, drSubscriber,CUPTI_CB_DOMAIN_DRIVER_API_TRACE);
		driver_enabled= true;
	}
	CUDA_CHECK_ERROR(err, "Cannot set Domain.\n");

	const char *names;
	const char **all_names;

	int nmetrics;
	TauMetrics_getCounterList(&all_names, &nmetrics);

	printf("number of metrics: %d.\n", nmetrics);

	for (int number = 0; number < nmetrics; number++)
	{
		names = TauMetrics_getMetricName(number);
		printf("Metrics: %s. #%d\n", names, number);
		string str (names);
		if (str.find(CUPTI_METRIC_INSTRUCTIONS) != string::npos)
		{
			track_instructions = true;
			printf("RECORDING number of instructions.\n");
			if (runtime_api == NULL)
			{
				printf("ERROR CUPTI metrics require the Runtime layer to be enabled. Please set TAU_CUPTI_RUNTIME.\n");
				exit(1);
			}
		}
   
	}
	Tau_gpu_init();
}

void Tau_cuda_onunload(void)
{
	//printf("in Tau_cuda_onunload.\n");
  CUresult err;
  err = cuptiUnsubscribe(rtSubscriber);
  err = cuptiUnsubscribe(drSubscriber);
  CUDA_CHECK_ERROR(err, "Cannot unsubscribe.\n");
	
	if (eventGroup != NULL)
	{
		CUptiResult cuptiErr;
    cuptiErr = cuptiEventGroupDisable(eventGroup);
    CUPTI_CHECK_ERROR(cuptiErr, "cuptiEventGroupDisable");

    cuptiErr = cuptiEventGroupDestroy(eventGroup);
    CUPTI_CHECK_ERROR(cuptiErr, "cuptiEventGroupDestroy");
	}
}
