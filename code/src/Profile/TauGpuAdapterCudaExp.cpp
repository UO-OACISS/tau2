#include "TauGpuAdapterCudaExp.h"
#include<dlfcn.h>

ToolsAPI gs_toolsapi;

int global_thread_id=0;
pthread_mutex_t event_mutex;
pthread_mutexattr_t event_mutex_attr;

int load_count=0;
bool tau_nexus=false;
bool clock_sync=false;
__thread int ltid=-1;
__thread EventManager *my_manager=NULL;
__thread bool registered=false;
bool user_events=false;

class cudaGpuId : public gpuId {

	NvU64 contextId;
	NvU32 deviceId;

public:
/*	cudaGpuId(const NvU64 cId, const NvU32 dId) :
		contextId(cId), deviceId(dId) {} */
	
	cudaGpuId(const NvU64 cId, const NvU32 dId) {
		contextId = cId;
		deviceId = dId;
	}
	
  char* printId();
	double id_p1() { return (double) contextId; }
	double id_p2() { return (double) deviceId; }
};

char* cudaGpuId::printId() 
{
		char *r;
		sprintf(r, "%f:%f", contextId, deviceId);
		return r;
}


class cuEventId : public eventId
{
	NvU64 contextId;
	NvU64 callId;

	public:
	cuEventId(const NvU64 a, const NvU64 b) :
		contextId(a), callId(b) {}
	
	bool operator<(const cuEventId& A) const
	{ 
		if (contextId == A.contextId)
		{
			return callId<A.callId; 
		}
		else
			return contextId<A.contextId;
	}
};

typedef map<cuEventId, bool> doubleMap;
doubleMap MemcpyEventMap;


#define SYNCH_LATENCY 1

/*
	Helper function to get Coretable
*/
inline cuToolsApi_Core* GetCoreTable(void)
{
	return gs_toolsapi.coreTable;
}

/*
	Helper function to get Devicetable
*/
inline cuToolsApi_Device* GetDeviceTable(void)
{
        return gs_toolsapi.deviceTable;
}
/*
	Helper function to get Contexttable
*/

inline cuToolsApi_Context* GetContextTable(void)
{
	return gs_toolsapi.contextTable;
}

/*
	Helper function to get Clock table
*/
double AlignedTime(int device, double raw_time)
{
#ifdef DEBUG_PROF
	printf("getting synced time on device %d, raw time: \t %f.\n", device, raw_time);
#endif
	double offset = gs_toolsapi.device_clocks[device].tau_end_time -
		gs_toolsapi.device_clocks[device].ref_gpu_end_time;
#ifdef DEBUG_PROF
	printf("device %d, \t \t \t synced time: \t %f.\n", device, (double) raw_time + offset);
#endif
	return (double) raw_time + offset;
}


void ClockSynch()
{
	double cpu_time1, cpu_time2;
	TAU64 ref_t1, ref_t2;
	GetDeviceTable()->DeviceGetCount(&(gs_toolsapi.device_count));
#ifdef DEBUG_PROF
	printf("Syncing clocks. %d devices available. \n", gs_toolsapi.device_count);
#endif
	
	for(int i=0;i<gs_toolsapi.device_count;i++)
	{
		cpu_time1=cpu_time();
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t1);
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t2);
		cpu_time2=cpu_time();
		gs_toolsapi.device_clocks[i].tau_end_time=(cpu_time1+cpu_time2)/2;
		gs_toolsapi.device_clocks[i].ref_gpu_end_time=((double)ref_t1+(double)ref_t2)/2e3;			

#ifdef DEBUG_PROF
	printf("Device [%d] CPU is running %f nanoseconds faster than the GPU.\n", i,
	gs_toolsapi.device_clocks[i].tau_end_time -
	gs_toolsapi.device_clocks[i].ref_gpu_end_time);
#endif

	}
}
/**************************************************************************************
	libcuda.so events are intercepted here as callback.
	This callback handler is executed in the thread context of the application 
	calling the API. The second argument is a GUID which is used to identify 
	different classes of callbacks. TAUCuda uses the callbacks for the API 
	entry/exit , memory profile and the kernel launch profiles.    
**************************************************************************************/
void CUDAAPI callback_handle(
    void* pUserData,
    const cuToolsApi_UUID* callbackId,
    const void* inParams)
{
	NvU64 contextId;
	/*
		For the first time when the Profiler callback is received we 
		synchronize the clock. Note that clock synchronization is done here as 
		synchronizing at the initialization stage created unexpected issues. It's a
		better place in the call back handler routine as this case occurs only 
		due to context synchronization or when the CUDA profile manager buffer is full.    
	*/
	if(!clock_sync)
	{
		ClockSynch();
		clock_sync=true;
	}
	
	// is the current thread registered otherwise increment the thread ID
	// A virtual thread ID is being generated here it might be of use in the future
	/*if(!registered)
	{
		gettid();
		registered=true;
	}*/
	if (*callbackId == cuToolsApi_CBID_EnterGeneric)
	{
		cuToolsApi_EnterGenericInParams* cParams = (cuToolsApi_EnterGenericInParams*) inParams;	
		//extract the device ID for the curent context
		GetContextTable()->CtxGetId(cParams->ctx, &contextId);
		cuEventId id(contextId, cParams->apiCallId);

		if(strncmp(cParams->functionName,"cuMemcpy", sizeof("cuMemcpy")-1)==0)
		{
			NvU32 device;
			GetContextTable()->CtxGetDevice(cParams->ctx,&device);
			cudaGpuId gId(contextId, device);
			bool memcpyType;
			if(strncmp(cParams->functionName,"cuMemcpyHtoD", sizeof("cuMemcpyHtoD")-1)==0)
			{
				memcpyType = MemcpyHtoD;
			}
			else if(strncmp(cParams->functionName,"cuMemcpyDtoH",sizeof("cuMemcpyDtoH")-1)==0)
			{
				memcpyType = MemcpyDtoH;
			}
			MemcpyEventMap.insert(make_pair(id, memcpyType));
			enter_memcpy_event(cParams->functionName, &id, &gId, memcpyType);
		}
		else
		{
			enter_event(cParams->functionName, &id);
		}
	}
	else if (*callbackId == cuToolsApi_CBID_ExitGeneric)
	{
		cuToolsApi_EnterGenericInParams* cParams = (cuToolsApi_EnterGenericInParams*) inParams;	
		//extract the device ID for the curent context
		GetContextTable()->CtxGetId(cParams->ctx, &contextId);
		cuEventId id(contextId, cParams->apiCallId);
		if(strncmp(cParams->functionName,"cuMemcpy", sizeof("cuMemcpy")-1)==0)
		{
			NvU32 device;
			GetContextTable()->CtxGetDevice(cParams->ctx,&device);
			cudaGpuId gId(contextId, device);
			bool memcpyType;
			if(strncmp(cParams->functionName,"cuMemcpyHtoD", sizeof("cuMemcpyHtoD")-1)==0)
			{
				memcpyType = MemcpyHtoD;
			}
			else if(strncmp(cParams->functionName,"cuMemcpyDtoH",sizeof("cuMemcpyDtoH")-1)==0)
			{
				memcpyType = MemcpyDtoH;
			}
			MemcpyEventMap.insert(make_pair(id, memcpyType));
			exit_memcpy_event(cParams->functionName, &id, &gId, memcpyType);
		}
		else
		{
		exit_event(cParams->functionName, &id);
		}
	}
	else if (*callbackId == cuToolsApi_CBID_ProfileLaunch)
	{
		cuToolsApi_ProfileLaunchInParams* cParams = (cuToolsApi_ProfileLaunchInParams*) inParams;	
		GetContextTable()->CtxGetId(cParams->ctx, &contextId);
		cuEventId id(contextId, cParams->apiCallId);
		NvU32 device;
		GetContextTable()->CtxGetDevice(cParams->ctx,&device);
		double startTime = AlignedTime((int)device, (double)cParams->startTime/1000);
		double endTime = AlignedTime((int)device, (double)cParams->endTime/1000);
		register_gpu_event(cParams->methodName, &id, startTime, endTime);
	}
	else if (*callbackId == cuToolsApi_CBID_ProfileMemcpy)
	{
		cuToolsApi_ProfileMemcpyInParams* cParams = (cuToolsApi_ProfileMemcpyInParams*) inParams;	
		GetContextTable()->CtxGetId(cParams->ctx, &contextId);
		cuEventId id(contextId, cParams->apiCallId);
		NvU32 device;
		GetContextTable()->CtxGetDevice(cParams->ctx,&device);
		cudaGpuId gId(contextId, device);
		double startTime = AlignedTime((int)device, (double)cParams->startTime/1000);
		double endTime = AlignedTime((int)device, (double)cParams->endTime/1000);
		doubleMap::const_iterator it = MemcpyEventMap.find(id);

		bool memcpyType;
	  if (it != MemcpyEventMap.end())
		{
			memcpyType = it->second;
			register_memcpy_event(&id, &gId, startTime, endTime, cParams->memTransferSize,
			memcpyType);
		}
		else
		{
    	printf("ERROR: cannot find matching memcopy event.\n");
		}
	}
}
/*
	This initializes the cuda callback mechanism
*/
inline int InitializeToolsApi(void)
{
#ifdef DEBUG_PROF
	printf("Initializing...\n");
#endif
	CUresult status;
	/*
		We first load the library explicitly with global flag which indicates 
		the library is shared across the application. After loading it the routine 
		symbol to extract the interface tables is extracted with dlsym. 
	*/
	gs_toolsapi.handle = dlopen("libcuda.so", RTLD_GLOBAL | RTLD_NOW);
	//gs_toolsapi.handle = dlopen("libcuda.so.190.42", RTLD_GLOBAL | RTLD_NOW);
	if (!gs_toolsapi.handle)
	{
		fprintf(stderr, "Failed to load libcuda.so >> %s\n", dlerror());
		return 1;
	}

	cuDriverGetExportTable_pfn getExportTable;
	getExportTable = (cuDriverGetExportTable_pfn) dlsym(gs_toolsapi.handle, "cuDriverGetExportTable");
	if (!getExportTable)
	{
		fprintf(stderr, "Failed to load function 'cuDriverGetExportTable' from libcuda.so\n");
		return 1;
	}

	/*
		Now we retrieve the required tables for TAUCuda callback.
		Contexttable and Device table serve a closely related purpose for finding 
		device related information. The coretable contains the interfaces for 
		controlling the profiler callbacks. 
	*/

	status = getExportTable(&cuToolsApi_ETID_Context, (const void**) &gs_toolsapi.contextTable);
	if (status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Failed to load Context table\n");
		return 1;
	}

	status = getExportTable(&cuToolsApi_ETID_Device, (const void**) &gs_toolsapi.deviceTable);
	if (status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Failed to load device table\n");
		return 1;
	}
	status = getExportTable(&cuToolsApi_ETID_Core, (const void**) &gs_toolsapi.coreTable);
	if (status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Failed to load core table %X\n", gs_toolsapi.coreTable);
		return 1;
	}

	if (!gs_toolsapi.coreTable->Construct())
	{
		fprintf(stderr, "Failed to initialize tools API\n");
		return 1;
	}

	/*
		initialize the synchonization mechanism
		This gurads updation in the global list of event managers. 
	*/
    /*
			 setting the node is helps in getting the correct profile file name.
			 Also top level timer is created here as the application event.  
    */ 
    /*
			 Here is some environment variable setting for CUDA to enable profiler and callbacks. 
			 This is a default setting which will deliver only the default GPU counters. If we want 
			 to have more counters , either environment variable or the configuration file needs to 
			 be setup.    
    */
		putenv("CUDA_PROFILE=1");
		putenv("CUDA_PROFILE_CALLBACKS=1");			   
    //_putenv("CUDA_PROFILE_CONFIG=C:\\sdk\\NVIDIA GPU Computing SDK-2.3\\C\\bin\\win32\\Debug\\profiler.cfg");
    	
    /*if (InitializeToolsApi() != 0)
    {
        return TAUCUDA_INIT_FAILED;
    }*/

    NvU64 subscriptionId = 0;
		/*
			 The driver library has been already initialized and we subscribe our 
        callback handler here. The callbacks will start only after enabling it. 	
    */ 
    if (!GetCoreTable()->SubscribeCallbacks(&callback_handle, 0, &subscriptionId))
    {
				printf("failed to subscribe callback.\n");
        return TAUCUDA_INIT_FAILED;
    }

    if (!GetCoreTable()->EnableCallbacks(true))
    {
				printf("failed to enable callback.\n");
        return TAUCUDA_INIT_FAILED;
    }
    /*
			 We create user events for keeping track of the memory transfer. 
			 Please note that the user events can not be defined anywhere. 
			 It will not work if you try to change the place where you define them. 
			 It's already defined by now and we are assigning them to the global pointers.  
    */
    if(!user_events)
    {
			/*send_message=Tau_get_userevent("TAUCUDA_MEM_SEND");
				rcv_message=Tau_get_userevent("TAUCUDA_MEM_RCV");											
				message_size=Tau_get_userevent("TAUCUDA_COPY_MEM_SIZE");
			 */
			user_events=true;
		}
		
		//tau specific initializations	
		tau_gpu_init();
		
		
		return TAUCUDA_SUCCESS;
}
void onload(void)
{
#ifdef DEBUG_PROF
	fprintf(stdout, "on load: %d.\n", load_count);
#endif
	if(load_count==0)
	{
		InitializeToolsApi();
		load_count++;
	}
	else
	{
		load_count++;
	}
}

void onunload(void)
{
}

void shutdown_tool_api(void)
{
	tau_gpu_exit();	
	//gs_toolsapi.managers.clear();
	if (gs_toolsapi.coreTable)
	{
		gs_toolsapi.coreTable->Destruct();
	}
	if (gs_toolsapi.handle)
	{
		dlclose(gs_toolsapi.handle);
	}
}


