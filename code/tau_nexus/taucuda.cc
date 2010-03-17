/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory                                        **
****************************************************************************/
/***************************************************************************
**      File            : taucuda.cc                                      **
**      Description     : TAU trace format reader library header files    **
**      Author          : Shangkar Mayanglambam                           **
**      Contact         : smeitei@cs.uoregon.edu                          **
***************************************************************************/



#include "taucuda.h"
#include "TAU.h"
#include <Profile/TauInit.h>

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

void *main_ptr, *gpu_ptr;
//static TauUserEvent* (*MemoryCopyEventHtoD)(void);
//static TauUserEvent* (*MemoryCopyEventDtoH)(void);
TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
int gpuTask;
bool firstEvent = true;

#include <linux/unistd.h>
#include<dlfcn.h>

extern void metric_set_gpu_timestamp(int tid, double value);

#include<map>
using namespace std;

#define SYNCH_LATENCY 1

//#define CPU_THREAD 0
//#define GPU_THREAD 1

/* create TAU callback routine to capture both CPU and GPU execution time 
	takes the thread id as a argument. */

// a seperate counter for each GPU.
//double gpu_timestamp[TAU_MAX_THREADS];
double cpu_start_time;

#define MemcpyHtoD false
#define MemcpyDtoH true

struct MemMapKey
{
	NvU64 contextId;
	NvU64 callId;

	MemMapKey(const NvU64 a, const NvU64 b) :
		contextId(a), callId(b) {}

	bool operator<(const MemMapKey& A) const
	{ 
		if (contextId == A.contextId)
		{
			return callId<A.callId; 
		}
		else
			return contextId<A.contextId;
	}
};

typedef map<MemMapKey, bool> doubleMap;
doubleMap MemcpyEventMap;

map<const char*, void*> events;

extern void metric_set_gpu_timestamp(int tid, double value);

double cpu_time()
{
	//get time from the CPU clock
	struct timeval tp;
	gettimeofday(&tp, 0);
	//printf("CPU time: %f \n", ((double)tp.tv_sec * 1e6 + tp.tv_usec));
	//printf("subtraction: %f \n", cpu_start_time);
	//printf("CPU time (2): %f \n", ((double)tp.tv_sec * 1e6 + tp.tv_usec) - cpu_start_time);
	return ((double)tp.tv_sec * 1e6 + tp.tv_usec);
}


extern x_uint64 TauTraceGetTimeStamp(int tid);

inline cuToolsApi_Device* GetDeviceTable(void)
{
        return gs_toolsapi.deviceTable;
}
inline ClockTable& GetClockTable(int device)
{
	return gs_toolsapi.device_clocks[device];
}
double AlignedTime(int device, double raw_time)
{
	double offset = gs_toolsapi.device_clocks[device].tau_end_time -
		gs_toolsapi.device_clocks[device].ref_gpu_end_time;
	/*printf("clock sync offset: \t%f.\n", offset);
	printf("raw time: \t\t%f.\n", raw_time);
	printf("adjusted time: \t\t%f.\n", raw_time + offset);*/
	return (double) raw_time + offset;
}


void ClockSynch()
{
	//timeval cpu_time1, cpu_time2;
	double cpu_time1, cpu_time2;
	TAU64 ref_t1, ref_t2;
	GetDeviceTable()->DeviceGetCount(&(gs_toolsapi.device_count));
	for(int i=0;i<gs_toolsapi.device_count;i++)
	{
		//gettimeofday(&cpu_time1,NULL);
		cpu_time1=cpu_time();
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t1);
		//GetDeviceTable()->DeviceGetTimestamp(i,&(gs_toolsapi.device_clocks[i].gpu_end_time));
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t2);
		//printf("GPU time [1]: %f.\n", (double) ref_t1);
		//printf("GPU time [2]: %f.\n", (double) ref_t2);
		//gettimeofday(&cpu_time2,NULL);				
		cpu_time2=cpu_time();
		//gs_toolsapi.device_clocks[i].tau_end_time=GetCPUTime(cpu_time1, cpu_time2);
		gs_toolsapi.device_clocks[i].tau_end_time=(cpu_time1+cpu_time2)/2;
		gs_toolsapi.device_clocks[i].ref_gpu_end_time=((double)ref_t1+(double)ref_t2)/2e3;			

		/*printf("for device %d, sync is \tCPU=%f \n\t\t\tGPU=%f", i,
				gs_toolsapi.device_clocks[i].tau_end_time,
				gs_toolsapi.device_clocks[i].ref_gpu_end_time);*/

	}
}
inline TAU32 gettid(void)
{
	if(ltid>=0)
		return ltid;
	pthread_mutex_lock(&event_mutex);
        ltid=global_thread_id++;
        pthread_mutex_unlock(&event_mutex);
	return ltid;
       // return syscall(__NR_gettid);	
}

/*
	Close down the interception mechanism 
*/

inline int ShutdownToolsApi()
{
	if (gs_toolsapi.coreTable)
	{
		gs_toolsapi.coreTable->Destruct();
	}
	if (gs_toolsapi.handle)
	{
		dlclose(gs_toolsapi.handle);
	}

	return 0;
}

/*
	Helper function to get Coretable
*/
inline cuToolsApi_Core* GetCoreTable(void)
{
	return gs_toolsapi.coreTable;
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


void EnterGenericEvent(cuToolsApi_EnterGenericInParams *clbkParameter)
{
	event_type type;
	/*
		If the API is memory transfer check up for the API name and identify if it's 
		transfer from the device or to the device. It would have been better if NVIDIA provided 
		IDs for the API name. We are just handling limited number of memcpy APIS here. We need 
		to take care of rest of the memcpy APIs too. 
	*/
	if(strncmp(clbkParameter->functionName,"cuMemcpy", sizeof("cuMemcpy")-1)==0)
	{
		type=DATA;
		NvU64 contextId;
		GetContextTable()->CtxGetId(clbkParameter->ctx, &contextId);

		if(strncmp(clbkParameter->functionName,"cuMemcpyHtoD", sizeof("cuMemcpyHtoD")-1)==0)
		{
			type=DATA2D;
			MemMapKey m(contextId, clbkParameter->apiCallId);
			MemcpyEventMap.insert(make_pair(m, MemcpyHtoD));
			//TAU_EVENT(MemoryCopyEventHtoD(), ((MemCpy2D *) clbkParameter->params)->count)
			
			/*printf("registering Memory copy Host to Device: %lld, %lld.\n", contextId,
			clbkParameter->apiCallId);*/
		}
		else if(strncmp(clbkParameter->functionName,"cuMemcpyDtoH",sizeof("cuMemcpyDtoH")-1)==0)
		{
			type=DATAFD;
			MemMapKey m(contextId, clbkParameter->apiCallId);
			MemcpyEventMap.insert(make_pair(m, MemcpyDtoH));
			//TAU_EVENT(MemoryCopyEventDtoH, ((MemCpy2D *) clbkParameter->params)->count)
			
			/*printf("registering Memory copy Device to Host: %lld, %lld.\n", contextId,
			clbkParameter->apiCallId);*/
		}
	}
	else
	{
		type=OTHERS;
		if(strncmp(clbkParameter->functionName,"cuLaunchGrid",sizeof("cuLaunchGrid")-1)==0)
		{
			type=KERNEL;
		}
	}
	if(type!=OTHERS)
	{	
		//TAU32 device_id;
		//extract the device ID for the curent context
		//GetContextTable()->CtxGetDevice(clbkParameter->ctx,&device_id);
		
	}
	tau_nexus=true;
	TAU_START((char *)clbkParameter->functionName);
	//TAU_REGISTER_EVENT(ev, "Thread accesses");
	//TAU_EVENT(ev, 480.00000);
	/* Do not have the size of the memory transfer at EnterGeneric.
	if (type==DATA2D)
	{
		TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory Copied from Host to Device");
		TAU_EVENT(MemoryCopyEventHtoD, clbkParameter->memTransferSize);
	}
	else if (type==DATAFD)
	{
		TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory Copied from Device to Host");
		TAU_EVENT(MemoryCopyEventDtoH, clbkParameter->memTransferSize);
	}*/
}

void ExitGenericEvent(cuToolsApi_EnterGenericInParams *clbkParameter)
{
	TAU_STOP((char *)clbkParameter->functionName);
}
void start_gpu_event(const char *name)
{
	//printf("staring %s event.\n", name);
	map<const char*,void*>::iterator it = events.find(name);
	if (it == events.end())
	{
		void *ptr;
		TAU_PROFILER_CREATE(ptr, name, "", TAU_USER);
		TAU_PROFILER_START_TASK(ptr, gpuTask);
		events[name] = ptr;
	} else
	{
		void *ptr = (*it).second;
		TAU_PROFILER_START_TASK(ptr, gpuTask);
	}
}
void stage_gpu_event(const char *name, double start_time, int device)
{
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

	metric_set_gpu_timestamp(gpuTask, AlignedTime(device, start_time));
	
	if (firstEvent)
	{
		//printf("first gpu event.\n");
		TAU_PROFILER_START_TASK(gpu_ptr, gpuTask);
		firstEvent = false;
	}
	//RtsLayer::LockDB();
	start_gpu_event(name);
}
void stop_gpu_event(const char *name)
{
	//printf("stopping %s event.\n", name);
	map<const char*,void*>::iterator it = events.find(name);
	if (it == events.end())
	{
		printf("FATAL ERROR in stopping GPU event.\n");
	} else
	{
		void *ptr = (*it).second;
		TAU_PROFILER_STOP_TASK(ptr, gpuTask);
	}
}
void break_gpu_event(const char *name, double stop_time, int device)
{
	metric_set_gpu_timestamp(gpuTask, AlignedTime(device, stop_time));
	stop_gpu_event(name);
}

void ProfileLaunchEvent(cuToolsApi_ProfileLaunchInParams *clbkParameter)
{
	TAU32 device;
	GetContextTable()->CtxGetDevice(clbkParameter->ctx,&device);
	
	stage_gpu_event(clbkParameter->methodName, 
		(double)clbkParameter->startTime/1000, (int) device);

	break_gpu_event(clbkParameter->methodName,
			(double)clbkParameter->endTime/1000, (int) device);
}

void ProfileMemcpyEvent(cuToolsApi_ProfileMemcpyInParams *clbkParameter)
{
	TAU32 device;
	GetContextTable()->CtxGetDevice(clbkParameter->ctx,&device);
	
	MemMapKey m(clbkParameter->contextId, clbkParameter->apiCallId);
	doubleMap::const_iterator it = MemcpyEventMap.find(m);
	/*printf("tiggering Memory copy: %lld, %lld\t",
			clbkParameter->contextId,
			clbkParameter->apiCallId);*/


	if (it != MemcpyEventMap.end())
	{
		if (it->second == MemcpyHtoD) {
			stage_gpu_event("cuda Memory copy Host to Device", 
					(double)clbkParameter->startTime/1000, (int) device);
			TAU_EVENT_THREAD(MemoryCopyEventHtoD(), (double)
					clbkParameter->memTransferSize, gpuTask);
			break_gpu_event("cuda Memory copy Host to Device",
					(double)clbkParameter->endTime/1000, (int) device);
		}
		else {
			stage_gpu_event("cuda Memory copy Device to Host", 
					(double)clbkParameter->startTime/1000, (int) device);
			TAU_EVENT_THREAD(MemoryCopyEventDtoH(), (double)
					clbkParameter->memTransferSize, gpuTask);
			break_gpu_event("cuda Memory copy Device to Host",
					(double)clbkParameter->endTime/1000, (int) device);
		}
	} else 
	{
		printf("ERROR: cannot find matching memcopy event.\n");
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
	// is the current thread registered otherwise increment the thread ID
	// A virtual thread ID is being generated here it might be of use in the future
	if(!registered)
	{
		gettid();
		registered=true;
	}
	if (*callbackId == cuToolsApi_CBID_EnterGeneric)
	{
		//printf("entering EnterGenericEvent.\n");
		EnterGenericEvent((cuToolsApi_EnterGenericInParams*) inParams);
	}
	else if (*callbackId == cuToolsApi_CBID_ExitGeneric)
	{
		//printf("entering ExitGenericEvent.\n");
		ExitGenericEvent((cuToolsApi_EnterGenericInParams*) inParams);
	}
	else if (*callbackId == cuToolsApi_CBID_ProfileLaunch)
	{
		ProfileLaunchEvent((cuToolsApi_ProfileLaunchInParams*) inParams);
	}
	else if (*callbackId == cuToolsApi_CBID_ProfileMemcpy)
	{
		ProfileMemcpyEvent((cuToolsApi_ProfileMemcpyInParams*) inParams);
	}
}

/*
	Routine executed when the library gets loaded 
*/

void onload(void)
{
	fprintf(stdout, "on load...\n");
	if(load_count==0)
	{
		tau_cuda_init();
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

/*
	This initializes the cuda callback mechanism
*/

inline int InitializeToolsApi(void)
{
	printf("Initializing...\n");
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

//	TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
//	TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
	/*
		initialize the synchonization mechanism
		This gurads updation in the global list of event managers. 
	*/
	TAU_PROFILE_SET_NODE(0);
	TAU_PROFILER_CREATE(main_ptr, "main", "", TAU_USER);
	TAU_PROFILER_CREATE(gpu_ptr, "gpu elapsed time", "", TAU_USER);
	//InitializeTAU();

	/* Create a seperate GPU task */
	TAU_CREATE_TASK(gpuTask);

	/* Register our time callback */
	//TAU_CREATE_USER_CLOCK("TAUCUDA_TIME", taucuda_time);

	printf("Created user clock.\n");
    
  TAU_PROFILER_START(main_ptr);	

	
	printf("started main.\n");
	return 0;
}
/*
	Initialization routine for taucuda
*/
int tau_cuda_init(void)
{
    int x;
    // check if current thread is accounted 
    // otherwise generate the first thread ID. 
    if(!registered)
    {
	gettid();
	registered=true;
    }
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
    	
    if (InitializeToolsApi() != 0)
    {
        return TAUCUDA_INIT_FAILED;
    }

    NvU64 subscriptionId = 0;
    /*
	The driver library has been already initialized and we subscribe our 
        callback handler here. The callbacks will start only after enabling it. 	
    */ 
    if (!GetCoreTable()->SubscribeCallbacks(&callback_handle, 0, &subscriptionId))
    {
        return TAUCUDA_INIT_FAILED;
    }

    if (!GetCoreTable()->EnableCallbacks(true))
    {
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
    return TAUCUDA_SUCCESS;
}


/*
	finalization routine for taucuda
*/
void tau_cuda_exit(void)
{
	/*const char **names;
	int num;
	TAU_GET_EVENT_NAMES(names, num);
	printf("exit events are: %s.\n", names[0]);*/
	
	if(!tau_nexus)
	{
		return;
	}
	/*
		stop the top level timer which is a dummy event 
		useful for profile/trace analysis
	*/
		printf("Stopping first gpu event.\n");
		TAU_PROFILE_EXIT("cuda");
		TAU_PROFILER_STOP_TASK(gpu_ptr, gpuTask);
		TAU_PROFILER_STOP(main_ptr);
	//Tau_stop_top_level_timer_if_necessary();
	//TAU_STATIC_PHASE_STOP(".TAUCudaApplication");					
	list<EventManager*>::iterator it;
	for(it=gs_toolsapi.managers.begin();it!=gs_toolsapi.managers.end();it++)
	{
		/*
			This is walking through the global list of event managers.
			and call the exit routine which will write out profiles. 
			Finally delete the event manager object.    
		*/
		//EventManager *event_manager=*it;
		//event_manager->ThreadExit();
		//delete event_manager;
	}
	gs_toolsapi.managers.clear();
	ShutdownToolsApi();
}
