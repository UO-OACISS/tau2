
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

int gpuTask;

#include <linux/unistd.h>
#include<dlfcn.h>

#define SYNCH_LATENCY 1

#define CPU_THREAD 0
#define GPU_THREAD 1

/* create TAU callback routine to capture both CPU and GPU execution time 
	takes the thread id as a argument. */

// a seperate counter for each GPU.
double gpu_timestamp[TAU_MAX_THREADS];

double taucuda_time(int tid)
{
	if (tid == CPU_THREAD)
	{	
		printf("CPU time\n");
		//get time from the CPU clock
		struct timeval tp;
	  gettimeofday(&tp, 0);
		return ((double)tp.tv_sec * 1e6 + tp.tv_usec);
	}
	// get time from the callback API 
	else
	{
		printf("GPU time\n");
		return gpu_timestamp[tid];
	}
}


extern x_uint64 TauTraceGetTimeStamp(int tid);

inline cuToolsApi_Device* GetDeviceTable(void)
{
        return gs_toolsapi.deviceTable;
}

void ClockSynch()
{
	//timeval cpu_time1, cpu_time2;
	TAU64 cpu_time1, cpu_time2;
	TAU64 ref_t1, ref_t2;
	GetDeviceTable()->DeviceGetCount(&(gs_toolsapi.device_count));
	for(int i=0;i<gs_toolsapi.device_count;i++)
	{
		//gettimeofday(&cpu_time1,NULL);
		cpu_time1=TauTraceGetTimeStamp(0);
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t1);
		GetDeviceTable()->DeviceGetTimestamp(i,&(gs_toolsapi.device_clocks[i].gpu_start_time));
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t2);
		//gettimeofday(&cpu_time2,NULL);
		cpu_time2=TauTraceGetTimeStamp(0);
		//gs_toolsapi.device_clocks[i].tau_start_time=GetCPUTime(cpu_time1, cpu_time2);
		gs_toolsapi.device_clocks[i].tau_start_time=(cpu_time1+cpu_time2)/2;
		gs_toolsapi.device_clocks[i].ref_gpu_start_time=(ref_t1+ref_t2)/2;			
		sleep(SYNCH_LATENCY);
		//gettimeofday(&cpu_time1,NULL);
		cpu_time1=TauTraceGetTimeStamp(0);
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t1);
		GetDeviceTable()->DeviceGetTimestamp(i,&(gs_toolsapi.device_clocks[i].gpu_end_time));
		GetDeviceTable()->DeviceGetTimestamp(0,&ref_t2);
		//gettimeofday(&cpu_time2,NULL);				
		cpu_time2=TauTraceGetTimeStamp(0);
		//gs_toolsapi.device_clocks[i].tau_end_time=GetCPUTime(cpu_time1, cpu_time2);
		gs_toolsapi.device_clocks[i].tau_end_time=(cpu_time1+cpu_time2)/2;
		gs_toolsapi.device_clocks[i].ref_gpu_end_time=(ref_t1+ref_t2)/2;			
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

inline ClockTable& GetClockTable(int device)
{
	return gs_toolsapi.device_clocks[device];
}

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
		if(strncmp(clbkParameter->functionName,"cuMemcpyHtoD", sizeof("cuMemcpyHtoD")-1)==0)
		{
			type=DATA2D;
		}
		else if(strncmp(clbkParameter->functionName,"cuMemcpyDtoH",sizeof("cuMemcpyDtoH")-1)==0)
		{
			type=DATAFD;
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
		TAU32 device_id;
		//extract the device ID for the curent context
		GetContextTable()->CtxGetDevice(clbkParameter->ctx,&device_id);
	}
	tau_nexus=true;
	TAU_START((char *)clbkParameter->functionName);
}

void ExitGenericEvent(cuToolsApi_EnterGenericInParams *clbkParameter)
{
	TAU_STOP((char *)clbkParameter->functionName);
}
void ProfileLaunchEvent(cuToolsApi_ProfileLaunchInParams *clbkParameter)
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
	
	gpu_timestamp[gpuTask] = ((double)clbkParameter->startTime/1000);
	TAU_START_TASK(clbkParameter->methodName, gpuTask);
	
	gpu_timestamp[gpuTask] = ((double)clbkParameter->endTime/1000);
	TAU_STOP_TASK(clbkParameter->methodName, gpuTask);

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

	/*
		initialize the synchonization mechanism
		This gurads updation in the global list of event managers. 
	*/
	

	/* Create a seperate GPU task */
	TAU_CREATE_TASK(gpuTask);

	/* Register our time callback */
	TAU_SET_USER_CLOCK_CALLBACK(taucuda_time);

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
    TAU_PROFILE_SET_NODE(0);
    Tau_create_top_level_timer_if_necessary();	
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
	
	if(!tau_nexus)
	{
		return;
	}
	/*
		stop the top level timer which is a dummy event 
		useful for profile/trace analysis
	*/
		TAU_PROFILE_EXIT("cuda");
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
