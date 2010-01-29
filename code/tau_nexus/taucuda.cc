
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
#include <stdio.h>
#include<stdlib.h>
#include<gpuevents.h>
#include<dlfcn.h>
#include<string.h>
#include <linux/unistd.h>
#include<pthread.h>
#include<TAU.h>
#include<iostream>
#include<sys/time.h>

ToolsAPI gs_toolsapi;
int global_thread_id=0;
pthread_mutex_t event_mutex;
pthread_mutexattr_t event_mutex_attr;
int load_count=0;
bool tau_nexus=false;
bool clock_sync=false;

extern x_uint64 TauTraceGetTimeStamp(int tid);
extern "C" void TauTraceEventOnly(long int ev, x_int64 par, int tid);

__thread int ltid=-1;
__thread EventManager *my_manager=NULL;
__thread bool registered=false;
bool user_events=false;

void * send_message, * rcv_message, * message_size;

/*

Generate unique tid for each thread

*/

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
	exit handler
*/
void handle_exit()
{
	Tau_stop_top_level_timer_if_necessary();						
}

/*
	initialize the handler for application exit.
*/
void set_exit()
{
	int status = atexit(handle_exit);
        if (status != 0) {
             perror("cannot set exit function\n");
        }
}

/*
	Intercepted pthread kickstart routine
*/
void* tau_start_routine(void *my_arg)
{	
	/*
		The thread registration is done in the kickstart routine as the new thread context 
		is visible from this routine. We can also have the top level event for the thread.
	*/
	//Tau_set_thread(gettid());
	TAU_REGISTER_THREAD();
	wrap_routine_arg *tau_arg=(wrap_routine_arg *)my_arg;
    	//Tau_create_top_level_timer_if_necessary();
        dfprintf(stdout, "Before calling APPSTART \n");
	//TAU_START("THREADAPP");	
	void * status= tau_arg->start_routine(tau_arg->arg);
        dfprintf(stdout, "Before calling APPSTOP \n");
	//TAU_STOP("THREADAPP");	
        dfprintf(stdout, "AFter calling APPSTOP \n");
	//Tau_stop_top_level_timer_if_necessary();					
	delete tau_arg;
	return status;	
}

/*
	Intercepting pthread create and replacing with our start routine
*/

int pthread_create(pthread_t * my_thread, const pthread_attr_t * my_attr, void *(*start_routine)(void*), void * my_arg)
{
	/*
		This library is ld_preloaded and intercepts the pthread create routine. 
		Internally it opens the pthread library and invokes the pthread_create function. 	
	*/
	void * handle = dlopen("libpthread.so.0", RTLD_GLOBAL | RTLD_NOW);
	if(!handle)
	{
                fprintf(stderr, "Failed to load libpthread.so >> %s\n", dlerror());
		return -1;
	} 	
	PTHREAD_CREATE_PTR pt_create=(PTHREAD_CREATE_PTR)dlsym(handle, "pthread_create");
	if(!pt_create)
	{
                fprintf(stderr, "Failed to get symbol pthread_create\n");
		return -1;
	}
	/*
		A wrapper is introduced in the thread routine to instrument with TAU APIs.
	*/
	wrap_routine_arg *tau_arg=new wrap_routine_arg;
	tau_arg->start_routine=start_routine;
	tau_arg->arg=my_arg;
	return pt_create(my_thread, my_attr, tau_start_routine, tau_arg);	
}

/*
   No interception of joing as of now might need later

*/

/*int pthread_join(pthread_t thread, void **value_ptr)
{
	int status;
	void * handle = dlopen("libpthread.so.0", RTLD_GLOBAL | RTLD_NOW);
	if(!handle)
	{
                fprintf(stderr, "Failed to load libpthread.so >> %s\n", dlerror());
		return -1;
	}
 	
	PTHREAD_JOIN_PTR pt_join=(PTHREAD_JOIN_PTR)dlsym(handle, "pthread_join");
	if(!pt_join)
	{
                fprintf(stderr, "Failed to get symbol pthread_join\n");
		return -1;
	}

	status=pt_join(thread, value_ptr);

	//Tau_stop_top_level_timer_if_necessary();

	return status;	
}*/

/*
  event manager pointer is maintained per thread as 
  thread local variable. It gets populated when called for the first time 
*/ 

inline EventManager * GetEventManager()
{
	if(my_manager)
		return my_manager;
	my_manager=new EventManager();
	TAU32 c_tid=gettid();
	my_manager->SetThread(c_tid);
	pthread_mutex_lock(&event_mutex);
	gs_toolsapi.managers.insert(gs_toolsapi.managers.end(),my_manager);
        pthread_mutex_unlock(&event_mutex);
	return my_manager;
}

/*
	This initializes the cuda callback mechanism
*/

inline int InitializeToolsApi(void)
{
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
	
	pthread_mutexattr_init(&event_mutex_attr);
        pthread_mutex_init(&event_mutex,&event_mutex_attr);
        return 0;
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

inline ClockTable& GetClockTable(int device)
{
        return gs_toolsapi.device_clocks[device];
}

#define SYNCH_LATENCY 1
/*
	not used currently
*/
inline TAU64 GetCPUTime(timeval & time1, timeval & time2)
{
	TAU64 my_time;
	my_time=time1.tv_usec;
	my_time+=time2.tv_usec;
	my_time+=time1.tv_sec*1000000;	
	my_time+=time2.tv_sec*1000000;
	my_time = my_time/2;
	return my_time;	
}

/* 
	Synchronizes the clocks. 
	This takes care of device to device or 
        CPU to device time alignment. 
*/
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

/*
	converts the device time into CPU aligned time. 
	It uses the clock table built in in the ClockSynch
*/

inline TAU64 GetCPUAlignedTime(int device, TAU64 raw_time)
{
	if(device<0 || device >gs_toolsapi.device_count)
		return raw_time;
	TAU64 time_val=(GetClockTable(device).tau_end_time-GetClockTable(device).tau_start_time);
	TAU64 time_tmp=(GetClockTable(device).gpu_end_time-GetClockTable(device).gpu_start_time);
	if(raw_time > GetClockTable(device).gpu_start_time)
	{
		time_val*=(raw_time-GetClockTable(device).gpu_start_time);
		time_val=time_val/time_tmp;	
		time_val+=GetClockTable(device).tau_start_time; 
	}
	else
	{
		time_val*=(GetClockTable(device).gpu_start_time-raw_time);
		time_val=time_val/time_tmp;	
		time_val=GetClockTable(device).tau_start_time-time_val; 	
	}
        dfprintf(stdout, "cpu_start=%llu cpu_end_time=%llu gpu_start=%llu raw_time =%llu\n",GetClockTable(device).tau_start_time,
					GetClockTable(device).tau_end_time , GetClockTable(device).gpu_start_time, raw_time );
	return time_val;							
}

/*
	It retrieves the time aligned with device 0 
*/

inline TAU64 GetGPUAlignedTime(int device, TAU64 raw_time)
{
	if(device<=0 || device >gs_toolsapi.device_count)
		return raw_time;			
	TAU64 time_val=(GetClockTable(device).ref_gpu_start_time+raw_time-GetClockTable(device).gpu_start_time);
	return time_val;	
}

TAU64 AlignedTime(int device,TAU64 raw_time)
{
	//return GetGPUAlignedTime(device,raw_time);
	return GetCPUAlignedTime(device,raw_time);
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
        cuToolsApi_EnterGenericInParams* clbkParameter = (cuToolsApi_EnterGenericInParams*) inParams;
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
			type=KERNEL;
	}
	if(type!=OTHERS)
	{	
		TAU32 device_id;
		//extract the device ID for the curent context
		GetContextTable()->CtxGetDevice(clbkParameter->ctx,&device_id);
		/*
			Hand down the information of the API to the EventManager. 
			Note that the event manager object is per thread and the pointter 
			is stored as thread local variable.  
		*/		
		GetEventManager()->APIEntry((TAU64)clbkParameter->ctx,(TAU64)clbkParameter->stream, 
					clbkParameter->apiCallId, type,(char *) clbkParameter->functionName,device_id);
	}
	tau_nexus=true;
	//TAU_STATIC_PHASE_START((char *)clbkParameter->functionName);
	/*
		Just trigger the start of event for TAU event
		This will generate TAU CPU events for the CUDA driver API calls. 
	*/ 
	TAU_START((char *)clbkParameter->functionName);
	if(type==DATA2D && GetEventManager()->IsTraceEnabled())
	{
		/*
			For tracing memory events extra care needs to be taken here. 
			We generate user defined event in the TAU trace to mark the start 
                        of memory transfer. Here we spit out three events as the API takes 
			only one parameter at a time while we need multiple parameters to 
			match up later with the TAUCuda trace.  
		*/

			MemCpy2D *my_params=(MemCpy2D *)clbkParameter->params;
			TauUserEvent *my_ev = (TauUserEvent*)send_message;
			TauUserEvent *my_size = (TauUserEvent*)message_size;
			
			TauTraceEventOnly(my_ev->GetEventId(),(TAU64)clbkParameter->ctx ,RtsLayer::getTid());	
			TauTraceEventOnly(my_ev->GetEventId(),(TAU64)clbkParameter->apiCallId ,RtsLayer::getTid());	
			TauTraceEventOnly(my_size->GetEventId(),my_params->count ,RtsLayer::getTid());
			
			//Tau_userevent(send_message,(double) ((TAU64)clbkParameter->ctx));		
			//Tau_userevent(send_message,(double)clbkParameter->apiCallId);		
			//Tau_userevent(message_size,(double)my_params->count);		
        }	
    }
    if (*callbackId == cuToolsApi_CBID_ExitGeneric) 
    {
	// We just trigger the end for the event which was started in the entry of the API above
        dfprintf(stdout, "cuToolsApi_CBID_ExitGeneric %x \n", inParams);
	cuToolsApi_EnterGenericInParams* clbkParameter = (cuToolsApi_EnterGenericInParams*) inParams;
	//TAU_STATIC_PHASE_STOP((char *)clbkParameter->functionName);
	TAU_STOP((char *)clbkParameter->functionName);
    }
    if (*callbackId == cuToolsApi_CBID_ProfileLaunch)
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
        cuToolsApi_ProfileLaunchInParams* clbkParameter = (cuToolsApi_ProfileLaunchInParams*) inParams;
        dfprintf(stdout, "ProfileLaunch\n");
        dfprintf(stdout, "\t%-40s%f\n", "occupancy", clbkParameter->occupancy);
	TAU32 g_size=clbkParameter->gridSizeX*clbkParameter->gridSizeY;
	TAU32 b_size=clbkParameter->blockSizeX*clbkParameter->blockSizeY*clbkParameter->blockSizeZ;

	if(GetEventManager()->IsTraceEnabled())
	{
		//If trace enabled then invoke trace manager
		GetEventManager()->KernelTraceEvent((TAU64)clbkParameter->ctx,clbkParameter->streamId,
			clbkParameter->apiCallId,(char *) clbkParameter->methodName, clbkParameter->startTime,clbkParameter->endTime,
			g_size , b_size,clbkParameter->staticSharedMemPerBlock,clbkParameter->dynamicSharedMemPerBlock,
			clbkParameter->registerPerThread, clbkParameter->occupancy);
	}
	else
	{
		/* if profiling then invoke profile manager
		   Two routines are used here which was a bad choice just to make things look better. 
		   Didnt change it as it's working fine. Also please note that we want to keep the 
		   eventmanager	take inputs which are independant of the CUDA environment. So in 
		   future changing CUDA callback mechinsm will need limited chgange in eventmanager module.   
		*/
		GetEventManager()->KernelProfileEvent((TAU64)clbkParameter->ctx,clbkParameter->streamId,
			clbkParameter->apiCallId,(char *) clbkParameter->methodName, clbkParameter->startTime,clbkParameter->endTime);
		GetEventManager()->UpdateKernelProfile(g_size , b_size,clbkParameter->staticSharedMemPerBlock,
			clbkParameter->dynamicSharedMemPerBlock,clbkParameter->registerPerThread, clbkParameter->occupancy);
	}
    }
    if (*callbackId == cuToolsApi_CBID_ProfileMemcpy)
    {
	/*
		again check for the clock to be synchronized.
	*/
	if(!clock_sync)
	{
    		ClockSynch();
		clock_sync=true;
	}
	cuToolsApi_ProfileMemcpyInParams *memParam = (cuToolsApi_ProfileMemcpyInParams *) inParams; 
	dfprintf(stdout, "cuToolsApi_CBID_ProfileMemcpy\n");
	/*
		hand down the memory profile information to the event manager.
		For this case the tracing and profiling options are taken by this routine. 
	*/
	GetEventManager()->MemProfileEvent((TAU64)memParam->ctx,memParam->streamId, 
		memParam->apiCallId,memParam->startTime,memParam->endTime, memParam->memTransferSize);
    }
}

/*
	Routine executed when the library gets loaded 
*/

void onload(void)
{
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
	send_message=Tau_get_userevent("TAUCUDA_MEM_SEND");
	rcv_message=Tau_get_userevent("TAUCUDA_MEM_RCV");											
	message_size=Tau_get_userevent("TAUCUDA_COPY_MEM_SIZE");											
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
		return;
	/*
		stop the top level timer which is a dummy event 
		useful for profile/trace analysis
	*/
	Tau_stop_top_level_timer_if_necessary();
	//TAU_STATIC_PHASE_STOP(".TAUCudaApplication");					
	list<EventManager*>::iterator it;
	for(it=gs_toolsapi.managers.begin();it!=gs_toolsapi.managers.end();it++)
	{
		/*
			This is walking through the global list of event managers.
			and call the exit routine which will write out profiles. 
			Finally delete the event manager object.    
		*/
		EventManager *event_manager=*it;
		event_manager->ThreadExit();
		delete event_manager;
	}
	gs_toolsapi.managers.clear();
	ShutdownToolsApi();
}
