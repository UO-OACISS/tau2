/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory                                        **
****************************************************************************/
/***************************************************************************
**      File            : TauGpu.cpp                                      **
**      Description     : TAU trace format reader library header files    **
**      Author          : Shangkar Mayanglambam                           **
**                      : Scott Biersdorff                                **
**      Contact         : scottb@cs.uoregon.edu                           **
***************************************************************************/

#include "TauGpu.h"
#include "TAU.h"
#include "Profile/TauTrace.h"
#include <TauInit.h>
#include <stdio.h>
#include <iostream>

void *main_ptr, *gpu_ptr;

TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventHtoD, "Bytes copied from Host to Device");
TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventDtoH, "Bytes copied from Device to Host");

int gpuTask;
bool firstEvent = true;

//The number of Memcpys called with unknown transfer size which should be given
//on the GPU thread.
int counted_memcpys = 0;

#include <linux/unistd.h>

extern void metric_set_gpu_timestamp(int tid, double value);

#include<map>
using namespace std;

double cpu_start_time;

struct EventName {
		const char *name;
		EventName(const char* n) :
			name(n) {}	
		bool operator<(const EventName &c1) const { return strcmp(name,c1.name) < 0; }
};

//typedef map<eventId, bool> doubleMap;
//doubleMap MemcpyEventMap;

map<EventName, void*> events;

extern void metric_set_gpu_timestamp(int tid, double value);


void check_gpu_event()
{
	if (firstEvent)
	{
#ifdef DEBUG_PROF
		cerr << "first gpu event" << endl;
#endif
		TAU_PROFILER_START_TASK(gpu_ptr, gpuTask);
		firstEvent = false;
	}
}

/* === Begin implementing the hooks === */

/* create TAU callback routine to capture both CPU and GPU execution time 
	takes the thread id as a argument. */

void Tau_gpu_enter_event(const char* name, eventId *id)
{
#ifdef DEBUG_PROF
	printf("entering cu event: %s.\n", name);
#endif
	TAU_START(name);
}
void Tau_gpu_enter_memcpy_event(const char *functionName, eventId *id, gpuId
*device, int transferSize, bool memcpyType)
{
#ifdef DEBUG_PROF
	//printf("entering cuMemcpy event: %s.\n", name);
#endif

	if (functionName == TAU_GPU_USE_DEFAULT_NAME)
	{
		if (memcpyType == MemcpyHtoD) {
			functionName = "Memory copy Host to Device";
		}
		else
		{
			functionName = "Memory copy Device to Host";
		}
	}

	// Place the Message into the trace in when the memcpy in entered if this
	// thread initiates the send otherwise wait until this event is exited.
	// This is too make the message lines in the trace to always point forward in
	// time.

	if (memcpyType == MemcpyHtoD) {
		TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, 0);
		if (transferSize != TAU_GPU_UNKNOW_TRANSFER_SIZE)
		{
			TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
		}
		else
		{
			counted_memcpys--;
		}
	}
	else
	{
		if (transferSize != TAU_GPU_UNKNOW_TRANSFER_SIZE)
		{
			TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
		}
		else
		{
			counted_memcpys--;
		}
	}
	
	TAU_START(functionName);
}
void Tau_gpu_exit_memcpy_event(const char * functionName, eventId *id, gpuId *device, bool
memcpyType)
{
#ifdef DEBUG_PROF
	//printf("exiting cuMemcpy event: %s.\n", name);
#endif

	if (functionName == TAU_GPU_USE_DEFAULT_NAME)
	{
		if (memcpyType == MemcpyHtoD) {
			functionName = "Memory copy Host to Device";
		}
		else
		{
			functionName = "Memory copy Device to Host";
		}
	}

	// Place the Message into the trace in when the memcpy in exited if this
	// thread receives the message otherwise do it when this event is entered.
	// This is too make the message lines in the trace to always point forward in
	// time.

	if (memcpyType == MemcpyDtoH) {
		TauTraceOneSidedMsg(MESSAGE_RECV, device, -1, 0);
	}
	
	TAU_STOP(functionName);

}

void Tau_gpu_exit_event(const char *name, eventId *id)
{
#ifdef DEBUG_PROF
	printf("exit cu event: %s.\n", name);
#endif
	TAU_STOP(name);
	if (strcmp(name, "cuCtxDetach") == 0)
	{
		//We're done with the gpu, stop the top level timer.
#ifdef DEBUG_PROF
		printf("in cuCtxDetach.\n");
#endif
		//TAU_PROFILER_STOP_TASK(gpu_ptr, gpuTask);
		//TAU_PROFILER_STOP(main_ptr);
	}
}
void start_gpu_event(const char *name)
{
#ifdef DEBUG_PROF
	printf("staring %s event.\n", name);
#endif
	map<EventName, void*>::iterator it = events.find(name);
	if (it == events.end())
	{
		void *ptr;
		TAU_PROFILER_CREATE(ptr, name, "", TAU_USER);
		TAU_PROFILER_START_TASK(ptr, gpuTask);
		events[EventName(name)] = ptr;
	} else
	{
		void *ptr = (*it).second;
		TAU_PROFILER_START_TASK(ptr, gpuTask);
	}
}
void stage_gpu_event(const char *name, double start_time)
{
#ifdef DEBUG_PROF
	cout << "setting gpu timestamp for start " << start_time << endl;
#endif
	metric_set_gpu_timestamp(gpuTask, start_time);

	check_gpu_event();
	start_gpu_event(name);
}
void stop_gpu_event(const char *name)
{
#ifdef DEBUG_PROF
	printf("stopping %s event.\n", name);
#endif
	map<EventName,void*>::iterator it = events.find(name);
	if (it == events.end())
	{
		printf("FATAL ERROR in stopping GPU event.\n");
	} else
	{
		void *ptr = (*it).second;
		TAU_PROFILER_STOP_TASK(ptr, gpuTask);
	}
}
void break_gpu_event(const char *name, double stop_time)
{
#ifdef DEBUG_PROF
	cout << "setting gpu timestamp for stop: " << stop_time << endl;
#endif
	metric_set_gpu_timestamp(gpuTask, stop_time);
	stop_gpu_event(name);
}

void Tau_gpu_register_gpu_event(const char *name, eventId *id, double startTime, double endTime)
{	
	//printf("in TauGpu.cpp.\n");
	//printf("Tau gpu name: %s.\n", name);
	stage_gpu_event(name, 
		startTime);
	//TAU_REGISTER_CONTEXT_EVENT(k1, "sample kernel data");
	//TAU_CONTEXT_EVENT(k1,1000);
	break_gpu_event(name,
			endTime);
}

void Tau_gpu_register_memcpy_event(const char *functionName, eventId *id, gpuId *device, double startTime, double
endTime, int transferSize, bool memcpyType)
{
	if (functionName == TAU_GPU_USE_DEFAULT_NAME)
	{
		if (memcpyType == MemcpyHtoD) {
			functionName = "Memory copy Host to Device";
		}
		else
		{
			functionName = "Memory copy Device to Host";
		}
	}

	printf("in Tau_gpu recording memcopy event.\n");
#ifdef DEBUG_PROF		
	printf("recording memcopy event.\n");
	printf("time is: %f:%f.\n", startTime, endTime);
#endif
	if (memcpyType == MemcpyHtoD) {
		stage_gpu_event(functionName,  
				startTime);
		//TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
		if (transferSize != TAU_GPU_UNKNOW_TRANSFER_SIZE)
		{
			counted_memcpys++;
			TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
		//TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread()); 
#ifdef DEBUG_PROF		
		printf("[%f] onesided event mem recv: %f, id: %s.\n", startTime, transferSize,
		device->printId());
#endif
		}
		break_gpu_event(functionName,
				endTime);
		TauTraceOneSidedMsg(MESSAGE_RECV, device, transferSize, gpuTask);
	}
	else {
		stage_gpu_event(functionName, 
				startTime);
		//TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
		if (transferSize != TAU_GPU_UNKNOW_TRANSFER_SIZE)
		{
			counted_memcpys++;
			TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
		printf("[%f] onesided event mem send: %f, id: %s\n", startTime, transferSize,
		device->printId());
#endif
		}
		//TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread()); 
		TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, gpuTask);
		break_gpu_event(functionName,
				endTime);
	}

}
/*
	Initialization routine for TAU
*/
int Tau_gpu_init(void)
{
		TAU_PROFILE_SET_NODE(0);
		TAU_PROFILER_CREATE(main_ptr, ".TAU application", "", TAU_USER);
		//TAU_PROFILER_CREATE(main_ptr, "main", "", TAU_USER);
		TAU_PROFILER_CREATE(gpu_ptr, ".TAU application", "", TAU_USER);

		/* Create a seperate GPU task */
		TAU_CREATE_TASK(gpuTask);


#ifdef DEBUG_PROF
		printf("Created user clock.\n");
#endif
			
		//TAU_PROFILER_START(main_ptr);	

		
#ifdef DEBUG_PROF
		printf("started main.\n");
#endif

}


/*
	finalization routine for TAU
*/
void Tau_gpu_exit(void)
{
		if (counted_memcpys != 0)
		{
			cerr << "TAU: warning not all bytes tranfered between CPU and GPU were recorded, some data maybe be incorrect." << endl;
		}
#ifdef DEBUG_PROF
		cerr << "stopping first gpu event.\n" << endl;
		printf("stopping level 0.\n");
#endif
		TAU_PROFILER_STOP_TASK(gpu_ptr, gpuTask);
#ifdef DEBUG_PROF
		printf("stopping level 1.\n");
#endif
		//TAU_PROFILER_STOP(main_ptr);
#ifdef DEBUG_PROF
		printf("stopping level 2.\n");
#endif
	  //TAU_PROFILE_EXIT("tau_gpu");
}
