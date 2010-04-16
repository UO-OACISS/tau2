/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory                                        **
****************************************************************************/
/***************************************************************************
**      File            : taucuda.cpp                                     **
**      Description     : TAU trace format reader library header files    **
**      Author          : Shangkar Mayanglambam                           **
**                      : Scott Biersdorff                                **
**      Contact         : scottb@cs.uoregon.edu                           **
***************************************************************************/

#include "taucuda_interface.h"
#include "TAU.h"
#include <Profile/TauInit.h>
#include <stdio.h>


void *main_ptr, *gpu_ptr;

TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
TAU_PROFILER_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");

int gpuTask;
bool firstEvent = true;

#include <linux/unistd.h>

extern void metric_set_gpu_timestamp(int tid, double value);

#include<map>
using namespace std;

double cpu_start_time;

#define MemcpyHtoD false
#define MemcpyDtoH true

struct EventName {
		const char *name;
		EventName(const char* n) :
			name(n) {}	
		bool operator<(const EventName &c1) const { return strcmp(name,c1.name) < 0; }
};

typedef map<cuEventId, bool> doubleMap;
doubleMap MemcpyEventMap;

map<EventName, void*> events;

extern void metric_set_gpu_timestamp(int tid, double value);


void check_gpu_event()
{
	if (firstEvent)
	{
#ifdef DEBUG_PROF
		printf("first gpu event.\n");
#endif
		TAU_PROFILER_START_TASK(gpu_ptr, gpuTask);
		firstEvent = false;
	}
}

/* === Begin implementing the hooks === */

/* create TAU callback routine to capture both CPU and GPU execution time 
	takes the thread id as a argument. */

void enter_cu_event(const char* name, cuEventId id)
{
#ifdef DEBUG_PROF
	printf("entering cu event: %s.\n", name);
#endif
	if(strncmp(name,"cuMemcpy", sizeof("cuMemcpy")-1)==0)
	{
		if(strncmp(name,"cuMemcpyHtoD", sizeof("cuMemcpyHtoD")-1)==0)
		{
			MemcpyEventMap.insert(make_pair(id, MemcpyHtoD));
		}
		else if(strncmp(name,"cuMemcpyDtoH",sizeof("cuMemcpyDtoH")-1)==0)
		{
			MemcpyEventMap.insert(make_pair(id, MemcpyDtoH));
		}
	}
	TAU_START(name);
}

void exit_cu_event(const char *name, cuEventId id)
{
#ifdef DEBUG_PROF
	printf("exit cu event: %s.\n", name);
#endif
	TAU_STOP(name);
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
	printf("setting gpu timestamp to: %ld.\n", start_time);
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
	printf("setting gpu timestamp to: %ld.\n", stop_time);
#endif
	metric_set_gpu_timestamp(gpuTask, stop_time);
	stop_gpu_event(name);
}

void register_gpu_event(const char *name, cuEventId id, double startTime, double endTime)
{
	stage_gpu_event(name, 
		startTime);

	break_gpu_event(name,
			endTime);
}

void register_memcpy_event(cuEventId id, double startTime, double
endTime, double transferSize)
{
	doubleMap::const_iterator it = MemcpyEventMap.find(id);

	if (it != MemcpyEventMap.end())
	{
		if (it->second == MemcpyHtoD) {
			stage_gpu_event("cuda Memory copy Host to Device", 
					startTime);
			TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
			break_gpu_event("cuda Memory copy Host to Device",
					endTime);
		}
		else {
			stage_gpu_event("cuda Memory copy Device to Host", 
					startTime);
			TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
			break_gpu_event("cuda Memory copy Device to Host",
					endTime);
		}
	} else 
	{
		printf("ERROR: cannot find matching memcopy event.\n");
	}

}
/*
	Initialization routine for taucuda
*/
int tau_cuda_init(void)
{
		TAU_PROFILE_SET_NODE(0);
		TAU_PROFILER_CREATE(main_ptr, "main", "", TAU_USER);
		TAU_PROFILER_CREATE(gpu_ptr, "gpu elapsed time", "", TAU_USER);

		/* Create a seperate GPU task */
		TAU_CREATE_TASK(gpuTask);


#ifdef DEBUG_PROF
		printf("Created user clock.\n");
#endif
			
		TAU_PROFILER_START(main_ptr);	

		
#ifdef DEBUG_PROF
		printf("started main.\n");
#endif
}


/*
	finalization routine for taucuda
*/
void tau_cuda_exit(void)
{
#ifdef debug_prof
		printf("stopping first gpu event.\n");
		printf("stopping level 0.\n");
#endif
		TAU_PROFILER_STOP_TASK(gpu_ptr, gpuTask);
#ifdef debug_prof
		printf("stopping level 1.\n");
#endif
		TAU_PROFILER_STOP(main_ptr);
#ifdef debug_prof
		printf("stopping level 2.\n");
#endif
}
