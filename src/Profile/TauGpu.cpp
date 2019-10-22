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
 **      Description     : TAU GPU layer that translates events that occur **
 **                        on the GPU to regular TAU events.               **
 **      Author          : Shangkar Mayanglambam                           **
 **                      : Scott Biersdorff                                **
 **                      : Robert Lim                                      **
 **      Contact         : scottb@cs.uoregon.edu                           **
 **                      : roblim1@cs.uoregon.edu                          **
 ***************************************************************************/

#include "TauGpu.h"
#include "TAU.h"
#include <TauInit.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <Profile/OpenMPLayer.h>
#include <stdlib.h>
#include <map>
#include <set>

// Moved from header file
using namespace std;

static TauContextUserEvent *MemoryCopyEventHtoD;
static TauContextUserEvent *MemoryCopyEventDtoH;
static TauContextUserEvent *MemoryCopyEventDtoD;
static TauContextUserEvent *UnifiedMemoryEventHtoD;
static TauContextUserEvent *UnifiedMemoryEventDtoH;
static TauContextUserEvent *UnifiedMemoryEventPageFault;

static TauContextUserEvent *FloatingPointOps;
static TauContextUserEvent *MemoryOps;
static TauContextUserEvent *ControlOps;
static TauContextUserEvent *SMClockEvent;
static TauContextUserEvent *MemoryClockEvent;
static TauContextUserEvent *PowerUtilizationEvent;

static uint32_t recentKernelId = -1;
static uint32_t recentCorrelationId = -1;

int number_of_tasks = 0;
int number_of_top_level_task_events = 0;

int get_task(GpuEvent *new_task);

bool gpuComp(GpuEvent* a, GpuEvent* b)
{
  return a->less_than(b);
}

//map of GPU to Profile id.
map<GpuEvent*, int, bool (*)(GpuEvent*, GpuEvent*)>& TheGpuEventMap(void)
{
  bool (*gpuCompFunc)(GpuEvent*, GpuEvent*) = gpuComp;
  static map<GpuEvent*, int, bool (*)(GpuEvent*, GpuEvent*)> GpuEventMap(gpuCompFunc);

  return GpuEventMap;
}

//The number of Memcpys called with unknown transfer size which should be given
//on the GPU thread.
int counted_memcpys = 0;
int counted_unifmems = 0;

#ifndef __APPLE__
#include <linux/unistd.h>
#endif /* __APPLE__ */

extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" void Tau_set_thread_fake(int tid);

extern "C" void Tau_create_top_level_timer_if_necessary_task(int tid);
extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);

extern "C" int Tau_is_thread_fake(int tid);

#include<map>
using namespace std;

double cpu_start_time;

static inline void record_context_event(TauContextUserEvent * e, TAU_EVENT_DATATYPE event_data, int task, double timestamp) {
  if(Tau_is_thread_fake(task)) {  
    TAU_CONTEXT_EVENT_THREAD_TS(e, event_data, task, timestamp);
  } else {
    TAU_CONTEXT_EVENT_THREAD(e, event_data, task);
  }
}

void check_gpu_event(int gpuTask)
{
  static std::set<int> seen;
  if (number_of_top_level_task_events < number_of_tasks) {
#ifdef DEBUG_PROF
    cerr << "first gpu event" << endl;
#endif
    if (gpuTask >= TAU_MAX_THREADS) {
      cerr << "TAU ERROR: The number of GPU entities exceeds the maximum: " << TAU_MAX_THREADS
          << ". Please reconfigure TAU with '-useropt=-DTAU_MAX_THREADS=<larger number>.'" << endl;
      exit(1);
    }
    if (seen.count(gpuTask) == 0) {
        Tau_create_top_level_timer_if_necessary_task(gpuTask);
        number_of_top_level_task_events++;
        seen.insert(gpuTask);
    }
  }
}

/* === Begin implementing the hooks === */

/* create TAU callback routine to capture both CPU and GPU execution time 
 takes the thread id as a argument. */

void Tau_gpu_enter_event(const char* name)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("entering cu event: %s.\n", name);
#endif
  //Tau_pure_start(name);
  void* handle = Tau_get_function_info(name, "", TAU_USER, "TAU_GPU_API");
  Tau_start_timer(handle, 0, Tau_get_thread());
}

void Tau_gpu_enter_memcpy_event(const char *functionName, GpuEvent *device, int transferSize, int memcpyType)
{
#ifdef DEBUG_PROF
  //TAU_VERBOSE("entering Memcpy event type: %d.\n", memcpyType);
#endif

  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0) {
    if (memcpyType == MemcpyHtoD) {
      functionName = "Memory copy Host to Device";
    } else if (memcpyType == MemcpyDtoH) {
      functionName = "Memory copy Device to Host";
    } else {
      functionName = "Memory copy Device to Device";
    }
    //printf("using default name: %s.\n", functionName);
  }

  //Tau_pure_start(functionName);
  void* handle = Tau_get_function_info(functionName, "", TAU_USER, "TAU_GPU_MEMORY_COPY");
  Tau_start_timer(handle, 0, Tau_get_thread());


  //Inorder to capture the entire memcpy transaction time start the send/recived
  //at the start of the event
  if (TauEnv_get_tracing()) {
    int threadId = RtsLayer::localThreadId();
    if (memcpyType == MemcpyDtoH) {
      TauTraceOneSidedMsg(MESSAGE_RECV, device, -1, threadId, 0UL);
    } else if (memcpyType == MemcpyHtoD) {
      TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, threadId, 0UL);
    } else {
      TauTraceOneSidedMsg(MESSAGE_UNKNOWN, device, transferSize, threadId, 0UL);
    }
  }
  if (memcpyType == MemcpyHtoD) {
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      TAU_CONTEXT_EVENT(MemoryCopyEventHtoD, transferSize);
      //TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
    } else {
      counted_memcpys--;
    }
  } else if (memcpyType == MemcpyDtoH) {
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      TAU_CONTEXT_EVENT(MemoryCopyEventDtoH, transferSize);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
    } else {
      counted_memcpys--;
    }
  } else if (memcpyType == MemcpyDtoD) {
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      TAU_CONTEXT_EVENT(MemoryCopyEventDtoD, transferSize);
    } else {
      counted_memcpys--;
    }
  }
}

void Tau_gpu_enter_unifmem_event(const char *functionName, GpuEvent *device, int transferSize, int unifmemType)
{
#ifdef DEBUG_PROF
  //TAU_VERBOSE("entering Memcpy event type: %d.\n", memcpyType);
#endif

  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0) {
    if (unifmemType == BytesHtoD) {
      functionName = "Unified Memory copy Host to Device";
    } else if (unifmemType == MemcpyDtoH) {
      functionName = "Unified Memory copy Device to Host";
    } else {
      functionName = "Unified Memory CPU Page Fault";
    }
    //printf("using default name: %s.\n", functionName);
  }

  //Tau_pure_start(functionName);
  void* handle = Tau_get_function_info(functionName, "", TAU_USER, "TAU_GPU_UNIFIED_MEMORY");
  Tau_start_timer(handle, 0, Tau_get_thread());


  //Inorder to capture the entire memcpy transaction time start the send/recived
  //at the start of the event
  if (TauEnv_get_tracing()) {
    int threadId = get_task(device);  
    if (unifmemType == BytesDtoH) {
      TauTraceOneSidedMsg(MESSAGE_RECV, device, -1, threadId, 0UL);
    } else if (unifmemType == BytesHtoD) {
      TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, threadId, 0UL);
    } else {
      TauTraceOneSidedMsg(MESSAGE_UNKNOWN, device, transferSize, threadId, 0UL);
    }
  }
  if (unifmemType == BytesHtoD) {
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      TAU_CONTEXT_EVENT(UnifiedMemoryEventHtoD, transferSize);
      //TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
    } else {
      counted_unifmems--;
    }
  } else if (unifmemType == BytesDtoH) {
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      TAU_CONTEXT_EVENT(UnifiedMemoryEventDtoH, transferSize);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
    } else {
      counted_unifmems--;
    }
  } else if (unifmemType == CPUPageFault) {
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      TAU_CONTEXT_EVENT(UnifiedMemoryEventPageFault, transferSize);
    } else {
      counted_unifmems--;
    }
  }
}

void Tau_gpu_exit_memcpy_event(const char * functionName, GpuEvent *device, int memcpyType)
{
#ifdef DEBUG_PROF
  //TAU_VERBOSE("exiting cuMemcpy event: %s.\n", name);
#endif

  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0) {
    if (memcpyType == MemcpyHtoD) {
      functionName = "Memory copy Host to Device";
    } else if (memcpyType == MemcpyDtoH) {
      functionName = "Memory copy Device to Host";
    } else {
      functionName = "Memory copy Device to Device";
    }
    //printf("using default name: %s.\n", functionName);
  }

  // Place the Message into the trace in when the memcpy in exited if this
  // thread receives the message otherwise do it when this event is entered.
  // This is too make the message lines in the trace to always point forward in
  // time.

  //Tau_pure_stop(functionName);
  void* handle = Tau_get_function_info(functionName, "", TAU_USER, "TAU_GPU_MEMORY_COPY");
  Tau_stop_timer(handle, Tau_get_thread());

}

void Tau_gpu_exit_unifmem_event(const char * functionName, GpuEvent *device, int unifmemType)
{
#ifdef DEBUG_PROF
  //TAU_VERBOSE("exiting cuMemcpy event: %s.\n", name);
#endif

  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0) {
    if (unifmemType == BytesHtoD) {
      functionName = "Unified Memory copy Host to Device";
    } else if (unifmemType == BytesDtoH) {
      functionName = "Unified Memory copy Device to Host";
    } else {
      functionName = "Unified Memory CPU Page Fault";
    }
    //printf("using default name: %s.\n", functionName);
  }

  // Place the Message into the trace in when the memcpy in exited if this
  // thread receives the message otherwise do it when this event is entered.
  // This is too make the message lines in the trace to always point forward in
  // time.

  //Tau_pure_stop(functionName);
  void* handle = Tau_get_function_info(functionName, "", TAU_USER, "TAU_GPU_UNIFIED_MEMORY");
  Tau_stop_timer(handle, Tau_get_thread());

}

void Tau_gpu_exit_event(const char *name)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("exit cu event: %s.\n", name);
#endif
  //Tau_pure_stop(name);
  void* handle = Tau_get_function_info(name, "", TAU_USER, "TAU_GPU_API");
  Tau_stop_timer(handle, Tau_get_thread());
}

void stage_gpu_event(const char *name, int gpuTask, double start_time, FunctionInfo* parent, const char * group)
{
#ifdef DEBUG_PROF
  cerr << "setting gpu timestamp for start " << setprecision(16) << start_time << endl;
#endif
  metric_set_gpu_timestamp(gpuTask, start_time);

  check_gpu_event(gpuTask);
  if (TauEnv_get_callpath()) {
    //printf("Profiler: %s \n", parent->GetName());
    if (parent != NULL) {
      Tau_start_timer(parent, 0, gpuTask);
    }
  }
  Tau_pure_start_task_group(name, gpuTask, group);
}

void break_gpu_event(const char *name, int gpuTask, double stop_time, FunctionInfo* parent)
{
#ifdef DEBUG_PROF
  cerr << "setting gpu timestamp for stop: " << setprecision(16) << stop_time << endl;
#endif
  metric_set_gpu_timestamp(gpuTask, stop_time);
  Tau_pure_stop_task(name, gpuTask);
  if (TauEnv_get_callpath()) {
    //printf("Profiler: %s \n", parent->GetName());
    double totalTime = 0;
    if (parent != NULL) {
      Tau_stop_timer(parent, gpuTask);
    }
  }
}
int get_task(GpuEvent *new_task)
{
  int task = 0;
  map<GpuEvent*, int>::iterator it = TheGpuEventMap().find(new_task);
  if (it == TheGpuEventMap().end()) {
    GpuEvent *create_task = new_task->getCopy();
    task = Tau_RtsLayer_createThread();
    //new task, record metadata.
    create_task->recordMetadata(task);
    TheGpuEventMap().insert(pair<GpuEvent *, int>(create_task, task));
    number_of_tasks++;
    Tau_set_thread_fake(task);
    //TAU_CREATE_TASK(task);
    //printf("new task: %s id: %d.\n", create_task->gpuIdentifier(), task);
  } else {
    task = (*it).second;
  }
  //printf("Using GPU task %d\n", task);

  return task;
}
/*
 GpuEvent Tau_gpu_create_gpu_event(const char *name, GpuEvent *device,
 FunctionInfo* callingSite, TauGpuContextMap* map)
 {
 return GpuEvent(name, device, callingSite, map);
 }

 GpuEvent Tau_gpu_create_gpu_event(const char *name, GpuEvent *device,
 FunctionInfo* callingSite)
 {
 return GpuEvent(name, device, callingSite, NULL);
 }
 */

/* Records a synchronous event on the GPU, as seen from CPU! */
void Tau_gpu_enter_event_from_cpu(const char* name, int tid, double start)
{
  stage_gpu_event(name, tid, start, NULL, "TAU_GPU_KERNEL");
}

/* Records a synchronous event on the GPU, as seen from CPU! */
void Tau_gpu_exit_event_from_cpu(const char* name, int tid, double end)
{
  break_gpu_event(name, tid, end, NULL);
}

void Tau_gpu_register_gpu_event(GpuEvent *id, double startTime, double endTime)
{
  int task = get_task(id);
  const double syncStartTime = startTime + id->syncOffset();
  const double syncEndTime = endTime + id->syncOffset();
  stage_gpu_event(id->getName(), task, syncStartTime, id->getCallingSite(), "TAU_GPU_KERNEL");
  GpuEventAttributes *attr;
  int number_of_attributes;
  id->getAttributes(attr, number_of_attributes);
  for (int i = 0; i < number_of_attributes; i++) {
    TauContextUserEvent* e = attr[i].userEvent;
    if (e) { 
      TAU_EVENT_DATATYPE event_data = attr[i].data;
      record_context_event(e, event_data, task, syncStartTime);
    } else {
      break;
    }
  }
  break_gpu_event(id->getName(), task, syncEndTime, id->getCallingSite());
}

void Tau_gpu_register_sync_event(GpuEvent *id, double startTime, double endTime)
{
  int task = get_task(id);
  const double syncStartTime = startTime + id->syncOffset();
  const double syncEndTime = endTime + id->syncOffset();
  stage_gpu_event(id->getName(), task, syncStartTime, id->getCallingSite(), "TAU_GPU_SYNCHRONIZATION");
  break_gpu_event(id->getName(), task, syncEndTime, id->getCallingSite());
}

void Tau_gpu_register_memcpy_event(GpuEvent *id, double startTime, double endTime, int transferSize, int memcpyType,
				   int direction)
{
  int task = get_task(id);  
  const char* functionName = id->getName();
  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0) {
    if (memcpyType == MemcpyHtoD) {
      functionName = "Memory copy Host to Device";
    } else if (memcpyType == MemcpyDtoH) {
      functionName = "Memory copy Device to Host";
    } else {
      functionName = "Memory copy Device to Device";
    }
    //printf("using default name: %s.\n", functionName);
  }

#ifdef DEBUG_PROF		
  TAU_VERBOSE("recording memcopy event.\n");
  TAU_VERBOSE("time is: %f:%f.\n", startTime, endTime);
  TAU_VERBOSE("kind is: %d.\n", memcpyType);
  TAU_VERBOSE("id is: %s.\n", id->gpuIdentifier());
#endif
  const double syncStartTime = startTime + id->syncOffset();
  const double syncEndTime = endTime + id->syncOffset();
  if (memcpyType == MemcpyHtoD) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite(), "TAU_GPU_MEMORY_COPY");
    //TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_memcpys++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(MemoryCopyEventHtoD, transferSize, task, syncStartTime);
      //TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
      //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem recv: %d, id: %s.\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    break_gpu_event(functionName, task, syncEndTime, id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_RECV, id, transferSize, task, syncEndTime);
    }
  } else if (memcpyType == MemcpyDtoH) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite(), "TAU_GPU_MEMORY_COPY");
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_memcpys++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(MemoryCopyEventDtoH, transferSize, task, syncStartTime);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //printf("TAU: putting message into trace file.\n");
    //printf("[%f] onesided event mem send: %f.\n", startTime, transferSize);
    break_gpu_event(functionName, task, syncEndTime, id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_SEND, id, transferSize, task, syncEndTime);
    }
  } else {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite(), "TAU_GPU_MEMORY_COPY");
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_SEND) {
      TauTraceOneSidedMsg(direction, id, transferSize, task, syncStartTime);
    }
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_memcpys++;
      record_context_event(MemoryCopyEventDtoD, transferSize, task, syncStartTime);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
    //TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, gpuTask, syncStartTime);
    break_gpu_event(functionName, task, syncEndTime, id->getCallingSite());
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_RECV) {
      TauTraceOneSidedMsg(direction, id, transferSize, task, syncEndTime);
    }
  }

}

void Tau_gpu_register_envt_event(GpuEvent *event, double startTime, double endTime, int transferSize, int dataType)
{
  int task = get_task(event);  
	const char* functionName = event->getName();
	if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0)
	{
		if (dataType == SMClock) {
			functionName = "SM Frequency (MHz)";
		}
		else if (dataType == MemoryClock)
		{
			functionName = "Memory Frequency (MHz)";
		}
		else if (dataType == GPUTemperature)
		{
			functionName = "GPU Temperature";
		}
		else if (dataType == PowerUtilization)
		  {
		    functionName = "Power Utilization (% mW)";
		  }
		else if (dataType == FanSpeed)
		  {
		    functionName == "Fan Speed";
		  }
	}

#ifdef DEBUG_PROF	
	TAU_VERBOSE("recording environment event.\n");
	TAU_VERBOSE("time is: %f:%f.\n", startTime, endTime);
	TAU_VERBOSE("kind is: %d.\n", dataType);
	TAU_VERBOSE("id is: %s.\n", event->gpuIdentifier());
#endif
	if (dataType == SMClock) {
	  TAU_CONTEXT_EVENT(SMClockEvent, transferSize);
	}
	else if (dataType == MemoryClock) {
	  TAU_CONTEXT_EVENT(MemoryClockEvent, transferSize);
	}
	else if (dataType == PowerUtilization) {
	  TAU_CONTEXT_EVENT(PowerUtilizationEvent, transferSize);
	}
}

void Tau_gpu_register_unifmem_event(GpuEvent *id, double startTime, double endTime, int transferSize, int unifmemType,
    int direction)
{
  int task = get_task(id);  
  const char* functionName = id->getName();
  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0) {
    if (unifmemType == BytesHtoD) {
      functionName = "Unified Memory copy Host to Device";
    } else if (unifmemType == BytesDtoH) {
      functionName = "Unified Memory copy Device to Host";
    } else {
      functionName = "Unified Memory CPU Page Fault";
    }
  }

#ifdef DEBUG_PROF		
  TAU_VERBOSE("recording unified memory event.\n");
  TAU_VERBOSE("time is: %f:%f.\n", startTime, endTime);
  TAU_VERBOSE("kind is: %d.\n", unifmemType);
  TAU_VERBOSE("id is: %s.\n", id->gpuIdentifier());
#endif
  const double syncStartTime = startTime + id->syncOffset();
  const double syncEndTime = endTime + id->syncOffset();
  if (unifmemType == BytesHtoD) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite(), "TAU_GPU_MEMORY_COPY");
    //TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_unifmems++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(UnifiedMemoryEventHtoD, transferSize, task, syncStartTime);
      //TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
      //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem recv: %d, id: %s.\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    break_gpu_event(functionName, task, syncEndTime, id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_RECV, id, transferSize, task, syncEndTime);
    }
  } else if (unifmemType == BytesDtoH) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite(), "TAU_GPU_MEMORY_COPY");
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_unifmems++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(UnifiedMemoryEventDtoH, transferSize, task, syncStartTime);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //printf("TAU: putting message into trace file.\n");
    //printf("[%f] onesided event mem send: %f.\n", startTime, transferSize);
    break_gpu_event(functionName, task, syncEndTime, id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_SEND, id, transferSize, task, syncEndTime);
    }
  } else {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite(), "TAU_GPU_MEMORY_COPY");
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_SEND) {
      TauTraceOneSidedMsg(direction, id, transferSize, task, syncStartTime);
    }
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_unifmems++;
      record_context_event(UnifiedMemoryEventPageFault, transferSize, task, syncStartTime);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
    //TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, gpuTask, syncStartTime);
    break_gpu_event(functionName, task, syncEndTime, id->getCallingSite());
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_RECV) {
      TauTraceOneSidedMsg(direction, id, transferSize, task, syncEndTime);
    }
  }

}

/* 
 * Callback for GPU atomic event.
 */
void Tau_gpu_register_gpu_atomic_event(GpuEvent *event)
{
#ifdef DEBUG_PROF		
  TAU_VERBOSE("registering atomic event.\n");
#endif //DEBUG_PROF
  int task = get_task(event);  
  GpuEventAttributes *attr;
  int number_of_attributes;
  event->getAttributes(attr, number_of_attributes);
  for (int i = 0; i < number_of_attributes; i++) {
    TauContextUserEvent* e = attr[i].userEvent;
    TAU_EVENT_DATATYPE event_data = attr[i].data;
    TAU_CONTEXT_EVENT_THREAD(e, event_data, task);
  }
}


void Tau_gpu_register_imix_event(GpuEvent *event, double startTime, double endTime, int transferSize, int dataType)
{
  int task = get_task(event);  
  printf("IMIX type is %d, contextId: %i.\n", dataType, task);
  const char* functionName = event->getName();
  if (strcmp(functionName, TAU_GPU_USE_DEFAULT_NAME) == 0)
    {
      if (dataType == FlPtOps) {
	functionName = "Floating Point Operations";
      }
      else if (dataType == MemOps)
	{
	  functionName = "Memory Operations";
	}
      else if (dataType == CtrlOps)
	{
	  functionName = "Control Operations";
	}
      printf("using default name: %s.\n", functionName);
    }
#ifdef DEBUG_PROF
  TAU_VERBOSE("recording instruction mix event.\n");
  TAU_VERBOSE("time is: %f:%f.\n", startTime, endTime);
  TAU_VERBOSE("kind is: %d.\n", dataType);
  TAU_VERBOSE("id is: %s.\n", event->gpuIdentifier());
#endif
  printf("Time: start: %f, end: %f, event->syncOffset(): %f\n", startTime, endTime, event->syncOffset());
  printf(" kind: %d, id: %s\n", dataType, event->gpuIdentifier());

  if (dataType == FlPtOps) {
    // stage_gpu_event(functionName, task,
    //              startTime + event->syncOffset(), event->getCallingSite());
    TAU_CONTEXT_EVENT(FloatingPointOps, transferSize);
    // TAU_CONTEXT_EVENT_THREAD(SMClockEvent, transferSize, task);
    // break_gpu_event(functionName, task,
    //              syncEndTime, event->getCallingSite());
  }
  else if (dataType == MemOps) {
    // stage_gpu_event(functionName, task,
    //              startTime + event->syncOffset(), event->getCallingSite());
    TAU_CONTEXT_EVENT(MemoryOps, transferSize);
    // TAU_CONTEXT_EVENT_THREAD(MemoryClockEvent, transferSize, task);
    // break_gpu_event(functionName, task,
    //              syncEndTime, event->getCallingSite());
  }
  else if (dataType == CtrlOps) {
    // stage_gpu_event(functionName, task,
    //              startTime + event->syncOffset(), event->getCallingSite);
    TAU_CONTEXT_EVENT(ControlOps, transferSize);
    // TAU_CONTEXT_EVENT_THREAD(PowerUtilizationEvent, transferSize, task);
    // break_gpu_event(functionName, task,
    //              syncEndTime, event->getCallingSite);
  }
}

/* Forward declaration of function to let TAU know we are done initializing */
void Tau_gpu_initialized(bool init);

/*
 Initialization routine for TAU
 */
void Tau_gpu_init(void)
{

  //init context event.
  Tau_get_context_userevent((void **)&MemoryCopyEventHtoD, "Bytes copied from Host to Device");
  Tau_get_context_userevent((void **)&MemoryCopyEventDtoH, "Bytes copied from Device to Host");
  Tau_get_context_userevent((void **)&MemoryCopyEventDtoD, "Bytes copied from Device to Device");
  Tau_get_context_userevent((void **)&UnifiedMemoryEventHtoD, "Unified Memory Bytes copied from Host to Device");
  Tau_get_context_userevent((void **)&UnifiedMemoryEventDtoH, "Unified Memory Bytes copied from Device to Host");
  Tau_get_context_userevent((void **)&UnifiedMemoryEventPageFault, "Unified Memory CPU Page Faults");
  Tau_get_context_userevent((void **) &FloatingPointOps, "Floating Point Operations");
  Tau_get_context_userevent((void **) &MemoryOps, "Memory Operations");
  Tau_get_context_userevent((void **) &ControlOps, "Control Operations");

  Tau_gpu_initialized(true);
}

/*
 finalization routine for TAU
 */
void Tau_gpu_exit(void)
{
  if (counted_memcpys != 0) {
    cerr << "TAU: warning not all bytes tranfered between CPU and GPU were recorded, some data maybe be incorrect." << endl;
  }
  cerr << "stopping first gpu event.\n" << endl;
#if not defined(PTHREADS)
  TAU_VERBOSE("stopping level %d tasks.\n", number_of_tasks);
  map<GpuEvent*, int>::iterator it;
  for (it = TheGpuEventMap().begin(); it != TheGpuEventMap().end(); it++) {
    Tau_stop_top_level_timer_if_necessary_task(it->second);
  }
#else
  printf("stopping level %d tasks.\n", TauEnv_get_cudaTotalThreads());
  for (int i=0; i < TauEnv_get_cudaTotalThreads(); i++) {
    // GpuThread gTemp = gThreads[i];
    // printf("[TauGpu]: Tau_gpu_exit, calling Tau_stop_top_level_timer_if_necessary_task for %i\n", gTemp.gpu_tid);    
    // Tau_stop_top_level_timer_if_necessary_task(gTemp.gpu_tid);
    Tau_stop_top_level_timer_if_necessary_task(i);
  }
#endif
  TAU_VERBOSE("stopping level 1.\n");
}

