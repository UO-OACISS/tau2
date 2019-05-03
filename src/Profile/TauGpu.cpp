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

// #define TAU_DEBUG_SASS 1

static TauContextUserEvent *MemoryCopyEventHtoD;
static TauContextUserEvent *MemoryCopyEventDtoH;
static TauContextUserEvent *MemoryCopyEventDtoD;
static TauContextUserEvent *UnifiedMemoryEventHtoD;
static TauContextUserEvent *UnifiedMemoryEventDtoH;
static TauContextUserEvent *UnifiedMemoryEventPageFault;

static TauContextUserEvent *FloatingPointOps;
static TauContextUserEvent *MemoryOps;
static TauContextUserEvent *ControlOps;

static uint32_t recentKernelId = -1;
static uint32_t recentCorrelationId = -1;
static int curr_device_id = 0;

int number_of_tasks = 0;
int number_of_top_level_task_events = 0;

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
  if (number_of_top_level_task_events < number_of_tasks) {
#ifdef DEBUG_PROF
    cerr << "first gpu event" << endl;
#endif
    if (gpuTask >= TAU_MAX_THREADS) {
      cerr << "TAU ERROR: The number of GPU entities exceeds the maximum: " << TAU_MAX_THREADS
          << ". Please reconfigure TAU with '-useropt=-DTAU_MAX_THREADS=<larger number>.'" << endl;
      exit(1);
    }
    //printf("starting top level timer total=%d map total=%d id=%d.\n", number_of_tasks, TheGpuEventMap().size(), gpuTask);
    //TAU_PROFILER_START_TASK(gpu_ptr, gpuTask);
    Tau_create_top_level_timer_if_necessary_task(gpuTask);
    number_of_top_level_task_events++;
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
  TAU_START(name);
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

  TAU_START(functionName);

  //Inorder to capture the entire memcpy transaction time start the send/recived
  //at the start of the event
  if (TauEnv_get_tracing()) {
    int threadId = 0;
#if defined(PTHREADS)
    threadId = device->getTaskId();
#endif
    if (memcpyType == MemcpyDtoH) {
      TauTraceOneSidedMsg(MESSAGE_RECV, device, -1, threadId);
    } else if (memcpyType == MemcpyHtoD) {
      TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, threadId);
    } else {
      TauTraceOneSidedMsg(MESSAGE_UNKNOWN, device, transferSize, threadId);
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

  TAU_START(functionName);

  //Inorder to capture the entire memcpy transaction time start the send/recived
  //at the start of the event
  if (TauEnv_get_tracing()) {
    int threadId = 0;
#if defined(PTHREADS)
    threadId = device->getTaskId();
#endif
    if (unifmemType == BytesDtoH) {
      TauTraceOneSidedMsg(MESSAGE_RECV, device, -1, threadId);
    } else if (unifmemType == BytesHtoD) {
      TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, threadId);
    } else {
      TauTraceOneSidedMsg(MESSAGE_UNKNOWN, device, transferSize, threadId);
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

  TAU_STOP(functionName);

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

  TAU_STOP(functionName);

}

void Tau_gpu_exit_event(const char *name)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("exit cu event: %s.\n", name);
#endif
  TAU_STOP(name);
}
void start_gpu_event(const char *name, int gpuTask)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("staring %s event.\n", name);
#endif
  TAU_START_TASK(name, gpuTask);
}
void stage_gpu_event(const char *name, int gpuTask, double start_time, FunctionInfo* parent)
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
  start_gpu_event(name, gpuTask);
}


void stop_gpu_event(const char *name, int gpuTask)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("stopping %s event.\n", name);
#endif
  /*
   map<EventName,void*>::iterator it = events.find(name);
   if (it == events.end())
   {
   printf("FATAL ERROR in stopping GPU event.\n");
   } else
   {
   void *ptr = (*it).second;
   TAU_PROFILER_STOP_TASK(ptr, gpuTask);
   }
   */
  TAU_STOP_TASK(name, gpuTask);
}

// FunctionInfo* Tau_make_cupti_sample_timer(const char * filename, const char * function, int lineno);

// void stop_gpu_event(const char *name, int gpuTask)
// {
// #ifdef DEBUG_PROF
// 	TAU_VERBOSE("stopping %s event.\n", name);
// #endif

// 	TAU_STOP_TASK(name, gpuTask);

// 	if (!instrSrcMap.empty()) {
// 	  // build profile

// 	  // filename, function, lineno
// 	  int tid = gpuTask;

// 	  // 0 is time, eventually want to index by stall reasons
// 	  int counter = 0;
// 	  char sass_level[] = "kernel";
// 	  // check if we're displaying at kernel level or source level:
// 	  if(strcmp(TauEnv_get_cuda_sass_type(), sass_level) == 0) {
// 	    // kernel level
// 	    for (std::map<uint32_t, FuncSampling>::iterator it=funcMap.begin(); 
// 		 it != funcMap.end(); ++it) {
// 	      FuncSampling fTemp = it->second;
// #ifdef TAU_DEBUG_SASS
// 	      printf("fTemp.fid: %i, fTemp.name: %s\n", fTemp.fid, fTemp.name);
// #endif
// 	      double kernel_exec_time;
// 	      uint32_t kernel_samples;
// 	      uint32_t kernel_launches;
// 	      const char* filename;
// 	      uint32_t lineno;
// 	      kernel_exec_time = getKernelExecutionTimes(fTemp.fid);
// 	      kernel_samples = getKernelSamples(fTemp.fid);
// 	      kernel_launches = getUniqueKernelLaunches(fTemp.fid);
// 	      filename = getKernelFilePath(fTemp.fid);
// 	      lineno = getKernelLineNo(fTemp.fid);
// 	      FunctionInfo* fi = Tau_make_cupti_sample_timer(filename, 
// 							     fTemp.demangled, 
// 							     lineno);	      
// #if TAU_DEBUG_SASS
// 	      printf("[TauGPU]:  Created filePath: %s, demangled: %s, lineno: %u\n", 
// 		     filename, fTemp.demangled, lineno);
// 	      // need samples, tstamp_delta
// 	      printf("[TauGPU]:  kernel_exec_time: %f, kernel_samples: %u, kernel_launches: %u\n", 
// 		     kernel_exec_time, kernel_samples, kernel_launches);
// 	      printf("[TauGPU]:  fi->GetCalls(tid): %u\n", fi->GetCalls(tid));	      
// #endif
// 	      // where tid is the gpu task/thread ID
// 	      // TODO:  Need to verify whether to include samples?
// 	      // fi->SetCalls(tid, kernel_launches+fi->GetCalls(tid)); // including samples
// 	      fi->SetCalls(tid, kernel_launches); // actual # kernel launches (MOST ACCURATE)
// 	      // where counter is the index of the metric (already in 1e3 scale)
// 	      // TODO:  Get time from kernelMap (CuptiActivity)
// 	      fi->AddInclTimeForCounter(kernel_exec_time, tid, counter); 
// 	      // where counter is the index of the metric
// 	      fi->AddExclTimeForCounter(kernel_exec_time, tid, counter); 
// 	      // resetKernelExecutionTimes(fTemp.fid); // verify whether needed??
// 	    }
// 	  }
// 	  else {
// 	    // source level
// 	    for (std::map<uint32_t, SourceSampling>::iterator it=srcLocMap.begin(); 
// 		 it != srcLocMap.end(); ++it) {
// 	      SourceSampling sTemp = it->second;
	    
// 	      if (funcMap.count(sTemp.fid)) {
// 		// lookup
// 		FuncSampling fTemp = funcMap.find(sTemp.fid)->second;
		
// // 		FunctionInfo* fi = Tau_make_cupti_sample_timer(sTemp.fileName, 
// // 							       fTemp.demangled, 
// // 							       sTemp.lineNumber);
		
// // #if TAU_DEBUG_SASS
// // 		// prepare fi object
// // 		printf("Created FunctionInfo, filePath: %s, demangled: %s, lineNumber: %i\n", sTemp.fileName, fTemp.demangled, sTemp.lineNumber);
// // 	      // need samples, tstamp_delta
// // 		printf("timestamp_delta: %f, samples: %u\n", 
// // 		       sTemp.timestamp_delta, sTemp.samples);
// // 		printf(" fi->GetCalls(tid): %u\n", fi->GetCalls(tid));
		
// // #endif
// 		// // where tid is the gpu task/thread ID
// 		// // fi->SetCalls(tid, sTemp.samples+fi->GetCalls(tid));
// 		// fi->SetCalls(tid, sTemp.samples);
// 		// // where counter is the index of the metric (already in 1e3 scale)
// 		// fi->AddInclTimeForCounter(sTemp.timestamp_recentacc, tid, counter); 
// 		// // where counter is the index of the metric
// 		// fi->AddExclTimeForCounter(sTemp.timestamp_recentacc, tid, counter); 
// 		// it->second.timestamp_recentacc=0; // reset
// 	      }
// 	    }
// 	  } // source level
// 	}
// 	else {
// 	  // srcLocMap is empty, can't build profile.
// 	}
// }


void break_gpu_event(const char *name, int gpuTask, double stop_time, FunctionInfo* parent)
{
#ifdef DEBUG_PROF
  cerr << "setting gpu timestamp for stop: " << setprecision(16) << stop_time << endl;
#endif
  metric_set_gpu_timestamp(gpuTask, stop_time);
  stop_gpu_event(name, gpuTask);
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


void Tau_gpu_register_gpu_event(GpuEvent *id, double startTime, double endTime)
{
  // for (int i=0; i < TauEnv_get_cudaTotalThreads(); i++) {
  //   GpuThread gTemp = gThreads[i];
  //   printf("[TauGpu]: systid %u, gputid %i, parentid %u, nodeid %i\n",
  // 	   gTemp.sys_tid, gTemp.gpu_tid, gTemp.parent_tid, gTemp.node_id);    
  // }

#if defined(PTHREADS) && defined(TAU_GPU)
  int task = id->getTaskId(); 
#else
  //int task = id->getTaskId();
  int task = get_task(id);
#endif
  const double syncStartTime = startTime + id->syncOffset();
  stage_gpu_event(id->getName(), task, syncStartTime, id->getCallingSite());
  GpuEventAttributes *attr;
  int number_of_attributes;
  id->getAttributes(attr, number_of_attributes);
//#ifndef TAU_ENABLE_OPENCL
  for (int i = 0; i < number_of_attributes; i++) {
    TauContextUserEvent* e = attr[i].userEvent;
    if (e) { 
      TAU_EVENT_DATATYPE event_data = attr[i].data;
      record_context_event(e, event_data, task, syncStartTime + 1);
    } else {
      break;
    }
  }
//#endif /* TAU_ENABLE_OPENCL: OpenCL crashes here! */
  /*
   if (id.contextEventMap != NULL)
   {
   for (TauGpuContextMap::iterator it = id.contextEventMap->begin();
   it != id.contextEventMap->end();
   it++)
   {
   TauContextUserEvent* e = it->first;
   TAU_EVENT_DATATYPE event_data = it->second;
   TAU_CONTEXT_EVENT_THREAD(e, event_data, task);
   }
   }
   */
  break_gpu_event(id->getName(), task, endTime + id->syncOffset(), id->getCallingSite());
}

void Tau_gpu_register_memcpy_event(GpuEvent *id, double startTime, double endTime, int transferSize, int memcpyType,
				   int direction)
{
  // for (int i=0; i < TauEnv_get_cudaTotalThreads(); i++) {
  //   GpuThread gTemp = gThreads[i];
  //   printf("[TauGpu]: systid %u, gputid %i, parentid %u, nodeid %i\n",
  // 	   gTemp.sys_tid, gTemp.gpu_tid, gTemp.parent_tid, gTemp.node_id);    
  // }
#if defined(PTHREADS) && defined(TAU_GPU)
  int task = id->getTaskId();
#else
  //int task = id->getTaskId();
  int task = get_task(id);  
#endif
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
  if (memcpyType == MemcpyHtoD) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite());
    //TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_memcpys++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(MemoryCopyEventHtoD, transferSize, task, syncStartTime + 1);
      //TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
      //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem recv: %d, id: %s.\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    break_gpu_event(functionName, task, endTime + id->syncOffset(), id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_RECV, id, transferSize, task);
    }
  } else if (memcpyType == MemcpyDtoH) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite());
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_memcpys++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(MemoryCopyEventDtoH, transferSize, task, syncStartTime + 1);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //printf("TAU: putting message into trace file.\n");
    //printf("[%f] onesided event mem send: %f.\n", startTime, transferSize);
    break_gpu_event(functionName, task, endTime + id->syncOffset(), id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_SEND, id, transferSize, task);
    }
  } else {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite());
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_SEND) {
      TauTraceOneSidedMsg(direction, id, transferSize, task);
    }
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_memcpys++;
      record_context_event(MemoryCopyEventDtoD, transferSize, task, syncStartTime + 1);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
    //TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, gpuTask);
    break_gpu_event(functionName, task, endTime + id->syncOffset(), id->getCallingSite());
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_RECV) {
      TauTraceOneSidedMsg(direction, id, transferSize, task);
    }
  }

}

void Tau_gpu_register_unifmem_event(GpuEvent *id, double startTime, double endTime, int transferSize, int unifmemType,
    int direction)
{
  int task = get_task(id);
  //int task = id->getTaskId();
  //printf("in Tau_gpu.\n");
  //printf("Memcpy type is %d.\n", memcpyType);
  const char* functionName = id->getName();
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

#ifdef DEBUG_PROF		
  TAU_VERBOSE("recording unified memory event.\n");
  TAU_VERBOSE("time is: %f:%f.\n", startTime, endTime);
  TAU_VERBOSE("kind is: %d.\n", unifmemType);
  TAU_VERBOSE("id is: %s.\n", id->gpuIdentifier());
#endif
  const double syncStartTime = startTime + id->syncOffset();
  if (unifmemType == BytesHtoD) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite());
    //TAU_REGISTER_EVENT(MemoryCopyEventHtoD, "Memory copied from Host to Device");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_unifmems++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(UnifiedMemoryEventHtoD, transferSize, task, syncStartTime + 1);
      //TAU_EVENT(MemoryCopyEventHtoD(), transferSize);
      //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem recv: %d, id: %s.\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    break_gpu_event(functionName, task, endTime + id->syncOffset(), id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_RECV, id, transferSize, task);
    }
  } else if (unifmemType == BytesDtoH) {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite());
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_unifmems++;
      //since these copies are record on the host, start the parent timer here
      record_context_event(UnifiedMemoryEventDtoH, transferSize, task, syncStartTime + 1);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //printf("TAU: putting message into trace file.\n");
    //printf("[%f] onesided event mem send: %f.\n", startTime, transferSize);
    break_gpu_event(functionName, task, endTime + id->syncOffset(), id->getCallingSite());
    //Inorder to capture the entire memcpy transaction time start the send/recived
    //at the start of the event
    if (TauEnv_get_tracing()) {
      TauTraceOneSidedMsg(MESSAGE_SEND, id, transferSize, task);
    }
  } else {
    stage_gpu_event(functionName, task, syncStartTime, id->getCallingSite());
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_SEND) {
      TauTraceOneSidedMsg(direction, id, transferSize, task);
    }
    if (transferSize != TAU_GPU_UNKNOWN_TRANSFER_SIZE) {
      counted_unifmems++;
      record_context_event(UnifiedMemoryEventPageFault, transferSize, task, syncStartTime + 1);
      //TAU_EVENT(MemoryCopyEventDtoH(), transferSize);
#ifdef DEBUG_PROF		
      TAU_VERBOSE("[%f] onesided event mem send: %d, id: %s\n", startTime, transferSize,
          id->gpuIdentifier());
#endif
    }
    //TAU_REGISTER_EVENT(MemoryCopyEventDtoH, "Memory copied from Device to Host");
    //TauTraceEventSimple(TAU_ONESIDED_MESSAGE_RECV, transferSize, RtsLayer::myThread());
    //TauTraceOneSidedMsg(MESSAGE_SEND, device, transferSize, gpuTask);
    break_gpu_event(functionName, task, endTime + id->syncOffset(), id->getCallingSite());
    if (TauEnv_get_tracing() && direction == MESSAGE_RECIPROCAL_RECV) {
      TauTraceOneSidedMsg(direction, id, transferSize, task);
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
  //int task = event->getTaskId();
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
  //int task = event->getTaskId();
  //task = contextId;
  
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
    //              endTime + event->syncOffset(), event->getCallingSite());
  }
  else if (dataType == MemOps) {
    // stage_gpu_event(functionName, task,
    //              startTime + event->syncOffset(), event->getCallingSite());
    TAU_CONTEXT_EVENT(MemoryOps, transferSize);
    // TAU_CONTEXT_EVENT_THREAD(MemoryClockEvent, transferSize, task);
    // break_gpu_event(functionName, task,
    //              endTime + event->syncOffset(), event->getCallingSite());
  }
  else if (dataType == CtrlOps) {
    // stage_gpu_event(functionName, task,
    //              startTime + event->syncOffset(), event->getCallingSite);
    TAU_CONTEXT_EVENT(ControlOps, transferSize);
    // TAU_CONTEXT_EVENT_THREAD(PowerUtilizationEvent, transferSize, task);
    // break_gpu_event(functionName, task,
    //              endTime + event->syncOffset(), event->getCallingSite);
  }
}


// // Make samples based on function/file/line:
// // 1) Read each instruction_event sample
// // 2) Check if existing source entry / line no (sid) in srcLocMap available. 
// // --- If so, samples+=1. Otherwise, ignore (return).
// // 3) Calculate timestamp_delta
// // --- Need to keep track of time spent in routine

// void Tau_gpu_register_instruction_event(GpuEvent *id, double start, double stop, double delta_tstamp, const char* name, uint32_t correlationId, uint32_t sourceLocatorId, uint32_t functionId, uint32_t pcOffset, uint32_t executed, uint32_t threadsExecuted) {

//   int task = get_task(id);

//   if (correlationId != 0) { // jibberish
//     // add to each
//     InstrSampling is;
//     is.sourceLocatorId = sourceLocatorId;
//     is.functionId = functionId;
//     is.pcOffset = pcOffset;
//     is.correlationId = correlationId;
//     is.executed = executed;
//     is.threadsExecuted = threadsExecuted;
//     is.timestamp_delta = delta_tstamp;
//     is.timestamp_current = stop;
//     is.timestamp_recent = start;
//     instrSrcMap[sourceLocatorId].push_back(is);
//     instrFuncMap[functionId].push_back(is);
//   }
//   else {
// #ifdef TAU_DEBUG_SASS
//     printf("[TauGpu]:  Jibberish\n");
// #endif
//   }
// }

// void Tau_gpu_register_source_event(GpuEvent *event, double timestamp, const char* name, uint32_t sourceId, const char *fileName, uint32_t lineNumber) {

//   int task = get_task(event);

//   SourceSampling sourcesamp;

//   // check if doesn't exist
//   if(!srcLocMap.count(sourceId)) {
// 	sourcesamp.fid = -1;	
//   }
//   sourcesamp.sid = sourceId;
//   sourcesamp.fileName = (char *)fileName;
//   sourcesamp.lineNumber = lineNumber;
//   sourcesamp.timestamp_delta = 1;
//   sourcesamp.samples = 1;
//   srcLocMap[sourceId] = sourcesamp;

// #if TAU_DEBUG_SASS
//   printf("In TauGpu.cpp, Tau_gpu_register_source_event\n");
//     printf("id->syncOffset(): %d\n", event->syncOffset());
//   //printf("SOURCE_MAP: srcLocMap[%i].filename: %s\n", sourceId, srcLocMap[sourceId].fileName);
//   printSourceMap();
// #endif


// }

// void Tau_gpu_register_func_event(GpuEvent *event, int deviceId, double timestamp, const char* name, uint32_t contextId, 
// 				 uint32_t functionIndex, uint32_t id, uint32_t moduleid, const char *kname, const char *demangled)
// {
// #if TAU_DEBUG_SASS
//   printf("[TauGpu]:  contextId: %i, functionIndex: %i, id: %i\n", contextId, functionIndex, id);
// #endif
//   // blank funcId=1
//   if (contextId == 0)
//     return;
//   curr_device_id = deviceId;
//   // printf("curr_device_id: %i\n", curr_device_id);

//   FuncSampling fs;

//   if(!funcMap.count(functionIndex)) {
//     fs.calls = 0;
//     fs.kernel_launches = 0;
//   }
//   fs.fid = id;
//   fs.contextId = contextId;
//   fs.moduleId = moduleid;
//   fs.functionIndex = functionIndex;
//   fs.name = strdup(kname);
//   fs.demangled = strdup(demangled);
//   fs.funcinfo_created = false;
//   funcMap[id] = fs;

// #if TAU_DEBUG_SASS
//   printf("In TauGpu.cpp, Tau_gpu_register_func_event\n");
//     printf("id->syncOffset(): %d\n", event->syncOffset());
//   printFuncMap();
// #endif

// }

/*
 Initialization routine for TAU
 */
void Tau_gpu_init(void)
{
  // #if not defined(PTHREADS) && not defined(TAU_GPU)
#if not defined(PTHREADS)
  Tau_create_top_level_timer_if_necessary();
#endif

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
  // #if not defined(PTHREADS) && not defined(TAU_GPU)
#if not defined(PTHREADS)
  printf("stopping level %d tasks.\n", number_of_tasks);
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

// // SASS Helper Functions
// double getKernelExecutionTimes(uint32_t functionIndex)
// {
//   double kernel_exec_time;
//   kernel_exec_time = 0;
  
//   std::list<InstrSampling> instrFunc_list = instrFuncMap.find(functionIndex)->second;

//   for (std::list<InstrSampling>::iterator it2 = instrFunc_list.begin();
//        it2 != instrFunc_list.end(); ++it2) {
//     kernel_exec_time += it2->timestamp_delta;
//   }

//   return kernel_exec_time;	
// }

// void resetKernelExecutionTimes(uint32_t functionIndex)
// {
// #ifdef TAU_DEBUG_SASS
//   printf("[TauGpu]:  About to remove items on list\n");
// #endif
//   //InstrSampling is_temp = instrFuncMap.find(functionIndex)->second.back();
//   instrFuncMap.find(functionIndex)->second.clear();
//   //instrFuncMap[functionIndex].push_back(is_temp);

// }


// uint32_t getKernelSamples(uint32_t functionIndex) 
// {
//   std::list<InstrSampling> instrFunc_list = instrFuncMap.find(functionIndex)->second;
//   return instrFunc_list.size();	
// }

// uint32_t getUniqueKernelLaunches(uint32_t functionIndex)
// {
//   std::list<InstrSampling> instrFunc_list = instrFuncMap.find(functionIndex)->second;
//   std::set<uint32_t> corrIdFromInstr_set;
//   for (std::list<InstrSampling>::iterator it=instrFunc_list.begin(); 
//        it != instrFunc_list.end(); ++it) {
//     InstrSampling is = *it;
//     corrIdFromInstr_set.insert(is.correlationId);
//   }
//   return corrIdFromInstr_set.size();
// }


// const char* getKernelFilePath(uint32_t functionIndex)
// {
//   // assuming first entry's filepath is same for all
//   uint32_t srcId_temp = instrFuncMap.find(functionIndex)->second.front().sourceLocatorId;
//   return srcLocMap.find(srcId_temp)->second.fileName;
// }

// uint32_t getKernelLineNo(uint32_t functionIndex)
// {
//   uint32_t lineNumber;
//   lineNumber = 99999999;
//   std::list<InstrSampling> instrFunc_list = instrFuncMap.find(functionIndex)->second;

//   std::set<uint32_t> srcIdFromInstr_set;
//   for (std::list<InstrSampling>::iterator it=instrFunc_list.begin(); 
//        it != instrFunc_list.end(); ++it) {
//     InstrSampling is = *it;
//     srcIdFromInstr_set.insert(is.sourceLocatorId);
//   }
//   for (std::set<uint32_t>::iterator it2=srcIdFromInstr_set.begin();
//        it2 != srcIdFromInstr_set.end(); it2++) { 
//     uint32_t srcId = *it2;
//     SourceSampling ss = srcLocMap.find(srcId)->second;
//     if (lineNumber > ss.lineNumber) {
//       lineNumber = ss.lineNumber;
//     }
//   }

//   return lineNumber;	

// }

// void printInstrMap(void)
// {
//   // we only care about each kernel, get lowest number line of source code reference
//   for(std::map<uint32_t, InstrSampling>::iterator iter = instrSamplingMap.begin();
//       iter != instrSamplingMap.end(); ++iter) {
//     InstrSampling instrTemp = iter->second;
    
//     printf("~~~ Instr Sampling MAP ~~~\n  tstamp_delta: %lu%, SourceLocatorID: %i\n  FunctionID: %i, UniqueKernels: %i, samples: %i\n~~~~~~\n",
// 	   instrTemp.timestamp_delta, instrTemp.sourceLocatorId, instrTemp.functionId, instrTemp.uniqueKernels, instrTemp.samples);
//   }
// }

// void printSourceMap(void)
// {
//   for (std::map<uint32_t, SourceSampling>::iterator it=srcLocMap.begin(); 
//        it != srcLocMap.end(); ++it) {
//     SourceSampling sTemp = it->second;
//     printf("~~~ srcLocMap ~~~\n  id %u, fileName %s, lineNumber %u\n  timestamp_delta %f, timestamp_recent %f, samples %i, functionId %i\n~~~~~~\n",
// 	   sTemp.sid, sTemp.fileName, sTemp.lineNumber, sTemp.timestamp_delta, 
// 	   sTemp.timestamp_recentacc, sTemp.samples, sTemp.fid);
//   }
// }

// void printFuncMap(void)
// {
//   // iterate map, print out
//   for (std::map<uint32_t, FuncSampling>::iterator it=funcMap.begin(); it != funcMap.end(); ++it) {
//     FuncSampling fTemp = it->second;
//     printf("~~~ funcMap ~~~\n  id %u, ctx %u, moduleId %u\n  functionIndex %u, name %s, demangled %s\n~~~~~~\n",
// 	   fTemp.fid, fTemp.contextId, fTemp.moduleId,fTemp.functionIndex, fTemp.name,
// 	   fTemp.demangled);
//   }
// }
