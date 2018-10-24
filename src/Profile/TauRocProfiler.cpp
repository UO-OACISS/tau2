/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include <hsa.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <atomic>
#include "rocprofiler.h"
#include "util/hsa_rsrc_factory.h"
#include "Profile/Profiler.h"
#include <dlfcn.h>

/*
#include "ctrl/run_kernel.h"
#include "ctrl/test_aql.h"
#include "ctrl/test_hsa.h"
#include "dummy_kernel/dummy_kernel.h"
#include "simple_convolution/simple_convolution.h"
#include "util/test_assert.h"
*/

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define TAU_MAX_ROCM_QUEUES 512
const char * pthread_orig_name = "libTAU-pthread.so";
static void *pthread_dso_handle = NULL;
static int tau_initialized_queues[TAU_MAX_ROCM_QUEUES] = { 0 };

extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" void Tau_metadata_task(const char *name, const char *value, int tid);
extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);
static unsigned long long tau_last_timestamp_ns = 0L;

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Tool is unloaded
volatile bool is_loaded = false;

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

// Context stored entry type
struct context_entry_t {
  bool valid;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
};

void Tau_add_metadata_for_task(int taskid) {
  char buf[1024];
  sprintf(buf, "%d", taskid);
  Tau_metadata_task("ROCM Task ID", buf, taskid);
}
// Dump stored context entry
void dump_context_entry(context_entry_t* entry) {
  TAU_VERBOSE("inside dump_context_entry\n");
  int taskid, queueid;
  unsigned long long timestamp = 0L;
  volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
  while (valid->load() == false) sched_yield();

  const std::string kernel_name = entry->data.kernel_name;
  const rocprofiler_dispatch_record_t* record = entry->data.record;

  if (!record) return; // there is nothing to do here. 

  fflush(stdout);
  queueid = entry->data.queue_id; 
  taskid = tau_initialized_queues[queueid]; 
  if (taskid == -1) { // not initialized
    TAU_CREATE_TASK(taskid);
    TAU_VERBOSE("dump_context_entry: associating queueid %d with taskid %d\n", queueid, taskid);
    tau_initialized_queues[queueid] = taskid;  
    timestamp = record->dispatch; 
    // Set the timestamp for TAUGPU_TIME:
    metric_set_gpu_timestamp(taskid, ((double)timestamp/1e3));
    Tau_create_top_level_timer_if_necessary_task(taskid); 
    Tau_add_metadata_for_task(taskid);
  }
  
  timestamp = record->begin;
  metric_set_gpu_timestamp(taskid, ((double)timestamp/1e3)); // convert to microseconds
  TAU_START_TASK(kernel_name.c_str(), taskid);

  timestamp = record->end;
  metric_set_gpu_timestamp(taskid, ((double)timestamp/1e3)); // convert to microseconds
  TAU_STOP_TASK(kernel_name.c_str(), taskid);
  tau_last_timestamp_ns = record->complete; 
  
#ifdef DEBUG_PROF
  fprintf(stdout, "kernel symbol(0x%lx) name(\"%s\") tid(%ld) queue-id(%u) gpu-id(%u) ",
    entry->data.kernel_object,
    kernel_name.c_str(),
    entry->data.thread_id,
    entry->data.queue_id,
    HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index);
  if (record) fprintf(stdout, "time(%lu,%lu,%lu,%lu)",
    record->dispatch,
    record->begin,
    record->end,
    record->complete);
  fprintf(stdout, "\n");
  fflush(stdout);
#endif /* DEBUG_PROF */

  rocprofiler_group_t& group = entry->group;
  if (group.context == NULL) {
    fprintf(stderr, "tool error: context is NULL\n");
    abort();
  }

  rocprofiler_close(group.context);
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool context_handler(rocprofiler_group_t group, void* arg) {
  TAU_VERBOSE("Inside context_handler\n");
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  dump_context_entry(entry);
  delete entry;

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* /*user_data*/,
                               rocprofiler_group_t* group) {
  TAU_VERBOSE("Inside dispatch_callback\n");
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

  // Profiling context
  rocprofiler_t* context = NULL;

  // Context entry
  context_entry_t* entry = new context_entry_t();

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = context_handler;
  properties.handler_arg = (void*)entry;

  // Open profiling context
  status = rocprofiler_open(callback_data->agent, NULL, 0,
                            &context, 0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);

  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  check_status(status);

  // Fill profiling context entry
  entry->agent = callback_data->agent;
  entry->group = *group;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

  return HSA_STATUS_SUCCESS;
}

void initialize() {
  TAU_VERBOSE("Inside initialize\n");
  // Getting GPU device info
  const AgentInfo* agent_info = NULL;
  if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) {
    fprintf(stderr, "GetGpuAgentInfo failed\n");
    abort();
  }

  // Adding dispatch observer
  rocprofiler_queue_callbacks_t callbacks_ptrs{};
  callbacks_ptrs.dispatch = dispatch_callback;
  int err=rocprofiler_set_queue_callbacks(callbacks_ptrs, NULL);
  TAU_VERBOSE("err=%d, rocprofiler_set_queue_callbacks\n", err);
}

void cleanup() {
  // Unregister dispatch callback
  TAU_VERBOSE("Inside cleanup\n");
  rocprofiler_remove_queue_callbacks();

  // Dump stored profiling output data
  fflush(stdout);
}



// Tool constructor
extern "C" PUBLIC_API void OnLoadToolProp(rocprofiler_settings_t* settings)
{
  TAU_VERBOSE("Inside OnLoadToolProp\n");
  Tau_init_initializeTAU();
  
  if (pthread_dso_handle == NULL) { }

/*
  if (pthread_dso_handle == NULL)
    pthread_dso_handle = (void *) dlopen(pthread_orig_name, RTLD_NOW);

  if (pthread_dso_handle == NULL) {
    perror("OnLoadToolProp: Error opening libTAU-pthread.so library in dlopen call");
    return;
  }
  TAU_VERBOSE("Successfully loaded libTAU-pthread.so\n");
*/
  for (int i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    tau_initialized_queues[i] = -1; // set it explicitly
  }
#if (!(defined (TAU_MPI) || (TAU_SHMEM)))
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI || TAU_SHMEM */
  //Tau_create_top_level_timer_if_necessary();
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (is_loaded) return;
  is_loaded = true;
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  // Enable timestamping
  settings->timestamp_on = true;

  // Initialize profiling
  initialize();
}

// Tool destructor
extern "C" PUBLIC_API void OnUnloadTool() {
  TAU_VERBOSE("Inside OnUnloadTool\n");
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (!is_loaded) return;
  is_loaded = false;
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  // Final resources cleanup
  cleanup();
  Tau_stop_top_level_timer_if_necessary(); // thread 0;
  for (int i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    if (tau_initialized_queues[i] != -1) { 
      //std::cout <<"Closing "<<i<<" last timestamp = "<<tau_last_timestamp_ns<<std::endl;
      metric_set_gpu_timestamp(i, ((double)tau_last_timestamp_ns/1e3)); // convert to microseconds
      Tau_stop_top_level_timer_if_necessary_task(i);
    }
  }
}

extern "C" DESTRUCTOR_API void destructor() {
  if (is_loaded == true) OnUnloadTool();
}
