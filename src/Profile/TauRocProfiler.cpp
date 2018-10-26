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
#include "Profile/rocprofiler.h"
#include "Profile/hsa_rsrc_factory.h"
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

extern "C" x_uint64 TauTraceGetTimeStamp();
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" void Tau_metadata_task(const char *name, const char *value, int tid);
extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);
static unsigned long long tau_last_timestamp_ns = 0L;
static unsigned long long offset_timestamp = 0L;
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


void metric_set_synchronized_gpu_timestamp(int tid, double value){
	if (offset_timestamp == 0L)
	{
		offset_timestamp=TauTraceGetTimeStamp() - ((double)value);
	}
	metric_set_gpu_timestamp(tid, offset_timestamp+value);
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
    metric_set_synchronized_gpu_timestamp(taskid, ((double)timestamp/1e3));
    Tau_create_top_level_timer_if_necessary_task(taskid); 
    Tau_add_metadata_for_task(taskid);
  }
  
  timestamp = record->begin;
  metric_set_synchronized_gpu_timestamp(taskid, ((double)timestamp/1e3)); // convert to microseconds
  TAU_START_TASK(kernel_name.c_str(), taskid);

  timestamp = record->end;
  metric_set_synchronized_gpu_timestamp(taskid, ((double)timestamp/1e3)); // convert to microseconds
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
  HsaRsrcFactory::Instance().PrintGpuAgents("ROCm");
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
      metric_set_synchronized_gpu_timestamp(i, ((double)tau_last_timestamp_ns/1e3)); // convert to microseconds
      Tau_stop_top_level_timer_if_necessary_task(i);
    }
  }
}

extern "C" DESTRUCTOR_API void destructor() {
  if (is_loaded == true) OnUnloadTool();
}

/* hsa_rsrc_factory.cpp */

/* #include "util/hsa_rsrc_factory.h" */

#include <dlfcn.h>
#include <fcntl.h>
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <hsa_ext_finalize.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Callback function to get available in the system agents
hsa_status_t HsaRsrcFactory::GetHsaAgentsCallback(hsa_agent_t agent, void* data) {
  hsa_status_t status = HSA_STATUS_ERROR;
  HsaRsrcFactory* hsa_rsrc = reinterpret_cast<HsaRsrcFactory*>(data);
  const AgentInfo* agent_info = hsa_rsrc->AddAgentInfo(agent);
  if (agent_info != NULL) status = HSA_STATUS_SUCCESS;
  return status;
}

// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.
static hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void* data, bool kern_arg) {
  hsa_status_t err;
  hsa_amd_segment_t segment;
  uint32_t flag;

  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  CHECK_STATUS("hsa_amd_memory_pool_get_info", err);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  CHECK_STATUS("hsa_amd_memory_pool_get_info", err);

  uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;

  if ((karg_st == 0 && kern_arg) || (karg_st != 0 && !kern_arg)) {
    return HSA_STATUS_SUCCESS;
  }

  *(reinterpret_cast<hsa_amd_memory_pool_t*>(data)) = pool;
  return HSA_STATUS_INFO_BREAK;
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindGlobalPool(pool, data, false);
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindGlobalPool(pool, data, true);
}

// Constructor of the class
HsaRsrcFactory::HsaRsrcFactory(bool initialize_hsa) : initialize_hsa_(initialize_hsa) {
  hsa_status_t status;

  cpu_pool_ = NULL;
  kern_arg_pool_ = NULL;

  // Initialize the Hsa Runtime
  if (initialize_hsa_) {
    status = hsa_init();
    CHECK_STATUS("Error in hsa_init", status);
  }

  // Discover the set of Gpu devices available on the platform
  status = hsa_iterate_agents(GetHsaAgentsCallback, this);
  CHECK_STATUS("Error Calling hsa_iterate_agents", status);
  if (cpu_pool_ == NULL) CHECK_STATUS("CPU memory pool is not found", HSA_STATUS_ERROR);
  if (kern_arg_pool_ == NULL) CHECK_STATUS("Kern-arg memory pool is not found", HSA_STATUS_ERROR);

  // Get AqlProfile API table
  aqlprofile_api_ = {0};
#ifdef ROCP_LD_AQLPROFILE
  status = LoadAqlProfileLib(&aqlprofile_api_);
#else
  status = hsa_system_get_extension_table(HSA_EXTENSION_AMD_AQLPROFILE, 1, 0, &aqlprofile_api_);
#endif
  CHECK_STATUS("aqlprofile API table load failed", status);

  // Get Loader API table
  loader_api_ = {0};
  status = hsa_system_get_extension_table(HSA_EXTENSION_AMD_LOADER, 1, 0, &loader_api_);
  CHECK_STATUS("loader API table query failed", status);

  // Instantiate HSA timer
  timer_ = new HsaTimer;
  CHECK_STATUS("HSA timer allocation failed",
    (timer_ == NULL) ? HSA_STATUS_ERROR : HSA_STATUS_SUCCESS);

  // System timeout
  timeout_ = (timeout_ns_ == HsaTimer::TIMESTAMP_MAX) ? timeout_ns_ : timer_->ns_to_sysclock(timeout_ns_);
}

// Destructor of the class
HsaRsrcFactory::~HsaRsrcFactory() {
  delete timer_;
  for (auto p : cpu_list_) delete p;
  for (auto p : gpu_list_) delete p;
  if (initialize_hsa_) {
    hsa_status_t status = hsa_shut_down();
    CHECK_STATUS("Error in hsa_shut_down", status);
  }
}

hsa_status_t HsaRsrcFactory::LoadAqlProfileLib(aqlprofile_pfn_t* api) {
  void* handle = dlopen(kAqlProfileLib, RTLD_NOW);
  if (handle == NULL) {
    fprintf(stderr, "Loading '%s' failed, %s\n", kAqlProfileLib, dlerror());
    return HSA_STATUS_ERROR;
  }
  dlerror(); /* Clear any existing error */

  api->hsa_ven_amd_aqlprofile_error_string =
      (decltype(::hsa_ven_amd_aqlprofile_error_string)*)dlsym(
          handle, "hsa_ven_amd_aqlprofile_error_string");
  api->hsa_ven_amd_aqlprofile_validate_event =
      (decltype(::hsa_ven_amd_aqlprofile_validate_event)*)dlsym(
          handle, "hsa_ven_amd_aqlprofile_validate_event");
  api->hsa_ven_amd_aqlprofile_start =
      (decltype(::hsa_ven_amd_aqlprofile_start)*)dlsym(handle, "hsa_ven_amd_aqlprofile_start");
  api->hsa_ven_amd_aqlprofile_stop =
      (decltype(::hsa_ven_amd_aqlprofile_stop)*)dlsym(handle, "hsa_ven_amd_aqlprofile_stop");
#ifdef AQLPROF_NEW_API
  api->hsa_ven_amd_aqlprofile_read =
      (decltype(::hsa_ven_amd_aqlprofile_read)*)dlsym(handle, "hsa_ven_amd_aqlprofile_read");
#endif
  api->hsa_ven_amd_aqlprofile_legacy_get_pm4 =
      (decltype(::hsa_ven_amd_aqlprofile_legacy_get_pm4)*)dlsym(
          handle, "hsa_ven_amd_aqlprofile_legacy_get_pm4");
  api->hsa_ven_amd_aqlprofile_get_info = (decltype(::hsa_ven_amd_aqlprofile_get_info)*)dlsym(
      handle, "hsa_ven_amd_aqlprofile_get_info");
  api->hsa_ven_amd_aqlprofile_iterate_data =
      (decltype(::hsa_ven_amd_aqlprofile_iterate_data)*)dlsym(
          handle, "hsa_ven_amd_aqlprofile_iterate_data");

  return HSA_STATUS_SUCCESS;
}

// Add system agent info
const AgentInfo* HsaRsrcFactory::AddAgentInfo(const hsa_agent_t agent) {
  // Determine if device is a Gpu agent
  hsa_status_t status;
  AgentInfo* agent_info = NULL;

  hsa_device_type_t type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  CHECK_STATUS("Error Calling hsa_agent_get_info", status);

  if (type == HSA_DEVICE_TYPE_CPU) {
    agent_info = new AgentInfo{};
    agent_info->dev_id = agent;
    agent_info->dev_type = HSA_DEVICE_TYPE_CPU;
    agent_info->dev_index = cpu_list_.size();

    status = hsa_amd_agent_iterate_memory_pools(agent, FindStandardPool, &agent_info->cpu_pool);
    if ((status == HSA_STATUS_INFO_BREAK) && (cpu_pool_ == NULL)) cpu_pool_ = &agent_info->cpu_pool;
    status = hsa_amd_agent_iterate_memory_pools(agent, FindKernArgPool, &agent_info->kern_arg_pool);
    if ((status == HSA_STATUS_INFO_BREAK) && (kern_arg_pool_ == NULL)) kern_arg_pool_ = &agent_info->kern_arg_pool;
    agent_info->gpu_pool = {};

    cpu_list_.push_back(agent_info);
    cpu_agents_.push_back(agent);
  }

  if (type == HSA_DEVICE_TYPE_GPU) {
    agent_info = new AgentInfo{};
    agent_info->dev_id = agent;
    agent_info->dev_type = HSA_DEVICE_TYPE_GPU;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_info->name);
    strncpy(agent_info->gfxip, agent_info->name, 4);
    agent_info->gfxip[4] = '\0';
    hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &agent_info->max_wave_size);
    hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &agent_info->max_queue_size);
    hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_info->profile);
    agent_info->is_apu = (agent_info->profile == HSA_PROFILE_FULL) ? true : false;
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                       &agent_info->cu_num);
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
                       &agent_info->waves_per_cu);
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU),
                       &agent_info->simds_per_cu);
    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES),
                       &agent_info->se_num);
    hsa_agent_get_info(agent,
                       static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE),
                       &agent_info->shader_arrays_per_se);

    agent_info->cpu_pool = {};
    agent_info->kern_arg_pool = {};
    status = hsa_amd_agent_iterate_memory_pools(agent, FindStandardPool, &agent_info->gpu_pool);
    CHECK_ITER_STATUS("hsa_amd_agent_iterate_memory_pools(gpu pool)", status);

    // Set GPU index
    agent_info->dev_index = gpu_list_.size();
    gpu_list_.push_back(agent_info);
    gpu_agents_.push_back(agent);
  }

  if (agent_info) agent_map_[agent.handle] = agent_info;

  return agent_info;
}

// Return systen agent info
const AgentInfo* HsaRsrcFactory::GetAgentInfo(const hsa_agent_t agent) {
  const AgentInfo* agent_info = NULL;
  auto it = agent_map_.find(agent.handle);
  if (it != agent_map_.end()) {
    agent_info = it->second;
  }
  return agent_info;
}

// Get the count of Hsa Gpu Agents available on the platform
//
// @return uint32_t Number of Gpu agents on platform
//
uint32_t HsaRsrcFactory::GetCountOfGpuAgents() { return uint32_t(gpu_list_.size()); }

// Get the count of Hsa Cpu Agents available on the platform
//
// @return uint32_t Number of Cpu agents on platform
//
uint32_t HsaRsrcFactory::GetCountOfCpuAgents() { return uint32_t(cpu_list_.size()); }

// Get the AgentInfo handle of a Gpu device
//
// @param idx Gpu Agent at specified index
//
// @param agent_info Output parameter updated with AgentInfo
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::GetGpuAgentInfo(uint32_t idx, const AgentInfo** agent_info) {
  // Determine if request is valid
  uint32_t size = uint32_t(gpu_list_.size());
  if (idx >= size) {
    return false;
  }

  // Copy AgentInfo from specified index
  *agent_info = gpu_list_[idx];

  return true;
}

// Get the AgentInfo handle of a Cpu device
//
// @param idx Cpu Agent at specified index
//
// @param agent_info Output parameter updated with AgentInfo
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::GetCpuAgentInfo(uint32_t idx, const AgentInfo** agent_info) {
  // Determine if request is valid
  uint32_t size = uint32_t(cpu_list_.size());
  if (idx >= size) {
    return false;
  }

  // Copy AgentInfo from specified index
  *agent_info = cpu_list_[idx];
  return true;
}

// Create a Queue object and return its handle. The queue object is expected
// to support user requested number of Aql dispatch packets.
//
// @param agent_info Gpu Agent on which to create a queue object
//
// @param num_Pkts Number of packets to be held by queue
//
// @param queue Output parameter updated with handle of queue object
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::CreateQueue(const AgentInfo* agent_info, uint32_t num_pkts,
                                 hsa_queue_t** queue) {
  hsa_status_t status;
  status = hsa_queue_create(agent_info->dev_id, num_pkts, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                            UINT32_MAX, UINT32_MAX, queue);
  return (status == HSA_STATUS_SUCCESS);
}

// Create a Signal object and return its handle.
// @param value Initial value of signal object
// @param signal Output parameter updated with handle of signal object
// @return bool true if successful, false otherwise
bool HsaRsrcFactory::CreateSignal(uint32_t value, hsa_signal_t* signal) {
  hsa_status_t status;
  status = hsa_signal_create(value, 0, NULL, signal);
  return (status == HSA_STATUS_SUCCESS);
}

// Allocate memory for use by a kernel of specified size in specified
// agent's memory region.
// @param agent_info Agent from whose memory region to allocate
// @param size Size of memory in terms of bytes
// @return uint8_t* Pointer to buffer, null if allocation fails.
uint8_t* HsaRsrcFactory::AllocateLocalMemory(const AgentInfo* agent_info, size_t size) {
  hsa_status_t status = HSA_STATUS_ERROR;
  uint8_t* buffer = NULL;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  status = hsa_amd_memory_pool_allocate(agent_info->gpu_pool, size, 0, reinterpret_cast<void**>(&buffer));
  uint8_t* ptr = (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
  return ptr;
}

// Allocate memory to pass kernel parameters.
// Memory is alocated accessible for all CPU agents and for GPU given by AgentInfo parameter.
// @param agent_info Agent from whose memory region to allocate
// @param size Size of memory in terms of bytes
// @return uint8_t* Pointer to buffer, null if allocation fails.
uint8_t* HsaRsrcFactory::AllocateKernArgMemory(const AgentInfo* agent_info, size_t size) {
  hsa_status_t status = HSA_STATUS_ERROR;
  uint8_t* buffer = NULL;
  if (!cpu_agents_.empty()) {
    size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
    status = hsa_amd_memory_pool_allocate(*kern_arg_pool_, size, 0, reinterpret_cast<void**>(&buffer));
    // Both the CPU and GPU can access the kernel arguments
    if (status == HSA_STATUS_SUCCESS) {
      hsa_agent_t ag_list[1] = {agent_info->dev_id};
      status = hsa_amd_agents_allow_access(1, ag_list, NULL, buffer);
    }
  }
  uint8_t* ptr = (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
  return ptr;
}

// Allocate system memory accessible by both CPU and GPU
// @param agent_info Agent from whose memory region to allocate
// @param size Size of memory in terms of bytes
// @return uint8_t* Pointer to buffer, null if allocation fails.
uint8_t* HsaRsrcFactory::AllocateSysMemory(const AgentInfo* agent_info, size_t size) {
  hsa_status_t status = HSA_STATUS_ERROR;
  uint8_t* buffer = NULL;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  if (!cpu_agents_.empty()) {
    status = hsa_amd_memory_pool_allocate(*cpu_pool_, size, 0, reinterpret_cast<void**>(&buffer));
    // Both the CPU and GPU can access the memory
    if (status == HSA_STATUS_SUCCESS) {
      hsa_agent_t ag_list[1] = {agent_info->dev_id};
      status = hsa_amd_agents_allow_access(1, ag_list, NULL, buffer);
    }
  }
  uint8_t* ptr = (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
  return ptr;
}

// Allocate memory for command buffer.
// @param agent_info Agent from whose memory region to allocate
// @param size Size of memory in terms of bytes
// @return uint8_t* Pointer to buffer, null if allocation fails.
uint8_t* HsaRsrcFactory::AllocateCmdMemory(const AgentInfo* agent_info, size_t size) {
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  uint8_t* ptr = (agent_info->is_apu && CMD_MEMORY_MMAP)
      ? reinterpret_cast<uint8_t*>(
            mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED | MAP_ANONYMOUS, 0, 0))
      : AllocateSysMemory(agent_info, size);
  return ptr;
}

// Wait signal
void HsaRsrcFactory::SignalWait(const hsa_signal_t& signal) const {
  while (1) {
    const hsa_signal_value_t signal_value =
      hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, timeout_, HSA_WAIT_STATE_BLOCKED);
    if (signal_value == 0) {
      break;
    } else {
      CHECK_STATUS("hsa_signal_wait_scacquire()", HSA_STATUS_ERROR);
    }
  }
}

// Wait signal with signal value restore
void HsaRsrcFactory::SignalWaitRestore(const hsa_signal_t& signal, const hsa_signal_value_t& signal_value) const {
  SignalWait(signal);
  hsa_signal_store_relaxed(const_cast<hsa_signal_t&>(signal), signal_value);
}

// Copy data from GPU to host memory
bool HsaRsrcFactory::Memcpy(const hsa_agent_t& agent, void* dst, const void* src, size_t size) {
  hsa_status_t status = HSA_STATUS_ERROR;
  if (!cpu_agents_.empty()) {
    hsa_signal_t s = {};
    status = hsa_signal_create(1, 0, NULL, &s);
    CHECK_STATUS("hsa_signal_create()", status);
    status = hsa_amd_memory_async_copy(dst, cpu_agents_[0], src, agent, size, 0, NULL, s);
    CHECK_STATUS("hsa_amd_memory_async_copy()", status);
    SignalWait(s);
    status = hsa_signal_destroy(s);
    CHECK_STATUS("hsa_signal_destroy()", status);
  }
  return (status == HSA_STATUS_SUCCESS);
}
bool HsaRsrcFactory::Memcpy(const AgentInfo* agent_info, void* dst, const void* src, size_t size) {
  return Memcpy(agent_info->dev_id, dst, src, size);
}

// Memory free method
bool HsaRsrcFactory::FreeMemory(void* ptr) {
  const hsa_status_t status = hsa_memory_free(ptr);
  CHECK_STATUS("hsa_memory_free", status);
  return (status == HSA_STATUS_SUCCESS);
}

// Loads an Assembled Brig file and Finalizes it into Device Isa
// @param agent_info Gpu device for which to finalize
// @param brig_path File path of the Assembled Brig file
// @param kernel_name Name of the kernel to finalize
// @param code_desc Handle of finalized Code Descriptor that could
// be used to submit for execution
// @return bool true if successful, false otherwise
bool HsaRsrcFactory::LoadAndFinalize(const AgentInfo* agent_info, const char* brig_path,
                                     const char* kernel_name, hsa_executable_t* executable,
                                     hsa_executable_symbol_t* code_desc) {
  hsa_status_t status = HSA_STATUS_ERROR;

  // Build the code object filename
  std::string filename(brig_path);
  std::clog << "Code object filename: " << filename << std::endl;

  // Open the file containing code object
  hsa_file_t file_handle = open(filename.c_str(), O_RDONLY);
  if (file_handle == -1) {
    std::cerr << "Error: failed to load '" << filename << "'" << std::endl;
    assert(false);
    return false;
  }

  // Create code object reader
  hsa_code_object_reader_t code_obj_rdr = {0};
  status = hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "Failed to create code object reader '" << filename << "'" << std::endl;
    return false;
  }

  // Create executable.
  status = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     NULL, executable);
  CHECK_STATUS("Error in creating executable object", status);

  // Load code object.
  status = hsa_executable_load_agent_code_object(*executable, agent_info->dev_id, code_obj_rdr,
                                                 NULL, NULL);
  CHECK_STATUS("Error in loading executable object", status);

  // Freeze executable.
  status = hsa_executable_freeze(*executable, "");
  CHECK_STATUS("Error in freezing executable object", status);

  // Get symbol handle.
  hsa_executable_symbol_t kernelSymbol;
  status = hsa_executable_get_symbol(*executable, NULL, kernel_name, agent_info->dev_id, 0,
                                     &kernelSymbol);
  CHECK_STATUS("Error in looking up kernel symbol", status);

  // Update output parameter
  *code_desc = kernelSymbol;
  return true;
}

// Print the various fields of Hsa Gpu Agents
bool HsaRsrcFactory::PrintGpuAgents(const std::string& header) {
  //std::clog << header << " :" << std::endl;

  char key[1024], value[1024];

  const AgentInfo* agent_info;
  int size = uint32_t(gpu_list_.size());
  for (int idx = 0; idx < size; idx++) {
    agent_info = gpu_list_[idx];

    sprintf(key, "ROCM_AGENT_%d_NAME", idx);
    sprintf(value, "%s", agent_info->name);
    TAU_METADATA(key, value);
//    std::clog << "> agent[" << idx << "] :" << std::endl;
//    std::clog << ">> Name : " << agent_info->name << std::endl;

    sprintf(key, "ROCM_AGENT_%d_IS_APU", idx);
    sprintf(value, "%d", agent_info->is_apu);
    TAU_METADATA(key, value);

//    std::clog << ">> APU : " << agent_info->is_apu << std::endl;
    sprintf(key, "ROCM_AGENT_%d_HSA_PROFILE", idx);
    sprintf(value, "%d", agent_info->profile);
    TAU_METADATA(key, value);
//    std::clog << ">> HSAIL profile : " << agent_info->profile << std::endl;

    sprintf(key, "ROCM_AGENT_%d_MAX_WAVE_SIZE", idx);
    sprintf(value, "%d", agent_info->max_wave_size);
    TAU_METADATA(key, value);

//    std::clog << ">> Max Wave Size : " << agent_info->max_wave_size << std::endl;

    sprintf(key, "ROCM_AGENT_%d_MAX_QUEUE_SIZE", idx);
    sprintf(value, "%d", agent_info->max_queue_size);
    TAU_METADATA(key, value);

//    std::clog << ">> Max Queue Size : " << agent_info->max_queue_size << std::endl;

    sprintf(key, "ROCM_AGENT_%d_CU_NUMBER", idx);
    sprintf(value, "%d", agent_info->cu_num);
    TAU_METADATA(key, value);
//    std::clog << ">> CU number : " << agent_info->cu_num << std::endl;

    sprintf(key, "ROCM_AGENT_%d_WAVES_PER_CU", idx);
    sprintf(value, "%d", agent_info->waves_per_cu);
    TAU_METADATA(key, value);

//    std::clog << ">> Waves per CU : " << agent_info->waves_per_cu << std::endl;

    sprintf(key, "ROCM_AGENT_%d_SIMDs_PER_CU", idx);
    sprintf(value, "%d", agent_info->simds_per_cu);
    TAU_METADATA(key, value);

//    std::clog << ">> SIMDs per CU : " << agent_info->simds_per_cu << std::endl;

    sprintf(key, "ROCM_AGENT_%d_SE_NUMBER", idx);
    sprintf(value, "%d", agent_info->se_num);
    TAU_METADATA(key, value);
//    std::clog << ">> SE number : " << agent_info->se_num << std::endl;

    sprintf(key, "ROCM_AGENT_%d_SHADER_ARRAYS_PER_SE", idx);
    sprintf(value, "%d", agent_info->shader_arrays_per_se);
    TAU_METADATA(key, value);
//    std::clog << ">> Shader Arrays per SE : " << agent_info->shader_arrays_per_se << std::endl;
  

  
  }
  return true;
}

uint64_t HsaRsrcFactory::Submit(hsa_queue_t* queue, const void* packet) {
  const uint32_t slot_size_b = 0x40;

  // adevance command queue
  const uint64_t write_idx = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, write_idx + 1);
  while ((write_idx - hsa_queue_load_read_index_relaxed(queue)) >= queue->size) {
    sched_yield();
  }

  uint32_t slot_idx = (uint32_t)(write_idx % queue->size);
  uint32_t* queue_slot = reinterpret_cast<uint32_t*>((uintptr_t)(queue->base_address) + (slot_idx * slot_size_b));
  const uint32_t* slot_data = reinterpret_cast<const uint32_t*>(packet);

  // Copy buffered commands into the queue slot.
  // Overwrite the AQL invalid header (first dword) last.
  // This prevents the slot from being read until it's fully written.
  memcpy(&queue_slot[1], &slot_data[1], slot_size_b - sizeof(uint32_t));
  std::atomic<uint32_t>* header_atomic_ptr =
      reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
  header_atomic_ptr->store(slot_data[0], std::memory_order_release);

  // ringdoor bell
  hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  return write_idx;
}

uint64_t HsaRsrcFactory::Submit(hsa_queue_t* queue, const void* packet, size_t size_bytes) {
  const uint32_t slot_size_b = 0x40;
  if ((size_bytes & (slot_size_b - 1)) != 0) {
    fprintf(stderr, "HsaRsrcFactory::Submit: Bad packet size %zx\n", size_bytes);
    abort();
  }

  const char* begin = reinterpret_cast<const char*>(packet);
  const char* end = begin + size_bytes;
  uint64_t write_idx = 0;
  for (const char* ptr = begin; ptr < end; ptr += slot_size_b) {
    write_idx = Submit(queue, ptr);
  }

  return write_idx;
}

HsaRsrcFactory* HsaRsrcFactory::instance_ = NULL;
HsaRsrcFactory::mutex_t HsaRsrcFactory::mutex_;
HsaRsrcFactory::timestamp_t HsaRsrcFactory::timeout_ns_ = HsaTimer::TIMESTAMP_MAX;
