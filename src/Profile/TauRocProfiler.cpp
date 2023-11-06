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
#include <Profile/TauRocm.h>
#include "Profile/rocprofiler.h"
#include "Profile/hsa_rsrc_factory.h"

#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

#include <atomic>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t rocm_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Tool is unloaded
volatile bool is_loaded = false;
volatile bool is_callback_loaded = false;
// Profiling features
//rocprofiler_feature_t* features = NULL;
//unsigned feature_count = 0;

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void Tau_rocm_check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

/*
// Context stored entry type
struct context_entry_t {
  bool valid;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
};
*/

// kernel properties structure
struct kernel_properties_t {
  uint32_t grid_size;
  uint32_t workgroup_size;
  uint32_t lds_size;
  uint32_t scratch_size;
  uint32_t vgpr_count;
  uint32_t sgpr_count;
  uint32_t fbarrier_count;
  hsa_signal_t signal;
  uint64_t object;
};

// Context stored entry type
struct context_entry_t {
  bool valid;
  bool active;
  uint32_t index;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_feature_t* features;
  unsigned feature_count;
  rocprofiler_callback_data_t data;
  kernel_properties_t kernel_properties;
  //HsaRsrcFactory::symbols_map_it_t kernel_name_it;
  FILE* file_handle;
};

// Context callback arg
struct callbacks_arg_t {
  rocprofiler_pool_t** pools;
};

static std::set<rocprofiler_pool_t*>& get_pool_set() {
    static std::set<rocprofiler_pool_t*> theset;
    return theset;
}

// Handler callback arg
struct handler_arg_t {
  rocprofiler_feature_t* features;
  unsigned feature_count;
};

// Dump stored context entry
void Tau_rocm_dump_context_entry(context_entry_t* entry, rocprofiler_feature_t* features, unsigned feature_count) {
  TAU_VERBOSE("inside Tau_rocm_dump_context_entry\n");
  int taskid, queueid;
  unsigned long long timestamp = 0L;
  static unsigned long long last_timestamp = Tau_get_last_timestamp_ns();

  volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
  while (valid->load() == false) sched_yield();

  const std::string kernel_name = entry->data.kernel_name;
  const rocprofiler_dispatch_record_t* record = entry->data.record;

  if (!record) return; // there is nothing to do here.

  fflush(stdout);
  queueid = entry->data.queue_id;
  taskid = Tau_get_initialized_queues(queueid);

  if (taskid == -1) { // not initialized
    TAU_CREATE_TASK(taskid);
    TAU_VERBOSE("Tau_rocm_dump_context_entry: associating queueid %d with taskid %d\n", queueid, taskid);
    Tau_set_initialized_queues(queueid, taskid);
    timestamp = record->dispatch;
    Tau_check_timestamps(last_timestamp, timestamp, "NEW QUEUE", taskid);
    last_timestamp = timestamp;
    // Set the timestamp for TAUGPU_TIME:
    Tau_metric_set_synchronized_gpu_timestamp(taskid, ((double)timestamp/1e3));
    Tau_create_top_level_timer_if_necessary_task(taskid);
    Tau_add_metadata_for_task("TAU_TASK_ID", taskid, taskid);
    Tau_add_metadata_for_task("ROCM_GPU_ID", HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index, taskid);
    Tau_add_metadata_for_task("ROCM_QUEUE_ID", entry->data.queue_id, taskid);
    Tau_add_metadata_for_task("ROCM_THREAD_ID", entry->data.thread_id, taskid);
  }

  TAU_VERBOSE(" --> NEW EVENT --> \n");
  struct TauRocmEvent e(kernel_name, record->begin, record->end, taskid);
  TAU_VERBOSE("KERNEL: name: %s entry: %lu exit: %lu ...\n",
        kernel_name.c_str(), record->begin, record->end);
  /* Capture these with counters! */
  if (!TauEnv_get_thread_per_gpu_stream()) {
    std::stringstream ss;
    void* ue = nullptr;
    double value;
    std::string tmp;
    ss << "Grid Size : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(entry->kernel_properties.grid_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Work Group Size : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(entry->kernel_properties.workgroup_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    const AgentInfo* agent_info = HsaRsrcFactory::Instance().GetAgentInfo(entry->agent);
    ss << "LDS Memory Size : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    static const uint32_t lds_block_size = 128 * 4;
    value = (double)((entry->kernel_properties.lds_size + (lds_block_size - 1)) & ~(lds_block_size - 1));
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Scratch Memory Size : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(entry->kernel_properties.scratch_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Vector Register Size (VGPR) : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)((entry->kernel_properties.vgpr_count + 1) * agent_info->vgpr_block_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "Scalar Register Size (SGPR) : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)((entry->kernel_properties.sgpr_count + agent_info->sgpr_block_dflt) * agent_info->sgpr_block_size);
    Tau_userevent_thread(ue, value, taskid);

    ss.str("");
    ss << "fbarrier count : " << kernel_name;
    tmp = ss.str();
    ue = Tau_get_userevent(tmp.c_str());
    value = (double)(entry->kernel_properties.fbarrier_count);
    Tau_userevent_thread(ue, value, taskid);
  }

#ifdef DEBUG_PROF
  e.printEvent();
  cout <<endl;
#endif /* DEBUG_PROF */
  Tau_process_rocm_events(e);
  timestamp = record->begin;

  last_timestamp = timestamp;
  timestamp = record->end;
  last_timestamp = timestamp;
  Tau_set_last_timestamp_ns(record->complete);


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
    fatal("context is NULL\n");
  }
  if (feature_count > 0) {
    hsa_status_t status = rocprofiler_group_get_data(&group);
    Tau_rocm_check_status(status);
    status = rocprofiler_get_metrics(group.context);
    Tau_rocm_check_status(status);
  }

#ifdef DEBUG_PROF
  for (unsigned i = 0; i < feature_count; ++i) {
    const rocprofiler_feature_t* p = &features[i];
    fprintf(stdout, ">  %s ", p->name);
    switch (p->data.kind) {
      // Output metrics results
      case ROCPROFILER_DATA_KIND_INT64:
        fprintf(stdout, "= (%lu)\n", p->data.result_int64);
        break;
      default:
        fprintf(stderr, "Undefined data kind(%u)\n", p->data.kind);
        abort();
    }
  }
#endif /* DEBUG_PROF */
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool Tau_rocm_context_handler(const rocprofiler_pool_entry_t* entry, void* arg) {
  // Context entry
  TAU_VERBOSE("Inside Tau_rocm_context_handler\n");

  context_entry_t* ctx_entry = reinterpret_cast<context_entry_t*>(entry->payload);
  handler_arg_t* handler_arg = reinterpret_cast<handler_arg_t*>(arg);

  if (pthread_mutex_lock(&rocm_mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (is_callback_loaded)  
	  Tau_rocm_dump_context_entry(ctx_entry, handler_arg->features, handler_arg->feature_count);

  if (pthread_mutex_unlock(&rocm_mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}
#if 0
// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool Tau_rocm_context_handler1(rocprofiler_group_t group, void* arg) {
  context_entry_t* ctx_entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  Tau_rocm_dump_context_entry(ctx_entry, features, feature_count);

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}
#endif

static const amd_kernel_code_t* GetKernelCode(uint64_t kernel_object) {
  const amd_kernel_code_t* kernel_code = NULL;
  hsa_status_t status =
      HsaRsrcFactory::Instance().LoaderApi()->hsa_ven_amd_loader_query_host_address(
          reinterpret_cast<const void*>(kernel_object),
          reinterpret_cast<const void**>(&kernel_code));
  if (HSA_STATUS_SUCCESS != status) {
    kernel_code = reinterpret_cast<amd_kernel_code_t*>(kernel_object);
  }
  return kernel_code;
}

// Setting kernel properties
void set_kernel_properties(const rocprofiler_callback_data_t* callback_data,
                           context_entry_t* entry)
{
  const hsa_kernel_dispatch_packet_t* packet = callback_data->packet;
  kernel_properties_t* kernel_properties_ptr = &(entry->kernel_properties);
  const amd_kernel_code_t* kernel_code = callback_data->kernel_code;

  entry->data = *callback_data;

  if (kernel_code == NULL) {
    const uint64_t kernel_object = callback_data->packet->kernel_object;
    kernel_code = GetKernelCode(kernel_object);
    //entry->kernel_name_it = HsaRsrcFactory::AcquireKernelNameRef(kernel_object);
  } else {
    entry->data.kernel_name = strdup(callback_data->kernel_name);
  }

  uint64_t grid_size = packet->grid_size_x * packet->grid_size_y * packet->grid_size_z;
  if (grid_size > UINT32_MAX) abort();
  kernel_properties_ptr->grid_size = (uint32_t)grid_size;
  uint64_t workgroup_size = packet->workgroup_size_x * packet->workgroup_size_y * packet->workgroup_size_z;
  if (workgroup_size > UINT32_MAX) abort();
  kernel_properties_ptr->workgroup_size = (uint32_t)workgroup_size;
  kernel_properties_ptr->lds_size = packet->group_segment_size;
  kernel_properties_ptr->scratch_size = packet->private_segment_size;
  kernel_properties_ptr->vgpr_count = AMD_HSA_BITS_GET(kernel_code->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT);
  kernel_properties_ptr->sgpr_count = AMD_HSA_BITS_GET(kernel_code->compute_pgm_rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT);
  kernel_properties_ptr->fbarrier_count = kernel_code->workgroup_fbarrier_count;
  kernel_properties_ptr->signal = callback_data->completion_signal;
  kernel_properties_ptr->object = callback_data->packet->kernel_object;
}

// Kernel disoatch callback
hsa_status_t Tau_rocm_dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* arg,
                               rocprofiler_group_t* group) {
  // Passed tool data
  TAU_VERBOSE("Inside Tau_rocm_dispatch_callback\n");

  hsa_agent_t agent = callback_data->agent;
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

#if 1
  // Open profiling context
  const unsigned gpu_id = HsaRsrcFactory::Instance().GetAgentInfo(agent)->dev_index;
  callbacks_arg_t* callbacks_arg = reinterpret_cast<callbacks_arg_t*>(arg);
  rocprofiler_pool_t* pool = callbacks_arg->pools[gpu_id];
  rocprofiler_pool_entry_t pool_entry{};
  status = rocprofiler_pool_fetch(pool, &pool_entry);
  Tau_rocm_check_status(status);
  // Profiling context entry
  rocprofiler_t* context = pool_entry.context;
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(pool_entry.payload);
  // Setting kernel properties
  set_kernel_properties(callback_data, entry);
#else
  // Open profiling context
  // context properties
  context_entry_t* entry = new context_entry_t{};
  rocprofiler_t* context = NULL;
  rocprofiler_properties_t properties{};
  properties.handler = Tau_rocm_context_handler1;
  properties.handler_arg = (void*)entry;
  status = rocprofiler_open(agent, features, feature_count,
                            &context, 0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  Tau_rocm_check_status(status);
#endif
  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  Tau_rocm_check_status(status);

  // Fill profiling context entry
  entry->agent = agent;
  entry->group = *group;
  //entry->data = *callback_data;
  //entry->data.kernel_name = strdup(callback_data->kernel_name);
  reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

  return HSA_STATUS_SUCCESS;
}

unsigned metrics_input(rocprofiler_feature_t** ret) {
  // Profiling feature objects
  const unsigned feature_count = 6;
  rocprofiler_feature_t* features = new rocprofiler_feature_t[feature_count];
  memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));

  // PMC events
  features[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[0].name = "GRBM_COUNT";
  features[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[1].name = "GRBM_GUI_ACTIVE";
  features[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[2].name = "GPUBusy";
  features[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[3].name = "SQ_WAVES";
  features[4].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[4].name = "SQ_INSTS_VALU";
  features[5].kind = ROCPROFILER_FEATURE_KIND_METRIC;
  features[5].name = "VALUInsts";
//  features[6].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  features[6].name = "TCC_HIT_sum";
//  features[7].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  features[7].name = "TCC_MISS_sum";
//  features[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
//  features[8].name = "WRITE_SIZE";

  *ret = features;
  return feature_count;
}

void Tau_rocm_initialize() {
  TAU_VERBOSE("Inside Tau_rocm_initialize\n");

  // Available GPU agents
  const unsigned gpu_count = HsaRsrcFactory::Instance().GetCountOfGpuAgents();

  // Getting profiling features
  rocprofiler_feature_t* features = NULL;
  // TAU doesn't support features or metrics yet! SSS
  //unsigned feature_count = metrics_input(&features);
  unsigned feature_count = 0;

  // Handler arg
  handler_arg_t* handler_arg = new handler_arg_t{};
  handler_arg->features = features;
  handler_arg->feature_count = feature_count;

  // Context properties
  rocprofiler_pool_properties_t properties{};
  properties.num_entries = 100;
  properties.payload_bytes = sizeof(context_entry_t);
  properties.handler = Tau_rocm_context_handler;
  properties.handler_arg = handler_arg;

  // Adding dispatch observer
  callbacks_arg_t* callbacks_arg = new callbacks_arg_t{};
  callbacks_arg->pools = new rocprofiler_pool_t* [gpu_count];
  for (unsigned gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
    // Getting GPU device info
    const AgentInfo* agent_info = NULL;
    if (HsaRsrcFactory::Instance().GetGpuAgentInfo(gpu_id, &agent_info) == false) {
      fprintf(stderr, "GetGpuAgentInfo failed\n");
      abort();
    }

    // Open profiling pool
    rocprofiler_pool_t* pool = NULL;
    hsa_status_t status = rocprofiler_pool_open(agent_info->dev_id, features, feature_count,
                                                &pool, 0/*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
    Tau_rocm_check_status(status);
    callbacks_arg->pools[gpu_id] = pool;
    get_pool_set().insert(pool);
  }

  rocprofiler_queue_callbacks_t callbacks_ptrs{};
  callbacks_ptrs.dispatch = Tau_rocm_dispatch_callback;
  int err=rocprofiler_set_queue_callbacks(callbacks_ptrs, callbacks_arg);
  TAU_VERBOSE("err=%d, rocprofiler_set_queue_callbacks\n", err);

}

void Tau_rocm_cleanup() {
  // Unregister dispatch callback
  if (pthread_mutex_lock(&rocm_mutex) != 0) {
      perror("pthread_mutex_lock");
      abort();
  }
  if (is_callback_loaded){
	  is_callback_loaded = false;
	  rocprofiler_remove_queue_callbacks();
  }
  if (pthread_mutex_unlock(&rocm_mutex) != 0) {
      perror("pthread_mutex_unlock");
      abort();
    }

  // CLose profiling pool
#if 0
  hsa_status_t status = rocprofiler_pool_flush(pool);
  Tau_rocm_check_status(status);
  status = rocprofiler_pool_close(pool);
  Tau_rocm_check_status(status);
#endif
}

void Tau_rocprofiler_pool_flush() {
    Tau_rocm_cleanup();
    if (pthread_mutex_lock(&rocm_mutex) != 0) {
      perror("pthread_mutex_lock");
      abort();
    }

    for (auto p : get_pool_set()) {
        TAU_VERBOSE("Flushing pool %p\n");
        hsa_status_t status = rocprofiler_pool_flush(p);
        Tau_rocm_check_status(status);
    }

    if (pthread_mutex_unlock(&rocm_mutex) != 0) {
      perror("pthread_mutex_unlock");
      abort();
    }
}

// Tool constructor
extern "C" PUBLIC_API void OnLoadToolProp(rocprofiler_settings_t* settings)
{
  TAU_VERBOSE("Inside OnLoadToolProp\n");
  Tau_init_initializeTAU();

#if (!(defined (TAU_MPI) || (TAU_SHMEM)))
  if (Tau_get_node() == -1) {
      TAU_PROFILE_SET_NODE(0);
  }
#endif /* TAU_MPI || TAU_SHMEM */

  if (pthread_mutex_lock(&rocm_mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (is_loaded) return;
  is_loaded = true;
  is_callback_loaded = true;
  if (pthread_mutex_unlock(&rocm_mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  // Enable timestamping
  settings->timestamp_on = true;

  // Initialize profiling
  Tau_rocm_initialize();
  HsaRsrcFactory::Instance().PrintGpuAgents("ROCm");

}

// Tool destructor
extern "C" PUBLIC_API void OnUnloadTool() {
  TAU_VERBOSE("Inside OnUnloadTool\n");

  if (pthread_mutex_lock(&rocm_mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (!is_loaded) return;
  is_loaded = false;
  if (pthread_mutex_unlock(&rocm_mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  Tau_stop_top_level_timer_if_necessary();
  // Final resources cleanup
  Tau_rocm_cleanup();
}

extern "C" CONSTRUCTOR_API void constructor() {
  TAU_VERBOSE("INTT constructor\n"); fflush(stdout);
}

extern "C" DESTRUCTOR_API void destructor() {
  if (is_loaded == true) OnUnloadTool();
}


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

  err = HsaRsrcFactory::HsaApi()->hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  CHECK_STATUS("hsa_amd_memory_pool_get_info", err);
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  err = HsaRsrcFactory::HsaApi()->hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
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

  InitHsaApiTable(NULL);

  // Initialize the Hsa Runtime
  if (initialize_hsa_) {
    status = hsa_api_.hsa_init();
    CHECK_STATUS("Error in hsa_init", status);
  }

  // Discover the set of Gpu devices available on the platform
  status = hsa_api_.hsa_iterate_agents(GetHsaAgentsCallback, this);
  CHECK_STATUS("Error Calling hsa_iterate_agents", status);
  if (cpu_pool_ == NULL) CHECK_STATUS("CPU memory pool is not found", HSA_STATUS_ERROR);
  if (kern_arg_pool_ == NULL) CHECK_STATUS("Kern-arg memory pool is not found", HSA_STATUS_ERROR);

  // Get AqlProfile API table
  aqlprofile_api_ = {0};
#ifdef ROCP_LD_AQLPROFILE
  status = LoadAqlProfileLib(&aqlprofile_api_);
#else
  status = hsa_api_.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_AQLPROFILE, hsa_ven_amd_aqlprofile_VERSION_MAJOR, sizeof(aqlprofile_api_), &aqlprofile_api_);
#endif
  CHECK_STATUS("aqlprofile API table load failed", status);

  // Get Loader API table
  loader_api_ = {0};
  status = hsa_api_.hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader_api_), &loader_api_);
  CHECK_STATUS("loader API table query failed", status);

  // Instantiate HSA timer
  timer_ = new HsaTimer(&hsa_api_);
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
    hsa_status_t status = hsa_api_.hsa_shut_down();
    CHECK_STATUS("Error in hsa_shut_down", status);
  }
}

void HsaRsrcFactory::InitHsaApiTable(HsaApiTable* table) {
  std::lock_guard<mutex_t> lck(mutex_);

  if (hsa_api_.hsa_init == NULL) {
    if (table != NULL) {
      hsa_api_.hsa_init = table->core_->hsa_init_fn;
      hsa_api_.hsa_shut_down = table->core_->hsa_shut_down_fn;
      hsa_api_.hsa_agent_get_info = table->core_->hsa_agent_get_info_fn;
      hsa_api_.hsa_iterate_agents = table->core_->hsa_iterate_agents_fn;

      hsa_api_.hsa_queue_create = table->core_->hsa_queue_create_fn;
      hsa_api_.hsa_queue_destroy = table->core_->hsa_queue_destroy_fn;
      hsa_api_.hsa_queue_load_write_index_relaxed = table->core_->hsa_queue_load_write_index_relaxed_fn;
      hsa_api_.hsa_queue_store_write_index_relaxed = table->core_->hsa_queue_store_write_index_relaxed_fn;
      hsa_api_.hsa_queue_load_read_index_relaxed = table->core_->hsa_queue_load_read_index_relaxed_fn;

      hsa_api_.hsa_signal_create = table->core_->hsa_signal_create_fn;
      hsa_api_.hsa_signal_destroy = table->core_->hsa_signal_destroy_fn;
      hsa_api_.hsa_signal_load_relaxed = table->core_->hsa_signal_load_relaxed_fn;
      hsa_api_.hsa_signal_store_relaxed = table->core_->hsa_signal_store_relaxed_fn;
      hsa_api_.hsa_signal_wait_scacquire = table->core_->hsa_signal_wait_scacquire_fn;
      hsa_api_.hsa_signal_store_screlease = table->core_->hsa_signal_store_screlease_fn;

      hsa_api_.hsa_code_object_reader_create_from_file = table->core_->hsa_code_object_reader_create_from_file_fn;
      hsa_api_.hsa_executable_create_alt = table->core_->hsa_executable_create_alt_fn;
      hsa_api_.hsa_executable_load_agent_code_object = table->core_->hsa_executable_load_agent_code_object_fn;
      hsa_api_.hsa_executable_freeze = table->core_->hsa_executable_freeze_fn;
      hsa_api_.hsa_executable_get_symbol = table->core_->hsa_executable_get_symbol_fn;
      hsa_api_.hsa_executable_symbol_get_info = table->core_->hsa_executable_symbol_get_info_fn;
      hsa_api_.hsa_executable_iterate_symbols = table->core_->hsa_executable_iterate_symbols_fn;

      hsa_api_.hsa_system_get_info = table->core_->hsa_system_get_info_fn;
      hsa_api_.hsa_system_get_major_extension_table = table->core_->hsa_system_get_major_extension_table_fn;

      hsa_api_.hsa_amd_agent_iterate_memory_pools = table->amd_ext_->hsa_amd_agent_iterate_memory_pools_fn;
      hsa_api_.hsa_amd_memory_pool_get_info = table->amd_ext_->hsa_amd_memory_pool_get_info_fn;
      hsa_api_.hsa_amd_memory_pool_allocate = table->amd_ext_->hsa_amd_memory_pool_allocate_fn;
      hsa_api_.hsa_amd_agents_allow_access = table->amd_ext_->hsa_amd_agents_allow_access_fn;
      hsa_api_.hsa_amd_memory_async_copy = table->amd_ext_->hsa_amd_memory_async_copy_fn;

      hsa_api_.hsa_amd_signal_async_handler = table->amd_ext_->hsa_amd_signal_async_handler_fn;
      hsa_api_.hsa_amd_profiling_set_profiler_enabled = table->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn;
      hsa_api_.hsa_amd_profiling_get_async_copy_time = table->amd_ext_->hsa_amd_profiling_get_async_copy_time_fn;
      hsa_api_.hsa_amd_profiling_get_dispatch_time = table->amd_ext_->hsa_amd_profiling_get_dispatch_time_fn;
    } else {
      hsa_api_.hsa_init = hsa_init;
      hsa_api_.hsa_shut_down = hsa_shut_down;
      hsa_api_.hsa_agent_get_info = hsa_agent_get_info;
      hsa_api_.hsa_iterate_agents = hsa_iterate_agents;

      hsa_api_.hsa_queue_create = hsa_queue_create;
      hsa_api_.hsa_queue_destroy = hsa_queue_destroy;
      hsa_api_.hsa_queue_load_write_index_relaxed = hsa_queue_load_write_index_relaxed;
      hsa_api_.hsa_queue_store_write_index_relaxed = hsa_queue_store_write_index_relaxed;
      hsa_api_.hsa_queue_load_read_index_relaxed = hsa_queue_load_read_index_relaxed;

      hsa_api_.hsa_signal_create = hsa_signal_create;
      hsa_api_.hsa_signal_destroy = hsa_signal_destroy;
      hsa_api_.hsa_signal_load_relaxed = hsa_signal_load_relaxed;
      hsa_api_.hsa_signal_store_relaxed = hsa_signal_store_relaxed;
      hsa_api_.hsa_signal_wait_scacquire = hsa_signal_wait_scacquire;
      hsa_api_.hsa_signal_store_screlease = hsa_signal_store_screlease;

      hsa_api_.hsa_code_object_reader_create_from_file = hsa_code_object_reader_create_from_file;
      hsa_api_.hsa_executable_create_alt = hsa_executable_create_alt;
      hsa_api_.hsa_executable_load_agent_code_object = hsa_executable_load_agent_code_object;
      hsa_api_.hsa_executable_freeze = hsa_executable_freeze;
      hsa_api_.hsa_executable_get_symbol = hsa_executable_get_symbol;
      hsa_api_.hsa_executable_symbol_get_info = hsa_executable_symbol_get_info;
      hsa_api_.hsa_executable_iterate_symbols = hsa_executable_iterate_symbols;

      hsa_api_.hsa_system_get_info = hsa_system_get_info;
      hsa_api_.hsa_system_get_major_extension_table = hsa_system_get_major_extension_table;

      hsa_api_.hsa_amd_agent_iterate_memory_pools = hsa_amd_agent_iterate_memory_pools;
      hsa_api_.hsa_amd_memory_pool_get_info = hsa_amd_memory_pool_get_info;
      hsa_api_.hsa_amd_memory_pool_allocate = hsa_amd_memory_pool_allocate;
      hsa_api_.hsa_amd_agents_allow_access = hsa_amd_agents_allow_access;
      hsa_api_.hsa_amd_memory_async_copy = hsa_amd_memory_async_copy;

      hsa_api_.hsa_amd_signal_async_handler = hsa_amd_signal_async_handler;
      hsa_api_.hsa_amd_profiling_set_profiler_enabled = hsa_amd_profiling_set_profiler_enabled;
      hsa_api_.hsa_amd_profiling_get_async_copy_time = hsa_amd_profiling_get_async_copy_time;
      hsa_api_.hsa_amd_profiling_get_dispatch_time = hsa_amd_profiling_get_dispatch_time;
    }
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
  status = hsa_api_.hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  CHECK_STATUS("Error Calling hsa_agent_get_info", status);

  if (type == HSA_DEVICE_TYPE_CPU) {
    agent_info = new AgentInfo{};
    agent_info->dev_id = agent;
    agent_info->dev_type = HSA_DEVICE_TYPE_CPU;
    agent_info->dev_index = cpu_list_.size();

    status = hsa_api_.hsa_amd_agent_iterate_memory_pools(agent, FindStandardPool, &agent_info->cpu_pool);
    if ((status == HSA_STATUS_INFO_BREAK) && (cpu_pool_ == NULL)) cpu_pool_ = &agent_info->cpu_pool;
    status = hsa_api_.hsa_amd_agent_iterate_memory_pools(agent, FindKernArgPool, &agent_info->kern_arg_pool);
    if ((status == HSA_STATUS_INFO_BREAK) && (kern_arg_pool_ == NULL)) kern_arg_pool_ = &agent_info->kern_arg_pool;
    agent_info->gpu_pool = {};

    cpu_list_.push_back(agent_info);
    cpu_agents_.push_back(agent);
  }

  if (type == HSA_DEVICE_TYPE_GPU) {
    agent_info = new AgentInfo{};
    agent_info->dev_id = agent;
    agent_info->dev_type = HSA_DEVICE_TYPE_GPU;
    hsa_api_.hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_info->name);
    strncpy(agent_info->gfxip, agent_info->name, 4);
    agent_info->gfxip[4] = '\0';
    hsa_api_.hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &agent_info->max_wave_size);
    hsa_api_.hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &agent_info->max_queue_size);
    hsa_api_.hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_info->profile);
    agent_info->is_apu = (agent_info->profile == HSA_PROFILE_FULL) ? true : false;
    hsa_api_.hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                       &agent_info->cu_num);
    hsa_api_.hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
                       &agent_info->waves_per_cu);
    hsa_api_.hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU),
                       &agent_info->simds_per_cu);
    hsa_api_.hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES),
                       &agent_info->se_num);
    hsa_api_.hsa_agent_get_info(agent,
                       static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE),
                       &agent_info->shader_arrays_per_se);

    agent_info->cpu_pool = {};
    agent_info->kern_arg_pool = {};
    status = hsa_api_.hsa_amd_agent_iterate_memory_pools(agent, FindStandardPool, &agent_info->gpu_pool);
    CHECK_ITER_STATUS("hsa_amd_agent_iterate_memory_pools(gpu pool)", status);

    // GFX8 and GFX9 SGPR/VGPR block sizes
    agent_info->sgpr_block_dflt = (strcmp(agent_info->gfxip, "gfx8") == 0) ? 1 : 2;
    agent_info->sgpr_block_size = 8;
    agent_info->vgpr_block_size = 4;

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
  status = hsa_api_.hsa_queue_create(agent_info->dev_id, num_pkts, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                            UINT32_MAX, UINT32_MAX, queue);
  return (status == HSA_STATUS_SUCCESS);
}

// Create a Signal object and return its handle.
// @param value Initial value of signal object
// @param signal Output parameter updated with handle of signal object
// @return bool true if successful, false otherwise
bool HsaRsrcFactory::CreateSignal(uint32_t value, hsa_signal_t* signal) {
  hsa_status_t status;
  status = hsa_api_.hsa_signal_create(value, 0, NULL, signal);
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
  status = hsa_api_.hsa_amd_memory_pool_allocate(agent_info->gpu_pool, size, 0, reinterpret_cast<void**>(&buffer));
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
    status = hsa_api_.hsa_amd_memory_pool_allocate(*kern_arg_pool_, size, 0, reinterpret_cast<void**>(&buffer));
    // Both the CPU and GPU can access the kernel arguments
    if (status == HSA_STATUS_SUCCESS) {
      hsa_agent_t ag_list[1] = {agent_info->dev_id};
      status = hsa_api_.hsa_amd_agents_allow_access(1, ag_list, NULL, buffer);
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
    status = hsa_api_.hsa_amd_memory_pool_allocate(*cpu_pool_, size, 0, reinterpret_cast<void**>(&buffer));
    // Both the CPU and GPU can access the memory
    if (status == HSA_STATUS_SUCCESS) {
      hsa_agent_t ag_list[1] = {agent_info->dev_id};
      status = hsa_api_.hsa_amd_agents_allow_access(1, ag_list, NULL, buffer);
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
      hsa_api_.hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, timeout_, HSA_WAIT_STATE_BLOCKED);
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
  hsa_api_.hsa_signal_store_relaxed(const_cast<hsa_signal_t&>(signal), signal_value);
}

// Copy data from GPU to host memory
bool HsaRsrcFactory::Memcpy(const hsa_agent_t& agent, void* dst, const void* src, size_t size) {
  hsa_status_t status = HSA_STATUS_ERROR;
  if (!cpu_agents_.empty()) {
    hsa_signal_t s = {};
    status = hsa_api_.hsa_signal_create(1, 0, NULL, &s);
    CHECK_STATUS("hsa_signal_create()", status);
    status = hsa_api_.hsa_amd_memory_async_copy(dst, cpu_agents_[0], src, agent, size, 0, NULL, s);
    CHECK_STATUS("hsa_amd_memory_async_copy()", status);
    SignalWait(s);
    status = hsa_api_.hsa_signal_destroy(s);
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
  status = hsa_api_.hsa_code_object_reader_create_from_file(file_handle, &code_obj_rdr);
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "Failed to create code object reader '" << filename << "'" << std::endl;
    return false;
  }

  // Create executable.
  status = hsa_api_.hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     NULL, executable);
  CHECK_STATUS("Error in creating executable object", status);

  // Load code object.
  status = hsa_api_.hsa_executable_load_agent_code_object(*executable, agent_info->dev_id, code_obj_rdr,
                                                 NULL, NULL);
  CHECK_STATUS("Error in loading executable object", status);

  // Freeze executable.
  status = hsa_api_.hsa_executable_freeze(*executable, "");
  CHECK_STATUS("Error in freezing executable object", status);

  // Get symbol handle.
  hsa_executable_symbol_t kernelSymbol;
  status = hsa_api_.hsa_executable_get_symbol(*executable, NULL, kernel_name, agent_info->dev_id, 0,
                                     &kernelSymbol);
  CHECK_STATUS("Error in looking up kernel symbol", status);

  // Update output parameter
  *code_desc = kernelSymbol;
  return true;
}

// Print the various fields of Hsa Gpu Agents
bool HsaRsrcFactory::PrintGpuAgents(const std::string& header) {
  std::cout << std::flush;
  //std::clog << header << " :" << std::endl;

  char key[1024], value[1024];

  const AgentInfo* agent_info;
  int size = uint32_t(gpu_list_.size());
  for (int idx = 0; idx < size; idx++) {
    agent_info = gpu_list_[idx];

/*  std::clog << "> agent[" << idx << "] :" << std::endl;
    std::clog << ">> Name : " << agent_info->name << std::endl;
    std::clog << ">> APU : " << agent_info->is_apu << std::endl;
    std::clog << ">> HSAIL profile : " << agent_info->profile << std::endl;
    std::clog << ">> Max Wave Size : " << agent_info->max_wave_size << std::endl;
    std::clog << ">> Max Queue Size : " << agent_info->max_queue_size << std::endl;
    std::clog << ">> CU number : " << agent_info->cu_num << std::endl;
    std::clog << ">> Waves per CU : " << agent_info->waves_per_cu << std::endl;
    std::clog << ">> SIMDs per CU : " << agent_info->simds_per_cu << std::endl;
    std::clog << ">> SE number : " << agent_info->se_num << std::endl;
    std::clog << ">> Shader Arrays per SE : " << agent_info->shader_arrays_per_se << std::endl;
*/

    sprintf(key, "ROCM_AGENT_%d_NAME", idx);
    sprintf(value, "%s", agent_info->name);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_IS_APU", idx);
    sprintf(value, "%d", agent_info->is_apu);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_HSA_PROFILE", idx);
    sprintf(value, "%d", agent_info->profile);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_MAX_WAVE_SIZE", idx);
    sprintf(value, "%d", agent_info->max_wave_size);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_MAX_QUEUE_SIZE", idx);
    sprintf(value, "%d", agent_info->max_queue_size);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_CU_NUMBER", idx);
    sprintf(value, "%d", agent_info->cu_num);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_WAVES_PER_CU", idx);
    sprintf(value, "%d", agent_info->waves_per_cu);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_SIMDs_PER_CU", idx);
    sprintf(value, "%d", agent_info->simds_per_cu);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_SE_NUMBER", idx);
    sprintf(value, "%d", agent_info->se_num);
    TAU_METADATA(key, value);

    sprintf(key, "ROCM_AGENT_%d_SHADER_ARRAYS_PER_SE", idx);
    sprintf(value, "%d", agent_info->shader_arrays_per_se);
    TAU_METADATA(key, value);

  }
  return true;
}

uint64_t HsaRsrcFactory::Submit(hsa_queue_t* queue, const void* packet) {
  const uint32_t slot_size_b = CMD_SLOT_SIZE_B;

  // adevance command queue
  const uint64_t write_idx = hsa_api_.hsa_queue_load_write_index_relaxed(queue);
  hsa_api_.hsa_queue_store_write_index_relaxed(queue, write_idx + 1);
  while ((write_idx - hsa_api_.hsa_queue_load_read_index_relaxed(queue)) >= queue->size) {
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
  hsa_api_.hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  return write_idx;
}

uint64_t HsaRsrcFactory::Submit(hsa_queue_t* queue, const void* packet, size_t size_bytes) {
  const uint32_t slot_size_b = CMD_SLOT_SIZE_B;
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

const char* HsaRsrcFactory::GetKernelName(uint64_t addr) {
  std::lock_guard<mutex_t> lck(mutex_);
  const auto it = symbols_map_->find(addr);
  if (it == symbols_map_->end()) {
    fprintf(stderr, "HsaRsrcFactory::kernel addr (0x%lx) is not found\n", addr);
    abort();
  }
  return strdup(it->second);
}

void HsaRsrcFactory::EnableExecutableTracking(HsaApiTable* table) {
  std::lock_guard<mutex_t> lck(mutex_);
  executable_tracking_on_ = true;
  table->core_->hsa_executable_freeze_fn = hsa_executable_freeze_interceptor;
}

hsa_status_t HsaRsrcFactory::executable_symbols_cb(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data) {
  hsa_symbol_kind_t value = (hsa_symbol_kind_t)0;
  hsa_status_t status = hsa_api_.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &value);
  CHECK_STATUS("Error in getting symbol info", status);
  if (value == HSA_SYMBOL_KIND_KERNEL) {
    uint64_t addr = 0;
    uint32_t len = 0;
    status = hsa_api_.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &addr);
    CHECK_STATUS("Error in getting kernel object", status);
    status = hsa_api_.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len);
    CHECK_STATUS("Error in getting name len", status);
    char *name = new char[len + 1];
    status = hsa_api_.hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
    CHECK_STATUS("Error in getting kernel name", status);
    name[len] = 0;
    auto ret = symbols_map_->insert({addr, name});
    if (ret.second == false) {
      delete[] ret.first->second;
      ret.first->second = name;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HsaRsrcFactory::hsa_executable_freeze_interceptor(hsa_executable_t executable, const char *options) {
  std::lock_guard<mutex_t> lck(mutex_);
  if (symbols_map_ == NULL) symbols_map_ = new symbols_map_t;
  hsa_status_t status = hsa_api_.hsa_executable_iterate_symbols(executable, executable_symbols_cb, NULL);
  CHECK_STATUS("Error in iterating executable symbols", status);
  return hsa_api_.hsa_executable_freeze(executable, options);;
}

std::atomic<HsaRsrcFactory*> HsaRsrcFactory::instance_{};
HsaRsrcFactory::mutex_t HsaRsrcFactory::mutex_;
HsaRsrcFactory::timestamp_t HsaRsrcFactory::timeout_ns_ = HsaTimer::TIMESTAMP_MAX;
hsa_pfn_t HsaRsrcFactory::hsa_api_{};
bool HsaRsrcFactory::executable_tracking_on_ = false;
HsaRsrcFactory::symbols_map_t* HsaRsrcFactory::symbols_map_ = NULL;
