/*
 * Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * */


#include <Profile/TauRocm.h>
#include <hip/hip_runtime.h>
#include <roctracer.h>
#include <stdlib.h>

#ifdef TAU_ENABLE_ROCTRACER_HSA
#ifndef TAU_HSA_TASK_ID
#define TAU_HSA_TASK_ID 501
#endif /* TAU_HSA_TASK_ID */
#include <roctracer_hsa.h>
#include <roctracer_hip.h>
#include <roctracer_hcc.h>
#include <ext/hsa_rt_utils.hpp>
#endif /* TAU_ENABLE_ROCTRACER_HSA */
#include <sys/syscall.h> 

#ifdef TAU_BFD
#define HAVE_DECL_BASENAME 1
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include <demangle.h>
#  endif /* HAVE_GNU_DEMANGLE */
// Add these definitions because the Binutils comedians think all the world uses autotools
#ifndef PACKAGE
#define PACKAGE TAU
#endif
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION 2.25
#endif
#  include <bfd.h>
#endif /* TAU_BFD */
#define TAU_INTERNAL_DEMANGLE_NAME(name, dem_name)  dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES); \
        if (dem_name == NULL) { \
          dem_name = name; \
        } \


#ifndef TAU_ROCTRACER_BUFFER_SIZE
#define TAU_ROCTRACER_BUFFER_SIZE 65536
#endif /* TAU_ROCTRACER_BUFFER_SIZE */

#ifndef TAU_ROCTRACER_HOST_TASKID 
#define TAU_ROCTRACER_HOST_TASKID 500
#endif /* TAU_ROCTRACER_HOST_TASKID */


// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)


std::string TauRocTracerNameDB[TAU_ROCTRACER_BUFFER_SIZE]; 

// Launch a kernel
void Tau_roctracer_register_activity(int id, const char *name) {
  TAU_VERBOSE("Inside Tau_roctracer_register_activity: id = %d, name = %s\n",
	id, name);
  TauRocTracerNameDB[id] = std::string(name); 
  TAU_VERBOSE("Tau_roctracer_register_activity: id = %d, name = %s\n",
	id, TauRocTracerNameDB[id].c_str());
  return; 
}

// Runtime API callback function
void Tau_roctracer_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  int task_id;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  const char *activity_name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
  TAU_VERBOSE("<%s id(%u)\tcorrelation_id(%lu) %s> ",
    activity_name,
    cid,
    data->correlation_id,
    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        TAU_VERBOSE("dst(%p) src(%p) size(0x%x) kind(%u)\n",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
	Tau_roctracer_register_activity(data->correlation_id, activity_name);
        break;
      case HIP_API_ID_hipMalloc:
        TAU_VERBOSE("ptr(%p) size(0x%x)\n",
          data->args.hipMalloc.ptr,
          (uint32_t)(data->args.hipMalloc.size));
	Tau_roctracer_register_activity(data->correlation_id, activity_name);
        break;
      case HIP_API_ID_hipFree:
        TAU_VERBOSE("ptr(%p)\n",
          data->args.hipFree.ptr);
	Tau_roctracer_register_activity(data->correlation_id, activity_name);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        TAU_VERBOSE("kernel(\"%s\") stream(%p)\n",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
	Tau_roctracer_register_activity(data->correlation_id, 
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f)); 
        break;
      default:
        break;
    }
  } else {
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        TAU_VERBOSE("*ptr(0x%p)",
          *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
  }
  TAU_VERBOSE("\n"); fflush(stdout);
}

// host based events 
void Tau_roctracer_hip_event(const roctracer_record_t *record, int task_id) {
  const char *event_name = roctracer_op_string(record->domain, record->op, record->kind);
  const char *name = event_name;
  bool dealloc_name = false;
  if (strcmp(name, "hipModuleLaunchKernel") == 0) {
    dealloc_name = true;
    const char * kernel_name = TauRocTracerNameDB[record->correlation_id].c_str();
    const char *demangled_name;
    TAU_INTERNAL_DEMANGLE_NAME(kernel_name, demangled_name);
    name = strdup((event_name + string(" ") + demangled_name).c_str());
    TAU_VERBOSE("Tau_roctracer_hip_event: name = %s\n", name);
  }
 
  TAU_VERBOSE("Tau_roctracer_hip_event: name=%s, cid=%lu, time_ns(%lu:%lu), task_id=%d, pid=%u, thread_id=%u\n",
    name, record->correlation_id, record->begin_ns, record->end_ns, 
    task_id, record->process_id, record->thread_id);
  Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(record->begin_ns)/1e3)); // convert to microseconds
  TAU_START_TASK(name, task_id);

  // stop
  Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(record->end_ns)/1e3)); // convert to microseconds
  TAU_STOP_TASK(name, task_id);
  TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", name, task_id, record->end_ns);
  if (dealloc_name) {
    free((void*)name);
  }
  Tau_set_last_timestamp_ns(record->end_ns);
}

// gpu based events 
void Tau_roctracer_hcc_event(const roctracer_record_t *record, int task_id) {
  const char * name = TauRocTracerNameDB[record->correlation_id].c_str();
  //const char * name = string(string(roctracer_op_string(record->domain, record->op, record->kind)) + " : " + TauRocTracerNameDB[record->correlation_id]).c_str();
  TAU_VERBOSE("Tau_roctracer_hcc_event: name=%s, cid=%lu, time_ns(%lu:%lu), device=%d, queue_id=%lu, task_id=%d\n",
    name, record->correlation_id, record->begin_ns, record->end_ns, 
    record->device_id, record->queue_id, task_id);
  Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(record->begin_ns)/1e3)); // convert to microseconds
  int status;
  const char *demangled_name;
  TAU_INTERNAL_DEMANGLE_NAME(name, demangled_name);
  char *joined_name ; 
  if ((record -> kind == 2) || (record->kind == 1) ) { //hipMemcpy 
    joined_name = (char *) roctracer_op_string(record->domain, record->op, record->kind);
  } else {
    joined_name = (char *) string(string(roctracer_op_string(record->domain, record->op, record->kind)) +" "+ demangled_name).c_str(); 
  }
  TAU_START_TASK(joined_name, task_id);
  TAU_VERBOSE("Started event %s on task %d timestamp = %lu \n", demangled_name, task_id, record->begin_ns);

  // and the end
  Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(record->end_ns)/1e3)); // convert to microseconds
  TAU_STOP_TASK(joined_name, task_id);
  TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", demangled_name, task_id, record->end_ns);
  Tau_set_last_timestamp_ns(record->end_ns);
   
  return;
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void Tau_roctracer_activity_callback(const char* begin, const char* end, void* arg) {
  int task_id=-1;
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
  TAU_VERBOSE("\tActivity records :\n"); fflush(stdout);
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    TAU_VERBOSE("\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
      name,
      record->correlation_id,
      record->begin_ns,
      record->end_ns
    );
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      int my_pid = getpid(); 
      TAU_VERBOSE(" ACTIVITY_DOMAIN_HIP_API: my_pid=%d\n", my_pid);
      //if ((record->process_id == my_pid) && (record->thread_id == my_pid)) {
        // We need to record events on this host thread. Check if it is created already. 
        int mytid = syscall(__NR_gettid);
        task_id = Tau_get_initialized_queues(mytid);
        if (task_id == -1) {
          TAU_VERBOSE(" ACTIVITY_DOMAIN_HIP_API: creating task\n");
          TAU_CREATE_TASK(task_id); 
          Tau_set_initialized_queues(mytid, task_id);
          Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)record->begin_ns/1e3));
          Tau_create_top_level_timer_if_necessary_task(task_id);
          Tau_add_metadata_for_task("TAU_TASK_ID", task_id, task_id);
          Tau_add_metadata_for_task("ROCM_HOST_PROCESS_ID", record->process_id, task_id);
          Tau_add_metadata_for_task("ROCM_HOST_THREAD_ID", record->thread_id, task_id);
        }
      //}
      TAU_VERBOSE(" process_id(%u) thread_id(%u) task_id(%d)\n",
        record->process_id,
        record->thread_id,
        task_id
      );
      Tau_roctracer_hip_event(record, task_id); // on the host 
    } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      TAU_VERBOSE(" ACTIVITY_DOMAIN_HCC_OPS\n");
      task_id = Tau_get_initialized_queues(record->queue_id); 
      if (task_id == -1) {
        TAU_VERBOSE("ACTIVITY_DOMAIN_HIP_API: creating task\n");
        TAU_CREATE_TASK(task_id);
        Tau_set_initialized_queues(record->queue_id, task_id);
        Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)record->begin_ns/1e3));
        Tau_create_top_level_timer_if_necessary_task(task_id);
        Tau_add_metadata_for_task("TAU_TASK_ID", task_id, task_id);
        Tau_add_metadata_for_task("ROCM_GPU_ID", record->device_id, task_id);
        Tau_add_metadata_for_task("ROCM_QUEUE_ID", record->queue_id, task_id);
      }
      TAU_VERBOSE(" device_id(%d) queue_id(%lu)\n",
        record->device_id,
        record->queue_id
      );
      Tau_roctracer_hcc_event(record, task_id);  // on the gpu
    } else {
      TAU_VERBOSE("Bad domain %d\n", record->domain);
      abort();
    }
    if (record->op == hc::HSA_OP_ID_COPY) TAU_VERBOSE(" bytes(0x%zx)", record->bytes);
    TAU_VERBOSE("\n");
    fflush(stdout);
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}

// Init tracing routine
int Tau_roctracer_init_tracing() {
  TAU_VERBOSE("# START #############################\n");
/*
  for (int i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    Tau_set_initialized_queues(i, -1); // set it explicitly
  }
  No need to do this. We use iterators now. 
*/
#if (!(defined (TAU_MPI) || (TAU_SHMEM)))
  TAU_PROFILE_SET_NODE(0);
#endif /* TAU_MPI || TAU_SHMEM */
  // Allocating tracing pool
  roctracer_properties_t properties{};
  properties.buffer_size = TAU_ROCTRACER_BUFFER_SIZE; 
  properties.buffer_callback_fun = Tau_roctracer_activity_callback;
  ROCTRACER_CALL(roctracer_open_pool(&properties));
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Start tracing routine
extern "C" void Tau_roctracer_start_tracing() {
  static int flag = Tau_roctracer_init_tracing(); 
  TAU_VERBOSE("# START #############################\n");
  // Enable HIP API callbacks
  ROCTRACER_CALL(roctracer_enable_callback(Tau_roctracer_api_callback, NULL));
  // Enable HIP activity tracing
  ROCTRACER_CALL(roctracer_enable_activity());
  // Enable HIP API callbacks
}

// Stop tracing routine
extern "C" void Tau_roctracer_stop_tracing() {
  ROCTRACER_CALL(roctracer_disable_callback());
  ROCTRACER_CALL(roctracer_disable_activity());
  ROCTRACER_CALL(roctracer_flush_activity());
  //Tau_stop_top_level_timer_if_necessary(); // check if this is the call for tasks.
  TAU_VERBOSE("# STOP  #############################\n");
}


#define PUBLIC_API __attribute__((visibility("default")))
extern "C" PUBLIC_API void OnUnloadTool() {
  TAU_VERBOSE("Inside OnUnloadTool\n");
}

extern "C" PUBLIC_API void OnLoadTool() {
  TAU_VERBOSE("Inside OnLoadTool\n");
}


#ifdef TAU_ENABLE_ROCTRACER_HSA
void Tau_roctracer_hsa_activity_callback(
  uint32_t op,
  activity_record_t* record,
  void* arg)
{
  TAU_VERBOSE("%lu:%lu async-copy%lu\n", record->begin_ns, record->end_ns, record->correlation_id);
  //printf("%lu:%lu async-copy%lu\n", record->begin_ns, record->end_ns, record->correlation_id);
}

typedef hsa_rt_utils::Timer::timestamp_t timestamp_t;
hsa_rt_utils::Timer* timer = NULL;
thread_local timestamp_t hsa_begin_timestamp = 0;
thread_local timestamp_t hip_begin_timestamp = 0;

// HSA API callback function
void Tau_roctracer_hsa_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
  const char *activity_name = roctracer_op_string(ACTIVITY_DOMAIN_HSA_API, cid, 0); 


  int tid = syscall(__NR_gettid);
  hsa_begin_timestamp = timer->timestamp_fn_ns();
  int task_id = Tau_get_initialized_queues(tid);
  if (task_id == -1) {
    TAU_VERBOSE("ACTIVITY_DOMAIN_HSA_API: creating task\n");
    TAU_CREATE_TASK(task_id);
    Tau_set_initialized_queues(tid, task_id);
    Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(hsa_begin_timestamp)/1e3));
    Tau_create_top_level_timer_if_necessary_task(task_id);
    Tau_add_metadata_for_task("TAU_TASK_ID", task_id, task_id);
    Tau_add_metadata_for_task("TAU_ROCM_THREAD_ID", tid, task_id);
    //printf("TAU_ROCM_THREAD_ID: %d on task id %d \n", tid, task_id);
  }

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(hsa_begin_timestamp)/1e3)); // convert to microseconds
    TAU_VERBOSE("Start: %s on tid=%d timestamp=%llu\n", activity_name, task_id,hsa_begin_timestamp);
    TAU_START_TASK(activity_name, task_id);
  } else {
    if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      //const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down) ? hsa_begin_timestamp : timer->timestamp_fn_ns();
      const timestamp_t end_timestamp = timer->timestamp_fn_ns();
      Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(end_timestamp)/1e3)); // convert to microseconds
      TAU_VERBOSE("Stop : %s on tid=%d timestamp=%llu\n", activity_name, task_id, end_timestamp);
      TAU_STOP_TASK(activity_name, task_id);
    }
  }
}


extern "C" PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count, const char* const* failed_tool_names) {
  timer = new hsa_rt_utils::Timer(table->core_->hsa_system_get_info_fn);

  // API trace vector
  std::vector<std::string> hsa_api_vec;

  // initialize HSA tracing
  roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*)table);
  roctracer::hsa_ops_properties_t ops_properties{
    table,
    reinterpret_cast<activity_async_callback_t>(Tau_roctracer_hsa_activity_callback),
    NULL, 
    NULL};
  roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);
  TAU_VERBOSE("TAU: HSA TRACING ENABLED\n");

  if (hsa_api_vec.size() != 0) {
    for (unsigned i = 0; i < hsa_api_vec.size(); ++i) {
      uint32_t cid = HSA_API_ID_NUMBER;
      const char* api = hsa_api_vec[i].c_str();
      ROCTRACER_CALL(roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api, &cid));
      ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_API, cid, Tau_roctracer_hsa_api_callback, NULL));
      TAU_VERBOSE(" %s", api);
    }
  } else {
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, Tau_roctracer_hsa_api_callback, NULL));
  }
  ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
  return true;
}

extern "C" PUBLIC_API void OnUnload() {
  printf("Inside OnUnload\n");
  TAU_VERBOSE("Inside OnUnload\n");
}
#endif /* TAU_ENABLE_ROCTRACER_HSA */



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
