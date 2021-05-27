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
#include <roctracer_roctx.h>
#include <roctracer_hsa.h>
#include <roctracer_hip.h>
#include <roctracer_hcc.h>
#include <stdlib.h>

//#define TAU_ENABLE_ROCTRACER_HSA
#ifdef TAU_ENABLE_ROCTRACER_HSA
#ifndef TAU_HSA_TASK_ID
#define TAU_HSA_TASK_ID 501
#endif /* TAU_HSA_TASK_ID */
#include <roctracer_hsa.h>
#include <roctracer_hip.h>
#include <roctracer_hcc.h>
//#include <src/core/loader.h>
//#include <src/core/trace_buffer.h>
#include <ext/hsa_rt_utils.hpp>
#endif /* TAU_ENABLE_ROCTRACER_HSA */
#include <sys/syscall.h>

#include <string>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <stack>

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

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)


#ifndef onload_debug
#define onload_debug false
#endif

int64_t deltaTimestamp = 0;

/* I know it's bad form to have this map just hanging out here,
 * but when I wrapped it with a getter function, it failed to work.
 * A regular map was always empty, and an unordered map would crash
 * at the find method.  Whatever.  Maybe it's a hipcc problem.
 * So, it's possible we can access this map after the program has
 * destroyed it, but that's a risk I am willing to take. */
using mapType = std::unordered_map<uint64_t, std::string>;
static mapType themap;
std::mutex mapLock;

// Launch a kernel
void Tau_roctracer_register_activity(uint64_t id, const char *name) {
    std::string n(name);
    mapLock.lock();
    themap.insert(std::pair<uint64_t, std::string>(id, n));
    mapLock.unlock();
    return;
}

// resolve a kernel
std::string Tau_roctracer_lookup_activity(uint64_t id) {
    std::string name("");
    mapLock.lock();
    auto i = themap.find(id);
    if (i != themap.end()) {
        name = i->second;
        themap.erase(id);
    }
    mapLock.unlock();
    return name;
}

bool run_once() {
    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    uint64_t startTimestampCPU = TauTraceGetTimeStamp();
    uint64_t startTimestampGPU;
    roctracer_get_timestamp(&startTimestampGPU);
    startTimestampCPU += TauTraceGetTimeStamp();
    startTimestampCPU = startTimestampCPU / 2;

    // assume CPU timestamp is greater than GPU
    deltaTimestamp = (int64_t)(startTimestampCPU) - (int64_t)(startTimestampGPU);
    return true;
}

void Tau_roctx_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
    static thread_local std::stack<std::string> timer_stack;
    static std::map<roctx_range_id_t, std::string> timer_map;
    static std::mutex map_lock;
    if (domain != ACTIVITY_DOMAIN_ROCTX) { return; }
    const roctx_api_data_t* data = (const roctx_api_data_t*)(callback_data);
    switch (cid) {
        case ROCTX_API_ID_roctxRangePushA:
            {
                std::stringstream ss;
                ss << "roctx: " << data->args.message;
                timer_stack.push(ss.str());
                TAU_START(ss.str().c_str());
                break;
            }
        case ROCTX_API_ID_roctxRangePop:
            {
                TAU_STOP(timer_stack.top().c_str());
                timer_stack.pop();
                break;
            }
        case ROCTX_API_ID_roctxRangeStartA:
            {
                std::stringstream ss;
                ss << "roctx: " << data->args.message;
                TAU_START(ss.str().c_str());
                const std::lock_guard<std::mutex> guard(map_lock);
                timer_map.insert(
                        std::pair<roctx_range_id_t, std::string>(
                            data->args.id, ss.str()));
                break;
            }
        case ROCTX_API_ID_roctxRangeStop:
            {
                const std::lock_guard<std::mutex> guard(map_lock);
                auto p = timer_map.find(data->args.id);
                if (p != timer_map.end()) {
                    TAU_STOP(p->second.c_str());
                    timer_map.erase(data->args.id);
                }
                break;
            }
        case ROCTX_API_ID_roctxMarkA:
            // we do nothing with marker events...for now
        default:
            break;
    }
    return;

}

// Runtime API callback function
void Tau_roctracer_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
    static bool dummy = run_once();
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
    TAU_START(activity_name);
    switch (cid) {
      case HIP_API_ID_hipLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address, data->args.hipLaunchKernel.stream));
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f));
        break;
      case HIP_API_ID_hipHccModuleLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRef(data->args.hipHccModuleLaunchKernel.f));
        break;
      case HIP_API_ID_hipExtModuleLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRef(data->args.hipExtModuleLaunchKernel.f));
        break;
      case HIP_API_ID_hipExtLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRefByPtr(data->args.hipExtLaunchKernel.function_address, data->args.hipLaunchKernel.stream));
        break;
      default:
        // not necessary.
        //Tau_roctracer_register_activity(data->correlation_id, activity_name);
        break;
    }
  } else {
    TAU_STOP(activity_name);
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

// gpu based events
void Tau_roctracer_hcc_event(const roctracer_record_t *record, int task_id, uint64_t begin_ns, uint64_t end_ns) {
  Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(begin_ns)/1e3)); // convert to microseconds
  int status;
  char *joined_name ;
  if ((record->op == HIP_OP_ID_DISPATCH)) {
    std::string name = Tau_roctracer_lookup_activity(record->correlation_id);
    const char *demangled_name;
    TAU_INTERNAL_DEMANGLE_NAME(name.c_str(), demangled_name);
    joined_name = (char *) string(string(roctracer_op_string(record->domain, record->op, record->kind)) +" "+ demangled_name).c_str();
  } else {
    joined_name = (char *) roctracer_op_string(record->domain, record->op, record->kind);
  }
  TAU_START_TASK(joined_name, task_id);
  TAU_VERBOSE("Started event %s on task %d timestamp = %lu \n", joined_name, task_id, begin_ns);

  // and the end
  Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(end_ns)/1e3)); // convert to microseconds
  TAU_STOP_TASK(joined_name, task_id);
  TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", joined_name, task_id, end_ns);
  Tau_set_last_timestamp_ns(end_ns);

  return;
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void Tau_roctracer_activity_callback(const char* begin, const char* end, void* arg) {
    static bool dummy = run_once();
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
    uint64_t begin_ns = record->begin_ns + deltaTimestamp;
    uint64_t end_ns = record->end_ns + deltaTimestamp;
    if (record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
      TAU_VERBOSE(" ACTIVITY_DOMAIN_HIP_OPS\n");
      task_id = Tau_get_initialized_queues(record->queue_id);
      if (task_id == -1) {
        TAU_VERBOSE("ACTIVITY_DOMAIN_HIP_API: creating task\n");
        TAU_CREATE_TASK(task_id);
        Tau_set_initialized_queues(record->queue_id, task_id);
        Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)begin_ns/1e3));
        Tau_create_top_level_timer_if_necessary_task(task_id);
        Tau_add_metadata_for_task("TAU_TASK_ID", task_id, task_id);
        Tau_add_metadata_for_task("ROCM_GPU_ID", record->device_id, task_id);
        Tau_add_metadata_for_task("ROCM_QUEUE_ID", record->queue_id, task_id);
      }
      TAU_VERBOSE(" device_id(%d) queue_id(%lu)\n",
        record->device_id,
        record->queue_id
      );
      Tau_roctracer_hcc_event(record, task_id, begin_ns, end_ns);  // on the gpu
    } else {
      TAU_VERBOSE("Bad domain %d\n", record->domain);
      abort();
    }
    if (record->op == HIP_OP_ID_COPY) TAU_VERBOSE(" bytes(0x%zx)", record->bytes);
    TAU_VERBOSE("\n");
    fflush(stdout);
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}



struct hip_api_trace_entry_t {
  uint32_t valid;
  uint32_t type;
  uint32_t domain;
  uint32_t cid;
  uint64_t begin;
  uint64_t end;
  uint32_t pid;
  uint32_t tid;
  hip_api_data_t data;
  const char* name;
  void* ptr;
};


void Tau_roctracer_hip_api_flush_cb(hip_api_trace_entry_t* entry) {
  const uint32_t domain = entry->domain;
  const uint32_t cid = entry->cid;
  const hip_api_data_t* data = &(entry->data);
  const uint64_t begin_timestamp = entry->begin;
  const uint64_t end_timestamp = entry->end;
  std::ostringstream oss;                                                                        \

  printf("Tau_roctracer_hip_api_flush_cb\n");
  const char* str = (domain < ACTIVITY_DOMAIN_NUMBER) ? roctracer_op_string(domain, cid, 0) : strdup("MARK");
  const char *demangled_name;
  oss << std::dec <<
    begin_timestamp << ":" << end_timestamp << " " << entry->pid << ":" << entry->tid << " " << str;

  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        TAU_VERBOSE("%s(dst(%p) src(%p) size(0x%x) kind(%u))\n",
          oss.str().c_str(),
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        TAU_VERBOSE("%s(ptr(%p) size(0x%x))\n",
          oss.str().c_str(),
          entry->ptr,
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        TAU_VERBOSE("%s(ptr(%p))\n",
          oss.str().c_str(),
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
        TAU_INTERNAL_DEMANGLE_NAME(entry->name, demangled_name);
        TAU_VERBOSE("%s(kernel(%s) stream(%p))\n",
          oss.str().c_str(),
          demangled_name,
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        TAU_VERBOSE("%s()\n", oss.str().c_str());
    }
  } else {
    TAU_VERBOSE("%s(name(%s))\n", oss.str().c_str(), entry->name);
  }

}

// Init tracing routine
int Tau_roctracer_init_tracing() {
  TAU_VERBOSE("# START init_tracing: #############################\n");
#if (!(defined (TAU_MPI) || (TAU_SHMEM)))
  if (Tau_get_node() == -1) {
      TAU_PROFILE_SET_NODE(0);
  }
#endif /* TAU_MPI || TAU_SHMEM */
  // set roctracer properties
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);
  // Allocating tracing pool
  roctracer_properties_t properties{};
  memset(&properties, 0, sizeof(roctracer_properties_t));
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
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, Tau_roctracer_api_callback, NULL));
  // Enable ROCTX Instrumentation API support
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, Tau_roctx_api_callback, NULL));
  // Enable HIP activity tracing
  ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  // Enable HIP API callbacks
}

// Stop tracing routine
extern "C" void Tau_roctracer_stop_tracing() {
  if (RtsLayer::myThread() != 0) return;
  ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
  ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  ROCTRACER_CALL(roctracer_flush_activity());
  //Tau_stop_top_level_timer_if_necessary(); // check if this is the call for tasks.
  TAU_VERBOSE("# STOP  #############################\n");
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
