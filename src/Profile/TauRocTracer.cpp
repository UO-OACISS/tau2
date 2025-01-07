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
#include <Profile/TauBfd.h>  // for name demangling
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
#include <chrono>

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

/* The delta timestamp is in nanoseconds. */
int64_t deltaTimestamp_ns = 0;

/* I know it's bad form to have this map just hanging out here,
 * but when I wrapped it with a getter function, it failed to work.
 * A regular map was always empty, and an unordered map would crash
 * at the find method.  Whatever.  Maybe it's a hipcc problem.
 * So, it's possible we can access this map after the program has
 * destroyed it, but that's a risk I am willing to take. */
using mapType = std::unordered_map<uint64_t, std::string>;
static mapType themap;
std::mutex mapLock;

// the user event for correlation IDs
static void* TraceCorrelationID;

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
    } else {
        TAU_VERBOSE("WARNING! Kernel name not found for correlation %lu\n", id);
    }
    mapLock.unlock();
    return name;
}

/* TAU uses microsecond clock for timestamps, but the GPU provides the
 * stamps in nanoseconds.  So, in order to compute the delta between
 * the CPU clock and GPU clock, we need to take a CPU timestamp in nanoseconds
 * and then get the delta.  The delta will be in nanoseconds.  So when we
 * adjust for the asynchronous activity, we will apply the nanosecond delta
 * and then convert to microseconds.
 */
#define MYCLOCK std::chrono::system_clock
static uint64_t time_point_to_nanoseconds(std::chrono::time_point<MYCLOCK> tp) {
    auto value = tp.time_since_epoch();
    uint64_t duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(value).count();
    return duration;
}
static uint64_t now_ns() {
    return time_point_to_nanoseconds(MYCLOCK::now());
}

bool run_once() {
    // synchronize timestamps
    // We'll take a CPU timestamp before and after taking a GPU timestmp, then
    // take the average of those two, hoping that it's roughly at the same time
    // as the GPU timestamp.
    uint64_t startTimestampCPU = now_ns(); //TauTraceGetTimeStamp(); // TAU is in microseconds!
    uint64_t startTimestampGPU;
    roctracer_get_timestamp(&startTimestampGPU);
    startTimestampCPU += now_ns(); //TauTraceGetTimeStamp(); // TAU is in microseconds!
    startTimestampCPU = startTimestampCPU / 2;

    // assume CPU timestamp is greater than GPU
    TAU_VERBOSE("HIP timestamp: %lu\n", startTimestampGPU);
    TAU_VERBOSE("CPU timestamp: %lu\n", startTimestampCPU);
    deltaTimestamp_ns = (int64_t)(startTimestampCPU) - (int64_t)(startTimestampGPU);
    TAU_VERBOSE("HIP delta timestamp: %ld\n", deltaTimestamp_ns);
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

inline void capture_correlation_id(uint32_t correlation_id) {
    if (TauEnv_get_thread_per_gpu_stream()) {
      TAU_TRIGGER_EVENT("Correlation ID", (double)(correlation_id));
    }
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
  /* Check for a couple of useless callbacks, we don't need to track them */
  if (cid == HIP_API_ID___hipPushCallConfiguration ||
      cid == HIP_API_ID___hipPopCallConfiguration) {
      return;
  }
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  const char *activity_name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
  /*
  TAU_VERBOSE("<%s id(%u)\tcorrelation_id(%lu) %s> ",
    activity_name,
    cid,
    data->correlation_id,
    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");
    */
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    TAU_START(activity_name);
    switch (cid) {
      /* If it's a kernel launch, map the correlation ID to the kernel name.
         also, capture the correlation ID */
      case HIP_API_ID_hipLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address,
          data->args.hipLaunchKernel.stream));
        capture_correlation_id(data->correlation_id);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f));
        capture_correlation_id(data->correlation_id);
        break;
      case HIP_API_ID_hipHccModuleLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRef(data->args.hipHccModuleLaunchKernel.f));
        capture_correlation_id(data->correlation_id);
        break;
      case HIP_API_ID_hipExtModuleLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRef(data->args.hipExtModuleLaunchKernel.f));
        capture_correlation_id(data->correlation_id);
        break;
      case HIP_API_ID_hipExtLaunchKernel:
        Tau_roctracer_register_activity(data->correlation_id,
          hipKernelNameRefByPtr(data->args.hipExtLaunchKernel.function_address,
          data->args.hipLaunchKernel.stream));
        capture_correlation_id(data->correlation_id);
        break;
      /* If it's a memcpy, capture the correlation ID */
      case HIP_API_ID_hipMemcpy:
      case HIP_API_ID_hipMemcpy2D:
      case HIP_API_ID_hipMemcpy2DAsync:
      case HIP_API_ID_hipMemcpy2DFromArray:
      case HIP_API_ID_hipMemcpy2DFromArrayAsync:
      case HIP_API_ID_hipMemcpy2DToArray:
      case HIP_API_ID_hipMemcpy2DToArrayAsync:
      case HIP_API_ID_hipMemcpy3D:
      case HIP_API_ID_hipMemcpy3DAsync:
      case HIP_API_ID_hipMemcpyAsync:
      case HIP_API_ID_hipMemcpyAtoH:
      case HIP_API_ID_hipMemcpyDtoD:
      case HIP_API_ID_hipMemcpyDtoDAsync:
      case HIP_API_ID_hipMemcpyDtoH:
      case HIP_API_ID_hipMemcpyDtoHAsync:
      case HIP_API_ID_hipMemcpyFromArray:
      case HIP_API_ID_hipMemcpyFromSymbol:
      case HIP_API_ID_hipMemcpyFromSymbolAsync:
      case HIP_API_ID_hipMemcpyHtoA:
      case HIP_API_ID_hipMemcpyHtoD:
      case HIP_API_ID_hipMemcpyHtoDAsync:
      case HIP_API_ID_hipMemcpyParam2D:
      case HIP_API_ID_hipMemcpyParam2DAsync:
      case HIP_API_ID_hipMemcpyPeer:
      case HIP_API_ID_hipMemcpyPeerAsync:
      case HIP_API_ID_hipMemcpyToArray:
      case HIP_API_ID_hipMemcpyToSymbol:
      case HIP_API_ID_hipMemcpyToSymbolAsync:
      case HIP_API_ID_hipMemcpyWithStream:
        capture_correlation_id(data->correlation_id);
        break;
      default:
        // not necessary.
        //Tau_roctracer_register_activity(data->correlation_id, activity_name);
        break;
    }
  } else {
    TAU_STOP(activity_name);
    /*
    switch (cid) {
      case HIP_API_ID_hipMalloc:
        TAU_VERBOSE("*ptr(0x%p)", *(data->args.hipMalloc.ptr));
        break;
      default:
        break;
    }
    */
  }
  //TAU_VERBOSE("\n"); fflush(stdout);
}

// gpu based events
void Tau_roctracer_hcc_event(const roctracer_record_t *record,
    int task_id, uint64_t begin_us, uint64_t end_us) {

  int status;
  char *joined_name;
  if ((record->op == HIP_OP_ID_DISPATCH)) {
    std::string name = Tau_roctracer_lookup_activity(record->correlation_id);
    char *demangled_name = Tau_demangle_name(name.c_str());
    std::stringstream ss;
    ss << roctracer_op_string(record->domain, record->op, record->kind);
    if (strlen(demangled_name) == 0) {
        ss << " " << name;
    } else {
        ss << " " << demangled_name;
    }
    free(demangled_name);
    joined_name = strdup(ss.str().c_str());
  } else {
    joined_name = strdup(roctracer_op_string(record->domain, record->op, record->kind));
  }
  // ignore marker events, not sure where they come from or what they are for
  if (strncmp(joined_name, "Marker", 6) == 0) return;
  // ignore FillBuffer events, not sure where they come from or what they are for
  if (strncmp(joined_name, "FillBuffer", 6) == 0) return;
  //TAU_VERBOSE("*** Begin: %lu, End: %lu\n", begin_us, end_us);
  //double ts = Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(begin_us))); // convert to microseconds
  metric_set_gpu_timestamp(task_id, ((double)(begin_us)));
  TAU_START_TASK(joined_name, task_id);
  //TAU_VERBOSE("Started event %s on task %d timestamp = %lu \n", joined_name, task_id, begin_us);

  // and the context ID
  if (TauEnv_get_thread_per_gpu_stream()) {
    double cid = (double)(record->correlation_id);
    TAU_CONTEXT_EVENT_THREAD_TS(TraceCorrelationID, cid, task_id, begin_us);
    //TAU_CONTEXT_EVENT_THREAD(TraceCorrelationID, cid, task_id);
  }

  // and the end
  //Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)(end_us))); // convert to microseconds
  metric_set_gpu_timestamp(task_id, ((double)(end_us)));
  TAU_STOP_TASK(joined_name, task_id);
  //TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", joined_name, task_id, end_us);
  Tau_set_last_timestamp_ns(end_us);
  free (joined_name);

  return;
}

//In some cases, such as srun, the pool is unsorted
uint64_t get_lowest_timestamp(const char* begin, const char* end)
{
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
  uint64_t lowest = record->begin_ns;
  while (record < end_record) {
    if(record->begin_ns < lowest)
      lowest = record->begin_ns;
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
  return lowest;
}


// Activity tracing callback
void Tau_roctracer_activity_callback(const char* begin, const char* end, void* arg) {
  TAU_VERBOSE("Tau_roctracer_activity_callback\n"); fflush(stdout);
  //bool dummy = run_once();  // actually, run it every time we process the buffer
  int dispatch_task_id=-1;
  int copy_task_id = -1;
  int barrier_task_id = -1;
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
  TAU_VERBOSE("\tActivity records :\n"); fflush(stdout);
  uint64_t lowest_timestamp = (get_lowest_timestamp(begin, end) + deltaTimestamp_ns) * 1.0e-3;

  //TAU_VERBOSE("\tActivity records :\n"); fflush(stdout);
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    // adjust the timestamp drift between CPU clock and GPU clock, and
    // TAU uses microseconds, so convert ns to us
    uint64_t begin_us = (record->begin_ns + deltaTimestamp_ns) * 1.0e-3;
    uint64_t end_us = (record->end_ns + deltaTimestamp_ns) * 1.0e-3;
    /*TAU_VERBOSE("\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)\n",
      name, record->correlation_id, begin_us, end_us);*/
    if (record->domain == ACTIVITY_DOMAIN_HIP_OPS) {
      dispatch_task_id = Tau_get_initialized_queues(record->queue_id);
      if (dispatch_task_id == -1) {
        // OK, this looks unusual but hear me out.
        // We need 3 threads, one for dispatch, one for memory transfers
        // (copy) and one for sync (barrier).  These event types can overlap.
        // So each one needs a unique virtual thread.
        // We lock the DB, so that we can create three consecutive thread
        // IDs with impunity.
        //TAU_VERBOSE("ACTIVITY_DOMAIN_HIP_API: creating task\n");
        // one for dispatch
        TAU_CREATE_TASK(dispatch_task_id);
        Tau_set_initialized_queues(record->queue_id, dispatch_task_id);
        if (TauEnv_get_thread_per_gpu_stream()) {
            // another one for copy
            TAU_CREATE_TASK(copy_task_id);
            // another one for barrier
            TAU_CREATE_TASK(barrier_task_id);
        } else {
            // when profiling, we just need one thread per device, really.
            copy_task_id = dispatch_task_id;
            barrier_task_id = dispatch_task_id;
        }
        //Tau_metric_set_synchronized_gpu_timestamp(task_id, ((double)begin_us));
        metric_set_gpu_timestamp(dispatch_task_id, ((double)(lowest_timestamp)));
        Tau_create_top_level_timer_if_necessary_task(dispatch_task_id);
        Tau_add_metadata_for_task("TAU_TASK_ID", dispatch_task_id, dispatch_task_id);
        Tau_add_metadata_for_task("ROCM_GPU_ID", record->device_id, dispatch_task_id);
        Tau_add_metadata_for_task("ROCM_QUEUE_ID", record->queue_id, dispatch_task_id);
        Tau_metadata_task("ROCM_QUEUE_TYPE", "Kernel Dispatch", dispatch_task_id);
        //Use the same timestamp to simplify things
        if (TauEnv_get_thread_per_gpu_stream()) {
            metric_set_gpu_timestamp(copy_task_id, ((double)(lowest_timestamp)));
            Tau_create_top_level_timer_if_necessary_task(copy_task_id);
            Tau_add_metadata_for_task("TAU_TASK_ID", copy_task_id, copy_task_id);
            Tau_add_metadata_for_task("ROCM_GPU_ID", record->device_id, copy_task_id);
            Tau_add_metadata_for_task("ROCM_QUEUE_ID", record->queue_id, copy_task_id);
            Tau_metadata_task("ROCM_QUEUE_TYPE", "Memory Transfers", copy_task_id);
            metric_set_gpu_timestamp(barrier_task_id, ((double)(lowest_timestamp)));
            Tau_create_top_level_timer_if_necessary_task(barrier_task_id);
            Tau_add_metadata_for_task("TAU_TASK_ID", barrier_task_id, barrier_task_id);
            Tau_add_metadata_for_task("ROCM_GPU_ID", record->device_id, barrier_task_id);
            Tau_add_metadata_for_task("ROCM_QUEUE_ID", record->queue_id, barrier_task_id);
            Tau_metadata_task("ROCM_QUEUE_TYPE", "Synchronization", barrier_task_id);
        }
      } else {
          if (TauEnv_get_thread_per_gpu_stream()) {
              // we locked when we created them, to they are consecutive
              copy_task_id = dispatch_task_id + 1;
              barrier_task_id = dispatch_task_id + 2;
          } else {
              // when profiling, we just need one thread per device, really.
              copy_task_id = dispatch_task_id;
              barrier_task_id = dispatch_task_id;
          }
      }
      /*TAU_VERBOSE(" device_id(%d) queue_id(%lu)\n",
        record->device_id,
        record->queue_id
      );*/
      if (record->op == HIP_OP_ID_COPY) {
        Tau_roctracer_hcc_event(record, copy_task_id, begin_us, end_us);  // on the gpu
      } else if (record->op == HIP_OP_ID_BARRIER) {
        Tau_roctracer_hcc_event(record, barrier_task_id, begin_us, end_us);  // on the gpu
      } else if (record->op == HIP_OP_ID_DISPATCH) {
        Tau_roctracer_hcc_event(record, dispatch_task_id, begin_us, end_us);  // on the gpu
      } else {
        TAU_VERBOSE("# roctracer Unhandled event! \n");
      }
    } else {
      TAU_VERBOSE("Bad domain %d\n", record->domain);
      abort();
    }
    //if (record->op == HIP_OP_ID_COPY) TAU_VERBOSE(" bytes(0x%zx)", record->bytes);
    //TAU_VERBOSE("\n");
    //fflush(stdout);
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
    // Why is this here?  Well, to make sure that this thread isn't
    // holding the lock!  FOR SOME REASON, the TauGpu code will finish
    // without releasing the lock...
    int tmp = RtsLayer::getNumDBLocks();
    while (tmp != 0) {
        //printf ("\n*\n*\n*\nHolding %d locks!\n*\n*\n*\n",tmp);
        //fflush(stdout);
        RtsLayer::UnLockDB();
        tmp = RtsLayer::getNumDBLocks();
    }
}


// Init tracing routine
int Tau_roctracer_init_tracing() {
  //return 0;
  TAU_VERBOSE("# START roctracer init_tracing: #############################\n");
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
  Tau_get_context_userevent(&TraceCorrelationID, "Correlation ID");
  return 0;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Start tracing routine
extern void Tau_roctracer_start_tracing() {
  //return;
  static int flag = Tau_roctracer_init_tracing();
  TAU_VERBOSE("# START roctracer start_tracing #############################\n");
  // Enable HIP API callbacks
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, Tau_roctracer_api_callback, NULL));
  // Enable ROCTX Instrumentation API support
  ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, Tau_roctx_api_callback, NULL));
  // Enable HIP activity tracing
  ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  // Enable HIP API callbacks
}

// Stop tracing routine
extern void Tau_roctracer_stop_tracing() {
  //return;
  if (RtsLayer::myThread() != 0) return;
  ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
  ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  ROCTRACER_CALL(roctracer_flush_activity());
  //Tau_stop_top_level_timer_if_necessary(); // check if this is the call for tasks.
  TAU_VERBOSE("# STOP roctracer stop_tracing  #############################\n");
}

// Flush tracing routine
extern void Tau_roctracer_flush_tracing() {
  //return;
  TAU_VERBOSE("# roctracer FLUSHING ASYNC! ###################\n");
  ROCTRACER_CALL(roctracer_flush_activity());
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
