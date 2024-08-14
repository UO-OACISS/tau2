#define _DEFAULT_SOURCE
#include <omp.h>

#ifdef _OPENMP
    #if (_OPENMP >= 202011)
        // #warning "Found _OPENMP version 5.1"
        #define TAU_OMPT_USE_TARGET_OFFLOAD
    #elif (_OPENMP == 201811)
        // #warning "Found _OPENMP version 5.0"
        #define TAU_OMPT_USE_TARGET_OFFLOAD
    #else
        #warning "Found _OPENMP version less than 5.0"
        #if defined (__GNUC__) && defined (__GNUC_MINOR__)
            #warning "Building OMPT support for GCC with LLVM library at runtime"
        #else
            #if defined (TAU_USE_OMPT_5_0)
                #undef TAU_USE_OMPT_5_0
            #endif
        #endif
    #endif
#endif

#if defined (TAU_USE_OMPT_5_0)

#include <omp-tools.h>
#include <stdio.h>
#include <sstream>
#include <inttypes.h>

#include <Profile/TauBfd.h>
#include <Profile/Profiler.h>
#include <Profile/TauPluginInternals.h>
#include <tau_internal.h>
//#include "kmp.h"
#include <execinfo.h>
#ifdef OMPT_USE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <assert.h>
#include <unordered_map>
#include <Profile/TauGpuAdapterOpenMP.h>
#include <Profile/TauMetaData.h>
#include <atomic>
#include <array>
#include <iostream>
#include <stack>

#include <pthread.h>

/* 16k buffer should be OK, if this is increased, please increase the size
 * of the circular buffer that maps target_id values to the thread IDs that
 * launched them - see TargetMap class, below! */
#define OMPT_BUFFER_REQUEST_SIZE 16*1024

static bool initializing = false;
static bool initialized = false;
static bool tau_initialized = false;
static thread_local bool is_master{false};
static bool tau_ompt_tracing = false;
/* This is to prevent libotf2 from registering a thread when our worker
 * thread tries to store the asynchronous activity. */
static thread_local bool internal_thread{false};

int get_ompt_tid(void) {
  if (is_master) return 0;
  return Tau_get_thread();
}

int Tau_set_tau_initialized() { tau_initialized = true; return 0;};

static const char* ompt_thread_type_t_values[] = {
  NULL,
  "ompt_thread_initial",
  "ompt_thread_worker",
  "ompt_thread_other"
};

static const char* ompt_task_status_t_values[] = {
  NULL,
  "ompt_task_complete",
  "ompt_task_yield",
  "ompt_task_cancel",
  "ompt_task_others"
};
static const char* ompt_cancel_flag_t_values[] = {
  "ompt_cancel_parallel",
  "ompt_cancel_sections",
  "ompt_cancel_do",
  "ompt_cancel_taskgroup",
  "ompt_cancel_activated",
  "ompt_cancel_detected",
  "ompt_cancel_discarded_task"
};

/* We need this method to make sure that we don't get any callbacks after
 * ompt_finaize_tool() has been called.  We *shouldn't* get any callbacks
 * after that, but with modern GCC implementations, it's happening, even
 * with the Intel/LLVM OpenMP runtime.  This is an insurance policy.  It
 * is only needed for the end of implicit tasks and thread exit events
 * (which can happen by the LLVM runtime when it harvests threads after
 * the program has exited and TAU has been mostly if not entirely destroyed. */
static bool Tau_ompt_finalized(bool changeValue = false) {
    static bool _finalized = false;
    if (changeValue) {
        _finalized = true;
    }
    return _finalized;
}

/* This is used to be able to register a callback for a plugin and still
 * prevent TAU from executing it's part of the callback.
 * The size is updated manually for now as I believe there is no way to get the
 * number of callbacks from an OMPT implementation.
 * The current number of callbacks is around 32 depending on the OMPT version. */
static bool Tau_ompt_callbacks_enabled[128] = {0};

static void format_task_type(int type, char* buffer)
{
  char* progress = buffer;
  if(type & ompt_task_initial) progress += sprintf(progress, "ompt_task_initial");
  if(type & ompt_task_implicit) progress += sprintf(progress, "ompt_task_implicit");
  if(type & ompt_task_explicit) progress += sprintf(progress, "ompt_task_explicit");
  if(type & ompt_task_target) progress += sprintf(progress, "ompt_task_target");
  if(type & ompt_task_undeferred) progress += sprintf(progress, "|ompt_task_undeferred");
  if(type & ompt_task_untied) progress += sprintf(progress, "|ompt_task_untied");
  if(type & ompt_task_final) progress += sprintf(progress, "|ompt_task_final");
  if(type & ompt_task_mergeable) progress += sprintf(progress, "|ompt_task_mergeable");
  if(type & ompt_task_merged) progress += sprintf(progress, "|ompt_task_merged");
}

/* Function pointers.  These are all queried from the runtime during
 * ompt_initialize() */
static ompt_set_callback_t ompt_set_callback = nullptr;
static ompt_finalize_tool_t ompt_finalize_tool = nullptr;
static ompt_get_task_info_t ompt_get_task_info = nullptr;
static ompt_get_thread_data_t ompt_get_thread_data = nullptr;
static ompt_get_parallel_info_t ompt_get_parallel_info = nullptr;
static ompt_get_unique_id_t ompt_get_unique_id = nullptr;
static ompt_get_num_places_t ompt_get_num_places = nullptr;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids = nullptr;
static ompt_get_place_num_t ompt_get_place_num = nullptr;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums = nullptr;
static ompt_get_proc_id_t ompt_get_proc_id = nullptr;
static ompt_enumerate_states_t ompt_enumerate_states = nullptr;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls = nullptr;
// new for target offload
static ompt_set_trace_ompt_t ompt_set_trace_ompt = nullptr;
static ompt_start_trace_t ompt_start_trace = nullptr;
static ompt_flush_trace_t ompt_flush_trace = nullptr;
static ompt_stop_trace_t ompt_stop_trace = nullptr;
static ompt_get_record_ompt_t ompt_get_record_ompt = nullptr;
static ompt_advance_buffer_cursor_t ompt_advance_buffer_cursor = nullptr;
static ompt_translate_time_t ompt_translate_time = nullptr;

/* Global map for mapping target_id data from the host callback to the device activity. */
class TargetMap {
    public:
        TargetMap(const TargetMap&) = delete;
        TargetMap(TargetMap&&) = delete;
        TargetMap& operator=(const TargetMap&) = delete;
        TargetMap& operator=(TargetMap&&) = delete;
        ~TargetMap() {};
        int get_tid() {
            static std::atomic<int> thread_count{0};
            return (++thread_count)-1;
        }
        void add_thread_id(ompt_id_t target_id, int device_id) {
            static thread_local int tid = get_tid();
            size_t location = (size_t)target_id % map_size;
            launching_thread[location] = tid;
            write_index = location;
            if (write_index == read_index) {
                std::cerr << "Error!  The OpenMP target_id map in TAU is too small."
                          << " Please increase the size of the map in "
                          << __FILE__ << std::endl;
                abort();
            }
            // make sure we have a virtual thread for the device activity for this target
            Tau_openmp_get_taskid_from_gpu_event(device_id, tid);
            //printf("%s %lu %d\n", __func__, location, tid);
        }
        int get_thread_id(ompt_id_t target_id) {
            size_t location = (size_t)target_id % map_size;
            read_index = location;
            int tid = launching_thread[location];
            //printf("%s %lu %d\n", __func__, location, tid);
            return tid;
        }
        static TargetMap& instance() {
            static TargetMap theMap;
            return theMap;
        }
    private:
        TargetMap() : write_index(0), read_index(UINT64_MAX) {};
        /* This size (1024) should be sufficient, as long as the buffer size
         * doesn't increase from 16*1024.  If the buffer is increased, please
         * increase the size of this circular buffer accordingly! */
        static constexpr size_t map_size = 1024;
        std::array<int,map_size> launching_thread;
        /* These are used for validation of the buffer size. */
        std::atomic<size_t> write_index;
        std::atomic<size_t> read_index;
};

std::map<int, ompt_device_t*>& getDeviceMap(void) {
    static std::map<int, ompt_device_t*> theMap; 
    return theMap;
}

uint64_t translateTime(int device_id, ompt_device_time_t time) {
    // this time is in seconds, relative to the host
    if (ompt_translate_time != nullptr) {
        double tmp = ompt_translate_time(getDeviceMap()[device_id], time);
        // convert to microseconds
        uint64_t converted = (uint64_t)(tmp * 1.0e6);
        return converted;
    } else {
        uint64_t converted = (uint64_t)(time) * 1.0e-3;
        return time;
    }
}

/*Externs*/
extern "C" char* Tau_ompt_resolve_callsite_eagerly(unsigned long addr, char * resolved_address);

#ifdef TAU_OMPT_USE_TARGET_OFFLOAD

/* Asynchronous device target offload support! */

// Simple print routine that this example uses while traversing
// through the trace records returned as part of the buffer-completion callback
static void print_record_ompt(ompt_record_ompt_t *rec) {
    if (rec == NULL) return;

    /*
    printf("rec=%p type=%d time=%lu thread_id=%lu target_id=%lu\n",
            rec, rec->type, rec->time, rec->thread_id, rec->target_id);
            */
    /* Save some data from the target region, because we don't have it when we
       process the target submit activity. */
    static int target_device_num = 0;
    static ompt_id_t task_id = 0;
    static ompt_id_t target_id = 0;
    static void *codeptr_ra = 0;
    static uint32_t thread_id = 0;
    uint32_t context = 0;
    ompt_device_time_t start = 0;
    ompt_device_time_t end = 0;
    int map_size = 0;
    int device_num = 0;
    std::string name;
    bool make_timer = true;
    static char resolved_address[1024];

    switch (rec->type) {
        case ompt_callback_target:
        case ompt_callback_target_emi: {
            ompt_record_target_t target_rec = rec->record.target;
            make_timer = false;
            /*
            printf("\tRecord Target: kind=%d endpoint=%d device=%d task_id=%lu target_id=%lu codeptr=%p\n",
                    target_rec.kind, target_rec.endpoint, target_rec.device_num,
                    target_rec.task_id, target_rec.target_id, target_rec.codeptr_ra);
                    */
            //printf("\tGPU Device: %d\n", target_rec.device_num);
            if (target_rec.endpoint == ompt_scope_begin) {
                target_device_num = target_rec.device_num;
                // validate the device ID, sometimes we get -1 for no known reason.
                if (target_device_num < 0) { target_device_num = 0; }
                task_id = target_rec.task_id;
                target_id = target_rec.target_id;
                thread_id = TargetMap::instance().get_thread_id(target_id);
                codeptr_ra = const_cast<void*>(target_rec.codeptr_ra);
                if(TauEnv_get_ompt_resolve_address_eagerly()) {
                    Tau_ompt_resolve_callsite_eagerly((unsigned long)codeptr_ra, resolved_address);
                } else {
                    snprintf(resolved_address, sizeof(resolved_address),  " : ADDR <%p>", codeptr_ra);
                }
            } else {
                // clear everything out
                target_device_num = 0;
                task_id = 0;
                target_id = 0;
                thread_id = 0;
                codeptr_ra = nullptr;
                memset(resolved_address, 0, 1024);
            }
            break;
        }
        case ompt_callback_target_data_op:
        case ompt_callback_target_data_op_emi: {
             ompt_record_target_data_op_t target_data_op_rec = rec->record.target_data_op;
             /*
             printf("\t  Record DataOp: host_op_id=%lu optype=%d src_addr=%p src_device=%d "
                     "dest_addr=%p dest_device=%d bytes=%lu end_time=%lu duration=%lu ns codeptr=%p\n",
                     target_data_op_rec.host_op_id, target_data_op_rec.optype,
                     target_data_op_rec.src_addr, target_data_op_rec.src_device_num,
                     target_data_op_rec.dest_addr, target_data_op_rec.dest_device_num,
                     target_data_op_rec.bytes, target_data_op_rec.end_time,
                     target_data_op_rec.end_time - rec->time,
                     target_data_op_rec.codeptr_ra);
                     */
            void * tmp_codeptr_ra;
            if (codeptr_ra != nullptr) {
                tmp_codeptr_ra = codeptr_ra;
            } else {
                tmp_codeptr_ra = const_cast<void*>(target_data_op_rec.codeptr_ra);
            }
            if(TauEnv_get_ompt_resolve_address_eagerly()) {
                Tau_ompt_resolve_callsite_eagerly((unsigned long)tmp_codeptr_ra, resolved_address);
            } else {
                snprintf(resolved_address, sizeof(resolved_address),  " : ADDR <%p>", tmp_codeptr_ra);
            }
            std::stringstream ss;
            // by default, the device num is the target device number
            device_num = target_device_num;
            ss << "GPU: OpenMP Target DataOp ";
            switch (target_data_op_rec.optype) {
                case ompt_target_data_alloc: {
                    ss << "Alloc";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_transfer_to_device: {
                    ss << "Xfer to Dev";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_transfer_from_device: {
                    ss << "Xfer from Dev";
                    device_num = target_data_op_rec.src_device_num;
                    break;
                }
                case ompt_target_data_delete: {
                    ss << "Delete";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_associate: {
                    ss << "Associate";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_disassociate: {
                    ss << "Disassociate";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_alloc_async: {
                    ss << "Alloc";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_transfer_to_device_async: {
                    ss << "Xfer to Dev Async";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
                case ompt_target_data_transfer_from_device_async: {
                    ss << "Xfer from Dev Async";
                    device_num = target_data_op_rec.src_device_num;
                    break;
                }
                case ompt_target_data_delete_async: {
                    ss << "Delete Async";
                    device_num = target_data_op_rec.dest_device_num;
                    break;
                }
            }
            ss << " " << resolved_address;
            name = ss.str();
            start = rec->time;
            end = target_data_op_rec.end_time;
            break;
        }
        case ompt_callback_target_submit:
        case ompt_callback_target_submit_emi: {
            ompt_record_target_kernel_t target_kernel_rec = rec->record.target_kernel;
            /*
            printf("\t  Record Submit: host_op_id=%lu requested_num_teams=%u granted_num_teams=%u "
                    "end_time=%lu duration=%lu ns\n",
                    target_kernel_rec.host_op_id, target_kernel_rec.requested_num_teams,
                    target_kernel_rec.granted_num_teams, target_kernel_rec.end_time,
                    target_kernel_rec.end_time - rec->time);
                    */

            std::stringstream ss;
            ss << "GPU: OpenMP Target Submit " << resolved_address;
            name = ss.str();
            start = rec->time;
            end = target_kernel_rec.end_time;
            device_num = target_device_num;
            break;
        }
        default: {
            break;
        }
    }
    if (make_timer) {
        // convert the start and end times
        Tau_openmp_register_gpu_event(name.c_str(), device_num, thread_id, task_id, target_id, nullptr, map_size, translateTime(device_num, start), translateTime(device_num, end));
    }
}

// Deallocation routine that will be called by the tool when a buffer
// previously allocated by the buffer-request callback is no longer required.
// The deallocation method must match the allocation routine. Here
// free is used for corresponding malloc
static void delete_buffer_ompt(ompt_buffer_t *buffer) {
  free(buffer);
  /*
  printf("Deallocated %p\n", buffer);
  */
}

// OMPT callbacks

// Trace record callbacks
// Allocation routine
static void on_ompt_callback_buffer_request (
  int device_num,
  ompt_buffer_t **buffer,
  size_t *bytes
) {
  TauInternalFunctionGuard protects_this_function;
  *bytes = OMPT_BUFFER_REQUEST_SIZE;
  *buffer = malloc(*bytes);
  /*
    printf("%s\n", __func__);
  printf("Allocated %lu bytes at %p in buffer request callback\n", *bytes, *buffer);
  */
}

// This function is called by an OpenMP runtime helper thread for
// returning trace records from a buffer.
// Note: This callback must handle a null begin cursor. Currently,
// ompt_get_record_ompt, print_record_ompt, and
// ompt_advance_buffer_cursor handle a null cursor.
static void on_ompt_callback_buffer_complete (
  int device_num,
  ompt_buffer_t *buffer,
  size_t bytes, /* bytes returned in this callback */
  ompt_buffer_cursor_t begin,
  int buffer_owned
) {
  TauInternalFunctionGuard protects_this_function;
/*
    printf("%s\n", __func__);
  printf("Executing buffer complete callback: %d %p %lu %p %d\n",
	 device_num, buffer, bytes, (void*)begin, buffer_owned);
     */
    //TAU_START("OMPT Activity handler");
    internal_thread = true;

  int status = 1;
  ompt_buffer_cursor_t current = begin;
  while (status) {
    ompt_record_ompt_t *rec = ompt_get_record_ompt(buffer, current);
    print_record_ompt(rec);
    status = ompt_advance_buffer_cursor(NULL, /* TODO device */
					buffer,
					bytes,
					current,
					&current);
  }
  if (buffer_owned) delete_buffer_ompt(buffer);
  //TAU_STOP("OMPT Activity handler");
}

// Utility routine to enable the desired tracing modes
ompt_set_result_t Tau_ompt_set_trace() {
    //printf("%s\n", __func__);
  if (!ompt_set_trace_ompt) return ompt_set_error;

  ompt_set_trace_ompt(0, 1, ompt_callback_target);
  ompt_set_trace_ompt(0, 1, ompt_callback_target_data_op);
  ompt_set_trace_ompt(0, 1, ompt_callback_target_submit);

  return ompt_set_always;
}

int Tau_ompt_start_trace() {
    //printf("%s\n", __func__);
  if (!ompt_start_trace) return 0;
  tau_ompt_tracing = true;
  return ompt_start_trace(0, &on_ompt_callback_buffer_request,
			  &on_ompt_callback_buffer_complete);
}

#endif // TAU_OMPT_USE_TARGET_OFFLOAD

int Tau_ompt_flush_trace() {
    //printf("%s\n", __func__);
  if (!ompt_flush_trace) return 0;
  if (!tau_ompt_tracing) return 0;
  return ompt_flush_trace(0);
}

int Tau_ompt_stop_trace() {
    //printf("%s\n", __func__);
  tau_ompt_tracing = false;
  if (!ompt_stop_trace) return 0;
  return ompt_stop_trace(0);
}

/* End Asynchronous device target offload support! */

/* IMPT NOTE: In general, we use Tau_global_stop() instead of TAU_PROFILER_STOP(handle) because it is
 * not possible to determine in advance which optional events are supported by the compiler used
 * to compile the application.
 * In general, we have noticed that some compilers generate the START* events but not the corresponding
 * STOP* optional events. As a result, it is impossible to know the right timer to stop.
 * We do NOT want to be adding ugly ifdef's in our code. Thus, the only solution is to hand the
 * responsbility to TAU with a certain loss in accuracy in measured values*/

/* Update: in an attempt to prevent weird errors from using Tau_global_stop(),
 * we will validate the data returned by the runtime, and abort. This behavior
 * can be optionally disabled. */
void stop_correct_timer(void * handle) {
    TAU_ASSERT(handle, "ERROR! OpenMP runtime didn't maintain our tool data! Complain to the vendor!");
    TAU_PROFILER_STOP(handle);
    //Tau_global_stop();
}

/*Parallel begin/end callbacks. We need context information (function name, filename, lineno) for these.
 * For context information the user has one of two options:
 *  a. Embed the address in the timername, and resolve the address when profile files are being written - cheapest and best way to do this
 *  b. Alternatively, resolve the address ONCE during the timer creation. This is useful for situations where TAU is
 *  instrumenting OMPT routines inside shared libraries that are unloaded before profile files get written.
 *  WARNING: Only use option (b) when absolutely necessary. We use shared data structures that need locking/unlocking for storing resolved addresses. This will lead
 *  to overheads that are best avoided. */
static void
on_ompt_callback_parallel_begin(
  ompt_data_t *parent_task_data,
  const ompt_frame_t *parent_task_frame,
  ompt_data_t* parallel_data,
  uint32_t requested_team_size,
  int flags,
  const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_parallel_begin] && Tau_init_check_initialized()) {
    char timerName[10240];
    char resolved_address[1024];

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

      /*Resolve addresses at runtime in case the user really wants to pay the price of doing so.
       *Enabling eager resolving of addresses is only useful in situations where
       *OpenMP routines are instrumented in shared libraries that get unloaded
       *before TAU has a chance to resolve addresses*/
      if (TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
        snprintf(timerName, sizeof(timerName),  "OpenMP_Parallel_Region %s", resolved_address);
      } else {
        snprintf(timerName, sizeof(timerName),  "OpenMP_Parallel_Region ADDR <%lx>", addr);
      }

      void *handle = NULL;
      TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
      parallel_data->ptr = (void*)handle;
      TAU_PROFILER_START(handle);
    }
  }

  if(Tau_plugins_enabled.ompt_parallel_begin) {
    Tau_plugin_event_ompt_parallel_begin_data_t plugin_data;

    plugin_data.encountering_task_data = parent_task_data;
    plugin_data.encountering_task_frame = parent_task_frame;
    plugin_data.parallel_data = parallel_data;
    plugin_data.requested_team_size = requested_team_size;
    plugin_data.flags = flags;
    plugin_data.codeptr_ra = codeptr_ra;
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_PARALLEL_BEGIN, "*", &plugin_data);
  }
}

/* TODO: Remove this and Remove changes to TauEnv.cpp once Intel and LLVM
 * runtime stop causing a deadlock on initialisazion */
static void tau_fix_initialize()
{
    char tmpstr[512];
    int value = omp_get_max_threads();
    snprintf(tmpstr, sizeof(tmpstr), "%d",value);
    TAU_METADATA("OMP_MAX_THREADS", tmpstr);

    value = omp_get_num_procs();
    snprintf(tmpstr, sizeof(tmpstr), "%d",value);
    TAU_METADATA("OMP_NUM_PROCS", tmpstr);
}

static void
on_ompt_callback_parallel_end(
  ompt_data_t *parallel_data,
  ompt_data_t *parent_task_data,
  int flags,
  const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_parallel_end] && Tau_init_check_initialized()) {
    static int once = 1;

    if(once) {
      tau_fix_initialize();
      once = 0;
    }

    if(codeptr_ra) {
      stop_correct_timer(parallel_data->ptr);
    }
  }

  if(Tau_plugins_enabled.ompt_parallel_end) {
    Tau_plugin_event_ompt_parallel_end_data_t plugin_data;

    plugin_data.parallel_data = parallel_data;
    plugin_data.encountering_task_data = parent_task_data;
    plugin_data.flags = flags;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_PARALLEL_END, "*", &plugin_data);
  }
}

/* Callback for task creation. Ideally, this should be traced and NOT profiled as there is no begin/end concept for task creation.
 * TODO: Implement the tracing logic. For now, we are storing this as a context event*/
static void
on_ompt_callback_task_create(
    ompt_data_t *parent_task_data,     /* id of parent task            */
    const ompt_frame_t *parent_frame,  /* frame data for parent task   */
    ompt_data_t* new_task_data,        /* id of created task           */
    int type,
    int has_dependences,
    const void *codeptr_ra)               /* pointer to outlined function */
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_task_create] && Tau_init_check_initialized()) {
    char buffer[2048];
    char timerName[10240];
    char resolved_address[1024];

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      format_task_type(type, buffer);

      /* TODO: Nick's advice: The ThreadTaskCreate/ThreadTaskSwitch/ThreadTaskComplete events are used in OTF2 to indicate creation of a task,
       * execution of a task, or completion of a task. The events are identified solely by a location and a uint64_t taskID.
       * There’s no region associated with it, so there’s no way within that event type, as I can tell, to specify that the task
       * corresponds to a particular region of code.
       * Based on this paper about the task support in OTF2 <http://apps.fz-juelich.de/jsc-pubsystem/aigaion/attachments/schmidl_iwomp2012.pdf-5d909613b453c6fdbf34af237b8d5e52.pdf>
       * it appears that these are supposed to be used in conjunction with Enter and Leave to assign code regions to the task. See Figure 1 in that paper.
       * I would recommend that you use Score-P to generate a trace using those event types and then look at the trace using otf2_print to figure out how it’s using the events.*/

      /* Srinivasan: IMPT - We do NOT start a timer for the task here. We merely create the task. The specification leaves the option
         of (not) starting a task as soon as it is created upto the implementation.
         The LLVM runtime does not start it when it is created.
         We assume this behavior in our implementation of the tool support.*/

      /*Create a timer for the task, and store handle in new_task_data*/
      if (TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
        snprintf(timerName, sizeof(timerName),  "OpenMP_Task %s", resolved_address);
      } else {
        snprintf(timerName, sizeof(timerName),  "OpenMP_Task ADDR <%lx>", addr);
      }

      void *handle = NULL;
      TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
      new_task_data->ptr = (void*)handle;
    }
  }

  if(Tau_plugins_enabled.ompt_task_create) {
    Tau_plugin_event_ompt_task_create_data_t plugin_data;

    plugin_data.encountering_task_data = parent_task_data;
    plugin_data.encountering_frame = parent_frame;
    plugin_data.new_task_data = new_task_data;
    plugin_data.type = type;
    plugin_data.has_dependences = has_dependences;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_TASK_CREATE, "*", &plugin_data);
  }
}

/* Callback for task schedule.
 * For now, we follow the simple logic of stopping the previous
 * task, and starting the next task */
static void
on_ompt_callback_task_schedule(
    ompt_data_t *prior_task_data,
    ompt_task_status_t prior_task_status,
    ompt_data_t *next_task_data)
{
    TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_task_schedule] && Tau_init_check_initialized()) {
    if(prior_task_data->ptr) {
      stop_correct_timer(prior_task_data->ptr);
    }

    if(next_task_data->ptr) {
      TAU_PROFILER_START(next_task_data->ptr);
    }
  }

  if(Tau_plugins_enabled.ompt_task_schedule) {
    Tau_plugin_event_ompt_task_schedule_data_t plugin_data;

    plugin_data.prior_task_data = prior_task_data;
    plugin_data.prior_task_status = prior_task_status;
    plugin_data.next_task_data = next_task_data;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_TASK_SCHEDULE, "*", &plugin_data);
  }
}

#if _OPENMP < 202011 && defined(ompt_callback_master)  // deprecated in 5.1+

/*Master thread begin/end callbacks. We need context information for these.
 * For context information the user has one of two options:
 *  a. Embed the address in the timername, and resolve the address when profile files are being written - cheapest and best way to do this
 *  b. Alternatively, resolve the address ONCE during the timer creation. This is useful for situations where TAU is
 *  instrumenting OMPT routines inside shared libraries that are unloaded before profile files get written.
 *  WARNING: Only use option (b) when absolutely necessary. We use shared data structures that need locking/unlocking for storing resolved addresses. This will lead
 *  to overheads that are best avoided. */
static void
on_ompt_callback_master(
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  /* Per-thread timer stack for this region type.
     this is necessary because we can't rely on the runtime
     to not overwrite the task_data pointer :( */
  thread_local static std::stack<void*> timer_stack;
  if(Tau_ompt_callbacks_enabled[ompt_callback_master] && Tau_init_check_initialized()) {
    char timerName[10240];
    char resolved_address[1024];
    void * codeptr_ra_copy;
    unsigned long addr;
    void *handle = NULL;

    if(codeptr_ra) {
      switch(endpoint)
      {
        case ompt_scope_begin:
          codeptr_ra_copy = (void*) codeptr_ra;
          addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
          /*Resolve addresses at runtime in case the user really wants to pay the price of doing so.
           *Enabling eager resolving of addresses is only useful in situations where
           *OpenMP routines are instrumented in shared libraries that get unloaded
           *before TAU has a chance to resolve addresses*/
          if (TauEnv_get_ompt_resolve_address_eagerly()) {
            Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
            snprintf(timerName, sizeof(timerName),  "OpenMP_Master %s", resolved_address);
          } else {
            snprintf(timerName, sizeof(timerName),  "OpenMP_Master ADDR <%lx>", addr);
          }

          TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
          TAU_PROFILER_START(handle);
          timer_stack.push(handle);
          break;
        case ompt_scope_end:
          handle = timer_stack.top();
          stop_correct_timer(handle);
          timer_stack.pop();
          break;
#if defined(ompt_scope_beginend)
        case ompt_scope_beginend:
#endif
        default:
          // This indicates a coincident beginning and end of scope. Do nothing?
          break;
      }
    }
  }

  if(Tau_plugins_enabled.ompt_master) {
    Tau_plugin_event_ompt_master_data_t plugin_data;

    plugin_data.endpoint = endpoint;
    plugin_data.parallel_data = parallel_data;
    plugin_data.task_data =  task_data;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_MASTER, "*", &plugin_data);
  }
}

#endif // TAU_OMPT_VERSION < 202011 && defined(ompt_callback_master)

/*Work section begin/end callbacks. We need context information for these.
 * For context information the user has one of two options:
 *  a. Embed the address in the timername, and resolve the address when profile files are being written - cheapest and best way to do this
 *  b. Alternatively, resolve the address ONCE during the timer creation. This is useful for situations where TAU is
 *  instrumenting OMPT routines inside shared libraries that are unloaded before profile files get written.
 *  WARNING: Only use option (b) when absolutely necessary. We use shared data structures that need locking/unlocking for storing resolved addresses. This will lead
 *  to overheads that are best avoided.
 *  NOTE: Building this tool with GNU (all versions) has a bug that leads to the single_executor callback not being invoked. LLVM folks
 *  are also aware of this issue. Remove the ifdef's once this bug is resolved.
 *  Tested with Intel/17 compilers. Works OK.*/
static void
on_ompt_callback_work(
  ompt_work_t wstype,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  uint64_t count,
  const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  /* Per-thread timer stack for this region type.
     this is necessary because we can't rely on the runtime
     to not overwrite the task_data pointer :( */
  thread_local static std::stack<void*> timer_stack;
  if(Tau_ompt_callbacks_enabled[ompt_callback_work] && Tau_init_check_initialized()) {
    void *handle = NULL;
    char timerName[10240];
    char resolved_address[1024];
    if(codeptr_ra) {

      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
      } else {
        snprintf(resolved_address, sizeof(resolved_address),  "ADDR <%lx>", addr);
      }

      switch(wstype)
      {
        case ompt_work_loop:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Loop %s", resolved_address);
          break;
        case ompt_work_sections:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Sections %s", resolved_address);
          break;
        case ompt_work_single_executor:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Single_Executor %s", resolved_address);
          break; /* WARNING: The ompt_scope_begin for this work type is triggered, but the corresponding ompt_scope_end is not triggered when using GNU to compile the tool code*/
        case ompt_work_single_other:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Single_Other %s", resolved_address);
          break;
        case ompt_work_workshare:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Workshare %s", resolved_address);
          break;
        case ompt_work_distribute:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Distribute %s", resolved_address);
          break;
        case ompt_work_taskloop:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Taskloop %s", resolved_address);
          break;
#if defined(ompt_work_scope) // why did Intel remove this?
        case ompt_work_scope:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Scope %s", resolved_address);
          break;
#endif
	default:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Work_Other %s", resolved_address);
          break;
      }

      switch(endpoint)
      {
        case ompt_scope_begin:
          TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
          TAU_PROFILER_START(handle);
          timer_stack.push(handle);
          break;
        case ompt_scope_end:
          handle = timer_stack.top();
          stop_correct_timer(handle);
          timer_stack.pop();
          break;
#if defined(ompt_scope_beginend) // why did Intel add this early?
        case ompt_scope_beginend:
#endif
        default:
          // This indicates a coincident beginning and end of scope. Do nothing?
          break;
      }
    }
  }

  if(Tau_plugins_enabled.ompt_work) {
    Tau_plugin_event_ompt_work_data_t plugin_data;

    plugin_data.wstype = wstype;
    plugin_data.endpoint = endpoint;
    plugin_data.parallel_data = parallel_data;
    plugin_data.task_data =  task_data;
    plugin_data.count = count;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_WORK, "*", &plugin_data);
  }
}

/*Thread begin/end callbacks. We do NOT need context information for these. If the user wants more context,
 * he/she can use TAU_CALLPATH=1 to get callpath weights. In our experiments, this leads to additional 10-20%
 * runtime overheads. Use with care. */
static void
on_ompt_callback_thread_begin(
  ompt_thread_t thread_type,
  ompt_data_t *thread_data)
{
  TauInternalFunctionGuard protects_this_function;
  if (internal_thread) { return; }
  if(Tau_ompt_callbacks_enabled[ompt_callback_thread_begin] && Tau_init_check_initialized()) {
    if (is_master ||
        thread_type & ompt_task_initial ) {
        return; // master thread can't be a new worker.
    }
    RtsLayer::RegisterThread();
    Tau_create_top_level_timer_if_necessary();
    void *handle = NULL;
    char timerName[100];
    snprintf(timerName, sizeof(timerName),  "OpenMP_Thread_Type_%s", ompt_thread_type_t_values[thread_type]);
    TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
    thread_data->ptr = (void*)handle;
    TAU_PROFILER_START(handle);
  }

  if(Tau_plugins_enabled.ompt_thread_begin) {
    Tau_plugin_event_ompt_thread_begin_data_t plugin_data;

    plugin_data.thread_type = thread_type;
    plugin_data.thread_data = thread_data;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_THREAD_BEGIN, "*", &plugin_data);
  }
}

static void
on_ompt_callback_thread_end(
  ompt_data_t *thread_data)
{
  TauInternalFunctionGuard protects_this_function;
  if (internal_thread) { return; }
  // Prevent against callbacks after finalization
  if (Tau_ompt_finalized()) { return; }
  if(Tau_ompt_callbacks_enabled[ompt_callback_thread_end] && Tau_init_check_initialized()) {
    if (is_master) return; // master thread can't be a new worker.
    int tid = RtsLayer::myThread();
    Tau_stop_all_timers(tid);
  }

  if(Tau_plugins_enabled.ompt_thread_end) {
    Tau_plugin_event_ompt_thread_end_data_t plugin_data;

    plugin_data.thread_data = thread_data;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_THREAD_END, "*", &plugin_data);
  }
}

/*Implicit task creation. This is a required event, but we do NOT need context.
 * TODO: This is an EXTREMELY high overhead call. The lines causing this overhead are the TAU_PROFILE_START/STOP calls.
 * We have verified that this is not due to any OpenMP runtime overheads, but inside TAU.
 * At the moment, this call is enabled only when using TAU_OMPT support in "full" mode.
 * Since this is a required event, we need to fix this overhead issue with high priority*/
static void
on_ompt_callback_implicit_task(
    ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data,
    ompt_data_t *task_data,
    unsigned int team_size,
    unsigned int thread_num)
{
  TauInternalFunctionGuard protects_this_function;
  /* Per-thread timer stack for this region type.
     this is necessary because we can't rely on the runtime
     to not overwrite the task_data pointer :( */
  thread_local static std::stack<void*> timer_stack;
  // protect against calls after finalization
  if(Tau_ompt_finalized()) { return; }
  if(Tau_ompt_callbacks_enabled[ompt_callback_implicit_task] && Tau_init_check_initialized()) {
    char timerName[100];
    snprintf(timerName, sizeof(timerName),  "OpenMP_Implicit_Task");
    void *handle = NULL;


    switch(endpoint)
    {
      case ompt_scope_begin:
        TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
        TAU_PROFILER_START(handle);
        //TAU_VERBOSE("********* Entering implicit task!\n");
        timer_stack.push(handle);
        break;
      case ompt_scope_end:
        if(task_data->ptr != NULL) {
          handle = timer_stack.top();
          stop_correct_timer(handle);
          timer_stack.pop();
        }
        break;
#if defined(ompt_scope_beginend)
        case ompt_scope_beginend:
#endif
        default:
        // This indicates a coincident beginning and end of scope. Do nothing?
        //TAU_VERBOSE("********* What?!\n");
        break;
    }
  }

  if(Tau_plugins_enabled.ompt_implicit_task) {
    Tau_plugin_event_ompt_implicit_task_data_t plugin_data;

    plugin_data.endpoint = endpoint;
    plugin_data.parallel_data = parallel_data;
    plugin_data.task_data = task_data;
    plugin_data.team_size = team_size;
    plugin_data.thread_num = thread_num;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_IMPLICIT_TASK, "*", &plugin_data);
  }
}

/*Synchronization callbacks (barriers, etc). This is not a required event, but we need context.
 * TODO: This is an EXTREMELY high overhead call. The lines causing this overhead are the TAU_PROFILE_START/STOP calls.
 * We have verified that this is not due to any OpenMP runtime overheads, but inside TAU.
 * At the moment, this call is enabled only when using TAU_OMPT support in "full" mode.
 * Fixing this overhead is relatively low priority, because this is an optional event. */

static void
on_ompt_callback_sync_region(
    ompt_sync_region_t kind,
    ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data,
    ompt_data_t *task_data,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  /* Per-thread timer stack for this region type.
     this is necessary because we can't rely on the runtime
     to not overwrite the task_data pointer :( */
  thread_local static std::stack<void*> timer_stack;
  if(Tau_ompt_callbacks_enabled[ompt_callback_sync_region] && Tau_init_check_initialized()) {
    void *handle = NULL;
    char timerName[10240];
    char resolved_address[1024];

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
      } else {
        snprintf(resolved_address, sizeof(resolved_address),  "ADDR <%lx>", addr);
      }

      switch(kind)
      {
        case ompt_sync_region_barrier:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier %s", resolved_address);
          break;
        case ompt_sync_region_barrier_implicit:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Implicit %s", resolved_address);
          break;
        case ompt_sync_region_barrier_explicit:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Explicit %s", resolved_address);
          break;
        case ompt_sync_region_barrier_implementation:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Implementation %s", resolved_address);
          break;
        case ompt_sync_region_taskwait:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Taskwait %s", resolved_address);
          break;
        case ompt_sync_region_taskgroup:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Taskgroup %s", resolved_address);
          break;
        case ompt_sync_region_reduction:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Reduction %s", resolved_address);
          break;
          // this wasn't in llvm 11
#if defined(ompt_sync_region_barrier_implicit_workshare)
        case ompt_sync_region_barrier_implicit_workshare:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Implicit_Workshare %s", resolved_address);
          break;
#endif
          // this wasn't in llvm 11
#if defined (ompt_sync_region_barrier_implicit_parallel)
        case ompt_sync_region_barrier_implicit_parallel:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Implicit_Parallel %s", resolved_address);
          break;
#endif
          // this wasn't in llvm 11
#if defined(ompt_sync_region_barrier_teams)
        case ompt_sync_region_barrier_teams:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Teams %s", resolved_address);
          break;
#endif
        // "Future proof?"
        default:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Sync_Region_Barrier_Other %s", resolved_address);
          break;
      }

      switch(endpoint)
      {
        case ompt_scope_begin:
          TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
          TAU_PROFILER_START(handle);
          timer_stack.push(handle);
          break;
        case ompt_scope_end:
          handle = timer_stack.top();
          stop_correct_timer(handle);
          timer_stack.pop();
          break;
#if defined(ompt_scope_beginend)
        case ompt_scope_beginend:
#endif
        default:
          // This indicates a coincident beginning and end of scope. Do nothing?
          break;
      }
    }
  }

  if(Tau_plugins_enabled.ompt_sync_region) {
    Tau_plugin_event_ompt_sync_region_data_t plugin_data;

    plugin_data.kind = kind;
    plugin_data.endpoint = endpoint;
    plugin_data.parallel_data = parallel_data;
    plugin_data.task_data = task_data;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_SYNC_REGION, "*", &plugin_data);
  }
}

/* Mutex event - optional event with context */
/* Currently with the LLVM-openmp implementation it seems these mutex events
  are only triggered when using the OpenMP API calls:
     omp_lock_t
     omp_init_lock
     omp_set_lock
     omp_unset_lock

   We ideally would also create timers for #pragma based mutexes, once LLVM-openmp
   implements callbacks for these directives.

   We create the following timers:

     OpenMP_Mutex_Waiting_... - represents time between lock request and acquisition
      entering locked region
     OpenMP_Mutex_Acquired_... - represents time between lock acquisition and release
*/

static void
on_ompt_callback_mutex_acquire(
    ompt_mutex_t kind,
    unsigned int hint,
    unsigned int impl,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_mutex_acquire] && Tau_init_check_initialized()) {
    char timerName[10240];
    char resolved_address[1024];
    void* mutex_waiting_handle=NULL;

    if(codeptr_ra) {

      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
      } else {
        snprintf(resolved_address, sizeof(resolved_address),  "ADDR <%lx>", addr);
      }

      switch(kind)
      {
        case ompt_mutex_lock:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Lock %s", resolved_address);
          break;
        case ompt_mutex_nest_lock:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Nest_Lock %s", resolved_address);
          break;
        case ompt_mutex_test_lock:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Test_Lock %s", resolved_address);
          break;
        case ompt_mutex_test_nest_lock:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Test_Nest_Lock %s", resolved_address);
          break;
        case ompt_mutex_critical:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Critical %s", resolved_address);
          break;
        case ompt_mutex_atomic:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Atomic %s", resolved_address);
          break;
        case ompt_mutex_ordered:
          snprintf(timerName, sizeof(timerName),  "OpenMP_Mutex_Waiting_Ordered %s", resolved_address);
          break;
      }

      // Start lock-wait timer
      TAU_PROFILER_CREATE(mutex_waiting_handle, timerName, " ", TAU_OPENMP);
      TAU_PROFILER_START(mutex_waiting_handle);
    }
  }

  if(Tau_plugins_enabled.ompt_mutex_acquire) {
    Tau_plugin_event_ompt_mutex_acquire_data_t plugin_data;

    plugin_data.kind = kind;
    plugin_data.hint = hint;
    plugin_data.impl = impl;
    plugin_data.wait_id = wait_id;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRE, "*", &plugin_data);
  }
}

static void
on_ompt_callback_mutex_acquired(
    ompt_mutex_t kind,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_mutex_acquired] && Tau_init_check_initialized()) {
    char acquiredtimerName[10240];
    char resolved_address[1024];
    void* mutex_acquired_handle=NULL;

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
      } else {
        snprintf(resolved_address, sizeof(resolved_address),  "OpenMP_Mutex_Acquired_Lock ADDR <%lx>", addr);
      }

      switch(kind)
      {
        case ompt_mutex_lock:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Lock %s", resolved_address);
          break;
        case ompt_mutex_nest_lock:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Nest_Lock %s", resolved_address);
          break;
        case ompt_mutex_test_lock:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Test_Lock %s", resolved_address);
          break;
        case ompt_mutex_test_nest_lock:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Test_Nest_Lock %s", resolved_address);
          break;
        case ompt_mutex_critical:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Critical %s", resolved_address);
          break;
        case ompt_mutex_atomic:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Atomic %s", resolved_address);
          break;
        case ompt_mutex_ordered:
          snprintf(acquiredtimerName, sizeof(acquiredtimerName),  "OpenMP_Mutex_Acquired_Ordered %s", resolved_address);
          break;
      }

      // Stop lock-wait timer
      Tau_global_stop();

      // Start lock timer
      TAU_PROFILER_CREATE(mutex_acquired_handle, acquiredtimerName, " ", TAU_OPENMP);
      TAU_PROFILER_START(mutex_acquired_handle);
    }
  }

  if(Tau_plugins_enabled.ompt_mutex_acquired) {
    Tau_plugin_event_ompt_mutex_acquired_data_t plugin_data;

    plugin_data.kind = kind;
    plugin_data.wait_id = wait_id;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRED, "*", &plugin_data);
  }
}

static void
on_ompt_callback_mutex_released(
    ompt_mutex_t kind,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_mutex_released] && Tau_init_check_initialized()) {
    if(codeptr_ra) {
      Tau_global_stop();
    }
  }

  if(Tau_plugins_enabled.ompt_mutex_released) {
    Tau_plugin_event_ompt_mutex_released_data_t plugin_data;

    plugin_data.kind = kind;
    plugin_data.wait_id = wait_id;
    plugin_data.codeptr_ra = codeptr_ra;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_MUTEX_RELEASED, "*", &plugin_data);
  }
}

// The device init callback must obtain the handles to the tracing
// entry points, if required.
static void on_ompt_callback_device_initialize (
  int device_num,
  const char *type,
  ompt_device_t *device,
  ompt_function_lookup_t lookup,
  const char *documentation
 ) {
    TauInternalFunctionGuard protects_this_function;
 /*
  printf("Init: device_num=%d type=%s device=%p lookup=%p doc=%p\n",
	 device_num, type, device, lookup, documentation);
     */
  if (!lookup) {
    printf("Trace collection disabled on device %d\n", device_num);
    return;
  }
  getDeviceMap().insert(std::pair<int,ompt_device_t*>(device_num, device));

#ifdef TAU_OMPT_USE_TARGET_OFFLOAD
  ompt_set_trace_ompt = (ompt_set_trace_ompt_t) lookup("ompt_set_trace_ompt");
  ompt_start_trace = (ompt_start_trace_t) lookup("ompt_start_trace");
  ompt_flush_trace = (ompt_flush_trace_t) lookup("ompt_flush_trace");
  ompt_stop_trace = (ompt_stop_trace_t) lookup("ompt_stop_trace");
  ompt_get_record_ompt = (ompt_get_record_ompt_t) lookup("ompt_get_record_ompt");
  ompt_advance_buffer_cursor = (ompt_advance_buffer_cursor_t) lookup("ompt_advance_buffer_cursor");
  ompt_translate_time = (ompt_translate_time_t) lookup("ompt_translate_time");

  // In many scenarios, this will be a good place to start the
  // trace. If start_trace is called from the main program before this
  // callback is dispatched, the start_trace handle will be null. This
  // is because this device_init callback is invoked during the first
  // target construct implementation.

  Tau_ompt_set_trace();
  Tau_ompt_start_trace();
#endif
}

// Called at device finalize
static void on_ompt_callback_device_finalize ( int device_num) {
    TauInternalFunctionGuard protects_this_function;
  //printf("Callback Fini: device_num=%d\n", device_num);
#ifdef TAU_OMPT_USE_TARGET_OFFLOAD
  Tau_ompt_flush_trace();
  Tau_ompt_stop_trace();
#endif
}

// Called at device load time
static void on_ompt_callback_device_load
    (
     int device_num,
     const char *filename,
     int64_t offset_in_file,
     void *vma_in_file,
     size_t bytes,
     void *host_addr,
     void *device_addr,
     uint64_t module_id
     ) {
    TauInternalFunctionGuard protects_this_function;
     /*
  printf("Load: device_num:%d filename:%s host_adddr:%p device_addr:%p bytes:%lu\n",
	 device_num, filename, host_addr, device_addr, bytes);
     */
}

/* TODO: These target callbacks strangely don't
 * seem to be called when registered by TAU, but
 * are called when registered by another tool. I
 * did not have the time to figure out why. */
static void on_ompt_callback_target(
    ompt_target_t kind,
    ompt_scope_endpoint_t endpoint,
    int device_num,
    ompt_data_t *task_data,
    ompt_id_t target_id,
    const void *codeptr_ra)
{
  /* Per-thread timer stack for this region type.
     this is necessary because we can't rely on the runtime
     to not overwrite the task_data pointer :( */
    thread_local static std::stack<void*> timer_stack;
    assert(codeptr_ra != 0);
    /*
    printf("Callback Target: target_id=%lu kind=%d endpoint=%d device_num=%d code=%p\n",
	    target_id, kind, endpoint, device_num, codeptr_ra);
    */
    TauInternalFunctionGuard protects_this_function;
    //printf("CPU Device: %d\n", device_num);
    void *handle = NULL;
    switch(endpoint) {
        case ompt_scope_begin: {
            char timerName[10240];
            char resolved_address[1024];
            void * codeptr_ra_copy = (void*) codeptr_ra;
            unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      /*Resolve addresses at runtime in case the user really wants to pay the price of doing so.
       *Enabling eager resolving of addresses is only useful in situations where
       *OpenMP routines are instrumented in shared libraries that get unloaded
       *before TAU has a chance to resolve addresses*/
            if (TauEnv_get_ompt_resolve_address_eagerly()) {
                Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
                snprintf(timerName, sizeof(timerName),  "OpenMP_Target %s", resolved_address);
            } else {
                snprintf(timerName, sizeof(timerName),  "OpenMP_Target ADDR <%lx>", addr);
            }

            TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
            timer_stack.push(handle);
            TAU_PROFILER_START(handle);
            TargetMap::instance().add_thread_id(target_id,
                (device_num == -1 ? 0 : device_num));
            break;
        }
        case ompt_scope_end: {
            handle = timer_stack.top();
            stop_correct_timer(handle);
            timer_stack.pop();
#ifndef TAU_INTEL_COMPILER // intel doesn't always provide a codeptr_ra value.
            // flush the trace to get async events for this target
            Tau_ompt_flush_trace();
#endif
            break;
        }
#if defined(ompt_scope_beginend)
        case ompt_scope_beginend:
#endif
        default: {
            // This indicates a coincident beginning and end of scope. Do nothing?
            break;
        }
    }

    if(Tau_plugins_enabled.ompt_target) {
        Tau_plugin_event_ompt_target_data_t plugin_data;

        plugin_data.kind = kind;
        plugin_data.endpoint = endpoint;
        plugin_data.device_num = device_num;
        plugin_data.task_data = task_data;
        plugin_data.target_id = target_id;
        plugin_data.codeptr_ra = codeptr_ra;

        Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_TARGET, "*", &plugin_data);
    }
}

static void
on_ompt_callback_target_data_op(
        ompt_id_t target_id,
        ompt_id_t host_op_id,
        ompt_target_data_op_t optype,
        void *src_addr,
        int src_device_num,
        void *dest_addr,
        int dest_device_num,
        size_t bytes,
        const void *codeptr_ra)
{
#ifndef TAU_INTEL_COMPILER // intel doesn't always provide a codeptr_ra value.
    assert(codeptr_ra != 0);
#endif
    // Both src and dest must not be null
    assert(src_addr != 0 || dest_addr != 0);
    /*
    printf("  Callback DataOp: target_id=%lu host_op_id=%lu optype=%d src=%p src_device_num=%d "
	    "dest=%p dest_device_num=%d bytes=%lu code=%p\n",
	    target_id, host_op_id, optype, src_addr, src_device_num,
	    dest_addr, dest_device_num, bytes, codeptr_ra);
    */
    TauInternalFunctionGuard protects_this_function;
    // printf("CPU Device: %d, %d\n", src_device_num, dest_device_num);

    static std::unordered_map<void*,double> allocations;
    static std::mutex allocation_lock;
    void * ue = nullptr;
    double d_bytes = (double)bytes;
    std::stringstream ss;
    // get the address and save the bytes transferred
    switch (optype) {
        case ompt_target_data_alloc: {
            static const char * _name = "OpenMP Target Data Alloc Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            std::unique_lock<std::mutex> l(allocation_lock);
            allocations[src_addr] = (double)bytes;
            break;
        }
        case ompt_target_data_transfer_to_device: {
            static const char * _name ="OpenMP Target Data Transfer to Device Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            std::unique_lock<std::mutex> l(allocation_lock);
            allocations[dest_addr] = (double)bytes;
            break;
        }
        case ompt_target_data_transfer_from_device: {
            static const char * _name ="OpenMP Target Data Transfer from Device Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            std::unique_lock<std::mutex> l(allocation_lock);
            allocations[dest_addr] = (double)bytes;
            break;
        }
        case ompt_target_data_delete: {
            {
                std::unique_lock<std::mutex> l(allocation_lock);
                d_bytes = allocations[src_addr];
                allocations.erase(src_addr);
            }
            static const char * _name ="OpenMP Target Data Delete Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            break;
        }
        case ompt_target_data_associate: {
            static const char * _name ="OpenMP Target Data Associate Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            break;
        }
        case ompt_target_data_disassociate: {
            static const char * _name ="OpenMP Target Data Disassociate Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            break;
        }
#ifdef TAU_OMPT_USE_TARGET_OFFLOAD
        case ompt_target_data_alloc_async: {
            static const char * _name ="OpenMP Target Data Alloc Async Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            std::unique_lock<std::mutex> l(allocation_lock);
            allocations[src_addr] = (double)bytes;
            break;
        }
        case ompt_target_data_transfer_to_device_async: {
            static const char * _name ="OpenMP Target Data Transfer to Device Async Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            std::unique_lock<std::mutex> l(allocation_lock);
            allocations[dest_addr] = (double)bytes;
            break;
        }
        case ompt_target_data_transfer_from_device_async: {
            static const char * _name ="OpenMP Target Data Transfer from Device Async Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            std::unique_lock<std::mutex> l(allocation_lock);
            allocations[dest_addr] = (double)bytes;
            break;
        }
        case ompt_target_data_delete_async: {
            {
                std::unique_lock<std::mutex> l(allocation_lock);
                d_bytes = allocations[src_addr];
                allocations.erase(src_addr);
            }
            static const char * _name ="OpenMP Target Data Delete Async Bytes";
            static void * _ue = Tau_get_userevent(_name);
            ue = _ue;
            ss << _name;
            break;
        }
#endif
        default:
            break;
    }

    if (ue != nullptr) {
        Tau_userevent(ue,d_bytes);
        // create a target-specific counter, too
#ifndef TAU_INTEL_COMPILER // intel doesn't always provide a codeptr_ra value.
        if (TauEnv_get_ompt_resolve_address_eagerly()) {
            char resolved_address[1024];
            void * codeptr_ra_copy = (void*) codeptr_ra;
            unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
            Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
            ss << " : " << resolved_address;
        } else {
            ss << " : ADDR <0x" << codeptr_ra << ">";
        }
#else
            ss << " : ADDR <0x" << codeptr_ra << ">";
#endif
        void * ue2 = Tau_get_userevent(ss.str().c_str());
        Tau_userevent(ue2,d_bytes);
    }

    if(Tau_plugins_enabled.ompt_target_data_op) {
        Tau_plugin_event_ompt_target_data_op_data_t plugin_data;

        plugin_data.target_id = target_id;
        plugin_data.host_op_id = host_op_id;
        plugin_data.optype = optype;
        plugin_data.src_addr = src_addr;
        plugin_data.src_device_num = src_device_num;
        plugin_data.dest_addr = dest_addr;
        plugin_data.dest_device_num = dest_device_num;
        plugin_data.bytes = bytes;
        plugin_data.codeptr_ra = codeptr_ra;

        Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_TARGET_DATA_OP, "*", &plugin_data);
    }
}

static void
on_ompt_callback_target_submit(
        ompt_id_t target_id,
        ompt_id_t host_op_id,
        unsigned int requested_num_teams)
{
    TauInternalFunctionGuard protects_this_function;

    static void * ue = Tau_get_userevent("OpenMP Target Submit Num Teams");
    Tau_userevent(ue,(double)(requested_num_teams));

    if(Tau_plugins_enabled.ompt_target_submit) {
        Tau_plugin_event_ompt_target_submit_data_t plugin_data;

        plugin_data.target_id = target_id;
        plugin_data.host_op_id = host_op_id;
        plugin_data.requested_num_teams = requested_num_teams;

        Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_TARGET_SUBMIT, "*", &plugin_data);
    }
}


/* Register callbacks. This function is invoked only from the ompt_start_tool
 * routine and the Tau_ompt_register_plugin_callbacks routine.
 * Callbacks that only have "ompt_set_always" are the required events that we HAVE to support */
inline static int register_callback(ompt_callbacks_t name, ompt_callback_t cb) {
  int ret = ompt_set_callback(name, cb);

  switch(ret) {
    case ompt_set_never:
      TAU_VERBOSE("TAU: WARNING: OMPT Callback for event %d could not be registered\n", name);
      break;
    case ompt_set_sometimes:
      TAU_VERBOSE("TAU: OMPT Callback for event %d registered with return value %s\n", name, "ompt_set_sometimes");
      break;
    case ompt_set_sometimes_paired:
      TAU_VERBOSE("TAU: OMPT Callback for event %d registered with return value %s\n", name, "ompt_set_sometimes_paired");
      break;
    case ompt_set_always:
      TAU_VERBOSE("TAU: OMPT Callback for event %d registered with return value %s\n", name, "ompt_set_always");
      break;
  }
  return ret;
}

/* Call the register_callback routine and set a flag indicating that TAU
 * requested this callback, and not just a plugin */
inline static void Tau_register_callback(ompt_callbacks_t name, ompt_callback_t cb) {
  int ret = register_callback(name, cb);

  if(ret != ompt_set_never)
    Tau_ompt_callbacks_enabled[name] = 1;
}


/* HACK ALERT: This function is only there to ensure that OMPT environment variables are initialized before the ompt_start_tool is invoked
 * We need this because we would need to register callbacks depending on the level of support that the user desires
 * Not sure how we else we can ensure initialization of OMPT environment variables.
 * TODO: Remove any function calls forcing initialization of OMPT variables by figuring out the root cause of the problem*/
/*********************************************************************
 * Parse a boolean value
 ********************************************************************/
static int parse_bool(const char *str, int default_value = 0) {

  if (str == NULL) {
    return default_value;
  }
  static char strbuf[128];
  char *ptr = strbuf;
  strncpy(strbuf, str, 128);
  while (*ptr) {
    *ptr = tolower(*ptr);
    ptr++;
  }
  if (strcmp(strbuf, "yes") == 0  ||
      strcmp(strbuf, "true") == 0 ||
      strcmp(strbuf, "on") == 0 ||
      strcmp(strbuf, "1") == 0) {
    return 1;
  } else {
    return 0;
  }
}

const char *getconf(const char *key);
void Tau_force_ompt_env_initialization() {

    TAU_VERBOSE("Inside Tau_force_ompt_env_initialization\n");
    const char* tmp = getconf("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY");

    if (parse_bool(tmp, 1)) {
      TauEnv_set_ompt_resolve_address_eagerly(1);
      TAU_VERBOSE("TAU: OMPT resolving addresses eagerly Enabled\n");
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "on");
      TAU_VERBOSE("TAU: Resolving OMPT addresses eagerly\n");
    } else {
      TauEnv_set_ompt_resolve_address_eagerly(0);
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "off");
      TAU_VERBOSE("TAU: NOT Resolving OMPT addresses eagerly\n");
    }

    TauEnv_set_ompt_support_level(0); // basic OMPT support is the default
    const char *omptSupportLevel = getconf("TAU_OMPT_SUPPORT_LEVEL");
    if (omptSupportLevel != NULL && 0 == strcasecmp(omptSupportLevel, "basic")) {
      TauEnv_set_ompt_support_level(0);
      TAU_VERBOSE("TAU: OMPT support will be basic - only required events supported\n");
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "basic");
    } else if (omptSupportLevel != NULL && 0 == strcasecmp(omptSupportLevel, "lowoverhead")) {
      TauEnv_set_ompt_support_level(1);
      TAU_VERBOSE("TAU: OMPT support will be for all required events along with optional low overhead events\n");
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "lowoverhead");
    } else if (omptSupportLevel != NULL && 0 == strcasecmp(omptSupportLevel, "full")) {
      TauEnv_set_ompt_support_level(2);
      TAU_VERBOSE("TAU: OMPT support will be full - all events will be supported\n");
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "full");
    } else {
      TAU_METADATA("TAU_OMPT_SUPPORT_LEVEL", "basic");
      TAU_VERBOSE("TAU: OMPT support will be basic - TAU_OMPT_SUPPORT_LEVEL runtime variable is not set");
    }
}

#define cb_t(name) (ompt_callback_t)&name

void * after_ompt_init_thread_fn(void * unused) {
    (void)unused;
    internal_thread = true;
    // The OpenMP runtime is initialized now, so we can fill OpenMP-runtime-related metadata
    Tau_metadata_fillOpenMPMetaData();
    return NULL;
}

/* Register callbacks for all events that we are interested in / have to support */
extern "C" int ompt_initialize(
  ompt_function_lookup_t lookup,
#if  defined (TAU_USE_OMPT_5_0)
  int initial_device_num,
#endif /* defined (TAU_USE_OMPT_5_0) */
  ompt_data_t* tool_data)
{
  Tau_init_initializeTAU();
  if (initialized || initializing) return 0;
  initializing = true;
  TauInternalFunctionGuard protects_this_function;
  if (!TauEnv_get_openmp_runtime_enabled()) return 0;
#ifndef TAU_MPI
  TAU_PROFILE_SET_NODE(0);
#endif
  Tau_create_top_level_timer_if_necessary();
  is_master = true;

  /* Srinivasan here: This is BAD idea. But we NEED to ensure that the OMPT env
   * variables are initialized correctly before registering callbacks */
  Tau_force_ompt_env_initialization();

/* Gather the required function pointers using the lookup tool */
  TAU_VERBOSE("Registering OMPT events...\n"); fflush(stderr);
  ompt_set_callback = (ompt_set_callback_t) lookup("ompt_set_callback");
  ompt_get_task_info = (ompt_get_task_info_t) lookup("ompt_get_task_info");
  ompt_get_thread_data = (ompt_get_thread_data_t) lookup("ompt_get_thread_data");
  ompt_get_parallel_info = (ompt_get_parallel_info_t) lookup("ompt_get_parallel_info");
  ompt_get_unique_id = (ompt_get_unique_id_t) lookup("ompt_get_unique_id");

  ompt_get_num_places = (ompt_get_num_places_t) lookup("ompt_get_num_places");
  ompt_get_place_proc_ids = (ompt_get_place_proc_ids_t) lookup("ompt_get_place_proc_ids");
  ompt_get_place_num = (ompt_get_place_num_t) lookup("ompt_get_place_num");
  ompt_get_partition_place_nums = (ompt_get_partition_place_nums_t) lookup("ompt_get_partition_place_nums");
  ompt_get_proc_id = (ompt_get_proc_id_t) lookup("ompt_get_proc_id");
  ompt_enumerate_states = (ompt_enumerate_states_t) lookup("ompt_enumerate_states");
  ompt_enumerate_mutex_impls = (ompt_enumerate_mutex_impls_t) lookup("ompt_enumerate_mutex_impls");
  ompt_finalize_tool = (ompt_finalize_tool_t) lookup("ompt_finalize_tool");

/* Required events */
  Tau_register_callback(ompt_callback_parallel_begin, cb_t(on_ompt_callback_parallel_begin));
  Tau_register_callback(ompt_callback_parallel_end, cb_t(on_ompt_callback_parallel_end));
  Tau_register_callback(ompt_callback_task_create, cb_t(on_ompt_callback_task_create));
  Tau_register_callback(ompt_callback_task_schedule, cb_t(on_ompt_callback_task_schedule));
  /* Intel doesn't provide the exit callback for implicit tasks... */
#if !defined(TAU_INTEL_COMPILER) && !defined(__ICC) && !defined(__clang__)
  //Tau_register_callback(ompt_callback_implicit_task, cb_t(on_ompt_callback_implicit_task)); //Sometimes high-overhead, but unfortunately we cannot avoid this as it is a required event
#endif
  Tau_register_callback(ompt_callback_thread_begin, cb_t(on_ompt_callback_thread_begin));
  Tau_register_callback(ompt_callback_thread_end, cb_t(on_ompt_callback_thread_end));

/* Target Events */
  Tau_register_callback(ompt_callback_device_initialize, cb_t(on_ompt_callback_device_initialize));
  Tau_register_callback(ompt_callback_device_finalize, cb_t(on_ompt_callback_device_finalize));
  Tau_register_callback(ompt_callback_device_load, cb_t(on_ompt_callback_device_load));
  Tau_register_callback(ompt_callback_target, cb_t(on_ompt_callback_target));
  Tau_register_callback(ompt_callback_target_data_op, cb_t(on_ompt_callback_target_data_op));
  Tau_register_callback(ompt_callback_target_submit, cb_t(on_ompt_callback_target_submit));

/* Optional events */

  if(TauEnv_get_ompt_support_level() >= 1) { /* Only support this when "lowoverhead" mode is enabled. Turns on all required events + other low overhead */
#ifndef __NVCOMPILER //NVIDIA does not provide endpoints for work callbacks as of 22.9
    Tau_register_callback(ompt_callback_work, cb_t(on_ompt_callback_work));
#endif
#if _OPENMP < 202011 && defined(ompt_callback_master)
    Tau_register_callback(ompt_callback_master, cb_t(on_ompt_callback_master));
#endif
  }

  if(TauEnv_get_ompt_support_level() == 2) { /* Only support this when "full" is enabled. This is a high overhead call */
    Tau_register_callback(ompt_callback_sync_region, cb_t(on_ompt_callback_sync_region));
    // TODO: Overheads unclear currently. Also, causing a hang with TAU mm example (other task-based examples also lead to the application becoming
    // unresponsive possibly due to extremely high overheads*/
    /*Tau_register_callback(ompt_callback_mutex_acquire, cb_t(on_ompt_callback_mutex_acquire));
    Tau_register_callback(ompt_callback_mutex_acquired, cb_t(on_ompt_callback_mutex_acquired));
    Tau_register_callback(ompt_callback_mutex_released, cb_t(on_ompt_callback_mutex_released));*/
  }

  // Overheads unclear currently

  // We can't make OpenMP calls inside an OMPT callback, but we have to defer reading
  // OpenMP parameters for metadata purposes until after OpenMP is initialized.
  // Here we launch another thread which will collect the metadata.
  pthread_t after_init_thread;
  pthread_create(&after_init_thread, NULL, after_ompt_init_thread_fn, NULL);
  pthread_detach(after_init_thread);

  initialized = true;
  initializing = false;
  return 1; //success
}

/* Register callbacks for plugins in the case that they are not already registered for TAU */
void Tau_ompt_register_plugin_callbacks(Tau_plugin_callbacks_active_t *Tau_plugins_enabled) {
  if(!initialized)
  {
    TAU_VERBOSE("TAU: WARNING: Could not register OMPT plugin callbacks as OMPT was not initialized.\n");
    return;
  }


  if (Tau_plugins_enabled->ompt_parallel_begin > Tau_ompt_callbacks_enabled[ompt_callback_parallel_begin])
    register_callback(ompt_callback_parallel_begin, cb_t(on_ompt_callback_parallel_begin));
  if (Tau_plugins_enabled->ompt_parallel_end > Tau_ompt_callbacks_enabled[ompt_callback_parallel_end])
    register_callback(ompt_callback_parallel_end, cb_t(on_ompt_callback_parallel_end));
  if (Tau_plugins_enabled->ompt_task_create > Tau_ompt_callbacks_enabled[ompt_callback_task_create])
    register_callback(ompt_callback_task_create, cb_t(on_ompt_callback_task_create));
  if (Tau_plugins_enabled->ompt_task_schedule > Tau_ompt_callbacks_enabled[ompt_callback_task_schedule])
    register_callback(ompt_callback_task_schedule, cb_t(on_ompt_callback_task_schedule));
  if (Tau_plugins_enabled->ompt_implicit_task > Tau_ompt_callbacks_enabled[ompt_callback_implicit_task])
    register_callback(ompt_callback_implicit_task, cb_t(on_ompt_callback_implicit_task));
  if (Tau_plugins_enabled->ompt_thread_begin > Tau_ompt_callbacks_enabled[ompt_callback_thread_begin])
    register_callback(ompt_callback_thread_begin, cb_t(on_ompt_callback_thread_begin));
  if (Tau_plugins_enabled->ompt_thread_end > Tau_ompt_callbacks_enabled[ompt_callback_thread_end])
    register_callback(ompt_callback_thread_end, cb_t(on_ompt_callback_thread_end));
  if (Tau_plugins_enabled->ompt_work > Tau_ompt_callbacks_enabled[ompt_callback_work])
    register_callback(ompt_callback_work, cb_t(on_ompt_callback_work));
#if _OPENMP < 202011 && defined(ompt_callback_master)
  if (Tau_plugins_enabled->ompt_master > Tau_ompt_callbacks_enabled[ompt_callback_master])
    register_callback(ompt_callback_master, cb_t(on_ompt_callback_master));
#endif
  if (Tau_plugins_enabled->ompt_sync_region > Tau_ompt_callbacks_enabled[ompt_callback_sync_region])
    register_callback(ompt_callback_sync_region, cb_t(on_ompt_callback_sync_region));
  if (Tau_plugins_enabled->ompt_mutex_acquire > Tau_ompt_callbacks_enabled[ompt_callback_mutex_acquire])
    register_callback(ompt_callback_mutex_acquire, cb_t(on_ompt_callback_mutex_acquire));
  if (Tau_plugins_enabled->ompt_mutex_acquired > Tau_ompt_callbacks_enabled[ompt_callback_mutex_acquired])
    register_callback(ompt_callback_mutex_acquired, cb_t(on_ompt_callback_mutex_acquired));
  if (Tau_plugins_enabled->ompt_mutex_released > Tau_ompt_callbacks_enabled[ompt_callback_mutex_released])
    register_callback(ompt_callback_mutex_released, cb_t(on_ompt_callback_mutex_released));
#if 0
  if (Tau_plugins_enabled->ompt_device_initialize > Tau_ompt_callbacks_enabled[ompt_callback_device_initialize])
    register_callback(ompt_callback_device_initialize, cb_t(on_ompt_callback_device_initialize));
  if (Tau_plugins_enabled->ompt_device_finalize > Tau_ompt_callbacks_enabled[ompt_callback_device_finalize])
    register_callback(ompt_callback_device_finalize, cb_t(on_ompt_callback_device_finalize));
  if (Tau_plugins_enabled->ompt_device_load > Tau_ompt_callbacks_enabled[ompt_callback_device_load])
    register_callback(ompt_callback_device_load, cb_t(on_ompt_callback_device_load));
#endif
  if (Tau_plugins_enabled->ompt_target > Tau_ompt_callbacks_enabled[ompt_callback_target])
    register_callback(ompt_callback_target, cb_t(on_ompt_callback_target));
  if (Tau_plugins_enabled->ompt_target_data_op > Tau_ompt_callbacks_enabled[ompt_callback_target_data_op])
    register_callback(ompt_callback_target_data_op, cb_t(on_ompt_callback_target_data_op));
  if (Tau_plugins_enabled->ompt_target_submit > Tau_ompt_callbacks_enabled[ompt_callback_target_submit])
    register_callback(ompt_callback_target_submit, cb_t(on_ompt_callback_target_submit));
}

/* I don't know why we have OMPT trigger events, but whatever... */
void Tau_ompt_finalize_trigger() {
    // make sure this only happens once.
    static bool once{false};
    if (once) return;
    once = true;
    if(Tau_plugins_enabled.ompt_finalize) {
        Tau_plugin_event_ompt_finalize_data_t plugin_data;
        plugin_data.null = 0;
        Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_FINALIZE, "*", &plugin_data);
    }
}

/* This is called by the Tau_destructor_trigger() to prevent
 * callbacks from happening after TAU is shut down */
void Tau_ompt_finalize(void) {
  //printf("Callback %s\n", __func__);
    if(Tau_ompt_finalized()) { return; }
    Tau_ompt_finalized(true);
#ifdef TAU_OMPT_USE_TARGET_OFFLOAD
    Tau_ompt_flush_trace();
#endif
    //Tau_ompt_stop_trace();
    if (TauEnv_get_ompt_force_finalize()) {
        if (ompt_finalize_tool != nullptr) {
            TAU_VERBOSE("Asking the OpenMP runtime to shut down...\n");
            ompt_finalize_tool();
        }
        else {
            //Tau_ompt_finalize_trigger();
        }
    }
}

/* This callback should come from the runtime when the runtime is shut down */
extern "C" void ompt_finalize(ompt_data_t* tool_data)
{
  TAU_VERBOSE("OpenMP runtime is shutting down...\n");
  /* Just in case... */
  Tau_destructor_trigger();
  Tau_ompt_finalize_trigger();
}

extern "C" ompt_start_tool_result_t * ompt_start_tool(
  unsigned int omp_version,
  const char *runtime_version)
{
  static ompt_start_tool_result_t result;
  result.initialize = &ompt_initialize;
  result.finalize = &ompt_finalize;
  result.tool_data.value = 0L;
  result.tool_data.ptr = NULL;
#ifdef _OPENMP
  TAU_VERBOSE("OMPT support configuring, TAU compiled with version %d, running with version %d\n", _OPENMP, omp_version);
#endif
  return &result;
}
#else /*  defined (TAU_USE_OMPT_5_0) */
#include <Profile/TauPluginInternals.h>

void Tau_ompt_register_plugin_callbacks(Tau_plugin_callbacks_active_t *Tau_plugins_enabled) {
  return;
}
#endif /*  defined (TAU_USE_OMPT_5_0) */
