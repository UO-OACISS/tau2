#define _BSD_SOURCE

#if defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)

#include <stdio.h>
#include <sstream>
#include <inttypes.h>
#include <omp.h>
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
#include <omp-tools.h>
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
#if defined (TAU_USE_OMPT_TR6)
#include <ompt.h>
#endif /* TAU_USE_OMPT_TR6 */

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

static bool initializing = false;
static bool initialized = false;
static bool tau_initialized = false;
#if defined (TAU_USE_TLS)
__thread bool is_master = false;
#elif defined (TAU_USE_DTLS)
__declspec(thread) bool is_master = false;
#elif defined (TAU_USE_PGS)
#include "pthread.h"
pthread_key_t thr_id_key;
#endif

/* Using typedefs here to avoid having too many #ifdef this file */
#ifdef TAU_USE_OMPT_TR7
typedef omp_frame_t ompt_frame_t;
typedef omp_wait_id_t ompt_wait_id_t;
#endif /* TAU_USE_OMPT_TR7 */

#ifdef TAU_USE_OMPT_TR6
typedef ompt_thread_type_t ompt_thread_t;
/* This should be un-commented for TR6 but needs to be commented for the TR6
 * lib that TAU downloads, and the TR6 support of llvm 7.0.1 */
/* typedef omp_frame_t ompt_frame_t; */
/* typedef omp_wait_id_t ompt_wait_id_t; */
typedef ompt_sync_region_kind_t ompt_sync_region_t;
typedef ompt_mutex_kind_t ompt_mutex_t;
typedef ompt_work_type_t ompt_work_t;
#endif /*TAU_USE_OMPT_TR6 */

int get_ompt_tid(void) {
#if defined (TAU_USE_TLS)
  if (is_master) return 0;
#elif defined (TAU_USE_DTLS)
  if (is_master) return 0;
#elif defined (TAU_USE_PGS)
  if (pthread_getspecific(thr_id_key) != NULL) return 0;
#endif
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

/* IMPT NOTE: In general, we use Tau_global_stop() instead of TAU_PROFILER_STOP(handle) because it is
 * not possible to determine in advance which optional events are supported by the compiler used
 * to compile the application.
 * In general, we have noticed that some compilers generate the START* events but not the corresponding
 * STOP* optional events. As a result, it is impossible to know the right timer to stop.
 * We do NOT want to be adding ugly ifdef's in our code. Thus, the only solution is to hand the
 * responsbility to TAU with a certain loss in accuracy in measured values*/

/*Externs*/
extern "C" char* Tau_ompt_resolve_callsite_eagerly(unsigned long addr, char * resolved_address);

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
#if defined (TAU_USE_OMPT_TR6)
  ompt_invoker_t invoker,
#endif /* TAU_USE_OMPT_TR6 */
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
  int flags,
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
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
        sprintf(timerName, "OpenMP_Parallel_Region %s", resolved_address);
      } else {
        sprintf(timerName, "OpenMP_Parallel_Region ADDR <%lx>", addr);
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
#if defined (TAU_USE_OMPT_TR6)
    plugin_data.invoker = invoker;
#endif /* TAU_USE_OMPT_TR6 */
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
    plugin_data.flags = flags;
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
    plugin_data.codeptr_ra = codeptr_ra;
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_PARALLEL_BEGIN, "*", &plugin_data);
  }
}

/* TODO: Remove this and Remove changes to TauEnv.cpp once Intel and LLVM runtime stop causing a deadlock on initialisazion */
static void tau_fix_initialize()
{
    char tmpstr[512];
    int value = omp_get_max_threads();
    sprintf(tmpstr,"%d",value);
    TAU_METADATA("OMP_MAX_THREADS", tmpstr);

    value = omp_get_num_procs();
    sprintf(tmpstr,"%d",value);
    TAU_METADATA("OMP_NUM_PROCS", tmpstr);
}

static void
on_ompt_callback_parallel_end(
  ompt_data_t *parallel_data,
  ompt_data_t *parent_task_data,
#if defined (TAU_USE_OMPT_TR6)
  ompt_invoker_t invoker,
#endif /* TAU_USE_OMPT_TR6 */
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
  int flags,
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
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
      //TAU_PROFILER_STOP(parallel_data->ptr);
      Tau_global_stop();
    }
  }

  if(Tau_plugins_enabled.ompt_parallel_end) {
    Tau_plugin_event_ompt_parallel_end_data_t plugin_data;

    plugin_data.parallel_data = parallel_data;
    plugin_data.encountering_task_data = parent_task_data;
#if defined (TAU_USE_OMPT_TR6)
    plugin_data.invoker = invoker;
#endif /* TAU_USE_OMPT_TR6 */
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
    plugin_data.flags = flags;
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
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
    char contextEventName[2058];
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
        sprintf(timerName, "OpenMP_Task %s", resolved_address);
      } else {
        sprintf(timerName, "OpenMP_Task ADDR <%lx>", addr);
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
  if(Tau_ompt_callbacks_enabled[ompt_callback_task_schedule] && Tau_init_check_initialized()) {
    if(prior_task_data->ptr) {
      //TAU_PROFILER_STOP(prior_task_data->ptr);
      Tau_global_stop();
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
            sprintf(timerName, "OpenMP_Master %s", resolved_address);
          } else {
            sprintf(timerName, "OpenMP_Master ADDR <%lx>", addr);
          }

          TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
          task_data->ptr = (void*)handle;
          TAU_PROFILER_START(handle);
          break;
        case ompt_scope_end:
          //TAU_PROFILER_STOP(task_data->ptr);
          Tau_global_stop();
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
  if(Tau_ompt_callbacks_enabled[ompt_callback_work] && Tau_init_check_initialized()) {
    void *handle = NULL;
    char timerName[10240];
    char resolved_address[1024];
    if(codeptr_ra) {

      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
        switch(wstype)
        {
          case ompt_work_loop:
            sprintf(timerName, "OpenMP_Work_Loop %s", resolved_address);
            break;
          case ompt_work_sections:
            sprintf(timerName, "OpenMP_Work_Sections %s", resolved_address);
            break;
          case ompt_work_single_executor:
            sprintf(timerName, "OpenMP_Work_Single_Executor %s", resolved_address);
            break; /* WARNING: The ompt_scope_begin for this work type is triggered, but the corresponding ompt_scope_end is not triggered when using GNU to compile the tool code*/
          case ompt_work_single_other:
            sprintf(timerName, "OpenMP_Work_Single_Other %s", resolved_address);
            break;
          case ompt_work_workshare:
            sprintf(timerName, "OpenMP_Work_Workshare %s", resolved_address);
            break;
          case ompt_work_distribute:
            sprintf(timerName, "OpenMP_Work_Distribute %s", resolved_address);
            break;
          case ompt_work_taskloop:
            sprintf(timerName, "OpenMP_Work_Taskloop %s", resolved_address);
            break;
        }
      } else {
        switch(wstype)
        {
          case ompt_work_loop:
            sprintf(timerName, "OpenMP_Work_Loop ADDR <%lx>", addr);
            break;
          case ompt_work_sections:
            sprintf(timerName, "OpenMP_Work_Sections ADDR <%lx>", addr);
            break;
          case ompt_work_single_executor:
            sprintf(timerName, "OpenMP_Work_Single_Executor ADDR <%lx>", addr);
            break; /* The ompt_scope_begin for this work type is triggered, but the corresponding ompt_scope_end is not triggered when using GNU to compile the tool code*/
          case ompt_work_single_other:
            sprintf(timerName, "OpenMP_Work_Single_Other ADDR <%lx>", addr);
            break;
          case ompt_work_workshare:
            sprintf(timerName, "OpenMP_Work_Workshare ADDR <%lx>", addr);
            break;
          case ompt_work_distribute:
            sprintf(timerName, "OpenMP_Work_Distribute ADDR <%lx>", addr);
            break;
          case ompt_work_taskloop:
            sprintf(timerName, "OpenMP_Work_Taskloop ADDR <%lx>", addr);
            break;
        }
      }
      switch(endpoint)
      {
        case ompt_scope_begin:
          TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
          TAU_PROFILER_START(handle);
          break;
        case ompt_scope_end:
          Tau_global_stop();
          //TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
          //TAU_PROFILER_STOP(handle);
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
  if(Tau_ompt_callbacks_enabled[ompt_callback_thread_begin] && Tau_init_check_initialized()) {
#if defined (TAU_USE_TLS)
    if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_DTLS)
    if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_PGS)
    if (pthread_getspecific(thr_id_key) != NULL) return; // master thread can't be a new worker.
#endif
    RtsLayer::RegisterThread();
    void *handle = NULL;
    char timerName[100];
    sprintf(timerName, "OpenMP_Thread_Type_%s", ompt_thread_type_t_values[thread_type]);
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
  // Prevent against callbacks after finalization
  if (Tau_ompt_finalized()) { return; }
  if(Tau_ompt_callbacks_enabled[ompt_callback_thread_end] && Tau_init_check_initialized()) {
#if defined (TAU_USE_TLS)
    if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_DTLS)
    if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_PGS)
    if (pthread_getspecific(thr_id_key) != NULL) return; // master thread can't be a new worker.
#endif
    //TAU_PROFILER_STOP(thread_data->ptr);
    Tau_global_stop();
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
  // protect against calls after finalization
  if(Tau_ompt_finalized()) { return; }
  if(Tau_ompt_callbacks_enabled[ompt_callback_implicit_task] && Tau_init_check_initialized()) {
    char timerName[100];
    sprintf(timerName, "OpenMP_Implicit_Task");
    void *handle = NULL;


    switch(endpoint)
    {
      case ompt_scope_begin:
        TAU_PROFILER_CREATE(handle, timerName, "", TAU_OPENMP);
        TAU_PROFILER_START(handle);
        task_data->ptr = (void*)handle;
        break;
      case ompt_scope_end:
        if(task_data->ptr != NULL) {
          //TAU_PROFILER_STOP(task_data->ptr);
          Tau_global_stop();
        }
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
  if(Tau_ompt_callbacks_enabled[ompt_callback_sync_region] && Tau_init_check_initialized()) {
    void *handle = NULL;
    char timerName[10240];
    char resolved_address[1024];

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
        switch(kind)
        {
          case ompt_sync_region_barrier:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier %s", resolved_address);
            break;
          case ompt_sync_region_taskwait:
            sprintf(timerName, "OpenMP_Sync_Region_Taskwait %s", resolved_address);
            break;
          case ompt_sync_region_taskgroup:
            sprintf(timerName, "OpenMP_Sync_Region_Taskgroup %s", resolved_address);
            break;
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
          case ompt_sync_region_barrier_implicit:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier_Implicit %s", resolved_address);
            break;
          case ompt_sync_region_barrier_explicit:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier_Explicit %s", resolved_address);
            break;
          case ompt_sync_region_barrier_implementation:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier_Implementation %s", resolved_address);
            break;
          case ompt_sync_region_reduction:
            sprintf(timerName, "OpenMP_Sync_Region_Reduction %s", resolved_address);
            break;
#endif /* defined (TAU_USE_OMPt_TR7) || defined (TAU_USE_OMPT_5_0) */
        }
      } else {
        switch(kind)
        {
          case ompt_sync_region_barrier:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier ADDR <%lx>", addr);
            break;
          case ompt_sync_region_taskwait:
            sprintf(timerName, "OpenMP_Sync_Region_Taskwait ADDR <%lx>", addr);
            break;
          case ompt_sync_region_taskgroup:
            sprintf(timerName, "OpenMP_Sync_Region_Taskgroup ADDR <%lx>", addr);
            break;
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
          case ompt_sync_region_barrier_implicit:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier_Implicit ADDR <%lx>", addr);
            break;
          case ompt_sync_region_barrier_explicit:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier_Explicit ADDR <%lx>", addr);
            break;
          case ompt_sync_region_barrier_implementation:
            sprintf(timerName, "OpenMP_Sync_Region_Barrier_Implementation ADDR <%lx>", addr);
            break;
          case ompt_sync_region_reduction:
            sprintf(timerName, "OpenMP_Sync_Region_Reduction ADDR <%lx>", addr);
            break;
#endif /* defined (TAU_USE_OMPt_TR7) || defined (TAU_USE_OMPT_5_0) */
        }
      }
      switch(endpoint)
      {
        case ompt_scope_begin:
          TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
          TAU_PROFILER_START(handle);
          break;
        case ompt_scope_end:
          //TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
          //TAU_PROFILER_STOP(task_data->ptr);
          Tau_global_stop();
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

/* Idle event - optional event that has low overhead and does not need context) */
#if defined (TAU_USE_OMPT_TR6)
static void
on_ompt_callback_idle(
    ompt_scope_endpoint_t endpoint)
{
  TauInternalFunctionGuard protects_this_function;
  if(Tau_ompt_callbacks_enabled[ompt_callback_idle] && Tau_init_check_initialized()) {
    const char *timerName= "OpenMP_Idle";

    TAU_PROFILE_TIMER(handle, timerName, " ", TAU_OPENMP);

    switch(endpoint)
    {
      case ompt_scope_begin:
        TAU_PROFILE_START(handle);
        break;
      case ompt_scope_end:
        TAU_PROFILE_STOP(handle);
        break;
    }
  }

  if(Tau_plugins_enabled.ompt_idle) {
    Tau_plugin_event_ompt_idle_data_t plugin_data;

    plugin_data.endpoint = endpoint;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_IDLE, "*", &plugin_data);
  }
}
#endif /* defined (TAU_USE_OMPT_TR6) */

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
        switch(kind)
        {
#if defined (TAU_USE_OMPT_TR6)
          case ompt_mutex:
            sprintf(timerName, "OpenMP_Mutex_Waiting %s", resolved_address);
            break;
#endif /* defined (TAU_USE_OMPT_TR6) */
          case ompt_mutex_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Lock %s", resolved_address);
            break;
          case ompt_mutex_nest_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Nest_Lock %s", resolved_address);
            break;
#if defined (TAU_USE_OMPT_5_0)
          case ompt_mutex_test_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Test_Lock %s", resolved_address);
            break;
          case ompt_mutex_test_nest_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Test_Nest_Lock %s", resolved_address);
            break;
#endif /* defined (TAU_USE_OMPT_5_0) */
          case ompt_mutex_critical:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Critical %s", resolved_address);
            break;
          case ompt_mutex_atomic:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Atomic %s", resolved_address);
            break;
          case ompt_mutex_ordered:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Ordered %s", resolved_address);
            break;
        }
      } else {
        switch(kind)
        {
#if defined (TAU_USE_OMPT_TR6)
          case ompt_mutex:
            sprintf(timerName, "OpenMP_Mutex_Waiting ADDR <%lx>", addr);
            break;
#endif /* defined (TAU_USE_OMPT_TR6) */
          case ompt_mutex_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Lock ADDR <%lx>", addr);
            break;
          case ompt_mutex_nest_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Nest_Lock ADDR <%lx>", addr);
            break;
#if defined (TAU_USE_OMPT_5_0)
          case ompt_mutex_test_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Test_Lock ADDR <%lx>", addr);
            break;
          case ompt_mutex_test_nest_lock:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Test_Nest_Lock ADDR <%lx>", addr);
            break;
#endif /* defined (TAU_USE_OMPT_5_0) */
          case ompt_mutex_critical:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Critical ADDR <%lx>", addr);
            break;
          case ompt_mutex_atomic:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Atomic ADDR <%lx>", addr);
            break;
          case ompt_mutex_ordered:
            sprintf(timerName, "OpenMP_Mutex_Waiting_Ordered ADDR <%lx>", addr);
            break;
        }
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
    char waitingtimerName[10240];
    char resolved_address[1024];
    void* mutex_acquired_handle=NULL;

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

      if(TauEnv_get_ompt_resolve_address_eagerly()) {
        Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
        switch(kind)
        {
#if defined (TAU_USE_OMPT_TR6)
          case ompt_mutex:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired %s", resolved_address);
            break;
#endif /* defined (TAU_USE_OMPT_TR6) */
          case ompt_mutex_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Lock %s", resolved_address);
            break;
          case ompt_mutex_nest_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Nest_Lock %s", resolved_address);
            break;
#if defined (TAU_USE_OMPT_5_0)
          case ompt_mutex_test_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Test_Lock %s", resolved_address);
            break;
          case ompt_mutex_test_nest_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Test_Nest_Lock %s", resolved_address);
            break;
#endif /* defined (TAU_USE_OMPT_5_0) */
          case ompt_mutex_critical:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Critical %s", resolved_address);
            break;
          case ompt_mutex_atomic:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Atomic %s", resolved_address);
            break;
          case ompt_mutex_ordered:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Ordered %s", resolved_address);
            break;
        }
      } else {
        switch(kind)
        {
#if defined (TAU_USE_OMPT_TR6)
          case ompt_mutex:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired ADDR <%lx>", addr);
            break;
#endif /* defined (TAU_USE_OMPT_TR6) */
          case ompt_mutex_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Lock ADDR <%lx>", addr);
            break;
          case ompt_mutex_nest_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Nest_Lock ADDR <%lx>", addr);
            break;
#if defined (TAU_USE_OMPT_5_0)
          case ompt_mutex_test_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Test_Lock ADDR <%lx>", addr);
            break;
          case ompt_mutex_test_nest_lock:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Test_Nest_Lock ADDR <%lx>", addr);
            break;
#endif /* defined (TAU_USE_OMPT_5_0) */
          case ompt_mutex_critical:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Critical ADDR <%lx>", addr);
            break;
          case ompt_mutex_atomic:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Atomic ADDR <%lx>", addr);
            break;
          case ompt_mutex_ordered:
            sprintf(acquiredtimerName, "OpenMP_Mutex_Acquired_Ordered ADDR <%lx>", addr);
            break;
        }
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
    char timerName[10240];
    char resolved_address[1024];

    if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      // Stop lock timer
      //TAU_PROFILER_STOP(mutex_acquired_handle);
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


/* TODO: These target callbacks strangely don't
 * seem to be called when registered by TAU, but
 * are called when registered by another tool. I
 * did not have the time to figure out why. */
static void
on_ompt_callback_target(
    ompt_target_t kind,
    ompt_scope_endpoint_t endpoint,
    int device_num,
    ompt_data_t *task_data,
    ompt_id_t target_id,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;

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
  TauInternalFunctionGuard protects_this_function;

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

/* Register callbacks for all events that we are interested in / have to support */
extern "C" int ompt_initialize(
  ompt_function_lookup_t lookup,
#if  defined (TAU_USE_OMPT_5_0)
  int initial_device_num,
#endif /* defined (TAU_USE_OMPT_5_0) */
  ompt_data_t* tool_data)
{
  int ret;
  Tau_init_initializeTAU();
  if (initialized || initializing) return 0;
  initializing = true;
  TauInternalFunctionGuard protects_this_function;
  if (!TauEnv_get_openmp_runtime_enabled()) return 0;
  if (Tau_get_node() == -1) {
      TAU_PROFILE_SET_NODE(0);
  }

#if defined (TAU_USE_TLS)
  is_master = true;
#elif defined (TAU_USE_DTLS)
  is_master = true;
#elif defined (TAU_USE_PGS)
  pthread_key_create(&thr_id_key, NULL);
  pthread_setspecific(thr_id_key, 1);
#endif

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
  Tau_register_callback(ompt_callback_implicit_task, cb_t(on_ompt_callback_implicit_task)); //Sometimes high-overhead, but unfortunately we cannot avoid this as it is a required event
  Tau_register_callback(ompt_callback_thread_begin, cb_t(on_ompt_callback_thread_begin));
  Tau_register_callback(ompt_callback_thread_end, cb_t(on_ompt_callback_thread_end));

/* Target Events */
  Tau_register_callback(ompt_callback_target, cb_t(on_ompt_callback_target));
  Tau_register_callback(ompt_callback_target_data_op, cb_t(on_ompt_callback_target_data_op));
  Tau_register_callback(ompt_callback_target_submit, cb_t(on_ompt_callback_target_submit));

/* Optional events */

  if(TauEnv_get_ompt_support_level() >= 1) { /* Only support this when "lowoverhead" mode is enabled. Turns on all required events + other low overhead */
    Tau_register_callback(ompt_callback_work, cb_t(on_ompt_callback_work));
    Tau_register_callback(ompt_callback_master, cb_t(on_ompt_callback_master));
#if defined (TAU_USE_OMPT_TR6)
    Tau_register_callback(ompt_callback_idle, cb_t(on_ompt_callback_idle));
#endif /* TAU_USE_OMPT_TR6 */
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

  initialized = true;
  initializing = false;
  return 1; //success
}

/* Register callbacks for plugins in the case that they are not already registered for TAU */
void Tau_ompt_register_plugin_callbacks(Tau_plugin_callbacks_active_t *Tau_plugins_enabled) {
  if(!initialized)
  {
    fprintf(stderr, "TAU: WARNING: Could not register OMPT plugin callbacks as OMPT was not initialized.\n");
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
  if (Tau_plugins_enabled->ompt_master > Tau_ompt_callbacks_enabled[ompt_callback_master])
    register_callback(ompt_callback_master, cb_t(on_ompt_callback_master));
#if defined (TAU_USE_OMPT_TR6)
  if (Tau_plugins_enabled->ompt_idle > Tau_ompt_callbacks_enabled[ompt_callback_idle])
    register_callback(ompt_callback_idle, cb_t(on_ompt_callback_idle));
#endif /* defined (TAU_USE_OMPT_TR6) */
  if (Tau_plugins_enabled->ompt_sync_region > Tau_ompt_callbacks_enabled[ompt_callback_sync_region])
    register_callback(ompt_callback_sync_region, cb_t(on_ompt_callback_sync_region));
  if (Tau_plugins_enabled->ompt_mutex_acquire > Tau_ompt_callbacks_enabled[ompt_callback_mutex_acquire])
    register_callback(ompt_callback_mutex_acquire, cb_t(on_ompt_callback_mutex_acquire));
  if (Tau_plugins_enabled->ompt_mutex_acquired > Tau_ompt_callbacks_enabled[ompt_callback_mutex_acquired])
    register_callback(ompt_callback_mutex_acquired, cb_t(on_ompt_callback_mutex_acquired));
  if (Tau_plugins_enabled->ompt_mutex_released > Tau_ompt_callbacks_enabled[ompt_callback_mutex_released])
    register_callback(ompt_callback_mutex_released, cb_t(on_ompt_callback_mutex_released));
  if (Tau_plugins_enabled->ompt_target > Tau_ompt_callbacks_enabled[ompt_callback_target])
    register_callback(ompt_callback_target, cb_t(on_ompt_callback_target));
  if (Tau_plugins_enabled->ompt_target_data_op > Tau_ompt_callbacks_enabled[ompt_callback_target_data_op])
    register_callback(ompt_callback_target_data_op, cb_t(on_ompt_callback_target_data_op));
  if (Tau_plugins_enabled->ompt_target_submit > Tau_ompt_callbacks_enabled[ompt_callback_target_submit])
    register_callback(ompt_callback_target_submit, cb_t(on_ompt_callback_target_submit));
}

/* This is called by the Tau_destructor_trigger() to prevent
 * callbacks from happening after TAU is shut down */
void Tau_ompt_finalize(void) {
    if(Tau_ompt_finalized()) { return; }
    Tau_ompt_finalized(true);
    if (ompt_finalize_tool != nullptr) {
        ompt_finalize_tool();
    }
}

/* This callback should come from the runtime when the runtime is shut down */
extern "C" void ompt_finalize(ompt_data_t* tool_data)
{
  TAU_VERBOSE("OpenMP runtime is shutting down...\n");
  /* Just in case... */
  Tau_destructor_trigger();

  if(Tau_plugins_enabled.ompt_finalize) {
    Tau_plugin_event_ompt_finalize_data_t plugin_data;

    plugin_data.null = 0;

    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_OMPT_FINALIZE, "*", &plugin_data);
  }
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
  return &result;
}
#else /*  defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
#include <Profile/TauPluginInternals.h>

void Tau_ompt_register_plugin_callbacks(Tau_plugin_callbacks_active_t *Tau_plugins_enabled) {
  return;
}
#endif /*  defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
