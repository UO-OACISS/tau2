#define _BSD_SOURCE

#ifdef TAU_USE_OMPT_TR6

#include <stdio.h>
#include <sstream>
#include <inttypes.h>
#include <omp.h>
#include <ompt.h>
#include <Profile/TauBfd.h>
#include <Profile/Profiler.h>
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

int Tau_set_tau_initialized() { tau_initialized = true; };

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
static ompt_set_callback_t ompt_set_callback;
static ompt_get_task_info_t ompt_get_task_info;
static ompt_get_thread_data_t ompt_get_thread_data;
static ompt_get_parallel_info_t ompt_get_parallel_info;
static ompt_get_unique_id_t ompt_get_unique_id;
static ompt_get_num_places_t ompt_get_num_places;
static ompt_get_place_proc_ids_t ompt_get_place_proc_ids;
static ompt_get_place_num_t ompt_get_place_num;
static ompt_get_partition_place_nums_t ompt_get_partition_place_nums;
static ompt_get_proc_id_t ompt_get_proc_id;
static ompt_enumerate_states_t ompt_enumerate_states;
static ompt_enumerate_mutex_impls_t ompt_enumerate_mutex_impls;

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
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  char timerName[1024];
  char resolved_address[1024];

  TauInternalFunctionGuard protects_this_function; 	
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

static void
on_ompt_callback_parallel_end(
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;

  if(codeptr_ra) {
    TAU_PROFILER_STOP(parallel_data->ptr);
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
  char contextEventName[2058];
  char buffer[2048];
  char timerName[1024];
  char resolved_address[1024];

  TauInternalFunctionGuard protects_this_function; 	
  if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      format_task_type(type, buffer);

      /*TODO: Srinivasan: This does not really fit in as a context event. Just keeping this here 
       * for the time being. It makes no sense to calculate any statistics for such events. 
       * Nick's advice: The ThreadTaskCreate/ThreadTaskSwitch/ThreadTaskComplete events are used in OTF2 to indicate creation of a task, 
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
      sprintf(contextEventName, "OpenMP_Task_Create %s ADDR <%lx> ", buffer, addr);

      TAU_REGISTER_CONTEXT_EVENT(event, contextEventName);
      TAU_EVENT_DISABLE_MAX(event);
      TAU_EVENT_DISABLE_MIN(event);
      TAU_EVENT_DISABLE_MEAN(event);
      TAU_EVENT_DISABLE_STDDEV(event);
      TAU_CONTEXT_EVENT(event, type);

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

/* Callback for task schedule */
static void 
on_ompt_callback_task_schedule(
    ompt_data_t *prior_task_data,
    ompt_task_status_t prior_task_status,
    ompt_data_t *next_task_data)
{
  if(prior_task_data->ptr) {
    TAU_PROFILER_STOP(prior_task_data->ptr);
  }

  if(next_task_data->ptr) {
    TAU_PROFILER_START(next_task_data->ptr);
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
  char timerName[1024];
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
        TAU_PROFILER_STOP(task_data->ptr);
        break;
    }
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
  ompt_work_type_t wstype,
  ompt_scope_endpoint_t endpoint,
  ompt_data_t *parallel_data,
  ompt_data_t *task_data,
  uint64_t count,
  const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  void *handle = NULL;
  char timerName[1024];
  char resolved_address[1024];
  if(codeptr_ra) {
    
   void * codeptr_ra_copy = (void*) codeptr_ra;
   unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
    switch(endpoint)
    {
      case ompt_scope_begin:
        task_data->ptr = NULL;
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
#ifndef __GNUG__ /*TODO: Remove this preprocessor check once a fix on our end has been identified.*/
              sprintf(timerName, "OpenMP_Work_Single_Executor %s", resolved_address);
              break; /* WARNING: The ompt_scope_begin for this work type is triggered, but the corresponding ompt_scope_end is not triggered when using GNU to compile the tool code*/ 
#else
	      return;
#endif
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
#ifndef __GNUG__ /*TODO: Remove this preprocessor check once a fix on our end has been identified.*/
              sprintf(timerName, "OpenMP_Work_Single_Executor ADDR <%lx>", addr);
              break; /* The ompt_scope_begin for this work type is triggered, but the corresponding ompt_scope_end is not triggered when using GNU to compile the tool code*/ 
#else
	      return;
#endif
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

        TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
        TAU_PROFILER_START(handle);
        task_data->ptr = (void*)handle;
        break;
      case ompt_scope_end: 
        if(task_data->ptr != NULL) {
	      TAU_PROFILER_STOP(task_data->ptr);
        }
	    break;
    }
  }
}

/*Thread begin/end callbacks. We do NOT need context information for these. If the user wants more context, 
 * he/she can use TAU_CALLPATH=1 to get callpath weights. In our experiments, this leads to additional 10-20% 
 * runtime overheads. Use with care. */
static void
on_ompt_callback_thread_begin(
  ompt_thread_type_t thread_type,
  ompt_data_t *thread_data)
{
  TauInternalFunctionGuard protects_this_function;
#if defined (TAU_USE_TLS)
  if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_DTLS)
  if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_PGS)
  if (pthread_getspecific(thr_id_key) != NULL) return; // master thread can't be a new worker.
#endif
  void *handle = NULL;
  char timerName[100];
  sprintf(timerName, "OpenMP_Thread_Type_%s", ompt_thread_type_t_values[thread_type]);
  TAU_PROFILER_CREATE(handle, timerName, "", TAU_DEFAULT);
  thread_data->ptr = (void*)handle;
  TAU_PROFILER_START(handle); 
}

static void
on_ompt_callback_thread_end(
  ompt_data_t *thread_data)
{
#if defined (TAU_USE_TLS)
  if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_DTLS)
  if (is_master) return; // master thread can't be a new worker.
#elif defined (TAU_USE_PGS)
  if (pthread_getspecific(thr_id_key) != NULL) return; // master thread can't be a new worker.
#endif
  TauInternalFunctionGuard protects_this_function;
  TAU_PROFILER_STOP(thread_data->ptr);
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
  const char *timerName= "OpenMP_Implicit_Task";

  TAU_PROFILE_TIMER(handle, timerName, "", TAU_DEFAULT);

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

/*Synchronization callbacks (barriers, etc). This is not a required event, but we need context.
 * TODO: This is an EXTREMELY high overhead call. The lines causing this overhead are the TAU_PROFILE_START/STOP calls. 
 * We have verified that this is not due to any OpenMP runtime overheads, but inside TAU.
 * At the moment, this call is enabled only when using TAU_OMPT support in "full" mode. 
 * Fixing this overhead is relatively low priority, because this is an optional event. */

static void
on_ompt_callback_sync_region(
    ompt_sync_region_kind_t kind,
    ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data,
    ompt_data_t *task_data,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  void *handle = NULL;
  char timerName[1024];
  char resolved_address[1024];

  if(codeptr_ra) {
    void * codeptr_ra_copy = (void*) codeptr_ra;
    unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

    switch(endpoint)
    {
      case ompt_scope_begin:
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
          }
        }
        TAU_PROFILER_CREATE(handle, timerName, " ", TAU_OPENMP);
        TAU_PROFILER_START(handle);
        task_data->ptr = (void*)handle;
        break;
      case ompt_scope_end:
        TAU_PROFILER_STOP(task_data->ptr);
        break;
    }

  }
}

/* Idle event - optional event that has low overhead and does not need context) */
static void
on_ompt_callback_idle(
    ompt_scope_endpoint_t endpoint)
{
  TauInternalFunctionGuard protects_this_function;
  const char *timerName= "OpenMP_Idle";

  TAU_PROFILE_TIMER(handle, timerName, "", TAU_DEFAULT);

  switch(endpoint)
  {
    case ompt_scope_begin:
      TAU_PROFILE_START(handle);
      break;
    case ompt_scope_end:
      TAU_PROFILE_STOP(handle);
      break;
  }

  return;
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

__thread void *mutex_waiting_handle;
__thread void *mutex_acquired_handle;

static void
on_ompt_callback_mutex_acquire(
    ompt_mutex_kind_t kind,
    unsigned int hint,
    unsigned int impl,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra) 
{
  TauInternalFunctionGuard protects_this_function;
  char timerName[1024];
  char resolved_address[1024];

  if(codeptr_ra) {

    void * codeptr_ra_copy = (void*) codeptr_ra;
    unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

    if(TauEnv_get_ompt_resolve_address_eagerly()) {
      Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
      switch(kind)
      {
        case ompt_mutex:
          sprintf(timerName, "OpenMP_Mutex_Waiting %s", resolved_address);
          break;
        case ompt_mutex_lock:
          sprintf(timerName, "OpenMP_Mutex_Waiting_Lock %s", resolved_address);
          break;
        case ompt_mutex_nest_lock:
          sprintf(timerName, "OpenMP_Mutex_Waiting_Nest_Lock %s", resolved_address);
          break;
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
        case ompt_mutex:
          sprintf(timerName, "OpenMP_Mutex_Waiting ADDR <%lx>", addr);
          break;
        case ompt_mutex_lock:
          sprintf(timerName, "OpenMP_Mutex_Waiting_Lock ADDR <%lx>", addr);
          break;
        case ompt_mutex_nest_lock:
          sprintf(timerName, "OpenMP_Mutex_Waiting_Nest_Lock ADDR <%lx>", addr);
          break;
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

    //printf("Mutex requested <%lx>\n", addr);

    // Start lock-wait timer
    TAU_PROFILER_CREATE(mutex_waiting_handle, timerName, " ", TAU_OPENMP);
    TAU_PROFILER_START(mutex_waiting_handle);
  }
}

static void
on_ompt_callback_mutex_acquired(
    ompt_mutex_kind_t kind,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  char timerName[1024];
  char resolved_address[1024];

  if(codeptr_ra) {
    void * codeptr_ra_copy = (void*) codeptr_ra;
    unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);

    if(TauEnv_get_ompt_resolve_address_eagerly()) {
      Tau_ompt_resolve_callsite_eagerly(addr, resolved_address);
      switch(kind)
      {
        case ompt_mutex:
          sprintf(timerName, "OpenMP_Mutex_Acquired %s", resolved_address);
          break;
        case ompt_mutex_lock:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Lock %s", resolved_address);
          break;
        case ompt_mutex_nest_lock:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Nest_Lock %s", resolved_address);
          break;
        case ompt_mutex_critical:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Critical %s", resolved_address);
          break;
        case ompt_mutex_atomic:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Atomic %s", resolved_address);
          break;
        case ompt_mutex_ordered:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Ordered %s", resolved_address);
          break;
      }
    } else {
      switch(kind)
      {
        case ompt_mutex:
          sprintf(timerName, "OpenMP_Mutex_Acquired ADDR <%lx>", addr);
          break;
        case ompt_mutex_lock:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Lock ADDR <%lx>", addr);
          break;
        case ompt_mutex_nest_lock:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Nest_Lock ADDR <%lx>", addr);
          break;
        case ompt_mutex_critical:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Critical ADDR <%lx>", addr);
          break;
        case ompt_mutex_atomic:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Atomic ADDR <%lx>", addr);
          break;
        case ompt_mutex_ordered:
          sprintf(timerName, "OpenMP_Mutex_Acquired_Ordered ADDR <%lx>", addr);
          break;
      }
    }
    //printf("Mutex acquired <%lx>\n", addr);

    // Stop lock-wait timer
    TAU_PROFILER_STOP(mutex_waiting_handle);

    // Start lock timer
    TAU_PROFILER_CREATE(mutex_acquired_handle, timerName, " ", TAU_OPENMP);
    TAU_PROFILER_START(mutex_acquired_handle);
  }

}

static void
on_ompt_callback_mutex_released(
    ompt_mutex_kind_t kind,
    ompt_wait_id_t wait_id,
    const void *codeptr_ra)
{
  TauInternalFunctionGuard protects_this_function;
  char timerName[1024];
  char resolved_address[1024];

  if(codeptr_ra) {
    void * codeptr_ra_copy = (void*) codeptr_ra;
    unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
    //printf("Mutex released <%lx>\n", addr);

    // Stop lock timer
    TAU_PROFILER_STOP(mutex_acquired_handle);
  }

}

/* Register callbacks. This function is invoked only from the ompt_start_tool routine.
 * Callbacks that only have "ompt_set_always" are the required events that we HAVE to support */
inline static void register_callback(ompt_callbacks_t name, ompt_callback_t cb) {
  int ret = ompt_set_callback(name, cb);

  switch(ret) { 
    case ompt_set_never:
      fprintf(stderr, "TAU: WARNING: Callback for event %s could not be registered\n", name); 
      break; 
    case ompt_set_sometimes: 
      TAU_VERBOSE("TAU: Callback for event %s registered with return value %s\n", name, "ompt_set_sometimes");
      break;
    case ompt_set_sometimes_paired:
      TAU_VERBOSE("TAU: Callback for event %s registered with return value %s\n", name, "ompt_set_sometimes_paired");
      break;
    case ompt_set_always:
      TAU_VERBOSE("TAU: Callback for event %s registered with return value %s\n", name, "ompt_set_always");
      break;
  }
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

void Tau_force_ompt_env_initialization() {

    const char* tmp = getenv("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY");

    if (parse_bool(tmp, 0)) {
      TauEnv_set_ompt_resolve_address_eagerly(1);
      TAU_VERBOSE("TAU: OMPT resolving addresses eagerly Enabled\n");
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "on");
      TAU_VERBOSE("TAU: Resolving OMPT addresses eagerly\n");
    } else {
      TAU_METADATA("TAU_OMPT_RESOLVE_ADDRESS_EAGERLY", "off");
    } 
    
    TauEnv_set_ompt_support_level(0); // Basic OMPT support is the default
    const char *omptSupportLevel = getenv("TAU_OMPT_SUPPORT_LEVEL");
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
    }
//#endif/* TAU_OMPT */
} 

#define cb_t(name) (ompt_callback_t)&name

/* Register callbacks for all events that we are interested in / have to support */
extern "C" int ompt_initialize(
  ompt_function_lookup_t lookup,
  ompt_data_t* tool_data)
{
  int ret;
  Tau_init_initializeTAU();
  if (initialized || initializing) return 0;
  initializing = true;
  TauInternalFunctionGuard protects_this_function;
  if (!TauEnv_get_openmp_runtime_enabled()) return 0;

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

/* Required events */
  register_callback(ompt_callback_parallel_begin, cb_t(on_ompt_callback_parallel_begin));
  register_callback(ompt_callback_parallel_end, cb_t(on_ompt_callback_parallel_end));
  register_callback(ompt_callback_task_create, cb_t(on_ompt_callback_task_create));
  register_callback(ompt_callback_task_schedule, cb_t(on_ompt_callback_task_schedule));
  register_callback(ompt_callback_implicit_task, cb_t(on_ompt_callback_implicit_task)); //Sometimes high-overhead, but unfortunately we cannot avoid this as it is a required event 
  register_callback(ompt_callback_thread_begin, cb_t(on_ompt_callback_thread_begin));
  register_callback(ompt_callback_thread_end, cb_t(on_ompt_callback_thread_end));

/* Optional events */

  if(TauEnv_get_ompt_support_level() >= 1) { /* Only support this when "lowoverhead" mode is enabled. Turns on all required events + other low overhead */
    register_callback(ompt_callback_work, cb_t(on_ompt_callback_work));
    register_callback(ompt_callback_master, cb_t(on_ompt_callback_master));
    register_callback(ompt_callback_idle, cb_t(on_ompt_callback_idle));
  }

  if(TauEnv_get_ompt_support_level() == 2) { /* Only support this when "full" is enabled. This is a high overhead call */
    register_callback(ompt_callback_sync_region, cb_t(on_ompt_callback_sync_region)); 
    // TODO: Overheads unclear currently. Also, causing a hang with TAU mm example
    /* register_callback(ompt_callback_mutex_acquire, cb_t(on_ompt_callback_mutex_acquire));
    register_callback(ompt_callback_mutex_acquired, cb_t(on_ompt_callback_mutex_acquired));
    register_callback(ompt_callback_mutex_released, cb_t(on_ompt_callback_mutex_released)); */
  }

  // Overheads unclear currently
  
  initialized = true;
  initializing = false;
  return 1; //success
}

extern "C" void ompt_finalize(ompt_data_t* tool_data)
{
  TAU_VERBOSE("OpenMP runtime is shutting down...\n");
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
#endif /* TAU_USE_OMPT_TR6 */
