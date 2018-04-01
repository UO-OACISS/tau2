#define _BSD_SOURCE
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
// #include "ompt-signal.h"
#include <Profile/Profiler.h>

//#define DEBUG_OUTPUT
#if defined(DEBUG_OUTPUT)
#define myprintf(...) printf(__VA_ARGS__)
#else
#define myprintf(...) 0
#endif

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

static void* openmp_parallel = NULL;
static void* openmp_task = NULL;
static void* openmp_thread = NULL;

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

static void
on_ompt_callback_parallel_begin(
  ompt_data_t *parent_task_data,
  const ompt_frame_t *parent_task_frame,
  ompt_data_t* parallel_data,
  uint32_t requested_team_size,
  ompt_invoker_t invoker,
  const void *codeptr_ra)
{
  char timerName[100];
  TauInternalFunctionGuard protects_this_function; 	
  if(codeptr_ra) {
      void * codeptr_ra_copy = (void*) codeptr_ra;
      unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
      sprintf(timerName, "OpenMP_Parallel_Region ADDR <%lx>", addr);
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
    void * codeptr_ra_copy = (void*) codeptr_ra;
    unsigned long addr = Tau_convert_ptr_to_unsigned_long(codeptr_ra_copy);
    TAU_PROFILER_STOP(parallel_data->ptr);
  }
}

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

#define cb_t(name) (ompt_callback_t)&name
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

  Tau_profile_c_timer(&openmp_task, "OpenMP task",  " ", TAU_OPENMP, "TAU_OPENMP");
  Tau_profile_c_timer(&openmp_thread, "OpenMP thread",  " ", TAU_OPENMP, "TAU_OPENMP");

/* Required events */

  register_callback(ompt_callback_parallel_begin, cb_t(on_ompt_callback_parallel_begin));
  register_callback(ompt_callback_parallel_end, cb_t(on_ompt_callback_parallel_end));

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
