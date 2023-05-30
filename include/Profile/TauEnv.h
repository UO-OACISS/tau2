/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2008  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauEnv.h 			        	   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Handle environment variables                     **
**                                                                         **
****************************************************************************/

#ifndef _TAU_ENV_H_
#define _TAU_ENV_H_

#include <tau_internal.h>

#define TAU_FORMAT_PROFILE 1
#define TAU_FORMAT_SNAPSHOT 2
#define TAU_FORMAT_MERGED 3
#define TAU_FORMAT_NONE 4
#define TAU_MAX_RECORDS 64*1024

#define TAU_ACTION_DUMP_PROFILES 1
#define TAU_ACTION_DUMP_CALLPATHS 2
#define TAU_ACTION_DUMP_BACKTRACES 3

#define TAU_PLUGIN_ENABLED 1

#ifndef TAU_EVENT_THRESHOLD
#define TAU_EVENT_THRESHOLD_DEFAULT -0.5
#endif /* TAU_EVENT_THRESHOLD */

#ifndef TAU_INTERRUPT_INTERVAL
#define TAU_INTERRUPT_INTERVAL_DEFAULT 10.0
#endif /* TAU_EVENT_THRESHOLD */

#define TAU_TRACE_FORMAT_TAU 0
#define TAU_TRACE_FORMAT_OTF2 1

#define TAU_EBS_RESOLUTION_FILE 0
#define TAU_EBS_RESOLUTION_FUNCTION 1
#define TAU_EBS_RESOLUTION_LINE 2
#define TAU_EBS_RESOLUTION_FUNCTION_LINE 3

#ifdef __cplusplus
extern "C" {
#endif

  void TAU_VERBOSE(const char *format, ...);

  void TAUDECL TauEnv_initialize();
  int  TAUDECL TauEnv_get_synchronize_clocks();
  int  TAUDECL TauEnv_get_verbose();
  int  TAUDECL TauEnv_get_throttle();
  void  TAUDECL TauEnv_set_throttle(int);
  int  TAUDECL TauEnv_get_profiling();
  int  TAUDECL TauEnv_get_tracing();
  int  TAUDECL TauEnv_get_thread_per_gpu_stream();
  int  TAUDECL TauEnv_get_trace_format();
  int  TAUDECL TauEnv_get_callpath();
  int  TAUDECL TauEnv_get_threadContext();
  int  TAUDECL TauEnv_get_callpath_depth();
  int  TAUDECL TauEnv_get_callsite();
  int  TAUDECL TauEnv_get_callsite_depth();
  int  TAUDECL TauEnv_get_callsite_offset();
  int  TAUDECL TauEnv_get_depth_limit();
  void TAUDECL TauEnv_set_depth_limit(int value);
  int  TAUDECL TauEnv_get_comm_matrix();
  int  TAUDECL TauEnv_get_current_timer_exit_params();
  int  TAUDECL TauEnv_get_track_message();
  int  TAUDECL TauEnv_get_lite_enabled();
  int  TAUDECL TauEnv_get_anonymize_enabled();
  int  TAUDECL TauEnv_get_compensate();
  int  TAUDECL TauEnv_get_level_zero_enable_api_tracing();

  int  TAUDECL TauEnv_get_track_load();
  int  TAUDECL TauEnv_get_track_memory_heap();
  int  TAUDECL TauEnv_get_track_memory_leaks();
  int  TAUDECL TauEnv_get_track_memory_headroom();
  int  TAUDECL TauEnv_get_track_io_params();
  int  TAUDECL TauEnv_get_track_signals();
  int  TAUDECL TauEnv_get_track_mpi_t_pvars();
  int  TAUDECL TauEnv_set_track_mpi_t_pvars(int value);
  int  TAUDECL TauEnv_get_ompt_resolve_address_eagerly();
  int  TAUDECL TauEnv_set_ompt_resolve_address_eagerly(int value);
  int  TAUDECL TauEnv_get_ompt_support_level();
  int  TAUDECL TauEnv_set_ompt_support_level(int value);
  int  TAUDECL TauEnv_get_ompt_force_finalize(void);
  int  TAUDECL TauEnv_get_signals_gdb();
  int  TAUDECL TauEnv_get_echo_backtrace();
  int  TAUDECL TauEnv_get_openmp_runtime_enabled();
  int  TAUDECL TauEnv_get_openmp_runtime_context();
  int  TAUDECL TauEnv_get_openmp_runtime_states_enabled();
  int  TAUDECL TauEnv_get_openmp_runtime_events_enabled();
  int  TAUDECL TauEnv_get_ebs_enabled();
  int  TAUDECL TauEnv_get_ebs_enabled_tau();
  int  TAUDECL TauEnv_get_ebs_keep_unresolved_addr();
  void  TAUDECL TauEnv_force_set_ebs_period(int period);
  int  TAUDECL TauEnv_get_ebs_period();
  int  TAUDECL TauEnv_get_ebs_inclusive();
  char *  TAUDECL Tau_check_dirname(const char *dirname);
  int  TAUDECL TauEnv_get_ebs_unwind();
  int  TAUDECL TauEnv_get_ebs_unwind_depth();
  int  TAUDECL TauEnv_get_ebs_resolution();
  int  TAUDECL TauEnv_get_stat_precompute();
  int  TAUDECL TauEnv_get_child_forkdirs();
  int  TAUDECL TauEnv_get_summary_only();
  int  TAUDECL TauEnv_get_ibm_bg_hwp_counters();
  double TAUDECL TauEnv_get_max_records();
  double TAUDECL TauEnv_get_evt_threshold();
  int TAUDECL TauEnv_get_interval();
  int TAUDECL TauEnv_get_disable_instrumentation();

  const char* TAUDECL TauEnv_get_ebs_source();
  const char* TAUDECL TauEnv_get_ebs_source_orig();
  void TAUDECL TauEnv_override_ebs_source(const char *newName);
  double      TAUDECL TauEnv_get_throttle_numcalls();
  double      TAUDECL TauEnv_get_throttle_percall();
  const char* TAUDECL TauEnv_get_profiledir();
  const char* TAUDECL TauEnv_get_tracedir();
  void TAUDECL TauEnv_set_profiledir(const char * new_profiledir);
  void TAUDECL TauEnv_set_tracedir(const char * new_tracedir);
  const char* TAUDECL TauEnv_get_metrics();
  const char* TAUDECL TauEnv_get_cvar_metrics();
  const char* TAUDECL TauEnv_get_cvar_values();
  const char* TAUDECL TauEnv_get_plugins_path();
  const char* TAUDECL TauEnv_get_plugins();
  int TAUDECL TauEnv_get_plugins_enabled();
  int TAUDECL TauEnv_get_track_mpi_t_comm_metric_values();
#ifndef TAU_WINDOWS
  const char  TAUDECL *TauEnv_get_mpi_t_comm_metric_values();
#endif
  int TAUDECL TauEnv_get_set_node();
  const char* TAUDECL TauEnv_get_cupti_api();
  const char* TAUDECL TauEnv_get_cuda_device_name();
  const char* TAUDECL TauEnv_get_cuda_instructions();
  int TAUDECL TauEnv_get_cuda_track_cdp();
  int TAUDECL TauEnv_get_cuda_track_unified_memory();
  int TAUDECL TauEnv_get_cuda_track_sass();
  const char* TAUDECL TauEnv_get_cuda_sass_type();
  int TAUDECL TauEnv_get_cuda_csv_output();
  int TAUDECL TauEnv_get_cuda_track_env();
  const char* TAUDECL TauEnv_get_cuda_binary_exe();
  int  TAUDECL TauEnv_get_cudaTotalThreads();
  void  TAUDECL TauEnv_set_cudaTotalThreads(int value);
  int TAUDECL TauEnv_get_tauCuptiAvail();
  void TAUDECL TauEnv_set_tauCuptiAvail(int value);
  int  TAUDECL TauEnv_get_nodeNegOneSeen();
  void  TAUDECL TauEnv_set_nodeNegOneSeen(int value);
  int TAUDECL TauEnv_get_mic_offload();
  int TAUDECL TauEnv_get_bfd_lookup();

  const char*  TAUDECL TauEnv_get_profile_prefix();
  int  TAUDECL TauEnv_get_profile_format();
  int  TAUDECL TauEnv_get_merge_metadata();
  int  TAUDECL TauEnv_get_disable_metadata();
  int  TAUDECL TauEnv_get_sigusr1_action();

  int TAUDECL TauEnv_get_memdbg();
  int TAUDECL TauEnv_get_memdbg_protect_above();
  void TAUDECL TauEnv_set_memdbg_protect_above(int);
  int TAUDECL TauEnv_get_memdbg_protect_below();
  void TAUDECL TauEnv_set_memdbg_protect_below(int);
  int TAUDECL TauEnv_get_memdbg_protect_free();
  void TAUDECL TauEnv_set_memdbg_protect_free(int);
  int TAUDECL TauEnv_get_memdbg_protect_gap();
  int TAUDECL TauEnv_get_memdbg_fill_gap();
  unsigned char TAUDECL TauEnv_get_memdbg_fill_gap_value();
  int TAUDECL TauEnv_get_memdbg_alloc_min();
  size_t TAUDECL TauEnv_get_memdbg_alloc_min_value();
  int TAUDECL TauEnv_get_memdbg_alloc_max();
  size_t TAUDECL TauEnv_get_memdbg_alloc_max_value();
  int TAUDECL TauEnv_get_memdbg_overhead();
  size_t TAUDECL TauEnv_get_memdbg_overhead_value();
  size_t TAUDECL TauEnv_get_memdbg_alignment();
  int TAUDECL TauEnv_get_memdbg_zero_malloc();
  int TAUDECL TauEnv_get_memdbg_attempt_continue();
  int TAUDECL TauEnv_get_pthread_stack_size();
  int TAUDECL TauEnv_get_alfred_port();
  int TAUDECL TauEnv_get_papi_multiplexing();
  int TAUDECL TauEnv_get_region_addresses();
  int TauEnv_get_show_memory_functions();
  int TAUDECL TauEnv_get_mem_callpath();
  const char * TAUDECL TauEnv_get_mem_classes();
  int TAUDECL TauEnv_get_mem_class_present(const char * name);
  const char * TAUDECL TauEnv_get_tau_exec_args();
  const char * TAUDECL TauEnv_get_tau_exec_path();
  int TAUDECL TauEnv_get_recycle_threads();
#ifdef __cplusplus
  void Tau_util_replaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace);
#endif /* __cplusplus */
  void Tau_util_replaceStringInPlaceC(char * subject, const char search,
                          const char replace);

#ifdef __cplusplus
}
#endif


#endif /* _TAU_ENV_H_ */
