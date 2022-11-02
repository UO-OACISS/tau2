/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.cs.uoregon.edu/research/tau             **
 * *****************************************************************************
 * **    Copyright 1997-2017                                                  **
 * **    Department of Computer and Information Science, University of Oregon **
 * **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauPluginTypes.h                                 **
 * **      Description     : Type definitions for the TAU Plugin System       **
 * **      Contact         : sramesh@cs.uoregon.edu                           **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 * ***************************************************************************/

#ifndef _TAU_PLUGIN_TYPES_H_
#define _TAU_PLUGIN_TYPES_H_

#include "TauMetaDataTypes.h"

#if defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
#if ((!defined(TAU_DISABLE_OMPT_5_0_IN_C_COMPILER)) || defined(__cplusplus))
#define TAU_PLUGIN_OMPT_ON
#endif /* TAU_DISABLE_OMPT_5_0_IN_C_COMPILER */
#endif /*  defined (TAU_USE_OMPT_TR6) || defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */

#ifdef TAU_PLUGIN_OMPT_ON
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
#include <omp-tools.h>
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
#if defined (TAU_USE_OMPT_TR6)
#include <ompt.h>
#endif /* TAU_USE_OMPT_TR6 */
#endif /* TAU_PLUGIN_OMPT_ON */

/* Using typedefs here to avoid having too many #ifdef this file */
#ifdef TAU_PLUGIN_OMPT_ON
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
#endif /* TAU_PLUGIN_OMPT_ON */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*Define data structures that define how TAU and the plugin exchange information*/
typedef struct Tau_plugin_event_function_registration_data {
   void * function_info_ptr;
   int tid;
} Tau_plugin_event_function_registration_data_t;

typedef struct Tau_plugin_event_metadata_registration_data {
   const char * name;
   Tau_metadata_value_t * value;
   int tid;
} Tau_plugin_event_metadata_registration_data_t;

/* GPU EVENTS BEGIN */
typedef struct Tau_plugin_event_gpu_init_data {
   int tid;
} Tau_plugin_event_gpu_init_data_t;

typedef struct Tau_plugin_event_gpu_finalize_data {
   int tid;
} Tau_plugin_event_gpu_finalize_data_t;

typedef struct Tau_plugin_event_gpu_kernel_exec_data {
   int tid;
   unsigned long int time;
} Tau_plugin_event_gpu_kernel_exec_data_t;

typedef struct Tau_plugin_event_gpu_memcpy_data {
   int tid;
   unsigned long int time;
   unsigned long int size;
   unsigned short int kind;
} Tau_plugin_event_gpu_memcpy_data_t;
/* GPU EVENTS END */

typedef struct Tau_plugin_event_post_init_data {
   int dummy;
   int tid;
} Tau_plugin_event_post_init_data_t;

typedef struct Tau_plugin_event_dump_data {
   int tid;
} Tau_plugin_event_dump_data_t;

typedef struct Tau_plugin_event_mpit_data {
   int pvar_index;
   char * pvar_name;
   long long int pvar_value;
} Tau_plugin_event_mpit_data_t;

typedef struct Tau_plugin_event_function_entry_data {
   const char * timer_name;
   const char * timer_group;
   unsigned int func_id;
   int tid;
   long unsigned int timestamp;
} Tau_plugin_event_function_entry_data_t;

typedef struct Tau_plugin_event_function_exit_data {
   const char * timer_name;
   const char * timer_group;
   unsigned int func_id;
   int tid;
   long unsigned int timestamp;
   double * metrics;
} Tau_plugin_event_function_exit_data_t;

typedef struct Tau_plugin_event_phase_entry_data {
   const char * phase_name;
} Tau_plugin_event_phase_entry_data_t;

typedef struct Tau_plugin_event_phase_exit_data {
   const char * phase_name;
} Tau_plugin_event_phase_exit_data_t;

typedef struct Tau_plugin_event_current_timer_exit_data {
   const char * name_prefix;
} Tau_plugin_event_current_timer_exit_data_t;

typedef struct Tau_plugin_event_send_data {
   long unsigned int message_tag;
   long unsigned int destination;
   long unsigned int bytes_sent;
   long unsigned int tid;
   long unsigned int timestamp;
} Tau_plugin_event_send_data_t;

typedef struct Tau_plugin_event_recv_data {
   long unsigned int message_tag;
   long unsigned int source;
   long unsigned int bytes_received;
   long unsigned int tid;
   long unsigned int timestamp;
} Tau_plugin_event_recv_data_t;

typedef struct Tau_plugin_event_atomic_event_registration_data {
   void * user_event_ptr;
   int tid;
} Tau_plugin_event_atomic_event_registration_data_t;

typedef struct Tau_plugin_event_atomic_event_trigger_data {
   const char * counter_name;
   int tid;
   long unsigned int value;
   long unsigned int timestamp;
} Tau_plugin_event_atomic_event_trigger_data_t;

typedef struct Tau_plugin_event_pre_end_of_execution_data {
   int tid;
} Tau_plugin_event_pre_end_of_execution_data_t;

typedef struct Tau_plugin_event_end_of_execution_data {
   int tid;
} Tau_plugin_event_end_of_execution_data_t;

typedef struct Tau_plugin_event_function_finalize_data {
   int junk;
} Tau_plugin_event_function_finalize_data_t;

typedef struct Tau_plugin_event_interrupt_trigger_data {
   int signum;
   int tid;
} Tau_plugin_event_interrupt_trigger_data_t ;

typedef struct Tau_plugin_event_trigger_data {
   void *data;
} Tau_plugin_event_trigger_data_t;

typedef struct Tau_plugin_event_ompt_parallel_begin_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_data_t *encountering_task_data;
   const ompt_frame_t *encountering_task_frame;
   ompt_data_t* parallel_data;
   uint32_t requested_team_size;
#if defined (TAU_USE_OMPT_TR6)
   ompt_invoker_t invoker;
#endif /* TAU_USE_OMPT_TR6 */
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
   int flags;
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_parallel_begin_data_t;

typedef struct Tau_plugin_event_ompt_parallel_end_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_data_t *parallel_data;
   ompt_data_t *encountering_task_data;
#if defined (TAU_USE_OMPT_TR6)
   ompt_invoker_t invoker;
#endif /* TAU_USE_OMPT_TR6 */
#if defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0)
   int flags;
#endif /* defined (TAU_USE_OMPT_TR7) || defined (TAU_USE_OMPT_5_0) */
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_parallel_end_data_t;

typedef struct Tau_plugin_event_ompt_task_create_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_data_t *encountering_task_data;
   const ompt_frame_t *encountering_frame;
   ompt_data_t* new_task_data;
   int type;
   int has_dependences;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_task_create_data_t;

typedef struct Tau_plugin_event_ompt_task_schedule_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_data_t *prior_task_data;
   ompt_task_status_t prior_task_status;
   ompt_data_t *next_task_data;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_task_schedule_data_t;

typedef struct Tau_plugin_event_ompt_implicit_task_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_scope_endpoint_t endpoint;
   ompt_data_t *parallel_data;
   ompt_data_t *task_data;
   unsigned int team_size;
   unsigned int thread_num;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_implicit_task_data_t;

typedef struct Tau_plugin_event_ompt_thread_begin_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_thread_t thread_type;
   ompt_data_t *thread_data;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_thread_begin_data_t;

typedef struct Tau_plugin_event_ompt_thread_end_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_data_t *thread_data;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_thread_end_data_t;

typedef struct Tau_plugin_event_ompt_work_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_work_t wstype;
   ompt_scope_endpoint_t endpoint;
   ompt_data_t *parallel_data;
   ompt_data_t *task_data;
   uint64_t count;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_work_data_t;

typedef struct Tau_plugin_event_ompt_master_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_scope_endpoint_t endpoint;
   ompt_data_t *parallel_data;
   ompt_data_t *task_data;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_master_data_t;

typedef struct Tau_plugin_event_ompt_idle_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_scope_endpoint_t endpoint;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_idle_data_t;

typedef struct Tau_plugin_event_ompt_sync_region_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_sync_region_t kind;
   ompt_scope_endpoint_t endpoint;
   ompt_data_t *parallel_data;
   ompt_data_t *task_data;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_sync_region_data_t;

typedef struct Tau_plugin_event_ompt_mutex_acquire_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_mutex_t kind;
   unsigned int hint;
   unsigned int impl;
   ompt_wait_id_t wait_id;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_mutex_acquire_data_t;

typedef struct Tau_plugin_event_ompt_mutex_acquired_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_mutex_t kind;
   ompt_wait_id_t wait_id;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_mutex_acquired_data_t;

typedef struct Tau_plugin_event_ompt_mutex_released_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_mutex_t kind;
   ompt_wait_id_t wait_id;
   const void *codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_mutex_released_data_t;

typedef struct Tau_plugin_event_ompt_device_initialize_data {
#ifdef TAU_PLUGIN_OMPT_ON
    int device_num;
    const char *type;
    ompt_device_t *device;
    ompt_function_lookup_t lookup;
    const char *documentation;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
    int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_device_initialize_data_t;

typedef struct Tau_plugin_event_ompt_device_finalize_data {
#ifdef TAU_PLUGIN_OMPT_ON
    int device_num;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
    int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_device_finalize_data_t;

typedef struct Tau_plugin_event_ompt_device_load_data {
#ifdef TAU_PLUGIN_OMPT_ON
    int device_num;
    const char *filename;
    int64_t offset_in_file;
    void *vma_in_file;
    size_t bytes;
    void *host_addr;
    void *device_addr;
    uint64_t module_id;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_device_load_data_t;

typedef struct Tau_plugin_event_ompt_target_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_target_t kind;
   ompt_scope_endpoint_t endpoint;
   int device_num;
   ompt_data_t * task_data;
   ompt_id_t target_id;
   const void * codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_target_data_t;

typedef struct Tau_plugin_event_ompt_target_data_op_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_id_t target_id;
   ompt_id_t host_op_id;
   ompt_target_data_op_t optype;
   void * src_addr;
   int src_device_num;
   void * dest_addr;
   int dest_device_num;
   size_t bytes;
   const void * codeptr_ra;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_target_data_op_data_t;

typedef struct Tau_plugin_event_ompt_target_submit_data {
#ifdef TAU_PLUGIN_OMPT_ON
   ompt_id_t target_id;
   ompt_id_t host_op_id;
   unsigned int requested_num_teams;
#else /* TAU_PLUGIN_OMPT_ON */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++.
    * This struct should never * be used if OMPT is not enabled */
   int null;
#endif /* TAU_PLUGIN_OMPT_ON */
} Tau_plugin_event_ompt_target_submit_data_t;

typedef struct Tau_plugin_event_ompt_finalize_data {
   /* TODO: Give a custom tool_data to tools and give it back in this callback */
   /* This is here for the sole purpose of preventing a warning saying that
    * empty struct have a size of 0 in C but 1 in C++ */
   int null;
} Tau_plugin_event_ompt_finalize_data_t;

/*Define callbacks for specific events*/
typedef int (*Tau_plugin_function_registration_complete)(Tau_plugin_event_function_registration_data_t*);
typedef int (*Tau_plugin_metadata_registration_complete)(Tau_plugin_event_metadata_registration_data_t*);
typedef int (*Tau_plugin_post_init)(Tau_plugin_event_post_init_data_t*);
typedef int (*Tau_plugin_dump)(Tau_plugin_event_dump_data_t*);
typedef int (*Tau_plugin_mpit)(Tau_plugin_event_mpit_data_t*);
typedef int (*Tau_plugin_function_entry)(Tau_plugin_event_function_entry_data_t*);
typedef int (*Tau_plugin_function_exit)(Tau_plugin_event_function_exit_data_t*);
typedef int (*Tau_plugin_phase_entry)(Tau_plugin_event_phase_entry_data_t*);
typedef int (*Tau_plugin_phase_exit)(Tau_plugin_event_phase_exit_data_t*);
typedef int (*Tau_plugin_current_timer_exit)(Tau_plugin_event_current_timer_exit_data_t*);
typedef int (*Tau_plugin_send)(Tau_plugin_event_send_data_t*);
typedef int (*Tau_plugin_recv)(Tau_plugin_event_recv_data_t*);
typedef int (*Tau_plugin_atomic_event_registration_complete)(Tau_plugin_event_atomic_event_registration_data_t*);
typedef int (*Tau_plugin_atomic_event_trigger)(Tau_plugin_event_atomic_event_trigger_data_t*);
typedef int (*Tau_plugin_pre_end_of_execution)(Tau_plugin_event_pre_end_of_execution_data_t*);
typedef int (*Tau_plugin_end_of_execution)(Tau_plugin_event_end_of_execution_data_t*);
typedef int (*Tau_plugin_function_finalize)(Tau_plugin_event_function_finalize_data_t*);
typedef int (*Tau_plugin_interrupt_trigger)(Tau_plugin_event_interrupt_trigger_data_t*);
typedef int (*Tau_plugin_trigger)(Tau_plugin_event_trigger_data_t*);
typedef int (*Tau_plugin_ompt_parallel_begin)(Tau_plugin_event_ompt_parallel_begin_data_t*);
typedef int (*Tau_plugin_ompt_parallel_end)(Tau_plugin_event_ompt_parallel_end_data_t*);
typedef int (*Tau_plugin_ompt_task_create)(Tau_plugin_event_ompt_task_create_data_t*);
typedef int (*Tau_plugin_ompt_task_schedule)(Tau_plugin_event_ompt_task_schedule_data_t*);
typedef int (*Tau_plugin_ompt_implicit_task)(Tau_plugin_event_ompt_implicit_task_data_t*);
typedef int (*Tau_plugin_ompt_thread_begin)(Tau_plugin_event_ompt_thread_begin_data_t*);
typedef int (*Tau_plugin_ompt_thread_end)(Tau_plugin_event_ompt_thread_end_data_t*);
typedef int (*Tau_plugin_ompt_work)(Tau_plugin_event_ompt_work_data_t*);
typedef int (*Tau_plugin_ompt_master)(Tau_plugin_event_ompt_master_data_t*);
typedef int (*Tau_plugin_ompt_idle)(Tau_plugin_event_ompt_idle_data_t*);
typedef int (*Tau_plugin_ompt_sync_region)(Tau_plugin_event_ompt_sync_region_data_t*);
typedef int (*Tau_plugin_ompt_mutex_acquire)(Tau_plugin_event_ompt_mutex_acquire_data_t*);
typedef int (*Tau_plugin_ompt_mutex_acquired)(Tau_plugin_event_ompt_mutex_acquired_data_t*);
typedef int (*Tau_plugin_ompt_mutex_released)(Tau_plugin_event_ompt_mutex_released_data_t*);
typedef int (*Tau_plugin_ompt_device_initialize)(Tau_plugin_event_ompt_device_initialize_data_t*);
typedef int (*Tau_plugin_ompt_device_finalize)(Tau_plugin_event_ompt_device_finalize_data_t*);
typedef int (*Tau_plugin_ompt_device_load)(Tau_plugin_event_ompt_device_load_data_t*);
typedef int (*Tau_plugin_ompt_target)(Tau_plugin_event_ompt_target_data_t*);
typedef int (*Tau_plugin_ompt_target_data_op)(Tau_plugin_event_ompt_target_data_op_data_t*);
typedef int (*Tau_plugin_ompt_target_submit)(Tau_plugin_event_ompt_target_submit_data_t*);
typedef int (*Tau_plugin_ompt_finalize)(Tau_plugin_event_ompt_finalize_data_t*);
/* GPU EVENTS BEGIN */
typedef int (*Tau_plugin_gpu_init)(Tau_plugin_event_gpu_init_data_t*);
typedef int (*Tau_plugin_gpu_finalize)(Tau_plugin_event_gpu_finalize_data_t*);
typedef int (*Tau_plugin_gpu_kernel_exec)(Tau_plugin_event_gpu_kernel_exec_data_t*);
typedef int (*Tau_plugin_gpu_memcpy)(Tau_plugin_event_gpu_memcpy_data_t*);
/* GPU EVENTS END */


/*Define the callback structure*/
typedef struct Tau_plugin_callbacks {
   Tau_plugin_function_registration_complete FunctionRegistrationComplete;
   Tau_plugin_metadata_registration_complete MetadataRegistrationComplete;
   Tau_plugin_post_init PostInit;
   Tau_plugin_dump Dump;
   Tau_plugin_mpit Mpit;
   Tau_plugin_function_entry FunctionEntry;
   Tau_plugin_function_exit FunctionExit;
   Tau_plugin_phase_entry PhaseEntry;
   Tau_plugin_phase_exit PhaseExit;
   Tau_plugin_current_timer_exit CurrentTimerExit;
   Tau_plugin_send Send;
   Tau_plugin_recv Recv;
   Tau_plugin_atomic_event_registration_complete AtomicEventRegistrationComplete;
   Tau_plugin_atomic_event_trigger AtomicEventTrigger;
   Tau_plugin_pre_end_of_execution PreEndOfExecution;
   Tau_plugin_end_of_execution EndOfExecution;
   Tau_plugin_function_finalize FunctionFinalize;
   Tau_plugin_interrupt_trigger InterruptTrigger;
   Tau_plugin_trigger Trigger;
   Tau_plugin_ompt_parallel_begin OmptParallelBegin;
   Tau_plugin_ompt_parallel_end OmptParallelEnd;
   Tau_plugin_ompt_task_create OmptTaskCreate;
   Tau_plugin_ompt_task_schedule OmptTaskSchedule;
   Tau_plugin_ompt_implicit_task OmptImplicitTask;
   Tau_plugin_ompt_thread_begin OmptThreadBegin;
   Tau_plugin_ompt_thread_end OmptThreadEnd;
   Tau_plugin_ompt_work OmptWork;
   Tau_plugin_ompt_master OmptMaster;
   Tau_plugin_ompt_idle OmptIdle;
   Tau_plugin_ompt_sync_region OmptSyncRegion;
   Tau_plugin_ompt_mutex_acquire OmptMutexAcquire;
   Tau_plugin_ompt_mutex_acquired OmptMutexAcquired;
   Tau_plugin_ompt_mutex_released OmptMutexReleased;
   Tau_plugin_ompt_device_initialize OmptDeviceInitialize;
   Tau_plugin_ompt_device_finalize OmptDeviceFinalize;
   Tau_plugin_ompt_device_load OmptDeviceLoad;
   Tau_plugin_ompt_target OmptTarget;
   Tau_plugin_ompt_target_data_op OmptTargetDataOp;
   Tau_plugin_ompt_target_submit OmptTargetSubmit;
   Tau_plugin_ompt_finalize OmptFinalize;
/* GPU EVENTS BEGIN */
   Tau_plugin_gpu_init GpuInit;
   Tau_plugin_gpu_finalize GpuFinalize;
   Tau_plugin_gpu_kernel_exec GpuKernelExec;
   Tau_plugin_gpu_memcpy GpuMemcpy;
/* GPU EVENTS END */
} Tau_plugin_callbacks_t;

/*Define all the events currently supported*/
typedef enum Tau_plugin_event {
   TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION,
   TAU_PLUGIN_EVENT_METADATA_REGISTRATION,
   TAU_PLUGIN_EVENT_POST_INIT,
   TAU_PLUGIN_EVENT_DUMP,
   TAU_PLUGIN_EVENT_MPIT,
   TAU_PLUGIN_EVENT_FUNCTION_ENTRY,
   TAU_PLUGIN_EVENT_FUNCTION_EXIT,
   TAU_PLUGIN_EVENT_PHASE_ENTRY,
   TAU_PLUGIN_EVENT_PHASE_EXIT,
   TAU_PLUGIN_EVENT_SEND,
   TAU_PLUGIN_EVENT_RECV,
   TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT,
   TAU_PLUGIN_EVENT_ATOMIC_EVENT_REGISTRATION,
   TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER,
   TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION,
   TAU_PLUGIN_EVENT_END_OF_EXECUTION,
   TAU_PLUGIN_EVENT_FUNCTION_FINALIZE,
   TAU_PLUGIN_EVENT_INTERRUPT_TRIGGER,
   TAU_PLUGIN_EVENT_TRIGGER,
   TAU_PLUGIN_EVENT_OMPT_PARALLEL_BEGIN,
   TAU_PLUGIN_EVENT_OMPT_PARALLEL_END,
   TAU_PLUGIN_EVENT_OMPT_TASK_CREATE,
   TAU_PLUGIN_EVENT_OMPT_TASK_SCHEDULE,
   TAU_PLUGIN_EVENT_OMPT_IMPLICIT_TASK,
   TAU_PLUGIN_EVENT_OMPT_THREAD_BEGIN,
   TAU_PLUGIN_EVENT_OMPT_THREAD_END,
   TAU_PLUGIN_EVENT_OMPT_WORK,
   TAU_PLUGIN_EVENT_OMPT_MASTER,
   TAU_PLUGIN_EVENT_OMPT_IDLE,
   TAU_PLUGIN_EVENT_OMPT_SYNC_REGION,
   TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRE,
   TAU_PLUGIN_EVENT_OMPT_MUTEX_ACQUIRED,
   TAU_PLUGIN_EVENT_OMPT_MUTEX_RELEASED,
   TAU_PLUGIN_EVENT_OMPT_DEVICE_INITIALIZE,
   TAU_PLUGIN_EVENT_OMPT_DEVICE_FINALIZE,
   TAU_PLUGIN_EVENT_OMPT_DEVICE_LOAD,
   TAU_PLUGIN_EVENT_OMPT_TARGET,
   TAU_PLUGIN_EVENT_OMPT_TARGET_DATA_OP,
   TAU_PLUGIN_EVENT_OMPT_TARGET_SUBMIT,
   TAU_PLUGIN_EVENT_OMPT_FINALIZE,
   /* GPU KERNEL START */
   TAU_PLUGIN_EVENT_GPU_INIT,
   TAU_PLUGIN_EVENT_GPU_FINALIZE,
   TAU_PLUGIN_EVENT_GPU_KERNEL_EXEC,
   TAU_PLUGIN_EVENT_GPU_MEMCPY,
   /* GPU KERNEL STOP */

   /* Max for number of events */
   NB_TAU_PLUGIN_EVENTS
} Tau_plugin_event_t;

/* Is the event registered with a callback? */
typedef struct Tau_plugin_callbacks_active {
    unsigned int function_registration;
    unsigned int metadata_registration;
    unsigned int post_init;
    unsigned int dump;
    unsigned int mpit;
    unsigned int function_entry;
    unsigned int function_exit;
    unsigned int phase_entry;
    unsigned int phase_exit;
    unsigned int send;
    unsigned int recv;
    unsigned int current_timer_exit;
    unsigned int atomic_event_registration;
    unsigned int atomic_event_trigger;
    unsigned int pre_end_of_execution;
    unsigned int end_of_execution;
    unsigned int function_finalize;
    unsigned int interrupt_trigger;
    unsigned int trigger;
    unsigned int ompt_parallel_begin;
    unsigned int ompt_parallel_end;
    unsigned int ompt_task_create;
    unsigned int ompt_task_schedule;
    unsigned int ompt_implicit_task;
    unsigned int ompt_thread_begin;
    unsigned int ompt_thread_end;
    unsigned int ompt_work;
    unsigned int ompt_master;
    unsigned int ompt_idle;
    unsigned int ompt_sync_region;
    unsigned int ompt_mutex_acquire;
    unsigned int ompt_mutex_acquired;
    unsigned int ompt_mutex_released;
    unsigned int ompt_device_initialize;
    unsigned int ompt_device_finalize;
    unsigned int ompt_device_load;
    unsigned int ompt_target;
    unsigned int ompt_target_data_op;
    unsigned int ompt_target_submit;
    unsigned int ompt_finalize;
    /* GPU KERNEL START */
    unsigned int gpu_init;
    unsigned int gpu_finalize;
    unsigned int gpu_kernel_exec;
    unsigned int gpu_memcpy;
    /* GPU KERNEL STOP */
} Tau_plugin_callbacks_active_t;

/*Deprecated*/

/*Define data structures to hold information about currently loaded plugins.
 * Only relevant for TAU internals - not a concern to plugins themselves*/
typedef struct Tau_plugin {
   char plugin_name[1024];
   void* handle;
   struct Tau_plugin * next;
} Tau_plugin_t;

typedef struct Tau_plugin_list {
   Tau_plugin_t * head;
} Tau_plugin_list_t;

typedef struct Tau_plugin_callback {
   Tau_plugin_callbacks_t cb;
   struct Tau_plugin_callback * next;
} Tau_plugin_callback_t;

typedef struct Tau_plugin_callback_list {
    Tau_plugin_callback_t * head;
} Tau_plugin_callback_list_t;

typedef struct PluginManager {
   Tau_plugin_list_t * plugin_list;
   Tau_plugin_callback_list_t * callback_list;
} PluginManager_t;

/* Deprecated */

typedef struct Tau_plugin_new {
   char plugin_name[1024];
   void* handle;
   unsigned int id;
} Tau_plugin_new_t;

typedef int (*PluginInitFunc) (int argc, char **argv, unsigned int plugin_id);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_TYPES_H_ */

