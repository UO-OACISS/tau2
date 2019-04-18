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
} Tau_plugin_event_metadata_registration_data_t;

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
typedef void (*Tau_plugin_start_async_plugin)(void *);


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
   Tau_plugin_start_async_plugin StartAsyncPlugin;
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
   TAU_PLUGIN_EVENT_START_ASYNC_PLUGIN
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
    unsigned int start_async_plugin;
} Tau_plugin_callbacks_active_t;

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

////NPD

typedef struct Tau_plugin_new {
   char plugin_name[1024];
   void* handle;
   unsigned int id;
} Tau_plugin_new_t; 

typedef int (*PluginInitFunc) (int argc, char **argv, unsigned int plugin_id);
////


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_TYPES_H_ */

