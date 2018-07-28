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

//Forward declarations
struct Tau_plugin_event_function_registration_data;
struct Tau_plugin_event_metadata_registration_data;
struct Tau_plugin_event_post_init_data;
struct Tau_plugin_event_dump;
struct Tau_plugin_event_mpit;
struct Tau_plugin_event_function_entry;
struct Tau_plugin_event_function_exit;
struct Tau_plugin_event_current_timer_exit;
struct Tau_plugin_event_send;
struct Tau_plugin_event_recv;
struct Tau_plugin_event_atomic_event_trigger_data;
struct Tau_plugin_event_atomic_event_registration_data;
struct Tau_plugin_event_pre_end_of_execution_data;
struct Tau_plugin_event_end_of_execution_data;
struct Tau_plugin_event_function_finalize_data;
struct Tau_plugin_event_interrupt_trigger_data;

/*Define callbacks for specific events*/
typedef int (*Tau_plugin_function_registration_complete)(struct Tau_plugin_event_function_registration_data);
typedef int (*Tau_plugin_metadata_registration_complete)(struct Tau_plugin_event_metadata_registration_data);
typedef int (*Tau_plugin_post_init)(struct Tau_plugin_event_post_init_data);
typedef int (*Tau_plugin_dump)(struct Tau_plugin_event_dump_data);
typedef int (*Tau_plugin_mpit)(struct Tau_plugin_event_mpit_data);
typedef int (*Tau_plugin_function_entry)(struct Tau_plugin_event_function_entry_data);
typedef int (*Tau_plugin_function_exit)(struct Tau_plugin_event_function_exit_data);
typedef int (*Tau_plugin_current_timer_exit)(struct Tau_plugin_event_current_timer_exit_data);
typedef int (*Tau_plugin_send)(struct Tau_plugin_event_send_data);
typedef int (*Tau_plugin_recv)(struct Tau_plugin_event_recv_data);
typedef int (*Tau_plugin_atomic_event_registration_complete)(struct Tau_plugin_event_atomic_event_registration_data);
typedef int (*Tau_plugin_atomic_event_trigger)(struct Tau_plugin_event_atomic_event_trigger_data);
typedef int (*Tau_plugin_pre_end_of_execution)(struct Tau_plugin_event_pre_end_of_execution_data);
typedef int (*Tau_plugin_end_of_execution)(struct Tau_plugin_event_end_of_execution_data);
typedef int (*Tau_plugin_function_finalize)(struct Tau_plugin_event_function_finalize_data);
typedef int (*Tau_plugin_interrupt_trigger)(struct Tau_plugin_event_interrupt_trigger_data);

/*Define the callback structure*/
struct Tau_plugin_callbacks {
   Tau_plugin_function_registration_complete FunctionRegistrationComplete;
   Tau_plugin_metadata_registration_complete MetadataRegistrationComplete;
   Tau_plugin_post_init PostInit;
   Tau_plugin_dump Dump;
   Tau_plugin_mpit Mpit;
   Tau_plugin_function_entry FunctionEntry;
   Tau_plugin_function_exit FunctionExit;
   Tau_plugin_current_timer_exit CurrentTimerExit;
   Tau_plugin_send Send;
   Tau_plugin_recv Recv;
   Tau_plugin_atomic_event_registration_complete AtomicEventRegistrationComplete;
   Tau_plugin_atomic_event_trigger AtomicEventTrigger;
   Tau_plugin_pre_end_of_execution PreEndOfExecution;
   Tau_plugin_end_of_execution EndOfExecution;
   Tau_plugin_function_finalize FunctionFinalize;
   Tau_plugin_interrupt_trigger InterruptTrigger;
};

/*Define all the events currently supported*/
typedef enum Tau_plugin_event {
   TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION,
   TAU_PLUGIN_EVENT_METADATA_REGISTRATION,
   TAU_PLUGIN_EVENT_POST_INIT,
   TAU_PLUGIN_EVENT_DUMP,
   TAU_PLUGIN_EVENT_MPIT,
   TAU_PLUGIN_EVENT_FUNCTION_ENTRY,
   TAU_PLUGIN_EVENT_FUNCTION_EXIT,
   TAU_PLUGIN_EVENT_CURRENT_TIMER_EXIT,
   TAU_PLUGIN_EVENT_SEND,
   TAU_PLUGIN_EVENT_RECV,
   TAU_PLUGIN_EVENT_ATOMIC_EVENT_REGISTRATION,
   TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER,
   TAU_PLUGIN_EVENT_PRE_END_OF_EXECUTION,
   TAU_PLUGIN_EVENT_END_OF_EXECUTION,
   TAU_PLUGIN_EVENT_FUNCTION_FINALIZE,
   TAU_PLUGIN_EVENT_INTERRUPT_TRIGGER
} Tau_plugin_event_t;

/*Define data structures that define how TAU and the plugin exchange information*/
struct Tau_plugin_event_function_registration_data {
   void * function_info_ptr;
   int tid;
};

struct Tau_plugin_event_metadata_registration_data {
   const char * name;
   Tau_metadata_value_t * value;
};

struct Tau_plugin_event_post_init_data {
   int dummy;
};

struct Tau_plugin_event_dump_data {
   int tid;
};

struct Tau_plugin_event_mpit_data {
   int pvar_index;
   char *pvar_name;
   long long int pvar_value;
};

struct Tau_plugin_event_function_entry_data {
   const char * timer_name;
   const char * timer_group;
   int tid;
   long unsigned int timestamp;
};

struct Tau_plugin_event_function_exit_data {
   const char * timer_name;
   const char * timer_group;
   int tid;
   long unsigned int timestamp;
};

struct Tau_plugin_event_current_timer_exit_data {
   const char * name_prefix;
};

struct Tau_plugin_event_send_data {
   long unsigned int message_tag;
   long unsigned int destination;
   long unsigned int bytes_sent;
   long unsigned int tid;
   long unsigned int timestamp;
};

struct Tau_plugin_event_recv_data {
   long unsigned int message_tag;
   long unsigned int source;
   long unsigned int bytes_received;
   long unsigned int tid;
   long unsigned int timestamp;
};

struct Tau_plugin_event_atomic_event_registration_data {
   void * user_event_ptr;
};

struct Tau_plugin_event_atomic_event_trigger_data {
   const char * counter_name;
   int tid;
   long unsigned int value;
   long unsigned int timestamp;
};

struct Tau_plugin_event_pre_end_of_execution_data {
   int tid;
};

struct Tau_plugin_event_end_of_execution_data {
   int tid;
};

struct Tau_plugin_event_function_finalize_data {
   int junk;
};

struct Tau_plugin_event_interrupt_trigger_data {
   int signum;
};

typedef struct Tau_plugin_event_function_registration_data Tau_plugin_event_function_registration_data;
typedef struct Tau_plugin_event_metadata_registration_data Tau_plugin_event_metadata_registration_data;
typedef struct Tau_plugin_event_post_init_data Tau_plugin_event_post_init_data;
typedef struct Tau_plugin_event_dump_data Tau_plugin_event_dump_data;
typedef struct Tau_plugin_event_mpit_data Tau_plugin_event_mpit_data;
typedef struct Tau_plugin_event_function_entry_data Tau_plugin_event_function_entry_data;
typedef struct Tau_plugin_event_function_exit_data Tau_plugin_event_function_exit_data;
typedef struct Tau_plugin_event_current_timer_exit_data Tau_plugin_event_current_timer_exit_data;
typedef struct Tau_plugin_event_send_data Tau_plugin_event_send_data;
typedef struct Tau_plugin_event_recv_data Tau_plugin_event_recv_data;
typedef struct Tau_plugin_event_atomic_event_registration_data Tau_plugin_event_atomic_event_registration_data;
typedef struct Tau_plugin_event_atomic_event_trigger_data Tau_plugin_event_atomic_event_trigger_data;
typedef struct Tau_plugin_event_pre_end_of_execution_data Tau_plugin_event_pre_end_of_execution_data;
typedef struct Tau_plugin_event_end_of_execution_data Tau_plugin_event_end_of_execution_data;
typedef struct Tau_plugin_event_function_finalize_data Tau_plugin_event_function_finalize_data;
typedef struct Tau_plugin_event_interrupt_trigger_data Tau_plugin_event_interrupt_trigger_data;

typedef struct Tau_plugin_callbacks Tau_plugin_callbacks;

/*Define data structures to hold information about currently loaded plugins. 
 * Only relevant for TAU internals - not a concern to plugins themselves*/
typedef struct Tau_plugin {
   char plugin_name[1024];
   void* handle;
   struct Tau_plugin * next;
} Tau_plugin;

typedef struct Tau_plugin_list {
   struct Tau_plugin * head;
} Tau_plugin_list;

typedef struct Tau_plugin_callback_ {
   Tau_plugin_callbacks cb;
   struct Tau_plugin_callback_ * next;
} Tau_plugin_callback_;

typedef struct Tau_plugin_callback_list {
    Tau_plugin_callback_ * head;
} Tau_plugin_callback_list;

typedef struct PluginManager {
   Tau_plugin_list * plugin_list;
   Tau_plugin_callback_list * callback_list;
} PluginManager;

typedef int (*PluginInitFunc) (int argc, char **argv);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_TYPES_H_ */

