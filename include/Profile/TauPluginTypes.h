#ifndef _TAU_PLUGIN_TYPES_H_
#define _TAU_PLUGIN_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
struct Tau_plugin_event_function_registration_data;
struct Tau_plugin_event_atomic_event_trigger_data;
struct Tau_plugin_event_end_of_execution_data;

/*Define callbacks for specific events*/
typedef int (*Tau_plugin_function_registration_complete)(struct Tau_plugin_event_function_registration_data);
typedef int (*Tau_plugin_atomic_event_trigger)(struct Tau_plugin_event_atomic_event_trigger_data);
typedef int (*Tau_plugin_end_of_execution)(struct Tau_plugin_event_end_of_execution_data);

struct Tau_plugin_callbacks {
   Tau_plugin_function_registration_complete FunctionRegistrationComplete;
   Tau_plugin_atomic_event_trigger AtomicEventTrigger;
   Tau_plugin_end_of_execution EndOfExecution;
};

enum Tau_plugin_event {
   TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION,
   TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER,
   TAU_PLUGIN_EVENT_END_OF_EXECUTION
};

struct Tau_plugin_event_function_registration_data {
   int data;
};

struct Tau_plugin_event_atomic_event_trigger_data {
   int data;
};

struct Tau_plugin_event_end_of_execution_data {
   int data;
};

typedef struct Tau_plugin_event_function_registration_data Tau_plugin_event_function_registration_data;
typedef struct Tau_plugin_event_atomic_event_trigger_data Tau_plugin_event_atomic_event_trigger_data;
typedef struct Tau_plugin_event_end_of_execution_data Tau_plugin_event_end_of_execution_data;

typedef struct Tau_plugin_callbacks Tau_plugin_callbacks;

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
   Tau_plugin_callback_ * next;
};

typedef struct Tau_plugin_callback_list {
    Tau_plugin_callback_ * head;
} Tau_plugin_callback_list;

typedef struct PluginManager {
   Tau_plugin_list * plugin_list;
   Tau_plugin_callback_list * callback_list;
} PluginManager;

typedef int (*PluginInitFunc) (PluginManager*);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_TYPES_H_ */

