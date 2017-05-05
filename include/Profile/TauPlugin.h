#ifndef _TAU_PLUGIN_H_
#define _TAU_PLUGIN_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define TAU_PLUGIN_PATH "TAU_PLUGIN_PATH"
#define TAU_PLUGINS "TAU_PLUGINS"
#define TAU_PLUGIN_INIT_FUNC "Tau_plugin_init_func"

/*Define callbacks for specific events*/
typedef int (*Tau_plugin_function_registration_complete)(void);
typedef int (*Tau_plugin_atomic_event_trigger)(void);
typedef int (*Tau_plugin_end_of_execution)(void);

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

int Tau_initialize_plugin_system();
int Tau_util_load_and_register_plugins(PluginManager* plugin_manager);
void* Tau_util_load_plugin(const char *name, const char *path, PluginManager* plugin_manager);
void* Tau_util_register_plugin(const char *name, void* handle, PluginManager* plugin_manager);

int Tau_util_cleanup_all_plugins(PluginManager* plugin_manager);

PluginManager* Tau_util_get_plugin_manager();

void Tau_util_init_tau_plugin_callbacks(Tau_plugin_callbacks * cb);
void Tau_util_plugin_register_callbacks(Tau_plugin_callbacks * cb);
void Tau_util_invoke_callbacks(Tau_plugin_event event, const void * data);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_H_ */
