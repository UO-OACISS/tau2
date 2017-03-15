#ifndef _TAU_PLUGIN_H_
#define _TAU_PLUGIN_H_

#define TAU_PLUGIN_PATH "TAU_PLUGIN_PATH"
#define TAU_PLUGINS "TAU_PLUGINS"
#define TAU_PLUGIN_INIT_FUNC "Tau_plugin_init_func"

typedef struct Plugin {
    char plugin_name[1024];
    void* handle;
    struct Plugin* next;
} Plugin;

typedef struct PluginList {
    struct Plugin* head;
} PluginList;


typedef int (*PluginRoleHook)(int argc, void** argv); /*To be answered: What do I pass into the rolehook?*/

typedef struct PluginRoleHookNode {
    char role_name[1024];
    PluginRoleHook role_hook;
    struct PluginRoleHookNode* next;
} PluginRoleHookNode;

typedef struct PluginRoleHookList {
    struct PluginRoleHookNode* head;
} PluginRoleHookList;

typedef struct PluginManager {
    PluginList* plugin_list;
    PluginRoleHookList *role_hook_list;
} PluginManager;

typedef int (*PluginInitFunc) (PluginManager*);

PluginManager* Tau_PluginManager_new();
int Tau_util_load_and_register_plugins(PluginManager* plugin_manager);
void* Tau_util_load_plugin(const char *name, const char *path, PluginManager* plugin_manager);
void* Tau_util_register_plugin(const char *name, void* handle, PluginManager* plugin_manager);

#ifdef __cplusplus
extern "C" void Tau_util_plugin_manager_register_role_hook(PluginManager* plugin_manager, const char* role_name, PluginRoleHook role_hook);
extern "C" void Tau_util_apply_role_hook(PluginManager* plugin_manager, const char* role_name, int argc, void** argv);
#else
void Tau_util_plugin_manager_register_role_hook(PluginManager* plugin_manager, const char* role_name, PluginRoleHook role_hook);
void Tau_util_apply_role_hook(PluginManager* plugin_manager, const char* role_name, int argc, void** argv);
#endif

int Tau_util_cleanup_all_plugins(PluginManager* plugin_manager);
#endif /* _TAU_PLUGIN_H_ */
