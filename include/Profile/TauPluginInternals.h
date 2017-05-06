#include <Profile/TauPluginTypes.h>

#ifndef _TAU_PLUGIN_INTERNALS_H_
#define _TAU_PLUGIN_INTERNALS_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define TAU_PLUGIN_PATH "TAU_PLUGIN_PATH"
#define TAU_PLUGINS "TAU_PLUGINS"
#define TAU_PLUGIN_INIT_FUNC "Tau_plugin_init_func"

int Tau_initialize_plugin_system();
int Tau_util_load_and_register_plugins(PluginManager* plugin_manager);
void* Tau_util_load_plugin(const char *name, const char *path, PluginManager* plugin_manager);
void* Tau_util_register_plugin(const char *name, void* handle, PluginManager* plugin_manager);

int Tau_util_cleanup_all_plugins(PluginManager* plugin_manager);

PluginManager* Tau_util_get_plugin_manager();
void Tau_util_invoke_callbacks(Tau_plugin_event event, const void * data);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_INTERNALS_H_ */

