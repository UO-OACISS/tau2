#include <Profile/TauPluginTypes.h>

#ifndef _TAU_PLUGIN_H_
#define _TAU_PLUGIN_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void Tau_util_init_tau_plugin_callbacks(Tau_plugin_callbacks * cb);
void Tau_util_plugin_register_callbacks(Tau_plugin_callbacks * cb);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_PLUGIN_H_ */
