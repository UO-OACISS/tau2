/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.cs.uoregon.edu/research/tau             **
 * *****************************************************************************
 * **    Copyright 1997-2017                                                  **
 * **    Department of Computer and Information Science, University of Oregon **
 * ****************************************************************************/
/***************************************************************************
 * **      File            : TauPlugin.h                                      **
 * **      Description     : Tau Plugin API                                   **
 * **      Contact         : sramesh@cs.uoregon.edu                           **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 * ***************************************************************************/

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
