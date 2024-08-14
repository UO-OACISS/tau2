/****************************************************************************
 **      TAU Portable Profiling Package         **
 **      http://www.cs.uoregon.edu/research/tau             **
 *****************************************************************************
 **    Copyright 2009                      **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/****************************************************************************
 **  File     : TauCAPI.h             **
 **  Description   : TAU Profiling Package API for C++       **
 **  Contact    : tau-team@cs.uoregon.edu                    **
 **  Documentation  : See http://www.cs.uoregon.edu/research/tau       **
 ****************************************************************************/

#ifndef _TAU_CPPAPI_H_
#define _TAU_CPPAPI_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DEBUG_PROF
#define DEBUGPROFMSG(msg) { cerr<< msg; }
#else
#define DEBUGPROFMSG(msg)
#endif // DEBUG_PROF

#define TAU_NEW(expr, size)        Tau_new(expr, size, __FILE__, __LINE__)
#define TAU_DELETE(expr, variable) Tau_track_memory_deallocation(variable, __FILE__, __LINE__) , expr

#define TAU_TYPE_STRING(profileString, str) static string profileString(str);

#ifdef NO_RTTI
#define TAU_CT(obj) string(" ")
#else /* RTTI is present */
#define TAU_CT(obj) RtsLayer::GetRTTI(typeid(obj).name())
#endif /* NO_RTTI */

struct Tau_Profile_Wrapper
{
  void * fInfo;

  Tau_Profile_Wrapper(void * fi, int phase = 0) : fInfo(fi) {
#ifndef TAU_PROFILEPHASE
    phase = 0;
#endif
    if (fi) {
      Tau_lite_start_timer(fi, phase);
    }
  }

  ~Tau_Profile_Wrapper() {
    if (fInfo) {
      Tau_lite_stop_timer(fInfo);
    }
  }
};

class TauInternalFunctionGuard
{
public:

  TauInternalFunctionGuard() : enabled(true) {
    Tau_global_incr_insideTAU();
  }

  TauInternalFunctionGuard(bool flag) : enabled(flag) {
    if (enabled) {
      Tau_global_incr_insideTAU();
    }
  }

  ~TauInternalFunctionGuard() {
    if (enabled) {
      Tau_global_decr_insideTAU();
    }
  }

private:

  // If false then the guard has no effect
  bool enabled;
};

#define TAU_PROFILE(name, type, group) \
  static void *tauFI = 0; \
  if (tauFI == 0) tauCreateFI(&tauFI, name, type, (TauGroup_t)group, #group); \
  Tau_Profile_Wrapper tauFProf(tauFI);

#define TAU_PHASE(name, type, group) \
  static void *tauFInfo = NULL; \
  static char *TauGroupNameUsed = Tau_phase_enable(#group); \
  tauCreateFI(&tauFInfo, name, type, (TauGroup_t)group, TauGroupNameUsed); \
  Tau_Profile_Wrapper tauFProf(tauFInfo, 1);

#define TAU_DYNAMIC_PROFILE(name, type, group) \
  static TauGroup_t tau_dy_group = group; \
  static int tau_timer_counter = 0; \
  void *tauFInfo = NULL; \
  char tau_timer_iteration_number[128]; \
  snprintf(tau_timer_iteration_number, sizeof(tau_timer_iteration_number),  " [%d]", ++tau_timer_counter); \
  tauCreateFI(&tauFInfo, string(name)+string(tau_timer_iteration_number), type, tau_dy_group, #group); \
  Tau_Profile_Wrapper tauFProf(tauFInfo);

#define TAU_DYNAMIC_PHASE(name, type, group) \
  static TauGroup_t tau_dy_group = group; \
  static int tau_timer_counter = 0; \
  void *tauFInfo = NULL; \
  char tau_timer_iteration_number[128]; \
  snprintf(tau_timer_iteration_number, sizeof(tau_timer_iteration_number),  " [%d]", ++tau_timer_counter); \
  tauCreateFI(&tauFInfo, string(name)+string(tau_timer_iteration_number), type, tau_dy_group, #group); \
  Tau_Profile_Wrapper tauFProf(tauFInfo, 1);

#endif /* _TAU_CPPAPI_H_ */
