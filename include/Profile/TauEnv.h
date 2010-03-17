/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2008  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauEnv.h 			        	   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Handle environment variables                     **
**                                                                         **
****************************************************************************/

#ifndef _TAU_ENV_H_
#define _TAU_ENV_H_

#include <tau_internal.h>

#ifdef __cplusplus
extern "C" {
#endif
  
  void TAU_VERBOSE(const char *format, ...);

  void TAUDECL TauEnv_initialize();
  int  TAUDECL TauEnv_get_synchronize_clocks();
  int  TAUDECL TauEnv_get_verbose();
  int  TAUDECL TauEnv_get_throttle();
  int  TAUDECL TauEnv_get_profiling();
  int  TAUDECL TauEnv_get_tracing();
  int  TAUDECL TauEnv_get_callpath();
  int  TAUDECL TauEnv_get_callpath_depth();
  int  TAUDECL TauEnv_get_depth_limit();
  void TAUDECL TauEnv_set_depth_limit(int value);
  int  TAUDECL TauEnv_get_comm_matrix();
  int  TAUDECL TauEnv_get_track_message();
  int  TAUDECL TauEnv_get_compensate();
  int  TAUDECL TauEnv_get_track_memory_heap();
  int  TAUDECL TauEnv_get_track_memory_headroom();
  int  TAUDECL TauEnv_get_extras();
  int  TAUDECL TauEnv_get_ebs_enabled();
  int  TAUDECL TauEnv_get_ebs_frequency();
  int  TAUDECL TauEnv_get_ebs_inclusive();

  const char* TAUDECL TauEnv_get_ebs_source();
  double      TAUDECL TauEnv_get_throttle_numcalls();
  double      TAUDECL TauEnv_get_throttle_percall();
  const char* TAUDECL TauEnv_get_profiledir();
  const char* TAUDECL TauEnv_get_tracedir();
  const char* TAUDECL TauEnv_get_metrics();


#define TAU_FORMAT_PROFILE 1
#define TAU_FORMAT_SNAPSHOT 2
#define TAU_FORMAT_MERGED 3
#define TAU_FORMAT_NONE 4
  int  TAUDECL TauEnv_get_profile_format();
  
#ifdef __cplusplus
}
#endif


#endif /* _TAU_ENV_H_ */
