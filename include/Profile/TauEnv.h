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

#if (defined(TAU_WINDOWS))
#define TAUDECL __cdecl
#else
#define TAUDECL
#endif /* TAU_WINDOWS */

#ifdef __cplusplus
extern "C" {
#endif
  
  void TAUDECL TauEnv_initialize();
  int TAUDECL TauEnv_get_synchronize_clocks();
  int TAUDECL TauEnv_get_verbose();
  int TAUDECL TauEnv_get_throttle();
  double TAUDECL TauEnv_get_throttle_numcalls();
  double TAUDECL TauEnv_get_throttle_percall();
  const char *TauEnv_get_profiledir();
  const char *TauEnv_get_tracedir();

#define TAU_FORMAT_PROFILE 1
#define TAU_FORMAT_SNAPSHOT 2
#define TAU_FORMAT_MERGED 3
  int TAUDECL TauEnv_get_profile_format();

  void TAU_VERBOSE(const char *format, ...);
  
#ifdef __cplusplus
}
#endif


#endif /* _TAU_ENV_H_ */
