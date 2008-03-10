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

#ifdef __cplusplus
extern "C" {
#endif
  
  void TauEnv_initialize();
  int TauEnv_get_synchronize_clocks();
  int TauEnv_get_verbose();
  int TauEnv_get_throttle();
  double TauEnv_get_throttle_numcalls();
  double TauEnv_get_throttle_percall();
  char *TauEnv_get_profiledir();
  char *TauEnv_get_tracedir();

#define TAU_FORMAT_PROFILE 1
#define TAU_FORMAT_SNAPSHOT 2
  int TauEnv_get_profile_format();

  void TAU_VERBOSE(char *format, ...);
  
#ifdef __cplusplus
}
#endif


#endif /* _TAU_ENV_H_ */
