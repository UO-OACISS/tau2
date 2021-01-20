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
**	File 		: TauInit.h 			        	   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : TAU Initialization                               **
**                                                                         **
****************************************************************************/

#ifndef _TAU_INIT_H_
#define _TAU_INIT_H_

#ifdef __cplusplus
extern "C" {
#endif

int Tau_init_initializeTAU();
int Tau_init_check_initialized();
int Tau_init_initializingTAU();
int Tau_get_inside_initialize(); 
void Tau_register_post_init_callback(void (*function)());


//call by wrappers/taupreload/dl_auditor.
void Tau_init_dl_initialized();
int Tau_init_check_dl_initialized();

#ifdef __cplusplus
}
#endif

#endif /* _TAU_INIT_H_ */
