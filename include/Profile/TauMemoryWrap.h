/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauMemoryWrap.h  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : memory wrapper                                   **
**                                                                         **
****************************************************************************/


int Tau_memorywrap_checkPassThrough();
void Tau_memorywrap_writeHook();
void Tau_memorywrap_checkInit();
void Tau_memorywrap_add_ptr (void *ptr, size_t size);
void Tau_memorywrap_remove_ptr (void *ptr);


