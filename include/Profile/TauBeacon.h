/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauUtil.h      				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : TAU's interface to Beacon pub-sub interface  **
**                                                                         **
****************************************************************************/

#ifndef _TAU_BEACON_H_
#define _TAU_BEACON_H_


#include <beacon.h> 

int TauBeaconInit(void);
int TauBeaconPublish(double value, char *units, char *topic, char *addtional_info); 
#endif /* _TAU_UTIL_H_ */
