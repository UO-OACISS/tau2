/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauSnapshot.h  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains all the snapshot stuff        **
**                                                                         **
****************************************************************************/


#ifndef _TAU_SNAPSHOT_H_
#define _TAU_SNAPSHOT_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int   TAUDECL Tau_snapshot_initialization();
int   TAUDECL Tau_snapshot_writeUnifiedBuffer();
int   TAUDECL Tau_snapshot_writeToBuffer(const char *name);
char* TAUDECL Tau_snapshot_getBuffer();
int   TAUDECL Tau_snapshot_getBufferLength();
int   TAUDECL Tau_snapshot_writeFinal(const char *name);
int   TAUDECL Tau_snapshot_writeIntermediate(const char *name);
int   TAUDECL Tau_snapshot_writeMetaDataBlock();


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _TAU_SNAPSHOT_H_ */
