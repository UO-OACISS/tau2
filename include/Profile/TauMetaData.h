/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauMetaData.h  				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains metadata related routines     **
**                                                                         **
****************************************************************************/


#ifndef _TAU_METADATA_H_
#define _TAU_METADATA_H_


#include <TauUtil.h>
#include <map>
std::map<std::string,std::string> &Tau_metadata_getMetaData();
int Tau_metadata_writeMetaData(Tau_util_outputDevice *out, int counter, int tid);
int Tau_metadata_writeMetaData(FILE *fp, int counter, int tid);
int Tau_metadata_writeMetaData(Tau_util_outputDevice *out, int tid);
int Tau_metadata_writeMetaData(Tau_util_outputDevice *out);
int Tau_metadata_fillMetaData();
Tau_util_outputDevice *Tau_metadata_generateMergeBuffer();
void Tau_metadata_removeDuplicates(char *buffer, int buflen);

void Tau_metadata_register(char *name, int value);
int Tau_metadata_mergeMetaData();

#endif /* _TAU_METADATA_H_ */
