/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauXML.h       				   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : This file contains xml related routines          **
**                                                                         **
****************************************************************************/


#ifndef _TAU_XML_H_
#define _TAU_XML_H_

#include <TauUtil.h>

void Tau_XML_writeString(Tau_util_outputDevice *out, const char *string);
void Tau_XML_writeTag(Tau_util_outputDevice *out, const char *tag, const char *string, bool newline);
void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const char *name, const char *value, bool newline);
void Tau_XML_writeAttribute(Tau_util_outputDevice *out, const char *name, const int value, bool newline);
int  Tau_XML_writeTime(Tau_util_outputDevice *out, bool newline);

#endif /* _TAU_XML_H_ */
