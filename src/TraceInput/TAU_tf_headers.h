/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2003  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Research Center Juelich, Germany                                     **
****************************************************************************/
/***************************************************************************
**	File 		: TAU_tf_headers.h				  **
**	Description 	: TAU trace format reader library header files	  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu 	                  **
***************************************************************************/
/* TAU Trace format */
#ifndef _TAU_TF_HEADERS_H_
#define _TAU_TF_HEADERS_H_

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h> 
#include <string.h>
   
#include <map>
#include <iostream>   
using namespace std;  

/* TAU specific headers */
#define PCXX_EVENT_SRC 
#include <Profile/pcxx_events.h>
#include <Profile/pcxx_ansi.h>

/* TAU trace library specific headers */
#include <TAU_tf_decl.h>

#endif /* _TAU_TF_HEADERS_H_ */

/********************************************************************************
 * $RCSfile: TAU_tf_headers.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 2003/11/13 00:09:31 $
 * TAU_VERSION_ID: $Id: TAU_tf_headers.h,v 1.1 2003/11/13 00:09:31 sameer Exp $ 
 *******************************************************************************/
