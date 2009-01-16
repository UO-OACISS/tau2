/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: ProfileHeaders.h			          **
**	Description 	: TAU Profiling Package include files	    	  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/
#ifndef _PROFILE_HEADERS_H_
#define _PROFILE_HEADERS_H_

#include <string.h>

#if (defined(TAU_DOT_H_LESS_HEADERS) || defined (TAU_STDCXXLIB))
#include <string>
#include <vector>
#include <utility>
#include <list>
#include <map>
using std::string;
#define TAU_STD_NAMESPACE std::
#ifdef TAU_LIBRARY_SOURCE
using std::vector;
using std::pair;
using std::list;
using std::map;
#endif /* TAU_LIBRARY_SOURCE */
#else
#define __BOOL_DEFINED 
#include "Profile/bstring.h"
#include <vector.h>
#include <map.h>
#include <list.h>
#include <pair.h>
#endif /* TAU_DOT_H_LESS_HEADERS  || TAU_STDCXXLIB */

#ifndef NO_RTTI /* RTTI is present */
#ifdef RTTI 
#include <typeinfo.h>
#else /* RTTI */
#include <typeinfo>
using std::type_info;
/* This is by default */ 
#endif /* RTTI */
#endif /* NO_RTTI */


#endif /* _PROFILE_HEADERS_H_ */
/***************************************************************************
 * $RCSfile: ProfileHeaders.h,v $   $Author: amorris $
 * $Revision: 1.8 $   $Date: 2009/01/16 00:46:32 $
 * POOMA_VERSION_ID: $Id: ProfileHeaders.h,v 1.8 2009/01/16 00:46:32 amorris Exp $ 
 ***************************************************************************/
