/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: ProfileHeaders.h			          **
**	Description 	: TAU Profiling Package include files	    	  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**			  -DSGI_TIMERS  for SGI fast nanosecs timer	  **
**			  -DTULIP_TIMERS for non-sgi Platform	 	  **
**			  -DPOOMA_STDSTL for using STD STL in POOMA src   **
**			  -DPOOMA_TFLOP for Intel Teraflop at SNL/NM 	  **
**			  -DPOOMA_KAI for KCC compiler 			  **
**			  -DDEBUG_PROF  for internal debugging messages   **
**                        -DPROFILE_CALLSTACK to enable callstack traces  **
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/
#ifndef _PROFILE_HEADERS_H_
#define _PROFILE_HEADERS_H_

#include <string.h>

#if (defined(POOMA_KAI) || defined (TAU_STDCXXLIB))
#include <string>
using std::string;
#else
#define __BOOL_DEFINED 
#include "Profile/bstring.h"
#endif /* POOMA_KAI */


#ifndef NO_RTTI /* RTTI is present  */
#include <typeinfo.h>
#endif /* NO_RTTI  */

#ifdef POOMA_STDSTL
#include <vector>
#include <utility>
#include <list>
#include <map>
using std::vector;
using std::pair;
using std::list;
using std::map;
#else
#include <vector.h>
#include <map.h>
#if ((!defined(POOMA_KAI)) && (!defined(TAU_STDCXXLIB)))
#include <pair.h>
#else
#include <utility.h>
#endif /* not POOMA_KAI */
#include <list.h>
#endif /* POOMA_STDSTL */

#endif /* _PROFILE_HEADERS_H_ */
/***************************************************************************
 * $RCSfile: ProfileHeaders.h,v $   $Author: sameer $
 * $Revision: 1.2 $   $Date: 1998/07/10 20:11:29 $
 * POOMA_VERSION_ID: $Id: ProfileHeaders.h,v 1.2 1998/07/10 20:11:29 sameer Exp $ 
 ***************************************************************************/
