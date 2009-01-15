
/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File 		: TauCAPI.h					   **
**	Description 	: TAU Profiling Package API for C		   **
**	Author		: Sameer Shende					   **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	   **
**	Flags		: Compile with				           **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)   **
**			  -DDEBUG_PROF  for internal debugging messages    **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
****************************************************************************/

#ifndef _TAU_CAPI_H_
#define _TAU_CAPI_H_

#if ((! defined( __cplusplus)) || defined (TAU_USE_C_API))
#ifdef TAU_USE_C_API
extern "C" {
#endif /* TAU_USE_C_API */
/* For C */
#include <stdio.h>
#include <Profile/ProfileGroups.h>
/* C API Definitions follow */
#if (defined(PROFILING_ON) || defined(TRACING_ON) )


/* These can't be used in C, only C++, so they are dummy macros here */
#define TAU_PROFILE(name, type, group) 
#define TAU_DYNAMIC_PROFILE(name, type, group) 
/* C doesn't support runtime type information */
#define CT(obj)
#define TYPE_STRING(profileString, str)



#define TAU_BCAST_DATA(data)  	                Tau_bcast_data(data)
#define TAU_REDUCE_DATA(data)  	                Tau_reduce_data(data)
#define TAU_ALLTOALL_DATA(data)                 Tau_alltoall_data(data) 
#define TAU_SCATTER_DATA(data)                  Tau_scatter_data(data) 
#define TAU_GATHER_DATA(data)  	                Tau_gather_data(data)
#define TAU_ALLREDUCE_DATA(data)  	        Tau_allreduce_data(data)
#define TAU_ALLGATHER_DATA(data)  	        Tau_allgather_data(data)
#define TAU_REDUCESCATTER_DATA(data)  	        Tau_reducescatter_data(data)
#define TAU_SCAN_DATA(data)  		        Tau_scan_data(data)




#endif /* PROFILING_ON */


#ifdef TAU_USE_C_API
}
#endif /* TAU_USE_C_API */

/* for consistency, we provide the long form */
#define TAU_STATIC_TIMER_START TAU_START
#define TAU_STATIC_TIMER_STOP TAU_STOP

#endif /* ! __cplusplus || TAU_C_API */
#endif /* _TAU_CAPI_H_ */

/***************************************************************************
 * $RCSfile: TauCAPI.h,v $   $Author: amorris $
 * $Revision: 1.61 $   $Date: 2009/01/15 00:27:15 $
 * POOMA_VERSION_ID: $Id: TauCAPI.h,v 1.61 2009/01/15 00:27:15 amorris Exp $
 ***************************************************************************/

