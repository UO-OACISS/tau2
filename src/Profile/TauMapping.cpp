/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauMapping.cpp				  **
**	Description 	: TAU Profiling Package				  **
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

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#include "Profile/Profiler.h"

//////////////////////////////////////////////////////////////////////
// This profiler is used for mapping between different layers.
//////////////////////////////////////////////////////////////////////
//Profiler *& TheTauMapProf()
//{ 
//  static Profiler *TauMapProf = (Profiler *) NULL; 
//
//  return TauMapProf;
//}

//////////////////////////////////////////////////////////////////////
// This global variable is used to keep the function information for
// mapping. It is passed to the Profiler.
//////////////////////////////////////////////////////////////////////
FunctionInfo *& TheTauMapFI(unsigned int Pgroup )
{ 
  //static FunctionInfo *TauMapFI = (FunctionInfo *) NULL;
  static map<unsigned int, FunctionInfo *> TauMapGroups;

  return TauMapGroups[Pgroup];
}
// EOF
