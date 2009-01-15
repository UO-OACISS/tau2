/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
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
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#include "Profile/Profiler.h"

#ifdef TAU_WINDOWS
using namespace std;
#endif

struct lTauGroup {
  bool operator()(const TauGroup_t s1, const TauGroup_t s2) const {
    return s1 < s2;
  }
};


//////////////////////////////////////////////////////////////////////
// This global variable is used to keep the function information for
// mapping. It is passed to the Profiler. It takes the key and returns
// the FunctionInfo * pointer that contains the id of the function 
// being mapped. The key is currently in the form of a profile group.
//////////////////////////////////////////////////////////////////////
void *& TheTauMapFI(TauGroup_t key) { 
  //static FunctionInfo *TauMapFI = (FunctionInfo *) NULL;
  static map<TauGroup_t, void*, lTauGroup > TauMapGroups;
  return TauMapGroups[key];
}
// EOF
