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
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
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
