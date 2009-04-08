/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauCallPath.cpp				  **
**	Description 	: TAU Profiling Package				  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF // For Debugging Messages from Profiler.cpp
#include "Profile/Profiler.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <string>
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

//////////////////////////////////////////////////////////////////////
// How deep should the callpath be? The default value is 2
//////////////////////////////////////////////////////////////////////
int& TauGetCallPathDepth(void) {
  static int value = 0;

#ifdef TAU_PROFILEPHASE
  value = 2;
  return value;
#endif /* TAU_PROFILEPHASE */

  if (value == 0) {
    value = TauEnv_get_callpath_depth();
  }
  return value;
}

//////////////////////////////////////////////////////////////////////
// Global variables (wrapped in routines for static initialization)
/////////////////////////////////////////////////////////////////////////
#define TAU_CALLPATH_MAP_TYPE long *, FunctionInfo *, TaultLong

/////////////////////////////////////////////////////////////////////////
/* The comparison function for callpath requires the TaultLong struct
 * to be defined. The operator() method in this struct compares two callpaths.
 * Since it only compares two arrays of longs (containing addresses), we can
 * look at the callpath depth as the first index in the two arrays and see if
 * they're equal. If they two arrays have the same depth, then we iterate
 * through the array and compare each array element till the end */
/////////////////////////////////////////////////////////////////////////
struct TaultLong {
  bool operator() (const long *l1, const long *l2) const {
   int i;
   /* first check 0th index (size) */
   if (l1[0] != l2[0]) return (l1[0] < l2[0]);
   /* they're equal, see the size and iterate */
   for (i = 1; i < l1[0] ; i++) {
     if (l1[i] != l2[i]) return l1[i] < l2[i];
   }
   return (l1[i] < l2[i]);
 }
};

/////////////////////////////////////////////////////////////////////////
// We use one global map to store the callpath information
/////////////////////////////////////////////////////////////////////////
map<TAU_CALLPATH_MAP_TYPE >& TheCallPathMap(void) {
  // to avoid initialization problems of non-local static variables
  static map<TAU_CALLPATH_MAP_TYPE > callpathmap;
  
  return callpathmap;
}

//////////////////////////////////////////////////////////////////////
long* TauFormulateComparisonArray(Profiler *p) {
  int depth = TauGetCallPathDepth();
  /* Create a long array with size depth+1. We need to put the depth
   * in it as the 0th index */
  long *ary = new long [depth+1];
  Profiler *current = p; /* argument */

  /* initialize the array */
  for (int j = 0; j < depth+1; j++) {
    ary[j] = 0L;
  }
  /* use the clean array now */
   
  if (ary) {
#ifdef TAU_PROFILEPHASE
/* if I'm in phase, go upto the profiler that has a phase. if you don't find
one then it is the top level profiler */
    ary[0] = 2; /* phase profiles are always 2 deep */
    ary[1] = (long) current->ThisFunction; 
    current = current->ParentProfiler;
    while (current != NULL) {
      ary[2] = (long) current->ThisFunction; 
      if (current->GetPhase()) { /* Found the parent phase! */
	break;
      } else {
	current = current->ParentProfiler;
      }
    }
#else /* TAU_PROFILEPHASE */
    int index = 1;
    while (current != NULL && depth != 0) {
      ary[index++] = (long) current->ThisFunction; 
      depth--;
      current = current->ParentProfiler;
    }
    ary[0] = index-1; /* set the depth */
#endif /* TAU_PROFILEPHASE */
  }
  return ary;
} 

//////////////////////////////////////////////////////////////////////
string *TauFormulateNameString(Profiler *p) {
  int depth = TauGetCallPathDepth();
  Profiler *current = p;
  string delimiter(" => ");
  string *name = new string("");

#ifdef TAU_PROFILEPHASE
  while (current != NULL) {
    if (current != p && (current->GetPhase() || (current->ParentProfiler == (Profiler *) NULL))) { 
      *name = current->ThisFunction->GetName() + string(" ") +
	current->ThisFunction->GetType() + delimiter + *name;
      break; /* come out of the loop, got phase name in */
    } else  {
      if (current == p) {
        *name = current->ThisFunction->GetName() + string (" ") + 
	  current->ThisFunction->GetType();
      }
      current = current->ParentProfiler;
    }
  }
  
#else /* TAU_PROFILEPHASE */
  while (current != NULL && depth != 0) {
    if (current != p) {
      *name =  current->ThisFunction->GetName() + string(" ") +
	current->ThisFunction->GetType() + delimiter + *name;
    } else {
      *name =  current->ThisFunction->GetName() + string (" ") + 
	current->ThisFunction->GetType();
    }
    current = current->ParentProfiler;
    depth --;
  }
#endif /* TAU_PROFILEPHASE */
  return name;
}



//////////////////////////////////////////////////////////////////////
// Member Function Definitions for class Profiler (contd).
//////////////////////////////////////////////////////////////////////

void Profiler::CallPathStart(int tid) {
  long *comparison = 0;
  // Start the callpath profiling
  if (ParentProfiler != NULL) { 
    // There is a callpath 
    if (ParentProfiler->CallPathFunction != 0) {
      ParentProfiler->CallPathFunction->IncrNumSubrs(tid);
    }
    DEBUGPROFMSG("Inside CallPath Start "<<ThisFunction->GetName()<<endl;);
    comparison = TauFormulateComparisonArray(this);
    
    map<TAU_CALLPATH_MAP_TYPE>::iterator it = TheCallPathMap().find(comparison);
    if (it == TheCallPathMap().end()) {
      RtsLayer::LockEnv();
      it = TheCallPathMap().find(comparison);
      if (it == TheCallPathMap().end()) {
	
	string *callpathname = TauFormulateNameString(this);
	DEBUGPROFMSG("Couldn't find string in map: "<<*comparison<<endl; );
	
	string grname = string("TAU_CALLPATH | ") + RtsLayer::PrimaryGroup(ThisFunction->GetAllGroups());
	CallPathFunction = new FunctionInfo(*callpathname, " ", 
					    ThisFunction->GetProfileGroup(), (const char*) grname.c_str(), true );
	TheCallPathMap().insert(map<TAU_CALLPATH_MAP_TYPE>::value_type(comparison, CallPathFunction));
      } else {
	CallPathFunction = (*it).second; 
	DEBUGPROFMSG("ROUTINE "<<(*it).second->GetName()<<" first = "<<(*it).first<<endl;);
	delete[] comparison; // free up memory when name is found
      }
      RtsLayer::UnLockEnv();
    } else {
      CallPathFunction = (*it).second; 
      DEBUGPROFMSG("ROUTINE "<<(*it).second->GetName()<<" first = "<<(*it).first<<endl;);
      delete[] comparison; // free up memory when name is found
    }
    
    DEBUGPROFMSG("FOUND Name = "<<CallPathFunction->GetName()<<endl;);
    
    // Set up metrics. Increment number of calls and subrs
    CallPathFunction->IncrNumCalls(tid);
    
    // Next, if this function is not already on the call stack, put it
    if (CallPathFunction->GetAlreadyOnStack(tid) == false) {
      AddInclCallPathFlag = true;
      // We need to add Inclusive time when it gets over as
      // it is not already on callstack.
      
      CallPathFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
    } else { 
      // the function is already on callstack, no need to add inclusive time
      AddInclCallPathFlag = false;
    }
    
  } else {
    // There's no callpath function when parentprofiler is null
    CallPathFunction = 0; 
  }
}

void Profiler::CallPathStop(double* TotalTime, int tid) {
  if (ParentProfiler != NULL) {
    if (AddInclCallPathFlag == true) { // The first time it came on call stack
      CallPathFunction->SetAlreadyOnStack(false, tid); // while exiting
      // And its ok to add both excl and incl times
      CallPathFunction->AddInclTime(TotalTime, tid);
    }
    
    CallPathFunction->AddExclTime(TotalTime, tid);  
    if (ParentProfiler->CallPathFunction != 0) {
      /* Increment the parent's NumSubrs and decrease its exclude time */
      ParentProfiler->CallPathFunction->ExcludeTime(TotalTime, tid);
    }
  }
}
  
/***************************************************************************
 * $RCSfile: TauCallPath.cpp,v $   $Author: amorris $
 * $Revision: 1.29 $   $Date: 2009/04/08 20:30:12 $
 * TAU_VERSION_ID: $Id: TauCallPath.cpp,v 1.29 2009/04/08 20:30:12 amorris Exp $ 
 ***************************************************************************/
