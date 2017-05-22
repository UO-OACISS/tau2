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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#include <map.h>
#include <string.h>
#include <sstream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <Profile/Profiler.h>

using namespace std;
using namespace tau;

#include <Profile/TauPin.h>


#define TAU_CALLPATH_DEPTH_MIN 2


///////////////////////////////////////////////////////////////////////////////
// Orders callpaths by comparing arrays of profiler addresses stored as longs.
// The first element of the array is the array length.
///////////////////////////////////////////////////////////////////////////////
struct CallpathMapCompare
{
  bool operator()(long const * l1, long const * l2) const
  {
    int i;
    long const len = l1[0];
    if (len != l2[0]) return len < l2[0];
    for (i=0; i<len; ++i) {
      if (l1[i] != l2[i]) return l1[i] < l2[i];
    }
    return (l1[i] < l2[i]);
  }
};

///////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////
struct CallpathMap : public std::map<long *, FunctionInfo *, CallpathMapCompare>
{
  virtual ~CallpathMap() {
    Tau_destructor_trigger();
  }
};

///////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////
static CallpathMap & TheCallpathMap(void)
{
  static CallpathMap map;
  return map;
}


///////////////////////////////////////////////////////////////////////////////
// How deep should the callpath be? The default value is 2
///////////////////////////////////////////////////////////////////////////////
static int GetCallpathDepth(void)
{
#ifdef TAU_PROFILEPHASE
  return TAU_CALLPATH_DEPTH_MIN;
#else
  static int value = 0;
  if (value) return value;

  value = TauEnv_get_callpath_depth();
  if (value < TAU_CALLPATH_DEPTH_MIN) {
    value = TAU_CALLPATH_DEPTH_MIN;
  }
  return value;
#endif /* TAU_PROFILEPHASE */
}


///////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////
long * TauFormulateComparisonArray(Profiler * current)
{
  int depth = GetCallpathDepth();
  /* Create a long array with size depth+1. We need to put the depth
   * in it as the 0th index */
  long * ary = new long[depth+1];
  memset(ary, 0, (depth+1)*sizeof(long));

#ifdef TAU_PROFILEPHASE

  // if I'm in phase, go upto the profiler that has a phase. if you don't find
  // one then it is the top level profiler
  ary[0] = 2;    // phase profiles are always 2 deep
  ary[1] = Tau_convert_ptr_to_long(current->ThisFunction);
  current = current->ParentProfiler;
  while (current) {
    ary[2] = Tau_convert_ptr_to_long(current->ThisFunction);
    if (current->GetPhase()) break; // Found the parent phase!
    current = current->ParentProfiler;
  }

#else /* TAU_PROFILEPHASE */

  int i=1;
  // start writing to index 1, we fill in the depth after
  for(; current && depth; ++i) {
    ary[i] = Tau_convert_ptr_to_long(current->ThisFunction);
    current = current->ParentProfiler;
    --depth;
  }
  ary[0] = i - 1; // set the depth

#endif /* TAU_PROFILEPHASE */

  return ary;
}


///////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////
string TauFormulateNameString(Profiler * current)
{
  ostringstream buff;

  int depth = GetCallpathDepth();
  int i=0;

  Profiler ** path = (Profiler**)malloc(depth*sizeof(Profiler*));

#ifdef TAU_PROFILEPHASE

  // Phase profiles are always 2 deep
  // Store reversed to avoid string copies
  path[1] = current;
  Profiler *it = current;  /* iterate */
  while (it != NULL) {
    if ( it != current && (it->GetPhase() || (it->ParentProfiler == (Profiler *) NULL))) {
      path[0] = it;
      break; /* come out of the loop, got phase name in */
    } 
    it = it ->ParentProfiler;
  }

#else /* TAU_PROFILEPHASE */

  // Reverse the callpath to avoid string copies
  for (i=depth-1; current && i >= 0; --i) {
    path[i] = current;
    current = current->ParentProfiler;
  }
  ++i;  // Bump back up to the first valid path index

#endif /* TAU_PROFILEPHASE */

  // Now we construct the name string by appending to the buffer
  FunctionInfo * fi;
  for (; i < depth-1; ++i) {
    fi = path[i]->ThisFunction;
    buff << fi->GetName();
    if (strlen(fi->GetType()) > 0)
      buff << " " << fi->GetType();
    buff << " => ";
  }
  fi = path[i]->ThisFunction;
  buff << fi->GetName();
  if (strlen(fi->GetType()) > 0)
    buff << " " << fi->GetType();

  free((void*)path);

  // Return a new string object.
  // A smart STL implementation will not allocate a new buffer.
  return buff.str();
}


//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////
void Profiler::CallPathStart(int tid)
{
  // Start the callpath profiling
  if (ParentProfiler != NULL) {
    // There is a callpath 
    if (ParentProfiler->CallPathFunction) {
      ParentProfiler->CallPathFunction->IncrNumSubrs(tid);
    }

    DEBUGPROFMSG("Inside CallPath Start "<<ThisFunction->GetName()<<endl;);

    long * comparison = TauFormulateComparisonArray(this);

    // *CWL* - Send the path key off to be registered with CallSite discovery
    //         for later processing.
#ifndef TAU_WINDOWS
#ifndef _AIX
    if (TauEnv_get_callsite()) {
      CallSiteAddPath(comparison, tid);
    }
#endif /* _AIX */
#endif /* TAU_WINDOWS */

    RtsLayer::LockDB();
    CallpathMap & pathMap = TheCallpathMap();
    CallpathMap::iterator it = pathMap.find(comparison);
    if (it == pathMap.end()) {
      string callpathname = TauFormulateNameString(this);
      string grname = string("TAU_CALLPATH|") + RtsLayer::PrimaryGroup(ThisFunction->GetAllGroups());
      CallPathFunction = new FunctionInfo(callpathname, "", ThisFunction->GetProfileGroup(), grname.c_str(), true);
      pathMap[comparison] = CallPathFunction;
    } else {
      CallPathFunction = it->second;
      delete[] comparison;    // free up memory when name is found
    }
    RtsLayer::UnLockDB();

    // Set up metrics. Increment number of calls and subrs
    CallPathFunction->IncrNumCalls(tid);

    // Next, if this function is not already on the call stack, put it
    if (!CallPathFunction->GetAlreadyOnStack(tid)) {
      // We need to add Inclusive time when it gets over as it is not already on callstack.
      AddInclCallPathFlag = true;
      CallPathFunction->SetAlreadyOnStack(true, tid);    // it is on callstack now
    } else {
      // the function is already on callstack, no need to add inclusive time
      AddInclCallPathFlag = false;
    }

  } else {
    // There's no callpath function when parentprofiler is null
    CallPathFunction = 0;
  }
}

void Profiler::CallPathStop(double * TotalTime, int tid)
{
  if (ParentProfiler) {
    if (AddInclCallPathFlag) {    // The first time it came on call stack
      CallPathFunction->SetAlreadyOnStack(false, tid);    // while exiting
      // And its ok to add both excl and incl times
      CallPathFunction->AddInclTime(TotalTime, tid);
    }
    CallPathFunction->AddExclTime(TotalTime, tid);
    if (ParentProfiler->CallPathFunction) {
      // Increment the parent's NumSubrs and decrease its exclude time
      ParentProfiler->CallPathFunction->ExcludeTime(TotalTime, tid);
    }
  }
}

/***************************************************************************
 * $RCSfile: TauCallPath.cpp,v $   $Author: amorris $
 * $Revision: 1.32 $   $Date: 2010/02/22 18:38:38 $
 * TAU_VERSION_ID: $Id: TauCallPath.cpp,v 1.32 2010/02/22 18:38:38 amorris Exp $ 
 ***************************************************************************/
