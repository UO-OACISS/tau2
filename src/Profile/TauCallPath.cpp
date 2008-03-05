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
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
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

#define TAU_DEFAULT_CALLPATH_DEPTH 2
//////////////////////////////////////////////////////////////////////
// How deep should the callpath be? The default value is 2
//////////////////////////////////////////////////////////////////////
int& TauGetCallPathDepth(void)
{
  char *depth; 
  static int value = 0;

#ifdef TAU_PROFILEPHASE
  value = 2;
  return value;
#endif /* TAU_PROFILEPHASE */

  if (value == 0)
  {
    if ((depth = getenv("TAU_CALLPATH_DEPTH")) != NULL)
    {
      value = atoi(depth);
      if (value > 1) 
      {
        return value;
      }
      else 
      {
        value = TAU_DEFAULT_CALLPATH_DEPTH;
        return value; /* default value */
      }
    }
    else 
    {
      value = TAU_DEFAULT_CALLPATH_DEPTH;
      return value;
    }
  }
  else
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
struct TaultLong
{
  bool operator() (const long *l1, const long *l2) const
 {
   int i;
   /* first check 0th index (size) */
   if (l1[0] != l2[0]) return (l1[0] < l2[0]);
   /* they're equal, see the size and iterate */
   for (i = 0; i < l1[0] ; i++)
   {
     if (l1[i] != l2[i]) return l1[i] < l2[i];
   }
   return (l1[i] < l2[i]);
 }
};

/////////////////////////////////////////////////////////////////////////
// We use one global map to store the callpath information
/////////////////////////////////////////////////////////////////////////
map<TAU_CALLPATH_MAP_TYPE >& TheCallPathMap(void)
{ // to avoid initialization problems of non-local static variables
  static map<TAU_CALLPATH_MAP_TYPE > callpathmap;

  return callpathmap;
}

//////////////////////////////////////////////////////////////////////
long* TauFormulateComparisonArray(Profiler *p)
{
  int depth = TauGetCallPathDepth();
  /* Create a long array with size depth+1. We need to put the depth
   * in it as the 0th index */
  long *ary = new long [depth+1];

  int i = 0;
  int j; 
  Profiler *current = p; /* argument */

  /* initialize the array */
  for (j = 0; j < depth+1; j++)
  {
    ary[j] = 0L;
  }
  /* use the clean array now */
   
  if (ary)
  {
    ary[0] = depth; /* this tells us how deep it is */
#ifdef TAU_PROFILEPHASE
/* if I'm in phase, go upto the profiler that has a phase. if you don't find
one then it is the top level profiler */
    ary[1] = (long) current->ThisFunction; 
    current = current->ParentProfiler;
    while (current != NULL)
    {
      ary[2] = (long) current->ThisFunction; 
#ifdef DEBUG_PROF
      cout <<"Name = "<< current->ThisFunction->GetName()<<" "<<
	current->ThisFunction->GetType() <<" ";
#endif /* DEBUG_PROF */
      if (current->GetPhase()) /* Found the parent phase! */
      {
	break;
      }
      else
	current = current->ParentProfiler;
    }
#else /* TAU_PROFILEPHASE */
    while (current != NULL && depth != 0)
    {
      i++; /* increment i */
      ary[i] = (long) current->ThisFunction; 
      depth --;
      current = current->ParentProfiler;
    }
#endif /* TAU_PROFILEPHASE */
  }
  return ary;
} 

//////////////////////////////////////////////////////////////////////
string * TauFormulateNameString(Profiler *p)
{
  DEBUGPROFMSG("Inside TauFormulateNameString()"<<endl;);
  int depth = TauGetCallPathDepth();
  Profiler *current = p;
  string delimiter(" => ");
  string *name = new string("");

#ifdef TAU_PROFILEPHASE
  while (current != NULL)
  {
    if (current != p && (current->GetPhase() || (current->ParentProfiler == (Profiler *) NULL)))
    { 
      *name =  current->ThisFunction->GetName() + string(" ") +
	       current->ThisFunction->GetType() + delimiter + *name;
      break; /* come out of the loop, got phase name in */
    }
    else 
    { /* keep going */
      if (current == p) /* initial name */
      {
        *name =  current->ThisFunction->GetName() + string (" ") + 
	       current->ThisFunction->GetType();
      }
      current = current->ParentProfiler;
    }
  }

#else /* TAU_PROFILEPHASE */
  while (current != NULL && depth != 0)
  {
    if (current != p)
      *name =  current->ThisFunction->GetName() + string(" ") +
	       current->ThisFunction->GetType() + delimiter + *name;
    else
      *name =  current->ThisFunction->GetName() + string (" ") + 
	       current->ThisFunction->GetType();
    current = current->ParentProfiler;
    depth --; 
  }
#endif /* TAU_PROFILEPHASE */
  DEBUGPROFMSG("TauFormulateNameString:Name: "<<*name <<endl;);
  return name;
}



//////////////////////////////////////////////////////////////////////
inline bool TauCallPathShouldBeProfiled(long *s)
{ 
  return true; // for now profile all callpaths
}


//////////////////////////////////////////////////////////////////////
// Member Function Definitions for class Profiler (contd).
//////////////////////////////////////////////////////////////////////

void Profiler::CallPathStart(int tid)
{
//  string *comparison = 0; 
  long *comparison = 0;
  // Start the callpath profiling
  if (ParentProfiler != NULL)
  { // There is a callpath 
    if (ParentProfiler->CallPathFunction != 0) {
      ParentProfiler->CallPathFunction->IncrNumSubrs(tid);
    }
    DEBUGPROFMSG("Inside CallPath Start "<<ThisFunction->GetName()<<endl;);
    comparison = TauFormulateComparisonArray(this);
    DEBUGPROFMSG("Comparison string = "<<*comparison<<endl;);


    // Should I profile this path? 
    if (TauCallPathShouldBeProfiled(comparison))
    {

      map<TAU_CALLPATH_MAP_TYPE>::iterator it = TheCallPathMap().find(comparison);
      if (it == TheCallPathMap().end())
      {
	RtsLayer::LockEnv();
	it = TheCallPathMap().find(comparison);
	if (it == TheCallPathMap().end())
	  {

	    string *callpathname = TauFormulateNameString(this);
	    DEBUGPROFMSG("Couldn't find string in map: "<<*comparison<<endl; );
	    
	    string grname = string("TAU_CALLPATH | ") + RtsLayer::PrimaryGroup(ThisFunction->GetAllGroups());
	    CallPathFunction = new FunctionInfo(*callpathname, " ", 
						ThisFunction->GetProfileGroup(), (const char*) grname.c_str(), true );
	    TheCallPathMap().insert(map<TAU_CALLPATH_MAP_TYPE>::value_type(comparison, CallPathFunction));
	  } 
	else
	  {
	    CallPathFunction = (*it).second; 
	    DEBUGPROFMSG("ROUTINE "<<(*it).second->GetName()<<" first = "<<(*it).first<<endl;);
	    delete comparison; // free up memory when name is found
	  }
	RtsLayer::UnLockEnv();
      } 
      else
      {
 	CallPathFunction = (*it).second; 
	DEBUGPROFMSG("ROUTINE "<<(*it).second->GetName()<<" first = "<<(*it).first<<endl;);
        delete comparison; // free up memory when name is found
      }

      DEBUGPROFMSG("FOUND Name = "<<CallPathFunction->GetName()<<endl;);

      // Set up metrics. Increment number of calls and subrs
      CallPathFunction->IncrNumCalls(tid);

      // Next, if this function is not already on the call stack, put it
      if (CallPathFunction->GetAlreadyOnStack(tid) == false)   {
        AddInclCallPathFlag = true;
        // We need to add Inclusive time when it gets over as
        // it is not already on callstack.

        CallPathFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
      }
      else { // the function is already on callstack, no need to add
             // inclusive time
        AddInclCallPathFlag = false;
      }

    } // should this path be profiled ?
  }
  else
    CallPathFunction = 0; 
    // There's no callpath function when parentprofiler is null
}

#ifdef TAU_MULTIPLE_COUNTERS
void Profiler::CallPathStop(double* TotalTime, int tid)
#else // single counter
void Profiler::CallPathStop(double TotalTime, int tid)
#endif // TAU_MULTIPLE_COUNTERS
{
  if (ParentProfiler != NULL)
  {
    DEBUGPROFMSG("Inside CallPath Stop "<<ThisFunction->GetName()<<endl;);
    if (AddInclCallPathFlag == true) { // The first time it came on call stack
      CallPathFunction->SetAlreadyOnStack(false, tid); // while exiting

      DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
       << RtsLayer::myContext() << "," << tid  << " "
       << "CallPathStop: After SetAlreadyOnStack Going for AddInclTime" <<endl; );

      // And its ok to add both excl and incl times
      CallPathFunction->AddInclTime(TotalTime, tid);
    }

    CallPathFunction->AddExclTime(TotalTime, tid);  
    DEBUGPROFMSG("Before IncrNumSubr"<<endl;);
    if (ParentProfiler->CallPathFunction != 0)
    { /* Increment the parent's NumSubrs and decrease its exclude time */
      ParentProfiler->CallPathFunction->ExcludeTime(TotalTime, tid);
    }
    DEBUGPROFMSG("After IncrNumSubr"<<endl;);

  }
}
  
/***************************************************************************
 * $RCSfile: TauCallPath.cpp,v $   $Author: amorris $
 * $Revision: 1.24 $   $Date: 2008/03/05 23:53:24 $
 * TAU_VERSION_ID: $Id: TauCallPath.cpp,v 1.24 2008/03/05 23:53:24 amorris Exp $ 
 ***************************************************************************/
