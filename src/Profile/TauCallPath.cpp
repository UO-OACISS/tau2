/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
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

int TauGetCallPathDepth(void)
{
  char *depth; 
  int value;
  if ((depth = getenv("TAU_CALLPATH_DEPTH")) != NULL)
  {
    value = atoi(depth);
    if (value > 1) 
    {
      return value;
    }
    else 
    {
      return 2; /* default value */
    }
  }
  else 
    return 2;
}

//////////////////////////////////////////////////////////////////////
// Global variables (wrapped in routines for static initialization)
/////////////////////////////////////////////////////////////////////////
#define TAU_CALLPATH_MAP_TYPE char *, FunctionInfo *, Taultstr
struct Taultstr
{
  bool operator()(const char* s1, const char* s2) const
  {
    return strcmp(s1, s2) < 0;
  }
};

map<TAU_CALLPATH_MAP_TYPE >& TheCallPathMap(void)
{ // to avoid initialization problems of non-local static variables
  static map<TAU_CALLPATH_MAP_TYPE > callpathmap;

  return callpathmap;
}

//////////////////////////////////////////////////////////////////////
string * TauFormulateComparisonString(Profiler *p)
{
  char str1[32];
  char str2[32];
  string *comparison = new string; 
  int depth = TauGetCallPathDepth();

  Profiler *current = p;
  while (current != NULL && depth != 0)
  {
    sprintf(str1, "%lx", current->ThisFunction);
    *comparison = *comparison + string(" ") + string(str1); 
    depth --;
    current = current->ParentProfiler;
  }
    
  DEBUGPROFMSG("Returning comparison = "<<*comparison<<endl;);
  return comparison;  
   
}

//////////////////////////////////////////////////////////////////////
string * TauFormulateNameString(Profiler *p)
{
  DEBUGPROFMSG("Inside TauFormulateNameString()"<<endl;);
  int depth = TauGetCallPathDepth();
  Profiler *current = p;
  string delimiter(" => ");
  string *name = new string("");

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
  DEBUGPROFMSG("TauFormulateNameString:Name: "<<*name <<endl;);
  return name;
}



//////////////////////////////////////////////////////////////////////
inline bool TauCallPathShouldBeProfiled(string *s)
{ 
  return true; // for now profile all callpaths
}



//////////////////////////////////////////////////////////////////////
// Member Function Definitions for class Profiler (contd).
//////////////////////////////////////////////////////////////////////

void Profiler::CallPathStart(int tid)
{
  string *comparison = 0; 
  // Start the callpath profiling
  if (ParentProfiler != NULL)
  { // There is a callpath 
    DEBUGPROFMSG("Inside CallPath Start "<<ThisFunction->GetName()<<endl;);
    comparison = TauFormulateComparisonString(this);
    DEBUGPROFMSG("Comparison string = "<<*comparison<<endl;);

    // Should I profile this path? 
    if (TauCallPathShouldBeProfiled(comparison))
    {
      map<TAU_CALLPATH_MAP_TYPE>::iterator it = TheCallPathMap().find((char *)(comparison->c_str()));
      if (it == TheCallPathMap().end())
      {
        string *callpathname = TauFormulateNameString(this);
        DEBUGPROFMSG("Couldn't find string in map: "<<*comparison<<endl; );
  	CallPathFunction = new FunctionInfo(*callpathname, " ", 
	  ThisFunction->GetProfileGroup(), "TAU_CALLPATH", true );
	TheCallPathMap().insert(map<TAU_CALLPATH_MAP_TYPE>::value_type((char *)(comparison->c_str()), CallPathFunction));
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
      ParentProfiler->CallPathFunction->IncrNumSubrs(tid);
      ParentProfiler->CallPathFunction->ExcludeTime(TotalTime, tid);
    }
    DEBUGPROFMSG("After IncrNumSubr"<<endl;);

  }
}
  
/***************************************************************************
 * $RCSfile: TauCallPath.cpp,v $   $Author: sameer $
 * $Revision: 1.8 $   $Date: 2003/05/13 23:30:59 $
 * TAU_VERSION_ID: $Id: TauCallPath.cpp,v 1.8 2003/05/13 23:30:59 sameer Exp $ 
 ***************************************************************************/
