/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: FunctionInfo.cpp				  **
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


#ifdef POOMA_USE_STANDARD_HEADERS
#include <iostream>
using namespace std;
#else
#include <iostream.h>
#endif

#include <stdio.h> 
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

#if (defined(POOMA_TFLOP) || !defined(TULIP_TIMERS))
#include <sys/time.h>
#else
#ifdef TULIP_TIMERS 
#include "Profile/TulipTimers.h"
#endif //TULIP_TIMERS 
#endif //POOMA_TFLOP

#ifdef TRACING_ON
#define PCXX_EVENT_SRC
#include "pcxx_events.h"
#endif // TRACING_ON 



//////////////////////////////////////////////////////////////////////
// Instead of using a global var., use static inside a function  to
// ensure that non-local static variables are initialised before being
// used (Ref: Scott Meyers, Item 47 Eff. C++).
//////////////////////////////////////////////////////////////////////
vector<FunctionInfo*>& TheFunctionDB(int threadid)
{
  static vector<FunctionInfo*> FunctionDB[TAU_MAX_THREADS];

  return FunctionDB[threadid];
}

//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class FunctionInfo
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// FunctionInfoInit is called by all four forms of FunctionInfo ctor
//////////////////////////////////////////////////////////////////////
void FunctionInfo::FunctionInfoInit(unsigned int ProfileGroup, 
	const char *ProfileGroupName)
{
#ifdef TRACING_ON
	GroupName = RtsLayer::PrimaryGroup(ProfileGroupName);
#endif //TRACING_ON
        NumCalls = 0;
        NumSubrs = 0;
  	ExclTime = 0;
  	InclTime = 0;

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
	SetAlreadyOnStack(false);

#ifdef PROFILE_STATS
	SumExclSqr = 0;
#endif //PROFILE_STATS

#ifdef PROFILE_CALLS
	ExclInclCallList = new list<pair<double, double> >();
#endif //PROFILE_CALLS
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	
        MyProfileGroup_ = ProfileGroup ;
	TheFunctionDB().push_back(this);
#ifdef TRACING_ON
	// Function Id is the index into the DB vector
	FunctionId = TheFunctionDB().size();
#endif //TRACING_ON
		
        DEBUGPROFMSG("Thr "<< RtsLayer::myNode() 
          << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName() 
	  << " Type : " << GetType() << endl;);

	return;
}
//////////////////////////////////////////////////////////////////////
FunctionInfo::FunctionInfo(const char *name, const char *type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const char *name, string& type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, const char * type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, string& type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;
	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const FunctionInfo& X) 
: Name(X.Name),
  Type(X.Type),
  NumCalls(X.NumCalls),
  NumSubrs(X.NumSubrs),
  ExclTime(X.ExclTime),
  InclTime(X.InclTime),
  MyProfileGroup_(X.MyProfileGroup_) 
{
	DEBUGPROFMSG("FunctionInfo::FunctionInfo (const FunctionInfo& X)"<<endl;);
	TheFunctionDB().push_back(this);
}
//////////////////////////////////////////////////////////////////////

FunctionInfo& FunctionInfo::operator= (const FunctionInfo& X) 
{
	DEBUGPROFMSG("FunctionInfo::operator= (const FunctionInfo& X)" << endl;);
   	Name = X.Name;
	Type = X.Type;
	NumCalls = X.NumCalls;
	NumSubrs = X.NumSubrs;
	ExclTime = X.ExclTime;
	InclTime = X.InclTime;
	MyProfileGroup_ = X.MyProfileGroup_;
	return (*this);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::~FunctionInfo()
{
// Don't delete Name, Type - if dtor of static object dumps the data
// after all these function objects are destroyed, it can't get the 
// name, and type.
//	delete [] Name;
//	delete [] Type;
}

#ifdef PROFILE_CALLS
//////////////////////////////////////////////////////////////////////

int FunctionInfo::AppendExclInclTimeThisCall(double ex, double in)
{
	ExclInclCallList->push_back(pair<double,double>(ex,in));
	return 1;
}

#endif //PROFILE_CALLS
//////////////////////////////////////////////////////////////////////

/***************************************************************************
 * $RCSfile: FunctionInfo.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 1998/04/24 00:06:49 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.cpp,v 1.1 1998/04/24 00:06:49 sameer Exp $ 
 ***************************************************************************/
