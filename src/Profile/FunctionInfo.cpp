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
#include "Profile/pcxx_events.h"
#endif // TRACING_ON 



//////////////////////////////////////////////////////////////////////
// Instead of using a global var., use static inside a function  to
// ensure that non-local static variables are initialised before being
// used (Ref: Scott Meyers, Item 47 Eff. C++).
//////////////////////////////////////////////////////////////////////
vector<FunctionInfo*>& TheFunctionDB(int threadid)
{ // FunctionDB contains pointers to each FunctionInfo static object
  static vector<FunctionInfo*> FunctionDB;

  return FunctionDB;
}

//////////////////////////////////////////////////////////////////////
// It is not safe to call Profiler::StoreData() after 
// FunctionInfo::~FunctionInfo has been called as names are null
//////////////////////////////////////////////////////////////////////
int& TheSafeToDumpData()
{ 
  static int SafeToDumpData=1;

  return SafeToDumpData;
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
   int tid = RtsLayer::myThread();

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
	SetAlreadyOnStack(false, tid);

#ifdef PROFILE_STATS
	SumExclSqr[tid] = 0;
#endif //PROFILE_STATS

#ifdef PROFILE_CALLS
	ExclInclCallList = new list<pair<double, double> >();
#endif //PROFILE_CALLS
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	
        MyProfileGroup_ = ProfileGroup ;
	// While accessing the global function database, lock it to ensure
	// an atomic operation in the push_back and size() operations. 
	// Important in the presence of concurrent threads.
	RtsLayer::LockDB();
	TheFunctionDB().push_back(this);
#ifdef TRACING_ON
	// FOR Tracing, we should make the two a single operation 
	// when threads are supported for traces. There needs to be 
	// a lock in RtsLayer that can be locked while the push_back
	// and size operations are done (this should be atomic). 
	// Function Id is the index into the DB vector
	FunctionId = TheFunctionDB().size();
#endif //TRACING_ON
	RtsLayer::UnLockDB();
		
        DEBUGPROFMSG("nct "<< RtsLayer::myNode() <<"," 
	  << RtsLayer::myContext() << ", " << tid 
          << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName() 
	  << " Type : " << GetType() << endl;);

	return;
}
//////////////////////////////////////////////////////////////////////
FunctionInfo::FunctionInfo(const char *name, const char *type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{

      DEBUGPROFMSG("FunctionInfo::FunctionInfo: MyProfileGroup_ = " << MyProfileGroup_ 
        << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const char *name, string& type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, const char * type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, string& type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;
	FunctionInfoInit(ProfileGroup, ProfileGroupName);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::~FunctionInfo()
{
// Don't delete Name, Type - if dtor of static object dumps the data
// after all these function objects are destroyed, it can't get the 
// name, and type.
//	delete [] Name;
//	delete [] Type;
  TheSafeToDumpData() = 0;
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
 * $Revision: 1.7 $   $Date: 1998/09/17 15:26:00 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.cpp,v 1.7 1998/09/17 15:26:00 sameer Exp $ 
 ***************************************************************************/
