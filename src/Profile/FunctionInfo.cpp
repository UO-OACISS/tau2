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


#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

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
void FunctionInfo::FunctionInfoInit(TauGroup_t ProfileGroup, 
	const char *ProfileGroupName, bool InitData)
{
#ifdef TRACING_ON
	GroupName = RtsLayer::PrimaryGroup(ProfileGroupName);
#endif //TRACING_ON
   int tid = RtsLayer::myThread();

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
	RtsLayer::LockDB();
// Use LockDB to avoid a possible race condition.
        if (InitData) 
        {
	  SetAlreadyOnStack(false, tid);
      	  for (int i=0; i < TAU_MAX_THREADS; i++)
   	  {
// don't initialize NumCalls and AlreadyOnStack as there could be 
// data corruption. Inspite of the lock, while one thread is being 
// initialized, other thread may have started executing and setting 
// these values? 
     	    NumCalls[i] = 0;
	//  SetAlreadyOnStack(false, i);
     	    NumSubrs[i] = 0;
       	    ExclTime[i] = 0;
       	    InclTime[i] = 0;
 	  }
	}

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
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData)
{

      DEBUGPROFMSG("FunctionInfo::FunctionInfo: MyProfileGroup_ = " << MyProfileGroup_ 
        << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const char *name, string& type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData)
{
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, const char * type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData)
{
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;

	FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, string& type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData)
{
      if (ProfileGroup & RtsLayer::TheProfileMask()) {

        Name = name;
  	Type = type;
	FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
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
long FunctionInfo::GetFunctionId(void) 
{
   // To avoid data races, we use a lock if the id has not been created
  	if (FunctionId == 0)
	{
#ifdef DEBUG_PROF
  	  printf("Fid = 0! tid = %d\n", RtsLayer::myThread());
#endif // DEBUG_PROF
	  while (FunctionId ==0)
	  {
	    RtsLayer::LockDB();
	    RtsLayer::UnLockDB();
	  }
	}
	return FunctionId;
}
	    
//////////////////////////////////////////////////////////////////////

/***************************************************************************
 * $RCSfile: FunctionInfo.cpp,v $   $Author: sameer $
 * $Revision: 1.18 $   $Date: 1999/06/22 22:33:11 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.cpp,v 1.18 1999/06/22 22:33:11 sameer Exp $ 
 ***************************************************************************/
