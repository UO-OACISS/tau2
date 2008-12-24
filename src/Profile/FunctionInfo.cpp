/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
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
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
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
#include <stdlib.h>

#if (!defined(TAU_WINDOWS))
 #include <unistd.h>
 #if (defined(POOMA_TFLOP) || !defined(TULIP_TIMERS))
  #include <sys/time.h>
 #else
  #ifdef TULIP_TIMERS 
   #include "Profile/TulipTimers.h"
  #endif //TULIP_TIMERS 
 #endif //POOMA_TFLOP

#else
  #include <vector>
#endif //TAU_WINDOWS

#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE 
#include <Profile/TauVampirTrace.h>
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
#include "elg_trc.h"
#else /* TAU_EPILOG */
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif // TRACING_ON 

#include <Profile/TauInit.h>
#include <Profile/TauCAPI.h>

//////////////////////////////////////////////////////////////////////
// The purpose of this subclass of vector is to give us a chance to execute
// some code when TheFunctionDB is destroyed.  For Dyninst, this is necessary
// when running with fortran programs
//////////////////////////////////////////////////////////////////////
class FIvector : public vector<FunctionInfo*> {
public: 
  ~FIvector() {
    Tau_destructor_trigger();
  }
};

//////////////////////////////////////////////////////////////////////
// Instead of using a global var., use static inside a function  to
// ensure that non-local static variables are initialised before being
// used (Ref: Scott Meyers, Item 47 Eff. C++).
//////////////////////////////////////////////////////////////////////
vector<FunctionInfo*>& TheFunctionDB(void)
{ // FunctionDB contains pointers to each FunctionInfo static object

  static int flag = InitializeTAU();

  // we now use the above FIvector, which subclasses vector
  //static vector<FunctionInfo*> FunctionDB;
  static FIvector FunctionDB;

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
// Set when uning Dyninst
//////////////////////////////////////////////////////////////////////
int& TheUsingDyninst()
{ 
  static int UsingDyninst=0;
  return UsingDyninst;
}

//////////////////////////////////////////////////////////////////////
// Set when uning Compiler Instrumentation
//////////////////////////////////////////////////////////////////////
int& TheUsingCompInst()
{ 
  static int UsingCompInst=0;
  return UsingCompInst;
}

#ifdef TAU_VAMPIRTRACE
//////////////////////////////////////////////////////////////////////
// Initialize VampirTrace Tracing package
//////////////////////////////////////////////////////////////////////
int TauInitVampirTrace(void)
{
  DEBUGPROFMSG("Calling vt_open"<<endl;);
  vt_open();
  return 1;
}
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_EPILOG 
//////////////////////////////////////////////////////////////////////
// Initialize EPILOG Tracing package
//////////////////////////////////////////////////////////////////////
int TauInitEpilog(void)
{
  DEBUGPROFMSG("Calling esd_open"<<endl;);
  esd_open();
  return 1;
}
#endif /* TAU_EPILOG */

//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class FunctionInfo
//////////////////////////////////////////////////////////////////////


static char *strip_tau_group(const char *ProfileGroupName) {
  char *source = strdup(ProfileGroupName);
  const char *find = "TAU_GROUP_";
  char *ptr;

  while (ptr = strstr(source,find)) {
    char *endptr = ptr+strlen(find);
    while (*endptr != NULL) {
      *ptr++ = *endptr++;
    }
  }
  return source;
}

//////////////////////////////////////////////////////////////////////
// FunctionInfoInit is called by all four forms of FunctionInfo ctor
//////////////////////////////////////////////////////////////////////
void FunctionInfo::FunctionInfoInit(TauGroup_t ProfileGroup, 
	const char *ProfileGroupName, bool InitData, int tid)
{
  //Need to keep track of all the groups this function is a member of.
  AllGroups = strip_tau_group(ProfileGroupName);

#ifdef TRACING_ON
  GroupName = RtsLayer::PrimaryGroup(AllGroups.c_str());
#endif //TRACING_ON

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
	RtsLayer::LockDB();
// Use LockDB to avoid a possible race condition.

	//Add function name to the name list.
	Profiler::theFunctionList(NULL, NULL, true, (const char *)GetName());

        if (InitData) {
      	  for (int i=0; i < TAU_MAX_THREADS; i++) {
     	    NumCalls[i] = 0;
	    SetAlreadyOnStack(false, i);
     	    NumSubrs[i] = 0;
#ifndef TAU_MULTIPLE_COUNTERS
	    ExclTime[i] = 0;
       	    InclTime[i] = 0;
#else //TAU_MULTIPLE_COUNTERS
	    for(int j=0;j<MAX_TAU_COUNTERS;j++){
	      ExclTime[i][j] = 0;
	      InclTime[i][j] = 0;
	    } 
#endif//TAU_MULTIPLE_COUNTERS
 	  }
	}

#ifdef PROFILE_STATS
	SumExclSqr[tid] = 0;
#endif //PROFILE_STATS

#ifdef PROFILE_CALLS
	ExclInclCallList = new list<pair<double, double> >();
#endif //PROFILE_CALLS
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	
	for (int i=0; i<TAU_MAX_THREADS; i++) {
	  MyProfileGroup_[i] = ProfileGroup;
	}
	// While accessing the global function database, lock it to ensure
	// an atomic operation in the push_back and size() operations. 
	// Important in the presence of concurrent threads.
	TheFunctionDB().push_back(this);
#ifdef TRACING_ON
#ifdef TAU_VAMPIRTRACE
        static int tau_vt_init=TauInitVampirTrace();
        string tau_vt_name(Name+" "+Type);
	FunctionId = vt_def_region(tau_vt_name.c_str(), VT_NO_ID, VT_NO_LNO,
		VT_NO_LNO, GroupName.c_str(), VT_FUNCTION);
	DEBUGPROFMSG("vt_def_region: "<<tau_vt_name<<": returns "<<FunctionId<<endl;);
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
        static int tau_elg_init=TauInitEpilog();
	string tau_elg_name(Name+" "+Type);
	FunctionId = esd_def_region(tau_elg_name.c_str(), ELG_NO_ID, ELG_NO_LNO,
		ELG_NO_LNO, GroupName.c_str(), ELG_FUNCTION);
	DEBUGPROFMSG("elg_def_region: "<<tau_elg_name<<": returns "<<FunctionId<<endl;);
#else /* TAU_EPILOG */
	// FOR Tracing, we should make the two a single operation 
	// when threads are supported for traces. There needs to be 
	// a lock in RtsLayer that can be locked while the push_back
	// and size operations are done (this should be atomic). 
	// Function Id is the index into the DB vector
	/* OLD:
	 * FunctionId = TheFunctionDB().size();
	 */
	FunctionId = RtsLayer::GenerateUniqueId();
	SetFlushEvents(tid);
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */
#endif //TRACING_ON
	RtsLayer::UnLockDB();
		
        DEBUGPROFMSG("nct "<< RtsLayer::myNode() <<"," 
	  << RtsLayer::myContext() << ", " << tid 
          << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName() 
          << " Group :  " << GetProfileGroup()
	  << " Type : " << GetType() << endl;);

#ifdef TAU_PROFILEMEMORY
	MemoryEvent = new TauUserEvent(string(Name+" "+Type+" - Heap Memory Used (KB)").c_str());
#endif /* TAU_PROFILEMEMORY */

#ifdef TAU_PROFILEHEADROOM
	HeadroomEvent = new TauUserEvent(string(Name+" "+Type+" - Memory Headroom Available (MB)").c_str());
#endif /* TAU_PROFILEHEADROOM */

#ifdef RENCI_STFF
#ifdef TAU_MULTIPLE_COUNTERS
    for (int t=0; t < TAU_MAX_THREADS; t++) {
        for (int i=0; i < MAX_TAU_COUNTERS; i++) {
            Signatures[t][i] = NULL;
        }
    }
 #else // TAU_MULTIPLE_COUNTERS
    for (int t=0; t < TAU_MAX_THREADS; t++) {
        Signatures[t] = NULL;
    }
#endif // TAU_MULTIPLE_COUNTERS
#endif //RENCI_STFF

	return;
}
//////////////////////////////////////////////////////////////////////
FunctionInfo::FunctionInfo(const char *name, const char *type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData,
	int tid)
{
      DEBUGPROFMSG("FunctionInfo::FunctionInfo: MyProfileGroup_ = " << ProfileGroup 
        << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
      Name = name;
      Type = type;

      FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData, tid);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const char *name, const string& type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData,
	int tid)
{
      Name = name;
      Type = type;

      FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData, tid);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const string& name, const char * type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData,
	int tid)
{
      Name = name;
      Type = type;

      FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData, tid);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const string& name, const string& type, 
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData,
	int tid)
{

      Name = name;
      Type = type;
 
      FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData, tid);
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

#ifdef TAU_MULTIPLE_COUNTERS
double * FunctionInfo::GetExclTime(int tid){
  double * tmpCharPtr = (double *) malloc( sizeof(double) * MAX_TAU_COUNTERS);
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    tmpCharPtr[i] = ExclTime[tid][i];
  }
  return tmpCharPtr;
}

double * FunctionInfo::GetInclTime(int tid){
  double * tmpCharPtr = (double *) malloc( sizeof(double) * MAX_TAU_COUNTERS);
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    tmpCharPtr[i] = InclTime[tid][i];
  }
  return tmpCharPtr;
}
#endif //TAU_MULTIPLE_COUNTERS


double *FunctionInfo::getInclusiveValues(int tid) {
  printf ("potentially evil\n");
#ifdef TAU_MULTIPLE_COUNTERS
  return InclTime[tid];
#else
  return &(InclTime[tid]);
#endif
}

double *FunctionInfo::getExclusiveValues(int tid) {
  printf ("potentially evil\n");
#ifdef TAU_MULTIPLE_COUNTERS
  return ExclTime[tid];
#else
  return &(ExclTime[tid]);
#endif
}

void FunctionInfo::getInclusiveValues(int tid, double *values) {
#ifdef TAU_MULTIPLE_COUNTERS
  for(int i=0; i<MAX_TAU_COUNTERS; i++) {
    values[i] = InclTime[tid][i];
  }
#else
  values[0] = InclTime[tid];
#endif
}

void FunctionInfo::getExclusiveValues(int tid, double *values) {
#ifdef TAU_MULTIPLE_COUNTERS
  for(int i=0; i<MAX_TAU_COUNTERS; i++) {
    values[i] = ExclTime[tid][i];
  }
#else
  values[0] = ExclTime[tid];
#endif
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
  	  printf("Fid = 0! \n");
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
void FunctionInfo::ResetExclTimeIfNegative(int tid) 
{ /* if exclusive time is negative (at Stop) we set it to zero during
     compensation. This function is used to reset it to zero for single
     and multiple counters */
#ifndef TAU_MULTIPLE_COUNTERS
	if (ExclTime[tid] < 0)
        {
          ExclTime[tid] = 0.0;
        }
#else /* TAU_MULTIPLE_COUNTERS */
	int i;
	for (i=0; i < MAX_TAU_COUNTERS; i++)
	{
	  if (ExclTime[tid][i] < 0)
          {
            ExclTime[tid][i] = 0.0; /* set each negative counter to zero */
          }
        }
#endif /* TAU_MULTIPLE_COUNTERS */
        return; 
}



//////////////////////////////////////////////////////////////////////
void tauCreateFI(FunctionInfo **ptr, const char *name, const char *type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName) {
  if (*ptr == 0) {

#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::LockDB();
#else
    RtsLayer::LockDB();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::UnLockDB();
#else
    RtsLayer::UnLockDB();
#endif

  }
}

void tauCreateFI(FunctionInfo **ptr, const char *name, const string& type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName) {
  if (*ptr == 0) {
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::LockDB();
#else
    RtsLayer::LockDB();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::UnLockDB();
#else
    RtsLayer::UnLockDB();
#endif
  }
}

void tauCreateFI(FunctionInfo **ptr, const string& name, const char *type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName) {
  if (*ptr == 0) {
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::LockDB();
#else
    RtsLayer::LockDB();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::UnLockDB();
#else
    RtsLayer::UnLockDB();
#endif
  }
}

void tauCreateFI(FunctionInfo **ptr, const string& name, const string& type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName) {
  if (*ptr == 0) {
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::LockDB();
#else
    RtsLayer::LockDB();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
      RtsLayer::UnLockDB();
#else
    RtsLayer::UnLockDB();
#endif
  }
}
/***************************************************************************
 * $RCSfile: FunctionInfo.cpp,v $   $Author: amorris $
 * $Revision: 1.57 $   $Date: 2008/12/24 09:50:08 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.cpp,v 1.57 2008/12/24 09:50:08 amorris Exp $ 
 ***************************************************************************/
