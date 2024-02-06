/****************************************************************************
e*			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: FunctionInfo.cpp				  **
**	Description 	: TAU Profiling Package				  **
*	Contact		: tau-team@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF 1
#include "Profile/Profiler.h"
#include <sstream>

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
#else
  #include <vector>
#endif //TAU_WINDOWS

#ifdef TAU_VAMPIRTRACE
#include <Profile/TauVampirTrace.h>
#else /* TAU_VAMPIRTRACE */
#ifdef TAU_EPILOG
#include "elg_trc.h"
#endif /* TAU_EPILOG */
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_SCOREP
#include <Profile/TauSCOREP.h>
#endif

#include <Profile/TauTrace.h>
#include <Profile/TauInit.h>
#include <Profile/TauUtil.h>
#include <Profile/TauPin.h>

#include <Profile/TauPluginInternals.h>

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

  // we now use the above FIvector, which subclasses vector
  //static vector<FunctionInfo*> FunctionDB;
  static FIvector FunctionDB;

  static int flag = 1;
  if (flag) {
    flag = 0;
    Tau_init_initializeTAU();
  }

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



//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class FunctionInfo
//////////////////////////////////////////////////////////////////////



static char *strip_tau_group(const char *ProfileGroupName) {
  char *source = strdup(ProfileGroupName);
  const char *find = "TAU_GROUP_";
  char *ptr;

  ptr = strstr(source,find);
  while (ptr != NULL) {
    char *endptr = ptr+strlen(find);
    while (*endptr != '\0') {
      *ptr++ = *endptr++;
    }
    *ptr = '\0';
    ptr = strstr(source,find);
  }
  return source;
}

// The next ID number for a FunctionInfo instance.
// This is used for the thread-local cache of FunctionMetrics.
std::atomic<uint64_t> FunctionInfo::next_id{0};

//thread_local FunctionInfo::FMetricListVector FunctionInfo::MetricThreadCache; //#Fixes opari bug, breaks pthreads
thread_local FunctionInfo::FMetricListVector_local FunctionInfo::MetricThreadCache;     //vector<FunctionInfo::FunctionMetrics*>* FunctionInfo::MetricThreadCache=new vector<FunctionInfo::FunctionMetrics*>; // One entry per instance
bool FunctionInfo::use_metric_tls = true;
//bool FunctionInfo::destructed = false;
//thread_local bool FunctionInfo::destructed_local = false;

//////////////////////////////////////////////////////////////////////
// FunctionInfoInit is called by all four forms of FunctionInfo ctor
//////////////////////////////////////////////////////////////////////
void FunctionInfo::FunctionInfoInit(TauGroup_t ProfileGroup, const char *ProfileGroupName, bool InitData)
{
  /* Make sure TAU is initialized */
  static bool flag = true;

  if (flag) {
    flag = false;
    Tau_init_initializeTAU();
#ifdef __PIN__
#if (!(defined (TAU_MPI) || defined(TAU_SHMEM)))
  if (RtsLayer::myNode() == -1) {
    TAU_PROFILE_SET_NODE(getpid());
  }
#endif /* MPI | SHMEM */
#endif /* __PIN__ */
  }
#ifdef TAU_SCOREP
  if (Tau_global_getLightsOut()) {
    TAU_VERBOSE("TAU<%d,%d>: FunctionInfoInit: Lights out... \n",RtsLayer::myNode(), RtsLayer::myThread());
    return;
  }
#endif /* TAU_SCOREP */


  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  // Use LockDB to avoid a possible race condition.
  RtsLayer::LockDB();

  function_info_id = ++next_id;

  //Need to keep track of all the groups this function is a member of.
  AllGroups = strip_tau_group(ProfileGroupName);

#ifndef TAU_WINDOWS
  // Necessary for signal-reentrancy to ensure the mmap memory manager
  //   is ready at this point.
#ifndef _AIX
  static bool mm_initialized = Tau_MemMgr_initIfNecessary();
  TAU_UNUSED(mm_initialized);
#endif /* _AIX */
#endif

  GroupName = strdup(RtsLayer::PrimaryGroup(AllGroups).c_str());

  // Since FunctionInfo constructor is called once for each function (static)
  // we know that it couldn't be already on the call stack.

  //Add function name to the name list.
  TauProfiler_theFunctionList(NULL, NULL, true, (const char *)GetName());
/*
 //TODO: Confirm that this init is no longer needed
  if (InitData) {
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      SetAlreadyOnStack(false, i); 
      SetCalls(i,0);
      NumCalls[i] = 0;
      NumSubrs[i] = 0;
      for (int j = 0; j < Tau_Global_numCounters; j++) {
        ExclTime[i][j] = 0;
        InclTime[i][j] = 0;
        dumpExclusiveValues[i][j] = 0;
        dumpInclusiveValues[i][j] = 0;
      }
    }
  } 
*/
  MyProfileGroup_ = ProfileGroup;

  // While accessing the global function database, lock it to ensure
  // an atomic operation in the push_back and size() operations.
  // Important in the presence of concurrent threads.
  TheFunctionDB().push_back(this);
  //DEBUG object address
  //printf("Address of function %s is %p !\n",(const char *)GetName(),(TheFunctionDB().back()));
  //DEBUG
  FunctionId = RtsLayer::GenerateUniqueId();

  StartAddr = 0;
  StopAddr = 0;

  // Initialize EBS structures. These will be created as and when necessary.
  //  pcHistogram = NULL;
  // *CWL* - this is an attempt to minimize the scenario where a sample
  //         requires the use of an actual malloc
  //         while in the middle of some other malloc call.
#ifndef TAU_WINDOWS
// #ifndef _AIX
  // create structure only if EBS is required.
  // Thread-safe, all (const char *) parameters. This check removes
  //   the need to create and allocate memory for EBS post-processed
  //   objects.
  if (TauEnv_get_ebs_enabled() &&
      !strstr(ProfileGroupName, "TAU_SAMPLE") &&
      !strstr(ProfileGroupName, "TAU_SAMPLE_CONTEXT") &&
      //!strstr(ProfileGroupName, "TAU_OMP_STATE") &&
      !strstr(ProfileGroupName, "TAU_UNWIND"))
 {//TODO: DYNAPROF. We need to flag this to do at initial thread initializations since we don't have all our threads at startup in a dynamic implementation
      setPathHistograms=true;
  }
    /*{
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      pathHistogram[i] = new TauPathHashTable<TauPathAccumulator>(i);
    }
  } else {
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      pathHistogram[i] = NULL;
    }
  }*/

  // Initialization of CallSite discovery structures.
  isCallSite = false;
  callSiteResolved = false;
  //  callSiteKeyId = 0; // Any value works.
  firstSpecializedFunction = NULL;

// #endif // _AIX
#endif // TAU_WINDOWS

#if defined(TAU_VAMPIRTRACE)
  string tau_vt_name(string(Name)+" "+string(Type));
  FunctionId = TAU_VT_DEF_REGION(tau_vt_name.c_str(), VT_NO_ID, VT_NO_LNO,
      VT_NO_LNO, GroupName, VT_FUNCTION);
  DEBUGPROFMSG("vt_def_region: "<<tau_vt_name<<": returns "<<FunctionId<<endl;);
#elif defined(TAU_EPILOG)
  string tau_elg_name(string(Name)+" "+string(Type));
  FunctionId = esd_def_region(tau_elg_name.c_str(), ELG_NO_ID, ELG_NO_LNO,
      ELG_NO_LNO, GroupName, ELG_FUNCTION);
  DEBUGPROFMSG("elg_def_region: "<<tau_elg_name<<": returns "<<FunctionId<<endl;);
#elif defined(TAU_SCOREP)
  string tau_silc_name(string(Name)+" "+string(Type));
  if (strstr(ProfileGroupName, "TAU_PHASE") != NULL) {
    FunctionId = SCOREP_Tau_DefineRegion( tau_silc_name.c_str(),
        SCOREP_TAU_INVALID_SOURCE_FILE,
        SCOREP_TAU_INVALID_LINE_NO,
        SCOREP_TAU_INVALID_LINE_NO,
        SCOREP_TAU_ADAPTER_COMPILER,
        SCOREP_TAU_REGION_PHASE
    );

  } else {
    FunctionId = SCOREP_Tau_DefineRegion( tau_silc_name.c_str(),
        SCOREP_TAU_INVALID_SOURCE_FILE,
        SCOREP_TAU_INVALID_LINE_NO,
        SCOREP_TAU_INVALID_LINE_NO,
        SCOREP_TAU_ADAPTER_COMPILER,
        SCOREP_TAU_REGION_FUNCTION
    );
  }
#endif

  DEBUGPROFMSG("nct "<< RtsLayer::myNode() <<","
      << RtsLayer::myContext() << ", " << RtsLayer::myThread()
      << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName()
      << " Group :  " << GetProfileGroup()
      << " Id :  " << GetFunctionId()
      << " Type : " << GetType() << endl;);

#ifdef TAU_PROFILEMEMORY
  {
    const int len = strlen(Name)+strlen(Type)+100;
    char * buff = new char[len];
    snprintf(buff, len,  "%s %s - Heap Memory Used (KB)", Name, Type);
    MemoryEvent = new tau::TauUserEvent(buff);
    delete buff;
  }
#endif

#ifdef TAU_PROFILEHEADROOM
  HeadroomEvent = new TauUserEvent(string(string(Name)+" "+Type+" - Memory Headroom Available (MB)").c_str());
#endif /* TAU_PROFILEHEADROOM */

//Already NULL by default
/*
#ifdef RENCI_STFF
  for (int t=0; t < TAU_MAX_THREADS; t++) {
    for (int i=0; i < TAU_MAX_COUNTERS; i++) {
      Signatures[t][i] = NULL;
    }
  }
#endif //RENCI_STFF
*/

  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.function_registration) {
    Tau_plugin_event_function_registration_data_t plugin_data;
    plugin_data.function_info_ptr = this;
    plugin_data.tid = RtsLayer::myThread();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_FUNCTION_REGISTRATION, GetName(), &plugin_data);
  }

  TauTraceSetFlushEvents(1);
  RtsLayer::UnLockDB();
}

//////////////////////////////////////////////////////////////////////
FunctionInfo::FunctionInfo(const char *name, const char *type, TauGroup_t ProfileGroup,
    const char *ProfileGroupName, bool InitData)
{
  DEBUGPROFMSG("FunctionInfo::FunctionInfo: MyProfileGroup_ = " << ProfileGroup << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
  Name = strdup(name);
  Type = strdup(type);
  FullName = NULL;
  DEBUGPROFMSG("FunctionInfo::FunctionInfo: MyProfileGroup_ = " << ProfileGroup << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
  FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const char *name, const string& type, TauGroup_t ProfileGroup,
    const char *ProfileGroupName, bool InitData)
{
  Name = strdup(name);
  Type = strdup(type.c_str());
  FullName = NULL;
  FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const string& name, const char * type,
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData) {
  Name = strdup(name.c_str());
  Type = strdup(type);
  FullName = NULL;

  FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const string& name, const string& type,
	TauGroup_t ProfileGroup , const char *ProfileGroupName, bool InitData) {

  Name = strdup(name.c_str());
  Type = strdup(type.c_str());
  FullName = NULL;

  FunctionInfoInit(ProfileGroup, ProfileGroupName, InitData);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::~FunctionInfo()
{
// Don't delete Name, Type - if dtor of static object dumps the data
// after all these function objects are destroyed, it can't get the
// name, and type.
//  delete [] Name;
//  delete [] Type;
  free(Name);
  free(Type);
  free(GroupName);
  free(AllGroups);
  Name = Type = GroupName = AllGroups = NULL;
#ifndef TAU_WINDOWS
#ifndef _AIX
  int maxThreads=getPathHistogramSize();
  for (int i = 0; i < maxThreads; i++) {
    delete GetPathHistogram(i);//pathHistogram[i];// TODO: DYNAPROF
  }
#endif /* _AIX */
#endif /* TAU_WINDOWS */
  TheSafeToDumpData() = 0;
}

double * FunctionInfo::GetExclTime(int tid){
  double * tmpCharPtr = (double *) malloc( sizeof(double) * Tau_Global_numCounters);
  for(int i=0;i<Tau_Global_numCounters;i++){
    tmpCharPtr[i] = GetExclTimeForCounter(tid,i);
  }
  return tmpCharPtr;
}

double * FunctionInfo::GetInclTime(int tid){
  double * tmpCharPtr = (double *) malloc( sizeof(double) * Tau_Global_numCounters);
  for(int i=0;i<Tau_Global_numCounters;i++){
    tmpCharPtr[i] = GetInclTimeForCounter(tid,i);
  }
  return tmpCharPtr;
}

//////////////////////////////////////////////////////////////////////
x_uint64 FunctionInfo::GetFunctionId(void) {
  // To avoid data races, we use a lock if the id has not been created
  if (FunctionId == 0) {
#ifdef DEBUG_PROF
    TAU_VERBOSE("Fid = 0! \n");
#endif // DEBUG_PROF
    while (FunctionId ==0) {
      RtsLayer::LockDB();
      RtsLayer::UnLockDB();
    }
  }
  return FunctionId;
}


//////////////////////////////////////////////////////////////////////
void FunctionInfo::ResetExclTimeIfNegative(int tid) {
  /* if exclusive time is negative (at Stop) we set it to zero during
     compensation. This function is used to reset it to zero for single
     and multiple counters */
  int i;
  for (i=0; i < Tau_Global_numCounters; i++) {
    if (GetExclTimeForCounter(tid,i) < 0) {
      SetExclTimeForCounter(0.0, tid, i); /* set each negative counter to zero */
    }
  }
  return;
}



//////////////////////////////////////////////////////////////////////
void tauCreateFI(void **ptr, const char *name, const char *type,
    TauGroup_t ProfileGroup, const char *ProfileGroupName)
{
  /* This is the entry point into TAU from PDT-instrumented C++ codes, so
   * make sure that TAU is ready to go before doing anything else! */
  static int do_this_once = Tau_init_initializeTAU();
  if (*ptr == 0) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

//Use The ENV lock here.
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::LockEnv();
#else
    RtsLayer::LockEnv();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::UnLockEnv();
#else
    RtsLayer::UnLockEnv();
#endif
  }
}

void tauCreateFI(void **ptr, const char *name, const string& type,
    TauGroup_t ProfileGroup, const char *ProfileGroupName)
{
  /* This is the entry point into TAU from PDT-instrumented C++ codes, so
   * make sure that TAU is ready to go before doing anything else! */
  static int do_this_once = Tau_init_initializeTAU();
  if (*ptr == 0) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::LockEnv();
#else
    RtsLayer::LockEnv();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::UnLockEnv();
#else
    RtsLayer::UnLockEnv();
#endif

  }
}

void tauCreateFI_signalSafe(void **ptr, const string& name, const char *type,
    TauGroup_t ProfileGroup, const char *ProfileGroupName)
{
  /* This is NOT the entry point into TAU from PDT-instrumented C++ codes!
   * this comes from creating the top level timer in TauCAPI.cpp. */
  if (*ptr == 0) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    /* KAH - Whoops!! We can't call "new" here, because malloc is not
     * safe in signal handling. therefore, use the special memory
     * allocation routines */
#if (!(defined (TAU_WINDOWS) || defined(_AIX) || defined(__PIN__)))
    *ptr = Tau_MemMgr_malloc(RtsLayer::unsafeThreadId(), sizeof(FunctionInfo));
    /*  now, use the pacement new function to create a object in
     *  pre-allocated memory. NOTE - this memory needs to be explicitly
     *  deallocated by explicitly calling the destructor.
     *  I think the best place for that is in the destructor for
     *  the hash table. */
    new(*ptr) FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
#else
    *ptr = (void *) new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
#endif
  }
}

void tauCreateFI(void **ptr, const string& name, const char *type,
    TauGroup_t ProfileGroup, const char *ProfileGroupName)
{
  /* This is the entry point into TAU from PDT-instrumented C++ codes, so
   * make sure that TAU is ready to go before doing anything else! */
  static int do_this_once = Tau_init_initializeTAU();
  if (*ptr == 0) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::LockEnv();
#else
    RtsLayer::LockEnv();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::UnLockEnv();
#else
    RtsLayer::UnLockEnv();
#endif
  }
}

void tauCreateFI(void **ptr, const string& name, const string& type, TauGroup_t ProfileGroup,
    const char *ProfileGroupName)
{
  /* This is the entry point into TAU from PDT-instrumented C++ codes, so
   * make sure that TAU is ready to go before doing anything else! */
  static int do_this_once = Tau_init_initializeTAU();
  if (*ptr == 0) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::LockEnv();
#else
    RtsLayer::LockEnv();
#endif
    if (*ptr == 0) {
      *ptr = new FunctionInfo(name, type, ProfileGroup, ProfileGroupName);
    }
#ifdef TAU_CHARM
    if (RtsLayer::myNode() != -1)
    RtsLayer::UnLockEnv();
#else
    RtsLayer::UnLockEnv();
#endif
  }
}


char const * FunctionInfo::GetFullName()
{
  if (!FullName) {
    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    ostringstream ostr;
    if (strlen(GetType()) > 0 && strcmp(GetType(), " ") != 0) {
      ostr << GetName() << " " << GetType() << ":GROUP:" << GetAllGroups();
    } else {
      ostr << GetName() << ":GROUP:" << GetAllGroups();
    }

    FullName = Tau_util_removeRuns(ostr.str().c_str());
  }
  return FullName;
}

/* EBS Sampling Profiles */

#ifndef TAU_WINDOWS
//#ifndef _AIX
void FunctionInfo::addPcSample(unsigned long *pcStack, int tid, double interval[TAU_MAX_COUNTERS])
{
  // Add to the mmap-ed histogram. We start with a temporary conversion. This
  //   becomes unnecessary once we stop using the vector.
  if (GetPathHistogram(tid) == NULL) return;
  TauPathAccumulator * accumulator = GetPathHistogram(tid)->get(pcStack);
  if (accumulator == NULL) {
    /* KAH - Whoops!! We can't call "new" here, because malloc is not
     * safe in signal handling. therefore, use the special memory
     * allocation routines */
    // accumulator = new TauPathAccumulator(1,interval);

    /* Use placement new to create a object in pre-allocated memory.
     * NOTE - this memory needs to be explicitly deallocated by calling the
     * destructor. I think the best place for that is in the destructor for
     * the hash table. */
    accumulator = (TauPathAccumulator*)Tau_MemMgr_malloc(tid, sizeof(TauPathAccumulator));
    new (accumulator) TauPathAccumulator(1, interval);

    bool success = GetPathHistogram(tid)->insert(pcStack, *accumulator);//pathHistogram[tid]->insert(pcStack, *accumulator);//TODO: DYNAPROF
    if (!success) {
      fprintf(stderr, "addPcSample: Failed to insert sample.\n");
    }
  } else {
    accumulator->count++;
    for (int i = 0; i < Tau_Global_numCounters; i++) {
      accumulator->accumulator[i] += interval[i];
    }
  }
}
//#endif // _AIX
#endif // TAU_WINDOWS

/***************************************************************************
 * $RCSfile: FunctionInfo.cpp,v $   $Author: amorris $
 * $Revision: 1.84 $   $Date: 2010/04/27 23:13:55 $
 * VERSION_ID: $Id: FunctionInfo.cpp,v 1.84 2010/04/27 23:13:55 amorris Exp $
 ***************************************************************************/
