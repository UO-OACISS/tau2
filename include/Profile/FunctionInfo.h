/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
 **	File 		: FunctionInfo.h				  **
 **	Description 	: TAU Profiling Package				  **
 **	Author		: Sameer Shende					  **
 **	Contact		: tau-bugs@cs.uoregon.edu                 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

#ifndef _FUNCTIONINFO_H_
#define _FUNCTIONINFO_H_

#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <atomic>

using namespace std;

/////////////////////////////////////////////////////////////////////
//
// class FunctionInfo
//
// This class is intended to be instantiated once per function
// (or other code block to be timed) as a static variable.
//
// It will be constructed the first time the function is called,
// and that constructor registers this object (and therefore the
// function) with the timer system.
//
//////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
extern int Tau_Global_numCounters;
extern int Tau_Global_numGPUCounters;
#ifdef __cplusplus
}
#endif /* __cplusplus */

//TODO: DYNATHREAD
#define TAU_STORAGE(type, variable, init) type variable = init 
#define TAU_MULTSTORAGE(type, variable, init) type variable[TAU_MAX_COUNTERS] = {init}

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
#include <Profile/KtauFuncInfo.h>
#endif /* TAUKTAU && TAUKTAU_MERGE */

#ifdef RENCI_STFF
#include "Profile/RenciSTFF.h"
#endif //RENCI_STFF

// For EBS Sampling Profiles with custom allocator support
#ifndef TAU_WINDOWS
#include <sys/types.h>
#include <unistd.h>
#include <map>

#include <Profile/TauPathHash.h>
#include "Profile/TauSampling.h"

#ifdef TAU_SS_ALLOC_SUPPORT
#include <Profile/TauSsAllocator.h>
#define SS_ALLOCATOR tau_ss_allocator
#else
#define SS_ALLOCATOR std::allocator
#endif //TAU_SS_ALLOC_SUPPORT
#endif //TAU_WINDOWS

// Forward declaration
namespace tau {
  class TauUserEvent;
}


class FunctionInfo
{
public:
  // Construct with the name of the function and its type.
  FunctionInfo(const char* name, const char * type,
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true);
  FunctionInfo(const char* name, const std::string& type,
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true);
  FunctionInfo(const std::string& name, const std::string& type,
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true);
  FunctionInfo(const std::string& name, const char * type,
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true);

  FunctionInfo(const FunctionInfo& X) ;
  // When we exit, we have to clean up.
  ~FunctionInfo();
  FunctionInfo& operator= (const FunctionInfo& X) ;

  void FunctionInfoInit(TauGroup_t PGroup, const char *PGroupName,
			bool InitData);

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
  KtauFuncInfo* GetKtauFuncInfo(int tid) { return &( getFunctionMetric(tid)->KernelFunc); }
#endif /* TAUKTAU && TAUKTAU_MERGE */

  inline void ExcludeTime(double *t, int tid);
  inline void AddInclTime(double *t, int tid);
  inline void AddExclTime(double *t, int tid);

  inline void IncrNumCalls(int tid);
  inline void IncrNumSubrs(int tid);
  inline bool GetAlreadyOnStack(int tid);
  inline void SetAlreadyOnStack(bool value, int tid);

#ifdef TAU_PROFILEMEMORY
  tau::TauUserEvent * MemoryEvent;
  tau::TauUserEvent * GetMemoryEvent(void) { return MemoryEvent; }
#endif // TAU_PROFILEMEMORY
#ifdef TAU_PROFILEHEADROOM
  tau::TauUserEvent * HeadroomEvent;
  tau::TauUserEvent * GetHeadroomEvent(void) { return HeadroomEvent; }
#endif // TAU_PROFILEHEADROOM

#ifdef RENCI_STFF
  // signatures for inclusive time for each counter in each thread
  ApplicationSignature** GetSignature(int tid) {
    return getFunctionMetric(tid)->Signatures;
  }
#endif //RENCI_STFF

private:
  bool setPathHistograms=false;
  // A record of the information unique to this function.
  // Statistics about calling this function.
  struct FunctionMetrics{//TODO: DYNATHREAD
#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
  TAU_STORAGE(KtauFuncInfo, KernelFunc);
#endif /* KTAU && KTAU_MERGE */

#ifdef RENCI_STFF
  // signatures for inclusive time for each counter in each thread
  TAU_MULTSTORAGE(ApplicationSignature*, Signatures, NULL);
#endif //RENCI_STFF

  TAU_STORAGE(long, NumCalls, 0);
  TAU_STORAGE(long, NumSubrs, 0);
  TAU_MULTSTORAGE(double, ExclTime, 0);
  TAU_MULTSTORAGE(double, InclTime, 0);
  TAU_STORAGE(bool, AlreadyOnStack, false);
  TAU_MULTSTORAGE(double, dumpExclusiveValues, 0);
  TAU_MULTSTORAGE(double, dumpInclusiveValues, 0);
  #if !defined(_AIX) && !defined(TAU_WINDOWS)
    //TODO: DYNAPROF the dynamic implementation of this needs more work
    TauPathHashTable<TauPathAccumulator> *pathHistogram=NULL;
  #endif /* _AIX */
  };


struct FMetricListVector : vector<FunctionMetrics *>{
    FMetricListVector() {
        // nothing
    }

    virtual ~FMetricListVector(){
	//destructed=true;
        Tau_destructor_trigger();
    }
};

// Metric list -- one entry per thread
FMetricListVector FMetricList;

// Mutex which protects FMetricList
std::mutex fInfoVectorMutex;

struct FMetricListVector_local : vector<FunctionMetrics *>{
    FMetricListVector_local() {
        // nothing
    }

    virtual ~FMetricListVector_local(){
        //destructed_local=true;
	//Tau_destructor_trigger();
    }
};

// Thread-local optimization for the FMetricList.
// We need a thread-local FMetricList, which means a thread-local member.
// C++ doesn't allow a thread_local member variable, only thread-local statics.
// Pthread TLS only allows a maximum of 1,024 such variables,
// and we need to support more timers than that.
//
// This keeps a sequential ID number for each FunctionInfo instance.
// This is used an an index into the static thread_local MetricThreadCache.
static thread_local FMetricListVector_local  MetricThreadCache;    //vector<FunctionMetrics*>* MetricThreadCache; // One entry per instance
//static thread_local FMetricListVector MetricThreadCache; // One entry per instance #Fixes opari bug, breaks pthreads
static std::atomic<uint64_t> next_id; // The next available ID; incremented when function_info_id is set.
uint64_t function_info_id; // This is set in FunctionInfo::FunctionInfoInit()
static bool use_metric_tls; // This is set to false to disable the thread-local cache during shutdown.
//static bool destructed;
//static thread_local bool destructed_local;
//bool& Tau_is_destroyed(void);
// getFunctionMetric(tid) returns the pointer to this instance's FunctionMetric 
// for the given tid. Uses thread-local cache if tid = this thread.
FunctionMetrics* getFunctionMetric(unsigned int tid){
    FunctionMetrics* MOut = NULL;
    /*if(destructed||destructed_local){
	    fprintf(stderr,"TAU Warning: getting function from destructed thread!\n");
	    return MOut;
    }*/
    static thread_local const unsigned int local_tid = RtsLayer::myThread();


    // Use thread-local optimization if the current thread is requesting its own metrics.
    // After the first time a thread requests its own metrics, we no longer have to lock.
    // (If requesting a *different* thread's metrics, we have to use the slow path.)
    // Also don't use the cache during shutdown -- it might have been destructed already,
    // but we can't put a destructor trigger on MetricThreadCache because they are *also*
    // destructed when a thread exits.
    if(use_metric_tls && tid == local_tid){//&& tid!=0 && use_metric_tls && !destructed && !destructed_local) {
        if(MetricThreadCache.size() > function_info_id) {
            MOut = MetricThreadCache.operator[](function_info_id);
            if(MOut != NULL) {
                return MOut;
            }
        }
    }
    // Not in thread-local cache, or cache not searched.
    // Create a new FunctionMetrics instance.
    std::lock_guard<std::mutex> guard(fInfoVectorMutex);
    while(FMetricList.size()<=tid){
	FMetricList.push_back(new FunctionMetrics());
#ifndef TAU_WINDOWS       
        if(setPathHistograms){//TODO: DYNAPROF
            int topThread=FMetricList.size()-1;
            FMetricList[topThread]->pathHistogram=new TauPathHashTable<TauPathAccumulator>(topThread);
        }
#endif        
    }
    
    MOut=FMetricList[tid];

    // Use thread-local optimization if the current thread is requesting its own metrics.
    if(use_metric_tls && tid == local_tid) {//tid !=0 && use_metric_tls && 
        // Ensure the FMetricList vector is long enough to accomodate the new cached item.
        while(MetricThreadCache.size() <= function_info_id) {
            MetricThreadCache.push_back(NULL);
        }    
        // Store the FunctionMetrics pointer in the thread-local cache
        MetricThreadCache.operator[](function_info_id) = MOut;
    }
    if(MOut==NULL){
	    fprintf(stderr,"TAU Warning: getFunctionMetric returning NULL!\n");
    }
    return MOut;

}

public:
  char *Name;
  char *Type;
  char *GroupName;
  char *AllGroups;
  char const * FullName;
  x_uint64 FunctionId;
  unsigned long StartAddr;
  unsigned long StopAddr;

  /* For EBS Sampling Profiles */
  // *CWL* - these need to be per-thread structures, just like the
  //         the data values above.
  //         They will also potentially need per-counter information
  //         eventually.
  //  map<unsigned long, unsigned int> *pcHistogram;
#ifndef TAU_WINDOWS
//#ifndef _AIX
  //TauPathHashTable<TauPathAccumulator> *pathHistogram[TAU_MAX_THREADS];

  // For CallSite discovery
  bool isCallSite;
  bool callSiteResolved;
  unsigned long callSiteKeyId;
  FunctionInfo *firstSpecializedFunction;
  char *ShortenedName;
  void SetShortName(std::string& str) { ShortenedName = strdup(str.c_str()); }
  const char* GetShortName() const { return ShortenedName; }
  inline TauPathHashTable<TauPathAccumulator>* GetPathHistogram(int tid){//TODO: DYNAPROF
    return getFunctionMetric(tid)->pathHistogram;
  }
  inline int getPathHistogramSize()
  {
	return FMetricList.size();
  }

  /* EBS Sampling Profiles */
  void addPcSample(unsigned long *pc, int tid, double interval[TAU_MAX_COUNTERS]);
//#endif /* _AIX */
#endif // TAU_WINDOWS

  inline double *getDumpExclusiveValues(int tid) {
    return getFunctionMetric(tid)->dumpExclusiveValues;
  }

  inline double *getDumpInclusiveValues(int tid) {
    return getFunctionMetric(tid)->dumpInclusiveValues;
  }

  // Cough up the information about this function.
  void SetName(std::string & str) { Name = strdup(str.c_str()); }
  const char* GetName() const { return Name; }

  void SetType(char const * str) {
    Type = strdup(str);
  }
  char const * GetType() const {
    return Type;
  }

  const char* GetPrimaryGroup() const { return GroupName; }
  const char* GetAllGroups() const { return AllGroups; }
  void SetPrimaryGroupName(const char *newname) {
    GroupName = strdup(newname);
    AllGroups = strdup(newname); /* to make it to the profile */
  }
  void SetPrimaryGroupName(std::string newname) {
    GroupName = strdup(newname.c_str());
    AllGroups = strdup(newname.c_str()); /* to make it to the profile */
  }

  char const * GetFullName(); /* created on demand, cached */

  x_uint64 GetFunctionId() ;
  long GetCalls(int tid) { return getFunctionMetric(tid)->NumCalls; }
  void SetCalls(int tid, long calls) { getFunctionMetric(tid)->NumCalls = calls; }
  long GetSubrs(int tid) {  return getFunctionMetric(tid)->NumSubrs; }
  void SetSubrs(int tid, long subrs) { getFunctionMetric(tid)->NumSubrs = subrs; }
  void ResetExclTimeIfNegative(int tid);


  double *getInclusiveValues(int tid){
    printf ("TAU: Warning, potentially evil function called\n");
    return getFunctionMetric(tid)->InclTime;
  }
  
  double *getExclusiveValues(int tid){
    printf ("TAU: Warning, potentially evil function called\n");
    return getFunctionMetric(tid)->ExclTime;
  }

  void getInclusiveValues(int tid, double *values){
    FunctionMetrics* tmp = getFunctionMetric(tid);
    if(tmp==NULL)return;
 	  
    for(int i=0; i<Tau_Global_numCounters; i++) {
        values[i] = tmp->InclTime[i];
    }
  }
  void getExclusiveValues(int tid, double *values){
    FunctionMetrics* tmp = getFunctionMetric(tid);
    if(tmp==NULL)return;
	  
    for(int i=0; i<Tau_Global_numCounters; i++) {
        values[i] = tmp->ExclTime[i];
    }
  }

  void SetExclTimeZero(int tid) {
    FunctionMetrics* tmp = getFunctionMetric(tid);
    if(tmp==NULL)return;
	  
    for(int i=0;i<Tau_Global_numCounters;i++) {
      tmp->ExclTime[i] = 0;
    }
  }
  void SetInclTimeZero(int tid) {
    FunctionMetrics* tmp = getFunctionMetric(tid);
    if(tmp==NULL)return;
	  
    for(int i=0;i<Tau_Global_numCounters;i++) {
      tmp->InclTime[i] = 0;
    }
  }

  //Returns the array of exclusive counter values.
  //double * GetExclTime(int tid) { return ExclTime[tid]; }
  double *GetExclTime(int tid);
  double *GetInclTime(int tid);
  inline void SetExclTime(int tid, double *excltime) {
    FunctionMetrics* tmp = getFunctionMetric(tid);
      if(tmp==NULL)return;
    for(int i=0;i<Tau_Global_numCounters;i++) {
      tmp->ExclTime[i] = excltime[i];
    }
  }
  inline void SetInclTime(int tid, double *incltime) {
    FunctionMetrics* tmp = getFunctionMetric(tid);
    if(tmp==NULL)return;	  
    for(int i=0;i<Tau_Global_numCounters;i++)
      tmp->InclTime[i] = incltime[i];
  }


  inline void AddInclTimeForCounter(double value, int tid, int counter) { getFunctionMetric(tid)->InclTime[counter] += value; }
  inline void AddExclTimeForCounter(double value, int tid, int counter) { getFunctionMetric(tid)->ExclTime[counter] += value; }
  inline void SetExclTimeForCounter(double value, int tid, int counter) { getFunctionMetric(tid)->ExclTime[counter] = value; }
  inline double GetInclTimeForCounter(int tid, int counter) { return getFunctionMetric(tid)->InclTime[counter]; }
  inline double GetExclTimeForCounter(int tid, int counter) { return getFunctionMetric(tid)->ExclTime[counter]; }

  TauGroup_t GetProfileGroup() const {return MyProfileGroup_; }
  void SetProfileGroup(TauGroup_t gr) {MyProfileGroup_ = gr; }

  bool IsThrottled() const {

	if (!RtsLayer::TheEnableInstrumentation()) {
		return true;
	}
	// Get a reference to the global blacklist mask.
  	const TauGroup_t& blacklist = RtsLayer::TheProfileBlackMask();
	const bool exclude_default =
        RtsLayer::TheExcludeDefaultGroup().load(std::memory_order_relaxed);

  	if (blacklist == 0 && !exclude_default ) {
    // ===================================================================
    // FAST PATH: The blacklist is not being used at all.
    // Just confirm that the profile group includes a bit in the mask
    // ===================================================================
    	return !(MyProfileGroup_ & RtsLayer::TheProfileMask());
  	} else {
    // ===================================================================
    // SLOW PATH: The blacklist is active.
    // This will only be executed if a user has explicitly disabled an event.
    // ===================================================================
    bool is_whitelisted = (MyProfileGroup_ & RtsLayer::TheProfileMask());
    bool is_blacklisted =  ((MyProfileGroup_ & blacklist) && (MyProfileGroup_ != TAU_DEFAULT)); //Everyt bit is set in TAU_DEFAULT so we exclude it from the TAU_EXCLUDE check
    if(is_blacklisted){
	    printf("FOUND BLACKLIST!!!\n");
    }
	
	bool blacklisted_as_default =
            (exclude_default && (MyProfileGroup_ == TAU_DEFAULT));
    return !is_whitelisted || is_blacklisted || blacklisted_as_default;
  }
  }

  static void disable_metric_cache() {
    use_metric_tls = false;
  }

private:
  TauGroup_t MyProfileGroup_;
};

// Global variables
std::vector<FunctionInfo*>& TheFunctionDB(void);
int& TheSafeToDumpData(void);
int& TheUsingDyninst(void);
int& TheUsingCompInst(void);

//
// For efficiency, make the timing updates inline.
//
inline void FunctionInfo::ExcludeTime(double *t, int tid) {
  // called by a function to decrease its parent functions time
  // exclude from it the time spent in child function
  FunctionMetrics* tmp = getFunctionMetric(tid);
  if(tmp==NULL)return;
  for (int i=0; i<Tau_Global_numCounters; i++) {
    tmp->ExclTime[i] -= t[i];
  }
}


inline void FunctionInfo::AddInclTime(double *t, int tid) {
  FunctionMetrics* tmp = getFunctionMetric(tid);
  if(tmp==NULL)return;
  for (int i=0; i<Tau_Global_numCounters; i++) {
    tmp->InclTime[i] += t[i]; // Add Inclusive time
  }
}

inline void FunctionInfo::AddExclTime(double *t, int tid) {
  FunctionMetrics* tmp = getFunctionMetric(tid);
  if(tmp==NULL)return;	
  for (int i=0; i<Tau_Global_numCounters; i++) {
    tmp->ExclTime[i] += t[i]; // Add Total Time to Exclusive time (-ve)
  }
}

inline void FunctionInfo::IncrNumCalls(int tid) {
  getFunctionMetric(tid)->NumCalls++; // Increment number of calls
} 

inline void FunctionInfo::IncrNumSubrs(int tid) {
  getFunctionMetric(tid)->NumSubrs++;  // increment # of subroutines
}

inline void FunctionInfo::SetAlreadyOnStack(bool value, int tid) {
  FunctionMetrics* tmp = getFunctionMetric(tid);
  if(tmp==NULL)return;
  tmp->AlreadyOnStack = value;
}

inline bool FunctionInfo::GetAlreadyOnStack(int tid) {
  return getFunctionMetric(tid)->AlreadyOnStack;
}


void tauCreateFI(void **ptr, const char *name, const char *type,
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const char *name, const std::string& type,
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const std::string& name, const char *type,
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const std::string& name, const std::string& type,
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI_signalSafe(void **ptr, const std::string& name, const char *type,
         TauGroup_t ProfileGroup, const char *ProfileGroupName);


#endif /* _FUNCTIONINFO_H_ */
/***************************************************************************
 * $RCSfile: FunctionInfo.h,v $   $Author: amorris $
 * $Revision: 1.57 $   $Date: 2010/03/19 00:21:13 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.h,v 1.57 2010/03/19 00:21:13 amorris Exp $
 ***************************************************************************/
