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

#define TAU_STORAGE(type, variable) type variable[TAU_MAX_THREADS]
#define TAU_MULTSTORAGE(type, variable) type variable[TAU_MAX_THREADS][TAU_MAX_COUNTERS]

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
  KtauFuncInfo* GetKtauFuncInfo(int tid) { return &(KernelFunc[tid]); }
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
  TAU_MULTSTORAGE(ApplicationSignature*, Signatures);
  ApplicationSignature** GetSignature(int tid) {
    return Signatures[tid];
  }
#endif //RENCI_STFF

private:
  // A record of the information unique to this function.
  // Statistics about calling this function.

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
  TAU_STORAGE(KtauFuncInfo, KernelFunc);
#endif /* KTAU && KTAU_MERGE */

  TAU_STORAGE(long, NumCalls);
  TAU_STORAGE(long, NumSubrs);
  TAU_MULTSTORAGE(double, ExclTime);
  TAU_MULTSTORAGE(double, InclTime);
  TAU_STORAGE(bool, AlreadyOnStack);
  TAU_MULTSTORAGE(double, dumpExclusiveValues);
  TAU_MULTSTORAGE(double, dumpInclusiveValues);

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
  TauPathHashTable<TauPathAccumulator> *pathHistogram[TAU_MAX_THREADS];

  // For CallSite discovery
  bool isCallSite;
  bool callSiteResolved;
  unsigned long callSiteKeyId;
  FunctionInfo *firstSpecializedFunction;
  char *ShortenedName;
  void SetShortName(std::string& str) { ShortenedName = strdup(str.c_str()); }
  const char* GetShortName() const { return ShortenedName; }

  /* EBS Sampling Profiles */
  void addPcSample(unsigned long *pc, int tid, double interval[TAU_MAX_COUNTERS]);
//#endif /* _AIX */
#endif // TAU_WINDOWS

  inline double *getDumpExclusiveValues(int tid) {
    return dumpExclusiveValues[tid];
  }

  inline double *getDumpInclusiveValues(int tid) {
    return dumpInclusiveValues[tid];
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
  long GetCalls(int tid) { return NumCalls[tid]; }
  void SetCalls(int tid, long calls) { NumCalls[tid] = calls; }
  long GetSubrs(int tid) { return NumSubrs[tid]; }
  void SetSubrs(int tid, long subrs) { NumSubrs[tid] = subrs; }
  void ResetExclTimeIfNegative(int tid);


  double *getInclusiveValues(int tid);
  double *getExclusiveValues(int tid);

  void getInclusiveValues(int tid, double *values);
  void getExclusiveValues(int tid, double *values);

  void SetExclTimeZero(int tid) {
    for(int i=0;i<Tau_Global_numCounters;i++) {
      ExclTime[tid][i] = 0;
    }
  }
  void SetInclTimeZero(int tid) {
    for(int i=0;i<Tau_Global_numCounters;i++) {
      InclTime[tid][i] = 0;
    }
  }

  //Returns the array of exclusive counter values.
  //double * GetExclTime(int tid) { return ExclTime[tid]; }
  double *GetExclTime(int tid);
  double *GetInclTime(int tid);
  inline void SetExclTime(int tid, double *excltime) {
    for(int i=0;i<Tau_Global_numCounters;i++) {
      ExclTime[tid][i] = excltime[i];
    }
  }
  inline void SetInclTime(int tid, double *incltime) {
    for(int i=0;i<Tau_Global_numCounters;i++)
      InclTime[tid][i] = incltime[i];
  }


  inline void AddInclTimeForCounter(double value, int tid, int counter) { InclTime[tid][counter] += value; }
  inline void AddExclTimeForCounter(double value, int tid, int counter) { ExclTime[tid][counter] += value; }
  inline double GetInclTimeForCounter(int tid, int counter) { return InclTime[tid][counter]; }
  inline double GetExclTimeForCounter(int tid, int counter) { return ExclTime[tid][counter]; }

  TauGroup_t GetProfileGroup() const {return MyProfileGroup_; }
  void SetProfileGroup(TauGroup_t gr) {MyProfileGroup_ = gr; }

  bool IsThrottled() const {
    return ! (RtsLayer::TheEnableInstrumentation() && (MyProfileGroup_ & RtsLayer::TheProfileMask()));
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
  for (int i=0; i<Tau_Global_numCounters; i++) {
    ExclTime[tid][i] -= t[i];
  }
}


inline void FunctionInfo::AddInclTime(double *t, int tid) {
  for (int i=0; i<Tau_Global_numCounters; i++) {
    InclTime[tid][i] += t[i]; // Add Inclusive time
  }
}

inline void FunctionInfo::AddExclTime(double *t, int tid) {
  for (int i=0; i<Tau_Global_numCounters; i++) {
    ExclTime[tid][i] += t[i]; // Add Total Time to Exclusive time (-ve)
  }
}

inline void FunctionInfo::IncrNumCalls(int tid) {
  NumCalls[tid]++; // Increment number of calls
}

inline void FunctionInfo::IncrNumSubrs(int tid) {
  NumSubrs[tid]++;  // increment # of subroutines
}

inline void FunctionInfo::SetAlreadyOnStack(bool value, int tid) {
  AlreadyOnStack[tid] = value;
}

inline bool FunctionInfo::GetAlreadyOnStack(int tid) {
  return AlreadyOnStack[tid];
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
