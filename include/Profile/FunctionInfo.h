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


extern int Tau_Global_numCounters;
#define TAU_STORAGE(type, variable) type variable[TAU_MAX_THREADS]
#ifdef TAU_MULTIPLE_COUNTERS
#define TAU_MULTSTORAGE(type, variable) type variable[TAU_MAX_THREADS][MAX_TAU_COUNTERS]
#endif//TAU_MULTIPLE_COUNTERS

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
#include <Profile/KtauFuncInfo.h>
#endif /* TAUKTAU && TAUKTAU_MERGE */

#ifdef RENCI_STFF
#include "Profile/RenciSTFF.h"
#endif //RENCI_STFF

class TauUserEvent; 

class FunctionInfo
{
public:
  // Construct with the name of the function and its type.
  FunctionInfo(const char* name, const char * type, 
	       TauGroup_t ProfileGroup = TAU_DEFAULT, 
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true,
	       int tid = RtsLayer::myThread());
  FunctionInfo(const char* name, const string& type, 
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true,
	       int tid = RtsLayer::myThread());
  FunctionInfo(const string& name, const string& type, 
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true,
	       int tid = RtsLayer::myThread());
  FunctionInfo(const string& name, const char * type, 
	       TauGroup_t ProfileGroup = TAU_DEFAULT,
	       const char *ProfileGroupName = "TAU_DEFAULT", bool InitData = true,
	       int tid = RtsLayer::myThread());
  
  FunctionInfo(const FunctionInfo& X) ;
  // When we exit, we have to clean up.
  ~FunctionInfo();
  FunctionInfo& operator= (const FunctionInfo& X) ;

  void FunctionInfoInit(TauGroup_t PGroup, const char *PGroupName, 
			bool InitData, int tid );

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
  KtauFuncInfo* GetKtauFuncInfo(int tid) { return &(KernelFunc[tid]); }
#endif /* TAUKTAU && TAUKTAU_MERGE */

#ifndef TAU_MULTIPLE_COUNTERS 
  // Tell it about a function call finishing.
  inline void ExcludeTime(double t, int tid);
  // Removing void IncludeTime(double t, int tid);
  // and replacing with 
  inline void AddInclTime(double t, int tid);
  inline void AddExclTime(double t, int tid);
#else//TAU_MULTIPLE_COUNTERS
  inline void ExcludeTime(double *t, int tid);
  inline void AddInclTime(double *t, int tid);
  inline void AddExclTime(double *t, int tid);
#endif//TAU_MULTIPLE_COUNTERS

  inline void IncrNumCalls(int tid);
  inline void IncrNumSubrs(int tid);
  inline bool GetAlreadyOnStack(int tid);
  inline void SetAlreadyOnStack(bool value, int tid);  

  // A container of all of these.
  // The ctor registers with this.
        
  //static TAU_STD_NAMESPACE vector<FunctionInfo*> FunctionDB[TAU_MAX_THREADS];

#ifdef TAU_PROFILEMEMORY
  TauUserEvent * MemoryEvent;
  TauUserEvent * GetMemoryEvent(void) { return MemoryEvent; }
#endif // TAU_PROFILEMEMORY
#ifdef TAU_PROFILEHEADROOM
  TauUserEvent * HeadroomEvent;
  TauUserEvent * GetHeadroomEvent(void) { return HeadroomEvent; }
#endif // TAU_PROFILEHEADROOM

#ifdef RENCI_STFF
#ifndef TAU_MULTIPLE_COUNTERS
  // signatures for inclusive time for each thread
  TAU_STORAGE(ApplicationSignature*, Signatures);
  ApplicationSignature* GetSignature(int tid) {
    return Signatures[tid];
  }
  void SetSignature(ApplicationSignature *sig, int tid) {
    Signatures[tid] = sig;
  }
#else // TAU_MULTIPLE_COUNTERS
  // signatures for inclusive time for each counter in each thread
  TAU_MULTSTORAGE(ApplicationSignature*, Signatures);
  ApplicationSignature** GetSignature(int tid) {
    return Signatures[tid];
  }
#endif // TAU_MULTIPLE_COUNTERS
#endif //RENCI_STFF

private:
  // A record of the information unique to this function.
  // Statistics about calling this function.
	
#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
  TAU_STORAGE(KtauFuncInfo, KernelFunc);
#endif /* KTAU && KTAU_MERGE */

  TAU_STORAGE(long, NumCalls);
  TAU_STORAGE(long, NumSubrs);
#ifndef TAU_MULTIPLE_COUNTERS
  TAU_STORAGE(double, ExclTime);
  TAU_STORAGE(double, InclTime);
#else //TAU_MULTIPLE_COUNTERS
  TAU_MULTSTORAGE(double, ExclTime);
  TAU_MULTSTORAGE(double, InclTime);
#endif//TAU_MULTIPLE_COUNTERS
  TAU_STORAGE(bool, AlreadyOnStack);

  double dumpExclusiveValues[TAU_MAX_THREADS][MAX_TAU_COUNTERS];
  double dumpInclusiveValues[TAU_MAX_THREADS][MAX_TAU_COUNTERS];

public:
  char *Name;
  char *Type;
  char *GroupName;
  char *AllGroups;
  long FunctionId;


  inline double *getDumpExclusiveValues(int tid) {
    return dumpExclusiveValues[tid];
  }

  inline double *getDumpInclusiveValues(int tid) {
    return dumpInclusiveValues[tid];
  }

  // Cough up the information about this function.
  void SetName(string& str) { Name = strdup(str.c_str()); }
  const char* GetName() const { return Name; }
  void SetType(string& str) { Type = strdup(str.c_str()); }
  const char* GetType() const { return Type; }
  const char* GetPrimaryGroup() const { return GroupName; }
  const char* GetAllGroups() const { return AllGroups; }
  void SetPrimaryGroupName(const char *newname) { 
    GroupName = strdup(newname);
    AllGroups = strdup(newname); /* to make it to the profile */
  }
  void SetPrimaryGroupName(string newname) { 
    GroupName = strdup(newname.c_str()); 
    AllGroups = strdup(newname.c_str()); /* to make it to the profile */
  }

  long GetFunctionId() ;
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
#ifdef TAU_MULTIPLE_COUNTERS
    for(int i=0;i<Tau_Global_numCounters;i++) {
      ExclTime[tid][i] = 0;
    }
#else
    ExclTime[tid] = 0;
#endif
  }
  void SetInclTimeZero(int tid) {
#ifdef TAU_MULTIPLE_COUNTERS
    for(int i=0;i<Tau_Global_numCounters;i++) {
      InclTime[tid][i] = 0;
    }
#else
    InclTime[tid] = 0;
#endif
  }

#ifndef TAU_MULTIPLE_COUNTERS
  inline double GetExclTime(int tid) { return ExclTime[tid]; }
  inline void SetExclTime(int tid, double excltime) { ExclTime[tid] = excltime; }
  inline double GetInclTime(int tid) { return InclTime[tid]; }
  inline void SetInclTime(int tid, double incltime) { InclTime[tid] = incltime; }
#else//TAU_MULTIPLE_COUNTERS
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
  inline double GetExclTimeForCounter(int tid, int counter) { return InclTime[tid][counter]; }
#endif//TAU_MULTIPLE_COUNTERS

  TauGroup_t GetProfileGroup(int tid = RtsLayer::myThread()) const {return MyProfileGroup_[tid]; }
  void SetProfileGroup(TauGroup_t gr, int tid = RtsLayer::myThread()) {MyProfileGroup_[tid] = gr; }

private:
  TauGroup_t MyProfileGroup_[TAU_MAX_THREADS];
};

// Global variables
TAU_STD_NAMESPACE vector<FunctionInfo*>& TheFunctionDB(void); 
int& TheSafeToDumpData(void);
int& TheUsingDyninst(void);
int& TheUsingCompInst(void);

//
// For efficiency, make the timing updates inline.
//
#ifndef TAU_MULTIPLE_COUNTERS
inline void FunctionInfo::ExcludeTime(double t, int tid) { 
  // called by a function to decrease its parent functions time
  ExclTime[tid] -= t; // exclude from it the time spent in child function
}
	

inline void FunctionInfo::AddInclTime(double t, int tid) {
  InclTime[tid] += t; // Add Inclusive time
}

inline void FunctionInfo::AddExclTime(double t, int tid) {
  ExclTime[tid] += t; // Add Total Time to Exclusive time (-ve)
}
#else //TAU_MULTIPLE_COUNTERS
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
#endif //TAU_MULTIPLE_COUNTERS

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
void tauCreateFI(void **ptr, const char *name, const string& type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const string& name, const char *type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const string& name, const string& type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);


#endif /* _FUNCTIONINFO_H_ */
/***************************************************************************
 * $RCSfile: FunctionInfo.h,v $   $Author: amorris $
 * $Revision: 1.53 $   $Date: 2009/01/31 01:26:16 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.h,v 1.53 2009/01/31 01:26:16 amorris Exp $ 
 ***************************************************************************/
