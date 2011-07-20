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


extern "C" int Tau_Global_numCounters;
#define TAU_STORAGE(type, variable) type variable[TAU_MAX_THREADS]
#define TAU_MULTSTORAGE(type, variable) type variable[TAU_MAX_THREADS][TAU_MAX_COUNTERS]

#if defined(TAUKTAU) && defined(TAUKTAU_MERGE)
#include <Profile/KtauFuncInfo.h>
#endif /* TAUKTAU && TAUKTAU_MERGE */

#ifdef RENCI_STFF
#include "Profile/RenciSTFF.h"
#endif //RENCI_STFF

// For EBS Sampling Profiles with custom allocator support
#include <sys/types.h>
#include <unistd.h>
#include <map>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4018) //signed/unsigned mismatch
#pragma warning(disable:4290) // exception spec ignored
#endif
#include <new>
#include <list>
#include <algorithm>

template <typename T>
class ss_storage
{
  enum ss_defaults{init_size = 0xfffff};
 public:
  ss_storage():size_(sizeof(link)>sizeof(T)?init_size*sizeof(link):init_size*sizeof(T)),
    step_(sizeof(link)>sizeof(T)?sizeof(link):sizeof(T))
      {
	ss_mem_.push_back(new char[size_]);
	link *l;
	head_ = l = reinterpret_cast<link *>(*ss_mem_.begin());
	for(int i = 1; i<init_size; ++i){
	  l->next_ = reinterpret_cast<link *>(*ss_mem_.begin() + i*step_);
	  l = l->next_;
	}
	l->next_ = 0;
      }
    
  T* allocate()
    {
    link *l = head_;
    if(!l) grow();
    head_ = head_->next_;
    return reinterpret_cast<T *>(l);
  }

  void* allocate(size_t n)
    {
      link *l = head_, *conn, *ret;
      if(n%step_) n = step_*(n/step_+1);
      while(1){
	if(verify_memory_contiguous(l->next_, n/step_)){
	  conn = ret = l->next_;
	  for(unsigned int i=0; i<n/step_; ++i) conn = conn->next_;
	  l->next_ = conn;
	  return ret;
	}
	if(l->next_) l = l->next_;
	else {
	  grow();
	  l = head_;
	}
      }
    }

  void deallocate(void *p, size_t n)
  {
    if(n<=step_){
      link *l = head_;
      head_ = reinterpret_cast<link*>(p);
      head_->next_=l;
    }
    else{
      link *l = head_, *conn;
      head_ = reinterpret_cast<link*>(p);
      conn = head_;
      for(unsigned int i=0; i<n/step_; ++i){
	conn->next_ = reinterpret_cast<link*>(p) + i;
	conn = conn->next_;
      }
      conn->next_ = l;
    }
  }
  ~ss_storage()
    {
      std::for_each(ss_mem_.begin(), ss_mem_.end(), killer());
    }
  
 private:
  struct link
  {
    link *next_;
  };
  void grow()
  {
    char *new_chunk = new char[size_];
    ss_mem_.push_back(new_chunk);
    link *old_head = head_;
    link *l = reinterpret_cast<link *>(new_chunk);
    head_ = l;
    for(int i = 1; i<init_size; ++i){
      l->next_ = reinterpret_cast<link *>(new_chunk + i*step_);
      l = l->next_;
    }
    l->next_ = old_head;
  }

  bool verify_memory_contiguous(link *l, int n)
  {
    if(!l) return false;
    for(int i=0; i<n; ++i){
      if(l->next_){
	if(reinterpret_cast<char*>(l->next_) - reinterpret_cast<char*>(l) == step_){
	  l = l->next_;
	}
	else{
	  return false;
	}
      }
      else{
	return false;
      }
    }
    return true;
  }
  struct killer
  {
    void operator()(char *p){delete [] p;}
  };
  size_t size_;
  size_t step_;
  std::list<char *> ss_mem_;
  link *head_;
};
#ifdef _WIN32
#pragma warning(pop)
#endif

template <typename T> class ss_allocator;
template <> class ss_allocator<void>
{
 public:
  typedef void* pointer;
  typedef const void* const_pointer;
  // reference to void members are impossible.
  typedef void value_type;
    template <class U>
      struct rebind { typedef ss_allocator<U> other; };
};

namespace ss_alloc {
  inline void destruct(char *){}
  inline void destruct(wchar_t*){}
    template <typename T>
      inline void destruct(T *t){t->~T();}
} // namespace ss_alloc

template <typename T>
class ss_allocator
{
 public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;

  template <class U>
    struct rebind { typedef ss_allocator<U> other; };
  
  ss_allocator(){}
  pointer address(reference x) const {return &x;}
  const_pointer address(const_reference x) const {return &x;}
  pointer allocate(size_type size, ss_allocator<void>::const_pointer hint = 0) const
  {
    if(size == 1) return mem_.allocate();
    return static_cast<pointer>(mem_.allocate(size*sizeof(T)));
  }
  
  // For Dinkumware (VC6SP5):                                                 
  char *_Charalloc(size_type n){return static_cast<char*>(mem_.allocate(n));}
  // end Dinkumware
  
  template <class U> ss_allocator(const ss_allocator<U>&){}
  ss_allocator(const ss_allocator<T>&){}
  void deallocate(pointer p, size_type n) const
  {
    mem_.deallocate(p, n);
  }
  void deallocate(void *p, size_type n) const
  {
    mem_.deallocate(p, n);
  }
  size_type max_size() const throw() {return size_t(-1) / sizeof(value_type);}
  void construct(pointer p, const T& val)
  {
    new(static_cast<void*>(p)) T(val);
  }
  void construct(pointer p)
  {
    new(static_cast<void*>(p)) T();
  }
  void destroy(pointer p){ss_alloc::destruct(p);}
  //    static void dump() {mem_.dump();}
 private:
  static ss_storage<T> mem_;
};

template <typename T> ss_storage<T> ss_allocator<T>::mem_;

template <typename T, typename U>
  inline bool operator==(const ss_allocator<T>&, const ss_allocator<U>){return true;}

template <typename T, typename U>
  inline bool operator!=(const ss_allocator<T>&, const ss_allocator<U>){return false;}

// For VC6/STLPort 4-5-3 see /stl/_alloc.h, line 464
// "If custom allocators are being used without member template classes support:                                                                              
// user (on purpose) is forced to define rebind/get operations !!!"
#ifdef _WIN32
#define SS_ALLOC_CDECL __cdecl
#else
#define SS_ALLOC_CDECL
#endif

namespace std {
  template <class _Tp1, class _Tp2>
    inline ss_allocator<_Tp2>& SS_ALLOC_CDECL
    __stl_alloc_rebind(ss_allocator<_Tp1>& __a, const _Tp2*)
  {
    return (ss_allocator<_Tp2>&)(__a);
  }
  
  template <class _Tp1, class _Tp2>
    inline ss_allocator<_Tp2> SS_ALLOC_CDECL
    __stl_alloc_create(const ss_allocator<_Tp1>&, const _Tp2*)
  {
    return ss_allocator<_Tp2>();
  }

} // namespace std
// end STLPort

using namespace std;
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

  inline void ExcludeTime(double *t, int tid);
  inline void AddInclTime(double *t, int tid);
  inline void AddExclTime(double *t, int tid);

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

  double dumpExclusiveValues[TAU_MAX_THREADS][TAU_MAX_COUNTERS];
  double dumpInclusiveValues[TAU_MAX_THREADS][TAU_MAX_COUNTERS];

public:
  char *Name;
  char *Type;
  char *GroupName;
  char *AllGroups;
  long FunctionId;
  string *FullName;

  /* For EBS Sampling Profiles */

  //  map<caddr_t, unsigned int> *pcHistogram;
  map<caddr_t, unsigned int, std::less<caddr_t>, ss_allocator< std::pair<caddr_t, unsigned int> > > *pcHistogram;
  // For Intermediate FunctionInfo objects for groups of samples
  FunctionInfo *ebsIntermediate;
  // For FunctionInfo objects created specially for sample-based profiling 
  FunctionInfo *parentTauContext;

  /* EBS Sampling Profiles */
  void addPcSample(caddr_t pc);

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

  string *GetFullName(); /* created on demand, cached */


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
void tauCreateFI(void **ptr, const char *name, const string& type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const string& name, const char *type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);
void tauCreateFI(void **ptr, const string& name, const string& type, 
		 TauGroup_t ProfileGroup , const char *ProfileGroupName);


#endif /* _FUNCTIONINFO_H_ */
/***************************************************************************
 * $RCSfile: FunctionInfo.h,v $   $Author: amorris $
 * $Revision: 1.57 $   $Date: 2010/03/19 00:21:13 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.h,v 1.57 2010/03/19 00:21:13 amorris Exp $ 
 ***************************************************************************/
