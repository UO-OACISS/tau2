/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: FunctionInfo.h				  **
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

class FunctionInfo
{
public:
	// Construct with the name of the function and its type.
	FunctionInfo(const char* name, const char * type, 
          unsigned int ProfileGroup = TAU_DEFAULT, 
	  const char *ProfileGroupName = "TAU_DEFAULT");
	FunctionInfo(const char* name, string& type, 
	  unsigned int ProfileGroup = TAU_DEFAULT,
	  const char *ProfileGroupName = "TAU_DEFAULT");
	FunctionInfo(string& name, string& type, 
	  unsigned int ProfileGroup = TAU_DEFAULT,
	  const char *ProfileGroupName = "TAU_DEFAULT");
	FunctionInfo(string& name, const char * type, 
	  unsigned int ProfileGroup = TAU_DEFAULT,
	  const char *ProfileGroupName = "TAU_DEFAULT");

	FunctionInfo(const FunctionInfo& X) ;
	// When we exit, we have to clean up.
	~FunctionInfo();
        FunctionInfo& operator= (const FunctionInfo& X) ;

	void FunctionInfoInit(unsigned int PGroup, const char *PGroupName);
        

	// Tell it about a function call finishing.
	inline void ExcludeTime(double t);
	// Removing void IncludeTime(double t);
 	// and replacing with 
        inline void AddInclTime(double t);
	inline void AddExclTime(double t);
	inline void IncrNumCalls(void);
        inline void IncrNumSubrs(void);
	inline bool GetAlreadyOnStack(void);
	inline void SetAlreadyOnStack(bool value);  

	// A container of all of these.
	// The ctor registers with this.
        
	//static vector<FunctionInfo*> FunctionDB[TAU_MAX_THREADS];

#ifdef PROFILE_CALLS
	list < pair<double,double> > *ExclInclCallList; 
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	// time spent in each call

	int AppendExclInclTimeThisCall(double ex, double in); 
	// to ExclInclCallList
#endif // PROFILE_CALLS




#ifdef PROFILE_CALLSTACK 
  	double InclTime_cs;
  	double ExclTime_cs;
#endif  // PROFILE_CALLSTACK

private:

	// A record of the information unique to this function.
	// Statistics about calling this function.
	long NumCalls;
	long NumSubrs;
	double ExclTime;
	double InclTime;
	bool AlreadyOnStack; 
#ifdef PROFILE_STATS
	double SumExclSqr;
#endif // PROFILE_STATS

public:
	string Name;
	string Type;
	string GroupName;
	long   FunctionId;
	// Cough up the information about this function.
	const char* GetName() const { return Name.c_str(); }
	const char* GetType() const { return Type.c_str(); }
	const char* GetPrimaryGroup() const { return GroupName.c_str(); }
	long GetFunctionId() const { return FunctionId; }
	long GetCalls() const { return NumCalls; }
	long GetSubrs() const { return NumSubrs; }
	double GetExclTime() const { return ExclTime; }
	double GetInclTime() const { return InclTime; }
	unsigned int GetProfileGroup() const {return MyProfileGroup_; }
#ifdef PROFILE_STATS 
	double GetSumExclSqr() const { return SumExclSqr; }
	void AddSumExclSqr(double ExclSqr) { SumExclSqr += ExclSqr; }
#endif // PROFILE_STATS 

private:
	unsigned int MyProfileGroup_;
};

vector<FunctionInfo*>& TheFunctionDB(int threadid=RtsLayer::myThread()); 

//
// For efficiency, make the timing updates inline.
//
inline void 
FunctionInfo::ExcludeTime(double t)
{ // called by a function to decrease its parent functions time
	ExclTime -= t; // exclude from it the time spent in child function
}
	

inline void 
FunctionInfo::AddInclTime(double t)
{
  	InclTime += t; // Add Inclusive time
}

inline void
FunctionInfo::AddExclTime(double t)
{
 	ExclTime += t; // Add Total Time to Exclusive time (-ve)
}

inline void
FunctionInfo::IncrNumCalls(void)
{
	NumCalls++; // Increment number of calls
} 


inline void
FunctionInfo::IncrNumSubrs(void)
{
  NumSubrs++;  // increment # of subroutines
}

inline void
FunctionInfo::SetAlreadyOnStack(bool value)
{
	AlreadyOnStack = value;
}

inline bool
FunctionInfo::GetAlreadyOnStack(void)
{
	return AlreadyOnStack;
}
#endif /* _FUNCTIONINFO_H_ */
/***************************************************************************
 * $RCSfile: FunctionInfo.h,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 1998/04/24 00:23:34 $
 * POOMA_VERSION_ID: $Id: FunctionInfo.h,v 1.1 1998/04/24 00:23:34 sameer Exp $ 
 ***************************************************************************/
