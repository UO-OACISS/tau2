/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.h					  **
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
**	Documentation	: See http://www.acl.lanl.gov/tau	          **
***************************************************************************/
#ifndef PROFILER_H
#define PROFILER_H

/* TAU PROFILING GROUPS. More will be added later.  */
#define TAU_DEFAULT 		0xffffffff   /* All profiling groups enabled*/
#define TAU_MESSAGE 		0x00000001   /* Message 'm'*/
#define TAU_PETE    		0x00000002   /* PETE    'p' */
#define TAU_VIZ     		0x00000004   /* ACLVIZ  'v' */
#define TAU_ASSIGN  		0x00000008   /* ASSIGN Expression Evaluation 'a' */
#define TAU_IO  		0x00000010   /* IO routines 'i' */
#define TAU_FIELD   		0x00000020   /* Field Classes 'f' */
#define TAU_LAYOUT  		0x00000040   /* Field Layout  'l' */
#define TAU_SPARSE  		0x00000080   /* Sparse Index  's' */
#define TAU_DOMAINMAP   	0x00000100   /* Domain Map    'd' */
#define TAU_UTILITY     	0x00000200   /* Utility       'Ut' */
#define TAU_REGION      	0x00000400   /* Region        'r' */
#define TAU_PARTICLE    	0x00000800   /* Particle      'pa' */
#define TAU_MESHES      	0x00001000   /* Meshes        'mesh' */
#define TAU_SUBFIELD    	0x00002000   /* SubField      'su' */
#define TAU_COMMUNICATION 	0x00004000   /* A++ Commm     'c' */
#define TAU_DESCRIPTOR_OVERHEAD 0x00008000   /* A++ Descriptor Overhead   'de' */
/*
SPACE for 			0x00010000
SPACE for 			0x00020000
SPACE for 			0x00040000
SPACE for 			0x00080000
*/
#define TAU_FFT 		0x00100000   /* FFT 'ff' */
#define TAU_ACLMPL 		0x00200000   /* ACLMPL 'ac' */
#define TAU_PAWS1		0x00400000   /* PAWS1  'paws1' */
#define TAU_PAWS2		0x00800000   /* PAWS2  'paws2' */
#define TAU_PAWS3 		0x01000000   /* PAWS3  'paws3' */
/* SPACE for			0x02000000
   SPACE for			0x04000000
*/
#define TAU_USER4   		0x08000000   /* User4 	      '4' */
#define TAU_USER3   		0x10000000   /* User3 	      '3' */	 
#define TAU_USER2   		0x20000000   /* User2 	      '2' */
#define TAU_USER1   		0x40000000   /* User1 	      '1' */
#define TAU_USER    		0x80000000   /* User 	      'u' */

#define TAU_MAX_THREADS 1024

#ifdef PROFILING_ON

#include <string.h>

#if (defined(POOMA_KAI) || defined (TAU_STDCXXLIB))
#include <string>
using std::string;
#else
#define __BOOL_DEFINED 
#include "Profile/bstring.h"
#endif /* POOMA_KAI */


#ifndef NO_RTTI /* RTTI is present  */
#include <typeinfo.h>
#endif /* NO_RTTI  */

#ifdef POOMA_STDSTL
#include <vector>
#include <utility>
#include <list>
using std::vector;
using std::pair;
using std::list;
#else
#include <vector.h>
#if ((!defined(POOMA_KAI)) && (!defined(TAU_STDCXXLIB)))
#include <pair.h>
#else
#include <utility.h>
#endif /* not POOMA_KAI */
#include <list.h>
#endif /* POOMA_STDSTL */


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
	FunctionInfo(const char* name, const char * type, unsigned int ProfileGroup = TAU_DEFAULT);
	FunctionInfo(const char* name, string& type, unsigned int ProfileGroup = TAU_DEFAULT);
	FunctionInfo(const FunctionInfo& X) ;
	// When we exit, we have to clean up.
	~FunctionInfo();
        FunctionInfo& operator= (const FunctionInfo& X) ;
        

	// Tell it about a function call finishing.
	void ExcludeTime(double t);
	// Removing void IncludeTime(double t);
 	// and replacing with 
        void AddInclTime(double t);
	void AddExclTime(double t);
	void IncrNumCalls(void);

	bool GetAlreadyOnStack(void);
	void SetAlreadyOnStack(bool value);  

	// A container of all of these.
	// The ctor registers with this.
        
	static vector<FunctionInfo*> FunctionDB[TAU_MAX_THREADS];

#ifdef PROFILE_CALLS
	list < pair<double,double> > *ExclInclCallList; 
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	// time spent in each call

	int AppendExclInclTimeThisCall(double ex, double in); 
	// to ExclInclCallList
#endif // PROFILE_CALLS

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
	// Cough up the information about this function.
	const char* GetName() const { return Name.c_str(); }
	const char* GetType() const { return Type.c_str(); }
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
	// There is a class that will do some initialization
	// of FunctionStack that can't be done with
	// just the constructor.
	//friend class ProfilerInitializer;
};

//
// For efficiency, make the timing updates inline.
//
inline void 
FunctionInfo::ExcludeTime(double t)
{ // called by a function to decrease its parent functions time
	++NumSubrs;
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
FunctionInfo::SetAlreadyOnStack(bool value)
{
	AlreadyOnStack = value;
}

inline bool
FunctionInfo::GetAlreadyOnStack(void)
{
	return AlreadyOnStack;
}


//////////////////////////////////////////////////////////////////////
//
// class Profiler
//
// This class is intended to be instantiated once per function
// (or other code block to be timed) as an auto variable.
//
// It will be constructed each time the block is entered
// and destroyed when the block is exited.  The constructor
// turns on the timer, and the destructor turns it off.
//
//////////////////////////////////////////////////////////////////////
class Profiler
{
public:
	Profiler(FunctionInfo * fi, unsigned int ProfileGroup = TAU_DEFAULT, 
	  bool StartStop = false);

	void Start();
	Profiler(const Profiler& X);
	Profiler& operator= (const Profiler& X);
	// Clean up data from this invocation.
	void Stop();
	~Profiler();
  	static void ProfileExit(const char *message=0);
	int StoreData(void); 

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) ) 
	int ExcludeTimeThisCall(double t);
	double ExclTimeThisCall; // for this invocation of the function
#endif // PROFILE_CALLS || PROFILE_STATS

	static Profiler * CurrentProfiler[TAU_MAX_THREADS];
	double StartTime;
	FunctionInfo * ThisFunction;
	Profiler * ParentProfiler; 

private:
	unsigned int MyProfileGroup_;
	bool	StartStopUsed_;
	bool 	AddInclFlag; 
	// There is a class that will do some initialization
	// of FunctionStack that can't be done with
	// just the constructor.
	//friend class ProfilerInitializer;
};

//////////////////////////////////////////////////////////////////////
//
// class Profiler
//
// This class is used for porting the TAU Profiling package to other
// platforms and software frameworks. It contains functions to get
// the node id, thread id etc. When Threads are implemented, myThread()
// method should return the thread id in {0..N-1} where N is the total
// number of threads. All interaction with the outside world should be
// restrained to this class. 
//////////////////////////////////////////////////////////////////////
class RtsLayer 
{ // Layer for Profiler to interact with the Runtime System
  public:
 	static unsigned int ProfileMask;
 	static int Node;
 	RtsLayer () { }  // defaults
	~RtsLayer () { } 

 	static unsigned int enableProfileGroup(unsigned int ProfileGroup) ;

        static unsigned int resetProfileGroup(void) ;

	static int setAndParseProfileGroups (char *prog, char *str) ;

        static bool isEnabled(unsigned int ProfileGroup) ; 

        static void ProfileInit(int argc, char **argv);

        static bool isCtorDtor(const char *name);

	inline
	static const char * CheckNotNull(const char * str) {
  	  if (str == 0) return "  ";
          else return str;
	}


  	static int 	SetEventCounter(void);
  	static double 	GetEventCounter(void);

	inline 
	static double   getUSecD(void); 

        // Set the node no.
        static int setMyNode(int NodeId) {
          Node  = NodeId;
          return Node;
        }

  	// Return the number of the 'current' node.
	static int myNode()  { return Node;}

	// Return the number of the 'current' context.
	static int myContext() { return 0; }

	// Return the number of the 'current' thread. 0..TAU_MAX_THREADS-1
	inline
	static int myThread() { return 0; }

}; 

//////////////////////////////////////////////////////////////////////
// TAU PROFILING API MACROS. 
// To ensure that Profiling does not add any runtime overhead when it 
// is turned off, these macros expand to null.
//////////////////////////////////////////////////////////////////////
#define TAU_TYPE_STRING(profileString, str) static string profileString(str);
#define TAU_PROFILE(name, type, group)   static FunctionInfo tauFI(name, type, group);\
				         Profiler tauFP(&tauFI, group); 
#define TAU_PROFILE_TIMER(var, name, type, group)   static FunctionInfo var##fi(name, type, group);\
				         Profiler var(&var##fi, group, true); 
// Construct a Profiler obj and a FunctionInfo obj with an extended name
// e.g., FunctionInfo loop1fi(); Profiler loop1(); 
#define TAU_PROFILE_START(var) var.Start();
#define TAU_PROFILE_STOP(var)  var.Stop();
#define TAU_PROFILE_STMT(stmt) stmt;
#define TAU_PROFILE_EXIT(msg)  Profiler::ProfileExit(msg); 
#define TAU_PROFILE_INIT(argc, argv) RtsLayer::ProfileInit(argc, argv);
#define TAU_PROFILE_SET_NODE(node) RtsLayer::setMyNode(node);

#ifdef NO_RTTI
#define CT(obj) string(#obj)
#else // RTTI is present
#define CT(obj) string(RtsLayer::CheckNotNull(typeid(obj).name())) 
#endif //NO_RTTI

// Use DEBUGPROFMSG macro as in 
// DEBUGPROF("Node" << RtsLayer::myNode() << " Message " << endl;);
// We're deliberately not using *PoomaInfo::Debug because some profiling
// debug messages come from the destructor of Profiler in main and by then
// PoomaInfo object may have been destroyed.

#else /* PROFILING_ON */
/* In the absence of profiling, define the functions as null */
#define TYPE_STRING(profileString, str)
#define PROFILED_BLOCK(name, type) 

#define TAU_TYPE_STRING(profileString, str) 
#define TAU_PROFILE(name, type, group) 
#define TAU_PROFILE_TIMER(var, name, type, group)
#define TAU_PROFILE_START(var)
#define TAU_PROFILE_STOP(var)
#define TAU_PROFILE_STMT(stmt) 
#define TAU_PROFILE_EXIT(msg)
#define TAU_PROFILE_INIT(argc, argv)
#define TAU_PROFILE_SET_NODE(node)
#define CT(obj)

#endif /* PROFILING_ON */





#endif /* PROFILER_H */
/***************************************************************************
 * $RCSfile: Profiler.h,v $   $Author: klindlan $
 * $Revision: 1.1.1.1 $   $Date: 1997/11/26 20:04:29 $
 * POOMA_VERSION_ID: $Id: Profiler.h,v 1.1.1.1 1997/11/26 20:04:29 klindlan Exp $ 
 ***************************************************************************/
