/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: Profiler.cpp					  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Flags		: Compile with				          **
**			  -DPROFILING_ON to enable profiling (ESSENTIAL)  **
**			  -DPROFILE_STATS for Std. Deviation of Excl Time **
**			  -DSGI_HW_COUNTERS for using SGI counters 	  **
**			  -DPROFILE_CALLS  for trace of each invocation   **
**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
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
#include "pcxx_events.h"
#endif // TRACING_ON 

//#define PROFILE_CALLS // Generate Excl Incl data for each call 
//#define DEBUG_PROF // For Debugging Messages from Profiler.cpp

#ifdef DEBUG_PROF
#define DEBUGPROFMSG(msg) { cout<< msg; }
#else
#define DEBUGPROFMSG(msg) 
#endif // DEBUG_PROF

//////////////////////////////////////////////////////////////////////
//Initialize static data
//////////////////////////////////////////////////////////////////////

vector<FunctionInfo*> FunctionInfo::FunctionDB[TAU_MAX_THREADS] ;
Profiler * Profiler::CurrentProfiler[] = {0}; // null to start with
// The rest of CurrentProfiler entries are initialized to null automatically
unsigned int RtsLayer::ProfileMask = TAU_DEFAULT;

// Default value of Node.
int RtsLayer::Node = -1;

//////////////////////////////////////////////////////////////////////
// Explicit Instantiations for templated entities needed for ASCI Red
//////////////////////////////////////////////////////////////////////

#ifdef POOMA_TFLOP
template void vector<FunctionInfo *>::insert_aux(vector<FunctionInfo *>::pointer, FunctionInfo *const &);
#ifndef POOMA_STDSTL
// need a few other function templates instantiated
template FunctionInfo** copy_backward(FunctionInfo**,FunctionInfo**,FunctionInfo**);
template FunctionInfo** uninitialized_copy(FunctionInfo**,FunctionInfo**,FunctionInfo**);
#endif // not POOMA_STDSTL
#endif //POOMA_TFLOP

//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class FunctionInfo
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
FunctionInfo::FunctionInfo(const char *name, const char *type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;
#ifdef TRACING_ON
	GroupName = RtsLayer::PrimaryGroup(ProfileGroupName);
#endif //TRACING_ON
        NumCalls = 0;
        NumSubrs = 0;
  	ExclTime = 0;
  	InclTime = 0;

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
	SetAlreadyOnStack(false);

#ifdef PROFILE_STATS
	SumExclSqr = 0;
#endif //PROFILE_STATS

#ifdef PROFILE_CALLS
	ExclInclCallList = new list<pair<double, double> >();
#endif //PROFILE_CALLS
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	
        MyProfileGroup_ = ProfileGroup ;
	FunctionDB[RtsLayer::myThread()].push_back(this);
#ifdef TRACING_ON
	// Function Id is the index into the DB vector
	FunctionId = FunctionDB[RtsLayer::myThread()].size();
#endif //TRACING_ON
		
        DEBUGPROFMSG("Thr "<< RtsLayer::myNode() 
          << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName() 
	  << " Type : " << GetType() << endl;);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const char *name, string& type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;
#ifdef TRACING_ON
	GroupName = RtsLayer::PrimaryGroup(ProfileGroupName);
#endif //TRACING_ON 
        NumCalls = 0;
        NumSubrs = 0;
  	ExclTime = 0;
  	InclTime = 0;

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
        SetAlreadyOnStack(false);

#ifdef PROFILE_STATS
	SumExclSqr = 0;
#endif //PROFILE_STATS

#ifdef PROFILE_CALLS
	ExclInclCallList = new list<pair<double, double> >();
#endif //PROFILE_CALLS
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	
        MyProfileGroup_ = ProfileGroup ;
	FunctionDB[RtsLayer::myThread()].push_back(this);
#ifdef TRACING_ON
	// Function Id is the index into the DB vector
	FunctionId = FunctionDB[RtsLayer::myThread()].size();
#endif //TRACING_ON
		
        DEBUGPROFMSG("Thr "<< RtsLayer::myNode() 
          << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName() 
	  << " Type : " << GetType() << endl;);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(string& name, string& type, 
	unsigned int ProfileGroup , const char *ProfileGroupName)
{
      if (ProfileGroup & RtsLayer::ProfileMask) {

        Name = name;
  	Type = type;
#ifdef TRACING_ON
	GroupName = RtsLayer::PrimaryGroup(ProfileGroupName);
#endif //TRACING_ON 
        NumCalls = 0;
        NumSubrs = 0;
  	ExclTime = 0;
  	InclTime = 0;

// Since FunctionInfo constructor is called once for each function (static)
// we know that it couldn't be already on the call stack.
        SetAlreadyOnStack(false);

#ifdef PROFILE_STATS
	SumExclSqr = 0;
#endif //PROFILE_STATS

#ifdef PROFILE_CALLS
	ExclInclCallList = new list<pair<double, double> >();
#endif //PROFILE_CALLS
	// Make this a ptr to a list so that ~FunctionInfo doesn't destroy it.
	
        MyProfileGroup_ = ProfileGroup ;
	FunctionDB[RtsLayer::myThread()].push_back(this);
#ifdef TRACING_ON
	// Function Id is the index into the DB vector
	FunctionId = FunctionDB[RtsLayer::myThread()].size();
#endif //TRACING_ON
		
        DEBUGPROFMSG("Thr "<< RtsLayer::myNode() 
          << " FunctionInfo::FunctionInfo(n,t) : Name : "<< GetName() 
	  << " Type : " << GetType() << endl;);
      }
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::FunctionInfo(const FunctionInfo& X) 
: Name(X.Name),
  Type(X.Type),
  NumCalls(X.NumCalls),
  NumSubrs(X.NumSubrs),
  ExclTime(X.ExclTime),
  InclTime(X.InclTime),
  MyProfileGroup_(X.MyProfileGroup_) 
{
	DEBUGPROFMSG("FunctionInfo::FunctionInfo (const FunctionInfo& X)"<<endl;);
	FunctionDB[RtsLayer::myThread()].push_back(this);
}
//////////////////////////////////////////////////////////////////////

FunctionInfo& FunctionInfo::operator= (const FunctionInfo& X) 
{
	DEBUGPROFMSG("FunctionInfo::operator= (const FunctionInfo& X)" << endl;);
   	Name = X.Name;
	Type = X.Type;
	NumCalls = X.NumCalls;
	NumSubrs = X.NumSubrs;
	ExclTime = X.ExclTime;
	InclTime = X.InclTime;
	MyProfileGroup_ = X.MyProfileGroup_;
	return (*this);
}

//////////////////////////////////////////////////////////////////////

FunctionInfo::~FunctionInfo()
{
// Don't delete Name, Type - if dtor of static object dumps the data
// after all these function objects are destroyed, it can't get the 
// name, and type.
//	delete [] Name;
//	delete [] Type;
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
 
//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class Profiler
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

void Profiler::Start(void)
{ 
     
      if (MyProfileGroup_ & RtsLayer::ProfileMask) {
	
#ifdef TRACING_ON
	pcxx_Event(ThisFunction->GetFunctionId(), 1); // 1 is for entry
#endif /* TRACING_ON */

#ifdef PROFILING_ON
	// First, increment the number of calls
	ThisFunction->IncrNumCalls();
        // now increment parent's NumSubrs()
        if (ParentProfiler != 0)
          ParentProfiler->ThisFunction->IncrNumSubrs();	

	// Next, if this function is not already on the call stack, put it
	if (ThisFunction->GetAlreadyOnStack() == false)   { 
	  AddInclFlag = true; 
	  // We need to add Inclusive time when it gets over as 
	  // it is not already on callstack.

	  ThisFunction->SetAlreadyOnStack(true); // it is on callstack now
	}
	else { // the function is already on callstack, no need to add
	       // inclusive time
	  AddInclFlag = false;
	}
	
	// Initialization is over, now record the time it started
	StartTime =  RtsLayer::getUSecD() ;
#endif // PROFILING_ON
  	ParentProfiler = CurrentProfiler[RtsLayer::myThread()] ;

	DEBUGPROFMSG("Thr " << RtsLayer::myNode() <<
	  " Profiler::Start (FunctionInfo * f)  : Name : " << 
	  ThisFunction->GetName() <<" Type : " << ThisFunction->GetType() 
	  << endl; );

	CurrentProfiler[RtsLayer::myThread()] = this;

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall = 0;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
      }  
}

//////////////////////////////////////////////////////////////////////

Profiler::Profiler( FunctionInfo * function, unsigned int ProfileGroup, bool StartStop)
{
      StartStopUsed_ = StartStop; // will need it later in ~Profiler
      MyProfileGroup_ = ProfileGroup ;
      ThisFunction = function ; 
      ParentProfiler = CurrentProfiler[RtsLayer::myThread()]; // Timers
      
      if(!StartStopUsed_) { // Profiler ctor/dtor interface used
	Start(); 
      }
}


//////////////////////////////////////////////////////////////////////

Profiler::Profiler( const Profiler& X)
: StartTime(X.StartTime),
  ThisFunction(X.ThisFunction),
  ParentProfiler(X.ParentProfiler),
  MyProfileGroup_(X.MyProfileGroup_),
  StartStopUsed_(X.StartStopUsed_)
{
	DEBUGPROFMSG("Profiler::Profiler(const Profiler& X)"<<endl;);

	CurrentProfiler[RtsLayer::myThread()] = this;

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall = X.ExclTimeThisCall;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
}

//////////////////////////////////////////////////////////////////////

Profiler& Profiler::operator= (const Profiler& X)
{
  	StartTime = X.StartTime;
	ThisFunction = X.ThisFunction;
	ParentProfiler = X.ParentProfiler; 
	MyProfileGroup_ = X.MyProfileGroup_;
 	StartStopUsed_ = X.StartStopUsed_;

	DEBUGPROFMSG(" Profiler& Profiler::operator= (const Profiler& X)" <<endl;);

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall = X.ExclTimeThisCall;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

	return (*this) ;

}

//////////////////////////////////////////////////////////////////////

void Profiler::Stop()
{
      if (MyProfileGroup_ & RtsLayer::ProfileMask) {

#ifdef TRACING_ON
	pcxx_Event(ThisFunction->GetFunctionId(), -1); // -1 is for exit 
#endif //TRACING_ON

#ifdef PROFILING_ON  // Calculations relevent to profiling only 
	double TotalTime = RtsLayer::getUSecD() - StartTime;

        DEBUGPROFMSG("Thr "<< RtsLayer::myNode() 
	  << " Profiler::Stop() : Name : "<< ThisFunction->GetName() 
	  << " Start : " <<StartTime <<" TotalTime : " << TotalTime<< endl;);

	if (AddInclFlag == true) { // The first time it came on call stack
	  ThisFunction->SetAlreadyOnStack(false); // while exiting

	  // And its ok to add both excl and incl times
	  ThisFunction->AddInclTime(TotalTime);
	  DEBUGPROFMSG("Thr "<< RtsLayer::myNode()
	    << " AddInclFlag true in Stop Name: "<< ThisFunction->GetName()
	    << " Type: " << ThisFunction->GetType() << endl; );
	} 
	// If its already on call stack, don't change AlreadyOnStack
	ThisFunction->AddExclTime(TotalTime);
	// In either case we need to add time to the exclusive time.

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS)|| defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall += TotalTime;
	DEBUGPROFMSG("Thr "<< RtsLayer::myNode() << "Profiler::Stop() : Name " 
	  << ThisFunction->GetName() << " ExclTimeThisCall = "
	  << ExclTimeThisCall << " InclTimeThisCall " << TotalTime << endl;);

#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

#ifdef PROFILE_CALLS
	ThisFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
#endif // PROFILE_CALLS

#ifdef PROFILE_STATS
	ThisFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall);
#endif // PROFILE_STATS

	if (ParentProfiler != 0) {

	  DEBUGPROFMSG("Thr " << RtsLayer::myNode() 
	    << " Profiler::Stop(): ParentProfiler Function Name : " 
	    << ParentProfiler->ThisFunction->GetName() << endl;);

	  ParentProfiler->ThisFunction->ExcludeTime(TotalTime);
#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	  ParentProfiler->ExcludeTimeThisCall(TotalTime);
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

	}
	
#endif //PROFILING_ON
	// First check if timers are overlapping.
	if (CurrentProfiler[RtsLayer::myThread()] != this) {
	  cout <<"ERROR: Timers Overlap. Illegal operation Profiler::Stop " 
	  << ThisFunction->GetName() << " " << ThisFunction->GetType() <<endl;
	}
	// While exiting, reset value of CurrentProfiler to reflect the parent
	CurrentProfiler[RtsLayer::myThread()] = ParentProfiler;

      } // if ProfileMask 
}

//////////////////////////////////////////////////////////////////////

Profiler::~Profiler() {

     if (!StartStopUsed_) {
	Stop();
      } // If ctor dtor interface is used then call Stop. 
	// If the Profiler object is going out of scope without Stop being
	// called, call it now!

      if (MyProfileGroup_ & RtsLayer::ProfileMask) {
	if (ParentProfiler == 0) {
	  if (!RtsLayer::isCtorDtor(ThisFunction->GetName())) {
	  // Not a destructor of a static object - its a function like main 
	    DEBUGPROFMSG("Thr " << RtsLayer::myNode() 
	      << " ~Profiler() : Reached top level function - dumping data" 
	      << endl;);

	    StoreData();
	  }
	// dump data here. Dump it only in the dtor of top level profiler. 
	}
      }
}

//////////////////////////////////////////////////////////////////////

void Profiler::ProfileExit(const char *message)
{
  Profiler *current;

  current = CurrentProfiler[RtsLayer::myThread()];

  DEBUGPROFMSG("Thr "<< RtsLayer::myNode() << " RtsLayer::ProfileExit called :"
    << message << endl;);

  while (current != 0) {
    DEBUGPROFMSG("Thr "<< RtsLayer::myNode() << " ProfileExit() calling Stop :" 
      << current->ThisFunction->GetName() << " " 
      << current->ThisFunction->GetType() << endl;);
    current->Stop(); // clean up 

    if (current->ParentProfiler == 0) {
      if (!RtsLayer::isCtorDtor(current->ThisFunction->GetName())) {
       // Not a destructor of a static object - its a function like main
         DEBUGPROFMSG("Thr " << RtsLayer::myNode()
           << " ProfileExit() : Reached top level function - dumping data"
           << endl;);

          current->StoreData();
      }
    }

    current = CurrentProfiler[RtsLayer::myThread()]; // Stop should set it
  }

}

//////////////////////////////////////////////////////////////////////

int Profiler::StoreData()
{
#ifdef PROFILING_ON 
  	vector<FunctionInfo*>::iterator it;
	char filename[1024], errormsg[1024];
	char *dirname;
	FILE* fp;
 	int numFunc;
#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	long listSize, numCalls;
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS


#ifdef TRACING_ON
	pcxx_EvClose();
	RtsLayer::DumpEDF();
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	sprintf(filename,"%s/profile.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), RtsLayer::myThread());
	DEBUGPROFMSG("Creating " << filename << endl;);
	if ((fp = fopen (filename, "w+")) == NULL) {
		sprintf(errormsg,"Error: Could not create %s",filename);
		perror(errormsg);
		return 0;
	}

	// Data format :
	// %d templated_functions
	// "%s %s" %ld %G %G  
	//  funcname type numcalls Excl Incl
	// %d aggregates
	// <aggregate info>
       
	// Recalculate number of funcs using ProfileGroup. Static objects 
        // constructed before setting Profile Groups have entries in FuncDB 
	// (TAU_DEFAULT) even if they are not supposed to be there.
	//numFunc = (int) FunctionInfo::FunctionDB[RtsLayer::myThread()].size();
	numFunc = 0;
 	for (it = FunctionInfo::FunctionDB[RtsLayer::myThread()].begin(); it != FunctionInfo::FunctionDB[RtsLayer::myThread()].end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::ProfileMask) { 
	    numFunc++;
	  }
	}

#ifdef SGI_HW_COUNTERS
	fprintf(fp,"%d templated_functions_hw_counters\n", numFunc);
#else  // SGI_TIMERS, TULIP_TIMERS 
	fprintf(fp,"%d templated_functions\n", numFunc);
#endif // SGI_HW_COUNTERS 

	// Send out the format string
	fprintf(fp,"# Name Calls Subrs Excl Incl ");
#ifdef PROFILE_STATS
	fprintf(fp,"SumExclSqr ");
#endif //PROFILE_STATS
	fprintf(fp,"ProfileCalls\n");
	
	
 	for (it = FunctionInfo::FunctionDB[RtsLayer::myThread()].begin(); it != FunctionInfo::FunctionDB[RtsLayer::myThread()].end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::ProfileMask) { 
  
  	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
  	      << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
              << (*it)->GetCalls() << " Subrs : "<< (*it)->GetSubrs() 
  	      << " Excl : " << (*it)->GetExclTime() << " Incl : " 
  	      << (*it)->GetInclTime() << endl;);
  	
  	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
  	      (*it)->GetType(), (*it)->GetCalls(), (*it)->GetSubrs(), 
  	      (*it)->GetExclTime(), (*it)->GetInclTime());
  
#ifdef PROFILE_STATS 
  	    fprintf(fp,"%.16G ", (*it)->GetSumExclSqr());
#endif //PROFILE_STATS
  
#ifdef PROFILE_CALLS
  	    listSize = (long) (*it)->ExclInclCallList->size(); 
  	    numCalls = (*it)->GetCalls();
  	    // Sanity check
  	    if (listSize != numCalls) 
  	    {
  	      fprintf(fp,"0 \n"); // don't write any invocation data
  	      DEBUGPROFMSG("Error *** list (profileCalls) size mismatch size "
  	        << listSize << " numCalls " << numCalls << endl;);
  	    }
  	    else { // List is maintained correctly
  	      fprintf(fp,"%ld \n", listSize); // no of records to follow
  	      for (iter = (*it)->ExclInclCallList->begin(); 
  	        iter != (*it)->ExclInclCallList->end(); iter++)
  	      {
  	        DEBUGPROFMSG("Node: " << RtsLayer::myNode() <<" Name "
  	          << (*it)->GetName() << " " << (*it)->GetType()
  	          << " ExclThisCall : "<< (*iter).first <<" InclThisCall : " 
  	          << (*iter).second << endl; );
  	        fprintf(fp,"%G %G\n", (*iter).first , (*iter).second);
  	      }
            } // sanity check 
#else  // PROFILE_CALLS
  	    fprintf(fp,"0 \n"); // Indicating - profile calls is turned off 
#endif // PROFILE_CALLS
	  } // ProfileGroup test 
	} // for loop. End of FunctionInfo data
	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
	// Change this when aggregate profiling in introduced in Pooma 

	fclose(fp);

#endif //PROFILING_ON
	return 1;
}

//////////////////////////////////////////////////////////////////////

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
int Profiler::ExcludeTimeThisCall(double t)
{
	ExclTimeThisCall -= t;
	return 1;
}
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Member Function Definitions For class RtsLayer
// Important for Porting to other platforms and frameworks.
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////

unsigned int RtsLayer::enableProfileGroup(unsigned int ProfileGroup) {
  ProfileMask |= ProfileGroup; // Add it to the mask
  DEBUGPROFMSG("enableProfileGroup " << ProfileGroup <<" Mask = " << ProfileMask << endl;);
  return ProfileMask;
}

/////////////////////////////////////////////////////////////////////////

unsigned int RtsLayer::resetProfileGroup(void) {
  ProfileMask = 0;
  return ProfileMask;
}

/////////////////////////////////////////////////////////////////////////
int RtsLayer::setMyNode(int NodeId) {
  Node  = NodeId;
// At this stage, we should create the trace file because we know the node id
#ifdef TRACING_ON
  char *dirname, tracefilename[1024];
  if ((dirname = getenv("TRACEDIR")) == NULL) {
  // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }
  sprintf(tracefilename, "%s/tau.####.trc",dirname); 
  pcxx_EvInit(tracefilename);
#endif // TRACING_ON
  return Node;
}

/////////////////////////////////////////////////////////////////////////

bool RtsLayer::isEnabled(unsigned int ProfileGroup) {
unsigned int res =  ProfileGroup & ProfileMask ;

  if (res > 0)
    return true;
  else
    return false;
}

//////////////////////////////////////////////////////////////////////

#ifdef SGI_HW_COUNTERS 
extern "C" {
  int start_counters( int e0, int e1 );
  int read_counters( int e0, long long *c0, int e1, long long *c1);
};
#endif // SGI_HW_COUNTERS

//////////////////////////////////////////////////////////////////////
#ifdef SGI_HW_COUNTERS 
int RtsLayer::SetEventCounter()
{
  int e0, e1;
  int start;


  e0 = 0;
  e1 = 0;


//  int x0, x1;
  // 
  // DO NOT remove the following two lines. Otherwise start_counters 
  // crashes with "prioctl PIOCENEVCTRS returns error: Invalid argument"

/*
  x0 = e0; 
  x1 = e1; 
*/

  if((start = start_counters(e0,e1)) < 0) {
    perror("start_counters");
    exit(0);
  }
  return start;
}
#endif // SGI_HW_COUNTERS

/////////////////////////////////////////////////////////////////////////
#ifdef SGI_HW_COUNTERS 
double RtsLayer::GetEventCounter()
{
  static int gen_start = SetEventCounter();
  int gen_read;
  int e0 = 0, e1 = 0;
  long long c0 , c1 ;
  static double accum = 0;

  if ((gen_read = read_counters(e0, &c0, e1, &c1)) < 0) {
    perror("read_counters");
  }

  if (gen_read != gen_start) {
    perror("lost counter! aborting...");
    exit(1);
  }

  accum += c0;
  DEBUGPROFMSG("Read counters e0 " << e0 <<" e1 "<< e1<<" gen_read " 
    << gen_read << " gen_start = " << gen_start << " accum "<< accum 
    << " c0 " << c0 << " c1 " << c1 << endl;);
  gen_start = SetEventCounter(); // Reset the counter

  return accum;
}
#endif //SGI_HW_COUNTERS

///////////////////////////////////////////////////////////////////////////

double RtsLayer::getUSecD () {

#ifdef SGI_HW_COUNTERS
  return RtsLayer::GetEventCounter();
#else  //SGI_HW_COUNTERS

#ifdef SGI_TIMERS
  struct timespec tp;
  clock_gettime(CLOCK_SGI_CYCLE,&tp);
  return (tp.tv_sec * 1e6 + (tp.tv_nsec * 1e-3)) ;

#else 
#if (defined(POOMA_TFLOP) || !defined(TULIP_TIMERS)) 
  struct timeval tp;
  gettimeofday (&tp, 0);
  return ( tp.tv_sec * 1000000 + tp.tv_usec );
#else  // TULIP_TIMERS by default.  
  return pcxx_GetUSecD();
#endif  //POOMA_TFLOP
#endif 	//SGI_TIMERS

#endif  // SGI_HW_COUNTERS
        }

///////////////////////////////////////////////////////////////////////////
//Note: This is similar to Tulip event classes during tracing
///////////////////////////////////////////////////////////////////////////
int RtsLayer::setAndParseProfileGroups(char *prog, char *str)
{
  char *end;
  
  if ( str )
  { 
    while (str && *str) 
    {
      if ( ( end = strchr (str, '+')) != NULL) *end = '\0';
 
      switch ( str[0] )
      {
        case 'a' :
	case 'A' : // Assign Expression Evaluation Group
	  if (strncasecmp(str,"ac", 2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_ACLMPL); 
	    // ACLMPL enabled 
	  } 
	  else 
	    RtsLayer::enableProfileGroup(TAU_ASSIGN);
	  break;
	case 'b' : 
	case 'B' : // Blitz++ profile group
	  RtsLayer::enableProfileGroup(TAU_BLITZ);
	  break; // Blitz++ enabled
        case 'f' :
	case 'F' : // Field Group
	  if (strncasecmp(str, "ff", 2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_FFT);
	    // FFT enabled 
	  }
	  else 
	    RtsLayer::enableProfileGroup(TAU_FIELD);
	    // Field enabled 
	  break;
	case 'c' :
	case 'C' : 
	  RtsLayer::enableProfileGroup(TAU_COMMUNICATION);
	  break;
        case 'i' :
	case 'I' : // DiskIO, Other IO 
	  RtsLayer::enableProfileGroup(TAU_IO);
	  break;
        case 'l' :
	case 'L' : // Field Layout Group
	  RtsLayer::enableProfileGroup(TAU_LAYOUT);
	  break;
	case 'm' : 
	case 'M' : 
          if (strncasecmp(str,"mesh", 4) == 0) {
  	    RtsLayer::enableProfileGroup(TAU_MESHES);
	    // Meshes enabled
 	  } 
 	  else 
	    RtsLayer::enableProfileGroup(TAU_MESSAGE);
	    // Message Profile Group enabled 
  	  break;
        case 'p' :
	case 'P' : 
          if (strncasecmp(str, "paws1", 5) == 0) {
	    RtsLayer::enableProfileGroup(TAU_PAWS1); 
	  } 
	  else {
	    if (strncasecmp(str, "paws2", 5) == 0) {
	      RtsLayer::enableProfileGroup(TAU_PAWS2); 
	    } 
	    else {
	      if (strncasecmp(str, "paws3", 5) == 0) {
	        RtsLayer::enableProfileGroup(TAU_PAWS3); 
	      } 
	      else {
	        if (strncasecmp(str,"pa",2) == 0) {
	          RtsLayer::enableProfileGroup(TAU_PARTICLE);
	          // Particle enabled 
	        } 
		else {
	          RtsLayer::enableProfileGroup(TAU_PETE);
	    	  // PETE Profile Group enabled 
	 	}
	      }
	    } 
 	  } 
	  
	  break;
  	case 'r' : 
	case 'R' : // Region Group 
	  RtsLayer::enableProfileGroup(TAU_REGION);
	  break;
        case 's' :
	case 'S' : 
	  if (strncasecmp(str,"su",2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_SUBFIELD);
	    // SubField enabled 
	  } 
 	  else
	    RtsLayer::enableProfileGroup(TAU_SPARSE);
	    // Sparse Index Group
	  break;
        case 'd' :
	case 'D' : // Domainmap Group
	  if (strncasecmp(str,"de",2) == 0) {
	    RtsLayer::enableProfileGroup(TAU_DESCRIPTOR_OVERHEAD);
	  } else  
	     RtsLayer::enableProfileGroup(TAU_DOMAINMAP);
	  break;
 	case 'u' :
        case 'U' : // User or Utility 
          if (strncasecmp(str,"ut", 2) == 0) { 
	    RtsLayer::enableProfileGroup(TAU_UTILITY);
	  }
	  else // default - for u is USER 
 	    RtsLayer::enableProfileGroup(TAU_USER);
	  break;
        case 'v' :
	case 'V' : // ACLVIZ Group
	  RtsLayer::enableProfileGroup(TAU_VIZ);
	  break;
	case '1' : // User1
	  RtsLayer::enableProfileGroup(TAU_USER1);
	  break; 
	case '2' : // User2
	  RtsLayer::enableProfileGroup(TAU_USER2);
	  break;
	case '3' : // User3
	  RtsLayer::enableProfileGroup(TAU_USER3);
	  break; 
	case '4' : // User4
	  RtsLayer::enableProfileGroup(TAU_USER4);
	  break;
	default  :
	  cout << prog << " : Invalid Profile Group " << str << endl;
	  break; 
      } 
      if (( str = end) != NULL) *str++ = '+';
    }
  }
  else 
    enableProfileGroup(TAU_DEFAULT); // Enable everything 
  return 1;
}

//////////////////////////////////////////////////////////////////////
void RtsLayer::ProfileInit(int argc, char **argv)
{
  int i;

  for(i=0; i < argc; i++) {
    if ( ( strcasecmp(argv[i], "--profile") == 0 ) ) {
        // Enable the profile groups
        if ( (i + 1) < argc && argv[i+1][0] != '-' )  { // options follow
           RtsLayer::resetProfileGroup(); // set it to blank
           RtsLayer::setAndParseProfileGroups(argv[0], argv[i+1]);
        }
    }
  }
  return;
}


//////////////////////////////////////////////////////////////////////
bool RtsLayer::isCtorDtor(const char *name)
{

  // If the destructor a static object is called, it could have a null name
  // after main is over. Treat it like a Dtor and return true.
  if (name[0] == 0) {
    DEBUGPROFMSG("isCtorDtor name is NULL" << endl;);
    return true; 
  }
  DEBUGPROFMSG("RtsLayer::isCtorDtor("<< name <<")" <<endl;);
  if (strchr(name,'~') == NULL) // a destructor 
    if (strchr(name,':') == NULL) // could be a constructor 
      return false;
    else  
      return true;
  else  
    return true;
}

//////////////////////////////////////////////////////////////////////
// PrimaryGroup returns the first group that the function belongs to.
// This is needed in tracing as Vampir can handle only one group per
// function. PrimaryGroup("TAU_FIELD | TAU_USER") should return "TAU_FIELD"
//////////////////////////////////////////////////////////////////////
string RtsLayer::PrimaryGroup(const char *ProfileGroupName) 
{
  string groups = ProfileGroupName;
  string primary; 
  string separators = " |"; 
  int start, stop, n;

  start = groups.find_first_not_of(separators);
  n = groups.length();
  stop = groups.find_first_of(separators, start); 

  if ((stop < 0) || (stop > n)) stop = n;

  primary = groups.substr(start, stop - start) ;
  return primary;

}

//////////////////////////////////////////////////////////////////////
// TraceSendMsg traces the message send
//////////////////////////////////////////////////////////////////////
void RtsLayer::TraceSendMsg(int type, int destination, int length)
{
#ifdef TRACING_ON 
  long int parameter, othernode;

  if (RtsLayer::isEnabled(TAU_MESSAGE))
  {
    parameter = 0L;
    /* for send, othernode is receiver or destination */
    othernode = (long int) destination;
    /* Format for parameter is
       31 ..... 24 23 ......16 15..............0
          other       type          length       
     */
  
    parameter = (length & 0x0000FFFF) | ((type & 0x000000FF)  << 16) | 
  	      (othernode << 24);
    pcxx_Event(TAU_MESSAGE_SEND, parameter); 
#ifdef DEBUG_PROF
    printf("Node %d TraceSendMsg, type %x dest %x len %x par %lx \n", 
  	RtsLayer::myNode(), type, destination, length, parameter);
#endif //DEBUG_PROF
  } 
#endif //TRACING_ON
}

  
//////////////////////////////////////////////////////////////////////
// TraceRecvMsg traces the message recv
//////////////////////////////////////////////////////////////////////
void RtsLayer::TraceRecvMsg(int type, int source, int length)
{
#ifdef TRACING_ON
  long int parameter, othernode;

  if (RtsLayer::isEnabled(TAU_MESSAGE)) 
  {
    parameter = 0L;
    /* for recv, othernode is sender or source*/
    othernode = (long int) source;
    /* Format for parameter is
       31 ..... 24 23 ......16 15..............0
          other       type          length       
     */
  
    parameter = (length & 0x0000FFFF) | ((type & 0x000000FF)  << 16) | 
  	      (othernode << 24);
    pcxx_Event(TAU_MESSAGE_RECV, parameter); 
  
#ifdef DEBUG_PROF
    printf("Node %d TraceRecvMsg, type %x src %x len %x par %lx \n", 
  	RtsLayer::myNode(), type, source, length, parameter);
#endif //DEBUG_PROF
  }
#endif //TRACING_ON
}

//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// DumpEDF() writes the function information in the edf.<node> file
// The function info consists of functionId, group, name, type, parameters
//////////////////////////////////////////////////////////////////////
int RtsLayer::DumpEDF(void)
{
#ifdef TRACING_ON 
  	vector<FunctionInfo*>::iterator it;
	char filename[1024], errormsg[1024];
	char *dirname;
	FILE* fp;
	int  numEvents, numExtra;


	if ((dirname = getenv("TRACEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}

	sprintf(filename,"%s/events.%d.edf",dirname, RtsLayer::myNode());
	DEBUGPROFMSG("Creating " << filename << endl;);
	if ((fp = fopen (filename, "w+")) == NULL) {
		sprintf(errormsg,"Error: Could not create %s",filename);
		perror(errormsg);
		return 0;
	}
	
	// Data Format 
	// <no.> events
	// # or \n ignored
	// %s %s %d "%s %s" %s 
	// id group tag "name type" parameters

	numExtra = 9; // Number of extra events
	numEvents = FunctionInfo::FunctionDB[RtsLayer::myThread()].size();

	numEvents += numExtra;

	fprintf(fp,"%d dynamic_trace_events\n", numEvents);

	fprintf(fp,"# FunctionId Group Tag \"Name Type\" Parameters\n");

 	for (it = FunctionInfo::FunctionDB[RtsLayer::myThread()].begin(); 
	  it != FunctionInfo::FunctionDB[RtsLayer::myThread()].end(); it++)
	{
  	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping EDF Id : " 
	    << (*it)->GetFunctionId() << " " << (*it)->GetPrimaryGroup() 
	    << " 0 " << (*it)->GetName() << " " << (*it)->GetType() 
	    << " EntryExit" << endl;); 
	
	  fprintf(fp, "%ld %s 0 \"%s %s\" EntryExit\n", (*it)->GetFunctionId(),
	    (*it)->GetPrimaryGroup(), (*it)->GetName(), (*it)->GetType() );
	}
	// Now add the nine extra events 
	fprintf(fp,"%ld TRACER 0 \"EV_INIT\" none\n", (long) PCXX_EV_INIT); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_ENTER\" none\n", (long) PCXX_EV_FLUSH_ENTER); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_EXIT\" none\n", (long) PCXX_EV_FLUSH_EXIT); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_CLOSE\" none\n", (long) PCXX_EV_CLOSE); 
	fprintf(fp,"%ld TRACER 0 \"FLUSH_INITM\" none\n", (long) PCXX_EV_INITM); 
	fprintf(fp,"%ld TRACER 0 \"WALL_CLOCK\" none\n", (long) PCXX_EV_WALL_CLOCK); 
	fprintf(fp,"%ld TRACER 0 \"CONT_EVENT\" none\n", (long) PCXX_EV_CONT_EVENT); 
	fprintf(fp,"%ld TAU_MESSAGE -7 \"MESSAGE_SEND\" par\n", (long) TAU_MESSAGE_SEND); 
	fprintf(fp,"%ld TAU_MESSAGE -8 \"MESSAGE_RECV\" par\n", (long) TAU_MESSAGE_RECV); 

  
	fclose(fp);
#endif //TRACING_ON
	return 1;
}

#ifdef PROFILE_CALLSTACK

//////////////////////////////////////////////////////////////////////
//  Profiler::CallStackTrace()
//
//  Author:  Mike Kaufman
//           mikek@cs.uoregon.edu
//  output stack of active Profiler objects
//////////////////////////////////////////////////////////////////////
void Profiler::CallStackTrace()
{
  char      *dirname;            // directory name of output file
  char      fname[1024];         // output file name 
  char      errormsg[1024];      // error message buffer
  FILE      *fp;
  Profiler  *curr;               // current Profiler object in stack traversal
  double    now;                 // current wallclock time 
  double    totalTime;           // now - profiler's start time
  double    prevTotalTime;       // inclusive time of last Profiler object 
                                 //   stack
  static int ncalls = 0;         // number of times CallStackTrace()
                                 //   has been called
 
  // get wallclock time
  now = RtsLayer::getUSecD();  

  DEBUGPROFMSG("CallStackTrace started at " << now << endl;);

  // increment num of calls to trace
  ncalls++;

  // set up output file
  if ((dirname = getenv("PROFILEDIR")) == NULL)
  {
    dirname = new char[8];
    strcpy (dirname, ".");
  }
  
  // create file name string
  sprintf(fname, "%s/callstack.%d.%d.%d", dirname, RtsLayer::myNode(),
	  RtsLayer::myContext(), RtsLayer::myThread());
  
  // traverse stack and set all FunctionInfo's *_cs fields to zero
  curr = CurrentProfiler[RtsLayer::myThread()];
  while (curr != 0)
  {
    curr->ThisFunction->ExclTime_cs = curr->ThisFunction->GetExclTime();
    curr = curr->ParentProfiler;
  }  

  prevTotalTime = 0;
  // calculate time info
  curr = CurrentProfiler[RtsLayer::myThread()];
  while (curr != 0 )
  {
    totalTime = now - curr->StartTime;
 
    // set profiler's inclusive time
    curr->InclTime_cs = totalTime;

    // calc Profiler's exclusive time
    curr->ExclTime_cs = totalTime + curr->ExclTimeThisCall
                      - prevTotalTime;
     
    if (curr->AddInclFlag == true)
    {
      // calculate inclusive time for profiler's FunctionInfo
      curr->ThisFunction->InclTime_cs = curr->ThisFunction->GetInclTime()  
                                      + totalTime;
    }
    
    // calculate exclusive time for each profiler's FunctionInfo
    curr->ThisFunction->ExclTime_cs += totalTime - prevTotalTime;

    // keep total of inclusive time
    prevTotalTime = totalTime;
 
    // next profiler
    curr = curr->ParentProfiler;

  }
 
  // open file
  if (ncalls == 1)
    fp = fopen(fname, "w+");
  else
    fp = fopen(fname, "a");
  if (fp == NULL)  // error opening file
  {
    sprintf(errormsg, "Error:  Could not create %s", fname);
    perror(errormsg);
    return;
  }

  if (ncalls == 1)
  {
    fprintf(fp,"%s%s","# Name Type Calls Subrs Prof-Incl ",
            "Prof-Excl Func-Incl Func-Excl\n");
    fprintf(fp, 
            "# -------------------------------------------------------------\n");
  }
  else
    fprintf(fp, "\n");

  // output time of callstack dump
  fprintf(fp, "%.16G\n", now);
  // output call stack info
  curr = CurrentProfiler[RtsLayer::myThread()];
  while (curr != 0 )
  {
    fprintf(fp, "\"%s %s\" %ld %ld %.16G %.16G %.16G %.16G\n",
            curr->ThisFunction->GetName(),  curr->ThisFunction->GetType(),
            curr->ThisFunction->GetCalls(), curr->ThisFunction->GetSubrs(),
            curr->InclTime_cs, curr->ExclTime_cs,
            curr->ThisFunction->InclTime_cs, curr->ThisFunction->ExclTime_cs);

    curr = curr->ParentProfiler;
    
  } 

  // close file
  fclose(fp);

}
/*-----------------------------------------------------------------*/
#endif //PROFILE_CALLSTACK


/***************************************************************************
 * $RCSfile: Profiler.cpp,v $   $Author: sameer $
 * $Revision: 1.7 $   $Date: 1998/03/20 21:46:31 $
 * POOMA_VERSION_ID: $Id: Profiler.cpp,v 1.7 1998/03/20 21:46:31 sameer Exp $ 
 ***************************************************************************/

	





