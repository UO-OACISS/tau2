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

//////////////////////////////////////////////////////////////////////
//Initialize static data
//////////////////////////////////////////////////////////////////////

// No need to initialize FunctionDB. using TheFunctionDB() instead.
// vector<FunctionInfo*> FunctionInfo::FunctionDB[TAU_MAX_THREADS] ;
Profiler * Profiler::CurrentProfiler[] = {0}; // null to start with
// The rest of CurrentProfiler entries are initialized to null automatically
//unsigned int RtsLayer::ProfileMask = TAU_DEFAULT;

// Default value of Node.
//int RtsLayer::Node = -1;

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
// Member Function Definitions For class Profiler
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

void Profiler::Start(void)
{ 
     
      if (MyProfileGroup_ & RtsLayer::TheProfileMask()) {
	
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
      if (MyProfileGroup_ & RtsLayer::TheProfileMask()) {

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

      } // if TheProfileMask() 
}

//////////////////////////////////////////////////////////////////////

Profiler::~Profiler() {

     if (!StartStopUsed_) {
	Stop();
      } // If ctor dtor interface is used then call Stop. 
	// If the Profiler object is going out of scope without Stop being
	// called, call it now!

      if (MyProfileGroup_ & RtsLayer::TheProfileMask()) {
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
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
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
	
	
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
  
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
 * $Revision: 1.10 $   $Date: 1998/04/26 07:29:23 $
 * POOMA_VERSION_ID: $Id: Profiler.cpp,v 1.10 1998/04/26 07:29:23 sameer Exp $ 
 ***************************************************************************/

	





