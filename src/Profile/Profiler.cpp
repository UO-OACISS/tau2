/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
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

//#define DEBUG_PROF // For Debugging Messages from Profiler.cpp
#include "Profile/Profiler.h"

#ifndef TAU_WINDOWS
extern "C" void Tau_shutdown(void);
#endif //TAU_WINDOWS

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

#endif //TAU_WINDOWS

#ifdef TRACING_ON
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif // TRACING_ON 

//#define PROFILE_CALLS // Generate Excl Incl data for each call 

//////////////////////////////////////////////////////////////////////
//Initialize static data
//////////////////////////////////////////////////////////////////////

// No need to initialize FunctionDB. using TheFunctionDB() instead.
// vector<FunctionInfo*> FunctionInfo::FunctionDB[TAU_MAX_THREADS] ;
Profiler * Profiler::CurrentProfiler[] = {0}; // null to start with
// The rest of CurrentProfiler entries are initialized to null automatically
//TauGroup_t RtsLayer::ProfileMask = TAU_DEFAULT;

// Default value of Node.
//int RtsLayer::Node = -1;

//////////////////////////////////////////////////////////////////////
// Explicit Instantiations for templated entities needed for ASCI Red
//////////////////////////////////////////////////////////////////////

#ifdef PGI
template
void vector<FunctionInfo *>::insert_aux(vector<FunctionInfo *>::pointer, FunctionInfo *const &);
// need a few other function templates instantiated
template
FunctionInfo** copy_backward(FunctionInfo**,FunctionInfo**,FunctionInfo**);
template
FunctionInfo** uninitialized_copy(FunctionInfo**,FunctionInfo**,FunctionInfo**);
//template <>
//std::basic_ostream<char, std::char_traits<char> > & std::operator<< (std::basic_ostream<char, std::char_traits<char> > &, const char * );
#endif /* PGI */

 
//////////////////////////////////////////////////////////////////////
// Member Function Definitions For class Profiler
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////

void Profiler::Start(int tid)
{ 
      DEBUGPROFMSG("Profiler::Start: MyProfileGroup_ = " << MyProfileGroup_ 
        << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
      if ((MyProfileGroup_ & RtsLayer::TheProfileMask()) 
	&& RtsLayer::TheEnableInstrumentation()) {
	if (ThisFunction == (FunctionInfo *) NULL) return; // Mapping
      DEBUGPROFMSG("Profiler::Start Entering " << ThisFunction->GetName()<<endl;);
	
#ifdef TRACING_ON
	TraceEvent(ThisFunction->GetFunctionId(), 1, tid); // 1 is for entry
#endif /* TRACING_ON */

#ifdef PROFILING_ON
	// First, increment the number of calls
	ThisFunction->IncrNumCalls(tid);
        // now increment parent's NumSubrs()
	if (ParentProfiler != 0)
          ParentProfiler->ThisFunction->IncrNumSubrs(tid);	

	// Next, if this function is not already on the call stack, put it
	if (ThisFunction->GetAlreadyOnStack(tid) == false)   { 
	  AddInclFlag = true; 
	  // We need to add Inclusive time when it gets over as 
	  // it is not already on callstack.

	  ThisFunction->SetAlreadyOnStack(true, tid); // it is on callstack now
	}
	else { // the function is already on callstack, no need to add
	       // inclusive time
	  AddInclFlag = false;
	}
	
	// Initialization is over, now record the time it started
#ifndef TAU_MULTIPLE_COUNTERS 
	StartTime =  RtsLayer::getUSecD(tid) ;
#else //TAU_MULTIPLE_COUNTERS
	//Initialize the array to zero, as some of the elements will
	//not be set by counting functions.
	for(int i=0;i<MAX_TAU_COUNTERS;i++){
	  StartTime[i]=0;
	}
	//Now get the start times.
	RtsLayer::getUSecD(tid, StartTime);	  
#endif//TAU_MULTIPLE_COUNTERS
	DEBUGPROFMSG("Start Time = "<< StartTime<<endl;);
#endif // PROFILING_ON
  	
	ParentProfiler = CurrentProfiler[tid] ;
	  


	DEBUGPROFMSG("nct "<< RtsLayer::myNode() << "," 
	  << RtsLayer::myContext() << ","  << tid 
	  << " Profiler::Start (tid)  : Name : " 
	  << ThisFunction->GetName() <<" Type : " << ThisFunction->GetType() 
	  << endl; );

	CurrentProfiler[tid] = this;
        if (ParentProfiler != 0) {
          DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
            << RtsLayer::myContext() << ","  << tid
	    << " Inside "<< ThisFunction->GetName()<< " Setting ParentProfiler "
	    << ParentProfiler->ThisFunction->GetName()<<endl
	    << " ParentProfiler = "<<ParentProfiler << " CurrProf = "
	    << CurrentProfiler[tid] << " = this = "<<this<<endl;);
        }

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall = 0;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

      }  
      else
      { /* If instrumentation is disabled, set the CurrentProfiler */

	ParentProfiler = CurrentProfiler[tid] ;
	CurrentProfiler[tid] = this;
      } /* this is so Stop can access CurrentProfiler as well */
}

//////////////////////////////////////////////////////////////////////

Profiler::Profiler( FunctionInfo * function, TauGroup_t ProfileGroup, 
	bool StartStop, int tid)
{

      StartStopUsed_ = StartStop; // will need it later in ~Profiler
      MyProfileGroup_ = ProfileGroup ;
      ThisFunction = function ; 
      ParentProfiler = CurrentProfiler[tid]; // Timers
      DEBUGPROFMSG("Profiler::Profiler: MyProfileGroup_ = " << MyProfileGroup_ 
        << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
      

      if(!StartStopUsed_) { // Profiler ctor/dtor interface used
	Start(tid); 
      }
}


//////////////////////////////////////////////////////////////////////

Profiler::Profiler( const Profiler& X)
: ThisFunction(X.ThisFunction),
  ParentProfiler(X.ParentProfiler),
  MyProfileGroup_(X.MyProfileGroup_),
  StartStopUsed_(X.StartStopUsed_)
{
#ifndef TAU_MULTIPLE_COUNTERS	
  StartTime = X.StartTime;
#else //TAU_MULTIPLE_COUNTERS
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    StartTime[i] = X.StartTime[i];
  }
#endif//TAU_MULTIPLE_COUNTERS

  DEBUGPROFMSG("Profiler::Profiler(const Profiler& X)"<<endl;);

	CurrentProfiler[RtsLayer::myThread()] = this;

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall = X.ExclTimeThisCall;
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK
}

//////////////////////////////////////////////////////////////////////

Profiler& Profiler::operator= (const Profiler& X)
{
#ifndef TAU_MULTIPLE_COUNTERS	
  StartTime = X.StartTime;
#else //TAU_MULTIPLE_COUNTERS
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    StartTime[i] = X.StartTime[i];
  }
#endif//TAU_MULTIPLE_COUNTERS

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

void Profiler::Stop(int tid)
{
      DEBUGPROFMSG("Profiler::Stop: MyProfileGroup_ = " << MyProfileGroup_ 
        << " Mask = " << RtsLayer::TheProfileMask() <<endl;);
      if ((MyProfileGroup_ & RtsLayer::TheProfileMask()) 
	  && RtsLayer::TheEnableInstrumentation()) {
	if (ThisFunction == (FunctionInfo *) NULL) return; // Mapping
        DEBUGPROFMSG("Profiler::Stop for routine = " << ThisFunction->GetName()<<endl;);
#ifdef TRACING_ON
	TraceEvent(ThisFunction->GetFunctionId(), -1, tid); // -1 is for exit
#endif //TRACING_ON

#ifdef PROFILING_ON  // Calculations relevent to profiling only 
#ifndef TAU_MULTIPLE_COUNTERS
	double TotalTime = RtsLayer::getUSecD(tid) - StartTime;
#else //TAU_MULTIPLE_COUNTERS
	double CurrentTime[MAX_TAU_COUNTERS];
	for(int j=0;j<MAX_TAU_COUNTERS;j++){
	  CurrentTime[j]=0;
	}
	//Get the current counter values.
	RtsLayer::getUSecD(tid, CurrentTime);

	double TotalTime[MAX_TAU_COUNTERS];
	for(int i=0;i<MAX_TAU_COUNTERS;i++){
	  TotalTime[i]=0;
	}

	for(int k=0;k<MAX_TAU_COUNTERS;k++){
	  TotalTime[k] = CurrentTime[k] - StartTime[k];
	}
	
#endif//TAU_MULTIPLE_COUNTERS


        DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << "," 
  	  << RtsLayer::myContext() << "," << tid 
	  << " Profiler::Stop() : Name : "<< ThisFunction->GetName() 
	  << " Start : " <<StartTime <<" TotalTime : " << TotalTime
	  << " AddInclFlag : " << AddInclFlag << endl;);

	if (AddInclFlag == true) { // The first time it came on call stack
	  ThisFunction->SetAlreadyOnStack(false, tid); // while exiting

          DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << "," 
  	    << RtsLayer::myContext() << "," << tid  << " "  
	    << "STOP: After SetAlreadyOnStack Going for AddInclTime" <<endl; );

	  // And its ok to add both excl and incl times
	  ThisFunction->AddInclTime(TotalTime, tid);
	  DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
	    << RtsLayer::myContext() << "," << tid
	    << " AddInclFlag true in Stop Name: "<< ThisFunction->GetName()
	    << " Type: " << ThisFunction->GetType() << endl; );
	} 
	// If its already on call stack, don't change AlreadyOnStack
	ThisFunction->AddExclTime(TotalTime, tid);
	// In either case we need to add time to the exclusive time.

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS)|| defined(PROFILE_CALLSTACK) )
	ExclTimeThisCall += TotalTime;
	DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
          << RtsLayer::myContext() << "," << tid  << " " 
  	  << "Profiler::Stop() : Name " 
	  << ThisFunction->GetName() << " ExclTimeThisCall = "
	  << ExclTimeThisCall << " InclTimeThisCall " << TotalTime << endl;);

#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

#ifdef PROFILE_CALLS
	ThisFunction->AppendExclInclTimeThisCall(ExclTimeThisCall, TotalTime);
#endif // PROFILE_CALLS

#ifdef PROFILE_STATS
	ThisFunction->AddSumExclSqr(ExclTimeThisCall*ExclTimeThisCall, tid);
#endif // PROFILE_STATS

	if (ParentProfiler != (Profiler *) NULL) {

	  DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
            << RtsLayer::myContext() << "," << tid  
	    << " Profiler::Stop(): ParentProfiler Function Name : " 
	    << ParentProfiler->ThisFunction->GetName() << endl;);
	  DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
            << RtsLayer::myContext() << "," << tid
	    << " Exiting from "<<ThisFunction->GetName() << " Returning to "
	    << ParentProfiler->ThisFunction->GetName() << endl;);

	  if (ParentProfiler->ThisFunction != (FunctionInfo *) NULL)
	    ParentProfiler->ThisFunction->ExcludeTime(TotalTime, tid);
          else {
	    cout <<"ParentProfiler's Function info is NULL" <<endl;
	  }

#if ( defined(PROFILE_CALLS) || defined(PROFILE_STATS) || defined(PROFILE_CALLSTACK) )
	  ParentProfiler->ExcludeTimeThisCall(TotalTime);
#endif //PROFILE_CALLS || PROFILE_STATS || PROFILE_CALLSTACK

	}
	
#endif //PROFILING_ON
	// First check if timers are overlapping.
	if (CurrentProfiler[tid] != this) {
	  DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
              << RtsLayer::myContext() << "," << tid
	      << " ERROR: Timers Overlap. Illegal operation Profiler::Stop " 
	      << ThisFunction->GetName() << " " 
	      << ThisFunction->GetType() <<endl;);
	  if (CurrentProfiler[tid] != (Profiler *) NULL) {
	    if (CurrentProfiler[tid]->ThisFunction != (FunctionInfo *)NULL) {
#ifdef TAU_OPENMP
#pragma omp critical
#endif /* TAU_OPENMP */
	      cout << "Overlapping function = "
                 << CurrentProfiler[tid]->ThisFunction->GetName () << " " 
		 << CurrentProfiler[tid]->ThisFunction->GetType() 
		 << " Other function " << this->ThisFunction->GetName()
	 	 << this->ThisFunction->GetType()<< " Tid = "<<tid<<endl;
	    } else {
	      cout <<"CurrentProfiler is not Null but its FunctionInfo is"<<endl;
	    }
	  }
	}
	// While exiting, reset value of CurrentProfiler to reflect the parent
	CurrentProfiler[tid] = ParentProfiler;
 	DEBUGPROFMSG("nct "<< RtsLayer::myNode() << ","
            << RtsLayer::myContext() << "," << tid
	    << " Stop: " << ThisFunction->GetName() 
	    << " TheSafeToDumpData() = " << TheSafeToDumpData()
	    << " CurrProf = "<<CurrentProfiler[tid] << " this = "
	    << this<<endl;);

        if (ParentProfiler == (Profiler *) NULL) {
	  // For Dyninst. tcf gets called after main and all the data structures may not be accessible
	  // after main exits. Still needed on Linux - we use TauProgramTermination()
	  if (strcmp(ThisFunction->GetName(), "_fini") == 0) TheSafeToDumpData() = 0;
	  #ifndef TAU_WINDOWS
	  atexit(Tau_shutdown);
	  #endif //TAU_WINDOWS
  	  if (TheSafeToDumpData()) {
            if (!RtsLayer::isCtorDtor(ThisFunction->GetName())) {
            // Not a destructor of a static object - its a function like main
              DEBUGPROFMSG("nct " << RtsLayer::myNode() << "," 
  	      << RtsLayer::myContext() << "," << tid  << " "
              << "Profiler::Stop() : Reached top level function: dumping data"
              << ThisFunction->GetName() <<endl;);
  
              StoreData(tid);
	    }
        // dump data here. Dump it only at the exit of top level profiler.
	  }
        }

      } // if TheProfileMask() 
      else 
      { /* set current profiler properly */
	CurrentProfiler[tid] = ParentProfiler; 
      }
}

//////////////////////////////////////////////////////////////////////

Profiler::~Profiler() {

     if (!StartStopUsed_) {
	Stop();
      } // If ctor dtor interface is used then call Stop. 
	// If the Profiler object is going out of scope without Stop being
	// called, call it now!

}

//////////////////////////////////////////////////////////////////////

void Profiler::ProfileExit(const char *message, int tid)
{
  Profiler *current;

  current = CurrentProfiler[tid];

  DEBUGPROFMSG("nct "<< RtsLayer::myNode() << " RtsLayer::ProfileExit called :"
    << message << endl;);
  if (current == 0) 
  {   
     DEBUGPROFMSG("Current is NULL, No need to store data TID = " << tid << endl;);
     //StoreData(tid);
  }
  else 
  {  
    while (current != 0) {
      DEBUGPROFMSG("Thr "<< RtsLayer::myNode() << " ProfileExit() calling Stop:"        << current->ThisFunction->GetName() << " " 
        << current->ThisFunction->GetType() << endl;);
      current->Stop(tid); // clean up 
  
      if (current->ParentProfiler == 0) {
        if (!RtsLayer::isCtorDtor(current->ThisFunction->GetName())) {
         // Not a destructor of a static object - its a function like main
           DEBUGPROFMSG("Thr " << RtsLayer::myNode()
             << " ProfileExit() : Reached top level function - dumping data"
             << endl;);
  
        //    StoreData(tid); // static now. Don't need current. 
        // The above Stop should call StoreData. We needn't do it again.
        }
      }
  
      current = CurrentProfiler[tid]; // Stop should set it
    }
  }

}

//////////////////////////////////////////////////////////////////////

void Profiler::theFunctionList(const char ***inPtr, int *numOfFunctions, bool addName, const char * inString)
{
  //static const char *const functionList[START_SIZE];
  static int numberOfFunctions = 0;
  static int sizeOfArray = 2;
  static const char **functionList = ( char const **) malloc( sizeof(char *) * 2);

  if(addName){
    //Note that the add only occurs when a thread is initializing a FunctionInfo
    //object.  As such, we already have a lock in progress.
    if(numberOfFunctions == sizeOfArray){
      //Increase the size of the array.
      sizeOfArray *= 2;
      functionList = (const char **) realloc(functionList, sizeof(char *) * sizeOfArray);
    }

    functionList[numberOfFunctions] = inString;
    numberOfFunctions++;
  }
  else{
    //We do not want to pass back internal pointers.
    *inPtr = ( char const **) malloc( sizeof(char *) * numberOfFunctions);

    for(int i=0;i<numberOfFunctions;i++)
      (*inPtr)[i] = functionList[i]; //Need the () in (*inPtr)[i] or the dereferrencing is
    //screwed up!

    *numOfFunctions = numberOfFunctions;
  }
}

void Profiler::dumpFunctionNames()
{
  char *filename, *dumpfile, *errormsg;
  char *dirname;
  FILE* fp;

  int numOfFunctions;
  const char ** functionList;

  Profiler::theFunctionList(&functionList, &numOfFunctions);

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  //Create temp write to file.
  filename = new char[1024];
  sprintf(filename,"%s/temp.%d.%d",dirname, RtsLayer::myNode(),
	  RtsLayer::myContext());
  if ((fp = fopen (filename, "w+")) == NULL) {
    errormsg = new char[1024];
    sprintf(errormsg,"Error: Could not create %s",filename);
    perror(errormsg);
    return;
  }

  //Write data, and close.
  fprintf(fp, "number of functions %d\n", numOfFunctions);
  for(int i =0;i<numOfFunctions;i++){
    fprintf(fp, "%s\n", functionList[i]);
  }
  fclose(fp);
  
  //Rename from the temp filename.
  dumpfile = new char[1024];
  sprintf(dumpfile,"%s/dump_functionnames_n,c.%d.%d",dirname, RtsLayer::myNode(),
                RtsLayer::myContext());
  rename(filename, dumpfile);
}

#ifndef TAU_MULTIPLE_COUNTERS
void Profiler::theCounterList(const char ***inPtr, int *numOfCounters)
{
  *inPtr = ( char const **) malloc( sizeof(char *) * 1);
  char *tmpChar = "default counter";
  (*inPtr)[0] = tmpChar; //Need the () in (*inPtr)[j] or the dereferrencing is
  //screwed up!
  *numOfCounters = 1;
}

void Profiler::getFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 double ***counterExclusiveValues,
				 double ***counterInclusiveValues,
				 int **numOfCalls,
				 int **numOfSubRoutines,
				 const char ***counterNames,
				 int *numOfCounters,
				 int tid)
{
  TAU_PROFILE("TAU_GET_FUNCTION_VALUES()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  
  bool functionCheck = false;
  int currentFuncPos = -1;
  const char *tmpFunctionName = NULL;

  int tmpNumberOfCounters;
  const char ** tmpCounterList;

  Profiler::theCounterList(&tmpCounterList,
			   &tmpNumberOfCounters);

  *numOfCounters = tmpNumberOfCounters;
  *counterNames = tmpCounterList;

  //Allocate memory for the lists.
  *counterExclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  *counterInclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  for(int memAlloc=0;memAlloc<numOfFuncs;memAlloc++){
    (*counterExclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * 1);
    (*counterInclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * 1);
  }
  *numOfCalls = (int *) malloc(sizeof(int) * numOfFuncs);
  *numOfSubRoutines = (int *) malloc(sizeof(int) * numOfFuncs);

  double tmpDoubleExcl;
  double tmpDoubleIncl;

  double currenttime = 0;
  double prevtime = 0;
  double total = 0;

  currenttime = RtsLayer::getUSecD(tid);

  RtsLayer::LockDB();
  
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
    //Check to see that it is one of the requested functions.
    functionCheck = false;
    currentFuncPos = -1;
    tmpFunctionName = (*it)->GetName();
    for(int fc=0;fc<numOfFuncs;fc++){
      if(strcmp(inFuncs[fc], tmpFunctionName) == 0){
	functionCheck = true;
	currentFuncPos = fc;
	break;
      }
    }

    if(functionCheck){
      if ((*it)->GetAlreadyOnStack(tid)){
	/* it is on the callstack. We need to do some processing. */
	/* Calculate excltime, incltime */
	Profiler *current;
	/* Traverse the Callstack */
	current = CurrentProfiler[tid];
	
	if (current == 0){ /* current is null */
	  DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	}
	else{ /* current is not null */
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);
	  
	  //Initialize what gets added for
	  //reducing from the parent profile
	  prevtime = 0;
	  total = 0;
	  
	  while (current != 0){
	    /* Traverse the stack */ 
	    if ((*it) == current->ThisFunction){ /* Match! */
	      DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			   <<endl;);
	      total = currenttime - current->StartTime;
	      tmpDoubleExcl += total - prevtime;
	      /* prevtime is the inclusive time of the subroutine that should
		 be subtracted from the current exclusive time */ 
	      /* If there is no instance of this function higher on the 	
		 callstack, we should add the total to the inclusive time */
	    }
	    prevtime = currenttime - current->StartTime;  
	    
	    /* to calculate exclusive time */
	    current = current->ParentProfiler; 
	  } /* We've reached the top! */
	  tmpDoubleIncl += total;//add this to the inclusive time
	  //prevtime and incltime are calculated
	} /* Current is not null */
      } /* On call stack */
      else{ /* it is not on the callstack. */
	tmpDoubleExcl = (*it)->GetExclTime(tid);
	tmpDoubleIncl = (*it)->GetInclTime(tid);
      }// Not on the Callstack

      //Copy the data.
      (*numOfCalls)[currentFuncPos] = (*it)->GetCalls(tid);
      (*numOfSubRoutines)[currentFuncPos] = (*it)->GetSubrs(tid);
      
      (*counterInclusiveValues)[currentFuncPos][0] = tmpDoubleIncl;
      (*counterExclusiveValues)[currentFuncPos][0] = tmpDoubleExcl;
    }
  }
  RtsLayer::UnLockDB();
#endif //PROFILING_ON
}

int Profiler::dumpFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 bool increment,
				 int tid){
  
  TAU_PROFILE("GET_FUNC_VALS()", " ", TAU_IO);
#ifdef PROFILING_ON
	vector<FunctionInfo*>::iterator it;
  	vector<TauUserEvent*>::iterator eit;
	char *filename, *dumpfile, *errormsg, *header;
	char *dirname;
	FILE* fp;
 	int numFunc = numOfFuncs; 
	int numEvents;

	bool functionCheck = false;
	const char *tmpFunctionName = NULL;

#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	long listSize, numCalls;
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS
 	double excltime, incltime; 
	double currenttime, prevtime, total;

	DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);

#ifdef TRACING_ON
	TraceEvClose(tid);
	RtsLayer::DumpEDF(tid);
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	currenttime = RtsLayer::getUSecD(tid); 
	RtsLayer::LockDB();
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	filename = new char[1024];
	sprintf(filename,"%s/temp.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	DEBUGPROFMSG("Creating " << filename << endl;);
	/* Changed: TRUNCATE dump file */ 
	if ((fp = fopen (filename, "w+")) == NULL) {
	 	errormsg = new char[1024];
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
	/*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	    numFunc++;
	  }
	}
	*/
	//numFunc = TheFunctionDB().size();
	header = new char[256];

#if (defined (SGI_HW_COUNTERS) || defined (TAU_PCL) \
	|| (defined (TAU_PAPI) && \
         (!(defined(TAU_PAPI_WALLCLOCKTIME) || (defined (TAU_PAPI_VIRTUAL))))))
	sprintf(header,"%d templated_functions_hw_counters\n", numFunc);
#else  // SGI_TIMERS, TULIP_TIMERS 
	sprintf(header,"%d templated_functions\n", numFunc);
#endif // SGI_HW_COUNTERS 

#ifdef TAU_PAPI
	static const char * papi_env_var = getenv("PAPI_EVENT");
	if (papi_env_var != NULL)
	  sprintf(header,"%d templated_functions_hw_counters\n", numFunc);
#endif // TAU_PAPI 

	// Send out the format string
	strcat(header,"# Name Calls Subrs Excl Incl ");
#ifdef PROFILE_STATS
	strcat(header,"SumExclSqr ");
#endif //PROFILE_STATS
	strcat(header,"ProfileCalls\n");
	int sz = strlen(header);
	int ret = fprintf(fp, "%s",header);	
	ret = fflush(fp);
	/*
	if (ret != sz) {
	  cout <<"ret not equal to strlen "<<endl;
 	}
        cout <<"Header: "<< tid << " : bytes " <<ret <<":"<<header ;
	*/
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
          /* if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask())
	     { 
	  */
	  
	  //Check to see that it is one of the requested functions.
	  functionCheck = false;
	  tmpFunctionName = (*it)->GetName();
	  for(int fc=0;fc<numOfFuncs;fc++){
	    if(strcmp(inFuncs[fc], tmpFunctionName) == 0){
	      functionCheck = true;
	      break;
	    }
	  }
	  if(functionCheck){
	    if ((*it)->GetAlreadyOnStack(tid)) { 
	      /* it is on the callstack. We need to do some processing. */
	      /* Calculate excltime, incltime */
	      Profiler *current; 
	      /* Traverse the Callstack */
	      current = CurrentProfiler[tid];
	      
	      if (current == 0){ /* current is null */
		DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	      }
	      else{ /* current is not null */
		incltime = (*it)->GetInclTime(tid); 
		excltime = (*it)->GetExclTime(tid); 
		total = 0;  /* Initialize what gets added */
		prevtime = 0; /* for reducing from the parent profiler */
		while (current != 0){
		  /* Traverse the stack */ 
		  if ((*it) == current->ThisFunction){ /* Match! */
		    DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
				 <<endl;);
		    total = currenttime - current->StartTime; 
		    excltime += total - prevtime; 
		    /* prevtime is the inclusive time of the subroutine that should
		       be subtracted from the current exclusive time */ 
		    /* If there is no instance of this function higher on the 	
		       callstack, we should add the total to the inclusive time */
		  }
		  prevtime = currenttime - current->StartTime;  
		  /* to calculate exclusive time */
		  
		  current = current->ParentProfiler; 
		} /* We've reached the top! */
		incltime += total; /* add this to the inclusive time */ 
		/* prevtime and incltime are calculated */
	      } /* Current is not null */
	    } /* On call stack */
	    else{ /* it is not on the callstack. */ 
	      excltime = (*it)->GetExclTime(tid);
	      incltime = (*it)->GetInclTime(tid); 
	    } // Not on the Callstack
	    
	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
			 << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
			 << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid) 
			 << " Excl : " << excltime << " Incl : " << incltime << endl;);
	    
	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
		    (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
		    excltime, incltime); 
	    
	    fprintf(fp,"0 \n"); // Indicating - profile calls is turned off 
	    /*
	      } // ProfileGroup test 
	    */
	  }
	} // for loop. End of FunctionInfo data
	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
	// Change this when aggregate profiling in introduced in Pooma 
	
	// Print UserEvent Data if any
	
	numEvents = 0;
 	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	  {
	    if ((*eit)->GetNumEvents(tid)) { 
	      numEvents++;
	    }
	  }
	
	if (numEvents > 0) {
	  // Data format 
	  // # % userevents
	  // # name numsamples max min mean sumsqr 
    	  fprintf(fp, "%d userevents\n", numEvents);
    	  fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	  
    	  vector<TauUserEvent*>::iterator it;
    	  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
	    {
	      
	      DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
			   (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid) 
			   << "\n Max " << (*it)->GetMax(tid) << "\n Mean " 
			   << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid) 
			   << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	      
	      fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
		      (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
		      (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
	    }
	}
	// End of userevents data 
	
	RtsLayer::UnLockDB();
	fclose(fp);
	dumpfile = new char[1024];
	if(increment){
	  //Place the date and time to the dumpfile name:
	  time_t theTime = time(NULL);
	  char *stringTime = ctime(&theTime);
	  tm *structTime = localtime(&theTime);
	  char *day = strtok(stringTime," ");
	  char *month = strtok(NULL," ");
	  char *dayInt = strtok(NULL," ");
	  char *time = strtok(NULL," ");
	  char *year = strtok(NULL," ");
	  //Get rid of the mewline.
	  year[4] = '\0';
	  char *newStringTime = new char[1024];
	  sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	  
	  sprintf(dumpfile,"%s/sel_dump__%s__.%d.%d.%d",dirname,
		  newStringTime,
		  RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	}
	else{
	  sprintf(dumpfile,"%s/dump.%d.%d.%d",dirname, RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	} 
	
	
#endif //PROFILING_ON
	return 1;
}

int Profiler::StoreData(int tid)
{
#ifdef PROFILING_ON 
	vector<FunctionInfo*>::iterator it;
  	vector<TauUserEvent*>::iterator eit;
	char *filename, *errormsg, *header;
	char *dirname;
	FILE* fp;
 	int numFunc, numEvents;
#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	long listSize, numCalls;
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS

	DEBUGPROFMSG("Profiler::StoreData( tid = "<<tid <<" ) "<<endl;);

#ifdef TRACING_ON
	TraceEvClose(tid);
	RtsLayer::DumpEDF(tid);
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	RtsLayer::LockDB();
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	filename = new char[1024];
	sprintf(filename,"%s/profile.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	DEBUGPROFMSG("Creating " << filename << endl;);
	if ((fp = fopen (filename, "w+")) == NULL) {
	 	errormsg = new char[1024];
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
	/*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	    numFunc++;
	  }
	}
	*/
	numFunc = TheFunctionDB().size();
	header = new char[256];

#if (defined (SGI_HW_COUNTERS) || defined (TAU_PCL) \
	|| (defined (TAU_PAPI) && \
         (!(defined(TAU_PAPI_WALLCLOCKTIME) || (defined (TAU_PAPI_VIRTUAL))))))
	sprintf(header,"%d templated_functions_hw_counters\n", numFunc);
#else  // SGI_TIMERS, TULIP_TIMERS 
	sprintf(header,"%d templated_functions\n", numFunc);
#endif // SGI_HW_COUNTERS 

#ifdef TAU_PAPI
	static const char * papi_env_var = getenv("PAPI_EVENT");
	if (papi_env_var != NULL)
	  sprintf(header,"%d templated_functions_hw_counters\n", numFunc);
#endif // TAU_PAPI 
	  
	
	// Send out the format string
	strcat(header,"# Name Calls Subrs Excl Incl ");
#ifdef PROFILE_STATS
	strcat(header,"SumExclSqr ");
#endif //PROFILE_STATS
	strcat(header,"ProfileCalls\n");
	int sz = strlen(header);
	int ret = fprintf(fp, "%s",header);	
	ret = fflush(fp);
	/*
	if (ret != sz) {
	  cout <<"ret not equal to strlen "<<endl;
 	}
        cout <<"Header: "<< tid << " : bytes " <<ret <<":"<<header ;
	*/
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
	/*
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	  */
  
  	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
  	      << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
              << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid) 
  	      << " Excl : " << (*it)->GetExclTime(tid) << " Incl : " 
  	      << (*it)->GetInclTime(tid) << endl;);
  	
  	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
  	      (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
  	      (*it)->GetExclTime(tid), (*it)->GetInclTime(tid));

#ifdef PROFILE_STATS 
  	    fprintf(fp,"%.16G ", (*it)->GetSumExclSqr(tid));
#endif //PROFILE_STATS
  
#ifdef PROFILE_CALLS
  	    listSize = (long) (*it)->ExclInclCallList->size(); 
  	    numCalls = (*it)->GetCalls(tid);
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
	    /*
	  } // ProfileGroup test 
	  */
	} // for loop. End of FunctionInfo data
	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
	RtsLayer::UnLockDB();
	// Change this when aggregate profiling in introduced in Pooma 

	// Print UserEvent Data if any
	
	numEvents = 0;
 	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
          if ((*eit)->GetNumEvents(tid)) { 
	    numEvents++;
	  }
	}

	if (numEvents > 0) {
    	// Data format 
    	// # % userevents
    	// # name numsamples max min mean sumsqr 
    	  fprintf(fp, "%d userevents\n", numEvents);
    	  fprintf(fp, "# eventname numevents max min mean sumsqr\n");

    	  vector<TauUserEvent*>::iterator it;
    	  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
    	  {
      
	    DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
              (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid) 
              << "\n Max " << (*it)->GetMax(tid) << "\n Mean " 
	      << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid) 
	      << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);

     	    fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
	    (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	    (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
    	  }
	}
	// End of userevents data 

	fclose(fp);

#endif //PROFILING_ON
	return 1;
}

int Profiler::DumpData(bool increment, int tid)
{
  	TAU_PROFILE("TAU_DUMP_DB()", " ", TAU_IO);
#ifdef PROFILING_ON
	vector<FunctionInfo*>::iterator it;
  	vector<TauUserEvent*>::iterator eit;
	char *filename, *dumpfile, *errormsg, *header;
	char *dirname;
	FILE* fp;
 	int numFunc, numEvents;
#endif //PROFILING_ON
#ifdef PROFILE_CALLS
	long listSize, numCalls;
	list<pair<double,double> >::iterator iter;
#endif // PROFILE_CALLS
 	double excltime, incltime; 
	double currenttime, prevtime, total;

	DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);

#ifdef TRACING_ON
	TraceEvClose(tid);
	RtsLayer::DumpEDF(tid);
#endif // TRACING_ON 

#ifdef PROFILING_ON 
	currenttime = RtsLayer::getUSecD(tid); 
	RtsLayer::LockDB();
	if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	   dirname  = new char[8];
	   strcpy (dirname,".");
	}
	 
	filename = new char[1024];
	sprintf(filename,"%s/temp.%d.%d.%d",dirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	DEBUGPROFMSG("Creating " << filename << endl;);
	/* Changed: TRUNCATE dump file */ 
	if ((fp = fopen (filename, "w+")) == NULL) {
	 	errormsg = new char[1024];
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
	/*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	    numFunc++;
	  }
	}
	*/
	numFunc = TheFunctionDB().size();
	header = new char[256];

#if (defined (SGI_HW_COUNTERS) || defined (TAU_PCL) \
	|| (defined (TAU_PAPI) && \
         (!(defined(TAU_PAPI_WALLCLOCKTIME) || (defined (TAU_PAPI_VIRTUAL))))))
	sprintf(header,"%d templated_functions_hw_counters\n", numFunc);
#else  // SGI_TIMERS, TULIP_TIMERS 
	sprintf(header,"%d templated_functions\n", numFunc);
#endif // SGI_HW_COUNTERS 

#ifdef TAU_PAPI
	static const char * papi_env_var = getenv("PAPI_EVENT");
	if (papi_env_var != NULL)
	  sprintf(header,"%d templated_functions_hw_counters\n", numFunc);
#endif // TAU_PAPI 

	// Send out the format string
	strcat(header,"# Name Calls Subrs Excl Incl ");
#ifdef PROFILE_STATS
	strcat(header,"SumExclSqr ");
#endif //PROFILE_STATS
	strcat(header,"ProfileCalls\n");
	int sz = strlen(header);
	int ret = fprintf(fp, "%s",header);	
	ret = fflush(fp);
	/*
	if (ret != sz) {
	  cout <<"ret not equal to strlen "<<endl;
 	}
        cout <<"Header: "<< tid << " : bytes " <<ret <<":"<<header ;
	*/
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
          /* if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask())
	  { 
	  */

	    if ((*it)->GetAlreadyOnStack(tid)) 
	    { 
	      /* it is on the callstack. We need to do some processing. */
	      /* Calculate excltime, incltime */
	      Profiler *current; 
	      /* Traverse the Callstack */
	      current = CurrentProfiler[tid];
	  
	      if (current == 0)
	      { /* current is null */
		DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	      }
	      else 
	      { /* current is not null */
		incltime = (*it)->GetInclTime(tid); 
		excltime = (*it)->GetExclTime(tid); 
		total = 0;  /* Initialize what gets added */
		prevtime = 0; /* for reducing from the parent profiler */
		while (current != 0) 
		{
		  /* Traverse the stack */ 
		  if ((*it) == current->ThisFunction) 
	 	  { /* Match! */
		    DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
	  	      <<endl;);
		    total = currenttime - current->StartTime; 
		    excltime += total - prevtime; 
		    /* prevtime is the inclusive time of the subroutine that should
		       be subtracted from the current exclusive time */ 
		    /* If there is no instance of this function higher on the 	
			callstack, we should add the total to the inclusive time */
		  }
        	  prevtime = currenttime - current->StartTime;  
		  /* to calculate exclusive time */

	          current = current->ParentProfiler; 
	        } /* We've reached the top! */
		incltime += total; /* add this to the inclusive time */ 
		/* prevtime and incltime are calculated */
	      } /* Current is not null */
	    } /* On call stack */
 	    else 
	    { /* it is not on the callstack. */ 
	      excltime = (*it)->GetExclTime(tid);
	      incltime = (*it)->GetInclTime(tid); 
	    } // Not on the Callstack

	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping " 
  	      << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : " 
              << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid) 
  	      << " Excl : " << excltime << " Incl : " << incltime << endl;);
  	
  	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(), 
  	      (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid), 
	      excltime, incltime); 

  	    fprintf(fp,"0 \n"); // Indicating - profile calls is turned off 
	    /*
	  } // ProfileGroup test 
	  */
	} // for loop. End of FunctionInfo data
	fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
	// Change this when aggregate profiling in introduced in Pooma 

	// Print UserEvent Data if any
	
	numEvents = 0;
 	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
          if ((*eit)->GetNumEvents(tid)) { 
	    numEvents++;
	  }
	}

	if (numEvents > 0) {
    	// Data format 
    	// # % userevents
    	// # name numsamples max min mean sumsqr 
    	  fprintf(fp, "%d userevents\n", numEvents);
    	  fprintf(fp, "# eventname numevents max min mean sumsqr\n");

    	  vector<TauUserEvent*>::iterator it;
    	  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
    	  {
      
	    DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
              (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid) 
              << "\n Max " << (*it)->GetMax(tid) << "\n Mean " 
	      << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid) 
	      << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);

     	    fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n", 
	    (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
	    (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
    	  }
	}
	// End of userevents data 

	RtsLayer::UnLockDB();
	fclose(fp);
	dumpfile = new char[1024];
	if(increment){
	  //Place the date and time to the dumpfile name:
	  time_t theTime = time(NULL);
	  char *stringTime = ctime(&theTime);
	  tm *structTime = localtime(&theTime);
	  char *day = strtok(stringTime," ");
	  char *month = strtok(NULL," ");
	  char *dayInt = strtok(NULL," ");
	  char *time = strtok(NULL," ");
	  char *year = strtok(NULL," ");
	  //Get rid of the mewline.
	  year[4] = '\0';
	  char *newStringTime = new char[1024];
	  sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	  
	  sprintf(dumpfile,"%s/dump__%s__.%d.%d.%d",dirname,
		  newStringTime,
		  RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	}
	else{
	  sprintf(dumpfile,"%s/dump.%d.%d.%d",dirname, RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	} 


#endif //PROFILING_ON
	return 1;
}

void Profiler::PurgeData(int tid)
{
	vector<FunctionInfo*>::iterator it;
	vector<TauUserEvent*>::iterator eit;
	Profiler *curr;

	DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
	RtsLayer::LockDB();

	// Reset The Function Database (save callstack entries)
	for(it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
// May be able to recycle fns which never get called again??
	    (*it)->SetCalls(tid,0);
	    (*it)->SetSubrs(tid,0);
	    (*it)->SetExclTime(tid,0);
	    (*it)->SetInclTime(tid,0);
#ifdef PROFILE_STATS
	    (*it)->SetSumExclSqr(tid,0);
#endif //PROFILE_STATS
#ifdef PROFILE_CALLS
	    (*it)->ExclInclCallList->clear();
#endif // PROFILE_CALLS
/*
	  }
*/
	}
	// Now Re-register callstack entries
	curr = CurrentProfiler[tid];
	curr->ThisFunction->IncrNumCalls(tid);
	curr = curr->ParentProfiler;
	while(curr != 0) {
	  curr->ThisFunction->IncrNumCalls(tid);
	  curr->ThisFunction->IncrNumSubrs(tid);
	  curr = curr->ParentProfiler;
	}

	// Reset the Event Database
	for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
	  (*eit)->LastValueRecorded[tid] = 0;
	  (*eit)->NumEvents[tid] = 0L;
	  (*eit)->MinValue[tid] = 9999999;
	  (*eit)->MaxValue[tid] = -9999999;
	  (*eit)->SumSqrValue[tid] = 0;
	  (*eit)->SumValue[tid] = 0;
	}

	RtsLayer::UnLockDB();
}
//////////////////////////////////////////////////////////////////////
#else //TAU_MULTIPLE_COUNTERS


bool Profiler::createDirectories(){

  char *dirname;
  static bool flag = true;
  RtsLayer::LockDB();
  if (flag) {
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      char *newdirname = new char[1024];
      char *rmdircommand = new char[1024];
      char *mkdircommand = new char[1024];
      
      if ((dirname = getenv("PROFILEDIR")) == NULL) {
	// Use default directory name .
	dirname  = new char[8];
	strcpy (dirname,".");
      }
      
      sprintf(newdirname,"%s/%s",dirname,tmpChar);
      sprintf(rmdircommand,"rm -rf %s",newdirname);
      sprintf(mkdircommand,"mkdir %s",newdirname);
    
      system(rmdircommand);
      system(mkdircommand);
    }
  }
    flag = false;
  }
  RtsLayer::UnLockDB();
  return true;
}

void Profiler::getFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 double ***counterExclusiveValues,
				 double ***counterInclusiveValues,
				 int **numOfCalls,
				 int **numOfSubRoutines,
				 const char ***counterNames,
				 int *numOfCounters,
				 int tid)
{
  TAU_PROFILE("TAU_GET_FUNCTION_VALUES()", " ", TAU_IO);

#ifdef PROFILING_ON

  vector<FunctionInfo*>::iterator it;

  bool functionCheck = false;
  int currentFuncPos = -1;
  const char *tmpFunctionName = NULL;
  bool memAllocated = false; //Used to help with memory cleanup.

  int tmpNumberOfCounters;
  bool * tmpCounterUsedList;
  const char ** tmpCounterList;

  MultipleCounterLayer::theCounterListInternal(&tmpCounterList,
					       &tmpNumberOfCounters,
					       &tmpCounterUsedList);

  *numOfCounters = tmpNumberOfCounters;
  *counterNames = tmpCounterList;

  //Allocate memory for the lists.
  *counterExclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  *counterInclusiveValues = ( double **) malloc( sizeof(double *) * numOfFuncs);
  for(int memAlloc=0;memAlloc<numOfFuncs;memAlloc++){
    (*counterExclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * tmpNumberOfCounters);
    (*counterInclusiveValues)[memAlloc] = ( double *) malloc( sizeof(double) * tmpNumberOfCounters);
  }
  *numOfCalls = (int *) malloc(sizeof(int) * numOfFuncs);
  *numOfSubRoutines = (int *) malloc(sizeof(int) * numOfFuncs);

  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

  double currenttime[MAX_TAU_COUNTERS];
  double prevtime[MAX_TAU_COUNTERS];
  double total[MAX_TAU_COUNTERS];

  for(int a=0;a<MAX_TAU_COUNTERS;a++){
    currenttime[a]=0;
    prevtime[a]=0;
    total[a]=0;
  }

  RtsLayer::getUSecD(tid, currenttime);

  RtsLayer::LockDB();
  
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
    //Check to see that it is one of the requested functions.
    functionCheck = false;
    currentFuncPos = -1;
    tmpFunctionName = (*it)->GetName();
    for(int fc=0;fc<numOfFuncs;fc++){
      if(strcmp(inFuncs[fc], tmpFunctionName) == 0){
	functionCheck = true;
	currentFuncPos = fc;
	break;
      }
    }
    if(functionCheck){
      if ((*it)->GetAlreadyOnStack(tid)){
	/* it is on the callstack. We need to do some processing. */
	/* Calculate excltime, incltime */
	Profiler *current;
	/* Traverse the Callstack */
	current = CurrentProfiler[tid];
	
	if (current == 0){ /* current is null */
	  DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	}
	else{ /* current is not null */
	  //These calls return pointers to new memory.
	  //Remember to free this memory after use!!!
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);
	  memAllocated = true;
	  
	  //Initialize what gets added for
	  //reducing from the parent profile
	  for(int j=0;j<MAX_TAU_COUNTERS;j++){
	    prevtime[j]=0;
	    total[j]=0;
	  }
	  
	  while (current != 0){
	    /* Traverse the stack */ 
	    if ((*it) == current->ThisFunction){ /* Match! */
	      DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			   <<endl;);
	      
	      for(int k=0;k<MAX_TAU_COUNTERS;k++){
		total[k] = currenttime[k] - current->StartTime[k];
		tmpDoubleExcl[k] += total[k] - prevtime[k];
	      }
	      /* prevtime is the inclusive time of the subroutine that should
		 be subtracted from the current exclusive time */ 
	      /* If there is no instance of this function higher on the 	
		 callstack, we should add the total to the inclusive time */
	    }
	    for(int l=0;l<MAX_TAU_COUNTERS;l++){
	      prevtime[l] = currenttime[l] - current->StartTime[l];  
	    }
	    /* to calculate exclusive time */
	    current = current->ParentProfiler; 
	  } /* We've reached the top! */
	  for(int m=0;m<MAX_TAU_COUNTERS;m++){
	    tmpDoubleIncl[m] += total[m];//add this to the inclusive time
	    //prevtime and incltime are calculated
	  }
	} /* Current is not null */
      } /* On call stack */
      else{ /* it is not on the callstack. */
	//These calls return pointers to new memory.
	//Remember to free this memory after use!!!
	tmpDoubleExcl = (*it)->GetExclTime(tid);
	tmpDoubleIncl = (*it)->GetInclTime(tid);
	memAllocated = true;
      }// Not on the Callstack

      //Copy the data.
      (*numOfCalls)[currentFuncPos] = (*it)->GetCalls(tid);
      (*numOfSubRoutines)[currentFuncPos] = (*it)->GetSubrs(tid);
      
      int posCounter = 0;
      if(memAllocated){
	for(int copyData=0;copyData<MAX_TAU_COUNTERS;copyData++){
	  if(tmpCounterUsedList[copyData]){
	    (*counterInclusiveValues)[currentFuncPos][posCounter] = tmpDoubleIncl[copyData];
	    (*counterExclusiveValues)[currentFuncPos][posCounter] = tmpDoubleExcl[copyData];
	    posCounter++;
	  }
	}
      }
      else{
	for(int copyData=0;copyData<MAX_TAU_COUNTERS;copyData++){
	  if(tmpCounterUsedList[copyData]){
	    (*counterInclusiveValues)[currentFuncPos][posCounter] = 0;
	    (*counterExclusiveValues)[currentFuncPos][posCounter] = 0;
	    posCounter++;
	  }
	}
      }
      //Free up the memory if it was allocated.
      if(memAllocated){
	free(tmpDoubleIncl);
	free(tmpDoubleExcl);
      }
    }
  }
  RtsLayer::UnLockDB();
#endif //PROFILING_ON
}

int Profiler::dumpFunctionValues(const char **inFuncs,
				 int numOfFuncs,
				 bool increment,
				 int tid){
  
  TAU_PROFILE("GET_FUNC_VALS()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;

  bool functionCheck = false;
  const char *tmpFunctionName = NULL;

  FILE* fp;
  char *dirname, *dumpfile;
  int numFunc = numOfFuncs; 
  int numEvents;

  bool memAllocated = false; //Used to help with memory cleanup.
  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

  double currenttime[MAX_TAU_COUNTERS];
  double prevtime[MAX_TAU_COUNTERS];
  double total[MAX_TAU_COUNTERS];

  for(int a=0;a<MAX_TAU_COUNTERS;a++){
    currenttime[a]=0;
    prevtime[a]=0;
    total[a]=0;
  }

  RtsLayer::getUSecD(tid, currenttime);

  DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);


  //Create directories for storage.
  static bool createFlag = createDirectories();

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      RtsLayer::LockDB();

      char *newdirname = new char[1024];
      char *filename = new char[1024];
      char *errormsg = new char[1024];
      char *header = new char[1024];

      sprintf(newdirname,"%s/%s",dirname,tmpChar);

      sprintf(filename,"%s/temp.%d.%d.%d",newdirname, RtsLayer::myNode(),
	      RtsLayer::myContext(), tid);

      DEBUGPROFMSG("Creating " << filename << endl;);
      if ((fp = fopen (filename, "w+")) == NULL) {
      errormsg = new char[1024];
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
      /*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
	if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	numFunc++;
	}
	}
      */

      //numFunc = TheFunctionDB().size();

      //Setting the header to the correct name.
      sprintf(header,"%d templated_functions_MULTI_%s\n", numFunc, tmpChar);
  
      strcat(header,"# Name Calls Subrs Excl Incl ");

      strcat(header,"ProfileCalls\n");
      int sz = strlen(header);
      int ret = fprintf(fp, "%s",header);
      ret = fflush(fp);

      for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
	//Check to see that it is one of the requested functions.
	functionCheck = false;
	tmpFunctionName = (*it)->GetName();
	for(int fc=0;fc<numOfFuncs;fc++){
	  if(strcmp(inFuncs[fc], tmpFunctionName) == 0){
	    functionCheck = true;
	    break;
	  }
	}
	if(functionCheck){
	  if ((*it)->GetAlreadyOnStack(tid)){
	    /* it is on the callstack. We need to do some processing. */
	    /* Calculate excltime, incltime */
	    Profiler *current;
	    /* Traverse the Callstack */
	    current = CurrentProfiler[tid];
	    
	    if (current == 0){ /* current is null */
	      DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	    }
	    else{ /* current is not null */
	      //These calls return pointers to new memory.
	      //Remember to free this memory after use!!!
	      tmpDoubleExcl = (*it)->GetExclTime(tid);
	      tmpDoubleIncl = (*it)->GetInclTime(tid);
	      memAllocated = true;
	      
	      //Initialize what gets added for
	      //reducing from the parent profile
	      for(int j=0;j<MAX_TAU_COUNTERS;j++){
		prevtime[j]=0;
		total[j]=0;
	      }
	      
	      while (current != 0){
		/* Traverse the stack */ 
		if ((*it) == current->ThisFunction){ /* Match! */
		  DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			       <<endl;);
		  
		  for(int k=0;k<MAX_TAU_COUNTERS;k++){
		    total[k] = currenttime[k] - current->StartTime[k];
		    tmpDoubleExcl[k] += total[k] - prevtime[k];
		  }
		  /* prevtime is the inclusive time of the subroutine that should
		     be subtracted from the current exclusive time */ 
		  /* If there is no instance of this function higher on the 	
		     callstack, we should add the total to the inclusive time */
		}
		for(int l=0;l<MAX_TAU_COUNTERS;l++){
		  prevtime[l] = currenttime[l] - current->StartTime[l];  
		}
		/* to calculate exclusive time */
		current = current->ParentProfiler; 
	      } /* We've reached the top! */
	      for(int m=0;m<MAX_TAU_COUNTERS;m++){
		tmpDoubleIncl[m] += total[m];//add this to the inclusive time
		//prevtime and incltime are calculated
	      }
	    } /* Current is not null */
	  } /* On call stack */
	  else{ /* it is not on the callstack. */
	    //These calls return pointers to new memory.
	    //Remember to free this memory after use!!!
	    tmpDoubleExcl = (*it)->GetExclTime(tid);
	    tmpDoubleIncl = (*it)->GetInclTime(tid);
	    memAllocated = true;
	  } // Not on the Callstack
	  
	  if(memAllocated){
	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
			 << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
			 << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
			 << " Excl : " << tmpDoubleExcl[i] << " Incl : "
			 << tmpDoubleIncl[i] << endl;);
	    
	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		    (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		    tmpDoubleExcl[i], tmpDoubleIncl[i]);
	  }
	  else{
	    DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
			 << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
			 << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
			 << " Excl : " << 0 << " Incl : "
			 << 0 << endl;);
	    
	    fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		    (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		    0, 0);
	  }
	  fprintf(fp,"0 \n"); // Indicating - profile calls is turned off
	  //Free up the memory if it was allocated.
	  if(memAllocated){
	    free(tmpDoubleIncl);
	    free(tmpDoubleExcl);
	  }
	}
      }
      fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
      
      numEvents = 0;
      for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
	  if ((*eit)->GetNumEvents(tid)) {
	    numEvents++;
	  }
	}
      
      if (numEvents > 0) {
	// Data format
	// # % userevents
	// # name numsamples max min mean sumsqr
	fprintf(fp, "%d userevents\n", numEvents);
	fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	
	vector<TauUserEvent*>::iterator it;
	for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++){
	  
	  DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
		       (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid)
		       << "\n Max " << (*it)->GetMax(tid) << "\n Mean "
		       << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid)
		       << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	  
	  fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n",
		  (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
		  (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
	}
      }
      
      // End of userevents data
      RtsLayer::UnLockDB();
      fclose(fp);
      
      dumpfile = new char[1024];
      if(increment){
	//Place the date and time to the dumpfile name:
	time_t theTime = time(NULL);
	char *stringTime = ctime(&theTime);
	tm *structTime = localtime(&theTime);
	char *day = strtok(stringTime," ");
	char *month = strtok(NULL," ");
	char *dayInt = strtok(NULL," ");
	char *time = strtok(NULL," ");
	char *year = strtok(NULL," ");
	//Get rid of the mewline.
	year[4] = '\0';
	char *newStringTime = new char[1024];
	sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	
	sprintf(dumpfile,"%s/sel_dump__%s__.%d.%d.%d",newdirname,
		newStringTime,
		RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	
	rename(filename, dumpfile);
      }
      else{
	sprintf(dumpfile,"%s/sel_dump.%d.%d.%d",newdirname, RtsLayer::myNode(),
		RtsLayer::myContext(), tid);
	rename(filename, dumpfile);
      }
    }
  }
#endif //PROFILING_ON
  return 1;
}

int Profiler::StoreData(int tid){
#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;
  FILE* fp;
  char *dirname;
  int numFunc, numEvents;
  
  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

#endif //PROFILING_ON

  DEBUGPROFMSG("Profiler::StoreData( tid = "<<tid <<" ) "<<endl;);

#ifdef PROFILING_ON
  
  //Create directories for storage.
  static bool createFlag = createDirectories();

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      RtsLayer::LockDB();
      
      char *newdirname = new char[1024];
      char *filename = new char[1024];
      char *errormsg = new char[1024];
      char *header = new char[1024];
      

      sprintf(newdirname,"%s/%s",dirname,tmpChar);

      sprintf(filename,"%s/profile.%d.%d.%d",newdirname, RtsLayer::myNode(),
            RtsLayer::myContext(), tid);

      DEBUGPROFMSG("Creating " << filename << endl;);
      if ((fp = fopen (filename, "w+")) == NULL) {
      errormsg = new char[1024];
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
      
      numFunc = TheFunctionDB().size();
      
      //Setting the header to the correct name.
      sprintf(header,"%d templated_functions_MULTI_%s\n", numFunc, tmpChar);
  
      strcat(header,"# Name Calls Subrs Excl Incl ");

      strcat(header,"ProfileCalls\n");
      int sz = strlen(header);
      int ret = fprintf(fp, "%s",header);
      ret = fflush(fp);

      

      for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
        {
	  //These calls return pointers to new memory.
	  //Remember to free this memory after use!!! 
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);

	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
		       << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
		       << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
		       << " Excl : " << tmpDoubleExcl[i] << " Incl : "
		       << tmpDoubleIncl[i] << endl;);
	  
	  fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		  (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		  tmpDoubleExcl[i], tmpDoubleIncl[i]);

	  fprintf(fp,"0 \n"); // Indicating - profile calls is turned off
	  
	  //Free up the memory.
	  free(tmpDoubleIncl);
	  free(tmpDoubleExcl);
	}

      fprintf(fp,"0 aggregates\n"); // For now there are no aggregates
  
      RtsLayer::UnLockDB();
      
      numEvents = 0;
      for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
	  if ((*eit)->GetNumEvents(tid)) {
	    numEvents++;
	  }
	}
      
      if (numEvents > 0) {
	// Data format
	// # % userevents
	// # name numsamples max min mean sumsqr
	fprintf(fp, "%d userevents\n", numEvents);
	fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	
	vector<TauUserEvent*>::iterator it;
	for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++){
	  
	  DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
		       (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid)
		       << "\n Max " << (*it)->GetMax(tid) << "\n Mean "
		       << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid)
		       << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	  
	  fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n",
		  (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
		  (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
	}
      }
      
      // End of userevents data

      fclose(fp);
    }
  }
#endif //PROFILING_ON
 
  return 1;
}
int Profiler::DumpData(bool increment, int tid){
  
  TAU_PROFILE("TAU_DUMP_DB()", " ", TAU_IO);

#ifdef PROFILING_ON
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;

  FILE* fp;
  char *dirname, *dumpfile;
  int numFunc, numEvents;

  bool memAllocated = false; //Used to help with memory cleanup.
  double * tmpDoubleExcl;
  double * tmpDoubleIncl;

  double currenttime[MAX_TAU_COUNTERS];
  double prevtime[MAX_TAU_COUNTERS];
  double total[MAX_TAU_COUNTERS];

  for(int a=0;a<MAX_TAU_COUNTERS;a++){
    currenttime[a]=0;
    prevtime[a]=0;
    total[a]=0;
  }

  RtsLayer::getUSecD(tid, currenttime);

  DEBUGPROFMSG("Profiler::DumpData( tid = "<<tid <<" ) "<<endl;);


  //Create directories for storage.
  static bool createFlag = createDirectories();

  if ((dirname = getenv("PROFILEDIR")) == NULL) {
    // Use default directory name .
    dirname  = new char[8];
    strcpy (dirname,".");
  }

  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    if(MultipleCounterLayer::getCounterUsed(i)){
      char * tmpChar = MultipleCounterLayer::getCounterNameAt(i);
      RtsLayer::LockDB();

      char *newdirname = new char[1024];
      char *filename = new char[1024];
      char *errormsg = new char[1024];
      char *header = new char[1024];

      sprintf(newdirname,"%s/%s",dirname,tmpChar);

      sprintf(filename,"%s/temp.%d.%d.%d",newdirname, RtsLayer::myNode(),
	      RtsLayer::myContext(), tid);

      DEBUGPROFMSG("Creating " << filename << endl;);
      if ((fp = fopen (filename, "w+")) == NULL) {
      errormsg = new char[1024];
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
      /*
	numFunc = 0;
 	for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++)
	{
	if ((*it)->GetProfileGroup() & RtsLayer::TheProfileMask()) { 
	numFunc++;
	}
	}
      */

      numFunc = TheFunctionDB().size();

      //Setting the header to the correct name.
      sprintf(header,"%d templated_functions_MULTI_%s\n", numFunc, tmpChar);
  
      strcat(header,"# Name Calls Subrs Excl Incl ");

      strcat(header,"ProfileCalls\n");
      int sz = strlen(header);
      int ret = fprintf(fp, "%s",header);
      ret = fflush(fp);

      for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++){
	if ((*it)->GetAlreadyOnStack(tid)){
	  /* it is on the callstack. We need to do some processing. */
	  /* Calculate excltime, incltime */
	  Profiler *current;
	  /* Traverse the Callstack */
	  current = CurrentProfiler[tid];

	  if (current == 0){ /* current is null */
	    DEBUGPROFMSG("Current is NULL when it should be on the stack! TID = " << tid << endl;);
	  }
	  else{ /* current is not null */
	    //These calls return pointers to new memory.
	    //Remember to free this memory after use!!!
	    tmpDoubleExcl = (*it)->GetExclTime(tid);
	    tmpDoubleIncl = (*it)->GetInclTime(tid);
	    memAllocated = true;
	    
	    //Initialize what gets added for
	    //reducing from the parent profile
	    for(int j=0;j<MAX_TAU_COUNTERS;j++){
	      prevtime[j]=0;
	      total[j]=0;
	    }

	    while (current != 0){
	      /* Traverse the stack */ 
	      if ((*it) == current->ThisFunction){ /* Match! */
		DEBUGPROFMSG("MATCH! Name :"<<current->ThisFunction->GetName()
			     <<endl;);

		for(int k=0;k<MAX_TAU_COUNTERS;k++){
		  total[k] = currenttime[k] - current->StartTime[k];
		  tmpDoubleExcl[k] += total[k] - prevtime[k];
		}
		/* prevtime is the inclusive time of the subroutine that should
		   be subtracted from the current exclusive time */ 
		/* If there is no instance of this function higher on the 	
		   callstack, we should add the total to the inclusive time */
	      }
	      for(int l=0;l<MAX_TAU_COUNTERS;l++){
		prevtime[l] = currenttime[l] - current->StartTime[l];  
	      }
	      /* to calculate exclusive time */
	      current = current->ParentProfiler; 
	    } /* We've reached the top! */
	    for(int m=0;m<MAX_TAU_COUNTERS;m++){
	      tmpDoubleIncl[m] += total[m];//add this to the inclusive time
	      //prevtime and incltime are calculated
	    }
	  } /* Current is not null */
	} /* On call stack */
	else{ /* it is not on the callstack. */
	  //These calls return pointers to new memory.
	  //Remember to free this memory after use!!!
	  tmpDoubleExcl = (*it)->GetExclTime(tid);
	  tmpDoubleIncl = (*it)->GetInclTime(tid);
	  memAllocated = true;
	} // Not on the Callstack

	if(memAllocated){
	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
		       << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
		       << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
		       << " Excl : " << tmpDoubleExcl[i] << " Incl : "
		       << tmpDoubleIncl[i] << endl;);
	  
	  fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		  (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		  tmpDoubleExcl[i], tmpDoubleIncl[i]);
	}
	else{
	  DEBUGPROFMSG("Node: "<< RtsLayer::myNode() <<  " Dumping "
		       << (*it)->GetName()<< " "  << (*it)->GetType() << " Calls : "
		       << (*it)->GetCalls(tid) << " Subrs : "<< (*it)->GetSubrs(tid)
		       << " Excl : " << 0 << " Incl : "
		       << 0 << endl;);
	  
	  fprintf(fp,"\"%s %s\" %ld %ld %.16G %.16G ", (*it)->GetName(),
		  (*it)->GetType(), (*it)->GetCalls(tid), (*it)->GetSubrs(tid),
		  0, 0);
	}
	fprintf(fp,"0 \n"); // Indicating - profile calls is turned off
	 //Free up the memory if it was allocated.
	if(memAllocated){
	  free(tmpDoubleIncl);
	  free(tmpDoubleExcl);
	}
      }
      fprintf(fp,"0 aggregates\n"); // For now there are no aggregates

      numEvents = 0;
      for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++)
	{
	  if ((*eit)->GetNumEvents(tid)) {
	    numEvents++;
	  }
	}
      
      if (numEvents > 0) {
	// Data format
	// # % userevents
	// # name numsamples max min mean sumsqr
	fprintf(fp, "%d userevents\n", numEvents);
	fprintf(fp, "# eventname numevents max min mean sumsqr\n");
	
	vector<TauUserEvent*>::iterator it;
	for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++){
	  
	  DEBUGPROFMSG("Thr "<< tid << " TauUserEvent "<<
		       (*it)->GetEventName() << "\n Min " << (*it)->GetMin(tid)
		       << "\n Max " << (*it)->GetMax(tid) << "\n Mean "
		       << (*it)->GetMean(tid) << "\n SumSqr " << (*it)->GetSumSqr(tid)
		       << "\n NumEvents " << (*it)->GetNumEvents(tid)<< endl;);
	  
	  fprintf(fp, "\"%s\" %ld %.16G %.16G %.16G %.16G\n",
		  (*it)->GetEventName(), (*it)->GetNumEvents(tid), (*it)->GetMax(tid),
		  (*it)->GetMin(tid), (*it)->GetMean(tid), (*it)->GetSumSqr(tid));
	}
      }
      
      // End of userevents data
      RtsLayer::UnLockDB();
      fclose(fp);

      dumpfile = new char[1024];
      	if(increment){
	  //Place the date and time to the dumpfile name:
	  time_t theTime = time(NULL);
	  char *stringTime = ctime(&theTime);
	  tm *structTime = localtime(&theTime);
	  char *day = strtok(stringTime," ");
	  char *month = strtok(NULL," ");
	  char *dayInt = strtok(NULL," ");
	  char *time = strtok(NULL," ");
	  char *year = strtok(NULL," ");
	  //Get rid of the mewline.
	  year[4] = '\0';
	  char *newStringTime = new char[1024];
	  sprintf(newStringTime,"%s-%s-%s-%s-%s",day,month,dayInt,time,year);
	  
	  sprintf(dumpfile,"%s/dump__%s__.%d.%d.%d",newdirname,
		  newStringTime,
		  RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);

	  rename(filename, dumpfile);
	}
	else{
	  sprintf(dumpfile,"%s/dump.%d.%d.%d",newdirname, RtsLayer::myNode(),
		  RtsLayer::myContext(), tid);
	  rename(filename, dumpfile);
	}
    }
  }
#endif //PROFILING_ON
  return 1;
}

void Profiler::PurgeData(int tid){
  
  vector<FunctionInfo*>::iterator it;
  vector<TauUserEvent*>::iterator eit;
  Profiler *curr;
  
  DEBUGPROFMSG("Profiler::PurgeData( tid = "<<tid <<" ) "<<endl;);
  RtsLayer::LockDB();
  
  // Reset The Function Database (save callstack entries)
  for(it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    // May be able to recycle fns which never get called again??
    (*it)->SetCalls(tid,0);
    (*it)->SetSubrs(tid,0);
    (*it)->SetExclTimeZero(tid);
    (*it)->SetInclTimeZero(tid);
  }
  // Now Re-register callstack entries
  curr = CurrentProfiler[tid];
  curr->ThisFunction->IncrNumCalls(tid);
  curr = curr->ParentProfiler;
  while(curr != 0) {
    curr->ThisFunction->IncrNumCalls(tid);
    curr->ThisFunction->IncrNumSubrs(tid);
    curr = curr->ParentProfiler;
  }
  
  // Reset the Event Database
  for (eit = TheEventDB().begin(); eit != TheEventDB().end(); eit++) {
    (*eit)->LastValueRecorded[tid] = 0;
    (*eit)->NumEvents[tid] = 0L;
    (*eit)->MinValue[tid] = 9999999;
    (*eit)->MaxValue[tid] = -9999999;
    (*eit)->SumSqrValue[tid] = 0;
    (*eit)->SumValue[tid] = 0;
  }
  
  RtsLayer::UnLockDB();
}
#endif//TAU_MULTIPLE_COUNTERS

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
void Profiler::CallStackTrace(int tid)
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
  now = RtsLayer::getUSecD(tid);  

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
	  RtsLayer::myContext(), tid);
  
  // traverse stack and set all FunctionInfo's *_cs fields to zero
  curr = CurrentProfiler[tid];
  while (curr != 0)
  {
    curr->ThisFunction->ExclTime_cs = curr->ThisFunction->GetExclTime(tid);
    curr = curr->ParentProfiler;
  }  

  prevTotalTime = 0;
  // calculate time info
  curr = CurrentProfiler[tid];
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
  curr = CurrentProfiler[tid];
  while (curr != 0 )
  {
    fprintf(fp, "\"%s %s\" %ld %ld %.16G %.16G %.16G %.16G\n",
            curr->ThisFunction->GetName(),  curr->ThisFunction->GetType(),
            curr->ThisFunction->GetCalls(tid),curr->ThisFunction->GetSubrs(tid),
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
 * $RCSfile: Profiler.cpp,v $   $Author: bertie $
 * $Revision: 1.71 $   $Date: 2002/03/29 01:06:57 $
 * POOMA_VERSION_ID: $Id: Profiler.cpp,v 1.71 2002/03/29 01:06:57 bertie Exp $ 
 ***************************************************************************/

	





