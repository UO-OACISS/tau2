/////////////////////////////////////////////////
//Definintions for MultipleCounters.
//
//Author:   Robert Bell
//Created:  March 2002
//
/////////////////////////////////////////////////

#include "Profile/Profiler.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef TAUKTAU_SHCTR
#include "Profile/KtauCounters.h"
#endif //TAUKTAU_SHCTR


#ifdef CPU_TIME
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif // CPU_TIME

#ifdef JAVA_CPU_TIME
#include "Profile/JavaThreadLayer.h"
#endif // JAVA_CPU_TIME

#ifdef CRAY_TIMERS
#ifndef TAU_CATAMOUNT
/* These header files are for Cray X1 */
#include <intrinsics.h>
#include <sys/param.h>
#endif /* TAU_CATAMOUNT */
#endif // CRAY_TIMERS

#ifdef BGL_TIMERS
#include <bglpersonality.h>
#include <rts.h>
#endif

#ifdef BGP_TIMERS
/* header files for BlueGene/P */
#include <bgp_personality.h>
#include <bgp_personality_inlines.h>
#include <kernel_interface.h>
#endif // BGP_TIMERS

#ifdef TRACING_ON
#ifdef TAU_EPILOG
#include "elg_trc.h"
#else /* TAU_EPILOG */
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif /* TAU_EPILOG */
#endif // TRACING_ON

#if (defined(__QK_USER__) || defined(__LIBCATAMOUNT__ ))
#ifndef TAU_CATAMOUNT
#define TAU_CATAMOUNT
#endif /* TAU_CATAMOUNT */
#include <catamount/dclock.h>
#endif /* __QK_USER__ || __LIBCATAMOUNT__ */



#ifdef TAU_MPI
extern TauUserEvent& TheSendEvent(void);
extern TauUserEvent& TheRecvEvent(void);
extern TauUserEvent& TheBcastEvent(void);
extern TauUserEvent& TheReduceEvent(void);
extern TauUserEvent& TheReduceScatterEvent(void);
extern TauUserEvent& TheScanEvent(void);
extern TauUserEvent& TheAllReduceEvent(void);
extern TauUserEvent& TheAlltoallEvent(void);
extern TauUserEvent& TheScatterEvent(void);
extern TauUserEvent& TheGatherEvent(void);
extern TauUserEvent& TheAllgatherEvent(void);

#endif /* TAU_MPI */


// Global Variable holding the number of counters
int Tau_Global_numCounters = -1;




//Initialize static members.
char MultipleCounterLayer::environment[25][10] = {
  {"COUNTER1"},{"COUNTER2"},{"COUNTER3"},{"COUNTER4"},{"COUNTER5"},
  {"COUNTER6"},{"COUNTER7"},{"COUNTER8"},{"COUNTER9"},{"COUNTER10"},
  {"COUNTER11"},{"COUNTER12"},{"COUNTER13"},{"COUNTER14"},{"COUNTER15"},
  {"COUNTER16"},{"COUNTER17"},{"COUNTER18"},{"COUNTER19"},{"COUNTER20"},
  {"COUNTER21"},{"COUNTER22"},{"COUNTER23"},{"COUNTER24"},{"COUNTER25"}};

int MultipleCounterLayer::gettimeofdayMCL_CP[1];
int MultipleCounterLayer::gettimeofdayMCL_FP;
#ifdef TAU_LINUX_TIMERS
int MultipleCounterLayer::linuxTimerMCL_CP[1];
int MultipleCounterLayer::linuxTimerMCL_FP;
#endif //TAU_LINUX_TIMERS

#ifdef CRAY_TIMERS
int MultipleCounterLayer::crayTimersMCL_CP[1];
int MultipleCounterLayer::crayTimersMCL_FP;
#endif // CRAY_TIMERS


#if (defined(BGL_TIMERS) || defined (BGP_TIMERS))
int MultipleCounterLayer::bglTimersMCL_CP[1];
int MultipleCounterLayer::bglTimersMCL_FP;
#endif // BGL_TIMERS || BGP_TIMERS

#ifdef SGI_TIMERS
int MultipleCounterLayer::sgiTimersMCL_CP[1];
int MultipleCounterLayer::sgiTimersMCL_FP;
#endif // SGI_TIMERS

#ifdef CPU_TIME
int MultipleCounterLayer::cpuTimeMCL_CP[1];
int MultipleCounterLayer::cpuTimeMCL_FP;
#endif // CPU_TIME

#ifdef JAVA_CPU_TIME
int MultipleCounterLayer::javaCpuTimeMCL_CP[1];
int MultipleCounterLayer::javaCpuTimeMCL_FP;
#endif // JAVA_CPU_TIME


#ifdef TAU_MUSE
int MultipleCounterLayer::tauMUSEMCL_CP[1];
int MultipleCounterLayer::tauMUSEMCL_FP;
#endif /* TAU_MUSE */

#ifdef TAU_MPI
int MultipleCounterLayer::tauMPIMessageSizeMCL_CP[1];
int MultipleCounterLayer::tauMPIMessageSizeMCL_FP;
#endif /* TAU_MPI */

#ifdef TAU_PAPI
int MultipleCounterLayer::papiMCL_CP[MAX_TAU_COUNTERS];
int MultipleCounterLayer::papiWallClockMCL_CP[1];
int MultipleCounterLayer::papiVirtualMCL_CP[1];
int MultipleCounterLayer::papiMCL_FP;
int MultipleCounterLayer::papiWallClockMCL_FP;
int MultipleCounterLayer::papiVirtualMCL_FP;
#endif//TAU_PAPI
#ifdef TAU_PCL
int MultipleCounterLayer::pclMCL_CP[MAX_TAU_COUNTERS];
int MultipleCounterLayer::pclMCL_FP;
int MultipleCounterLayer::numberOfPCLHWCounters;
int MultipleCounterLayer::PCL_CounterCodeList[MAX_TAU_COUNTERS];
unsigned int MultipleCounterLayer::PCL_Mode = PCL_MODE_USER;
PCL_DESCR_TYPE MultipleCounterLayer::descr;
bool MultipleCounterLayer::threadInit[TAU_MAX_THREADS];
PCL_CNT_TYPE MultipleCounterLayer::CounterList[MAX_TAU_COUNTERS];
PCL_FP_CNT_TYPE MultipleCounterLayer::FpCounterList[MAX_TAU_COUNTERS];
#endif//TAU_PCL

#ifdef TAUKTAU_SHCTR
int MultipleCounterLayer::ktauMCL_CP[MAX_TAU_COUNTERS];
int MultipleCounterLayer::ktauMCL_FP;
#endif//TAUKTAU_SHCTR

#ifdef TRACING_ON
TauUserEvent **MultipleCounterLayer::counterEvents; 
#endif /* TRACING_ON */

firstListType MultipleCounterLayer::initArray[] = {gettimeofdayMCLInit,
						   linuxTimerMCLInit,
						   bglTimersMCLInit,
						   sgiTimersMCLInit,
						   cpuTimeMCLInit,
						   javaCpuTimeMCLInit,
						   crayTimersMCLInit,
						   tauMUSEMCLInit,
						   tauMPIMessageSizeMCLInit,
						   papiMCLInit,
						   papiWallClockMCLInit,
						   papiVirtualMCLInit,
						   pclMCLInit,
						   ktauMCLInit};

int MultipleCounterLayer::numberOfActiveFunctions = 0;
secondListType MultipleCounterLayer::functionArray[] = { };
char * MultipleCounterLayer::names[] = { };
int MultipleCounterLayer::numberOfCounters[] = { };
bool MultipleCounterLayer::counterUsed[] = { };

bool MultipleCounterLayer::initializeMultiCounterLayer(void)
{
  static bool flag = true;
  bool returnValue = true;
  int functionPosition = 0;

  RtsLayer::LockDB();
  if (flag)
  { 
    flag = false;

    //Initializing data.
    for(int a=0; a<MAX_TAU_COUNTERS; a++){
      functionArray[a] = NULL;
      MultipleCounterLayer::names[a] = NULL;
      MultipleCounterLayer::counterUsed[a] = false; //Don't use setter function as we are already in RtsLayer::LockDB();
      MultipleCounterLayer::numberOfCounters[a] = 0;
#ifdef TAU_PAPI 
      MultipleCounterLayer::papiMCL_CP[a] = -1;
#endif//TAU_PAPI
#ifdef TAU_PCL
      MultipleCounterLayer::pclMCL_CP[a] = -1;
      MultipleCounterLayer::PCL_CounterCodeList[a] = -1;
      MultipleCounterLayer::CounterList[a] = 0;
      MultipleCounterLayer::FpCounterList[a] = 0;      
#endif//TAU_PCL
#ifdef TAUKTAU_SHCTR
      MultipleCounterLayer::ktauMCL_CP[a] = -1;
#endif//TAUKTAU_SHCTR
    }

#ifdef TAU_PCL
    for(int a=0; a<TAU_MAX_THREADS; a++){
      MultipleCounterLayer::threadInit[a] = false;
    }
#endif

    MultipleCounterLayer::gettimeofdayMCL_CP[0] = -1;
    MultipleCounterLayer::gettimeofdayMCL_FP = -1;
#ifdef TAU_LINUX_TIMERS
    MultipleCounterLayer::linuxTimerMCL_CP[0] = -1;
    MultipleCounterLayer::linuxTimerMCL_FP = -1;
#endif //TAU_LINUX_TIMERS

#if (defined(BGL_TIMERS) || defined(BGP_TIMERS))
    MultipleCounterLayer::bglTimersMCL_CP[0] = -1;
    MultipleCounterLayer::bglTimersMCL_FP = -1;
#endif // BGL_TIMERS || BGP_TIMERS

#ifdef SGI_TIMERS
    MultipleCounterLayer::sgiTimersMCL_CP[0] = -1;
    MultipleCounterLayer::sgiTimersMCL_FP = -1;
#endif // SGI_TIMERS

#ifdef CPU_TIME
    MultipleCounterLayer::cpuTimeMCL_CP[0] = -1;
    MultipleCounterLayer::cpuTimeMCL_FP = -1;
#endif // CPU_TIME

#ifdef JAVA_CPU_TIME
    MultipleCounterLayer::javaCpuTimeMCL_CP[0] = -1;
    MultipleCounterLayer::javaCpuTimeMCL_FP = -1;
#endif // JAVA_CPU_TIME

#ifdef TAU_MUSE
    MultipleCounterLayer::tauMUSEMCL_CP[0] = -1;
    MultipleCounterLayer::tauMUSEMCL_FP = -1;
#endif /* TAU_MUSE */

#ifdef TAU_MPI
    MultipleCounterLayer::tauMPIMessageSizeMCL_CP[0] = -1;
    MultipleCounterLayer::tauMPIMessageSizeMCL_FP = -1;
#endif /* TAU_MPI */

#ifdef CRAY_TIMERS
    MultipleCounterLayer::crayTimersMCL_CP[0] = -1;
    MultipleCounterLayer::crayTimersMCL_FP = -1;
#endif // CRAY_TIMERS


#ifdef TAU_PAPI
    MultipleCounterLayer::papiWallClockMCL_CP[0] = -1;
    MultipleCounterLayer::papiVirtualMCL_CP[0] = -1;
    MultipleCounterLayer::papiMCL_FP = -1;
    MultipleCounterLayer::papiWallClockMCL_FP = -1;
    MultipleCounterLayer::papiVirtualMCL_FP = -1;
#endif//TAU_PAPI
#ifdef TAU_PCL
    MultipleCounterLayer::numberOfPCLHWCounters = 0;
    MultipleCounterLayer::pclMCL_FP = -1;
#endif//TAU_PCL
#ifdef TAUKTAU_SHCTR
    MultipleCounterLayer::ktauMCL_FP = -1;
#endif//TAUKTAU_SHCTR


    //Get the counter names from the environment.
    bool counterFound = false;
    for (int c=0; c<MAX_TAU_COUNTERS; c++) {
      MultipleCounterLayer::names[c] = getenv(environment[c]);
      if (MultipleCounterLayer::names[c]) {
	counterFound = true;
	MultipleCounterLayer::names[c] = strdup(MultipleCounterLayer::names[c]);
      }
    }
    
    if (!counterFound) {
      char *counter = "GET_TIME_OF_DAY";
#if defined(TAU_USE_PAPI_TIMER) && defined(TAU_PAPI)
      counter = "P_WALL_CLOCK_TIME";
#else
      fprintf (stderr, "TAU: No counters environment variables defined (COUNTER1, COUNTER2, ...), using COUNTER1=GET_TIME_OF_DAY\n");
#endif
      MultipleCounterLayer::names[0] = strdup(counter);
    }

    //Initialize the function array with the correct active functions.
    for (int e=0; e<SIZE_OF_INIT_ARRAY; e++) {
      if (MultipleCounterLayer::initArray[e](functionPosition)) {
	//If this check is true, then this function is active,
	//and has taken a position in the function array.
	//Update the function array position.
	functionPosition++;
	  //Update the number of active functions.
	numberOfActiveFunctions++;
	//cout <<"numberOfActiveFunctions = "<<numberOfActiveFunctions<<endl;
	
	// cout << "Adding function to position: " 
	//      << e << " of the init array." << endl;
      } else {
	//cout << "Not function to position: " 
	//     << e << " of the init array." << endl;
      }
    }

    //Check to see that we have at least one counter defined.
    //Give a warning of not.  It should not break the system,
    //but it is nice to give a warning.
    if (numberOfActiveFunctions == 0) {
      fprintf (stderr, "Warning: No multi counter fncts active ... are the env variables COUNTER<1-N> set?\n");
    }
#ifdef TRACING_ON
   int countersUsed = getNumberOfCountersUsed();
   counterEvents = new TauUserEvent * [countersUsed] ; 
   /* We obtain the timestamp from COUNTER1, so we only need to trigger 
      COUNTER2-N or i=1 through no. of active functions not through 0 */
   RtsLayer::UnLockDB(); // mutual exclusion primitive AddEventToDB locks it
   for (int i = 1; i < countersUsed; i++) {
     counterEvents[i] = new TauUserEvent(names[i], true);
     /* the second arg is MonotonicallyIncreasing which is true (HW counters)*/ 
   }
   RtsLayer::LockDB(); // We do this to prevent a deadlock. Lock it again!
#endif /* TRACING_ON */
  }
  RtsLayer::UnLockDB(); // mutual exclusion primitive
  


  /* Temporary hack until this code is cleaned up */

  int count = 0;
  int i;
  for (i=0;i<MAX_TAU_COUNTERS;i++){
    char *tmpChar = getCounterNameAt(i);
    if((tmpChar != NULL) && (MultipleCounterLayer::getCounterUsed(i))){
      count = i+1;
    }
  }


//   Tau_Global_numCounters = getNumberOfCountersUsed();
  Tau_Global_numCounters = count;
//   printf ("set Tau_Global_numCounters to %d\n", Tau_Global_numCounters);

  return returnValue;
}

bool *MultipleCounterLayer::getCounterUsedList() {
  bool *tmpPtr = (bool *) malloc(sizeof(bool *) * MAX_TAU_COUNTERS);

  RtsLayer::LockDB();
  for(int i=0;i< MAX_TAU_COUNTERS;i++) {
    tmpPtr[i] = MultipleCounterLayer::counterUsed[i];
  }
  RtsLayer::UnLockDB();

  return tmpPtr;
}

bool MultipleCounterLayer::getCounterUsed(int inPosition) {
  bool tmpBool = false;

  if (inPosition < MAX_TAU_COUNTERS) {
    tmpBool = MultipleCounterLayer::counterUsed[inPosition];
  }

  return tmpBool;

}

void MultipleCounterLayer::setCounterUsed(bool inValue, int inPosition) {
  RtsLayer::LockDB();
  if(inPosition < MAX_TAU_COUNTERS)
    MultipleCounterLayer::counterUsed[inPosition] = inValue;
  RtsLayer::UnLockDB();
}

void MultipleCounterLayer::getCounters(int tid, double values[]) {
  static bool initFlag = initializeMultiCounterLayer();

  //Just cycle through the list of function in the active function array.
  for(int i=0; i<numberOfActiveFunctions; i++){
    if(functionArray[i] != NULL) //Need this check just in case a function is deactivated.
      MultipleCounterLayer::functionArray[i](tid, values);
  }
}

// a low overhead way to get a single counter without getting all the others
double MultipleCounterLayer::getSingleCounter(int tid, int counter) {
  static bool initFlag = initializeMultiCounterLayer();
  static double values[MAX_TAU_COUNTERS];;

  if (functionArray[counter] == NULL) {
    return 0.0;
  }
  
  MultipleCounterLayer::functionArray[counter](tid, values);
  return values[counter];
}

char * MultipleCounterLayer::getCounterNameAt(int position)
{
  if(position < MAX_TAU_COUNTERS)
    return MultipleCounterLayer::names[position];
  else
    return NULL;
}

void MultipleCounterLayer::theCounterList(const char ***inPtr, int *numOfCounters) {
  static const char **counterList = ( char const **) malloc( sizeof(char *) * MAX_TAU_COUNTERS);
  int numberOfCounters = 0;

  //With a look toward future developements, this list might
  //change from call to call.  Thus, build it each time.
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    char *tmpChar = getCounterNameAt(i);
    if((tmpChar != NULL) && (MultipleCounterLayer::getCounterUsed(i))){
      counterList[i] = tmpChar;
      numberOfCounters++;
    }
  }

  //We do not want to pass back references to internal pointers.
  *inPtr = ( char const **) malloc( sizeof(char *) * numberOfCounters);
  for(int j=0;j<numberOfCounters;j++){
    (*inPtr)[j] = counterList[j]; //Need the () in (*inPtr)[j] or the dereferrencing is
    //screwed up!

    *numOfCounters = numberOfCounters;
  }
}

void MultipleCounterLayer::theCounterListInternal(const char ***inPtr,
						  int *numOfCounters,
						  bool **tmpPtr) {
  //For situations where a consistency is needed between the elements
  //in the counterUsed array and those in the counter names array.
  //As such, we grab the counter used list array atomically,
  //and then only grab the counter names that were active when the
  //counter used array was obtained.  This consistency is really only
  //required internally.  The external interface should use theCounterList
  //function above.
  bool * tmpCounterUsedList;

  tmpCounterUsedList = MultipleCounterLayer::getCounterUsedList();

  static const char **counterList = ( char const **) malloc( sizeof(char *) * MAX_TAU_COUNTERS);
  int numberOfCounters = 0;
  for(int i=0;i<MAX_TAU_COUNTERS;i++){
    char *tmpChar = getCounterNameAt(i);
    if((tmpChar != NULL) && (tmpCounterUsedList[i])){
      counterList[i] = tmpChar;
      numberOfCounters++;
    }
  }

  //We do not want to pass back internal pointers.
  *inPtr = ( char const **) malloc( sizeof(char *) * numberOfCounters);
  for(int j=0;j<numberOfCounters;j++){
    (*inPtr)[j] = counterList[j]; //Need the () in (*inPtr)[j] or the dereferrencing is
    //screwed up!

    *numOfCounters = numberOfCounters;
  }

  *numOfCounters = numberOfCounters;
  *tmpPtr = tmpCounterUsedList;
}

bool MultipleCounterLayer::gettimeofdayMCLInit(int functionPosition){
  for (int i=0; i<MAX_TAU_COUNTERS; i++) {
      if (MultipleCounterLayer::names[i] != NULL) {
	if (strcmp(MultipleCounterLayer::names[i], "GET_TIME_OF_DAY") == 0) {
	  gettimeofdayMCL_CP[0] = i;
	  MultipleCounterLayer::counterUsed[i] = true;
	  MultipleCounterLayer::numberOfCounters[i] = 1;
	  MultipleCounterLayer::functionArray[functionPosition] = gettimeofdayMCL;
	  gettimeofdayMCL_FP = functionPosition;
	  return true;
	}
      }
    }
  return false;
}

bool MultipleCounterLayer::bglTimersMCLInit(int functionPosition){
#if (defined(BGL_TIMERS) || defined(BGP_TIMERS))
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if((strcmp(MultipleCounterLayer::names[i], "BGL_TIMERS") == 0) || 
         (strcmp(MultipleCounterLayer::names[i], "BGP_TIMERS") == 0)) {
	bglTimersMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = bglTimersMCL;
	bglTimersMCL_FP = functionPosition;
	return true;
      }
    }
  }
  return false;
#else //BGL_TIMERS || BGP_TIMERS
  return false;
#endif//BGL_TIMERS || BGP_TIMERS
}

bool MultipleCounterLayer::sgiTimersMCLInit(int functionPosition){
#ifdef SGI_TIMERS
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if(strcmp(MultipleCounterLayer::names[i], "SGI_TIMERS") == 0){
	sgiTimersMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = sgiTimersMCL;
	sgiTimersMCL_FP = functionPosition;
	return true;
      }
    }
  }
  return false;
#else //SGI_TIMERS
  return false;
#endif//SGI_TIMERS
}

bool MultipleCounterLayer::crayTimersMCLInit(int functionPosition){
#ifdef CRAY_TIMERS
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if(strcmp(MultipleCounterLayer::names[i], "CRAY_TIMERS") == 0){
	crayTimersMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = crayTimersMCL;
	crayTimersMCL_FP = functionPosition;
	return true;
      }
    }
  }
  return false;
#else //CRAY_TIMERS
  return false;
#endif//CRAY_TIMERS
}

bool MultipleCounterLayer::cpuTimeMCLInit(int functionPosition){
#ifdef CPU_TIME
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if(strcmp(MultipleCounterLayer::names[i], "CPU_TIME") == 0){
	cpuTimeMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = cpuTimeMCL;
	cpuTimeMCL_FP = functionPosition;
	return true;
      }
    }
  }
  return false;
#else //CPU_TIME
  return false;
#endif//CPU_TIME
}

bool MultipleCounterLayer::javaCpuTimeMCLInit(int functionPosition){
#ifdef JAVA_CPU_TIME
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if(strcmp(MultipleCounterLayer::names[i], "JAVA_CPU_TIME") == 0){
	javaCpuTimeMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = javaCpuTimeMCL;
	javaCpuTimeMCL_FP = functionPosition;
	return true;
      }
    }
  }
  return false;
#else //JAVA_CPU_TIME
  return false;
#endif//JAVA_CPU_TIME
}

bool MultipleCounterLayer::tauMUSEMCLInit(int functionPosition){
#ifdef TAU_MUSE
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if(strcmp(MultipleCounterLayer::names[i], "TAU_MUSE") == 0){
	tauMUSEMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = tauMUSEMCL;
	tauMUSEMCL_FP = functionPosition;
	return true;
      }
    }
  }
  return false;
#else //TAU_MUSE
  return false;
#endif//TAU_MUSE
}

bool MultipleCounterLayer::tauMPIMessageSizeMCLInit(int functionPosition){
#ifdef TAU_MPI
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if(strcmp(MultipleCounterLayer::names[i], "TAU_MPI_MESSAGE_SIZE") == 0){
	tauMPIMessageSizeMCL_CP[0] = i;
	MultipleCounterLayer::counterUsed[i] = true;
	MultipleCounterLayer::numberOfCounters[i] = 1;
	MultipleCounterLayer::functionArray[functionPosition] = tauMPIMessageSizeMCL;
	tauMPIMessageSizeMCL_FP = functionPosition;
	return true;
      }
    }
  }
  //If we are here, then this function is not active.
  return false;
#else //TAU_MPI
  return false;
#endif//TAU_MPI
}


bool MultipleCounterLayer::papiMCLInit(int functionPosition){
#ifdef TAU_PAPI

  int rc = PapiLayer::initializePapiLayer(false); // do not lock, it's already locked
  if (rc != 0) {
    return false;
  }

  bool returnValue = false;
  for (int i=0; i<MAX_TAU_COUNTERS; i++) {
    if (MultipleCounterLayer::names[i] != NULL) {
      if (strstr(MultipleCounterLayer::names[i],"PAPI") != NULL) {
	//Reset the name if this is a native event.
	if (strstr(MultipleCounterLayer::names[i],"NATIVE") != NULL) {
	  //Shift the string down.
	  int counter = 0;
	  while (names[i][12+counter]!='\0') {
	    MultipleCounterLayer::names[i][counter]=MultipleCounterLayer::names[i][12+counter];
	    counter++;
	  }
	  MultipleCounterLayer::names[i][counter]='\0';
#ifdef DEBUG_PROF
	  cout << "Adjusted counter name is: " << names[i] << endl;
#endif /* DEBUG_PROF */
	}
	
	int counterID = PapiLayer::addCounter(MultipleCounterLayer::names[i]);

	if (counterID >= 0) {
	  papiMCL_CP[counterID] = i;
	  MultipleCounterLayer::counterUsed[i] = true;
	  MultipleCounterLayer::numberOfCounters[i] = 1;
	  returnValue = true;
	}
      }
    }
  }
  

  if (returnValue) { // at least one PAPI counter was available
    MultipleCounterLayer::functionArray[functionPosition] = papiMCL;
    papiMCL_FP = functionPosition;
    return true;
  }
  return false;
#else //TAU_PAPI
  return false;
#endif//TAU_PAPI
}

bool MultipleCounterLayer::papiWallClockMCLInit(int functionPosition){
#ifdef TAU_PAPI
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "P_WALL_CLOCK_TIME") == 0){
	  PapiLayer::initializePapiLayer(false);
	  papiWallClockMCL_CP[0] = i;
	  MultipleCounterLayer::counterUsed[i] = true;
	  MultipleCounterLayer::numberOfCounters[i] = 1;
	  MultipleCounterLayer::functionArray[functionPosition] = papiWallClockMCL;
	  papiWallClockMCL_FP = functionPosition;
	  return true;
	}
      }
    }
  return false;
#else  // TAU_PAPI
  return false;
#endif // TAU_PAPI
}

bool MultipleCounterLayer::papiVirtualMCLInit(int functionPosition){
#ifdef TAU_PAPI
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
        if(strcmp(MultipleCounterLayer::names[i], "P_VIRTUAL_TIME") == 0){
	  PapiLayer::initializePapiLayer(false);
          papiVirtualMCL_CP[0] = i;
	  MultipleCounterLayer::counterUsed[i] = true;
	  MultipleCounterLayer::numberOfCounters[i] = 1;
          MultipleCounterLayer::functionArray[functionPosition] = papiVirtualMCL;
	  papiVirtualMCL_FP = functionPosition;
          return true;
        }
      }
    }
  return false;
#else  // TAU_PAPI
  return false;
#endif // TAU_PAPI
}

bool MultipleCounterLayer::pclMCLInit(int functionPosition){
#ifdef  TAU_PCL
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
    if(MultipleCounterLayer::names[i] != NULL){
      if (strstr(MultipleCounterLayer::names[i],"PCL") != NULL) {
	PCL_Layer::multiCounterPCLInit(&MultipleCounterLayer::descr);
	int tmpCode = PCL_Layer::map_eventnames(MultipleCounterLayer::names[i]);
	if(tmpCode != -1){
	  pclMCL_CP[numberOfPCLHWCounters] = i;
	  MultipleCounterLayer::PCL_CounterCodeList[numberOfPCLHWCounters] = tmpCode;
	  numberOfPCLHWCounters++;
	}
      }
    }
  }
  if(numberOfPCLHWCounters != 0){
    //Check whether these pcl events are possible.
    if(PCLquery(descr, PCL_CounterCodeList, numberOfPCLHWCounters, PCL_Mode) == PCL_SUCCESS){
      for(int j=0;j<numberOfPCLHWCounters;j++){
	MultipleCounterLayer::counterUsed[pclMCL_CP[j]] = true;
	MultipleCounterLayer::numberOfCounters[pclMCL_CP[j]] = 1;
      }
      MultipleCounterLayer::functionArray[functionPosition] = pclMCL;
      return true;
    }
    else{
      cout << "Requested pcl events, or event combination not possible!" << endl;
      return false;
    } 
  }

#else //TAU_PCL
  return false;
#endif//TAU_PCL
}


bool MultipleCounterLayer::ktauMCLInit(int functionPosition){
#ifdef TAUKTAU_SHCTR

  int rc = KtauCounters::initializeKtauCounters(false); // do not lock, it's already locked
  if (rc != 0) {
    return false;
  }

  bool returnValue = false;
  for (int i=0; i<MAX_TAU_COUNTERS; i++) {
    int counterID = -1;
    int cType = 0, nOffset = 0;
    if (MultipleCounterLayer::names[i] != NULL) {
      if (strstr(MultipleCounterLayer::names[i],"KTAU_") != NULL) {
        if (strstr(MultipleCounterLayer::names[i],"KTAU_INCL_") != NULL) {
          //remember type as INCL and set OFFSET as 10
          cType = KTAU_SHCTR_TYPE_INCL;
          nOffset = 5;
        } else if (strstr(MultipleCounterLayer::names[i],"KTAU_NUM_") != NULL) {
          //remember type as NUM and set OFFSET as 9
          cType = KTAU_SHCTR_TYPE_NUM;
          nOffset = 5;
        } else {
          //remember type as EXCL and set OFFSET as 5
          cType = KTAU_SHCTR_TYPE_EXCL;
          nOffset = 5;
        }
	//Reset the name  to get symbol name
	//Shift the string down.
	int counter = 0;
	while (names[i][nOffset+counter]!='\0') {
	  MultipleCounterLayer::names[i][counter]=MultipleCounterLayer::names[i][nOffset+counter];
	  counter++;
	}
        MultipleCounterLayer::names[i][counter]='\0';
#ifdef DEBUG_PROF
        cout << "Adjusted counter name is: " << names[i] << endl;
#endif /* DEBUG_PROF */
        counterID = KtauCounters::addCounter(MultipleCounterLayer::names[i], cType);
      }
	
      if (counterID >= 0) {
        ktauMCL_CP[counterID] = i;
        MultipleCounterLayer::counterUsed[i] = true;
        MultipleCounterLayer::numberOfCounters[i] = 1;
        returnValue = true;
      }
    }
  }
  
  if (returnValue) { // at least one KTAU counter was available
    MultipleCounterLayer::functionArray[functionPosition] = ktauMCL;
    ktauMCL_FP = functionPosition;
    return true;
  }
  return false;
#else //TAUKTAU_SHCTR
  return false;
#endif//TAUKTAU_SHCTR
}


bool MultipleCounterLayer::linuxTimerMCLInit(int functionPosition){
#ifdef  TAU_LINUX_TIMERS
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "LINUX_TIMERS") == 0){
	  linuxTimerMCL_CP[0] = i;
	  MultipleCounterLayer::counterUsed[i] = true;
	  MultipleCounterLayer::numberOfCounters[i] = 1;
	  MultipleCounterLayer::functionArray[functionPosition] = linuxTimerMCL;
	  linuxTimerMCL_FP = functionPosition;
	  return true;
	}
      }
    }
  return false;
#else //TAU_LINUX_TIMERS
return false;
#endif//TAU_LINUX_TIMERS
}

void MultipleCounterLayer::gettimeofdayMCL(int tid, double values[]){
  struct timeval tp;
  gettimeofday (&tp, 0);
  values[gettimeofdayMCL_CP[0]] = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
}

void MultipleCounterLayer::bglTimersMCL(int tid, double values[]){
#if (defined(BGL_TIMERS) || defined(BGP_TIMERS))
#ifdef TAU_BGL
   static double bgl_clockspeed = 0.0;

   if (bgl_clockspeed == 0.0)
   {
     BGLPersonality mybgl;
     rts_get_personality(&mybgl, sizeof(BGLPersonality));
     bgl_clockspeed = 1.0e6/(double)BGLPersonality_clockHz(&mybgl);
   }
   values[bglTimersMCL_CP[0]] = (rts_get_timebase() * bgl_clockspeed);
#endif /* TAU_BGL */
#ifdef TAU_BGP
  static double bgp_clockspeed = 0.0;

  if (bgp_clockspeed == 0.0)
  {
    _BGP_Personality_t mybgp;
    Kernel_GetPersonality(&mybgp, sizeof(_BGP_Personality_t));
    bgp_clockspeed = 1.0/(double)BGP_Personality_clockMHz(&mybgp);
  }
   values[bglTimersMCL_CP[0]] =  (_bgp_GetTimeBase() * bgp_clockspeed);
#endif /* TAU_BGP */
#endif//BGL_TIMERS || BGP_TIMERS
}

void MultipleCounterLayer::sgiTimersMCL(int tid, double values[]){
#ifdef  SGI_TIMERS
  struct timespec tp;
  clock_gettime(CLOCK_SGI_CYCLE,&tp);
  values[sgiTimersMCL_CP[0]] = (tp.tv_sec * 1e6 + (tp.tv_nsec * 1e-3)) ;
#endif//SGI_TIMERS
}

void MultipleCounterLayer::crayTimersMCL(int tid, double values[]){
#ifdef  CRAY_TIMERS
#ifdef TAU_CATAMOUNT /* for Cray XT3 */
  values[crayTimersMCL_CP[0]] = dclock()*1.0e6;
#else /* for Cray X1 */
  long long tick = _rtc();
  values[crayTimersMCL_CP[0]] = (double) tick/HZ;
#endif /* TAU_CATAMOUNT */
#endif // CRAY_TIMERS
}

void MultipleCounterLayer::cpuTimeMCL(int tid, double values[]){
#ifdef  CPU_TIME
  struct rusage current_usage;

  getrusage (RUSAGE_SELF, &current_usage);

  values[cpuTimeMCL_CP[0]] = (current_usage.ru_utime.tv_sec + current_usage.ru_stime.tv_sec)* 1e6 
  + (current_usage.ru_utime.tv_usec + current_usage.ru_stime.tv_usec);
#endif//CPU_TIME
}

void MultipleCounterLayer::javaCpuTimeMCL(int tid, double values[]){
#ifdef  JAVA_CPU_TIME
  struct rusage current_usage;

  getrusage (RUSAGE_SELF, &current_usage);

  values[javaCpuTimeMCL_CP[0]] = JavaThreadLayer::getCurrentThreadCpuTime();
#endif//JAVA_CPU_TIME
}


void MultipleCounterLayer::tauMUSEMCL(int tid, double values[]){
#ifdef TAU_MUSE 
  values[tauMUSEMCL_CP[0]] = TauMuseQuery();
#endif//TAU_MUSE
}

void MultipleCounterLayer::tauMPIMessageSizeMCL(int tid, double values[]){
#ifdef TAU_MPI
  values[tauMPIMessageSizeMCL_CP[0]] = TheSendEvent().GetSumValue(tid) 
	+ TheRecvEvent().GetSumValue(tid) 
	+ TheBcastEvent().GetSumValue(tid)
	+ TheReduceEvent().GetSumValue(tid)
	+ TheReduceScatterEvent().GetSumValue(tid)
	+ TheScanEvent().GetSumValue(tid)
	+ TheAllReduceEvent().GetSumValue(tid)
	+ TheAlltoallEvent().GetSumValue(tid)
	+ TheScatterEvent().GetSumValue(tid)
	+ TheGatherEvent().GetSumValue(tid)
	+ TheAllgatherEvent().GetSumValue(tid);
 //Currently TAU_EVENT_DATATYPE is a double.
#endif//TAU_MPI
}

void MultipleCounterLayer::papiMCL(int tid, double values[]){
#ifdef TAU_PAPI

  int numPapiValues;
  long long *papiValues = PapiLayer::getAllCounters(tid, &numPapiValues);
  
  if (papiValues) {
    for(int i=0; i<numPapiValues; i++) {
      values[papiMCL_CP[i]] = papiValues[i];
    }
  }

#endif//TAU_PAPI
}

void MultipleCounterLayer::papiWallClockMCL(int tid, double values[]){
static long long oldvalue = 0L;
static long long offset = 0;
long long newvalue = 0L;
#ifdef TAU_PAPI
  newvalue = PAPI_get_real_usec();
  if (newvalue < oldvalue) 
  {
    offset += UINT_MAX;
    DEBUGPROFMSG("WARNING: papi counter overflow. Fixed in TAU! new = "
	 <<newvalue <<" old = " <<oldvalue<<" offset = "<<offset <<endl;);
    DEBUGPROFMSG("Returning "<<newvalue + offset<<endl;);
  }
  oldvalue = newvalue;
  values[papiWallClockMCL_CP[0]] = newvalue + offset; 
#endif // TAU_PAPI
}
void MultipleCounterLayer::papiVirtualMCL(int tid, double values[]){
#ifdef TAU_PAPI
  values[papiVirtualMCL_CP[0]] = PAPI_get_virt_usec();
#endif // TAU_PAPI
}

void MultipleCounterLayer::pclMCL(int tid, double values[]){
#ifdef  TAU_PCL
  //******************************************
  //Start peformance counting.
  //This section is run once for each thread.
  //******************************************
  if(threadInit[tid] == 0)
    {
      //Since this is also the first call to
      //getCounters for this thread, just return
      //zero.
      if(tid >= TAU_MAX_THREADS){
	cout << "Exceeded max thread count of TAU_MAX_THREADS" << endl;
      }
      threadInit[tid] = 1;

      //Starting the counter.
      if((PCLstart(descr, PCL_CounterCodeList,
		   numberOfPCLHWCounters, PCL_Mode)) != PCL_SUCCESS){
	cout << "Error starting PCL counters!" << endl;
      }

      //Initialize the array the Pcl portion of the passed in values
      //array to zero.
      for(int i=0;i<numberOfPCLHWCounters;i++){
	values[pclMCL_CP[i]] = 0;
      }
    }
  else{
    //If here, it means that the thread has already been registered
    //and we need to just read and update the counters.
  
    //*****************************************
    //Reading the performance counters and
    //outputting the counter values.
    //*****************************************
    if( PCLread(descr, CounterList, FpCounterList, numberOfPCLHWCounters) != PCL_SUCCESS){
      cout << "Error reading PCL counters!" << endl;
    }

    for(int i=0;i<numberOfPCLHWCounters;i++){
      if(PCL_CounterCodeList[i]<PCL_MFLOPS){
	values[pclMCL_CP[i]] = CounterList[i];
      }
      else{
	values[pclMCL_CP[i]] = FpCounterList[i];
      }
    }
  }
#endif//TAU_PCL
}


void MultipleCounterLayer::ktauMCL(int tid, double values[]){
#ifdef TAUKTAU_SHCTR
  //declaration (this is implemented in RtsLater.cpp)
  extern double KTauGetMHz(void);

  int numKtauValues;
  long long *ktauValues = KtauCounters::getAllCounters(tid, &numKtauValues);
  
  if (ktauValues) {
    for(int i=0; i<numKtauValues; i++) {
      //sometimes due to double-precision issues the below
      //division can result in very small negative exclusive
      //times. Currently there is no fix implemented for this.
      //The thing to do maybe is to add a check in Profiler.cpp
      //to make sure no negative values are set.
      if(KtauCounters::counterType[i] != KTAU_SHCTR_TYPE_NUM) {
        values[ktauMCL_CP[i]] = ktauValues[i]/KTauGetMHz();
      } else {
        values[ktauMCL_CP[i]] = ktauValues[i];
      }
    }
  }

#endif//TAUKTAU_SHCTR
}

#ifdef TAU_LINUX_TIMERS

///////////////////////////////////////////////////////////////////////////
inline double TauGetMHzRatingsMCL(void)
{
  FILE *f;
  bool isApple = false;
  FILE *fd;
  double rating;
  char *cmd1 = "cat /proc/cpuinfo | egrep -i '^cpu MHz' | head -1 | sed 's/^.*: //'";
  char *cmd2 = "sysctl hw.cpufrequency | sed 's/^.*: //'"; /* For Apple */

  char buf[BUFSIZ];

  if ((fd = fopen("/proc/cpuinfo", "r")) == NULL) {
    /* Assume Mac OS X. There is no /proc/cpuinfo on Darwin.  */
    f = popen(cmd2,"r");
    isApple = true;
  }
  else 
  { /* Linux */
    f = popen(cmd1,"r");
  }
  fclose(fd); /* for testing /proc/cpuinfo */

  if (f!=NULL) {
    while (fgets(buf, BUFSIZ, f) != NULL)
    {
      rating = atof(buf);
    }
  }
  pclose(f);
#ifdef DEBUG_PROF
  printf("Rating = %g Mhz\n", rating);
#endif /* DEBUG_PROF */

  if (isApple) {
    return rating/1E6; /* Apple returns Hz not MHz. Convert to MHz */
  }
  else
  {
    return rating; /* in MHz */
  }
}

  
///////////////////////////////////////////////////////////////////////////
inline double TauGetMHzMCL(void)
{
  static double ratings = TauGetMHzRatingsMCL();
  return ratings;
}
///////////////////////////////////////////////////////////////////////////
// Fix Intel compiler does not support this asm statements
extern "C" unsigned long long getLinuxHighResolutionTscCounter(void);
inline unsigned long long getLinuxHighResolutionTscCounterMCL(void)
{
  return getLinuxHighResolutionTscCounter();
}
///////////////////////////////////////////////////////////////////////////

#endif //TAU_LINUX_TIMERS
  
///////////////////////////////////////////////////////////////////////////
void MultipleCounterLayer::linuxTimerMCL(int tid, double values[]){
#ifdef TAU_LINUX_TIMERS
  values[linuxTimerMCL_CP[0]] = 
    (double) getLinuxHighResolutionTscCounterMCL()/TauGetMHzMCL();
#endif //TAU_LINUX_TIMERS
}

/////////////////////////////////////////////////
// Get number of counters
/////////////////////////////////////////////////
int MultipleCounterLayer::getNumberOfCountersUsed(void) {
  int i, numberOfCounters=0;
  for(i=0;i<MAX_TAU_COUNTERS;i++){
    char *tmpChar = getCounterNameAt(i);
    if((tmpChar != NULL) && (MultipleCounterLayer::getCounterUsed(i))){
      numberOfCounters++;
    }
  }
  return numberOfCounters; 
}

#if ( defined(TAU_MULTIPLE_COUNTERS) && defined(TRACING_ON))

/////////////////////////////////////////////////
// Trigger user defined events associated with each counter 
/////////////////////////////////////////////////
void MultipleCounterLayer::triggerCounterEvents(unsigned long long timestamp, double *values, int tid)
{
  int i;
  static int countersUsed = MultipleCounterLayer::getNumberOfCountersUsed();
#ifndef TAU_EPILOG
  for (i = 1; i < countersUsed; i++)
  { /* for each event */
    TraceEvent(counterEvents[i]->GetEventId(), (long long) values[i], tid, timestamp, 1);
    // 1 in the last parameter is for use timestamp 
  }
#endif /* TAU_EPILOG */

}
#endif /* TAU_MULTIPLE_COUNTERS && TRACING_ON */

/////////////////////////////////////////////////
//
//End - Definintions for MultipleCounters.
//
/////////////////////////////////////////////////
