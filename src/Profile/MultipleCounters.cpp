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

#ifdef CPU_TIME
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif // CPU_TIME

#ifdef CRAY_TIMERS
#include <intrinsics.h>
#include <sys/param.h>
#endif // CRAY_TIMERS

#ifdef TAU_MPI
extern "C" TauUserEvent sendevent;
#endif /* TAU_MPI */

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


#ifdef SGI_TIMERS
int MultipleCounterLayer::sgiTimersMCL_CP[1];
int MultipleCounterLayer::sgiTimersMCL_FP;
#endif // SGI_TIMERS

#ifdef CPU_TIME
int MultipleCounterLayer::cpuTimeMCL_CP[1];
int MultipleCounterLayer::cpuTimeMCL_FP;
#endif // CPU_TIME

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
int MultipleCounterLayer::numberOfPapiHWCounters;
int MultipleCounterLayer::PAPI_CounterCodeList[MAX_TAU_COUNTERS];
ThreadValue * MultipleCounterLayer::ThreadList[TAU_MAX_THREADS];
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

firstListType MultipleCounterLayer::initArray[] = {gettimeofdayMCLInit,
						   linuxTimerMCLInit,
						   sgiTimersMCLInit,
						   cpuTimeMCLInit,
						   crayTimersMCLInit,
						   tauMUSEMCLInit,
						   tauMPIMessageSizeMCLInit,
						   papiMCLInit,
						   papiWallClockMCLInit,
						   papiVirtualMCLInit,
						   pclMCLInit};

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
      MultipleCounterLayer::PAPI_CounterCodeList[a] = -1;
      MultipleCounterLayer::ThreadList[TAU_MAX_THREADS] = NULL;
#endif//TAU_PAPI
#ifdef TAU_PCL
      MultipleCounterLayer::pclMCL_CP[a] = -1;
      MultipleCounterLayer::PCL_CounterCodeList[a] = -1;
      MultipleCounterLayer::threadInit[TAU_MAX_THREADS] = false;
      MultipleCounterLayer::CounterList[MAX_TAU_COUNTERS] = 0;
      MultipleCounterLayer::FpCounterList[MAX_TAU_COUNTERS] = 0;      
#endif//TAU_PCL
    }

    MultipleCounterLayer::gettimeofdayMCL_CP[0] = -1;
    MultipleCounterLayer::gettimeofdayMCL_FP = -1;
#ifdef TAU_LINUX_TIMERS
    MultipleCounterLayer::linuxTimerMCL_CP[0] = -1;
    MultipleCounterLayer::linuxTimerMCL_FP = -1;
#endif //TAU_LINUX_TIMERS

#ifdef SGI_TIMERS
    MultipleCounterLayer::sgiTimersMCL_CP[0] = -1;
    MultipleCounterLayer::sgiTimersMCL_FP = -1;
#endif // SGI_TIMERS

#ifdef CPU_TIME
    MultipleCounterLayer::cpuTimeMCL_CP[0] = -1;
    MultipleCounterLayer::cpuTimeMCL_FP = -1;
#endif // CPU_TIME

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
    MultipleCounterLayer::numberOfPapiHWCounters = 0;
#endif//TAU_PAPI
#ifdef TAU_PCL
    MultipleCounterLayer::numberOfPCLHWCounters = 0;
    MultipleCounterLayer::pclMCL_FP = -1;
#endif//TAU_PCL

  //Get the counter names from the environment.
    for(int c=0; c<MAX_TAU_COUNTERS; c++)
    {
      MultipleCounterLayer::names[c] = getenv(environment[c]);
    }

    //Initialize the function array with the correct active functions.
    for(int e=0; e<SIZE_OF_INIT_ARRAY; e++)
    {
      if(MultipleCounterLayer::initArray[e](functionPosition)){
	  //If this check is true, then this function is active,
	  //and has taken a position in the function array.
	  //Update the function array position.
	  functionPosition++;
	  //Update the number of active functions.
	  numberOfActiveFunctions++;

	  //cout << "Adding function to position: " 
	  //     << e << " of the init array." << endl;
      }
      else{
	//cout << "Not function to position: " 
	//     << e << " of the init array." << endl;
      }
    }

    //Check to see that we have at least one counter defined.
    //Give a warning of not.  It should not break the system,
    //but it is nice to give a warning.
    if(numberOfActiveFunctions == 0)
      cout << "Warning: No multi counter fncts active ... are the env variables COUNTER<1-N> set?" << endl;
  }
  RtsLayer::UnLockDB(); // mutual exclusion primitive
   
  return returnValue;
}

bool * MultipleCounterLayer::getCounterUsedList()
{
  bool *tmpPtr = (bool *) malloc(sizeof(bool *) * MAX_TAU_COUNTERS);

  RtsLayer::LockDB();
  for(int i=0;i< MAX_TAU_COUNTERS;i++){
    tmpPtr[i] = MultipleCounterLayer::counterUsed[i];
  }
  RtsLayer::UnLockDB();

  return tmpPtr;

}

bool MultipleCounterLayer::getCounterUsed(int inPosition)
{
  bool tmpBool = false;

  //RtsLayer::LockDB();
  if(inPosition < MAX_TAU_COUNTERS)
    tmpBool = MultipleCounterLayer::counterUsed[inPosition];

  //RtsLayer::UnLockDB();

  return tmpBool;

}

void MultipleCounterLayer::setCounterUsed(bool inValue, int inPosition)
{
  RtsLayer::LockDB();
  if(inPosition < MAX_TAU_COUNTERS)
    MultipleCounterLayer::counterUsed[inPosition] = inValue;
  RtsLayer::UnLockDB();
}

void MultipleCounterLayer::getCounters(int tid, double values[])
{
  static bool initFlag = initializeMultiCounterLayer();

  //Just cycle through the list of function in the active function array.
  for(int i=0; i<numberOfActiveFunctions; i++){
    if(functionArray[i] != NULL) //Need this check just in case a function is deactivated.
      MultipleCounterLayer::functionArray[i](tid, values);
  }
}

char * MultipleCounterLayer::getCounterNameAt(int position)
{
  if(position < MAX_TAU_COUNTERS)
    return MultipleCounterLayer::names[position];
  else
    return NULL;
}

void MultipleCounterLayer::theCounterList(const char ***inPtr, int *numOfCounters)
{
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
						  bool **tmpPtr)
{
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
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "GET_TIME_OF_DAY") == 0){
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
  bool returnValue = false;
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
	if (strstr(MultipleCounterLayer::names[i],"PAPI") != NULL) {
	  //Reset the name if this is a native event.
	  if (strstr(MultipleCounterLayer::names[i],"NATIVE") != NULL) {
	    //Shift the string down.
	    int counter = 0;
	    while(names[i][12+counter]!='\0'){
	      MultipleCounterLayer::names[i][counter]=MultipleCounterLayer::names[i][12+counter];
	      counter++;
	    }
	    MultipleCounterLayer::names[i][counter]='\0';
	    cout << "Adjusted counter name is: " << names[i] << endl;
	  }
	  PapiLayer::multiCounterPapiInit();
	  int tmpCode = PapiLayer::map_eventnames(MultipleCounterLayer::names[i]);
	  if(tmpCode != -1){
	    if((PAPI_query_event(tmpCode) == PAPI_OK)){//Check if this is possible on this machine!
	      papiMCL_CP[numberOfPapiHWCounters] = i;
	      MultipleCounterLayer::PAPI_CounterCodeList[numberOfPapiHWCounters] = tmpCode;//Set the counter code.
	      numberOfPapiHWCounters++;//Update the number of Papi counters.
	      MultipleCounterLayer::counterUsed[i] = true;
	      MultipleCounterLayer::numberOfCounters[i] = 1;
	      returnValue = true;
	    }
	    else{
	      cout << MultipleCounterLayer::names[i] << " is not available!" << endl;
	    }
	  }
	}
      }
    }
  if(returnValue){
    MultipleCounterLayer::functionArray[functionPosition] = papiMCL;
    papiMCL_FP = functionPosition;
  }
  return returnValue;
#else //TAU_PAPI
  return false;
#endif//TAU_PAPI
}

bool MultipleCounterLayer::papiWallClockMCLInit(int functionPosition){
#ifdef TAU_PAPI
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "P_WALL_CLOCK_TIME") == 0){
	  PapiLayer::multiCounterPapiInit();
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
          PapiLayer::multiCounterPapiInit();
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
  values[gettimeofdayMCL_CP[0]] = ( (double) tp.tv_sec * 1e6 + tp.tv_usec );

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
  long long tick = _rtc();
  values[crayTimersMCL_CP[0]] = (double) tick/HZ; 
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

void MultipleCounterLayer::tauMUSEMCL(int tid, double values[]){
#ifdef TAU_MUSE 
  values[tauMUSEMCL_CP[0]] = TauMuseQuery();
#endif//TAU_MUSE
}

void MultipleCounterLayer::tauMPIMessageSizeMCL(int tid, double values[]){
#ifdef TAU_MPI
  values[tauMPIMessageSizeMCL_CP[0]] = sendevent.GetSumValue(tid); //Currently TAU_EVENT_DATATYPE is a double.
#endif//TAU_MPI
}

void MultipleCounterLayer::papiMCL(int tid, double values[]){
#ifdef TAU_PAPI
  //******************************************
  //Start peformance counting.
  //This section is run once for each thread.
  //******************************************

  //First check to see if the thread is already
  //present.  If not, register the thread and
  //then start the counters.
  if(ThreadList[tid] == NULL){
    //Register thread and start the counters.
    //Since this is also the first call to
    //getCounters for this thread, just return
    //zero.
    if(tid >= TAU_MAX_THREADS){
      cout << "Exceeded max thread count of TAU_MAX_THREADS" << endl;
    }
    else{
      ThreadList[tid] = new ThreadValue;
      ThreadList[tid]->ThreadID = tid;
      ThreadList[tid]->EventSet = PAPI_NULL;
      ThreadList[tid]->CounterValues = new long long[numberOfPapiHWCounters];
      
      PAPI_create_eventset(&(ThreadList[tid]->EventSet));
      
#ifndef PAPI_VERSION
/* PAPI 2 support goes here */
      int resultCode = PAPI_add_events(&(ThreadList[tid]->EventSet),
				       PAPI_CounterCodeList,
				       numberOfPapiHWCounters);
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)
/* PAPI 3 support goes here */
      int resultCode = PAPI_add_events(ThreadList[tid]->EventSet,
				       PAPI_CounterCodeList,
				       numberOfPapiHWCounters);
#else
/* PAPI future support goes here */
#error "Compiling against a not yet released PAPI version"
#endif 

      if(resultCode != PAPI_OK){
	cout << "Error adding Papi events!" << endl;
	if(resultCode == PAPI_ECNFLCT){
	  cout <<"The events you have chosed conflict."<<endl;
	  cout <<"This could be a limit on either the number of events" << endl;
	  cout <<"allowed by this hardware, or the combination of events chosen." << endl;
	  cout << endl;
	}
	cout <<"The papi layer calls are being disabled!" << endl;
	cout <<"Deleting papiMCL in position: " << papiMCL_FP << endl;
	MultipleCounterLayer::functionArray[papiMCL_FP] = NULL;
	cout <<"Setting papi flags in counterUsed array to false ... " << endl;
	for(int h=0;h<numberOfPapiHWCounters;h++){
	  MultipleCounterLayer::setCounterUsed(false, papiMCL_CP[h]);
	  //MultipleCounterLayer::counterUsed[papiMCL_CP[h]] = false;
	  cout <<"counterUsed[" << papiMCL_CP[h] << "] is now false ..." << endl;
	  cout <<"Finished disabling papi layer calls!" << endl;
	}
      }

      if((PAPI_start(ThreadList[tid]->EventSet)) != PAPI_OK){
	  cout << "Error starting Papi counters!" << endl;
      }

      //Initialize the array the Papi portion of the passed in values
      //array to zero.
      for(int i=0;i<numberOfPapiHWCounters;i++){
	values[papiMCL_CP[i]] = 0;
      }
    }    
  }
  else{
    //If here, it means that the thread has already been registered
    //and we need to just read and update the counters.
    
    //*****************************************
    //Reading the performance counters and
    //outputting the counter values.
    //*****************************************
    if((PAPI_read(ThreadList[tid]->EventSet, ThreadList[tid]->CounterValues)) != PAPI_OK){
	cout << "Error reading the Papi counters" << endl;
      }
    
    for(int i=0;i<numberOfPapiHWCounters;i++){
	values[papiMCL_CP[i]] = ThreadList[tid]->CounterValues[i];
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

#ifdef TAU_LINUX_TIMERS

///////////////////////////////////////////////////////////////////////////
inline double TauGetMHzRatingsMCL(void)
{
  FILE *f;
  double rating;
  char *cmd = "cat /proc/cpuinfo | egrep -i '^cpu MHz' | head -1 | sed 's/^.*: //'";
  char buf[BUFSIZ];
  if ((f = popen(cmd,"r"))!=NULL)
  while (fgets(buf, BUFSIZ, f) != NULL)
  {
    rating = atof(buf);
  }
  pclose(f);
#ifdef DEBUG_PROF
  printf("Rating = %g Mhz\n", rating);
#endif /* DEBUG_PROF */
  return rating;
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
//
//End - Definintions for MultipleCounters.
//
/////////////////////////////////////////////////
