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


//Initialize static members.
char MultipleCounterLayer::environment[25][10] = {
  {"COUNTER1"},{"COUNTER2"},{"COUNTER3"},{"COUNTER4"},{"COUNTER5"},
  {"COUNTER6"},{"COUNTER7"},{"COUNTER8"},{"COUNTER9"},{"COUNTER10"},
  {"COUNTER11"},{"COUNTER12"},{"COUNTER13"},{"COUNTER14"},{"COUNTER15"},
  {"COUNTER16"},{"COUNTER17"},{"COUNTER18"},{"COUNTER19"},{"COUNTER20"},
  {"COUNTER21"},{"COUNTER22"},{"COUNTER23"},{"COUNTER24"},{"COUNTER25"}};

int MultipleCounterLayer::gettimeofdayMCL_CP[1];
#ifdef TAU_PAPI
int MultipleCounterLayer::papiMCL_CP[MAX_TAU_COUNTERS];
int MultipleCounterLayer::papiWallClockMCL_CP[1];
int MultipleCounterLayer::papiVirtualMCL_CP[1];
int MultipleCounterLayer::numberOfPapiHWCounters;
int MultipleCounterLayer::PAPI_CounterCodeList[MAX_TAU_COUNTERS];
#endif//TAU_PAPI
#ifdef TAU_PCL
int MultipleCounterLayer::pclMCL_CP[MAX_TAU_COUNTERS];
int MultipleCounterLayer::numberOfPCLHWCounters;
int MultipleCounterLayer::PCL_CounterCodeList[MAX_TAU_COUNTERS];
#endif//TAU_PCL
int MultipleCounterLayer::linuxTimerMCL_CP[1];

firstListType MultipleCounterLayer::initArray[] = {gettimeofdayMCLInit,
						   papiMCLInit,
						   papiWallClockMCLInit,
						   papiVirtualMCLInit,
						   pclMCLInit,
						   linuxTimerMCLInit};

int MultipleCounterLayer::numberOfActiveFunctions = 0;
secondListType MultipleCounterLayer::functionArray[] = { };
char * MultipleCounterLayer::names[] = { };
bool MultipleCounterLayer::counterUsed[] = { };


bool MultipleCounterLayer::initializeMultiCounterLayer(void)
{
  bool returnValue = true;
  int functionPosition = 0;

  //Initializing data.
  for(int a=0; a<MAX_TAU_COUNTERS; a++){
      functionArray[a] = NULL;
      MultipleCounterLayer::names[a] = NULL;
      MultipleCounterLayer::counterUsed[a] = false;
#ifdef TAU_PAPI 
      MultipleCounterLayer::papiMCL_CP[a] = -1;
      MultipleCounterLayer::PAPI_CounterCodeList[a] = -1;
#endif//TAU_PAPI
#ifdef TAU_PCL
      MultipleCounterLayer::pclMCL_CP[a] = -1;
      MultipleCounterLayer::PCL_CounterCodeList[a] = -1;
#endif//TAU_PCL
    }

  MultipleCounterLayer::gettimeofdayMCL_CP[0] = -1;
#ifdef TAU_PAPI
  MultipleCounterLayer::papiWallClockMCL_CP[0] = -1;
  MultipleCounterLayer::papiVirtualMCL_CP[0] = -1;
  MultipleCounterLayer::numberOfPapiHWCounters = 0;
#endif//TAU_PAPI
#ifdef TAU_PCL
  MultipleCounterLayer::numberOfPCLHWCounters = 0;
#endif//TAU_PCL
  MultipleCounterLayer::linuxTimerMCL_CP[0] = -1;

  //Get the counter names from the environment.
  for(int c=0; c<MAX_TAU_COUNTERS; c++)
    {
      MultipleCounterLayer::names[c] = getenv(environment[c]);
    }

  
  cout << "The names obtained were:" << endl;
  for(int d=0; d<MAX_TAU_COUNTERS; d++)
    {
      cout << "COUNTER" << d << " = " ;

      if(MultipleCounterLayer::names[d] != NULL)
	cout << MultipleCounterLayer::names[d] << endl;
      else
	cout << endl;
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

	  //       cout << "Adding function specified by in init function in position: " << e << " of the init array." << endl;
	}
      else{
	//	cout << "Not adding function specified by in init function in position: " << e << " of the init array." << endl;
      }
    }

  //Check to see that we have at least one counter defined.
  //Give a warning of not.  It should not break the system,
  //but it is nice to give a warning.
  if(numberOfActiveFunctions == 0)
    cout << "Warning: No functions active ... no profiles will be created!" << endl;
  return returnValue;
}

void MultipleCounterLayer::getCounters(int tid, double values[])
{
  static bool initFlag = initializeMultiCounterLayer();

  //Just cycle through the list of function in the active function array.
  for(int i=0; i<numberOfActiveFunctions; i++)
    {
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

bool MultipleCounterLayer::gettimeofdayMCLInit(int functionPosition)
{
  //This function uses the Unix gettimeofday function.
  //Checks for GETTIMEOFDAY

  for(int i=0; i<MAX_TAU_COUNTERS; i++)
    {
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "GET_TIME_OF_DAY") == 0){
	
	  //	  cout << "gettimeofdayMCL is active." << endl;

	  //Set the counter position.
	  gettimeofdayMCL_CP[0] = i;

	  //Indicate that this function is being used.
	  MultipleCounterLayer::counterUsed[i] = true;

	  //Update the functionArray.
	  //	  cout << "Inserting gettimeofdayMCL in position: " << functionPosition << endl;
	  MultipleCounterLayer::functionArray[functionPosition] = gettimeofdayMCL;
	  //Now just return with beginCountersPosition incremented.
	  return true;
	}
      }
    }

  //If we are here, then this function is not active.
  //cout << "gettimeofdayMCL is not active." << endl;
  return false;
}

bool MultipleCounterLayer::papiMCLInit(int functionPosition){
#ifdef TAU_PAPI
  //This function uses the papi layer counters.
  
  bool returnValue = false;
  
  for(int i=0; i<MAX_TAU_COUNTERS; i++){
      if(MultipleCounterLayer::names[i] != NULL){
	if (strstr(MultipleCounterLayer::names[i],"PAPI") != NULL) {
	  
	  int tmpCode = PapiLayer::map_eventnames(MultipleCounterLayer::names[i]);
	  
	  if(tmpCode != -1){
	 
	    cout << "Found a papi counter: " << MultipleCounterLayer::names[i] << endl;

	    PapiLayer::multiCounterPapiInit();

	    //Check if this is possible on this machine!
	    if((PAPI_query_event(tmpCode) == PAPI_OK)){
	      
	      //Set the counter position.
	      papiMCL_CP[numberOfPapiHWCounters] = i;
	      
	      //Set the counter code.
	      MultipleCounterLayer::PAPI_CounterCodeList[numberOfPapiHWCounters] = tmpCode;
	      
	      //Update the number of Papi counters.
	      numberOfPapiHWCounters++;
	      
	      //Indicate that this position is being used.
	      MultipleCounterLayer::counterUsed[i] = true;
	      
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
    //If we found viable Papi events, update the function array.
    cout << "Inserting papiMCL in position: " << functionPosition << endl;
    MultipleCounterLayer::functionArray[functionPosition] = papiMCL;
  }

  if(!returnValue)
    cout << "papiMCL is not active." << endl;
  return returnValue;
#else //TAU_PAPI
  return false;
#endif//TAU_PAPI
}

bool MultipleCounterLayer::papiWallClockMCLInit(int functionPosition)
{
#ifdef TAU_PAPI
  //This function uses the papi PAPI_get_real_usec()  function.
  //Checks for P_WALL_CLOCK

  for(int i=0; i<MAX_TAU_COUNTERS; i++)
    {
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "P_WALL_CLOCK") == 0){

	  cout << "papiWallClockMCL is active." << endl;
	  PapiLayer::multiCounterPapiInit();

	  //Set the counter position.
	  papiWallClockMCL_CP[0] = i;

	  //Indicate that this function is being used.
	  MultipleCounterLayer::counterUsed[i] = true;

	  //Update the functionArray.
	  cout << "Inserting papiWallClockMCL in position: " << functionPosition << endl;
	  MultipleCounterLayer::functionArray[functionPosition] = papiWallClockMCL;
	  //Now just return with beginCountersPosition incremented.
	  return true;
	}
      }
    }

  //If we are here, then this function is not active.
  cout << "papiWallClockMCL is not active." << endl;
  return false;
#else  // TAU_PAPI
  return false;
#endif // TAU_PAPI
}

bool MultipleCounterLayer::papiVirtualMCLInit(int functionPosition)
{
#ifdef TAU_PAPI
  //This function uses the papi PAPI_get_virt_usec()  function.
  //Checks for P_VIRTUAL

  for(int i=0; i<MAX_TAU_COUNTERS; i++)
    {
      if(MultipleCounterLayer::names[i] != NULL){
        if(strcmp(MultipleCounterLayer::names[i], "P_VIRTUAL") == 0){

          cout << "papiVirtualMCL is active." << endl;
          PapiLayer::multiCounterPapiInit();

          //Set the counter position.
          papiVirtualMCL_CP[0] = i;

	  //Indicate that this function is being used.
	  MultipleCounterLayer::counterUsed[i] = true;

          //Update the functionArray.
          cout << "Inserting papiVirtualMCL in position: " << functionPosition << endl;
          MultipleCounterLayer::functionArray[functionPosition] = papiVirtualMCL;
          //Now just return with beginCountersPosition incremented.
          return true;
        }
      }
    }

  //If we are here, then this function is not active.
  cout << "papiVirtualMCL is not active." << endl;
  return false;
#else  // TAU_PAPI
  return false;
#endif // TAU_PAPI
}

bool MultipleCounterLayer::pclMCLInit(int functionPosition){
#ifdef  TAU_PCL
  return false;
#else //TAU_PCL
  return false;
#endif//TAU_PCL
}

bool MultipleCounterLayer::linuxTimerMCLInit(int functionPosition){return false;}

void MultipleCounterLayer::gettimeofdayMCL(int tid, double values[]){

  //cout << endl;
  //cout << "gettimeofdayMCL" << endl;
  //cout << "Storing value in position: " << gettimeofdayMCL_CP[0] << endl;

  struct timeval tp;
  gettimeofday (&tp, 0);
  values[gettimeofdayMCL_CP[0]] = ( (double) tp.tv_sec * 1e6 + tp.tv_usec );

}

void MultipleCounterLayer::papiMCL(int tid, double values[]){
#ifdef TAU_PAPI
  static ThreadValue * ThreadList[TAU_MAX_THREADS];

  static int PAPI_CounterList[];


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
      
      if((PAPI_add_events(&(ThreadList[tid]->EventSet),
			  PAPI_CounterCodeList,
			  numberOfPapiHWCounters)) != PAPI_OK){
	cout << "Error adding Papi events!" << endl;
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
#ifdef TAU_PAPI
  //cout << endl;
  //cout << "PapiWallClockMCL" << endl;
  //cout << "Storing value in position: " << papiWallClockMCL_CP[0] << endl;

  values[papiWallClockMCL_CP[0]] = PAPI_get_real_usec();
#endif // TAU_PAPI
}
void MultipleCounterLayer::papiVirtualMCL(int tid, double values[]){
#ifdef TAU_PAPI
  //cout << endl;
  //cout << "PapiVirtualMCL" << endl;
  //cout << "Storing value in position: " << papiVirtualMCL_CP[0] << endl;

  values[papiVirtualMCL_CP[0]] = PAPI_get_virt_usec();
#endif // TAU_PAPI
}

void MultipleCounterLayer::pclMCL(int tid, double values[]){
#ifdef  TAU_PCL
#endif//TAU_PCL
}

void MultipleCounterLayer::linuxTimerMCL(int tid, double values[]){}

/////////////////////////////////////////////////
//
//End - Definintions for MultipleCounters.
//
/////////////////////////////////////////////////
