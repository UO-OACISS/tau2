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

int MultipleCounterLayer::gettimeofdayMCL_CP[1] = {-1};
int MultipleCounterLayer::papiWallClockMCL_CP[1] = {-1};
int MultipleCounterLayer::papiVirtualMCL_CP[1] = {-1};
int MultipleCounterLayer::linuxTimerMCL_CP[1] = {-1};

firstListType MultipleCounterLayer::initArray[] = {gettimeofdayMCLInit,
				    				     papiWallClockMCLInit,
								     papiVirtualMCLInit,
								     linuxTimerMCLInit};

int MultipleCounterLayer::numberOfActiveFunctions = 0;
secondListType MultipleCounterLayer::functionArray[] = { };
char * MultipleCounterLayer::names[] = { };


bool MultipleCounterLayer::initializeMultiCounterLayer(void)
{
  bool returnValue = true;
  int functionPosition = 0;

  //Initialize the function array to NULL.
  for(int a=0; a<MAX_TAU_COUNTERS; a++)
    {
      functionArray[a] = NULL;
    }

  //Initialize the names array to NULL.
  for(int b=0; b<MAX_TAU_COUNTERS; b++)
    {
      MultipleCounterLayer::names[b] = NULL;
    }

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

       cout << "Adding function specified by in init function in position: " << e << " of the init array." << endl;
	}
      else{
	cout << "Not adding function specified by in init function in position: " << e << " of the init array." << endl;
      }
    }
  return returnValue;
}

void MultipleCounterLayer::getCounters(int tid, double values[])
{
  static bool initFlag = initializeMultiCounterLayer();

  //Just cycle through the list of function in the active function array.
  for(int i=0; i<numberOfActiveFunctions; i++)
    {
      MultipleCounterLayer::functionArray[i](values);
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
	
	  cout << "gettimeofdayMCL is active." << endl;

	  //Set the counter position.
	  gettimeofdayMCL_CP[0] = i;

	  //Update the functionArray.
	  cout << "Inserting gettimeofdayMCL in position: " << functionPosition << endl;
	  MultipleCounterLayer::functionArray[functionPosition] = gettimeofdayMCL;
	  //Now just return with beginCountersPosition incremented.
	  return true;
	}
      }
    }

  //If we are here, then this function is not active.
  cout << "gettimeofdayMCL is not active." << endl;
  return false;
}

bool MultipleCounterLayer::linuxTimerMCLInit(int functionPosition){return false;}


bool MultipleCounterLayer::papiWallClockMCLInit(int functionPosition)
{
#ifdef TAU_PAPI
  //This function uses the papi PAPI_get_real_usec()  function.
  //Checks for PAPI_WALL_CLOCK

  for(int i=0; i<MAX_TAU_COUNTERS; i++)
    {
      if(MultipleCounterLayer::names[i] != NULL){
	if(strcmp(MultipleCounterLayer::names[i], "PAPI_WALL_CLOCK") == 0){

	  cout << "papiWallClockMCL is active." << endl;
	  PapiLayer::multiCounterPapiInit();

	  //Set the counter position.
	  papiWallClockMCL_CP[0] = i;

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
  //Checks for PAPI_VIRTUAL

  for(int i=0; i<MAX_TAU_COUNTERS; i++)
    {
      if(MultipleCounterLayer::names[i] != NULL){
        if(strcmp(MultipleCounterLayer::names[i], "PAPI_VIRTUAL") == 0){

          cout << "papiVirtualMCL is active." << endl;
          PapiLayer::multiCounterPapiInit();

          //Set the counter position.
          papiVirtualMCL_CP[0] = i;

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

void MultipleCounterLayer::gettimeofdayMCL(double values[]){

  cout << endl;
  cout << "gettimeofdayMCL" << endl;
  cout << "Storing value in position: " << gettimeofdayMCL_CP[0] << endl;

  struct timeval tp;
  gettimeofday (&tp, 0);
  values[gettimeofdayMCL_CP[0]] = ( (double) tp.tv_sec * 1e6 + tp.tv_usec );

}
void MultipleCounterLayer::linuxTimerMCL(int functionPosition){}
void MultipleCounterLayer::papiWallClockMCL(double values[]){
#ifdef TAU_PAPI
  cout << endl;
  cout << "PapiWallClockMCL" << endl;
  cout << "Storing value in position: " << papiWallClockMCL_CP[0] << endl;

  values[papiWallClockMCL_CP[0]] = PAPI_get_real_usec();
#endif // TAU_PAPI
}
void MultipleCounterLayer::papiVirtualMCL(double values[]){
#ifdef TAU_PAPI
  cout << endl;
  cout << "PapiVirtualMCL" << endl;
  cout << "Storing value in position: " << papiVirtualMCL_CP[0] << endl;

  values[papiVirtualMCL_CP[0]] = PAPI_get_virt_usec();
#endif // TAU_PAPI
}

/////////////////////////////////////////////////
//
//End - Definintions for MultipleCounters.
//
/////////////////////////////////////////////////
