/////////////////////////////////////////////////
//Function definintion file for the Papi_Layer class.
//
//Author:   Robert Ansell-Bell
//Created:  February 2000
//
/////////////////////////////////////////////////

#include "Profile/Profiler.h"
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

//Some useful defines.
#define NUMBER_OF_COUNTERS 1 //At the moment TAU only supports one counter.


//Helper function used to determine the counter value
//from the event name.
int PapiLayer::map_eventnames(char *name)
{
  int i;

  if((PAPI_event_name_to_code(name, &i)) != PAPI_OK)
    return -1;
  else
    return i;
}

////////////////////////////////////////////////////
// Accessing hardware performance counters for both
// single and multithreaded cases
////////////////////////////////////////////////////
long long PapiLayer::getCounters(int tid)
{
  static ThreadValue * ThreadList[TAU_MAX_THREADS];
  
  //Defining a static test boolean.
  static bool RunBefore = false;
  
  //PAPI init stuff.
  
  //Defining a static element which indicates the list of counters
  //to be measured.  This array is initialized in the run-once section
  //below.
  static int PAPI_CounterList[NUMBER_OF_COUNTERS];

  //A test declaration used to test the success of function calls.
  //It does not store any counter values.
  int Result = -1;

  //******************************************
  //This section is only run once.
  //******************************************
  if(!RunBefore)
    {
      //Initializing the static thread list.
      for(int i=0;i<TAU_MAX_THREADS;i++)
	ThreadList[i] = NULL;

      //Initialize the counter list.
      for(int j=0;j<NUMBER_OF_COUNTERS;j++)
	{
	  PAPI_CounterList[j] = -1;
	}
      
      
      int papi_ver = PAPI_library_init(PAPI_VER_CURRENT);
      if (papi_ver < 0) {
	cout << "PAPI Initialization error: " << papi_ver << endl;
	return -1;
      }
      
      if (papi_ver != PAPI_VER_CURRENT) {
	cout << "Wrong version of PAPI library" << endl;
	return -1;
      }

#ifndef PAPI_VERSION
/* PAPI 2 support goes here */
#ifdef __alpha
    if (false)
#else
    if(PAPI_thread_init((unsigned long (*)(void))(RtsLayer::myThread),0) != PAPI_OK)
#endif /* alpha PAPI thread problem */
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)
/* PAPI 3 support goes here */
#ifdef __alpha
    if (false)
#else
    if(PAPI_thread_init((unsigned long (*)(void))(RtsLayer::myThread)) != PAPI_OK)
#endif /* alpha PAPI thread problem */
#else
/* PAPI future support goes here */
#error "Compiling against a not yet released PAPI version"
#endif 
       {
	 cout << "There was a problem with papi's thread init" << endl;
	 return -1;
       }

      char * EnvironmentVariable = NULL;
      EnvironmentVariable = getenv("PAPI_EVENT");
      if(EnvironmentVariable != NULL)
	{
	  PAPI_CounterList[0] = map_eventnames(EnvironmentVariable); 
	  if(PAPI_CounterList[0] == -1)
	    {
	      cout << "Unable to determine event type" << endl;
	      return -1;
	    }
	}
	else
	{
#ifndef TAU_EPILOG
	  cout << "Error - You must define the PAPI_EVENT environment variable." << endl;
#endif /* TAU_EPILOG */

	  return -1;
	}


      //Check if this is possible on this machine!
      if((PAPI_query_event(PAPI_CounterList[0])) != PAPI_OK)
	{
	  cout << "Requested events not possible" << endl;
	  return -1;
	}

      RunBefore = true;
    }

  //******************************************

  //******************************************
  //Start peformance counting.
  //This section is run once for each thread.
  //******************************************

  //First check to see if the thread is already
  //present.  If not, register the thread and
  //then start the counters.
  if(ThreadList[tid] == NULL)
    {
      //Register thread and start the counters.
      //Since this is also the first call to
      //getCounters for this thread, just return
      //zero.
      if(tid >= TAU_MAX_THREADS)
	{
	  cout << "Exceeded max thread count of TAU_MAX_THREADS" << endl;
	  return -1;
	}
      ThreadList[tid] = new ThreadValue;
      ThreadList[tid]->ThreadID = tid;
      ThreadList[tid]->EventSet = PAPI_NULL;
      ThreadList[tid]->CounterValues = new long long[NUMBER_OF_COUNTERS];

      PAPI_create_eventset(&(ThreadList[tid]->EventSet));

      //Adding the events and starting the counter.
#ifndef PAPI_VERSION
/* PAPI 2 support goes here */
      Result = PAPI_add_events(&(ThreadList[tid]->EventSet) ,PAPI_CounterList ,NUMBER_OF_COUNTERS);
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)
/* PAPI 3 support goes here */
      Result = PAPI_add_events(ThreadList[tid]->EventSet ,PAPI_CounterList ,NUMBER_OF_COUNTERS);
#else
/* PAPI future support goes here */
#error "Compiling against a not yet released PAPI version"
#endif 
      if(Result != PAPI_OK)
	{
	  cout << "Something went wrong adding events!" << endl;
	  return -1;
	}

      Result = PAPI_start(ThreadList[tid]->EventSet);

      if(Result != PAPI_OK)
	{
	  cout << "Something went wrong" << endl;
	  return -1;
	}

      //Now return zero as this thread has only just begun
      //counting(as mentioned before).
      return 0;
    }

  //If here, it means that the thread has already been registered
  //and we need to just read and update the counters.
  
  //*****************************************
  //Reading the performance counters and
  //outputting the counter values.
  //*****************************************
  if((PAPI_read(ThreadList[tid]->EventSet, ThreadList[tid]->CounterValues)) != PAPI_OK)
    {
      cout << "There were problems reading the counters" << endl;
      return -1;
    }

  return ThreadList[tid]->CounterValues[0];  //At the moment, TAU can only deal
                                             //with one value being returned.
}


/////////////////////////////////////////////////
int PapiLayer::PapiLayerInit(bool lock)
{ 
  static bool flag = true;

  if(lock)
    RtsLayer::LockDB();
  if (flag)
  {
    flag = false; 
    // Initialization routine for PAPI timers
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
      cout << "Wrong version of PAPI library" << endl;
      return -1;
    }
 
#ifndef PAPI_VERSION
/* PAPI 2 support goes here */
#ifdef __alpha
    if (false)
#else
    if(PAPI_thread_init((unsigned long (*)(void))(RtsLayer::myThread),0) != PAPI_OK)
#endif /* alpha PAPI thread problem */
#elif (PAPI_VERSION_MAJOR(PAPI_VERSION) == 3)
/* PAPI 3 support goes here */
#ifdef __alpha
    if (false)
#else
    if(PAPI_thread_init((unsigned long (*)(void))(RtsLayer::myThread)) != PAPI_OK)
#endif /* alpha PAPI thread problem */
#else
/* PAPI future support goes here */
#error "Compiling against a not yet released PAPI version"
#endif 
    {
      cout << "There was a problem with papi's thread init" << endl;
      return -1;
    }
     
  }
  if(lock)
    RtsLayer::UnLockDB();
  return 0;
}

void PapiLayer::multiCounterPapiInit(void)
{
  //This function has the possibility if being called
  //one or more times by MultipleCounter routines.
  //Only want to do the init. once however.
  static int initFlag = PapiLayerInit(false);
}

/////////////////////////////////////////////////
long long PapiLayer::getWallClockTime(void)
{ // Returns the wall clock time from PAPI interface
  static int initflag = PapiLayerInit();
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
  return (newvalue + offset);
#endif // TAU_PAPI


  // OLD code: return PAPI_get_real_usec();
}

/////////////////////////////////////////////////
long long PapiLayer::getVirtualTime(void)
{ // Returns the virtual (user) time from PAPI interface
  static int initflag = PapiLayerInit();

  return PAPI_get_virt_usec();
}

/////////////////////////////////////////////////
//
//End PapiLayer class functions definition file.
//
/////////////////////////////////////////////////



