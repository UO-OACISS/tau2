/////////////////////////////////////////////////
//Function definintion file for the Papi_Layer class.
//
//Author:   Robert Ansell-Bell
//Created:  February 2000
//
/////////////////////////////////////////////////

#include "Profile/Profiler.h"
#include <iostream>

//Helper function used to determine the counter value
//from the event name.
int map_eventnames(char *name)
{
  int i;

  if((PAPI_event_name_to_code(name, &i)) != PAPI_OK)
    return -1;
  else
    return i;
}

/////////////////////////
//If mutex locking of resourses is desired
//define PAPI_MUTEX_LOCK and the following version
//of getCounters will be used.
/////////////////////////
#ifdef PAPI_MUTEX_LOCK

#include <pthread.h>




long long PapiLayer::getCounters(int tid)
{
  static ThreadValue * ThreadList[TAU_MAX_THREADS];
  
  //Defining some thread locking stuff.
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

  //Defining a static test boolean.
  static bool RunBefore = false;
  
  //PAPI init stuff.
  int PAPI_CounterList[1] = {0};
  long long values[1] = {-1};
  int Result;

  //******************************************
  //This section is only run once.
  //******************************************
  if((pthread_mutex_lock(&lock) != 0))
     {
       cout << "There was a problem locking or unlocking a resource" << endl;
       return -1;
     }
  
  if(!RunBefore)
    {
      //Initializing the static thread list.
      for(int i=0;i<TAU_MAX_THREADS;i++)
	ThreadList[i] = NULL;

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
	  cout << "Error - You must define the PAPI_EVENT environment variable." << endl;
	  if((pthread_mutex_unlock(&lock) != 0))
	     {
	       cout << "There was a problem locking or unlocking a resource" << endl;
	       return -1;
	     }
	  return -1;
	}

      //Check if this is possible on this machine!
      if((PAPI_query_event(PAPI_CounterList[0])) != PAPI_OK)
	{
	  cout << "Requested events not possible" << endl;
	  if((pthread_mutex_unlock(&lock) != 0))
	     {
	       cout << "There was a problem locking or unlocking a resource" << endl;
	       return -1;
	     }
	  return -1;
	}

      RunBefore = true;
    }
  if((pthread_mutex_unlock(&lock) != 0))
     {
       cout << "There was a problem locking or unlocking a resource" << endl;
       return -1;
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
      ThreadList[tid]->CounterValue = 0;
      
      //cout << "New thread added to the thread list." << endl;

      //Starting the counter.
      if((pthread_mutex_lock(&lock) != 0))
	 {
	   cout << "There was a problem locking or unlocking a resource" << endl;
	   return -1;
	 }
      Result = PAPI_start_counters(&PAPI_CounterList[0], 1);
      if((pthread_mutex_unlock(&lock) != 0))
	 {
	   cout << "There was a problem locking or unlocking a resource" << endl;
	   return -1;
	 }
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

  if((pthread_mutex_lock(&lock) != 0))
     {
       cout << "There was a problem locking or unlocking a resource" << endl;
       return -1;
     }

  if((PAPI_read_counters(&values[0], 1)) != PAPI_OK)
    {
      cout << "There were problems reading the counters" << endl;
      if((pthread_mutex_unlock(&lock) != 0))
	 {
	   cout << "There was a problem locking or unlocking a resource" << endl;
	   return -1;
	 }
      return -1;
    }

  if((pthread_mutex_unlock(&lock) != 0))
     {
       cout << "There was a problem locking or unlocking a resource" << endl;
       return -1;
     }

  
    //Update the ThreadList and return the counter value.
    ThreadList[tid]->CounterValue = ThreadList[tid]->CounterValue  + values[0];
  
  //cout << "The thread ID is: " << tid << endl;
  //cout << "The value being returned is: " << (ThreadList[tid]->CounterValue) << endl;
  return ThreadList[tid]->CounterValue;

}

/////////////////////////
//The following is the default if no sychronization
//is required.
/////////////////////////
#else

long long PapiLayer::getCounters(int tid)
{
  static ThreadValue * ThreadList[TAU_MAX_THREADS];
  
  //Defining a static test boolean.
  static bool RunBefore = false;
  
  //PAPI init stuff.
  int PAPI_CounterList[1] = {0};
  long long values[1] = {-1};
  int Result;

  //******************************************
  //This section is only run once.
  //******************************************
  if(!RunBefore)
    {
      //Initializing the static thread list.
      for(int i=0;i<TAU_MAX_THREADS;i++)
	ThreadList[i] = NULL;

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
	  cout << "Error - You must define the PAPI_EVENT environment variable." << endl;

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
      ThreadList[tid]->CounterValue = 0;
      
      //cout << "New thread added to the thread list." << endl;

      //Starting the counter.
      Result = PAPI_start_counters(&PAPI_CounterList[0], 1);

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
  if((PAPI_read_counters(&values[0], 1)) != PAPI_OK)
    {
      cout << "There were problems reading the counters" << endl;
      return -1;
    }


    //Update the ThreadList and return the counter value.
    ThreadList[tid]->CounterValue = ThreadList[tid]->CounterValue  + values[0];
  
  //cout << "The thread ID is: " << tid << endl;
  //cout << "The value being returned is: " << (ThreadList[tid]->CounterValue) << endl;
  return ThreadList[tid]->CounterValue;
}

#endif

/////////////////////////////////////////////////
//
//End PapiLayer class functions definition file.
//
/////////////////////////////////////////////////



