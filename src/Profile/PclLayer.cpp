/////////////////////////////////////////////////
//Function definintion file for the PCL_Layer class.
//
//Author:   Robert Ansell-Bell
//Created:  July 1999
//
/////////////////////////////////////////////////

#include "Profile/Profiler.h"
#include <iostream.h>

//Helper function used to determine the counter value
//from the event name.
int map_eventnames(char *name)
{
  int i;

  /* search all names */
  for (i = 0; i < PCL_MAX_EVENT; ++i)
    if (!strcmp(name, PCLeventname(i)))
      return i;

  /* not found */
  return -1;
}
/////////////////////////
//If mutex locking of resourses is desired
//define PCL_MUTEX_LOCK and the following version
//of getCounters will be used.
/////////////////////////
#ifdef PCL_MUTEX_LOCK

#include <pthread.h>




PCL_FP_CNT_TYPE PCL_Layer::getCounters(int tid)
{
  static ThreadValue * ThreadList[TAU_MAX_THREADS];
  
  //Defining some thread locking stuff.
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

  //Defining a static test boolean.
  static bool RunBefore = false;
  
  //PCl init stuff.
  static int PCL_CounterList[1] = {0};
  static unsigned int PCL_Mode = PCL_MODE_USER;
  static PCL_DESCR_TYPE descr;
  PCL_CNT_TYPE ResultsList[1];
  PCL_FP_CNT_TYPE FpResultsList[1];
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
      //Call PCLinit.
      if(PCLinit(&descr) != PCL_SUCCESS)
	{
	  cout << "unable to allocate PCL handle" << endl;
	  return;
	}

      //Initializing the static thread list.
      for(int i=0;i<TAU_MAX_THREADS;i++)
	ThreadList[i] = NULL;

      char * EnvironmentVariable = NULL;
      EnvironmentVariable = getenv("PCL_EVENT");
      if(EnvironmentVariable != NULL)
	{
	  PCL_CounterList[0] = map_eventnames(EnvironmentVariable); 
	  if(PCL_CounterList[0] == -1)
	    {
	      cout << "Unable to determine event type" << endl;
	      return -1;
	    }
	}
	else
	{
	  cout << "Error - You must define the PCL_EVENT environment variable." << endl;
	  if((pthread_mutex_unlock(&lock) != 0))
	     {
	       cout << "There was a problem locking or unlocking a resource" << endl;
	       return -1;
	     }
	  return -1;
	}

      //Check if this is possible on this machine!
      if(PCLquery(descr, PCL_CounterList, 1, PCL_Mode) != PCL_SUCCESS)
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
      Result = PCLstart(descr, PCL_CounterList, 1, PCL_Mode);
      if((pthread_mutex_unlock(&lock) != 0))
	 {
	   cout << "There was a problem locking or unlocking a resource" << endl;
	   return -1;
	 }
      if(Result != PCL_SUCCESS)
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

  if( PCLread(descr, &ResultsList[0], &FpResultsList[0], 1) != PCL_SUCCESS)
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

  
  if ( PCL_CounterList[0] < PCL_MFLOPS )
  {
    //Update the ThreadList and return the counter value.
    ThreadList[tid]->CounterValue = ResultsList[0];
  } 
  else
  { // FP result 
    //Update the ThreadList and return the counter value.
    ThreadList[tid]->CounterValue = FpResultsList[0];
  }
  
  //cout << "The thread ID is: " << tid << endl;
  //cout << "The value being returned is: " << (ThreadList[tid]->CounterValue) << endl;
  return ThreadList[tid]->CounterValue;

}

/////////////////////////
//The following is the default if no sychronization
//is required.
/////////////////////////
#else

PCL_FP_CNT_TYPE PCL_Layer::getCounters(int tid)
{
  static ThreadValue * ThreadList[TAU_MAX_THREADS];
  
  //Defining a static test boolean.
  static bool RunBefore = false;
  
  //PCl init stuff.
  static int PCL_CounterList[1] = {0};
  static unsigned int PCL_Mode = PCL_MODE_USER;
  static PCL_DESCR_TYPE descr;
  PCL_CNT_TYPE ResultsList[1];
  PCL_FP_CNT_TYPE FpResultsList[1];
  int Result;

  //******************************************
  //This section is only run once.
  //******************************************
  if(!RunBefore)
    {
      //Call PCLinit.
      if(PCLinit(&descr) != PCL_SUCCESS)
	{
	  cout << "unable to allocate PCL handle" << endl;
	  return -1;
	}

      //Initializing the static thread list.
      for(int i=0;i<TAU_MAX_THREADS;i++)
	ThreadList[i] = NULL;


      char * EnvironmentVariable = NULL;
      EnvironmentVariable = getenv("PCL_EVENT");

      if(EnvironmentVariable != NULL)
	{
	  PCL_CounterList[0] = map_eventnames(EnvironmentVariable); 
	  if(PCL_CounterList[0] == -1)
	    {
	      cout << "Unable to determine event type" << endl;
	      return -1;
	    }
	}
      else
	{
	  cout << "Error - You must define the PCL_EVENT environment variable." << endl;
	  return -1;
	}

      //Check if this is possible on this machine!
      if(PCLquery(descr, PCL_CounterList, 1, PCL_Mode) != PCL_SUCCESS)
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
      Result = PCLstart(descr, PCL_CounterList, 1, PCL_Mode);

      if(Result != PCL_SUCCESS)
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
  if( PCLread(descr, &ResultsList[0], &FpResultsList[0], 1) != PCL_SUCCESS)
    {
      cout << "There were problems reading the counters" << endl;
      return -1;
    }


  if ( PCL_CounterList[0] < PCL_MFLOPS )
  {
    //Update the ThreadList and return the counter value.
    //ThreadList[tid]->CounterValue = 
    //ThreadList[tid]->CounterValue + ResultsList[0];

    ThreadList[tid]->CounterValue = ResultsList[0];
  } 
  else
  { // FP result 
    //Update the ThreadList and return the counter value.
    ThreadList[tid]->CounterValue = FpResultsList[0];
 
    
  }

  //cout << "The thread ID is: " << tid << endl;
  //cout << "The value being returned is: " << (ThreadList[tid]->CounterValue) << endl;

  return ThreadList[tid]->CounterValue;
}

#endif

/////////////////////////////////////////////////
//
//End PCL_Layer class functions definition file.
//
/////////////////////////////////////////////////
