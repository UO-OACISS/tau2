/************************************************************************
 ************************************************************************/

/************************************************************************
	stack.cc
	Author: Ryan K. Harris
 
	Source file for stack class.
	*************************************************************************
 
	Member functions:
		push -- push one tray onto top of the stack
************************************************************************/

#include "stack.h"

#include <iostream>
using namespace std;

/************************************************************************
	stack::push
		Push one tray onto top of the stack.
		When it does this, it proceeds based on the exchange type:
			If it is type (1) (a call to a routine), it leaves this
			function at the top of the stack. It then calculates the
			difference between the previousTray's timeStamp and the
			current tray on top; this equals the exclusive time for
			the previousTray.
 
			If it is type (2) a return from a subroutine, it
			removes the new type (2) tray and the associated
			type (1) tray which should be the previous tray, i.e.
			should be on the stack directly before the current type (2).
			When it removes these two, it calculates the difference
			between them, this equals the inclusive time.
 
		After making either of these calculations, the calculated amount
		is added to either the exclusive or inclusive totals stored as
		private variables inside of the 'function' type nodes within
		the associated 'cpu' node.
	
	Operations marked with 'Equi' are sample manipulations equivalent to the
	timestamp operations being performed.
 
	Parameters: functionID, timeStamp, exchangeType, a_samples, n_samplesdim
************************************************************************/
void stack::push(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim, int n_exchangeType)
{

  /*
    If only a part of the trace is being analyzed the startMets and endMets
    values must be initialized like startTime and endTime
  */
  if(startTime != -1)
    {
      if((startset == 0) && (n_timeStamp >= startTime))
	{
	  startset = 1;
	  startMets = a_samples;
	}

      if((endset == 0) && (n_timeStamp >= endTime))
	{
	  endset = 1;
	  endMets = a_samples;
	}
    }

  if(n_exchangeType == 5)
    {
      newtray(n_functionID,n_timeStamp, a_samples, n_samplesdim);
    }
  else
    if(n_exchangeType == 1)
      {
	routine(n_functionID,n_timeStamp, a_samples, n_samplesdim);
      }	
    else
      if(n_exchangeType == 2)
	{
	  returnfrom(n_functionID,n_timeStamp, a_samples, n_samplesdim);
	}
	
}//push
	
void stack::newtray(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim)
{

  /* Build a new tray.
     We have a special build process when a n_exhcangeType == 5
     is received. This means that there was an 'Upfrom' event,
     but we want stack class to be able to handle this without
     major modification. I originally designed this stack class
     around 'Upto' and 'Downto' events since those were the
     original examples I was given. Therefore this forces
     'Upfrom' to be a special case. Basically I just alter some
     variables as though an 'Upto' event was received, after
     making some safety checks.
  */

  //if(n_exchangeType == 5)
  //{
  if(topTray != 0)
    {
      int topID = topTray->getFunctionID();
      int topExchange = topTray->getExchangeType();
      if((topID == n_functionID) && (topExchange == 1))
	{
	  tray * topPrev = topTray->getPrevious();
	  if (topPrev)
	    { /* has a parent tray */
	      n_functionID = topPrev->getFunctionID();
	    }
	  else
	    { /* detected termination on the given thread */
	      /* set n_functionID to -1 which is checked later */
	      n_functionID = -1;
	    }
	  returnfrom(n_functionID, n_timeStamp,a_samples, n_samplesdim);
	  //n_exchangeType = 2;//???
	}
      else
	{
	  cerr << "received 'Upfrom' exchangeType(5) event, "
	       << "but the topTray in 'stack' has diff funcID or top.exchangeType != 1.\n";
	  parent->stackError = 1;
	  return;
	}
    }
  else
    {
      cerr << "received 'Upfrom' exchangeType(5) event, "
	   << "but the topTray in 'stack' is a NULL pointer.\n";
      parent->stackError = 1;
      return;
    }

	
}//newtray



/* If this is a call to a routine. */
//if(n_exchangeType == 1)
void stack::routine(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim)
{
	
  tray * tempTray = new tray(	n_functionID,
				n_timeStamp, a_samples, n_samplesdim,
				1);

  /* The following is for timeStamp error detection. */
  if(parent->lastTimeOne > n_timeStamp)
    {
      parent->totalOutOrder += 1;
      parent->lastTimeTwo = parent->lastTimeOne;
      parent->lastTimeOne = n_timeStamp;
    }//if
  else
    {
      if((parent->lastTimeTwo == n_timeStamp) && (parent->lastTimeOne < n_timeStamp))
	{
	  parent->totalRepeats += 1;
	}//if
      parent->lastTimeTwo = parent->lastTimeOne;
      parent->lastTimeOne = n_timeStamp;

    }//else


  /* Check whether stack is empty. */
  if(topTray == 0)
    {
      topTray = tempTray;
      lastEntry = tempTray;
      int tempFuncID = topTray->getFunctionID();
      node * tempNode = parent->funcTree->getFuncNode(tempFuncID);
      double tempTime = tempTray->getTimeStamp();
      //int * tempSamps = tempTray->getSamples();

      if((startTime != -1) && (endTime != -1))
	{
	  if((startTime <= tempTime) && (tempTime <= endTime))
	    {
	      tempNode->incCount();
	    }//if
	}//if
      else
	{
	  tempNode->incCount();
	}

      topTray->setLastEntrance(n_timeStamp);
      topTray->setLastSamples(a_samples);//Equi!

      return;
    }
  else
    {
      tempTray->setLastEntrance(n_timeStamp);
      tempTray->setLastSamples(a_samples);//Equi!

      topTray->setNext(tempTray);
      tempTray->setPrevious(topTray);

      /* Now calculate the exclusive time for previous function. */

      /* New addition: we now implement the ability for the user
	 to specify time intervals, so we first determine if
	 this event is in their requested interval. */
      double tempTime = tempTray->getTimeStamp();
      unsigned long long * tempSamps = tempTray->getSamples();//Equi!
      double tempExclusive = -1;	// Holds difference to be added.
      unsigned long long * tempSampExclusive = new unsigned long long[n_samplesdim];//Equi!

      //double lastTime = lastEntry->getTimeStamp();
      double lastTime = topTray->getLastEntrance();
      unsigned long long * lastSamples = topTray->getLastSamples();//Equi!

      /* If endTime == -1, then the interval option isn't set and
	 we are profiling the entire tracefile. */

      if(endTime != -1)
	{
	  if((startTime <= tempTime) && (lastTime <= endTime))
	    {
	      /* Check that lastTime is within the interval. */
	      if(lastTime < startTime)
		{
		  lastTime = startTime;
		  lastSamples = startMets;
		}//if
	      if(endTime < tempTime)
		{
		  tempTime = endTime;
		  tempSamps = endMets;
		}//if

	      tempExclusive = tempTime - lastTime;
	      for(int i = 0; i<n_samplesdim; i++)
		{
		  tempSampExclusive[i] = tempSamps[i]-lastSamples[i];
		}//for-Equi!

	    }//if
	  else
	    {
	      /* Nothing. The interval is set, but this
		 event doesn't take place inside of it.
		 The exclusive amount to be added is 0. */
	    }//else


	}//if
      else
	{
	  /* Else, no interval was set. */
	  tempExclusive = tempTime - lastTime;


	  for(int i = 0; i<n_samplesdim; i++)
	    {
	      tempSampExclusive[i] = tempSamps[i]-lastSamples[i];

	    }//for-Equi!
	}//else

      /* Now add the exclusive time to the appropriate node.
	 First, we need to get the functionID from the previous
	 tray. */

      /* This used to say:
	 int funcID = lastEntry->getFuncionID();
	 but I'm changing it to match obsolstack, I
	 believe it makes more sense to add this time
	 to the last tray which was on top. */

      int funcID = topTray->getFunctionID();

      /* Now set to the appropriate function node. */

      node * funcNode = (*parent).funcTree->getFuncNode(funcID);
      if(funcNode != 0)
	{
	  double valueToAdd = tempExclusive;
	  if (valueToAdd < 0) {
	    valueToAdd = 0;
	  }
	  funcNode->addExclusive(valueToAdd);
	  funcNode->addExcSamples(tempSampExclusive,n_samplesdim);//Equi!
	}

      /* Now make the topTray pointer point to the newest
	 addition to the stack (the top tray). */
      //delete topTray;
      topTray = tempTray;
      lastEntry = tempTray;

      /* Now incSubrs of previous function in topTray.previous. */
      tempTray = tempTray->getPrevious();
      //delete lastEntry;
      int tempFuncID = tempTray->getFunctionID();
      node * tempNode = parent->funcTree->getFuncNode(tempFuncID);

      if(tempNode != 0)
	{
	  // tempNode->incSubrs();

#ifdef DEBUG
	    printf ("\nstartTime == %G\n", startTime);
	    printf ("endTime == %G\n", endTime);
	    printf ("tempExclusive == %G\n", tempExclusive);
#endif /* DEBUG */
	  /* I'm not yet positive whether the next
	     conditional is required or not, I don't have
	     the "if(startTime != -1.....endTime != -1)
	     but I'll leave it here in stack for now. */
	  if ((startTime != -1) && (endTime != -1)) {
	    if((tempExclusive >= 0) && ((startTime <= n_timeStamp) && (n_timeStamp <= endTime))) {
	      tempNode->incSubrs();
	    }
	  } else if(tempExclusive >= 0){
	    tempNode->incSubrs();
	  }
	}

      /* The following is a new method for incrementing the function
	 call counters, based on the new interval implementation.
	 The idea is that we only increment the call counter if
	 we have made an addition to the time sums, i.e. only
	 if the call was made within our chosen interval. */
      tempFuncID = topTray->getFunctionID();
      tempNode = parent->funcTree->getFuncNode(tempFuncID);

      /* However, after more thought, I don't want to call
	 incCount based on tempExclusive alone, because this
	 doesn't necessarily mean that the function we're looking
	 at is within interval(I). Need to check also that
	 'n_timeStamp' is in (I). */

      if((startTime != -1) && (endTime != -1))
	{
	  if((startTime <= n_timeStamp) && (n_timeStamp <= endTime))
	    {
	      tempNode->incCount();
	    }//if
	}//if
      else
	{
	  tempNode->incCount();
	}//else
			
      return;
    }
}//exchangetype(1)




/* If this is a return from a routine. */
//if(n_exchangeType == 2)
void stack::returnfrom(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim)
{
  /* The following is for timeStamp error detection. */
  if(parent->lastTimeOne > n_timeStamp)
    {
      parent->totalOutOrder += 1;
      parent->lastTimeTwo = parent->lastTimeOne;
      parent->lastTimeOne = n_timeStamp;
    }//if
  else
    {
      if((parent->lastTimeTwo == n_timeStamp) && (parent->lastTimeOne < n_timeStamp))
	{
	  parent->totalRepeats += 1;
	}//if
      parent->lastTimeTwo = parent->lastTimeOne;
      parent->lastTimeOne = n_timeStamp;

    }//else
  /* Check whether stack is empty. */
  if(topTray == 0)
    {
      cerr << "received exchangeType(2) event when stack is "
	   << "already empty\n";
      parent->stackError = 1;
      return;
    }


  /* Now check that topTray is exchangeType(1). */
  if(topTray->getExchangeType() != 1)
    {
      cerr << "missmatched exchangeTypes in traceStack!\n";
      parent->stackError = 1;
      return;
    }

  else
    {
		
      //Set tempTray's 'lastEntrance' time.
      tray * prevTray = topTray->getPrevious();
      if(prevTray != 0)
	{
	  if((prevTray->getFunctionID()) != (n_functionID))
	    {
	      cerr << "missmatched functionID's in the trays\n";
	    }//if
	  else
	    {
	      prevTray->setLastEntrance(n_timeStamp);
	      prevTray->setLastSamples(a_samples);//Equi!
	    }//else
	}//if

      if(n_functionID == -1)
	{
	  //delete tempTray;
	  lastreturn(n_functionID, n_timeStamp, a_samples, n_samplesdim);
	  return;
	}

      /* Verify that the topTray.previousTray is an exchangeType(1)
	 with matching functionID. */

      tray * the_prev = topTray->getPrevious();
      if(the_prev != 0)
	{
	  if( (the_prev->getFunctionID()) != n_functionID )
	    {
	      cerr << "topTray.previous.functionID != "
		   << "tempTray.functionID in traceStack!\n\n";
	      parent->stackError = 1;
	      return;
	    }//if

	  else
	    {   
	      tray * tempTray = new tray(	n_functionID,
						n_timeStamp, a_samples, n_samplesdim,
						2);

	      double tempTime = tempTray->getTimeStamp();
	      double tempExclusive = 0;		// Holds difference to be added.
	      //double lastTime = lastEntry->getTimeStamp();
	      double lastTime = topTray->getLastEntrance();

	      //unsigned long * tempSamps = tempTray->getSamples();//Equi!
	      unsigned long long * tempSamps = new unsigned long long[n_samplesdim];
	      for(int i = 0; i< n_samplesdim;i++)
		{
		  tempSamps[i] = tempTray->getSamples()[i];
		}
					
	      unsigned long long * tempSampExclusive = new unsigned long long[n_samplesdim];//Equi!
	      unsigned long long * lastSamples = topTray->getLastSamples();//Equ

	      /* If endTime == -1, then the interval option isn't set and
		 we are profiling the entire tracefile. */

	      if(endTime != -1)
		{

		  if((tempTime < startTime) || (endTime < lastTime))
		    {
		      // Do nothing, doesn't occur within interval.
		    }//if
		  else
		    {
		      /*
			cerr << "startTime: " << startTime << "\n";
			cerr << "endTime: " << endTime << "\n";
			cerr << "tempTime: " << tempTime << "\n";
			cerr << "lastTime: " << lastTime << "\n";
		      */


		      if(endTime < tempTime)
			{
			  tempTime = endTime;
			  tempSamps = endMets;
			}
		      if(lastTime < startTime)
			{
			  lastTime = startTime;
			  lastSamples = startMets;
			}

		      tempExclusive = tempTime - lastTime;
		      for(int i = 0; i<n_samplesdim; i++)
			{
			  tempSampExclusive[i] = tempSamps[i]-lastSamples[i];
			}//for-Equi!

		    }//else

		}//if
	      else
		{
		  /* Else, no interval was set. */
		  tempExclusive = tempTime - lastTime;
		  for(int i = 0; i<n_samplesdim; i++)
		    {
		      tempSampExclusive[i] = tempSamps[i]-lastSamples[i];
		    }//for-Equi!
						

		}//else
	      delete[] tempSamps;
	      node * funcNode =
		(*parent).funcTree->getFuncNode(topTray->getFunctionID());


	      if(funcNode != 0)
		{
		  funcNode->addExclusive(tempExclusive);
		  funcNode->addExcSamples(tempSampExclusive,n_samplesdim);//Equi!
		}

	      /* Inclusive time needs to be handled (calculated)
		 more carefully when recursion is present in the
		 program; so for every exchangetype 2 event which
		 comes to our stack, we now verify whether there
		 already exists a tray on the stack representing
		 the function which is currently being pushed.
		 If an instance of the function already exists,
		 it implies either recursion or subsequent calls
		 to identical functions: we don't want to add the
		 same inclusive time to the same function multiple
		 times. */

	      tray * searchTray = topTray->getPrevious();
	      int inclusiveID = topTray->getFunctionID();
	      int recursionPresent = 0;
	      while(searchTray != 0)
		{
		  if( (searchTray->getFunctionID()) == inclusiveID)
		    {
		      recursionPresent = 1;
		    }//if

		  searchTray = searchTray->getPrevious();
		}//while

	      if(!recursionPresent)
		{
		  norec(funcNode, n_samplesdim, tempTray);
		}//if(!recursionPresent)


	      /* Garbage collect tempTray. */
	      //delete lastEntry;
	      lastEntry = tempTray;

	      /* Assign tempTray pointer = topTray for garbage collection
		 purposes. */
	      tempTray = topTray;


	      /* Now make topTray pointer point to topTray.previous. */
	      topTray = topTray->getPrevious();

	      /* Now make sure topTray.nextTray == NULL. */
	      //delete topTray->getNext();
	      topTray->setNext(0);

	      /* garbage collect what was topTray. */
	      //delete tempTray;
	      //delete[] tempSampExclusive;
	      //tempSampExclusive = NULL;
	      return;
	    }//else
	}//if
      else
	{
	  cerr << "error: the_prev == NULL in stack.\n";
	  parent->stackError = 1;
	  return;
	}//else
    }//else
	
  return;
}


void stack::norec(node * funcNode, int n_samplesdim, tray * tempTray)
{
  /*****THE FOLLOWING IS FOR INTERVAL CALCULATIONS*****/
  double tempTime = tempTray->getTimeStamp();
  double tempInclusive = 0;		// Holds difference to be added.
  double lastTime = topTray->getTimeStamp();

  //unsigned long * tempSamps = tempTray->getSamples();//Equi
  unsigned long long * tempSamps = new unsigned long long[n_samplesdim];
  for(int i = 0; i< n_samplesdim;i++)
    {
      tempSamps[i] = tempTray->getSamples()[i];
    }
  unsigned long long * tempSampInclusive = new unsigned long long[n_samplesdim];//Equi
  unsigned long long * lastSamples = topTray->getSamples();//Equi


  if(endTime != -1)
    {
      if(tempTime < startTime)
	{
	  /* Do nothing, doesn't occur within interval. */
	}//if
      else
	{
	  if(endTime < lastTime)
	    {
	      /* Do nothing, doesn't occur within interval. */
	    }//if
	  else
	    {
	      /* At this point we know we've found something
		 inside of the interval. */
	      if(endTime < tempTime)
		{
		  tempTime = endTime;
		  tempSamps = endMets;
		}//if
	      if(lastTime < startTime)
		{
		  lastTime = startTime;
		  lastSamples = startMets;
		}//if

	      tempInclusive = tempTime - lastTime;
	      for(int i = 0; i<n_samplesdim; i++)
		{
		  tempSampInclusive[i] = tempSamps[i]-lastSamples[i];
		}//for-Equi!

	    }//else
	}//else
    }//if
  else
    {
      /* Else, no interval was set. */
      tempInclusive = tempTime - lastTime;
      for(int i = 0; i<n_samplesdim; i++)
	{
	  tempSampInclusive[i] = tempSamps[i]-lastSamples[i];
	}//for-Equi!

    }//else

  /*
    double tempInclusive =
    (tempTray->getTimeStamp()) - (topTray->getTimeStamp());
  */

  if(funcNode != 0)
    {
      funcNode->addInclusive(tempInclusive);
      funcNode->addIncSamples(tempSampInclusive,n_samplesdim);//Equi!
    }
  delete[] tempSamps;
  delete tempTray;
  tempTray = 0;
}


/* When we receive the last return call and remove the
   last tray from the stack, the functionID on that
   return call will == -1. We need to check for that,
   and if it is the case, we cannot access
   topTray.previous as it is a NULL pointer. */
//if(n_functionID == -1)
void stack::lastreturn(int n_functionID, double n_timeStamp, unsigned long long * a_samples, int n_samplesdim)
{
  /* The purpose of having a seperate if statement for the
     case when functionID == -1 is that: when we set
     topTray = topTray.previous, we cannot in turn
     (don't need to actually) set the topTray
     previousTray and nextTray pointers, as topTray
     is now a NULL pointer. */
		
  tray * tempTray = new tray(	n_functionID,
				n_timeStamp, a_samples, n_samplesdim,
				2);


  double tempTime = tempTray->getTimeStamp();
  unsigned long long * tempSamps = tempTray->getSamples();//Equi
  double tempExclusive = 0;		// Holds difference to be added.
  unsigned long long * tempSampExclusive = new unsigned long long[n_samplesdim];//Equi
  //double lastTime = lastEntry->getTimeStamp();
  double lastTime = topTray->getLastEntrance();
  unsigned long long * lastSamples = topTray->getLastSamples();//Equi

  /* If endTime == -1, then the interval option isn't set and
     we are profiling the entire tracefile. */

  if(endTime != -1)
    {

      if((tempTime < startTime) || (endTime < lastTime))
	{
	  // Do nothing, doesn't occur within interval.
	}//if
      else
	{
	  /*
	    cerr << "startTime: " << startTime << "\n";
	    cerr << "endTime: " << endTime << "\n";
	    cerr << "tempTime: " << tempTime << "\n";
	    cerr << "lastTime: " << lastTime << "\n";
	  */

	  if(endTime < tempTime)
	    {
	      tempTime = endTime;
	      tempSamps = endMets;
	    }
	  if(lastTime < startTime)
	    {
	      lastTime = startTime;
	      lastSamples = startMets;
	    }

	  tempExclusive = tempTime - lastTime;
	  for(int i = 0; i<n_samplesdim; i++)
	    {
	      tempSampExclusive[i] = tempSamps[i]-lastSamples[i];
	    }//for-Equi!
	}//else

    }//if
  else
    {
      /* Else, no interval was set. */
      tempExclusive = tempTime - lastTime;
      for(int i = 0; i<n_samplesdim; i++)
	{
	  tempSampExclusive[i] = tempSamps[i]-lastSamples[i];
	}//for-Equi!

    }//else

  node * funcNode =
    (*parent).funcTree->getFuncNode(topTray->getFunctionID());

  if(funcNode != 0)
    {
      funcNode->addExclusive(tempExclusive);
      funcNode->addExcSamples(tempSampExclusive,n_samplesdim);//Equi!
    }


  /* Inclusive time needs to be handled (calculated)
     more carefully when recursion is present in the
     program; so for every exchangetype 2 event which
     comes to our stack, we now verify whether there
     already exists a tray on the stack representing
     the function which is currently being pushed.
     If an instance of the function already exists,
     it implies either recursion or subsequent calls
     to identical functions: we don't want to add the
     same inclusive time to the same function multiple
     times. */

  tray * searchTray = topTray->getPrevious();
  int inclusiveID = topTray->getFunctionID();
  int recursionPresent = 0;
  while(searchTray != 0)
    {
      if( (searchTray->getFunctionID()) == inclusiveID)
	{
	  recursionPresent = 1;
	}//if

      searchTray = searchTray->getPrevious();
    }//while

  if(!recursionPresent)
    {
      norec(funcNode, n_samplesdim, tempTray);
    }//if(!recursionPresent)

  /* Garbage collect tempTray. */
  //delete lastEntry;
  //lastEntry = 0;
  lastEntry = tempTray;

  /* Assign tempTray pointer = topTray for garbage collection
     purposes. */
  tempTray = topTray;


  /* Now make topTray pointer point to topTray.previous. */
  topTray = topTray->getPrevious();

  /* garbage collect what was topTray. */
  delete tempTray;
  //delete[] tempSampExclusive;
  //tempSampExclusive = NULL;
  return;
}//functionID == -1
