/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: UserEvent.cpp					  **
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
// Note: The default behavior of this library is to calculate all the
// statistics (min, max, mean, stddev, etc.) If the user wishes to 
// override these settings, SetDisableXXX routines can be used to do so
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

//#define DEBUG_PROF

#include "Profile/Profiler.h"


#include <stdio.h>
#include <fcntl.h>

//#include <math.h>
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */


#ifdef PGI
template void vector<TauUserEvent *>::insert_aux(vector<TauUserEvent *>::iterator, TauUserEvent *const &);
template TauUserEvent** copy_backward(TauUserEvent**,TauUserEvent**,TauUserEvent**);
template TauUserEvent** uninitialized_copy(TauUserEvent**,TauUserEvent**,TauUserEvent**);
#endif // PGI

vector<TauUserEvent*>& TheEventDB(int threadid)
{
  static vector<TauUserEvent*> EventDB;

  return EventDB;
}

// Add User Event to the EventDB
void TauUserEvent::AddEventToDB()
{
  RtsLayer::LockDB();
  TheEventDB().push_back(this);
  DEBUGPROFMSG("Successfully registered event " << GetEventName() << endl;);
  DEBUGPROFMSG("Size of eventDB is " << TheEventDB().size() <<endl);
  RtsLayer::UnLockDB();
  return;
}

// Constructor 
TauUserEvent::TauUserEvent(const char * EName)
{
  DEBUGPROFMSG("Inside ctor of TauUserEvent EName = "<< EName << endl;);

  EventName 	= EName;
  // Assign event name and then set the default values
  DisableMin 	= false; 	// Min 	    is calculated 
  DisableMax 	= false; 	// Max      is calculated 
  DisableMean 	= false; 	// Mean     is calculated 
  DisableStdDev = false; 	// StdDev   is calculated

  for(int i=0; i < TAU_MAX_THREADS; i++) 
  {
    LastValueRecorded[i] = 0;  	// null to start with
    NumEvents[i] = 0L; 		// initialize
    MinValue[i]  = 9999999;  	// Least -ve value? limits.h
    MaxValue[i]  = -9999999;		// Greatest  +ve value?    "
    SumSqrValue[i]  = 0;		// initialize
    SumValue[i]     = 0; 		// initialize
  }

  AddEventToDB();
  // Register this event in the main event database 
}

// Copy Constructor 
TauUserEvent::TauUserEvent(TauUserEvent& X)
{
  DEBUGPROFMSG("Inside copy ctor TauUserEvent::TauUserEvent()" << endl;);

  EventName 	= X.EventName;
  DisableMin	= X.DisableMin;
  DisableMax 	= X.DisableMax;
  DisableMean	= X.DisableMean;
  DisableStdDev = X.DisableStdDev;
/* Do we really need these? 
  LastValueRecorded = X.LastValueRecorded;
  NumEvents	= X.NumEvents;
  MinValue	= X.MinValue;
  MaxValue	= X.MaxValue;
  SumSqrValue	= X.SumSqrValue;
  SumValue	= X.SumValue;
 */

  AddEventToDB(); 
  //Register this event
}

// Default constructor
TauUserEvent::TauUserEvent()
{
  EventName 	= string("No Name");
  DisableMin 	= false; 	// Min 	    is calculated 
  DisableMax 	= false; 	// Max      is calculated 
  DisableMean 	= false; 	// Mean     is calculated 
  DisableStdDev = false; 	// StdDev   is calculated

  for (int i=0; i < TAU_MAX_THREADS; i++)
  {
    LastValueRecorded[i] = 0;  	// null to start with
    NumEvents[i] = 0L; 		// initialize
    MinValue[i]  = 9999999;  	// Least -ve value? limits.h
    MaxValue[i]  = -9999999;		// Greatest  +ve value?    "
    SumSqrValue[i]  = 0;		// initialize
    SumValue[i]     = 0; 		// initialize
  } 

  AddEventToDB();
  // Register this event in the main event database 
}

// Assignment operator
TauUserEvent& TauUserEvent::operator= (const TauUserEvent& X)
{

  DEBUGPROFMSG("Inside TauUserEvent::operator= (const TauUserEvent& X)" << endl;);

  EventName 	= X.EventName;
  DisableMin	= X.DisableMin;
  DisableMax 	= X.DisableMax;
  DisableMean	= X.DisableMean;
  DisableStdDev = X.DisableStdDev;
/* do we really need these? 
  LastValueRecorded = X.LastValueRecorded;
  NumEvents	= X.NumEvents;
  MinValue	= X.MinValue;
  MaxValue	= X.MaxValue;
  SumSqrValue	= X.SumSqrValue;
  SumValue	= X.SumValue;
*/

  return *this;
}

///////////////////////////////////////////////////////////
// TriggerEvent records the value of data in the UserEvent
///////////////////////////////////////////////////////////

void TauUserEvent::TriggerEvent(TAU_EVENT_DATATYPE data, int tid)
{ 
  // Record this value  
  LastValueRecorded[tid] = data;

  // Increment number of events
  NumEvents[tid] ++;

  // Compute relevant statistics for the data 
  if (!GetDisableMin()) 
  {  // Min is not disabled
     if (NumEvents[tid] > 1) {
     	MinValue[tid] = data < MinValue[tid] ? data : MinValue[tid];
     } else
	MinValue[tid] = data;
  }
  
  if (!GetDisableMax())
  {  // Max is not disabled
     if (NumEvents[tid] > 1) {
       MaxValue[tid] = MaxValue[tid] < data ? data : MaxValue[tid];
     } else
       MaxValue[tid] = data;
  }

  if (!GetDisableMean())
  {  // Mean is not disabled 
     SumValue[tid] += data; 
  }
     
  if (!GetDisableStdDev())
  {  // Standard Deviation is not disabled
     SumSqrValue[tid] += data*data; 
  }

  return; // completed calculating statistics for this event
}

// Return the data stored in the class
TAU_EVENT_DATATYPE TauUserEvent::GetMin(int tid)
{ 
  if (NumEvents[tid] != 0L)
  { 
    return MinValue[tid];
  }
  else
    return 0;
}

TAU_EVENT_DATATYPE TauUserEvent::GetMax(int tid)
{
  if (NumEvents[tid] != 0L)
  {
    return MaxValue[tid];
  }
  else
    return 0;
}

TAU_EVENT_DATATYPE TauUserEvent::GetSumValue(int tid)
{  
  if (NumEvents[tid] != 0L)
  {
    return SumValue[tid];
  }
  else
    return 0;
}

TAU_EVENT_DATATYPE TauUserEvent::GetMean(int tid)
{
  if (NumEvents[tid] != 0L) 
  {
    return (SumValue[tid]/NumEvents[tid]);
  } 
  else
    return 0;
}

double TauUserEvent::GetSumSqr(int tid)
{
  return (SumSqrValue[tid]);
}

long TauUserEvent::GetNumEvents(int tid)
{
  return NumEvents[tid];
}

// Get the event name
const char * TauUserEvent::GetEventName (void) const
{
  return EventName.c_str();
}

bool TauUserEvent::GetDisableMin(void)
{ 
  return DisableMin;
}

bool TauUserEvent::GetDisableMax(void)
{
  return DisableMax;
}

bool TauUserEvent::GetDisableMean(void)
{
  return DisableMean;
}

bool TauUserEvent::GetDisableStdDev(void)
{
  return DisableStdDev;
}

// Set Routines
void TauUserEvent::SetDisableMin(bool value)
{
  DisableMin = value;
  return;
}

void TauUserEvent::SetDisableMax(bool value)
{
  DisableMax = value;
  return;
}

void TauUserEvent::SetDisableMean(bool value)
{
  DisableMean = value;
  return;
}

void TauUserEvent::SetDisableStdDev(bool value)
{
  DisableStdDev = value;
  return;
}

TauUserEvent::~TauUserEvent(void)
{
  DEBUGPROFMSG(" DTOR CALLED for " << GetEventName() << endl;); 
}

void TauUserEvent::ReportStatistics(bool ForEachThread)
{
  TAU_EVENT_DATATYPE TotalNumEvents, TotalSumValue, Minima, Maxima;
  vector<TauUserEvent*>::iterator it;

    cout << "TAU Runtime Statistics" <<endl;
    cout << "*************************************************************" << endl;

  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++)
  {
    DEBUGPROFMSG("Thr "<< RtsLayer::myThread()<< " TauUserEvent "<< 
      (*it)->GetEventName() << "\n Min " << (*it)->GetMin() << "\n Max " <<
      (*it)->GetMax() << "\n Mean " << (*it)->GetMean() << "\n Sum Sqr " <<
      (*it)->GetSumSqr() << "\n NumEvents " << (*it)->GetNumEvents()<< endl;);
      
    TotalNumEvents = TotalSumValue = 0;

    for (int tid = 0; tid < TAU_MAX_THREADS; tid++)
    { 
      if ((*it)->GetNumEvents(tid) > 0)
      { // There were some events on this thread 
        TotalNumEvents += (*it)->GetNumEvents(tid); 
	TotalSumValue  += (*it)->GetSumValue(tid);

        if (!(*it)->GetDisableMin())
        { // Min is not disabled
	  // take the lesser of Minima and the min on that thread
	  if (tid > 0) 
	  { // more than one thread
	    Minima = (*it)->GetMin(tid) < Minima ? (*it)->GetMin(tid) : Minima;
	  } 
	  else 
	  { // this is the first thread. Initialize Minima to the min on it.
	    Minima = (*it)->GetMin(tid);
	  }
	} 

	if (!(*it)->GetDisableMax())
	{ // Max is not disabled
	  // take the maximum of Maxima and max on that thread
	  if (tid > 0)
	  { // more than one thread 
	    Maxima = (*it)->GetMax(tid) > Maxima ? (*it)->GetMax(tid) : Maxima;
	  } 
	  else
	  { // this is the first thread. Initialize Maxima to the max on it.
	    Maxima = (*it)->GetMax(tid);
	  }
	}   
	  

	if (ForEachThread) 
	{ // true, print statistics for this thread
	  cout <<  "n,c,t "<<RtsLayer::myNode() <<"," <<RtsLayer::myContext()
	       <<  "," << tid << " : Event : "<< (*it)->GetEventName() << endl
	       <<  " Number : " << (*it)->GetNumEvents(tid) <<endl
	       <<  " Min    : " << (*it)->GetMin(tid) << endl
	       <<  " Max    : " << (*it)->GetMax(tid) << endl
	       <<  " Mean   : " << (*it)->GetMean(tid) << endl
	       <<  " Sum    : " << (*it)->GetSumValue(tid) << endl << endl;
	}
	
      } // there were no events on this thread 
    } // for all threads 
    


    cout << "*************************************************************" << endl;
    cout << "Cumulative Statistics over all threads for Node: "
	 << RtsLayer::myNode() << " Context: " << RtsLayer::myContext() << endl;
    cout << "*************************************************************" << endl;
    cout << "Event Name     = " << (*it)->GetEventName() << endl;
	        
    cout << "Total Number   = " << TotalNumEvents << endl;
    cout << "Total Value    = " << TotalSumValue << endl; 
    cout << "Minimum Value  = " << Minima << endl;
    cout << "Maximum Value  = " << Maxima << endl;
    cout << "-------------------------------------------------------------" <<endl;
    cout << endl;
 
  } // For all events
}

/***************************************************************************
 * $RCSfile: UserEvent.cpp,v $   $Author: sameer $
 * $Revision: 1.7 $   $Date: 1999/06/20 17:34:41 $
 * POOMA_VERSION_ID: $Id: UserEvent.cpp,v 1.7 1999/06/20 17:34:41 sameer Exp $ 
 ***************************************************************************/
