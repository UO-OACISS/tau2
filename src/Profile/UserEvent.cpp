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

#define DEBUG_PROF

#include "Profile/Profiler.h"

// UserEvent structure definition
#include "Profile/UserEvent.h"

#include <math.h>
#include <iostream.h>

#ifndef TAU_STDCXXLIB // It'd be better to use #ifdef SGICC 
//template void vector<TauUserEvent *,alloc>::insert_aux(vector<TauUserEvent *,alloc>::iterator, TauUserEvent *const &);
#endif // TAU_STDCXXLIB

vector<TauUserEvent*>& TheEventDB(int threadid)
{
  static vector<TauUserEvent*> EventDB[TAU_MAX_THREADS];

  return EventDB[threadid];
}

// Add User Event to the EventDB
void TauUserEvent::AddEventToDB()
{
  TheEventDB().push_back(this);
  DEBUGPROFMSG("Successfully registered event " << GetEventName() << endl;);
  DEBUGPROFMSG("Size of eventDB is " << TheEventDB().size() <<endl);
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

  LastValueRecorded = 0;  	// null to start with
  NumEvents = 0L; 		// initialize
  MinValue  = 9999999;  	// Least -ve value? limits.h
  MaxValue  = -9999999;		// Greatest  +ve value?    "
  SumSqrValue  = 0;		// initialize
  SumValue     = 0; 		// initialize

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
  LastValueRecorded = X.LastValueRecorded;
  NumEvents	= X.NumEvents;
  MinValue	= X.MinValue;
  MaxValue	= X.MaxValue;
  SumSqrValue	= X.SumSqrValue;
  SumValue	= X.SumValue;

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

  LastValueRecorded = 0;  	// null to start with
  NumEvents = 0L; 		// initialize
  MinValue  = 9999999;  	// Least -ve value? limits.h
  MaxValue  = -9999999;		// Greatest  +ve value?    "
  SumSqrValue  = 0;		// initialize
  SumValue     = 0; 		// initialize

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
  LastValueRecorded = X.LastValueRecorded;
  NumEvents	= X.NumEvents;
  MinValue	= X.MinValue;
  MaxValue	= X.MaxValue;
  SumSqrValue	= X.SumSqrValue;
  SumValue	= X.SumValue;

  return *this;
}

///////////////////////////////////////////////////////////
// TriggerEvent records the value of data in the UserEvent
///////////////////////////////////////////////////////////

void TauUserEvent::TriggerEvent(TAU_EVENT_DATATYPE data)
{ 
  // Record this value  
  LastValueRecorded = data;

  // Increment number of events
  NumEvents ++;

  // Compute relevant statistics for the data 
  if (!GetDisableMin()) 
  {  // Min is not disabled
     if (NumEvents > 1) {
     	MinValue = data < MinValue ? data : MinValue;
     } else
	MinValue = data;
  }
  
  if (!GetDisableMax())
  {  // Max is not disabled
     if (NumEvents > 1) {
       MaxValue = MaxValue < data ? data : MaxValue;
     } else
       MaxValue = data;
  }

  if (!GetDisableMean())
  {  // Mean is not disabled 
     SumValue += data; 
  }
     
  if (!GetDisableStdDev())
  {  // Standard Deviation is not disabled
     SumSqrValue += data*data; 
  }

  return; // completed calculating statistics for this event
}

// Return the data stored in the class
TAU_EVENT_DATATYPE TauUserEvent::GetMin(void)
{ 
  return MinValue;
}

TAU_EVENT_DATATYPE TauUserEvent::GetMax(void)
{
  return MaxValue;
}

TAU_EVENT_DATATYPE TauUserEvent::GetMean(void)
{
  return (SumValue/NumEvents);
}

double TauUserEvent::GetSumSqr(void)
{
  return (SumSqrValue);
}

double TauUserEvent::GetStdDev(void)
{
  double stddev = 0;
  // Formula to calculate StdDev is 
  // sigma = sqrt( (sumvalue^2/n) - (mean^2)) 

  stddev = sqrt(fabs( (SumSqrValue/NumEvents) - (SumValue/NumEvents) 
	        * (SumValue/NumEvents) ) ); 

  return stddev;
}

// Get Routines
long TauUserEvent::GetNumEvents(void)
{
  return NumEvents;
}

// Get the event name
const char * TauUserEvent::GetEventName(void)
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
 
}

void TauUserEvent::StoreData(void)
{

  vector<TauUserEvent*>::iterator it;
  for(it  = TheEventDB().begin(); 
      it != TheEventDB().end(); it++)
  {
    cout << "Thr "<< RtsLayer::myThread()<< " TauUserEvent "<< 
      (*it)->GetEventName() << "\n Min " << (*it)->GetMin() << "\n Max " <<
      (*it)->GetMax() << "\n Mean " << (*it)->GetMean() << "\n Std Dev " <<
      (*it)->GetStdDev() << "\n NumEvents " << (*it)->GetNumEvents() << endl;
  }
}

/***************************************************************************
 * $RCSfile: UserEvent.cpp,v $   $Author: sameer $
 * $Revision: 1.1 $   $Date: 1998/04/24 00:01:06 $
 * POOMA_VERSION_ID: $Id: UserEvent.cpp,v 1.1 1998/04/24 00:01:06 sameer Exp $ 
 ***************************************************************************/
