/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997-2009	          			   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: UserEvent.cpp					  **
**	Description 	: TAU Profiling Package				  **
**	Contact		: tau-bugs@cs.uoregon.edu 		 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Note: The default behavior of this library is to calculate all the
// statistics (min, max, mean, stddev, etc.) If the user wishes to 
// override these settings, SetDisableXXX routines can be used to do so
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////

#ifdef TAU_CRAYXMT
#pragma mta instantiate used
#endif /* TAU_CRAYXMT */

//#define DEBUG_PROF

#include "Profile/Profiler.h"
#include <tau_internal.h>

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>

//#include <math.h>
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#ifdef TAU_VAMPIRTRACE
#include <otf.h>
#include "Profile/TauVampirTrace.h"
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_EPILOG
#include "elg_trc.h"
#endif /* TAU_EPILOG */

#include <TauTrace.h>
#include <TauInit.h>


#ifdef PGI
template void vector<TauUserEvent *>::insert_aux(vector<TauUserEvent *>::iterator, TauUserEvent *const &);
template TauUserEvent** copy_backward(TauUserEvent**,TauUserEvent**,TauUserEvent**);
template TauUserEvent** uninitialized_copy(TauUserEvent**,TauUserEvent**,TauUserEvent**);
#endif // PGI



class AtomicEventDB : public vector<TauUserEvent*> {
public :
  ~AtomicEventDB() {
    Tau_destructor_trigger();
  }
};


vector<TauUserEvent*>& TheEventDB(void) {
  static AtomicEventDB EventDB;

  static int flag = 1;
  if (flag) {
    flag = 0;
    Tau_init_initializeTAU();
  }

  return EventDB;
}

// Add User Event to the EventDB
void TauUserEvent::AddEventToDB() {
  RtsLayer::LockDB();
  TheEventDB().push_back(this);
  DEBUGPROFMSG("Successfully registered event " << GetEventName() << endl;);
  DEBUGPROFMSG("Size of eventDB is " << TheEventDB().size() <<endl);
  /* Set user event id */
  EventId = RtsLayer::GenerateUniqueId();
#ifdef TAU_VAMPIRTRACE
  uint32_t gid = vt_def_counter_group("TAU Events");
  EventId = vt_def_counter(GetEventName(), OTF_COUNTER_TYPE_ABS|OTF_COUNTER_SCOPE_NEXT, gid, "#");
#endif /* TAU_VAMPIRTRACE */
  RtsLayer::UnLockDB();
  return;
}

long TauUserEvent::GetEventId(void) {
  return EventId;
}

extern "C" long TauUserEvent_GetEventId(TauUserEvent *evt) {
  return evt->GetEventId();
}

TauUserEvent::TauUserEvent(const char * EName, bool increasing) {
  DEBUGPROFMSG("Inside ctor of TauUserEvent EName = "<< EName << endl;);

  EventName 	= EName;
  // Assign event name and then set the default values
  DisableMin 	= false; 	// Min 	    is calculated 
  DisableMax 	= false; 	// Max      is calculated 
  DisableMean 	= false; 	// Mean     is calculated 
  DisableStdDev = false; 	// StdDev   is calculated
  MonotonicallyIncreasing = increasing; // By default it is false 

  for(int i=0; i < TAU_MAX_THREADS; i++) {
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
TauUserEvent::TauUserEvent(TauUserEvent& X) {
  DEBUGPROFMSG("Inside copy ctor TauUserEvent::TauUserEvent()" << endl;);

  EventName 	= X.EventName;
  DisableMin	= X.DisableMin;
  DisableMax 	= X.DisableMax;
  DisableMean	= X.DisableMean;
  DisableStdDev = X.DisableStdDev;
  MonotonicallyIncreasing = X.MonotonicallyIncreasing;
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
TauUserEvent::TauUserEvent() {
  EventName 	= string("No Name");
  DisableMin 	= false; 	// Min 	    is calculated 
  DisableMax 	= false; 	// Max      is calculated 
  DisableMean 	= false; 	// Mean     is calculated 
  DisableStdDev = false; 	// StdDev   is calculated
  MonotonicallyIncreasing = false; // By default it does not have any constraints

  for (int i=0; i < TAU_MAX_THREADS; i++) {
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
TauUserEvent& TauUserEvent::operator= (const TauUserEvent& X) {

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
// GetMonotonicallyIncreasing
///////////////////////////////////////////////////////////
bool TauUserEvent::GetMonotonicallyIncreasing(void) {
  return MonotonicallyIncreasing; 
}

///////////////////////////////////////////////////////////
// SetMonotonicallyIncreasing
///////////////////////////////////////////////////////////
void TauUserEvent::SetMonotonicallyIncreasing(bool value) {
  MonotonicallyIncreasing = value; 
}

///////////////////////////////////////////////////////////
// TriggerEvent records the value of data in the UserEvent
///////////////////////////////////////////////////////////

void TauUserEvent::TriggerEvent(TAU_EVENT_DATATYPE data, int tid) { 
  TriggerEvent(data, tid, 0, 0);
}

void TauUserEvent::TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts) { 
#ifdef TAU_VAMPIRTRACE
  uint64_t time;
  uint64_t cval;
  int id = GetEventId();
  time = vt_pform_wtime();
  cval = (uint64_t) data;
  vt_count(&time, id, 0);
  time = vt_pform_wtime();
  vt_count(&time, id, cval);
  time = vt_pform_wtime();
  vt_count(&time, id, 0);
#else /* TAU_VAMPIRTRACE */
#ifndef TAU_EPILOG
  if (TauEnv_get_tracing()) {
    TauTraceEvent(GetEventId(), (x_uint64) 0, tid, (x_uint64) timestamp, use_ts); 
    TauTraceEvent(GetEventId(), (x_uint64) data, tid, (x_uint64) timestamp, use_ts); 
    TauTraceEvent(GetEventId(), (x_uint64) 0, tid, (x_uint64) timestamp, use_ts); 
  }
#endif /* TAU_EPILOG */
  /* Timestamp is 0, and use_ts is 0, so tracing layer gets timestamp */
#endif /* TAU_VAMPIRTRACE */

#ifdef PROFILING_ON
  // Record this value  
  LastValueRecorded[tid] = data;

  // Increment number of events
  NumEvents[tid] ++;

  // Compute relevant statistics for the data 
  if (!GetDisableMin()) {  
    // Min is not disabled
     if (NumEvents[tid] > 1) {
       MinValue[tid] = data < MinValue[tid] ? data : MinValue[tid];
     } else {
	MinValue[tid] = data;
     }
  }
  
  if (!GetDisableMax()) {  
    // Max is not disabled
     if (NumEvents[tid] > 1) {
       MaxValue[tid] = MaxValue[tid] < data ? data : MaxValue[tid];
     } else {
       MaxValue[tid] = data;
     }
  }

  if (!GetDisableMean()) {  
    // Mean is not disabled 
     SumValue[tid] += data; 
  }
     
  if (!GetDisableStdDev()) {  
    // Standard Deviation is not disabled
     SumSqrValue[tid] += data*data; 
  }

#endif /* PROFILING_ON */
  return; // completed calculating statistics for this event
}

// Return the data stored in the class
TAU_EVENT_DATATYPE TauUserEvent::GetMin(int tid) { 
  if (NumEvents[tid] != 0L) { 
    return MinValue[tid];
  } else {
    return 0;
  }
}

TAU_EVENT_DATATYPE TauUserEvent::GetMax(int tid) {
  if (NumEvents[tid] != 0L) {
    return MaxValue[tid];
  } else {
    return 0;
  }
}

TAU_EVENT_DATATYPE TauUserEvent::GetSumValue(int tid) {  
  if (NumEvents[tid] != 0L) {
    return SumValue[tid];
  } else {
    return 0;
  }
}

TAU_EVENT_DATATYPE TauUserEvent::GetMean(int tid) {
  if (NumEvents[tid] != 0L) {
    return (SumValue[tid]/NumEvents[tid]);
  } else {
    return 0;
  }
}

double TauUserEvent::GetSumSqr(int tid) {
  return (SumSqrValue[tid]);
}

long TauUserEvent::GetNumEvents(int tid) {
  return NumEvents[tid];
}

// Get the event name
const char *TauUserEvent::GetEventName (void) const {
  return EventName.c_str();
}

// Set the event name
void TauUserEvent::SetEventName (const char *newname) {
  EventName = newname;
}

// Set the event name
void TauUserEvent::SetEventName (string newname) {
  EventName = newname;
}

bool TauUserEvent::GetDisableMin(void) { 
  return DisableMin;
}

bool TauUserEvent::GetDisableMax(void) {
  return DisableMax;
}

bool TauUserEvent::GetDisableMean(void) {
  return DisableMean;
}

bool TauUserEvent::GetDisableStdDev(void) {
  return DisableStdDev;
}

// Set Routines
void TauUserEvent::SetDisableMin(bool value) {
  DisableMin = value;
  return;
}

void TauUserEvent::SetDisableMax(bool value) {
  DisableMax = value;
  return;
}

void TauUserEvent::SetDisableMean(bool value) {
  DisableMean = value;
  return;
}

void TauUserEvent::SetDisableStdDev(bool value) {
  DisableStdDev = value;
  return;
}

TauUserEvent::~TauUserEvent(void) {
  DEBUGPROFMSG(" DTOR CALLED for " << GetEventName() << endl;); 
  Tau_destructor_trigger();
}

void TauUserEvent::ReportStatistics(bool ForEachThread) {
  TAU_EVENT_DATATYPE TotalNumEvents, TotalSumValue, Minima, Maxima ;
  vector<TauUserEvent*>::iterator it;

  Maxima = Minima = 0;
  cout << "TAU Runtime Statistics" <<endl;
  cout << "*************************************************************" << endl;

  for(it  = TheEventDB().begin(); it != TheEventDB().end(); it++) {
    DEBUGPROFMSG("TauUserEvent "<< 
      (*it)->GetEventName() << "\n Min " << (*it)->GetMin() << "\n Max " <<
      (*it)->GetMax() << "\n Mean " << (*it)->GetMean() << "\n Sum Sqr " <<
      (*it)->GetSumSqr() << "\n NumEvents " << (*it)->GetNumEvents()<< endl;);
      
    TotalNumEvents = TotalSumValue = 0;

    for (int tid = 0; tid < TAU_MAX_THREADS; tid++) { 
      if ((*it)->GetNumEvents(tid) > 0) { 
	// There were some events on this thread 
        TotalNumEvents += (*it)->GetNumEvents(tid); 
	TotalSumValue  += (*it)->GetSumValue(tid);

        if (!(*it)->GetDisableMin()) { 
	  // Min is not disabled
	  // take the lesser of Minima and the min on that thread
	  if (tid > 0) { 
	    // more than one thread
	    Minima = (*it)->GetMin(tid) < Minima ? (*it)->GetMin(tid) : Minima;
	  } else { 
	    // this is the first thread. Initialize Minima to the min on it.
	    Minima = (*it)->GetMin(tid);
	  }
	} 

	if (!(*it)->GetDisableMax()) {
	  // Max is not disabled
	  // take the maximum of Maxima and max on that thread
	  if (tid > 0) {
	    // more than one thread 
	    Maxima = (*it)->GetMax(tid) > Maxima ? (*it)->GetMax(tid) : Maxima;
	  } else { 
	    // this is the first thread. Initialize Maxima to the max on it.
	    Maxima = (*it)->GetMax(tid);
	  }
	}   
	  

	if (ForEachThread) {
	  // true, print statistics for this thread
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

////////////////////////////////////////////////////////////////////////////
// We now implement support for user defined events that link with callpaths
////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////
// The datatypes and routines for maintaining a context map
////////////////////////////////////////////////////////////////////////////
#define TAU_CONTEXT_MAP_TYPE long *, TauUserEvent *, TaultUserEventLong

/////////////////////////////////////////////////////////////////////////
/* The comparison function for callpath requires the TaultUserEventLong struct
 * to be defined. The operator() method in this struct compares two callpaths.
 * Since it only compares two arrays of longs (containing addresses), we can
 * look at the callpath depth as the first index in the two arrays and see if
 * they're equal. If they two arrays have the same depth, then we iterate
 * through the array and compare each array element till the end */
/////////////////////////////////////////////////////////////////////////
struct TaultUserEventLong {
  bool operator() (const long *l1, const long *l2) const {
   int i;
   /* first check 0th index (size) */
   if (l1[0] != l2[0]) return (l1[0] < l2[0]);
   /* they're equal, see the size and iterate */
   for (i = 1; i < l1[0] ; i++) {
     if (l1[i] != l2[i]) return l1[i] < l2[i];
   }
   return (l1[i] < l2[i]);
 }
};



class ContextEventMap : public map<TAU_CONTEXT_MAP_TYPE > {
public :
  ~ContextEventMap() {
    Tau_destructor_trigger();
  }
};



/////////////////////////////////////////////////////////////////////////
// We use one global map to store the callpath information
/////////////////////////////////////////////////////////////////////////
ContextEventMap& TheContextMap(void) { 
  // to avoid initialization problems of non-local static variables
  static ContextEventMap contextmap;
  return contextmap;
}

////////////////////////////////////////////////////////////////////////////
// Formulate Context Comparison Array
//////////////////////////////////////////////////////////////////////
long* TauFormulateContextComparisonArray(Profiler *p, TauUserEvent *uevent) {
  int depth = TauEnv_get_callpath_depth();
  /* Create a long array with size depth+2. We need to put the depth
   * in it as the 0th index, the user event goes as the tail element */

  long *ary = new long [depth+2];
  if (!ary) return NULL;
  /* initialize the array */
  for (int j = 0; j < depth+2; j++) {
    ary[j] = 0L;
  }

  Profiler *current = p; /* argument */
  int index = 1; /* start writing to index 1, we fill in the depth after */

  while (current != NULL && depth != 0) {
    ary[index++] = (long) current->ThisFunction;
    depth--;
    current = current->ParentProfiler;
  }
  
  ary[index++] = (long) uevent;
  ary[0] = index-1; /* set the depth */
  return ary;
}

////////////////////////////////////////////////////////////////////////////
// Formulate Context Callpath name string
////////////////////////////////////////////////////////////////////////////
string * TauFormulateContextNameString(Profiler *p) {
  int depth = TauEnv_get_callpath_depth();
  Profiler *current = p;
  string delimiter(" => ");
  string *name = new string("");

  while (current != NULL && depth != 0) {
    if (current != p) {
      *name = current->ThisFunction->GetName() + string(" ") +
	current->ThisFunction->GetType() + delimiter + *name;
    } else {
      *name = current->ThisFunction->GetName() + string (" ") +
	current->ThisFunction->GetType();
    }
    current = current->ParentProfiler;
    depth --;
  }
  return name;
}



////////////////////////////////////////////////////////////////////////////
// Ctor for TauContextUserEvent 
////////////////////////////////////////////////////////////////////////////
TauContextUserEvent::TauContextUserEvent(const char *EName, bool MonoIncr) {
  /* create the event */
  uevent = new TauUserEvent(EName, MonoIncr);
  DisableContext = false; /* context tracking is enabled by default */

  int depth = TauEnv_get_callpath_depth();
  if (depth == 0) {
    DisableContext = true;
  }

  MonotonicallyIncreasing = MonoIncr;
}

////////////////////////////////////////////////////////////////////////////
// Dtor for TauContextUserEvent 
////////////////////////////////////////////////////////////////////////////
TauContextUserEvent::~TauContextUserEvent() {
  delete uevent; 
  delete contextevent; 
}

////////////////////////////////////////////////////////////////////////////
// SetDisableContext for TauContextUserEvent 
////////////////////////////////////////////////////////////////////////////
void TauContextUserEvent::SetDisableContext(bool value) {
  DisableContext = value;
}

////////////////////////////////////////////////////////////////////////////
// GetEventName() returns the name of the context event 
////////////////////////////////////////////////////////////////////////////
const char * TauContextUserEvent::GetEventName(void) {
  if (contextevent) {
    return contextevent->GetEventName();
  } else {
    return (const char *)NULL;
  }
}
////////////////////////////////////////////////////////////////////////////
// Trigger the context event
////////////////////////////////////////////////////////////////////////////

void TauContextUserEvent::TriggerEvent( TAU_EVENT_DATATYPE data, int tid) {
  TriggerEvent(data, tid, 0, 0);
}

void TauContextUserEvent::TriggerEvent( TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts) {
  Tau_global_incr_insideTAU();

  if (!DisableContext) {
    long *comparison = 0;
    TauUserEvent *ue;
    /* context tracking is enabled */
    Profiler *current = TauInternal_CurrentProfiler(tid);
    comparison = TauFormulateContextComparisonArray(current, uevent); 

    map<TAU_CONTEXT_MAP_TYPE>::iterator it = TheContextMap().find(comparison);

    if (it == TheContextMap().end()) {
      RtsLayer::LockEnv();
      it = TheContextMap().find(comparison);
      if (it == TheContextMap().end()) {
	string *ctxname = TauFormulateContextNameString(current);
	string contextname(uevent->EventName  + " : " + *ctxname);
	
	ue = new TauUserEvent((const char *)(contextname.c_str()), MonotonicallyIncreasing);

	TheContextMap().insert(map<TAU_CONTEXT_MAP_TYPE>::value_type(comparison, ue));
	
	ue->ctxevt = this; /* store the current object in the user event */
	delete ctxname; /* free up the string memory */
      } else {
	/* found it! Get the user defined event from the map */
	ue = (*it).second;
	delete[] comparison; // free up memory when name is found
      }
      RtsLayer::UnLockEnv();
    } else {
      /* found it! Get the user defined event from the map */
      ue = (*it).second;
      delete[] comparison; // free up memory when name is found
    }

    /* Now we trigger this event */
    if (ue) { 
      /* it is not null, trigger it */
      contextevent = ue;
      /* store this context event, so we can get its name */
      contextevent->TriggerEvent(data, tid, timestamp, use_ts);
    }
  }
  uevent->TriggerEvent(data, tid, timestamp, use_ts);

  Tau_global_decr_insideTAU();
}

/***************************************************************************
 * $RCSfile: UserEvent.cpp,v $   $Author: amorris $
 * $Revision: 1.46 $   $Date: 2010/05/07 22:16:23 $
 * POOMA_VERSION_ID: $Id: UserEvent.cpp,v 1.46 2010/05/07 22:16:23 amorris Exp $ 
 ***************************************************************************/
