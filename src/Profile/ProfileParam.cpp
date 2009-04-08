/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: ProfileParam.cpp				  **
**	Description 	: TAU Profiling Package				  **
**	Contact		: tau-team@cs.uoregon.edu 		 	  **
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

//#define DEBUG_PROF

#include "Profile/Profiler.h"

#include "tau_internal.h"


#include <stdio.h>
#include <fcntl.h>

//#include <math.h>
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

////////////////////////////////////////////////////////////////////////////
// The datatypes and routines for maintaining a context map
////////////////////////////////////////////////////////////////////////////
#define TAU_PROFILE_PARAM_TYPE long *, FunctionInfo *, TaultProfileParamLong

/////////////////////////////////////////////////////////////////////////
/* The comparison function for callpath requires the TaultUserEventLong struct
 * to be defined. The operator() method in this struct compares two callpaths.
 * Since it only compares two arrays of longs (containing addresses), we can
 * look at the callpath depth as the first index in the two arrays and see if
 * they're equal. If they two arrays have the same depth, then we iterate
 * through the array and compare each array element till the end */
/////////////////////////////////////////////////////////////////////////
struct TaultProfileParamLong {
  bool operator() (const long *l1, const long *l2) const {
   int i;
   /* first check 0th index (size) */
   if (l1[0] != l2[0]) return (l1[0] < l2[0]);
   /* they're equal, see the size and iterate */
   for (i = 0; i < l1[0] ; i++)
   {
     if (l1[i] != l2[i]) return l1[i] < l2[i];
   }
   return (l1[i] < l2[i]);
 }
};


/////////////////////////////////////////////////////////////////////////
// We use one global map to store the callpath information
/////////////////////////////////////////////////////////////////////////
map<TAU_PROFILE_PARAM_TYPE >& TheTimerProfileParamMap(void) { 
  // to avoid initialization problems of non-local static variables
  static map<TAU_PROFILE_PARAM_TYPE > timerappdatamap;

  return timerappdatamap;
}

long * TauCreateProfileParamArray(long FuncId, long key) {
  int depth = 3; 
  long *retary = new long[depth+1]; 
  retary[0] = depth; /* encode the depth first */
  retary[1] = FuncId; /* the id of the current timer */
  retary[2] = key;   /* data */
  return retary;
}

FunctionInfo * TauGetProfileParamFI(int tid, long key, string& keyname) {
  /* Get the FunctionInfo Object of the current Profiler */
  Profiler *current = TauInternal_CurrentProfiler(tid);
  if (!current) return NULL; /* not in a valid profiler */
  FunctionInfo *f = current->ThisFunction; 
  if (!f) return NULL;  /* proceed if we are in a valid function */
  
  /* we have a timer definition. We need to examine the key and see if
   * it has appeared before. If not, we need to create a new functionInfo 
   * and a mapping between the key and the newly created functionInfo */
  
  long *ary = TauCreateProfileParamArray((long) f, key);
  
  /* We've set the key */
  map<TAU_PROFILE_PARAM_TYPE >::iterator it = TheTimerProfileParamMap().find(ary);
  
  if (it == TheTimerProfileParamMap().end()) {
    /* Couldn't find it */
    char keystr[256]; 
    sprintf(keystr, "%ld", key); 
    
    string name ( f->GetName() + string(" ") + f->GetType()+ string(" [ <")
		  +keyname+ string("> = <")+ keystr + string("> ]")); 
    DEBUGPROFMSG("Name created = "<<name<<endl;);
    string grname = string("TAU_PARAM | ") + RtsLayer::PrimaryGroup(f->GetAllGroups());
    
    FunctionInfo *fnew = new FunctionInfo(name, " ", 
					  f->GetProfileGroup(),
					  (const char *)grname.c_str(), true, tid); 
    TheTimerProfileParamMap().insert(map<TAU_PROFILE_PARAM_TYPE >::value_type(ary, fnew)); /* Add it to the map */
    return fnew; 
  } else { 
    /* found it. (*it).second refers to the functionInfo object corresponding
       to our particular instance */
    DEBUGPROFMSG("Found name = "<<(*it).second->GetName()<<endl;);
    return (*it).second; 
  }
}
	
void TauProfiler_AddProfileParamData(long key, const char *keyname) {
  string keystring(keyname);
  int tid = RtsLayer::myThread();
  FunctionInfo *f = TauGetProfileParamFI(tid, key, keystring);
  Profiler *current = TauInternal_CurrentProfiler(tid);
  if (!current) return; 
  current->ProfileParamFunction = f; 

  /* set add incl flag for this function info object */
  if (f->GetAlreadyOnStack(tid) == false) { /* is it on the callstack? */
    current->AddInclProfileParamFlag = true; 
    f->SetAlreadyOnStack(true, tid); // it is on callstack now 
  } else {
    current->AddInclProfileParamFlag = false; // no need to add incl time
  }
 
  return;  
}
	

void Profiler::ProfileParamStop(double* TotalTime, int tid) {
  if (ProfileParamFunction) {
    DEBUGPROFMSG("Inside ProfileParamStop "<<ThisFunction->GetName()<<endl;);
    if (AddInclProfileParamFlag == true) { // The first time it came on call stack
      ProfileParamFunction->SetAlreadyOnStack(false, tid); // while exiting

      DEBUGPROFMSG("nct "<< RtsLayer::myNode()  << ","
       << RtsLayer::myContext() << "," << tid  << " "
       << "ProfileParamStop: After SetAlreadyOnStack Going for AddInclTime" <<endl; );

      // And its ok to add both excl and incl times
      ProfileParamFunction->IncrNumCalls(tid);
      ProfileParamFunction->AddInclTime(TotalTime, tid);
    }

    ProfileParamFunction->AddExclTime(TotalTime, tid);  
  }
}
  
/***************************************************************************
 * $RCSfile: ProfileParam.cpp,v $   $Author: amorris $
 * $Revision: 1.7 $   $Date: 2009/04/08 20:30:12 $
 * TAU_VERSION_ID: $Id: ProfileParam.cpp,v 1.7 2009/04/08 20:30:12 amorris Exp $ 
 ***************************************************************************/
