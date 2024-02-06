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
// Include Files
//////////////////////////////////////////////////////////////////////
#ifdef TAU_CRAYXMT
#pragma mta instantiate used
#endif /* TAU_CRAYXMT */

#ifdef TAU_BEACON
#include <Profile/TauBeacon.h>
#endif /* TAU_BEACON */

#ifndef TAU_DISABLE_MARKERS
#define TAU_USE_EVENT_THRESHOLDS 1
#endif /* TAU_DISABLE_MARKERS */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>

#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
#include <sstream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */
#include <iostream.h>
#include <sstream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

#ifdef TAU_VAMPIRTRACE
#include <otf.h>
#include "Profile/TauVampirTrace.h"
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_EPILOG
#include "elg_trc.h"
#endif /* TAU_EPILOG */

#ifdef TAU_SCOREP
#include <scorep/SCOREP_Tau.h>
#endif

#include <Profile/Profiler.h>
#include <Profile/TauTrace.h>
#include <Profile/TauInit.h>
#include <Profile/UserEvent.h>
#include <tau_internal.h>

#include <Profile/TauEnv.h>
#include <Profile/TauPluginInternals.h>
#include <Profile/TauPin.h>

using namespace tau;

#ifdef PGI
template void AtomicEventDB::insert_aux(AtomicEventDB::iterator, TauUserEvent *const &);
template TauUserEvent** copy_backward(TauUserEvent**,TauUserEvent**,TauUserEvent**);
template TauUserEvent** uninitialized_copy(TauUserEvent**,TauUserEvent**,TauUserEvent**);
#endif // PGI

namespace tau
{

// Orders callpaths by comparing arrays of profiler addresses stored as longs.
// The first element of the array is the array length.
struct ContextEventMapCompare
{
  bool operator()(long const * l1, long const * l2) const
  {
    int i = 0;
    for (i=0; (i<=l1[0] && i<=l2[0]) ; i++) {
        //printf("%d: %p, %p\t", i, l1[i], l2[i]);
      if (l1[i] != l2[i]) {
      /*
          if (l1[i] < l2[i]) {
              printf("\nleft <  right\n"); fflush(stdout);
          } else {
              printf("\nleft >= right\n"); fflush(stdout);
          }
          */
          return l1[i] < l2[i];
      }
    }
    //printf("\nEqual!\n"); fflush(stdout);
    return false;
  }
};

struct ContextEventMap : public std::map<long *, TauUserEvent *, ContextEventMapCompare, TauSignalSafeAllocator<std::pair<long* const, TauUserEvent *> > >
{
  ~ContextEventMap() {
    Tau_destructor_trigger();
  }
};

AtomicEventDB & TheEventDB(void)
{
  static AtomicEventDB eventDB;
  return eventDB;
}

// Add User Event to the EventDB
void TauUserEvent::AddEventToDB()
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  RtsLayer::LockDB();
  TheEventDB().push_back(this);
  DEBUGPROFMSG("Successfully registered event " << GetName() << endl;);
  DEBUGPROFMSG("Size of eventDB is " << TheEventDB().size() <<endl);

  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.atomic_event_registration) {
    Tau_plugin_event_atomic_event_registration_data_t plugin_data;
    plugin_data.user_event_ptr = this;
    plugin_data.tid = Tau_get_thread();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_ATOMIC_EVENT_REGISTRATION, GetName().c_str(), &plugin_data);
  }

  /* Set user event id */
  eventId = RtsLayer::GenerateUniqueId();
#ifdef TAU_VAMPIRTRACE
#ifdef TAU_VAMPIRTRACE_5_12_API
  uint32_t gid = vt_def_counter_group(VT_CURRENT_THREAD, "TAU Events");
  eventId = vt_def_counter(VT_CURRENT_THREAD, GetName().c_str(), OTF_COUNTER_TYPE_ABS|OTF_COUNTER_SCOPE_NEXT, gid, "#");
#else
  uint32_t gid = vt_def_counter_group("TAU Events");
  eventId = vt_def_counter(GetName().c_str(), OTF_COUNTER_TYPE_ABS|OTF_COUNTER_SCOPE_NEXT, gid, "#");
#endif /* TAU_VAMPIRTRACE_5_12_API */
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_SCOREP
  SCOREP_Tau_MetricHandle handle = SCOREP_TAU_INIT_METRIC_HANDLE;

  SCOREP_Tau_InitMetric( &handle, GetName().c_str(), "units");
  eventId=handle;
#endif
  RtsLayer::UnLockDB();

}

// Used when the timestamp is 0, but we need it for the plugin processing
extern "C" x_uint64 TauTraceGetTimeStamp(int tid);

///////////////////////////////////////////////////////////
// TriggerEvent records the value of data in the UserEvent
///////////////////////////////////////////////////////////
void TauUserEvent::TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts)
{
  if (!Tau_global_getLightsOut()) {
#ifdef TAU_VAMPIRTRACE
    // *CWL* - x_uint64 (unsigned long long) violates the vampirtrace interface which expects
    //         unsigned long (previously uint64_t). The change from uint64_t to x_uint64 was
    //         previously made in response to problems with SCORE-P but was done as a global
    //         cut-and-paste which turned out to be unsafe. Since the use of time and cval
    //         are guarded for just vampirtrace, it should be safe to revert the changes
    //         for just vampirtrace.
    //
    //         Keep an eye on this. We should expect trouble as long as we cannot provide
    //         a proper abstraction for what constitutes a 64-bit unsigned integer in TAU.
    //         This should be a TODO item.
    // x_uint64 time;
    // x_uint64 cval;
    uint64_t time;
    uint64_t cval;
    int id = eventId;
    time = vt_pform_wtime();
    // cval = (x_uint64) data;
    cval = (uint64_t) data;
#ifdef TAU_VAMPIRTRACE_5_12_API
    vt_count(VT_CURRENT_THREAD, &time, id, 0);
#else
    vt_count(&time, id, 0);
#endif /* TAU_VAMPIRTRACE_5_12_API */
    time = vt_pform_wtime();
#ifdef TAU_VAMPIRTRACE_5_12_API
    vt_count(VT_CURRENT_THREAD, &time, id, cval);
#else
    vt_count(&time, id, cval);
#endif /* TAU_VAMPIRTRACE_5_12_API */
    time = vt_pform_wtime();
#ifdef TAU_VAMPIRTRACE_5_12_API
    vt_count(VT_CURRENT_THREAD, &time, id, 0);
#else
    vt_count(&time, id, 0);
#endif /* TAU_VAMPIRTRACE_5_12_API */

#else /* TAU_VAMPIRTRACE */
#ifndef TAU_EPILOG
    if (TauEnv_get_tracing()) {
#ifdef TAU_OTF2
      if(TauEnv_get_trace_format() == TAU_TRACE_FORMAT_OTF2) {
        TauTraceEvent(eventId, (x_uint64)data, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
      } else {
        TauTraceEvent(eventId, (x_uint64)0, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
        TauTraceEvent(eventId, (x_uint64)data, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
        TauTraceEvent(eventId, (x_uint64)0, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
      }
#else
      TauTraceEvent(eventId, (x_uint64)0, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
      TauTraceEvent(eventId, (x_uint64)data, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
      TauTraceEvent(eventId, (x_uint64)0, tid, (x_uint64)timestamp, use_ts, TAU_TRACE_EVENT_KIND_USEREVENT);
#endif /* TAU_OTF2 */
    }
#endif /* TAU_EPILOG */
    /* Timestamp is 0, and use_ts is 0, so tracing layer gets timestamp */
#endif /* TAU_VAMPIRTRACE */

#ifdef TAU_SCOREP
    SCOREP_Tau_TriggerMetricDouble( eventId, data );
#endif /*TAU_SCOREP*/

    TAU_ASSERT(this != NULL, "this == NULL in TauUserEvent::TriggerEvent!  Make sure all databases are appropriately locked.\n");

#ifdef PROFILING_ON
    Data & d = ThreadData(tid);

    // Record this value
    d.lastVal = data;

    // Increment number of events
    ++d.nEvents;

    // Compute relevant statistics for the data
    if (minEnabled && data < d.minVal) {
#ifdef TAU_USE_EVENT_THRESHOLDS
      if ((TauEnv_get_evt_threshold() > 0.0) && (d.nEvents > 1) && data <= (1.0 - TauEnv_get_evt_threshold()) * d.minVal) {
        if (name[0] != '[') { //re-entrant
#if !(defined(TAU_WINDOWS) || defined(TAU_NEC_SX))
          char ename[20 + name.length()];
#else /* TAU_WINDOWS || TAU_NEC_SX */
	  char ename[2048];
#endif /* TAU_WINDOWS || TAU_NEC_SX */
          snprintf(ename, sizeof(ename),  "[GROUP=MIN_MARKER] %s", name.c_str());
          if (name.find("=>") == std::string::npos) {
            //DEBUGPROFMSG("Marker: "<<ename<<"  d.minVal = "<<d.minVal<<" data = "<<data<<" d.nEvents = "<<d.nEvents<<endl;);
#ifdef TAU_SCOREP
            TAU_TRIGGER_EVENT(ename, data);
#else /* TAU_SCOREP */
            TAU_TRIGGER_CONTEXT_EVENT_THREAD(ename, data, tid);
#endif /* TAU_SCOREP */
#ifdef TAU_BEACON
            TauBeaconPublish(data,"counts", "MIN_MARKER", ename);
#endif /* TAU_BEACON */
          }
        }
      }
#endif /* TAU_USE_EVENT_THRESHOLDS */
      d.minVal = data;
    }

    if (maxEnabled && data > d.maxVal) {
#ifdef TAU_USE_EVENT_THRESHOLDS
      if ((TauEnv_get_evt_threshold() > 0.0) && (d.nEvents > 1) && data >= (1.0 + TauEnv_get_evt_threshold()) * d.maxVal) {
        if (name[0] != '[') { //re-entrant
#if !(defined(TAU_WINDOWS) || defined(TAU_NEC_SX))
          char ename[20 + name.length()];
#else /* TAU_WINDOWS || TAU_NEC_SX */
          char ename[2048];
#endif /* TAU_WINDOWS || TAU_NEC_SX */

          snprintf(ename, sizeof(ename),  "[GROUP=MAX_MARKER] %s", name.c_str());
          if (name.find("=>") == std::string::npos) {
            //DEBUGPROFMSG("Marker: "<<ename<<"  d.maxVal = "<<d.maxVal<<" data = "<<data<<" d.nEvents = "<<d.nEvents<<endl;);
#ifdef TAU_SCOREP
            TAU_TRIGGER_EVENT(ename, data);
#else /* TAU_SCOREP */
            TAU_TRIGGER_CONTEXT_EVENT_THREAD(ename, data, tid);
#endif /* TAU_SCOREP */
#ifdef TAU_BEACON
            TauBeaconPublish(data,"counts", "MAX_MARKER", ename);
#endif /* TAU_BEACON */
          }
        }
      }
#endif /* TAU_USE_EVENT_THRESHOLDS */
      d.maxVal = data;
    }

    if (meanEnabled) {
      d.sumVal += data;
    }
    if (stdDevEnabled) {
      d.sumSqrVal += data * data;
    }
#endif /* PROFILING_ON */
  /*Invoke plugins only if both plugin path and plugins are specified*/
    /* and only output the counter if it's not a context counter */
    if(Tau_plugins_enabled.atomic_event_trigger) {
      if ((name[0] != '[')
            && (name.find(" : ") == std::string::npos)
            && (name.find("=>") == std::string::npos)) {
        Tau_plugin_event_atomic_event_trigger_data_t plugin_data;
        plugin_data.counter_name = name.c_str();
        plugin_data.tid = tid;
        plugin_data.timestamp = (timestamp == 0 ? TauTraceGetTimeStamp(tid) : timestamp);
        plugin_data.value = (uint64_t)data;
        Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_ATOMIC_EVENT_TRIGGER, name.c_str(), &plugin_data);
      }
    }
  } // Tau_global_getLightsOut

}


void TauUserEvent::ReportStatistics(bool ForEachThread)
{
  TAU_EVENT_DATATYPE TotalNumEvents, TotalSumValue, Minima, Maxima;
  AtomicEventDB::iterator it;

  Maxima = Minima = 0;
  cout << "TAU Runtime Statistics" << endl;
  cout << "*************************************************************" << endl;

  for (it = TheEventDB().begin(); it != TheEventDB().end(); it++) {
    DEBUGPROFMSG("TauUserEvent "<<
        (*it)->GetName() << "\n Min " << (*it)->GetMin() << "\n Max " <<
        (*it)->GetMax() << "\n Mean " << (*it)->GetMean() << "\n Sum Sqr " <<
        (*it)->GetSumSqr() << "\n NumEvents " << (*it)->GetNumEvents()<< endl;);

    TotalNumEvents = TotalSumValue = 0;
    int maxThreads=(*it)->ThreadDataSize();
    for (int tid = 0; tid < maxThreads; tid++) { //TODO: DYNATHREAD
      if ((*it)->GetNumEvents(tid) > 0) {
        // There were some events on this thread
        TotalNumEvents += (*it)->GetNumEvents(tid);
        TotalSumValue += (*it)->GetSum(tid);

        if ((*it)->IsMinEnabled()) {
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

        if ((*it)->IsMaxEnabled()) {
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
          cout << "n,c,t " << RtsLayer::myNode() << "," << RtsLayer::myContext() << "," << tid
              << " : Event : " << (*it)->GetName() << endl
              << " Number : " << (*it)->GetNumEvents(tid) << endl
              << " Min    : " << (*it)->GetMin(tid) << endl
              << " Max    : " << (*it)->GetMax(tid) << endl
              << " Mean   : " << (*it)->GetMean(tid) << endl
              << " Sum    : " << (*it)->GetSum(tid) << endl << endl;
        }

      }    // there were no events on this thread
    }    // for all threads

    cout << "*************************************************************" << endl;
    cout << "Cumulative Statistics over all threads for Node: " << RtsLayer::myNode() << " Context: " << RtsLayer::myContext() << endl;
    cout << "*************************************************************" << endl;
    cout << "Event Name     = " << (*it)->GetName() << endl;
    cout << "Total Number   = " << TotalNumEvents << endl;
    cout << "Total Value    = " << TotalSumValue << endl;
    cout << "Minimum Value  = " << Minima << endl;
    cout << "Maximum Value  = " << Maxima << endl;
    cout << "-------------------------------------------------------------" << endl;
    cout << endl;
  }    // For all events
}


////////////////////////////////////////////////////////////////////////////
// Formulate Context Comparison Array, an array of addresses with size depth+2.
// The callpath depth is the 0th index, the user event goes is the last index
//////////////////////////////////////////////////////////////////////
void TauContextUserEvent::FormulateContextComparisonArray(Profiler * in_current, long * comparison)
{
  int tid = RtsLayer::myThread();
  int depth = Tau_get_current_stack_depth(tid);
  Profiler * current = in_current;
  if (depth > TAU_MAX_CALLPATH_DEPTH) {
      // oh, no...  super-deep callpath.  Warn the user and abort.  Bummer.
      fprintf(stderr, "ERROR! The callstack depth has exceeded a hard-coded limit in TAU.  Please reconfigure TAU with the option '-useropt=-DTAU_MAX_CALLPATH_DEPTH=X' where X is greater than %d\n", TAU_MAX_CALLPATH_DEPTH);
  }

  int i=1;
  // we might be processing virtual thread data, in which case tid is bogus
  // (because we are processing the asynchronous gpu data on a different thread)
  // so do some special handling
  if (current != NULL && depth == 0) {
    comparison[i] = Tau_convert_ptr_to_long(current->ThisFunction);
    i++;
  } else {
    // start writing to index 1, we fill in the depth after
    for(; current && depth; ++i) {
        comparison[i] = Tau_convert_ptr_to_long(current->ThisFunction);
        current = current->ParentProfiler;
        --depth;
    }
  }
  comparison[i] = Tau_convert_ptr_to_long(userEvent);
  comparison[0] = i; // set the depth

  return;
}

////////////////////////////////////////////////////////////////////////////
// Formulate Context Callpath name string
////////////////////////////////////////////////////////////////////////////
std::string TauContextUserEvent::FormulateContextNameString(Profiler * in_current)
{
  int tid = RtsLayer::myThread();
  Profiler * current = in_current;
  if (current) {
      //std::basic_stringstream<char, std::char_traits<char>, TauSignalSafeAllocator<char> > buff;
      std::stringstream buff;
      buff << userEvent->GetName();

      int depth = Tau_get_current_stack_depth(tid);
      FunctionInfo * fi;
      if (depth > 0) {
          Profiler ** path = new Profiler*[depth];
          // Reverse the callpath to avoid string copies
          int i=depth-1;
          for (; current && i >= 0; --i) {
            path[i] = current;
            current = current->ParentProfiler;
          }
          // Now we can construct the name string by appending rather than prepending
          buff  << " : ";
          for (++i; i < depth-1; ++i) {
            fi = path[i]->ThisFunction;
            buff << fi->GetName();
            if (strlen(fi->GetType()) > 0)
              buff << " " << fi->GetType();
            buff << " => ";
          }
          if (depth == 0) {
            fi = current->ThisFunction;
          } else {
            fi = path[i]->ThisFunction;
          }
          buff << fi->GetName();
          if (strlen(fi->GetType()) > 0)
            buff << " " << fi->GetType();

          //delete[] path;
      } else {
          fi = current->ThisFunction;
          buff << " : " << fi->GetName();
          if (strlen(fi->GetType()) > 0) {
            buff << " " << fi->GetType();
          }
      }
      // Return a new string object.
      // A smart STL implementation will not allocate a new buffer.
      return buff.str().c_str();
  } else {
      return "";
  }
}

////////////////////////////////////////////////////////////////////////////
// Trigger the context event
////////////////////////////////////////////////////////////////////////////
void TauContextUserEvent::TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts)
{
  static ContextEventMap contextMap;
  if (!Tau_global_getLightsOut()) {

    // Protect TAU from itself
    TauInternalFunctionGuard protects_this_function;

    if (contextEnabled) {
      Profiler * current = TauInternal_CurrentProfiler(tid);
      if (current) {
        //printf("**** Looking for : %p %s\n", current, FormulateContextNameString(current).c_str()); fflush(stdout);
        long comparison[TAU_MAX_CALLPATH_DEPTH] = {0};
        FormulateContextComparisonArray(current, comparison);
        //printf("Searching: %lu, %lu or %p (should be %p)\n", comparison[0], comparison[1], comparison[1], current);

        RtsLayer::LockDB();
        ContextEventMap::const_iterator it = contextMap.find(comparison);
#if 0
	bool cuda_ctx_seen = true;
	if (it != contextMap.end()) {
	  FunctionInfo* fi;
	  fi = current->ThisFunction;
	  std::istringstream userEventPrevSS((std::string)(it->second->GetName().c_str()));
	  std::string tok;
	  vector<std::string> userEventPrevVec;
	  while (std::getline(userEventPrevSS, tok, ':')) {
	    userEventPrevVec.push_back(tok);
	  }
	  userEventPrevVec[0].erase(userEventPrevVec[0].length()-1, userEventPrevVec[0].length());
	  userEventPrevVec[1].erase(0, 1);
	  if (!((std::string)(userEvent->GetName().c_str())).compare(userEventPrevVec[0])) {
	    if (((std::string)(fi->GetName())).compare(userEventPrevVec[1])) {
	      cuda_ctx_seen = false;
	    }
	  }
	}
        if (it == contextMap.end() || !cuda_ctx_seen) {
#else
        if (it == contextMap.end()) {
#endif
          contextEvent = new TauUserEvent(
              FormulateContextNameString(current).c_str(),
              userEvent->IsMonotonicallyIncreasing());
          // need to make a heap copy of our comparison array. Otherwise it gets
          // corrupted, because right now this is a stack variable.
          // It needs to be a stack variable so that searching each time we have
          // a counter doesn't eat up the whole memory map.
          int depth = comparison[0];
          int size = sizeof(long)*(depth+2);
          long * ary = (long*)malloc(size);
          int i;
          for (i = 0 ; i <= depth ; i++) {
              ary[i] = comparison[i];
          }
          contextMap[ary] = contextEvent;
        } else {
          contextEvent = it->second;
          //printf("**** FOUND **** %s \n", contextEvent->GetName().c_str()); fflush(stdout);
        }
        RtsLayer::UnLockDB();
        contextEvent->TriggerEvent(data, tid, timestamp, use_ts);
      } else {
        // do nothing - there is no context.
      }
    }
    // regardless of the context, trigger the UserEvent.
    userEvent->TriggerEvent(data, tid, timestamp, use_ts);
  }
}

} // END namespace tau


////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////
extern "C"
x_uint64 TauUserEvent_GetEventId(TauUserEvent const * evt)
{
  return evt->GetId();
}


/***************************************************************************
 * $RCSfile: UserEvent.cpp,v $   $Author: amorris $
 * $Revision: 1.46 $   $Date: 2010/05/07 22:16:23 $
 * POOMA_VERSION_ID: $Id: UserEvent.cpp,v 1.46 2010/05/07 22:16:23 amorris Exp $
 ***************************************************************************/
