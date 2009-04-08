//////////////////////////////////////////////////////////////////////
//
// RENCI STFF Layer
//
// This layer is used for capturing signtaures using the RENCI
// Scalable Time-series Filter Framework. Data collected by the
// TAU profiler can be passed to this layer, and signatures are
// made per function/callsite/counter.
// 
// Scalable Traces compress runtime trace data using least 
// squares and other fitting methods.
// 
// For more information about STFF contact:
//
// Todd Gamblin tgamblin@cs.unc.edu
// Rob Fowler   rjf@renci.org
//
// Or, visit http://www.renci.org
// 
//////////////////////////////////////////////////////////////////////

#include "Profile/RenciSTFF.h"

#include "stff-config.h"
using namespace renci_stff;

#include <Profile/Profiler.h>

#define CHECK_INIT if (!inited && !init()) return;

#ifdef DEBUG_PROF
#include <iostream>
#include <cstdio>
#define DEBUGPRINTF(msg, val) { printf(msg, val); }
#else //DEBUG_PROF
#define DEBUGPRINTF(msg, val)
#endif //DEBUG_PROF

#include <sstream>
using namespace std;


// ------------------------------------------------------------------
// Static Storage
// ------------------------------------------------------------------
bool RenciSTFF::inited = false;
const char **RenciSTFF::counterList;	
int RenciSTFF::numCounters;
std::vector<ApplicationSignature*> RenciSTFF::signatures;

extern "C" int PMPI_Initialized(int *inited);

// function object class to delete elements in a container
struct AppSigDelete {
  void operator() (ApplicationSignature*& appSig) const {
    if(appSig) {
      delete appSig; appSig = 0;
    }
  }
};

// funtion object class to stop signature application
struct AppSigStop {
  void operator() ( ApplicationSignature* appSig) const {
    if(appSig) {
      appSig->stopBuildingSignature(true);
    }
  }
};


/**
 * Use this to lock and unlock the rts db during initialization.
 */
struct Lock {
  Lock() { RtsLayer::LockDB(); }
  ~Lock() { RtsLayer::UnLockDB(); }
};


string RenciSTFF::handle_to_string(int handle) {
  return counterList[handle];
}

/** CounterList for when we don't have multiple counters. */
static const char *INCL_TIME = "InclusiveTime";


// Initializes the signature layer
bool RenciSTFF::init() {
#ifdef STFF_USE_MPI
  //wait until mpi is inited to do anything, if STFF expects MPI.
  int mpiInited;
  PMPI_Initialized(&mpiInited);
  if (!mpiInited) return false;
#endif //STFF_USE_MPI

  Lock rtsLock;
  if (inited) return true; // make sure we weren't inited while waiting for lock

  DEBUGPROFMSG("Initializing RENCI STFF" << endl);

  //Set up signatures for all the counters in use
  MultipleCounterLayer::theCounterList(&counterList, &numCounters);

  SigId::set_handle_to_string(handle_to_string);

  DEBUGPROFMSG("Done with RenciSTFF::init()" << endl);
  inited = true;
  return true;
}


ApplicationSignature *RenciSTFF::createSignature(
						 const Point &point, const FunctionInfo *info, int tid, int metricId
						 ) {
  DEBUGPROFMSG("<NewSig: ");
  //make a new signature
  SigId id(metricId, static_cast<const void*>(info));
  ApplicationSignature *signature = new ApplicationSignature(point, id);

  //Associate metric metadata with the signature
  MetaData* theMetaData = signature->getMetaData();
  const char * metricName = counterList[id.getHandle()];
  if (metricName) theMetaData->put(MetaData::PERFMETRIC, metricName);
    
  //Associate some extra metadata with the signature
  ostringstream tidStream;
  tidStream << tid;
  theMetaData->put(MetaData::THREAD_ID, tidStream.str());
  theMetaData->put(MetaData::CALL_PATH, info->GetName());

  DEBUGPROFMSG(signature << ">");
  signatures.push_back(signature);
  return signature;
}


void RenciSTFF::recordValues(FunctionInfo *function, 
			     double timestamp, 
			     const double *metricValue, int tid) {
  CHECK_INIT;
  if (!function) return;
  
  DEBUGPRINTF("    t%d", tid); 
  DEBUGPRINTF("\t%20.0lf", timestamp);
  ApplicationSignature **signatures = function->GetSignature(tid);
  for (int i=0; i < numCounters; i++) {
    Point sample(timestamp, metricValue[i]);
    if (!signatures[i]) {
      signatures[i] = createSignature(sample, function, tid, i);
    } else {
      signatures[i]->addObservation(sample);
    }
    DEBUGPRINTF(", %20.0lf", metricValue[i]);
  }
  DEBUGPROFMSG(endl);
}


void RenciSTFF::cleanup() {
  DEBUGPROFMSG("Cleaning up after signature layer." << endl);
  for_each(signatures.begin(), signatures.end(), AppSigStop());
  for_each(signatures.begin(), signatures.end(), AppSigDelete());
  signatures.clear();
  
  delete [] counterList;
  
  inited = false;
}
