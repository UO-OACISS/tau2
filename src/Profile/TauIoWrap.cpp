/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 2010  						   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/****************************************************************************
 **	File 		: TauIoWrap.cpp  				   **
 **	Description 	: TAU Profiling Package				   **
 **	Contact		: tau-bugs@cs.uoregon.edu               	   **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
 **                                                                         **
 **      Description     : io wrapper                                       **
 **                                                                         **
 ****************************************************************************/
#include <TAU.h>
#include <Profile/TauInit.h>
#include <Profile/TauIoWrap.h>
#include <Profile/TauPin.h>
#include <stdio.h>
#include <vector>

using namespace std;
using namespace tau;



#define dprintf TAU_VERBOSE

const char * iowrap_event_names[NUM_EVENTS] = {
    "Write Bandwidth (MB/s)",
    "Bytes Written",
    "Read Bandwidth (MB/s)",
    "Bytes Read"
};

void * global_write_bandwidth = 0;
void * global_read_bandwidth = 0;
void * global_bytes_written = 0;
void * global_bytes_read = 0;

/*********************************************************************
 * IOvector subclasses vector to provide custom constructor/destructor 
 * to enable/disable wrapping
 ********************************************************************/
static int lightsOut = 0;
struct IOvector : public vector<AtomicEventDB>
{
  IOvector(int farg) : vector<AtomicEventDB>(farg) {
    lightsOut = 0;
  }
  ~IOvector() {
    lightsOut = 1;
  }
};

/*********************************************************************
 * Return the set of events, must be done as a static returned because
 * the wrapped routines may be called during the initializers of other
 * shared libraries and may be before our global_ctors_aux() call
 ********************************************************************/
static IOvector & TheIoWrapEvents()
{
  static IOvector iowrap_events(4);
  return iowrap_events;
}

/*********************************************************************
 * return whether we should pass through and not track the IO
 ********************************************************************/
extern "C"
int Tau_iowrap_checkPassThrough()
{
  return lightsOut || Tau_init_initializingTAU() || !Tau_init_check_initialized() || (Tau_global_get_insideTAU() > 0);
}

/*********************************************************************
 * register the different events (read/write/etc) for a file descriptor
 ********************************************************************/
extern "C"
void Tau_iowrap_registerEvents(int fid, const char *pathname)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  RtsLayer::LockDB();

  IOvector & iowrap_events = TheIoWrapEvents();
  dprintf("Asked to register %d with %s (current size=%d)\n", fid, pathname, TheIoWrapEvents()[0].size());
  fid++;    // skip the "unknown" descriptor

  for (int i = 0; i < NUM_EVENTS; i++) {
    TauUserEvent *unknown_ptr = 0;
    if (iowrap_events[i].size() > 0) {
      unknown_ptr = iowrap_events[i][0];
    }
    while ((int)(iowrap_events[i].size()) <= fid) {
      iowrap_events[i].push_back(unknown_ptr);
      if ((int)(iowrap_events[i].size() - 1) != fid) {
        dprintf("Registering %d with unknown\n", iowrap_events[i].size() - 2);
      }
    }
    void *event = 0;
    char ename[4096];
    sprintf(ename,"%s <file=%s>", iowrap_event_names[i], pathname);
    Tau_pure_context_userevent(&event, ename);
    iowrap_events[i][fid] = (TauUserEvent*)event;
  }
  dprintf("Registering %d with %s\n", fid - 1, pathname);
  RtsLayer::UnLockDB();
}

/*********************************************************************
 * unregister events for a file descriptor
 ********************************************************************/
extern "C" void Tau_iowrap_unregisterEvents(unsigned int fid)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  RtsLayer::LockDB();

  IOvector & iowrap_events = TheIoWrapEvents();
  dprintf("Un-registering %d\n", fid);
  fid++;    // skip the "unknown" descriptor

  for (int i = 0; i < NUM_EVENTS; i++) {
    TauUserEvent *unknown_ptr = 0;
    if (iowrap_events[i].size() >= 1) {
      unknown_ptr = iowrap_events[i][0];
    }
    while (iowrap_events[i].size() <= fid) {
      iowrap_events[i].push_back(unknown_ptr);
    }
    iowrap_events[i][fid] = unknown_ptr;
  }
  RtsLayer::UnLockDB();
}

/*********************************************************************
 * Tau_iowrap_dupEvents takes care of the associating the events with the 
 * new file descriptor obtained by using dup/dup2 calls.
 ********************************************************************/
extern "C" void Tau_iowrap_dupEvents(unsigned int oldfid, unsigned int newfid)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;

  RtsLayer::LockDB();

  IOvector & iowrap_events = TheIoWrapEvents();
  dprintf("dup (old=%d, new=%d)\n", oldfid, newfid);
  oldfid++;    // skip the "unknown" descriptor
  newfid++;

  for (int i = 0; i < NUM_EVENTS; i++) {
    while (iowrap_events[i].size() <= newfid) {
      iowrap_events[i].push_back(0);
    }
    iowrap_events[i][newfid] = iowrap_events[i][oldfid];
  }
  RtsLayer::UnLockDB();
}

/*********************************************************************
 * initializer
 ********************************************************************/
extern "C" void Tau_iowrap_checkInit()
{
  static int init = 0;
  if (init) return;
  init = 1;

  global_write_bandwidth = 0;
  global_read_bandwidth = 0;
  global_bytes_written = 0;
  global_bytes_read = 0;

  Tau_init_initializeTAU();
  Tau_iowrap_registerEvents(-1, "unknown");
  Tau_iowrap_registerEvents(0, "stdin");
  Tau_iowrap_registerEvents(1, "stdout");
  Tau_iowrap_registerEvents(2, "stderr");
  Tau_get_context_userevent(&global_write_bandwidth, "Write Bandwidth (MB/s)");
  Tau_get_context_userevent(&global_read_bandwidth, "Read Bandwidth (MB/s)");
  Tau_get_context_userevent(&global_bytes_written, "Bytes Written");
  Tau_get_context_userevent(&global_bytes_read, "Bytes Read");
  Tau_create_top_level_timer_if_necessary();
}

/*********************************************************************
 * Get the user event for the given type of event and file descriptor
 ********************************************************************/
extern "C" void *Tau_iowrap_getEvent(event_type type, unsigned int fid)
{
  IOvector const & iowrap_events = TheIoWrapEvents();
  fid++;    // skip the "unknown" descriptor

  if (fid >= iowrap_events[(int)type].size()) {
    dprintf("************** unknown fid! %d\n", fid - 1);
    fid = 0;    // use the "unknown" descriptor
  }
  return iowrap_events[(int)type][fid];
}
