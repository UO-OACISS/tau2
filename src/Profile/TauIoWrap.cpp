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

#include <vector>
using namespace std;

#define dprintf TAU_VERBOSE


const char *iowrap_event_names[NUM_EVENTS] = {"Write Bandwidth (MB/s)", "Bytes Written", "Read Bandwidth (MB/s)", "Bytes Read"};


/*********************************************************************
 * IOvector subclasses vector to provide custom constructor/destructor 
 * to enable/disable wrapping
 ********************************************************************/
static int lightsOut = 0;
class IOvector : public vector<vector<TauUserEvent*> > {
public:
  IOvector(int farg) : vector<vector<TauUserEvent*> >(farg) {
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
static IOvector &TheIoWrapEvents() {
  static IOvector iowrap_events(4);
  return iowrap_events;
}

void *global_write_bandwidth=0, *global_read_bandwidth=0, *global_bytes_written=0, *global_bytes_read=0;


/*********************************************************************
 * return whether we should pass through and not track the IO
 ********************************************************************/
extern "C" int Tau_iowrap_checkPassThrough() {
  if (Tau_global_get_insideTAU() > 0 || lightsOut) {
    return 1;
  } else {
    return 0;
  }
}

/*********************************************************************
 * register the different events (read/write/etc) for a file descriptor
 ********************************************************************/
extern "C" void Tau_iowrap_registerEvents(int fid, const char *pathname) {
  RtsLayer::LockDB();
  dprintf ("Asked to registering %d with %s (current size=%d)\n", fid, pathname, TheIoWrapEvents()[0].size());
  fid++; // skip the "unknown" descriptor

  for (int i=0; i<NUM_EVENTS; i++) {
    TauUserEvent *unknown_ptr = 0;
    if (TheIoWrapEvents()[i].size() >= 1) {
      unknown_ptr = TheIoWrapEvents()[i][0];
    }
    while (TheIoWrapEvents()[i].size() <= fid) {
      TheIoWrapEvents()[i].push_back(unknown_ptr);
      if (TheIoWrapEvents()[i].size()-1 != fid) {
	dprintf ("Registering %d with unknown\n", TheIoWrapEvents()[i].size()-2);
      }
    }
    void *event = 0;
    string name = string(iowrap_event_names[i]) + " <file=\"" + pathname + "\">";
    Tau_get_context_userevent(&event, strdup((char*)name.c_str()));
    TheIoWrapEvents()[i][fid] = (TauUserEvent*)event;
  }
  dprintf ("Registering %d with %s\n", fid-1, pathname);
  RtsLayer::UnLockDB();
}

/*********************************************************************
 * unregister events for a file descriptor
 ********************************************************************/
extern "C" void Tau_iowrap_unregisterEvents(int fid) {
  RtsLayer::LockDB();
  dprintf ("Un-registering %d\n", fid);
  fid++; // skip the "unknown" descriptor

  for (int i=0; i<NUM_EVENTS; i++) {
    TauUserEvent *unknown_ptr = 0;
    if (TheIoWrapEvents()[i].size() >= 1) {
      unknown_ptr = TheIoWrapEvents()[i][0];
    }
    while (TheIoWrapEvents()[i].size() <= fid) {
      TheIoWrapEvents()[i].push_back(unknown_ptr);
    }
    TheIoWrapEvents()[i][fid] = unknown_ptr;
  }
  RtsLayer::UnLockDB();
}


/*********************************************************************
 * Tau_iowrap_dupEvents takes care of the associating the events with the 
 * new file descriptor obtained by using dup/dup2 calls.
 ********************************************************************/
extern "C" void Tau_iowrap_dupEvents(int oldfid, int newfid) {
  RtsLayer::LockDB();
  dprintf ("dup (old=%d, new=%d)\n", oldfid, newfid);
  oldfid++; // skip the "unknown" descriptor
  newfid++;

  for (int i=0; i<NUM_EVENTS; i++) {
    while (TheIoWrapEvents()[i].size() <= newfid) {
      TheIoWrapEvents()[i].push_back(0);
    }
    TheIoWrapEvents()[i][newfid] = TheIoWrapEvents()[i][oldfid];
  }
  RtsLayer::UnLockDB();
}



/*********************************************************************
 * initializer
 ********************************************************************/
extern "C" void Tau_iowrap_checkInit() {
  static int init = 0;
  if (init == 1) {
    return;
  }
  init = 1;

  global_write_bandwidth=0;
  global_read_bandwidth=0;
  global_bytes_written=0;
  global_bytes_read=0;

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
extern "C" void *Tau_iowrap_getEvent(event_type type, int fid) {
  fid++; // skip the "unknown" descriptor
  if (fid >= TheIoWrapEvents()[(int)type].size()) {
    dprintf ("************** unknown fid! %d\n", fid-1);
    fid = 0; // use the "unknown" descriptor
  }
  return TheIoWrapEvents()[(int)type][fid];
}
