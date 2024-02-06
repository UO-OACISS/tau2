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
#include <deque>
#include <utility>
#include <sstream>
#include <string>
#ifndef TAU_WINDOWS
#include <sys/time.h>
#endif


using namespace std;
using namespace tau;

struct TauFidMap : public std::map<int, const char *> {
    ~TauFidMap() {
        Tau_destructor_trigger();
    }
};

TauFidMap & TheFidMap() {
    static TauFidMap fidMap;
    return fidMap;
}

const char * Tau_get_pathname_from_fid(int fid) {
    static const char * empty = "";
    if (fid == 0) {
        return empty;
    }
    if (TheFidMap().count(fid) == 0) {
        return empty;
    }
    return TheFidMap()[fid];
}


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
  // save the pathname so we can look it up later
  TheFidMap()[fid] = strdup(pathname);

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
    std::stringstream ss;
    ss << iowrap_event_names[i] << " <file=" << pathname << ">";
    std::string ename(ss.str());
    Tau_get_context_userevent(&event, ename.c_str());
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
  static thread_local bool seen{false};
  if (init) {
    // don't re-register thread 0!
    if (!seen) {
        if (Tau_init_check_initialized() && !Tau_global_getLightsOut()) {
            Tau_register_thread();
            Tau_create_top_level_timer_if_necessary();
            seen = true;
        }
    }
    return;
  }
  init = 1;
  seen = true;

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

enum io_event_kind {
    TAU_IO_EVENT_KIND_READ,
    TAU_IO_EVENT_KIND_WRITE,
    TAU_IO_NUM_EVENT_KINDS
};

struct tau_io_wrapper_event {
    struct timeval t1;
    struct timeval t2;
};

#if defined(TAU_USE_TLS) && !defined(TAU_INTEL_COMPILER)
// thread local storage
static tau_io_wrapper_event * tau_get_io_event_record(void)
{
  static __thread tau_io_wrapper_event * io_wrapper_event_tls = NULL;
  if(io_wrapper_event_tls == NULL) {
    io_wrapper_event_tls = new tau_io_wrapper_event[TAU_IO_NUM_EVENT_KINDS];
  }
  return io_wrapper_event_tls;
}

typedef std::map<std::string, void*> tfio_write_bytes_map_t;

// thread local storage
static tfio_write_bytes_map_t * tau_tfio_write_bytes_map(void)
{
  static __thread tfio_write_bytes_map_t * tfio_write_bytes_map_tls = NULL;
  if(tfio_write_bytes_map_tls == NULL) {
    tfio_write_bytes_map_tls = new tfio_write_bytes_map_t();
  }
  return tfio_write_bytes_map_tls;
}


typedef std::map<std::string, void*> tfio_write_bw_map_t;

// thread local storage
static tfio_write_bw_map_t * tau_tfio_write_bw_map(void)
{
  static __thread tfio_write_bw_map_t * tfio_write_bw_map_tls = NULL;
  if(tfio_write_bw_map_tls == NULL) {
    tfio_write_bw_map_tls = new tfio_write_bw_map_t();
  }
  return tfio_write_bw_map_tls;
}

typedef std::map<std::string, void*> tfio_read_bytes_map_t;

// thread local storage
static tfio_read_bytes_map_t * tau_tfio_read_bytes_map(void)
{
  static __thread tfio_read_bytes_map_t * tfio_read_bytes_map_tls = NULL;
  if(tfio_read_bytes_map_tls == NULL) {
    tfio_read_bytes_map_tls = new tfio_read_bytes_map_t();
  }
  return tfio_read_bytes_map_tls;
}


typedef std::map<std::string, void*> tfio_read_bw_map_t;

// thread local storage
static tfio_read_bw_map_t * tau_tfio_read_bw_map(void)
{
  static __thread tfio_read_bw_map_t * tfio_read_bw_map_tls = NULL;
  if(tfio_read_bw_map_tls == NULL) {
    tfio_read_bw_map_tls = new tfio_read_bw_map_t();
  }
  return tfio_read_bw_map_tls;
}



extern "C" {

void Tau_app_report_file_read_start(const char * name, size_t size) {
    TAU_START("TensorFlow File Read");
    tau_io_wrapper_event * tau_io_event_record_arr = tau_get_io_event_record();
    gettimeofday(&(tau_io_event_record_arr[TAU_IO_EVENT_KIND_READ].t1), 0);
    tfio_read_bytes_map_t * tfio_read_bytes_map = tau_tfio_read_bytes_map();
    std::string nameStr = std::string(name);
    if(tfio_read_bytes_map->find(nameStr) == tfio_read_bytes_map->end()) {
        void *event = 0;
        char ename[4096];
        snprintf(ename, sizeof(ename), "TensorFlow File Read Bytes <file=%s>", name);
        Tau_get_context_userevent(&event, ename);
        tfio_read_bytes_map->insert(std::pair<std::string, void *>(nameStr, event));
    }
    tfio_read_bw_map_t * tfio_read_bw_map = tau_tfio_read_bw_map();
    if(tfio_read_bw_map->find(nameStr) == tfio_read_bw_map->end()) {
        void *event = 0;
        char ename[4096];
        snprintf(ename, sizeof(ename), "TensorFlow File Read Bandwidth <file=%s>", name);
        Tau_get_context_userevent(&event, ename);
        tfio_read_bw_map->insert(std::pair<std::string, void *>(nameStr, event));
    }
}

void Tau_app_report_file_read_stop(const char * name, size_t size) {
    TAU_STOP("TensorFlow File Read");
    tau_io_wrapper_event * tau_io_event_record_arr = tau_get_io_event_record();
    gettimeofday(&(tau_io_event_record_arr[TAU_IO_EVENT_KIND_READ].t2), 0);

    tfio_read_bytes_map_t * tfio_read_bytes_map = tau_tfio_read_bytes_map();
    std::string nameStr = std::string(name);
    tfio_read_bytes_map_t::const_iterator it = tfio_read_bytes_map->find(nameStr);
    if(it == tfio_read_bytes_map->end()) {
        fprintf(stderr, "TAU: ERROR: File read stop seen for %s without start!\n", name);
        return;
    }
    void * bytesEvent = it->second;

    tfio_read_bw_map_t * tfio_read_bw_map = tau_tfio_read_bw_map();
    tfio_read_bw_map_t::const_iterator it2 = tfio_read_bw_map->find(nameStr);
    if(it == tfio_read_bw_map->end()) {
        fprintf(stderr, "TAU: ERROR: File read stop seen for %s without start!\n", name);
        return;
    }
    void * bwEvent = it2->second;

    struct timeval t1 = tau_io_event_record_arr[TAU_IO_EVENT_KIND_READ].t1;
    struct timeval t2 = tau_io_event_record_arr[TAU_IO_EVENT_KIND_READ].t2;
    double readTime = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
    double bw = size/readTime;

    TAU_CONTEXT_EVENT(bytesEvent, size);
    TAU_CONTEXT_EVENT(bwEvent, bw);

}

void Tau_app_report_file_write_start(const char * name, size_t size) {
    TAU_START("TensorFlow File Write");
    tau_io_wrapper_event * tau_io_event_record_arr = tau_get_io_event_record();
    gettimeofday(&(tau_io_event_record_arr[TAU_IO_EVENT_KIND_WRITE].t1), 0);
    tfio_write_bytes_map_t * tfio_write_bytes_map = tau_tfio_write_bytes_map();
    std::string nameStr = std::string(name);
    if(tfio_write_bytes_map->find(nameStr) == tfio_write_bytes_map->end()) {
        void *event = 0;
        char ename[4096];
        snprintf(ename, sizeof(ename), "TensorFlow File Write Bytes <file=%s>", name);
        Tau_get_context_userevent(&event, ename);
        tfio_write_bytes_map->insert(std::pair<std::string, void *>(nameStr, event));
    }
    tfio_write_bw_map_t * tfio_write_bw_map = tau_tfio_write_bw_map();
    if(tfio_write_bw_map->find(nameStr) == tfio_write_bw_map->end()) {
        void *event = 0;
        char ename[4096];
        snprintf(ename, sizeof(ename), "TensorFlow File Write Bandwidth <file=%s>", name);
        Tau_get_context_userevent(&event, ename);
        tfio_write_bw_map->insert(std::pair<std::string, void *>(nameStr, event));
    }
}

void Tau_app_report_file_write_stop(const char * name, size_t size) {
    TAU_STOP("TensorFlow File Write");
    tau_io_wrapper_event * tau_io_event_record_arr = tau_get_io_event_record();
    gettimeofday(&(tau_io_event_record_arr[TAU_IO_EVENT_KIND_WRITE].t2), 0);

    tfio_write_bytes_map_t * tfio_write_bytes_map = tau_tfio_write_bytes_map();
    std::string nameStr = std::string(name);
    tfio_write_bytes_map_t::const_iterator it = tfio_write_bytes_map->find(nameStr);
    if(it == tfio_write_bytes_map->end()) {
        fprintf(stderr, "TAU: ERROR: File write stop seen for %s without start!\n", name);
        return;
    }
    void * bytesEvent = it->second;

    tfio_write_bw_map_t * tfio_write_bw_map = tau_tfio_write_bw_map();
    tfio_write_bw_map_t::const_iterator it2 = tfio_write_bw_map->find(nameStr);
    if(it == tfio_write_bw_map->end()) {
        fprintf(stderr, "TAU: ERROR: File write stop seen for %s without start!\n", name);
        return;
    }
    void * bwEvent = it2->second;

    struct timeval t1 = tau_io_event_record_arr[TAU_IO_EVENT_KIND_WRITE].t1;
    struct timeval t2 = tau_io_event_record_arr[TAU_IO_EVENT_KIND_WRITE].t2;
    double writeTime = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
    double bw = size/writeTime;

    TAU_CONTEXT_EVENT(bytesEvent, size);
    TAU_CONTEXT_EVENT(bwEvent, bw);
}

void Tau_app_report_file_open_start(const char * name) {
    TAU_START("TensorFlow File Open");
}

void Tau_app_report_file_open_stop(const char * name) {
    TAU_STOP("TensorFlow File Open");
}

void Tau_app_report_file_close_start(const char * name) {
    TAU_START("TensorFlow File Close");
}

void Tau_app_report_file_close_stop(const char * name) {
    TAU_STOP("TensorFlow File Close");
}

void Tau_app_report_file_flush_start(const char * name) {
    TAU_START("TensorFlow File Flush");
}

void Tau_app_report_file_flush_stop(const char * name) {
    TAU_STOP("TensorFlow File Flush");
}


}
#endif // TAU_USE_TLS
