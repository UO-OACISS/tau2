/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997-2009					   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: UserEvent.h					  **
 **	Description 	: TAU Profiling Package				  **
 *	Contact		: tau-team@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

#ifndef _USEREVENT_H_
#define _USEREVENT_H_

#include <string>
#include <limits>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <Profile/TauInit.h>
#include <Profile/TauEnv.h>
#include <Profile/TauMmapMemMgr.h>


namespace tau {
//=============================================================================

// Forward declaration
class Profiler;

//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////
typedef double TAU_EVENT_DATATYPE;

//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////
class TauUserEvent
{
public:

  struct Data
  {
    Data() :
      minVal(std::numeric_limits<TAU_EVENT_DATATYPE>::max()),
      maxVal(-std::numeric_limits<TAU_EVENT_DATATYPE>::max()),
      sumVal(0), sumSqrVal(0), lastVal(0), userVal(0), nEvents(0)
    { }

    TAU_EVENT_DATATYPE minVal;
    TAU_EVENT_DATATYPE maxVal;
    TAU_EVENT_DATATYPE sumVal;
    TAU_EVENT_DATATYPE sumSqrVal;
    TAU_EVENT_DATATYPE lastVal;
    TAU_EVENT_DATATYPE userVal;
    size_t nEvents;
  };

  static void ReportStatistics(bool ForEachThread=false);

public:

  TauUserEvent() :
      user_event_id(next_user_event_id.fetch_add(1, std::memory_order_relaxed)),
      eventId(0), name("No Name"),
      minEnabled(true), maxEnabled(true), meanEnabled(true),
      stdDevEnabled(true), monoIncreasing(false), writeAsMetric(false)
  {
    //printf("Constructed UserEvent: %s\n", name.c_str()); fflush(stdout);
    AddEventToDB();
  }

  TauUserEvent(TauUserEvent const & e) :
      user_event_id(next_user_event_id.fetch_add(1, std::memory_order_relaxed)),
      eventId(0), name(e.name),
      minEnabled(e.minEnabled), maxEnabled(e.maxEnabled),
      meanEnabled(e.meanEnabled), stdDevEnabled(e.stdDevEnabled),
      monoIncreasing(e.monoIncreasing), writeAsMetric(false)
  {
    //printf("Constructed UserEvent: %s\n", name.c_str()); fflush(stdout);
    AddEventToDB();
  }

  //TauUserEvent(std::string const & name, bool increasing=false) :
  TauUserEvent(const char * name, bool increasing=false) :
      user_event_id(next_user_event_id.fetch_add(1, std::memory_order_relaxed)),
      eventId(0), name(name), minEnabled(true), maxEnabled(true),
      meanEnabled(true), stdDevEnabled(true), monoIncreasing(increasing), writeAsMetric(false)
  {
    //printf("Constructed UserEvent: %s\n", name); fflush(stdout);
    AddEventToDB();
  }

  ~TauUserEvent(void) {
    Tau_destructor_trigger();
  }

  TauUserEvent & operator=(const TauUserEvent & e) {
    // Why isn't eventId copied?
    name.assign(e.name);
    minEnabled = e.minEnabled;
    maxEnabled = e.maxEnabled;
    meanEnabled = e.meanEnabled;
    stdDevEnabled = e.stdDevEnabled;
    return *this;
  }

  x_uint64 GetId(void) const {
    return eventId;
  }

  std::string const & GetName(void) {
    return name;
  }

  //void SetName(std::string const & value) {
    //name = value.c_str();
  //}

  void SetName(std::string const & value) {
    name = value;
  }

  bool IsMinEnabled(void) const {
    return minEnabled;
  }
  void SetMinEnabled(bool value) {
    minEnabled = value;
  }

  bool IsMaxEnabled(void) const {
    return maxEnabled;
  }
  void SetMaxEnabled(bool value) {
    maxEnabled = value;
  }

  bool IsMeanEnabled(void) const {
    return meanEnabled;
  }
  void SetMeanEnabled(bool value) {
    meanEnabled = value;
  }

  bool IsStdDevEnabled(void) const {
    return stdDevEnabled;
  }
  void SetStdDevEnabled(bool value) {
    stdDevEnabled = value;
  }

  bool IsMonotonicallyIncreasing(void) const {
    return monoIncreasing;
  }
  void SetMonotonicallyIncreasing(bool value) {
    monoIncreasing = value;
  }

  void SetWriteAsMetric(bool value) {
    writeAsMetric = value;
  }

  bool GetWriteAsMetric() {
    return writeAsMetric;
  }

  TAU_EVENT_DATATYPE GetMin(void) {
    Data const & d = ThreadData();
    return d.nEvents ? d.minVal : 0;
  }
  TAU_EVENT_DATATYPE GetMin(int tid) {
    Data const & d = ThreadData(tid);
    return d.nEvents ? d.minVal : 0;
  }

  TAU_EVENT_DATATYPE GetMax(void) {
    Data const & d = ThreadData();
    return d.nEvents ? d.maxVal : 0;
  }
  TAU_EVENT_DATATYPE GetMax(int tid) {
    Data const & d = ThreadData(tid);
    return d.nEvents ? d.maxVal : 0;
  }

  TAU_EVENT_DATATYPE GetSum(void) {
    return ThreadData().sumVal;
  }
  TAU_EVENT_DATATYPE GetSum(int tid) {
    return ThreadData(tid).sumVal;
  }

  TAU_EVENT_DATATYPE GetSumSqr(void) {
    return ThreadData().sumSqrVal;
  }
  TAU_EVENT_DATATYPE GetSumSqr(int tid) {
    return ThreadData(tid).sumSqrVal;
  }

  TAU_EVENT_DATATYPE GetMean(void) {
    Data const & d = ThreadData();
    return d.nEvents ? (d.sumVal / d.nEvents) : 0;
  }
  TAU_EVENT_DATATYPE GetMean(int tid) {
    Data const & d = ThreadData(tid);
    return d.nEvents ? (d.sumVal / d.nEvents) : 0;
  }

  size_t GetNumEvents(void) {
    return ThreadData().nEvents;
  }
  size_t GetNumEvents(int tid) {
    return ThreadData(tid).nEvents;
  }

  void ResetData(void) {
    ThreadData() = Data();
  }
  void ResetData(int tid) {
    ThreadData(tid) = Data();
  }

  void TriggerEvent(TAU_EVENT_DATATYPE data) {
    TriggerEvent(data, RtsLayer::myThread(), 0, 0);
  }
  void TriggerEvent(TAU_EVENT_DATATYPE data, int tid) {
    TriggerEvent(data, tid, 0, 0);
  }
  void TriggerEventTS(TAU_EVENT_DATATYPE data, int tid, double ts) {
    TriggerEvent(data, tid, ts, 1);
  }
  void TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts);

  // Disable the TLS data cache globally (call during TAU shutdown, alongside
  // FunctionInfo::disable_metric_cache()).
  static void disable_data_cache() {
    use_data_tls.store(false, std::memory_order_relaxed);
  }

private:

  // Per-instance ID used to index into the per-thread DataThreadCache vector.
  // Assigned once at construction from a global atomic counter; never changes.
  static std::atomic<uint64_t> next_user_event_id;
  uint64_t user_event_id;

  // Guards uDataList growth in the ThreadData() slow path.
  std::mutex DataVectorMutex;

  // Thread-local cache: DataThreadCache[user_event_id] holds this thread's
  // Data* for a TauUserEvent instance.  Eliminates DataVectorMutex contention
  // after the first call per (thread × event) in the steady state.
  struct DataThreadCacheList : vector<Data*> {
    DataThreadCacheList() {}
    virtual ~DataThreadCacheList() {
      // Set flag BEFORE clearing so any signal handler or other code that
      // fires during teardown skips the cache instead of reading freed memory.
      TauUserEvent::DataThreadCacheDestroyed = true;
      this->clear();
    }
  };
  static thread_local DataThreadCacheList DataThreadCache;
  // Trivial TLS (no destructor); its lifetime outlasts DataThreadCache so it
  // remains readable after DataThreadCache's destructor runs.
  static thread_local bool DataThreadCacheDestroyed;
  // Global kill-switch: set to false during TAU shutdown to suppress all TLS
  // cache accesses (mirrors FunctionInfo::use_metric_tls).
  static std::atomic<bool> use_data_tls;

  Data & ThreadData() {
    int tid = RtsLayer::myThread();
    return ThreadData(tid);
  }

  Data & ThreadData(int tid) {
    static thread_local const unsigned int local_tid = RtsLayer::myThread();
    // Fast path: return cached Data* without any synchronization.
    if (use_data_tls.load(std::memory_order_relaxed) &&
        (unsigned int)tid == local_tid &&
        !DataThreadCacheDestroyed) {
      if (DataThreadCache.size() > user_event_id) {
        Data* d = DataThreadCache[user_event_id];
        if (d != nullptr) return *d;
      }
    }
    // Slow path: grow uDataList under the per-instance mutex.
    Data* d;
    {
      std::lock_guard<std::mutex> guard(DataVectorMutex);
      while (uDataList.size() <= (unsigned int)tid) {
        uDataList.push_back(new Data());
      }
      d = uDataList[tid];
    } // lock released here

    // Populate TLS cache outside the lock.  d points to a stable heap object
    // that will not be moved or freed while this TauUserEvent is alive.
    if (use_data_tls.load(std::memory_order_relaxed) &&
        (unsigned int)tid == local_tid &&
        !DataThreadCacheDestroyed) {
      while (DataThreadCache.size() <= user_event_id) {
        DataThreadCache.push_back(nullptr);
      }
      DataThreadCache[user_event_id] = d;
    }
    return *d;
  }

  int ThreadDataSize() {
    std::lock_guard<std::mutex> guard(DataVectorMutex);
    return (int)uDataList.size();
  }

  void AddEventToDB();

  struct UDataList : vector<Data*> {
    UDataList(const UDataList&) = delete;
    UDataList& operator=(const UDataList&) = delete;
    UDataList() {}
    virtual ~UDataList() {
      Tau_destructor_trigger();
    }
  };
  UDataList uDataList;

  x_uint64 eventId;
  std::string name;
  bool minEnabled;
  bool maxEnabled;
  bool meanEnabled;
  bool stdDevEnabled;
  bool monoIncreasing;
  bool writeAsMetric;
};


//////////////////////////////////////////////////////////////////////
// Don't inherit TauUserEvent to avoid virtual function overhead
//////////////////////////////////////////////////////////////////////
class TauContextUserEvent
{
public:

  TauContextUserEvent(char const * name, bool monoIncr=false) :
#ifdef TAU_SCOREP
      contextEnabled(false),
#else
      contextEnabled(TauEnv_get_callpath_depth() != 0),
#endif
      contextEvent(NULL)
  {
    //printf("Constructing ContextUserEvent: %s\n", name); fflush(stdout);
    /* KAH - Whoops!! We can't call "new" here, because malloc is not
     * safe in signal handling. therefore, use the special memory
     * allocation routines */
#if (!(defined (TAU_WINDOWS) || defined(_AIX)))
    userEvent = (TauUserEvent*)Tau_MemMgr_malloc(RtsLayer::unsafeThreadId(), sizeof(TauUserEvent));
    /*  now, use the pacement new function to create a object in
     *  pre-allocated memory. NOTE - this memory needs to be explicitly
     *  deallocated by explicitly calling the destructor.
     *  I think the best place for that is in the destructor for
     *  the hash table. */
    new(userEvent) TauUserEvent(name, monoIncr);
#else
      userEvent = new TauUserEvent(name, monoIncr);
#endif
  }

  TauContextUserEvent(const TauContextUserEvent &c) :
	  contextEnabled(c.contextEnabled),
      userEvent(c.userEvent),
      contextEvent(c.contextEvent)
  { }

  TauContextUserEvent & operator=(const TauContextUserEvent &rhs) {
      userEvent = rhs.userEvent;
      contextEvent = rhs.contextEvent;
      contextEnabled = rhs.contextEnabled;
      return *this;
  }

  ~TauContextUserEvent() {
    // Because of the above "fixes" for pre-allocating memory, this delete
    // method now crashes. Let's not and say we did, ok?
    //
    //delete userEvent;
  }

  void SetContextEnabled(bool value) {
    contextEnabled = value;
  }

  std::string const & GetUserEventName() const {
    return userEvent->GetName();
  }

  void SetAllEventName(std::string const & value) {
    userEvent->SetName(std::string(value.c_str()));
    if (contextEvent != NULL)
    {
      std::size_t sep_pos = contextEvent->GetName().find(':');
      if (sep_pos != std::string::npos)
      {
        std::string context_portion = contextEvent->GetName().substr(sep_pos, contextEvent->GetName().length()-sep_pos);
        //form new string
        //contextEvent = userEvent;
        std::string new_context = userEvent->GetName();
        new_context += std::string(" ");
        new_context += context_portion;
        contextEvent->SetName(std::string(new_context.c_str()));
      }
      else {
        contextEvent->SetName(std::string(value.c_str()));
      }
    }

  }

  std::string const & GetName() const {
    if (contextEnabled && contextEvent != NULL) {
        return contextEvent->GetName();
    } else {
        return userEvent->GetName();
    }
  }

  void SetName(std::string const & value) {
    contextEvent->SetName(std::string(value.c_str()));
  }

  TauUserEvent *getContextUserEvent() {
    return contextEvent;
  }

  TauUserEvent *getUserEvent() {
    return userEvent;
  }

  void TriggerEvent(TAU_EVENT_DATATYPE data) {
    TriggerEvent(data, RtsLayer::myThread(), 0, 0);
  }
  void TriggerEvent(TAU_EVENT_DATATYPE data, int tid) {
    TriggerEvent(data, tid, 0, 0);
  }
  void TriggerEventTS(TAU_EVENT_DATATYPE data, int tid, double ts) {
    TriggerEvent(data, tid, ts, 1);
  }
  void TriggerEvent(TAU_EVENT_DATATYPE data, int tid, double timestamp, int use_ts);

private:

  void FormulateContextComparisonArray(Profiler * current, long * comparison);
  std::string FormulateContextNameString(Profiler * current);

  bool contextEnabled;
  TauUserEvent * userEvent;
  TauUserEvent * contextEvent;
};


//////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////
struct AtomicEventDB : public std::vector<tau::TauUserEvent*>
{
  AtomicEventDB() {
    Tau_init_initializeTAU();
  }
  ~AtomicEventDB() {
    Tau_destructor_trigger();
  }
};

AtomicEventDB & TheEventDB(void);


//=============================================================================
} // END namespace tau


#endif /* _USEREVENT_H_ */

/***************************************************************************
 * $RCSfile: UserEvent.h,v $   $Author: amorris $
 * $Revision: 1.17 $   $Date: 2009/09/28 18:28:48 $
 * POOMA_VERSION_ID: $Id: UserEvent.h,v 1.17 2009/09/28 18:28:48 amorris Exp $
 ***************************************************************************/
