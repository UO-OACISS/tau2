/****************************************************************************
 * **                      TAU Portable Profiling Package                     **
 * **                      http://www.cs.uoregon.edu/research/tau             **
 * *****************************************************************************
 * **    Department of Computer and Information Science, University of Oregon **
 * **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 * **    Forschungszentrum Juelich                                            **
 * ****************************************************************************/
/*******************************************************************************
 * **      File            : TauRocm.cpp                                      **
 * **      Description     : TAU Rocm routines                                **
 * **      Author          : Sameer Shende                                    **
 * **      Contact         : tau-bugs@cs.uoregon.edu                          **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 * **                                                                         **
 * **                                                                         **
 * ****************************************************************************/

#include <Profile/TauRocm.h>
#include <Profile/Profiler.h>

//#define DEBUG_PROF 1

std::list<struct TauRocmEvent> TauRocmList;
static TAU_METRIC_TYPE tau_last_timestamp_published = 0;
static TAU_METRIC_TYPE tau_last_timestamp_ns = 0L;
static unsigned long long offset_timestamp = 0L;

// Compare to TauRocmEvents based on their timestamps (for sorting)
bool Tau_compare_rocm_events (struct TauRocmEvent one, struct TauRocmEvent two) {
  if (one.entry.counters[0] < two.entry.counters[0]) return true;
  else return false;
/*
  if (one.appearsBefore(two)) return true; // Less than
  else return false; // one does not appear before two.
*/
}


#ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
static int tau_initialized_queues[TAU_MAX_ROCM_QUEUES];
#else
static std::map<int, int, less<int> >& TheTauInitializedQueues() {
  static std::map<int, int, less<int> > initialized_queues;
  return initialized_queues;
}
#endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */


// Dispatch callbacks and context handlers synchronization
// Tool is unloaded

int Tau_initialize_queues(void) {
  int i;
  for (i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    tau_initialized_queues[i] = -1;
  }
  return 1;
}
// access tau_initialized_queues through get and set access functions
int Tau_get_initialized_queues(int queue_id) {
  //TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d ", queue_id);
#ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
  static int flag = Tau_initialize_queues();
  //TAU_VERBOSE("value = %d\n", tau_initialized_queues[queue_id]);
  return tau_initialized_queues[queue_id];
#else

  std::map<int, int, less<int> >::iterator it;
  it = TheTauInitializedQueues().find(queue_id);
  if (it == TheTauInitializedQueues().end()) { // not found!
    TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d not found. Returning -1\n", queue_id);
    TAU_VERBOSE("value = -1\n");
    return -1;
  } else {
    TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d found. Returning %d\n", queue_id, it->second);
    TAU_VERBOSE("value = %d\n", it->second);
    return it->second;
  }
#endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */
  /*
  TAU_VERBOSE("Tau_get_initialized_queues: queue_id = %d, value = %d\n", queue_id,  TheTauInitializedQueues()[queue_id]);
  return TheTauInitializedQueues()[queue_id];
  */
}

void Tau_set_initialized_queues(int queue_id, int value) {
  TAU_VERBOSE("Tau_set_initialized_queues: queue_id = %d, value = %d\n", queue_id, value);
#ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
  tau_initialized_queues[queue_id]=value;
#else
  TheTauInitializedQueues()[queue_id]=value;
  TAU_VERBOSE("Tau_set_initialized_queues: queue_id = %d, value = %d\n", queue_id,  TheTauInitializedQueues()[queue_id]);
#endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */
  return;
}

double Tau_metric_set_synchronized_gpu_timestamp(int tid, double value){
  //printf("value = %f\n", value);
  if (offset_timestamp == 0L) {
    offset_timestamp=TauTraceGetTimeStamp() - ((double)value);
  }
  metric_set_gpu_timestamp(tid, offset_timestamp+value);
  //Tau_create_top_level_timer_if_necessary_task(tid);
  TAU_VERBOSE("metric_set_gpu_timestamp = %llu + %f = %f\n", offset_timestamp, value, offset_timestamp+value);
  return offset_timestamp+value;
}

// access tau_last_timestamp_ns through get and set access functions
TAU_METRIC_TYPE Tau_get_last_timestamp_ns(void) {
  return tau_last_timestamp_ns;
}

void Tau_set_last_timestamp_ns(TAU_METRIC_TYPE timestamp) {
  tau_last_timestamp_ns = timestamp;
}

void Tau_add_metadata_for_task(const char *key, int value, int taskid) {
  char buf[1024];
  snprintf(buf, sizeof(buf),  "%d", value);
  Tau_metadata_task(key, buf, taskid);
  TAU_VERBOSE("Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
}

bool Tau_is_thread_id_rocm_task(int thread_id) {
  // Just for checking!
/* Not needed with iterators */
#ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
  for (int i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    if (Tau_get_initialized_queues(i) == thread_id) {
      //TAU_VERBOSE("TauIsThreadRocmTask: Tau_get_initialized_queues(%d) = %d matches thread_id %d. Returning true\n", i, Tau_get_initialized_queues(i), thread_id);
      return true;
    }
  }
#else

  std::map<int, int, less<int> >::iterator it;
  for (it = TheTauInitializedQueues().begin(); it != TheTauInitializedQueues().end(); it++) {
    if (it -> second == thread_id) { // match found!
      //TAU_VERBOSE("TauIsThreadRocmTask: Tau_get_initialized_queues(%d) = %d matches thread_id %d. Returning true\n", it->first, it->second, thread_id);
      return true;
    }
  }
#endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */
  return false;
}


bool Tau_check_timestamps(unsigned long long last_timestamp, unsigned long long current_timestamp, const char *debug_str, int taskid) {
  TAU_VERBOSE("Taskid<%d>: Tau_check_timestamps: Checking last_timestamp = %llu, current_timestamp = %llu at %s\n", taskid, last_timestamp, current_timestamp, debug_str);
  if (last_timestamp > current_timestamp) {
    TAU_VERBOSE("Taskid<%d>: Tau_check_timestamps: Timestamps are not monotonically increasing! last_timestamp = %llu, current_timestamp = %llu at %s\n", taskid, last_timestamp, current_timestamp, debug_str);
    return false;
  }
  else
    return true;
}

void TauPublishEvent(struct TauRocmEvent event) {
  std::list<struct TauRocmEvent>::iterator it;
  TAU_METRIC_TYPE timestamp;

#ifdef DEBUG_PROF
  cout <<"Publishing event ";
  event.printEvent();
  cout <<endl;

  cout <<endl<<" ------------- List -----------------: last: "<<tau_last_timestamp_published<<" current entry: "<<
	event.entry.counters[0]<<endl;
  for (it = TauRocmList.begin(); it != TauRocmList.end(); it++) {
    it->printEvent();
    cout <<endl;
  }

  cout <<" ------------- List -----------------"<<endl<<endl;
#endif /* DEBUG_PROF */
  // First the entry
  if (event.entry.counters[0] < tau_last_timestamp_published) {

    TAU_VERBOSE("ERROR: TauPublishEvent: Event to be published has a timestamp = %llu that is earlier than the last published timestamp %llu\n",
	event.entry.counters[0], tau_last_timestamp_published);
    TAU_VERBOSE("ERROR: please re-configure TAU with -useropt=-DTAU_ROCM_LOOK_AHEAD=16 (or some higher value) for a window of events to sort, make install and retry the command\n");
    TAU_VERBOSE("Ignoring this event:\n");
#ifdef DEBUG_PROF
    event.printEvent();
#endif /* DEBUG_PROF */
    return;
  }
  timestamp = event.entry.counters[0]; // Using first element of array
  double timestamp_entry = Tau_metric_set_synchronized_gpu_timestamp(event.taskid, ((double)timestamp/1e3)); // convert to microseconds


  // then the exit
  timestamp = event.exit.counters[0]; // Using first element of array
  tau_last_timestamp_published = timestamp;
  double timestamp_exit = Tau_metric_set_synchronized_gpu_timestamp(event.taskid, ((double)timestamp/1e3)); // convert to microseconds
  

  metric_set_gpu_timestamp(event.taskid, timestamp_entry);
  TAU_START_TASK(event.name.c_str(), event.taskid);
  TAU_VERBOSE("Started event %s on task %d timestamp = %lu \n", event.name.c_str(), event.taskid, timestamp_entry);
  
  metric_set_gpu_timestamp(event.taskid, timestamp_exit);
  TAU_STOP_TASK(event.name.c_str(), event.taskid);
  TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", event.name.c_str(), event.taskid, timestamp_exit);
}


void Tau_process_rocm_events(struct TauRocmEvent e) {
  std::list<struct TauRocmEvent>::iterator it;

#ifdef DEBUG_PROF
  cout <<"        Pushing event to the list: ";
  e.printEvent();
  cout <<endl;
#endif /* DEBUG_PROF */

  TauRocmList.push_back(e);

  TauRocmList.sort(Tau_compare_rocm_events);
  int listsize = TauRocmList.size();

  if (listsize < TAU_ROCM_LOOK_AHEAD) {

    return; // don't do anything else
  } else {
    // There are elements in the list. What are they?
    TauPublishEvent(TauRocmList.front());
    //TauRocmList.pop_front();
#ifdef DEBUG_PROF
    cout <<"   before popping, size = "<<TauRocmList.size()<<endl;
    cout <<" *********   TauRocmList BEFORE ******* "<<endl;
    for(it=TauRocmList.begin(); it != TauRocmList.end(); it++) {

      it->printEvent();
      cout <<endl;
    }
    cout <<" *********   TauRocmList BEFORE ******* "<<endl;
#endif /* DEBUG_PROF */
/*
    it=TauRocmList.begin();
    TauRocmList.erase(it);
*/
    TauRocmList.pop_front();

#ifdef DEBUG_PROF
    int listsize = TauRocmList.size();
    cout <<"   After popping, size = "<<listsize<<endl;
    cout <<" *********   TauRocmList AFTER ******* "<<endl;
    for(it=TauRocmList.begin(); it != TauRocmList.end(); it++) {
      it->printEvent();
      cout <<endl;
    }
    cout <<" *********   TauRocmList AFTER ******* "<<endl;
#endif /* DEBUG_PROF */
  }
  return;
}


extern void TauFlushRocmEventsIfNecessary() {

  //static bool reentrant_flag = false;
  /*if (reentrant_flag) {
    TAU_VERBOSE("TauFlushRocmEventsIfNecessary: reentrant_flag = true, tid = %d\n", thread_id);
    return;
  }
  if (Tau_is_thread_id_rocm_task(thread_id)) {
    reentrant_flag = true;
  }
  else {
    return;
  }*/
  //TAU_VERBOSE("Inside TauFlushRocmEventsIfNecessary: tid = %d\n", thread_id);
  TAU_VERBOSE("Inside TauFlushRocmEventsIfNecessary\n");
  //if(!Tau_is_thread_id_rocm_task(thread_id)) return ;

  if (TauRocmList.empty()) return;
  TAU_VERBOSE("Inside unload! publishing...\n");
  TauRocmList.sort(Tau_compare_rocm_events);
  while (!TauRocmList.empty()) {
    //TauPublishEvent(TauRocmList.front(), thread_id);
    TauPublishEvent(TauRocmList.front());
    TauRocmList.pop_front();
  }

//  Tau_stop_top_level_timer_if_necessary();

#ifndef TAU_ROCM_USE_MAP_FOR_INIT_QUEUES
  for (int i=0; i < TAU_MAX_ROCM_QUEUES; i++) {
    if (Tau_get_initialized_queues(i) != -1) {
      RtsLayer::LockDB();
      if (Tau_get_initialized_queues(i) != -1) {  // contention. Is it still -1?
        TAU_VERBOSE("Closing thread id: %d last timestamp = %llu\n", Tau_get_initialized_queues(i), tau_last_timestamp_ns);
        Tau_metric_set_synchronized_gpu_timestamp(i, ((double)tau_last_timestamp_ns/1e3)); // convert to microseconds
        Tau_stop_top_level_timer_if_necessary_task(Tau_get_initialized_queues(i));
        Tau_set_initialized_queues(i, -1);
      }
      RtsLayer::UnLockDB();
    }
  }
#else
  std::map<int, int, less<int> >::iterator it;
  for (it = TheTauInitializedQueues().begin(); it != TheTauInitializedQueues().end(); it++) {
    if (it -> second != -1) {
      RtsLayer::LockDB();
      int i = it->first;
      if (Tau_get_initialized_queues(i) != -1) {  // contention. Is it still -1?
        TAU_VERBOSE("Closing thread id: %d last timestamp = %llu\n", Tau_get_initialized_queues(i), tau_last_timestamp_ns);
        Tau_metric_set_synchronized_gpu_timestamp(i, ((double)tau_last_timestamp_ns/1e3)); // convert to microseconds
        Tau_stop_top_level_timer_if_necessary_task(Tau_get_initialized_queues(i));
	TAU_VERBOSE("Setting initialized_queues i=%d to -1: \n", i);
        Tau_set_initialized_queues(i, -1);
      }
      RtsLayer::UnLockDB();
    }
  }
#endif /* TAU_ROCM_USE_MAP_FOR_INIT_QUEUES */
  //reentrant_flag = false;  /* safe */

}


