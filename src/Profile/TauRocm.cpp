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

//#define DEBUG_PROF 1 

std::list<struct TauRocmEvent> TauRocmList;
static TAU_METRIC_TYPE tau_last_timestamp_published = 0;
//static unsigned long long tau_last_timestamp_ns = 0L;
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


static int tau_initialized_queues[TAU_MAX_ROCM_QUEUES] = { 0 };

// Dispatch callbacks and context handlers synchronization
// Tool is unloaded

// access tau_initialized_queues through get and set access functions
int Tau_get_initialized_queues(int queue_id) {
  return tau_initialized_queues[queue_id]; 
}

void Tau_set_initialized_queues(int queue_id, int value) {
  tau_initialized_queues[queue_id]=value; 
  return;
}

void Tau_metric_set_synchronized_gpu_timestamp(int tid, double value){
	if (offset_timestamp == 0L)
	{
		offset_timestamp=TauTraceGetTimeStamp() - ((double)value);
	}
	metric_set_gpu_timestamp(tid, offset_timestamp+value);
}


void Tau_add_metadata_for_task(const char *key, int value, int taskid) {
  char buf[1024];
  sprintf(buf, "%d", value);
  Tau_metadata_task(key, buf, taskid);
  TAU_VERBOSE("Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
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
  Tau_metric_set_synchronized_gpu_timestamp(event.taskid, ((double)timestamp/1e3)); // convert to microseconds
  TAU_START_TASK(event.name.c_str(), event.taskid);
  TAU_VERBOSE("Started event %s on task %d timestamp = %lu \n", event.name.c_str(), event.taskid, timestamp);

  // then the exit
  timestamp = event.exit.counters[0]; // Using first element of array 
  tau_last_timestamp_published = timestamp;
  Tau_metric_set_synchronized_gpu_timestamp(event.taskid, ((double)timestamp/1e3)); // convert to microseconds
  TAU_STOP_TASK(event.name.c_str(), event.taskid);
  TAU_VERBOSE("Stopped event %s on task %d timestamp = %lu \n", event.name.c_str(), event.taskid, timestamp);
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


