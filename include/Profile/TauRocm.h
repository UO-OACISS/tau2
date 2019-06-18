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


#ifndef _TAU_ROCM_H_
#define _TAU_ROCM_H_

#include <string.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <atomic>
#include "Profile/Profiler.h"
#include <dlfcn.h>
using namespace std;

#define TAU_METRIC_TYPE unsigned long long

#ifndef TAU_ROCM_LOOK_AHEAD
#define TAU_ROCM_LOOK_AHEAD 128
#endif /* TAU_ROCM_LOOK_AHEAD */

#ifndef TAU_MAX_ROCM_QUEUES
#define TAU_MAX_ROCM_QUEUES 512
#endif /* TAU_MAX_ROCM_QUEUES */

//static TAU_METRIC_TYPE tau_last_timestamp_published = 0;

struct TauRocmCounter {
  TAU_METRIC_TYPE counters[TAU_MAX_COUNTERS];
};

struct TauRocmEvent {
  struct TauRocmCounter entry;
  struct TauRocmCounter exit;
  string name;
  int taskid;

  TauRocmEvent(): taskid(0) {}
  TauRocmEvent(string event_name, TAU_METRIC_TYPE begin, TAU_METRIC_TYPE end, int t) : name(event_name), taskid(t)
  {
    entry.counters[0] = begin;
    exit.counters[0]  = end;
  }
  void printEvent() {
    std::cout <<name<<" Task: "<<taskid<<", \t\tEntry: "<<entry.counters[0]<<" , Exit = "<<exit.counters[0];
  }
  bool appearsBefore(struct TauRocmEvent other_event) {
    if ((taskid == other_event.taskid) &&
        (entry.counters[0] < other_event.entry.counters[0]) &&
        (exit.counters[0] < other_event.entry.counters[0]))  {
      // both entry and exit of my event is before the entry of the other event. 
      return true;
    } else
      return false;
  }

} ;

extern std::list<struct TauRocmEvent> TauRocmList;
extern void Tau_process_rocm_events(struct TauRocmEvent e);
extern bool Tau_compare_rocm_events (struct TauRocmEvent one, struct TauRocmEvent two); 
extern void Tau_process_rocm_events(struct TauRocmEvent e);
extern int Tau_get_initialized_queues(int queue_id);
extern void Tau_set_initialized_queues(int queue_id, int value); 
extern void Tau_metric_set_synchronized_gpu_timestamp(int tid, double value); 
extern void Tau_add_metadata_for_task(const char *key, int value, int taskid);
extern bool Tau_check_timestamps(unsigned long long last_timestamp, unsigned long long current_timestamp, const char *debug_str, int taskid); 
extern void TauPublishEvent(struct TauRocmEvent event); 
extern void Tau_process_rocm_events(struct TauRocmEvent e); 
extern void TauFlushRocmEventsIfNecessary(int thread_id);
extern TAU_METRIC_TYPE Tau_get_last_timestamp_ns(void); 
extern void Tau_set_last_timestamp_ns(TAU_METRIC_TYPE timestamp);

extern "C" x_uint64 TauTraceGetTimeStamp();
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" void Tau_metadata_task(const char *name, const char *value, int tid);
extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);


#endif /* _TAU_ROCM_H */
