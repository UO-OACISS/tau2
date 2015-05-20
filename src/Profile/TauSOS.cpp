
#include "Profile/Profiler.h"
#include "TauMetrics.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <cassert>
#include "sos.h"
#include "stdio.h"
#include "error.h"
#include "errno.h"
#include <pthread.h>
/*
#ifdef TAU_MPI
#include "VMPI.h"
#endif
*/
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>

#define INTERRUPT_PERIOD 1 // one second

SOS_pub_handle *pub = NULL;
unsigned long fi_count = 0;
bool done = false;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_t worker_thread;
bool _threaded = false;

inline void init_lock(void) {
    if (!_threaded) return;
    pthread_mutexattr_t Attr;
    pthread_mutexattr_init(&Attr);
    pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
    int rc;
    if ((rc = pthread_mutex_init(&_my_mutex, &Attr)) != 0) {
        errno = rc;
        perror("pthread_mutex_init error");
        exit(1);
    }
}
inline void do_lock(void) {
    if (!_threaded) return;
    int rc;
    if ((rc = pthread_mutex_lock(&_my_mutex)) != 0) {
        errno = rc;
        perror("pthread_mutex_lock error");
        exit(1);
    }
}

inline void do_unlock(void) {
    if (!_threaded) return;
    int rc;
    if ((rc = pthread_mutex_unlock(&_my_mutex)) != 0) {
        errno = rc;
        perror("pthread_mutex_unlock error");
        exit(1);
    }
}

extern "C" void TAU_SOS_send_data(void);

void * Tau_sos_thread_function(void* data) {
    while (!done) {
        sleep(1);
        TAU_SOS_send_data();
    }
    pthread_exit((void*)0L);
}

extern "C" void TAU_SOS_init(int argc, char ** argv, int threaded) {
    static bool initialized = false;
    if (!initialized) {
        _threaded = threaded > 0 ? true : false;
            init_lock();
            do_lock();
        SOS_init(&argc, &argv, SOS_APP);
        SOS_comm_split();
        pub = SOS_new_pub((char *)"TAU Application");
        if (_threaded) {
            int ret = pthread_create(&worker_thread, NULL, &Tau_sos_thread_function, NULL);
            if (ret != 0) {
                errno = ret;
                perror("Error: pthread_create (1) fails\n");
                exit(1);
            }
        }
        initialized = true;
        do_unlock();
    }
}

extern "C" void TAU_SOS_finalize(void) {
    static bool finalized = false;
    if (finalized) return;
    do_lock();
    done = true;
    SOS_finalize();
    finalized = true;
    do_unlock();
    if (_threaded) {
        int ret = pthread_join(worker_thread, NULL);
        if (ret != 0) {
            errno = ret;
            perror("Error: pthread_join (1) fails\n");
            exit(1);
        }
    }
}

extern "C" int TauProfiler_updateAllIntermediateStatistics(void);

extern "C" void TAU_SOS_send_data(void) {
    assert(pub);
    do_lock();
    if (done) return;
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::iterator it;
  const char **counterNames;
  int numCounters;
  TauMetrics_getCounterList(&counterNames, &numCounters);
  RtsLayer::LockDB();
  bool keys_added = false;
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    // get the number of calls
    int tid = 0; // todo: get ALL thread data.
    SOS_val calls, inclusive, exclusive;
    calls.d_val = 0.0;
    inclusive.d_val = 0.0;
    inclusive.d_val = 0.0;
    for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
        calls.d_val += fi->GetCalls(tid);
    }
    std::stringstream calls_str;
    calls_str << "TAU::calls::" << fi->GetName();
    const std::string& tmpcalls = calls_str.str();
    SOS_pack(pub, tmpcalls.c_str(), SOS_DOUBLE, calls);
    // todo - subroutines
    // iterate over metrics 
    std::stringstream incl_str;
    std::stringstream excl_str;
    for (int m = 0; m < Tau_Global_numCounters; m++) {
        incl_str.clear();
        incl_str << "TAU::inclusive::" << counterNames[m] << "::" << fi->GetName();
        const std::string& tmpincl = incl_str.str();
        excl_str.clear();
        excl_str << "TAU::exclusive::" << counterNames[m] << "::" << fi->GetName();
        const std::string& tmpexcl = excl_str.str();
        for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            inclusive.d_val += fi->getDumpInclusiveValues(tid)[m];
            exclusive.d_val += fi->getDumpInclusiveValues(tid)[m];
        }
        SOS_pack(pub, tmpincl.c_str(), SOS_DOUBLE, inclusive);
        SOS_pack(pub, tmpexcl.c_str(), SOS_DOUBLE, exclusive);
    }
  }
  if (TheFunctionDB().size() > fi_count) {
    keys_added = true;
    fi_count = TheFunctionDB().size();
  }
  RtsLayer::UnLockDB();
  if (keys_added) {
    SOS_announce(pub);
  }
  SOS_publish(pub);
  do_unlock();
}

