
#include "Profile/TauSOS.h"
#include "Profile/Profiler.h"
#include "Profile/UserEvent.h"
#include "Profile/TauMetrics.h"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <cassert>
#include "stdio.h"
#include "error.h"
#include "errno.h"
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>

#ifdef TAU_MPI
#include <mpi.h>
#endif

#include "sos.h"

SOS_pub *tau_sos_pub = NULL;
unsigned long fi_count = 0;
unsigned long ue_count = 0;
static bool done = false;
static SOS_runtime * _runtime = NULL;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
bool _threaded = false;

void init_lock(void) {
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
    if ((rc = pthread_cond_init(&_my_cond, NULL)) != 0) {
        errno = rc;
        perror("pthread_cond_init error");
        exit(1);
    }
}

void * Tau_sos_thread_function(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;

    while (!done) {
        // wait 2 seconds for the next batch.
        gettimeofday(&tp, NULL);
        ts.tv_sec  = (tp.tv_sec + 2);
        ts.tv_nsec = (1000 * tp.tv_usec);
        pthread_mutex_lock(&_my_mutex);
        int rc = pthread_cond_timedwait(&_my_cond, &_my_mutex, &ts);
        if (rc == ETIMEDOUT) {
            TAU_VERBOSE("%d Sending data from TAU thread...\n", RtsLayer::myNode()); fflush(stderr);
            TAU_SOS_send_data();
            TAU_VERBOSE("%d Done.\n", RtsLayer::myNode()); fflush(stderr);
        } else if (rc == EINVAL) {
            printf("Invalid timeout!\n"); fflush(stdout);
        } else if (rc == EPERM) {
            printf("Mutex not locked!\n"); fflush(stdout);
        }
    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    TAU_VERBOSE("TAU SOS thread exiting.\n"); fflush(stderr);
    pthread_exit((void*)0L);
}

void TAU_SOS_make_pub() {
        char pub_name[SOS_DEFAULT_STRING_LEN] = {0};
        char app_version[SOS_DEFAULT_STRING_LEN] = {0};

        TAU_VERBOSE("[TAU_SOS_init]: Creating new pub...\n");

#ifdef TAU_MPI
        int rank;
        int commsize;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &commsize);
        _runtime->config.comm_rank = rank;
        _runtime->config.comm_size = commsize;
#endif

/* Fixme! Replace these with values from TAU metadata. */
        sprintf(pub_name, "TAU_SOS_SUPPORT");
        sprintf(app_version, "v0.alpha");
/* Fixme! Replace these with values from TAU metadata. */
        tau_sos_pub = SOS_pub_create(_runtime, pub_name, SOS_NATURE_DEFAULT);

        strcpy(tau_sos_pub->prog_ver, app_version);
        tau_sos_pub->meta.channel       = 1;
        tau_sos_pub->meta.layer         = SOS_LAYER_LIB;
        // tau_sos_pub->meta.pri_hint      = SOS_PRI_IMMEDIATE;
        // tau_sos_pub->meta.scope_hint    = SOS_SCOPE_SELF;
        // tau_sos_pub->meta.retain_hint   = SOS_RETAIN_SESSION;

        TAU_VERBOSE("[TAU_SOS_init]:   ... done.  (pub->guid == %ld)\n", tau_sos_pub->guid);
}

extern "C" void TAU_SOS_init(int * argc, char *** argv, bool threaded) {
    static bool initialized = false;
    if (!TauEnv_get_sos_enabled()) { TAU_VERBOSE("*** SOS NOT ENABLED! ***\n"); return; }
    if (!initialized) {
        _threaded = threaded > 0 ? true : false;
        init_lock();
        _runtime = SOS_init(argc, argv, SOS_ROLE_CLIENT, SOS_LAYER_LIB);

        if (_threaded) {
            TAU_VERBOSE("Spawning thread for SOS.\n");
            int ret = pthread_create(&worker_thread, NULL, &Tau_sos_thread_function, NULL);
            if (ret != 0) {
                errno = ret;
                perror("Error: pthread_create (1) fails\n");
                exit(1);
            }
        }
        initialized = true;
        /* Fixme! Insert all the data that was collected into Metadata */
    }
}

extern "C" void TAU_SOS_stop_worker(void) {
    //printf("%s\n", __func__); fflush(stdout);
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    if (_threaded) {
        TAU_VERBOSE("TAU SOS thread joining...\n"); fflush(stderr);
        pthread_cond_signal(&_my_cond);
        int ret = pthread_join(worker_thread, NULL);
        if (ret != 0) {
            switch (ret) {
                case ESRCH:
                    // already exited.
                    break;
                case EINVAL:
                    // Didn't exist?
                    break;
                case EDEADLK:
                    // trying to join with itself?
                    break;
                default:
                    errno = ret;
                    perror("Warning: pthread_join failed\n");
                    break;
            }
        }
        pthread_cond_destroy(&_my_cond);
        pthread_mutex_destroy(&_my_mutex);
    }
}

extern "C" void TAU_SOS_finalize(void) {
    static bool finalized = false;
    if (!TauEnv_get_sos_enabled()) { return; }
    //printf("%s\n", __func__); fflush(stdout);
    if (finalized) return;
    if (!done) {
        TAU_SOS_stop_worker();
    }
    SOS_finalize(_runtime);
    finalized = true;
}

extern "C" int TauProfiler_updateAllIntermediateStatistics(void);

extern "C" void TAU_SOS_send_data(void) {
    if (tau_sos_pub == NULL) {
        TAU_SOS_make_pub();
    }
    assert(tau_sos_pub);
    if (done) { return; }
    //TauTrackPowerHere(); // get a power measurement
    Tau_global_incr_insideTAU();
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::iterator it;
  const char **counterNames;
  int numCounters;
  TauMetrics_getCounterList(&counterNames, &numCounters);
  RtsLayer::LockDB();
  bool keys_added = false;

  //foreach: TIMER
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    // get the number of calls
    int tid = 0; // todo: get ALL thread data.
    int calls;
    double inclusive, exclusive;
    calls = 0;
    inclusive = 0.0;
    exclusive = 0.0;

    //foreach: THREAD
    for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
        calls = fi->GetCalls(tid);
        std::stringstream calls_str;
        calls_str << "TAU::" << tid << "::calls::" << fi->GetName();
        const std::string& tmpcalls = calls_str.str();

        SOS_pack(tau_sos_pub, tmpcalls.c_str(), SOS_VAL_TYPE_INT, &calls);

        // todo - subroutines
        // iterate over metrics 
        std::stringstream incl_str;
        std::stringstream excl_str;
        for (int m = 0; m < Tau_Global_numCounters; m++) {
            incl_str.str(std::string());
            incl_str << "TAU::" << tid << "::inclusive_" << counterNames[m] << "::" << fi->GetName();
            const std::string& tmpincl = incl_str.str();
            excl_str.str(std::string());
            excl_str << "TAU::" << tid << "::exclusive_" << counterNames[m] << "::" << fi->GetName();
            const std::string& tmpexcl = excl_str.str();

            inclusive = fi->getDumpInclusiveValues(tid)[m];
            exclusive = fi->getDumpExclusiveValues(tid)[m];
            
            SOS_pack(tau_sos_pub, tmpincl.c_str(), SOS_VAL_TYPE_DOUBLE, &inclusive);
            SOS_pack(tau_sos_pub, tmpexcl.c_str(), SOS_VAL_TYPE_DOUBLE, &exclusive);
        }
    }
  }
  if (TheFunctionDB().size() > fi_count) {
    keys_added = true;
    fi_count = TheFunctionDB().size();
  }
  // do the same with counters.
  std::vector<tau::TauUserEvent*>::iterator it2;
  int numEvents;
  double max, min, mean, sumsqr;
  std::stringstream tmp_str;
  for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
    tau::TauUserEvent *ue = (*it2);
    int tid = 0;
    for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
      if (ue && ue->GetNumEvents(tid) == 0) continue;
      //if (ue && ue->GetWriteAsMetric()) continue;
      numEvents = ue->GetNumEvents(tid);
      tmp_str << "TAU::" << tid << "::NumEvents::" << ue->GetName();
      SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_INT, &numEvents);
      tmp_str.str(std::string());
      max = ue->GetMax(tid);
      tmp_str << "TAU::" << tid << "::Max::" << ue->GetName();
      SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &max);
      tmp_str.str(std::string());
      min = ue->GetMin(tid);
      tmp_str << "TAU::" << tid << "::Min::" << ue->GetName();
      SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &min);
      tmp_str.str(std::string());
      mean = ue->GetMean(tid);
      tmp_str << "TAU::" << tid << "::Mean::" << ue->GetName();
      SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &mean);
      tmp_str.str(std::string());
      sumsqr = ue->GetSumSqr(tid);
      tmp_str << "TAU::" << tid << "::SumSqr::" << ue->GetName();
      SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &sumsqr);
      tmp_str.str(std::string());
    }
  }
  if (tau::TheEventDB().size() > ue_count) {
    keys_added = true;
    ue_count = tau::TheEventDB().size();
  }
  if ((ue_count + fi_count) > SOS_DEFAULT_ELEM_MAX) {
      TAU_VERBOSE("DANGER, WILL ROBINSON! EXCEEDING MAX ELEMENTS IN SOS. Bad things might happen?\n");
  }
  RtsLayer::UnLockDB();
  if (keys_added) {
      TAU_VERBOSE("[TAU_SOS_send_data]: Announcing the pub...\n");
      //SOS_announce(tau_sos_pub);
      TAU_VERBOSE("[TAU_SOS_send_data]:   ...done.\n");
  }
  TAU_VERBOSE("[TAU_SOS_send_data]: Publishing the values...\n");
  TAU_VERBOSE("MY RANK IS: %d/%d\n", _runtime->config.comm_rank, _runtime->config.comm_size);
  SOS_publish(tau_sos_pub);
  TAU_VERBOSE("[TAU_SOS_send_data]:   ...done.\n");
  Tau_global_decr_insideTAU();
}

