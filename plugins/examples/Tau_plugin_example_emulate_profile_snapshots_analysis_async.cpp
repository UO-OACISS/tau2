/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>

#ifdef TAU_USE_STDCXX11

#include <Profile/TauEnv.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauCollate.h>
#include <Profile/TauUtil.h>
#include <Profile/TauXML.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <semaphore.h>

#ifdef TAU_MPI
#include <mpi.h>

#include <list>
#include <vector>

#include <Profile/TauPlugin.h>
#include <Profile/TauMemory.h>

#include <Profile/TauTrace.h>
#include <sys/resource.h>

#include <sys/types.h>
#include <unistd.h>

MPI_Comm comm;
MPI_Comm newcomm;
pthread_t worker_thread;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
int period_microseconds = 2000000;
bool _threaded = true;

using namespace std;

typedef struct snapshot_buffer {
} snapshot_buffer_t;

#define N_SNAPSHOTS 2000
snapshot_buffer_t s_buffer[N_SNAPSHOTS]; //Store upto N_SNAPSHOTS snapshots

int done = 0;
pid_t process_id;
ifstream memusage;

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

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  //sem_destroy(&mutex);

  return 0;
}

void * Tau_plugin_threaded_analytics(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;

    int rank = RtsLayer::myNode();

    string line;

    while (!done) {
        // wait x microseconds for the next batch.
        gettimeofday(&tp, NULL);
        const int one_second = 1000000;
        // first, add the period to the current microseconds
        int tmp_usec = tp.tv_usec + period_microseconds;
        int flow_sec = 0;
        if (tmp_usec > one_second) { // did we overflow?
            flow_sec = tmp_usec / one_second; // how many seconds?
            tmp_usec = tmp_usec % one_second; // get the remainder
        }
        ts.tv_sec  = (tp.tv_sec + flow_sec);
        ts.tv_nsec = (1000 * tmp_usec);
        pthread_mutex_lock(&_my_mutex);
        int rc = pthread_cond_timedwait(&_my_cond, &_my_mutex, &ts);
        if (rc == ETIMEDOUT) {

           struct rusage r_usage;
           long result;
           getrusage(RUSAGE_SELF,&r_usage);
           PMPI_Reduce(&(r_usage.ru_maxrss), &result, 1, MPI_LONG, MPI_MAX, 0, newcomm);

           if(!rank)
             fprintf(stderr, "Max Memory usage = %ld\n", result);

        } else if (rc == EINVAL) {
            TAU_VERBOSE("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("Mutex not locked!\n"); fflush(stderr);
        }
    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    pthread_exit((void*)0L);
	return(NULL);
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

#ifdef TAU_MPI
  PMPI_Comm_dup(MPI_COMM_WORLD, &newcomm);
#endif

  init_lock();

  int ret = pthread_create(&worker_thread, NULL, &Tau_plugin_threaded_analytics, NULL);
            if (ret != 0) {
                errno = ret;
                perror("Error: pthread_create (1) fails\n");
                exit(1);
  }


  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

#endif
#endif
