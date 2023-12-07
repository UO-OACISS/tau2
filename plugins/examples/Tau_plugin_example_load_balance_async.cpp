/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifdef TAU_USE_STDCXX11

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>

#include <pthread.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>

#ifdef TAU_MPI
  #include <mpi.h>

pthread_t worker_thread;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
int period_microseconds = 2000000;
bool _threaded = true;


#define N_SNAPSHOTS 2000
int imbalance_history[N_SNAPSHOTS];
int index_ = 0;

int latest_work = 0;
int should_rebalance = 0;

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  #ifdef TAU_MPI
  #endif

  return 0;
}

int done = 0;

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

void * Tau_plugin_threaded_analytics(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;

    int rank; int size;
    int global_min, global_max;
    int global_sum; float sum_, avg_, min_, max_;

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
        } else if (rc == EINVAL) {
            TAU_VERBOSE("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("Mutex not locked!\n"); fflush(stderr);
        }

  #ifdef TAU_MPI
  /* Core logic */

  PMPI_Reduce(&latest_work, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&latest_work, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  PMPI_Reduce(&latest_work, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  rank = RtsLayer::myNode();

  if(rank == 0) {
    sum_ = global_sum;
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    avg_ = (sum_ / size);
    min_ = global_min;
    max_ = global_max;

    if((max_ - min_) > 0.10 * avg_) {
      imbalance_history[index_] = 1;
    } else {
      imbalance_history[index_] = 0;
    }

    if(index_ > 5) {
     int count = 0;
     for(int i = index_ - 5; i < index_; i++) {
       count = count + imbalance_history[i];
     }

     if(count == 5) {
       should_rebalance = 1;
       fprintf(stderr, "Rebalancing as load imbalance exists for %d iterations..\n", count);
     } else {
       should_rebalance = 0;
     }
    }
  }

  PMPI_Bcast(&should_rebalance, 1, MPI_INT, 0, MPI_COMM_WORLD);

  #endif
    index_++;

    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    pthread_exit((void*)0L);
	return(NULL);
}

int Tau_plugin_event_trigger(Tau_plugin_event_trigger_data_t* data) {

  int local = *((int*)(data->data));

  latest_work = local;

  *((int*)(data->data)) = should_rebalance;

  return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->Trigger = Tau_plugin_event_trigger;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

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
