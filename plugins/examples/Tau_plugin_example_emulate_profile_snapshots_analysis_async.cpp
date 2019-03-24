/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <utility>

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

#include <Profile/TauTrace.h>

MPI_Comm comm;
MPI_Comm newcomm;
pthread_t worker_thread;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
int period_microseconds = 2000000;
bool _threaded = true;

int analytics_complete = 1;
sem_t mutex;

typedef struct snapshot_buffer {
  double ***gExcl, ***gIncl;
  double_int **gExcl_min, **gIncl_min;
  double_int **gExcl_max, **gIncl_max;
  double **gNumCalls, **gNumSubr;
  double ***sExcl, ***sIncl;
  double **sNumCalls, **sNumSubr;
  double **gAtomicMin, **gAtomicMax;
  double_int *gAtomicMin_min, *gAtomicMax_max;
  double **gAtomicCalls, **gAtomicMean;
  double **gAtomicSumSqr;
  double **sAtomicMin, **sAtomicMax;
  double **sAtomicCalls, **sAtomicMean;
  double **sAtomicSumSqr;
  Tau_unify_object_t *functionUnifier;
  Tau_unify_object_t *atomicUnifier;
  int *numEventThreads;
  int *globalEventMap;
  int *numAtomicEventThreads;
  int *globalAtomicEventMap;
  std::vector <int> top_5_excl_time_mean;
} snapshot_buffer_t;

#define N_SNAPSHOTS 2000
snapshot_buffer_t s_buffer[N_SNAPSHOTS]; //Store upto N_SNAPSHOTS snapshots

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

void Tau_stop_worker(void) {

    fprintf(stderr, "Entering stop worker routine...\n");
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
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

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  //sem_destroy(&mutex);

  return 0;
}

void * Tau_plugin_threaded_analytics(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;
    int dummy_array[100];
    int min_array[100];
    int max_array[100];
    int sum_array[100];

    int rank;

    PMPI_Comm_rank(newcomm, &rank);

    while (!done) {
        // wait x microseconds for the next batch.
        gettimeofday(&tp, NULL);
        fprintf(stderr, "Inside thread...\n");
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
          for(int i = 0 ; i < 100; i++) {
               dummy_array[i] = i*2;
          }

          PMPI_Reduce(dummy_array, min_array, 100, MPI_INT, MPI_MIN, 0, newcomm);
          PMPI_Reduce(dummy_array, max_array, 100, MPI_INT, MPI_MAX, 0, newcomm);
          PMPI_Reduce(dummy_array, sum_array, 100, MPI_INT, MPI_SUM, 0, newcomm);
   
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

