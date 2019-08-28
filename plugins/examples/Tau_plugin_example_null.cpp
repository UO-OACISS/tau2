/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <pthread.h>
#include <errno.h>

#ifdef TAU_MPI
#include <mpi.h>
#endif

pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
bool done;

void stop_worker(void) {
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    printf("TAU ADIOS2 thread joining...\n"); fflush(stderr);
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
}

int Tau_plugin_event_end_of_execution_null(Tau_plugin_event_end_of_execution_data_t *data) {
    printf("NULL PLUGIN %s\n", __func__);
    stop_worker();
    //pthread_cond_destroy(&_my_cond);
    //pthread_mutex_destroy(&_my_mutex);
    return 0;
}

int Tau_plugin_event_trigger_null(Tau_plugin_event_trigger_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_dump_null(Tau_plugin_event_dump_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_metadata_registration_complete_null(Tau_plugin_event_metadata_registration_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_null(Tau_plugin_event_post_init_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}  

int Tau_plugin_event_send_null(Tau_plugin_event_send_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_recv_null(Tau_plugin_event_recv_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_entry_null(Tau_plugin_event_function_entry_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_exit_null(Tau_plugin_event_function_exit_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_atomic_trigger_null(Tau_plugin_event_atomic_event_trigger_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

void * threaded_function(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;
    Tau_pure_start(__func__);

    while (!done) {
        // wait x microseconds for the next batch.
        gettimeofday(&tp, NULL);
        const int one_second = 1000000;
        // first, add the period to the current microseconds
        int tmp_usec = tp.tv_usec + one_second;
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
            //printf("%d Timeout from plugin.\n", RtsLayer::myNode()); fflush(stderr);
        } else if (rc == EINVAL) {
            printf("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            printf("Mutex not locked!\n"); fflush(stderr);
        }
    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    Tau_pure_start(__func__);
    pthread_exit((void*)0L);
	return(NULL);
}

void init_lock(pthread_mutex_t * _mutex) {
    pthread_mutexattr_t Attr;
    pthread_mutexattr_init(&Attr);
    pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
    int rc;
    if ((rc = pthread_mutex_init(_mutex, &Attr)) != 0) {
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

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    done = false;
    init_lock(&_my_mutex);
    printf("Spawning thread.\n");
    int ret = pthread_create(&worker_thread, NULL, &threaded_function, NULL);
    if (ret != 0) {
        errno = ret;
        perror("Error: pthread_create (1) fails\n");
        exit(1);
    }

    /* Required event support */
    cb->Trigger = Tau_plugin_event_trigger_null;
    cb->Dump = Tau_plugin_event_dump_null;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_null;
    cb->PostInit = Tau_plugin_event_post_init_null;
    cb->EndOfExecution = Tau_plugin_event_end_of_execution_null;

    /* Trace events */
    cb->Send = Tau_plugin_event_send_null;
    cb->Recv = Tau_plugin_event_recv_null;
    cb->FunctionEntry = Tau_plugin_event_function_entry_null;
    cb->FunctionExit = Tau_plugin_event_function_exit_null;
    cb->AtomicEventTrigger = Tau_plugin_event_atomic_trigger_null;

    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

    return 0;
}

