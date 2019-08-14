/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <pthread.h>

#ifdef TAU_PAPI
#include "papi.h"

namespace tau {
    namespace papi_plugin {
        /* Simple class to aid in converting/storing component event data */
        class papi_event {
            public:
                papi_event(const char * ename, const char * eunits, int ecode, int data_type) : 
                    name(ename), units(eunits), code(ecode), type(data_type), conversion(1.0) {}
                std::string name;
                std::string units;
                int code;
                int type;
                double conversion;
        };
        /* Simple class to aid in processing PAPI components */
        class papi_component {
            public:
                papi_component(int cid, const PAPI_component_info_t *cinfo) : 
                    event_set(PAPI_NULL), initialized(false), id(cid), info(cinfo) {}
                std::vector<papi_event> events;
                int event_set;
                bool initialized;
                int id;
                const PAPI_component_info_t *info;
        };
    }
}

typedef tau::papi_plugin::papi_component ppc;
typedef tau::papi_plugin::papi_event ppe;

std::vector<ppc*> components;

pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
bool done;

void initialize_papi_events(void) {
    PapiLayer::initializePapiLayer();
    int num_components = PAPI_num_components();
    const PAPI_component_info_t *comp_info;
    int retval = PAPI_OK;
    // are there any components?
    for (int component_id = 0 ; component_id < num_components ; component_id++) {
        comp_info = PAPI_get_component_info(component_id);
        if (comp_info == NULL) {
            fprintf(stderr, "PAPI component info unavailable, no power measurements will be done.\n");
            return;
        }
        ppc * comp = new ppc(component_id, comp_info);
        /* Skip the perf_event component, that's standard PAPI */
        if (strstr(comp_info->name, "perf_event") != NULL) {
            continue;
        }
        printf("Found %s component...\n", comp_info->name);
        /* Does this component have available events? */
        if (comp_info->num_native_events == 0) {
            fprintf(stderr, "No %s events found.\n", comp_info->name);
            if (comp_info->disabled != 0) {
                fprintf(stderr, "%s.\n", comp_info->disabled_reason);
            }        
            continue;
        }
        /* Construct the event set and populate it */
        retval = PAPI_create_eventset(&comp->event_set);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error creating PAPI eventset for %s component.\n", comp_info->name);
            continue;
        }
        int code = PAPI_NATIVE_MASK;
        int event_modifier = PAPI_ENUM_FIRST;
        for ( int ii=0; ii< comp_info->num_native_events; ii++ ) {
            // get the event
            retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
            event_modifier = PAPI_ENUM_EVENTS;
            if ( retval != PAPI_OK ) {
                fprintf( stderr, "%s %d %s %d\n", __FILE__,
                        __LINE__, "PAPI_event_code_to_name", retval );
                continue;
            }
            // get the event name
            char event_name[PAPI_MAX_STR_LEN];
            retval = PAPI_event_code_to_name( code, event_name );
            if (retval != PAPI_OK) {
                fprintf(stderr, "%s %d %s %d\n", __FILE__,
                        __LINE__, "Error getting event name\n",retval);
                continue;
            }
            // skip the counter events...
            // if (strstr(event_name, "_CNT") != NULL) { continue; }
            // get the event info
            PAPI_event_info_t evinfo;
            retval = PAPI_get_event_info(code,&evinfo);
            if (retval != PAPI_OK) {
                fprintf(stderr, "%s %d %s %d\n", __FILE__,
                        __LINE__, "Error getting event info\n",retval);
                continue;
            }
            // get the event units
            char unit[PAPI_MAX_STR_LEN] = {0};
            strncpy(unit,evinfo.units,PAPI_MAX_STR_LEN);
            // save the event info
            //printf("Found event '%s (%s)'\n", event_name, unit);
            ppe this_event(event_name, unit, code, evinfo.data_type);
            if(strcmp(unit, "nJ") == 0) {
                this_event.units = "J";
                this_event.conversion = 1.0e-9;
            }
            if(strcmp(unit, "mW") == 0) {
                this_event.units = "W";
                this_event.conversion = 1.0e-3;
            }
            if(this_event.units.size() > 0) {
                std::stringstream ss;
                ss << this_event.name << " (" 
                   << this_event.units << ")";
                this_event.name = ss.str();
            }
            retval = PAPI_add_event(comp->event_set, code);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error adding RAPL event.\n");
                return;
            }
            comp->events.push_back(std::move(this_event));
        }
        /* Start the event set */
        retval = PAPI_start(comp->event_set);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error starting PAPI eventset.\n");
            return;
        }
        comp->initialized = true;
        components.push_back(comp);
    }
}

void read_papi_components(void) {
    for (size_t index = 0; index < components.size() ; index++) {
        if (components[index]->initialized) {
            ppc * comp = components[index];
            long long * values = (long long *)calloc(comp->events.size(), sizeof(long long));
            int retval = PAPI_read(comp->event_set, values);
            if (retval != PAPI_OK) {
                fprintf(stderr, "Error reading PAPI RAPL eventset.\n");
                return;
            }
            for (size_t i = 0 ; i < comp->events.size() ; i++) {
                void * ue = Tau_get_userevent(comp->events[i].name.c_str());
                Tau_userevent_thread(ue, ((double)values[i]) * comp->events[i].conversion, 0);
            }
            free(values);
        }
    }
    return;
}

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

int Tau_plugin_event_end_of_execution_papi_component(Tau_plugin_event_end_of_execution_data_t *data) {
    printf("PAPI Component PLUGIN %s\n", __func__);
    stop_worker();
    //pthread_cond_destroy(&_my_cond);
    //pthread_mutex_destroy(&_my_mutex);
    return 0;
}

int Tau_plugin_event_trigger_papi_component(Tau_plugin_event_trigger_data_t* data) {
    printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_dump_papi_component(Tau_plugin_event_dump_data_t* data) {
    printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_metadata_registration_complete_papi_component(Tau_plugin_event_metadata_registration_data_t* data) {
    //printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_papi_component(Tau_plugin_event_post_init_data_t* data) {
    printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}  

int Tau_plugin_event_send_papi_component(Tau_plugin_event_send_data_t* data) {
    printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_recv_papi_component(Tau_plugin_event_recv_data_t* data) {
    printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_entry_papi_component(Tau_plugin_event_function_entry_data_t* data) {
    //printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_exit_papi_component(Tau_plugin_event_function_exit_data_t* data) {
    //printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_atomic_trigger_papi_component(Tau_plugin_event_atomic_event_trigger_data_t* data) {
    //printf("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

void * Tau_papi_component_plugin_threaded_function(void* data) {
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
            printf("%d Timeout from plugin.\n", RtsLayer::myNode()); fflush(stderr);
            read_papi_components();
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

    /* get ready to read metrics! */
    initialize_papi_events();

    done = false;
    init_lock(&_my_mutex);
    printf("Spawning thread.\n");
    int ret = pthread_create(&worker_thread, NULL,
        &Tau_papi_component_plugin_threaded_function, NULL);
    if (ret != 0) {
        errno = ret;
        perror("Error: pthread_create (1) fails\n");
        exit(1);
    }

    /* Required event support */
    cb->Trigger = Tau_plugin_event_trigger_papi_component;
    cb->Dump = Tau_plugin_event_dump_papi_component;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_papi_component;
    cb->PostInit = Tau_plugin_event_post_init_papi_component;
    cb->EndOfExecution = Tau_plugin_event_end_of_execution_papi_component;

    /* Trace events */
    cb->Send = Tau_plugin_event_send_papi_component;
    cb->Recv = Tau_plugin_event_recv_papi_component;
    cb->FunctionEntry = Tau_plugin_event_function_entry_papi_component;
    cb->FunctionExit = Tau_plugin_event_function_exit_papi_component;
    cb->AtomicEventTrigger = Tau_plugin_event_atomic_trigger_papi_component;

    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

    return 0;
}

#endif // ifdef TAU_PAPI
