/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <regex>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <pthread.h>

#ifdef TAU_MPI
#include "mpi.h"
#endif

#ifdef TAU_PAPI
#include "papi.h"
#else
#define PAPI_NULL -1
#endif

#include "json.h"
using json = nlohmann::json;
json configuration;

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
                papi_component(int cid) : 
                    event_set(PAPI_NULL), initialized(false), id(cid) {}
                std::vector<papi_event> events;
                int event_set;
                bool initialized;
                int id;
        };

        class CPUStat {
            public:
                CPUStat() : user(0LL), nice(0LL), system(0LL),
                    idle(0LL), iowait(0LL), irq(0LL), softirq(0LL),
                    steal(0LL), guest(0LL) {}
                char name[32];
                long long user;
                long long nice;
                long long system;
                long long idle;
                long long iowait;
                long long irq;
                long long softirq;
                long long steal;
                long long guest;
        };
    }
}

typedef tau::papi_plugin::papi_component ppc;
typedef tau::papi_plugin::papi_event ppe;
typedef tau::papi_plugin::CPUStat cpustats_t;
std::vector<ppc*> components;

std::vector<cpustats_t*> * previous_stats;

pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
bool done;
int rank_getting_system_data;
int my_rank = 0;

void * find_user_event(const std::string& name) {
    void * ue = NULL;
    /* I can't believe I need a local map to do this... */
    static std::map<std::string, void*> event_map;
    auto search = event_map.find(name);
    if (search == event_map.end()) {
        ue = Tau_get_userevent(name.c_str());
        event_map.insert({name, ue});
    } else {
        ue = search->second;
    }
    return ue;
}

bool include_event(const char * component, const char * event_name) {
    if (configuration.count(component)) {
        auto json_component = configuration[component];
        if (json_component.count("include")) {
            bool found = false;
            auto json_include = json_component["include"];
            for (auto i : json_include) {
                try {
                    std::string needle(i);
                    std::regex re(needle);
                    std::string haystack(event_name);
                    if (std::regex_search(haystack, re)) {
                        //std::cout << "including " << event_name << std::endl;
                        return true;
                    }
                } catch (std::regex_error& e) {
                    std::cerr << "Error in regular expression: " << i << std::endl;
                }
            }
            if (!found) { return false; }
        }
        if (json_component.count("exclude")) {
            auto json_include = json_component["exclude"];
            for (auto i : json_include) {
                try {
                    std::string needle(i);
                    std::regex re(needle);
                    std::string haystack(event_name);
                    if (std::regex_search(haystack, re)) {
                        //std::cout << "excluding " << event_name << std::endl;
                        return false;
                    }
                } catch (std::regex_error& e) {
                    std::cerr << "Error in regular expression: " << i << std::endl;
                }
            }
        }
    }
    return true;
}

#ifdef TAU_PAPI
void initialize_papi_events(void) {
    PapiLayer::initializePapiLayer();
    int num_components = PAPI_num_components();
    const PAPI_component_info_t *comp_info;
    int retval = PAPI_OK;
    // are there any components?
    for (int component_id = 0 ; component_id < num_components ; component_id++) {
        comp_info = PAPI_get_component_info(component_id);
        if (comp_info == NULL) {
            TAU_VERBOSE("Error: PAPI component info unavailable, no power measurements will be done.\n");
            return;
        }
        ppc * comp = new ppc(component_id);
        /* Skip the perf_event component, that's standard PAPI */
        if (strstr(comp_info->name, "perf_event") != NULL) {
            continue;
        }
        TAU_VERBOSE("Found %s component...\n", comp_info->name);
        /* Does this component have available events? */
        if (comp_info->num_native_events == 0) {
            TAU_VERBOSE("Error: No %s events found.\n", comp_info->name);
            if (comp_info->disabled != 0) {
                TAU_VERBOSE("Error: %s.\n", comp_info->disabled_reason);
            }        
            continue;
        }
        /* Construct the event set and populate it */
        retval = PAPI_create_eventset(&comp->event_set);
        if (retval != PAPI_OK) {
            TAU_VERBOSE("Error: Error creating PAPI eventset for %s component.\n", comp_info->name);
            continue;
        }
        int code = PAPI_NATIVE_MASK;
        int event_modifier = PAPI_ENUM_FIRST;
        for ( int ii=0; ii< comp_info->num_native_events; ii++ ) {
            // get the event
            retval = PAPI_enum_cmp_event( &code, event_modifier, component_id );
            event_modifier = PAPI_ENUM_EVENTS;
            if ( retval != PAPI_OK ) {
                TAU_VERBOSE("Error: %s %d %s %d\n", __FILE__,
                        __LINE__, "PAPI_event_code_to_name", retval );
                continue;
            }
            // get the event name
            char event_name[PAPI_MAX_STR_LEN];
            retval = PAPI_event_code_to_name( code, event_name );
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: %s %d %s %d\n", __FILE__,
                        __LINE__, "Error getting event name\n",retval);
                continue;
            }
            if (!include_event(comp_info->name, event_name)) {
                continue;
            }
            // get the event info
            PAPI_event_info_t evinfo;
            retval = PAPI_get_event_info(code,&evinfo);
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: %s %d %s %d\n", __FILE__,
                        __LINE__, "Error getting event info\n",retval);
                continue;
            }
            // get the event units
            char unit[PAPI_MAX_STR_LEN] = {0};
            strncpy(unit,evinfo.units,PAPI_MAX_STR_LEN);
            // save the event info
            TAU_VERBOSE("Found event '%s (%s)'\n", event_name, unit);
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
                TAU_VERBOSE("Error: Error adding PAPI %s event %s.\n", comp_info->name, event_name);
                return;
            }
            comp->events.push_back(std::move(this_event));
        }
        /* Start the event set */
        retval = PAPI_start(comp->event_set);
        if (retval != PAPI_OK) {
            TAU_VERBOSE("Error: Error starting PAPI eventset.\n");
            return;
        }
        comp->initialized = true;
        components.push_back(comp);
    }
}
#endif

std::vector<cpustats_t*> * read_cpu_stats() {
    std::vector<cpustats_t*> * cpu_stats = new std::vector<cpustats_t*>();
    /*  Reading proc/stat as a file  */
    FILE * pFile;
    char line[128];
    char dummy[32];
    pFile = fopen ("/proc/stat","r");
    if (pFile == nullptr) {
        perror ("Error opening file");
        return NULL;
    } else {
        while ( fgets( line, 128, pFile)) {
            if ( strncmp (line, "cpu", 3) == 0 ) {
                cpustats_t * cpu_stat = new(cpustats_t);
                /*  Note, this will only work on linux 2.6.24 through 3.5  */
                sscanf(line, "%s %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
                       cpu_stat->name, &cpu_stat->user, &cpu_stat->nice,
                       &cpu_stat->system, &cpu_stat->idle,
                       &cpu_stat->iowait, &cpu_stat->irq, &cpu_stat->softirq,
                       &cpu_stat->steal, &cpu_stat->guest);
                if (!include_event("/proc/stat", cpu_stat->name)) {
                    continue;
                }
                cpu_stats->push_back(cpu_stat);
            }
        }
    }
    return cpu_stats;
}

int choose_volunteer_rank() {
#ifdef TAU_MPI
    // figure out who should get system stats for this node
    int i;
    my_rank = 0;
    int comm_size = 1;
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // get my hostname
    const int hostlength = 128;
    char hostname[hostlength] = {0};
    gethostname(hostname, sizeof(char)*hostlength);
    // make array for all hostnames
    char * allhostnames = (char*)calloc(hostlength * comm_size, sizeof(char));
    // copy my name into the big array
    char * host_index = allhostnames + (hostlength * my_rank);
    strncpy(host_index, hostname, hostlength);
    // get all hostnames
    PMPI_Allgather(hostname, hostlength, MPI_CHAR, allhostnames, 
                   hostlength, MPI_CHAR, MPI_COMM_WORLD);
    int volunteer = 0;
    // point to the head of the array
    host_index = allhostnames;
    // find the lowest rank with my hostname
    for (i = 0 ; i < comm_size ; i++) {
        //printf("%d:%d comparing '%s' to '%s'\n", rank, size, hostname, host_index);
        if (strncmp(hostname, host_index, hostlength) == 0) {
            volunteer = i;
            break;
        }
        host_index = host_index + hostlength;
    }
    free(allhostnames);
    return volunteer;
#else
    return 0;
#endif
}

void sample_value(const char * cpu, const char * name, const double value, const long long total) {
    std::stringstream ss;
    ss << cpu << ":" << name;
    void * ue = find_user_event(ss.str());
    // double-check the value...
    double tmp;
    if (total == 0LL) {
        tmp = 0.0;
    } else {
        tmp = (value / (double)(total)) * 100.0;
    }
    Tau_userevent_thread(ue, tmp, 0);
}

void update_cpu_stats(void) {
    /* get the current stats */
    std::vector<cpustats_t*> * new_stats = read_cpu_stats();
    if (new_stats == NULL) return;
    for (int i = 0 ; i < new_stats->size() ; i++) {
        /* we need to take the difference from the last read */
        cpustats_t diff;
        diff.user = (*new_stats)[i]->user - (*previous_stats)[i]->user;
        diff.nice = (*new_stats)[i]->nice - (*previous_stats)[i]->nice;
        diff.system = (*new_stats)[i]->system - (*previous_stats)[i]->system;
        diff.idle = (*new_stats)[i]->idle - (*previous_stats)[i]->idle;
        diff.iowait = (*new_stats)[i]->iowait - (*previous_stats)[i]->iowait;
        diff.irq = (*new_stats)[i]->irq - (*previous_stats)[i]->irq;
        diff.softirq = (*new_stats)[i]->softirq - (*previous_stats)[i]->softirq;
        diff.steal = (*new_stats)[i]->steal - (*previous_stats)[i]->steal;
        diff.guest = (*new_stats)[i]->guest - (*previous_stats)[i]->guest;
        double total = (double)(diff.user + diff.nice + diff.system +
                diff.idle + diff.iowait + diff.irq + diff.softirq +
                diff.steal + diff.guest);
        long long lltotal = (diff.user + diff.nice + diff.system +
                diff.idle + diff.iowait + diff.irq + diff.softirq +
                diff.steal + diff.guest);
        sample_value((*new_stats)[i]->name, " User %",     (double)(diff.user), total);
        sample_value((*new_stats)[i]->name, " Nice %",     (double)(diff.nice), total);
        sample_value((*new_stats)[i]->name, " System %",   (double)(diff.system), total);
        sample_value((*new_stats)[i]->name, " Idle %",     (double)(diff.idle), total);
        sample_value((*new_stats)[i]->name, " I/O Wait %", (double)(diff.iowait), total);
        sample_value((*new_stats)[i]->name, " IRQ %",      (double)(diff.irq), total);
        sample_value((*new_stats)[i]->name, " soft IRQ %", (double)(diff.softirq), total);
        sample_value((*new_stats)[i]->name, " Steal %",    (double)(diff.steal), total);
        sample_value((*new_stats)[i]->name, " Guest %",    (double)(diff.guest), total);
    }
    delete(previous_stats);
    previous_stats = new_stats;
}

void read_papi_components(void) {
    Tau_pure_start(__func__);
#ifdef TAU_PAPI
    for (size_t index = 0; index < components.size() ; index++) {
        if (components[index]->initialized) {
            ppc * comp = components[index];
            long long * values = (long long *)calloc(comp->events.size(), sizeof(long long));
            int retval = PAPI_read(comp->event_set, values);
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: Error reading PAPI RAPL eventset.\n");
                return;
            }
            for (size_t i = 0 ; i < comp->events.size() ; i++) {
                void * ue = find_user_event(comp->events[i].name);
                Tau_userevent_thread(ue, ((double)values[i]) * comp->events[i].conversion, 0);
            }
            free(values);
        }
    }
#endif
    /* Also read some OS level metrics. */

    /* records the heap, with no context, even though it says "here". */
    Tau_track_memory_here();
#if !defined(__APPLE__)
    /* records the rss/hwm, without context. */
    Tau_track_memory_rss_and_hwm();
#endif

    if (my_rank == rank_getting_system_data) {
#if !defined(__APPLE__)
        /* records the load, without context */
        Tau_track_load();
#endif
        /* records the power, without context */
        Tau_track_power();
#if !defined(__APPLE__)
        /* Get the current CPU statistics for the node */
        update_cpu_stats();
#endif
    }

    Tau_pure_stop(__func__);
    return;
}

#ifdef TAU_PAPI
void free_papi_components(void) {
    for (size_t index = 0; index < components.size() ; index++) {
        ppc * comp = components[index];
        if (comp->initialized) {
            long long * values = (long long *)calloc(comp->events.size(), sizeof(long long));
            int retval = PAPI_stop(comp->event_set, values);
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: Error reading PAPI RAPL eventset.\n");
                return;
            }
            free(values);
            /* Done, clean up */
            retval = PAPI_cleanup_eventset(comp->event_set);
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: %s %d %s %d\n", __FILE__, __LINE__,
                        "PAPI_cleanup_eventset()",retval);
            }

            retval = PAPI_destroy_eventset(&(comp->event_set));
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: %s %d %s %d\n", __FILE__, __LINE__,
                        "PAPI_destroy_eventset()",retval);
            }
            comp->initialized = false;
        }
        delete(comp);
    }
    components.clear();
}
#endif

void stop_worker(void) {
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    TAU_VERBOSE("TAU ADIOS2 thread joining...\n"); fflush(stderr);
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
            TAU_VERBOSE("%d Timeout from plugin.\n", RtsLayer::myNode()); fflush(stderr);
            if (!done) {
                read_papi_components();
            }
        } else if (rc == EINVAL) {
            TAU_VERBOSE("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("Mutex not locked!\n"); fflush(stderr);
        }
    }

    /* Get an exit reading */
    read_papi_components();

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

int Tau_plugin_event_end_of_execution_papi_component(Tau_plugin_event_end_of_execution_data_t *data) {
    TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    stop_worker();
#ifdef TAU_PAPI
    /* clean up papi */
    if (my_rank == rank_getting_system_data) {
        free_papi_components();
    }
#endif
    /* Why do these deadlock on exit? */
    //pthread_cond_destroy(&_my_cond);
    //pthread_mutex_destroy(&_my_mutex);
    return 0;
}

int Tau_plugin_event_trigger_papi_component(Tau_plugin_event_trigger_data_t* data) {
    TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_dump_papi_component(Tau_plugin_event_dump_data_t* data) {
    TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_metadata_registration_complete_papi_component(Tau_plugin_event_metadata_registration_data_t* data) {
    //TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_papi_component(Tau_plugin_event_post_init_data_t* data) {
    TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);

    rank_getting_system_data = choose_volunteer_rank();

    if (my_rank == rank_getting_system_data) {
#ifdef TAU_PAPI
        /* get ready to read metrics! */
        initialize_papi_events();
#endif
#if !defined(__APPLE__)
        previous_stats = read_cpu_stats();
#endif
    }
    /* spawn the worker thread to do the reading */
    init_lock(&_my_mutex);
    TAU_VERBOSE("Spawning thread.\n");
    int ret = pthread_create(&worker_thread, NULL,
        &Tau_papi_component_plugin_threaded_function, NULL);
    if (ret != 0) {
        errno = ret;
        perror("Error: pthread_create (1) fails\n");
        exit(1);
    }
    return 0;
}  

int Tau_plugin_event_send_papi_component(Tau_plugin_event_send_data_t* data) {
    TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_recv_papi_component(Tau_plugin_event_recv_data_t* data) {
    TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_entry_papi_component(Tau_plugin_event_function_entry_data_t* data) {
    //TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_function_exit_papi_component(Tau_plugin_event_function_exit_data_t* data) {
    //TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_atomic_trigger_papi_component(Tau_plugin_event_atomic_event_trigger_data_t* data) {
    //TAU_VERBOSE("PAPI Component PLUGIN %s\n", __func__);
    return 0;
}

void read_config_file(void) {
    try {
            std::ifstream cfg("components.json");
            cfg >> configuration;
            cfg.close();
        } catch (...) {
            // fail silently, nothing to do
        }
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    done = false;

    read_config_file();

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

