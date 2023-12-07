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
#include <set>
#include <vector>
#include <regex>
#include <algorithm>
#include <math.h>

#ifdef TAU_MPI
#include "mpi.h"
#endif

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>
#include <pthread.h>

#ifdef TAU_PAPI
#include "papi.h"
#else
#define PAPI_NULL -1
#endif

#include "Tau_scoped_timer.h"
#include "json.h"
using json = nlohmann::json;
json configuration;

#ifdef CUPTI
#include "tau_nvml.hpp"
#endif

#ifdef TAU_ROCM_SMI
#include "tau_rocm_smi.hpp"
#endif

#define ONE_BILLION  1e9
#define ONE_MILLION  1e6
#define ONE_BILLIONF 1000000000.0

inline void _plugin_assert(const char* expression, const char* file, int line)
{
    fprintf(stderr, "Assertion '%s' failed, file '%s' line '%d' on node '%d', thread '%d'.",
        expression, file, line, RtsLayer::myNode(), RtsLayer::myThread());
    abort();
}

#ifdef NDEBUG
#define PLUGIN_ASSERT(EXPRESSION) ((void)0)
#else
#define PLUGIN_ASSERT(EXPRESSION) ((EXPRESSION) ? (void)0 : \
    _plugin_assert(#EXPRESSION, __FILE__, __LINE__))
#endif

bool& main_thread() {
    static thread_local bool thebool = false;
    return thebool;
}

/* Provide a default configuration,
 * to avoid collecting too much data by default */

const char * default_configuration = R"(
{
  "periodic": false,
  "node_data_from_all_ranks": false,
  "monitor_counter_prefix": "",
  "periodicity seconds": 10.0,
  "scatterplot": false,
  "PAPI metrics": [],
  "/proc/stat": {
    "disable": false,
    "comment": "This will exclude all core-specific readings.",
    "exclude": ["^cpu[0-9]+.*"]
  },
  "/proc/meminfo": {
    "disable": false,
    "comment": "This will include three readings.",
    "include": [".*MemAvailable.*", ".*MemFree.*", ".*MemTotal.*"]
  },
  "/proc/net/dev": {
    "disable": false,
    "comment": "This will include only the first ethernet device.",
    "include": [".*eno1.*"]
  },
  "/proc/self/net/dev": {
    "disable": true,
    "comment": "This will include only the first ethernet device (from network namespace of which process is a member).",
    "include": [".*eno1.*"]
  },
  "lmsensors": {
    "disable": true,
    "comment": "This will include all power readings.",
    "include": [".*power.*"]
  },
  "net": {
    "disable": true,
    "comment": "This will include only the first ethernet device.",
    "include": [".*eno1.*"]
  },
  "nvml": {
    "disable": true,
    "comment": "This will include only the utilization metrics.",
    "include": [".*utilization.*"]
  },
  "rsmi": {
    "disable": false,
    "comment": "This will include only the utilization metrics.",
    "include": [".*Device Busy.*", ".*Memory Activity.*"]
  }
}
)";

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
                papi_component(const char * cname, int cid) :
                    name(cname), event_set(PAPI_NULL), initialized(false), id(cid) {}
                std::string name;
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
                char name[32] = {0};
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

        class NetStat {
            public:
                NetStat() : recv_bytes(0LL), recv_packets(0LL),
                    recv_errors(0LL), recv_drops(0LL), recv_fifo(0LL),
                    recv_frame(0LL), recv_compressed(0LL), recv_multicast(0LL),
                    transmit_bytes(0LL), transmit_packets(0LL),
                    transmit_errors(0LL), transmit_drops(0LL),
                    transmit_fifo(0LL), transmit_collisions(0LL),
                    transmit_carrier(0LL), transmit_compressed(0LL) {}
                char name[32] = {0};
                long long recv_bytes;
                long long recv_packets;
                long long recv_errors;
                long long recv_drops;
                long long recv_fifo;
                long long recv_frame;
                long long recv_compressed;
                long long recv_multicast;
                long long transmit_bytes;
                long long transmit_packets;
                long long transmit_errors;
                long long transmit_drops;
                long long transmit_fifo;
                long long transmit_collisions;
                long long transmit_carrier;
                long long transmit_compressed;
        };


        // trim from left
        inline std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
        {
            s.erase(0, s.find_first_not_of(t));
            return s;
        }

        // trim from right
        inline std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
        {
            s.erase(s.find_last_not_of(t) + 1);
            return s;
        }

        // trim from left & right
        inline std::string& trim(std::string& s, const char* t = " \t\n\r\f\v")
        {
            return ltrim(rtrim(s, t), t);
        }

    }
}

typedef tau::papi_plugin::papi_component ppc;
typedef tau::papi_plugin::papi_event ppe;
typedef tau::papi_plugin::CPUStat cpustats_t;
typedef tau::papi_plugin::NetStat netstats_t;
typedef std::map<std::string, long long> iostats_t;

// Globals should be defined static so they can't be seen outside this compilation unit (i.e. the plugin library)
static std::vector<ppc*> components;
#ifdef TAU_PAPI
static int papi_periodic_event_set = {PAPI_NULL};
static long long * papi_periodic_values;
static size_t num_metrics = 0;
#endif

static std::vector<cpustats_t*> * previous_cpu_stats = nullptr;
static std::map<std::string, netstats_t*> * previous_net_stats = nullptr;
static std::map<std::string, netstats_t*> * previous_self_net_stats = nullptr;
static iostats_t * previous_io_stats = nullptr;
static bool _attached{true};

static pthread_mutex_t _my_mutex; // for initialization, termination
static pthread_cond_t _my_cond; // for timer
static pthread_t worker_thread;
static std::atomic<bool> done{false};
static std::atomic<bool> worker_working{false};
static int rank_getting_system_data;
static std::stringstream csv_output;
static uint64_t periodic_index;
#ifdef CUPTI
tau::nvml::monitor& get_nvml_reader() {
    static tau::nvml::monitor nvml_reader;
    return nvml_reader;
}
#endif
#ifdef TAU_ROCM_SMI
tau::rsmi::monitor& get_rsmi_reader() {
    static tau::rsmi::monitor rsmi_reader;
    return rsmi_reader;
}
#endif

int& get_my_rank() {
    static int my_rank = 0;
    return my_rank;
}

void * find_user_event(const std::string& name) {
    void * ue = NULL;
    /* I can't believe I need a local map to do this... */
    static std::map<std::string, void*> event_map;
    std::string tmpname{
        configuration.count("monitor_counter_prefix") > 0 ?
        configuration["monitor_counter_prefix"] : ""};
    tmpname += name;
    auto search = event_map.find(tmpname);
    if (search == event_map.end()) {
        ue = Tau_get_userevent(tmpname.c_str());
        event_map.insert({tmpname, ue});
    } else {
        ue = search->second;
    }
    return ue;
}

void write_scatterplot_point(const std::string& name, double value) {
    if (configuration.count("scatterplot") > 0) {
        if (configuration["scatterplot"]) {
            csv_output << get_my_rank() << ",\"" << name << "\","
                << periodic_index << "," << value << "\n";
        }
    }
}

void sample_user_event(const std::string& name, double value) {
    if (TauEnv_get_thread_per_gpu_stream()) {
        std::string tmpname{
            configuration.count("monitor_counter_prefix") > 0 ?
            configuration["monitor_counter_prefix"] : ""};
        tmpname += name;
        Tau_trigger_userevent(tmpname.c_str(), value);
    } else {
        void * ue = find_user_event(name);
        Tau_userevent_thread(ue, value, 0);
    }
    write_scatterplot_point(name, value);
}

/* Older versions of Clang++ won't compile this regular expression
 * code.  Apple implementations will never use this function, because
 * they don't have PAPI support or /proc filesystem.  So it's safe
 * to just give it a dummy implementation. */
#if defined(__APPLE__)
bool include_event(const char * component, const char * event_name) {
    return false;
}
bool include_component(const char * component) {
    return false;
}
#else
bool include_event(const char * component, const char * event_name) {
    if (configuration.count(component)) {
        auto json_component = configuration[component];
        if (json_component.count("include")) {
            auto json_include = json_component["include"];
            for (auto i : json_include) {
                std::string needle(i);
                needle.erase(std::remove(needle.begin(),needle.end(),'\"'),needle.end());
                try {
                    std::regex re(needle);
                    std::string haystack(event_name);
                    if (std::regex_search(haystack, re)) {
                        //std::cout << "including " << event_name << std::endl;
                        return true;
                    }
                } catch (std::regex_error& e) {
// PGI will barf on the otherwise fine regular expression,
// no need to write out a ton of errors
#if !defined(__PGI)
                    std::cerr << "Error: '" << e.what() << "' in regular expression: " << needle << std::endl;
                    switch (e.code()) {
                        case std::regex_constants::error_collate:
                            std::cerr << "collate" << std::endl;
                            break;
                        case std::regex_constants::error_ctype:
                            std::cerr << "ctype" << std::endl;
                            break;
                        case std::regex_constants::error_escape:
                            std::cerr << "escape" << std::endl;
                            break;
                        case std::regex_constants::error_backref:
                            std::cerr << "backref" << std::endl;
                            break;
                        case std::regex_constants::error_brack:
                            std::cerr << "brack" << std::endl;
                            break;
                        case std::regex_constants::error_paren:
                            std::cerr << "paren" << std::endl;
                            break;
                        case std::regex_constants::error_brace:
                            std::cerr << "brace" << std::endl;
                            break;
                        case std::regex_constants::error_badbrace:
                            std::cerr << "badbrace" << std::endl;
                            break;
                        case std::regex_constants::error_range:
                            std::cerr << "range" << std::endl;
                            break;
                        case std::regex_constants::error_space:
                            std::cerr << "space" << std::endl;
                            break;
                        case std::regex_constants::error_badrepeat:
                            std::cerr << "badrepeat" << std::endl;
                            break;
                        case std::regex_constants::error_complexity:
                            std::cerr << "complexity" << std::endl;
                            break;
                        case std::regex_constants::error_stack:
                            std::cerr << "stack" << std::endl;
                            break;
                        default:
                            std::cerr << "unknown" << std::endl;
                            break;
                     }
#endif
                }
            }
            return false;
        }
        if (json_component.count("exclude")) {
            auto json_exclude = json_component["exclude"];
            for (auto i : json_exclude) {
                std::string needle(i);
                needle.erase(std::remove(needle.begin(),needle.end(),'\"'),needle.end());
                try {
                    std::regex re(needle);
                    std::string haystack(event_name);
                    if (std::regex_search(haystack, re)) {
                        //std::cout << "excluding " << event_name << std::endl;
                        return false;
                    }
                } catch (std::regex_error& e) {
// PGI will barf on the otherwise fine regular expression,
// no need to write out a ton of errors
#if !defined(__PGI)
                    std::cerr << "Error: '" << e.what() << "' in regular expression: " << needle << std::endl;
                    switch (e.code()) {
                        case std::regex_constants::error_collate:
                            std::cerr << "collate" << std::endl;
                            break;
                        case std::regex_constants::error_ctype:
                            std::cerr << "ctype" << std::endl;
                            break;
                        case std::regex_constants::error_escape:
                            std::cerr << "escape" << std::endl;
                            break;
                        case std::regex_constants::error_backref:
                            std::cerr << "backref" << std::endl;
                            break;
                        case std::regex_constants::error_brack:
                            std::cerr << "brack" << std::endl;
                            break;
                        case std::regex_constants::error_paren:
                            std::cerr << "paren" << std::endl;
                            break;
                        case std::regex_constants::error_brace:
                            std::cerr << "brace" << std::endl;
                            break;
                        case std::regex_constants::error_badbrace:
                            std::cerr << "badbrace" << std::endl;
                            break;
                        case std::regex_constants::error_range:
                            std::cerr << "range" << std::endl;
                            break;
                        case std::regex_constants::error_space:
                            std::cerr << "space" << std::endl;
                            break;
                        case std::regex_constants::error_badrepeat:
                            std::cerr << "badrepeat" << std::endl;
                            break;
                        case std::regex_constants::error_complexity:
                            std::cerr << "complexity" << std::endl;
                            break;
                        case std::regex_constants::error_stack:
                            std::cerr << "stack" << std::endl;
                            break;
                        default:
                            std::cerr << "unknown" << std::endl;
                            break;
                     }
#endif
                }
            }
        }
    }
    return true;
}
bool include_component(const char * component) {
    if (configuration.count(component)) {
        auto json_component = configuration[component];
        if (json_component.count("disable")) {
            bool tmp = json_component["disable"];
            if(tmp) {
                return false;
            }
        }
    }
    return true;
}
#endif

#ifdef TAU_PAPI
void initialize_papi_events(bool do_components) {
    PapiLayer::initializePapiLayer();
    int num_components = PAPI_num_components();
    const PAPI_component_info_t *comp_info;
    int retval = PAPI_OK;
    const char * tau_metrics = TauEnv_get_metrics();
    if (tau_metrics != NULL && strlen(tau_metrics) > 0) {
        TAU_VERBOSE("WARNING: TAU_METRICS is set to '%s'.\n", tau_metrics);
        TAU_VERBOSE("TAU can't enable process-wide hardware PAPI measurements in the monitoring plugin.\n");
    }
    if ((tau_metrics == NULL || strlen(tau_metrics) == 0) &&
        configuration.count("PAPI metrics")) {
        // Set the default granularity for the cpu component - this won't work.
        if ((retval = PAPI_set_cmp_granularity(PAPI_GRN_PROC, 0)) != PAPI_OK)
            TAU_VERBOSE("Error: PAPI_set_granularity: %d %s\n", retval, PAPI_strerror(retval));

        // Enable multiplexing to allow everything to be captured
        if ((retval = PAPI_multiplex_init()) != PAPI_OK)
            printf("Error: PAPI_mulitplex_init: %d %s\n", retval, PAPI_strerror(retval));

        /* Set up counters, if desired */
        if ((retval = PAPI_create_eventset(&papi_periodic_event_set)) != PAPI_OK)
            printf("Error: PAPI_create_eventset: %d %s\n", retval, PAPI_strerror(retval));

        if ((retval = PAPI_assign_eventset_component(papi_periodic_event_set, 0)) != PAPI_OK)
            printf("Error: PAPI_create_eventset: %d %s\n", retval, PAPI_strerror(retval));

        if ((retval = PAPI_set_multiplex(papi_periodic_event_set)) != PAPI_OK)
            printf("Error: PAPI_set_multiplex: %d %s\n", retval, PAPI_strerror(retval));

        auto metrics = configuration["PAPI metrics"];
        for (auto i : metrics) {
            std::string metric(i);
            // PAPI is either const char * or char * depending on version.  Fun.
            if ((retval = PAPI_add_named_event(papi_periodic_event_set, (char*)(metric.c_str()))) != PAPI_OK) {
                printf("Error: PAPI_add_event: %d %s %s\n", retval, PAPI_strerror(retval), metric.c_str());
	    } else {
                num_metrics++;
	    }
        }
        if (metrics.size() > 0) {
            papi_periodic_values = (long long*)(calloc(metrics.size(), sizeof(long long)));

            if ((retval = PAPI_start(papi_periodic_event_set)) != PAPI_OK)
                printf("Error: PAPI_start: %d %s\n", retval, PAPI_strerror(retval));
        }
    }

    // if this rank isn't doing system stats, return
    if (!do_components) { return; }

    // are there any components?
    for (int component_id = 0 ; component_id < num_components ; component_id++) {
        comp_info = PAPI_get_component_info(component_id);
        if (comp_info == NULL) {
            TAU_VERBOSE("Warning: PAPI component info unavailable, no measurements will be done.\n");
            return;
        }
        //fprintf(stdout, "Found %s PAPI component with %d native events.\n", comp_info->name, comp_info->num_native_events);
        /* Skip the perf_event component, that's standard PAPI */
        if (strstr(comp_info->name, "perf_event") != NULL) {
            continue;
        }
        /* Skip the example component, that's worthless and will break things */
        if (strstr(comp_info->name, "example") != NULL) {
            continue;
        }
        /* Skip the perf_event_uncore component, it has security problems */
        if (strstr(comp_info->name, "perf_event_uncore") != NULL) {
            continue;
        }
        /* Skip the nvml component, for non-CUPTI configurations */
#ifndef TAU_ROCM_SMI
        if (strstr(comp_info->name, "nvml") != NULL) {
            continue;
        }
#endif
#ifndef CUPTI
        if (strstr(comp_info->name, "cuda") != NULL) {
            continue;
        }
#endif
        if (!include_component(comp_info->name)) { return; }
        if (get_my_rank() == 0) TAU_VERBOSE("Found %s component...\n", comp_info->name);
        /* Does this component have available events? */
        if (comp_info->num_native_events == 0) {
            TAU_VERBOSE("Error: No %s events found.\n", comp_info->name);
            if (comp_info->disabled != 0) {
                TAU_VERBOSE("Error: %s.\n", comp_info->disabled_reason);
            }
            continue;
        }
        ppc * comp = new ppc(comp_info->name, component_id);
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
            if (get_my_rank() == 0) TAU_VERBOSE("Found event '%s (%s)'\n", event_name, unit);
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
            TAU_VERBOSE("Error: Error starting PAPI eventset for component %s.\n", comp_info->name);
            continue;
        }
        comp->initialized = true;
        components.push_back(comp);
    }
}
#endif

std::vector<cpustats_t*> * read_cpu_stats() {
    static const char * source = "/proc/stat";
    if (!include_component(source)) { return NULL; }
    if (get_my_rank() == 0) TAU_VERBOSE("Found %s component...\n", source);
    std::vector<cpustats_t*> * cpu_stats = new std::vector<cpustats_t*>();
    /*  Reading proc/stat as a file  */
    FILE * pFile;
    char line[128] = {0};
    pFile = fopen (source,"r");
    if (pFile == nullptr) {
        perror ("Error opening file");
        return NULL;
    } else {
        while ( fgets( line, 128, pFile)) {
            if ( strncmp (line, "cpu", 3) == 0 ) {
                cpustats_t * cpu_stat = new(cpustats_t);
                std::string tmp{line};
                // trim the newline
                tmp = tau::papi_plugin::trim(tmp);
                // create a stringstream from the string
                std::stringstream ss(tmp);
                std::vector<std::string> out;
                std::string s;
                while (std::getline(ss, s, ' ')) {
                    if (s.size() == 0) { continue; }
                    out.push_back(s);
                }
                snprintf(cpu_stat->name, 32, "%s", out[0].c_str());
                cpu_stat->user = stol(out[1]);
                cpu_stat->nice = stol(out[2]);
                cpu_stat->system =stol(out[3]);
                cpu_stat->idle = stol(out[4]);
                cpu_stat->iowait =stol(out[5]);
                cpu_stat->irq =stol(out[6]);
                cpu_stat->softirq = stol(out[7]);
                cpu_stat->steal =stol(out[8]);
                cpu_stat->guest = stol(out[9]);
                /* PGI Compiler is non-standard.  It can't handle regular expressions
                 * with range values, so we can't filter out the per-cpu results.
                 * So, we'll just read the first line of the file and quit
                 * for all cases. */
#if !defined(__PGI)
                if (!include_event(source, cpu_stat->name)) {
                    continue;
                }
#endif
                cpu_stats->push_back(cpu_stat);
            }
//#if defined(__PGI)
            // only do the first line.
            break;
//#endif
        }
    }
    fclose(pFile);
    return cpu_stats;
}

std::map<std::string, netstats_t*> * read_net_stats(const char * source) {
    if (!include_component(source)) { return NULL; }
    std::map<std::string, netstats_t*> * net_stats = new std::map<std::string, netstats_t*>();
    /*  Reading proc/stat as a file  */
    FILE * pFile;
    char line[512] = {0};
    /* Do we want per-process readings? */
    pFile = fopen (source,"r");
    if (pFile == nullptr) {
        perror ("Error opening file");
        return NULL;
    }
    char * rc = fgets(line, 512, pFile); // skip this line
    if (rc == nullptr) {
        fclose(pFile);
        return NULL;
    }
    rc = fgets(line, 512, pFile); // skip this line
    if (rc == nullptr) {
        fclose(pFile);
        return NULL;
    }
    /* Read each device */
    while (fgets(line, 512, pFile)) {
        std::string outer_tmp(line);
        std::size_t found = outer_tmp.find(":");
        if (found == std::string::npos) { continue; }
        outer_tmp = tau::papi_plugin::trim(outer_tmp);
        netstats_t * net_stat = new(netstats_t);
        int nf = sscanf( line,
            "%s: %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
            net_stat->name, &net_stat->recv_bytes,
            &net_stat->recv_packets, &net_stat->recv_errors,
            &net_stat->recv_drops, &net_stat->recv_fifo,
            &net_stat->recv_frame, &net_stat->recv_compressed,
            &net_stat->recv_multicast, &net_stat->transmit_bytes,
            &net_stat->transmit_packets, &net_stat->transmit_errors,
            &net_stat->transmit_drops, &net_stat->transmit_fifo,
            &net_stat->transmit_collisions, &net_stat->transmit_carrier,
            &net_stat->transmit_compressed);
        if (nf == 0) {
            printf("%s",line);
            continue; // error!
        }
        (*net_stats)[net_stat->name] = net_stat;
    }
    fclose(pFile);
    return net_stats;
}

iostats_t * read_io_stats(const char * source) {
    if (!include_component(source)) { return NULL; }
    iostats_t * io_stats = new iostats_t();
    /*  Reading proc/stat as a file  */
    FILE * pFile;
    char line[512] = {0};
    pFile = fopen (source,"r");
    if (pFile == nullptr) {
        perror ("Error opening file");
        return NULL;
    }
    /* Read each line */
    while (fgets(line, 512, pFile)) {
        char dummy[32] = {0};
        long long tmplong = 0LL;
        int nf = sscanf( line, "%s: %lld\n", dummy, &tmplong);
        if (nf == 0) continue; // error!
        std::string name(dummy);
        (*io_stats)[name] = tmplong;
    }
    fclose(pFile);
    return io_stats;
}

int choose_volunteer_rank() {
#ifdef TAU_MPI
    // figure out who should get system stats for this node
    int i;
    int my_rank = RtsLayer::myNode();
    int comm_size = tau_totalnodes(0,1);

    // save it somewhere the NVML component can access it
    get_my_rank() = my_rank;

    /* If the user wants node health data from all ranks,
       then every rank participates in reading that data. */
    if (configuration.count("node_data_from_all_ranks") > 0 &&
        configuration["node_data_from_all_ranks"]) {
        return my_rank;
    }

    // get my hostname
    const int hostlength = MPI_MAX_PROCESSOR_NAME;
    char hostname[hostlength] = {0};
    //gethostname(hostname, sizeof(char)*hostlength);
    int namelength = 0;
    MPI_Get_processor_name(hostname, &namelength);
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
    std::set<std::string> hostnames;
    bool found = false;
    for (i = 0 ; i < comm_size ; i++) {
        //printf("%d:%d comparing '%s' to '%s'\n", rank, size, hostname, host_index);
        if (!found && (strncmp(hostname, host_index, hostlength) == 0)) {
            volunteer = i;
            found = true;
        }
        hostnames.insert(std::string(hostname));
        host_index = host_index + hostlength;
    }
    // Set some metadata to help with analysis later
    Tau_metadata("MPI Comm World Size", std::to_string(comm_size).c_str());
    Tau_metadata("MPI Unique Hosts", std::to_string(hostnames.size()).c_str());
    Tau_metadata("MPI Host Name", hostname);
    Tau_metadata("MPI Comm World Rank", std::to_string(my_rank).c_str());
    free(allhostnames);
    return volunteer;
#else
    return 0;
#endif
}

void parse_proc_meminfo() {
  static const char * source = "/proc/meminfo";
  if (!include_component(source)) { return; }
  FILE *f = fopen(source, "r");
  if (f) {
    char line[4096] = {0};
    while ( fgets( line, 4096, f)) {
        std::string tmp(line);
        std::istringstream iss(tmp);
        std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                         std::istream_iterator<std::string>());
        std::string& value = results[1];
        char* pEnd;
        double d1 = strtod (value.c_str(), &pEnd);
        if (pEnd) {
            std::stringstream ss;
            /* trim the trailing : */
            ss << "meminfo:" << results[0].substr(0,results[0].size()-1);
            if (results.size() == 3) {
                if(results[2].compare("kB") == 0 && d1 > 10000.0) {
                    ss << " (MB)";
                    d1 = d1 * 1.0e-3;
                } else {
                    ss << " (" << results[2] << ")";
                }
            }
            if (include_event(source, ss.str().c_str())) {
                sample_user_event(ss.str().c_str(), d1);
            }
        }
    }
    fclose(f);
  }
  return;
}

void sample_value(const char * component, const char * cpu, const char * name,
        const double value, const long long total) {
    std::stringstream ss;
    ss << cpu << ":" << name;
    /* If we are not including this event, continue */
    if (!include_event(component, ss.str().c_str())) { return; }
    // double-check the value...
    double tmp;
    if (total == 0LL) {
        tmp = 0.0;
    } else {
        tmp = (value / (double)(total)) * 100.0;
    }
    sample_user_event(ss.str().c_str(), tmp);
}

extern "C" void Tau_metadata_task(char *name, const char* value, int tid);

void parse_proc_self_stat() {
  static const char * source = "/proc/self/stat";
  if (!include_component(source)) { return; }
  FILE *f = fopen(source, "r");
  if (f) {
    char line[4096] = {0};
    while ( fgets( line, 4096, f)) {
        std::string tmp(line);
        std::istringstream iss(tmp);
        // split the string on whitespace
        std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                         std::istream_iterator<std::string>());
        if (results.size() >= 52) {
            /* Remember that the string is 0-indexed, so subtract 1 from the
             * indexes on https://man7.org/linux/man-pages/man5/proc.5.html */
            sample_user_event("Page faults (minor)", atof(results[9].c_str()));
            sample_user_event("Page faults (major)", atof(results[11].c_str()));
            sample_user_event("Pages swapped (not maintained)", atof(results[35].c_str()));
            //sample_user_event("Threads (stat)", atof(results[19].c_str()), true);
        }
    }
    fclose(f);
  }
  return;
}

void parse_proc_self_status() {
  static const char * source = "/proc/self/status";
  static bool first = true;
  if (!include_component(source)) { return; }
  FILE *f = fopen(source, "r");
  if (f) {
    char line[4096] = {0};
    while ( fgets( line, 4096, f)) {
        std::string tmp(line);
        std::istringstream iss(tmp);
        // split the string on whitespace
        std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                         std::istream_iterator<std::string>());
        // remove the trailing colon from the label
        results[0].erase(std::remove(results[0].begin(),
            results[0].end(), ':'), results[0].end());
        // if this is the first time calling this function, save
        // metadata fields (that don't change)
        if (first) {
            if (results[0].find("_allowed_list") != std::string::npos) {
                Tau_metadata_task(const_cast<char*>(results[0].c_str()),
                const_cast<char*>(results[1].c_str()), 0);
            }
        }
    }
    fclose(f);
  }
  first = false;
  return;
}

void parse_proc_self_statm() {
  static const char * source = "/proc/self/statm";
  if (!include_component(source)) { return; }
  FILE *f = fopen(source, "r");
  if (f) {
    char line[4096] = {0};
    while ( fgets( line, 4096, f)) {
        std::string tmp(line);
        std::istringstream iss(tmp);
        std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                         std::istream_iterator<std::string>());
        std::string& value = results[0];
        char* pEnd;
        double d1 = strtod (value.c_str(), &pEnd);
        if (pEnd) {
            if (include_event(source, "program size (pages)")) {
                sample_user_event("program size (pages)", d1);
            }
        }
        value = results[1];
        d1 = strtod (value.c_str(), &pEnd);
        if (pEnd) {
            if (include_event(source, "resident set size (pages)")) {
                sample_user_event("resident set size (pages)", d1);
            }
        }
        value = results[2];
        d1 = strtod (value.c_str(), &pEnd);
        if (pEnd) {
            if (include_event(source, "resident shared pages")) {
                sample_user_event("resident shared pages", d1);
            }
        }
    }
    fclose(f);
  }
  return;
}

void update_cpu_stats(void) {
    static const char * source = "/proc/stat";
    PLUGIN_ASSERT(previous_cpu_stats != nullptr);
    if (!include_component(source)) { return; }
    /* get the current stats */
    std::vector<cpustats_t*> * new_stats = read_cpu_stats();
    if (new_stats == NULL) return;
    for (size_t i = 0 ; i < new_stats->size() ; i++) {
        /* we need to take the difference from the last read */
        cpustats_t diff;
        diff.user = (*new_stats)[i]->user - (*previous_cpu_stats)[i]->user;
        diff.nice = (*new_stats)[i]->nice - (*previous_cpu_stats)[i]->nice;
        diff.system = (*new_stats)[i]->system - (*previous_cpu_stats)[i]->system;
        diff.idle = (*new_stats)[i]->idle - (*previous_cpu_stats)[i]->idle;
        diff.iowait = (*new_stats)[i]->iowait - (*previous_cpu_stats)[i]->iowait;
        diff.irq = (*new_stats)[i]->irq - (*previous_cpu_stats)[i]->irq;
        diff.softirq = (*new_stats)[i]->softirq - (*previous_cpu_stats)[i]->softirq;
        diff.steal = (*new_stats)[i]->steal - (*previous_cpu_stats)[i]->steal;
        diff.guest = (*new_stats)[i]->guest - (*previous_cpu_stats)[i]->guest;
        double total = (double)(diff.user + diff.nice + diff.system +
                diff.idle + diff.iowait + diff.irq + diff.softirq +
                diff.steal + diff.guest);
        sample_value(source, (*new_stats)[i]->name, " User %",
            (double)(diff.user), total);
        sample_value(source, (*new_stats)[i]->name, " Nice %",
            (double)(diff.nice), total);
        sample_value(source, (*new_stats)[i]->name, " System %",
            (double)(diff.system), total);
        sample_value(source, (*new_stats)[i]->name, " Idle %",
            (double)(diff.idle), total);
        sample_value(source, (*new_stats)[i]->name, " I/O Wait %",
            (double)(diff.iowait), total);
        sample_value(source, (*new_stats)[i]->name, " IRQ %",
            (double)(diff.irq), total);
        sample_value(source, (*new_stats)[i]->name, " soft IRQ %",
            (double)(diff.softirq), total);
        sample_value(source, (*new_stats)[i]->name, " Steal %",
            (double)(diff.steal), total);
        sample_value(source, (*new_stats)[i]->name, " Guest %",
            (double)(diff.guest), total);
    }
    for (auto it : *previous_cpu_stats) {
        delete it;
    }
    delete previous_cpu_stats;
    previous_cpu_stats = new_stats;
}

std::map<std::string, netstats_t*> * update_net_stats(const char * source,
std::map<std::string, netstats_t*> *previous) {
    if (!include_component(source)) { return previous; }
    PLUGIN_ASSERT(previous != nullptr);
    /* get the current stats */
    std::map<std::string, netstats_t*> * new_stats = read_net_stats(source);
    if (new_stats == NULL) return previous;
    for (auto itr : *new_stats) {
        // get the name
        std::string name = itr.first;
        netstats_t * ns = itr.second;
        netstats_t * ps = nullptr;
        // if the previous read didn't have this value, create zeros for the last read
        auto tmp = previous->find(name);
        if (tmp == previous->end()) {
            ps = new(netstats_t);
        } else {
            ps = tmp->second;
        }
        /* we need to take the difference from the last read */
        netstats_t diff;
        diff.recv_bytes = ns->recv_bytes - ps->recv_bytes;
        sample_value(source, name.c_str(), "rx:bytes", (double)(diff.recv_bytes), 1LL);
        diff.recv_packets = ns->recv_packets - ps->recv_packets;
        sample_value(source, name.c_str(), "rx:packets", (double)(diff.recv_packets), 1LL);
        diff.recv_errors = ns->recv_errors - ps->recv_errors;
        sample_value(source, name.c_str(), "rx:errors", (double)(diff.recv_errors), 1LL);
        diff.recv_drops = ns->recv_drops - ps->recv_drops;
        sample_value(source, name.c_str(), "rx:drops", (double)(diff.recv_drops), 1LL);
        diff.recv_fifo = ns->recv_fifo - ps->recv_fifo;
        sample_value(source, name.c_str(), "rx:fifo", (double)(diff.recv_fifo), 1LL);
        diff.recv_frame = ns->recv_frame - ps->recv_frame;
        sample_value(source, name.c_str(), "rx:frames", (double)(diff.recv_frame), 1LL);
        diff.recv_compressed = ns->recv_compressed - ps->recv_compressed;
        sample_value(source, name.c_str(), "rx:compressed", (double)(diff.recv_compressed), 1LL);
        diff.recv_multicast = ns->recv_multicast - ps->recv_multicast;
        sample_value(source, name.c_str(), "rx:multicast", (double)(diff.recv_multicast), 1LL);
        diff.transmit_bytes = ns->transmit_bytes - ps->transmit_bytes;
        sample_value(source, name.c_str(), "tx:bytes", (double)(diff.transmit_bytes), 1LL);
        diff.transmit_packets = ns->transmit_packets - ps->transmit_packets;
        sample_value(source, name.c_str(), "tx:packets", (double)(diff.transmit_packets), 1LL);
        diff.transmit_errors = ns->transmit_errors - ps->transmit_errors;
        sample_value(source, name.c_str(), "tx:errors", (double)(diff.transmit_errors), 1LL);
        diff.transmit_drops = ns->transmit_drops - ps->transmit_drops;
        sample_value(source, name.c_str(), "tx:drops", (double)(diff.transmit_drops), 1LL);
        diff.transmit_fifo = ns->transmit_fifo - ps->transmit_fifo;
        sample_value(source, name.c_str(), "tx:fifo", (double)(diff.transmit_fifo), 1LL);
        diff.transmit_collisions = ns->transmit_collisions - ps->transmit_collisions;
        sample_value(source, name.c_str(), "tx:collisions", (double)(diff.transmit_collisions), 1LL);
        diff.transmit_carrier = ns->transmit_carrier - ps->transmit_carrier;
        sample_value(source, name.c_str(), "tx:carrier", (double)(diff.transmit_carrier), 1LL);
        diff.transmit_compressed = ns->transmit_compressed - ps->transmit_compressed;
        sample_value(source, name.c_str(), "tx:compressed", (double)(diff.transmit_compressed), 1LL);
    }
    /* If the previous reading had  value we didn't see this time,
     * move it to the new reading. Otherwise, delete it. */
    for (auto it : *previous) {
        std::string name = it.first;
        auto tmp = new_stats->find(name);
        if (tmp == new_stats->end()) {
            (*new_stats)[name] = (*previous)[name];
        } else {
            delete it.second;
        }
    }
    delete previous;
    return new_stats;
}

void update_io_stats(const char * source) {
    if (!include_component(source)) { return; }
    PLUGIN_ASSERT(previous_io_stats != nullptr);
    /* get the current stats */
    iostats_t * new_stats = read_io_stats(source);
    if (new_stats == NULL) return;
    //for (size_t i = 0 ; i < new_stats->size() ; i++) {
    for(auto itr : *new_stats) {
        std::string name = itr.first;
        long long new_value = itr.second;
        long long old_value = 0;
        auto tmp = previous_io_stats->find(name);
        if (tmp != previous_io_stats->end()) {
            old_value = tmp->second;
        }
        /* we need to take the difference from the last read */
        long long tmplong = new_value - old_value;
        sample_value(source, "io", name.c_str(), (double)(tmplong), 1LL);
    }
    for (auto it : *previous_io_stats) {
        std::string name = it.first;
        auto tmp = new_stats->find(name);
        if (tmp == new_stats->end()) {
            (*new_stats)[name] = (*previous_io_stats)[name];
        }
    }

    delete previous_io_stats;
    previous_io_stats = new_stats;
}

void read_components(void) {
    TAU_VERBOSE("Tau monitoring : %s\n", __func__);
    static std::string tmpstr{configuration.count("monitor_counter_prefix") > 0 ?
        configuration["monitor_counter_prefix"] : ""};
    static const char * prefix = tmpstr.c_str();
#ifdef TAU_PAPI
    TAU_VERBOSE("Tau monitoring : reading papi\n");
    for (size_t index = 0; index < components.size() ; index++) {
        if (components[index]->initialized) {
            ppc * comp = components[index];
            long long * values = (long long *)calloc(comp->events.size(), sizeof(long long));
            int retval = PAPI_read(comp->event_set, values);
            if (retval != PAPI_OK) {
                TAU_VERBOSE("Error: Error reading PAPI %s eventset.\n", comp->name.c_str());
                return;
            }
            for (size_t i = 0 ; i < comp->events.size() ; i++) {
                sample_user_event(comp->events[i].name,
                    ((double)values[i]) * comp->events[i].conversion);
            }
            free(values);
        }
    }
    if (num_metrics > 0) {
	// reset the counters to zero
	memset(papi_periodic_values, 0, (sizeof(long long) * num_metrics));
        int retval = PAPI_accum(papi_periodic_event_set, papi_periodic_values);
        if (retval != PAPI_OK) {
            TAU_VERBOSE("Error: PAPI_read: %d %s\n", retval, PAPI_strerror(retval));
        } else {
            auto metrics = configuration["PAPI metrics"];
            int index = 0;
            for (auto i : metrics) {
                std::string metric(i);
                if (papi_periodic_values[index] < 0LL) {
                    TAU_VERBOSE("Bogus (probably derived/multiplexed) value: %s %lld\n", metric.c_str(), papi_periodic_values[index]);
                    papi_periodic_values[index] = 0LL;
                }
                sample_user_event(metric, (double)papi_periodic_values[index]);
                // set it back to zero so we can use PAPI_accum and not PAPI_reset
                papi_periodic_values[index] = 0;
                index++;
            }
        }
    }
#endif
    /* Also read some OS level metrics. */

    /* records the heap, with no context, even though it says "here". */
    TauTrackMemoryHere(prefix);
#if !defined(__APPLE__)
    /* records the rss/hwm, without context. */
    TauTrackMemoryFootPrint(prefix);
    /* Get current io stats for the process */
    update_io_stats("/proc/self/io");
    /* Parse overall stats */
    parse_proc_self_status();
    /* Parse memory stats */
    parse_proc_self_statm();
    /* Parse general stat */
    parse_proc_self_stat();
    /* Get current net stats for the process */
    previous_self_net_stats = update_net_stats("/proc/self/net/dev", previous_self_net_stats);
#endif

    if (get_my_rank() == rank_getting_system_data) {
#ifdef CUPTI
        if (include_component("nvml")) {
            get_nvml_reader().query();
        }
#endif
#ifdef TAU_ROCM_SMI
        if (include_component("rsmi")) {
            get_rsmi_reader().query();
        }
#endif

#if !defined(__APPLE__)
        /* records the load, without context */
        TauTrackLoad(prefix);
#endif
        /* records the power, without context */
        TauTrackPower(prefix);
#if !defined(__APPLE__)
        /* Get the current CPU statistics for the node */
        update_cpu_stats();
        /* Get current meminfo stats for the node */
        parse_proc_meminfo();
        /* Get current net stats for the node */
        previous_net_stats = update_net_stats("/proc/net/dev", previous_net_stats);
#endif
    }

    TAU_VERBOSE("Tau monitoring : reading done!\n");
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
                TAU_VERBOSE("Error: Error reading PAPI %s eventset.\n", comp->name.c_str());
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
    if (done) return;
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    // If the thread isn't attached, we can't join it - so just sleep the usual
    // measurement period and hope it gets the message that we're done.
    if (!_attached) {
        // wait for the worker thread to exit.
        while(worker_working) {}
        /*
        int microseconds{1};
        if (configuration.count("periodicity seconds")) {
            double floating = configuration["periodicity seconds"];
            double integer;
            double fractional = modf(floating, &integer);
            seconds = (int)(integer);
            int microseconds = (int)(fractional * ONE_MILLIONF);
            usleep(microseconds);
        }
        */
        return;
    }
    if (get_my_rank() == 0) TAU_VERBOSE("TAU Monitoring thread joining...\n"); fflush(stderr);
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

void * Tau_monitoring_plugin_threaded_function(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;
    gettimeofday(&tp, NULL);

    /* If we are tracing while doing periodic measurements, then register
     * this thread so we don't collide with events on thread 0 */
    if (TauEnv_get_thread_per_gpu_stream()) {
        Tau_register_thread();
    }
    periodic_index = 0;

    while (!done) {
        TAU_VERBOSE("Tau monitoring thread: waking up\n");
        // take a reading...
        worker_working = true;
        TAU_VERBOSE("Tau monitoring thread: reading components\n");
        read_components();
        worker_working = false;
        // wait x seconds for the next batch.  Can be floating!
        int seconds = 1;
        int nanoseconds = 0;
        // convert the floating seconds to seconds and nanoseconds
        if (configuration.count("periodicity seconds")) {
            double floating = configuration["periodicity seconds"];
            double integer;
            double fractional = modf(floating, &integer);
            seconds = (int)(integer);
            nanoseconds = (int)(fractional * ONE_BILLIONF);
        }
        // add our nanoseconds of delay
        ts.tv_nsec = (1000 * tp.tv_usec) + nanoseconds;
        // check for overflow
        if (ts.tv_nsec > ONE_BILLION){
            ts.tv_nsec = ts.tv_nsec - ONE_BILLION;
            seconds = seconds + 1;
        }
        // add our seconds of delay
        ts.tv_sec  = (tp.tv_sec + seconds);
        // before we lock and wait, make sure we haven't exited in the meantime
        if (done) { break; }
        pthread_mutex_lock(&_my_mutex);
        // wait the time period.
        int rc = pthread_cond_timedwait(&_my_cond, &_my_mutex, &ts);
        if (rc == ETIMEDOUT) {
            //TAU_VERBOSE("%d Timeout from plugin.\n", RtsLayer::myNode()); fflush(stderr);
        } else if (rc == EINVAL) {
            TAU_VERBOSE("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("Mutex not locked!\n"); fflush(stderr);
        }
        periodic_index++;
        tp.tv_usec = tp.tv_usec + (nanoseconds/1000);
        // check for overflow
        if (tp.tv_usec > ONE_MILLION){
            tp.tv_usec = tp.tv_usec - ONE_MILLION;
            seconds = seconds + 1;
        }
        // add our seconds of delay
        tp.tv_sec  = (tp.tv_sec + seconds);
    }
    if (get_my_rank() == 0) TAU_VERBOSE("TAU Monitoring thread exiting...\n"); fflush(stderr);

    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    //pthread_exit((void*)0L);
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

static void do_cleanup() {
    static bool clean = false;
    if (clean) return;
    // if no thread, return
    if (configuration.count("periodic") == 0 || !configuration["periodic"]) {
        read_components();
    } else {
        stop_worker();
    }
#ifdef TAU_PAPI
    /* clean up papi */
    if (get_my_rank() == rank_getting_system_data) {
        free_papi_components();
    }
#endif
    if (previous_cpu_stats != nullptr) {
        for (auto it : *previous_cpu_stats) {
            delete it;
        }
        delete previous_cpu_stats;
        previous_cpu_stats = nullptr;
    }
    if (previous_net_stats != nullptr) {
        for (auto it : *previous_net_stats) {
            delete it.second;
        }
        delete previous_net_stats;
        previous_net_stats = nullptr;
    }
    if (previous_self_net_stats != nullptr) {
        for (auto it : *previous_self_net_stats) {
            delete it.second;
        }
        delete previous_self_net_stats;
        previous_self_net_stats = nullptr;
    }
    if (previous_io_stats != nullptr) {
        delete previous_io_stats;
        previous_io_stats = nullptr;
    }
    /* Why do these deadlock on exit? */
    //pthread_cond_destroy(&_my_cond);
    //pthread_mutex_destroy(&_my_mutex);
    clean = true;
}

// Macro to check MPI calls status
#define MPI_CALL(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char errstr[MPI_MAX_ERROR_STRING]; \
            int errlen; \
            MPI_Error_string(err, errstr, &errlen); \
            fprintf(stderr, "%s\n", errstr); \
            MPI_Abort(MPI_COMM_WORLD, 999); \
        } \
    } while (0)

inline char filesystem_separator()
{
#if defined _WIN32 || defined __CYGWIN__
    return '\\';
#else
    return '/';
#endif
}

void reduce_scatterplot(std::stringstream& csv_output, std::string filename) {
    int commrank = 0;
    int commsize = 1;
#if defined(TAU_MPI)
    int mpi_initialized = 0;
    MPI_CALL(MPI_Initialized( &mpi_initialized ));
    if (mpi_initialized) {
        commrank = RtsLayer::myNode();
        commsize = tau_totalnodes(0,1);
    }
#endif
    // if nothing to reduce, just write the data.
    if (commsize == 1) {
        std::ofstream csvfile;
        std::stringstream csvname;
        csvname << TauEnv_get_profiledir();
        csvname << filesystem_separator() << filename;
        std::cout << "Writing: " << csvname.str() << std::endl;
        csvfile.open(csvname.str(), std::ios::out);
        csvfile << "\"rank\",\"name\",\"step\",\"value\"\n";
        csvfile << csv_output.str();
        csvfile.close();
        return;
    }

    size_t length{csv_output.str().size()};
    size_t max_length{length};
    // get the longest string from all ranks
#if defined(TAU_MPI)
    if (mpi_initialized && commsize > 1) {
        MPI_CALL(PMPI_Allreduce(&length, &max_length, 1,
            MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD));
    }
    // so we don't have to specially handle the first string which will append
    // the second string without a null character (zero).
    max_length = max_length + 1;
#endif
    // allocate the send buffer
    char * sbuf = (char*)calloc(max_length, sizeof(char));
    // copy into the send buffer
    strncpy(sbuf, csv_output.str().c_str(), length);
    // allocate the memory to hold all output
    char * rbuf = nullptr;
    if (commrank == 0) {
#if defined(TAU_MPI)
        rbuf = (char*)calloc(max_length * commsize, sizeof(char));
#else
        rbuf = sbuf;
#endif
    }

#if defined(TAU_MPI)
    MPI_Gather(sbuf, max_length, MPI_CHAR, rbuf, max_length, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

    if (commrank == 0) {
        std::ofstream csvfile;
        std::stringstream csvname;
        csvname << TauEnv_get_profiledir();
        csvname << filesystem_separator() << filename;
        std::cout << "Writing: " << csvname.str();
        csvfile.open(csvname.str(), std::ios::out);
        csvfile << "\"rank\",\"name\",\"step\",\"value\"\n";
        char * index = rbuf;
        for (auto i = 0 ; i < commsize ; i++) {
            index = rbuf+(i*max_length);
            std::string tmpstr{index};
            csvfile << tmpstr;
            csvfile.flush();
        }
        csvfile.close();
        std::cout << "...done." << std::endl;
    }
    free(sbuf);
    free(rbuf);
}

int Tau_plugin_event_pre_end_of_execution_monitoring(Tau_plugin_event_pre_end_of_execution_data_t *data) {
    if (configuration.count("scatterplot") > 0) {
        if (configuration["scatterplot"]) {
            reduce_scatterplot(csv_output, "monitoring.csv");
        }
    }
    if (get_my_rank() == 0)
        TAU_VERBOSE("Monitoring Component PLUGIN %s\n", __func__);
    //if (RtsLayer::myThread() == 0) {
    if (main_thread()) {
        // only the main thread should cleanup
        do_cleanup();
    }
    return 0;
}

int Tau_plugin_event_end_of_execution_monitoring(Tau_plugin_event_end_of_execution_data_t *data) {
    if (get_my_rank() == 0)
        TAU_VERBOSE("Monitoring Component PLUGIN %s\n", __func__);
    //if (RtsLayer::myThread() == 0) {
    if (main_thread()) {
        // only the main thread should cleanup
        do_cleanup();
    }
    return 0;
}

int Tau_plugin_metadata_registration_complete_monitoring(Tau_plugin_event_metadata_registration_data_t* data) {
    //TAU_VERBOSE("Monitoring Component PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_monitoring(Tau_plugin_event_post_init_data_t* data) {
    if (get_my_rank() == 0) TAU_VERBOSE("Monitoring Component PLUGIN %s\n", __func__);

    rank_getting_system_data = choose_volunteer_rank();

#ifdef TAU_PAPI
    /* get ready to read metrics! */
    if (configuration.count("PAPI metrics")) {
        auto metrics = configuration["PAPI metrics"];
        if (metrics.size() > 0) {
            initialize_papi_events(get_my_rank() == rank_getting_system_data);
        }
    }
#endif

    if (get_my_rank() == rank_getting_system_data) {
#if !defined(__APPLE__)
        previous_cpu_stats = read_cpu_stats();
        /* Parse initial node network data */
        previous_net_stats = read_net_stats("/proc/net/dev");
#endif
    }
#if !defined(__APPLE__)
    /* Parse status metadata */
    parse_proc_self_status();
    /* Parse general stat */
    /* don't do this yet - wait until the thread does it so we don't
       have counters on thread 0 */
    if (!TauEnv_get_thread_per_gpu_stream()) {
        parse_proc_self_stat();
    }
    /* Parse initial process io data */
    previous_io_stats = read_io_stats("/proc/self/io");
    /* Parse initial process network data */
    previous_self_net_stats = read_net_stats("/proc/self/net/dev");
#endif
    if (configuration.count("periodic") &&
        configuration["periodic"]) {
        /* spawn the worker thread to do the reading */
        init_lock(&_my_mutex);
        if (get_my_rank() == 0) TAU_VERBOSE("Spawning thread.\n");
        int ret = pthread_create(&worker_thread, NULL,
        &Tau_monitoring_plugin_threaded_function, NULL);
        if (ret != 0) {
            errno = ret;
            perror("Error: pthread_create (1) fails\n");
            exit(1);
        }
        // be free, little thread!
        ret = pthread_detach(worker_thread);
        if (ret != 0) {
            switch (ret) {
                case ESRCH:
                    // no thread found?
                case EINVAL:
                    // not joinable?
                default:
                    errno = ret;
                    perror("Warning: pthread_detach failed\n");
                    return 0;
            }
        }
        if (get_my_rank() == 0) TAU_VERBOSE("Detached thread.\n");
        _attached = false;
    }
    return 0;
}

void read_config_file(void) {
    try {
            std::ifstream cfg("tau_monitoring.json");
            // if the file doesn't exist, return
            if (!cfg.good()) {
            	// fail silently, nothing to do but use defaults
                configuration = json::parse(default_configuration);
                return;
            }
            cfg >> configuration;
            cfg.close();
        } catch (...) {
            TAU_VERBOSE("Error reading tau_monitoring.json file!");
            configuration = json::parse(default_configuration);
        }
}

int Tau_plugin_dump_monitoring(Tau_plugin_event_dump_data_t* data) {
    tau::plugins::ScopedTimer(__func__);
    //printf("Monitoring Component PLUGIN %s\n", __func__);
    // take a reading...
    read_components();
    return 0;
}


/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    done = false;
    worker_working = false;
    main_thread() = true;

    read_config_file();

    /* Required event support */
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_monitoring;
    cb->PostInit = Tau_plugin_event_post_init_monitoring;
    cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution_monitoring;
    cb->EndOfExecution = Tau_plugin_event_end_of_execution_monitoring;
    cb->Dump = Tau_plugin_dump_monitoring;

    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);
    free (cb);

    return 0;
}

