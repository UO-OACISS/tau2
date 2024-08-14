/****************************************************************************
 **                      TAU Portable Profiling Package                     **
 **                      http://www.cs.uoregon.edu/research/tau             **
 *****************************************************************************
 **    Copyright 2009                                                       **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 **    Forschungszentrum Juelich                                            **
 ****************************************************************************/
/****************************************************************************
 **      File            : TauMetrics.cpp                                   **
 **      Description     : TAU Profiling Package                            **
 **      Contact         : tau-bugs@cs.uoregon.edu                          **
 **      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
 **                                                                         **
 **      Description     : Handles Metrics                                  **
 **                                                                         **
 ****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tau_internal.h>
#include <Profile/TauMetrics.h>
#include <Profile/Profiler.h>
#include <Profile/TauTrace.h>
#ifdef CUPTI
#include <Profile/CuptiLayer.h>
#endif //CUPTI

#ifdef TAUKTAU_SHCTR
#include "Profile/KtauCounters.h"
#endif //TAUKTAU_SHCTR

#ifdef TAU_WINDOWS
#define strcasecmp stricmp
#endif

#ifdef TAU_LIKWID
#include <likwid.h>
#endif /* TAU_LIKWID */

//using namespace std;
using namespace tau;

// This would be more useful in a utility header somewhere, but the way people slap 'extern "C"'
// on everything means we'll probably wind up with an C-linked template at some point...
template<typename T>
struct ScopedArray {
    ScopedArray(size_t count) :
            size(count * sizeof(T)), ptr(new T[count]) {
    }
    ~ScopedArray() {
        if (ptr)
            delete[] ptr;
    }
    operator T*() const {
        return ptr;
    }
    size_t size;
    T * const ptr;
};

void metric_read_nullClock(int tid, int idx, double values[]);
void metric_write_userClock(int tid, double value);
int metric_get_num_clocks();
void metric_read_userClock(int tid, int idx, double values[]);
void metric_read_logicalClock(int tid, int idx, double values[]);
void metric_read_gettimeofday(int tid, int idx, double values[]);
void metric_read_clock_gettime(int tid, int idx, double values[]);
void metric_read_linuxtimers(int tid, int idx, double values[]);
void metric_read_bgtimers(int tid, int idx, double values[]);
void metric_read_craytimers(int tid, int idx, double values[]);
void metric_read_cputime(int tid, int idx, double values[]);
void metric_read_cpuenergy(int tid, int idx, double values[]);
void metric_read_accelenergy(int tid, int idx, double values[]);
void metric_read_messagesize(int tid, int idx, double values[]);
void metric_read_papivirtual(int tid, int idx, double values[]);
void metric_read_papiwallclock(int tid, int idx, double values[]);
void metric_read_papi(int tid, int idx, double values[]);
void metric_read_likwid(int tid, int idx, double values[]);
void metric_read_ktau(int tid, int idx, double values[]);
void metric_read_cudatime(int tid, int idx, double values[]);
void metric_read_cupti(int tid, int idx, double values[]);
void metric_read_memory(int tid, int idx, double values[]);

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

int TauMetrics_init();
int Tau_init_check_initialized(void);

static void metricv_add(const char *name);
static void read_env_vars();
static void initialize_functionArray();
static bool functionsInitialized(false);

#ifndef TAU_MAX_METRICS
#define TAU_MAX_METRICS 25
#endif
/* Global Variable holding the number of counters */
int Tau_Global_numCounters = -1;
int Tau_Global_numGPUCounters = 0;

static TauUserEvent **traceCounterEvents;

typedef void (*tau_function)(int, int, double[]);

static char *metricv[TAU_MAX_METRICS];
static int nmetrics = 0;
static TauMetricCuptiFlag cumetric[TAU_MAX_METRICS];
static int eventsv[TAU_MAX_METRICS] = {0};
static double defaults[TAU_MAX_METRICS] = {0}; // used for values read before initialization

/* nfunctions can be different from nmetrics because
 a single call to PAPI can provide several metrics */
static int nfunctions = 0;

/* traceMetric in the index used for the trace metric (might not be zero) */
static int traceMetric = 0;

/* array of function pointers used to get metric data */
static tau_function functionArray[TAU_MAX_METRICS];

/* gtod based initial timestamp, used for snapshots and other stuff */
static x_uint64 initialTimeStamp = 0L;
/* gtod based final timestamp, so all threads agree when we exited */
static x_uint64 finalTimeStamp = 0L;

/* flags for atomic metrics */
char *TauMetrics_atomicMetrics[TAU_MAX_METRICS] = { NULL };

#ifdef CUPTI
static int cuda_device_count() {
    int deviceCount;
    CUresult result = cuDeviceGetCount(&deviceCount);
    if (result == CUDA_ERROR_NOT_INITIALIZED) {
        cuInit(0);
        result = cuDeviceGetCount(&deviceCount);
    }
    if (result != CUDA_SUCCESS) {
        char const * err_str;
        cuGetErrorString(result, &err_str);
        //fprintf(stderr, "cuDeviceGetCount failed: %s\n", err_str);
        return 0;
    }
    return deviceCount;
}
#endif

static void check_max_metrics() {
    if (nmetrics >= TAU_MAX_METRICS) {
        fprintf(stderr,
                "Number of counters exceeds TAU_MAX_METRICS (%d), "
                        "please reconfigure TAU with -useropt=-DTAU_MAX_METRICS=<higher number>.\n",
                TAU_MAX_METRICS);
        exit (EXIT_FAILURE);
    }
}

/*********************************************************************
 * Add a metric to the metrics vector
 ********************************************************************/
static void metricv_add(const char *name) {
    int cupti_metric = 0;

    TAU_VERBOSE("entering metricv_add, adding metric %s\n", name);

    // Don't add metrics twice
    for (int i = 0; i < nmetrics; ++i) {
        if (strcasecmp(metricv[i], name) == 0) {
            return;
        }
    }

    check_max_metrics();

#ifdef CUPTI
    char const * const tau_cuda_device_name = TauEnv_get_cuda_device_name();

    // Get events required to compute CUPTI metric
    for(int dev=0; dev<cuda_device_count(); ++dev) {
        /* skip the usual suspects */
        if (strncmp(name, "TAU", 3) == 0) { continue; }
        if (strncmp(name, "TIME", 4) == 0) { continue; }
        if (strncmp(name, "PAPI", 4) == 0) { continue; }

        CUptiResult result;
        CUdevice device;
        cudaDeviceProp deviceProps;
        if (cuDeviceGet(&device, dev) != CUDA_SUCCESS) {
            fprintf(stderr, "Could not get device %d.\n", dev);
            continue;
        }

        // Check if metric is a CUPTI metric we can calculate on this device
        CUpti_MetricID metricID;
        result = cuptiMetricGetIdFromName(device, name, &metricID);
        cupti_metric = (result == CUPTI_SUCCESS);
        if (!cupti_metric) continue;

        // Get the device name to be used in the event name below
        cudaGetDeviceProperties(&deviceProps, dev);
        std::string device_name = deviceProps.name;
        //std::replace(device_name.begin(), device_name.end(), ' ', '_');
        // PGI compiler has problems with -c++11
        Tau_util_replaceStringInPlace(device_name, " ", "_");
        if (tau_cuda_device_name && strcmp(tau_cuda_device_name, device_name.c_str())) {
            continue;
        }

        // Get the list of events required to calculate this metric on this device
        uint32_t numMetricEvents;
        result = cuptiMetricGetNumEvents(metricID, &numMetricEvents);
        if (result != CUPTI_SUCCESS) {
            fprintf(stderr, "cuptiMetricGetNumEvents failed on device %d\n", dev);
            continue;
        }
        CUpti_EventID metricEvents[numMetricEvents];
        size_t metricEvents_size = numMetricEvents * sizeof(CUpti_EventID);
        result = cuptiMetricEnumEvents(metricID, &metricEvents_size, metricEvents);
        if (result != CUPTI_SUCCESS) {
            fprintf(stderr, "cuptiMetricEnumEvents failed on device %d\n", dev);
            continue;
        }

        // add events to metricv

        for (int i = 0; i < numMetricEvents; i++) {
            CUpti_EventID event = metricEvents[i];
            char buff[TAU_CUPTI_MAX_NAME];
            size_t buff_size = sizeof(buff);
            result = cuptiEventGetAttribute(event, CUPTI_EVENT_ATTR_NAME, &buff_size, buff);
            if (result != CUPTI_SUCCESS) {
                fprintf(stderr, "cuptiEventGetAttribute failed for event %d on device %d\n", event, dev);
                continue;
            }
            if (buff_size == sizeof(buff)) {
                fprintf(stderr, "TAU_CUPTI_MAX_NAME=%d is too small for event name!\n", TAU_CUPTI_MAX_NAME);
                exit(EXIT_FAILURE);
            }
            if (numMetricEvents == 1) {
                snprintf(buff, sizeof(buff),  "%s", name);
            } else {
                // some events don't have proper names, so just use event id instead
                if (std::string(buff).compare("event_name") == 0) {
                    snprintf(buff, sizeof(buff),  "%s.%d", name, event);
                    //sprintf(buff, "CUpti_EventID.%s", name);
                }
            }

            std::string event_name = "CUDA." + device_name + '.' + std::string(buff);

            if (!Tau_CuptiLayer_is_cupti_counter(event_name.c_str())) {
                // double check because it just got initialized
                if (!Tau_CuptiLayer_is_cupti_counter(event_name.c_str())) {
                    CuptiCounterEvent* ev = new CuptiCounterEvent(dev, event, buff);
                    Tau_CuptiLayer_Counter_Map().insert(std::make_pair(event_name, ev));
                }
            }

            TAU_VERBOSE("%s: %s\n", name, event_name.c_str());

            // Add event to metricv if it's not already on the list.
            bool found = false;
            for (int k=0; k<nmetrics; ++k) {
                if (strcasecmp(metricv[k], event_name.c_str()) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                check_max_metrics();
                metricv[nmetrics] = strdup(event_name.c_str());
                eventsv[nmetrics] = event; // This looks weird... is this right?
                cumetric[nmetrics] = TAU_METRIC_CUPTI_EVENT;
                nmetrics++;
                Tau_Global_numGPUCounters++;
            }
        } // for (event)
    } // for (dev)
#endif //CUPTI

    check_max_metrics();
    metricv[nmetrics] = strdup(name);
    eventsv[nmetrics] = 0;
    cumetric[nmetrics] =
            cupti_metric ? TAU_METRIC_CUPTI_METRIC : TAU_METRIC_NOT_CUPTI;
    if (cumetric[nmetrics] == TAU_METRIC_CUPTI_METRIC) Tau_Global_numGPUCounters++;
    nmetrics++;
    TAU_VERBOSE("exiting metricv_add, adding metric %s\n", name);
}

/*********************************************************************
 * This routine will reorder the metrics so that the PAPI ones all come last
 * Note: traceMetric must already be set
 ********************************************************************/
static void reorder_metrics(const char *match) {
    char *newMetricV[TAU_MAX_METRICS];
    int idx = 0;
    int newTraceMetric = 0;

    for (int i = 0; i < nmetrics; i++) {
        if (strncmp(match, metricv[i], strlen(match)) != 0) {
            newMetricV[idx++] = metricv[i];
        }
    }

    for (int i = 0; i < nmetrics; i++) {
        if (strncmp(match, metricv[i], strlen(match)) == 0) {
            newMetricV[idx++] = metricv[i];
        }
    }

    for (int i = 0; i < nmetrics; i++) {
        if (strcasecmp(newMetricV[i], metricv[traceMetric]) == 0) {
            newTraceMetric = i;
        }
    }

    for (int i = 0; i < nmetrics; i++) {
        metricv[i] = newMetricV[i];
    }

    traceMetric = newTraceMetric;
}

/*********************************************************************
 * Read the configuration entries into the metric vector
 ********************************************************************/
static void read_env_vars() {
    const char *token;
    const char *taumetrics = TauEnv_get_metrics();
    char *ptr, *ptr2;
    int len = strlen(taumetrics);
    int i;
    bool alt_delimiter_found = false;

    if (taumetrics && len == 0) {
        taumetrics = NULL;
    }

    if (taumetrics) {
        char *metrics = strdup(taumetrics);
        for (i = 0; i < len; i++) {
            if ((taumetrics[i] == ',') || (taumetrics[i] == '|')) {
                alt_delimiter_found = true;
                //printf("ALT delimiter found: taumetrics[%d] = %c\n", i, taumetrics[i]);
                break;
            }
        }
        for (ptr = metrics; *ptr; ptr++) {
            if (*ptr == '\\') {
                /* escaped, skip over */
                for (ptr2 = ptr; *(ptr2); ptr2++) {
                    *ptr2 = *(ptr2 + 1);
                }
                ptr++;
            } else {
                if (alt_delimiter_found) {
                    //printf("Alt_delimiter = %d, ptr = %c\n", alt_delimiter_found, *ptr);
                    if ((*ptr == '|') || (*ptr == ',')) {
                        // printf("Checking for | or , in %s\n", metrics);
                        *ptr = '^';
                    }
                } else {
                    // printf("Alt_delimiter = %d, ptr = %c\n", alt_delimiter_found, *ptr);
                    if (*ptr == ':') {
                        // printf("Checking for : in %s\n", metrics);
                        *ptr = '^';
                    }
                }
            }
        }

        token = strtok(metrics, "^");
        while (token) {
            metricv_add(token);
            token = strtok(NULL, "^");
        }
    } else {
        char counterName[256];
        for (int i = 1; i < 26; i++) {
            snprintf(counterName, sizeof(counterName),  "COUNTER%d", i);
            char *metric = getenv(counterName);
            if (metric && strlen(metric) == 0) {
                metric = NULL;
            }
            if (metric) {
                metricv_add(metric);
            }
        }

        if (nmetrics == 0) {
#if defined(BGL_TIMERS)
            metricv_add("BGL_TIMERS");
#elif defined(BGP_TIMERS)
            metricv_add("BGP_TIMERS");
#elif defined(BGQ_TIMERS)
            metricv_add("BGQ_TIMERS");
#else
            // NOTE: Probably need more #elif defined() for other platforms...
            metricv_add("TIME");
#endif
        }
    }
}

/*********************************************************************
 * Initialize KTAU metrics
 ********************************************************************/
#ifdef TAUKTAU_SHCTR
static void TauMetrics_initializeKTAU() {
    for (int i = 0; i < nmetrics; i++) {
        int cType = 0;
        if (strncmp("KTAU", metricv[i], 4) == 0) {
            if (strstr(metricv[i], "KTAU_INCL_") != NULL) {
                cType = KTAU_SHCTR_TYPE_INCL;
            } else if (strstr(metricv[i], "KTAU_NUM_") != NULL) {
                cType = KTAU_SHCTR_TYPE_NUM;
            } else {
                cType = KTAU_SHCTR_TYPE_EXCL;
            }
            char *metric = strdup(metricv[i]);
            metric = metric + 5; /* strip "KTAU_" */
            KtauCounters::addCounter(metric, cType);
        }
    }
}
#endif

/*********************************************************************
 * Query if a string is a PAPI metric
 ********************************************************************/
static int is_papi_metric(char *str) {
    if (strncmp("PAPI", str, 4) == 0) {
        if (strcasecmp(str, "PAPI_TIME") != 0
                && strcasecmp(str, "PAPI_VIRTUAL_TIME") != 0) {
            return 1;
        }
    }
    return 0;
}

/*********************************************************************
 * Query if a string is a LIKWID metric
 ********************************************************************/
static int is_likwid_metric(char *str) {
    if (strncmp("LIKWID", str, 6) == 0) {

        return 1;

    }
    return 0;
}

#ifdef CUPTI
/*********************************************************************
 * Query if a string is a CUPTI event
 ********************************************************************/
static int is_cupti_event(char const * str)
{
    if (strncmp("CUDA", str, 4) == 0 && Tau_CuptiLayer_is_cupti_counter(str)) {
        return 1;
    }
    return 0;
}

/*********************************************************************
 * Query if a string is a CUPTI metric
 ********************************************************************/
static int is_cupti_metric(char const * const str)
{
    if (strncmp(str, "TAU", 3) == 0) { return 0; }
    if (strncmp(str, "TIME", 4) == 0) { return 0; }
    if (strncmp(str, "PAPI", 4) == 0) { return 0; }
    CUpti_MetricID metricid;
    CUdevice device;

    for (int dev=0; dev<cuda_device_count(); ++dev) {
        if (cuDeviceGet(&device, dev) != CUDA_SUCCESS) {
            fprintf(stderr, "Could not get device %d.\n", dev);
            return 0;
        }
        return (cuptiMetricGetIdFromName(device, str, &metricid) == CUPTI_SUCCESS);
    }
    return 0;
}

#endif //CUPTI

/*********************************************************************
 * Initialize the function array
 ********************************************************************/
static void initialize_functionArray() {
    int pos = 0;
    int found = 0;
    int ktau = 0;
#ifdef TAUKTAU_SHCTR
    ktau = 1;
#endif

    int papi_available = 0;
#ifdef TAU_PAPI
    int usingPAPI = 0;
    papi_available = 1;
#endif

    int likwid_available = 0;
#ifdef TAU_LIKWID
    int usingLIKWID = 0;
    likwid_available = 1;
    int add = 0;
    int temp[1] = {0};
    TAU_VERBOSE("Before perfmon_init\n");
    int ret = perfmon_init(1, (int*)temp);
    TAU_VERBOSE("perfmon_init %d\n", ret);
    for (int i = 0; i < nmetrics; i++) {
        if (is_likwid_metric(metricv[i]) && (!strstr(metricv[i], ":"))) {
            TAU_VERBOSE("Adding %s temporarily\n", &metricv[i][7]);
            int gid = perfmon_addEventSet(&metricv[i][7]);
            TAU_VERBOSE("GID %d numEvents %d\n", gid, perfmon_getNumberOfEvents(gid));
            for (int j = 0; j < perfmon_getNumberOfEvents(gid); j++)
            {
                char foo[1024];
                int ret = snprintf(foo, 1023, "LIKWID_%s:%s", perfmon_getEventName(gid, j), perfmon_getCounterName(gid, j));
                if (ret > 0)
                {
                    foo[ret] = '\0';

                    TAU_VERBOSE("Adding %s at %d\n", foo, nmetrics + add);
                    metricv[nmetrics + add] = strdup(foo);
                    add++;
                }
            }
            free(metricv[i]);
            metricv[i] = metricv[nmetrics];
            for (int j = nmetrics; j < nmetrics+add-1; j++)
            {
                metricv[j] = metricv[j+1];
            }
            add--;
        }
    }
    perfmon_finalize();
    TAU_VERBOSE("Expand metrics list from %d to %d\n", nmetrics, nmetrics + add);
    nmetrics += add;
#endif


    for (int i = 0; i < nmetrics; i++) {
        found = 1;
        if (strcasecmp(metricv[i], "LOGICAL_CLOCK") == 0) {
            functionArray[pos++] = metric_read_logicalClock;
        } else if (strcasecmp(metricv[i], "USER_CLOCK") == 0) {
            functionArray[pos++] = metric_read_userClock;
        } else if (strcasecmp(metricv[i], "GET_TIME_OF_DAY") == 0) {
            functionArray[pos++] = metric_read_gettimeofday;
        } else if (strcasecmp(metricv[i], "CLOCK_GET_TIME") == 0) {
            functionArray[pos++] = metric_read_clock_gettime;
        } else if (strcasecmp(metricv[i], "TIME") == 0) {
            functionArray[pos++] = metric_read_gettimeofday;
        } else if (strcasecmp(metricv[i], "CPU_TIME") == 0) {
            functionArray[pos++] = metric_read_cputime;
        } else if (strcasecmp(metricv[i], "ENERGY") == 0) {
            functionArray[pos++] = metric_read_cpuenergy;
        } else if (strcasecmp(metricv[i], "ACCEL_ENERGY") == 0) {
            functionArray[pos++] = metric_read_accelenergy;
#ifdef TAU_LINUX_TIMERS
        } else if (strcasecmp(metricv[i], "LINUX_TIMERS") == 0) {
            functionArray[pos++] = metric_read_linuxtimers;
#endif
        } else if (strcasecmp(metricv[i], "BGL_TIMERS") == 0) {
            functionArray[pos++] = metric_read_bgtimers;
        } else if (strcasecmp(metricv[i], "BGP_TIMERS") == 0) {
            functionArray[pos++] = metric_read_bgtimers;
        } else if (strcasecmp(metricv[i], "BGQ_TIMERS") == 0) {
            functionArray[pos++] = metric_read_bgtimers;
        } else if (strcasecmp(metricv[i], "CRAY_TIMERS") == 0) {
            functionArray[pos++] = metric_read_craytimers;
        } else if (strcasecmp(metricv[i], "TAU_MPI_MESSAGE_SIZE") == 0) {
            functionArray[pos++] = metric_read_messagesize;
#ifdef CUPTI
        } else if (is_cupti_event(metricv[i])) {
            /* CUPTI handled separately */
            /* setup CUPTI metrics */
            //functionArray[pos++] = metric_read_cupti;
            Tau_CuptiLayer_register_string(metricv[i], pos);
            TauMetrics_atomicMetrics[pos] = metricv[i];
            functionArray[pos++] = metric_read_cupti;
        } else if(is_cupti_metric(metricv[i])) {
            /* Cupti metrics handled separately */
#endif //CUPTI
#ifdef TAU_PAPI
        } else if (strcasecmp(metricv[i], "P_WALL_CLOCK_TIME") == 0) {
            usingPAPI = 1;
            functionArray[pos++] = metric_read_papiwallclock;
        } else if (strcasecmp(metricv[i], "PAPI_TIME") == 0) {
            usingPAPI = 1;
            functionArray[pos++] = metric_read_papiwallclock;
        } else if (strcasecmp(metricv[i], "P_VIRTUAL_TIME") == 0) {
            usingPAPI = 1;
            functionArray[pos++] = metric_read_papivirtual;
        } else if (strcasecmp(metricv[i], "PAPI_VIRTUAL_TIME") == 0) {
            usingPAPI = 1;
            functionArray[pos++] = metric_read_papivirtual;
#endif /* TAU_PAPI */
        } else if (strcasecmp(metricv[i], "TAUGPU_TIME") == 0) {
            functionArray[pos++] = metric_read_cudatime;
        } else if (strcasecmp(metricv[i], "MEMORY_DELTA") == 0) {
            functionArray[pos++] = metric_read_memory;
        } else {
            if (papi_available && is_papi_metric(metricv[i])) {
                /* PAPI handled separately */
            } else if (likwid_available && is_likwid_metric(metricv[i])) {
                /* LIKWID handled separately */
            } else if (ktau && strncmp("KTAU", metricv[i], 4) == 0) {
                /* KTAU handled separately */
            } else {
                fprintf(stderr, "TAU: Error: Unknown metric: %s\n", metricv[i]);

                /* Delete the metric */
                for (int j = i; j < nmetrics - 1; j++) {
                    metricv[j] = metricv[j + 1];
                }
                nmetrics--;
                i--;
                found = 0;

                /* old: null clock
                 functionArray[pos++] = metric_read_nullClock;
                 */
            }
        }
        if (found) {
            TAU_VERBOSE("TAU: Using metric: %s\n", metricv[i]);
        }
    }

    /* check if we are using PAPI */
    for (int i = 0; i < nmetrics; i++) {
        if (is_papi_metric(metricv[i])) {
            functionArray[pos++] = metric_read_papi;
#ifdef TAU_PAPI
            usingPAPI = 1;
#endif /* TAU_PAPI */
            break;
        }
    }
        /* check if we are using LIKWID */
        for (int i = 0; i < nmetrics; i++) {
            if (is_likwid_metric(metricv[i])) {
                functionArray[pos++] = metric_read_likwid;
                //printf("Added Likwid to functionArray\n");
#ifdef TAU_LIKWID
                usingLIKWID = 1;
#endif /* TAU_LIKWID */
                break;
            }
        }

#ifdef TAUKTAU_SHCTR
        for (int i = 0; i < nmetrics; i++) {
            if (strncmp("KTAU", metricv[i], 4) == 0) {
                functionArray[pos++] = metric_read_ktau;
                break;
            }
        }
        TauMetrics_initializeKTAU();
#endif

#ifdef TAU_PAPI
        if (usingPAPI) {
            PapiLayer::initializePapiLayer();
        }
#endif

#ifdef TAU_LIKWID
        if (usingLIKWID) {
            LikwidLayer::initializeLikwidLayer();
        }
#endif
        //printf("adding %d metrics\n", nmetrics);
#ifdef TAU_LIKWID
        bool firstLikwidString=true;
        int numLikwidEvents=0;
#endif
        std::string likwidEventString;
        for (int i = 0; i < nmetrics; i++) {

            if (is_likwid_metric(metricv[i])) {
                //printf("IS LIKWID METRIC: %s\n", metricv[i]);
                if (strstr(metricv[i], "LIKWID") != NULL) {
                    char *metricString = strdup(metricv[i]);
                    int idx = 0;
                    while (metricString[7 + idx] != '\0') {
                        metricString[idx] = metricString[7 + idx];
                        idx++;
                    }
                    metricString[idx] = '\0';
#ifdef TAU_LIKWID
                    if (usingLIKWID)
                    {
                        //printf("adding metric string: %s\n",metricString);
                        if(firstLikwidString){
                            likwidEventString = std::string(metricString);
                            firstLikwidString=false;
                          }
                        else{
                            likwidEventString=likwidEventString+","+std::string(metricString);
                        }
                        numLikwidEvents++;
                    }
#endif
                    free(metricString);
                }
            } else if (is_papi_metric(metricv[i])) {
                if (strstr(metricv[i], "PAPI") != NULL) {
                    char *metricString = strdup(metricv[i]);

                    if (strstr(metricString, "NATIVE") != NULL) {
                        /* Fix the name for a native event */
                        int idx = 0;
                        while (metricString[12 + idx] != '\0') {
                            metricString[idx] = metricString[12 + idx];
                            idx++;
                        }
                        metricString[idx] = '\0';
                    }

#ifdef TAU_PAPI
                    int counterID = PapiLayer::addCounter(metricString);
                    if (counterID == -1) {
                        /* Delete the metric */
                        for (int j = i; j < nmetrics - 1; j++) {
                            metricv[j] = metricv[j + 1];
                        }
                        nmetrics--;
                    }
#endif
                    free(metricString);
                }
            }
        }

#ifdef TAU_LIKWID
                    if (usingLIKWID)
                    {
                        LikwidLayer::addEvents(likwidEventString.c_str()); //"L2_LINES_IN_ALL:PMC0,L2_TRANS_L2_WB:PMC1");    LIKWID_L1D_REPLACEMENT:PMC0
                    }
#endif

    nfunctions = pos;
    functionsInitialized = true;
}

/*********************************************************************
 * Returns metric name for an index
 ********************************************************************/
extern "C" const char *TauMetrics_getMetricName(int metric) {
    char const * metric_name = metricv[metric];
#ifdef CUPTI
    // Don't bother checking if it's a CUPTI counter if it's obviously not.
    if( (strncmp(metric_name, "TAU", 3) == 0)
            || (strncmp(metric_name, "TIME", 4) == 0)
            || (strncmp(metric_name, "PAPI", 4) == 0) ) {
        return metric_name;
    }
    int event_id = Tau_CuptiLayer_get_cupti_event_id(metric);
    if (Tau_CuptiLayer_is_cupti_counter(metric_name) &&
        event_id < Tau_CuptiLayer_get_num_events()) {
        return Tau_CuptiLayer_get_event_name(event_id);
    }
#endif
    return metric_name;
}

/*********************************************************************
 * Query if a metric is used
 ********************************************************************/
int TauMetrics_getMetricUsed(int metric) {
    if (metric < nmetrics) {
        return 1;
    } else {
        return 0;
    }
}

/*********************************************************************
 * Query if a metric is Cupti metric or event
 ********************************************************************/
int TauMetrics_getIsCuptiMetric(int metric) {
    return cumetric[metric];
}

/*********************************************************************
 * Query if a metric is atomic
 ********************************************************************/
const char* TauMetrics_getMetricAtomic(int metric) {
    return TauMetrics_atomicMetrics[metric];
}

/*********************************************************************
 * Get id of time metric
 ********************************************************************/
int TauMetrics_getTimeMetric() {
#ifdef CUPTI
    char const * const time = "TAUGPU_TIME";
#else
    char const * const time = "TIME";
#endif
    for (int i = 0; i < nmetrics; i++) {
        if (strcasecmp(metricv[i], time) == 0)
            return i;
    }
    return -1;
}

/*********************************************************************
 * Get event id
 ********************************************************************/
int TauMetrics_getEventId(int metric) {
    return eventsv[metric];
}

/*********************************************************************
 * Get event index from event id
 ********************************************************************/
int TauMetrics_getEventIndex(int eventid) {
    for (int i = 0; i < nmetrics; i++) {
        if (eventid == eventsv[i])
            return i;
    }
    return -1;
}

/*********************************************************************
 * Read the metrics
 ********************************************************************/
extern "C" bool TauCompensateInitialized(void);
void TauMetrics_getMetrics(int tid, double values[], int reversed) {
        if (!functionsInitialized) {
           TauMetrics_init();
        }
    if (functionsInitialized) {
        if (reversed) {
            for (int i=nfunctions-1; i >= 0; --i) {
                functionArray[i](tid, i, values);
            }
        } else {
            for (int i=0; i < nfunctions; i++) {
                functionArray[i](tid, i, values);
            }
        }
    } else {
        // *CWL* - Safe only if Compensation is safely initialized. Otherwise
        //         we would be in the middle of re-entrant behavior and
        //         would be re-initializing metrics each time.
        fprintf(stderr,"TAU: ERROR: TauMetrics not initialized!\n");
        if (TauCompensateInitialized()) {
            TauMetrics_init();
        }
        // we need to give some value to the tracer or it will take 0
        // as the default value which will mess up traces. This is
        // seen in roctracer (AMD).
                metric_read_gettimeofday(tid, 0, &values[0]);
    }
}

void TauMetrics_getDefaults(int tid, double values[], int reversed) {
    if (functionsInitialized) {
        if (reversed) {
            for (int i=nfunctions-1; i >= 0; --i) {
                values[i] = defaults[i];
            }
        } else {
            for (int i=0; i < nfunctions; i++) {
                values[i] = defaults[i];
            }
        }
    }
}

extern "C" void TauMetrics_internal_alwaysSafeToGetMetrics(int tid,
        double values[]) {
    for (int i = 0; i < nfunctions; i++) {
        functionArray[i](tid, i, values);
    }
}

extern "C" x_uint64 TauMetrics_getInitialTimeStamp() {
    return initialTimeStamp;
}

extern "C" x_uint64 TauMetrics_getFinalTimeStamp() {
    return finalTimeStamp;
}

extern "C" x_uint64 TauMetrics_getTimeOfDay() {
    x_uint64 timestamp;
#ifdef TAU_WINDOWS
    timestamp = TauWindowsUsecD();
#else
    struct timeval tp;
    gettimeofday(&tp, 0);
    timestamp = (x_uint64) tp.tv_sec * (x_uint64) 1e6 + (x_uint64) tp.tv_usec;
#endif
    return timestamp;
}

/*********************************************************************
 * Initialize the metrics module
 ********************************************************************/
int TauMetrics_init() {
    // Why lock?  Well, other threads can get spawned by this metric
    // initialization process, so don't allow anyone to try to take
    // measurements until after metrics are ready.
    RtsLayer::LockDB();

    TAU_VERBOSE("entering TauMetrics_init\n");

    int i;

    initialTimeStamp = TauMetrics_getTimeOfDay();

    if (TauEnv_get_ebs_enabled()) {
        // *CWL* - keep an eye on this. *must* we do this? Or can we
        //         keep PAPI overflow signal triggers separate from
        //         user-selected metrics to be measured.
        if (strcasecmp(TauEnv_get_ebs_source(), "itimer") != 0) {
            metricv_add (TauEnv_get_ebs_source());}
    }

		/* Set the user clock values to 0 */
    int numClocks=metric_get_num_clocks();
	for (i = 0; i < numClocks; i++) {
		metric_write_userClock(i, 0);
	}

    read_env_vars();
    //for(i = 0; i < nmetrics; i++)
    //  printf("metricv[%d] = %s\n", i, metricv[i]);

    traceMetric = 0;
    reorder_metrics("PAPI\0");
    reorder_metrics("KTAU\0");

    initialize_functionArray();
    // set the "default" values for timers started before we were ready
    TauMetrics_getMetrics(Tau_get_thread(), defaults, 0);

    Tau_Global_numCounters = nmetrics;

    /* Create atomic events for tracing */
    if (TauEnv_get_tracing()) {
        traceCounterEvents = new TauUserEvent *[nmetrics];
        /* We obtain the timestamp from COUNTER1, so we only need to trigger
         COUNTER2-N or i=1 through no. of active functions not through 0 */
        std::string illegalChars("/\\?%*:|\"<> ");
        for (i = 1; i < nmetrics; i++) {
            //sanitize metricName before using it to create a name
            std::string metricStr = std::string(metricv[i]);
            size_t found;
            found = metricStr.find_first_of(illegalChars, 0);
            while (found != std::string::npos) {
                metricStr[found] = '_';
                found = metricStr.find_first_of(illegalChars, found + 1);
            }

            //traceCounterEvents[i] = new TauUserEvent(metricv[i], true);
            traceCounterEvents[i] = new TauUserEvent(metricStr.c_str(), true);
            /* the second arg is MonotonicallyIncreasing which is true (HW counters)*/
        }
    }

    TAU_VERBOSE("exiting TauMetrics_init\n");
    RtsLayer::UnLockDB();
    return 0;
}

/*********************************************************************
 * Finalize the metrics module
 ********************************************************************/
void TauMetrics_finalize() {
    if (finalTimeStamp == 0L) {
        finalTimeStamp = TauMetrics_getTimeOfDay();
    }
}

/*********************************************************************
 * Trigger atomic events for each metric
 ********************************************************************/
void TauMetrics_triggerAtomicEvents(unsigned long long timestamp,
        double *values, int tid) {
    int i;
#ifndef TAU_EPILOG
    for (i = 1; i < nmetrics; i++) {
        TauTraceEvent(traceCounterEvents[i]->GetId(), (long long) values[i],
                tid, timestamp, 1, TAU_TRACE_EVENT_KIND_USEREVENT);
        // 1 in the last parameter is for use timestamp
    }
#endif /* TAU_EPILOG */
}

/*********************************************************************
 * Returns a duplicated list of counter names, and writes the number
 * of counters in numCounters
 ********************************************************************/
void TauMetrics_getCounterList(const char ***counterNames, int *numCounters) {
    *numCounters = nmetrics;
    *counterNames = (char const **) malloc(sizeof(char *) * nmetrics);
    for (int i = 0; i < nmetrics; i++) {
        (*counterNames)[i] = strdup(TauMetrics_getMetricName(i));
    }
}

/*********************************************************************
 * Returns the index of the trace metric
 ********************************************************************/
double TauMetrics_getTraceMetricIndex() {
    return traceMetric;
}

/*********************************************************************
 * Returns the value of the trace metric
 ********************************************************************/
double TauMetrics_getTraceMetricValue(int tid) {
    double values[TAU_MAX_COUNTERS];
    TauMetrics_getMetrics(tid, values, 0);
    return values[traceMetric];
}

/**********************************************************************
 * Returns the index in the metric vector of a particular metric string
 *    *CWL* Intended for EBS to resolve the index location of the metric
 *    source at data output time. This assumes the metric vector does
 *    not change between the time it is read and when it is used. This
 *    is not true, I believe, so we need to keep an eye on this.
 *    There are also other issues of name matching ... EBS (this ought
 *    to be changed) uses "itimer" instead of "TIME". So, we are going
 *    to do the unethical practice of assuming it is TIME if there is
 *    a failure to match the presented name against the names in the
 *    metric vector.
 **********************************************************************/
int TauMetrics_getMetricIndexFromName(const char *metricString) {
    for (int i = 0; i < nmetrics; i++) {
        if (strcasecmp(metricv[i], metricString) == 0) {
            return i;
        }
    }
    /* EBS ONLY:
     Try again assuming TIME. The reason we loop again is because in the
     general case, TIME isn't necessarily in position 0. */
    if (TauEnv_get_ebs_enabled()) {
        for (int i = 0; i < nmetrics; i++) {
            if (strcasecmp(metricv[i], "TIME") == 0) {
                return i;
            }
        }
    }
    /* This is a bad failure value. EBS cannot handle this. Other features
     might. */
    return -1;
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

