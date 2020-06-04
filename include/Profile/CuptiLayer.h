#ifndef CUPTI_LAYER_H
#define CUPTI_LAYER_H

#ifdef __GNUC__
#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

#include <cupti_version.h>
#include <cupti_events.h>
#include <cupti_metrics.h>
#include <cuda_runtime_api.h>
#include "cupti.h"

/* Specific errors from CUDA lib */
#define CHECK_CU_ERROR(err, cufunc) \
if (err != CUDA_SUCCESS) \
{ \
		const char* err_name; \
    const char* err_str; \
		cuGetErrorName(err, &err_name); \
		cuGetErrorString(err, &err_str); \
		fprintf(stderr, "CUDA driver error %s: %s\n", err_name, err_str); \
printf ("[%s:%d] Error %d for CUDA Driver API function '%s'. cuptiQuery failed\n", __FILE__, __LINE__, err, cufunc); \
}

/* Specific errors from CuPTI lib */
#define CHECK_CUPTI_ERROR(err, cuptifunc) \
if (err != CUPTI_SUCCESS) \
{ \
const char * tmpstr;  \
cuptiGetResultString(err, &tmpstr); \
printf ("[%s:%d] Error %d for CUPTI API function '%s'. cuptiQuery failed\n%s\n", __FILE__, __LINE__, err, cuptifunc, tmpstr); \
}

#define TAU_CUPTI_MAX_NAME 40
#define TAU_CUPTI_MAX_DESCRIPTION 480

#define TAU_CUPTI_COUNTER_ACTUAL 0
#define TAU_CUPTI_COUNTER_BOUNDED 1
#define TAU_CUPTI_COUNTER_AVERAGED 2

//This setting will aggregate the event values collected across all event
//domains. Thus the event results will report values as if all SM had an event
//domain available to collect this value. WARNING: If the kernel being measured
//is not large enough to utilize all the available SMs on a device this
//aggregation will result in skewed data.
#define TAU_CUPTI_NORMALIZE_EVENTS_ACROSS_ALL_SMS

//#define DISABLE_CUPTI


// These really should come from TauInit.h and TauAPI.h but TAU's include files are so
// jacked up that including those files causes weird errors.  Just work around the problem...
extern "C" int Tau_init_initializeTAU();
extern "C" void Tau_destructor_trigger();

struct CuptiCounterEvent
{
    static void printHeader();

    CuptiCounterEvent(int device_n, int event_n, const char * name);

    CUdevice device;
	CUpti_EventID event;

	std::string device_name;
	std::string event_name;
	std::string event_description;
	std::string tag; // string presented to the user.

	void print();
};

struct CuptiCounterMap: public std::map<std::string, CuptiCounterEvent*>
{
    CuptiCounterMap() {
        Tau_init_initializeTAU();
    }
    ~CuptiCounterMap() {
        Tau_destructor_trigger();
    }
};
typedef CuptiCounterMap counter_map_t;
typedef CuptiCounterMap::iterator counter_map_it;

struct CuptiCounterVector: public std::vector<CuptiCounterEvent*>
{
    CuptiCounterVector() {
        Tau_init_initializeTAU();
    }
    ~CuptiCounterVector() {
        Tau_destructor_trigger();
    }
};
typedef CuptiCounterVector counter_vec_t;

struct CuptiCounterIdMap : public std::map<int, int>
{
    CuptiCounterIdMap() {
        Tau_init_initializeTAU();
    }
    ~CuptiCounterIdMap() {
        Tau_destructor_trigger();
    }
};
typedef CuptiCounterIdMap counter_id_map_t;

counter_vec_t & Tau_CuptiLayer_Added_counters(void);

struct CuptiMetric
{
    static void printHeader();

    CuptiMetric(int device_n, int metric_n);

    CUdevice device;
    CUpti_EventDomainID domain;
    CUpti_MetricID metric;

    std::string device_name;
    std::string metric_name;
    std::string metric_description;
    std::string tag; // string presented to the user.

    void print();
};

struct CuptiMetricMap: public std::map<std::string, CuptiMetric*>
{
    CuptiMetricMap() {
        Tau_init_initializeTAU();
    }
    ~CuptiMetricMap() {
        Tau_destructor_trigger();
    }
};
typedef CuptiMetricMap metric_map_t;
typedef CuptiMetricMap::iterator metric_map_it;

struct CuptiMetricVector: public std::vector<CuptiMetric*>
{
    CuptiMetricVector() {
        Tau_init_initializeTAU();
    }
    ~CuptiMetricVector() {
        Tau_destructor_trigger();
    }
};
typedef CuptiMetricVector metric_vec_t;



#ifdef DISABLE_CUPTI

extern int Tau_CuptiLayer_get_num_events() {}

extern bool Tau_CuptiLayer_is_initialized() { return false;}

extern void Tau_CuptiLayer_init() {}

extern void Tau_CuptiLayer_finalize() {}

extern void Tau_CuptiLayer_enable() {}

extern void Tau_CuptiLayer_disable() {}

extern void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev) {}

extern int Tau_CuptiLayer_Initialize_callbacks();

extern void Tau_CuptiLayer_Initialize_Map();

counter_map_t Counter_Map;

/* mapping the metric number to the cupti metric number */
counter_id_map_t internal_id_map;
extern counter_id_map_t internal_id_map() {return internal_id_map;}
counter_id_map_t internal_id_map_backwards;
extern counter_id_map_t internal_id_map_backwards() {return internal_id_map_backwards;}
#else

extern bool Tau_CuptiLayer_is_initialized();

extern void Tau_CuptiLayer_enable();

extern void Tau_CuptiLayer_disable();

extern void Tau_CuptiLayer_init();

extern void Tau_CuptiLayer_finalize();

extern void Tau_CuptiLayer_register_all_counters();

extern void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev);

extern void Tau_CuptiLayer_Initialize_callbacks();

extern void Tau_CuptiLayer_Initialize_Map(int off);

extern counter_map_t& Tau_CuptiLayer_Counter_Map();

extern metric_map_t& Tau_CuptiLayer_Metric_Map();

extern counter_id_map_t interal_id_map();
#endif

#endif //__GNUC__

/*
 * C interface between TauMetrics, TauReadMetrics, and CuptiLayer. A C interface
 * is needed because while TauMetrics, TauReadMetrics along with the rest of TAU
 * maybe compiled with any compiler, CuptiLayer must be compiled by g++.
*/

#include <stdint.h>

void Tau_cupti_post_init(void);
void Tau_CuptiLayer_enable_eventgroup(void);
void Tau_CuptiLayer_setup_eventgroup(void);

extern "C" int Tau_CuptiLayer_get_num_events();

extern "C" void Tau_CuptiLayer_set_num_events(int n);

extern "C" void Tau_CuptiLayer_set_event_name(int metric_n, int type);

extern "C" char const * Tau_CuptiLayer_get_event_name(int metric_n);

extern "C" int Tau_CuptiLayer_get_cupti_event_id(int metric_n);

extern "C" int Tau_CuptiLayer_get_metric_event_id(int metric_n);

extern "C" void Tau_CuptiLayer_read_counters(int d, int t, uint64_t *cb);

extern "C" uint64_t Tau_CuptiLayer_read_counter(int metric_n);

extern "C" bool Tau_CuptiLayer_is_cupti_counter(char const * str);

extern "C" void Tau_CuptiLayer_register_string(char const * str, int metric_n);

extern "C" void Tau_cuda_Event_Synchonize();

#endif //CUPTI_LAYER_H
