#ifdef __GNUC__
#include "cupti_version.h"
#include "cupti_events.h"
#include "cupti_metrics.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

// Putting "using namespace" statements in header files can create ambiguity
// between user-defined symbols and std symbols, creating unparsable code
// or even changing the behavior of user codes.  This is also widely considered
// to be bad practice.  Here's a code PDT can't parse because of this line:
//   EX: #include <complex>
//   EX: typedef double real;
//
//using namespace std;

/* Specific errors from CUDA lib */
#define CHECK_CU_ERROR(err, cufunc) \
if (err != CUDA_SUCCESS) \
{ \
printf ("[%s:%d] Error %d for CUDA Driver API function '%s'. cuptiQuery failed\n", __FILE__, __LINE__, err, cufunc); \
}

/* Specific errors from CuPTI lib */
#define CHECK_CUPTI_ERROR(err, cuptifunc) \
if (err != CUPTI_SUCCESS) \
{ \
printf ("[%s:%d] Error %d for CUPTI API function '%s'. cuptiQuery failed\n", __FILE__, __LINE__, err, cuptifunc); \
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

class CuptiCounterEvent
{

public:
	CUdevice device;
	CUpti_EventDomainID domain;
	CUpti_EventID event;

	std::string device_name;
	std::string domain_name;
	std::string event_name;
	std::string event_description;
	std::string tag; // string presented to the user.

	CuptiCounterEvent(int device_n, int domain_n, int event_n);

	void create_tag();

	static void printHeader();
	void print();
		
};

typedef std::map<std::string, CuptiCounterEvent*> counter_map_t;
typedef std::vector<CuptiCounterEvent*> counter_vec_t;
typedef std::map<std::string, CuptiCounterEvent*>::iterator counter_map_it;
typedef std::map<int, int> counter_id_map_t;

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

extern int Tau_CuptiLayer_Initialize_callbacks();

extern void Tau_CuptiLayer_Initialize_Map();

extern counter_map_t& Tau_CuptiLayer_Counter_Map();

extern counter_id_map_t interal_id_map();
#endif
#endif //__GNUC__

/*
 * C interface between TauMetrics, TauReadMetrics, and CuptiLayer. A C interface
 * is needed because while TauMetrics, TauReadMetrics along with the rest of TAU
 * maybe compiled with any compiler, CuptiLayer must be compiled by g++.
*/

#include <stdint.h>



extern "C" int Tau_CuptiLayer_get_num_events();

extern "C" void Tau_CuptiLayer_set_event_name(int metric_n, int type);

extern "C" const char* Tau_CuptiLayer_get_event_name(int metric_n);

extern "C" int Tau_CuptiLayer_get_cupti_event_id(int metric_n);

extern "C" int Tau_CuptiLayer_get_metric_event_id(int metric_n);

extern "C" void Tau_CuptiLayer_read_counters(int d, uint64_t *cb);

extern "C" uint64_t Tau_CuptiLayer_read_counter(int metric_n);

extern "C" bool Tau_CuptiLayer_is_cupti_counter(char *str);

extern "C" void Tau_CuptiLayer_register_string(char *str, int metric_n);

//extern counter_map_t& Tau_CuptiLayer_Counter_Map();

//counter_map_it Tau_CuptiLayer_counters_iterator();

