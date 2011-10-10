#ifdef __GNUC__
#include "cupti_events.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
using namespace std;

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
#define TAU_CUPTI_MAX_EVENTS 160
		
//#define DISABLE_CUPTI

class CuptiCounterEvent
{

public:
	CUdevice device;
	CUpti_EventDomainID domain;
	CUpti_EventID event;

	string device_name;
	string domain_name;
	string event_name;
	string event_description;
	string tag; // string presented to the user.

	CuptiCounterEvent(int device_n, int domain_n, int event_n);

	void create_tag();

	static void printHeader();
	void print();
		
};

typedef map<std::string, CuptiCounterEvent*> counter_map_t;
typedef vector<CuptiCounterEvent*> counter_vec_t;
typedef map<std::string, CuptiCounterEvent*>::iterator counter_map_it; 

#ifdef DISABLE_CUPTI

extern int Tau_CuptiLayer_get_num_events() {}

extern bool Tau_CuptiLayer_is_initialized() { return false;}

extern void Tau_CuptiLayer_init() {}

extern void Tau_CuptiLayer_finalize() {}

extern void Tau_CuptiLayer_enable() {}

extern void Tau_CuptiLayer_disable() {}

extern void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev) {}

extern void Tau_CuptiLayer_read_counters(uint64_t * cBuffer) {}

counter_map_t Tau_CuptiLayer_Counter_Map;
extern counter_map_t Tau_CuptiLayer_map() {return Tau_CuptiLayer_Counter_Map;}

#else

extern int Tau_CuptiLayer_get_num_events();

extern bool Tau_CuptiLayer_is_initialized();

extern void Tau_CuptiLayer_enable();

extern void Tau_CuptiLayer_disable();

extern void Tau_CuptiLayer_init();

extern void Tau_CuptiLayer_finalize();

extern void Tau_CuptiLayer_register_counter(CuptiCounterEvent* ev);

extern void Tau_CuptiLayer_read_counters(uint64_t * cBuffer);

extern counter_map_t Tau_CuptiLayer_map();

#endif
#endif //__GNUC__

/*
 * C interface between TauMetrics, TauReadMetrics, and CuptiLayer. A C interface
 * is needed because while TauMetrics, TauReadMetrics along with the rest of TAU
 * maybe compiled with any compiler, CuptiLayer must be compiled by g++.
*/

#include <stdint.h>

extern "C" int Tau_CuptiLayer_get_num_events();

extern "C" void Tau_CuptiLayer_read_counters(void * cBuffer);

extern "C" bool Tau_CuptiLayer_is_cupti_counter(char *str);

extern "C" void Tau_CuptiLayer_register_string(char *str);

//counter_map_it Tau_CuptiLayer_counters_iterator();

