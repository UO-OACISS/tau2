/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://tau.uoregon.edu												     **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory				                                 **
****************************************************************************/
/***************************************************************************
**      File            : nVidia_exp_adapter.h                         	  **
**      Description     : Adapter nVidia's experimenat 190.38 driver to   **
**                      : allow TAU to profile events                     **
**      Author          : Shangkar Mayanglambam                           **
**                      : Scott Biersdorff                                **
**      Contact         : scottb@cs.uoregon.edu                           **
***************************************************************************/

#pragma once
#define cuToolsApi_INITGUID

#include <cuda_toolsapi.h>
#include <cuda_toolsapi_tau.h>

#include "taucuda_interface.h"

#include<iostream>
#include<list>
#include<map>
#include<string>
#include<fstream>
#include <TAU_tf.h>
#include "TAU_tf_headers.h"
using namespace std;


/* ============ Driver version specific stuff below here =============== */


enum eventType{DATA,DATA2D,DATAFD,KERNEL,ALL,OTHERS};
typedef unsigned long long TAU64;
typedef unsigned int TAU32;
typedef unsigned short TAU16;
typedef unsigned char TAU8;

#define TAUCUDA_SUCCESS 0
#define TAUCUDA_INIT_FAILED -1

class EventManager;


/* For cpu/gpu synchorization */
struct ClockTable{
	double ref_gpu_start_time;
	double ref_gpu_end_time;
	double gpu_start_time;
	double gpu_end_time;
	double tau_start_time;
	double tau_end_time;
};

struct MemCpy2D{
	void* ptr1;
	void* ptr2; 
	unsigned int count;	
};

double cpu_time()
{
	//get time from the CPU clock
	struct timeval tp;
	gettimeofday(&tp, 0);
	//printf("CPU time: %f \n", ((double)tp.tv_sec * 1e6 + tp.tv_usec));
	//printf("subtraction: %f \n", cpu_start_time);
	//printf("CPU time (2): %f \n", ((double)tp.tv_sec * 1e6 + tp.tv_usec) - cpu_start_time);
	return ((double)tp.tv_sec * 1e6 + tp.tv_usec);
}

#define MAX_DEVICES 32
 
void shutdown_tool_api(void);

/* This is the guts of the callback layer. This struct is created at
 * initalization and when destory it will shutdown the callback layer. */
struct ToolsAPI {
	void * handle;
	cuToolsApi_Core* coreTable;
	cuToolsApi_Device* deviceTable;
	cuToolsApi_Context* contextTable;
	int device_count;
	ClockTable device_clocks[MAX_DEVICES];	
	//EventManager *manager;
	//list<EventManager *> managers;
	~ToolsAPI(){
		//shutdown the callback layer when this is destoryed
		shutdown_tool_api();
	};
};

void __attribute__ ((constructor)) onload(void);
void __attribute__ ((destructor)) onunload(void);

struct wrap_routine_arg{
	void *(*start_routine)(void*);
	void * arg;
};

extern "C" {
	typedef int (* PTHREAD_CREATE_PTR)(pthread_t *restrict, const pthread_attr_t
	*_restrict, void *(*start_routine)(void*), void *restrict1);
	typedef int (*PTHREAD_JOIN_PTR)(pthread_t, void **); 
};
 
class cudaGpuId : public gpuId {

	NvU64 contextId;
	NvU32 deviceId;

public:
	cudaGpuId(const NvU64 cId, const NvU32 dId) :
		contextId(cId), deviceId(dId) {}
	
	char* printId()
	{
		char *r;
		sprintf(r, "%f:%f", contextId, deviceId);
		return r;
	}
	double id_p1() { return (double) contextId; }
	double id_p2() { return (double) deviceId; }
};

class cuEventId : public eventId
{
	NvU64 contextId;
	NvU64 callId;

	public:
	cuEventId(const NvU64 a, const NvU64 b) :
		contextId(a), callId(b) {}
	
	bool operator<(const cuEventId& A) const
	{ 
		if (contextId == A.contextId)
		{
			return callId<A.callId; 
		}
		else
			return contextId<A.contextId;
	}
};

