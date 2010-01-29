
/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory				           **
****************************************************************************/
/***************************************************************************
**      File            : taucuda.h                                	  **
**      Description     : TAU trace format reader library header files    **
**      Author          : Shangkar Mayanglambam                           **
**      Contact         : smeitei@cs.uoregon.edu                          **
***************************************************************************/



#pragma once
#define cuToolsApi_INITGUID
#include <cuda_toolsapi.h>
#include <cuda_toolsapi_tau.h>
#include<iostream>
#include<list>
#include<pthread.h>
#include "gpuevents.h"
using namespace std;

#define TAUCUDA_SUCCESS 0
#define TAUCUDA_INIT_FAILED -1

//extern TAUCUDA_API int ntaucuda;
//extern "C"{
 int tau_cuda_init(void);
 void tau_cuda_exit(void);
//};


class EventManager;

struct ClockTable{
	TAU64 ref_gpu_start_time;
	TAU64 ref_gpu_end_time;
	TAU64 gpu_start_time;
	TAU64 gpu_end_time;
	TAU64 tau_start_time;
	TAU64 tau_end_time;
};

struct MemCpy2D{
	//unsigned int f_id;
	void* ptr1;
	void* ptr2; 
	unsigned int count;	
};

#define MAX_DEVICES 32
  
struct ToolsAPI {
        void * handle;
        cuToolsApi_Core* coreTable;
        cuToolsApi_Device* deviceTable;
        cuToolsApi_Context* contextTable;
	int device_count;
	ClockTable device_clocks[MAX_DEVICES];	
	//EventManager *manager;
	list<EventManager *> managers;
	~ToolsAPI(){
		tau_cuda_exit();
	};
};

void __attribute__ ((constructor)) onload(void);
void __attribute__ ((destructor)) onunload(void);

struct wrap_routine_arg{
	void *(*start_routine)(void*);
	void * arg;
};

extern "C" {
	typedef int (* PTHREAD_CREATE_PTR)(pthread_t *restrict, const pthread_attr_t *restrict, void *(*start_routine)(void*), void *restrict1);
	typedef int (*PTHREAD_JOIN_PTR)(pthread_t, void **); 
};
 
