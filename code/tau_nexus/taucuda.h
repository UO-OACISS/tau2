
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
//#include<pthread.h>
#include<list>
#include<map>
#include<string>
#include<fstream>
#include <TAU_tf.h>
#include "TAU_tf_headers.h"

using namespace std;

enum event_type{DATA,DATA2D,DATAFD,KERNEL,ALL,OTHERS};
typedef unsigned long long TAU64;
typedef unsigned int TAU32;
typedef unsigned short TAU16;
typedef unsigned char TAU8;

#define TAUCUDA_SUCCESS 0
#define TAUCUDA_INIT_FAILED -1

//extern TAUCUDA_API int ntaucuda;
//extern "C"{
 int tau_cuda_init(void);
 void tau_cuda_exit(void);
//};


class EventManager;

struct ClockTable{
	double ref_gpu_start_time;
	double ref_gpu_end_time;
	double gpu_start_time;
	double gpu_end_time;
	double tau_start_time;
	double tau_end_time;
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
	typedef int (* PTHREAD_CREATE_PTR)(pthread_t *restrict, const pthread_attr_t
	*_restrict, void *(*start_routine)(void*), void *restrict1);
	typedef int (*PTHREAD_JOIN_PTR)(pthread_t, void **); 
};
 
