#include "TauGpu.h"
#include <stdio.h>
#include <CL/cl.h>
#include <string.h>

#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif

#define TAU_MAX_FUNCTIONNAME 200

using namespace std;

void Tau_opencl_init();

void Tau_opencl_exit();

void Tau_opencl_enter_memcpy_event(const char *name, int id, int size, int MemcpyType);

void Tau_opencl_exit_memcpy_event(const char *name, int id, int MemcpyType);

void Tau_opencl_register_gpu_event(const char *name, int id, double start,
double stop);

void Tau_opencl_register_memcpy_event(int id, double start, double stop, int
transferSize, int MemcpyType);


//Memcpy event callback

void CL_CALLBACK Tau_opencl_memcpy_callback(cl_event event, cl_int command_stat, void
*memcpy_type);

void CL_CALLBACK Tau_opencl_kernel_callback(cl_event event, cl_int command_stat, void
*kernel_type);

typedef struct memcpy_callback_data_t 
{
	char name[TAU_MAX_FUNCTIONNAME];
	FunctionInfo* callingSite;
	int memcpy_type;
} memcpy_callback_data; 
int memcpy_data_size = sizeof(memcpy_callback_data);

typedef struct kernel_callback_data_t 
{
	char name[TAU_MAX_FUNCTIONNAME];
	FunctionInfo* callingSite;
} kernel_callback_data;
int kernel_data_size = sizeof(kernel_callback_data);


double Tau_opencl_sync_clocks(cl_command_queue commandQueue, cl_context context);

bool Tau_opencl_is_callbacks_supported();

//Enqueue events

enum EnqueueEvents { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER };


