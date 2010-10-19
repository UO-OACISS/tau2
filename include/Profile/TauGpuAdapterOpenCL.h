#include "TauGpu.h"
#include <stdio.h>
#include <CL/opencl.h>

void Tau_opencl_init();

void Tau_opencl_exit();

void Tau_opencl_enter_memcpy_event(int id, bool MemcpyType);

void Tau_opencl_exit_memcpy_event(int id, bool MemcpyType);

void Tau_opencl_register_gpu_event(const char *name, int id, double start,
double stop);

void Tau_opencl_register_memcpy_event(int id, double start, double stop, int
transferSize, bool MemcpyType);


//Memcpy event callback

void CL_CALLBACK Tau_opencl_memcpy_callback(cl_event event, cl_int command_stat, void
*memcpy_type);

void CL_CALLBACK Tau_opencl_kernel_callback(cl_event event, cl_int command_stat, void
*kernel_type);

//Enqueue events

enum EnqueueEvents { ENQUEUE_READ_BUFFER, ENQUEUE_WRITE_BUFFER };


