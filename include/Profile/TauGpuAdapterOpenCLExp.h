#include "TauGpu.h"
#include <stdio.h>


void Tau_opencl_init();

void Tau_opencl_exit();

void Tau_opencl_enter_memcpy_event(int id, bool MemcpyType);

void Tau_opencl_exit_memcpy_event(int id, bool MemcpyType);

void Tau_opencl_register_gpu_event(const char *name, int id, double start,
double stop);

void Tau_opencl_register_memcpy_event(int id, double start, double stop, int
transferSize, bool MemcpyType);

