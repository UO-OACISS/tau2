#include "TauGpu.h"
#include <cuda_runtime_api.h>

#define TAU_MAX_FUNCTIONNAME 200


void Tau_cuda_init();

void Tau_cuda_exit();

void Tau_cuda_enter_memcpy_event(const char *name, int id, int size, int MemcpyType);

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType);

void Tau_cuda_register_gpu_event(const char *name, int id, double start,
double stop);

void Tau_cuda_register_memcpy_event(int id, double start, double stop, int
transferSize, int MemcpyType);

void Tau_cuda_enqueue_kernel_enter_event(const char *name, int id);

void Tau_cuda_enqueue_kernel_exit_event(const char *name, int id);

void Tau_cuda_register_sync_event();
