#include "TauGpu.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#define TAU_MAX_FUNCTIONNAME 200


class cudaGpuId : public gpuId {
	int device;
	cudaStream_t stream;
public:
	cudaGpuId(const int d, cudaStream_t s)
	{
		device = d;
		//dr_stream = (CUstream) s;
		//rt_stream = (cudaStream_t) s;
		stream = s;
	}
	
	cudaGpuId *getCopy();
	char* printId();
	//x_uint64 id_p1() { return device; }
	//x_uint64 id_p2() { return  stream; }
	bool operator<(const cudaGpuId& other) const;
	bool equals(const gpuId *other) const;
	CUstream get_dr_stream();
	cudaStream_t get_rt_stream();
};
class cudaEventId : public eventId
{
	int id;
public:
	cudaEventId(const int a);
	
	bool operator<(const cudaEventId& A) const;
};

void Tau_cuda_init();

void Tau_cuda_exit();

void Tau_cuda_enter_memcpy_event(const char *name, int id, int size, int MemcpyType);

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType);

void Tau_cuda_register_gpu_event(const char *name, int id, double start,
double stop);

void Tau_cuda_register_memcpy_event(int id, double start, double stop, int
transferSize, int MemcpyType);

void Tau_cuda_enqueue_kernel_enter_event(const char *name, 
cudaStream_t s);

void Tau_cuda_enqueue_kernel_exit_event(const char *name, 
cudaStream_t s);

void Tau_cuda_register_sync_event();
