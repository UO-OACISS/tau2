#include "TauGpu.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#define TAU_MAX_FUNCTIONNAME 200


class cudaGpuId : public gpuId {
public:
	cudaGpuId() {}
	
	virtual cudaGpuId *getCopy() = 0;
	virtual char* printId() = 0;
	virtual x_uint64 id_p1() = 0;
	virtual x_uint64 id_p2() = 0;
	//virtual bool operator<(const cudaGpuId& other) const = 0;
	virtual bool equals(const gpuId *other) const = 0;
	virtual cudaStream_t getStream() = 0;
	virtual int getDevice() = 0;
	virtual CUcontext getContext() = 0;
};

class cudaRuntimeGpuId : public cudaGpuId {
public:
	int device;
	cudaStream_t stream;
	cudaRuntimeGpuId()
	{
		device = 0;
		stream = 0;
	}
	cudaRuntimeGpuId(const cudaRuntimeGpuId& cpy)
	{
		device = cpy.device;
		stream = cpy.stream;
	}
	cudaRuntimeGpuId(const int d, cudaStream_t s)
	{
		device = d;
		stream = s;
	}
	
	cudaRuntimeGpuId *getCopy();
	char* printId();
	x_uint64 id_p1(); 
	x_uint64 id_p2();
	//bool operator<(const cudaGpuId& other) const;
	bool equals(const gpuId *other) const;
	cudaStream_t getStream();
	int getDevice();
	CUcontext getContext();
};
class cudaDriverGpuId : public cudaGpuId {
public:
	int device;
	CUcontext context;
	cudaStream_t stream;
	cudaDriverGpuId()
	{
		device = 0;
		stream = 0;
		context = 0;
	}
	cudaDriverGpuId(const cudaDriverGpuId& cpy)
	{
		device = cpy.device;
		stream = cpy.stream;
		context = cpy.context;
	}
	cudaDriverGpuId(const int d, CUcontext c, cudaStream_t s)
	{
		device = d;
		context = 0;
		stream = s;
	}
	
	cudaDriverGpuId *getCopy();
	char* printId();
	x_uint64 id_p1(); 
	x_uint64 id_p2();
	//bool operator<(const cudaGpuId& other) const;
	bool equals(const gpuId *other) const;
	cudaStream_t getStream();
	int getDevice();
	CUcontext getContext();
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

void Tau_cuda_register_gpu_event(const char *name, cudaGpuId* id, double start,
double stop);

void Tau_cuda_register_memcpy_event(int id, double start, double stop, int
transferSize, int MemcpyType);

void Tau_cuda_enqueue_kernel_enter_event(const char *name, 
cudaGpuId* id);

void Tau_cuda_enqueue_kernel_exit_event(const char *name, 
cudaGpuId* id);

void Tau_cuda_register_sync_event();
