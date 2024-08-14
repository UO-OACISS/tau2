#ifndef TAU_GPU_ADAPTER_CUDA_H
#define TAU_GPU_ADAPTER_CUDA_H

#include "TauGpu.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

#define TAU_MAX_FUNCTIONNAME 200

//CPU timestamp at the first cuEvent.
double sync_offset = 0;

class CudaGpuEvent : public GpuEvent {
public:
	const char *name;
	int taskId;
	FunctionInfo *callingSite;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
	CudaGpuEvent() {}
	/*CudaGpuEvent *getCopy()
	{
		CudaGpuEvent *c = new CudaGpuEvent(this);
		return c;
	}*/
	/*
	CudaGpuEvent(const CudaGpuEvent &other)
	{
		printf("in CudaGpuEvent copy constructor.\n");
		char * o_name = new char[strlen(other.name) + 1];
		strcpy(name, o_name);
		callingSite = new FunctionInfo(*other.callingSite);
		startEvent = other.startEvent;
		stopEvent = other.stopEvent;
	}*/
	CudaGpuEvent(const char* name, FunctionInfo* fi) : 
	name(name), callingSite(fi), taskId(-1) {};
	virtual cudaStream_t getStream() = 0;
/*
	cudaStream_t getStream()
	{
		CudaGpuEvent* cuDevice = static_cast<CudaGpuEvent*>(device); 
		return cuDevice->getStream();
	}
*/
	int enqueue_start_event()
	{
		cudaError_t err;
		err = cudaEventCreate(&startEvent);
		if (err != cudaSuccess)
		{
			printf("Error creating kernel event, error #: %d.\n", err);
			return 1;
		}
		err = cudaEventRecord(startEvent, 0);
		if (err != cudaSuccess)
		{
			printf("Error recording kernel event (0), error #: %d.\n", err);
			return 1;
		}
		err = cudaEventRecord(startEvent, getStream());
		if (err != cudaSuccess)
		{
			printf("Error recording kernel event, error #: %d.\n", err);
			return 1;
		}
		//clear error buffer.
		err == cudaGetLastError();
		return 0;
	}

int enqueue_stop_event()
{
	cudaError_t err;
	err = cudaEventCreate(&stopEvent);
	if (err != cudaSuccess)
	{
		printf("Error creating kernel event, error #: %d.\n", err);
		return 1;
	}
	err = cudaEventRecord(stopEvent, 0);
	if (err != cudaSuccess)
	{
		printf("Error recording kernel event (0), error #: %d.\n", err);
		return 1;
	}
	err = cudaEventRecord(stopEvent, getStream());
	if (err != cudaSuccess)
	{
		printf("Error recording kernel event, error #: %d.\n", err);
		return 1;
	}
	//clear error buffer.
	err == cudaGetLastError();
	return 0;
}
	
	virtual x_uint64 id_p1() const = 0;
	virtual x_uint64 id_p2() const = 0;
	//virtual bool operator<(const CudaGpuEvent& other) const = 0;
	virtual bool less_than(const GpuEvent *other) const = 0;
	virtual double syncOffset() const = 0;
	virtual int getDevice() = 0;
	virtual CUcontext getContext() = 0;

	const char *getName() const { return name; }
	int getTaskId() const { return taskId; }
	FunctionInfo *getCallingSite() const { return callingSite; }
	
	// GPU attributes not implemented for CUDA.
	void getAttributes(GpuEventAttributes *&gA, int &num) const
	{
		num = 0;
		gA = NULL;
	}
	void recordMetadata(int i) const {}

};

class CudaRuntimeGpuEvent : public CudaGpuEvent {
public:
	int device;
	cudaStream_t stream;
	CudaRuntimeGpuEvent()
	{
		device = 0;
		stream = 0;
	}
		
	CudaRuntimeGpuEvent(const CudaRuntimeGpuEvent& cpy) : CudaGpuEvent(cpy)
	{
		device = cpy.device;
		stream = cpy.stream;
	}
	
	CudaRuntimeGpuEvent(const char *n, const int d, cudaStream_t s)
	{
		name = n;
		device = d;
		stream = s;
	}
	double syncOffset() const
	{
		return sync_offset;
	}
	CudaRuntimeGpuEvent *getCopy() const {
		CudaRuntimeGpuEvent *c = new CudaRuntimeGpuEvent(*this);
		return c;
	}
	const char* gpuIdentifier() const
	{
			char *rtn = (char*) malloc(50*sizeof(char));
			snprintf(rtn, 50*sizeof(char),  "[%d:%d]", device, stream);
			return rtn;
	}
	x_uint64 id_p1(void) const { return device; }
	x_uint64 id_p2(void) const { return (x_uint64) stream; }
	cudaStream_t getStream() { return stream; }
	int getDevice() { return device; }
	CUcontext getContext() { return 0; }
	//bool operator<(const CudaGpuEvent& other) const;
	bool less_than(const GpuEvent *o) const 
	{
		CudaRuntimeGpuEvent *other = (CudaRuntimeGpuEvent *) o;
		//cout << "checking" << gpuIdentifier() << " < " << other->gpuIdentifier() << endl;
		if (this->device == other->device)
		{
			return this->stream < other->stream;
		}
		else
		{
			return this->device < other->device;
		}	
	}
};

class CudaDriverGpuEvent : public CudaGpuEvent {
public:
	int device;
	CUcontext context;
	cudaStream_t stream;
	CudaDriverGpuEvent()
	{
		device = 0;
		stream = 0;
		context = 0;
	}
	CudaDriverGpuEvent(const CudaDriverGpuEvent& cpy)
	{
		device = cpy.device;
		stream = cpy.stream;
		context = cpy.context;
	}
	CudaDriverGpuEvent(const int d, CUcontext c, cudaStream_t s)
	{
		device = d;
		context = 0;
		stream = s;
	}
	
	CudaDriverGpuEvent *getCopy() const { 
			CudaDriverGpuEvent *c = new CudaDriverGpuEvent(*this);
			return c;
	}
	bool less_than(const GpuEvent *o) const 
	{
		//cout << "in equals." << endl;
		CudaDriverGpuEvent *other = (CudaDriverGpuEvent *) o;
		if (this->device == other->device)
		{
			if (this->stream == other->stream)
			{
				return this->context < other->context;
			}
			else
			{	
				return this->stream < other->stream;
			}
		}
		else
		{
			return this->device < other->device;
		}
	}
	const char* gpuIdentifier() const
	{
			char *rtn = (char*) malloc(50*sizeof(char));
			snprintf(rtn, 50*sizeof(char),  "%d:%d:%d (Device,Context,Stream)", device, context, stream);
			return rtn;
	}
	x_uint64 id_p1(void) const { return device; }
	x_uint64 id_p2(void) const { return (x_uint64) stream; }
	cudaStream_t getStream() { return stream; }
	int getDevice() { return device; }
	CUcontext getContext() { return context; }
	double syncOffset() const
	{
		return sync_offset;
	}
};

void Tau_cuda_init();

void Tau_cuda_exit();

void Tau_cuda_enter_memcpy_event(const char *name, int id, int size, int MemcpyType);

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType);

void Tau_cuda_register_gpu_event(CudaGpuEvent* id, double start, double stop);

void Tau_cuda_register_memcpy_event(const char *name, CudaGpuEvent* id, double start, double stop, int
				    transferSize, int MemcpyType);

void Tau_cuda_enqueue_kernel_enter_event(CudaGpuEvent *id);

void Tau_cuda_enqueue_kernel_exit_event(CudaGpuEvent *id);

void Tau_cuda_register_sync_event();

#endif //TAU_GPU_ADAPTER_CUDA_H
