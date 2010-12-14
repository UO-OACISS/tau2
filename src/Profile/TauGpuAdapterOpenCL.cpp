//if callbacks are supported make sure do locking in the callbacks.
#ifdef TAU_ENABLE_CL_CALLBACK
#define TAU_OPENCL_LOCKING
#endif
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "TauGpuAdapterOpenCL.h"
#ifdef TAU_OPENCL_LOCKING
#include <pthread.h>
#else
#pragma message ("Warning: Compiling without pthread support, callbacks may give \
overlapping timers errors.")
#endif
void __attribute__ ((constructor)) Tau_opencl_init(void);
//void __attribute__ ((destructor)) Tau_opencl_exit(void);

class openCLGpuId : public gpuId {

	int id;

public:
/*	cudaGpuId(const NvU64 cId, const NvU32 dId) :
		contextId(cId), deviceId(dId) {} */
	
	openCLGpuId(const int i) {
		id = i;
	}
	openCLGpuId *getCopy() { 
			openCLGpuId *c = new openCLGpuId(*this);
			return c;
	}

	bool equals(const gpuId *other) const
	{
		return id  == ((openCLGpuId *)other)->id;
	}
	
  char* printId();
	x_uint64 id_p1() { return id; }
	x_uint64 id_p2() { return 0; }
};

char* openCLGpuId::printId() 
{
		/*char *r;
		sprintf(r, "%d", id);
		return r;*/
		return "";
}

/* CUDA Event are uniquely identified as the pair of two other ids:
 * context and call (API).
 */
class openCLEventId : public eventId
{
	int id;
	public:
	openCLEventId(const int a) :
		id(a) {}
	
	// for use in STL Maps	
	bool operator<(const openCLEventId& A) const
	{ 
			return id<A.id; 
	}
};

// HACK need to not profile clGetEventProfilingInfo inside a callback.
extern cl_int clGetEventProfilingInfo_noinst(cl_event a1, cl_profiling_info a2, size_t
a3, void * a4, size_t * a5);

extern cl_mem clCreateBuffer_noinst(cl_context a1, cl_mem_flags a2, size_t a3, void *
a4, cl_int * a5);

extern cl_int clEnqueueWriteBuffer_noinst(cl_command_queue a1, cl_mem a2, cl_bool a3,
size_t a4, size_t a5, const void * a6, cl_uint a7, const cl_event * a8, cl_event
* a9);

extern cl_int clWaitForEvents_noinst(cl_uint a1, const cl_event * a2);


//Lock for the callbacks
pthread_mutex_t callback_lock;

int init_callback() 
{
#ifdef TAU_OPENCL_LOCKING
	printf("initalize pthread locking.\n");
	pthread_mutexattr_t lock_attr;
	pthread_mutexattr_init(&lock_attr);
	pthread_mutex_init(&callback_lock, &lock_attr);
	return 1;
#endif
}

void lock_callback()
{
#ifdef TAU_OPENCL_LOCKING
  static int initflag = init_callback();
	pthread_mutex_lock(&callback_lock);
#endif
}

void release_callback()
{
#ifdef TAU_OPENCL_LOCKING
	pthread_mutex_unlock(&callback_lock);
#endif
}

//The time in microseconds that the GPU is ahead of the CPU clock.
double sync_offset = 0;

double Tau_opencl_sync_clocks(cl_command_queue commandQueue, cl_context context)
{
	int d = 0;
	void *data = &d;
	cl_mem buffer;
	cl_int err;
	buffer = clCreateBuffer_noinst(context, CL_MEM_READ_WRITE |
	CL_MEM_ALLOC_HOST_PTR, sizeof(void *), NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Cannot Create Sync Buffer.\n");
	  exit(1);	
	}

	double cpu_timestamp;	
  struct timeval tp;
	cl_ulong gpu_timestamp;

	cl_event sync_event;
	err = clEnqueueWriteBuffer_noinst(commandQueue, buffer, CL_TRUE, 0, sizeof(void*), data,  0, NULL, &sync_event);
	if (err != CL_SUCCESS)
	{
		printf("Cannot Enqueue Sync Kernel: %d.\n", err);
	  exit(1);	
	}

	//clWaitForEvents_noinst(1, &sync_event);
	//get CPU timestamp.
  gettimeofday(&tp, 0);
  cpu_timestamp = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
	//get GPU timestamp for finish.
  err = clGetEventProfilingInfo_noinst(sync_event, CL_PROFILING_COMMAND_END,
															 sizeof(cl_ulong), &gpu_timestamp, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get end time for Sync event.\n");
	  exit(1);	
	}

	//printf("SYNC: CPU= %f GPU= %f.\n", cpu_timestamp, ((double)gpu_timestamp/1e3)); 
	sync_offset = (((double) gpu_timestamp)/1e3) - cpu_timestamp;

	return sync_offset;
}


void Tau_opencl_init()
{
	//printf("in Tau_opencl_init().\n");
	Tau_gpu_init();
}

void Tau_opencl_exit()
{
	//printf("in Tau_opencl_exit().\n");
	Tau_gpu_exit();
}

void Tau_opencl_enter_memcpy_event(const char *name, int id, int size, int MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_enter_memcpy_event(name, evId, gId, size, MemcpyType);
	else
		Tau_gpu_enter_memcpy_event(name, evId, gId, size, MemcpyType);
}

void Tau_opencl_exit_memcpy_event(const char *name, int id, int MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_exit_memcpy_event(name, evId, gId, MemcpyType);
	else
		Tau_gpu_exit_memcpy_event(name, evId, gId, MemcpyType);
}

void Tau_opencl_register_gpu_event(const char *name, int id, double start,
double stop)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	lock_callback();
	//printf("locked for: %s.\n", name);
	Tau_gpu_register_gpu_event(name, evId, gId, start/1e3 - sync_offset, stop/1e3 - sync_offset);
	//printf("released for: %s.\n", name);
	release_callback();
}

void Tau_opencl_register_memcpy_event(const char *name, int id, double start, double stop, int
transferSize, int MemcpyType)
{
	//printf("in Tau_open.\n");
	//printf("Memcpy type is %d.\n", MemcpyType);
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	lock_callback();
	//printf("locked for: %s.\n", name);
	Tau_gpu_register_memcpy_event(name, evId, gId, start/1e3 - sync_offset, stop/1e3 - sync_offset, transferSize, MemcpyType);
	//printf("released for: %s.\n", name);
	release_callback();

}

void CL_CALLBACK Tau_opencl_memcpy_callback(cl_event event, cl_int command_stat, void
*data)
{
	memcpy_callback_data *memcpy_data = (memcpy_callback_data*) malloc(memcpy_data_size);
	memcpy(memcpy_data, data, memcpy_data_size);
	//printf("in memcpy callback!\n");
	//printf("in TauGpuAdapt, name: %s", memcpy_data->name);
	cl_ulong startTime, endTime;
	cl_int err;
  err = clGetEventProfilingInfo_noinst(event, CL_PROFILING_COMMAND_START,
															 sizeof(cl_ulong), &startTime, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get start time for Memcpy event.\n");
	  exit(1);	
	}
  err = clGetEventProfilingInfo_noinst(event, CL_PROFILING_COMMAND_END,
															 sizeof(cl_ulong), &endTime, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get end time for Memcpy event.\n");
	  exit(1);	
	}
	//printf("DtoH calling Tau_open.\n");
	Tau_opencl_register_memcpy_event(memcpy_data->name, 0, (double) startTime,
	(double) endTime, TAU_GPU_UNKNOW_TRANSFER_SIZE, memcpy_data->memcpy_type);
	
	free(data);
}

void CL_CALLBACK Tau_opencl_kernel_callback(cl_event event, cl_int command_stat, void
*data)
{
	kernel_callback_data *kernel_data = (kernel_callback_data*) malloc(kernel_data_size);
	//printf("memcpy size  %d.\n", kernel_data_size);
	memcpy(kernel_data, data, kernel_data_size);
	//printf("in kernel callback!\n");
	cl_ulong startTime, endTime;
	cl_int err;
  err = clGetEventProfilingInfo_noinst(event, CL_PROFILING_COMMAND_START,
															 sizeof(cl_ulong), &startTime, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get start time for Kernel event.\n");
	  exit(1);	
	}
  err = clGetEventProfilingInfo_noinst(event, CL_PROFILING_COMMAND_END,
															 sizeof(cl_ulong), &endTime, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get end time for Kernel event.\n");
	  exit(1);	
	}
		//printf("OpenCL.cpp: start timestamp: %.7f stop time %.7f.", (double) startTime, (double)endTime);
	  //printf("in TauGpuAdapt name: %s.\n", kernel_data->name);
		Tau_opencl_register_gpu_event(kernel_data->name, 0, (double) startTime,
		(double) endTime);
	free(data);
}

