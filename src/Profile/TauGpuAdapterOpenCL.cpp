#include "TauGpuAdapterOpenCL.h"
#include <stdlib.h>
#include <string.h>
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




void Tau_opencl_init()
{
	printf("in Tau_opencl_init().\n");
	Tau_gpu_init();
}

void Tau_opencl_exit()
{
	printf("in Tau_opencl_exit().\n");
	Tau_gpu_exit();
}

void Tau_opencl_enter_memcpy_event(const char *name, int id, int size, bool MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_enter_memcpy_event(name, evId, gId, size, MemcpyType);
	else
		Tau_gpu_enter_memcpy_event(name, evId, gId, size, MemcpyType);
}

void Tau_opencl_exit_memcpy_event(const char *name, int id, bool MemcpyType)
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
	
	Tau_gpu_register_gpu_event(name, evId, start/1e3, stop/1e3);
}

void Tau_opencl_register_memcpy_event(const char *name, int id, double start, double stop, int
transferSize, bool MemcpyType)
{
	//printf("in Tau_open.\n");
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
		Tau_gpu_register_memcpy_event(name, evId, gId, start/1e3, stop/1e3, transferSize, MemcpyType);

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
	(double) endTime, 0 /*TAU_GPU_UNKNOW_TRANSFER_SIZE*/, memcpy_data->memcpy_type);
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
}

