#include "TauGpuAdapterOpenCLExp.h"
#include <stdlib.h>

void __attribute__ ((constructor)) Tau_opencl_init(void);
void __attribute__ ((destructor)) Tau_opencl_exit(void);

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
		char *r;
		sprintf(r, "%d", id);
		return r;
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

void Tau_opencl_enter_memcpy_event(int id, bool MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_enter_memcpy_event(evId, gId, MemcpyType);
	else
		Tau_gpu_enter_memcpy_event(evId, gId, MemcpyType);
}

void Tau_opencl_exit_memcpy_event(int id, bool MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_exit_memcpy_event(evId, gId, MemcpyType);
	else
		Tau_gpu_exit_memcpy_event(evId, gId, MemcpyType);
}

void Tau_opencl_register_gpu_event(const char *name, int id, double start,
double stop)
{
	openCLEventId *evId = new openCLEventId(id);
	
	Tau_gpu_register_gpu_event(name, evId, start, stop);
}

void Tau_opencl_register_memcpy_event(int id, double start, double stop, int
transferSize, bool MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLGpuId *gId = new openCLGpuId(0);
		Tau_gpu_register_memcpy_event(evId, gId, start, stop, transferSize, MemcpyType);

}

void Tau_opencl_callback_memcpy(cl_event event, cl_int command_stat, void
*memcpy_type)
{
	printf("in memcpy callback!\n");
	/*cl_ulong startTime, endTime;
	cl_int err;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
															 sizeof(cl_ulong), &startTime, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get start time for Memcpy event.\n");
	  exit(1);	
	}
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
															 sizeof(cl_ulong), &endTime, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get end time for Memcpy event.\n");
	  exit(1);	
	}
	if (*(bool*)memcpy_type == MemcpyDtoH)
	{
		Tau_opencl_register_memcpy_event(0, (double) startTime,
		(double) endTime, 0, MemcpyDtoH);
	}
	else if (*(bool*)memcpy_type == MemcpyHtoD)
	{
		Tau_opencl_register_memcpy_event(1, (double) startTime,
		(double) endTime, 0, MemcpyHtoD);
	}*/
}


