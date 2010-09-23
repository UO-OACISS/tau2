#include "TauGpuAdaperOpenCLExp.h"

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

	Tau_gpu_init();
}

void Tau_opencl_exit()
{

	Tau_gpu_exit();
}

void Tau_opencl_enter_memcpy_event(int id, bool MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLEventGpuId *gId = new openCLEventGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_enter_memcpy_event("Memcpy Host to Device (CPU)", evId, gId, MemcpyType);
	else
		Tau_gpu_enter_memcpy_event("Memcpy Device to Host (CPU)", evId, gId, MemcpyType);
}

void Tau_opencl_exit_memcpy_event(int id, bool MemcpyType)
{
	openCLEventId *evId = new openCLEventId(id);
	openCLEventGpuId *gId = new openCLEventGpuId(0);
	if (MemcpyType == MemcpyHtoD) 
		Tau_gpu_exit_memcpy_event("Memcpy Host to Device (CPU)", evId, gId, MemcpyType);
	else
		Tau_gpu_exit_memcpy_event("Memcpy Device to Host (CPU)", evId, gId, MemcpyType);
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
	openCLEventGpuId *gId = new openCLEventGpuId(0);
		Tau_gpu_register_memcpy_event(evId, gId, start, stop, tranferSize, MemcpyType);

}
