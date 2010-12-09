#include "TauGpuAdapterCUDA.h"
#include <stdio.h>
#include <queue>
#include <sys/time.h>
using namespace std;

//CPU timestamp at the first cuEvent.
int sync_offset = 0;

class cudaGpuId : public gpuId {

	int id;

public:
/*	cudaGpuId(const NvU64 cId, const NvU32 dId) :
		contextId(cId), deviceId(dId) {} */
	
	cudaGpuId(const int i) {
		id = i;
	}
	
  char* printId();
	x_uint64 id_p1() { return id; }
	x_uint64 id_p2() { return 0; }
};

char* cudaGpuId::printId() 
{
		/*char *r;
		sprintf(r, "%d", id);
		return r;*/
		return "";
}

/* CUDA Event are uniquely identified as the pair of two other ids:
 * context and call (API).
 */
class cudaEventId : public eventId
{
	int id;
	public:
	cudaEventId(const int a) :
		id(a) {}
	
	// for use in STL Maps	
	bool operator<(const cudaEventId& A) const
	{ 
			return id<A.id; 
	}
};

class KernelEvent
{

	public: 
	int id;
	cudaStream_t stream;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
	int blocksPerGrid;
	int threadsPerBlock;
};

static cudaEvent_t lastEvent;
static float lastEventTime = 0;

static queue<KernelEvent> KernelBuffer;



void Tau_cuda_init()
{
	//printf("in Tau_cuda_init().\n");
	static bool init = false;
	if (!init)
	{
		cudaEvent_t initEvent;
		cudaStream_t stream;
		cudaError err = cudaStreamCreate(&stream);
		
  	struct timeval tp;

		if (err != cudaSuccess)
		{
			printf("Error creating stream, error #: %d.\n", err);
			//exit(1);
		}
		err = cudaEventCreate(&initEvent); 
		if (err != cudaSuccess)
		{
			printf("Error creating Event, error #: %d.\n", err);
			//exit(1);
		}
		err = cudaEventRecord(initEvent, stream);
		if (err != cudaSuccess)
		{
			printf("Error recording Event, error #: %d.\n", err);
			//exit(1);
		}
		err = cudaEventSynchronize(initEvent);
  	gettimeofday(&tp, 0);
		
		if (err != cudaSuccess)
		{
			printf("Error syncing Event, error #: %d.\n", err);
			//exit(1);
		}
  	sync_offset = ((double)tp.tv_sec * 1e6 + tp.tv_usec);

		lastEvent = initEvent;
		init = true;
		Tau_gpu_init();
	}
}

void Tau_cuda_exit()
{
	//printf("in Tau_cuda_exit().\n");
	Tau_gpu_exit();
}

void Tau_cuda_enter_memcpy_event(const char *name, int id, int size, int MemcpyType)
{
	Tau_gpu_enter_memcpy_event(name, &cudaEventId(id), &cudaGpuId(0), size, MemcpyType);
}

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType)
{
	Tau_gpu_exit_memcpy_event(name, &cudaEventId(id), &cudaGpuId(0), MemcpyType);
}

void Tau_cuda_register_gpu_event(const char *name, int id, double start,
double stop)
{
	Tau_gpu_register_gpu_event(name, &cudaEventId(id), start/1e3 + sync_offset, stop/1e3 + sync_offset);
}

void Tau_cuda_register_memcpy_event(const char *name, int id, double start, double stop, int
transferSize, int MemcpyType)
{
	Tau_gpu_register_memcpy_event(name, &cudaEventId(id), &cudaGpuId(id), start/1e3 + sync_offset, stop/1e3 + sync_offset, transferSize, MemcpyType);
}


KernelEvent curKernel;

void Tau_cuda_enqueue_kernel_enter_event(const char *name, int id)
{

	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	curKernel.startEvent = startEvent;
	curKernel.stopEvent = stopEvent;
	curKernel.id = id;

	cudaError err = cudaStreamCreate(&curKernel.stream);
	
	if (err != cudaSuccess)
	{
		printf("Error creating stream, error #: %d.\n", err);
		//exit(1);
	}

	KernelBuffer.push(curKernel);

  err = cudaEventRecord(curKernel.startEvent, curKernel.stream);

	if (err != cudaSuccess)
	{
		printf("Error recording kernel event, error #: %d.\n", err);
		//exit(1);
	}
 

}

void Tau_cuda_enqueue_kernel_exit_event(const char* name, int id)
{


  cudaError err = cudaEventRecord(curKernel.stopEvent, curKernel.stream);
	if (err != cudaSuccess)
	{
		printf("Error recording kernel event, error #: %d.\n", err);
		//exit(1);
	}
}

void Tau_cuda_register_sync_event()
{
	//cudaError err = cudaEventQuery(KernelBuffer.front().stopEvent);
	//printf("buffer front is: %d\n", err);

	float start_sec, stop_sec;

	while (!KernelBuffer.empty() && cudaEventQuery(KernelBuffer.front().stopEvent) == cudaSuccess)
	{
		KernelEvent kernel = KernelBuffer.front();
		printf("kernel buffer size = %d.\n", KernelBuffer.size());

		cudaEventElapsedTime(&start_sec, lastEvent, kernel.startEvent);
		printf("kernel event [start] = %f.\n", start_sec + lastEventTime);

		cudaEventElapsedTime(&stop_sec, lastEvent, kernel.stopEvent);
		printf("kernel event [stop] = %f.\n", stop_sec + lastEventTime );

		Tau_cuda_register_gpu_event("Kernel", kernel.id, (double) start_sec,
		(double) stop_sec);

		lastEvent = kernel.stopEvent;
		lastEventTime += stop_sec;

		KernelBuffer.pop();

	}

}
