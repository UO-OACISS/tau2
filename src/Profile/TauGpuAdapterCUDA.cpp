#include "TauGpuAdapterCUDA.h"
#ifdef CUPTI
#include "CuptiLayer.h"
#endif //CUPTI
#include <stdio.h>
#include <iostream>
#include <queue>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
using namespace std;

#ifdef TAU_BFD
#define HAVE_DECL_BASENAME 1
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include <demangle.h>
#  endif /* HAVE_GNU_DEMANGLE */
#  include <bfd.h>
#endif /* TAU_BFD */


//CPU timestamp at the first cuEvent.
double sync_offset = 0;


cudaRuntimeGpuId *cudaRuntimeGpuId::getCopy() { 
		//printf("in runtime, getCopy.\n");
		//return this;
		cudaRuntimeGpuId *c = new cudaRuntimeGpuId(*this);
		return c;
}
/*
bool cudaRuntimeGpuId::operator<(const cudaGpuId& other) const
{
	if (device == other.device)
	{
		return stream < other.stream;
	}
	else
		return device < other.device;
}*/
bool cudaRuntimeGpuId::equals(const gpuId *o) const 
{
	//cout << "in equals." << endl;
	cudaRuntimeGpuId *other = (cudaRuntimeGpuId *) o;
	return (this->device == other->device && this->stream == other->stream);
}

char* cudaRuntimeGpuId::printId() 
{
		char *rtn = (char*) malloc(50*sizeof(char));
		sprintf(rtn, "[%d:%d]", device, stream);
		return rtn;
}
x_uint64 cudaRuntimeGpuId::id_p1(void) { return device; }
x_uint64 cudaRuntimeGpuId::id_p2(void) { return (x_uint64) stream; }
cudaStream_t cudaRuntimeGpuId::getStream() { return stream; }
int cudaRuntimeGpuId::getDevice() { return device; }
CUcontext cudaRuntimeGpuId::getContext() { return 0; }

cudaDriverGpuId *cudaDriverGpuId::getCopy() { 
		//printf("in driver, getCopy.\n");
		//return this;
		cudaDriverGpuId *c = new cudaDriverGpuId(*this);
		return c;
}
/*
bool cudaDriverGpuId::operator<(const cudaGpuId& other) const
{
	if (device == other.device)
	{
		if (context == other.context)
			return stream < other.stream;
		else 
			return context < other.context;
	}
	else
		return device < other.device;
}*/
bool cudaDriverGpuId::equals(const gpuId *o) const 
{
	//cout << "in equals." << endl;
	cudaDriverGpuId *other = (cudaDriverGpuId *) o;
	return (this->device == other->device && this->stream == other->stream &&
					this->context == other->context); 
}

char* cudaDriverGpuId::printId() 
{
		char *rtn = (char*) malloc(50*sizeof(char));
		sprintf(rtn, "[%d:%d:%d]", device, context, stream);
		return rtn;
}
x_uint64 cudaDriverGpuId::id_p1(void) { return device; }
x_uint64 cudaDriverGpuId::id_p2(void) { return (x_uint64) stream; }
cudaStream_t cudaDriverGpuId::getStream() { return stream; }
int cudaDriverGpuId::getDevice() { return device; }
CUcontext cudaDriverGpuId::getContext() { return context; }
/*
CUstream cudaGpuId::get_dr_stream(void)
{
	if (dr_stream != NULL)
	{
		return dr_stream;
	}
	else
	{
		CUstream st;
		cuStreamCreate(&st, 0);
		return st; 
	}
}
cudaStream_t cudaGpuId::get_rt_stream(void)
{
	if (rt_stream != NULL)
	{	
		printf("working correctly.\n");
		return rt_stream;
	}
	else
	{
		cudaStream_t st;
		cudaStreamCreate(&st);
		return st;
	}
}*/
//cudaEventId::cudaEventId(const int a) :
//		id(a) {}
	
	// for use in STL Maps	
bool cudaEventId::operator<(const cudaEventId& A) const
{ 
		return id<A.id; 
}

class KernelEvent : public eventId
{
	public: 
	//const char *name;
	int blocksPerGrid;
	int threadsPerBlock;
	//cudaGpuId* device;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;

	KernelEvent(const char* name, cudaGpuId* tmp, FunctionInfo* fi) : 
		eventId(name, tmp, fi) {
		//overide device member 
		//device = tmp;
		//device = new cudaGpuId(tmp);
		//delete tmp;
		};

	cudaStream_t getStream()
	{
		cudaGpuId* cuDevice = static_cast<cudaGpuId*>(device); 
		return cuDevice->getStream();
	}

	int enqueue_start_event()
	{
		cudaError_t err;
		cudaEventCreate(&startEvent);
		err = cudaEventRecord(startEvent, 0);
		err = cudaEventRecord(startEvent, getStream());
		if (err != cudaSuccess)
		{
			printf("Error recording kernel event, error #: %d.\n", err);
			return 1;
		}
		return 0;
	}
	int enqueue_stop_event()
	{
		cudaError_t err;
		cudaEventCreate(&stopEvent);
		err = cudaEventRecord(stopEvent, 0);
		err = cudaEventRecord(stopEvent, getStream());
		if (err != cudaSuccess)
		{
			printf("Error recording kernel event, error #: %d.\n", err);
			return 1;
		}
		return 0;
	}
};

static cudaEvent_t lastEvent;
static double lastEventTime = 0;

static queue<KernelEvent> KernelBuffer;



void Tau_cuda_init()
{
	static bool init = false;
	if (!init)
	{
		//printf("in Tau_cuda_init().\n");
		cudaEvent_t initEvent;
		//cudaStream_t stream;
		cudaError err = cudaSuccess; //= cudaStreamCreate(&stream);
		
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
		err = cudaEventRecord(initEvent, 0);
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
  	sync_offset = (double)(tp.tv_sec * 1e6 + tp.tv_usec);
		//printf("sync offset: %lf.\n", sync_offset);

		lastEvent = initEvent;
		lastEventTime = sync_offset / 1e3;  
		//printf("last event time: %lf.\n", lastEventTime);
		init = true;
		Tau_gpu_init();
#ifdef CUPTI
		Tau_CuptiLayer_init();
#endif
	}
}

void Tau_cuda_exit()
{
	//printf("in Tau_cuda_exit().\n");

	//Tau_cuda_register_sync_event();
	Tau_gpu_exit();
}

void Tau_cuda_enter_memcpy_event(const char *name, int id, int size, int MemcpyType)
{
	Tau_gpu_enter_memcpy_event(name, &cudaDriverGpuId(0,0,0), size, MemcpyType);
}

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType)
{
	Tau_gpu_exit_memcpy_event(name, &cudaDriverGpuId(0,0,0), MemcpyType);
}

/*void Tau_cuda_register_gpu_event(KernelEvent k, double start,
double stop)
{
	//printf("sync'ed \t start: %lf.\n \t \t \t stop: %lf.\n", start+sync_offset, stop+sync_offset);
	//FunctionInfo *p = TauInternal_CurrentProfiler(RtsLayer::myNode())->ThisFunction;
	//eventId c = Tau_gpu_create_gpu_event(name, id, p);
	Tau_gpu_register_gpu_event(k, start + sync_offset, stop + sync_offset);
}*/

void Tau_cuda_register_memcpy_event(const char *name, cudaGpuId* id, double start, double stop, int
transferSize, int MemcpyType)
{
	FunctionInfo *p = TauInternal_CurrentProfiler(RtsLayer::myNode())->ThisFunction;
	eventId c = Tau_gpu_create_gpu_event(name, id, p);
	Tau_gpu_register_memcpy_event(c, start/1e3 + sync_offset, stop/1e3 + sync_offset, transferSize, MemcpyType);
}


KernelEvent *curKernel;

void Tau_cuda_enqueue_kernel_enter_event(const char *name, cudaGpuId* id,
FunctionInfo* callingSite)
{
	//printf("recording start for %s.\n", name);

	curKernel = new KernelEvent(name, id, callingSite);
	
	const char *dem_name = 0;

#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
	//printf("demangling name....\n");
	dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE |
	DMGL_TYPES);
#else
	dem_name = name;
#endif /* HAVE_GPU_DEMANGLE */


	//printf("final kernel name is: %s.\n", dem_name);

	curKernel->name = dem_name;
	curKernel->device = id->getCopy();

	curKernel->enqueue_start_event();
 
	//printf("Successfully recorded start.\n");

}

void Tau_cuda_enqueue_kernel_exit_event()
{

	//printf("recording stop for %s.\n", name);

	curKernel->enqueue_stop_event();
	KernelBuffer.push(*curKernel);

 
	//printf("Successfully recorded stop.\n");
}

void Tau_cuda_register_sync_event()
{
	//printf("in sync event, buffer size: %d.\n", KernelBuffer.size());	
	
	if (KernelBuffer.size() > 0 && KernelBuffer.front().stopEvent != NULL)
	{
		//printf("buffer front stop: %d.\n", KernelBuffer.front().stopEvent == NULL);
		cudaError err = cudaEventQuery(KernelBuffer.front().stopEvent);
		//printf("buffer front is: %d\n", err);
	}
	float start_sec, stop_sec;

	while (!KernelBuffer.empty() && cudaEventQuery(KernelBuffer.front().stopEvent) == cudaSuccess)
	{
		KernelEvent kernel = KernelBuffer.front();
		//printf("kernel buffer size = %d.\n", KernelBuffer.size());

		cudaError_t err;
		err = cudaEventElapsedTime(&start_sec, lastEvent, kernel.startEvent);
		//printf("kernel event [start] = %lf.\n", (((double) start_sec) + lastEventTime)*1e3);

		if (err != cudaSuccess)
		{
			printf("Error calculating kernel event start, error #: %d.\n", err);
		}

		err = cudaEventElapsedTime(&stop_sec, lastEvent, kernel.stopEvent);
		//printf("kernel event [stop] = %lf.\n", (((double) stop_sec) + lastEventTime)*1e3 );

		if (err != cudaSuccess)
		{
			printf("Error calculating kernel event stop, error #: %d.\n", err);
		}

		//Create cudaGpuId for stream.
		//cudaGpuId *id = new cudaGpuId(kernel.id.getDevice(), kernel.id.getContext(), kernel.id.getStream());
		//cout << "in sync event, stream id is: " << id->printId() << endl;
		//printf("last event time: %f.\n", lastEventTime);
		//printf("stop time: %f.\n", stop_sec);
		Tau_gpu_register_gpu_event(kernel, 
															 (((double) start_sec) + lastEventTime)*1e3,
															 (((double) stop_sec)  + lastEventTime)*1e3);
		//Tau_cuda_register_gpu_event(kernel.name, kernel.id, 
		//													 (((double) start_sec) + lastEventTime)*1e3,
		//													 (((double) stop_sec)  + lastEventTime)*1e3);

		//delete id;

		lastEvent = kernel.stopEvent;
		lastEventTime += (double) stop_sec;

		KernelBuffer.pop();

	}
	
}
