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
// Add these definitions because the Binutils comedians think all the world uses autotools
#ifndef PACKAGE
#define PACKAGE TAU
#endif
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION 2.25
#endif
#  include <bfd.h>
#endif /* TAU_BFD */


static cudaEvent_t lastEvent;
static double lastEventTime = 0;

//call to cudaEventQuery that does not register a sync event.
extern cudaError_t cudaEventQuery_nosync(cudaEvent_t a);


/*
bool CudaRuntimeGpuEvent::operator<(const CudaGpuEvent& other) const
{
	if (device == other.device)
	{
		return stream < other.stream;
	}
	else
		return device < other.device;
}*/



/*
bool CudaDriverGpuEvent::operator<(const CudaGpuEvent& other) const
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

/*
CUstream CudaGpuEvent::get_dr_stream(void)
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
cudaStream_t CudaGpuEvent::get_rt_stream(void)
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



static queue<CudaGpuEvent *> KernelBuffer;



void Tau_cuda_init()
{
	//printf("in Tau_cuda_init.\n");
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
		//lastEventTime = sync_offset / 1e3;  
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
        CudaDriverGpuEvent x(0,0,0);
	//Tau_gpu_enter_memcpy_event(name, &CudaDriverGpuEvent(0,0,0), size, MemcpyType);
	Tau_gpu_enter_memcpy_event(name, &x, size, MemcpyType);
}

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType)
{
        CudaDriverGpuEvent x(0,0,0);
	//Tau_gpu_exit_memcpy_event(name, &CudaDriverGpuEvent(0,0,0), MemcpyType);
	Tau_gpu_exit_memcpy_event(name, &x, MemcpyType);
}

// void Tau_cuda_register_gpu_event(KernelEvent k, double start,
// double stop)
// {
// 	//printf("sync'ed \t start: %lf.\n \t \t \t stop: %lf.\n", start+sync_offset, stop+sync_offset);
// 	//FunctionInfo *p = TauInternal_CurrentProfiler(RtsLayer::myNode())->ThisFunction;
// 	//eventId c = Tau_gpu_create_gpu_event(name, id, p);
// 	Tau_gpu_register_gpu_event(k, start + sync_offset, stop + sync_offset);
// }

void Tau_cuda_register_memcpy_event(const char *name, CudaGpuEvent* id, double start, double stop, int
				    transferSize, int MemcpyType)
{
	FunctionInfo *p = TauInternal_CurrentProfiler(RtsLayer::myThread())->ThisFunction;
	//eventId c = Tau_gpu_create_gpu_event(name, id, p, NULL);
	id->name = name;
	id->callingSite = p;
	Tau_gpu_register_memcpy_event(id, start/1e3, stop/1e3, transferSize, MemcpyType, MESSAGE_UNKNOWN);
}

//CudaGpuEvent *curKernel;
//needed for pycuda for some reason.
string curName;

void Tau_cuda_enqueue_kernel_enter_event(CudaGpuEvent* id)
{
	FunctionInfo* callingSite;
	const char *name = id->getName();
	if (TauInternal_CurrentProfiler(RtsLayer::myThread()) == NULL)
	{
		callingSite = NULL;
	}
	else
	{
		callingSite = TauInternal_CurrentProfiler(RtsLayer::myThread())->CallPathFunction;
	}
	//printf("recording start for %s.\n", name);

	id->callingSite = callingSite;
	
	const char *dem_name = 0;
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
	//printf("demangling name....\n");
	dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE |
	DMGL_TYPES);
  //revert to original string if demangle fails.
  if (dem_name == NULL)
  {
    dem_name = name;
  }
#else
	dem_name = name;
#endif /* HAVE_GPU_DEMANGLE */

	//printf("final kernel name is: %s.\n", dem_name);

	id->name = dem_name;
	//curKernel->device = id->getCopy();

	id->enqueue_start_event();
	curName = string(dem_name); 
	//printf("Successfully recorded start.\n");
	//KernelBuffer.push(*curKernel);

	//curKernel = id->getCopy();
}

void Tau_cuda_enqueue_kernel_exit_event(CudaGpuEvent *curKernel)
{

	//printf("recording stop for: %s.\n", curName.c_str());

	const int len = sizeof(char) * (curName.length() + 1);
	char *device_name = (char*) malloc(len);
	strncpy(device_name,  curName.c_str(), len); 
	curKernel->name = device_name;
	curKernel->enqueue_stop_event();

	KernelBuffer.push(curKernel);

 
	//printf("Successfully recorded stop.\n");
}

static int in_sync_event = 0;

void Tau_cuda_register_sync_event()
{
	//printf("sync flag: %d.\n", in_sync_event);
	if (in_sync_event)
	{
		return;
	}
	in_sync_event = 1;
	//printf("in sync event, buffer size: %d.\n", KernelBuffer.size());	
	
	if (KernelBuffer.size() > 0 && KernelBuffer.front()->stopEvent != NULL)
	{
		//printf("buffer front stop: %d.\n", KernelBuffer.front().stopEvent == NULL);
		cudaError err = cudaEventQuery(KernelBuffer.front()->stopEvent);
		//printf("buffer front is: %d\n", err);
	}
	float start_sec, stop_sec;

	while (!KernelBuffer.empty() && cudaEventQuery(KernelBuffer.front()->stopEvent) == cudaSuccess)
	{
		CudaGpuEvent* kernel = KernelBuffer.front();
		//printf("kernel buffer size = %d.\n", KernelBuffer.size());

		cudaError_t err;
		err = cudaEventElapsedTime(&start_sec, lastEvent, kernel->startEvent);
		//printf("kernel event [start] = %lf.\n", (((double) start_sec))*1e3);
		//printf("w last event [start] = %lf.\n", (((double) start_sec) + lastEventTime)*1e3);

		if (err != cudaSuccess)
		{
			printf("Error calculating kernel event start, error #: %d.\n", err);
		}

		err = cudaEventElapsedTime(&stop_sec, lastEvent, kernel->stopEvent);
		//printf("kernel event [name]  = %s.\n", kernel.name);
		//printf("kernel event [stop]  = %lf.\n", (((double) stop_sec))*1e3 );
		//printf("w last event [stop]  = %lf.\n", (((double) stop_sec) + lastEventTime)*1e3 );

		if (err != cudaSuccess)
		{
			printf("Error calculating kernel event stop, error #: %d.\n", err);
		}
		//printf("kernel event [sync]  = %lf.\n", kernel.device->syncOffset());

		//Create CudaGpuEvent for stream.
		//CudaGpuEvent *id = new cudaGpuEvent(kernel.id.getDevice(), kernel.id.getContext(), kernel.id.getStream());
		//cout << "in sync event, stream id is: " << kernel.device->gpuIdentifier() << endl;
		//printf("last event time: %f.\n", lastEventTime);
		//printf("stop time: %f.\n", stop_sec);

		//kernel.device->sync_offset = lastEventTime * 1e3;

	  //printf("in tau_cuda_register_sync_event #1");
		
		//size_t f1, f2;
		//string s = string(kernel->getName());
	    //f1 = s.find('(');
		//f2 = s.find(')');
		//if (f1 != string::npos && f2 != string::npos)	
		{
			// Tau_gpu_register_gpu_event(kernel, 
			// 													 ((double) start_sec + lastEventTime)*1e3,
			// 													 ((double) stop_sec + lastEventTime)*1e3);
		  int deviceid = 0;
		  Tau_gpu_register_gpu_event(kernel, 
					     ((double) start_sec + lastEventTime)*1e3,
					     ((double) stop_sec + lastEventTime)*1e3);

		}
	  //printf("in tau_cuda_register_sync_event #2");
		//Tau_cuda_register_gpu_event(kernel.name, kernel.id, 
		//													 (((double) start_sec) + lastEventTime)*1e3,
		//													 (((double) stop_sec)  + lastEventTime)*1e3);

		//delete id;
		lastEvent = kernel->stopEvent;
		lastEventTime += (double) stop_sec;

		free((void *)(kernel->name));
		delete kernel;
		KernelBuffer.pop();

	}
		in_sync_event = 0;
	
}
