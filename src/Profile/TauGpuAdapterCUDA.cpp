#include "TauGpuAdapterCUDA.h"
#include <stdio.h>
#include <iostream>
#include <queue>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
using namespace std;

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
cudaEventId::cudaEventId(const int a) :
		id(a) {}
	
	// for use in STL Maps	
bool cudaEventId::operator<(const cudaEventId& A) const
{ 
		return id<A.id; 
}

class KernelEvent
{
	public: 
	const char *name;
	int blocksPerGrid;
	int threadsPerBlock;
	cudaGpuId* id;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;

	KernelEvent() {}

	int enqueue_start_event()
	{
		cudaError_t err;
		cudaEventCreate(&startEvent);
		cudaEventRecord(startEvent, id->getStream());
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
		cudaEventRecord(stopEvent, id->getStream());
		if (err != cudaSuccess)
		{
			printf("Error recording kernel event, error #: %d.\n", err);
			return 1;
		}
		return 0;
	}
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
		//cudaStream_t stream;
		cudaError err; //= cudaStreamCreate(&stream);
		
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
		printf("sync offset: %lf.\n", sync_offset);

		lastEvent = initEvent;
		init = true;
		Tau_gpu_init();
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
	Tau_gpu_enter_memcpy_event(name, &cudaEventId(id), &cudaDriverGpuId(0,0,0), size, MemcpyType);
}

void Tau_cuda_exit_memcpy_event(const char *name, int id, int MemcpyType)
{
	Tau_gpu_exit_memcpy_event(name, &cudaEventId(id), &cudaDriverGpuId(0,0,0), MemcpyType);
}

void Tau_cuda_register_gpu_event(const char *name, cudaGpuId *id, double start,
double stop)
{
	//printf("sync'ed \t start: %lf.\n \t \t \t stop: %lf.\n", start+sync_offset, stop+sync_offset);
	Tau_gpu_register_gpu_event(name, &cudaEventId(0), id, start + sync_offset, stop + sync_offset);
}

void Tau_cuda_register_memcpy_event(const char *name, cudaGpuId* id, double start, double stop, int
transferSize, int MemcpyType)
{
	Tau_gpu_register_memcpy_event(name, &cudaEventId(0), id, start/1e3 + sync_offset, stop/1e3 + sync_offset, transferSize, MemcpyType);
}

#define TAU_KERNEL_STRING_SIZE 1024

/* Function to parse CUDA kernel names. Borrowed from VampirTrace */

const char *parse_kernel_name(const char *devFunc)
{
	/*
	char *kernelName;
	kernelName = (char*) malloc(TAU_KERNEL_STRING_SIZE*sizeof(char));
	sprintf(kernelName, "<addr=%p>", devFunc);

	//printf("kernelName = %s.\n", kernelName);
	return kernelName;

 May use this later */

	char kernelName[TAU_KERNEL_STRING_SIZE];
  int i = 0;       /* position in device function (source string) */
  int nlength = 0; /* length of namespace or kernel */
  int ePos = 0;    /* position in final kernel string */
  char *curr_elem, kn_templates[TAU_KERNEL_STRING_SIZE];
  char *tmpEnd, *tmpElemEnd;

  printf("[CUDA] device funtion name: %s.\n", devFunc);

  /* init for both cases: namespace available or not */
  if(devFunc[2] == 'N'){
    nlength = atoi(&devFunc[3]); /* get length of first namespace */
    i = 4;
  }else{
    nlength = atoi(&devFunc[2]); /* get length of kernel */
    i = 3;
  }

  /* unless string null termination */
  while(devFunc[i] != '\0'){
    /* found either namespace or kernel name (no digits) */
    if(devFunc[i] < '0' || devFunc[i] > '9'){
      /* copy name to kernel function */
      if((ePos + nlength) < TAU_KERNEL_STRING_SIZE){
        (void)strncpy(&kernelName[ePos], &devFunc[i], nlength);
        ePos += nlength; /* set next position to write */
      }else{
        nlength = TAU_KERNEL_STRING_SIZE - ePos;
        (void)strncpy(&kernelName[ePos], &devFunc[i], nlength);
        printf("[CUDA]: kernel name '%s' contains more than %d chars!",
                      devFunc, TAU_KERNEL_STRING_SIZE);
        return kernelName;
      }

      i += nlength; /* jump over name */
      nlength = atoi(&devFunc[i]); /* get length of next namespace or kernel */

      /* finish if no digit after namespace or kernel */
      if(nlength == 0){
        kernelName[ePos] = '\0'; /* set string termination */
        break;
      }else{
        if((ePos + 3) < TAU_KERNEL_STRING_SIZE){
          (void)strncpy(&kernelName[ePos], "::\0", 3);
          ePos += 2;
        }else{
          printf("[CUDA]: kernel name '%s' contains more than %d chars!",
                        devFunc, TAU_KERNEL_STRING_SIZE);
          return kernelName;
        }
      }
    }else i++;
  }

  /* copy the end of the kernel name string to extract templates */
  if(-1 == snprintf(kn_templates, TAU_KERNEL_STRING_SIZE, "%s", &devFunc[i+1]))
    printf( "[CUDA]: Error parsing kernel '%s'", devFunc);
  curr_elem = kn_templates; /* should be 'L' */

  /* search templates (e.g. "_Z10cptCurrentILb1ELi10EEv6SField8SParListifff") */
  tmpEnd=strstr(curr_elem,"EE");
  /* check for templates: curr_elem[0] points to 'L' AND string contains "EE" */
  if(tmpEnd != NULL && curr_elem[0]=='L'){ /* templates exist */
    tmpEnd[1] = '\0'; /* set 2nd 'E' to \0 as string end marker */

    /* write at postion 'I' with '<' */
    /* elem->name[ePos]='<'; */
    if(-1 == snprintf(&(kernelName[ePos]),TAU_KERNEL_STRING_SIZE-ePos,"<"))
      printf("[CUDA] Parsing templates of kernel '%s' failed!", devFunc);
    ePos++; /* continue with next character */

    do{
      int res;
      curr_elem++; /* set pointer to template type length or template type */
      /* find end of template element */
      tmpElemEnd = strchr(curr_elem + atoi(curr_elem), 'E');
      tmpElemEnd[0] = '\0'; /* set termination char after template element */
      /* find next non-digit char */
      while(*curr_elem >= '0' && *curr_elem <= '9') curr_elem++;
      /* append template value to kernel name */
      if(-1 == (res = snprintf(&(kernelName[ePos]),
                               TAU_KERNEL_STRING_SIZE-ePos,"%s,",curr_elem)))
        printf("[CUDA]: Parsing templates of kernel '%s' crashed!", devFunc);
      ePos += res; /* continue after template value */
      curr_elem =tmpElemEnd + 1; /* set current element to begin of next template */
    }while(tmpElemEnd < tmpEnd);
    if((ePos-1) < TAU_KERNEL_STRING_SIZE) (void)strncpy(&kernelName[ePos-1], ">\0", 2);
    else printf("[CUDA]: Templates of '%s' too long for internal buffer!", devFunc);
  } /* else: kernel has no templates */


  printf("[CUDA] funtion name: %s.\n",kernelName);
	return kernelName; 
}


KernelEvent *curKernel;

void Tau_cuda_enqueue_kernel_enter_event(const char *name, cudaGpuId* id)
{
	//printf("recording start for %s.\n", parse_kernel_name(name));

	curKernel = new KernelEvent();
	curKernel->name = parse_kernel_name(name);
	curKernel->id = id->getCopy();

	curKernel->enqueue_start_event();
 
	//printf("Successfully recorded start.\n");

}

void Tau_cuda_enqueue_kernel_exit_event(const char* name, cudaGpuId* id)
{

	//printf("recording stop for %s.\n", parse_kernel_name(name));

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

		err = cudaEventElapsedTime(&stop_sec, lastEvent, kernel.stopEvent);
		//printf("kernel event [stop] = %lf.\n", (((double) stop_sec) + lastEventTime)*1e3 );

		if (err != cudaSuccess)
		{
			printf("Error calculating kernel event, error #: %d.\n", err);
		}

		//Create cudaGpuId for stream.
		//cudaGpuId *id = new cudaGpuId(kernel.id.getDevice(), kernel.id.getContext(), kernel.id.getStream());
		//cout << "in sync event, stream id is: " << id->printId() << endl;
		Tau_cuda_register_gpu_event(kernel.name, kernel.id, 
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
