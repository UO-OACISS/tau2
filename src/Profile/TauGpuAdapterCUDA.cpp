#include "TauGpuAdapterCUDA.h"
#include <stdio.h>
#include <queue>
#include <stdlib.h>
#include <string.h>
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
	const char *name;
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

	Tau_cuda_register_sync_event();
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

#define TAU_KERNEL_STRING_SIZE 1024

/* Function to parse CUDA kernel names. Borrowed from VampirTrace */

const char *parse_kernel_name(const char *devFunc)
{

	char *kernelName;
	kernelName = (char*) malloc(TAU_KERNEL_STRING_SIZE*sizeof(char));
	sprintf(kernelName, "<addr=%p>", devFunc);

	//printf("kernelName = %s.\n", kernelName);
	return kernelName;

/* May use this later */

#ifdef FALSE 
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


  //printf("[CUDA] funtion name: %s",kernelName);
	return kernelName; 
#endif
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
	curKernel.name = parse_kernel_name(name);

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
		//printf("kernel buffer size = %d.\n", KernelBuffer.size());

		cudaEventElapsedTime(&start_sec, lastEvent, kernel.startEvent);
		//printf("kernel event [start] = %f.\n", start_sec + lastEventTime);

		cudaEventElapsedTime(&stop_sec, lastEvent, kernel.stopEvent);
		//printf("kernel event [stop] = %f.\n", stop_sec + lastEventTime );

		Tau_cuda_register_gpu_event(kernel.name, kernel.id, (double) start_sec,
		(double) stop_sec);

		lastEvent = kernel.stopEvent;
		lastEventTime += stop_sec;

		KernelBuffer.pop();

	}

}
