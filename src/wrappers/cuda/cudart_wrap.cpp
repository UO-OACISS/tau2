#include <cuda_runtime_api.h>
#include <Profile/Profiler.h>
#include <Profile/TauGpuAdapterCUDA.h>
#include <dlfcn.h>
#include <stdio.h>
#include <queue>
#include <iostream>
using namespace std;

#define TRACK_MEMORY
#define TRACK_KERNEL
#define KERNEL_EVENT_BUFFFER 4096

#define CUDART_API TAU_USER
#define CUDA_SYNC TAU_USER

#ifdef CUPTI
extern void Tau_CuptiLayer_finalize();
#endif //CUPTI

const char * cudart_orig_libname = "libcudart.so";
static void *cudart_handle = NULL;
cudaStream_t curr_stream;

void tau_track_memory(int kind, int count)
{
	static bool init = false;
	static TauContextUserEvent *MemoryCopyEventHtoD;
	static TauContextUserEvent *MemoryCopyEventDtoH;
	static TauContextUserEvent *MemoryCopyEventDtoD;
	if (!init)
	{
		
		Tau_get_context_userevent((void **) &MemoryCopyEventHtoD, "Bytes copied from Host to Device");
		Tau_get_context_userevent((void **) &MemoryCopyEventDtoH, "Bytes copied from Device to Host");
		Tau_get_context_userevent((void **) &MemoryCopyEventDtoD, "Bytes copied (Other)");
		init = true;
	}
	/*printf("initalize counters. Number of events: %ld, %ld, %ld.\n", 
	MemoryCopyEventHtoD->GetNumEvents(0),
	MemoryCopyEventDtoH->GetNumEvents(0),
	MemoryCopyEventDtoD->GetNumEvents(0));*/
	//printf("tracking memory.... %ld.\n", count);
	if (kind == cudaMemcpyHostToDevice)
		TAU_CONTEXT_EVENT(MemoryCopyEventHtoD, count);
	if (kind == cudaMemcpyDeviceToHost)
		TAU_CONTEXT_EVENT(MemoryCopyEventDtoH, count);
	if (kind == cudaMemcpyDeviceToDevice)
		TAU_CONTEXT_EVENT(MemoryCopyEventDtoD, count);
}	

//cudaThreadExit is depercated to be replaced with cudaDeviceReset.

#if CUDART_VERSION == 4000
cudaError_t cudaDeviceReset() {

  typedef cudaError_t (*cudaDeviceReset_p) ();
  static cudaDeviceReset_p cudaDeviceReset_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaDeviceReset(void) C", "", CUDA_SYNC);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaDeviceReset_h == NULL)
	cudaDeviceReset_h = (cudaDeviceReset_p) dlsym(cudart_handle,"cudaDeviceReset"); 
    if (cudaDeviceReset_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
	//printf("in cudaDeviceReset(), check for kernel events.\n");
#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif 
  TAU_PROFILE_START(t);
#ifdef CUPTI
	Tau_CuptiLayer_finalize();
#endif //CUPTI
  retval  =  (*cudaDeviceReset_h)();
  TAU_PROFILE_STOP(t);

#ifdef TRACK_KERNEL
	Tau_cuda_exit();
#endif
  }
  return retval;

}


#endif



cudaError_t cudaThreadExit() {

  typedef cudaError_t (*cudaThreadExit_p) ();
  static cudaThreadExit_p cudaThreadExit_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadExit(void) C", "", CUDA_SYNC);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadExit_h == NULL)
	cudaThreadExit_h = (cudaThreadExit_p) dlsym(cudart_handle,"cudaThreadExit"); 
    if (cudaThreadExit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
	//printf("in cudaThreadExit(), check for kernel events.\n");
#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif 
  TAU_PROFILE_START(t);
#ifdef CUPTI
	Tau_CuptiLayer_finalize();
#endif //CUPTI
  retval  =  (*cudaThreadExit_h)();
  TAU_PROFILE_STOP(t);

#ifdef TRACK_KERNEL
	Tau_cuda_exit();
#endif
  }
  return retval;

}

cudaError_t cudaThreadSynchronize() {

  typedef cudaError_t (*cudaThreadSynchronize_p) ();
  static cudaThreadSynchronize_p cudaThreadSynchronize_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadSynchronize(void) C", "", CUDA_SYNC);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadSynchronize_h == NULL)
	cudaThreadSynchronize_h = (cudaThreadSynchronize_p) dlsym(cudart_handle,"cudaThreadSynchronize"); 
    if (cudaThreadSynchronize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadSynchronize_h)();
  TAU_PROFILE_STOP(t);

#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif

  }
  return retval;

}
/*
cudaError_t cudaThreadSetLimit(enum cudaLimit a1, size_t a2) {

  typedef cudaError_t (*cudaThreadSetLimit_p) (enum cudaLimit, size_t);
  static cudaThreadSetLimit_p cudaThreadSetLimit_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadSetLimit(enum cudaLimit, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadSetLimit_h == NULL)
	cudaThreadSetLimit_h = (cudaThreadSetLimit_p) dlsym(cudart_handle,"cudaThreadSetLimit"); 
    if (cudaThreadSetLimit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadSetLimit_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaThreadGetLimit(size_t * a1, enum cudaLimit a2) {

  typedef cudaError_t (*cudaThreadGetLimit_p) (size_t *, enum cudaLimit);
  static cudaThreadGetLimit_p cudaThreadGetLimit_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadGetLimit(size_t *, enum cudaLimit) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadGetLimit_h == NULL)
	cudaThreadGetLimit_h = (cudaThreadGetLimit_p) dlsym(cudart_handle,"cudaThreadGetLimit"); 
    if (cudaThreadGetLimit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadGetLimit_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache * a1) {

  typedef cudaError_t (*cudaThreadGetCacheConfig_p) (enum cudaFuncCache *);
  static cudaThreadGetCacheConfig_p cudaThreadGetCacheConfig_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadGetCacheConfig_h == NULL)
	cudaThreadGetCacheConfig_h = (cudaThreadGetCacheConfig_p) dlsym(cudart_handle,"cudaThreadGetCacheConfig"); 
    if (cudaThreadGetCacheConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadGetCacheConfig_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache a1) {

  typedef cudaError_t (*cudaThreadSetCacheConfig_p) (enum cudaFuncCache);
  static cudaThreadSetCacheConfig_p cudaThreadSetCacheConfig_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadSetCacheConfig_h == NULL)
	cudaThreadSetCacheConfig_h = (cudaThreadSetCacheConfig_p) dlsym(cudart_handle,"cudaThreadSetCacheConfig"); 
    if (cudaThreadSetCacheConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadSetCacheConfig_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}*/

cudaError_t cudaGetLastError() {

  typedef cudaError_t (*cudaGetLastError_p) ();
  static cudaGetLastError_p cudaGetLastError_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetLastError(void) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetLastError_h == NULL)
	cudaGetLastError_h = (cudaGetLastError_p) dlsym(cudart_handle,"cudaGetLastError"); 
    if (cudaGetLastError_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetLastError_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaPeekAtLastError() {

  typedef cudaError_t (*cudaPeekAtLastError_p) ();
  static cudaPeekAtLastError_p cudaPeekAtLastError_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaPeekAtLastError(void) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaPeekAtLastError_h == NULL)
	cudaPeekAtLastError_h = (cudaPeekAtLastError_p) dlsym(cudart_handle,"cudaPeekAtLastError"); 
    if (cudaPeekAtLastError_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaPeekAtLastError_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

const char * cudaGetErrorString(cudaError_t a1) {

  typedef const char * (*cudaGetErrorString_p) (cudaError_t);
  static cudaGetErrorString_p cudaGetErrorString_h = NULL;
  const char * retval;
  TAU_PROFILE_TIMER(t,"const char *cudaGetErrorString(cudaError_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetErrorString_h == NULL)
	cudaGetErrorString_h = (cudaGetErrorString_p) dlsym(cudart_handle,"cudaGetErrorString"); 
    if (cudaGetErrorString_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetErrorString_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetDeviceCount(int * a1) {

  typedef cudaError_t (*cudaGetDeviceCount_p) (int *);
  static cudaGetDeviceCount_p cudaGetDeviceCount_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetDeviceCount(int *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetDeviceCount_h == NULL)
	cudaGetDeviceCount_h = (cudaGetDeviceCount_p) dlsym(cudart_handle,"cudaGetDeviceCount"); 
    if (cudaGetDeviceCount_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetDeviceCount_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * a1, int a2) {

  typedef cudaError_t (*cudaGetDeviceProperties_p) (struct cudaDeviceProp *, int);
  static cudaGetDeviceProperties_p cudaGetDeviceProperties_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *, int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetDeviceProperties_h == NULL)
	cudaGetDeviceProperties_h = (cudaGetDeviceProperties_p) dlsym(cudart_handle,"cudaGetDeviceProperties"); 
    if (cudaGetDeviceProperties_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetDeviceProperties_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaChooseDevice(int * a1, const struct cudaDeviceProp * a2) {

  typedef cudaError_t (*cudaChooseDevice_p) (int *, const struct cudaDeviceProp *);
  static cudaChooseDevice_p cudaChooseDevice_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaChooseDevice(int *, const struct cudaDeviceProp *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaChooseDevice_h == NULL)
	cudaChooseDevice_h = (cudaChooseDevice_p) dlsym(cudart_handle,"cudaChooseDevice"); 
    if (cudaChooseDevice_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaChooseDevice_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDevice(int a1) {

  typedef cudaError_t (*cudaSetDevice_p) (int);
  static cudaSetDevice_p cudaSetDevice_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDevice(int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDevice_h == NULL)
	cudaSetDevice_h = (cudaSetDevice_p) dlsym(cudart_handle,"cudaSetDevice"); 
    if (cudaSetDevice_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetDevice(int * a1) {

  typedef cudaError_t (*cudaGetDevice_p) (int *);
  static cudaGetDevice_p cudaGetDevice_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetDevice(int *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetDevice_h == NULL)
	cudaGetDevice_h = (cudaGetDevice_p) dlsym(cudart_handle,"cudaGetDevice"); 
    if (cudaGetDevice_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetValidDevices(int * a1, int a2) {

  typedef cudaError_t (*cudaSetValidDevices_p) (int *, int);
  static cudaSetValidDevices_p cudaSetValidDevices_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetValidDevices(int *, int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetValidDevices_h == NULL)
	cudaSetValidDevices_h = (cudaSetValidDevices_p) dlsym(cudart_handle,"cudaSetValidDevices"); 
    if (cudaSetValidDevices_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetValidDevices_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDeviceFlags(unsigned int a1) {

  typedef cudaError_t (*cudaSetDeviceFlags_p) (unsigned int);
  static cudaSetDeviceFlags_p cudaSetDeviceFlags_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDeviceFlags(unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDeviceFlags_h == NULL)
	cudaSetDeviceFlags_h = (cudaSetDeviceFlags_p) dlsym(cudart_handle,"cudaSetDeviceFlags"); 
    if (cudaSetDeviceFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDeviceFlags_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamCreate(cudaStream_t * a1) {

  typedef cudaError_t (*cudaStreamCreate_p) (cudaStream_t *);
  static cudaStreamCreate_p cudaStreamCreate_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamCreate(cudaStream_t *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamCreate_h == NULL)
	cudaStreamCreate_h = (cudaStreamCreate_p) dlsym(cudart_handle,"cudaStreamCreate"); 
    if (cudaStreamCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamCreate_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamDestroy(cudaStream_t a1) {

  typedef cudaError_t (*cudaStreamDestroy_p) (cudaStream_t);
  static cudaStreamDestroy_p cudaStreamDestroy_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamDestroy(cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamDestroy_h == NULL)
	cudaStreamDestroy_h = (cudaStreamDestroy_p) dlsym(cudart_handle,"cudaStreamDestroy"); 
    if (cudaStreamDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamWaitEvent(cudaStream_t a1, cudaEvent_t a2, unsigned int a3) {

  typedef cudaError_t (*cudaStreamWaitEvent_p) (cudaStream_t, cudaEvent_t, unsigned int);
  static cudaStreamWaitEvent_p cudaStreamWaitEvent_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamWaitEvent_h == NULL)
	cudaStreamWaitEvent_h = (cudaStreamWaitEvent_p) dlsym(cudart_handle,"cudaStreamWaitEvent"); 
    if (cudaStreamWaitEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamWaitEvent_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamSynchronize(cudaStream_t a1) {

  typedef cudaError_t (*cudaStreamSynchronize_p) (cudaStream_t);
  static cudaStreamSynchronize_p cudaStreamSynchronize_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamSynchronize(cudaStream_t) C", "", CUDA_SYNC);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamSynchronize_h == NULL)
	cudaStreamSynchronize_h = (cudaStreamSynchronize_p) dlsym(cudart_handle,"cudaStreamSynchronize"); 
    if (cudaStreamSynchronize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamSynchronize_h)( a1);
  TAU_PROFILE_STOP(t);
	
#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif
  }
  return retval;

}

cudaError_t cudaStreamQuery(cudaStream_t a1) {

  typedef cudaError_t (*cudaStreamQuery_p) (cudaStream_t);
  static cudaStreamQuery_p cudaStreamQuery_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamQuery(cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamQuery_h == NULL)
	cudaStreamQuery_h = (cudaStreamQuery_p) dlsym(cudart_handle,"cudaStreamQuery"); 
    if (cudaStreamQuery_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamQuery_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventCreate(cudaEvent_t * a1) {

  typedef cudaError_t (*cudaEventCreate_p) (cudaEvent_t *);
  static cudaEventCreate_p cudaEventCreate_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventCreate(cudaEvent_t *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventCreate_h == NULL)
	cudaEventCreate_h = (cudaEventCreate_p) dlsym(cudart_handle,"cudaEventCreate"); 
    if (cudaEventCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventCreate_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t * a1, unsigned int a2) {

  typedef cudaError_t (*cudaEventCreateWithFlags_p) (cudaEvent_t *, unsigned int);
  static cudaEventCreateWithFlags_p cudaEventCreateWithFlags_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventCreateWithFlags_h == NULL)
	cudaEventCreateWithFlags_h = (cudaEventCreateWithFlags_p) dlsym(cudart_handle,"cudaEventCreateWithFlags"); 
    if (cudaEventCreateWithFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventCreateWithFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventRecord(cudaEvent_t a1, cudaStream_t a2) {

  typedef cudaError_t (*cudaEventRecord_p) (cudaEvent_t, cudaStream_t);
  static cudaEventRecord_p cudaEventRecord_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventRecord_h == NULL)
	cudaEventRecord_h = (cudaEventRecord_p) dlsym(cudart_handle,"cudaEventRecord"); 
    if (cudaEventRecord_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventRecord_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventQuery(cudaEvent_t a1) {

  typedef cudaError_t (*cudaEventQuery_p) (cudaEvent_t);
  static cudaEventQuery_p cudaEventQuery_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventQuery(cudaEvent_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventQuery_h == NULL)
	cudaEventQuery_h = (cudaEventQuery_p) dlsym(cudart_handle,"cudaEventQuery"); 
    if (cudaEventQuery_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventQuery_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventSynchronize(cudaEvent_t a1) {

  typedef cudaError_t (*cudaEventSynchronize_p) (cudaEvent_t);
  static cudaEventSynchronize_p cudaEventSynchronize_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventSynchronize(cudaEvent_t) C", "", CUDA_SYNC);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventSynchronize_h == NULL)
	cudaEventSynchronize_h = (cudaEventSynchronize_p) dlsym(cudart_handle,"cudaEventSynchronize"); 
    if (cudaEventSynchronize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventSynchronize_h)( a1);
  TAU_PROFILE_STOP(t);

#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif

  }
  return retval;

}

cudaError_t cudaEventDestroy(cudaEvent_t a1) {

  typedef cudaError_t (*cudaEventDestroy_p) (cudaEvent_t);
  static cudaEventDestroy_p cudaEventDestroy_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventDestroy(cudaEvent_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventDestroy_h == NULL)
	cudaEventDestroy_h = (cudaEventDestroy_p) dlsym(cudart_handle,"cudaEventDestroy"); 
    if (cudaEventDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventElapsedTime(float * a1, cudaEvent_t a2, cudaEvent_t a3) {

  typedef cudaError_t (*cudaEventElapsedTime_p) (float *, cudaEvent_t, cudaEvent_t);
  static cudaEventElapsedTime_p cudaEventElapsedTime_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventElapsedTime_h == NULL)
	cudaEventElapsedTime_h = (cudaEventElapsedTime_p) dlsym(cudart_handle,"cudaEventElapsedTime"); 
    if (cudaEventElapsedTime_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventElapsedTime_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaConfigureCall(dim3 a1, dim3 a2, size_t a3, cudaStream_t a4) {

  typedef cudaError_t (*cudaConfigureCall_p) (dim3, dim3, size_t, cudaStream_t);
  static cudaConfigureCall_p cudaConfigureCall_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaConfigureCall(dim3, dim3, size_t, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaConfigureCall_h == NULL)
	cudaConfigureCall_h = (cudaConfigureCall_p) dlsym(cudart_handle,"cudaConfigureCall"); 
    if (cudaConfigureCall_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }

	//cout << "in cudaConfigure... stream is " << a4 << endl;
	/*
	if (a4 == 0)
	{
		cudaStreamCreate(&curr_stream);
	}
	else
	{
		curr_stream = a4;
	}*/

	curr_stream = a4;
	
  TAU_PROFILE_START(t);
  retval  =  (*cudaConfigureCall_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetupArgument(const void * a1, size_t a2, size_t a3) {

  typedef cudaError_t (*cudaSetupArgument_p) (const void *, size_t, size_t);
  static cudaSetupArgument_p cudaSetupArgument_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetupArgument(const void *, size_t, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetupArgument_h == NULL)
	cudaSetupArgument_h = (cudaSetupArgument_p) dlsym(cudart_handle,"cudaSetupArgument"); 
    if (cudaSetupArgument_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetupArgument_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
/*
cudaError_t cudaFuncSetCacheConfig(const char * a1, enum cudaFuncCache a2) {

  typedef cudaError_t (*cudaFuncSetCacheConfig_p) (const char *, enum cudaFuncCache);
  static cudaFuncSetCacheConfig_p cudaFuncSetCacheConfig_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFuncSetCacheConfig(const char *, enum cudaFuncCache) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFuncSetCacheConfig_h == NULL)
	cudaFuncSetCacheConfig_h = (cudaFuncSetCacheConfig_p) dlsym(cudart_handle,"cudaFuncSetCacheConfig"); 
    if (cudaFuncSetCacheConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFuncSetCacheConfig_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}*/

char *kernelName = "";

/*
 * This function is being called before execution of a cuda program for every
 * cuda kernel (host_runtime.h)
 * Borrowed from VampirTrace.
 */
extern "C" void __cudaRegisterFunction(void ** a1, const char * a2, char * a3, const char * a4, int a5, uint3 * a6, uint3 * a7, dim3 * a8, dim3 * a9, int * a10);
extern "C" void __cudaRegisterFunction(void ** a1, const char * a2, char * a3, const char * a4, int a5, uint3 * a6, uint3 * a7, dim3 * a8, dim3 * a9, int * a10) {

	//printf("*** in __cudaRegisterFunction.\n");
	//printf("Kernel name is: %s.\n", a3);

  typedef void (*__cudaRegisterFunction_p_h) (void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
  static __cudaRegisterFunction_p_h __cudaRegisterFunction_h = NULL;
  TAU_PROFILE_TIMER(t,"void __cudaRegisterFunction(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (__cudaRegisterFunction_h == NULL)
	__cudaRegisterFunction_h = (__cudaRegisterFunction_p_h) dlsym(cudart_handle,"__cudaRegisterFunction"); 
    if (__cudaRegisterFunction_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
	
	kernelName = a3;

  (*__cudaRegisterFunction_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);
  }

}

cudaError_t cudaLaunch(const char * a1) {

  typedef cudaError_t (*cudaLaunch_p) (const char *);
  static cudaLaunch_p cudaLaunch_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaLaunch(const char *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaLaunch_h == NULL)
	cudaLaunch_h = (cudaLaunch_p) dlsym(cudart_handle,"cudaLaunch"); 
    if (cudaLaunch_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
		
		//printf("in cudaLaunch, TAU wrap.\n");
		//printf("cuda kernel: %s.\n", kernelName);
  
		TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
		//printf("tracking kernel on node: %d.\n", RtsLayer::myNode());
		FunctionInfo* parent;
		if (TauInternal_CurrentProfiler(RtsLayer::getTid()) == NULL)
		{
			parent = NULL;
		}
		else
		{
			parent = TauInternal_CurrentProfiler(RtsLayer::getTid())->CallPathFunction;
		}
		Tau_cuda_init();
		int device;
		cudaGetDevice(&device);
		Tau_cuda_enqueue_kernel_enter_event(kernelName,
			&cudaRuntimeGpuId(device,curr_stream), parent);
		/*Tau_cuda_enqueue_kernel_enter_event(kernelName,
			&cudaRuntimeGpuId(device,curr_stream),
			TauInternal_CurrentProfiler(RtsLayer::myNode())->CallPathFunction);*/
#endif
		retval  =  (*cudaLaunch_h)( a1);
#ifdef TRACK_KERNEL
		Tau_cuda_enqueue_kernel_exit_event();
#endif
		TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * a1, const char * a2) {

  typedef cudaError_t (*cudaFuncGetAttributes_p) (struct cudaFuncAttributes *, const char *);
  static cudaFuncGetAttributes_p cudaFuncGetAttributes_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *, const char *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFuncGetAttributes_h == NULL)
	cudaFuncGetAttributes_h = (cudaFuncGetAttributes_p) dlsym(cudart_handle,"cudaFuncGetAttributes"); 
    if (cudaFuncGetAttributes_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFuncGetAttributes_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDoubleForDevice(double * a1) {

  typedef cudaError_t (*cudaSetDoubleForDevice_p) (double *);
  static cudaSetDoubleForDevice_p cudaSetDoubleForDevice_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDoubleForDevice(double *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDoubleForDevice_h == NULL)
	cudaSetDoubleForDevice_h = (cudaSetDoubleForDevice_p) dlsym(cudart_handle,"cudaSetDoubleForDevice"); 
    if (cudaSetDoubleForDevice_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDoubleForDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDoubleForHost(double * a1) {

  typedef cudaError_t (*cudaSetDoubleForHost_p) (double *);
  static cudaSetDoubleForHost_p cudaSetDoubleForHost_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDoubleForHost(double *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDoubleForHost_h == NULL)
	cudaSetDoubleForHost_h = (cudaSetDoubleForHost_p) dlsym(cudart_handle,"cudaSetDoubleForHost"); 
    if (cudaSetDoubleForHost_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDoubleForHost_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMalloc(void ** a1, size_t a2) {

  typedef cudaError_t (*cudaMalloc_p) (void **, size_t);
  static cudaMalloc_p cudaMalloc_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMalloc(void **, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMalloc_h == NULL)
	cudaMalloc_h = (cudaMalloc_p) dlsym(cudart_handle,"cudaMalloc"); 
    if (cudaMalloc_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMalloc_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMallocHost(void ** a1, size_t a2) {

  typedef cudaError_t (*cudaMallocHost_p) (void **, size_t);
  static cudaMallocHost_p cudaMallocHost_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMallocHost(void **, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMallocHost_h == NULL)
	cudaMallocHost_h = (cudaMallocHost_p) dlsym(cudart_handle,"cudaMallocHost"); 
    if (cudaMallocHost_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMallocHost_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMallocPitch(void ** a1, size_t * a2, size_t a3, size_t a4) {

  typedef cudaError_t (*cudaMallocPitch_p) (void **, size_t *, size_t, size_t);
  static cudaMallocPitch_p cudaMallocPitch_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMallocPitch_h == NULL)
	cudaMallocPitch_h = (cudaMallocPitch_p) dlsym(cudart_handle,"cudaMallocPitch"); 
    if (cudaMallocPitch_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMallocPitch_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
cudaError_t cudaMallocArray(struct cudaArray ** a1, const struct cudaChannelFormatDesc * a2, size_t a3, size_t a4, unsigned int a5) {

  typedef cudaError_t (*cudaMallocArray_p) (struct cudaArray **, const struct cudaChannelFormatDesc *, size_t, size_t, unsigned int);
  static cudaMallocArray_p cudaMallocArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMallocArray(struct cudaArray **, const struct cudaChannelFormatDesc *, size_t, size_t, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMallocArray_h == NULL)
	cudaMallocArray_h = (cudaMallocArray_p) dlsym(cudart_handle,"cudaMallocArray"); 
    if (cudaMallocArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMallocArray_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
cudaError_t cudaFree(void * a1) {

  typedef cudaError_t (*cudaFree_p) (void *);
  static cudaFree_p cudaFree_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFree(void *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFree_h == NULL)
	cudaFree_h = (cudaFree_p) dlsym(cudart_handle,"cudaFree"); 
    if (cudaFree_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFree_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaFreeHost(void * a1) {

  typedef cudaError_t (*cudaFreeHost_p) (void *);
  static cudaFreeHost_p cudaFreeHost_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFreeHost(void *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFreeHost_h == NULL)
	cudaFreeHost_h = (cudaFreeHost_p) dlsym(cudart_handle,"cudaFreeHost"); 
    if (cudaFreeHost_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFreeHost_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaFreeArray(struct cudaArray * a1) {

  typedef cudaError_t (*cudaFreeArray_p) (struct cudaArray *);
  static cudaFreeArray_p cudaFreeArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFreeArray(struct cudaArray *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFreeArray_h == NULL)
	cudaFreeArray_h = (cudaFreeArray_p) dlsym(cudart_handle,"cudaFreeArray"); 
    if (cudaFreeArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFreeArray_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaHostAlloc(void ** a1, size_t a2, unsigned int a3) {

  typedef cudaError_t (*cudaHostAlloc_p) (void **, size_t, unsigned int);
  static cudaHostAlloc_p cudaHostAlloc_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaHostAlloc(void **, size_t, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaHostAlloc_h == NULL)
	cudaHostAlloc_h = (cudaHostAlloc_p) dlsym(cudart_handle,"cudaHostAlloc"); 
    if (cudaHostAlloc_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaHostAlloc_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaHostGetDevicePointer(void ** a1, void * a2, unsigned int a3) {

  typedef cudaError_t (*cudaHostGetDevicePointer_p) (void **, void *, unsigned int);
  static cudaHostGetDevicePointer_p cudaHostGetDevicePointer_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaHostGetDevicePointer(void **, void *, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaHostGetDevicePointer_h == NULL)
	cudaHostGetDevicePointer_h = (cudaHostGetDevicePointer_p) dlsym(cudart_handle,"cudaHostGetDevicePointer"); 
    if (cudaHostGetDevicePointer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaHostGetDevicePointer_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaHostGetFlags(unsigned int * a1, void * a2) {

  typedef cudaError_t (*cudaHostGetFlags_p) (unsigned int *, void *);
  static cudaHostGetFlags_p cudaHostGetFlags_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaHostGetFlags(unsigned int *, void *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaHostGetFlags_h == NULL)
	cudaHostGetFlags_h = (cudaHostGetFlags_p) dlsym(cudart_handle,"cudaHostGetFlags"); 
    if (cudaHostGetFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaHostGetFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr * a1, struct cudaExtent a2) {

  typedef cudaError_t (*cudaMalloc3D_p) (struct cudaPitchedPtr *, struct cudaExtent);
  static cudaMalloc3D_p cudaMalloc3D_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMalloc3D(struct cudaPitchedPtr *, struct cudaExtent) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMalloc3D_h == NULL)
	cudaMalloc3D_h = (cudaMalloc3D_p) dlsym(cudart_handle,"cudaMalloc3D"); 
    if (cudaMalloc3D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMalloc3D_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMalloc3DArray(struct cudaArray ** a1, const struct cudaChannelFormatDesc * a2, struct cudaExtent a3, unsigned int a4) {

  typedef cudaError_t (*cudaMalloc3DArray_p) (struct cudaArray **, const struct cudaChannelFormatDesc *, struct cudaExtent, unsigned int);
  static cudaMalloc3DArray_p cudaMalloc3DArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMalloc3DArray(struct cudaArray **, const struct cudaChannelFormatDesc *, struct cudaExtent, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMalloc3DArray_h == NULL)
	cudaMalloc3DArray_h = (cudaMalloc3DArray_p) dlsym(cudart_handle,"cudaMalloc3DArray"); 
    if (cudaMalloc3DArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMalloc3DArray_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * a1) {

  typedef cudaError_t (*cudaMemcpy3D_p) (const struct cudaMemcpy3DParms *);
  static cudaMemcpy3D_p cudaMemcpy3D_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy3D_h == NULL)
	cudaMemcpy3D_h = (cudaMemcpy3D_p) dlsym(cudart_handle,"cudaMemcpy3D"); 
    if (cudaMemcpy3D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }

  TAU_PROFILE_START(t);
// cannot find example of cudaMemcpy3D to test memory tracking
#ifdef TRACK_MEMORY
#endif //TRACK_MEMORY
  retval  =  (*cudaMemcpy3D_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * a1, cudaStream_t a2) {

  typedef cudaError_t (*cudaMemcpy3DAsync_p) (const struct cudaMemcpy3DParms *, cudaStream_t);
  static cudaMemcpy3DAsync_p cudaMemcpy3DAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy3DAsync_h == NULL)
	cudaMemcpy3DAsync_h = (cudaMemcpy3DAsync_p) dlsym(cudart_handle,"cudaMemcpy3DAsync"); 
    if (cudaMemcpy3DAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy3DAsync_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemGetInfo(size_t * a1, size_t * a2) {

  typedef cudaError_t (*cudaMemGetInfo_p) (size_t *, size_t *);
  static cudaMemGetInfo_p cudaMemGetInfo_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemGetInfo(size_t *, size_t *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemGetInfo_h == NULL)
	cudaMemGetInfo_h = (cudaMemGetInfo_p) dlsym(cudart_handle,"cudaMemGetInfo"); 
    if (cudaMemGetInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemGetInfo_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy(void * a1, const void * a2, size_t a3, enum cudaMemcpyKind a4) {

  typedef cudaError_t (*cudaMemcpy_p) (void *, const void *, size_t, enum cudaMemcpyKind);
  static cudaMemcpy_p cudaMemcpy_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy_h == NULL)
	cudaMemcpy_h = (cudaMemcpy_p) dlsym(cudart_handle,"cudaMemcpy"); 
    if (cudaMemcpy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_MEMORY
	tau_track_memory(a4, a3);
#endif //TRACK_MEMORY
  retval  =  (*cudaMemcpy_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToArray(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, enum cudaMemcpyKind a6) {

  typedef cudaError_t (*cudaMemcpyToArray_p) (struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind);
  static cudaMemcpyToArray_p cudaMemcpyToArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToArray(struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToArray_h == NULL)
	cudaMemcpyToArray_h = (cudaMemcpyToArray_p) dlsym(cudart_handle,"cudaMemcpyToArray"); 
    if (cudaMemcpyToArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_MEMORY
	tau_track_memory(a6, a5);
#endif //TRACK_MEMORY
  retval  =  (*cudaMemcpyToArray_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromArray(void * a1, const struct cudaArray * a2, size_t a3, size_t a4, size_t a5, enum cudaMemcpyKind a6) {

  typedef cudaError_t (*cudaMemcpyFromArray_p) (void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpyFromArray_p cudaMemcpyFromArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromArray(void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromArray_h == NULL)
	cudaMemcpyFromArray_h = (cudaMemcpyFromArray_p) dlsym(cudart_handle,"cudaMemcpyFromArray"); 
    if (cudaMemcpyFromArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TAU_TRACK_MEMORY
	tau_track_memory(a6, a5);
#endif // TAU_TRACK_MEMORY
  retval  =  (*cudaMemcpyFromArray_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyArrayToArray(struct cudaArray * a1, size_t a2, size_t a3, const struct cudaArray * a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8) {

  typedef cudaError_t (*cudaMemcpyArrayToArray_p) (struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpyArrayToArray_p cudaMemcpyArrayToArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyArrayToArray(struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyArrayToArray_h == NULL)
	cudaMemcpyArrayToArray_h = (cudaMemcpyArrayToArray_p) dlsym(cudart_handle,"cudaMemcpyArrayToArray"); 
    if (cudaMemcpyArrayToArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TAU_TRACK_MEMORY
	tau_track_memory(a8, a7);
#endif // TAU_TRACK_MEMORY
  retval  =  (*cudaMemcpyArrayToArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2D(void * a1, size_t a2, const void * a3, size_t a4, size_t a5, size_t a6, enum cudaMemcpyKind a7) {

  typedef cudaError_t (*cudaMemcpy2D_p) (void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpy2D_p cudaMemcpy2D_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2D_h == NULL)
	cudaMemcpy2D_h = (cudaMemcpy2D_p) dlsym(cudart_handle,"cudaMemcpy2D"); 
    if (cudaMemcpy2D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
#ifdef TRACK_MEMORY
	//Seg fault in UserEvent::~UserEvent when tracking this event
	//printf("array size: %d, by %dx%d.\n", sizeof(a3), a5, a6);
	tau_track_memory(a7, sizeof(a3)*a5*a6);
#endif //TRACK_MEMORY
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2D_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DToArray(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8) {

  typedef cudaError_t (*cudaMemcpy2DToArray_p) (struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpy2DToArray_p cudaMemcpy2DToArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DToArray(struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DToArray_h == NULL)
	cudaMemcpy2DToArray_h = (cudaMemcpy2DToArray_p) dlsym(cudart_handle,"cudaMemcpy2DToArray"); 
    if (cudaMemcpy2DToArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
#ifdef TRACK_MEMORY
	//Seg fault in UserEvent::~UserEvent when tracking this event
	tau_track_memory(a8, sizeof(a4)*a6*a7);
#endif //TRACK_MEMORY
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DToArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DFromArray(void * a1, size_t a2, const struct cudaArray * a3, size_t a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8) {

  typedef cudaError_t (*cudaMemcpy2DFromArray_p) (void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpy2DFromArray_p cudaMemcpy2DFromArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DFromArray(void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DFromArray_h == NULL)
	cudaMemcpy2DFromArray_h = (cudaMemcpy2DFromArray_p) dlsym(cudart_handle,"cudaMemcpy2DFromArray"); 
    if (cudaMemcpy2DFromArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DFromArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray * a1, size_t a2, size_t a3, const struct cudaArray * a4, size_t a5, size_t a6, size_t a7, size_t a8, enum cudaMemcpyKind a9) {

  typedef cudaError_t (*cudaMemcpy2DArrayToArray_p) (struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpy2DArrayToArray_p cudaMemcpy2DArrayToArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DArrayToArray_h == NULL)
	cudaMemcpy2DArrayToArray_h = (cudaMemcpy2DArrayToArray_p) dlsym(cudart_handle,"cudaMemcpy2DArrayToArray"); 
    if (cudaMemcpy2DArrayToArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DArrayToArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToSymbol(const char * a1, const void * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5) {

  typedef cudaError_t (*cudaMemcpyToSymbol_p) (const char *, const void *, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpyToSymbol_p cudaMemcpyToSymbol_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToSymbol_h == NULL)
	cudaMemcpyToSymbol_h = (cudaMemcpyToSymbol_p) dlsym(cudart_handle,"cudaMemcpyToSymbol"); 
    if (cudaMemcpyToSymbol_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_MEMORY
	tau_track_memory(a5, a3);
#endif //TRACK_MEMORY
  retval  =  (*cudaMemcpyToSymbol_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromSymbol(void * a1, const char * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5) {

  typedef cudaError_t (*cudaMemcpyFromSymbol_p) (void *, const char *, size_t, size_t, enum cudaMemcpyKind);
  static cudaMemcpyFromSymbol_p cudaMemcpyFromSymbol_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t, enum cudaMemcpyKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromSymbol_h == NULL)
	cudaMemcpyFromSymbol_h = (cudaMemcpyFromSymbol_p) dlsym(cudart_handle,"cudaMemcpyFromSymbol"); 
    if (cudaMemcpyFromSymbol_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_MEMORY
	tau_track_memory(a5, a3);
#endif //TRACK_MEMORY
  retval  =  (*cudaMemcpyFromSymbol_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyAsync(void * a1, const void * a2, size_t a3, enum cudaMemcpyKind a4, cudaStream_t a5) {

  typedef cudaError_t (*cudaMemcpyAsync_p) (void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpyAsync_p cudaMemcpyAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyAsync(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyAsync_h == NULL)
	cudaMemcpyAsync_h = (cudaMemcpyAsync_p) dlsym(cudart_handle,"cudaMemcpyAsync"); 
    if (cudaMemcpyAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_MEMORY
	tau_track_memory(a4, a3);
#endif //TRACK_MEMORY
  retval  =  (*cudaMemcpyAsync_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToArrayAsync(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, enum cudaMemcpyKind a6, cudaStream_t a7) {

  typedef cudaError_t (*cudaMemcpyToArrayAsync_p) (struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpyToArrayAsync_p cudaMemcpyToArrayAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToArrayAsync_h == NULL)
	cudaMemcpyToArrayAsync_h = (cudaMemcpyToArrayAsync_p) dlsym(cudart_handle,"cudaMemcpyToArrayAsync"); 
    if (cudaMemcpyToArrayAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TAU_TRACK_MEMORY
	tau_track_memory(a6, a5);
#endif // TAU_TRACK_MEMORY
  retval  =  (*cudaMemcpyToArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromArrayAsync(void * a1, const struct cudaArray * a2, size_t a3, size_t a4, size_t a5, enum cudaMemcpyKind a6, cudaStream_t a7) {

  typedef cudaError_t (*cudaMemcpyFromArrayAsync_p) (void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpyFromArrayAsync_p cudaMemcpyFromArrayAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromArrayAsync(void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromArrayAsync_h == NULL)
	cudaMemcpyFromArrayAsync_h = (cudaMemcpyFromArrayAsync_p) dlsym(cudart_handle,"cudaMemcpyFromArrayAsync"); 
    if (cudaMemcpyFromArrayAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TAU_TRACK_MEMORY
	tau_track_memory(a6, a5);
#endif // TAU_TRACK_MEMORY
  retval  =  (*cudaMemcpyFromArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DAsync(void * a1, size_t a2, const void * a3, size_t a4, size_t a5, size_t a6, enum cudaMemcpyKind a7, cudaStream_t a8) {

  typedef cudaError_t (*cudaMemcpy2DAsync_p) (void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpy2DAsync_p cudaMemcpy2DAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DAsync_h == NULL)
	cudaMemcpy2DAsync_h = (cudaMemcpy2DAsync_p) dlsym(cudart_handle,"cudaMemcpy2DAsync"); 
    if (cudaMemcpy2DAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8, cudaStream_t a9) {

  typedef cudaError_t (*cudaMemcpy2DToArrayAsync_p) (struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpy2DToArrayAsync_p cudaMemcpy2DToArrayAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DToArrayAsync_h == NULL)
	cudaMemcpy2DToArrayAsync_h = (cudaMemcpy2DToArrayAsync_p) dlsym(cudart_handle,"cudaMemcpy2DToArrayAsync"); 
    if (cudaMemcpy2DToArrayAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DToArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DFromArrayAsync(void * a1, size_t a2, const struct cudaArray * a3, size_t a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8, cudaStream_t a9) {

  typedef cudaError_t (*cudaMemcpy2DFromArrayAsync_p) (void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpy2DFromArrayAsync_p cudaMemcpy2DFromArrayAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DFromArrayAsync_h == NULL)
	cudaMemcpy2DFromArrayAsync_h = (cudaMemcpy2DFromArrayAsync_p) dlsym(cudart_handle,"cudaMemcpy2DFromArrayAsync"); 
    if (cudaMemcpy2DFromArrayAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DFromArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToSymbolAsync(const char * a1, const void * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5, cudaStream_t a6) {

  typedef cudaError_t (*cudaMemcpyToSymbolAsync_p) (const char *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpyToSymbolAsync_p cudaMemcpyToSymbolAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToSymbolAsync_h == NULL)
	cudaMemcpyToSymbolAsync_h = (cudaMemcpyToSymbolAsync_p) dlsym(cudart_handle,"cudaMemcpyToSymbolAsync"); 
    if (cudaMemcpyToSymbolAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TAU_TRACK_MEMORY
	tau_track_memory(a5, a3);
#endif // TAU_TRACK_MEMORY
  retval  =  (*cudaMemcpyToSymbolAsync_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromSymbolAsync(void * a1, const char * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5, cudaStream_t a6) {

  typedef cudaError_t (*cudaMemcpyFromSymbolAsync_p) (void *, const char *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
  static cudaMemcpyFromSymbolAsync_p cudaMemcpyFromSymbolAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromSymbolAsync_h == NULL)
	cudaMemcpyFromSymbolAsync_h = (cudaMemcpyFromSymbolAsync_p) dlsym(cudart_handle,"cudaMemcpyFromSymbolAsync"); 
    if (cudaMemcpyFromSymbolAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TAU_TRACK_MEMORY
	tau_track_memory(a5, a3);
#endif // TAU_TRACK_MEMORY
  retval  =  (*cudaMemcpyFromSymbolAsync_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset(void * a1, int a2, size_t a3) {

  typedef cudaError_t (*cudaMemset_p) (void *, int, size_t);
  static cudaMemset_p cudaMemset_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset(void *, int, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset_h == NULL)
	cudaMemset_h = (cudaMemset_p) dlsym(cudart_handle,"cudaMemset"); 
    if (cudaMemset_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset2D(void * a1, size_t a2, int a3, size_t a4, size_t a5) {

  typedef cudaError_t (*cudaMemset2D_p) (void *, size_t, int, size_t, size_t);
  static cudaMemset2D_p cudaMemset2D_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset2D_h == NULL)
	cudaMemset2D_h = (cudaMemset2D_p) dlsym(cudart_handle,"cudaMemset2D"); 
    if (cudaMemset2D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset2D_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset3D(struct cudaPitchedPtr a1, int a2, struct cudaExtent a3) {

  typedef cudaError_t (*cudaMemset3D_p) (struct cudaPitchedPtr, int, struct cudaExtent);
  static cudaMemset3D_p cudaMemset3D_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset3D(struct cudaPitchedPtr, int, struct cudaExtent) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset3D_h == NULL)
	cudaMemset3D_h = (cudaMemset3D_p) dlsym(cudart_handle,"cudaMemset3D"); 
    if (cudaMemset3D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset3D_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemsetAsync(void * a1, int a2, size_t a3, cudaStream_t a4) {

  typedef cudaError_t (*cudaMemsetAsync_p) (void *, int, size_t, cudaStream_t);
  static cudaMemsetAsync_p cudaMemsetAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemsetAsync_h == NULL)
	cudaMemsetAsync_h = (cudaMemsetAsync_p) dlsym(cudart_handle,"cudaMemsetAsync"); 
    if (cudaMemsetAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemsetAsync_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset2DAsync(void * a1, size_t a2, int a3, size_t a4, size_t a5, cudaStream_t a6) {

  typedef cudaError_t (*cudaMemset2DAsync_p) (void *, size_t, int, size_t, size_t, cudaStream_t);
  static cudaMemset2DAsync_p cudaMemset2DAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset2DAsync(void *, size_t, int, size_t, size_t, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset2DAsync_h == NULL)
	cudaMemset2DAsync_h = (cudaMemset2DAsync_p) dlsym(cudart_handle,"cudaMemset2DAsync"); 
    if (cudaMemset2DAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset2DAsync_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr a1, int a2, struct cudaExtent a3, cudaStream_t a4) {

  typedef cudaError_t (*cudaMemset3DAsync_p) (struct cudaPitchedPtr, int, struct cudaExtent, cudaStream_t);
  static cudaMemset3DAsync_p cudaMemset3DAsync_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr, int, struct cudaExtent, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset3DAsync_h == NULL)
	cudaMemset3DAsync_h = (cudaMemset3DAsync_p) dlsym(cudart_handle,"cudaMemset3DAsync"); 
    if (cudaMemset3DAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset3DAsync_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetSymbolAddress(void ** a1, const char * a2) {

  typedef cudaError_t (*cudaGetSymbolAddress_p) (void **, const char *);
  static cudaGetSymbolAddress_p cudaGetSymbolAddress_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetSymbolAddress(void **, const char *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetSymbolAddress_h == NULL)
	cudaGetSymbolAddress_h = (cudaGetSymbolAddress_p) dlsym(cudart_handle,"cudaGetSymbolAddress"); 
    if (cudaGetSymbolAddress_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetSymbolAddress_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetSymbolSize(size_t * a1, const char * a2) {

  typedef cudaError_t (*cudaGetSymbolSize_p) (size_t *, const char *);
  static cudaGetSymbolSize_p cudaGetSymbolSize_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetSymbolSize(size_t *, const char *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetSymbolSize_h == NULL)
	cudaGetSymbolSize_h = (cudaGetSymbolSize_p) dlsym(cudart_handle,"cudaGetSymbolSize"); 
    if (cudaGetSymbolSize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetSymbolSize_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
/*
cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t a1) {

  typedef cudaError_t (*cudaGraphicsUnregisterResource_p) (cudaGraphicsResource_t);
  static cudaGraphicsUnregisterResource_p cudaGraphicsUnregisterResource_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGraphicsUnregisterResource_h == NULL)
	cudaGraphicsUnregisterResource_h = (cudaGraphicsUnregisterResource_p) dlsym(cudart_handle,"cudaGraphicsUnregisterResource"); 
    if (cudaGraphicsUnregisterResource_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGraphicsUnregisterResource_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t a1, unsigned int a2) {

  typedef cudaError_t (*cudaGraphicsResourceSetMapFlags_p) (cudaGraphicsResource_t, unsigned int);
  static cudaGraphicsResourceSetMapFlags_p cudaGraphicsResourceSetMapFlags_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGraphicsResourceSetMapFlags_h == NULL)
	cudaGraphicsResourceSetMapFlags_h = (cudaGraphicsResourceSetMapFlags_p) dlsym(cudart_handle,"cudaGraphicsResourceSetMapFlags"); 
    if (cudaGraphicsResourceSetMapFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGraphicsResourceSetMapFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGraphicsMapResources(int a1, cudaGraphicsResource_t * a2, cudaStream_t a3) {

  typedef cudaError_t (*cudaGraphicsMapResources_p) (int, cudaGraphicsResource_t *, cudaStream_t);
  static cudaGraphicsMapResources_p cudaGraphicsMapResources_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGraphicsMapResources(int, cudaGraphicsResource_t *, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGraphicsMapResources_h == NULL)
	cudaGraphicsMapResources_h = (cudaGraphicsMapResources_p) dlsym(cudart_handle,"cudaGraphicsMapResources"); 
    if (cudaGraphicsMapResources_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGraphicsMapResources_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGraphicsUnmapResources(int a1, cudaGraphicsResource_t * a2, cudaStream_t a3) {

  typedef cudaError_t (*cudaGraphicsUnmapResources_p) (int, cudaGraphicsResource_t *, cudaStream_t);
  static cudaGraphicsUnmapResources_p cudaGraphicsUnmapResources_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGraphicsUnmapResources(int, cudaGraphicsResource_t *, cudaStream_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGraphicsUnmapResources_h == NULL)
	cudaGraphicsUnmapResources_h = (cudaGraphicsUnmapResources_p) dlsym(cudart_handle,"cudaGraphicsUnmapResources"); 
    if (cudaGraphicsUnmapResources_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGraphicsUnmapResources_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGraphicsResourceGetMappedPointer(void ** a1, size_t * a2, cudaGraphicsResource_t a3) {

  typedef cudaError_t (*cudaGraphicsResourceGetMappedPointer_p) (void **, size_t *, cudaGraphicsResource_t);
  static cudaGraphicsResourceGetMappedPointer_p cudaGraphicsResourceGetMappedPointer_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGraphicsResourceGetMappedPointer(void **, size_t *, cudaGraphicsResource_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGraphicsResourceGetMappedPointer_h == NULL)
	cudaGraphicsResourceGetMappedPointer_h = (cudaGraphicsResourceGetMappedPointer_p) dlsym(cudart_handle,"cudaGraphicsResourceGetMappedPointer"); 
    if (cudaGraphicsResourceGetMappedPointer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGraphicsResourceGetMappedPointer_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGraphicsSubResourceGetMappedArray(struct cudaArray ** a1, cudaGraphicsResource_t a2, unsigned int a3, unsigned int a4) {

  typedef cudaError_t (*cudaGraphicsSubResourceGetMappedArray_p) (struct cudaArray **, cudaGraphicsResource_t, unsigned int, unsigned int);
  static cudaGraphicsSubResourceGetMappedArray_p cudaGraphicsSubResourceGetMappedArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGraphicsSubResourceGetMappedArray(struct cudaArray **, cudaGraphicsResource_t, unsigned int, unsigned int) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGraphicsSubResourceGetMappedArray_h == NULL)
	cudaGraphicsSubResourceGetMappedArray_h = (cudaGraphicsSubResourceGetMappedArray_p) dlsym(cudart_handle,"cudaGraphicsSubResourceGetMappedArray"); 
    if (cudaGraphicsSubResourceGetMappedArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGraphicsSubResourceGetMappedArray_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * a1, const struct cudaArray * a2) {

  typedef cudaError_t (*cudaGetChannelDesc_p) (struct cudaChannelFormatDesc *, const struct cudaArray *);
  static cudaGetChannelDesc_p cudaGetChannelDesc_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *, const struct cudaArray *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetChannelDesc_h == NULL)
	cudaGetChannelDesc_h = (cudaGetChannelDesc_p) dlsym(cudart_handle,"cudaGetChannelDesc"); 
    if (cudaGetChannelDesc_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetChannelDesc_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
/* Do not instrument: Called before process is started.
struct cudaChannelFormatDesc cudaCreateChannelDesc(int a1, int a2, int a3, int a4, enum cudaChannelFormatKind a5) {

  typedef struct cudaChannelFormatDesc (*cudaCreateChannelDesc_p) (int, int, int, int, enum cudaChannelFormatKind);
  static cudaCreateChannelDesc_p cudaCreateChannelDesc_h = NULL;
  struct cudaChannelFormatDesc retval;
  TAU_PROFILE_TIMER(t,"struct cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, enum cudaChannelFormatKind) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaCreateChannelDesc_h == NULL)
	cudaCreateChannelDesc_h = (cudaCreateChannelDesc_p) dlsym(cudart_handle,"cudaCreateChannelDesc"); 
    if (cudaCreateChannelDesc_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaCreateChannelDesc_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaBindTexture(size_t * a1, const struct textureReference * a2, const void * a3, const struct cudaChannelFormatDesc * a4, size_t a5) {

  typedef cudaError_t (*cudaBindTexture_p) (size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t);
  static cudaBindTexture_p cudaBindTexture_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaBindTexture(size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaBindTexture_h == NULL)
	cudaBindTexture_h = (cudaBindTexture_p) dlsym(cudart_handle,"cudaBindTexture"); 
    if (cudaBindTexture_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaBindTexture_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaBindTexture2D(size_t * a1, const struct textureReference * a2, const void * a3, const struct cudaChannelFormatDesc * a4, size_t a5, size_t a6, size_t a7) {

  typedef cudaError_t (*cudaBindTexture2D_p) (size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t, size_t, size_t);
  static cudaBindTexture2D_p cudaBindTexture2D_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaBindTexture2D(size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t, size_t, size_t) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaBindTexture2D_h == NULL)
	cudaBindTexture2D_h = (cudaBindTexture2D_p) dlsym(cudart_handle,"cudaBindTexture2D"); 
    if (cudaBindTexture2D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaBindTexture2D_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaBindTextureToArray(const struct textureReference * a1, const struct cudaArray * a2, const struct cudaChannelFormatDesc * a3) {

  typedef cudaError_t (*cudaBindTextureToArray_p) (const struct textureReference *, const struct cudaArray *, const struct cudaChannelFormatDesc *);
  static cudaBindTextureToArray_p cudaBindTextureToArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaBindTextureToArray(const struct textureReference *, const struct cudaArray *, const struct cudaChannelFormatDesc *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaBindTextureToArray_h == NULL)
	cudaBindTextureToArray_h = (cudaBindTextureToArray_p) dlsym(cudart_handle,"cudaBindTextureToArray"); 
    if (cudaBindTextureToArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaBindTextureToArray_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaUnbindTexture(const struct textureReference * a1) {

  typedef cudaError_t (*cudaUnbindTexture_p) (const struct textureReference *);
  static cudaUnbindTexture_p cudaUnbindTexture_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaUnbindTexture(const struct textureReference *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaUnbindTexture_h == NULL)
	cudaUnbindTexture_h = (cudaUnbindTexture_p) dlsym(cudart_handle,"cudaUnbindTexture"); 
    if (cudaUnbindTexture_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaUnbindTexture_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetTextureAlignmentOffset(size_t * a1, const struct textureReference * a2) {

  typedef cudaError_t (*cudaGetTextureAlignmentOffset_p) (size_t *, const struct textureReference *);
  static cudaGetTextureAlignmentOffset_p cudaGetTextureAlignmentOffset_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetTextureAlignmentOffset(size_t *, const struct textureReference *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetTextureAlignmentOffset_h == NULL)
	cudaGetTextureAlignmentOffset_h = (cudaGetTextureAlignmentOffset_p) dlsym(cudart_handle,"cudaGetTextureAlignmentOffset"); 
    if (cudaGetTextureAlignmentOffset_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetTextureAlignmentOffset_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetTextureReference(const struct textureReference ** a1, const char * a2) {

  typedef cudaError_t (*cudaGetTextureReference_p) (const struct textureReference **, const char *);
  static cudaGetTextureReference_p cudaGetTextureReference_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetTextureReference(const struct textureReference **, const char *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetTextureReference_h == NULL)
	cudaGetTextureReference_h = (cudaGetTextureReference_p) dlsym(cudart_handle,"cudaGetTextureReference"); 
    if (cudaGetTextureReference_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetTextureReference_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaBindSurfaceToArray(const struct surfaceReference * a1, const struct cudaArray * a2, const struct cudaChannelFormatDesc * a3) {

  typedef cudaError_t (*cudaBindSurfaceToArray_p) (const struct surfaceReference *, const struct cudaArray *, const struct cudaChannelFormatDesc *);
  static cudaBindSurfaceToArray_p cudaBindSurfaceToArray_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *, const struct cudaArray *, const struct cudaChannelFormatDesc *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaBindSurfaceToArray_h == NULL)
	cudaBindSurfaceToArray_h = (cudaBindSurfaceToArray_p) dlsym(cudart_handle,"cudaBindSurfaceToArray"); 
    if (cudaBindSurfaceToArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaBindSurfaceToArray_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetSurfaceReference(const struct surfaceReference ** a1, const char * a2) {

  typedef cudaError_t (*cudaGetSurfaceReference_p) (const struct surfaceReference **, const char *);
  static cudaGetSurfaceReference_p cudaGetSurfaceReference_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetSurfaceReference(const struct surfaceReference **, const char *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetSurfaceReference_h == NULL)
	cudaGetSurfaceReference_h = (cudaGetSurfaceReference_p) dlsym(cudart_handle,"cudaGetSurfaceReference"); 
    if (cudaGetSurfaceReference_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetSurfaceReference_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaDriverGetVersion(int * a1) {

  typedef cudaError_t (*cudaDriverGetVersion_p) (int *);
  static cudaDriverGetVersion_p cudaDriverGetVersion_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaDriverGetVersion(int *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaDriverGetVersion_h == NULL)
	cudaDriverGetVersion_h = (cudaDriverGetVersion_p) dlsym(cudart_handle,"cudaDriverGetVersion"); 
    if (cudaDriverGetVersion_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaDriverGetVersion_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaRuntimeGetVersion(int * a1) {

  typedef cudaError_t (*cudaRuntimeGetVersion_p) (int *);
  static cudaRuntimeGetVersion_p cudaRuntimeGetVersion_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaRuntimeGetVersion(int *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaRuntimeGetVersion_h == NULL)
	cudaRuntimeGetVersion_h = (cudaRuntimeGetVersion_p) dlsym(cudart_handle,"cudaRuntimeGetVersion"); 
    if (cudaRuntimeGetVersion_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  // needed for HMPP 
	Tau_global_incr_insideTAU();
  Tau_create_top_level_timer_if_necessary();
  Tau_global_decr_insideTAU();

  TAU_PROFILE_SET_NODE(0);
  TAU_PROFILE_START(t);
  retval  =  (*cudaRuntimeGetVersion_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
/*
cudaError_t cudaGetExportTable(const void ** a1, const cudaUUID_t * a2) {

  typedef cudaError_t (*cudaGetExportTable_p) (const void **, const cudaUUID_t *);
  static cudaGetExportTable_p cudaGetExportTable_h = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetExportTable(const void **, const cudaUUID_t *) C", "", CUDART_API);
  if (cudart_handle == NULL) 
    cudart_handle = (void *) dlopen(cudart_orig_libname, RTLD_NOW); 

  if (cudart_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetExportTable_h == NULL)
	cudaGetExportTable_h = (cudaGetExportTable_p) dlsym(cudart_handle,"cudaGetExportTable"); 
    if (cudaGetExportTable_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetExportTable_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}*/
