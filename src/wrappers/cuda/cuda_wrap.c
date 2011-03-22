#include <cuda.h>
#include <Profile/Profiler.h>
#include <Profile/TauGpuAdapterCUDA.h>
#include <stdio.h>
#include <dlfcn.h>

#define TRACK_KERNEL

const char * tau_orig_libname = "libcuda.so";
static void *tau_handle = NULL;

CUresult cuInit(unsigned int a1) {

  typedef CUresult (*cuInit_p_h) (unsigned int);
  static cuInit_p_h cuInit_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuInit(unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuInit_h == NULL)
	cuInit_h = (cuInit_p_h) dlsym(tau_handle,"cuInit"); 
    if (cuInit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuInit_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDriverGetVersion(int * a1) {

  typedef CUresult (*cuDriverGetVersion_p_h) (int *);
  static cuDriverGetVersion_p_h cuDriverGetVersion_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDriverGetVersion(int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDriverGetVersion_h == NULL)
	cuDriverGetVersion_h = (cuDriverGetVersion_p_h) dlsym(tau_handle,"cuDriverGetVersion"); 
    if (cuDriverGetVersion_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDriverGetVersion_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceGet(CUdevice * a1, int a2) {

  typedef CUresult (*cuDeviceGet_p_h) (CUdevice *, int);
  static cuDeviceGet_p_h cuDeviceGet_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGet(CUdevice *, int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceGet_h == NULL)
	cuDeviceGet_h = (cuDeviceGet_p_h) dlsym(tau_handle,"cuDeviceGet"); 
    if (cuDeviceGet_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceGet_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceGetCount(int * a1) {

  typedef CUresult (*cuDeviceGetCount_p_h) (int *);
  static cuDeviceGetCount_p_h cuDeviceGetCount_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetCount(int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceGetCount_h == NULL)
	cuDeviceGetCount_h = (cuDeviceGetCount_p_h) dlsym(tau_handle,"cuDeviceGetCount"); 
    if (cuDeviceGetCount_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceGetCount_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceGetName(char * a1, int a2, CUdevice a3) {

  typedef CUresult (*cuDeviceGetName_p_h) (char *, int, CUdevice);
  static cuDeviceGetName_p_h cuDeviceGetName_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetName(char *, int, CUdevice) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceGetName_h == NULL)
	cuDeviceGetName_h = (cuDeviceGetName_p_h) dlsym(tau_handle,"cuDeviceGetName"); 
    if (cuDeviceGetName_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceGetName_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceComputeCapability(int * a1, int * a2, CUdevice a3) {

  typedef CUresult (*cuDeviceComputeCapability_p_h) (int *, int *, CUdevice);
  static cuDeviceComputeCapability_p_h cuDeviceComputeCapability_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceComputeCapability(int *, int *, CUdevice) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceComputeCapability_h == NULL)
	cuDeviceComputeCapability_h = (cuDeviceComputeCapability_p_h) dlsym(tau_handle,"cuDeviceComputeCapability"); 
    if (cuDeviceComputeCapability_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceComputeCapability_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceTotalMem_v2(size_t * a1, CUdevice a2) {

  typedef CUresult (*cuDeviceTotalMem_v2_p_h) (size_t *, CUdevice);
  static cuDeviceTotalMem_v2_p_h cuDeviceTotalMem_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceTotalMem_v2(size_t *, CUdevice) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceTotalMem_v2_h == NULL)
	cuDeviceTotalMem_v2_h = (cuDeviceTotalMem_v2_p_h) dlsym(tau_handle,"cuDeviceTotalMem_v2"); 
    if (cuDeviceTotalMem_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceTotalMem_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceGetProperties(CUdevprop * a1, CUdevice a2) {

  typedef CUresult (*cuDeviceGetProperties_p_h) (CUdevprop *, CUdevice);
  static cuDeviceGetProperties_p_h cuDeviceGetProperties_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetProperties(CUdevprop *, CUdevice) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceGetProperties_h == NULL)
	cuDeviceGetProperties_h = (cuDeviceGetProperties_p_h) dlsym(tau_handle,"cuDeviceGetProperties"); 
    if (cuDeviceGetProperties_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceGetProperties_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuDeviceGetAttribute(int * a1, CUdevice_attribute a2, CUdevice a3) {

  typedef CUresult (*cuDeviceGetAttribute_p_h) (int *, CUdevice_attribute, CUdevice);
  static cuDeviceGetAttribute_p_h cuDeviceGetAttribute_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuDeviceGetAttribute_h == NULL)
	cuDeviceGetAttribute_h = (cuDeviceGetAttribute_p_h) dlsym(tau_handle,"cuDeviceGetAttribute"); 
    if (cuDeviceGetAttribute_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuDeviceGetAttribute_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxCreate_v2(CUcontext * a1, unsigned int a2, CUdevice a3) {

  typedef CUresult (*cuCtxCreate_v2_p_h) (CUcontext *, unsigned int, CUdevice);
  static cuCtxCreate_v2_p_h cuCtxCreate_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxCreate_v2_h == NULL)
	cuCtxCreate_v2_h = (cuCtxCreate_v2_p_h) dlsym(tau_handle,"cuCtxCreate_v2"); 
    if (cuCtxCreate_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxCreate_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxDestroy(CUcontext a1) {

  typedef CUresult (*cuCtxDestroy_p_h) (CUcontext);
  static cuCtxDestroy_p_h cuCtxDestroy_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxDestroy(CUcontext) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxDestroy_h == NULL)
	cuCtxDestroy_h = (cuCtxDestroy_p_h) dlsym(tau_handle,"cuCtxDestroy"); 
    if (cuCtxDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxAttach(CUcontext * a1, unsigned int a2) {

  typedef CUresult (*cuCtxAttach_p_h) (CUcontext *, unsigned int);
  static cuCtxAttach_p_h cuCtxAttach_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxAttach(CUcontext *, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxAttach_h == NULL)
	cuCtxAttach_h = (cuCtxAttach_p_h) dlsym(tau_handle,"cuCtxAttach"); 
    if (cuCtxAttach_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxAttach_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxDetach(CUcontext a1) {

  typedef CUresult (*cuCtxDetach_p_h) (CUcontext);
  static cuCtxDetach_p_h cuCtxDetach_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxDetach(CUcontext) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxDetach_h == NULL)
	cuCtxDetach_h = (cuCtxDetach_p_h) dlsym(tau_handle,"cuCtxDetach"); 
    if (cuCtxDetach_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxDetach_h)( a1);
  TAU_PROFILE_STOP(t);

#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
	Tau_cuda_exit();
#endif
  }
  return retval;

}

CUresult cuCtxPushCurrent(CUcontext a1) {

  typedef CUresult (*cuCtxPushCurrent_p_h) (CUcontext);
  static cuCtxPushCurrent_p_h cuCtxPushCurrent_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxPushCurrent(CUcontext) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxPushCurrent_h == NULL)
	cuCtxPushCurrent_h = (cuCtxPushCurrent_p_h) dlsym(tau_handle,"cuCtxPushCurrent"); 
    if (cuCtxPushCurrent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxPushCurrent_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxPopCurrent(CUcontext * a1) {

  typedef CUresult (*cuCtxPopCurrent_p_h) (CUcontext *);
  static cuCtxPopCurrent_p_h cuCtxPopCurrent_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxPopCurrent(CUcontext *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxPopCurrent_h == NULL)
	cuCtxPopCurrent_h = (cuCtxPopCurrent_p_h) dlsym(tau_handle,"cuCtxPopCurrent"); 
    if (cuCtxPopCurrent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxPopCurrent_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxGetDevice(CUdevice * a1) {

  typedef CUresult (*cuCtxGetDevice_p_h) (CUdevice *);
  static cuCtxGetDevice_p_h cuCtxGetDevice_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetDevice(CUdevice *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxGetDevice_h == NULL)
	cuCtxGetDevice_h = (cuCtxGetDevice_p_h) dlsym(tau_handle,"cuCtxGetDevice"); 
    if (cuCtxGetDevice_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxGetDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxSynchronize() {

  typedef CUresult (*cuCtxSynchronize_p_h) ();
  static cuCtxSynchronize_p_h cuCtxSynchronize_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxSynchronize(void) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxSynchronize_h == NULL)
	cuCtxSynchronize_h = (cuCtxSynchronize_p_h) dlsym(tau_handle,"cuCtxSynchronize"); 
    if (cuCtxSynchronize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif 
  retval  =  (*cuCtxSynchronize_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxSetLimit(CUlimit a1, size_t a2) {

  typedef CUresult (*cuCtxSetLimit_p_h) (CUlimit, size_t);
  static cuCtxSetLimit_p_h cuCtxSetLimit_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxSetLimit(CUlimit, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxSetLimit_h == NULL)
	cuCtxSetLimit_h = (cuCtxSetLimit_p_h) dlsym(tau_handle,"cuCtxSetLimit"); 
    if (cuCtxSetLimit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxSetLimit_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxGetLimit(size_t * a1, CUlimit a2) {

  typedef CUresult (*cuCtxGetLimit_p_h) (size_t *, CUlimit);
  static cuCtxGetLimit_p_h cuCtxGetLimit_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetLimit(size_t *, CUlimit) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxGetLimit_h == NULL)
	cuCtxGetLimit_h = (cuCtxGetLimit_p_h) dlsym(tau_handle,"cuCtxGetLimit"); 
    if (cuCtxGetLimit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxGetLimit_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxGetCacheConfig(CUfunc_cache * a1) {

  typedef CUresult (*cuCtxGetCacheConfig_p_h) (CUfunc_cache *);
  static cuCtxGetCacheConfig_p_h cuCtxGetCacheConfig_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetCacheConfig(CUfunc_cache *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxGetCacheConfig_h == NULL)
	cuCtxGetCacheConfig_h = (cuCtxGetCacheConfig_p_h) dlsym(tau_handle,"cuCtxGetCacheConfig"); 
    if (cuCtxGetCacheConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxGetCacheConfig_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxSetCacheConfig(CUfunc_cache a1) {

  typedef CUresult (*cuCtxSetCacheConfig_p_h) (CUfunc_cache);
  static cuCtxSetCacheConfig_p_h cuCtxSetCacheConfig_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxSetCacheConfig(CUfunc_cache) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxSetCacheConfig_h == NULL)
	cuCtxSetCacheConfig_h = (cuCtxSetCacheConfig_p_h) dlsym(tau_handle,"cuCtxSetCacheConfig"); 
    if (cuCtxSetCacheConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxSetCacheConfig_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuCtxGetApiVersion(CUcontext a1, unsigned int * a2) {

  typedef CUresult (*cuCtxGetApiVersion_p_h) (CUcontext, unsigned int *);
  static cuCtxGetApiVersion_p_h cuCtxGetApiVersion_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetApiVersion(CUcontext, unsigned int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuCtxGetApiVersion_h == NULL)
	cuCtxGetApiVersion_h = (cuCtxGetApiVersion_p_h) dlsym(tau_handle,"cuCtxGetApiVersion"); 
    if (cuCtxGetApiVersion_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuCtxGetApiVersion_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleLoad(CUmodule * a1, const char * a2) {

  typedef CUresult (*cuModuleLoad_p_h) (CUmodule *, const char *);
  static cuModuleLoad_p_h cuModuleLoad_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoad(CUmodule *, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleLoad_h == NULL)
	cuModuleLoad_h = (cuModuleLoad_p_h) dlsym(tau_handle,"cuModuleLoad"); 
    if (cuModuleLoad_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleLoad_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleLoadData(CUmodule * a1, const void * a2) {

  typedef CUresult (*cuModuleLoadData_p_h) (CUmodule *, const void *);
  static cuModuleLoadData_p_h cuModuleLoadData_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoadData(CUmodule *, const void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleLoadData_h == NULL)
	cuModuleLoadData_h = (cuModuleLoadData_p_h) dlsym(tau_handle,"cuModuleLoadData"); 
    if (cuModuleLoadData_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleLoadData_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleLoadDataEx(CUmodule * a1, const void * a2, unsigned int a3, CUjit_option * a4, void ** a5) {

  typedef CUresult (*cuModuleLoadDataEx_p_h) (CUmodule *, const void *, unsigned int, CUjit_option *, void **);
  static cuModuleLoadDataEx_p_h cuModuleLoadDataEx_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoadDataEx(CUmodule *, const void *, unsigned int, CUjit_option *, void **) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleLoadDataEx_h == NULL)
	cuModuleLoadDataEx_h = (cuModuleLoadDataEx_p_h) dlsym(tau_handle,"cuModuleLoadDataEx"); 
    if (cuModuleLoadDataEx_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleLoadDataEx_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleLoadFatBinary(CUmodule * a1, const void * a2) {

  typedef CUresult (*cuModuleLoadFatBinary_p_h) (CUmodule *, const void *);
  static cuModuleLoadFatBinary_p_h cuModuleLoadFatBinary_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoadFatBinary(CUmodule *, const void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleLoadFatBinary_h == NULL)
	cuModuleLoadFatBinary_h = (cuModuleLoadFatBinary_p_h) dlsym(tau_handle,"cuModuleLoadFatBinary"); 
    if (cuModuleLoadFatBinary_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleLoadFatBinary_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleUnload(CUmodule a1) {

  typedef CUresult (*cuModuleUnload_p_h) (CUmodule);
  static cuModuleUnload_p_h cuModuleUnload_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleUnload(CUmodule) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleUnload_h == NULL)
	cuModuleUnload_h = (cuModuleUnload_p_h) dlsym(tau_handle,"cuModuleUnload"); 
    if (cuModuleUnload_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleUnload_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleGetFunction(CUfunction * a1, CUmodule a2, const char * a3) {

  typedef CUresult (*cuModuleGetFunction_p_h) (CUfunction *, CUmodule, const char *);
  static cuModuleGetFunction_p_h cuModuleGetFunction_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleGetFunction_h == NULL)
	cuModuleGetFunction_h = (cuModuleGetFunction_p_h) dlsym(tau_handle,"cuModuleGetFunction"); 
    if (cuModuleGetFunction_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleGetFunction_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleGetGlobal_v2(CUdeviceptr * a1, size_t * a2, CUmodule a3, const char * a4) {

  typedef CUresult (*cuModuleGetGlobal_v2_p_h) (CUdeviceptr *, size_t *, CUmodule, const char *);
  static cuModuleGetGlobal_v2_p_h cuModuleGetGlobal_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetGlobal_v2(CUdeviceptr *, size_t *, CUmodule, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleGetGlobal_v2_h == NULL)
	cuModuleGetGlobal_v2_h = (cuModuleGetGlobal_v2_p_h) dlsym(tau_handle,"cuModuleGetGlobal_v2"); 
    if (cuModuleGetGlobal_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleGetGlobal_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleGetTexRef(CUtexref * a1, CUmodule a2, const char * a3) {

  typedef CUresult (*cuModuleGetTexRef_p_h) (CUtexref *, CUmodule, const char *);
  static cuModuleGetTexRef_p_h cuModuleGetTexRef_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetTexRef(CUtexref *, CUmodule, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleGetTexRef_h == NULL)
	cuModuleGetTexRef_h = (cuModuleGetTexRef_p_h) dlsym(tau_handle,"cuModuleGetTexRef"); 
    if (cuModuleGetTexRef_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleGetTexRef_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuModuleGetSurfRef(CUsurfref * a1, CUmodule a2, const char * a3) {

  typedef CUresult (*cuModuleGetSurfRef_p_h) (CUsurfref *, CUmodule, const char *);
  static cuModuleGetSurfRef_p_h cuModuleGetSurfRef_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetSurfRef(CUsurfref *, CUmodule, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuModuleGetSurfRef_h == NULL)
	cuModuleGetSurfRef_h = (cuModuleGetSurfRef_p_h) dlsym(tau_handle,"cuModuleGetSurfRef"); 
    if (cuModuleGetSurfRef_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuModuleGetSurfRef_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemGetInfo_v2(size_t * a1, size_t * a2) {

  typedef CUresult (*cuMemGetInfo_v2_p_h) (size_t *, size_t *);
  static cuMemGetInfo_v2_p_h cuMemGetInfo_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemGetInfo_v2(size_t *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemGetInfo_v2_h == NULL)
	cuMemGetInfo_v2_h = (cuMemGetInfo_v2_p_h) dlsym(tau_handle,"cuMemGetInfo_v2"); 
    if (cuMemGetInfo_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemGetInfo_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemAlloc_v2(CUdeviceptr * a1, size_t a2) {

  typedef CUresult (*cuMemAlloc_v2_p_h) (CUdeviceptr *, size_t);
  static cuMemAlloc_v2_p_h cuMemAlloc_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemAlloc_v2(CUdeviceptr *, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemAlloc_v2_h == NULL)
	cuMemAlloc_v2_h = (cuMemAlloc_v2_p_h) dlsym(tau_handle,"cuMemAlloc_v2"); 
    if (cuMemAlloc_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemAlloc_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemAllocPitch_v2(CUdeviceptr * a1, size_t * a2, size_t a3, size_t a4, unsigned int a5) {

  typedef CUresult (*cuMemAllocPitch_v2_p_h) (CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
  static cuMemAllocPitch_v2_p_h cuMemAllocPitch_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemAllocPitch_v2(CUdeviceptr *, size_t *, size_t, size_t, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemAllocPitch_v2_h == NULL)
	cuMemAllocPitch_v2_h = (cuMemAllocPitch_v2_p_h) dlsym(tau_handle,"cuMemAllocPitch_v2"); 
    if (cuMemAllocPitch_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemAllocPitch_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemFree_v2(CUdeviceptr a1) {

  typedef CUresult (*cuMemFree_v2_p_h) (CUdeviceptr);
  static cuMemFree_v2_p_h cuMemFree_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemFree_v2(CUdeviceptr) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemFree_v2_h == NULL)
	cuMemFree_v2_h = (cuMemFree_v2_p_h) dlsym(tau_handle,"cuMemFree_v2"); 
    if (cuMemFree_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemFree_v2_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemGetAddressRange_v2(CUdeviceptr * a1, size_t * a2, CUdeviceptr a3) {

  typedef CUresult (*cuMemGetAddressRange_v2_p_h) (CUdeviceptr *, size_t *, CUdeviceptr);
  static cuMemGetAddressRange_v2_p_h cuMemGetAddressRange_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemGetAddressRange_v2(CUdeviceptr *, size_t *, CUdeviceptr) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemGetAddressRange_v2_h == NULL)
	cuMemGetAddressRange_v2_h = (cuMemGetAddressRange_v2_p_h) dlsym(tau_handle,"cuMemGetAddressRange_v2"); 
    if (cuMemGetAddressRange_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemGetAddressRange_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemAllocHost_v2(void ** a1, size_t a2) {

  typedef CUresult (*cuMemAllocHost_v2_p_h) (void **, size_t);
  static cuMemAllocHost_v2_p_h cuMemAllocHost_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemAllocHost_v2(void **, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemAllocHost_v2_h == NULL)
	cuMemAllocHost_v2_h = (cuMemAllocHost_v2_p_h) dlsym(tau_handle,"cuMemAllocHost_v2"); 
    if (cuMemAllocHost_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemAllocHost_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemFreeHost(void * a1) {

  typedef CUresult (*cuMemFreeHost_p_h) (void *);
  static cuMemFreeHost_p_h cuMemFreeHost_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemFreeHost(void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemFreeHost_h == NULL)
	cuMemFreeHost_h = (cuMemFreeHost_p_h) dlsym(tau_handle,"cuMemFreeHost"); 
    if (cuMemFreeHost_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemFreeHost_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemHostAlloc(void ** a1, size_t a2, unsigned int a3) {

  typedef CUresult (*cuMemHostAlloc_p_h) (void **, size_t, unsigned int);
  static cuMemHostAlloc_p_h cuMemHostAlloc_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemHostAlloc(void **, size_t, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemHostAlloc_h == NULL)
	cuMemHostAlloc_h = (cuMemHostAlloc_p_h) dlsym(tau_handle,"cuMemHostAlloc"); 
    if (cuMemHostAlloc_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemHostAlloc_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr * a1, void * a2, unsigned int a3) {

  typedef CUresult (*cuMemHostGetDevicePointer_v2_p_h) (CUdeviceptr *, void *, unsigned int);
  static cuMemHostGetDevicePointer_v2_p_h cuMemHostGetDevicePointer_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *, void *, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemHostGetDevicePointer_v2_h == NULL)
	cuMemHostGetDevicePointer_v2_h = (cuMemHostGetDevicePointer_v2_p_h) dlsym(tau_handle,"cuMemHostGetDevicePointer_v2"); 
    if (cuMemHostGetDevicePointer_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemHostGetDevicePointer_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemHostGetFlags(unsigned int * a1, void * a2) {

  typedef CUresult (*cuMemHostGetFlags_p_h) (unsigned int *, void *);
  static cuMemHostGetFlags_p_h cuMemHostGetFlags_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemHostGetFlags(unsigned int *, void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemHostGetFlags_h == NULL)
	cuMemHostGetFlags_h = (cuMemHostGetFlags_p_h) dlsym(tau_handle,"cuMemHostGetFlags"); 
    if (cuMemHostGetFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemHostGetFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyHtoD_v2(CUdeviceptr a1, const void * a2, size_t a3) {

  typedef CUresult (*cuMemcpyHtoD_v2_p_h) (CUdeviceptr, const void *, size_t);
  static cuMemcpyHtoD_v2_p_h cuMemcpyHtoD_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyHtoD_v2_h == NULL)
	cuMemcpyHtoD_v2_h = (cuMemcpyHtoD_v2_p_h) dlsym(tau_handle,"cuMemcpyHtoD_v2"); 
    if (cuMemcpyHtoD_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyHtoD_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyDtoH_v2(void * a1, CUdeviceptr a2, size_t a3) {

  typedef CUresult (*cuMemcpyDtoH_v2_p_h) (void *, CUdeviceptr, size_t);
  static cuMemcpyDtoH_v2_p_h cuMemcpyDtoH_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyDtoH_v2_h == NULL)
	cuMemcpyDtoH_v2_h = (cuMemcpyDtoH_v2_p_h) dlsym(tau_handle,"cuMemcpyDtoH_v2"); 
    if (cuMemcpyDtoH_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyDtoH_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyDtoD_v2(CUdeviceptr a1, CUdeviceptr a2, size_t a3) {

  typedef CUresult (*cuMemcpyDtoD_v2_p_h) (CUdeviceptr, CUdeviceptr, size_t);
  static cuMemcpyDtoD_v2_p_h cuMemcpyDtoD_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyDtoD_v2_h == NULL)
	cuMemcpyDtoD_v2_h = (cuMemcpyDtoD_v2_p_h) dlsym(tau_handle,"cuMemcpyDtoD_v2"); 
    if (cuMemcpyDtoD_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyDtoD_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyDtoA_v2(CUarray a1, size_t a2, CUdeviceptr a3, size_t a4) {

  typedef CUresult (*cuMemcpyDtoA_v2_p_h) (CUarray, size_t, CUdeviceptr, size_t);
  static cuMemcpyDtoA_v2_p_h cuMemcpyDtoA_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoA_v2(CUarray, size_t, CUdeviceptr, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyDtoA_v2_h == NULL)
	cuMemcpyDtoA_v2_h = (cuMemcpyDtoA_v2_p_h) dlsym(tau_handle,"cuMemcpyDtoA_v2"); 
    if (cuMemcpyDtoA_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyDtoA_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyAtoD_v2(CUdeviceptr a1, CUarray a2, size_t a3, size_t a4) {

  typedef CUresult (*cuMemcpyAtoD_v2_p_h) (CUdeviceptr, CUarray, size_t, size_t);
  static cuMemcpyAtoD_v2_p_h cuMemcpyAtoD_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoD_v2(CUdeviceptr, CUarray, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyAtoD_v2_h == NULL)
	cuMemcpyAtoD_v2_h = (cuMemcpyAtoD_v2_p_h) dlsym(tau_handle,"cuMemcpyAtoD_v2"); 
    if (cuMemcpyAtoD_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyAtoD_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyHtoA_v2(CUarray a1, size_t a2, const void * a3, size_t a4) {

  typedef CUresult (*cuMemcpyHtoA_v2_p_h) (CUarray, size_t, const void *, size_t);
  static cuMemcpyHtoA_v2_p_h cuMemcpyHtoA_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoA_v2(CUarray, size_t, const void *, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyHtoA_v2_h == NULL)
	cuMemcpyHtoA_v2_h = (cuMemcpyHtoA_v2_p_h) dlsym(tau_handle,"cuMemcpyHtoA_v2"); 
    if (cuMemcpyHtoA_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyHtoA_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyAtoH_v2(void * a1, CUarray a2, size_t a3, size_t a4) {

  typedef CUresult (*cuMemcpyAtoH_v2_p_h) (void *, CUarray, size_t, size_t);
  static cuMemcpyAtoH_v2_p_h cuMemcpyAtoH_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoH_v2(void *, CUarray, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyAtoH_v2_h == NULL)
	cuMemcpyAtoH_v2_h = (cuMemcpyAtoH_v2_p_h) dlsym(tau_handle,"cuMemcpyAtoH_v2"); 
    if (cuMemcpyAtoH_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyAtoH_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyAtoA_v2(CUarray a1, size_t a2, CUarray a3, size_t a4, size_t a5) {

  typedef CUresult (*cuMemcpyAtoA_v2_p_h) (CUarray, size_t, CUarray, size_t, size_t);
  static cuMemcpyAtoA_v2_p_h cuMemcpyAtoA_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoA_v2(CUarray, size_t, CUarray, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyAtoA_v2_h == NULL)
	cuMemcpyAtoA_v2_h = (cuMemcpyAtoA_v2_p_h) dlsym(tau_handle,"cuMemcpyAtoA_v2"); 
    if (cuMemcpyAtoA_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyAtoA_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D * a1) {

  typedef CUresult (*cuMemcpy2D_v2_p_h) (const CUDA_MEMCPY2D *);
  static cuMemcpy2D_v2_p_h cuMemcpy2D_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpy2D_v2_h == NULL)
	cuMemcpy2D_v2_h = (cuMemcpy2D_v2_p_h) dlsym(tau_handle,"cuMemcpy2D_v2"); 
    if (cuMemcpy2D_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpy2D_v2_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * a1) {

  typedef CUresult (*cuMemcpy2DUnaligned_v2_p_h) (const CUDA_MEMCPY2D *);
  static cuMemcpy2DUnaligned_v2_p_h cuMemcpy2DUnaligned_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpy2DUnaligned_v2_h == NULL)
	cuMemcpy2DUnaligned_v2_h = (cuMemcpy2DUnaligned_v2_p_h) dlsym(tau_handle,"cuMemcpy2DUnaligned_v2"); 
    if (cuMemcpy2DUnaligned_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpy2DUnaligned_v2_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D * a1) {

  typedef CUresult (*cuMemcpy3D_v2_p_h) (const CUDA_MEMCPY3D *);
  static cuMemcpy3D_v2_p_h cuMemcpy3D_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpy3D_v2_h == NULL)
	cuMemcpy3D_v2_h = (cuMemcpy3D_v2_p_h) dlsym(tau_handle,"cuMemcpy3D_v2"); 
    if (cuMemcpy3D_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpy3D_v2_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr a1, const void * a2, size_t a3, CUstream a4) {

  typedef CUresult (*cuMemcpyHtoDAsync_v2_p_h) (CUdeviceptr, const void *, size_t, CUstream);
  static cuMemcpyHtoDAsync_v2_p_h cuMemcpyHtoDAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr, const void *, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyHtoDAsync_v2_h == NULL)
	cuMemcpyHtoDAsync_v2_h = (cuMemcpyHtoDAsync_v2_p_h) dlsym(tau_handle,"cuMemcpyHtoDAsync_v2"); 
    if (cuMemcpyHtoDAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyHtoDAsync_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyDtoHAsync_v2(void * a1, CUdeviceptr a2, size_t a3, CUstream a4) {

  typedef CUresult (*cuMemcpyDtoHAsync_v2_p_h) (void *, CUdeviceptr, size_t, CUstream);
  static cuMemcpyDtoHAsync_v2_p_h cuMemcpyDtoHAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoHAsync_v2(void *, CUdeviceptr, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyDtoHAsync_v2_h == NULL)
	cuMemcpyDtoHAsync_v2_h = (cuMemcpyDtoHAsync_v2_p_h) dlsym(tau_handle,"cuMemcpyDtoHAsync_v2"); 
    if (cuMemcpyDtoHAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyDtoHAsync_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr a1, CUdeviceptr a2, size_t a3, CUstream a4) {

  typedef CUresult (*cuMemcpyDtoDAsync_v2_p_h) (CUdeviceptr, CUdeviceptr, size_t, CUstream);
  static cuMemcpyDtoDAsync_v2_p_h cuMemcpyDtoDAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyDtoDAsync_v2_h == NULL)
	cuMemcpyDtoDAsync_v2_h = (cuMemcpyDtoDAsync_v2_p_h) dlsym(tau_handle,"cuMemcpyDtoDAsync_v2"); 
    if (cuMemcpyDtoDAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyDtoDAsync_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyHtoAAsync_v2(CUarray a1, size_t a2, const void * a3, size_t a4, CUstream a5) {

  typedef CUresult (*cuMemcpyHtoAAsync_v2_p_h) (CUarray, size_t, const void *, size_t, CUstream);
  static cuMemcpyHtoAAsync_v2_p_h cuMemcpyHtoAAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoAAsync_v2(CUarray, size_t, const void *, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyHtoAAsync_v2_h == NULL)
	cuMemcpyHtoAAsync_v2_h = (cuMemcpyHtoAAsync_v2_p_h) dlsym(tau_handle,"cuMemcpyHtoAAsync_v2"); 
    if (cuMemcpyHtoAAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyHtoAAsync_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpyAtoHAsync_v2(void * a1, CUarray a2, size_t a3, size_t a4, CUstream a5) {

  typedef CUresult (*cuMemcpyAtoHAsync_v2_p_h) (void *, CUarray, size_t, size_t, CUstream);
  static cuMemcpyAtoHAsync_v2_p_h cuMemcpyAtoHAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoHAsync_v2(void *, CUarray, size_t, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpyAtoHAsync_v2_h == NULL)
	cuMemcpyAtoHAsync_v2_h = (cuMemcpyAtoHAsync_v2_p_h) dlsym(tau_handle,"cuMemcpyAtoHAsync_v2"); 
    if (cuMemcpyAtoHAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpyAtoHAsync_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * a1, CUstream a2) {

  typedef CUresult (*cuMemcpy2DAsync_v2_p_h) (const CUDA_MEMCPY2D *, CUstream);
  static cuMemcpy2DAsync_v2_p_h cuMemcpy2DAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpy2DAsync_v2_h == NULL)
	cuMemcpy2DAsync_v2_h = (cuMemcpy2DAsync_v2_p_h) dlsym(tau_handle,"cuMemcpy2DAsync_v2"); 
    if (cuMemcpy2DAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpy2DAsync_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * a1, CUstream a2) {

  typedef CUresult (*cuMemcpy3DAsync_v2_p_h) (const CUDA_MEMCPY3D *, CUstream);
  static cuMemcpy3DAsync_v2_p_h cuMemcpy3DAsync_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemcpy3DAsync_v2_h == NULL)
	cuMemcpy3DAsync_v2_h = (cuMemcpy3DAsync_v2_p_h) dlsym(tau_handle,"cuMemcpy3DAsync_v2"); 
    if (cuMemcpy3DAsync_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemcpy3DAsync_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD8_v2(CUdeviceptr a1, unsigned char a2, size_t a3) {

  typedef CUresult (*cuMemsetD8_v2_p_h) (CUdeviceptr, unsigned char, size_t);
  static cuMemsetD8_v2_p_h cuMemsetD8_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD8_v2(CUdeviceptr, unsigned char, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD8_v2_h == NULL)
	cuMemsetD8_v2_h = (cuMemsetD8_v2_p_h) dlsym(tau_handle,"cuMemsetD8_v2"); 
    if (cuMemsetD8_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD8_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD16_v2(CUdeviceptr a1, unsigned short a2, size_t a3) {

  typedef CUresult (*cuMemsetD16_v2_p_h) (CUdeviceptr, unsigned short, size_t);
  static cuMemsetD16_v2_p_h cuMemsetD16_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD16_v2(CUdeviceptr, unsigned short, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD16_v2_h == NULL)
	cuMemsetD16_v2_h = (cuMemsetD16_v2_p_h) dlsym(tau_handle,"cuMemsetD16_v2"); 
    if (cuMemsetD16_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD16_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD32_v2(CUdeviceptr a1, unsigned int a2, size_t a3) {

  typedef CUresult (*cuMemsetD32_v2_p_h) (CUdeviceptr, unsigned int, size_t);
  static cuMemsetD32_v2_p_h cuMemsetD32_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD32_v2(CUdeviceptr, unsigned int, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD32_v2_h == NULL)
	cuMemsetD32_v2_h = (cuMemsetD32_v2_p_h) dlsym(tau_handle,"cuMemsetD32_v2"); 
    if (cuMemsetD32_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD32_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD2D8_v2(CUdeviceptr a1, size_t a2, unsigned char a3, size_t a4, size_t a5) {

  typedef CUresult (*cuMemsetD2D8_v2_p_h) (CUdeviceptr, size_t, unsigned char, size_t, size_t);
  static cuMemsetD2D8_v2_p_h cuMemsetD2D8_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D8_v2(CUdeviceptr, size_t, unsigned char, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD2D8_v2_h == NULL)
	cuMemsetD2D8_v2_h = (cuMemsetD2D8_v2_p_h) dlsym(tau_handle,"cuMemsetD2D8_v2"); 
    if (cuMemsetD2D8_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD2D8_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD2D16_v2(CUdeviceptr a1, size_t a2, unsigned short a3, size_t a4, size_t a5) {

  typedef CUresult (*cuMemsetD2D16_v2_p_h) (CUdeviceptr, size_t, unsigned short, size_t, size_t);
  static cuMemsetD2D16_v2_p_h cuMemsetD2D16_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D16_v2(CUdeviceptr, size_t, unsigned short, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD2D16_v2_h == NULL)
	cuMemsetD2D16_v2_h = (cuMemsetD2D16_v2_p_h) dlsym(tau_handle,"cuMemsetD2D16_v2"); 
    if (cuMemsetD2D16_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD2D16_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD2D32_v2(CUdeviceptr a1, size_t a2, unsigned int a3, size_t a4, size_t a5) {

  typedef CUresult (*cuMemsetD2D32_v2_p_h) (CUdeviceptr, size_t, unsigned int, size_t, size_t);
  static cuMemsetD2D32_v2_p_h cuMemsetD2D32_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D32_v2(CUdeviceptr, size_t, unsigned int, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD2D32_v2_h == NULL)
	cuMemsetD2D32_v2_h = (cuMemsetD2D32_v2_p_h) dlsym(tau_handle,"cuMemsetD2D32_v2"); 
    if (cuMemsetD2D32_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD2D32_v2_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD8Async(CUdeviceptr a1, unsigned char a2, size_t a3, CUstream a4) {

  typedef CUresult (*cuMemsetD8Async_p_h) (CUdeviceptr, unsigned char, size_t, CUstream);
  static cuMemsetD8Async_p_h cuMemsetD8Async_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD8Async_h == NULL)
	cuMemsetD8Async_h = (cuMemsetD8Async_p_h) dlsym(tau_handle,"cuMemsetD8Async"); 
    if (cuMemsetD8Async_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD8Async_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD16Async(CUdeviceptr a1, unsigned short a2, size_t a3, CUstream a4) {

  typedef CUresult (*cuMemsetD16Async_p_h) (CUdeviceptr, unsigned short, size_t, CUstream);
  static cuMemsetD16Async_p_h cuMemsetD16Async_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD16Async_h == NULL)
	cuMemsetD16Async_h = (cuMemsetD16Async_p_h) dlsym(tau_handle,"cuMemsetD16Async"); 
    if (cuMemsetD16Async_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD16Async_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD32Async(CUdeviceptr a1, unsigned int a2, size_t a3, CUstream a4) {

  typedef CUresult (*cuMemsetD32Async_p_h) (CUdeviceptr, unsigned int, size_t, CUstream);
  static cuMemsetD32Async_p_h cuMemsetD32Async_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD32Async(CUdeviceptr, unsigned int, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD32Async_h == NULL)
	cuMemsetD32Async_h = (cuMemsetD32Async_p_h) dlsym(tau_handle,"cuMemsetD32Async"); 
    if (cuMemsetD32Async_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD32Async_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD2D8Async(CUdeviceptr a1, size_t a2, unsigned char a3, size_t a4, size_t a5, CUstream a6) {

  typedef CUresult (*cuMemsetD2D8Async_p_h) (CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream);
  static cuMemsetD2D8Async_p_h cuMemsetD2D8Async_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D8Async(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD2D8Async_h == NULL)
	cuMemsetD2D8Async_h = (cuMemsetD2D8Async_p_h) dlsym(tau_handle,"cuMemsetD2D8Async"); 
    if (cuMemsetD2D8Async_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD2D8Async_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD2D16Async(CUdeviceptr a1, size_t a2, unsigned short a3, size_t a4, size_t a5, CUstream a6) {

  typedef CUresult (*cuMemsetD2D16Async_p_h) (CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream);
  static cuMemsetD2D16Async_p_h cuMemsetD2D16Async_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D16Async(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD2D16Async_h == NULL)
	cuMemsetD2D16Async_h = (cuMemsetD2D16Async_p_h) dlsym(tau_handle,"cuMemsetD2D16Async"); 
    if (cuMemsetD2D16Async_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD2D16Async_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuMemsetD2D32Async(CUdeviceptr a1, size_t a2, unsigned int a3, size_t a4, size_t a5, CUstream a6) {

  typedef CUresult (*cuMemsetD2D32Async_p_h) (CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream);
  static cuMemsetD2D32Async_p_h cuMemsetD2D32Async_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuMemsetD2D32Async_h == NULL)
	cuMemsetD2D32Async_h = (cuMemsetD2D32Async_p_h) dlsym(tau_handle,"cuMemsetD2D32Async"); 
    if (cuMemsetD2D32Async_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuMemsetD2D32Async_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuArrayCreate_v2(CUarray * a1, const CUDA_ARRAY_DESCRIPTOR * a2) {

  typedef CUresult (*cuArrayCreate_v2_p_h) (CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
  static cuArrayCreate_v2_p_h cuArrayCreate_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArrayCreate_v2(CUarray *, const CUDA_ARRAY_DESCRIPTOR *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuArrayCreate_v2_h == NULL)
	cuArrayCreate_v2_h = (cuArrayCreate_v2_p_h) dlsym(tau_handle,"cuArrayCreate_v2"); 
    if (cuArrayCreate_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuArrayCreate_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * a1, CUarray a2) {

  typedef CUresult (*cuArrayGetDescriptor_v2_p_h) (CUDA_ARRAY_DESCRIPTOR *, CUarray);
  static cuArrayGetDescriptor_v2_p_h cuArrayGetDescriptor_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *, CUarray) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuArrayGetDescriptor_v2_h == NULL)
	cuArrayGetDescriptor_v2_h = (cuArrayGetDescriptor_v2_p_h) dlsym(tau_handle,"cuArrayGetDescriptor_v2"); 
    if (cuArrayGetDescriptor_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuArrayGetDescriptor_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuArrayDestroy(CUarray a1) {

  typedef CUresult (*cuArrayDestroy_p_h) (CUarray);
  static cuArrayDestroy_p_h cuArrayDestroy_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArrayDestroy(CUarray) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuArrayDestroy_h == NULL)
	cuArrayDestroy_h = (cuArrayDestroy_p_h) dlsym(tau_handle,"cuArrayDestroy"); 
    if (cuArrayDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuArrayDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuArray3DCreate_v2(CUarray * a1, const CUDA_ARRAY3D_DESCRIPTOR * a2) {

  typedef CUresult (*cuArray3DCreate_v2_p_h) (CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
  static cuArray3DCreate_v2_p_h cuArray3DCreate_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArray3DCreate_v2(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuArray3DCreate_v2_h == NULL)
	cuArray3DCreate_v2_h = (cuArray3DCreate_v2_p_h) dlsym(tau_handle,"cuArray3DCreate_v2"); 
    if (cuArray3DCreate_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuArray3DCreate_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * a1, CUarray a2) {

  typedef CUresult (*cuArray3DGetDescriptor_v2_p_h) (CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
  static cuArray3DGetDescriptor_v2_p_h cuArray3DGetDescriptor_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *, CUarray) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuArray3DGetDescriptor_v2_h == NULL)
	cuArray3DGetDescriptor_v2_h = (cuArray3DGetDescriptor_v2_p_h) dlsym(tau_handle,"cuArray3DGetDescriptor_v2"); 
    if (cuArray3DGetDescriptor_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuArray3DGetDescriptor_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuStreamCreate(CUstream * a1, unsigned int a2) {

  typedef CUresult (*cuStreamCreate_p_h) (CUstream *, unsigned int);
  static cuStreamCreate_p_h cuStreamCreate_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamCreate(CUstream *, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuStreamCreate_h == NULL)
	cuStreamCreate_h = (cuStreamCreate_p_h) dlsym(tau_handle,"cuStreamCreate"); 
    if (cuStreamCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuStreamCreate_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuStreamWaitEvent(CUstream a1, CUevent a2, unsigned int a3) {

  typedef CUresult (*cuStreamWaitEvent_p_h) (CUstream, CUevent, unsigned int);
  static cuStreamWaitEvent_p_h cuStreamWaitEvent_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuStreamWaitEvent_h == NULL)
	cuStreamWaitEvent_h = (cuStreamWaitEvent_p_h) dlsym(tau_handle,"cuStreamWaitEvent"); 
    if (cuStreamWaitEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuStreamWaitEvent_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuStreamQuery(CUstream a1) {

  typedef CUresult (*cuStreamQuery_p_h) (CUstream);
  static cuStreamQuery_p_h cuStreamQuery_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamQuery(CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuStreamQuery_h == NULL)
	cuStreamQuery_h = (cuStreamQuery_p_h) dlsym(tau_handle,"cuStreamQuery"); 
    if (cuStreamQuery_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuStreamQuery_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuStreamSynchronize(CUstream a1) {

  typedef CUresult (*cuStreamSynchronize_p_h) (CUstream);
  static cuStreamSynchronize_p_h cuStreamSynchronize_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamSynchronize(CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuStreamSynchronize_h == NULL)
	cuStreamSynchronize_h = (cuStreamSynchronize_p_h) dlsym(tau_handle,"cuStreamSynchronize"); 
    if (cuStreamSynchronize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif 
  retval  =  (*cuStreamSynchronize_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuStreamDestroy(CUstream a1) {

  typedef CUresult (*cuStreamDestroy_p_h) (CUstream);
  static cuStreamDestroy_p_h cuStreamDestroy_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamDestroy(CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuStreamDestroy_h == NULL)
	cuStreamDestroy_h = (cuStreamDestroy_p_h) dlsym(tau_handle,"cuStreamDestroy"); 
    if (cuStreamDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuStreamDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuEventCreate(CUevent * a1, unsigned int a2) {

  typedef CUresult (*cuEventCreate_p_h) (CUevent *, unsigned int);
  static cuEventCreate_p_h cuEventCreate_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventCreate(CUevent *, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuEventCreate_h == NULL)
	cuEventCreate_h = (cuEventCreate_p_h) dlsym(tau_handle,"cuEventCreate"); 
    if (cuEventCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuEventCreate_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuEventRecord(CUevent a1, CUstream a2) {

  typedef CUresult (*cuEventRecord_p_h) (CUevent, CUstream);
  static cuEventRecord_p_h cuEventRecord_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventRecord(CUevent, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuEventRecord_h == NULL)
	cuEventRecord_h = (cuEventRecord_p_h) dlsym(tau_handle,"cuEventRecord"); 
    if (cuEventRecord_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuEventRecord_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuEventQuery(CUevent a1) {

  typedef CUresult (*cuEventQuery_p_h) (CUevent);
  static cuEventQuery_p_h cuEventQuery_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventQuery(CUevent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuEventQuery_h == NULL)
	cuEventQuery_h = (cuEventQuery_p_h) dlsym(tau_handle,"cuEventQuery"); 
    if (cuEventQuery_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuEventQuery_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuEventSynchronize(CUevent a1) {

  typedef CUresult (*cuEventSynchronize_p_h) (CUevent);
  static cuEventSynchronize_p_h cuEventSynchronize_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventSynchronize(CUevent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuEventSynchronize_h == NULL)
	cuEventSynchronize_h = (cuEventSynchronize_p_h) dlsym(tau_handle,"cuEventSynchronize"); 
    if (cuEventSynchronize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
	Tau_cuda_register_sync_event();
#endif 
  retval  =  (*cuEventSynchronize_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuEventDestroy(CUevent a1) {

  typedef CUresult (*cuEventDestroy_p_h) (CUevent);
  static cuEventDestroy_p_h cuEventDestroy_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventDestroy(CUevent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuEventDestroy_h == NULL)
	cuEventDestroy_h = (cuEventDestroy_p_h) dlsym(tau_handle,"cuEventDestroy"); 
    if (cuEventDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuEventDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuEventElapsedTime(float * a1, CUevent a2, CUevent a3) {

  typedef CUresult (*cuEventElapsedTime_p_h) (float *, CUevent, CUevent);
  static cuEventElapsedTime_p_h cuEventElapsedTime_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventElapsedTime(float *, CUevent, CUevent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuEventElapsedTime_h == NULL)
	cuEventElapsedTime_h = (cuEventElapsedTime_p_h) dlsym(tau_handle,"cuEventElapsedTime"); 
    if (cuEventElapsedTime_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuEventElapsedTime_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuFuncSetBlockShape(CUfunction a1, int a2, int a3, int a4) {

  typedef CUresult (*cuFuncSetBlockShape_p_h) (CUfunction, int, int, int);
  static cuFuncSetBlockShape_p_h cuFuncSetBlockShape_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncSetBlockShape(CUfunction, int, int, int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuFuncSetBlockShape_h == NULL)
	cuFuncSetBlockShape_h = (cuFuncSetBlockShape_p_h) dlsym(tau_handle,"cuFuncSetBlockShape"); 
    if (cuFuncSetBlockShape_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuFuncSetBlockShape_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuFuncSetSharedSize(CUfunction a1, unsigned int a2) {

  typedef CUresult (*cuFuncSetSharedSize_p_h) (CUfunction, unsigned int);
  static cuFuncSetSharedSize_p_h cuFuncSetSharedSize_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncSetSharedSize(CUfunction, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuFuncSetSharedSize_h == NULL)
	cuFuncSetSharedSize_h = (cuFuncSetSharedSize_p_h) dlsym(tau_handle,"cuFuncSetSharedSize"); 
    if (cuFuncSetSharedSize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuFuncSetSharedSize_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuFuncGetAttribute(int * a1, CUfunction_attribute a2, CUfunction a3) {

  typedef CUresult (*cuFuncGetAttribute_p_h) (int *, CUfunction_attribute, CUfunction);
  static cuFuncGetAttribute_p_h cuFuncGetAttribute_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncGetAttribute(int *, CUfunction_attribute, CUfunction) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuFuncGetAttribute_h == NULL)
	cuFuncGetAttribute_h = (cuFuncGetAttribute_p_h) dlsym(tau_handle,"cuFuncGetAttribute"); 
    if (cuFuncGetAttribute_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuFuncGetAttribute_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuFuncSetCacheConfig(CUfunction a1, CUfunc_cache a2) {

  typedef CUresult (*cuFuncSetCacheConfig_p_h) (CUfunction, CUfunc_cache);
  static cuFuncSetCacheConfig_p_h cuFuncSetCacheConfig_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncSetCacheConfig(CUfunction, CUfunc_cache) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuFuncSetCacheConfig_h == NULL)
	cuFuncSetCacheConfig_h = (cuFuncSetCacheConfig_p_h) dlsym(tau_handle,"cuFuncSetCacheConfig"); 
    if (cuFuncSetCacheConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuFuncSetCacheConfig_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuParamSetSize(CUfunction a1, unsigned int a2) {

  typedef CUresult (*cuParamSetSize_p_h) (CUfunction, unsigned int);
  static cuParamSetSize_p_h cuParamSetSize_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetSize(CUfunction, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuParamSetSize_h == NULL)
	cuParamSetSize_h = (cuParamSetSize_p_h) dlsym(tau_handle,"cuParamSetSize"); 
    if (cuParamSetSize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuParamSetSize_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuParamSeti(CUfunction a1, int a2, unsigned int a3) {

  typedef CUresult (*cuParamSeti_p_h) (CUfunction, int, unsigned int);
  static cuParamSeti_p_h cuParamSeti_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSeti(CUfunction, int, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuParamSeti_h == NULL)
	cuParamSeti_h = (cuParamSeti_p_h) dlsym(tau_handle,"cuParamSeti"); 
    if (cuParamSeti_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuParamSeti_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuParamSetf(CUfunction a1, int a2, float a3) {

  typedef CUresult (*cuParamSetf_p_h) (CUfunction, int, float);
  static cuParamSetf_p_h cuParamSetf_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetf(CUfunction, int, float) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuParamSetf_h == NULL)
	cuParamSetf_h = (cuParamSetf_p_h) dlsym(tau_handle,"cuParamSetf"); 
    if (cuParamSetf_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuParamSetf_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuParamSetv(CUfunction a1, int a2, void * a3, unsigned int a4) {

  typedef CUresult (*cuParamSetv_p_h) (CUfunction, int, void *, unsigned int);
  static cuParamSetv_p_h cuParamSetv_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetv(CUfunction, int, void *, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuParamSetv_h == NULL)
	cuParamSetv_h = (cuParamSetv_p_h) dlsym(tau_handle,"cuParamSetv"); 
    if (cuParamSetv_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuParamSetv_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuLaunch(CUfunction a1) {

  typedef CUresult (*cuLaunch_p_h) (CUfunction);
  static cuLaunch_p_h cuLaunch_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuLaunch(CUfunction) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuLaunch_h == NULL)
	cuLaunch_h = (cuLaunch_p_h) dlsym(tau_handle,"cuLaunch"); 
    if (cuLaunch_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
		Tau_cuda_init();
		int device;
		cuCtxGetDevice(&device);
		CUcontext ctx;
		cuCtxPopCurrent(&ctx);
		cuCtxPushCurrent(ctx);
		Tau_cuda_enqueue_kernel_enter_event((const char*) a1, 
																				&cudaDriverGpuId(device, ctx, 0),
					TauInternal_CurrentProfiler(RtsLayer::myNode())->CallPathFunction);
#endif
  	retval  =  (*cuLaunch_h)( a1);
#ifdef TRACK_KERNEL
		Tau_cuda_enqueue_kernel_exit_event(), 
#endif
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuLaunchGrid(CUfunction a1, int a2, int a3) {

  typedef CUresult (*cuLaunchGrid_p_h) (CUfunction, int, int);
  static cuLaunchGrid_p_h cuLaunchGrid_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuLaunchGrid(CUfunction, int, int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuLaunchGrid_h == NULL)
	cuLaunchGrid_h = (cuLaunchGrid_p_h) dlsym(tau_handle,"cuLaunchGrid"); 
    if (cuLaunchGrid_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
		Tau_cuda_init();
		int device;
		cuCtxGetDevice(&device);
		CUcontext ctx;
		cuCtxPopCurrent(&ctx);
		cuCtxPushCurrent(ctx);
		Tau_cuda_enqueue_kernel_enter_event((const char*)a1,
			&cudaDriverGpuId(device,ctx,0),
			TauInternal_CurrentProfiler(RtsLayer::myNode())->CallPathFunction);
#endif
  	retval  =  (*cuLaunchGrid_h)( a1,  a2,  a3);
#ifdef TRACK_KERNEL
		Tau_cuda_enqueue_kernel_exit_event();
#endif
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuLaunchGridAsync(CUfunction a1, int a2, int a3, CUstream a4) {

  typedef CUresult (*cuLaunchGridAsync_p_h) (CUfunction, int, int, CUstream);
  static cuLaunchGridAsync_p_h cuLaunchGridAsync_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuLaunchGridAsync(CUfunction, int, int, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuLaunchGridAsync_h == NULL)
	cuLaunchGridAsync_h = (cuLaunchGridAsync_p_h) dlsym(tau_handle,"cuLaunchGridAsync"); 
    if (cuLaunchGridAsync_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
#ifdef TRACK_KERNEL
		Tau_cuda_init();
		int device;
		cuCtxGetDevice(&device);
		CUcontext ctx;
		cuCtxPopCurrent(&ctx);
		cuCtxPushCurrent(ctx);
		Tau_cuda_enqueue_kernel_enter_event((const char*)a1,
			&cudaDriverGpuId(device,ctx,a4),
			TauInternal_CurrentProfiler(RtsLayer::myNode())->CallPathFunction);
#endif
  	retval  =  (*cuLaunchGridAsync_h)( a1,  a2,  a3,  a4);
#ifdef TRACK_KERNEL
		Tau_cuda_enqueue_kernel_exit_event();
#endif
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuParamSetTexRef(CUfunction a1, int a2, CUtexref a3) {

  typedef CUresult (*cuParamSetTexRef_p_h) (CUfunction, int, CUtexref);
  static cuParamSetTexRef_p_h cuParamSetTexRef_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetTexRef(CUfunction, int, CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuParamSetTexRef_h == NULL)
	cuParamSetTexRef_h = (cuParamSetTexRef_p_h) dlsym(tau_handle,"cuParamSetTexRef"); 
    if (cuParamSetTexRef_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuParamSetTexRef_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetArray(CUtexref a1, CUarray a2, unsigned int a3) {

  typedef CUresult (*cuTexRefSetArray_p_h) (CUtexref, CUarray, unsigned int);
  static cuTexRefSetArray_p_h cuTexRefSetArray_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetArray(CUtexref, CUarray, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetArray_h == NULL)
	cuTexRefSetArray_h = (cuTexRefSetArray_p_h) dlsym(tau_handle,"cuTexRefSetArray"); 
    if (cuTexRefSetArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetArray_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetAddress_v2(size_t * a1, CUtexref a2, CUdeviceptr a3, size_t a4) {

  typedef CUresult (*cuTexRefSetAddress_v2_p_h) (size_t *, CUtexref, CUdeviceptr, size_t);
  static cuTexRefSetAddress_v2_p_h cuTexRefSetAddress_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetAddress_v2(size_t *, CUtexref, CUdeviceptr, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetAddress_v2_h == NULL)
	cuTexRefSetAddress_v2_h = (cuTexRefSetAddress_v2_p_h) dlsym(tau_handle,"cuTexRefSetAddress_v2"); 
    if (cuTexRefSetAddress_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetAddress_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetAddress2D_v2(CUtexref a1, const CUDA_ARRAY_DESCRIPTOR * a2, CUdeviceptr a3, size_t a4) {

  typedef CUresult (*cuTexRefSetAddress2D_v2_p_h) (CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
  static cuTexRefSetAddress2D_v2_p_h cuTexRefSetAddress2D_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetAddress2D_v2(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetAddress2D_v2_h == NULL)
	cuTexRefSetAddress2D_v2_h = (cuTexRefSetAddress2D_v2_p_h) dlsym(tau_handle,"cuTexRefSetAddress2D_v2"); 
    if (cuTexRefSetAddress2D_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetAddress2D_v2_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetFormat(CUtexref a1, CUarray_format a2, int a3) {

  typedef CUresult (*cuTexRefSetFormat_p_h) (CUtexref, CUarray_format, int);
  static cuTexRefSetFormat_p_h cuTexRefSetFormat_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetFormat(CUtexref, CUarray_format, int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetFormat_h == NULL)
	cuTexRefSetFormat_h = (cuTexRefSetFormat_p_h) dlsym(tau_handle,"cuTexRefSetFormat"); 
    if (cuTexRefSetFormat_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetFormat_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetAddressMode(CUtexref a1, int a2, CUaddress_mode a3) {

  typedef CUresult (*cuTexRefSetAddressMode_p_h) (CUtexref, int, CUaddress_mode);
  static cuTexRefSetAddressMode_p_h cuTexRefSetAddressMode_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetAddressMode(CUtexref, int, CUaddress_mode) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetAddressMode_h == NULL)
	cuTexRefSetAddressMode_h = (cuTexRefSetAddressMode_p_h) dlsym(tau_handle,"cuTexRefSetAddressMode"); 
    if (cuTexRefSetAddressMode_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetAddressMode_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetFilterMode(CUtexref a1, CUfilter_mode a2) {

  typedef CUresult (*cuTexRefSetFilterMode_p_h) (CUtexref, CUfilter_mode);
  static cuTexRefSetFilterMode_p_h cuTexRefSetFilterMode_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetFilterMode(CUtexref, CUfilter_mode) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetFilterMode_h == NULL)
	cuTexRefSetFilterMode_h = (cuTexRefSetFilterMode_p_h) dlsym(tau_handle,"cuTexRefSetFilterMode"); 
    if (cuTexRefSetFilterMode_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetFilterMode_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefSetFlags(CUtexref a1, unsigned int a2) {

  typedef CUresult (*cuTexRefSetFlags_p_h) (CUtexref, unsigned int);
  static cuTexRefSetFlags_p_h cuTexRefSetFlags_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetFlags(CUtexref, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefSetFlags_h == NULL)
	cuTexRefSetFlags_h = (cuTexRefSetFlags_p_h) dlsym(tau_handle,"cuTexRefSetFlags"); 
    if (cuTexRefSetFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefSetFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefGetAddress_v2(CUdeviceptr * a1, CUtexref a2) {

  typedef CUresult (*cuTexRefGetAddress_v2_p_h) (CUdeviceptr *, CUtexref);
  static cuTexRefGetAddress_v2_p_h cuTexRefGetAddress_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetAddress_v2(CUdeviceptr *, CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefGetAddress_v2_h == NULL)
	cuTexRefGetAddress_v2_h = (cuTexRefGetAddress_v2_p_h) dlsym(tau_handle,"cuTexRefGetAddress_v2"); 
    if (cuTexRefGetAddress_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefGetAddress_v2_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefGetArray(CUarray * a1, CUtexref a2) {

  typedef CUresult (*cuTexRefGetArray_p_h) (CUarray *, CUtexref);
  static cuTexRefGetArray_p_h cuTexRefGetArray_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetArray(CUarray *, CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefGetArray_h == NULL)
	cuTexRefGetArray_h = (cuTexRefGetArray_p_h) dlsym(tau_handle,"cuTexRefGetArray"); 
    if (cuTexRefGetArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefGetArray_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefGetAddressMode(CUaddress_mode * a1, CUtexref a2, int a3) {

  typedef CUresult (*cuTexRefGetAddressMode_p_h) (CUaddress_mode *, CUtexref, int);
  static cuTexRefGetAddressMode_p_h cuTexRefGetAddressMode_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetAddressMode(CUaddress_mode *, CUtexref, int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefGetAddressMode_h == NULL)
	cuTexRefGetAddressMode_h = (cuTexRefGetAddressMode_p_h) dlsym(tau_handle,"cuTexRefGetAddressMode"); 
    if (cuTexRefGetAddressMode_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefGetAddressMode_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefGetFilterMode(CUfilter_mode * a1, CUtexref a2) {

  typedef CUresult (*cuTexRefGetFilterMode_p_h) (CUfilter_mode *, CUtexref);
  static cuTexRefGetFilterMode_p_h cuTexRefGetFilterMode_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetFilterMode(CUfilter_mode *, CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefGetFilterMode_h == NULL)
	cuTexRefGetFilterMode_h = (cuTexRefGetFilterMode_p_h) dlsym(tau_handle,"cuTexRefGetFilterMode"); 
    if (cuTexRefGetFilterMode_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefGetFilterMode_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefGetFormat(CUarray_format * a1, int * a2, CUtexref a3) {

  typedef CUresult (*cuTexRefGetFormat_p_h) (CUarray_format *, int *, CUtexref);
  static cuTexRefGetFormat_p_h cuTexRefGetFormat_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetFormat(CUarray_format *, int *, CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefGetFormat_h == NULL)
	cuTexRefGetFormat_h = (cuTexRefGetFormat_p_h) dlsym(tau_handle,"cuTexRefGetFormat"); 
    if (cuTexRefGetFormat_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefGetFormat_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefGetFlags(unsigned int * a1, CUtexref a2) {

  typedef CUresult (*cuTexRefGetFlags_p_h) (unsigned int *, CUtexref);
  static cuTexRefGetFlags_p_h cuTexRefGetFlags_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetFlags(unsigned int *, CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefGetFlags_h == NULL)
	cuTexRefGetFlags_h = (cuTexRefGetFlags_p_h) dlsym(tau_handle,"cuTexRefGetFlags"); 
    if (cuTexRefGetFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefGetFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefCreate(CUtexref * a1) {

  typedef CUresult (*cuTexRefCreate_p_h) (CUtexref *);
  static cuTexRefCreate_p_h cuTexRefCreate_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefCreate(CUtexref *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefCreate_h == NULL)
	cuTexRefCreate_h = (cuTexRefCreate_p_h) dlsym(tau_handle,"cuTexRefCreate"); 
    if (cuTexRefCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefCreate_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuTexRefDestroy(CUtexref a1) {

  typedef CUresult (*cuTexRefDestroy_p_h) (CUtexref);
  static cuTexRefDestroy_p_h cuTexRefDestroy_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefDestroy(CUtexref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuTexRefDestroy_h == NULL)
	cuTexRefDestroy_h = (cuTexRefDestroy_p_h) dlsym(tau_handle,"cuTexRefDestroy"); 
    if (cuTexRefDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuTexRefDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuSurfRefSetArray(CUsurfref a1, CUarray a2, unsigned int a3) {

  typedef CUresult (*cuSurfRefSetArray_p_h) (CUsurfref, CUarray, unsigned int);
  static cuSurfRefSetArray_p_h cuSurfRefSetArray_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuSurfRefSetArray(CUsurfref, CUarray, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuSurfRefSetArray_h == NULL)
	cuSurfRefSetArray_h = (cuSurfRefSetArray_p_h) dlsym(tau_handle,"cuSurfRefSetArray"); 
    if (cuSurfRefSetArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuSurfRefSetArray_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuSurfRefGetArray(CUarray * a1, CUsurfref a2) {

  typedef CUresult (*cuSurfRefGetArray_p_h) (CUarray *, CUsurfref);
  static cuSurfRefGetArray_p_h cuSurfRefGetArray_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuSurfRefGetArray(CUarray *, CUsurfref) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuSurfRefGetArray_h == NULL)
	cuSurfRefGetArray_h = (cuSurfRefGetArray_p_h) dlsym(tau_handle,"cuSurfRefGetArray"); 
    if (cuSurfRefGetArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuSurfRefGetArray_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource a1) {

  typedef CUresult (*cuGraphicsUnregisterResource_p_h) (CUgraphicsResource);
  static cuGraphicsUnregisterResource_p_h cuGraphicsUnregisterResource_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsUnregisterResource(CUgraphicsResource) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGraphicsUnregisterResource_h == NULL)
	cuGraphicsUnregisterResource_h = (cuGraphicsUnregisterResource_p_h) dlsym(tau_handle,"cuGraphicsUnregisterResource"); 
    if (cuGraphicsUnregisterResource_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGraphicsUnregisterResource_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray * a1, CUgraphicsResource a2, unsigned int a3, unsigned int a4) {

  typedef CUresult (*cuGraphicsSubResourceGetMappedArray_p_h) (CUarray *, CUgraphicsResource, unsigned int, unsigned int);
  static cuGraphicsSubResourceGetMappedArray_p_h cuGraphicsSubResourceGetMappedArray_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsSubResourceGetMappedArray(CUarray *, CUgraphicsResource, unsigned int, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGraphicsSubResourceGetMappedArray_h == NULL)
	cuGraphicsSubResourceGetMappedArray_h = (cuGraphicsSubResourceGetMappedArray_p_h) dlsym(tau_handle,"cuGraphicsSubResourceGetMappedArray"); 
    if (cuGraphicsSubResourceGetMappedArray_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGraphicsSubResourceGetMappedArray_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * a1, size_t * a2, CUgraphicsResource a3) {

  typedef CUresult (*cuGraphicsResourceGetMappedPointer_v2_p_h) (CUdeviceptr *, size_t *, CUgraphicsResource);
  static cuGraphicsResourceGetMappedPointer_v2_p_h cuGraphicsResourceGetMappedPointer_v2_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *, size_t *, CUgraphicsResource) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGraphicsResourceGetMappedPointer_v2_h == NULL)
	cuGraphicsResourceGetMappedPointer_v2_h = (cuGraphicsResourceGetMappedPointer_v2_p_h) dlsym(tau_handle,"cuGraphicsResourceGetMappedPointer_v2"); 
    if (cuGraphicsResourceGetMappedPointer_v2_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGraphicsResourceGetMappedPointer_v2_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource a1, unsigned int a2) {

  typedef CUresult (*cuGraphicsResourceSetMapFlags_p_h) (CUgraphicsResource, unsigned int);
  static cuGraphicsResourceSetMapFlags_p_h cuGraphicsResourceSetMapFlags_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource, unsigned int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGraphicsResourceSetMapFlags_h == NULL)
	cuGraphicsResourceSetMapFlags_h = (cuGraphicsResourceSetMapFlags_p_h) dlsym(tau_handle,"cuGraphicsResourceSetMapFlags"); 
    if (cuGraphicsResourceSetMapFlags_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGraphicsResourceSetMapFlags_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGraphicsMapResources(unsigned int a1, CUgraphicsResource * a2, CUstream a3) {

  typedef CUresult (*cuGraphicsMapResources_p_h) (unsigned int, CUgraphicsResource *, CUstream);
  static cuGraphicsMapResources_p_h cuGraphicsMapResources_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsMapResources(unsigned int, CUgraphicsResource *, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGraphicsMapResources_h == NULL)
	cuGraphicsMapResources_h = (cuGraphicsMapResources_p_h) dlsym(tau_handle,"cuGraphicsMapResources"); 
    if (cuGraphicsMapResources_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGraphicsMapResources_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGraphicsUnmapResources(unsigned int a1, CUgraphicsResource * a2, CUstream a3) {

  typedef CUresult (*cuGraphicsUnmapResources_p_h) (unsigned int, CUgraphicsResource *, CUstream);
  static cuGraphicsUnmapResources_p_h cuGraphicsUnmapResources_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsUnmapResources(unsigned int, CUgraphicsResource *, CUstream) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGraphicsUnmapResources_h == NULL)
	cuGraphicsUnmapResources_h = (cuGraphicsUnmapResources_p_h) dlsym(tau_handle,"cuGraphicsUnmapResources"); 
    if (cuGraphicsUnmapResources_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGraphicsUnmapResources_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

CUresult cuGetExportTable(const void ** a1, const CUuuid * a2) {

  typedef CUresult (*cuGetExportTable_p_h) (const void **, const CUuuid *);
  static cuGetExportTable_p_h cuGetExportTable_h = NULL;
  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGetExportTable(const void **, const CUuuid *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cuGetExportTable_h == NULL)
	cuGetExportTable_h = (cuGetExportTable_p_h) dlsym(tau_handle,"cuGetExportTable"); 
    if (cuGetExportTable_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cuGetExportTable_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

