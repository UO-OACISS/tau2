#include <cuda_wrap.h>
#include <TAU.h>
CUresult  tau_cuInit(unsigned int a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuInit(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuInit(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDriverGetVersion(int * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDriverGetVersion(int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDriverGetVersion(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceGet(CUdevice * a1, int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGet(CUdevice *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceGet(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceGetCount(int * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetCount(int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceGetCount(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceGetName(char * a1, int a2, CUdevice a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetName(char *, int, CUdevice) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceGetName(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceComputeCapability(int * a1, int * a2, CUdevice a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceComputeCapability(int *, int *, CUdevice) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceComputeCapability(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceTotalMem_v2(size_t * a1, CUdevice a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceTotalMem_v2(size_t *, CUdevice) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceTotalMem_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceGetProperties(CUdevprop * a1, CUdevice a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetProperties(CUdevprop *, CUdevice) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceGetProperties(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuDeviceGetAttribute(int * a1, CUdevice_attribute a2, CUdevice a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuDeviceGetAttribute(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxCreate_v2(CUcontext * a1, unsigned int a2, CUdevice a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxCreate_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxDestroy(CUcontext a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxDestroy(CUcontext) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxDestroy(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxAttach(CUcontext * a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxAttach(CUcontext *, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxAttach(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxDetach(CUcontext a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxDetach(CUcontext) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxDetach(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxPushCurrent(CUcontext a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxPushCurrent(CUcontext) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxPushCurrent(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxPopCurrent(CUcontext * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxPopCurrent(CUcontext *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxPopCurrent(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxGetDevice(CUdevice * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetDevice(CUdevice *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxGetDevice(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxSynchronize() {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxSynchronize(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxSynchronize();
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxSetLimit(CUlimit a1, size_t a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxSetLimit(CUlimit, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxSetLimit(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxGetLimit(size_t * a1, CUlimit a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetLimit(size_t *, CUlimit) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxGetLimit(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxGetCacheConfig(CUfunc_cache * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetCacheConfig(CUfunc_cache *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxGetCacheConfig(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxSetCacheConfig(CUfunc_cache a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxSetCacheConfig(CUfunc_cache) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxSetCacheConfig(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuCtxGetApiVersion(CUcontext a1, unsigned int * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuCtxGetApiVersion(CUcontext, unsigned int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuCtxGetApiVersion(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleLoad(CUmodule * a1, const char * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoad(CUmodule *, const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleLoad(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleLoadData(CUmodule * a1, const void * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoadData(CUmodule *, const void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleLoadData(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleLoadDataEx(CUmodule * a1, const void * a2, unsigned int a3, CUjit_option * a4, void ** a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoadDataEx(CUmodule *, const void *, unsigned int, CUjit_option *, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleLoadDataEx(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleLoadFatBinary(CUmodule * a1, const void * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleLoadFatBinary(CUmodule *, const void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleLoadFatBinary(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleUnload(CUmodule a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleUnload(CUmodule) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleUnload(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleGetFunction(CUfunction * a1, CUmodule a2, const char * a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleGetFunction(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleGetGlobal_v2(CUdeviceptr * a1, size_t * a2, CUmodule a3, const char * a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetGlobal_v2(CUdeviceptr *, size_t *, CUmodule, const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleGetGlobal_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleGetTexRef(CUtexref * a1, CUmodule a2, const char * a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetTexRef(CUtexref *, CUmodule, const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleGetTexRef(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuModuleGetSurfRef(CUsurfref * a1, CUmodule a2, const char * a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuModuleGetSurfRef(CUsurfref *, CUmodule, const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuModuleGetSurfRef(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemGetInfo_v2(size_t * a1, size_t * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemGetInfo_v2(size_t *, size_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemGetInfo_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemAlloc_v2(CUdeviceptr * a1, size_t a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemAlloc_v2(CUdeviceptr *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemAlloc_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemAllocPitch_v2(CUdeviceptr * a1, size_t * a2, size_t a3, size_t a4, unsigned int a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemAllocPitch_v2(CUdeviceptr *, size_t *, size_t, size_t, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemAllocPitch_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemFree_v2(CUdeviceptr a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemFree_v2(CUdeviceptr) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemFree_v2(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemGetAddressRange_v2(CUdeviceptr * a1, size_t * a2, CUdeviceptr a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemGetAddressRange_v2(CUdeviceptr *, size_t *, CUdeviceptr) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemGetAddressRange_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemAllocHost_v2(void ** a1, size_t a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemAllocHost_v2(void **, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemAllocHost_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemFreeHost(void * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemFreeHost(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemFreeHost(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemHostAlloc(void ** a1, size_t a2, unsigned int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemHostAlloc(void **, size_t, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemHostAlloc(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemHostGetDevicePointer_v2(CUdeviceptr * a1, void * a2, unsigned int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *, void *, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemHostGetDevicePointer_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemHostGetFlags(unsigned int * a1, void * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemHostGetFlags(unsigned int *, void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemHostGetFlags(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyHtoD_v2(CUdeviceptr a1, const void * a2, size_t a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyHtoD_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyDtoH_v2(void * a1, CUdeviceptr a2, size_t a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyDtoH_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyDtoD_v2(CUdeviceptr a1, CUdeviceptr a2, size_t a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyDtoD_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyDtoA_v2(CUarray a1, size_t a2, CUdeviceptr a3, size_t a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoA_v2(CUarray, size_t, CUdeviceptr, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyDtoA_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyAtoD_v2(CUdeviceptr a1, CUarray a2, size_t a3, size_t a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoD_v2(CUdeviceptr, CUarray, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyAtoD_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyHtoA_v2(CUarray a1, size_t a2, const void * a3, size_t a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoA_v2(CUarray, size_t, const void *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyHtoA_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyAtoH_v2(void * a1, CUarray a2, size_t a3, size_t a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoH_v2(void *, CUarray, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyAtoH_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyAtoA_v2(CUarray a1, size_t a2, CUarray a3, size_t a4, size_t a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoA_v2(CUarray, size_t, CUarray, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyAtoA_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpy2D_v2(const CUDA_MEMCPY2D * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpy2D_v2(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpy2DUnaligned_v2(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpy3D_v2(const CUDA_MEMCPY3D * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpy3D_v2(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyHtoDAsync_v2(CUdeviceptr a1, const void * a2, size_t a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr, const void *, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyHtoDAsync_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyDtoHAsync_v2(void * a1, CUdeviceptr a2, size_t a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoHAsync_v2(void *, CUdeviceptr, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyDtoHAsync_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyDtoDAsync_v2(CUdeviceptr a1, CUdeviceptr a2, size_t a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyDtoDAsync_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyHtoAAsync_v2(CUarray a1, size_t a2, const void * a3, size_t a4, CUstream a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyHtoAAsync_v2(CUarray, size_t, const void *, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyHtoAAsync_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpyAtoHAsync_v2(void * a1, CUarray a2, size_t a3, size_t a4, CUstream a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpyAtoHAsync_v2(void *, CUarray, size_t, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpyAtoHAsync_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * a1, CUstream a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpy2DAsync_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * a1, CUstream a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemcpy3DAsync_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD8_v2(CUdeviceptr a1, unsigned char a2, size_t a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD8_v2(CUdeviceptr, unsigned char, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD8_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD16_v2(CUdeviceptr a1, unsigned short a2, size_t a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD16_v2(CUdeviceptr, unsigned short, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD16_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD32_v2(CUdeviceptr a1, unsigned int a2, size_t a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD32_v2(CUdeviceptr, unsigned int, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD32_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD2D8_v2(CUdeviceptr a1, size_t a2, unsigned char a3, size_t a4, size_t a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D8_v2(CUdeviceptr, size_t, unsigned char, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD2D8_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD2D16_v2(CUdeviceptr a1, size_t a2, unsigned short a3, size_t a4, size_t a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D16_v2(CUdeviceptr, size_t, unsigned short, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD2D16_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD2D32_v2(CUdeviceptr a1, size_t a2, unsigned int a3, size_t a4, size_t a5) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D32_v2(CUdeviceptr, size_t, unsigned int, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD2D32_v2(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD8Async(CUdeviceptr a1, unsigned char a2, size_t a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD8Async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD16Async(CUdeviceptr a1, unsigned short a2, size_t a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD16Async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD32Async(CUdeviceptr a1, unsigned int a2, size_t a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD32Async(CUdeviceptr, unsigned int, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD32Async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD2D8Async(CUdeviceptr a1, size_t a2, unsigned char a3, size_t a4, size_t a5, CUstream a6) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D8Async(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD2D8Async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD2D16Async(CUdeviceptr a1, size_t a2, unsigned short a3, size_t a4, size_t a5, CUstream a6) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D16Async(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD2D16Async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuMemsetD2D32Async(CUdeviceptr a1, size_t a2, unsigned int a3, size_t a4, size_t a5, CUstream a6) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuMemsetD2D32Async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuArrayCreate_v2(CUarray * a1, const CUDA_ARRAY_DESCRIPTOR * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArrayCreate_v2(CUarray *, const CUDA_ARRAY_DESCRIPTOR *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuArrayCreate_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * a1, CUarray a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *, CUarray) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuArrayGetDescriptor_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuArrayDestroy(CUarray a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArrayDestroy(CUarray) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuArrayDestroy(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuArray3DCreate_v2(CUarray * a1, const CUDA_ARRAY3D_DESCRIPTOR * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArray3DCreate_v2(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuArray3DCreate_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * a1, CUarray a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *, CUarray) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuArray3DGetDescriptor_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuStreamCreate(CUstream * a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamCreate(CUstream *, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuStreamCreate(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuStreamWaitEvent(CUstream a1, CUevent a2, unsigned int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuStreamWaitEvent(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuStreamQuery(CUstream a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamQuery(CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuStreamQuery(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuStreamSynchronize(CUstream a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamSynchronize(CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuStreamSynchronize(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuStreamDestroy(CUstream a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuStreamDestroy(CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuStreamDestroy(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuEventCreate(CUevent * a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventCreate(CUevent *, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuEventCreate(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuEventRecord(CUevent a1, CUstream a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventRecord(CUevent, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuEventRecord(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuEventQuery(CUevent a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventQuery(CUevent) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuEventQuery(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuEventSynchronize(CUevent a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventSynchronize(CUevent) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuEventSynchronize(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuEventDestroy(CUevent a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventDestroy(CUevent) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuEventDestroy(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuEventElapsedTime(float * a1, CUevent a2, CUevent a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuEventElapsedTime(float *, CUevent, CUevent) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuEventElapsedTime(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuFuncSetBlockShape(CUfunction a1, int a2, int a3, int a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncSetBlockShape(CUfunction, int, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuFuncSetBlockShape(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuFuncSetSharedSize(CUfunction a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncSetSharedSize(CUfunction, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuFuncSetSharedSize(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuFuncGetAttribute(int * a1, CUfunction_attribute a2, CUfunction a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncGetAttribute(int *, CUfunction_attribute, CUfunction) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuFuncGetAttribute(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuFuncSetCacheConfig(CUfunction a1, CUfunc_cache a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuFuncSetCacheConfig(CUfunction, CUfunc_cache) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuFuncSetCacheConfig(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuParamSetSize(CUfunction a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetSize(CUfunction, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuParamSetSize(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuParamSeti(CUfunction a1, int a2, unsigned int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSeti(CUfunction, int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuParamSeti(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuParamSetf(CUfunction a1, int a2, float a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetf(CUfunction, int, float) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuParamSetf(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuParamSetv(CUfunction a1, int a2, void * a3, unsigned int a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetv(CUfunction, int, void *, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuParamSetv(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuLaunch(CUfunction a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuLaunch(CUfunction) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuLaunch(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuLaunchGrid(CUfunction a1, int a2, int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuLaunchGrid(CUfunction, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuLaunchGrid(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuLaunchGridAsync(CUfunction a1, int a2, int a3, CUstream a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuLaunchGridAsync(CUfunction, int, int, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuLaunchGridAsync(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuParamSetTexRef(CUfunction a1, int a2, CUtexref a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuParamSetTexRef(CUfunction, int, CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuParamSetTexRef(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetArray(CUtexref a1, CUarray a2, unsigned int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetArray(CUtexref, CUarray, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetArray(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetAddress_v2(size_t * a1, CUtexref a2, CUdeviceptr a3, size_t a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetAddress_v2(size_t *, CUtexref, CUdeviceptr, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetAddress_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetAddress2D_v2(CUtexref a1, const CUDA_ARRAY_DESCRIPTOR * a2, CUdeviceptr a3, size_t a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetAddress2D_v2(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetAddress2D_v2(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetFormat(CUtexref a1, CUarray_format a2, int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetFormat(CUtexref, CUarray_format, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetFormat(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetAddressMode(CUtexref a1, int a2, CUaddress_mode a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetAddressMode(CUtexref, int, CUaddress_mode) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetAddressMode(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetFilterMode(CUtexref a1, CUfilter_mode a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetFilterMode(CUtexref, CUfilter_mode) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetFilterMode(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefSetFlags(CUtexref a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefSetFlags(CUtexref, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefSetFlags(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefGetAddress_v2(CUdeviceptr * a1, CUtexref a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetAddress_v2(CUdeviceptr *, CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefGetAddress_v2(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefGetArray(CUarray * a1, CUtexref a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetArray(CUarray *, CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefGetArray(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefGetAddressMode(CUaddress_mode * a1, CUtexref a2, int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetAddressMode(CUaddress_mode *, CUtexref, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefGetAddressMode(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefGetFilterMode(CUfilter_mode * a1, CUtexref a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetFilterMode(CUfilter_mode *, CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefGetFilterMode(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefGetFormat(CUarray_format * a1, int * a2, CUtexref a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetFormat(CUarray_format *, int *, CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefGetFormat(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefGetFlags(unsigned int * a1, CUtexref a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefGetFlags(unsigned int *, CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefGetFlags(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefCreate(CUtexref * a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefCreate(CUtexref *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefCreate(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuTexRefDestroy(CUtexref a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuTexRefDestroy(CUtexref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuTexRefDestroy(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuSurfRefSetArray(CUsurfref a1, CUarray a2, unsigned int a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuSurfRefSetArray(CUsurfref, CUarray, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuSurfRefSetArray(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuSurfRefGetArray(CUarray * a1, CUsurfref a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuSurfRefGetArray(CUarray *, CUsurfref) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuSurfRefGetArray(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGraphicsUnregisterResource(CUgraphicsResource a1) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsUnregisterResource(CUgraphicsResource) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGraphicsUnregisterResource(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGraphicsSubResourceGetMappedArray(CUarray * a1, CUgraphicsResource a2, unsigned int a3, unsigned int a4) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsSubResourceGetMappedArray(CUarray *, CUgraphicsResource, unsigned int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGraphicsSubResourceGetMappedArray(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * a1, size_t * a2, CUgraphicsResource a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *, size_t *, CUgraphicsResource) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGraphicsResourceGetMappedPointer_v2(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGraphicsResourceSetMapFlags(CUgraphicsResource a1, unsigned int a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGraphicsResourceSetMapFlags(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGraphicsMapResources(unsigned int a1, CUgraphicsResource * a2, CUstream a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsMapResources(unsigned int, CUgraphicsResource *, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGraphicsMapResources(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGraphicsUnmapResources(unsigned int a1, CUgraphicsResource * a2, CUstream a3) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGraphicsUnmapResources(unsigned int, CUgraphicsResource *, CUstream) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGraphicsUnmapResources(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}

CUresult  tau_cuGetExportTable(const void ** a1, const CUuuid * a2) {

  CUresult retval;
  TAU_PROFILE_TIMER(t,"CUresult cuGetExportTable(const void **, const CUuuid *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  cuGetExportTable(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

