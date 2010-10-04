#ifndef _TAU_cuda_H_
#define _TAU_cuda_H_

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif /*  __cplusplus */

#define  cuInit(a1) tau_cuInit(a1)
extern CUresult tau_cuInit(unsigned int a1);

#define  cuDriverGetVersion(a1) tau_cuDriverGetVersion(a1)
extern CUresult tau_cuDriverGetVersion(int * a1);

#define  cuDeviceGet(a1, a2) tau_cuDeviceGet(a1, a2)
extern CUresult tau_cuDeviceGet(CUdevice * a1, int a2);

#define  cuDeviceGetCount(a1) tau_cuDeviceGetCount(a1)
extern CUresult tau_cuDeviceGetCount(int * a1);

#define  cuDeviceGetName(a1, a2, a3) tau_cuDeviceGetName(a1, a2, a3)
extern CUresult tau_cuDeviceGetName(char * a1, int a2, CUdevice a3);

#define  cuDeviceComputeCapability(a1, a2, a3) tau_cuDeviceComputeCapability(a1, a2, a3)
extern CUresult tau_cuDeviceComputeCapability(int * a1, int * a2, CUdevice a3);

#define  cuDeviceTotalMem_v2(a1, a2) tau_cuDeviceTotalMem_v2(a1, a2)
extern CUresult tau_cuDeviceTotalMem_v2(size_t * a1, CUdevice a2);

#define  cuDeviceGetProperties(a1, a2) tau_cuDeviceGetProperties(a1, a2)
extern CUresult tau_cuDeviceGetProperties(CUdevprop * a1, CUdevice a2);

#define  cuDeviceGetAttribute(a1, a2, a3) tau_cuDeviceGetAttribute(a1, a2, a3)
extern CUresult tau_cuDeviceGetAttribute(int * a1, CUdevice_attribute a2, CUdevice a3);

#define  cuCtxCreate_v2(a1, a2, a3) tau_cuCtxCreate_v2(a1, a2, a3)
extern CUresult tau_cuCtxCreate_v2(CUcontext * a1, unsigned int a2, CUdevice a3);

#define  cuCtxDestroy(a1) tau_cuCtxDestroy(a1)
extern CUresult tau_cuCtxDestroy(CUcontext a1);

#define  cuCtxAttach(a1, a2) tau_cuCtxAttach(a1, a2)
extern CUresult tau_cuCtxAttach(CUcontext * a1, unsigned int a2);

#define  cuCtxDetach(a1) tau_cuCtxDetach(a1)
extern CUresult tau_cuCtxDetach(CUcontext a1);

#define  cuCtxPushCurrent(a1) tau_cuCtxPushCurrent(a1)
extern CUresult tau_cuCtxPushCurrent(CUcontext a1);

#define  cuCtxPopCurrent(a1) tau_cuCtxPopCurrent(a1)
extern CUresult tau_cuCtxPopCurrent(CUcontext * a1);

#define  cuCtxGetDevice(a1) tau_cuCtxGetDevice(a1)
extern CUresult tau_cuCtxGetDevice(CUdevice * a1);

#define  cuCtxSynchronize() tau_cuCtxSynchronize()
extern CUresult tau_cuCtxSynchronize();

#define  cuCtxSetLimit(a1, a2) tau_cuCtxSetLimit(a1, a2)
extern CUresult tau_cuCtxSetLimit(CUlimit a1, size_t a2);

#define  cuCtxGetLimit(a1, a2) tau_cuCtxGetLimit(a1, a2)
extern CUresult tau_cuCtxGetLimit(size_t * a1, CUlimit a2);

#define  cuCtxGetCacheConfig(a1) tau_cuCtxGetCacheConfig(a1)
extern CUresult tau_cuCtxGetCacheConfig(CUfunc_cache * a1);

#define  cuCtxSetCacheConfig(a1) tau_cuCtxSetCacheConfig(a1)
extern CUresult tau_cuCtxSetCacheConfig(CUfunc_cache a1);

#define  cuCtxGetApiVersion(a1, a2) tau_cuCtxGetApiVersion(a1, a2)
extern CUresult tau_cuCtxGetApiVersion(CUcontext a1, unsigned int * a2);

#define  cuModuleLoad(a1, a2) tau_cuModuleLoad(a1, a2)
extern CUresult tau_cuModuleLoad(CUmodule * a1, const char * a2);

#define  cuModuleLoadData(a1, a2) tau_cuModuleLoadData(a1, a2)
extern CUresult tau_cuModuleLoadData(CUmodule * a1, const void * a2);

#define  cuModuleLoadDataEx(a1, a2, a3, a4, a5) tau_cuModuleLoadDataEx(a1, a2, a3, a4, a5)
extern CUresult tau_cuModuleLoadDataEx(CUmodule * a1, const void * a2, unsigned int a3, CUjit_option * a4, void ** a5);

#define  cuModuleLoadFatBinary(a1, a2) tau_cuModuleLoadFatBinary(a1, a2)
extern CUresult tau_cuModuleLoadFatBinary(CUmodule * a1, const void * a2);

#define  cuModuleUnload(a1) tau_cuModuleUnload(a1)
extern CUresult tau_cuModuleUnload(CUmodule a1);

#define  cuModuleGetFunction(a1, a2, a3) tau_cuModuleGetFunction(a1, a2, a3)
extern CUresult tau_cuModuleGetFunction(CUfunction * a1, CUmodule a2, const char * a3);

#define  cuModuleGetGlobal_v2(a1, a2, a3, a4) tau_cuModuleGetGlobal_v2(a1, a2, a3, a4)
extern CUresult tau_cuModuleGetGlobal_v2(CUdeviceptr * a1, size_t * a2, CUmodule a3, const char * a4);

#define  cuModuleGetTexRef(a1, a2, a3) tau_cuModuleGetTexRef(a1, a2, a3)
extern CUresult tau_cuModuleGetTexRef(CUtexref * a1, CUmodule a2, const char * a3);

#define  cuModuleGetSurfRef(a1, a2, a3) tau_cuModuleGetSurfRef(a1, a2, a3)
extern CUresult tau_cuModuleGetSurfRef(CUsurfref * a1, CUmodule a2, const char * a3);

#define  cuMemGetInfo_v2(a1, a2) tau_cuMemGetInfo_v2(a1, a2)
extern CUresult tau_cuMemGetInfo_v2(size_t * a1, size_t * a2);

#define  cuMemAlloc_v2(a1, a2) tau_cuMemAlloc_v2(a1, a2)
extern CUresult tau_cuMemAlloc_v2(CUdeviceptr * a1, size_t a2);

#define  cuMemAllocPitch_v2(a1, a2, a3, a4, a5) tau_cuMemAllocPitch_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemAllocPitch_v2(CUdeviceptr * a1, size_t * a2, size_t a3, size_t a4, unsigned int a5);

#define  cuMemFree_v2(a1) tau_cuMemFree_v2(a1)
extern CUresult tau_cuMemFree_v2(CUdeviceptr a1);

#define  cuMemGetAddressRange_v2(a1, a2, a3) tau_cuMemGetAddressRange_v2(a1, a2, a3)
extern CUresult tau_cuMemGetAddressRange_v2(CUdeviceptr * a1, size_t * a2, CUdeviceptr a3);

#define  cuMemAllocHost_v2(a1, a2) tau_cuMemAllocHost_v2(a1, a2)
extern CUresult tau_cuMemAllocHost_v2(void ** a1, size_t a2);

#define  cuMemFreeHost(a1) tau_cuMemFreeHost(a1)
extern CUresult tau_cuMemFreeHost(void * a1);

#define  cuMemHostAlloc(a1, a2, a3) tau_cuMemHostAlloc(a1, a2, a3)
extern CUresult tau_cuMemHostAlloc(void ** a1, size_t a2, unsigned int a3);

#define  cuMemHostGetDevicePointer_v2(a1, a2, a3) tau_cuMemHostGetDevicePointer_v2(a1, a2, a3)
extern CUresult tau_cuMemHostGetDevicePointer_v2(CUdeviceptr * a1, void * a2, unsigned int a3);

#define  cuMemHostGetFlags(a1, a2) tau_cuMemHostGetFlags(a1, a2)
extern CUresult tau_cuMemHostGetFlags(unsigned int * a1, void * a2);

#define  cuMemcpyHtoD_v2(a1, a2, a3) tau_cuMemcpyHtoD_v2(a1, a2, a3)
extern CUresult tau_cuMemcpyHtoD_v2(CUdeviceptr a1, const void * a2, size_t a3);

#define  cuMemcpyDtoH_v2(a1, a2, a3) tau_cuMemcpyDtoH_v2(a1, a2, a3)
extern CUresult tau_cuMemcpyDtoH_v2(void * a1, CUdeviceptr a2, size_t a3);

#define  cuMemcpyDtoD_v2(a1, a2, a3) tau_cuMemcpyDtoD_v2(a1, a2, a3)
extern CUresult tau_cuMemcpyDtoD_v2(CUdeviceptr a1, CUdeviceptr a2, size_t a3);

#define  cuMemcpyDtoA_v2(a1, a2, a3, a4) tau_cuMemcpyDtoA_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyDtoA_v2(CUarray a1, size_t a2, CUdeviceptr a3, size_t a4);

#define  cuMemcpyAtoD_v2(a1, a2, a3, a4) tau_cuMemcpyAtoD_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyAtoD_v2(CUdeviceptr a1, CUarray a2, size_t a3, size_t a4);

#define  cuMemcpyHtoA_v2(a1, a2, a3, a4) tau_cuMemcpyHtoA_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyHtoA_v2(CUarray a1, size_t a2, const void * a3, size_t a4);

#define  cuMemcpyAtoH_v2(a1, a2, a3, a4) tau_cuMemcpyAtoH_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyAtoH_v2(void * a1, CUarray a2, size_t a3, size_t a4);

#define  cuMemcpyAtoA_v2(a1, a2, a3, a4, a5) tau_cuMemcpyAtoA_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemcpyAtoA_v2(CUarray a1, size_t a2, CUarray a3, size_t a4, size_t a5);

#define  cuMemcpy2D_v2(a1) tau_cuMemcpy2D_v2(a1)
extern CUresult tau_cuMemcpy2D_v2(const CUDA_MEMCPY2D * a1);

#define  cuMemcpy2DUnaligned_v2(a1) tau_cuMemcpy2DUnaligned_v2(a1)
extern CUresult tau_cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D * a1);

#define  cuMemcpy3D_v2(a1) tau_cuMemcpy3D_v2(a1)
extern CUresult tau_cuMemcpy3D_v2(const CUDA_MEMCPY3D * a1);

#define  cuMemcpyHtoDAsync_v2(a1, a2, a3, a4) tau_cuMemcpyHtoDAsync_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyHtoDAsync_v2(CUdeviceptr a1, const void * a2, size_t a3, CUstream a4);

#define  cuMemcpyDtoHAsync_v2(a1, a2, a3, a4) tau_cuMemcpyDtoHAsync_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyDtoHAsync_v2(void * a1, CUdeviceptr a2, size_t a3, CUstream a4);

#define  cuMemcpyDtoDAsync_v2(a1, a2, a3, a4) tau_cuMemcpyDtoDAsync_v2(a1, a2, a3, a4)
extern CUresult tau_cuMemcpyDtoDAsync_v2(CUdeviceptr a1, CUdeviceptr a2, size_t a3, CUstream a4);

#define  cuMemcpyHtoAAsync_v2(a1, a2, a3, a4, a5) tau_cuMemcpyHtoAAsync_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemcpyHtoAAsync_v2(CUarray a1, size_t a2, const void * a3, size_t a4, CUstream a5);

#define  cuMemcpyAtoHAsync_v2(a1, a2, a3, a4, a5) tau_cuMemcpyAtoHAsync_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemcpyAtoHAsync_v2(void * a1, CUarray a2, size_t a3, size_t a4, CUstream a5);

#define  cuMemcpy2DAsync_v2(a1, a2) tau_cuMemcpy2DAsync_v2(a1, a2)
extern CUresult tau_cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D * a1, CUstream a2);

#define  cuMemcpy3DAsync_v2(a1, a2) tau_cuMemcpy3DAsync_v2(a1, a2)
extern CUresult tau_cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D * a1, CUstream a2);

#define  cuMemsetD8_v2(a1, a2, a3) tau_cuMemsetD8_v2(a1, a2, a3)
extern CUresult tau_cuMemsetD8_v2(CUdeviceptr a1, unsigned char a2, size_t a3);

#define  cuMemsetD16_v2(a1, a2, a3) tau_cuMemsetD16_v2(a1, a2, a3)
extern CUresult tau_cuMemsetD16_v2(CUdeviceptr a1, unsigned short a2, size_t a3);

#define  cuMemsetD32_v2(a1, a2, a3) tau_cuMemsetD32_v2(a1, a2, a3)
extern CUresult tau_cuMemsetD32_v2(CUdeviceptr a1, unsigned int a2, size_t a3);

#define  cuMemsetD2D8_v2(a1, a2, a3, a4, a5) tau_cuMemsetD2D8_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemsetD2D8_v2(CUdeviceptr a1, size_t a2, unsigned char a3, size_t a4, size_t a5);

#define  cuMemsetD2D16_v2(a1, a2, a3, a4, a5) tau_cuMemsetD2D16_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemsetD2D16_v2(CUdeviceptr a1, size_t a2, unsigned short a3, size_t a4, size_t a5);

#define  cuMemsetD2D32_v2(a1, a2, a3, a4, a5) tau_cuMemsetD2D32_v2(a1, a2, a3, a4, a5)
extern CUresult tau_cuMemsetD2D32_v2(CUdeviceptr a1, size_t a2, unsigned int a3, size_t a4, size_t a5);

#define  cuMemsetD8Async(a1, a2, a3, a4) tau_cuMemsetD8Async(a1, a2, a3, a4)
extern CUresult tau_cuMemsetD8Async(CUdeviceptr a1, unsigned char a2, size_t a3, CUstream a4);

#define  cuMemsetD16Async(a1, a2, a3, a4) tau_cuMemsetD16Async(a1, a2, a3, a4)
extern CUresult tau_cuMemsetD16Async(CUdeviceptr a1, unsigned short a2, size_t a3, CUstream a4);

#define  cuMemsetD32Async(a1, a2, a3, a4) tau_cuMemsetD32Async(a1, a2, a3, a4)
extern CUresult tau_cuMemsetD32Async(CUdeviceptr a1, unsigned int a2, size_t a3, CUstream a4);

#define  cuMemsetD2D8Async(a1, a2, a3, a4, a5, a6) tau_cuMemsetD2D8Async(a1, a2, a3, a4, a5, a6)
extern CUresult tau_cuMemsetD2D8Async(CUdeviceptr a1, size_t a2, unsigned char a3, size_t a4, size_t a5, CUstream a6);

#define  cuMemsetD2D16Async(a1, a2, a3, a4, a5, a6) tau_cuMemsetD2D16Async(a1, a2, a3, a4, a5, a6)
extern CUresult tau_cuMemsetD2D16Async(CUdeviceptr a1, size_t a2, unsigned short a3, size_t a4, size_t a5, CUstream a6);

#define  cuMemsetD2D32Async(a1, a2, a3, a4, a5, a6) tau_cuMemsetD2D32Async(a1, a2, a3, a4, a5, a6)
extern CUresult tau_cuMemsetD2D32Async(CUdeviceptr a1, size_t a2, unsigned int a3, size_t a4, size_t a5, CUstream a6);

#define  cuArrayCreate_v2(a1, a2) tau_cuArrayCreate_v2(a1, a2)
extern CUresult tau_cuArrayCreate_v2(CUarray * a1, const CUDA_ARRAY_DESCRIPTOR * a2);

#define  cuArrayGetDescriptor_v2(a1, a2) tau_cuArrayGetDescriptor_v2(a1, a2)
extern CUresult tau_cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR * a1, CUarray a2);

#define  cuArrayDestroy(a1) tau_cuArrayDestroy(a1)
extern CUresult tau_cuArrayDestroy(CUarray a1);

#define  cuArray3DCreate_v2(a1, a2) tau_cuArray3DCreate_v2(a1, a2)
extern CUresult tau_cuArray3DCreate_v2(CUarray * a1, const CUDA_ARRAY3D_DESCRIPTOR * a2);

#define  cuArray3DGetDescriptor_v2(a1, a2) tau_cuArray3DGetDescriptor_v2(a1, a2)
extern CUresult tau_cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR * a1, CUarray a2);

#define  cuStreamCreate(a1, a2) tau_cuStreamCreate(a1, a2)
extern CUresult tau_cuStreamCreate(CUstream * a1, unsigned int a2);

#define  cuStreamWaitEvent(a1, a2, a3) tau_cuStreamWaitEvent(a1, a2, a3)
extern CUresult tau_cuStreamWaitEvent(CUstream a1, CUevent a2, unsigned int a3);

#define  cuStreamQuery(a1) tau_cuStreamQuery(a1)
extern CUresult tau_cuStreamQuery(CUstream a1);

#define  cuStreamSynchronize(a1) tau_cuStreamSynchronize(a1)
extern CUresult tau_cuStreamSynchronize(CUstream a1);

#define  cuStreamDestroy(a1) tau_cuStreamDestroy(a1)
extern CUresult tau_cuStreamDestroy(CUstream a1);

#define  cuEventCreate(a1, a2) tau_cuEventCreate(a1, a2)
extern CUresult tau_cuEventCreate(CUevent * a1, unsigned int a2);

#define  cuEventRecord(a1, a2) tau_cuEventRecord(a1, a2)
extern CUresult tau_cuEventRecord(CUevent a1, CUstream a2);

#define  cuEventQuery(a1) tau_cuEventQuery(a1)
extern CUresult tau_cuEventQuery(CUevent a1);

#define  cuEventSynchronize(a1) tau_cuEventSynchronize(a1)
extern CUresult tau_cuEventSynchronize(CUevent a1);

#define  cuEventDestroy(a1) tau_cuEventDestroy(a1)
extern CUresult tau_cuEventDestroy(CUevent a1);

#define  cuEventElapsedTime(a1, a2, a3) tau_cuEventElapsedTime(a1, a2, a3)
extern CUresult tau_cuEventElapsedTime(float * a1, CUevent a2, CUevent a3);

#define  cuFuncSetBlockShape(a1, a2, a3, a4) tau_cuFuncSetBlockShape(a1, a2, a3, a4)
extern CUresult tau_cuFuncSetBlockShape(CUfunction a1, int a2, int a3, int a4);

#define  cuFuncSetSharedSize(a1, a2) tau_cuFuncSetSharedSize(a1, a2)
extern CUresult tau_cuFuncSetSharedSize(CUfunction a1, unsigned int a2);

#define  cuFuncGetAttribute(a1, a2, a3) tau_cuFuncGetAttribute(a1, a2, a3)
extern CUresult tau_cuFuncGetAttribute(int * a1, CUfunction_attribute a2, CUfunction a3);

#define  cuFuncSetCacheConfig(a1, a2) tau_cuFuncSetCacheConfig(a1, a2)
extern CUresult tau_cuFuncSetCacheConfig(CUfunction a1, CUfunc_cache a2);

#define  cuParamSetSize(a1, a2) tau_cuParamSetSize(a1, a2)
extern CUresult tau_cuParamSetSize(CUfunction a1, unsigned int a2);

#define  cuParamSeti(a1, a2, a3) tau_cuParamSeti(a1, a2, a3)
extern CUresult tau_cuParamSeti(CUfunction a1, int a2, unsigned int a3);

#define  cuParamSetf(a1, a2, a3) tau_cuParamSetf(a1, a2, a3)
extern CUresult tau_cuParamSetf(CUfunction a1, int a2, float a3);

#define  cuParamSetv(a1, a2, a3, a4) tau_cuParamSetv(a1, a2, a3, a4)
extern CUresult tau_cuParamSetv(CUfunction a1, int a2, void * a3, unsigned int a4);

#define  cuLaunch(a1) tau_cuLaunch(a1)
extern CUresult tau_cuLaunch(CUfunction a1);

#define  cuLaunchGrid(a1, a2, a3) tau_cuLaunchGrid(a1, a2, a3)
extern CUresult tau_cuLaunchGrid(CUfunction a1, int a2, int a3);

#define  cuLaunchGridAsync(a1, a2, a3, a4) tau_cuLaunchGridAsync(a1, a2, a3, a4)
extern CUresult tau_cuLaunchGridAsync(CUfunction a1, int a2, int a3, CUstream a4);

#define  cuParamSetTexRef(a1, a2, a3) tau_cuParamSetTexRef(a1, a2, a3)
extern CUresult tau_cuParamSetTexRef(CUfunction a1, int a2, CUtexref a3);

#define  cuTexRefSetArray(a1, a2, a3) tau_cuTexRefSetArray(a1, a2, a3)
extern CUresult tau_cuTexRefSetArray(CUtexref a1, CUarray a2, unsigned int a3);

#define  cuTexRefSetAddress_v2(a1, a2, a3, a4) tau_cuTexRefSetAddress_v2(a1, a2, a3, a4)
extern CUresult tau_cuTexRefSetAddress_v2(size_t * a1, CUtexref a2, CUdeviceptr a3, size_t a4);

#define  cuTexRefSetAddress2D_v2(a1, a2, a3, a4) tau_cuTexRefSetAddress2D_v2(a1, a2, a3, a4)
extern CUresult tau_cuTexRefSetAddress2D_v2(CUtexref a1, const CUDA_ARRAY_DESCRIPTOR * a2, CUdeviceptr a3, size_t a4);

#define  cuTexRefSetFormat(a1, a2, a3) tau_cuTexRefSetFormat(a1, a2, a3)
extern CUresult tau_cuTexRefSetFormat(CUtexref a1, CUarray_format a2, int a3);

#define  cuTexRefSetAddressMode(a1, a2, a3) tau_cuTexRefSetAddressMode(a1, a2, a3)
extern CUresult tau_cuTexRefSetAddressMode(CUtexref a1, int a2, CUaddress_mode a3);

#define  cuTexRefSetFilterMode(a1, a2) tau_cuTexRefSetFilterMode(a1, a2)
extern CUresult tau_cuTexRefSetFilterMode(CUtexref a1, CUfilter_mode a2);

#define  cuTexRefSetFlags(a1, a2) tau_cuTexRefSetFlags(a1, a2)
extern CUresult tau_cuTexRefSetFlags(CUtexref a1, unsigned int a2);

#define  cuTexRefGetAddress_v2(a1, a2) tau_cuTexRefGetAddress_v2(a1, a2)
extern CUresult tau_cuTexRefGetAddress_v2(CUdeviceptr * a1, CUtexref a2);

#define  cuTexRefGetArray(a1, a2) tau_cuTexRefGetArray(a1, a2)
extern CUresult tau_cuTexRefGetArray(CUarray * a1, CUtexref a2);

#define  cuTexRefGetAddressMode(a1, a2, a3) tau_cuTexRefGetAddressMode(a1, a2, a3)
extern CUresult tau_cuTexRefGetAddressMode(CUaddress_mode * a1, CUtexref a2, int a3);

#define  cuTexRefGetFilterMode(a1, a2) tau_cuTexRefGetFilterMode(a1, a2)
extern CUresult tau_cuTexRefGetFilterMode(CUfilter_mode * a1, CUtexref a2);

#define  cuTexRefGetFormat(a1, a2, a3) tau_cuTexRefGetFormat(a1, a2, a3)
extern CUresult tau_cuTexRefGetFormat(CUarray_format * a1, int * a2, CUtexref a3);

#define  cuTexRefGetFlags(a1, a2) tau_cuTexRefGetFlags(a1, a2)
extern CUresult tau_cuTexRefGetFlags(unsigned int * a1, CUtexref a2);

#define  cuTexRefCreate(a1) tau_cuTexRefCreate(a1)
extern CUresult tau_cuTexRefCreate(CUtexref * a1);

#define  cuTexRefDestroy(a1) tau_cuTexRefDestroy(a1)
extern CUresult tau_cuTexRefDestroy(CUtexref a1);

#define  cuSurfRefSetArray(a1, a2, a3) tau_cuSurfRefSetArray(a1, a2, a3)
extern CUresult tau_cuSurfRefSetArray(CUsurfref a1, CUarray a2, unsigned int a3);

#define  cuSurfRefGetArray(a1, a2) tau_cuSurfRefGetArray(a1, a2)
extern CUresult tau_cuSurfRefGetArray(CUarray * a1, CUsurfref a2);

#define  cuGraphicsUnregisterResource(a1) tau_cuGraphicsUnregisterResource(a1)
extern CUresult tau_cuGraphicsUnregisterResource(CUgraphicsResource a1);

#define  cuGraphicsSubResourceGetMappedArray(a1, a2, a3, a4) tau_cuGraphicsSubResourceGetMappedArray(a1, a2, a3, a4)
extern CUresult tau_cuGraphicsSubResourceGetMappedArray(CUarray * a1, CUgraphicsResource a2, unsigned int a3, unsigned int a4);

#define  cuGraphicsResourceGetMappedPointer_v2(a1, a2, a3) tau_cuGraphicsResourceGetMappedPointer_v2(a1, a2, a3)
extern CUresult tau_cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr * a1, size_t * a2, CUgraphicsResource a3);

#define  cuGraphicsResourceSetMapFlags(a1, a2) tau_cuGraphicsResourceSetMapFlags(a1, a2)
extern CUresult tau_cuGraphicsResourceSetMapFlags(CUgraphicsResource a1, unsigned int a2);

#define  cuGraphicsMapResources(a1, a2, a3) tau_cuGraphicsMapResources(a1, a2, a3)
extern CUresult tau_cuGraphicsMapResources(unsigned int a1, CUgraphicsResource * a2, CUstream a3);

#define  cuGraphicsUnmapResources(a1, a2, a3) tau_cuGraphicsUnmapResources(a1, a2, a3)
extern CUresult tau_cuGraphicsUnmapResources(unsigned int a1, CUgraphicsResource * a2, CUstream a3);

#define  cuGetExportTable(a1, a2) tau_cuGetExportTable(a1, a2)
extern CUresult tau_cuGetExportTable(const void ** a1, const CUuuid * a2);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /*  _TAU_cuda_H_ */
