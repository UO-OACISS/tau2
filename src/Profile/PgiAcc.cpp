#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <map>
using namespace std;

// extern "C" {
// #include <nv.h>
// }

/*
 * typedefs used to communication with cuda libraries
 */
typedef unsigned int uint1;

typedef struct{
  uint1 x,y,z;
}dim3;

typedef struct{
  uint1 x,y,z;
}uint3;

typedef int CUresult;
typedef int CUdevice;
/*
 * NOTE: cuda.h defines CUdeviceptr as 'unsigned int',
 *       but it is actually an 8-byte pointer, and must be passed
 *       as 8 bytes, so we use unsigned long here.
 */
typedef unsigned long CUdeviceptr;
typedef unsigned int CUdeviceptrx;
typedef void* CUarray;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;



TAU_GLOBAL_TIMER(pgi_acc_region_timer, "pgi accelerator region", "", TAU_DEFAULT);

static map<CUfunction,string> functionMap;

extern "C" void __pgi_cu_init_p();
extern "C" void __pgi_cu_init() {
  TAU_GLOBAL_TIMER_START(pgi_acc_region_timer);
  TAU_PROFILE("__pgi_cu_init","",TAU_DEFAULT);
  __pgi_cu_init_p();
}


extern "C" void __pgi_cu_sync_p();
extern "C" void __pgi_cu_sync() {
  TAU_GLOBAL_TIMER_START(pgi_acc_region_timer);
  TAU_PROFILE("__pgi_cu_sync","",TAU_DEFAULT);
  __pgi_cu_sync_p();
}

extern "C" void __pgi_cu_fini_p();
extern "C" void __pgi_cu_fini() {
  TAU_PROFILE("__pgi_cu_fini","",TAU_DEFAULT);
  __pgi_cu_fini_p();
}

extern "C" void __pgi_cu_module_p(void *image);
extern "C" void __pgi_cu_module(void *image) {
  TAU_PROFILE("__pgi_cu_module","",TAU_DEFAULT);
  __pgi_cu_module_p(image);
}

extern "C" CUfunction __pgi_cu_module_function_p(char *name);
extern "C" CUfunction __pgi_cu_module_function(char *name) {
  TAU_PROFILE("__pgi_cu_module_function","",TAU_DEFAULT);
  CUfunction func = __pgi_cu_module_function_p(name);
  functionMap[func] = name;
  return func;
}

extern "C" CUdeviceptr __pgi_cu_alloc_p(size_t size);
extern "C" CUdeviceptr __pgi_cu_alloc(size_t size) {
  TAU_PROFILE("__pgi_cu_alloc","",TAU_DEFAULT);
  return __pgi_cu_alloc_p(size);
}


extern "C" void __pgi_cu_upload_p( CUdeviceptr devptr, void* hostptr, size_t size );
extern "C" void __pgi_cu_upload( CUdeviceptr devptr, void* hostptr, size_t size ) {
  TAU_PROFILE("__pgi_cu_upload","",TAU_DEFAULT);
  __pgi_cu_upload_p(devptr, hostptr, size );
}


extern "C" void __pgi_cu_upload1_p(CUdeviceptr devptr, void* hostptr,
				   size_t devx, size_t hostx,
				   size_t size, size_t hoststride, size_t elementsize);
extern "C" void __pgi_cu_upload1(CUdeviceptr devptr, void* hostptr,
				 size_t devx, size_t hostx,
				 size_t size, size_t hoststride, size_t elementsize) {
  TAU_PROFILE("__pgi_cu_upload1","",TAU_DEFAULT);
  __pgi_cu_upload1_p(devptr, hostptr,
		     devx, hostx,
		     size, hoststride, elementsize);
}

extern "C" void __pgi_cu_upload2_p(CUdeviceptr devptr, void* hostptr,
				    size_t devx, size_t devy, size_t hostx, size_t hosty,
				    size_t size1, size_t size2, size_t devstride2,
				    size_t hoststride1, size_t hoststride2, size_t elementsize);
extern "C" void __pgi_cu_upload2(CUdeviceptr devptr, void* hostptr,
				  size_t devx, size_t devy, size_t hostx, size_t hosty,
				  size_t size1, size_t size2, size_t devstride2,
				  size_t hoststride1, size_t hoststride2, size_t elementsize) {
  TAU_PROFILE("__pgi_cu_upload2","",TAU_DEFAULT);
  __pgi_cu_upload2_p(devptr, hostptr,
		     devx, devy, hostx, hosty,
		     size1, size2, devstride2,
		     hoststride1, hoststride2, elementsize);
}

extern "C" void __pgi_cu_upload3_p( CUdeviceptr devptr, void* hostptr,
			      size_t devx, size_t devy, size_t devz,
			      size_t hostx, size_t hosty, size_t hostz,
			      size_t size1, size_t size2, size_t size3,
			      size_t devstride2, size_t devstride3,
			      size_t hoststride1, size_t hoststride2, size_t hoststride3,
			      size_t elementsize );

extern "C" void __pgi_cu_upload3( CUdeviceptr devptr, void* hostptr,
				  size_t devx, size_t devy, size_t devz,
				  size_t hostx, size_t hosty, size_t hostz,
				  size_t size1, size_t size2, size_t size3,
				  size_t devstride2, size_t devstride3,
				  size_t hoststride1, size_t hoststride2, size_t hoststride3,
				  size_t elementsize ) {
  TAU_PROFILE("__pgi_cu_upload3","",TAU_DEFAULT);
  __pgi_cu_upload3_p(devptr, hostptr,
		     devx, devy, devz,
		     hostx, hosty, hostz,
		     size1, size2, size3,
		     devstride2, devstride3,
		     hoststride1, hoststride2, hoststride3,
		     elementsize);
}


extern "C" void __pgi_cu_paramset_p( CUfunction func, void* ptr, unsigned long bytes, unsigned long sharedbytes );
extern "C" void __pgi_cu_paramset( CUfunction func, void* ptr, unsigned long bytes, unsigned long sharedbytes ) {
  TAU_PROFILE("__pgi_cu_paramset","",TAU_DEFAULT);
  __pgi_cu_paramset_p(func, ptr, bytes, sharedbytes );
}

extern "C" void __pgi_cu_launch_p( CUfunction func, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz );
extern "C" void __pgi_cu_launch( CUfunction func, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz ) {
  //  printf ("gridx = %d, gridy = %d, gridz = %d, blockx = %d, blocky = %d, blockz = %d\n", gridx, gridy, gridz, blockx, blocky, blockz);
  //  TAU_PROFILE("__pgi_cu_launch","",TAU_DEFAULT);
  
  string name = functionMap[func];
  char routine[4096];
  sprintf (routine, "__pgi_cu_launch(%s,gx=%d,gy=%d,gz=%d,bx=%d,by=%d,bz=%d)",name.c_str(),gridx,gridy,gridz,blockx,blocky,blockz);
 
  TAU_PROFILE_TIMER_DYNAMIC(stimer, routine, "", TAU_DEFAULT);
  TAU_PROFILE_START(stimer);
  __pgi_cu_launch_p(func, gridx, gridy, gridz, blockx, blocky, blockz);
  TAU_PROFILE_STOP(stimer);
}




extern "C" void __pgi_cu_download_p( CUdeviceptr devptr, void* hostptr, size_t size );
extern "C" void __pgi_cu_download( CUdeviceptr devptr, void* hostptr, size_t size ) {
  TAU_PROFILE("__pgi_cu_download","",TAU_DEFAULT);
  __pgi_cu_download_p(devptr, hostptr, size );
}


extern "C" void __pgi_cu_download1_p(CUdeviceptr devptr, void* hostptr,
				   size_t devx, size_t hostx,
				   size_t size, size_t hoststride, size_t elementsize);
extern "C" void __pgi_cu_download1(CUdeviceptr devptr, void* hostptr,
				 size_t devx, size_t hostx,
				 size_t size, size_t hoststride, size_t elementsize) {
  TAU_PROFILE("__pgi_cu_download1","",TAU_DEFAULT);
  __pgi_cu_download1_p(devptr, hostptr,
		     devx, hostx,
		     size, hoststride, elementsize);
}

extern "C" void __pgi_cu_download2_p(CUdeviceptr devptr, void* hostptr,
				    size_t devx, size_t devy, size_t hostx, size_t hosty,
				    size_t size1, size_t size2, size_t devstride2,
				    size_t hoststride1, size_t hoststride2, size_t elementsize);
extern "C" void __pgi_cu_download2(CUdeviceptr devptr, void* hostptr,
				  size_t devx, size_t devy, size_t hostx, size_t hosty,
				  size_t size1, size_t size2, size_t devstride2,
				  size_t hoststride1, size_t hoststride2, size_t elementsize) {
  TAU_PROFILE("__pgi_cu_download2","",TAU_DEFAULT);
  __pgi_cu_download2_p(devptr, hostptr,
		     devx, devy, hostx, hosty,
		     size1, size2, devstride2,
		     hoststride1, hoststride2, elementsize);
}

extern "C" void __pgi_cu_download3_p( CUdeviceptr devptr, void* hostptr,
			      size_t devx, size_t devy, size_t devz,
			      size_t hostx, size_t hosty, size_t hostz,
			      size_t size1, size_t size2, size_t size3,
			      size_t devstride2, size_t devstride3,
			      size_t hoststride1, size_t hoststride2, size_t hoststride3,
			      size_t elementsize );

extern "C" void __pgi_cu_download3( CUdeviceptr devptr, void* hostptr,
				  size_t devx, size_t devy, size_t devz,
				  size_t hostx, size_t hosty, size_t hostz,
				  size_t size1, size_t size2, size_t size3,
				  size_t devstride2, size_t devstride3,
				  size_t hoststride1, size_t hoststride2, size_t hoststride3,
				  size_t elementsize ) {
  TAU_PROFILE("__pgi_cu_download3","",TAU_DEFAULT);
  __pgi_cu_download3_p(devptr, hostptr,
		     devx, devy, devz,
		     hostx, hosty, hostz,
		     size1, size2, size3,
		     devstride2, devstride3,
		     hoststride1, hoststride2, hoststride3,
		     elementsize);
}


extern "C" void __pgi_cu_free_p( CUdeviceptr ptr );
extern "C" void __pgi_cu_free( CUdeviceptr ptr ) {
  TAU_PROFILE("__pgi_cu_free","",TAU_DEFAULT);
  __pgi_cu_free_p(ptr);
}


extern "C" void __pgi_cu_module_unload_p();
extern "C" void __pgi_cu_module_unload() {
  TAU_PROFILE_TIMER(timer,"__pgi_cu_module_unload","",TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  __pgi_cu_module_unload_p();
  TAU_PROFILE_STOP(timer);
  TAU_GLOBAL_TIMER_STOP();
}
