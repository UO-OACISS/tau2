/* This wrapper works for PGI 12.3+ */
#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <map>
using namespace std;


#ifndef TAU_WINDOWS
#define TAU_LLONG long
#else /* for Windows */
#define TAU_LLONG long long
#endif /* TAU_WINDOWS */

typedef size_t memint; /* check */
typedef int cuda_result;
typedef int cuda_device;
typedef unsigned long cuda_deviceptr;
typedef void* cuda_array;
typedef void* cuda_context;
typedef void* cuda_module;
typedef void* cuda_event_info;
typedef void* cuda_function;
typedef void* cuda_stream;
struct __pgi_cu_paraminfo; /* check */




//TAU_GLOBAL_TIMER(pgi_acc_region_timer, "pgi accelerator region", "", TAU_DEFAULT);

static map<cuda_function,string> functionMap;
static char* TauPgiFile; 
static char* TauPgiFunc; 
#define TAU_PGI_ACC_NAME_LEN 4096

extern "C" void __pgi_cu_init_p(char* file, char* func, long lineno, long startlineno, long endlineno);
extern "C" void __pgi_cu_init(char* file, char* func, long lineno, long startlineno, long endlineno) {
  //TAU_GLOBAL_TIMER_START(pgi_acc_region_timer);
  TauPgiFile = file;
  TauPgiFunc = func;
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_init %s (enter a region) [{%s}{%ld,0}-{%ld,0}]", TauPgiFunc, TauPgiFile, startlineno, endlineno);
  
  TAU_START(sourceinfo);
  __pgi_cu_init_p(file, func, lineno, startlineno, endlineno);
  TAU_STOP(sourceinfo);
}


// extern "C" void __pgi_cu_close_p(void);
// extern "C" void __pgi_cu_close(void) {
//   TAU_GLOBAL_TIMER_START(pgi_acc_region_timer);
//   char sourceinfo[TAU_PGI_ACC_NAME_LEN];
//   sprintf(sourceinfo, "__pgi_cu_close %s [{%s}]", TauPgiFunc, TauPgiFile);
  
//   TAU_START(sourceinfo);
//   __pgi_cu_close_p();
//   TAU_STOP(sourceinfo);
// }


typedef struct __pgi_cuda_module{
        unsigned int capability;/* major*CAPM + minor, so 2003 is 2.3 */
        unsigned int code;      /* code type, cubin, or elf, or ptx */
        size_t length;          /* length in bytes of the code module */
        char* image;            /* the actual code */
    }__pgi_cuda_module;

/* load a cuda binary to the device */
extern "C" void __pgi_cu_module3_p( void* modlist, long lineno );
extern "C" void __pgi_cu_module3( void* modlist, long lineno ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_module %s (load a cuda binary to the device) [{%s}{%ld}]", TauPgiFunc, TauPgiFile, lineno);
  TAU_START(sourceinfo);
  __pgi_cu_module3_p(modlist, lineno);
  TAU_STOP(sourceinfo);
}


/* return function handle */
extern "C" cuda_function __pgi_cu_module_function3_p( char* Name, long lineno, char* Argname, long Argsize, char* Varname, long Varsize, long SWcachesize, void* oldHandle );

extern "C" cuda_function __pgi_cu_module_function3( char* Name, long lineno, char* Argname, long Argsize, char* Varname, long Varsize, long SWcachesize, void* oldHandle ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_module_function3 %s (return function handle) name=%s argname=%s, argsize=%ld, varname=%s, varsize=%ld, SWcachesize=%ld [{%s}{%ld}]", TauPgiFunc, Name, Argname, Argsize, Varname, Varsize, SWcachesize, TauPgiFile, lineno);
  TAU_START(sourceinfo);
  cuda_function func = __pgi_cu_module_function3_p(Name, lineno, Argname, Argsize, Varname, Varsize, SWcachesize, oldHandle);
  functionMap[func] = Name;
  TAU_STOP(sourceinfo);
  return func;

}

/* allocate user data */
extern "C" cuda_deviceptr __pgi_cu_alloc_a_p(cuda_deviceptr * ptr, size_t size, long lineno, char *name, void *hostptr, long flags, long async);

extern "C" cuda_deviceptr __pgi_cu_alloc_a(cuda_deviceptr * ptr, size_t size, long lineno, char *name, void *hostptr, long flags, long async)
 {  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_alloc_a %s (allocate user data) size=%d, name=%s, flags=%ld, async=%ld, [{%s}{%ld}]", TauPgiFunc, size, name, flags, async, TauPgiFile, lineno);
  cuda_deviceptr ret;
  TAU_START(sourceinfo);
  ret = __pgi_cu_alloc_a_p(ptr, size, lineno, name, hostptr, flags, async);
  TAU_STOP(sourceinfo);
  return ret;
}

/* actually does the allocate */
extern "C" cuda_deviceptr __pgi_cu_allocx_p(cuda_deviceptr * ptr, size_t size, long lineno, char *name, void *hostptr);
extern "C" cuda_deviceptr __pgi_cu_allocx(cuda_deviceptr * ptr, size_t size, long lineno, char *name, void *hostptr)
 {  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_allocx %s (actually does the allocate) size=%d, name=%s [{%s}{%ld}]", TauPgiFunc, size, name, TauPgiFile, lineno);
  cuda_deviceptr ret;
  TAU_START(sourceinfo);   
  ret = __pgi_cu_allocx_p(ptr, size, lineno, name, hostptr);
  TAU_STOP(sourceinfo);
  return ret;
}


/* enqueue a deallocate, may be asynchronous */
extern "C" void __pgi_cu_free_a_p(cuda_deviceptr ptr, long lineno, char* name, long flags, long async);
extern "C" void __pgi_cu_free_a(cuda_deviceptr ptr, long lineno, char* name, long flags, long async) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_free_a %s (enqueue a deallocate, may be asynchronous) name=%s, flags=%ld, async=%ld [{%s}{%ld}]", TauPgiFunc, name, flags, async, TauPgiFile, lineno);
  TAU_START(sourceinfo);
  __pgi_cu_free_a_p(ptr, lineno, name, flags, async);
  TAU_STOP(sourceinfo);
}

/* actually do the deallocate */
extern "C" void __pgi_cu_freex_p(cuda_deviceptr ptr, long lineno, char* name);
extern "C" void __pgi_cu_freex(cuda_deviceptr ptr, long lineno, char* name) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_free %s (actually do the deallocate) name=%s [{%s}{%ld}]", TauPgiFunc, name, TauPgiFile, lineno);
  TAU_START(sourceinfo);
  __pgi_cu_freex_p(ptr, lineno, name);
  TAU_STOP(sourceinfo);
}

/* allocate a compiler temp */
extern "C" cuda_deviceptr __pgi_cu_alloc_p(size_t size, long lineno, char *name);
extern "C" cuda_deviceptr __pgi_cu_alloc(size_t size, long lineno, char *name) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_alloc %s (allocate a compiler temp) [{%s}{%ld}]", TauPgiFunc, TauPgiFile, lineno);
  cuda_deviceptr ret;
  TAU_START(sourceinfo);
  ret = __pgi_cu_alloc_p(size, lineno, name);
  TAU_STOP(sourceinfo);
  return ret;
}

/* enqueue launch of kernel */
extern "C" void __pgi_cu_launch_a_p( cuda_function func, void* ptr, memint bytes, memint sharedbytes, __pgi_cu_paraminfo* descr, int nargs, memint* sharedarray, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int flags, long lineno, long async );

extern "C" void __pgi_cu_launch_a( cuda_function func, void* ptr, memint bytes, memint sharedbytes, __pgi_cu_paraminfo* descr, int nargs, memint* sharedarray, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int flags, long lineno, long async ) {

  string name = functionMap[func];
  char routine[TAU_PGI_ACC_NAME_LEN];
  snprintf (routine, sizeof(routine),  "__pgi_cu_launch_a %s (enqueue launch of kernel) (%s,bytes=%ld,sharedbytes=%ld,nargs=%d, gx=%d,gy=%d,gz=%d,bx=%d,by=%d,bz=%d,flags=%ld,async=%ld) [{%s}{%ld}]",
    TauPgiFunc, name.c_str(), bytes, sharedbytes, nargs, gridx, gridy, gridz, 
    blockx, blocky, blockz, flags, async,
    TauPgiFile, lineno);
 
  TAU_START(routine);

  __pgi_cu_launch_a_p( func, ptr, bytes, sharedbytes, descr, nargs, 
	sharedarray, gridx, gridy, gridz, blockx, blocky, blockz, flags, 
	lineno, async );

  TAU_STOP(routine);
}

/* actually launch of kernel */
extern "C" void __pgi_cu_launchx_p( cuda_function func, void* ptr, memint bytes, memint sharedbytes, __pgi_cu_paraminfo* descr, int nargs, memint* sharedarray, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int flags, long lineno, cuda_event_info* eventinfo);

extern "C" void __pgi_cu_launchx( cuda_function func, void* ptr, memint bytes, memint sharedbytes, __pgi_cu_paraminfo* descr, int nargs, memint* sharedarray, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int flags, long lineno, cuda_event_info* eventinfo) {

  string name = functionMap[func];
  char routine[TAU_PGI_ACC_NAME_LEN];
  snprintf (routine, sizeof(routine),  "__pgi_cu_launchx_p %s (actually launch of kernel) (%s,bytes=%ld,sharedbytes=%ld,nargs=%d, gx=%d,gy=%d,gz=%d,bx=%d,by=%d,bz=%d,flags=%ld) [{%s}{%ld}]",
    TauPgiFunc, name.c_str(), bytes, sharedbytes, nargs, gridx, gridy, gridz,     blockx, blocky, blockz, flags, 
    TauPgiFile, lineno);

  TAU_START(routine);

  __pgi_cu_launchx_p( func, ptr, bytes, sharedbytes, descr, nargs,         
	sharedarray, gridx, gridy, gridz, blockx, blocky, blockz, flags, lineno, 
	eventinfo ); 
  TAU_STOP(routine);

}


extern "C" void __pgi_cu_launch2_p(cuda_function func, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int flag, long lineno );
extern "C" void __pgi_cu_launch2(cuda_function func, int gridx, int gridy, int gridz, int blockx, int blocky, int blockz, int flag, long lineno ) {

  string name = functionMap[func];
  char routine[TAU_PGI_ACC_NAME_LEN];
  snprintf (routine, sizeof(routine),  "__pgi_cu_launch %s (%s,gx=%d,gy=%d,gz=%d,bx=%d,by=%d,bz=%d,flag=%d) [{%s}{%ld}]",
    TauPgiFunc, name.c_str(), gridx, gridy, gridz, blockx, blocky, blockz, flag,
    TauPgiFile, lineno);

  TAU_START(routine);
  __pgi_cu_launch2_p(func, gridx, gridy, gridz, blockx, blocky, blockz, flag, lineno);
  TAU_STOP(routine);
}

/* old routines */
/* enqueue end of block of data moves */
/* 
extern "C" void __pgi_cu_datadone_a_p( long a, long b);

extern "C" void __pgi_cu_datadone_a( long a, long b);
extern "C" void __pgi_cu_datadone( );
extern "C" void __pgi_cu_upstartx( cuda_event_info*, void* );
extern "C" cuda_deviceptr __pgi_cu_mirroralloc( size_t size, size_t elemsize, long lineno, char* name );
extern "C" void __pgi_cu_mirrordealloc( cuda_deviceptr ptr, long lineno, char* name );
extern "C" void __pgi_cu_datastart_a( long, long );
extern "C" void __pgi_cu_datastart( );
extern "C" void __pgi_cu_datastartx( cuda_event_info*, void* );

*/


/* check */
struct __pgi_nv_pdata;
typedef struct __pgi_nv_data{
        size_t devx, devstride, hostx, hoststride, size;
    }__pgi_nv_data;

typedef struct __pgi_nv_xdata{
        TAU_LLONG devx, devstride, hostx, hoststride, size, extent;
    }__pgi_nv_xdata; 

/* enqueue a data download, may be asynchrnonous */
extern "C" void __pgi_cu_downloadx_a_p( cuda_deviceptr devptr, void* hostptr, int dims,
        __pgi_nv_xdata* desc, long elementsize, long lineno, char* name, long flags, long async);

extern "C" void __pgi_cu_downloadx_a( cuda_deviceptr devptr, void* hostptr, int dims,
        __pgi_nv_xdata* desc, long elementsize, long lineno, char* name, long flags, long async) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_downloadx_a %s (enqueue a data download, may be asynchrnonous) var=%s, dims=%d, desc.devx=%ld, desc.devstride=%ld, desc.hoststride=%ld, desc.size=%ld, desc.extent=%ld, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]", 
	TauPgiFunc, name, dims, desc->devx, desc->devstride, 
        desc->hoststride, desc->size, desc->extent, elementsize, flags, async,
        TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_cu_downloadx_a_p(devptr, hostptr, dims, desc, elementsize, lineno, name, flags, async);
  TAU_STOP(sourceinfo);
}

/* actually do the data download */
extern "C" void __pgi_cu_downloadxx_p( cuda_deviceptr devptr, void* hostptr, int dims,
        __pgi_nv_xdata* desc, long elementsize, long lineno, char* name, long flags, cuda_event_info* eventinfo);

extern "C" void __pgi_cu_downloadxx( cuda_deviceptr devptr, void* hostptr, int dims,
        __pgi_nv_xdata* desc, long elementsize, long lineno, char* name, long flags, cuda_event_info* eventinfo) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_downloadxx %s (actually do the data download) var=%s, dims=%d, desc.devx=%ld, desc.devstride=%ld, desc.hoststride=%ld, desc.size=%ld, desc.extent=%ld, elementsize=%ld, flags=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, desc->devx, desc->devstride,
        desc->hoststride, desc->size, desc->extent, elementsize, flags,         TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_cu_downloadxx_p(devptr, hostptr, dims, desc, elementsize, lineno, name, flags, eventinfo);
  TAU_STOP(sourceinfo);
}

/* allocate, upload, check for or add to presence at start of region */
extern "C" void __pgi_acc_dataon_p( cuda_deviceptr* devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async );

extern "C" void __pgi_acc_dataon( cuda_deviceptr* devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async ) {

  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
/*
  sprintf(sourceinfo, "__pgi_acc_dataon %s (allocate, upload, check for or add to presence at start of region) var=%s, dims=%d, desc.devx=%ld, desc.devstride=%ld, desc.hoststride=%ld, desc.size=%ld, desc.extent=%ld, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, desc->devx, desc->devstride,
        desc->hoststride, desc->size, desc->extent, elementsize, flags, async, 
	TauPgiFile, lineno);
*/
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_acc_dataon %s (allocate, upload, check for or add to presence at start of region) var=%s, dims=%d, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, elementsize, flags, async, 
	TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_acc_dataon_p(devptr, hostptr, dims, desc, elementsize, lineno, name, flags, async);
  TAU_STOP(sourceinfo);

}

/* deallocate, download, clear presence at end of region */
extern "C" void __pgi_acc_dataoff_p( cuda_deviceptr* devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async );

extern "C" void __pgi_acc_dataoff( cuda_deviceptr* devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_acc_dataoff %s (deallocate, download, clear presence at end of region) var=%s, dims=%d, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, elementsize, flags, async, TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_acc_dataoff_p(devptr, hostptr, dims, desc, elementsize, lineno, name, flags, async);
  TAU_STOP(sourceinfo);

}

/* enqueue upload data */
extern "C" void __pgi_acc_dataup_p( cuda_deviceptr devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async );

extern "C" void __pgi_acc_dataup( cuda_deviceptr devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_acc_dataup %s (enqueue upload data) var=%s, dims=%d, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, elementsize, flags, async, TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_acc_dataup_p(devptr, hostptr, dims, desc, elementsize, lineno, name, flags, async);
  TAU_STOP(sourceinfo);

}

/* actually upload data */
extern "C" void __pgi_acc_dataupx_p( cuda_deviceptr devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, cuda_event_info* eventinfo );

extern "C" void __pgi_acc_dataupx( cuda_deviceptr devptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, cuda_event_info* eventinfo ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_acc_dataupx %s (actually upload data) var=%s, dims=%d, elementsize=%ld, [{%s}{%ld}]",
        TauPgiFunc, name, dims, elementsize, TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_acc_dataupx_p(devptr, hostptr, dims, desc, elementsize, lineno, name, eventinfo);
  TAU_STOP(sourceinfo);


}

/* enqueue download data */
extern "C" void __pgi_acc_datadown_p ( cuda_deviceptr indevptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async );

extern "C" void __pgi_acc_datadown ( cuda_deviceptr indevptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, long async ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_acc_datadown %s (enqueue download data) var=%s, dims=%d, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, elementsize, flags, async, TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_acc_datadown_p(indevptr, hostptr, dims, desc, elementsize, lineno, name, flags, async);
  TAU_STOP(sourceinfo);

}

/* actually download data */
/* THERE IS NO __pgi_acc_datadownx_p in PGI 12.3 */

#ifdef TAU_PGI_ACC_DATADOWNX
extern "C" void __pgi_acc_datadownx_p ( cuda_deviceptr indevptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, cuda_event_info* eventinfo);
 extern "C" void __pgi_acc_datadownx ( cuda_deviceptr indevptr, void* hostptr, int dims, __pgi_nv_pdata* desc, long elementsize, long lineno, char* name, long flags, cuda_event_info* eventinfo) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_acc_datadownx %s (actually download data) var=%s, dims=%d, elementsize=%ld, flags=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, elementsize, flags, TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_acc_datadownx_p(indevptr, hostptr, dims, desc, elementsize, lineno, name, flags, eventinfo);

  TAU_STOP(sourceinfo);

}
#endif /* TAU_PGI_ACC_DATADOWNX */

/* enqueue upload of compiler temp data */ 
extern "C" void __pgi_cu_uploadc_a_p( char* name, void* hostptr, size_t size, long lineno, size_t offset, long flags, long async );

extern "C" void __pgi_cu_uploadc_a( char* name, void* hostptr, size_t size, long lineno, size_t offset, long flags, long async ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_uploadc_a %s (enqueue upload of compiler temp data) var=%s, size=%ld, offset=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, size, offset, flags, async,
        TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_cu_uploadc_a_p(name, hostptr, size, lineno, offset, flags, async);

  TAU_STOP(sourceinfo);

}

/* actually upload compiler temp data */ 
extern "C" void __pgi_cu_uploadcx_p( char* name, void* hostptr, size_t size, long lineno, size_t offset, cuda_event_info* eventinfo);

extern "C" void __pgi_cu_uploadcx( char* name, void* hostptr, size_t size, long lineno, size_t offset, cuda_event_info* eventinfo ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_uploadcx %s (actually upload compiler temp data) var=%s, size=%ld, offset=%ld, [{%s}{%ld}]",
        TauPgiFunc, name, size, offset, TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_cu_uploadcx_p(name, hostptr, size, lineno, offset, eventinfo);

  TAU_STOP(sourceinfo);

}
/* enqueue download of compiler temp data */ 
extern "C" void __pgi_cu_downloadc_a_p( char* name, void* hostptr, size_t size, long lineno, size_t offset, long flags, long async );

extern "C" void __pgi_cu_downloadc_a( char* name, void* hostptr, size_t size, long lineno, size_t offset, long flags, long async ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_downloadc %s (enqueue download of compiler temp data) var=%s, size=%ld, offset=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, size, offset, flags, async,
        TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_cu_downloadc_a_p(name, hostptr, size, lineno, offset, flags, async);

  TAU_STOP(sourceinfo);

}

/* actually download compiler temp data */
extern "C" void __pgi_cu_downloadcx_p( char* name, void* hostptr, size_t size, long lineno, size_t offset, cuda_event_info* eventinfo); 

extern "C" void __pgi_cu_downloadcx( char* name, void* hostptr, size_t size, long lineno, size_t offset, cuda_event_info* eventinfo) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_downloadcx %s (actually download compiler temp data) var=%s, size=%ld [{%s}{%ld}]",
        TauPgiFunc, name, size,
        TauPgiFile, lineno);

  TAU_START(sourceinfo);

  __pgi_cu_downloadcx_p(name, hostptr, size, lineno, offset, eventinfo);

  TAU_STOP(sourceinfo);


}

/* enqueue a data upload, may be asynchronous */
extern "C" void __pgi_cu_uploadx_a_p( cuda_deviceptr devptr, void* hostptr, 
	int dims, __pgi_nv_xdata* desc, long elementsize, long lineno, 
	char* name, long flags, long async );

extern "C" void __pgi_cu_uploadx_a( cuda_deviceptr devptr, void* hostptr, 
	int dims, __pgi_nv_xdata* desc, long elementsize, long lineno, 
	char* name, long flags, long async ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_uploadx_a %s (enqueue a data upload, may be asynchronous) var=%s, dims=%d, desc.devx=%ld, desc.devstride=%ld, desc.hoststride=%ld, desc.size=%ld, elementsize=%ld, flags=%ld, async=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, desc->devx, desc->devstride,
        desc->hoststride, desc->size, elementsize, flags, async, TauPgiFile, lineno);

  TAU_START(sourceinfo);
  __pgi_cu_uploadx_a_p( devptr, hostptr, dims, desc, elementsize, lineno, name, flags, async);
  TAU_STOP(sourceinfo);

}

/* actually do the data upload */
extern "C" void __pgi_cu_uploadxx_p( cuda_deviceptr devptr, void* hostptr,
        int dims, __pgi_nv_xdata* desc, long elementsize, long lineno,
        char* name, long flags, cuda_event_info* eventinfo);

extern "C" void __pgi_cu_uploadxx( cuda_deviceptr devptr, void* hostptr,
        int dims, __pgi_nv_xdata* desc, long elementsize, long lineno,
        char* name, long flags, cuda_event_info* eventinfo ) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  snprintf(sourceinfo, sizeof(sourceinfo),  "__pgi_cu_uploadxx %s (actually do the data upload) var=%s, dims=%d, desc.devx=%ld, desc.devstride=%ld, desc.hoststride=%ld, desc.size=%ld, elementsize=%ld, flags=%ld [{%s}{%ld}]",
        TauPgiFunc, name, dims, desc->devx, desc->devstride,
        desc->hoststride, desc->size, elementsize, flags, TauPgiFile, lineno);

  TAU_START(sourceinfo);
  __pgi_cu_uploadxx_p( devptr, hostptr, dims, desc, elementsize, lineno, name, flags, eventinfo);
  TAU_STOP(sourceinfo);

}



/*  These routines are supposed to be part of 8.0-6, but they're not there! */

/*
extern "C" void __pgi_cu_uploadc_p(char *name, void* hostptr, size_t size, long lineno);
extern "C" void __pgi_cu_uploadc(char *name, void* hostptr, size_t size, long lineno) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  sprintf(sourceinfo, "__pgi_cu_uploadc %s [{%s}{%ld}]", TauPgiFunc, TauPgiFile, lineno);
  TAU_START(sourceinfo);
  __pgi_cu_uploadc_p(name, hostptr, size, lineno);
  TAU_STOP(sourceinfo);
}


typedef struct __pgi_nv_data {
  size_t devx, devstride, hostx, hoststride, size;
} __pgi_nv_data;



extern "C" void __pgi_cu_uploadn_p(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name);
extern "C" void __pgi_cu_uploadn(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  sprintf(sourceinfo, "__pgi_cu_uploadn %s [{%s}]", TauPgiFunc, TauPgiFile);
  TAU_START(sourceinfo);
  __pgi_cu_uploadn_p(devptr, hostptr, dims, desc, elementsize, lineno, name);
  TAU_STOP(sourceinfo);
}


extern "C" void __pgi_cu_uploadp_p(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name);
extern "C" void __pgi_cu_uploadp(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  sprintf(sourceinfo, "__pgi_cu_uploadp %s [{%s}]", TauPgiFunc, TauPgiFile);
  TAU_START(sourceinfo);
  __pgi_cu_uploadp_p(devptr, hostptr, dims, desc, elementsize, lineno, name);
  TAU_STOP(sourceinfo);
}


extern "C" void __pgi_cu_downloadn_p(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name);
extern "C" void __pgi_cu_downloadn(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  sprintf(sourceinfo, "__pgi_cu_downloadn %s [{%s}]", TauPgiFunc, TauPgiFile);
  TAU_START(sourceinfo);
  __pgi_cu_downloadn_p(devptr, hostptr, dims, desc, elementsize, lineno, name);
  TAU_STOP(sourceinfo);
}


extern "C" void __pgi_cu_downloadp_p(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name);
extern "C" void __pgi_cu_downloadp(cuda_deviceptr devptr, void* hostptr, int dims, 
				 __pgi_nv_data* desc, size_t elementsize, long lineno, char* name) {
  char sourceinfo[TAU_PGI_ACC_NAME_LEN];
  sprintf(sourceinfo, "__pgi_cu_downloadp %s [{%s}]", TauPgiFunc, TauPgiFile);
  TAU_START(sourceinfo);
  __pgi_cu_downloadp_p(devptr, hostptr, dims, desc, elementsize, lineno, name);
  TAU_STOP(sourceinfo);
}

*/
