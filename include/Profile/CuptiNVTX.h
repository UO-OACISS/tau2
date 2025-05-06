#ifndef CUPTI_NVTX
#define CUPTI_NVTX
#include <cuda.h>
#include <cupti.h>
#include <map>
#include <stack>
#include <iostream>
#include <sstream>
//nvtx3 was available since 10.0, but CUpti_NvtxData does not have
// functionReturnValue, so only use the nvtx3 since 12.0
#if CUDA_VERSION  >= 12000
#include "nvtx3/nvToolsExt.h"
#include "nvtx3/nvToolsExtSync.h"
#include "nvtx3/nvToolsExtCuda.h"
#include "nvtx3/nvToolsExtCudaRt.h"
#else
#include <nvToolsExt.h>
#include <nvToolsExtSync.h>
#warning "Using TAU_BROKEN_CUPTI_NVTX_CALLBACKS"
#define TAU_BROKEN_CUPTI_NVTX_CALLBACKS
#endif

#include <generated_nvtx_meta.h>

// Ranges defined by Start/End can overlap. 
// Therefore, they are not adapted for TAU and no measurement is performed.
// Ranges defined by Push/Pop are safe.
//Apex CUpti_trace.cpp
/* Fun!  CUPTI doesn't do callbacks for end or push events.  Wheeeeee
 * So, what we'll do is wrap the functions instead of having callbacks. */
#ifdef TAU_BROKEN_CUPTI_NVTX_CALLBACKS

typedef nvtxDomainHandle_t (*nvtxDomainCreateA_p)(const char * name);
typedef nvtxDomainHandle_t (*nvtxDomainCreateW_p)(const wchar_t* name);


typedef NVTX_DECLSPEC int NVTX_API (*nvtxDomainRangePushEx_p)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);

//Typedefs for wrappers


typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePushA_p)(const char * message);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePushW_p)(const wchar_t * message);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePushEx_p)(const nvtxEventAttributes_t *eventAttrib);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePop_p)(void);

typedef void (*nvtxDomainDestroy_p)(nvtxDomainHandle_t domain);

typedef NVTX_DECLSPEC int NVTX_API (*nvtxDomainRangePop_p)(nvtxDomainHandle_t domain);

#endif //TAU_BROKEN_CUPTI_NVTX_CALLBACKS


//Auxiliary 

std::map<nvtxDomainHandle_t, std::string>& get_domain_map();
//std::map<nvtxRangeId_t, std::string>& get_range_map();
std::stack<std::string>& get_range_stack();
std::map<nvtxStringHandle_t, std::string>* get_domain_string_map(nvtxDomainHandle_t handle);
static void * get_system_function_handle(char const * name, void * caller);
std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib, nvtxDomainHandle_t handle);

void tau_nvtxRangePush (const std::string name);
void tau_nvtxRangePop ();




//Wrappers & Interceptors
#ifdef TAU_BROKEN_CUPTI_NVTX_CALLBACKS
nvtxDomainHandle_t tau_nvtxDomainCreateA_wrapper (nvtxDomainCreateA_p nvtxDomainCreateA_call, const char* name);
nvtxDomainHandle_t tau_nvtxDomainCreateW_wrapper (nvtxDomainCreateW_p nvtxDomainCreateW_call, const wchar_t* name);
NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateA(const char* name);
NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateW(const wchar_t* name);


int tau_nvtxDomainRangePushEx_wrapper (nvtxDomainRangePushEx_p nvtxDomainRangePushEx_call, nvtxDomainHandle_t domain, const nvtxEventAttributes_t* eventAttrib);
NVTX_DECLSPEC int NVTX_API nvtxDomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);


int tau_nvtxRangePushA_wrapper (nvtxRangePushA_p nvtxRangePushA_call, const char * message);
int tau_nvtxRangePushW_wrapper (nvtxRangePushW_p nvtxRangePushW_call, const wchar_t *message);
int tau_nvtxRangePushEx_wrapper (nvtxRangePushEx_p nvtxRangePushEx_call, const nvtxEventAttributes_t *eventAttrib);
int tau_nvtxRangePop_wrapper (nvtxRangePop_p nvtxRangePop_call);

NVTX_DECLSPEC int NVTX_API nvtxRangePushA (const char *message);
NVTX_DECLSPEC int NVTX_API nvtxRangePushW (const wchar_t *message);
NVTX_DECLSPEC int NVTX_API nvtxRangePushEx (const nvtxEventAttributes_t *eventAttrib);
NVTX_DECLSPEC int NVTX_API nvtxRangePop (void);


void tau_nvtxDomainDestroy_wrapper (nvtxDomainDestroy_p nvtxDomainDestroy_call, nvtxDomainHandle_t domain);
NVTX_DECLSPEC void NVTX_API nvtxDomainDestroy(nvtxDomainHandle_t domain);



int tau_nvtxDomainRangePop_wrapper (nvtxDomainRangePop_p nvtxDomainRangePop_call, const wchar_t *message);

NVTX_DECLSPEC int NVTX_API nvtxDomainRangePop(nvtxDomainHandle_t domain);

#endif //TAU_BROKEN_CUPTI_NVTX_CALLBACKS


//Handle for CUPTI_CB_DOMAIN_NVTX
extern void handle_nvtx_callback(CUpti_CallbackId id, const void *cbdata);


#endif //CUPTI_NVTX
