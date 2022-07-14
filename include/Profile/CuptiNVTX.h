#ifndef CUPTI_NVTX
#define CUPTI_NVTX
#include <cuda.h>
#include <cupti.h>
#include <map>
#include <stack>
#include <iostream>
#include <sstream>
#include <nvToolsExt.h>
#include <nvToolsExtSync.h>
#include <generated_nvtx_meta.h>
/*
https://gitlab.com/nvidia/headers/cuda-individual/nvtx/-/blob/0570d3f68e5bd2f3c97b0c928930534ca0230f9a/nvToolsExt.h
https://nvidia.github.io/NVTX/doxygen/group___m_a_r_k_e_r_s___a_n_d___r_a_n_g_e_s.html
https://github.com/UO-OACISS/apex/blob/eda271d93bd6871028cbcc037ce173e29bf9e934/src/apex/cupti_trace.cpp
https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvtx_api_events.htm
https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx

https://github.com/UO-OACISS/apex/blob/eda271d93bd6871028cbcc037ce173e29bf9e934/src/apex/cupti_trace.cpp


store_sync_counter_data(nullptr, tmp, 0); --> Tau_userevent
https://github.com/UO-OACISS/tau2/blob/dd9689beb1360632576ac38deaa0dc5de62e077e/src/Profile/TauCAPI.cpp#L2036-L2040
https://gitlab.com/nvidia/headers/cuda-individual/nvtx/-/blob/0570d3f68e5bd2f3c97b0c928930534ca0230f9a/nvToolsExt.h#L481-491
*/

//Apex CUpti_trace.cpp
/* Fun!  CUPTI doesn't do callbacks for end or push events.  Wheeeeee
 * So, what we'll do is wrap the functions instead of having callbacks. */

#define TAU_BROKEN_CUPTI_NVTX_PUSH_POP 1


// Ranges defined by Start/End can overlap. 
// Therefore, they are not adapted for TAU and no measurement is performed.
// Ranges defined by Push/Pop are safe.




//Typedefs for wrappers
#ifdef TAU_BROKEN_CUPTI_NVTX_PUSH_POP

typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePushA_p)(const char * message);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePushW_p)(const wchar_t * message);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePushEx_p)(const nvtxEventAttributes_t *eventAttrib);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxRangePop_p)(void);
typedef nvtxDomainHandle_t (*nvtxDomainCreateA_p)(const char * name);
typedef nvtxDomainHandle_t (*nvtxDomainCreateW_p)(const wchar_t* name);
typedef void (*nvtxDomainDestroy_p)(nvtxDomainHandle_t domain);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxDomainRangePushEx_p)(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);
typedef NVTX_DECLSPEC int NVTX_API (*nvtxDomainRangePop_p)(nvtxDomainHandle_t domain);

#endif //TAU_BROKEN_CUPTI_NVTX_PUSH_POP


//Auxiliary 

std::map<nvtxDomainHandle_t, std::string>& get_domain_map();
//std::map<nvtxRangeId_t, std::string>& get_range_map();
std::stack<std::string>& get_range_stack();

static void * get_system_function_handle(char const * name, void * caller);
std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib);

void tau_nvtxRangePush (const std::string name);
void tau_nvtxRangePop ();




//Wrappers & Interceptors


#ifdef TAU_BROKEN_CUPTI_NVTX_PUSH_POP
int tau_nvtxRangePushA_wrapper (nvtxRangePushA_p nvtxRangePushA_call, const char * message);
int tau_nvtxRangePushW_wrapper (nvtxRangePushW_p nvtxRangePushW_call, const wchar_t *message);
int tau_nvtxRangePushEx_wrapper (nvtxRangePushEx_p nvtxRangePushEx_call, const nvtxEventAttributes_t *eventAttrib);
int tau_nvtxRangePop_wrapper (nvtxRangePop_p nvtxRangePop_call);

NVTX_DECLSPEC int NVTX_API nvtxRangePushA (const char *message);
NVTX_DECLSPEC int NVTX_API nvtxRangePushW (const wchar_t *message);
NVTX_DECLSPEC int NVTX_API nvtxRangePushEx (const nvtxEventAttributes_t *eventAttrib);
NVTX_DECLSPEC int NVTX_API nvtxRangePop (void);

//TO TEST

nvtxDomainHandle_t tau_nvtxDomainCreateA_wrapper (nvtxDomainCreateA_p nvtxDomainCreateA_call, const char* name);
nvtxDomainHandle_t tau_nvtxDomainCreateW_wrapper (nvtxDomainCreateW_p nvtxDomainCreateW_call, const wchar_t* name);
NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateA(const char* name);
NVTX_DECLSPEC nvtxDomainHandle_t NVTX_API nvtxDomainCreateW(const wchar_t* name);

void tau_nvtxDomainDestroy_wrapper (nvtxDomainDestroy_p nvtxDomainDestroy_call, nvtxDomainHandle_t domain);
NVTX_DECLSPEC void NVTX_API nvtxDomainDestroy(nvtxDomainHandle_t domain);


int tau_nvtxDomainRangePushEx_wrapper (nvtxDomainRangePushEx_p nvtxDomainRangePushEx_call, nvtxDomainHandle_t domain, const nvtxEventAttributes_t* eventAttrib);
int tau_nvtxDomainRangePop_wrapper (nvtxDomainRangePop_p nvtxDomainRangePop_call, const wchar_t *message);
NVTX_DECLSPEC int NVTX_API nvtxDomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);
NVTX_DECLSPEC int NVTX_API nvtxDomainRangePop(nvtxDomainHandle_t domain);

#endif //TAU_BROKEN_CUPTI_NVTX_PUSH_POP


//Handle for CUPTI_CB_DOMAIN_NVTX
void handle_nvtx_callback(CUpti_CallbackId id, const void *cbdata);

#endif //CUPTI_NVTX
