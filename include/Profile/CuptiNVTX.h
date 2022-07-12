#ifndef CUPTI_NVTX
#define CUPTI_NVTX
#include <cuda.h>
#include <cupti.h>
#include <map>
#include <iostream>
#include <sstream>
#include <nvToolsExt.h>
#include <nvToolsExtSync.h>
#include <generated_nvtx_meta.h>

std::map<nvtxDomainHandle_t, std::string>& get_domain_map();
std::string get_nvtx_message(const nvtxEventAttributes_t * eventAttrib);
NVTX_DECLSPEC int NVTX_API nvtxRangePushA (const char *message);
NVTX_DECLSPEC int NVTX_API nvtxRangePushW (const wchar_t *message);
NVTX_DECLSPEC int NVTX_API nvtxRangePushEx (const nvtxEventAttributes_t *eventAttrib);
NVTX_DECLSPEC int NVTX_API nvtxRangePop (void);
NVTX_DECLSPEC nvtxRangeId_t NVTX_API nvtxDomainRangeStartEx  (nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib);
NVTX_DECLSPEC void NVTX_API 	nvtxDomainRangeEnd (nvtxDomainHandle_t domain, nvtxRangeId_t id);

#endif //CUPTI_NVTX
