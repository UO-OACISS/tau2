#ifndef CUPTI_PCSAMPLING_H
#define CUPTI_PCSAMPLING_H

// System headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <sstream>
// CUDA headers
#include <cuda.h>

#define CUDA_MIN 12050
#if CUDA_VERSION  >= CUDA_MIN
#include <string>
#include <vector>
#include <inttypes.h>
#include <unordered_set>
#include <mutex>
#include <map>
#include <queue>
#include <thread>
#include <cxxabi.h>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#include "detours.h"
#else
#include <unistd.h>
#include <pthread.h>
#endif



// CUPTI headers
#include <cupti_pcsampling_util.h>
#include <cupti_pcsampling.h>
#include <cupti.h>
#include "Profile/Profiler.h"
#include <Profile/TauEnv.h>

extern "C" x_uint64 TauTraceGetTimeStamp(int tid);
extern "C" void Tau_metadata_task(const char *name, const char* value, int tid);

using namespace CUPTI::PcSamplingUtil;

// Macros
#define THREAD_SLEEP_TIME 100 // in ms

#define DRIVER_API_CALL(apiFunctionCall)                                            \
do                                                                                  \
{                                                                                   \
    CUresult _status = apiFunctionCall;                                             \
    if (_status != CUDA_SUCCESS)                                                    \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuGetErrorString(_status, &pErrorString);                                   \
                                                                                    \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Function "   \
        << #apiFunctionCall << " failed with error(" << _status << "): "            \
        << pErrorString << ".\n\n";                                                 \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define RUNTIME_API_CALL(apiFunctionCall)                                           \
do                                                                                  \
{                                                                                   \
    cudaError_t _status = apiFunctionCall;                                          \
    if (_status != cudaSuccess)                                                     \
    {                                                                               \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Function "   \
        << #apiFunctionCall << " failed with error(" << _status << "): "            \
        << cudaGetErrorString(_status) << ".\n\n";                                  \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CUPTI_API_CALL(apiFunctionCall)                                             \
do                                                                                  \
{                                                                                   \
    CUptiResult _status = apiFunctionCall;                                          \
    if (_status != CUPTI_SUCCESS)                                                   \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuptiGetResultString(_status, &pErrorString);                               \
                                                                                    \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Function "   \
        << #apiFunctionCall << " failed with error(" << _status << "): "            \
        << pErrorString << ".\n\n";                                                 \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CUPTI_API_CALL_VERBOSE(apiFunctionCall)                                     \
do                                                                                  \
{                                                                                   \
    std::cout << "Calling CUPTI API: " << #apiFunctionCall << "\n";                 \
                                                                                    \
    CUptiResult _status = apiFunctionCall;                                          \
    if (_status != CUPTI_SUCCESS)                                                   \
    {                                                                               \
        const char *pErrorString;                                                   \
        cuptiGetResultString(_status, &pErrorString);                               \
                                                                                    \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Function "   \
        << #apiFunctionCall << " failed with error(" << _status << "): "            \
        << pErrorString << ".\n\n";                                                 \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CUPTI_UTIL_CALL(apiFunctionCall)                                            \
do                                                                                  \
{                                                                                   \
    CUptiUtilResult _status = apiFunctionCall;                                      \
    if (_status != CUPTI_UTIL_SUCCESS)                                              \
    {                                                                               \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Function "   \
        << #apiFunctionCall << " failed with error: " << _status << "\n\n";         \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define NVPW_API_CALL(apiFunctionCall)                                              \
do                                                                                  \
{                                                                                   \
    NVPA_Status _status = apiFunctionCall;                                          \
    if (_status != NVPA_STATUS_SUCCESS)                                             \
    {                                                                               \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Function "   \
        << #apiFunctionCall << " failed with error: " << _status << "\n\n";         \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define MEMORY_ALLOCATION_CALL(variable)                                            \
do                                                                                  \
{                                                                                   \
    if (variable == NULL)                                                           \
    {                                                                               \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ <<                 \
        " Memory allocation failed.\n\n";                                           \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CHECK_CONDITION(condition)                                                  \
do                                                                                  \
{                                                                                   \
    if (!(condition))                                                               \
    {                                                                               \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Condition "  \
        << #condition << " failed.\n\n";                                            \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

#define CHECK_INTEGER_CONDITION(argument1, operator, argument2)                     \
do                                                                                  \
{                                                                                   \
    if (!(argument1 operator argument2))                                            \
    {                                                                               \
        std::cerr << "\n\nError:" << __FILE__ << ":" << __LINE__ << ": Condition "  \
        << #argument1 << " " << #operator << " " << #argument2 << " fails. " <<     \
        #argument1 << " = " << argument1 << ", " << #argument2 << " = " <<          \
        argument2 << "\n\n";                                                        \
                                                                                    \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

typedef struct ModuleDetails_st
{
    uint32_t cubinSize;
    void *pCubinImage;
} ModuleDetails;

typedef struct ContextInfo_st
{
    uint32_t contextUid;
    CUpti_PCSamplingData pcSamplingData;
    std::vector<CUpti_PCSamplingConfigurationInfo> pcSamplingConfigurationInfo;
    PcSamplingStallReasons pcSamplingStallReasons;
} ContextInfo;

typedef struct TAUCuptiSamples_st
{
    uint64_t cubinCrc;
    uint64_t pcOffset;
    std::string functionName;
    size_t stallReasonCount;
    std::vector<std::pair<uint32_t, uint32_t>> stallReason;
    uint32_t contextUid;
} TAUCuptiSamples;


typedef struct TAUCuptiStalls_st
{
    size_t stallReasonCount;
    std::map<uint32_t, uint32_t> stallReason;
} TAUCuptiStalls;

typedef struct TAUCuptiIdSamples_st
{
    uint64_t cubinCrc;
    uint64_t pcOffset;
    std::string functionName;
    uint32_t contextUid;
    uint32_t functionIndex;

    bool operator!=(const TAUCuptiIdSamples_st &o) const{
        return std:: tie(cubinCrc, functionIndex, pcOffset, contextUid)
                != std:: tie(o.cubinCrc, o.functionIndex, o.pcOffset, o.contextUid);
    }
    bool operator==(const TAUCuptiIdSamples_st &o) const{
        return std:: tie(cubinCrc, functionIndex, pcOffset, contextUid)
                == std:: tie(o.cubinCrc, o.functionIndex, o.pcOffset, o.contextUid);
    }
    bool operator>(const TAUCuptiIdSamples_st &o) const{
        return std:: tie(cubinCrc, functionIndex, pcOffset, contextUid)
                > std:: tie(o.cubinCrc, o.functionIndex, o.pcOffset, o.contextUid);
    }
    bool operator<(const TAUCuptiIdSamples_st &o) const{
        return std:: tie(cubinCrc, functionIndex, pcOffset, contextUid)
                < std:: tie(o.cubinCrc, o.functionIndex, o.pcOffset, o.contextUid);
    }
} TAUCuptiIdSamples;



static std::string GetStallReason( uint32_t pcSamplingStallReasonIndex );
static bool GetPcSamplingDataFromCupti( CUpti_PCSamplingGetDataParams &pcSamplingGetDataParams, ContextInfo *pContextInfo );
static void StorePcSampDataInFileThread();
static void PreallocateBuffersForRecords();
static void FreePreallocatedMemory();
void ConfigureActivity( CUcontext cuCtx );
void CallbackHandler( void *pUserdata, CUpti_CallbackDomain domain, CUpti_CallbackId callbackId, void *pCallbackData );

#endif //CUDA_VERSION  >= CUDA_MIN
void cupti_pcsampling_init();
void cupti_pcsampling_exit();

#endif //CUPTI_PCSAMPLING_H
