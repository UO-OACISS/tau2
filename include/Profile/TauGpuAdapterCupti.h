#include <Profile/TauGpu.h>
#include <cupti_events.h>
#include <cupti_callbacks.h>
#include <cupti_runtime_cbid.h>

#define CUDA_CHECK_ERROR(err, str) \
	if (err != CUDA_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

// Structure to hold API parameters
#define cudaMemcpy cudaMemcpy
typedef struct cudaMemcpy_params_st {
    void *dst;
    const void *src;
    size_t count;
    unsigned int kind;
}cudaMemcpy_params;

#define cudaMemcpyToArray cudaMemcpyToArray
typedef struct cudaMemcpyToArray_params_st {
    void *dst;
		size_t wOffset;
		size_t hOffset;
    const void *src;
    size_t count;
    unsigned int kind;
}cudaMemcpyToArray_params;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
    CUpti_RuntimeTraceApi traceInfo;
    uint64_t startTimestamp;
    uint64_t endTimestamp;
    cudaMemcpy_params memcpy_params;
} RuntimeApiTrace_t;


enum launchOrder{ MEMCPY_H2D1, MEMCPY_H2D2, MEMCPY_D2H, KERNEL, THREAD_SYNC, LAUNCH_LAST};

