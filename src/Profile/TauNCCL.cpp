#include <stdio.h>
#include <pthread.h>
#include <cstring>
#include <chrono>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdlib>

#include <nccl.h>
#include <nccl/common.h>
#include <nccl/nccl_profiler.h>
#include <nccl/profiler/profiler_v6.h>

// arrays for different event objects
struct context_st {
  const char* commName;
  uint64_t commHash;
  int nranks;
  int rank;

  // CE event tracking for poller
  //struct ceEventList ceEvents;

  int groupApiPoolSize;
  int groupApiPoolBase;
  int groupApiPoolIndex;
  struct groupApi* groupApiPool;

  int collApiPoolSize;
  int collApiPoolBase;
  int collApiPoolIndex;
  struct collApi* collApiPool;

  int p2pApiPoolSize;
  int p2pApiPoolBase;
  int p2pApiPoolIndex;
  struct p2pApi* p2pApiPool;

  int kernelLaunchPoolSize;
  int kernelLaunchPoolBase;
  int kernelLaunchPoolIndex;
  struct kernelLaunch* kernelLaunchPool;

  int groupPoolSize;
  int groupPoolBase;
  int groupPoolIndex;
  struct group* groupPool;

  int collPoolSize;
  int collPoolBase;
  int collPoolIndex;
  struct collective* collPool;

  int p2pPoolSize;
  int p2pPoolBase;
  int p2pPoolIndex;
  struct p2p* p2pPool;

  int proxyCtrlPoolSize;
  int proxyCtrlPoolBase;
  int proxyCtrlPoolIndex;
  struct proxyCtrl* proxyCtrlPool;

  // CE event pools
  int ceCollPoolSize;
  int ceCollPoolBase;
  int ceCollPoolIndex;
  struct ceColl* ceCollPool;

  int ceSyncPoolSize;
  int ceSyncPoolBase;
  int ceSyncPoolIndex;
  struct ceSync* ceSyncPool;

  int ceBatchPoolSize;
  int ceBatchPoolBase;
  int ceBatchPoolIndex;
  struct ceBatch* ceBatchPool;
};

#define __hidden __attribute__ ((visibility("hidden")))

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int plug_init = 0;

static ncclDebugLogger_t logFn = nullptr;

// Init
__hidden ncclResult_t exampleProfilerInit(void** context, uint64_t commId,
                                          int* eActivationMask,
                                          const char* commName, int nNodes,
                                          int nranks, int rank,
                                          ncclDebugLogger_t logfn) {
    //printf("[ExampleProfiler] Init tid=%ld\n", (long)syscall(SYS_gettid));
    
    struct context_st* ctx = (struct context_st *)calloc(1, sizeof(*ctx));
    ctx->commName = commName;
    ctx->commHash = commId;
    //ctx->nNodes = nNodes;
    ctx->nranks = nranks;
    ctx->rank = rank;
    *context = ctx;

    pthread_mutex_lock(&lock);
    if (plug_init) {
        
        pthread_mutex_unlock(&lock);
        return ncclSuccess;
    }
    plug_init=1;
    printf("[ExampleProfiler] Init- tid=%ld\n", (long)syscall(SYS_gettid));
    const char* str = getenv("NCCL_PROFILE_EVENT_MASK");
    *eActivationMask= (str!=NULL) ? atoi(str) : 0;
    pthread_mutex_unlock(&lock);

    logFn = logfn;

    return ncclSuccess;
}

// Finalize
__hidden ncclResult_t exampleProfilerFinalize(void* context) {
    printf("[ExampleProfiler] Finalize tid=%ld\n", (long)syscall(SYS_gettid));
    struct context_st* ctx = (struct context_st *)context;
    if (ctx != NULL) {
        free(ctx);
    }
    return ncclSuccess;
}

// StartEvent
__hidden ncclResult_t exampleProfilerStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
    printf("[ExampleProfiler] exampleProfilerStartEvent tid=%ld\n", (long)syscall(SYS_gettid));
    *eHandle = NULL;
    struct context_st* ctx = (struct context_st *)context;
    if (ctx == NULL) {
        return ncclSuccess;
    }

    if(eDescr->type == ncclProfileGroupApi)
    {

    }

    return ncclSuccess;
}

// StopEvent
__hidden ncclResult_t exampleProfilerStopEvent(void* eHandle) {
    printf("[ExampleProfiler] exampleProfilerStopEvent tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;



    return ncclSuccess;
}

__hidden ncclResult_t exampleProfilerRecordEventState(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* eStateArgs) {
    printf("[ExampleProfiler] exampleProfilerRecordEventState tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;



    return ncclSuccess;
}

ncclProfiler_v6_t ncclProfiler_v6 = {
    "ExampleProfiler-v6",
    exampleProfilerInit,
    exampleProfilerStartEvent,
    exampleProfilerStopEvent,
    exampleProfilerRecordEventState,
    exampleProfilerFinalize
};
