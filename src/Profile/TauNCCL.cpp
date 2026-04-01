#include <stdio.h>
#include <pthread.h>
#include <cstring>
#include <chrono>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdlib>
#include <map>
#include <TAU.h>

#include <nccl.h>
#include <nccl/common.h>
#include <nccl/nccl_profiler.h>
#include <nccl/profiler/profiler_v6.h>

#define __hidden __attribute__ ((visibility("hidden")))

static pthread_mutex_t rank_init_lock = PTHREAD_MUTEX_INITIALIZER;

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


struct tau_nccl_event {
  int type;
  int in_use;
};

//Size in the examples is 8 for each pool, 16 for the ProxyCtrl, but has variables to increase
// the size if needed. As the memory requirement is not high, make the default size 64,
// May need to create an environmental variable to increase the size in case we need more?
// Or make it dynamic size?
//This is needed as each StopEvent needs input information, this input information is initialized
// using the StartEvent, but we need to initialize the variables for each event, to avoid as many
// mallocs and frees as events, better to use pools of events.
static const int defaultNCCLPoolSize = 64;
static int realNCCLPoolSize;

static struct tau_nccl_event* GroupApiPool;
static int index_gap_pool;

static struct tau_nccl_event* CollApiPool;
static int index_ca_pool;

static struct tau_nccl_event* P2pApiPool;
static int index_p2pa_pool;

static struct tau_nccl_event* KernelLaunchPool;
static int index_kl_pool;

static struct tau_nccl_event* GroupPool;
static int index_g_pool;

static struct tau_nccl_event* CollPool;
static int index_c_pool;

static struct tau_nccl_event* P2pPool;
static int index_p2p_pool;

static struct tau_nccl_event* ProxyOpPool;
static int index_po_pool;

static struct tau_nccl_event* ProxyStepPool;
static int index_ps_pool;

static struct tau_nccl_event* ProxyCtrlPool;
static int index_pc_pool;

static struct tau_nccl_event* KernelChPool;
static int index_kc_pool;

static struct tau_nccl_event* NetPluginPool;
static int index_np_pool;

static pthread_mutex_t nccl_event_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclDebugLogger_t logFn = nullptr;

//We want that each NCCL rank intializes only once, the tid that initializes or profiles, is not always the same
// so better use a map, to check if the profiler is initialized, to avoid multiple threads trying to initialize
// the plugin.
static std::map<int, bool> nccl_rank_init;

void initializeNCCLPools()
{
    static int initialize_pool = 0;
    if(initialize_pool ==1)
        return;
    //Make an environmental variable to change size if needed
    realNCCLPoolSize = defaultNCCLPoolSize;
    GroupApiPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*GroupApiPool));
    CollApiPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*CollApiPool));
    P2pApiPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*P2pApiPool));
    KernelLaunchPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*KernelLaunchPool));
    GroupPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*GroupPool));
    CollPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*CollPool));
    P2pPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*P2pPool));
    ProxyOpPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*ProxyOpPool));
    ProxyStepPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*ProxyStepPool));
    ProxyCtrlPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*ProxyCtrlPool));
    KernelChPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*KernelChPool));
    NetPluginPool = (struct tau_nccl_event *)calloc( realNCCLPoolSize, sizeof(*NetPluginPool));
}

// Init
__hidden ncclResult_t exampleProfilerInit(void** context, uint64_t commId,
                                          int* eActivationMask,
                                          const char* commName, int nNodes,
                                          int nranks, int rank,
                                          ncclDebugLogger_t logfn) {
    
    struct context_st* ctx = (struct context_st *)calloc(1, sizeof(*ctx));
    ctx->commName = commName;
    ctx->commHash = commId;
    //ctx->nNodes = nNodes;
    ctx->nranks = nranks;
    ctx->rank = rank;
    *context = ctx;

    pthread_mutex_lock(&rank_init_lock);
    auto it = nccl_rank_init.find(rank);
    if (it != nccl_rank_init.end()) {
        
        pthread_mutex_unlock(&rank_init_lock);
        return ncclSuccess;
    }

    nccl_rank_init[rank]=1;
    initializeNCCLPools();
    pthread_mutex_unlock(&rank_init_lock);

    TAU_VERBOSE("[ExampleProfiler] Init- tid=%ld with rank %d of %d\n", (long)syscall(SYS_gettid), rank, nranks);
    const char* str = getenv("NCCL_PROFILE_EVENT_MASK");
    *eActivationMask= (str!=NULL) ? atoi(str) : 0;

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

int check_event_pool(struct tau_nccl_event* event)
{
    if(event->in_use)
    {
        printf("Increase the pool size with: \n");
        return 1;
    }
    return 0;
}

struct tau_nccl_event* create_nccl_event(uint64_t type, struct tau_nccl_event* e_pool, int* e_pool_indexer)
{
    pthread_mutex_lock(&nccl_event_lock);
    struct tau_nccl_event* event = &e_pool[*e_pool_indexer];
    check_event_pool(event);
    event->type = type;
    event->in_use = 1;
    *e_pool_indexer = (*e_pool_indexer+1)%realNCCLPoolSize;
    pthread_mutex_unlock(&nccl_event_lock);
    return event;
}

// StartEvent
__hidden ncclResult_t exampleProfilerStartEvent(void* context, void** eHandle, ncclProfilerEventDescr_t* eDescr) {
    //Rank here is the rank of the NCCL communicator
    TAU_VERBOSE("[ExampleProfiler] exampleProfilerStartEvent tid=%ld c_rank=%d rank=%d\n", (long)syscall(SYS_gettid), eDescr->rank, eDescr->rank);
    *eHandle = NULL;
    struct context_st* ctx = (struct context_st *)context;
    if (ctx == NULL) {
        return ncclSuccess;
    }
    if(eDescr->type == ncclProfileGroupApi)
    {
        TAU_VERBOSE("ncclProfileGroupApi ");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileGroupApi");

        *eHandle = create_nccl_event( eDescr->type, GroupApiPool, &index_gap_pool );
    }
    else if (eDescr->type == ncclProfileCollApi) 
    {
        TAU_VERBOSE("ncclProfileCollApi\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileCollApi");

        *eHandle = create_nccl_event( eDescr->type, CollApiPool, &index_ca_pool);
    }
    else if (eDescr->type == ncclProfileP2pApi)
    {
        TAU_VERBOSE("ncclProfileP2pApi\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileP2pApi");

        *eHandle = create_nccl_event( eDescr->type, P2pApiPool, &index_p2pa_pool);
    }
    else if (eDescr->type == ncclProfileKernelLaunch)
    {
        TAU_VERBOSE("ncclProfileKernelLaunch\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileKernelLaunch");

        *eHandle = create_nccl_event( eDescr->type, KernelLaunchPool, &index_kl_pool);
    }
    else if (eDescr->type == ncclProfileGroup)
    {
        TAU_VERBOSE("ncclProfileGroup\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileGroup");

        *eHandle = create_nccl_event( eDescr->type, GroupPool, &index_g_pool);
    }
    else if (eDescr->type == ncclProfileColl)
    {
        TAU_VERBOSE("ncclProfileColl\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileColl");

        *eHandle = create_nccl_event( eDescr->type, CollPool, &index_c_pool);
    }
    else if (eDescr->type == ncclProfileP2p)
    {
        TAU_VERBOSE("ncclProfileP2p\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileP2p");

        *eHandle = create_nccl_event( eDescr->type, P2pPool, &index_p2p_pool);
    }
    else if (eDescr->type == ncclProfileProxyOp)
    {
        TAU_VERBOSE("ncclProfileProxyOp\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileProxyOp");

        *eHandle = create_nccl_event( eDescr->type, ProxyOpPool, &index_po_pool);
    }
    else if (eDescr->type == ncclProfileProxyStep)
    {
        TAU_VERBOSE("ncclProfileProxyStep\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileProxyStep");

        *eHandle = create_nccl_event( eDescr->type, ProxyStepPool, &index_ps_pool);
    }
    else if (eDescr->type == ncclProfileProxyCtrl)
    {
        TAU_VERBOSE("ncclProfileProxyCtrl\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileProxyCtrl");

        *eHandle = create_nccl_event( eDescr->type, ProxyCtrlPool, &index_pc_pool);
    }
    else if (eDescr->type == ncclProfileKernelCh)
    {
        TAU_VERBOSE("ncclProfileKernelCh\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileKernelCh");

        *eHandle = create_nccl_event( eDescr->type, KernelChPool, &index_kc_pool);
    }
    else if (eDescr->type == ncclProfileNetPlugin)
    {
        TAU_VERBOSE("ncclProfileNetPlugin\n");
        TAU_VERBOSE("\n");
        TAU_START("ncclProfileNetPlugin");

        *eHandle = create_nccl_event( eDescr->type, NetPluginPool, &index_np_pool);
    }

    return ncclSuccess;
}

int discard_nccl_event( struct tau_nccl_event* event)
{
    pthread_mutex_lock(&nccl_event_lock);
    if(event->in_use == 0)
    {
        printf("Event already stoped, this should not happen!");
        pthread_mutex_unlock(&nccl_event_lock);
        return 1;
    }
    event->in_use = 0;
    pthread_mutex_unlock(&nccl_event_lock);
    return 1;
}

// StopEvent
__hidden ncclResult_t exampleProfilerStopEvent(void* eHandle) {
    TAU_VERBOSE("[ExampleProfiler] exampleProfilerStopEvent tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;

    struct tau_nccl_event* eDescr = (tau_nccl_event *) eHandle;
    uint64_t e_type = eDescr->type;
    if(discard_nccl_event(eDescr)==0)
        return ncclSuccess;

    if(e_type == ncclProfileGroupApi)
    {
        TAU_VERBOSE("ncclProfileGroupApi ");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileGroupApi");        
    }
    else if (e_type == ncclProfileCollApi) 
    {
        TAU_VERBOSE("ncclProfileCollApi\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileCollApi");    
    }
    else if (e_type == ncclProfileP2pApi)
    {
        TAU_VERBOSE("ncclProfileP2pApi\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileP2pApi");    
    }
    else if (e_type == ncclProfileKernelLaunch)
    {
        TAU_VERBOSE("ncclProfileKernelLaunch\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileKernelLaunch");    
    }
    else if (e_type == ncclProfileGroup)
    {
        TAU_VERBOSE("ncclProfileGroup\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileGroup");    
    }
    else if (e_type == ncclProfileColl)
    {
        TAU_VERBOSE("ncclProfileColl\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileColl");    
    }
    else if (e_type == ncclProfileP2p)
    {
        TAU_VERBOSE("ncclProfileP2p\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileP2p");    
    }
    else if (e_type == ncclProfileProxyOp)
    {
        TAU_VERBOSE("ncclProfileProxyOp\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileProxyOp");    
    }
    else if (e_type == ncclProfileProxyStep)
    {
        TAU_VERBOSE("ncclProfileProxyStep\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileProxyStep");    
    }
    else if (e_type == ncclProfileProxyCtrl)
    {
        TAU_VERBOSE("ncclProfileProxyCtrl\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileProxyCtrl");    
    }
    else if (e_type == ncclProfileKernelCh)
    {
        TAU_VERBOSE("ncclProfileKernelCh\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileKernelCh");    
    }
    else if (e_type == ncclProfileNetPlugin)
    {
        TAU_VERBOSE("ncclProfileNetPlugin\n");
        TAU_VERBOSE("\n");
        TAU_STOP("ncclProfileNetPlugin");    
    }

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
