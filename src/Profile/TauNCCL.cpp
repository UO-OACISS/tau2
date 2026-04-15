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

//#define NCCL_DEBUG

#define __hidden __attribute__ ((visibility("hidden")))
using namespace tau;
static pthread_mutex_t rank_init_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t init_free_lock = PTHREAD_MUTEX_INITIALIZER;


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
    int in_use = 0;
    uint64_t type;
    char type_name[1024];
    double start_ts;
    size_t rank;
    size_t count;
    uint8_t datatype_size;
    #ifdef NCCL_DEBUG
    int p_index;
    #endif
};

//Size in the examples is 8 for each pool, 16 for the ProxyCtrl, but has variables to increase
// the size if needed. As the memory requirement is not high, make the default size 64,
// May need to create an environmental variable to increase the size in case we need more?
// Or make it dynamic size?
//This is needed as each StopEvent needs input information, this input information is initialized
// using the StartEvent, but we need to initialize the variables for each event, to avoid as many
// mallocs and frees as events, better to use pools of events.
static const int defaultNCCLPoolSize = 256;
static int realNCCLPoolSize;

static struct tau_nccl_event* GroupApiPool;
static int index_gap_pool = 0;

static struct tau_nccl_event* CollApiPool;
static int index_ca_pool = 0;

static struct tau_nccl_event* P2pApiPool;
static int index_p2pa_pool = 0;

static struct tau_nccl_event* KernelLaunchPool;
static int index_kl_pool = 0;

static struct tau_nccl_event* GroupPool;
static int index_g_pool = 0;

static struct tau_nccl_event* CollPool;
static int index_c_pool = 0;

static struct tau_nccl_event* P2pPool;
static int index_p2p_pool = 0;

static struct tau_nccl_event* ProxyOpPool;
static int index_po_pool = 0;

static struct tau_nccl_event* ProxyStepPool;
static int index_ps_pool = 0;

static struct tau_nccl_event* ProxyCtrlPool;
static int index_pc_pool = 0;

static struct tau_nccl_event* KernelChPool;
static int index_kc_pool = 0;

static struct tau_nccl_event* NetPluginPool;
static int index_np_pool = 0;

static struct tau_nccl_event invalid_event{};

static pthread_mutex_t nccl_event_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t nccl_queue_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclDebugLogger_t logFn = nullptr;

struct tau_nccl_queue_ts {
    int queue_id;
    uint64_t timestamp;
 
};

//Map a rank and "stream" to a task
static std::map<int, std::vector<tau_nccl_queue_ts>> nccl_timestamps;


//We want that each NCCL rank intializes only once, the tid that initializes or profiles, is not always the same
// so better use a map, to check if the profiler is initialized, to avoid multiple threads trying to initialize
// the plugin.
static std::map<int, bool> nccl_rank_init;


//extern "C" void Tau_stop_top_level_timer_if_necessary_task(int tid);
extern "C" void metric_set_gpu_timestamp(int tid, double value);
extern "C" x_uint64 TauTraceGetTimeStamp(int tid);
extern "C" void Tau_metadata_task(const char *name, const char *value, int tid);

void Tau_add_metadata_for_task(const char *key, int value, int taskid) {
  char buf[1024];
  snprintf(buf, sizeof(buf),  "%d", value);
  Tau_metadata_task(key, buf, taskid);
  TAU_VERBOSE("Adding Metadata: %s, %d, for task %d\n", key, value, taskid);
}

void initializeNCCLPools()
{
    pthread_mutex_lock(&init_free_lock);
    static int initialize_pool = 0;
    if(initialize_pool == 1)
    {
        pthread_mutex_unlock(&init_free_lock);
        return;
    }
    //Make an environmental variable to change size if needed
    if(TauEnv_get_tauNCCL_poolsize()>defaultNCCLPoolSize)
    {
        printf("[TAU] NCCL pool size changed to: %d \n", TauEnv_get_tauNCCL_poolsize());
        realNCCLPoolSize = TauEnv_get_tauNCCL_poolsize();
    }
    else
    {
        realNCCLPoolSize = defaultNCCLPoolSize;
    }
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
    initialize_pool=1;
    pthread_mutex_unlock(&init_free_lock);
}

void freeNCCLPools()
{
    pthread_mutex_lock(&init_free_lock);
    static int freed_pool = 0;
    if(freed_pool == 1)
    {
        pthread_mutex_unlock(&init_free_lock);
        return;
    }
    free(GroupApiPool);
    free(CollApiPool);
    free(P2pApiPool);
    free(KernelLaunchPool);
    free(GroupPool);
    free(CollPool);
    free(P2pPool);
    free(ProxyOpPool);
    free(ProxyStepPool);
    free(ProxyCtrlPool);
    free(KernelChPool);
    free(NetPluginPool);
    freed_pool=1;
    pthread_mutex_unlock(&init_free_lock);
}


__hidden ncclResult_t tauNcclProfilerInit_v2(void** context, int* eActivationMask)
{
    
    struct context_st* ctx = (struct context_st *)calloc(1, sizeof(*ctx));
    ctx->commName = "";
    ctx->commHash = -1;
    //ctx->nNodes = nNodes;
    ctx->nranks = -1;
    ctx->rank = -1;
    *context = ctx;
    invalid_event.in_use = -1;                                       
    initializeNCCLPools();
    TAU_VERBOSE("[tauNcclProfiler v2] Init- tid=%ld\n", (long)syscall(SYS_gettid));
    const char* em_str = getenv("NCCL_PROFILE_EVENT_MASK");
    //printf("NCCL_PROFILE_EVENT_MASK %d\n", atoi(em_str));
    *eActivationMask= (em_str!=NULL) ? atoi(em_str) : 0;
    return ncclSuccess;
}

__hidden ncclResult_t tauNcclProfilerInit_v4(void** context, int* eActivationMask, 
                                             const char* commName, uint64_t commHash, 
                                             int nNodes, int nranks, int rank, ncclDebugLogger_t logfn)
{
    
    struct context_st* ctx = (struct context_st *)calloc(1, sizeof(*ctx));
    ctx->commName = commName;
    ctx->commHash = commHash;
    //ctx->nNodes = nNodes;
    ctx->nranks = nranks;
    ctx->rank = rank;
    *context = ctx;
    invalid_event.in_use = -1;                                       
    pthread_mutex_lock(&rank_init_lock);
    auto it = nccl_rank_init.find(rank);
    if (it != nccl_rank_init.end()) {
        
        pthread_mutex_unlock(&rank_init_lock);
        return ncclSuccess;
    }

    nccl_rank_init[rank]=1;
    initializeNCCLPools();
    pthread_mutex_unlock(&rank_init_lock);

    TAU_VERBOSE("[tauNcclProfiler v4] Init- tid=%ld with rank %d of %d\n", (long)syscall(SYS_gettid), rank, nranks);
    const char* em_str = getenv("NCCL_PROFILE_EVENT_MASK");
    //printf("NCCL_PROFILE_EVENT_MASK %d\n", atoi(em_str));
    *eActivationMask= (em_str!=NULL) ? atoi(em_str) : 0;

    logFn = logfn;

    return ncclSuccess;
}



// Init
__hidden ncclResult_t tauNcclProfilerInit_v5(void** context, uint64_t commId,
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
    invalid_event.in_use = -1;                                       
    pthread_mutex_lock(&rank_init_lock);
    auto it = nccl_rank_init.find(rank);
    if (it != nccl_rank_init.end()) {
        
        pthread_mutex_unlock(&rank_init_lock);
        return ncclSuccess;
    }

    nccl_rank_init[rank]=1;
    initializeNCCLPools();
    pthread_mutex_unlock(&rank_init_lock);

    TAU_VERBOSE("[tauNcclProfiler v5] Init- tid=%ld with rank %d of %d\n", (long)syscall(SYS_gettid), rank, nranks);
    const char* em_str = getenv("NCCL_PROFILE_EVENT_MASK");
    //printf("NCCL_PROFILE_EVENT_MASK %d\n", atoi(em_str));
    *eActivationMask= (em_str!=NULL) ? atoi(em_str) : 0;

    logFn = logfn;

    return ncclSuccess;
}

// Finalize
__hidden ncclResult_t tauNcclProfilerFinalize(void* context) {
    TAU_VERBOSE("[tauNcclProfiler] Finalize tid=%ld\n", (long)syscall(SYS_gettid));
    struct context_st* ctx = (struct context_st *)context;
    if (ctx != NULL) {
        free(ctx);
    }
    freeNCCLPools();
    return ncclSuccess;
}

//https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html ncclDataType_t
//As we are comparing a string, there is no need for version checking, unless they make a type
// that changes per version
static uint8_t ncclStringDataToSize(const char* dt) {
  if (0 == strcmp(dt, "ncclInt8")) return 1;
  if (0 == strcmp(dt, "ncclChar")) return 1;
  if (0 == strcmp(dt, "ncclUint8")) return 1;
  if (0 == strcmp(dt, "ncclInt32")) return 4;
  if (0 == strcmp(dt, "ncclInt")) return 4;
  if (0 == strcmp(dt, "ncclUint32")) return 4;
  if (0 == strcmp(dt, "ncclInt64")) return 8;
  if (0 == strcmp(dt, "ncclUint64")) return 8;
  if (0 == strcmp(dt, "ncclFloat16")) return 2;
  if (0 == strcmp(dt, "ncclHalf")) return 2;
  if (0 == strcmp(dt, "ncclFloat32")) return 4;
  if (0 == strcmp(dt, "ncclFloat")) return 4;
  if (0 == strcmp(dt, "ncclFloat64")) return 8;
  if (0 == strcmp(dt, "ncclDouble")) return 8;
  if (0 == strcmp(dt, "ncclBfloat16")) return 2;
  if (0 == strcmp(dt, "ncclFloat8e4m3")) return 1;
  if (0 == strcmp(dt, "ncclFloat8e5m2")) return 1;
  return ncclFloat64;
}

//There are some events that take way longer than others, check if all the available events are in use or not
// if there are available events, assign one and fill the data. If not, show a message and use the invalid event,
// which will be discarded.
struct tau_nccl_event* create_nccl_event(int cur_rank, struct tau_nccl_event* e_pool, 
    int* e_pool_indexer, char* type_name, size_t d_count, uint8_t datatype_size)
{
    pthread_mutex_lock(&nccl_event_lock);
    struct tau_nccl_event* event = &e_pool[*e_pool_indexer];
    //printf("pool index %d event %d %s\n", *e_pool_indexer, event->in_use, type_name);
    int in_use_counter = 0;
    while(event->in_use)
    {
        //printf("!");
        if(in_use_counter == realNCCLPoolSize)
        {
            printf("\nCurrent NCCL event will be discarded. Increase the pool size with: TAU_NCCL_POOLSIZE, curret size %d\n", realNCCLPoolSize);
            pthread_mutex_unlock(&nccl_event_lock);
            return &invalid_event;
        }
        *e_pool_indexer = (*e_pool_indexer+1)%realNCCLPoolSize;
        event = &e_pool[*e_pool_indexer];
        in_use_counter++;
    }
    event->start_ts = (double)TauTraceGetTimeStamp(0);
    event->rank = cur_rank;
    event->datatype_size = datatype_size;
    event->count = d_count;
    strncpy(event->type_name, type_name, sizeof(event->type_name));
    //event->task_id = task_id;
    event->in_use = 1;
    #ifdef NCCL_DEBUG
    event->p_index = *e_pool_indexer;
    #endif
    *e_pool_indexer = (*e_pool_indexer+1)%realNCCLPoolSize;
    pthread_mutex_unlock(&nccl_event_lock);
    return event;
}

//Ideally we would want to do something such as
//|-----group---------------| thread1
//|.|call1|....|call2|......| thread1
//But as we can have multiple overlaps, better to have a simple 
// approach where overlapping calls are considered to be executed
// using a  different thread/stream
//Execute this inside a locked region, in this case the event lock
//We compare the start timestamp of the current event to the
// registered ending timestamps, if all the registered timestamps
// are higher the the start,create a new task, 
// as we would have an overlap, if not, reuse one.
size_t nccl_check_overlap(int rank, uint64_t cur_b_timestamp, double cur_e_timestamp)
{
    auto& vec = nccl_timestamps[rank];
    for (size_t i = 0; i < vec.size(); ++i)
    {
        auto &t = vec[i];
        if(cur_b_timestamp>=t.timestamp)
        {
            t.timestamp = cur_e_timestamp;
            return t.queue_id;
        }
    }
    //If we reach this point, it means there is overlap
    // or there are no queues available
    //printf("Need new stream\n");
    int new_taskid;
    TAU_CREATE_TASK(new_taskid);
    struct tau_nccl_queue_ts cur_nccl_qts;
    cur_nccl_qts.queue_id = new_taskid;
    cur_nccl_qts.timestamp = cur_e_timestamp;
    vec.push_back(cur_nccl_qts);
    Tau_add_metadata_for_task("TAU_TASK_ID", new_taskid, new_taskid);
    Tau_add_metadata_for_task("NCCL_RANK_ID", rank, new_taskid);
    Tau_add_metadata_for_task("NCCL_STREAM_ID", vec.size()-1, new_taskid);
    Tau_create_top_level_timer_if_necessary_task(new_taskid);
    return new_taskid;
}

void nccl_register_task(char * task_name, int rank, double start_ts, size_t d_count, uint8_t datatype_size)
{
    double cur_timestamp = TauTraceGetTimeStamp(0);
    int cur_task_id = nccl_check_overlap( rank, start_ts, cur_timestamp);
    metric_set_gpu_timestamp(cur_task_id, start_ts);
    TAU_START_TASK(task_name, cur_task_id);
    metric_set_gpu_timestamp(cur_task_id, cur_timestamp);
    TAU_STOP_TASK(task_name, cur_task_id);
    if(d_count != 0)
    {
        void* ue = nullptr;
        char nccl_str_user_event[1024];
        snprintf(nccl_str_user_event, sizeof(nccl_str_user_event), "NCCL %s message size", task_name);
        Tau_get_context_userevent(&ue, nccl_str_user_event);
        TAU_CONTEXT_EVENT_THREAD_TS(ue, d_count*datatype_size, cur_task_id, cur_timestamp);
    }
}

void free_nccl_event( struct tau_nccl_event* event)
{
    pthread_mutex_lock(&nccl_event_lock);
    nccl_register_task(event->type_name, event->rank, event->start_ts, event->count, event->datatype_size );
    event->in_use = 0;
    pthread_mutex_unlock(&nccl_event_lock);
}

__hidden ncclResult_t tauNcclProfilerStartEvent_v2(void* context, void** eHandle, ncclProfilerEventDescr_v2_t* eDescr) {
    //TAU_VERBOSE("[tauNcclProfiler] tauNcclProfilerStartEvent_v4 tid=%ld c_rank=%d rank=%d\n", (long)syscall(SYS_gettid), eDescr->rank, eDescr->rank);
    *eHandle = NULL;
    struct context_st* ctx = (struct context_st *)context;
    if (ctx == NULL) {
        //printf("ctx == NULL\n");
        return ncclSuccess;
    }
    if (eDescr->type == ncclProfileGroup)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileGroupApi");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, GroupApiPool, &index_gap_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileGroupApi] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileColl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileColl %s", eDescr->coll.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, CollPool, &index_c_pool, type_name, 
                                                         eDescr->coll.count, ncclStringDataToSize(eDescr->coll.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileColl] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileP2p)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileP2p %s", eDescr->p2p.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, P2pPool, &index_p2p_pool, type_name,
                                                         eDescr->p2p.count, ncclStringDataToSize(eDescr->p2p.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileP2p] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyCtrl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyCtrl");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyCtrlPool, &index_pc_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyCtrl] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyOp)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyOp");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyOpPool, &index_po_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyOp] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyStep)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyStep");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyStepPool, &index_ps_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyStep] %s\n", type_name);
    }
    return ncclSuccess;
}

__hidden ncclResult_t tauNcclProfilerStartEvent_v3(void* context, void** eHandle, ncclProfilerEventDescr_v3_t* eDescr) {
    //TAU_VERBOSE("[tauNcclProfiler] tauNcclProfilerStartEvent_v4 tid=%ld c_rank=%d rank=%d\n", (long)syscall(SYS_gettid), eDescr->rank, eDescr->rank);
    *eHandle = NULL;
    struct context_st* ctx = (struct context_st *)context;
    if (ctx == NULL) {
        //printf("ctx == NULL\n");
        return ncclSuccess;
    }
    if (eDescr->type == ncclProfileGroup)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileGroupApi");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, GroupApiPool, &index_gap_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileGroupApi] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileColl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileColl %s", eDescr->coll.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, CollPool, &index_c_pool, type_name, 
                                                         eDescr->coll.count, ncclStringDataToSize(eDescr->coll.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileColl] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileP2p)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileP2p %s", eDescr->p2p.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, P2pPool, &index_p2p_pool, type_name,
                                                         eDescr->p2p.count, ncclStringDataToSize(eDescr->p2p.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileP2p] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyCtrl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyCtrl");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyCtrlPool, &index_pc_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyCtrl] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyOp)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyOp");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyOpPool, &index_po_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyOp] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyStep)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyStep");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyStepPool, &index_ps_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyStep] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileNetPlugin)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileNetPlugin");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, NetPluginPool, &index_np_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileNetPlugin] %s\n", type_name);
    }
    return ncclSuccess;
}

__hidden ncclResult_t tauNcclProfilerStartEvent_v4(void* context, void** eHandle, ncclProfilerEventDescr_v4_t* eDescr) {
    //TAU_VERBOSE("[tauNcclProfiler] tauNcclProfilerStartEvent_v4 tid=%ld c_rank=%d rank=%d\n", (long)syscall(SYS_gettid), eDescr->rank, eDescr->rank);
    *eHandle = NULL;
    struct context_st* ctx = (struct context_st *)context;
    if (ctx == NULL) {
        //printf("ctx == NULL\n");
        return ncclSuccess;
    }
    if (eDescr->type == ncclProfileGroup)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileGroupApi");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, GroupApiPool, &index_gap_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileGroupApi] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileColl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileColl %s", eDescr->coll.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, CollPool, &index_c_pool, type_name, 
                                                         eDescr->coll.count, ncclStringDataToSize(eDescr->coll.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileColl] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileP2p)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileP2p %s", eDescr->p2p.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, P2pPool, &index_p2p_pool, type_name,
                                                         eDescr->p2p.count, ncclStringDataToSize(eDescr->p2p.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileP2p] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyCtrl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyCtrl");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyCtrlPool, &index_pc_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyCtrl] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyOp)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyOp");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyOpPool, &index_po_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyOp] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileProxyStep)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyStep");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyStepPool, &index_ps_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyStep] %s\n", type_name);
    }
    else if (eDescr->type == ncclProfileNetPlugin)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileNetPlugin");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, NetPluginPool, &index_np_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileNetPlugin] %s\n", type_name);
    }
    return ncclSuccess;
}


// StartEvent
__hidden ncclResult_t tauNcclProfilerStartEvent_v5(void* context, void** eHandle, ncclProfilerEventDescr_v5_t* eDescr) {
    //Rank here is the rank of the NCCL communicator
    
    //TAU_VERBOSE("[tauNcclProfiler] tauNcclProfilerStartEvent tid=%ld c_rank=%d rank=%d\n", (long)syscall(SYS_gettid), eDescr->rank, eDescr->rank);
    *eHandle = NULL;
    
    struct context_st* ctx = (struct context_st *)context;
    if (ctx == NULL) {
        //printf("ctx == NULL\n");
        return ncclSuccess;
    }
    if(eDescr->type == ncclProfileGroupApi)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileGroupApi");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, GroupApiPool, &index_gap_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileGroupApi] %s\n", type_name);
        //TAU_START("ncclProfileGroupApi");
    }
    else if (eDescr->type == ncclProfileCollApi) 
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileCollApi %s", eDescr->collApi.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, CollApiPool, &index_ca_pool, type_name, 
                                                         eDescr->collApi.count, ncclStringDataToSize(eDescr->p2p.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileCollApi] %s\n", type_name);
        //TAU_START(type_name);
        //tau_nccl_userevents(eDescr->collApi.func, eDescr->collApi.count, eDescr->collApi.datatype);
    }
    else if (eDescr->type == ncclProfileP2pApi)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileP2pApi %s", eDescr->p2pApi.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, P2pApiPool, &index_p2pa_pool, type_name,
                                                         eDescr->p2pApi.count, ncclStringDataToSize(eDescr->p2pApi.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        #ifdef NCCL_DEBUG
            TAU_VERBOSE("[+ncclProfileP2pApi %d] %s\n", start_event->p_index, type_name);
        #else
            TAU_VERBOSE("[+ncclProfileP2pApi] %s\n", type_name);
        #endif
        //TAU_START(type_name);
        //tau_nccl_userevents(eDescr->p2pApi.func, eDescr->p2pApi.count, eDescr->p2pApi.datatype);
    }
    //Cuda profiles this part, if enabled, we have duplicated information
    /*else if (eDescr->type == ncclProfileKernelLaunch)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileKernelLaunch");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, KernelLaunchPool, &index_kl_pool, type_name );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileKernelLaunch] %s\n", type_name);
        //TAU_START("ncclProfileKernelLaunch");
    }*/
    else if (eDescr->type == ncclProfileGroup)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileGroup");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, GroupPool, &index_g_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileGroup] %s\n", type_name);
        //TAU_START("ncclProfileGroup");
    }
    else if (eDescr->type == ncclProfileColl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileColl %s", eDescr->coll.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, CollPool, &index_c_pool, type_name, 
                                                         eDescr->coll.count, ncclStringDataToSize(eDescr->coll.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileColl] %s\n", type_name);
        //TAU_START(type_name);
        //char event_name[1024];
        //snprintf(event_name, sizeof(type_name), "ncclProfileColl %s[%s,%s]", eDescr->coll.func, eDescr->coll.algo, eDescr->coll.proto);
        //tau_nccl_userevents(eDescr->coll.func, eDescr->coll.count, eDescr->coll.datatype);
    }
    else if (eDescr->type == ncclProfileP2p)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileP2p %s", eDescr->p2p.func);
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, P2pPool, &index_p2p_pool, type_name,
                                                         eDescr->p2p.count, ncclStringDataToSize(eDescr->p2p.datatype) );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        #ifdef NCCL_DEBUG
            TAU_VERBOSE("[+ncclProfileP2p pid %d index %d origin %d destination %d size %d] %s\n", (long)syscall(SYS_gettid), start_event->p_index, eDescr->rank, eDescr->p2p.peer, eDescr->p2p.count, type_name);
        #else
            TAU_VERBOSE("[+ncclProfileP2p] %s\n", type_name);
        #endif
        //TAU_START(type_name);
        //char event_name[1024];
        //snprintf(event_name, sizeof(type_name), "ncclProfileP2p %s", eDescr->p2p.func);
        //tau_nccl_userevents(eDescr->p2p.func, eDescr->p2p.count, eDescr->p2p.datatype);
    }
    else if (eDescr->type == ncclProfileProxyOp)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyOp");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyOpPool, &index_po_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyOp] %s\n", type_name);
        //TAU_START("ncclProfileProxyOp");
    }
    else if (eDescr->type == ncclProfileProxyStep)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyStep");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyStepPool, &index_ps_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyStep] %s\n", type_name);
        //TAU_START("ncclProfileProxyStep");
    }
    else if (eDescr->type == ncclProfileProxyCtrl)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileProxyCtrl");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, ProxyCtrlPool, &index_pc_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileProxyCtrl] %s\n", type_name);
        //TAU_START("ncclProfileProxyCtrl");
    }
    //Cuda profiles this part, if enabled, we have duplicated information
    /*else if (eDescr->type == ncclProfileKernelCh)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileKernelCh");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, KernelChPool, &index_kc_pool, type_name );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileKernelCh] %s\n", type_name);
        //TAU_START("ncclProfileKernelCh");
    }*/
    else if (eDescr->type == ncclProfileNetPlugin)
    {
        char type_name[1024];
        snprintf(type_name, sizeof(type_name), "ncclProfileNetPlugin");
        tau_nccl_event* start_event = create_nccl_event( eDescr->rank, NetPluginPool, &index_np_pool, type_name, 0 ,0 );
        *eHandle = (void*)start_event;
        if(start_event->in_use ==-1)
            return ncclSuccess;
        TAU_VERBOSE("[+ncclProfileNetPlugin] %s\n", type_name);
        //TAU_START("ncclProfileNetPlugin");
    }
    /*else
    {
        //printf("??\n");
    }*/

    return ncclSuccess;
}


// StopEvent
__hidden ncclResult_t tauNcclProfilerStopEvent(void* eHandle) {
    //TAU_VERBOSE("[tauNcclProfiler] tauNcclProfilerStopEvent tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL)
    {
        TAU_VERBOSE("eHandle == NULL\n");
        return ncclSuccess;
    }

    struct tau_nccl_event* eDescr = (tau_nccl_event *) eHandle;
    if(eDescr->in_use==-1)
    {
        TAU_VERBOSE("Invalid event, discarded\n");
        return ncclSuccess;
    }
    uint64_t e_type = eDescr->type;

    if(e_type == ncclProfileGroupApi)
    {
        TAU_VERBOSE("[-ncclProfileGroupApi] %s\n", eDescr->type_name);
        //TAU_STOP(eDescr->type_name);        
    }
    else if (e_type == ncclProfileCollApi) 
    {
        TAU_VERBOSE("[-ncclProfileCollApi] %s\n", eDescr->type_name);
        //TAU_STOP(eDescr->type_name);    
    }
    else if (e_type == ncclProfileP2pApi)
    {
        TAU_VERBOSE("[-ncclProfileP2pApi] %s\n", eDescr->type_name);
        //TAU_STOP(eDescr->type_name);    
    }
    /*else if (e_type == ncclProfileKernelLaunch)
    {
        TAU_VERBOSE("[-ncclProfileKernelLaunch] %s\n", eDescr->type_name);
        //TAU_STOP("ncclProfileKernelLaunch");    
    }*/
    else if (e_type == ncclProfileGroup)
    {
        TAU_VERBOSE("[-ncclProfileGroup] %s\n", eDescr->type_name);
        //TAU_STOP(eDescr->type_name);    
    }
    else if (e_type == ncclProfileColl)
    {
        TAU_VERBOSE("[-ncclProfileColl] %s\n",eDescr->type_name);
        //TAU_STOP(eDescr->type_name);    
    }
    else if (e_type == ncclProfileP2p)
    {
        
        #ifdef NCCL_DEBUG
            TAU_VERBOSE("[-ncclProfileP2p %d] %s\n", eDescr->p_index, eDescr->type_name);
        #else
            TAU_VERBOSE("[-ncclProfileP2p] %s\n", eDescr->type_name);
        #endif
        //TAU_STOP(eDescr->type_name);    
    }
    else if (e_type == ncclProfileProxyOp)
    {
        TAU_VERBOSE("[-ncclProfileProxyOp] %s\n", eDescr->type_name);
        //TAU_STOP("ncclProfileProxyOp");    
    }
    else if (e_type == ncclProfileProxyStep)
    {
        TAU_VERBOSE("[-ncclProfileProxyStep] %s\n", eDescr->type_name);
        //TAU_STOP("ncclProfileProxyStep");    
    }
    else if (e_type == ncclProfileProxyCtrl)
    {
        TAU_VERBOSE("[-ncclProfileProxyCtrl] %s\n", eDescr->type_name);
        //TAU_STOP("ncclProfileProxyCtrl");    
    }
    /*else if (e_type == ncclProfileKernelCh)
    {
        TAU_VERBOSE("[-ncclProfileKernelCh] %s\n", eDescr->type_name);
        //TAU_STOP("ncclProfileKernelCh");    
    }*/
    else if (e_type == ncclProfileNetPlugin)
    {
        TAU_VERBOSE("[-ncclProfileNetPlugin] %s\n", eDescr->type_name);
        //TAU_STOP(eDescr->type_name);    
    }
    free_nccl_event(eDescr);
    return ncclSuccess;
}

__hidden ncclResult_t tauNcclProfilerRecordEventState_v2(void* eHandle, ncclProfilerEventState_v2_t eState, ncclProfilerEventStateArgs_v2_t* eStateArgs) {
    //printf("[tauNcclProfiler] tauNcclProfilerRecordEventState tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;

    return ncclSuccess;
}

__hidden ncclResult_t tauNcclProfilerRecordEventState_v3(void* eHandle, ncclProfilerEventState_v3_t eState, ncclProfilerEventStateArgs_v3_t* eStateArgs) {
    //printf("[tauNcclProfiler] tauNcclProfilerRecordEventState tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;

    return ncclSuccess;
}

__hidden ncclResult_t tauNcclProfilerRecordEventState_v4(void* eHandle, ncclProfilerEventState_v4_t eState, ncclProfilerEventStateArgs_v4_t* eStateArgs) {
    //printf("[tauNcclProfiler] tauNcclProfilerRecordEventState tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;

    return ncclSuccess;
}
__hidden ncclResult_t tauNcclProfilerRecordEventState_v5(void* eHandle, ncclProfilerEventState_v5_t eState, ncclProfilerEventStateArgs_v5_t* eStateArgs) {
    //printf("[tauNcclProfiler] tauNcclProfilerRecordEventState tid=%ld\n", (long)syscall(SYS_gettid));
    // the event handle might be null if we run out of events
    if (eHandle == NULL) return ncclSuccess;

    return ncclSuccess;
}


/*
__hidden ncclResult_t tauNcclProfilerStartEvent_v6(void* context, void** eHandle, ncclProfilerEventDescr_v6_t* eDescr) 
{
  struct context_st* ctx = (struct context_st *)context;
  return tauNcclProfilerStartEvent_v5(context, eHandle, (ncclProfilerEventDescr_v5_t*)eDescr);
}

__hidden ncclResult_t tauNcclProfilerStopEvent_v6(void* eHandle)
{
    if (!eHandle) return ncclSuccess;
    return tauNcclProfilerStopEvent(eHandle);
}
*/

//To test if a profiler version works, just comment the newer ncclProfiler_v*
//v1 and v6 do not have documentation, do not use them at this moment

ncclProfiler_v2_t ncclProfiler_v2 = {
  "TAU-NCCLProfiler-v2",
  tauNcclProfilerInit_v2,
  tauNcclProfilerStartEvent_v2,
  tauNcclProfilerStopEvent,
  tauNcclProfilerRecordEventState_v2,
  tauNcclProfilerFinalize,
};

ncclProfiler_v3_t ncclProfiler_v3 = {
  "TAU-NCCLProfiler-v3",
  tauNcclProfilerInit_v2,
  tauNcclProfilerStartEvent_v3,
  tauNcclProfilerStopEvent,
  tauNcclProfilerRecordEventState_v3,
  tauNcclProfilerFinalize,
};


ncclProfiler_v4_t ncclProfiler_v4 = {
  "TAU-NCCLProfiler-v4",
  tauNcclProfilerInit_v4,
  tauNcclProfilerStartEvent_v4,
  tauNcclProfilerStopEvent,
  tauNcclProfilerRecordEventState_v4,
  tauNcclProfilerFinalize,
};


ncclProfiler_v5_t ncclProfiler_v5 = {
    "TAU-NCCLProfiler-v5",
    tauNcclProfilerInit_v5,
    tauNcclProfilerStartEvent_v5,
    tauNcclProfilerStopEvent,
    tauNcclProfilerRecordEventState_v5,
    tauNcclProfilerFinalize
};
/*
ncclProfiler_v6_t ncclProfiler_v6 = {
    "TAU-NCCLProfiler-v6",
    tauNcclProfilerInit_v5,
    tauNcclProfilerStartEvent_v6,
    tauNcclProfilerStopEvent,
    tauNcclProfilerRecordEventState_v5,
    tauNcclProfilerFinalize
};
*/