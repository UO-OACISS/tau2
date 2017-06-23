/****************************************************************************
**			TAU Portable Profiling Package                     **
**			http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2009-2017  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TracerOTF2.cpp 			        	   **
**	Description 	: TAU Tracing for native OTF2 generation			   **
**  Author      : Nicholas Chaimov
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : TAU Tracing                                      **
**                                                                         **
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>

#include <tau_internal.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauTrace.h>
#include <Profile/TauTraceOTF2.h>
#include <Profile/TauMetrics.h>

#include <iostream>

#include <otf2/otf2.h>
#ifdef PTHREADS
#include <otf2/OTF2_Pthread_Locks.h>
#endif
#ifdef TAU_OPENMP
#include <otf2/OTF2_OpenMP_Locks.h>
#endif

#define OTF2_EC(call) { \
    OTF2_ErrorCode ec = call; \
    if (ec != OTF2_SUCCESS) { \
        printf("TAU: OTF2 Error (%s:%d): %s, %s\n", __FILE__, __LINE__, OTF2_Error_GetName(ec), OTF2_Error_GetDescription (ec)); \
        abort(); \
    } \
}

using namespace std;
using namespace tau;

static bool otf2_initialized = false;
static bool otf2_finished = false;
static OTF2_Archive * otf2_archive = NULL;
static x_uint64 start_time = 0;
static x_uint64 end_time = 0;

extern "C" x_uint64 TauTraceGetTimeStamp(int tid);
extern "C" int tau_totalnodes(int set_or_get, int value);

// Collective Callbacks -- GetSize and GetRank are mandatory
// others are only needed when using SION substrate

static OTF2_CallbackCode tau_OTF2GetSize(void * userData, 
        OTF2_CollectiveContext * commContext, uint32_t * size) {
    *size = tau_totalnodes(0, 0);
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2GetRank (void * userData, 
        OTF2_CollectiveContext *commContext, uint32_t * rank) {
    int myNode = RtsLayer::myNode();
    *rank = myNode == -1 ? 0 : myNode;
    return OTF2_CALLBACK_SUCCESS;
}

// Stubs for optional collective callbacks
static OTF2_CallbackCode tau_OTF2CreateLocalComm (void *userData,
        OTF2_CollectiveContext **localCommContext, OTF2_CollectiveContext
        *globalCommContext, uint32_t globalRank, uint32_t globalSize, uint32_t
        localRank, uint32_t localSize, uint32_t fileNumber, uint32_t numberOfFiles) {
    /* Create a new disjoint partitioning of the the globalCommContext
        communication context. numberOfFiles denotes the number of the partitions.
        fileNumber denotes in which of the partitions this OTF2_Archive should belong.
        localSize is the size of this partition and localRank the rank of this
        OTF2_Archive in the partition. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2FreeLocalComm (void *userData,
        OTF2_CollectiveContext *localCommContext) {
    /* Destroys the communication context previous created by the
        OTF2CreateLocalComm callback. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2Barrier (void *userData,
        OTF2_CollectiveContext *commContext) {
    /* Performs a barrier collective on the given communication context. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2Bcast (void *userData,
        OTF2_CollectiveContext *commContext, void *data, uint32_t numberElements,
        OTF2_Type type, uint32_t root) {
    /* Performs a broadcast collective on the given communication context. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2Gather (void *userData,
        OTF2_CollectiveContext *commContext, const void *inData, void *outData,
        uint32_t numberElements, OTF2_Type type, uint32_t root) {
    /* Performs a gather collective on the given communication context where
        each ranks contribute the same number of elements. outData is only valid at
        rank root. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2Gatherv (void *userData,
        OTF2_CollectiveContext *commContext, const void *inData, uint32_t inElements,
        void *outData, const uint32_t *outElements, OTF2_Type type, uint32_t root) {
    /* Performs a gather collective on the given communication context where
        each ranks contribute different number of elements. outData and outElements are
        only valid at rank root. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2Scatter (void *userData,
        OTF2_CollectiveContext *commContext, const void *inData, void *outData,
        uint32_t numberElements, OTF2_Type type, uint32_t root) {
    /* Performs a scatter collective on the given communication context where
        each ranks contribute the same number of elements. inData is only valid at rank
        root. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CallbackCode tau_OTF2Scatterv (void *userData,
        OTF2_CollectiveContext *commContext, const void *inData, const uint32_t
        *inElements, void *outData, uint32_t outElements, OTF2_Type type, uint32_t
        root) {
    /* Performs a scatter collective on the given communication context where
        each ranks contribute different number of elements. inData and inElements are
        only valid at rank root. */
    return OTF2_CALLBACK_SUCCESS;
}

static OTF2_CollectiveCallbacks * get_tau_collective_callbacks() {
    static OTF2_CollectiveCallbacks cb;
    cb.otf2_release = NULL;
    cb.otf2_get_size = tau_OTF2GetSize;
    cb.otf2_get_rank = tau_OTF2GetRank;
    cb.otf2_create_local_comm = NULL;
    cb.otf2_free_local_comm = NULL;
    cb.otf2_barrier = tau_OTF2Barrier;
    cb.otf2_bcast = tau_OTF2Bcast;
    cb.otf2_gather = tau_OTF2Gather;
    cb.otf2_gatherv = tau_OTF2Gatherv;
    cb.otf2_scatter = tau_OTF2Scatter;
    cb.otf2_scatterv = tau_OTF2Scatterv;
    return &cb;                                                 
}

// Flush Callbacks -- both mandatory

static OTF2_FlushType tau_OTF2PreFlush( void* userData, OTF2_FileType fileType,
        OTF2_LocationRef location, void* callerData, bool final ) {
    return OTF2_FLUSH;
}

static OTF2_TimeStamp tau_OTF2PostFlush(void* userData, OTF2_FileType fileType,
        OTF2_LocationRef location ) { 
    return TauTraceGetTimeStamp(0);
}

static OTF2_FlushCallbacks * get_tau_flush_callbacks() {
   static OTF2_FlushCallbacks cb;
   cb.otf2_pre_flush = tau_OTF2PreFlush;
   cb.otf2_post_flush = tau_OTF2PostFlush;
   return &cb;
}


// Helper functions

static inline int my_location() {
    const int myNode = RtsLayer::myNode();
    const int myThread = RtsLayer::myThread();
    return myNode == -1 ? myThread : (myNode * TAU_MAX_THREADS) + myThread;
}

// Tau Tracing API calls for OTF2

/* Flush the trace buffer */
void TauTraceOTF2FlushBuffer(int tid)
{
  // Protect TAU from itself
  TauInternalFunctionGuard protects_this_function;
  std::cerr << "TauTraceOTF2FlushBuffer " << tid << std::endl;
}

/* Initialize tracing. */
int TauTraceOTF2Init(int tid) {
  return TauTraceOTF2InitTS(tid, TauTraceGetTimeStamp(tid));
}

int TauTraceOTF2InitTS(int tid, x_uint64 ts)
{
  if(otf2_initialized || otf2_finished) {
      return 0;
  }
  start_time = ts;
  TauInternalFunctionGuard protects_this_function;     
  std::cerr << "TauTraceOTF2Init " << tid << std::endl;
  otf2_archive = OTF2_Archive_Open(TauEnv_get_tracedir() /* path */,
                             "trace" /* filename */,
                             OTF2_FILEMODE_WRITE,
                             OTF2_CHUNK_SIZE_EVENTS_DEFAULT,
                             OTF2_CHUNK_SIZE_DEFINITIONS_DEFAULT,
                             OTF2_SUBSTRATE_POSIX,
                             OTF2_COMPRESSION_NONE);
  fprintf(stderr, "opened archive\n");
  if(otf2_archive == NULL) {
    std::cerr << "TAU: Error: Unable to create OTF2 archive at " << TauEnv_get_tracedir() << "/trace" << std::endl;    abort();
  }

  OTF2_EC(OTF2_Archive_SetFlushCallbacks(otf2_archive, get_tau_flush_callbacks(), NULL));
  OTF2_EC(OTF2_Archive_SetCollectiveCallbacks(otf2_archive, get_tau_collective_callbacks(), NULL, NULL, NULL));
  uint32_t size = 40;
  tau_OTF2GetRank(NULL, NULL, &size);
  OTF2_EC(OTF2_Archive_SetCreator(otf2_archive, "TAU"));
#if defined(TAU_OPENMP)
  OTF2_EC(OTF2_OpenMP_Archive_SetLockingCallbacks(otf2_archive));
#elif defined(PTHREADS)
  OTF2_EC(OTF2_Pthread_Archive_SetLockingCallbacks(otf2_archive, NULL));
#endif
  // If going to use a threading model other than OpenMP or Pthreads,
  // a set of custom locking callbacks will need to be defined.

  OTF2_EC(OTF2_Archive_OpenEvtFiles(otf2_archive));
  OTF2_EC(OTF2_Archive_OpenDefFiles(otf2_archive));

  OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter(otf2_archive, my_location());

  otf2_initialized = true;
  return 0; 
}

/* This routine is typically invoked when multiple SET_NODE calls are 
   encountered for a multi-threaded program */ 
void TauTraceOTF2Reinitialize(int oldid, int newid, int tid) {
  std::cerr << "TauTraceOTF2Reinitialize " << tid << std::endl;
  return ;
}

/* Reset the trace */
void TauTraceOTF2UnInitialize(int tid) {
  /* to set the trace as uninitialized and clear the current buffers (for forked
     child process, trying to clear its parent records) */
  std::cerr << "TauTraceOTF2UnInitialize " << tid << std::endl;
}


/* Write event to buffer */
void TauTraceOTF2EventSimple(long int ev, x_int64 par, int tid, int kind) {
  TauTraceOTF2Event(ev, par, tid, 0, 0, kind);
}

/* Write event to buffer */
void TauTraceOTF2EventWithNodeId(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts, int node_id, int kind)
{
  TauInternalFunctionGuard protects_this_function;
  std::cerr << "TauTraceOTF2WithNodeId(" << ev << ", " << par << ", " << tid << ", " << ts << ", " << use_ts << ", " << node_id << ", " << kind << ")" << std::endl;
  if(!otf2_initialized) {
    if(use_ts) {
        TauTraceOTF2InitTS(tid, ts); 
    } else {
        TauTraceOTF2Init(tid);
    }
  }
  if(otf2_finished) {
    return;
  }
  if(kind == TAU_TRACE_EVENT_KIND_FUNC) {
    int loc = my_location();
    OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter(otf2_archive, loc);
    if(par == 1) { // Enter
      OTF2_EvtWriter_Enter(evt_writer, NULL, use_ts ? ts : TauTraceGetTimeStamp(tid), ev);
    } else if(par == -1) { // Exit
      OTF2_EvtWriter_Leave(evt_writer, NULL, use_ts ? ts : TauTraceGetTimeStamp(tid), ev);
    }
  }
}


extern "C" void TauTraceOTF2Msg(int send_or_recv, int type, int other_id, int length, x_uint64 ts, int use_ts, int node_id) {
  std::cerr << "TauTraceOTF2Msg( " << send_or_recv << ", " << type << ", " << other_id << ", " << length << ", " << ts << ", " << use_ts << ", " << node_id << ")" << std::endl;
}

/* Write event to buffer */
void TauTraceOTF2Event(long int ev, x_int64 par, int tid, x_uint64 ts, int use_ts, int kind) {
  TauTraceOTF2EventWithNodeId(ev, par, tid, ts, use_ts, RtsLayer::myNode(), kind);
}

/* Close the trace */
void TauTraceOTF2Close(int tid) {
    std::cerr << "TauTraceOTF2Close " << tid << std::endl;
    if(tid != 0 || otf2_finished || !otf2_initialized) {
        return;
    }

    otf2_finished = true;
    otf2_initialized = false;
    end_time = TauTraceGetTimeStamp(0);

    // Write definitions file
    
    OTF2_GlobalDefWriter * global_def_writer = OTF2_Archive_GetGlobalDefWriter(otf2_archive);
    if(global_def_writer == NULL) {
        fprintf(stderr, "TAU: Error: Couldn't get global def writer.\n");
        abort();
    }

    OTF2_GlobalDefWriter_WriteClockProperties(global_def_writer, 1000000, start_time, end_time - start_time);

    // Write a Location for each thread within each Node (which has a LocationGroup and SystemTreeNode)
        
    int nextString = 1;
    const int emptyString = 0;
    OTF2_EC(OTF2_GlobalDefWriter_WriteString(global_def_writer, 0, "" ));

    if(RtsLayer::myNode() < 1) { // If master or only node
        const int nodes = tau_totalnodes(0, 0);
        for(int node = 0; node < nodes; ++node) {
            // System Tree Node
            char namebuf[256];
            snprintf(namebuf, 256, "node %d", node);                                  
            int nodeName = nextString++;
            OTF2_EC(OTF2_GlobalDefWriter_WriteString(global_def_writer, nodeName, namebuf));
            OTF2_EC(OTF2_GlobalDefWriter_WriteSystemTreeNode(global_def_writer, node, nodeName, emptyString, OTF2_UNDEFINED_SYSTEM_TREE_NODE));        

            // Location Group
            snprintf(namebuf, 256, "group %d", node);
            int groupName = nextString++;
            OTF2_EC(OTF2_GlobalDefWriter_WriteString(global_def_writer, groupName, namebuf));
            OTF2_EC(OTF2_GlobalDefWriter_WriteLocationGroup(global_def_writer, node, groupName, OTF2_LOCATION_GROUP_TYPE_PROCESS, node));

            // TODO Need to get actual number of locations from each node
            const int locs = nodes * RtsLayer::getTotalThreads();
            for(int loc = 0; loc < locs; ++loc) {
                snprintf(namebuf, 256, "location %d", loc);
                int locName = nextString++;
                OTF2_EvtWriter* evt_writer = OTF2_Archive_GetEvtWriter(otf2_archive, loc);
                uint64_t num_events = 0;
                OTF2_EC(OTF2_EvtWriter_GetNumberOfEvents(evt_writer, &num_events));
                OTF2_EC(OTF2_GlobalDefWriter_WriteString(global_def_writer, locName, namebuf));
                OTF2_EC(OTF2_GlobalDefWriter_WriteLocation(global_def_writer, loc, locName, OTF2_LOCATION_TYPE_CPU_THREAD, num_events, node));
            }

        }

        // Write all the functions out as Regions
        for (vector<FunctionInfo*>::iterator it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
            FunctionInfo *fi = *it;
            int thisFuncName = nextString++;
            OTF2_EC(OTF2_GlobalDefWriter_WriteString(global_def_writer, thisFuncName, fi->GetName()));
            OTF2_EC(OTF2_GlobalDefWriter_WriteRegion(global_def_writer, fi->GetFunctionId(), thisFuncName, thisFuncName, emptyString, OTF2_REGION_ROLE_FUNCTION, OTF2_PARADIGM_USER, OTF2_REGION_FLAG_NONE, 0, 0, 0));
        }

    }


    
    OTF2_EC(OTF2_Archive_CloseGlobalDefWriter(otf2_archive, global_def_writer));
    OTF2_EC(OTF2_Archive_CloseEvtFiles(otf2_archive));
    OTF2_EC(OTF2_Archive_Close(otf2_archive));

}


