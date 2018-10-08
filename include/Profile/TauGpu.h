#ifndef _TAU_GPU_INTERFACE
#define _TAU_GPU_INTERFACE

#define MESSAGE_SEND 0
#define MESSAGE_RECV 1
#define MESSAGE_UNKNOWN 2
#define MESSAGE_RECIPROCAL_SEND 3
#define MESSAGE_RECIPROCAL_RECV 4

enum Memcpy { MemcpyHtoD = 0, MemcpyDtoH = 1, MemcpyDtoD = 2, MemcpyUnknown = 3 };
enum Unifmem { BytesHtoD = 0, BytesDtoH = 1, CPUPageFault = 2, UnifmemUnknown = 3 };
enum Instrmix { FlPtOps = 0, MemOps = 1, CtrlOps = 2 };

#define TAU_GPU_UNKNOWN_TRANSFER_SIZE -1
#define TAU_GPU_USE_DEFAULT_NAME ""

#define TAU_MAX_NUMBER_OF_GPU_THREADS TAU_MAX_THREADS

#include <Profile/tau_types.h>
#ifdef __cplusplus

#include <Profile/Profiler.h>
#include <map>
#include <stdint.h>
using namespace tau;

//typedef map<TauContextUserEvent*, TAU_EVENT_DATATYPE> TauGpuContextMap;

/* Struct to contain the user event data for each GPU event. */
typedef struct {
	tau::TauContextUserEvent *userEvent;
	tau::TAU_EVENT_DATATYPE data;
} GpuEventAttributes;

/* Struct to contain the metadata for each GPU. */
typedef struct {
	char *name;
	const char *value;

} GpuMetadata;

#define GPU_ATTRIBUTE(attr, event, data) \
attr.userEvent = event; \
attr.data = data;

/*
 * GPU Event class. This a virtual class that is extended by each GPU adapter.
 * It contains all the information that TAU needs to know about the GPU and the
 * events that run on it.
 */
class GpuEvent {

public:
	// method to get a copy of this class.
	virtual GpuEvent *getCopy() const = 0;

	// method for comparison of GpuEvents by GPU identification. Used for mapping
	// GPUs to profile tasks. 
	virtual bool less_than(const GpuEvent *other) const = 0;

	//Name for this event.
	virtual const char *getName() const = 0;

	//Task id for this event.
	virtual int getTaskId() const = 0;

	//CPU event callsite for this event. Used to create a the GPU callsite event paths.
	virtual FunctionInfo* getCallingSite() const = 0;

	//GPU event attributes, used to create context user events for each GPU Event.
	//Warning: atr can be set to NULL.
	virtual void getAttributes(GpuEventAttributes *&atr, int &numberOfAttributes) const = 0;

	//GPU metadata, calling this routine should register the GPU metadata on the
	//profile id given.
	virtual void recordMetadata(int profile_id) const = 0;

	//Synchronization offset for this GPU. 
	virtual double syncOffset() const = 0; 

	//the GPU identification (for debugging purposes).
	virtual const char *gpuIdentifier() const = 0;

	//return the conponents of the gpu identifier, used for store the gpu id in
	//the trace files.
	virtual x_uint64 id_p1() const = 0;
	virtual x_uint64 id_p2() const = 0;
	
};

/* /\* BEGIN: PC Sampling structs *\/ */

/* class InstrSampling */
/* { */
/*  public: */
/*   uint32_t sourceLocatorId; */
/*   uint32_t functionId; */
/*   uint32_t pcOffset; */
/*   uint32_t correlationId; */
/*   uint32_t executed; */
/*   uint32_t threadsExecuted; */
/*   double timestamp_delta; */
/*   double timestamp_current; */
/*   double timestamp_recent; */
/*   }; */

/* class FuncSampling */
/* { */
/*  public: */
/*   uint32_t fid; */
/*   uint32_t contextId; */
/*   uint32_t moduleId; */
/*   uint32_t functionIndex; */
/*   char* name; */
/*   const char* demangled; */
/*   uint32_t calls; // unique function calls */
/*   uint32_t kernel_launches; // total kernel launches for this function */
/*   bool funcinfo_created;  // track whether FunctionInfo created for pprof */
/* }; */

/* class SourceSampling */
/* { */
/*  public: */
/*   uint32_t sid; */
/*   uint32_t fid; */
/*   char* fileName; */
/*   uint32_t lineNumber; */
/*   double timestamp_delta; */
/*   double timestamp_recentacc; */
/*   uint32_t samples; // # samples for sid (file/line) */
/* }; */


/* static std::map<uint32_t, std::list<InstrSampling> > instrSrcMap; */
/* static std::map<uint32_t, std::list<InstrSampling> > instrFuncMap; */
/* static std::map<uint32_t, SourceSampling> srcLocMap; */
/* static std::map<uint32_t, FuncSampling> funcMap; */
/* END: PC Sampling structs */

/************************************************************************
 * Performance Hooks. The following routines are hooks into the execution
 * of GPU applications. 
 */

/* Initialization to be executed at the start of the application */
extern "C" void Tau_gpu_init(void);

/* Initialization of each gpu/device. */
extern "C" void Tau_gpu_device_init(GpuEvent *gpu);

/* Stuff to be performed at the end of the application */
extern "C" void Tau_gpu_exit(void);

/* Entry point for CPU routines */
extern "C" void Tau_gpu_enter_event(const char *functionName);

/* Entry point for CPU routines that initiate a memory copy to the GPU */
extern "C" void Tau_gpu_enter_memcpy_event(const char *functionName,
GpuEvent *gpu, int transferSize, int memcpyType);

/* Entry point for CPU routines that initiate a unified memory copy to the GPU */
extern "C" void Tau_gpu_enter_unifmem_event(const char *functionName,
GpuEvent *gpu, int transferSize, int unifmemType);

/* Exit point for CPU routines */
extern "C" void Tau_gpu_exit_event(const char *functionName);

/* Exit point for CPU routines that initiate a memory copy to the GPU */
extern "C" void Tau_gpu_exit_memcpy_event(const char *functionName,
GpuEvent *gpu, int memcpyType);

/* Exit point for CPU routines that initiate a unified memory copy to the GPU */
extern "C" void Tau_gpu_exit_unifmem_event(const char *functionName,
GpuEvent *gpu, int unifmemType);

/* Creates a GPU event that to be passed on to the register calls later. */
//eventId Tau_gpu_create_gpu_event(const char* name, gpuId *device, FunctionInfo* callingSite, TauGpuContextMap* m);

/* Callback for a GPU event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_gpu_event(GpuEvent *event, double startTime, double endTime);

/* Callback for a Memcpy event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_memcpy_event(GpuEvent *event, double startTime, double endTime, int transferSize, int memcpyType, int direction);

/* Callback for a UnifMem event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_unifmem_event(GpuEvent *event, double startTime, double endTime, int transferSize, int unifmemType, int direction);

/* Callback for a GPU atomic event that is associated with this gpu event. */
extern "C" void Tau_gpu_register_gpu_atomic_event(GpuEvent *event);

extern "C" void TauTraceOneSidedMsg(int type, GpuEvent *gpu, int length, int thread);

/* /\* Callback for a SASS event that occurred earlier in the execution of the                                    */
/*  * program. Timestamps from GPU clock. *\/ */
/* extern "C" void Tau_gpu_register_func_event(GpuEvent *event, int deviceId, double timeStamp, const char* name, uint32_t contextId, uint32_t functionIndex, uint32_t id, uint32_t moduleid, const char *kname, const char *demangled); */

/* extern "C" void Tau_gpu_register_instruction_event(GpuEvent *event, double start, double stop, double delta_tstamp, const char* name, uint32_t correlationId, uint32_t sourceLocatorId, uint32_t functionId, uint32_t pcOffset, uint32_t executed, uint32_t threadsExecuted); */

/* extern "C" void Tau_gpu_register_source_event(GpuEvent *event, double timestamp, const char* name, uint32_t sourceId, const char *fileName, uint32_t lineNumber); */

/* //extern "C" void printInstrMap(void); */
/* extern "C" void printSourceMap(void); */
/* extern "C" void printFuncMap(void); */

/* // routines for calculating kernel level stats */
/* extern "C" double getKernelExecutionTimes(uint32_t functionIndex); */
/* extern "C" uint32_t getKernelSamples(uint32_t functionIndex); */
/* extern "C" const char* getKernelFilePath(uint32_t functionIndex); */
/* extern "C" uint32_t getKernelLineNo(uint32_t functionIndex); */
/* extern "C" void resetKernelExecutionTimes(uint32_t functionIndex); */
/* extern "C" uint32_t getUniqueKernelLaunches(uint32_t functionIndex); */


#endif // __cplusplus
#endif // _TAU_GPU_INTERFACE


