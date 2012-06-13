#ifndef _TAU_GPU_INTERFACE
#define _TAU_GPU_INTERFACE

#define MESSAGE_SEND 0
#define MESSAGE_RECV 1

enum Memcpy { MemcpyHtoD = 0, MemcpyDtoH = 1, MemcpyDtoD = 2, MemcpyUnknown = 3 };

#define TAU_GPU_UNKNOW_TRANSFER_SIZE -1
#define TAU_GPU_USE_DEFAULT_NAME ""

#define TAU_MAX_NUMBER_OF_GPU_THREADS TAU_MAX_THREADS

#include <Profile/tau_types.h>
#ifdef __cplusplus

#include <Profile/Profiler.h>
using namespace tau;

//typedef map<TauContextUserEvent*, TAU_EVENT_DATATYPE> TauGpuContextMap;

/* Struct to contain the user event data for each GPU event. */
typedef struct {
	TauContextUserEvent *userEvent;
	TAU_EVENT_DATATYPE data;

} GpuEventAttributes;

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

	//CPU event callsite for this event. Used to create a the GPU callsite event paths.
	virtual FunctionInfo* getCallingSite() const = 0;

	//GPU event attributes, used to create context user events for each GPU Event.
	//Warning: atr can be set to NULL.
	virtual void getAttributes(GpuEventAttributes *&atr, int &numberOfAttributes) const = 0;

	//Synchronization offset for this GPU. 
	virtual double syncOffset() const = 0; 

	//the GPU identification (for debugging purposes).
	virtual const char *gpuIdentifier() const = 0;

	//return the conponents of the gpu identifier, used for store the gpu id in
	//the trace files.
	virtual const x_uint64 id_p1() const = 0;
	virtual const x_uint64 id_p2() const = 0;

};

/*
class gpuId {

public:
	virtual gpuId *getCopy() const = 0;
	virtual char * printId() const = 0;
	virtual x_uint64 id_p1() const = 0;
	virtual x_uint64 id_p2() const = 0;
	virtual bool less_than(const gpuId *other) const = 0;
	virtual double syncOffset() = 0;
	//virtual bool operator<(const gpuId& A) const;
};
	
typedef map<TauContextUserEvent*, TAU_EVENT_DATATYPE> TauGpuContextMap;

class eventId {
public:
	//virtual bool operator<(const eventId& A) const;
	gpuId *device;
	const char *name;
	// rountine where this gpu Kernel was launched.
	FunctionInfo* callingSite;

	//map of context event to be trigger with the kernel
	TauGpuContextMap* contextEventMap;

	eventId(const char* n, gpuId* d, FunctionInfo *s,
	TauGpuContextMap* map) {
		name = n;
		device = d;
		callingSite = s;
		contextEventMap = map;
	}
};
*/
/************************************************************************
 * Performance Hooks. The following routines are hooks into the execution
 * of GPU applications. 
 */

/* Initialization to be executed at the start of the application */
extern "C" void Tau_gpu_init(void);

/* Stuff to be performed at the end of the application */
extern "C" void Tau_gpu_exit(void);

/* Entry point for CPU routines */
extern "C" void Tau_gpu_enter_event(const char *functionName);

/* Entry point for CPU routines that initiate a memory copy to the GPU */
extern "C" void Tau_gpu_enter_memcpy_event(const char *functionName,
GpuEvent *gpu, int transferSize, int memcpyType);

/* Exit point for CPU routines */
extern "C" void Tau_gpu_exit_event(const char *functionName);

/* Exit point for CPU routines that initiate a memory copy to the GPU */
extern "C" void Tau_gpu_exit_memcpy_event(const char *functionName,
GpuEvent *gpu, int memcpyType);

/* Creates a GPU event that to be passed on to the register calls later. */
//eventId Tau_gpu_create_gpu_event(const char* name, gpuId *device, FunctionInfo* callingSite, TauGpuContextMap* m);

/* Callback for a GPU event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_gpu_event(GpuEvent *event, double startTime, double endTime);

/* Callback for a Memcpy event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_memcpy_event(GpuEvent *event, double startTime, double endTime, int transferSize, int memcpyType);

extern "C" void TauTraceOneSidedMsg(bool type, GpuEvent *gpu, int length, int thread);

#endif // __cplusplus
#endif // _TAU_GPU_INTERFACE


