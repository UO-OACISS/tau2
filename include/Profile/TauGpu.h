#ifndef _TAU_GPU_INTERFACE
#define _TAU_GPU_INTERFACE

#define MESSAGE_SEND 0
#define MESSAGE_RECV 1

#define MemcpyHtoD false
#define MemcpyDtoH true

#include<Profile/tau_types.h>

/**********************************************
	* Callback into the driver adapter to retrive information about the device ids
	* and event ids 
	*********************************************/

#ifdef __cplusplus
class gpuId {

public:
	virtual char * printId() = 0;
	virtual x_uint64 id_p1() = 0;
	virtual x_uint64 id_p2() = 0;
};

class eventId {

};

/************************************************************************
 * Performance Hooks. The following routines are hooks into the executaion
 * of GPU applications. 
 */

/* Initialization to be executed at the start of the application */
extern "C" int Tau_gpu_init(void);

/* Stuff to be performed at the end of the application */
extern "C" void Tau_gpu_exit(void);

/* Entry point for CPU routines */
extern "C" void Tau_gpu_enter_event(const char *functionName, eventId *id);

/* Entry point for CPU routines that initiate a memory copy to the GPU */
extern "C" void Tau_gpu_enter_memcpy_event(eventId *id,
gpuId *device, bool memcpyType);

/* Exit point for CPU routines */
extern "C" void Tau_gpu_exit_event(const char *functionName, eventId *id);

/* Exit point for CPU routines that initiate a memory copy to the GPU */
extern "C" void Tau_gpu_exit_memcpy_event(eventId *id,
gpuId *device, bool memcpyType);

/* Callback for a GPU event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_gpu_event(const char *functionName, eventId *id, double startTime, double
endTime);

/* Callback for a Memcpy event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void Tau_gpu_register_memcpy_event(eventId *id, gpuId *device, double startTime, double
endTime, int transferSize, bool memcpyType);

#endif // __cplusplus
#endif // _TAU_GPU_INTERFACE

