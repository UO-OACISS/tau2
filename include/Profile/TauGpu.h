#ifndef _TAU_GPU_INTERFACE
#define _TAU_GPU_INTERFACE

#define MESSAGE_SEND 0
#define MESSAGE_RECV 1

#define MemcpyHtoD false
#define MemcpyDtoH true


/**********************************************
	* Callback into the driver adapter to retrive information about the device ids
	* and event ids 
	*********************************************/

#ifdef __cplusplus
class gpuId {

public:
	virtual char * printId() = 0;
	virtual double id_p1() = 0;
	virtual double id_p2() = 0;
};

class eventId {

};
#endif // __cplusplus

/************************************************************************
 * Performance Hooks. The following routines are hooks into the executaion
 * of GPU applications. 
 */

/* Initialization to be executed at the start of the application */
extern "C" int tau_gpu_init(void);

/* Stuff to be performed at the end of the application */
extern "C" void tau_gpu_exit(void);

/* Entry point for CPU routines */
extern "C" void enter_event(const char *functionName, eventId *id);

/* Entry point for CPU routines that initiate a memory copy to the GPU */
extern "C" void enter_memcpy_event(const char *functionName, eventId *id,
gpuId *device, bool memcpyType);

/* Exit point for CPU routines */
extern "C" void exit_event(const char *functionName, eventId *id);

/* Exit point for CPU routines that initiate a memory copy to the GPU */
extern "C" void exit_memcpy_event(const char *functionName, eventId *id,
gpuId *device, bool memcpyType);

/* Callback for a GPU event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void register_gpu_event(const char *functionName, eventId *id, double startTime, double
endTime);

/* Callback for a Memcpy event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void register_memcpy_event(eventId *id, gpuId *device, double startTime, double
endTime, double transferSize, bool memcpyType);

#endif // _TAU_GPU_INTERFACE

