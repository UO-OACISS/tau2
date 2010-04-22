#ifndef _TAU_CUDA_INTERFACE
#define _TAU_CUDA_INTERFACE

typedef unsigned long long NvU64; /* 0 to 18446744073709551615               */
typedef unsigned int NvU32;

#define MESSAGE_SEND 0
#define MESSAGE_RECV 1


/* cu Event ids are complex, both a context and api call id*/
struct cuEventId
{
	NvU64 contextId;
	NvU64 callId;

	cuEventId(const NvU64 a, const NvU64 b) :
		contextId(a), callId(b) {}

	bool operator<(const cuEventId& A) const
	{ 
		if (contextId == A.contextId)
		{
			return callId<A.callId; 
		}
		else
			return contextId<A.contextId;
	}
};

struct gpuId
{
	NvU64 contextId;
	NvU32 deviceId;

	gpuId(const NvU64 a, const NvU32 b) :
		contextId(a), deviceId(b) {}

	bool operator<(const gpuId& A) const
	{ 
		if (contextId == A.contextId)
		{
			return deviceId<A.deviceId; 
		}
		else
			return contextId<A.contextId;
	}
};

#endif // _TAU_CUDA_INTERFACE


/************************************************************************
 * Performance Hooks. The following routines are hooks into the executaion
 * of CUDA applications. 
 */

/* Initialization to be executed at load time */
extern "C" int tau_cuda_init(void);

/* Stuff to be performed when the library is destroyed */
extern "C" void tau_cuda_exit(void);

/* Entry point for cu* routines */
extern "C" void enter_cu_event(const char *functionName, cuEventId id);

/* Entry point for cu* routines that initiate memory copies. */
extern "C" void enter_cu_memcpy_event(const char *functionName, cuEventId id,
gpuId device);

/* Exit point for cu* routines */
extern "C" void exit_cu_event(const char *functionName, cuEventId id);

/* Callback for a GPU event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void register_gpu_event(const char *functionName, cuEventId id, double startTime, double
endTime);

/* Callback for a Memcpy event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void register_memcpy_event(cuEventId id, gpuId device, double startTime, double
endTime, double transferSize);


