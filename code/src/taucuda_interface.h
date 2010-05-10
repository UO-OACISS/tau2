#ifndef _TAU_CUDA_INTERFACE
#define _TAU_CUDA_INTERFACE

typedef unsigned long long NvU64; /* 0 to 18446744073709551615               */
typedef unsigned int NvU32;

#define MESSAGE_SEND 0
#define MESSAGE_RECV 1


/* cu Event ids are complex, both a context and api call id
struct eventId
{
	NvU64 contextId;
	NvU64 callId;

	eventId(const NvU64 a, const NvU64 b) :
		contextId(a), callId(b) {}
#ifdef __cplusplus
	bool operator<(const eventId& A) const
	{ 
		if (contextId == A.contextId)
		{
			return callId<A.callId; 
		}
		else
			return contextId<A.contextId;
	}
#endif // __cplusplus
};

struct gpuId
{
	NvU64 contextId;
	NvU32 deviceId;

	gpuId(const NvU64 a, const NvU32 b) :
		contextId(a), deviceId(b) {}

#ifdef __cplusplus
	bool operator<(const gpuId& A) const
	{ 
		if (contextId == A.contextId)
		{
			return deviceId<A.deviceId; 
		}
		else
			return contextId<A.contextId;
	}
#endif // __cplusplus
};
*/


/**********************************************
	* Callback into the driver adapter to retrive information about the device ids
	* and event ids 
	*********************************************/

#ifdef __cplusplus
class gpuId {

public:
	char * printId();
	double id_p1();
	double id_p2();
};

class eventId {

public:
	bool operator<(const eventId& A) const;
};
#endif // __cplusplus

/************************************************************************
 * Performance Hooks. The following routines are hooks into the executaion
 * of CUDA applications. 
 */

/* Initialization to be executed at load time */
extern "C" int tau_cuda_init(void);

/* Stuff to be performed when the library is destroyed */
extern "C" void tau_cuda_exit(void);

/* Entry point for cu* routines */
extern "C" void enter_cu_event(const char *functionName, eventId id);

/* Entry point for cu* routines that initiate memory copies. */
extern "C" void enter_cu_memcpy_event(const char *functionName, eventId id,
gpuId device);

/* Exit point for cu* routines */
extern "C" void exit_cu_event(const char *functionName, eventId id);

/* Callback for a GPU event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void register_gpu_event(const char *functionName, eventId id, double startTime, double
endTime);

/* Callback for a Memcpy event that occurred earlier in the execution of the
 * program. Times are pre-aligned to the CPU clock. */
extern "C" void register_memcpy_event(eventId id, gpuId device, double startTime, double
endTime, double transferSize);

#endif // _TAU_CUDA_INTERFACE

