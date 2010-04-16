#ifndef _TAU_CUDA_INTERFACE
#define _TAU_CUDA_INTERFACE

typedef unsigned long long NvU64; /* 0 to 18446744073709551615               */

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

#endif // _TAU_CUDA_INTERFACE


/************************************************************************
 * Performance Hooks. The following routines are hooks into the executaion
 * of CUDA applications. 
 */

/* Initialization to be executed at load time */
int tau_cuda_init(void);

/* Stuff to be performed when the library is destroyed */
void tau_cuda_exit(void);

/* Entry point for cu* routines */
void enter_cu_event(const char *functionName, cuEventId id);

/* Exit point for cu* routines */
void exit_cu_event(const char *functionName, cuEventId id);

/* Callback for a GPU event that occured earlier in the execution of the
 * program. Times are prealignied to the CPU clock. */
void register_gpu_event(const char *functionName, cuEventId id, double startTime, double
endTime);

/* Callback for a Memcpy event that occured earlier in the execution of the
 * program. Times are prealignied to the CPU clock. */
void register_memcpy_event(cuEventId id, double startTime, double
endTime, double transferSize);


