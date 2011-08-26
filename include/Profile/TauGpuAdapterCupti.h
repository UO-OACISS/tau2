#include <Profile/TauGpu.h>
#include <cuda.h>
#include <cupti.h>

#ifdef TAU_BFD
#define HAVE_DECL_BASENAME 1
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include <demangle.h>
#  endif /* HAVE_GNU_DEMANGLE */
#  include <bfd.h>
#endif /* TAU_BFD */

#define CUDA_CHECK_ERROR(err, str) \
	if (err != CUDA_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

#define CUPTI_CHECK_ERROR(err, str) \
	if (err != CUPTI_SUCCESS) \
  { \
		fprintf(stderr, str); \
		exit(1); \
	} \

#define ACTIVITY_BUFFER_SIZE (4096 * 32)

uint8_t *activityBuffer;
CUpti_SubscriberHandle subscriber;

void Tau_cupti_register_sync_event();

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params);

void Tau_cupti_record_activity(CUpti_Activity *record);

void __attribute__ ((constructor)) Tau_cupti_onload(void);
void __attribute__ ((destructor)) Tau_cupti_onunload(void);

int getMemcpyType(int kind);
const char* demangleName(const char *n);

class cuptiGpuId : public gpuId
{
	uint32_t streamId;

public:

	cuptiGpuId(uint32_t s) { streamId = s; };
	cuptiGpuId *getCopy() { 
		cuptiGpuId *c = new cuptiGpuId(*this);
		return c; 
	};
	char* printId() {
		char *rtn = (char*) malloc(50*sizeof(char));
		sprintf(rtn, "%d", streamId);
		return rtn;
	};
	x_uint64 id_p1() {
		return streamId;
	};
	x_uint64 id_p2() { return 0; };

	bool equals(const gpuId *other) const
	{
		return streamId == ((cuptiGpuId *)other)->id_p1();
	};

	double syncOffset() { return 0; };
};


class cuptiRecord : public eventId {

	cuptiGpuId *device;
	const char *name;
	FunctionInfo *callingSite;

public:
	cuptiRecord(const char* n, cuptiGpuId *id, FunctionInfo *site) : eventId(n, id, site)
	{
	};
	cuptiRecord(const char* n, uint32_t id) : eventId(n, &cuptiGpuId(id), NULL) {};

};
