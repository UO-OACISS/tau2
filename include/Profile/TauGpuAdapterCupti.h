#include <Profile/TauGpu.h>
#include <cuda.h>
#include <cupti.h>
#include <sstream>
#include <vector>

#if CUPTI_API_VERSION >= 2

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

#define ACTIVITY_BUFFER_SIZE (4096 * 1024)

uint8_t *activityBuffer;
CUpti_SubscriberHandle subscriber;

int number_of_streams;
vector<int> streamIds;

void Tau_cupti_register_sync_event(CUcontext c, uint32_t stream);

void Tau_cupti_callback_dispatch(void *ud, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void *params);

void Tau_cupti_record_activity(CUpti_Activity *record);

void __attribute__ ((constructor)) Tau_cupti_onload(void);
void __attribute__ ((destructor)) Tau_cupti_onunload(void);

void get_values_from_memcpy(const CUpti_CallbackData *info, CUpti_CallbackId id, CUpti_CallbackDomain domain, int &kind, int &count);

int getMemcpyType(int kind);
const char* demangleName(const char *n);

int getParentFunction(uint32_t id);

bool function_is_sync(CUpti_CallbackId id);
bool function_is_memcpy(CUpti_CallbackId id, CUpti_CallbackDomain domain);
bool function_is_launch(CUpti_CallbackId id);
bool function_is_exit(CUpti_CallbackId id);

bool registered_sync = false;

bool cupti_api_runtime();
bool cupti_api_driver();

map<uint32_t, FunctionInfo*> functionInfoMap;

class CuptiGpuEvent : public GpuEvent
{
public:
	uint32_t streamId;
	uint32_t contextId;
	uint32_t correlationId;
	
	const char *name;
	//FunctionInfo *callingSite;
	GpuEventAttributes *gpu_event_attributes;
	int number_of_gpu_attributes;

	/*CuptiGpuEvent(uint32_t s, uint32_t cn, uint32_t c) { streamId = s; contextId = cn ; correlationId = c; };*/
	CuptiGpuEvent *getCopy() const { 
		CuptiGpuEvent *c = new CuptiGpuEvent(*this);
		return c; 
	};
	CuptiGpuEvent(const char* n, uint32_t stream, uint32_t context, uint32_t correlation, GpuEventAttributes *m, int m_size) : name(n), streamId(stream), contextId(context), correlationId(correlation), gpu_event_attributes(m), number_of_gpu_attributes(m_size) {};

	const char* getName() const { return name; }

	const char* gpuIdentifier() const {
		char *rtn = (char*) malloc(50*sizeof(char));
		sprintf(rtn, "%d/%d", streamId, correlationId);
		return rtn;
	};
	const x_uint64 id_p1() const {
		return correlationId;
	};
	const x_uint64 id_p2() const { 
		return RtsLayer::myNode(); 
	};

	bool less_than(const GpuEvent *other) const
	{
		if (contextId == ((CuptiGpuEvent *)other)->context()) {
			return streamId < ((CuptiGpuEvent *)other)->stream();
		} else {
			return contextId < ((CuptiGpuEvent *)other)->context();
		}
		/*
		if (ret) { printf("%s equals %s.\n", printId(), ((CuptiGpuEvent *)other)->printId()); }
		else { printf("%s does not equal %s.\n", printId(), ((CuptiGpuEvent *)other)->printId());}
		return ret;
		*/
	};

	void getAttributes(GpuEventAttributes *&gA, int &num) const
	{
		num = number_of_gpu_attributes;
		gA = gpu_event_attributes;
	}

	double syncOffset() const { return 0; };
	uint32_t stream() { return streamId; };
	uint32_t context() { return contextId; };
	
	FunctionInfo* getCallingSite() const
	{
		FunctionInfo *funcInfo = NULL;
		map<uint32_t, FunctionInfo*>::iterator it = functionInfoMap.find(correlationId);
		if (it != functionInfoMap.end())
		{
			funcInfo = it->second;
		}
		return funcInfo;
	};
};

#define CAST_TO_RUNTIME_MEMCPY_TYPE_AND_CALL(name, id, info, kind, count) \
	if ((id) == CUPTI_RUNTIME_TRACE_CBID_##name##_v3020) \
	{ \
		kind = ((name##_v3020_params *) info->functionParams)->kind; \
		count = ((name##_v3020_params *) info->functionParams)->count; \
	}

#define S(x) #x
#define SX(x) S(x)
#define RECORD_DEVICE_METADATA(name, device) \
  std::ostringstream str_##name; \
	str_##name << device->name; \
	Tau_metadata("GPU " SX(name), str_##name.str().c_str()); 

#endif

