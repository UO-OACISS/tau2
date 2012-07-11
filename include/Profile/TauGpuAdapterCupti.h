#include <Profile/TauGpu.h>


#define uint32_t unsigned int

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
	x_uint64 id_p1() const {
		return correlationId;
	};
	x_uint64 id_p2() const { 
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
