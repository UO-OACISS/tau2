#include "Profile/CuptiLayer.h"
#include <Profile/TauGpu.h>
#include <stdlib.h>

extern "C" void Tau_metadata_task(char *name, const char* value, int tid);

#define uint32_t unsigned int

struct {
	GpuMetadata *list;
	int length;
	} typedef metadata_struct;

static std::map<uint32_t, FunctionInfo*> host_map;
std::map<uint32_t, FunctionInfo*>& functionInfoMap_hostLaunch() {
  return host_map;
}
static std::map<int64_t, FunctionInfo*> device_map;
std::map<int64_t, FunctionInfo*>& functionInfoMap_deviceLaunch() {
  return device_map;
}

std::map<uint32_t, metadata_struct> deviceInfoMap;
std::map<uint32_t, FunctionInfo *> kernelContextMap;

class CuptiGpuEvent : public GpuEvent
{
public:
	uint32_t streamId;
	uint32_t contextId;
	uint32_t deviceId;
	uint32_t correlationId;
  int64_t parentGridId;
  uint32_t taskId;
  //CDP kernels can overlap with other kernels so each one needs to be in a
  //seperate 'thread' of execution.
  uint32_t cdpId;
  static uint32_t cdpCount;

	//This event is tied to the entire deivce not a particular stream or context.
	//Used for recording device metadata.
	bool deviceContainer;
	
	const char *name;
	//FunctionInfo *callingSite;
	GpuEventAttributes *gpu_event_attributes;
	int number_of_gpu_attributes;

  static double beginTimestamp;

	/*CuptiGpuEvent(uint32_t s, uint32_t cn, uint32_t c) { streamId = s; contextId = cn ; correlationId = c; };*/
	CuptiGpuEvent *getCopy() const { 
		CuptiGpuEvent *c = new CuptiGpuEvent(*this);
		return c; 
	};
 CuptiGpuEvent(const char* n, uint32_t device, GpuEventAttributes *m, int m_size) : name(n), deviceId(device), gpu_event_attributes(m), number_of_gpu_attributes(m_size) {
		deviceContainer = true;
		streamId = 0;
		contextId = 0;
		correlationId = -1;
    cdpId = 0;
    parentGridId = 0;
    taskId = -1;
	};
 CuptiGpuEvent(const char* n, uint32_t device, uint32_t stream, uint32_t context, uint32_t correlation, int64_t pId, GpuEventAttributes *m, int m_size, uint32_t task_id) : name(n), deviceId(device), streamId(stream), contextId(context), correlationId(correlation), parentGridId(pId), gpu_event_attributes(m), number_of_gpu_attributes(m_size), taskId(task_id) {
		deviceContainer = false;
    cdpId = 0;
	};

  void setCdp() { cdpId = ++cdpCount; }
  
	const char* getName() const { return name; }
	int getTaskId() const { return taskId; }

	const char* gpuIdentifier() const {
		char *rtn = (char*) malloc(50*sizeof(char));
		sprintf(rtn, "%d/%d/%d/%d/%d", deviceId, streamId, contextId, correlationId, taskId);
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
		if (deviceContainer || ((CuptiGpuEvent *)other)->deviceContainer) {
			return deviceId < ((CuptiGpuEvent *)other)->deviceId;
		}
		else {
			if (contextId == ((CuptiGpuEvent *)other)->context()) {
        if (streamId == ((CuptiGpuEvent *)other)->stream()) {
          return cdpId < ((CuptiGpuEvent *)other)->cdp();
        } else {
				  return streamId < ((CuptiGpuEvent *)other)->stream();
        }
			} else {
				return contextId < ((CuptiGpuEvent *)other)->context();
			}
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

	void recordMetadata(int id) const
	{
		std::map<uint32_t, metadata_struct>::iterator it = deviceInfoMap.find(deviceId);
		if (it != deviceInfoMap.end())
		{
			GpuMetadata *gpu_metadata = it->second.list;
			int number_of_gpu_metadata = it->second.length;
			//printf("recording %d.\n", number_of_gpu_metadata);
			for (int i=0;i<number_of_gpu_metadata;i++)
			{
				Tau_metadata_task(gpu_metadata[i].name, gpu_metadata[i].value, id);
			}
		}
	}

	double syncOffset() const 
  { 
    return (double) beginTimestamp; 
  };
	uint32_t stream() { return streamId; };
	uint32_t context() { return contextId; };
	uint32_t cdp() { return cdpId; };
	FunctionInfo* getCallingSite() const
	{
    //printf("cdp id is: %d.\n", cdpId);
    FunctionInfo *funcInfo = NULL;
    if (cdpId != 0)
    {
      std::map<int64_t, FunctionInfo*>::iterator it = functionInfoMap_deviceLaunch().find(parentGridId);
      if (it != functionInfoMap_deviceLaunch().end())
      { 
        funcInfo = it->second;
        //printf("found device launch site: %s.\n", funcInfo->GetName());
      }
    } else {
      std::map<uint32_t, FunctionInfo*>::iterator it = functionInfoMap_hostLaunch().find(correlationId);
      if (it != functionInfoMap_hostLaunch().end())
      {
        funcInfo = it->second;
        //printf("found host launch site: %s.\n", funcInfo->GetName());
      }
    }
    if (funcInfo != NULL) {
      funcInfo->SetPrimaryGroupName("TAU_REMOTE");
    }
    return funcInfo;
	};

	~CuptiGpuEvent()
	{
		free(gpu_event_attributes);
	}

};

uint32_t CuptiGpuEvent::cdpCount = 0;
double CuptiGpuEvent::beginTimestamp = 0;
