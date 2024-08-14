#ifndef TAU_GPU_ADAPTER_CUPTI_H
#define TAU_GPU_ADAPTER_CUPTI_H

#include "Profile/CuptiLayer.h"
#include <Profile/TauGpu.h>
#include <stdlib.h>
#include <mutex>

extern "C" void Tau_metadata_task(char *name, const char* value, int tid);

#define uint32_t unsigned int

struct {
	GpuMetadata *list;
	int length;
	} typedef metadata_struct;

struct HostMap : public std::map<uint32_t, FunctionInfo*> {
    HostMap() {
        Tau_init_initializeTAU();
    }

    ~HostMap() {
        Tau_destructor_trigger();
    }
};

HostMap & functionInfoMap_hostLaunch() {
  static HostMap host_map;
  return host_map;
}

struct DeviceMap : public std::map<int64_t, FunctionInfo*> {
    DeviceMap() {
        Tau_init_initializeTAU();
    }

    ~DeviceMap() {
        Tau_destructor_trigger();
    }
};

DeviceMap & functionInfoMap_deviceLaunch() {
  static DeviceMap device_map;
  return device_map;
}

std::mutex & functionInfoMap_mutex() {
  static std::mutex mtx;
  return mtx;
}

struct DeviceInfoMap : public std::map<uint32_t, metadata_struct> {
    DeviceInfoMap() {
        Tau_init_initializeTAU();
    }

    ~DeviceInfoMap() {
        Tau_destructor_trigger();
    }
};

DeviceInfoMap & TheDeviceInfoMap() {
    static DeviceInfoMap device_info_map;
    return device_info_map;
}
//std::map<uint32_t, metadata_struct> deviceInfoMap;

struct KernelContextMap : public std::map<uint32_t, FunctionInfo *> {
    KernelContextMap() {
        Tau_init_initializeTAU();
    }

    ~KernelContextMap() {
        Tau_destructor_trigger();
    }
};

KernelContextMap & TheKernelContextMap() {
    static KernelContextMap kernel_context_map;
    return kernel_context_map;
}
//std::map<uint32_t, FunctionInfo *> kernelContextMap;

class CuptiGpuEvent : public GpuEvent
{
private:
  static double beginTimestamp;
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

	const char *name;
	//FunctionInfo *callingSite;
	GpuEventAttributes *gpu_event_attributes;
	int number_of_gpu_attributes;

	/*CuptiGpuEvent(uint32_t s, uint32_t cn, uint32_t c) { streamId = s; contextId = cn ; correlationId = c; };*/
	CuptiGpuEvent *getCopy() const {
		CuptiGpuEvent *c = new CuptiGpuEvent(*this);
		return c;
	};
 CuptiGpuEvent(const char* n, uint32_t device, uint32_t stream, uint32_t context, uint32_t correlation, int64_t pId, GpuEventAttributes *m, int m_size, uint32_t task_id) : name(n), deviceId(device), streamId(stream), contextId(context), correlationId(correlation), parentGridId(pId), gpu_event_attributes(m), number_of_gpu_attributes(m_size), taskId(task_id) {
    cdpId = 0;
	};

  void setCdp() { cdpId = ++cdpCount; }

	const char* getName() const { return name; }
	int getTaskId() const { return taskId; }

	const char* gpuIdentifier() const {
		char *rtn = (char*) malloc(50*sizeof(char));
		snprintf(rtn, 50*sizeof(char),  "Dev%d/Ctx%d/Strm%d/cdp%d/cor%d/task%d", deviceId, contextId, streamId, cdpId, correlationId, taskId);
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
        // if we are not tracing, only the device ID matters.
        if (!TauEnv_get_thread_per_gpu_stream()) {
            // Devices are different, return
            return deviceId < ((CuptiGpuEvent *)other)->deviceId;
        }
        /* First, check if we are running on different devices */
        if (deviceId != ((CuptiGpuEvent *)other)->deviceId) {
            /* Devices are different, return */
            return deviceId < ((CuptiGpuEvent *)other)->deviceId;
        } else {
            /* same device */
            /* Are we in the same context? */
            if (contextId != ((CuptiGpuEvent *)other)->context()) {
                return contextId < ((CuptiGpuEvent *)other)->context();
            } else {
                /* same context */
                /* Are we on different streams? */
                if (streamId != ((CuptiGpuEvent *)other)->stream()) {
                    return streamId < ((CuptiGpuEvent *)other)->stream();
                } else {
                    /* same stream */
                    /* Are we using CDP kernels? */
                    return cdpId < ((CuptiGpuEvent *)other)->cdp();
                }
            }
        }
    };

	void getAttributes(GpuEventAttributes *&gA, int &num) const
	{
		num = number_of_gpu_attributes;
		gA = gpu_event_attributes;
	}

	void recordMetadata(int id) const
	{
		std::map<uint32_t, metadata_struct>::iterator it = TheDeviceInfoMap().find(deviceId);
		if (it != TheDeviceInfoMap().end())
		{
			GpuMetadata *gpu_metadata = it->second.list;
			int number_of_gpu_metadata = it->second.length;
			//printf("recording %d.\n", number_of_gpu_metadata);
			for (int i=0;i<number_of_gpu_metadata;i++)
			{
				Tau_metadata_task(gpu_metadata[i].name, gpu_metadata[i].value, id);
			}
		}

        char tmpVal[32] = {0};
        snprintf(tmpVal, sizeof(tmpVal),  "%u", deviceId);
        Tau_metadata_task("CUDA Device", tmpVal, id);
        snprintf(tmpVal, sizeof(tmpVal),  "%u", contextId);
        Tau_metadata_task("CUDA Context", tmpVal, id);
        if (TauEnv_get_thread_per_gpu_stream()) {
            snprintf(tmpVal, sizeof(tmpVal),  "%u", streamId);
            Tau_metadata_task("CUDA Stream", tmpVal, id);
        }
        if (cdpId > 0) {
            snprintf(tmpVal, sizeof(tmpVal),  "%u", cdpId);
            Tau_metadata_task("CUDA cdpId", tmpVal, id);
        }
	}

	double syncOffset() const
  {
    return (double) beginTimestamp;
  };
	static void setSyncOffset(double offset)
  {
    beginTimestamp = offset;
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
      // lock required to prevent multithreaded access to the tree
      functionInfoMap_mutex().lock();
      std::map<int64_t, FunctionInfo*>::iterator it = functionInfoMap_deviceLaunch().find(parentGridId);
      if (it != functionInfoMap_deviceLaunch().end())
      {
        funcInfo = it->second;
        //printf("found device launch site: %s.\n", funcInfo->GetName());
      }
      functionInfoMap_mutex().unlock();
    } else {
      // lock required to prevent multithreaded access to the tree
      functionInfoMap_mutex().lock();
      std::map<uint32_t, FunctionInfo*>::iterator it = functionInfoMap_hostLaunch().find(correlationId);
      if (it != functionInfoMap_hostLaunch().end())
      {
        funcInfo = it->second;
        //printf("found host launch site: %s.\n", funcInfo->GetName());
      }
      functionInfoMap_mutex().unlock();
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

#endif //TAU_GPU_ADAPTER_CUPTI_H
