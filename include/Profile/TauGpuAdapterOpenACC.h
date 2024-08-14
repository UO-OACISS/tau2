#ifndef TAU_GPU_ADAPTER_OPENACC_H
#define TAU_GPU_ADAPTER_OPENACC_H

#include <Profile/TauGpu.h>
#include <stdlib.h>
#include <stdint.h>

//copy-pasta and rename from TauGpuAdapterCupti that I feel kind of bad about but including is worse


class OpenACCGpuEvent : public GpuEvent
{
	private:
		static double offset;
		/*
		struct HostMap : public std::map<uint32_t, FunctionInfo*> {
				HostMap() {
						Tau_init_initializeTAU();
				}

				~HostMap() {
						Tau_destructor_trigger();
				}
		};


		HostMap &functionInfoMap_hostLaunch() {
			static HostMap host_map;
			return host_map;
		}
*/
	public:
		uint32_t stream_id;
		uint32_t context_id;
		uint32_t device_id;
		uint32_t task_id;
		uint32_t correlation_id;

		const char* name;

		GpuEventAttributes* event_attrs;
		int num_event_attrs;

		OpenACCGpuEvent(const char* n, uint32_t device, GpuEventAttributes* evt_attrs, int num_evt_attrs) :
			name(n), device_id(device), event_attrs(evt_attrs), num_event_attrs(num_evt_attrs)
		{
			stream_id = 0;
			context_id = 0;
			task_id = -1;
			correlation_id = 0;
		}

		OpenACCGpuEvent(const char* n, uint32_t device, uint32_t stream, uint32_t context, uint32_t task, uint32_t corr_id,
				GpuEventAttributes* evt_attrs, int num_evt_attrs) :
			name(n), device_id(device), stream_id(stream), context_id(context), task_id(task), correlation_id(corr_id),
				event_attrs(evt_attrs), num_event_attrs(num_evt_attrs)
		{}

		OpenACCGpuEvent* getCopy() const
		{
			OpenACCGpuEvent* c = new OpenACCGpuEvent(*this);
			return c;
		}

		bool less_than(const GpuEvent* other) const
		{
			if (!TauEnv_get_thread_per_gpu_stream() || device_id != ((OpenACCGpuEvent*) other)->device_id) {
				return device_id < ((OpenACCGpuEvent*) other)->device_id;
			}
			else if (context_id != ((OpenACCGpuEvent*) other)->context_id) {
				return context_id < ((OpenACCGpuEvent*) other)->context_id;
			}
			else {
				return stream_id < ((OpenACCGpuEvent*) other)->stream_id;
			}
		}

		const char* getName() const
		{
			return name;
		}

		int getTaskId() const
		{
			return task_id;
		}

		FunctionInfo* getCallingSite() const
		{
			FunctionInfo* funcInfo = NULL;
			/*// mostly copy-pasta from the non-cdp case of TauGpuAdapaterCupti, this does not work
			RtsLayer::LockDB();
			std::map<uint32_t, FunctionInfo*>::iterator it = functionInfoMap_hostLaunch().find(correlation_id);
      if (it != functionInfoMap_hostLaunch().end())
      {
        funcInfo = it->second;
        //printf("found host launch site: %s.\n", funcInfo->GetName());
      }
      RtsLayer::UnLockDB();
*/
			if (funcInfo != NULL) {
				funcInfo->SetPrimaryGroupName("TAU_REMOTE");
			}
			return funcInfo;
		}

		void getAttributes(GpuEventAttributes *&attrs, int &num) const
		{
			num = num_event_attrs;
			attrs = event_attrs;
		}

		void recordMetadata(int id) const
		{

		}

		double syncOffset() const
		{
			return offset;
		}

		const char* gpuIdentifier() const
		{
			char* id = (char*) malloc(50*sizeof(char));
			snprintf(id, 50*sizeof(char),  "Dev%d/Ctx%d/Strm%d/cor%d/task%d", device_id, stream_id, context_id, correlation_id, task_id);

			return id;
		}

		// copy-pasta from TauGpuAdapterCupti, what the heck do these even do?
		x_uint64 id_p1() const
		{
			return correlation_id;
		}

		x_uint64 id_p2() const
		{
			return RtsLayer::myNode();
		}

		~OpenACCGpuEvent()
		{
			if (event_attrs) {
				free(event_attrs);
			}
		}
};

//double OpenACCGpuEvent::offset = 0;
/*
void Tau_openacc_register_gpu_event(                                                                                                     const char* name,
  uint32_t device,
  uint32_t stream,
  uint32_t context,
  uint32_t task,
	uint32_t corr_id,
  GpuEventAttributes* event_attrs,
  int num_event_attrs,
  double start,
  double stop);
*/
#endif //TAU_GPU_ADAPTER_OPENACC_H
