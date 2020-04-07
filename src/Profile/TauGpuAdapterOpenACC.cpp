#include <Profile/TauGpuAdapterOpenACC.h>
#include <Profile/TauGpu.h>

double OpenACCGpuEvent::offset = 0;

extern "C" void Tau_openacc_register_gpu_event(
	const char* name,
	uint32_t device,
	uint32_t stream,
	uint32_t context,
	uint32_t task,
	uint32_t corr_id,
	GpuEventAttributes* event_attrs,
	int num_event_attrs,
	double start,
	double stop)
{
	OpenACCGpuEvent event = OpenACCGpuEvent(name, device, stream, context, task, corr_id, event_attrs, num_event_attrs);
	Tau_gpu_register_gpu_event(&event, start, stop);
}
