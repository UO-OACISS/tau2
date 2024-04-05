#ifdef TAU_GPU
#include <Profile/TauGpuAdapterOpenMP.h>
#include <Profile/TauGpu.h>

double OpenMPGpuEvent::offset = 0;

void Tau_openmp_register_gpu_event(
	const char* name,
	uint32_t device,
	uint32_t thread_id,
	uint32_t task,
	uint32_t corr_id,
	GpuEventAttributes* event_attrs,
	int num_event_attrs,
	double start,
	double stop)
{
	OpenMPGpuEvent event = OpenMPGpuEvent(name, device, thread_id, task, corr_id, event_attrs, num_event_attrs);
	Tau_gpu_register_gpu_event(&event, start, stop);
}

/* The one and only way to get the virtual thread ID is to create a dummy
 * GPU event and get the task with a lookup.  The lookup logic is defined
 * in TauGpuAdapterCupti::less_than() */
int get_task(GpuEvent *new_task); // defined in TauGpu.cpp
int Tau_openmp_get_taskid_from_gpu_event(uint32_t deviceId, uint32_t threadId) {
    OpenMPGpuEvent gpu_event = OpenMPGpuEvent("", deviceId, threadId, 0, 0, NULL, 0);
    return get_task(&gpu_event);
}
#endif
