#include <Profile/TauGpuAdapterCupti.h>
#include <Profile/TauMetrics.h>

extern void *Tau_pure_search_for_function(const char *name, int create);

extern "C" void Tau_cupti_set_offset(double cpu_gpu_offset) {
    // printf("setting offset to %f.\n", cpu_gpu_offset);
    CuptiGpuEvent::setSyncOffset(cpu_gpu_offset);
}

extern "C" void Tau_cupti_find_context_event(
        TauContextUserEvent** u,
        const char *name, bool context) {
    Tau_pure_context_userevent((void **) u, name);
    (*u)->SetContextEnabled(context);
}


extern "C" void Tau_cupti_register_metadata(
        uint32_t deviceId,
        GpuMetadata *metadata,
        int metadata_size) {
    metadata_struct m;
    m.list = metadata;
    m.length = metadata_size;
    TheDeviceInfoMap()[deviceId] = m;
}
extern "C" void Tau_cupti_register_host_calling_site(
        uint32_t correlationId,
        const char *name) {
    //find thread with launch event.
    FunctionInfo* launch = (FunctionInfo *) Tau_pure_search_for_function(name, 0);
    for (int i=0; i<TAU_MAX_THREADS; i++)
    {
        if (TauInternal_CurrentProfiler(i) != NULL &&
            launch == TauInternal_CurrentProfiler(i)->ThisFunction &&
            TauInternal_CurrentProfiler(i)->CallPathFunction != NULL)
        {
            // lock required to prevent multithreaded access to the tree
            RtsLayer::LockDB();
            functionInfoMap_hostLaunch()[correlationId] =
                TauInternal_CurrentProfiler(i)->CallPathFunction;
            RtsLayer::UnLockDB();
            break;
        }
    }
    //functionInfoMap_hostLaunch()[correlationId] =
    //  TauInternal_CurrentProfiler(RtsLayer::myThread())->CallPathFunction;
}

extern "C" void Tau_cupti_register_device_calling_site(
        int64_t correlationId,
        const char *name) {
    // lock required to prevent multithreaded access to the tree
    RtsLayer::LockDB();
    functionInfoMap_deviceLaunch()[correlationId] = (FunctionInfo *) Tau_pure_search_for_function(name, 0);
    RtsLayer::UnLockDB();
}
extern "C" void Tau_cupti_register_sync_site(
        uint32_t correlationId,
        uint64_t *counters,
        int number_of_counters
        ) {
}


extern "C" void Tau_cupti_enter_memcpy_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t contextId,
        uint32_t correlationId,
        int bytes_copied,
        int memcpy_type,
        int taskId) {
    //Empty list of gpu attributes
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId, streamId, contextId, 0, correlationId, NULL, 0, taskId);
    Tau_gpu_enter_memcpy_event(name, &gpu_event, bytes_copied, memcpy_type);
}

extern "C" void Tau_cupti_exit_memcpy_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t contextId,
        uint32_t correlationId,
        int bytes_copied,
        int memcpy_type,
        int taskId) {
    //Empty list of gpu attributes
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId, streamId, contextId, 0, correlationId, NULL, 0, taskId);
    Tau_gpu_exit_memcpy_event(name, &gpu_event, memcpy_type);
}

extern "C" void Tau_cupti_register_memcpy_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t contextId,
        uint32_t correlationId,
        double start,
        double stop,
        int bytes_copied,
        int memcpy_type,
        int direction,
        int taskId) {
    //Empty list of gpu attributes
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId, streamId, contextId, correlationId, correlationId, NULL, 0, taskId);
    Tau_gpu_register_memcpy_event(&gpu_event,
            start, stop, bytes_copied, memcpy_type, direction);
}


extern "C" void Tau_cupti_register_unifmem_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t processId,
        uint64_t start,
        uint64_t end,
        uint64_t value,
        int unifmem_type,
        int direction,
        int taskId) {
    //Empty list of gpu attributes
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId, streamId, 0, 0, -1, NULL, 0, taskId);
    // start/stop times set to timestamp
    Tau_gpu_register_unifmem_event(&gpu_event, start, end, value, unifmem_type, direction);
}

extern "C" void Tau_cupti_register_gpu_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t contextId,
        uint32_t correlationId,
        int64_t parentGridId,
        bool cdp,
        GpuEventAttributes *gpu_attributes,
        int number_of_attributes,
        double start,
        double stop,
        int taskId) {
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId,
            streamId,
            contextId,
            correlationId,
            parentGridId, gpu_attributes, number_of_attributes, taskId);
    if (cdp) {
        //printf("setting CDP flag.\n");
        gpu_event.setCdp();
    }
    Tau_gpu_register_gpu_event(&gpu_event, start, stop);
}

extern "C" void Tau_cupti_register_gpu_sync_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t contextId,
        uint32_t correlationId,
        double start,
        double stop,
        int taskId) {
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId,
            streamId,
            contextId,
            correlationId,
            0, NULL, 0, taskId);
    Tau_gpu_register_sync_event(&gpu_event, start, stop);
}

extern "C" void Tau_cupti_register_gpu_atomic_event(
        const char *name,
        uint32_t deviceId,
        uint32_t streamId,
        uint32_t contextId,
        uint32_t correlationId,
        GpuEventAttributes *gpu_attributes,
        int number_of_attributes,
        int taskId) {
    CuptiGpuEvent gpu_event = CuptiGpuEvent(name,
            deviceId, streamId, contextId, correlationId, 0, gpu_attributes,
            number_of_attributes, taskId);
    Tau_gpu_register_gpu_atomic_event(&gpu_event);
}

/* The one and only way to get the virtual thread ID is to create a dummy
 * GPU event and get the task with a lookup.  The lookup logic is defined
 * in TauGpuAdapterCupti::less_than() */
int get_task(GpuEvent *new_task); // defined in TauGpu.cpp
int get_taskid_from_gpu_event(uint32_t deviceId, uint32_t streamId,
    uint32_t contextId, bool cdp) {
    CuptiGpuEvent gpu_event = CuptiGpuEvent("", deviceId, streamId,
        contextId, 0, 0, NULL, 0, 0);
    if (cdp) { gpu_event.setCdp(); }
    return get_task(&gpu_event);
}

/* Records a synchronous event on the GPU, as seen from CPU! */
extern "C" void Tau_cupti_gpu_enter_event_from_cpu(const char* name, int tid)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("entering cu event: %s on virtual thread %d.\n", name, tid);
#endif
  // get a timestamp from the CPU thread
  double startTime = (double)TauMetrics_getTraceMetricValue(RtsLayer::myThread());
  // adjust it.
  CuptiGpuEvent gpu_event = CuptiGpuEvent("", 0, 0, 0, 0, 0, NULL, 0, 0);
  const double syncStartTime = startTime + gpu_event.syncOffset();
  Tau_gpu_enter_event_from_cpu(name, tid, syncStartTime);
}

/* Records a synchronous event on the GPU, as seen from CPU! */
extern "C" void Tau_cupti_gpu_exit_event_from_cpu(const char* name, int tid)
{
#ifdef DEBUG_PROF
  TAU_VERBOSE("exiting cu event: %s on virtual thread %d.\n", name, tid);
#endif
  // get a timestamp from the CPU thread
  double endTime = (double)TauMetrics_getTraceMetricValue(RtsLayer::myThread());
  // adjust it.
  CuptiGpuEvent gpu_event = CuptiGpuEvent("", 0, 0, 0, 0, 0, NULL, 0, 0);
  const double syncEndTime = endTime + gpu_event.syncOffset();
  Tau_gpu_exit_event_from_cpu(name, tid, syncEndTime);
}



