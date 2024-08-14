#ifndef TAU_GPU_ADAPTER_OPENMP_H
#define TAU_GPU_ADAPTER_OPENMP_H

#include <Profile/TauGpu.h>
#include <stdlib.h>
#include <stdint.h>

//copy-pasta and rename from TauGpuAdapterCupti that I feel kind of bad about but including is worse


class OpenMPGpuEvent : public GpuEvent
{
    private:
        static double offset;
    public:
        uint32_t device_id;
        uint32_t thread_id;
        uint32_t task_id;
        uint32_t correlation_id;

        const char* name;

        GpuEventAttributes* event_attrs;
        int num_event_attrs;

        OpenMPGpuEvent(const char* n, uint32_t device, uint32_t _thread,
            GpuEventAttributes* evt_attrs, int num_evt_attrs) :
            device_id(device), thread_id(_thread), name(n),
            event_attrs(evt_attrs), num_event_attrs(num_evt_attrs)
        {
            task_id = 0;
            correlation_id = 0;
        }

        OpenMPGpuEvent(const char* n, uint32_t device, uint32_t _thread,
            uint32_t task, uint32_t corr_id, GpuEventAttributes* evt_attrs,
            int num_evt_attrs) : device_id(device), thread_id(_thread),
            task_id(task), correlation_id(corr_id), name(n),
            event_attrs(evt_attrs), num_event_attrs(num_evt_attrs)
        {}

        OpenMPGpuEvent* getCopy() const
        {
            OpenMPGpuEvent* c = new OpenMPGpuEvent(*this);
            return c;
        }

        bool less_than(const GpuEvent* other) const
        {
            if (!TauEnv_get_thread_per_gpu_stream() ||
                device_id != ((OpenMPGpuEvent*) other)->device_id) {
                return device_id < ((OpenMPGpuEvent*) other)->device_id;
            } else {
                return thread_id < ((OpenMPGpuEvent*) other)->thread_id;
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
        { }

        double syncOffset() const
        {
            return offset;
        }

        const char* gpuIdentifier() const
        {
            char* id = (char*) malloc(50*sizeof(char));
            snprintf(id, 50*sizeof(char),  "Dev%d/Thrd%d/cor%d/task%d", device_id, thread_id, correlation_id, task_id);

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

        ~OpenMPGpuEvent()
        {
            if (event_attrs) {
                free(event_attrs);
            }
        }
};

//double OpenMPGpuEvent::offset = 0;
void Tau_openmp_register_gpu_event(
    const char* name,
    uint32_t device,
    uint32_t thread_id,
    uint32_t task,
    uint32_t corr_id,
    GpuEventAttributes* event_attrs,
    int num_event_attrs,
    double start,
    double stop);

int Tau_openmp_get_taskid_from_gpu_event(uint32_t deviceId, uint32_t threadId);

#endif //TAU_GPU_ADAPTER_OPENMP_H
