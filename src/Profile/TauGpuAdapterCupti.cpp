#include <Profile/TauGpuAdapterCupti.h>

extern "C" void Tau_cupti_set_offset(
            double timestamp
            ) {
              CuptiGpuEvent::beginTimestamp = timestamp;
            }

extern "C" void Tau_cupti_find_context_event(
						TauContextUserEvent** u, 
						const char *name
						) {
							Tau_pure_context_userevent((void **) u, name);
							(*u)->SetContextEnabled(false);
						}
      

extern "C" void Tau_cupti_register_metadata(
						uint32_t deviceId,
						GpuMetadata *metadata, 
						int metadata_size
						) {
							metadata_struct m; 
							m.list = metadata;
							m.length = metadata_size;
							deviceInfoMap[deviceId] = m;
						}
extern "C" void Tau_cupti_register_calling_site(
						uint32_t correlationId,
						FunctionInfo *current_function
						) {
							functionInfoMap[correlationId] = current_function;	
						}	

extern "C" void Tau_cupti_enter_memcpy_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						int bytes_copied,
						int memcpy_type
						) {
							//Empty list of gpu attributes
							CuptiGpuEvent gpu_event = CuptiGpuEvent(name, 
								deviceId, streamId, contextId, correlationId, NULL, 0);
							Tau_gpu_enter_memcpy_event(name, &gpu_event, bytes_copied, memcpy_type);
						}

extern "C" void Tau_cupti_exit_memcpy_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						int bytes_copied,
						int memcpy_type
						) {
							//Empty list of gpu attributes
							CuptiGpuEvent gpu_event = CuptiGpuEvent(name, 
								deviceId, streamId, contextId, correlationId, NULL, 0);
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
						int memcpy_type
						) {
							//Empty list of gpu attributes
							CuptiGpuEvent gpu_event = CuptiGpuEvent(name, 
								deviceId, streamId, contextId, correlationId, NULL, 0);
							Tau_gpu_register_memcpy_event(&gpu_event, 
								start, stop, bytes_copied, memcpy_type);
						}

extern "C" void Tau_cupti_register_gpu_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
            bool cdp,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes,
						double start,
						double stop
						) {
							CuptiGpuEvent gpu_event = CuptiGpuEvent(name, 
								deviceId, streamId, contextId, correlationId, gpu_attributes, number_of_attributes);
              if (cdp) {
                printf("setting CDP flag.\n");
							  gpu_event.setCdp();
              }
							Tau_gpu_register_gpu_event(&gpu_event, start, stop);
						}

extern "C" void Tau_cupti_register_gpu_atomic_event(
						const char *name,
						uint32_t deviceId,
						uint32_t streamId,
						uint32_t contextId,
						uint32_t correlationId,
						GpuEventAttributes *gpu_attributes,
						int number_of_attributes
						) {
							CuptiGpuEvent gpu_event = CuptiGpuEvent(name, 
								deviceId, streamId, contextId, correlationId, gpu_attributes, number_of_attributes);
							Tau_gpu_register_gpu_atomic_event(&gpu_event);
						}
