#ifndef TAU_OPENACC_H
#define TAU_OPENACC_H

#include <Profile/TauGpu.h>
//#include <Profile/TauGpuAdapterOpenACC.h>
#include <cuda.h>
#include <cupti.h>

extern "C" void Tau_openacc_register_gpu_event(
  const char* name,                                                                                                                      uint32_t device,                                                                                                                       uint32_t stream,                                                                                                                       uint32_t context,
  uint32_t task,                                                                                                                         uint32_t corr_id,                                                                                                                      GpuEventAttributes* event_attrs,
  int num_event_attrs,
  double start,
  double stop);

#endif
