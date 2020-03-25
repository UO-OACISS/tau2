#ifndef TAU_CUDA_THREAD_H
#define TAU_CUDA_THREAD_H

struct CudaThread
{
  unsigned int sys_tid;     // pthread
  int parent_tid;
  int tau_vtid;    // virtual tid, write profiles
  // callback info
  const char* function_name;
  unsigned int context_id;
  unsigned int correlation_id;
  unsigned int device_id;
};

typedef struct Cuda_thread_device {
  int threadid;
  int deviceid;
  int tau_vtid;
} cuda_thread_device_t;

#endif //TAU_CUDA_THREAD_H
