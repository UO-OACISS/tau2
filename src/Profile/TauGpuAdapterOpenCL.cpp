#include <dlfcn.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstring>
#include <queue>
#include "TauGpuAdapterOpenCL.h"

typedef std::queue<OpenCLGpuEvent*> gpu_event_queue_t;

typedef std::map<cl_command_queue, OpenCLGpuEvent*> queue_event_map_t;

extern "C" void metric_set_gpu_timestamp(int tid, double value);

void __attribute__ ((constructor)) Tau_opencl_init()
{
  Tau_gpu_init();
}

void __attribute__ ((destructor)) Tau_opencl_exit()
{
  Tau_destructor_trigger();
}

static gpu_event_queue_t & KernelBuffer()
{
  static gpu_event_queue_t queue;
  return queue;
}


static queue_event_map_t & IdentityMap()
{
  static queue_event_map_t map;
  return map;
}

/*
 * Given a cl code and return a string represenation
 */
static const char* clGetErrorString(int errorCode) {
    switch (errorCode) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69: return "CL_INVALID_PIPE_SIZE";
        case -70: return "CL_INVALID_DEVICE_QUEUE";
        case -71: return "CL_INVALID_SPEC_ID";
        case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
        case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
        case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        default: return "CL_UNKNOWN_ERROR";
    }
}


cl_mem clCreateBuffer_noinst(cl_context a1, cl_mem_flags a2, size_t a3, void * a4, cl_int * a5)
{
  HANDLE(cl_mem, clCreateBuffer, cl_context, cl_mem_flags, size_t, void *, cl_int *);
  return clCreateBuffer_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clGetEventProfilingInfo_noinst(cl_event a1, cl_profiling_info a2, size_t a3, void * a4, size_t * a5)
{
  HANDLE(cl_int, clGetEventProfilingInfo, cl_event, cl_profiling_info, size_t, void *, size_t *);
  return clGetEventProfilingInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clEnqueueWriteBuffer_noinst(cl_command_queue a1, cl_mem a2, cl_bool a3, size_t a4, size_t a5, const void * a6,
                                   cl_uint a7, const cl_event * a8, cl_event * a9)
{
  HANDLE(cl_int, clEnqueueWriteBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint,
         const cl_event *, cl_event *);
  return clEnqueueWriteBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
}

cl_int clGetEventInfo_noinst(cl_event a1, cl_event_info a2, size_t a3, void * a4, size_t * a5)
{
  HANDLE(cl_int, clGetEventInfo, cl_event, cl_event_info, size_t, void *, size_t *);
  return clGetEventInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clReleaseEvent_noinst(cl_event a1)
{
  HANDLE(cl_int, clReleaseEvent, cl_event);
  return clReleaseEvent_h(a1);
}

cl_int clFinish_noinst(cl_command_queue a1)
{
  HANDLE(cl_int, clFinish, cl_command_queue);
  return clFinish_h(a1);
}

static double Tau_opencl_get_gpu_timestamp(cl_command_queue commandQueue, cl_context context) {
  int d = 0;
  void *data = &d;
  cl_mem buffer;
  cl_int err;
  buffer = clCreateBuffer_noinst(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(void*), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Cannot Create Sync Buffer: %d.\n", err);
    if (err == CL_INVALID_CONTEXT) {
      printf("Invalid context.\n");
    }
    abort();
  }

  struct timeval tp;
  cl_ulong gpu_timestamp;

  cl_event sync_event;
  err = clEnqueueWriteBuffer_noinst(commandQueue, buffer, CL_TRUE, 0, sizeof(void*), data,  0, NULL, &sync_event);
  if (err != CL_SUCCESS) {
    printf("Cannot Enqueue Sync Kernel: %d.\n", err);
    abort();
  }

  clFinish_noinst(commandQueue);
  //get GPU timestamp for finish.
  err = clGetEventProfilingInfo_noinst(sync_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gpu_timestamp, NULL);
  if (err != CL_SUCCESS) {
    printf("Cannot get end time for Sync event: %s\n", clGetErrorString(err));
    abort();
  }

  return gpu_timestamp;
}

static double Tau_opencl_get_cpu_timestamp() {
  double cpu_timestamp;
  struct timeval tp;
  gettimeofday(&tp, 0);
  cpu_timestamp = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
  return cpu_timestamp;
}

static double Tau_opencl_sync_clocks(cl_command_queue commandQueue, cl_context context)
{
  int d = 0;
  void *data = &d;
  cl_mem buffer;
  cl_int err;
  buffer = clCreateBuffer_noinst(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(void*), NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Cannot Create Sync Buffer: %d.\n", err);
    if (err == CL_INVALID_CONTEXT) {
      printf("Invalid context.\n");
    }
    abort();
  }

  double cpu_timestamp;
  struct timeval tp;
  cl_ulong gpu_timestamp;

  cl_event sync_event;
  err = clEnqueueWriteBuffer_noinst(commandQueue, buffer, CL_TRUE, 0, sizeof(void*), data,  0, NULL, &sync_event);
  if (err != CL_SUCCESS) {
    printf("Cannot Enqueue Sync Kernel: %d.\n", err);
    abort();
  }

  //get CPU timestamp.
  gettimeofday(&tp, 0);
  cpu_timestamp = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
  //get GPU timestamp for finish.
  clFinish_noinst(commandQueue);
  err = clGetEventProfilingInfo_noinst(sync_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gpu_timestamp, NULL);
  if (err != CL_SUCCESS) {
    printf("Cannot get end time for Sync event: %s\n", clGetErrorString(err));
    abort();
  }

  //printf("SYNC: CPU= %f GPU= %f.\n", cpu_timestamp, ((double)gpu_timestamp/1e3));
  return cpu_timestamp - (((double)gpu_timestamp)/1e3);
}

void * Tau_opencl_get_handle(char const * fnc_name)
{
#ifdef __APPLE__
  static char const * libname = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
  static char const * libname = TAU_OPENCL_LIBRARY;
#endif /* __APPLE__ */

  static void * handle = NULL;
  if (!handle) {
    handle = (void *)dlopen(libname, RTLD_NOW);
  }
  if (!handle) {
    perror("Error opening library in dlopen call");
    return NULL;
  }

  void * fnc_sym = dlsym(handle, fnc_name);
  if (!fnc_sym) {
    perror("Error obtaining symbol info from dlopen'ed lib");
    return NULL;
  }
  return fnc_sym;
}


OpenCLGpuEvent * Tau_opencl_retrieve_gpu(cl_command_queue q)
{
  queue_event_map_t & id_map = IdentityMap();
  queue_event_map_t::iterator it = id_map.find(q);
  if (it != id_map.end()) {
#ifdef TAU_DEBUG_OPENCL
    OpenCLGpuEvent * evt = it->second;
    fprintf(stderr, "Found existing GPU event: id=%lu, name=%s, taskid=%d\n", evt->id, evt->name, evt->getTaskId());
#endif
    return it->second;
  }

  cl_device_id id;
  cl_context context;
  cl_int err;
  cl_uint vendor;
  err = clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(cl_device_id), &id, NULL);
  if (err != CL_SUCCESS)
  {
    printf("error in clGetCommandQueueInfo DEVICE.\n");
    if (err == CL_INVALID_COMMAND_QUEUE)
      printf("invalid command queue.\n");
  }
  err = clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
  if (err != CL_SUCCESS)
  {	printf("error in clGetCommandQueueInfo CONTEXT.\n"); }

  char deviceName[256];
  char deviceVendor[256];

  err = clGetDeviceInfo(id, CL_DEVICE_NAME, 256, deviceName, NULL);
  if (err != CL_SUCCESS)
  {	printf("error in clGetDeviceInfo CL_DEVICE_NAME.\n"); }

  err = clGetDeviceInfo(id, CL_DEVICE_VENDOR, 256, deviceVendor, NULL);
  if (err != CL_SUCCESS)
  {	printf("error in clGetDeviceInfo CL_DEVICE_VENDOR.\n"); }

  printf("device name: %s\n", deviceName);
  printf("vendor name: %s\n", deviceVendor);
  printf("command id: %lld\n", q);
  double sync_offset = Tau_opencl_sync_clocks(q, context);
#if defined(PTHREADS) || defined(TAU_OPENMP)
  // Create a virtual thread for this command queue
  int taskid = TAU_CREATE_TASK(taskid);
  double cpu_timestamp = Tau_opencl_get_cpu_timestamp();
  metric_set_gpu_timestamp(taskid, cpu_timestamp);
  Tau_create_top_level_timer_if_necessary_task(taskid);
  OpenCLGpuEvent *gId = new OpenCLGpuEvent(id, (x_uint64) q, sync_offset, taskid);
#ifdef TAU_DEBUG_OPENCL
  fprintf(stderr, "Created OpenCLGpuEvent with taskid %d\n", taskid);
#endif // TAU_DEBUG_OPENCL
#else
  OpenCLGpuEvent *gId = new OpenCLGpuEvent(id, (x_uint64) q, sync_offset);
#endif // defined(PTHREADS) || defined(TAU_OPENMP)


  id_map[q] = gId;

  return gId;
  //printf("New device id found: %d.\n", id);
}

OpenCLGpuEvent * Tau_opencl_new_gpu_event(cl_command_queue queue, char const * name, int memcpy_type)
{
  Profiler * p = TauInternal_CurrentProfiler(RtsLayer::myThread());
  if (p) {
    OpenCLGpuEvent * gpu_event = Tau_opencl_retrieve_gpu(queue)->getCopy();
    gpu_event->name = name;
    gpu_event->event = NULL;
    gpu_event->callingSite = p->CallPathFunction;
    gpu_event->memcpy_type = memcpy_type;
    return gpu_event;
  }
  return NULL;
}

void Tau_opencl_enter_memcpy_event(const char *name, OpenCLGpuEvent *id, int size, int MemcpyType)
{
  Tau_gpu_enter_memcpy_event(name, id->getCopy(), size, MemcpyType);
}

void Tau_opencl_exit_memcpy_event(const char *name, OpenCLGpuEvent *id, int MemcpyType)
{
  Tau_gpu_exit_memcpy_event(name, id->getCopy(), MemcpyType);
}

void Tau_opencl_register_gpu_event(OpenCLGpuEvent *evId, double start, double stop)
{
  Tau_gpu_register_gpu_event(evId, start/1e3, stop/1e3);
}

void Tau_opencl_register_memcpy_event(OpenCLGpuEvent *evId, double start, double stop, int transferSize, int MemcpyType)
{
  Tau_gpu_register_memcpy_event(evId, start/1e3, stop/1e3, transferSize, MemcpyType, MESSAGE_UNKNOWN);
}

void Tau_opencl_enqueue_event(OpenCLGpuEvent * event)
{
  KernelBuffer().push(event);
}

void Tau_opencl_register_sync_event()
{
  gpu_event_queue_t & event_queue = KernelBuffer();

#ifdef TAU_DEBUG_OPENCL
  printf("\nin Tau_opencl_register_sync_event\n");
  printf("TAU (opencl): registering sync.\n");
  printf("TAU (opencl): empty buffer? %d.\n", event_queue.empty());
  printf("TAU (opencl): size of buffer: %d.\n", event_queue.size());
#endif // TAU_DEBUG_OPENCL

  while(!event_queue.empty())
  {
    cl_int err;
    cl_ulong startTime, endTime, queuedTime, submitTime;
    OpenCLGpuEvent* kernel_data = event_queue.front();

//    printf("Checking event: %p (%s)\n", kernel_data->event, kernel_data->name);
//    printf("Size of buffer: %d.\n", event_queue.size());
//    printf("Front of buffer is: %s\n", kernel_data->name);

    cl_int status;
    err = clGetEventInfo_noinst(kernel_data->event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
    if (err != CL_SUCCESS) {
      printf("Fatal error: calling clGetEventInfo, exiting.\n");
      abort();
    }
    if (status != CL_COMPLETE) continue;
//    printf("Complete: %p (%s)\n", kernel_data->event, kernel_data->name);

    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong), &queuedTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get queued time for Kernel event.\n");
      abort();
    }
    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_SUBMIT,
        sizeof(cl_ulong), &submitTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get submit time for Kernel event.\n");
      abort();
    }
    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &startTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get start time for Kernel event.\n");
      abort();
    }

    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &endTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get end time for Kernel event.\n");
      abort();
    }

    //Add context events to gpu event.
    GpuEventAttributes *map;
    static TauContextUserEvent *qt;
    static TauContextUserEvent *st;
    Tau_get_context_userevent((void**) &qt, "Time in Queue (us)");
    Tau_get_context_userevent((void**) &st, "Time in Submitted (us)");
    map = (GpuEventAttributes *) malloc(sizeof(GpuEventAttributes) * 2);
    map[0].userEvent = qt;
    map[0].data = (startTime - queuedTime)/1e3;
    map[1].userEvent = st;
    map[1].data = (startTime - submitTime)/1e3;
    kernel_data->gpu_event_attr = map;

    if (kernel_data->isMemcpy()) {
#ifdef TAU_DEBUG_OPENCL
      printf("TAU (opencl): isMemcpy kind: %d.\n", kernel_data->memcpy_type);
#endif // TAU_DEBUG_OPENCL
      Tau_opencl_register_memcpy_event(kernel_data, (double)startTime, (double)endTime,
          TAU_GPU_UNKNOWN_TRANSFER_SIZE, kernel_data->memcpy_type);
    } else {
#ifdef TAU_DEBUG_OPENCL
      printf("TAU (opencl): isKernel.\n");
#endif
      Tau_opencl_register_gpu_event(kernel_data, (double)startTime, (double)endTime);
    }
    event_queue.pop();
    clReleaseEvent_noinst(kernel_data->event);
//    printf("Popped buffer.\n");
  }
//  printf("\n\n");
}
