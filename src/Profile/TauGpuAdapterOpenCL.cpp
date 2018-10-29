#include <dlfcn.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstring>
#include <queue>
#include "TauGpuAdapterOpenCL.h"

typedef std::queue<OpenCLGpuEvent*> gpu_event_queue_t;

typedef std::map<cl_command_queue, OpenCLGpuEvent*> queue_event_map_t;


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
    exit(1);	
  }

  double cpu_timestamp;
  struct timeval tp;
  cl_ulong gpu_timestamp;

  cl_event sync_event;
  err = clEnqueueWriteBuffer_noinst(commandQueue, buffer, CL_TRUE, 0, sizeof(void*), data,  0, NULL, &sync_event);
  if (err != CL_SUCCESS) {
    printf("Cannot Enqueue Sync Kernel: %d.\n", err);
    exit(1);
  }

  //get CPU timestamp.
  gettimeofday(&tp, 0);
  cpu_timestamp = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
  //get GPU timestamp for finish.
  err = clGetEventProfilingInfo_noinst(sync_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gpu_timestamp, NULL);
  if (err != CL_SUCCESS) {
    printf("Cannot get end time for Sync event.\n");
    exit(1);	
  }

  //printf("SYNC: CPU= %f GPU= %f.\n", cpu_timestamp, ((double)gpu_timestamp/1e3)); 
  return cpu_timestamp - (((double)gpu_timestamp)/1e3);
}

void * Tau_opencl_get_handle(char const * fnc_name)
{
#ifdef __APPLE__
  static char const * libname = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
  static char const * libname = "libOpenCL.so";
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


OpenCLGpuEvent * Tau_opencl_retrive_gpu(cl_command_queue q)
{
  queue_event_map_t & id_map = IdentityMap();
  queue_event_map_t::iterator it = id_map.find(q);
  if (it != id_map.end())
    return it->second;

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

  //err = clGetDeviceInfo(id, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor, NULL);


  //printf("device id: %d.\n", id);
  //printf("command id: %lld.\n", q);
  //printf("vendor id: %d.\n", vendor);
  double sync_offset = Tau_opencl_sync_clocks(q, context);
  OpenCLGpuEvent *gId = new OpenCLGpuEvent(id, (x_uint64) q, sync_offset);
  id_map[q] = gId;

  return gId;
  //printf("New device id found: %d.\n", id);
}

OpenCLGpuEvent * Tau_opencl_new_gpu_event(cl_command_queue queue, char const * name, int memcpy_type)
{
  Profiler * p = TauInternal_CurrentProfiler(RtsLayer::myThread());
  if (p) {
    OpenCLGpuEvent * gpu_event = Tau_opencl_retrive_gpu(queue)->getCopy();
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
#ifndef PTHREADS
  Tau_gpu_register_gpu_event(evId, start/1e3, stop/1e3);
#endif /* PTHREADS */
}

void Tau_opencl_register_memcpy_event(OpenCLGpuEvent *evId, double start, double stop, int transferSize, int MemcpyType)
{
#ifndef PTHREADS
  Tau_gpu_register_memcpy_event(evId, start/1e3, stop/1e3, transferSize, MemcpyType, MESSAGE_UNKNOWN);
#endif /* PTHREADS */
}
 
void Tau_opencl_enqueue_event(OpenCLGpuEvent * event)
{
  KernelBuffer().push(event);
}

void Tau_opencl_register_sync_event()
{
  gpu_event_queue_t & event_queue = KernelBuffer();

//  printf("\nin Tau_opencl_register_sync_event\n");
//  printf("TAU (opencl): registering sync.\n");
//  printf("TAU (opencl): empty buffer? %d.\n", event_queue.empty());
//  printf("TAU (opencl): size of buffer: %d.\n", event_queue.size());

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
      exit(1);
    }
    if (status != CL_COMPLETE) continue;
//    printf("Complete: %p (%s)\n", kernel_data->event, kernel_data->name);

    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong), &queuedTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get queued time for Kernel event.\n");
      exit(1);	
    }
    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_SUBMIT,
        sizeof(cl_ulong), &submitTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get submit time for Kernel event.\n");
      exit(1);	
    }
    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &startTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get start time for Kernel event.\n");
      exit(1);	
    }

    err = clGetEventProfilingInfo_noinst(kernel_data->event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &endTime, NULL);
    if (err != CL_SUCCESS) {
      printf("Cannot get end time for Kernel event.\n");
      exit(1);	
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
//      printf("TAU (opencl): isMemcpy kind: %d.\n", kernel_data->memcpy_type);
      Tau_opencl_register_memcpy_event(kernel_data, (double)startTime, (double)endTime,
          TAU_GPU_UNKNOWN_TRANSFER_SIZE, kernel_data->memcpy_type);
    } else {
//      printf("TAU (opencl): isKernel.\n");
      Tau_opencl_register_gpu_event(kernel_data, (double)startTime, (double)endTime);
    }
    event_queue.pop();
    clReleaseEvent_noinst(kernel_data->event);
//    printf("Popped buffer.\n");
  }
//  printf("\n\n");
}
