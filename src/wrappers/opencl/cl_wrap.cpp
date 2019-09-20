#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <unistd.h>

#include <Profile/TauGpuAdapterOpenCL.h>

#ifdef TAU_BFD
#define HAVE_DECL_BASENAME 1
#  if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#    include <demangle.h>
#  endif /* HAVE_GNU_DEMANGLE */
// Add these definitions because the Binutils comedians think all the world uses autotools
#ifndef PACKAGE
#define PACKAGE TAU
#endif
#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION 2.25
#endif
#  include <bfd.h>
#endif /* TAU_BFD */

#define TAU_INTERNAL_DEMANGLE_NAME(name, dem_name)  dem_name = cplus_demangle(name, DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES); \
        if (dem_name == NULL) { \
          dem_name = name; \
        } \


void MemoryCopyEventHtoD(size_t bytes)
{
  static TauContextUserEvent * event = NULL;
  if (!event) {
    Tau_get_context_userevent((void **)&event, "Bytes copied from Host to Device");
  }
  TAU_CONTEXT_EVENT(event, bytes);
}

void MemoryCopyEventDtoH(size_t bytes)
{
  static TauContextUserEvent * event = NULL;
  if (!event) {
    Tau_get_context_userevent((void **)&event, "Bytes copied from Device to Host");
  }
  TAU_CONTEXT_EVENT(event, bytes);
}

void MemoryCopyEventDtoD(size_t bytes)
{
  static TauContextUserEvent * event = NULL;
  if (!event) {
    Tau_get_context_userevent((void **)&event, "Bytes copied (Other)");
  }
  TAU_CONTEXT_EVENT(event, bytes);
}

cl_int clGetPlatformIDs(cl_uint a1, cl_platform_id * a2, cl_uint * a3) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetPlatformIDs, cl_uint, cl_platform_id *, cl_uint *);
  return clGetPlatformIDs_h(a1,  a2,  a3);
}

cl_int clGetPlatformInfo(cl_platform_id a1, cl_platform_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetPlatformInfo, cl_platform_id, cl_platform_info, size_t, void *, size_t *);
  return clGetPlatformInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clGetDeviceIDs(cl_platform_id a1, cl_device_type a2, cl_uint a3, cl_device_id * a4, cl_uint * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetDeviceIDs, cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  return clGetDeviceIDs_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clGetDeviceInfo(cl_device_id a1, cl_device_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetDeviceInfo, cl_device_id, cl_device_info, size_t, void *, size_t *);
  return clGetDeviceInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_context clCreateContext(const cl_context_properties * a1, cl_uint a2, const cl_device_id * a3, 
                           void (*a4)(const char *, const void *, size_t, void *), void * a5, cl_int * a6) 
{
  HANDLE_AND_AUTOTIMER(cl_context, clCreateContext, const cl_context_properties *, cl_uint, const cl_device_id *, 
                   void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
  return clCreateContext_h(a1,  a2,  a3,  a4,  a5,  a6);
}

cl_context clCreateContextFromType(const cl_context_properties * a1, cl_device_type a2, 
                                   void (*a3)(const char *, const void *, size_t, void *), void * a4, cl_int * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_context, clCreateContextFromType, const cl_context_properties *, cl_device_type, 
                   void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
  return clCreateContextFromType_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clRetainContext(cl_context a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainContext, cl_context);
  return clRetainContext_h(a1);
}

cl_int clReleaseContext(cl_context a1) 
{
  HANDLE_AND_TIMER(cl_int, clReleaseContext, cl_context);
  TAU_PROFILE_START(t);
  cl_int retval = clReleaseContext_h(a1);
  TAU_PROFILE_STOP(t);
  return retval;
}

cl_int clGetContextInfo(cl_context a1, cl_context_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetContextInfo, cl_context, cl_context_info, size_t, void *, size_t *);
  return clGetContextInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_command_queue clCreateCommandQueue(cl_context a1, cl_device_id a2, cl_command_queue_properties a3, cl_int * a4) 
{
  HANDLE_AND_AUTOTIMER(cl_command_queue, clCreateCommandQueue, cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
  // Make sure this command queue allows profiling
  a3 |= CL_QUEUE_PROFILING_ENABLE;
  return clCreateCommandQueue_h(a1,  a2,  a3,  a4);
}

#ifdef CL_VERSION_2_0
cl_command_queue clCreateCommandQueueWithProperties(cl_context a1, cl_device_id a2, const cl_queue_properties * a3, cl_int * a4) {
  // OpenCL 2.0 replaces clCreateCommandQueue with clCreateCommandQueueWithProperties.
  // The difference is that while clCreateCommandQueue allows a bitmask for cl_command_queue_properties,
  // clCreateCommandQueueWithProperties has a list of cl_queue_properties (which is actually an unsigned long).
  // The a3 arguemnt can be:
  //   - NULL, in which case the defaults are used; or,
  //   - a list specifying the properties.
  // If a list is used, the list is:
  //   - Any number of [property ID, property value]
  //   - followed by 0
  // Where the old cl_command_queue_properties bitmask is provided by CL_QUEUE_PROPERTIES and then the bitmask.
  // So there are three cases that have to be handled:
  //   - If NULL was provided, instead provide {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0}.
  //   - If a list was provided that has CL_QUEUE_PROPERTIES, add CL_QUEUE_PROFILING_ENABLE to the bitmask
  //     (the next entry) by logical OR that entry with CL_QUEUE_PROFILING_ENABLE.
  //   - If a list was provided, but it doesn't have CL_QUEUE_PROPERTIES, extend the list to add
  //     {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE} before the 0 terminating the list.
  HANDLE_AND_AUTOTIMER(cl_command_queue, clCreateCommandQueueWithProperties, cl_context, cl_device_id, const cl_command_queue_properties *, cl_int *);
  if(a3 == NULL) {
    // If no properties were provided, create a new list that specifies CL_QUEUE_PROFILING_ENABLE.
#ifdef TAU_DEBUG_OPENCL
    fprintf(stderr, "clCreateCommandQueueWithProperties with NULL list\n");
#endif
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    return clCreateCommandQueueWithProperties_h(a1, a2, props, a4);
  } else {
#ifdef TAU_DEBUG_OPENCL
    fprintf(stderr, "clCreateCommandQueueWithProperties with non-NULL list\n");
#endif
    // If a property list was specified, determine if it already has CL_QUEUE_PROPERTIES in it.
    size_t size;
    ssize_t prop_bitmask_index = -1;
    for(size = 0; a3[size] != 0; ++size) {
      if(a3[size] == CL_QUEUE_PROPERTIES) {
        prop_bitmask_index = size+1;
      }
    }
    // If so, change the corresponding value to include CL_QUEUE_PROFILING_ENABLE.
    if(prop_bitmask_index != -1) {
#ifdef TAU_DEBUG_OPENCL
    fprintf(stderr, "clCreateCommandQueueWithProperties list contained CL_QUEUE_PROFILING_EVENT\n");
#endif
      cl_queue_properties props[size+1];
      for(size_t i = 0; i < size; ++i) {
        if(i == prop_bitmask_index) {
          props[i] = a3[i] | CL_QUEUE_PROFILING_ENABLE;
        } else {
          props[i] = a3[i];
        }
      }
      props[size] = 0;
      return clCreateCommandQueueWithProperties_h(a1, a2, props, a4);
    } else {
#ifdef TAU_DEBUG_OPENCL
    fprintf(stderr, "clCreateCommandQueueWithProperties list did NOT contain CL_QUEUE_PROFILING_EVENT\n");
#endif
      // If not, extend the list and add CL_QUEUE_PROPERTIES CL_QUEUE_PROFILING_ENABLE to the end.
      cl_queue_properties props[size+3];  
      for(size_t i = 0; i < size; ++i) {
          props[i] = a3[i];
      }
      props[size] = CL_QUEUE_PROPERTIES;
      props[size+1] = CL_QUEUE_PROFILING_ENABLE;
      props[size+2] = 0;
      return clCreateCommandQueueWithProperties_h(a1, a2, props, a4);
    }
  }
}
#endif

cl_int clRetainCommandQueue(cl_command_queue a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainCommandQueue, cl_command_queue);
  return clRetainCommandQueue_h(a1);
}

cl_int clReleaseCommandQueue(cl_command_queue a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clReleaseCommandQueue, cl_command_queue);
  return clReleaseCommandQueue_h(a1);
}

cl_int clGetCommandQueueInfo(cl_command_queue a1, cl_command_queue_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetCommandQueueInfo, cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
  return clGetCommandQueueInfo_h( a1,  a2,  a3,  a4,  a5);
}

cl_int clSetCommandQueueProperty(cl_command_queue a1, cl_command_queue_properties a2, cl_bool a3, cl_command_queue_properties * a4) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clSetCommandQueueProperty, cl_command_queue, cl_command_queue_properties, cl_bool, cl_command_queue_properties *);
  return clSetCommandQueueProperty_h(a1,  a2,  a3,  a4);
}

cl_mem clCreateBuffer(cl_context a1, cl_mem_flags a2, size_t a3, void * a4, cl_int * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_mem, clCreateBuffer, cl_context, cl_mem_flags, size_t, void *, cl_int *);
  return clCreateBuffer_h(a1,  a2,  a3,  a4,  a5);
}

cl_mem clCreateImage2D(cl_context a1, cl_mem_flags a2, const cl_image_format * a3, size_t a4, size_t a5, size_t a6, void * a7, cl_int * a8) 
{
  HANDLE_AND_AUTOTIMER(cl_mem, clCreateImage2D, cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *);
  return clCreateImage2D_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
}

cl_mem clCreateImage3D(cl_context a1, cl_mem_flags a2, const cl_image_format * a3, size_t a4, size_t a5, size_t a6, size_t a7, size_t a8, void * a9, cl_int * a10) 
{
  HANDLE_AND_AUTOTIMER(cl_mem, clCreateImage3D, cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *);
  return clCreateImage3D_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);
}

cl_int clRetainMemObject(cl_mem a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainMemObject, cl_mem);
  return clRetainMemObject_h(a1);
}

cl_int clGetSupportedImageFormats(cl_context a1, cl_mem_flags a2, cl_mem_object_type a3, cl_uint a4, cl_image_format * a5, cl_uint * a6) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetSupportedImageFormats, cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *);
  return clGetSupportedImageFormats_h(a1,  a2,  a3,  a4,  a5,  a6);
}

cl_int clGetMemObjectInfo(cl_mem a1, cl_mem_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetMemObjectInfo, cl_mem, cl_mem_info, size_t, void *, size_t *);
  return clGetMemObjectInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clGetImageInfo(cl_mem a1, cl_image_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetImageInfo, cl_mem, cl_image_info, size_t, void *, size_t *);
  return clGetImageInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_sampler clCreateSampler(cl_context a1, cl_bool a2, cl_addressing_mode a3, cl_filter_mode a4, cl_int * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_sampler, clCreateSampler, cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *);
  return clCreateSampler_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clRetainSampler(cl_sampler a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainSampler, cl_sampler);
  return clRetainSampler_h( a1);
}

cl_int clReleaseSampler(cl_sampler a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clReleaseSampler, cl_sampler);
  return clReleaseSampler_h( a1);
}

cl_int clGetSamplerInfo(cl_sampler a1, cl_sampler_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetSamplerInfo, cl_sampler, cl_sampler_info, size_t, void *, size_t *);
  return clGetSamplerInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_program clCreateProgramWithSource(cl_context a1, cl_uint a2, const char ** a3, const size_t * a4, cl_int * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_program, clCreateProgramWithSource, cl_context, cl_uint, const char **, const size_t *, cl_int *);
  return clCreateProgramWithSource_h( a1,  a2,  a3,  a4,  a5);
}

cl_program clCreateProgramWithBinary(cl_context a1, cl_uint a2, const cl_device_id * a3, const size_t * a4, const unsigned char ** a5, cl_int * a6, cl_int * a7) 
{
  HANDLE_AND_AUTOTIMER(cl_program, clCreateProgramWithBinary, cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
  return clCreateProgramWithBinary_h(a1,  a2,  a3,  a4,  a5,  a6,  a7);
}

cl_int clRetainProgram(cl_program a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainProgram, cl_program);
  return clRetainProgram_h(a1);
}

cl_int clReleaseProgram(cl_program a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clReleaseProgram, cl_program);
  return clReleaseProgram_h(a1);
}

cl_int clBuildProgram(cl_program a1, cl_uint a2, const cl_device_id * a3, const char * a4, void (*a5)(cl_program, void *), void * a6) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clBuildProgram, cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *);
  return clBuildProgram_h(a1,  a2,  a3,  a4,  a5,  a6);
}

cl_int clUnloadCompiler() 
{
  HANDLE_AND_AUTOTIMER(cl_int, clUnloadCompiler);
  return clUnloadCompiler_h();
}

cl_int clGetProgramInfo(cl_program a1, cl_program_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetProgramInfo, cl_program, cl_program_info, size_t, void *, size_t *);
  return clGetProgramInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clGetProgramBuildInfo(cl_program a1, cl_device_id a2, cl_program_build_info a3, size_t a4, void * a5, size_t * a6) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetProgramBuildInfo, cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
  return clGetProgramBuildInfo_h(a1,  a2,  a3,  a4,  a5,  a6);
}

cl_kernel clCreateKernel(cl_program a1, const char * a2, cl_int * a3) 
{
  HANDLE_AND_AUTOTIMER(cl_kernel, clCreateKernel, cl_program, const char *, cl_int *);
  return clCreateKernel_h( a1,  a2,  a3);
}

cl_int clCreateKernelsInProgram(cl_program a1, cl_uint a2, cl_kernel * a3, cl_uint * a4) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clCreateKernelsInProgram, cl_program, cl_uint, cl_kernel *, cl_uint *);
  return clCreateKernelsInProgram_h( a1,  a2,  a3,  a4);
}

cl_int clRetainKernel(cl_kernel a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainKernel, cl_kernel);
  return clRetainKernel_h(a1);
}

cl_int clReleaseKernel(cl_kernel a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clReleaseKernel, cl_kernel);
  return clReleaseKernel_h(a1);
}

cl_int clSetKernelArg(cl_kernel a1, cl_uint a2, size_t a3, const void * a4) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clSetKernelArg, cl_kernel, cl_uint, size_t, const void *);
  return clSetKernelArg_h( a1,  a2,  a3,  a4);
}

cl_int clGetKernelInfo(cl_kernel a1, cl_kernel_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetKernelInfo, cl_kernel, cl_kernel_info, size_t, void *, size_t *);
  return clGetKernelInfo_h( a1,  a2,  a3,  a4,  a5);
}

cl_int clGetKernelWorkGroupInfo(cl_kernel a1, cl_device_id a2, cl_kernel_work_group_info a3, size_t a4, void * a5, size_t * a6) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetKernelWorkGroupInfo, cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);
  return clGetKernelWorkGroupInfo_h(a1,  a2,  a3,  a4,  a5,  a6);
}

cl_int clWaitForEvents(cl_uint a1, const cl_event * a2) 
{
  HANDLE_AND_TIMER(cl_int, clWaitForEvents, cl_uint, const cl_event *);
  TAU_PROFILE_START(t);
  cl_int retval = clWaitForEvents_h(a1,  a2);
  TAU_PROFILE_STOP(t);
  Tau_opencl_register_sync_event();
  return retval;
}

cl_int clGetEventInfo(cl_event a1, cl_event_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetEventInfo, cl_event, cl_event_info, size_t, void *, size_t *);
  return clGetEventInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clRetainEvent(cl_event a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clRetainEvent, cl_event);
  return clRetainEvent_h(a1);
}

cl_int clRetainEvent_noinst(cl_event a1)
{
  HANDLE(cl_int, clRetainEvent, cl_event);
  return clRetainEvent_h(a1);
}

cl_int clReleaseEvent(cl_event a1) 
{
  HANDLE_AND_TIMER(cl_int, clReleaseEvent, cl_event);
  TAU_PROFILE_START(t);
  cl_int retval = clReleaseEvent_h(a1);
  TAU_PROFILE_STOP(t);
  return retval;
}

cl_int clGetEventProfilingInfo(cl_event a1, cl_profiling_info a2, size_t a3, void * a4, size_t * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clGetEventProfilingInfo, cl_event, cl_profiling_info, size_t, void *, size_t *);
  return clGetEventProfilingInfo_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clFlush(cl_command_queue a1) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clFlush, cl_command_queue);
  return clFlush_h(a1);
}

cl_int clFinish(cl_command_queue a1) 
{
  HANDLE_AND_TIMER(cl_int, clFinish, cl_command_queue);
  TAU_PROFILE_START(t);
  cl_int retval = clFinish_h(a1);
  TAU_PROFILE_STOP(t);
  Tau_opencl_register_sync_event();
  return retval;
}

cl_int clEnqueueReadBuffer(cl_command_queue a1, cl_mem a2, cl_bool a3, size_t a4, size_t a5, void * a6, 
                           cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_TIMER(cl_int, clEnqueueReadBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t, 
                   void *, cl_uint, const cl_event *, cl_event *);
  static char const * timer_name = TIMER_NAME(cl_int, clEnqueueReadBuffer, cl_command_queue, cl_mem, 
                                              cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);

  OpenCLGpuEvent * gId = Tau_opencl_new_gpu_event(a1, "ReadBuffer", MemcpyDtoH);
  if (!gId) {
    return clEnqueueReadBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  }

  if (!a9) {
    a9 = &gId->event;
  }

  MemoryCopyEventDtoH(a5);

  Tau_opencl_enter_memcpy_event(timer_name, gId, a5, MemcpyDtoH);
  cl_int retval = clEnqueueReadBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  Tau_opencl_exit_memcpy_event(timer_name, gId, MemcpyDtoH);

  if (!gId->event) {
    gId->event = *a9;
    clRetainEvent_noinst(gId->event);
  }
  Tau_opencl_enqueue_event(gId);

  Tau_opencl_register_sync_event();
  return retval;
}

cl_int clEnqueueWriteBuffer(cl_command_queue a1, cl_mem a2, cl_bool a3, size_t a4, size_t a5, const void * a6, 
                            cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_TIMER(cl_int, clEnqueueWriteBuffer, cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                   const void *, cl_uint, const cl_event *, cl_event *);
  static char const * timer_name = TIMER_NAME(cl_int, clEnqueueWriteBuffer, cl_command_queue, cl_mem, cl_bool,
                                              size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);

  OpenCLGpuEvent * gId = Tau_opencl_new_gpu_event(a1, "WriteBuffer", MemcpyHtoD);
  if (!gId) {
    return clEnqueueWriteBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  }

  if (!a9) {
    a9 = &gId->event;
  }

  MemoryCopyEventHtoD(a5);

  Tau_opencl_enter_memcpy_event(timer_name, gId, a5, MemcpyHtoD); 
  cl_int retval = clEnqueueWriteBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  Tau_opencl_exit_memcpy_event(timer_name, gId, MemcpyHtoD); 

  if (!gId->event) {
    gId->event = *a9;
    clRetainEvent_noinst(gId->event);
  }
  Tau_opencl_enqueue_event(gId);

  Tau_opencl_register_sync_event();
  return retval;
}

cl_int clEnqueueCopyBuffer(cl_command_queue a1, cl_mem a2, cl_mem a3, size_t a4, size_t a5, size_t a6, 
                           cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_TIMER(cl_int, clEnqueueCopyBuffer, cl_command_queue, cl_mem, cl_mem, 
                   size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
  static char const * timer_name = TIMER_NAME(cl_int, clEnqueueCopyBuffer, cl_command_queue, cl_mem, cl_mem, 
                                              size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);

  OpenCLGpuEvent * gId = Tau_opencl_new_gpu_event(a1, "CopyBuffer", MemcpyDtoD);
  if (!gId) {
    return clEnqueueCopyBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  }

  if (!a9) {
    a9 = &gId->event;
  }

  MemoryCopyEventDtoD(a5);

  Tau_opencl_enter_memcpy_event(timer_name, gId, a5, MemcpyDtoD); 
  cl_int retval = clEnqueueCopyBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  Tau_opencl_exit_memcpy_event(timer_name, gId, MemcpyDtoD); 

  if (!gId->event) {
    gId->event = *a9;
    clRetainEvent_noinst(gId->event);
  }
  Tau_opencl_enqueue_event(gId);

  Tau_opencl_register_sync_event();
  return retval;
}

cl_int clEnqueueReadImage(cl_command_queue a1, cl_mem a2, cl_bool a3, const size_t * a4, const size_t * a5, size_t a6, 
                          size_t a7, void * a8, cl_uint a9, const cl_event * a10, cl_event * a11)
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueReadImage, cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, 
                     size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
  return clEnqueueReadImage_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10,  a11);
}

cl_int clEnqueueWriteImage(cl_command_queue a1, cl_mem a2, cl_bool a3, const size_t * a4, const size_t * a5, size_t a6, 
                           size_t a7, const void * a8, cl_uint a9, const cl_event * a10, cl_event * a11) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueWriteImage, cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, 
                     size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
  return clEnqueueWriteImage_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10,  a11);
}

cl_int clEnqueueCopyImage(cl_command_queue a1, cl_mem a2, cl_mem a3, const size_t * a4, const size_t * a5,
                          const size_t * a6, cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueCopyImage, cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, 
                     const size_t *, cl_uint, const cl_event *, cl_event *);
  return clEnqueueCopyImage_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
}

cl_int clEnqueueCopyImageToBuffer(cl_command_queue a1, cl_mem a2, cl_mem a3, const size_t * a4, const size_t * a5, 
                                  size_t a6, cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueCopyImageToBuffer, cl_command_queue, cl_mem, cl_mem, const size_t *, 
                     const size_t *, size_t, cl_uint, const cl_event *, cl_event *);
  return clEnqueueCopyImageToBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
}

cl_int clEnqueueCopyBufferToImage(cl_command_queue a1, cl_mem a2, cl_mem a3, size_t a4, const size_t * a5, 
                                  const size_t * a6, cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueCopyBufferToImage, cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, 
                     const size_t *, cl_uint, const cl_event *, cl_event *);
  return clEnqueueCopyBufferToImage_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
}

void * clEnqueueMapBuffer(cl_command_queue a1, cl_mem a2, cl_bool a3, cl_map_flags a4, size_t a5, size_t a6, 
                          cl_uint a7, const cl_event * a8, cl_event * a9, cl_int * a10) 
{
  HANDLE_AND_TIMER(void *, clEnqueueMapBuffer, cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, 
                   size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
  static char const * timer_name = TIMER_NAME(void *, clEnqueueMapBuffer, cl_command_queue, cl_mem, cl_bool, 
                                              cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *);

  OpenCLGpuEvent * gId = Tau_opencl_new_gpu_event(a1, "MapBuffer", MemcpyHtoD);
  if (!gId) {
    return clEnqueueMapBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9, a10);
  }

  if (!a9) {
    a9 = &gId->event;
  }

  MemoryCopyEventHtoD(a6);

  Tau_opencl_enter_memcpy_event(timer_name, gId, a6, MemcpyHtoD); 
  void * retval = clEnqueueMapBuffer_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9, a10);
  Tau_opencl_exit_memcpy_event(timer_name, gId, MemcpyHtoD); 

  if (!gId->event) {
    gId->event = *a9;
    clRetainEvent_noinst(gId->event);
  }
  Tau_opencl_enqueue_event(gId);

  Tau_opencl_register_sync_event();
  return retval;
}

void * clEnqueueMapImage(cl_command_queue a1, cl_mem a2, cl_bool a3, cl_map_flags a4, const size_t * a5, 
                         const size_t * a6, size_t * a7, size_t * a8, cl_uint a9, const cl_event * a10, 
                         cl_event * a11, cl_int * a12) 
{
  HANDLE_AND_AUTOTIMER(void *, clEnqueueMapImage, cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, 
                     const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *);
  return clEnqueueMapImage_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10,  a11,  a12);
}

#ifdef TAU_ENABLE_OPENCL_ENQUEUE_UNMAP_MEM_OBJECT
cl_int clEnqueueUnmapMemObject(cl_command_queue a1, cl_mem a2, void * a3, cl_uint a4, const cl_event * a5, cl_event * a6)
{
  HANDLE_AND_TIMER(cl_int, clEnqueueUnmapMemObject, cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
  static char const * timer_name = TIMER_NAME(cl_int, clEnqueueUnmapMemObject, cl_command_queue, cl_mem, void *, cl_uint, 
                                              const cl_event *, cl_event *);

  OpenCLGpuEvent * gId = Tau_opencl_new_gpu_event(a1, "UnmapBuffer", MemcpyDtoH);
  if (!gId) {
    return clEnqueueUnmapMemObject(a1,  a2,  a3,  a4,  a5,  a6);
  }

  if (!a6) {
    a6 = &gId->event;
  }

  MemoryCopyEventDtoH(0);

  Tau_opencl_enter_memcpy_event(timer_name, gId, 0, MemcpyDtoH); 
  cl_int retval = clEnqueueUnmapMemObject(a1,  a2,  a3,  a4,  a5,  a6);
  Tau_opencl_exit_memcpy_event(timer_name, gId, MemcpyDtoH); 

  if (!gId->event) {
    gId->event = *a6;
    clRetainEvent_noinst(gId->event);
  }
  Tau_opencl_enqueue_event(gId);

  Tau_opencl_register_sync_event();
  return retval;
}
#endif /* TAU_ENABLE_OPENCL_ENQUEUE_UNMAP_MEM_OBJECT */

cl_int clEnqueueNDRangeKernel(cl_command_queue a1, cl_kernel a2, cl_uint a3, const size_t * a4, const size_t * a5, 
                              const size_t * a6, cl_uint a7, const cl_event * a8, cl_event * a9) 
{
  HANDLE_AND_TIMER(cl_int, clEnqueueNDRangeKernel, cl_command_queue, cl_kernel, cl_uint, const size_t *, 
                   const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);

  char buf[4096];
  size_t len;
  char const * name;
  cl_int err = clGetKernelInfo(a2, CL_KERNEL_FUNCTION_NAME, sizeof(buf), buf, &len);
  if (err != CL_SUCCESS) {
    name = "NAME ERROR";
  } else {
    name = new char[len+1];
    strncpy(const_cast<char*>(name), buf, len+1);
  }
  const char *dem_name = 0;
#if defined(TAU_BFD) && defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
  TAU_INTERNAL_DEMANGLE_NAME(name, dem_name);
  const char * typeinfo_prefix = "typeinfo name for ";
  if(strncmp(dem_name, typeinfo_prefix, strlen(typeinfo_prefix)) == 0) {
    dem_name = dem_name + strlen(typeinfo_prefix);
  }
#else
  dem_name = name; 
#endif /* HAVE_GNU_DEMANGLE */

  OpenCLGpuEvent * gId = Tau_opencl_new_gpu_event(a1, dem_name, -1);
  if (!gId) {
    return clEnqueueNDRangeKernel_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  }

  if (!a9) {
    a9 = &gId->event;
  }

  TAU_PROFILE_START(t);
  cl_int retval = clEnqueueNDRangeKernel_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);

  if (!gId->event) {
    gId->event = *a9;
    clRetainEvent_noinst(gId->event);
  }

  Tau_opencl_enqueue_event(gId);

  return retval;
}

cl_int clEnqueueTask(cl_command_queue a1, cl_kernel a2, cl_uint a3, const cl_event * a4, cl_event * a5) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueTask, cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *);
  return clEnqueueTask_h(a1,  a2,  a3,  a4,  a5);
}

cl_int clEnqueueNativeKernel(cl_command_queue a1, void (*a2)(void *), void * a3, size_t a4, cl_uint a5, 
                             const cl_mem * a6, const void ** a7, cl_uint a8, const cl_event * a9, cl_event * a10)
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueNativeKernel, cl_command_queue, void (*)(void *), void *, size_t, cl_uint,  
                       const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *);
  return clEnqueueNativeKernel_h(a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);
}

cl_int clEnqueueMarker(cl_command_queue a1, cl_event * a2) 
{
  HANDLE_AND_AUTOTIMER(cl_int, clEnqueueMarker, cl_command_queue, cl_event *);
  return clEnqueueMarker_h(a1,  a2);
}

cl_int clEnqueueWaitForEvents(cl_command_queue a1, cl_uint a2, const cl_event * a3) 
{
  HANDLE_AND_TIMER(cl_int, clEnqueueWaitForEvents, cl_command_queue, cl_uint, const cl_event *);
  TAU_PROFILE_START(t);
  cl_int retval = clEnqueueWaitForEvents_h(a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  Tau_opencl_register_sync_event();
  return retval;
}

cl_int clEnqueueBarrier(cl_command_queue a1) 
{
  HANDLE_AND_TIMER(cl_int, clEnqueueBarrier, cl_command_queue);
  TAU_PROFILE_START(t);
  cl_int retval = clEnqueueBarrier_h(a1);
  TAU_PROFILE_STOP(t);
  Tau_opencl_register_sync_event();
  return retval;
}

void * clGetExtensionFunctionAddress(const char * a1) 
{
  HANDLE_AND_AUTOTIMER(void *, clGetExtensionFunctionAddress, const char *);
  return clGetExtensionFunctionAddress_h(a1);
}

