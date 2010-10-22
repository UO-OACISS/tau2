#include <Profile/Profiler.h>
#include <Profile/TauGpuAdapterOpenCL.h>
#include <dlfcn.h>
#include <stdio.h>
#include <CL/cl.h>

const char * tau_orig_libname = "libOpenCL.so";
static void *tau_handle = NULL;

static TauUserEvent *MemoryCopyEventHtoD;
static TauUserEvent *MemoryCopyEventDtoH;
static TauUserEvent *MemoryCopyEventDtoD;

void check_memory_init()
{

	static bool init = false;
	if (!init)
	{
		MemoryCopyEventHtoD = (TauUserEvent *) Tau_get_userevent("Bytes copied from Host to Device");
		MemoryCopyEventDtoH = (TauUserEvent *) Tau_get_userevent("Bytes copied from Device to Host");
		MemoryCopyEventDtoD = (TauUserEvent *) Tau_get_userevent("Bytes copied from Device to Device");
		init = true;
	}
}


cl_int clGetPlatformIDs(cl_uint a1, cl_platform_id * a2, cl_uint * a3) {

  typedef cl_int (*clGetPlatformIDs_p) (cl_uint, cl_platform_id *, cl_uint *);
  static clGetPlatformIDs_p clGetPlatformIDs_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetPlatformIDs_h == NULL)
	clGetPlatformIDs_h = (clGetPlatformIDs_p) dlsym(tau_handle,"clGetPlatformIDs"); 
    if (clGetPlatformIDs_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetPlatformIDs_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetPlatformInfo(cl_platform_id a1, cl_platform_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetPlatformInfo_p) (cl_platform_id, cl_platform_info, size_t, void *, size_t *);
  static clGetPlatformInfo_p clGetPlatformInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetPlatformInfo_h == NULL)
	clGetPlatformInfo_h = (clGetPlatformInfo_p) dlsym(tau_handle,"clGetPlatformInfo"); 
    if (clGetPlatformInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetPlatformInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetDeviceIDs(cl_platform_id a1, cl_device_type a2, cl_uint a3, cl_device_id * a4, cl_uint * a5) {

  typedef cl_int (*clGetDeviceIDs_p) (cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  static clGetDeviceIDs_p clGetDeviceIDs_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetDeviceIDs_h == NULL)
	clGetDeviceIDs_h = (clGetDeviceIDs_p) dlsym(tau_handle,"clGetDeviceIDs"); 
    if (clGetDeviceIDs_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetDeviceIDs_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetDeviceInfo(cl_device_id a1, cl_device_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetDeviceInfo_p) (cl_device_id, cl_device_info, size_t, void *, size_t *);
  static clGetDeviceInfo_p clGetDeviceInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetDeviceInfo_h == NULL)
	clGetDeviceInfo_h = (clGetDeviceInfo_p) dlsym(tau_handle,"clGetDeviceInfo"); 
    if (clGetDeviceInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetDeviceInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_context clCreateContext(const cl_context_properties * a1, cl_uint a2, const cl_device_id * a3, void (*a4)(const char *, const void *, size_t, void *), void * a5, cl_int * a6) {

  typedef cl_context (*clCreateContext_p) (const cl_context_properties *, cl_uint, const cl_device_id *, void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
  static clCreateContext_p clCreateContext_h = NULL;
  cl_context retval;
  TAU_PROFILE_TIMER(t,"cl_context clCreateContext(const cl_context_properties *, cl_uint, const cl_device_id *, void (*)(const char *, const void *, size_t, void *) C, void *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateContext_h == NULL)
	clCreateContext_h = (clCreateContext_p) dlsym(tau_handle,"clCreateContext"); 
    if (clCreateContext_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateContext_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_context clCreateContextFromType(const cl_context_properties * a1, cl_device_type a2, void (*a3)(const char *, const void *, size_t, void *), void * a4, cl_int * a5) {

  typedef cl_context (*clCreateContextFromType_p) (const cl_context_properties *, cl_device_type, void (*)(const char *, const void *, size_t, void *), void *, cl_int *);
  static clCreateContextFromType_p clCreateContextFromType_h = NULL;
  cl_context retval;
  TAU_PROFILE_TIMER(t,"cl_context clCreateContextFromType(const cl_context_properties *, cl_device_type, void (*)(const char *, const void *, size_t, void *) C, void *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateContextFromType_h == NULL)
	clCreateContextFromType_h = (clCreateContextFromType_p) dlsym(tau_handle,"clCreateContextFromType"); 
    if (clCreateContextFromType_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateContextFromType_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainContext(cl_context a1) {

  typedef cl_int (*clRetainContext_p) (cl_context);
  static clRetainContext_p clRetainContext_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainContext(cl_context) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainContext_h == NULL)
	clRetainContext_h = (clRetainContext_p) dlsym(tau_handle,"clRetainContext"); 
    if (clRetainContext_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainContext_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clReleaseContext(cl_context a1) {

  typedef cl_int (*clReleaseContext_p) (cl_context);
  static clReleaseContext_p clReleaseContext_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseContext(cl_context) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseContext_h == NULL)
	clReleaseContext_h = (clReleaseContext_p) dlsym(tau_handle,"clReleaseContext"); 
    if (clReleaseContext_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseContext_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetContextInfo(cl_context a1, cl_context_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetContextInfo_p) (cl_context, cl_context_info, size_t, void *, size_t *);
  static clGetContextInfo_p clGetContextInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetContextInfo(cl_context, cl_context_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetContextInfo_h == NULL)
	clGetContextInfo_h = (clGetContextInfo_p) dlsym(tau_handle,"clGetContextInfo"); 
    if (clGetContextInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetContextInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_command_queue clCreateCommandQueue(cl_context a1, cl_device_id a2, cl_command_queue_properties a3, cl_int * a4) {

  typedef cl_command_queue (*clCreateCommandQueue_p) (cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
  static clCreateCommandQueue_p clCreateCommandQueue_h = NULL;
  cl_command_queue retval;
  TAU_PROFILE_TIMER(t,"cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateCommandQueue_h == NULL)
	clCreateCommandQueue_h = (clCreateCommandQueue_p) dlsym(tau_handle,"clCreateCommandQueue"); 
    if (clCreateCommandQueue_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateCommandQueue_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainCommandQueue(cl_command_queue a1) {

  typedef cl_int (*clRetainCommandQueue_p) (cl_command_queue);
  static clRetainCommandQueue_p clRetainCommandQueue_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainCommandQueue(cl_command_queue) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainCommandQueue_h == NULL)
	clRetainCommandQueue_h = (clRetainCommandQueue_p) dlsym(tau_handle,"clRetainCommandQueue"); 
    if (clRetainCommandQueue_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainCommandQueue_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clReleaseCommandQueue(cl_command_queue a1) {

  typedef cl_int (*clReleaseCommandQueue_p) (cl_command_queue);
  static clReleaseCommandQueue_p clReleaseCommandQueue_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseCommandQueue(cl_command_queue) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseCommandQueue_h == NULL)
	clReleaseCommandQueue_h = (clReleaseCommandQueue_p) dlsym(tau_handle,"clReleaseCommandQueue"); 
    if (clReleaseCommandQueue_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseCommandQueue_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetCommandQueueInfo(cl_command_queue a1, cl_command_queue_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetCommandQueueInfo_p) (cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
  static clGetCommandQueueInfo_p clGetCommandQueueInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetCommandQueueInfo(cl_command_queue, cl_command_queue_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetCommandQueueInfo_h == NULL)
	clGetCommandQueueInfo_h = (clGetCommandQueueInfo_p) dlsym(tau_handle,"clGetCommandQueueInfo"); 
    if (clGetCommandQueueInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetCommandQueueInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clSetCommandQueueProperty(cl_command_queue a1, cl_command_queue_properties a2, cl_bool a3, cl_command_queue_properties * a4) {

  typedef cl_int (*clSetCommandQueueProperty_p) (cl_command_queue, cl_command_queue_properties, cl_bool, cl_command_queue_properties *);
  static clSetCommandQueueProperty_p clSetCommandQueueProperty_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clSetCommandQueueProperty(cl_command_queue, cl_command_queue_properties, cl_bool, cl_command_queue_properties *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clSetCommandQueueProperty_h == NULL)
	clSetCommandQueueProperty_h = (clSetCommandQueueProperty_p) dlsym(tau_handle,"clSetCommandQueueProperty"); 
    if (clSetCommandQueueProperty_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clSetCommandQueueProperty_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_mem clCreateBuffer(cl_context a1, cl_mem_flags a2, size_t a3, void * a4, cl_int * a5) {

  typedef cl_mem (*clCreateBuffer_p) (cl_context, cl_mem_flags, size_t, void *, cl_int *);
  static clCreateBuffer_p clCreateBuffer_h = NULL;
  cl_mem retval;
  TAU_PROFILE_TIMER(t,"cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateBuffer_h == NULL)
	clCreateBuffer_h = (clCreateBuffer_p) dlsym(tau_handle,"clCreateBuffer"); 
    if (clCreateBuffer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateBuffer_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_mem clCreateImage2D(cl_context a1, cl_mem_flags a2, const cl_image_format * a3, size_t a4, size_t a5, size_t a6, void * a7, cl_int * a8) {

  typedef cl_mem (*clCreateImage2D_p) (cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *);
  static clCreateImage2D_p clCreateImage2D_h = NULL;
  cl_mem retval;
  TAU_PROFILE_TIMER(t,"cl_mem clCreateImage2D(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateImage2D_h == NULL)
	clCreateImage2D_h = (clCreateImage2D_p) dlsym(tau_handle,"clCreateImage2D"); 
    if (clCreateImage2D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateImage2D_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_mem clCreateImage3D(cl_context a1, cl_mem_flags a2, const cl_image_format * a3, size_t a4, size_t a5, size_t a6, size_t a7, size_t a8, void * a9, cl_int * a10) {

  typedef cl_mem (*clCreateImage3D_p) (cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *);
  static clCreateImage3D_p clCreateImage3D_h = NULL;
  cl_mem retval;
  TAU_PROFILE_TIMER(t,"cl_mem clCreateImage3D(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, size_t, size_t, void *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateImage3D_h == NULL)
	clCreateImage3D_h = (clCreateImage3D_p) dlsym(tau_handle,"clCreateImage3D"); 
    if (clCreateImage3D_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateImage3D_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainMemObject(cl_mem a1) {

  typedef cl_int (*clRetainMemObject_p) (cl_mem);
  static clRetainMemObject_p clRetainMemObject_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainMemObject(cl_mem) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainMemObject_h == NULL)
	clRetainMemObject_h = (clRetainMemObject_p) dlsym(tau_handle,"clRetainMemObject"); 
    if (clRetainMemObject_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainMemObject_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
/*
cl_int clReleaseMemObject(cl_mem a1) {

  typedef cl_int (*clReleaseMemObject_p) (cl_mem);
  static clReleaseMemObject_p clReleaseMemObject_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseMemObject(cl_mem) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseMemObject_h == NULL)
	clReleaseMemObject_h = (clReleaseMemObject_p) dlsym(tau_handle,"clReleaseMemObject"); 
    if (clReleaseMemObject_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseMemObject_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}*/

cl_int clGetSupportedImageFormats(cl_context a1, cl_mem_flags a2, cl_mem_object_type a3, cl_uint a4, cl_image_format * a5, cl_uint * a6) {

  typedef cl_int (*clGetSupportedImageFormats_p) (cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *);
  static clGetSupportedImageFormats_p clGetSupportedImageFormats_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetSupportedImageFormats(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_image_format *, cl_uint *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetSupportedImageFormats_h == NULL)
	clGetSupportedImageFormats_h = (clGetSupportedImageFormats_p) dlsym(tau_handle,"clGetSupportedImageFormats"); 
    if (clGetSupportedImageFormats_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetSupportedImageFormats_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetMemObjectInfo(cl_mem a1, cl_mem_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetMemObjectInfo_p) (cl_mem, cl_mem_info, size_t, void *, size_t *);
  static clGetMemObjectInfo_p clGetMemObjectInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetMemObjectInfo(cl_mem, cl_mem_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetMemObjectInfo_h == NULL)
	clGetMemObjectInfo_h = (clGetMemObjectInfo_p) dlsym(tau_handle,"clGetMemObjectInfo"); 
    if (clGetMemObjectInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetMemObjectInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetImageInfo(cl_mem a1, cl_image_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetImageInfo_p) (cl_mem, cl_image_info, size_t, void *, size_t *);
  static clGetImageInfo_p clGetImageInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetImageInfo(cl_mem, cl_image_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetImageInfo_h == NULL)
	clGetImageInfo_h = (clGetImageInfo_p) dlsym(tau_handle,"clGetImageInfo"); 
    if (clGetImageInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetImageInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_sampler clCreateSampler(cl_context a1, cl_bool a2, cl_addressing_mode a3, cl_filter_mode a4, cl_int * a5) {

  typedef cl_sampler (*clCreateSampler_p) (cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *);
  static clCreateSampler_p clCreateSampler_h = NULL;
  cl_sampler retval;
  TAU_PROFILE_TIMER(t,"cl_sampler clCreateSampler(cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateSampler_h == NULL)
	clCreateSampler_h = (clCreateSampler_p) dlsym(tau_handle,"clCreateSampler"); 
    if (clCreateSampler_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateSampler_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainSampler(cl_sampler a1) {

  typedef cl_int (*clRetainSampler_p) (cl_sampler);
  static clRetainSampler_p clRetainSampler_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainSampler(cl_sampler) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainSampler_h == NULL)
	clRetainSampler_h = (clRetainSampler_p) dlsym(tau_handle,"clRetainSampler"); 
    if (clRetainSampler_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainSampler_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clReleaseSampler(cl_sampler a1) {

  typedef cl_int (*clReleaseSampler_p) (cl_sampler);
  static clReleaseSampler_p clReleaseSampler_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseSampler(cl_sampler) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseSampler_h == NULL)
	clReleaseSampler_h = (clReleaseSampler_p) dlsym(tau_handle,"clReleaseSampler"); 
    if (clReleaseSampler_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseSampler_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetSamplerInfo(cl_sampler a1, cl_sampler_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetSamplerInfo_p) (cl_sampler, cl_sampler_info, size_t, void *, size_t *);
  static clGetSamplerInfo_p clGetSamplerInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetSamplerInfo(cl_sampler, cl_sampler_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetSamplerInfo_h == NULL)
	clGetSamplerInfo_h = (clGetSamplerInfo_p) dlsym(tau_handle,"clGetSamplerInfo"); 
    if (clGetSamplerInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetSamplerInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_program clCreateProgramWithSource(cl_context a1, cl_uint a2, const char ** a3, const size_t * a4, cl_int * a5) {

  typedef cl_program (*clCreateProgramWithSource_p) (cl_context, cl_uint, const char **, const size_t *, cl_int *);
  static clCreateProgramWithSource_p clCreateProgramWithSource_h = NULL;
  cl_program retval;
  TAU_PROFILE_TIMER(t,"cl_program clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateProgramWithSource_h == NULL)
	clCreateProgramWithSource_h = (clCreateProgramWithSource_p) dlsym(tau_handle,"clCreateProgramWithSource"); 
    if (clCreateProgramWithSource_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateProgramWithSource_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_program clCreateProgramWithBinary(cl_context a1, cl_uint a2, const cl_device_id * a3, const size_t * a4, const unsigned char ** a5, cl_int * a6, cl_int * a7) {

  typedef cl_program (*clCreateProgramWithBinary_p) (cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *);
  static clCreateProgramWithBinary_p clCreateProgramWithBinary_h = NULL;
  cl_program retval;
  TAU_PROFILE_TIMER(t,"cl_program clCreateProgramWithBinary(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateProgramWithBinary_h == NULL)
	clCreateProgramWithBinary_h = (clCreateProgramWithBinary_p) dlsym(tau_handle,"clCreateProgramWithBinary"); 
    if (clCreateProgramWithBinary_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateProgramWithBinary_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainProgram(cl_program a1) {

  typedef cl_int (*clRetainProgram_p) (cl_program);
  static clRetainProgram_p clRetainProgram_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainProgram(cl_program) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainProgram_h == NULL)
	clRetainProgram_h = (clRetainProgram_p) dlsym(tau_handle,"clRetainProgram"); 
    if (clRetainProgram_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainProgram_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clReleaseProgram(cl_program a1) {

  typedef cl_int (*clReleaseProgram_p) (cl_program);
  static clReleaseProgram_p clReleaseProgram_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseProgram(cl_program) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseProgram_h == NULL)
	clReleaseProgram_h = (clReleaseProgram_p) dlsym(tau_handle,"clReleaseProgram"); 
    if (clReleaseProgram_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseProgram_h)( a1);
  TAU_PROFILE_STOP(t);
	Tau_opencl_exit();
  }
  return retval;

}

cl_int clBuildProgram(cl_program a1, cl_uint a2, const cl_device_id * a3, const char * a4, void (*a5)(cl_program, void *), void * a6) {

  typedef cl_int (*clBuildProgram_p) (cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *), void *);
  static clBuildProgram_p clBuildProgram_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void (*)(cl_program, void *) C, void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clBuildProgram_h == NULL)
	clBuildProgram_h = (clBuildProgram_p) dlsym(tau_handle,"clBuildProgram"); 
    if (clBuildProgram_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clBuildProgram_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clUnloadCompiler() {

  typedef cl_int (*clUnloadCompiler_p) ();
  static clUnloadCompiler_p clUnloadCompiler_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clUnloadCompiler() C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clUnloadCompiler_h == NULL)
	clUnloadCompiler_h = (clUnloadCompiler_p) dlsym(tau_handle,"clUnloadCompiler"); 
    if (clUnloadCompiler_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clUnloadCompiler_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetProgramInfo(cl_program a1, cl_program_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetProgramInfo_p) (cl_program, cl_program_info, size_t, void *, size_t *);
  static clGetProgramInfo_p clGetProgramInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetProgramInfo(cl_program, cl_program_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetProgramInfo_h == NULL)
	clGetProgramInfo_h = (clGetProgramInfo_p) dlsym(tau_handle,"clGetProgramInfo"); 
    if (clGetProgramInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetProgramInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetProgramBuildInfo(cl_program a1, cl_device_id a2, cl_program_build_info a3, size_t a4, void * a5, size_t * a6) {

  typedef cl_int (*clGetProgramBuildInfo_p) (cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
  static clGetProgramBuildInfo_p clGetProgramBuildInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetProgramBuildInfo_h == NULL)
	clGetProgramBuildInfo_h = (clGetProgramBuildInfo_p) dlsym(tau_handle,"clGetProgramBuildInfo"); 
    if (clGetProgramBuildInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetProgramBuildInfo_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_kernel clCreateKernel(cl_program a1, const char * a2, cl_int * a3) {

  typedef cl_kernel (*clCreateKernel_p) (cl_program, const char *, cl_int *);
  static clCreateKernel_p clCreateKernel_h = NULL;
  cl_kernel retval;
  TAU_PROFILE_TIMER(t,"cl_kernel clCreateKernel(cl_program, const char *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateKernel_h == NULL)
	clCreateKernel_h = (clCreateKernel_p) dlsym(tau_handle,"clCreateKernel"); 
    if (clCreateKernel_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateKernel_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clCreateKernelsInProgram(cl_program a1, cl_uint a2, cl_kernel * a3, cl_uint * a4) {

  typedef cl_int (*clCreateKernelsInProgram_p) (cl_program, cl_uint, cl_kernel *, cl_uint *);
  static clCreateKernelsInProgram_p clCreateKernelsInProgram_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clCreateKernelsInProgram(cl_program, cl_uint, cl_kernel *, cl_uint *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clCreateKernelsInProgram_h == NULL)
	clCreateKernelsInProgram_h = (clCreateKernelsInProgram_p) dlsym(tau_handle,"clCreateKernelsInProgram"); 
    if (clCreateKernelsInProgram_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clCreateKernelsInProgram_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainKernel(cl_kernel a1) {

  typedef cl_int (*clRetainKernel_p) (cl_kernel);
  static clRetainKernel_p clRetainKernel_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainKernel(cl_kernel) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainKernel_h == NULL)
	clRetainKernel_h = (clRetainKernel_p) dlsym(tau_handle,"clRetainKernel"); 
    if (clRetainKernel_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainKernel_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clReleaseKernel(cl_kernel a1) {

  typedef cl_int (*clReleaseKernel_p) (cl_kernel);
  static clReleaseKernel_p clReleaseKernel_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseKernel(cl_kernel) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseKernel_h == NULL)
	clReleaseKernel_h = (clReleaseKernel_p) dlsym(tau_handle,"clReleaseKernel"); 
    if (clReleaseKernel_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseKernel_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clSetKernelArg(cl_kernel a1, cl_uint a2, size_t a3, const void * a4) {

  typedef cl_int (*clSetKernelArg_p) (cl_kernel, cl_uint, size_t, const void *);
  static clSetKernelArg_p clSetKernelArg_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clSetKernelArg_h == NULL)
	clSetKernelArg_h = (clSetKernelArg_p) dlsym(tau_handle,"clSetKernelArg"); 
    if (clSetKernelArg_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clSetKernelArg_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetKernelInfo(cl_kernel a1, cl_kernel_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetKernelInfo_p) (cl_kernel, cl_kernel_info, size_t, void *, size_t *);
  static clGetKernelInfo_p clGetKernelInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetKernelInfo(cl_kernel, cl_kernel_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetKernelInfo_h == NULL)
	clGetKernelInfo_h = (clGetKernelInfo_p) dlsym(tau_handle,"clGetKernelInfo"); 
    if (clGetKernelInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetKernelInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetKernelWorkGroupInfo(cl_kernel a1, cl_device_id a2, cl_kernel_work_group_info a3, size_t a4, void * a5, size_t * a6) {

  typedef cl_int (*clGetKernelWorkGroupInfo_p) (cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *);
  static clGetKernelWorkGroupInfo_p clGetKernelWorkGroupInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetKernelWorkGroupInfo(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetKernelWorkGroupInfo_h == NULL)
	clGetKernelWorkGroupInfo_h = (clGetKernelWorkGroupInfo_p) dlsym(tau_handle,"clGetKernelWorkGroupInfo"); 
    if (clGetKernelWorkGroupInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetKernelWorkGroupInfo_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clWaitForEvents(cl_uint a1, const cl_event * a2) {

  typedef cl_int (*clWaitForEvents_p) (cl_uint, const cl_event *);
  static clWaitForEvents_p clWaitForEvents_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clWaitForEvents(cl_uint, const cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clWaitForEvents_h == NULL)
	clWaitForEvents_h = (clWaitForEvents_p) dlsym(tau_handle,"clWaitForEvents"); 
    if (clWaitForEvents_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clWaitForEvents_h)( a1,  a2);
  TAU_PROFILE_STOP(t);


  }
  return retval;

}

cl_int clGetEventInfo(cl_event a1, cl_event_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetEventInfo_p) (cl_event, cl_event_info, size_t, void *, size_t *);
  static clGetEventInfo_p clGetEventInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetEventInfo(cl_event, cl_event_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetEventInfo_h == NULL)
	clGetEventInfo_h = (clGetEventInfo_p) dlsym(tau_handle,"clGetEventInfo"); 
    if (clGetEventInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetEventInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clRetainEvent(cl_event a1) {

  typedef cl_int (*clRetainEvent_p) (cl_event);
  static clRetainEvent_p clRetainEvent_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clRetainEvent(cl_event) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clRetainEvent_h == NULL)
	clRetainEvent_h = (clRetainEvent_p) dlsym(tau_handle,"clRetainEvent"); 
    if (clRetainEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clRetainEvent_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clReleaseEvent(cl_event a1) {

  typedef cl_int (*clReleaseEvent_p) (cl_event);
  static clReleaseEvent_p clReleaseEvent_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clReleaseEvent(cl_event) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clReleaseEvent_h == NULL)
	clReleaseEvent_h = (clReleaseEvent_p) dlsym(tau_handle,"clReleaseEvent"); 
    if (clReleaseEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clReleaseEvent_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clGetEventProfilingInfo(cl_event a1, cl_profiling_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetEventProfilingInfo_p) (cl_event, cl_profiling_info, size_t, void *, size_t *);
  static clGetEventProfilingInfo_p clGetEventProfilingInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetEventProfilingInfo_h == NULL)
	clGetEventProfilingInfo_h = (clGetEventProfilingInfo_p) dlsym(tau_handle,"clGetEventProfilingInfo"); 
    if (clGetEventProfilingInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetEventProfilingInfo_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
cl_int clGetEventProfilingInfo_noinst(cl_event a1, cl_profiling_info a2, size_t a3, void * a4, size_t * a5) {

  typedef cl_int (*clGetEventProfilingInfo_p) (cl_event, cl_profiling_info, size_t, void *, size_t *);
  static clGetEventProfilingInfo_p clGetEventProfilingInfo_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void *, size_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetEventProfilingInfo_h == NULL)
	clGetEventProfilingInfo_h = (clGetEventProfilingInfo_p) dlsym(tau_handle,"clGetEventProfilingInfo"); 
    if (clGetEventProfilingInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  retval  =  (*clGetEventProfilingInfo_h)( a1,  a2,  a3,  a4,  a5);
  }
  return retval;

}

cl_int clFlush(cl_command_queue a1) {

  typedef cl_int (*clFlush_p) (cl_command_queue);
  static clFlush_p clFlush_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clFlush(cl_command_queue) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clFlush_h == NULL)
	clFlush_h = (clFlush_p) dlsym(tau_handle,"clFlush"); 
    if (clFlush_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clFlush_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clFinish(cl_command_queue a1) {

  typedef cl_int (*clFinish_p) (cl_command_queue);
  static clFinish_p clFinish_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clFinish(cl_command_queue) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clFinish_h == NULL)
	clFinish_h = (clFinish_p) dlsym(tau_handle,"clFinish"); 
    if (clFinish_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clFinish_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueReadBuffer(cl_command_queue a1, cl_mem a2, cl_bool a3, size_t a4, size_t a5, void * a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueReadBuffer_p) (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueReadBuffer_p clEnqueueReadBuffer_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueReadBuffer_h == NULL)
	clEnqueueReadBuffer_h = (clEnqueueReadBuffer_p) dlsym(tau_handle,"clEnqueueReadBuffer"); 
    if (clEnqueueReadBuffer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
	if (a9 == NULL)
	{
		//printf("cl_event is null.\n");
		cl_event new_event;
		a9 = &new_event;
	}
	memcpy_callback_data *mem_data = (memcpy_callback_data*) malloc(memcpy_data_size);
	strcpy(mem_data->name, "ReadBuffer");
	mem_data->memcpy_type = MemcpyDtoH;
	printf("name %s.\n", mem_data->name);
  TAU_PROFILE_START(t);
	check_memory_init();
	TAU_EVENT(MemoryCopyEventDtoH, a5);
  retval  =  (*clEnqueueReadBuffer_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
	clSetEventCallback((*a9), CL_COMPLETE, Tau_opencl_memcpy_callback, mem_data);
  TAU_PROFILE_STOP(t);
	//free(mem_data);
  }
  return retval;

}

cl_int clEnqueueWriteBuffer(cl_command_queue a1, cl_mem a2, cl_bool a3, size_t a4, size_t a5, const void * a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueWriteBuffer_p) (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueWriteBuffer_p clEnqueueWriteBuffer_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueWriteBuffer_h == NULL)
	clEnqueueWriteBuffer_h = (clEnqueueWriteBuffer_p) dlsym(tau_handle,"clEnqueueWriteBuffer"); 
    if (clEnqueueWriteBuffer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
	memcpy_callback_data *mem_data = (memcpy_callback_data*) malloc(memcpy_data_size);
	strcpy(mem_data->name, "WriteBuffer");
	mem_data->memcpy_type = MemcpyHtoD;
	if (a9 == NULL)
	{
		//printf("cl_event is null.\n");
		cl_event new_event;
		a9 = &new_event;
	}
  TAU_PROFILE_START(t);
	check_memory_init();
	TAU_EVENT(MemoryCopyEventHtoD, a5);
  retval  =  (*clEnqueueWriteBuffer_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
	clSetEventCallback((*a9), CL_COMPLETE, Tau_opencl_memcpy_callback, mem_data);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}
/* Assuming copy is between two devices. */
cl_int clEnqueueCopyBuffer(cl_command_queue a1, cl_mem a2, cl_mem a3, size_t a4, size_t a5, size_t a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueCopyBuffer_p) (cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
  static clEnqueueCopyBuffer_p clEnqueueCopyBuffer_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueCopyBuffer(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueCopyBuffer_h == NULL)
	clEnqueueCopyBuffer_h = (clEnqueueCopyBuffer_p) dlsym(tau_handle,"clEnqueueCopyBuffer"); 
    if (clEnqueueCopyBuffer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
	memcpy_callback_data *mem_data = (memcpy_callback_data*) malloc(memcpy_data_size);
	strcpy(mem_data->name, "CopyBuffer");
	mem_data->memcpy_type = MemcpyHtoD;
	if (a9 == NULL)
	{
		//printf("cl_event is null.\n");
		cl_event new_event;
		a9 = &new_event;
	}
  TAU_PROFILE_START(t);
	check_memory_init();
	TAU_EVENT(MemoryCopyEventDtoD, a6);
  retval  =  (*clEnqueueCopyBuffer_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
	clSetEventCallback((*a9), CL_COMPLETE, Tau_opencl_memcpy_callback, mem_data);
  TAU_PROFILE_STOP(t);
	//free(mem_data);
  }
  return retval;

}

//No example found -- not implemented.
cl_int clEnqueueReadImage(cl_command_queue a1, cl_mem a2, cl_bool a3, const size_t * a4, const size_t * a5, size_t a6, size_t a7, void * a8, cl_uint a9, const cl_event * a10, cl_event * a11) {

  typedef cl_int (*clEnqueueReadImage_p) (cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueReadImage_p clEnqueueReadImage_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueReadImage(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueReadImage_h == NULL)
	clEnqueueReadImage_h = (clEnqueueReadImage_p) dlsym(tau_handle,"clEnqueueReadImage"); 
    if (clEnqueueReadImage_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueReadImage_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10,  a11);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

//No example found -- not implemented.
cl_int clEnqueueWriteImage(cl_command_queue a1, cl_mem a2, cl_bool a3, const size_t * a4, const size_t * a5, size_t a6, size_t a7, const void * a8, cl_uint a9, const cl_event * a10, cl_event * a11) {

  typedef cl_int (*clEnqueueWriteImage_p) (cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueWriteImage_p clEnqueueWriteImage_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueWriteImage(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueWriteImage_h == NULL)
	clEnqueueWriteImage_h = (clEnqueueWriteImage_p) dlsym(tau_handle,"clEnqueueWriteImage"); 
    if (clEnqueueWriteImage_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueWriteImage_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10,  a11);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

//No example found -- not implemented.
cl_int clEnqueueCopyImage(cl_command_queue a1, cl_mem a2, cl_mem a3, const size_t * a4, const size_t * a5, const size_t * a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueCopyImage_p) (cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueCopyImage_p clEnqueueCopyImage_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueCopyImage(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueCopyImage_h == NULL)
	clEnqueueCopyImage_h = (clEnqueueCopyImage_p) dlsym(tau_handle,"clEnqueueCopyImage"); 
    if (clEnqueueCopyImage_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueCopyImage_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

//No example found -- not implemented.
cl_int clEnqueueCopyImageToBuffer(cl_command_queue a1, cl_mem a2, cl_mem a3, const size_t * a4, const size_t * a5, size_t a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueCopyImageToBuffer_p) (cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, size_t, cl_uint, const cl_event *, cl_event *);
  static clEnqueueCopyImageToBuffer_p clEnqueueCopyImageToBuffer_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueCopyImageToBuffer(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, size_t, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueCopyImageToBuffer_h == NULL)
	clEnqueueCopyImageToBuffer_h = (clEnqueueCopyImageToBuffer_p) dlsym(tau_handle,"clEnqueueCopyImageToBuffer"); 
    if (clEnqueueCopyImageToBuffer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueCopyImageToBuffer_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

//No example found -- not implemented.
cl_int clEnqueueCopyBufferToImage(cl_command_queue a1, cl_mem a2, cl_mem a3, size_t a4, const size_t * a5, const size_t * a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueCopyBufferToImage_p) (cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueCopyBufferToImage_p clEnqueueCopyBufferToImage_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueCopyBufferToImage(cl_command_queue, cl_mem, cl_mem, size_t, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueCopyBufferToImage_h == NULL)
	clEnqueueCopyBufferToImage_h = (clEnqueueCopyBufferToImage_p) dlsym(tau_handle,"clEnqueueCopyBufferToImage"); 
    if (clEnqueueCopyBufferToImage_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueCopyBufferToImage_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

void * clEnqueueMapBuffer(cl_command_queue a1, cl_mem a2, cl_bool a3, cl_map_flags a4, size_t a5, size_t a6, cl_uint a7, const cl_event * a8, cl_event * a9, cl_int * a10) {

  typedef void * (*clEnqueueMapBuffer_p) (cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *);
  static clEnqueueMapBuffer_p clEnqueueMapBuffer_h = NULL;
  void * retval;
  TAU_PROFILE_TIMER(t,"void *clEnqueueMapBuffer(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueMapBuffer_h == NULL)
	clEnqueueMapBuffer_h = (clEnqueueMapBuffer_p) dlsym(tau_handle,"clEnqueueMapBuffer"); 
    if (clEnqueueMapBuffer_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
	//Seg. fault when tracked.
	check_memory_init();
	TAU_EVENT(MemoryCopyEventDtoH, a5);
  retval  =  (*clEnqueueMapBuffer_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

//No example found -- not implemented.
void * clEnqueueMapImage(cl_command_queue a1, cl_mem a2, cl_bool a3, cl_map_flags a4, const size_t * a5, const size_t * a6, size_t * a7, size_t * a8, cl_uint a9, const cl_event * a10, cl_event * a11, cl_int * a12) {

  typedef void * (*clEnqueueMapImage_p) (cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *);
  static clEnqueueMapImage_p clEnqueueMapImage_h = NULL;
  void * retval;
  TAU_PROFILE_TIMER(t,"void *clEnqueueMapImage(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *, const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *, cl_int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueMapImage_h == NULL)
	clEnqueueMapImage_h = (clEnqueueMapImage_p) dlsym(tau_handle,"clEnqueueMapImage"); 
    if (clEnqueueMapImage_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueMapImage_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10,  a11,  a12);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueUnmapMemObject(cl_command_queue a1, cl_mem a2, void * a3, cl_uint a4, const cl_event * a5, cl_event * a6) {

  typedef cl_int (*clEnqueueUnmapMemObject_p) (cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueUnmapMemObject_p clEnqueueUnmapMemObject_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueUnmapMemObject(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueUnmapMemObject_h == NULL)
	clEnqueueUnmapMemObject_h = (clEnqueueUnmapMemObject_p) dlsym(tau_handle,"clEnqueueUnmapMemObject"); 
    if (clEnqueueUnmapMemObject_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueUnmapMemObject_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueNDRangeKernel(cl_command_queue a1, cl_kernel a2, cl_uint a3, const size_t * a4, const size_t * a5, const size_t * a6, cl_uint a7, const cl_event * a8, cl_event * a9) {

  typedef cl_int (*clEnqueueNDRangeKernel_p) (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
  static clEnqueueNDRangeKernel_p clEnqueueNDRangeKernel_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueNDRangeKernel_h == NULL)
	clEnqueueNDRangeKernel_h = (clEnqueueNDRangeKernel_p) dlsym(tau_handle,"clEnqueueNDRangeKernel"); 
    if (clEnqueueNDRangeKernel_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }


	if (a9 == NULL)
	{
		//printf("cl_event is null.\n");
		cl_event new_event;
		a9 = &new_event;
	}
	kernel_callback_data *kernel_data = (kernel_callback_data*) malloc(memcpy_data_size);
	int err;
	err = clGetKernelInfo(a2, CL_KERNEL_FUNCTION_NAME,
	sizeof(char[TAU_MAX_FUNCTIONNAME]), kernel_data->name, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Cannot get Kernel name.\n");
	  exit(1);	
	}
	//printf("name returned from KernelInfo: %s.\n", name_returned);
	//kernel_data.name = "NDRangeKernel";
	printf("name: %s.\n", kernel_data->name);
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueNDRangeKernel_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
	clSetEventCallback((*a9), CL_COMPLETE, Tau_opencl_kernel_callback, kernel_data);
  TAU_PROFILE_STOP(t);
	//free(kernel_data);
  }
  return retval;

}

cl_int clEnqueueTask(cl_command_queue a1, cl_kernel a2, cl_uint a3, const cl_event * a4, cl_event * a5) {

  typedef cl_int (*clEnqueueTask_p) (cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *);
  static clEnqueueTask_p clEnqueueTask_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueTask(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueTask_h == NULL)
	clEnqueueTask_h = (clEnqueueTask_p) dlsym(tau_handle,"clEnqueueTask"); 
    if (clEnqueueTask_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueTask_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueNativeKernel(cl_command_queue a1, void (*a2)(void *), void * a3, size_t a4, cl_uint a5, const cl_mem * a6, const void ** a7, cl_uint a8, const cl_event * a9, cl_event * a10) {

  typedef cl_int (*clEnqueueNativeKernel_p) (cl_command_queue, void (*)(void *), void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *);
  static clEnqueueNativeKernel_p clEnqueueNativeKernel_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueNativeKernel(cl_command_queue, void (*)(void *) C, void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueNativeKernel_h == NULL)
	clEnqueueNativeKernel_h = (clEnqueueNativeKernel_p) dlsym(tau_handle,"clEnqueueNativeKernel"); 
    if (clEnqueueNativeKernel_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueNativeKernel_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueMarker(cl_command_queue a1, cl_event * a2) {

  typedef cl_int (*clEnqueueMarker_p) (cl_command_queue, cl_event *);
  static clEnqueueMarker_p clEnqueueMarker_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueMarker(cl_command_queue, cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueMarker_h == NULL)
	clEnqueueMarker_h = (clEnqueueMarker_p) dlsym(tau_handle,"clEnqueueMarker"); 
    if (clEnqueueMarker_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueMarker_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueWaitForEvents(cl_command_queue a1, cl_uint a2, const cl_event * a3) {

  typedef cl_int (*clEnqueueWaitForEvents_p) (cl_command_queue, cl_uint, const cl_event *);
  static clEnqueueWaitForEvents_p clEnqueueWaitForEvents_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueWaitForEvents(cl_command_queue, cl_uint, const cl_event *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueWaitForEvents_h == NULL)
	clEnqueueWaitForEvents_h = (clEnqueueWaitForEvents_p) dlsym(tau_handle,"clEnqueueWaitForEvents"); 
    if (clEnqueueWaitForEvents_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueWaitForEvents_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cl_int clEnqueueBarrier(cl_command_queue a1) {

  typedef cl_int (*clEnqueueBarrier_p) (cl_command_queue);
  static clEnqueueBarrier_p clEnqueueBarrier_h = NULL;
  cl_int retval;
  TAU_PROFILE_TIMER(t,"cl_int clEnqueueBarrier(cl_command_queue) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clEnqueueBarrier_h == NULL)
	clEnqueueBarrier_h = (clEnqueueBarrier_p) dlsym(tau_handle,"clEnqueueBarrier"); 
    if (clEnqueueBarrier_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clEnqueueBarrier_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

void * clGetExtensionFunctionAddress(const char * a1) {

  typedef void * (*clGetExtensionFunctionAddress_p) (const char *);
  static clGetExtensionFunctionAddress_p clGetExtensionFunctionAddress_h = NULL;
  void * retval;
  TAU_PROFILE_TIMER(t,"void *clGetExtensionFunctionAddress(const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (clGetExtensionFunctionAddress_h == NULL)
	clGetExtensionFunctionAddress_h = (clGetExtensionFunctionAddress_p) dlsym(tau_handle,"clGetExtensionFunctionAddress"); 
    if (clGetExtensionFunctionAddress_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*clGetExtensionFunctionAddress_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

