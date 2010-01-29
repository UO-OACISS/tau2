#include <cuda_runtime_api.h>
#include <Profile/Profiler.h>
#include <dlfcn.h>

#include <stdio.h>

const char * tau_orig_libname = "libcudart.so";
static void *tau_handle = NULL;

/* void __attribute__ ((constructor)) on_tauwrap_load(void); */
/* void __attribute__ ((destructor)) on_tauwrap_unload(void); */

/* void on_tauwrap_load() */
/* { */
/*   fprintf(stderr,"[pid=%d]###############on_wrap_load\n", getpid()); */
/*   //TAU_REGISTER_THREAD(); */
/*   //TAU_START("TAUAPP"); */
/*   TAU_PROFILE_SET_NODE(0); */
/*   Tau_create_top_level_timer_if_necessary(); */
/*   //TAU_STATIC_PHASE_START(".TAUCudaApplication"); */
/* } */

/* void on_tauwrap_unload() */
/* { */
/*   fprintf(stderr,"[pid=%d]###################on_wrap_unload\n", getpid()); */
/*   Tau_stop_top_level_timer_if_necessary(); */
/*   //TAU_STATIC_PHASE_STOP(".TAUCudaApplication"); */
/*   //TAU_STOP("TAUAPP"); */
/* } */

void atexit_handler() {
    Tau_create_top_level_timer_if_necessary();
  //TAU_STOP("WRAPPER");
}

void checkinit() {
  static int init = 1;
  if (init) {
    init = 0;
    Tau_create_top_level_timer_if_necessary();
    //TAU_START("WRAPPER");
    atexit(atexit_handler);
  }
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr * a1, struct cudaExtent a2) {
  checkinit();
  static cudaError_t (*cudaMalloc3D_h) (struct cudaPitchedPtr *, struct cudaExtent) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMalloc3D(struct cudaPitchedPtr *, struct cudaExtent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMalloc3D_h == NULL)
	cudaMalloc3D_h = dlsym(tau_handle,"cudaMalloc3D"); 
    if (cudaMalloc3D_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMalloc3D_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMalloc3DArray(struct cudaArray ** a1, const struct cudaChannelFormatDesc * a2, struct cudaExtent a3) {
  checkinit();

  static cudaError_t (*cudaMalloc3DArray_h) (struct cudaArray **, const struct cudaChannelFormatDesc *, struct cudaExtent) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMalloc3DArray(struct cudaArray **, const struct cudaChannelFormatDesc *, struct cudaExtent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMalloc3DArray_h == NULL)
	cudaMalloc3DArray_h = dlsym(tau_handle,"cudaMalloc3DArray"); 
    if (cudaMalloc3DArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMalloc3DArray_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset3D(struct cudaPitchedPtr a1, int a2, struct cudaExtent a3) {
  checkinit();

  static cudaError_t (*cudaMemset3D_h) (struct cudaPitchedPtr, int, struct cudaExtent) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset3D(struct cudaPitchedPtr, int, struct cudaExtent) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset3D_h == NULL)
	cudaMemset3D_h = dlsym(tau_handle,"cudaMemset3D"); 
    if (cudaMemset3D_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset3D_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms * a1) {
  checkinit();

  static cudaError_t (*cudaMemcpy3D_h) (const struct cudaMemcpy3DParms *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy3D_h == NULL)
	cudaMemcpy3D_h = dlsym(tau_handle,"cudaMemcpy3D"); 
    if (cudaMemcpy3D_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy3D_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms * a1, cudaStream_t a2) {
  checkinit();

  static cudaError_t (*cudaMemcpy3DAsync_h) (const struct cudaMemcpy3DParms *, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy3DAsync_h == NULL)
	cudaMemcpy3DAsync_h = dlsym(tau_handle,"cudaMemcpy3DAsync"); 
    if (cudaMemcpy3DAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy3DAsync_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMalloc(void ** a1, size_t a2) {
  checkinit();

  static cudaError_t (*cudaMalloc_h) (void **, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMalloc(void **, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMalloc_h == NULL)
	cudaMalloc_h = dlsym(tau_handle,"cudaMalloc"); 
    if (cudaMalloc_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMalloc_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMallocHost(void ** a1, size_t a2) {
  checkinit();

  static cudaError_t (*cudaMallocHost_h) (void **, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMallocHost(void **, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMallocHost_h == NULL)
	cudaMallocHost_h = dlsym(tau_handle,"cudaMallocHost"); 
    if (cudaMallocHost_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMallocHost_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMallocPitch(void ** a1, size_t * a2, size_t a3, size_t a4) {
  checkinit();

  static cudaError_t (*cudaMallocPitch_h) (void **, size_t *, size_t, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMallocPitch_h == NULL)
	cudaMallocPitch_h = dlsym(tau_handle,"cudaMallocPitch"); 
    if (cudaMallocPitch_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMallocPitch_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMallocArray(struct cudaArray ** a1, const struct cudaChannelFormatDesc * a2, size_t a3, size_t a4) {
  checkinit();

  static cudaError_t (*cudaMallocArray_h) (struct cudaArray **, const struct cudaChannelFormatDesc *, size_t, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMallocArray(struct cudaArray **, const struct cudaChannelFormatDesc *, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMallocArray_h == NULL)
	cudaMallocArray_h = dlsym(tau_handle,"cudaMallocArray"); 
    if (cudaMallocArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMallocArray_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaFree(void * a1) {
  checkinit();

  static cudaError_t (*cudaFree_h) (void *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFree(void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFree_h == NULL)
	cudaFree_h = dlsym(tau_handle,"cudaFree"); 
    if (cudaFree_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFree_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaFreeHost(void * a1) {
  checkinit();

  static cudaError_t (*cudaFreeHost_h) (void *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFreeHost(void *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFreeHost_h == NULL)
	cudaFreeHost_h = dlsym(tau_handle,"cudaFreeHost"); 
    if (cudaFreeHost_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFreeHost_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaFreeArray(struct cudaArray * a1) {
  checkinit();

  static cudaError_t (*cudaFreeArray_h) (struct cudaArray *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaFreeArray(struct cudaArray *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaFreeArray_h == NULL)
	cudaFreeArray_h = dlsym(tau_handle,"cudaFreeArray"); 
    if (cudaFreeArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaFreeArray_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy(void * a1, const void * a2, size_t a3, enum cudaMemcpyKind a4) {
  checkinit();

  static cudaError_t (*cudaMemcpy_h) (void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy_h == NULL)
	cudaMemcpy_h = dlsym(tau_handle,"cudaMemcpy"); 
    if (cudaMemcpy_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToArray(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, enum cudaMemcpyKind a6) {
  checkinit();

  static cudaError_t (*cudaMemcpyToArray_h) (struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToArray(struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToArray_h == NULL)
	cudaMemcpyToArray_h = dlsym(tau_handle,"cudaMemcpyToArray"); 
    if (cudaMemcpyToArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyToArray_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromArray(void * a1, const struct cudaArray * a2, size_t a3, size_t a4, size_t a5, enum cudaMemcpyKind a6) {
  checkinit();

  static cudaError_t (*cudaMemcpyFromArray_h) (void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromArray(void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromArray_h == NULL)
	cudaMemcpyFromArray_h = dlsym(tau_handle,"cudaMemcpyFromArray"); 
    if (cudaMemcpyFromArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyFromArray_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyArrayToArray(struct cudaArray * a1, size_t a2, size_t a3, const struct cudaArray * a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8) {
  checkinit();

  static cudaError_t (*cudaMemcpyArrayToArray_h) (struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyArrayToArray(struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyArrayToArray_h == NULL)
	cudaMemcpyArrayToArray_h = dlsym(tau_handle,"cudaMemcpyArrayToArray"); 
    if (cudaMemcpyArrayToArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyArrayToArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2D(void * a1, size_t a2, const void * a3, size_t a4, size_t a5, size_t a6, enum cudaMemcpyKind a7) {
  checkinit();

  static cudaError_t (*cudaMemcpy2D_h) (void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2D_h == NULL)
	cudaMemcpy2D_h = dlsym(tau_handle,"cudaMemcpy2D"); 
    if (cudaMemcpy2D_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2D_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DToArray(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8) {
  checkinit();

  static cudaError_t (*cudaMemcpy2DToArray_h) (struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DToArray(struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DToArray_h == NULL)
	cudaMemcpy2DToArray_h = dlsym(tau_handle,"cudaMemcpy2DToArray"); 
    if (cudaMemcpy2DToArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DToArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DFromArray(void * a1, size_t a2, const struct cudaArray * a3, size_t a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8) {
  checkinit();

  static cudaError_t (*cudaMemcpy2DFromArray_h) (void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DFromArray(void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DFromArray_h == NULL)
	cudaMemcpy2DFromArray_h = dlsym(tau_handle,"cudaMemcpy2DFromArray"); 
    if (cudaMemcpy2DFromArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DFromArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray * a1, size_t a2, size_t a3, const struct cudaArray * a4, size_t a5, size_t a6, size_t a7, size_t a8, enum cudaMemcpyKind a9) {
  checkinit();

  static cudaError_t (*cudaMemcpy2DArrayToArray_h) (struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *, size_t, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DArrayToArray_h == NULL)
	cudaMemcpy2DArrayToArray_h = dlsym(tau_handle,"cudaMemcpy2DArrayToArray"); 
    if (cudaMemcpy2DArrayToArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DArrayToArray_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToSymbol(const char * a1, const void * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5) {
  checkinit();

  static cudaError_t (*cudaMemcpyToSymbol_h) (const char *, const void *, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToSymbol_h == NULL)
	cudaMemcpyToSymbol_h = dlsym(tau_handle,"cudaMemcpyToSymbol"); 
    if (cudaMemcpyToSymbol_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyToSymbol_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromSymbol(void * a1, const char * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5) {
  checkinit();

  static cudaError_t (*cudaMemcpyFromSymbol_h) (void *, const char *, size_t, size_t, enum cudaMemcpyKind) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t, enum cudaMemcpyKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromSymbol_h == NULL)
	cudaMemcpyFromSymbol_h = dlsym(tau_handle,"cudaMemcpyFromSymbol"); 
    if (cudaMemcpyFromSymbol_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyFromSymbol_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyAsync(void * a1, const void * a2, size_t a3, enum cudaMemcpyKind a4, cudaStream_t a5) {
  checkinit();

  static cudaError_t (*cudaMemcpyAsync_h) (void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyAsync(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyAsync_h == NULL)
	cudaMemcpyAsync_h = dlsym(tau_handle,"cudaMemcpyAsync"); 
    if (cudaMemcpyAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyAsync_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToArrayAsync(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, enum cudaMemcpyKind a6, cudaStream_t a7) {
  checkinit();

  static cudaError_t (*cudaMemcpyToArrayAsync_h) (struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *, size_t, size_t, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToArrayAsync_h == NULL)
	cudaMemcpyToArrayAsync_h = dlsym(tau_handle,"cudaMemcpyToArrayAsync"); 
    if (cudaMemcpyToArrayAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyToArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromArrayAsync(void * a1, const struct cudaArray * a2, size_t a3, size_t a4, size_t a5, enum cudaMemcpyKind a6, cudaStream_t a7) {
  checkinit();

  static cudaError_t (*cudaMemcpyFromArrayAsync_h) (void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromArrayAsync(void *, const struct cudaArray *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromArrayAsync_h == NULL)
	cudaMemcpyFromArrayAsync_h = dlsym(tau_handle,"cudaMemcpyFromArrayAsync"); 
    if (cudaMemcpyFromArrayAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyFromArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DAsync(void * a1, size_t a2, const void * a3, size_t a4, size_t a5, size_t a6, enum cudaMemcpyKind a7, cudaStream_t a8) {
  checkinit();

  static cudaError_t (*cudaMemcpy2DAsync_h) (void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DAsync_h == NULL)
	cudaMemcpy2DAsync_h = dlsym(tau_handle,"cudaMemcpy2DAsync"); 
    if (cudaMemcpy2DAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray * a1, size_t a2, size_t a3, const void * a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8, cudaStream_t a9) {
  checkinit();

  static cudaError_t (*cudaMemcpy2DToArrayAsync_h) (struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DToArrayAsync_h == NULL)
	cudaMemcpy2DToArrayAsync_h = dlsym(tau_handle,"cudaMemcpy2DToArrayAsync"); 
    if (cudaMemcpy2DToArrayAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DToArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpy2DFromArrayAsync(void * a1, size_t a2, const struct cudaArray * a3, size_t a4, size_t a5, size_t a6, size_t a7, enum cudaMemcpyKind a8, cudaStream_t a9) {
  checkinit();

  static cudaError_t (*cudaMemcpy2DFromArrayAsync_h) (void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const struct cudaArray *, size_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpy2DFromArrayAsync_h == NULL)
	cudaMemcpy2DFromArrayAsync_h = dlsym(tau_handle,"cudaMemcpy2DFromArrayAsync"); 
    if (cudaMemcpy2DFromArrayAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpy2DFromArrayAsync_h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyToSymbolAsync(const char * a1, const void * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5, cudaStream_t a6) {
  checkinit();

  static cudaError_t (*cudaMemcpyToSymbolAsync_h) (const char *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyToSymbolAsync_h == NULL)
	cudaMemcpyToSymbolAsync_h = dlsym(tau_handle,"cudaMemcpyToSymbolAsync"); 
    if (cudaMemcpyToSymbolAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyToSymbolAsync_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemcpyFromSymbolAsync(void * a1, const char * a2, size_t a3, size_t a4, enum cudaMemcpyKind a5, cudaStream_t a6) {
  checkinit();

  static cudaError_t (*cudaMemcpyFromSymbolAsync_h) (void *, const char *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemcpyFromSymbolAsync_h == NULL)
	cudaMemcpyFromSymbolAsync_h = dlsym(tau_handle,"cudaMemcpyFromSymbolAsync"); 
    if (cudaMemcpyFromSymbolAsync_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemcpyFromSymbolAsync_h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset(void * a1, int a2, size_t a3) {
  checkinit();

  static cudaError_t (*cudaMemset_h) (void *, int, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset(void *, int, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset_h == NULL)
	cudaMemset_h = dlsym(tau_handle,"cudaMemset"); 
    if (cudaMemset_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaMemset2D(void * a1, size_t a2, int a3, size_t a4, size_t a5) {
  checkinit();

  static cudaError_t (*cudaMemset2D_h) (void *, size_t, int, size_t, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaMemset2D_h == NULL)
	cudaMemset2D_h = dlsym(tau_handle,"cudaMemset2D"); 
    if (cudaMemset2D_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaMemset2D_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetSymbolAddress(void ** a1, const char * a2) {

  static cudaError_t (*cudaGetSymbolAddress_h) (void **, const char *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetSymbolAddress(void **, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetSymbolAddress_h == NULL)
	cudaGetSymbolAddress_h = dlsym(tau_handle,"cudaGetSymbolAddress"); 
    if (cudaGetSymbolAddress_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetSymbolAddress_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetSymbolSize(size_t * a1, const char * a2) {

  static cudaError_t (*cudaGetSymbolSize_h) (size_t *, const char *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetSymbolSize(size_t *, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetSymbolSize_h == NULL)
	cudaGetSymbolSize_h = dlsym(tau_handle,"cudaGetSymbolSize"); 
    if (cudaGetSymbolSize_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetSymbolSize_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetDeviceCount(int * a1) {

  static cudaError_t (*cudaGetDeviceCount_h) (int *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetDeviceCount(int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetDeviceCount_h == NULL)
	cudaGetDeviceCount_h = dlsym(tau_handle,"cudaGetDeviceCount"); 
    if (cudaGetDeviceCount_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetDeviceCount_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * a1, int a2) {

  static cudaError_t (*cudaGetDeviceProperties_h) (struct cudaDeviceProp *, int) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *, int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetDeviceProperties_h == NULL)
	cudaGetDeviceProperties_h = dlsym(tau_handle,"cudaGetDeviceProperties"); 
    if (cudaGetDeviceProperties_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetDeviceProperties_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaChooseDevice(int * a1, const struct cudaDeviceProp * a2) {

  static cudaError_t (*cudaChooseDevice_h) (int *, const struct cudaDeviceProp *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaChooseDevice(int *, const struct cudaDeviceProp *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaChooseDevice_h == NULL)
	cudaChooseDevice_h = dlsym(tau_handle,"cudaChooseDevice"); 
    if (cudaChooseDevice_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaChooseDevice_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDevice(int a1) {

  static cudaError_t (*cudaSetDevice_h) (int) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDevice(int) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDevice_h == NULL)
	cudaSetDevice_h = dlsym(tau_handle,"cudaSetDevice"); 
    if (cudaSetDevice_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetDevice(int * a1) {

  static cudaError_t (*cudaGetDevice_h) (int *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetDevice(int *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetDevice_h == NULL)
	cudaGetDevice_h = dlsym(tau_handle,"cudaGetDevice"); 
    if (cudaGetDevice_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaBindTexture(size_t * a1, const struct textureReference * a2, const void * a3, const struct cudaChannelFormatDesc * a4, size_t a5) {

  static cudaError_t (*cudaBindTexture_h) (size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaBindTexture(size_t *, const struct textureReference *, const void *, const struct cudaChannelFormatDesc *, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaBindTexture_h == NULL)
	cudaBindTexture_h = dlsym(tau_handle,"cudaBindTexture"); 
    if (cudaBindTexture_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaBindTexture_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaBindTextureToArray(const struct textureReference * a1, const struct cudaArray * a2, const struct cudaChannelFormatDesc * a3) {

  static cudaError_t (*cudaBindTextureToArray_h) (const struct textureReference *, const struct cudaArray *, const struct cudaChannelFormatDesc *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaBindTextureToArray(const struct textureReference *, const struct cudaArray *, const struct cudaChannelFormatDesc *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaBindTextureToArray_h == NULL)
	cudaBindTextureToArray_h = dlsym(tau_handle,"cudaBindTextureToArray"); 
    if (cudaBindTextureToArray_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaBindTextureToArray_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaUnbindTexture(const struct textureReference * a1) {

  static cudaError_t (*cudaUnbindTexture_h) (const struct textureReference *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaUnbindTexture(const struct textureReference *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaUnbindTexture_h == NULL)
	cudaUnbindTexture_h = dlsym(tau_handle,"cudaUnbindTexture"); 
    if (cudaUnbindTexture_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaUnbindTexture_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetTextureAlignmentOffset(size_t * a1, const struct textureReference * a2) {

  static cudaError_t (*cudaGetTextureAlignmentOffset_h) (size_t *, const struct textureReference *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetTextureAlignmentOffset(size_t *, const struct textureReference *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetTextureAlignmentOffset_h == NULL)
	cudaGetTextureAlignmentOffset_h = dlsym(tau_handle,"cudaGetTextureAlignmentOffset"); 
    if (cudaGetTextureAlignmentOffset_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetTextureAlignmentOffset_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetTextureReference(const struct textureReference ** a1, const char * a2) {

  static cudaError_t (*cudaGetTextureReference_h) (const struct textureReference **, const char *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetTextureReference(const struct textureReference **, const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetTextureReference_h == NULL)
	cudaGetTextureReference_h = dlsym(tau_handle,"cudaGetTextureReference"); 
    if (cudaGetTextureReference_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetTextureReference_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc * a1, const struct cudaArray * a2) {

  static cudaError_t (*cudaGetChannelDesc_h) (struct cudaChannelFormatDesc *, const struct cudaArray *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *, const struct cudaArray *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetChannelDesc_h == NULL)
	cudaGetChannelDesc_h = dlsym(tau_handle,"cudaGetChannelDesc"); 
    if (cudaGetChannelDesc_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetChannelDesc_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int a1, int a2, int a3, int a4, enum cudaChannelFormatKind a5) {
  checkinit();

  static struct cudaChannelFormatDesc (*cudaCreateChannelDesc_h) (int, int, int, int, enum cudaChannelFormatKind) = NULL;
  struct cudaChannelFormatDesc retval;
  TAU_PROFILE_TIMER(t,"struct cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, enum cudaChannelFormatKind) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaCreateChannelDesc_h == NULL)
	cudaCreateChannelDesc_h = dlsym(tau_handle,"cudaCreateChannelDesc"); 
    if (cudaCreateChannelDesc_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaCreateChannelDesc_h)( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaGetLastError() {

  static cudaError_t (*cudaGetLastError_h) () = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaGetLastError(void) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetLastError_h == NULL)
	cudaGetLastError_h = dlsym(tau_handle,"cudaGetLastError"); 
    if (cudaGetLastError_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetLastError_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

const char * cudaGetErrorString(cudaError_t a1) {

  static const char * (*cudaGetErrorString_h) (cudaError_t) = NULL;
  const char * retval;
  TAU_PROFILE_TIMER(t,"const char *cudaGetErrorString(cudaError_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaGetErrorString_h == NULL)
	cudaGetErrorString_h = dlsym(tau_handle,"cudaGetErrorString"); 
    if (cudaGetErrorString_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaGetErrorString_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaConfigureCall(dim3 a1, dim3 a2, size_t a3, cudaStream_t a4) {

  static cudaError_t (*cudaConfigureCall_h) (dim3, dim3, size_t, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaConfigureCall(dim3, dim3, size_t, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaConfigureCall_h == NULL)
	cudaConfigureCall_h = dlsym(tau_handle,"cudaConfigureCall"); 
    if (cudaConfigureCall_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaConfigureCall_h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetupArgument(const void * a1, size_t a2, size_t a3) {

  static cudaError_t (*cudaSetupArgument_h) (const void *, size_t, size_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetupArgument(const void *, size_t, size_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetupArgument_h == NULL)
	cudaSetupArgument_h = dlsym(tau_handle,"cudaSetupArgument"); 
    if (cudaSetupArgument_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetupArgument_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaLaunch(const char * a1) {

  static cudaError_t (*cudaLaunch_h) (const char *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaLaunch(const char *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaLaunch_h == NULL)
	cudaLaunch_h = dlsym(tau_handle,"cudaLaunch"); 
    if (cudaLaunch_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaLaunch_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamCreate(cudaStream_t * a1) {

  static cudaError_t (*cudaStreamCreate_h) (cudaStream_t *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamCreate(cudaStream_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamCreate_h == NULL)
	cudaStreamCreate_h = dlsym(tau_handle,"cudaStreamCreate"); 
    if (cudaStreamCreate_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamCreate_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamDestroy(cudaStream_t a1) {

  static cudaError_t (*cudaStreamDestroy_h) (cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamDestroy(cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamDestroy_h == NULL)
	cudaStreamDestroy_h = dlsym(tau_handle,"cudaStreamDestroy"); 
    if (cudaStreamDestroy_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamSynchronize(cudaStream_t a1) {

  static cudaError_t (*cudaStreamSynchronize_h) (cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamSynchronize(cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamSynchronize_h == NULL)
	cudaStreamSynchronize_h = dlsym(tau_handle,"cudaStreamSynchronize"); 
    if (cudaStreamSynchronize_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamSynchronize_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaStreamQuery(cudaStream_t a1) {

  static cudaError_t (*cudaStreamQuery_h) (cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaStreamQuery(cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaStreamQuery_h == NULL)
	cudaStreamQuery_h = dlsym(tau_handle,"cudaStreamQuery"); 
    if (cudaStreamQuery_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaStreamQuery_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventCreate(cudaEvent_t * a1) {

  static cudaError_t (*cudaEventCreate_h) (cudaEvent_t *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventCreate(cudaEvent_t *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventCreate_h == NULL)
	cudaEventCreate_h = dlsym(tau_handle,"cudaEventCreate"); 
    if (cudaEventCreate_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventCreate_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventRecord(cudaEvent_t a1, cudaStream_t a2) {

  static cudaError_t (*cudaEventRecord_h) (cudaEvent_t, cudaStream_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventRecord_h == NULL)
	cudaEventRecord_h = dlsym(tau_handle,"cudaEventRecord"); 
    if (cudaEventRecord_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventRecord_h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventQuery(cudaEvent_t a1) {

  static cudaError_t (*cudaEventQuery_h) (cudaEvent_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventQuery(cudaEvent_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventQuery_h == NULL)
	cudaEventQuery_h = dlsym(tau_handle,"cudaEventQuery"); 
    if (cudaEventQuery_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventQuery_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventSynchronize(cudaEvent_t a1) {

  static cudaError_t (*cudaEventSynchronize_h) (cudaEvent_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventSynchronize(cudaEvent_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventSynchronize_h == NULL)
	cudaEventSynchronize_h = dlsym(tau_handle,"cudaEventSynchronize"); 
    if (cudaEventSynchronize_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventSynchronize_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventDestroy(cudaEvent_t a1) {

  static cudaError_t (*cudaEventDestroy_h) (cudaEvent_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventDestroy(cudaEvent_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventDestroy_h == NULL)
	cudaEventDestroy_h = dlsym(tau_handle,"cudaEventDestroy"); 
    if (cudaEventDestroy_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventDestroy_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaEventElapsedTime(float * a1, cudaEvent_t a2, cudaEvent_t a3) {

  static cudaError_t (*cudaEventElapsedTime_h) (float *, cudaEvent_t, cudaEvent_t) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaEventElapsedTime_h == NULL)
	cudaEventElapsedTime_h = dlsym(tau_handle,"cudaEventElapsedTime"); 
    if (cudaEventElapsedTime_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaEventElapsedTime_h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDoubleForDevice(double * a1) {

  static cudaError_t (*cudaSetDoubleForDevice_h) (double *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDoubleForDevice(double *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDoubleForDevice_h == NULL)
	cudaSetDoubleForDevice_h = dlsym(tau_handle,"cudaSetDoubleForDevice"); 
    if (cudaSetDoubleForDevice_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDoubleForDevice_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaSetDoubleForHost(double * a1) {

  static cudaError_t (*cudaSetDoubleForHost_h) (double *) = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaSetDoubleForHost(double *) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaSetDoubleForHost_h == NULL)
	cudaSetDoubleForHost_h = dlsym(tau_handle,"cudaSetDoubleForHost"); 
    if (cudaSetDoubleForHost_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaSetDoubleForHost_h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaThreadExit() {

  static cudaError_t (*cudaThreadExit_h) () = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadExit(void) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadExit_h == NULL)
	cudaThreadExit_h = dlsym(tau_handle,"cudaThreadExit"); 
    if (cudaThreadExit_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadExit_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

cudaError_t cudaThreadSynchronize() {

  static cudaError_t (*cudaThreadSynchronize_h) () = NULL;
  cudaError_t retval;
  TAU_PROFILE_TIMER(t,"cudaError_t cudaThreadSynchronize(void) C", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    fprintf(stderr,"Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (cudaThreadSynchronize_h == NULL)
	cudaThreadSynchronize_h = dlsym(tau_handle,"cudaThreadSynchronize"); 
    if (cudaThreadSynchronize_h == NULL) {
      fprintf(stderr,"Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*cudaThreadSynchronize_h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

