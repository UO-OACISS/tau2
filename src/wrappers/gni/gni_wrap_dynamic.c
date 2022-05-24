#include <gni_pub.h>
#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>

#include <dlfcn.h>

static const char * tau_orig_libname = "libugni.so";
static void *tau_handle = NULL;


#ifndef TAU_GROUP_TAU_GNI
#define TAU_GROUP_TAU_GNI TAU_GET_PROFILE_GROUP("TAU_GNI")
#endif /* TAU_GROUP_TAU_GNI */ 

/**********************************************************
   GNI_DqueueInit
 **********************************************************/

extern gni_return_t  __wrap_GNI_DqueueInit(gni_nic_handle_t a1, gni_dqueue_in_attr_t * a2, gni_dqueue_out_attr_t * a3, gni_dqueue_handle_t * a4) ;
extern gni_return_t  GNI_DqueueInit(gni_nic_handle_t a1, gni_dqueue_in_attr_t * a2, gni_dqueue_out_attr_t * a3, gni_dqueue_handle_t * a4)  {

  static gni_return_t (*GNI_DqueueInit_h) (gni_nic_handle_t, gni_dqueue_in_attr_t *, gni_dqueue_out_attr_t *, gni_dqueue_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_DqueueInit(gni_nic_handle_t, gni_dqueue_in_attr_t *, gni_dqueue_out_attr_t *, gni_dqueue_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_DqueueInit_h == NULL)
      GNI_DqueueInit_h = dlsym(tau_handle,"GNI_DqueueInit"); 
    if (GNI_DqueueInit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_DqueueInit_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_DqueueFini
 **********************************************************/

extern gni_return_t  __wrap_GNI_DqueueFini(gni_dqueue_handle_t a1) ;
extern gni_return_t  GNI_DqueueFini(gni_dqueue_handle_t a1)  {

  static gni_return_t (*GNI_DqueueFini_h) (gni_dqueue_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_DqueueFini(gni_dqueue_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_DqueueFini_h == NULL)
      GNI_DqueueFini_h = dlsym(tau_handle,"GNI_DqueueFini"); 
    if (GNI_DqueueFini_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_DqueueFini_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_DqueueConnect
 **********************************************************/

extern gni_return_t  __wrap_GNI_DqueueConnect(gni_dqueue_handle_t a1, gni_dqueue_out_attr_t * a2, uint32_t a3) ;
extern gni_return_t  GNI_DqueueConnect(gni_dqueue_handle_t a1, gni_dqueue_out_attr_t * a2, uint32_t a3)  {

  static gni_return_t (*GNI_DqueueConnect_h) (gni_dqueue_handle_t, gni_dqueue_out_attr_t *, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_DqueueConnect(gni_dqueue_handle_t, gni_dqueue_out_attr_t *, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_DqueueConnect_h == NULL)
      GNI_DqueueConnect_h = dlsym(tau_handle,"GNI_DqueueConnect"); 
    if (GNI_DqueueConnect_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_DqueueConnect_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_DqueuePut
 **********************************************************/

extern gni_return_t  __wrap_GNI_DqueuePut(gni_dqueue_handle_t a1, void * a2, uint32_t a3, uint32_t a4, uint32_t a5) ;
extern gni_return_t  GNI_DqueuePut(gni_dqueue_handle_t a1, void * a2, uint32_t a3, uint32_t a4, uint32_t a5)  {

  static gni_return_t (*GNI_DqueuePut_h) (gni_dqueue_handle_t, void *, uint32_t, uint32_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_DqueuePut(gni_dqueue_handle_t, void *, uint32_t, uint32_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_DqueuePut_h == NULL)
      GNI_DqueuePut_h = dlsym(tau_handle,"GNI_DqueuePut"); 
    if (GNI_DqueuePut_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_DqueuePut_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_DqueueProgress
 **********************************************************/

extern gni_return_t  __wrap_GNI_DqueueProgress(gni_dqueue_handle_t a1) ;
extern gni_return_t  GNI_DqueueProgress(gni_dqueue_handle_t a1)  {

  static gni_return_t (*GNI_DqueueProgress_h) (gni_dqueue_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_DqueueProgress(gni_dqueue_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_DqueueProgress_h == NULL)
      GNI_DqueueProgress_h = dlsym(tau_handle,"GNI_DqueueProgress"); 
    if (GNI_DqueueProgress_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_DqueueProgress_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CdmCreate
 **********************************************************/

extern gni_return_t  __wrap_GNI_CdmCreate(uint32_t a1, uint8_t a2, uint32_t a3, uint32_t a4, gni_cdm_handle_t * a5) ;
extern gni_return_t  GNI_CdmCreate(uint32_t a1, uint8_t a2, uint32_t a3, uint32_t a4, gni_cdm_handle_t * a5)  {

  // If we found a node ID already from PMI, use it; otherwise, use the CdmCreate inst_id
  if(Tau_get_node() < 0) {
    Tau_set_node(a1); // inst_id
  }

  static gni_return_t (*GNI_CdmCreate_h) (uint32_t, uint8_t, uint32_t, uint32_t, gni_cdm_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CdmCreate(uint32_t, uint8_t, uint32_t, uint32_t, gni_cdm_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CdmCreate_h == NULL)
      GNI_CdmCreate_h = dlsym(tau_handle,"GNI_CdmCreate"); 
    if (GNI_CdmCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CdmCreate_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CdmDestroy
 **********************************************************/

extern gni_return_t  __wrap_GNI_CdmDestroy(gni_cdm_handle_t a1) ;
extern gni_return_t  GNI_CdmDestroy(gni_cdm_handle_t a1)  {

  static gni_return_t (*GNI_CdmDestroy_h) (gni_cdm_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CdmDestroy(gni_cdm_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CdmDestroy_h == NULL)
      GNI_CdmDestroy_h = dlsym(tau_handle,"GNI_CdmDestroy"); 
    if (GNI_CdmDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CdmDestroy_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CdmHold
 **********************************************************/

extern gni_return_t  __wrap_GNI_CdmHold(uint8_t a1, uint32_t a2, int * a3) ;
extern gni_return_t  GNI_CdmHold(uint8_t a1, uint32_t a2, int * a3)  {

  static gni_return_t (*GNI_CdmHold_h) (uint8_t, uint32_t, int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CdmHold(uint8_t, uint32_t, int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CdmHold_h == NULL)
      GNI_CdmHold_h = dlsym(tau_handle,"GNI_CdmHold"); 
    if (GNI_CdmHold_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CdmHold_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CdmRelease
 **********************************************************/

extern gni_return_t  __wrap_GNI_CdmRelease(int a1) ;
extern gni_return_t  GNI_CdmRelease(int a1)  {

  static gni_return_t (*GNI_CdmRelease_h) (int) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CdmRelease(int) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CdmRelease_h == NULL)
      GNI_CdmRelease_h = dlsym(tau_handle,"GNI_CdmRelease"); 
    if (GNI_CdmRelease_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CdmRelease_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CdmGetNicAddress
 **********************************************************/

extern gni_return_t  __wrap_GNI_CdmGetNicAddress(uint32_t a1, uint32_t * a2, uint32_t * a3) ;
extern gni_return_t  GNI_CdmGetNicAddress(uint32_t a1, uint32_t * a2, uint32_t * a3)  {

  static gni_return_t (*GNI_CdmGetNicAddress_h) (uint32_t, uint32_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CdmGetNicAddress(uint32_t, uint32_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CdmGetNicAddress_h == NULL)
      GNI_CdmGetNicAddress_h = dlsym(tau_handle,"GNI_CdmGetNicAddress"); 
    if (GNI_CdmGetNicAddress_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CdmGetNicAddress_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CdmAttach
 **********************************************************/

extern gni_return_t  __wrap_GNI_CdmAttach(gni_cdm_handle_t a1, uint32_t a2, uint32_t * a3, gni_nic_handle_t * a4) ;
extern gni_return_t  GNI_CdmAttach(gni_cdm_handle_t a1, uint32_t a2, uint32_t * a3, gni_nic_handle_t * a4)  {

  static gni_return_t (*GNI_CdmAttach_h) (gni_cdm_handle_t, uint32_t, uint32_t *, gni_nic_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CdmAttach(gni_cdm_handle_t, uint32_t, uint32_t *, gni_nic_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CdmAttach_h == NULL)
      GNI_CdmAttach_h = dlsym(tau_handle,"GNI_CdmAttach"); 
    if (GNI_CdmAttach_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CdmAttach_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SuspendJob
 **********************************************************/

extern gni_return_t  __wrap_GNI_SuspendJob(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4, uint32_t a5) ;
extern gni_return_t  GNI_SuspendJob(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4, uint32_t a5)  {

  static gni_return_t (*GNI_SuspendJob_h) (uint32_t, uint64_t, uint8_t, uint32_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SuspendJob(uint32_t, uint64_t, uint8_t, uint32_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SuspendJob_h == NULL)
      GNI_SuspendJob_h = dlsym(tau_handle,"GNI_SuspendJob"); 
    if (GNI_SuspendJob_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SuspendJob_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ResumeJob
 **********************************************************/

extern gni_return_t  __wrap_GNI_ResumeJob(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4) ;
extern gni_return_t  GNI_ResumeJob(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4)  {

  static gni_return_t (*GNI_ResumeJob_h) (uint32_t, uint64_t, uint8_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ResumeJob(uint32_t, uint64_t, uint8_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ResumeJob_h == NULL)
      GNI_ResumeJob_h = dlsym(tau_handle,"GNI_ResumeJob"); 
    if (GNI_ResumeJob_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ResumeJob_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ConfigureNTT
 **********************************************************/

extern gni_return_t  __wrap_GNI_ConfigureNTT(int a1, gni_ntt_descriptor_t * a2, uint32_t * a3) ;
extern gni_return_t  GNI_ConfigureNTT(int a1, gni_ntt_descriptor_t * a2, uint32_t * a3)  {

  static gni_return_t (*GNI_ConfigureNTT_h) (int, gni_ntt_descriptor_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ConfigureNTT(int, gni_ntt_descriptor_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ConfigureNTT_h == NULL)
      GNI_ConfigureNTT_h = dlsym(tau_handle,"GNI_ConfigureNTT"); 
    if (GNI_ConfigureNTT_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ConfigureNTT_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ConfigureJob
 **********************************************************/

extern gni_return_t  __wrap_GNI_ConfigureJob(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4, gni_job_limits_t * a5) ;
extern gni_return_t  GNI_ConfigureJob(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4, gni_job_limits_t * a5)  {

  static gni_return_t (*GNI_ConfigureJob_h) (uint32_t, uint64_t, uint8_t, uint32_t, gni_job_limits_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ConfigureJob(uint32_t, uint64_t, uint8_t, uint32_t, gni_job_limits_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ConfigureJob_h == NULL)
      GNI_ConfigureJob_h = dlsym(tau_handle,"GNI_ConfigureJob"); 
    if (GNI_ConfigureJob_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ConfigureJob_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ConfigureJobFd
 **********************************************************/

extern gni_return_t  __wrap_GNI_ConfigureJobFd(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4, gni_job_limits_t * a5, int * a6) ;
extern gni_return_t  GNI_ConfigureJobFd(uint32_t a1, uint64_t a2, uint8_t a3, uint32_t a4, gni_job_limits_t * a5, int * a6)  {

  static gni_return_t (*GNI_ConfigureJobFd_h) (uint32_t, uint64_t, uint8_t, uint32_t, gni_job_limits_t *, int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ConfigureJobFd(uint32_t, uint64_t, uint8_t, uint32_t, gni_job_limits_t *, int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ConfigureJobFd_h == NULL)
      GNI_ConfigureJobFd_h = dlsym(tau_handle,"GNI_ConfigureJobFd"); 
    if (GNI_ConfigureJobFd_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ConfigureJobFd_h) ( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ConfigureNTTandJob
 **********************************************************/

extern gni_return_t  __wrap_GNI_ConfigureNTTandJob(int a1, uint64_t a2, uint8_t a3, uint32_t a4, gni_job_limits_t * a5, gni_ntt_descriptor_t * a6, uint32_t * a7) ;
extern gni_return_t  GNI_ConfigureNTTandJob(int a1, uint64_t a2, uint8_t a3, uint32_t a4, gni_job_limits_t * a5, gni_ntt_descriptor_t * a6, uint32_t * a7)  {

  static gni_return_t (*GNI_ConfigureNTTandJob_h) (int, uint64_t, uint8_t, uint32_t, gni_job_limits_t *, gni_ntt_descriptor_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ConfigureNTTandJob(int, uint64_t, uint8_t, uint32_t, gni_job_limits_t *, gni_ntt_descriptor_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ConfigureNTTandJob_h == NULL)
      GNI_ConfigureNTTandJob_h = dlsym(tau_handle,"GNI_ConfigureNTTandJob"); 
    if (GNI_ConfigureNTTandJob_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ConfigureNTTandJob_h) ( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetCapabilities
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetCapabilities(gni_revision_info_t * a1) ;
extern gni_return_t  GNI_GetCapabilities(gni_revision_info_t * a1)  {

  static gni_return_t (*GNI_GetCapabilities_h) (gni_revision_info_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetCapabilities(gni_revision_info_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetCapabilities_h == NULL)
      GNI_GetCapabilities_h = dlsym(tau_handle,"GNI_GetCapabilities"); 
    if (GNI_GetCapabilities_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetCapabilities_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ValidateCapabilities
 **********************************************************/

extern gni_return_t  __wrap_GNI_ValidateCapabilities(gni_revision_info_t a1, gni_revision_info_t a2) ;
extern gni_return_t  GNI_ValidateCapabilities(gni_revision_info_t a1, gni_revision_info_t a2)  {

  static gni_return_t (*GNI_ValidateCapabilities_h) (gni_revision_info_t, gni_revision_info_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ValidateCapabilities(gni_revision_info_t, gni_revision_info_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ValidateCapabilities_h == NULL)
      GNI_ValidateCapabilities_h = dlsym(tau_handle,"GNI_ValidateCapabilities"); 
    if (GNI_ValidateCapabilities_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ValidateCapabilities_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpCreate
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpCreate(gni_nic_handle_t a1, gni_cq_handle_t a2, gni_ep_handle_t * a3) ;
extern gni_return_t  GNI_EpCreate(gni_nic_handle_t a1, gni_cq_handle_t a2, gni_ep_handle_t * a3)  {

  static gni_return_t (*GNI_EpCreate_h) (gni_nic_handle_t, gni_cq_handle_t, gni_ep_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpCreate(gni_nic_handle_t, gni_cq_handle_t, gni_ep_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpCreate_h == NULL)
      GNI_EpCreate_h = dlsym(tau_handle,"GNI_EpCreate"); 
    if (GNI_EpCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpCreate_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpSetEventData
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpSetEventData(gni_ep_handle_t a1, uint32_t a2, uint32_t a3) ;
extern gni_return_t  GNI_EpSetEventData(gni_ep_handle_t a1, uint32_t a2, uint32_t a3)  {

  static gni_return_t (*GNI_EpSetEventData_h) (gni_ep_handle_t, uint32_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpSetEventData(gni_ep_handle_t, uint32_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpSetEventData_h == NULL)
      GNI_EpSetEventData_h = dlsym(tau_handle,"GNI_EpSetEventData"); 
    if (GNI_EpSetEventData_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpSetEventData_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpBind
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpBind(gni_ep_handle_t a1, uint32_t a2, uint32_t a3) ;
extern gni_return_t  GNI_EpBind(gni_ep_handle_t a1, uint32_t a2, uint32_t a3)  {

  static gni_return_t (*GNI_EpBind_h) (gni_ep_handle_t, uint32_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpBind(gni_ep_handle_t, uint32_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpBind_h == NULL)
      GNI_EpBind_h = dlsym(tau_handle,"GNI_EpBind"); 
    if (GNI_EpBind_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpBind_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpUnbind
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpUnbind(gni_ep_handle_t a1) ;
extern gni_return_t  GNI_EpUnbind(gni_ep_handle_t a1)  {

  static gni_return_t (*GNI_EpUnbind_h) (gni_ep_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpUnbind(gni_ep_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpUnbind_h == NULL)
      GNI_EpUnbind_h = dlsym(tau_handle,"GNI_EpUnbind"); 
    if (GNI_EpUnbind_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpUnbind_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpIdle
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpIdle(gni_ep_handle_t a1) ;
extern gni_return_t  GNI_EpIdle(gni_ep_handle_t a1)  {

  static gni_return_t (*GNI_EpIdle_h) (gni_ep_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpIdle(gni_ep_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpIdle_h == NULL)
      GNI_EpIdle_h = dlsym(tau_handle,"GNI_EpIdle"); 
    if (GNI_EpIdle_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpIdle_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpDestroy
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpDestroy(gni_ep_handle_t a1) ;
extern gni_return_t  GNI_EpDestroy(gni_ep_handle_t a1)  {

  static gni_return_t (*GNI_EpDestroy_h) (gni_ep_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpDestroy(gni_ep_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpDestroy_h == NULL)
      GNI_EpDestroy_h = dlsym(tau_handle,"GNI_EpDestroy"); 
    if (GNI_EpDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpDestroy_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostData
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostData(gni_ep_handle_t a1, void * a2, uint16_t a3, void * a4, uint16_t a5) ;
extern gni_return_t  GNI_EpPostData(gni_ep_handle_t a1, void * a2, uint16_t a3, void * a4, uint16_t a5)  {

  static gni_return_t (*GNI_EpPostData_h) (gni_ep_handle_t, void *, uint16_t, void *, uint16_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostData(gni_ep_handle_t, void *, uint16_t, void *, uint16_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostData_h == NULL)
      GNI_EpPostData_h = dlsym(tau_handle,"GNI_EpPostData"); 
    if (GNI_EpPostData_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostData_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataWId
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataWId(gni_ep_handle_t a1, void * a2, uint16_t a3, void * a4, uint16_t a5, uint64_t a6) ;
extern gni_return_t  GNI_EpPostDataWId(gni_ep_handle_t a1, void * a2, uint16_t a3, void * a4, uint16_t a5, uint64_t a6)  {

  static gni_return_t (*GNI_EpPostDataWId_h) (gni_ep_handle_t, void *, uint16_t, void *, uint16_t, uint64_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataWId(gni_ep_handle_t, void *, uint16_t, void *, uint16_t, uint64_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataWId_h == NULL)
      GNI_EpPostDataWId_h = dlsym(tau_handle,"GNI_EpPostDataWId"); 
    if (GNI_EpPostDataWId_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataWId_h) ( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataTest
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataTest(gni_ep_handle_t a1, gni_post_state_t * a2, uint32_t * a3, uint32_t * a4) ;
extern gni_return_t  GNI_EpPostDataTest(gni_ep_handle_t a1, gni_post_state_t * a2, uint32_t * a3, uint32_t * a4)  {

  static gni_return_t (*GNI_EpPostDataTest_h) (gni_ep_handle_t, gni_post_state_t *, uint32_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataTest(gni_ep_handle_t, gni_post_state_t *, uint32_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataTest_h == NULL)
      GNI_EpPostDataTest_h = dlsym(tau_handle,"GNI_EpPostDataTest"); 
    if (GNI_EpPostDataTest_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataTest_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataTestById
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataTestById(gni_ep_handle_t a1, uint64_t a2, gni_post_state_t * a3, uint32_t * a4, uint32_t * a5) ;
extern gni_return_t  GNI_EpPostDataTestById(gni_ep_handle_t a1, uint64_t a2, gni_post_state_t * a3, uint32_t * a4, uint32_t * a5)  {

  static gni_return_t (*GNI_EpPostDataTestById_h) (gni_ep_handle_t, uint64_t, gni_post_state_t *, uint32_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataTestById(gni_ep_handle_t, uint64_t, gni_post_state_t *, uint32_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataTestById_h == NULL)
      GNI_EpPostDataTestById_h = dlsym(tau_handle,"GNI_EpPostDataTestById"); 
    if (GNI_EpPostDataTestById_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataTestById_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataWait
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataWait(gni_ep_handle_t a1, uint32_t a2, gni_post_state_t * a3, uint32_t * a4, uint32_t * a5) ;
extern gni_return_t  GNI_EpPostDataWait(gni_ep_handle_t a1, uint32_t a2, gni_post_state_t * a3, uint32_t * a4, uint32_t * a5)  {

  static gni_return_t (*GNI_EpPostDataWait_h) (gni_ep_handle_t, uint32_t, gni_post_state_t *, uint32_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataWait(gni_ep_handle_t, uint32_t, gni_post_state_t *, uint32_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataWait_h == NULL)
      GNI_EpPostDataWait_h = dlsym(tau_handle,"GNI_EpPostDataWait"); 
    if (GNI_EpPostDataWait_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataWait_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataWaitById
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataWaitById(gni_ep_handle_t a1, uint64_t a2, uint32_t a3, gni_post_state_t * a4, uint32_t * a5, uint32_t * a6) ;
extern gni_return_t  GNI_EpPostDataWaitById(gni_ep_handle_t a1, uint64_t a2, uint32_t a3, gni_post_state_t * a4, uint32_t * a5, uint32_t * a6)  {

  static gni_return_t (*GNI_EpPostDataWaitById_h) (gni_ep_handle_t, uint64_t, uint32_t, gni_post_state_t *, uint32_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataWaitById(gni_ep_handle_t, uint64_t, uint32_t, gni_post_state_t *, uint32_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataWaitById_h == NULL)
      GNI_EpPostDataWaitById_h = dlsym(tau_handle,"GNI_EpPostDataWaitById"); 
    if (GNI_EpPostDataWaitById_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataWaitById_h) ( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_PostDataProbe
 **********************************************************/

extern gni_return_t  __wrap_GNI_PostDataProbe(gni_nic_handle_t a1, uint32_t * a2, uint32_t * a3) ;
extern gni_return_t  GNI_PostDataProbe(gni_nic_handle_t a1, uint32_t * a2, uint32_t * a3)  {

  static gni_return_t (*GNI_PostDataProbe_h) (gni_nic_handle_t, uint32_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_PostDataProbe(gni_nic_handle_t, uint32_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_PostDataProbe_h == NULL)
      GNI_PostDataProbe_h = dlsym(tau_handle,"GNI_PostDataProbe"); 
    if (GNI_PostDataProbe_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_PostDataProbe_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_PostDataProbeById
 **********************************************************/

extern gni_return_t  __wrap_GNI_PostDataProbeById(gni_nic_handle_t a1, uint64_t * a2) ;
extern gni_return_t  GNI_PostDataProbeById(gni_nic_handle_t a1, uint64_t * a2)  {

  static gni_return_t (*GNI_PostDataProbeById_h) (gni_nic_handle_t, uint64_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_PostDataProbeById(gni_nic_handle_t, uint64_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_PostDataProbeById_h == NULL)
      GNI_PostDataProbeById_h = dlsym(tau_handle,"GNI_PostDataProbeById"); 
    if (GNI_PostDataProbeById_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_PostDataProbeById_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_PostdataProbeWaitById
 **********************************************************/

extern gni_return_t  __wrap_GNI_PostdataProbeWaitById(gni_nic_handle_t a1, uint32_t a2, uint64_t * a3) ;
extern gni_return_t  GNI_PostdataProbeWaitById(gni_nic_handle_t a1, uint32_t a2, uint64_t * a3)  {

  static gni_return_t (*GNI_PostdataProbeWaitById_h) (gni_nic_handle_t, uint32_t, uint64_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_PostdataProbeWaitById(gni_nic_handle_t, uint32_t, uint64_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_PostdataProbeWaitById_h == NULL)
      GNI_PostdataProbeWaitById_h = dlsym(tau_handle,"GNI_PostdataProbeWaitById"); 
    if (GNI_PostdataProbeWaitById_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_PostdataProbeWaitById_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataCancel
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataCancel(gni_ep_handle_t a1) ;
extern gni_return_t  GNI_EpPostDataCancel(gni_ep_handle_t a1)  {

  static gni_return_t (*GNI_EpPostDataCancel_h) (gni_ep_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataCancel(gni_ep_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataCancel_h == NULL)
      GNI_EpPostDataCancel_h = dlsym(tau_handle,"GNI_EpPostDataCancel"); 
    if (GNI_EpPostDataCancel_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataCancel_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpPostDataCancelById
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpPostDataCancelById(gni_ep_handle_t a1, uint64_t a2) ;
extern gni_return_t  GNI_EpPostDataCancelById(gni_ep_handle_t a1, uint64_t a2)  {

  static gni_return_t (*GNI_EpPostDataCancelById_h) (gni_ep_handle_t, uint64_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpPostDataCancelById(gni_ep_handle_t, uint64_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpPostDataCancelById_h == NULL)
      GNI_EpPostDataCancelById_h = dlsym(tau_handle,"GNI_EpPostDataCancelById"); 
    if (GNI_EpPostDataCancelById_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpPostDataCancelById_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MemRegister
 **********************************************************/

extern gni_return_t  __wrap_GNI_MemRegister(gni_nic_handle_t a1, uint64_t a2, uint64_t a3, gni_cq_handle_t a4, uint32_t a5, uint32_t a6, gni_mem_handle_t * a7) ;
extern gni_return_t  GNI_MemRegister(gni_nic_handle_t a1, uint64_t a2, uint64_t a3, gni_cq_handle_t a4, uint32_t a5, uint32_t a6, gni_mem_handle_t * a7)  {

  static gni_return_t (*GNI_MemRegister_h) (gni_nic_handle_t, uint64_t, uint64_t, gni_cq_handle_t, uint32_t, uint32_t, gni_mem_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MemRegister(gni_nic_handle_t, uint64_t, uint64_t, gni_cq_handle_t, uint32_t, uint32_t, gni_mem_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MemRegister_h == NULL)
      GNI_MemRegister_h = dlsym(tau_handle,"GNI_MemRegister"); 
    if (GNI_MemRegister_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MemRegister_h) ( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MemRegisterSegments
 **********************************************************/

extern gni_return_t  __wrap_GNI_MemRegisterSegments(gni_nic_handle_t a1, gni_mem_segment_t * a2, uint32_t a3, gni_cq_handle_t a4, uint32_t a5, uint32_t a6, gni_mem_handle_t * a7) ;
extern gni_return_t  GNI_MemRegisterSegments(gni_nic_handle_t a1, gni_mem_segment_t * a2, uint32_t a3, gni_cq_handle_t a4, uint32_t a5, uint32_t a6, gni_mem_handle_t * a7)  {

  static gni_return_t (*GNI_MemRegisterSegments_h) (gni_nic_handle_t, gni_mem_segment_t *, uint32_t, gni_cq_handle_t, uint32_t, uint32_t, gni_mem_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MemRegisterSegments(gni_nic_handle_t, gni_mem_segment_t *, uint32_t, gni_cq_handle_t, uint32_t, uint32_t, gni_mem_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MemRegisterSegments_h == NULL)
      GNI_MemRegisterSegments_h = dlsym(tau_handle,"GNI_MemRegisterSegments"); 
    if (GNI_MemRegisterSegments_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MemRegisterSegments_h) ( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SetMddResources
 **********************************************************/

extern gni_return_t  __wrap_GNI_SetMddResources(gni_nic_handle_t a1, uint32_t a2) ;
extern gni_return_t  GNI_SetMddResources(gni_nic_handle_t a1, uint32_t a2)  {

  static gni_return_t (*GNI_SetMddResources_h) (gni_nic_handle_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SetMddResources(gni_nic_handle_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SetMddResources_h == NULL)
      GNI_SetMddResources_h = dlsym(tau_handle,"GNI_SetMddResources"); 
    if (GNI_SetMddResources_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SetMddResources_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MemDeregister
 **********************************************************/

extern gni_return_t  __wrap_GNI_MemDeregister(gni_nic_handle_t a1, gni_mem_handle_t * a2) ;
extern gni_return_t  GNI_MemDeregister(gni_nic_handle_t a1, gni_mem_handle_t * a2)  {

  static gni_return_t (*GNI_MemDeregister_h) (gni_nic_handle_t, gni_mem_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MemDeregister(gni_nic_handle_t, gni_mem_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MemDeregister_h == NULL)
      GNI_MemDeregister_h = dlsym(tau_handle,"GNI_MemDeregister"); 
    if (GNI_MemDeregister_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MemDeregister_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MemHndlQueryAttr
 **********************************************************/

extern gni_return_t  __wrap_GNI_MemHndlQueryAttr(gni_mem_handle_t * a1, gni_mem_handle_attr_t a2, int * a3) ;
extern gni_return_t  GNI_MemHndlQueryAttr(gni_mem_handle_t * a1, gni_mem_handle_attr_t a2, int * a3)  {

  static gni_return_t (*GNI_MemHndlQueryAttr_h) (gni_mem_handle_t *, gni_mem_handle_attr_t, int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MemHndlQueryAttr(gni_mem_handle_t *, gni_mem_handle_attr_t, int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MemHndlQueryAttr_h == NULL)
      GNI_MemHndlQueryAttr_h = dlsym(tau_handle,"GNI_MemHndlQueryAttr"); 
    if (GNI_MemHndlQueryAttr_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MemHndlQueryAttr_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_RebuildMemHndl
 **********************************************************/

extern gni_return_t  __wrap_GNI_RebuildMemHndl(gni_mem_handle_t * a1, uint32_t a2, gni_mem_handle_t * a3) ;
extern gni_return_t  GNI_RebuildMemHndl(gni_mem_handle_t * a1, uint32_t a2, gni_mem_handle_t * a3)  {

  static gni_return_t (*GNI_RebuildMemHndl_h) (gni_mem_handle_t *, uint32_t, gni_mem_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_RebuildMemHndl(gni_mem_handle_t *, uint32_t, gni_mem_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_RebuildMemHndl_h == NULL)
      GNI_RebuildMemHndl_h = dlsym(tau_handle,"GNI_RebuildMemHndl"); 
    if (GNI_RebuildMemHndl_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_RebuildMemHndl_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MemQueryHndls
 **********************************************************/

extern gni_return_t  __wrap_GNI_MemQueryHndls(gni_nic_handle_t a1, int a2, gni_mem_handle_t * a3, uint64_t * a4, uint64_t * a5) ;
extern gni_return_t  GNI_MemQueryHndls(gni_nic_handle_t a1, int a2, gni_mem_handle_t * a3, uint64_t * a4, uint64_t * a5)  {

  static gni_return_t (*GNI_MemQueryHndls_h) (gni_nic_handle_t, int, gni_mem_handle_t *, uint64_t *, uint64_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MemQueryHndls(gni_nic_handle_t, int, gni_mem_handle_t *, uint64_t *, uint64_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MemQueryHndls_h == NULL)
      GNI_MemQueryHndls_h = dlsym(tau_handle,"GNI_MemQueryHndls"); 
    if (GNI_MemQueryHndls_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MemQueryHndls_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqCreate
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqCreate(gni_nic_handle_t a1, uint32_t a2, uint32_t a3, gni_cq_mode_t a4, void (*a5)(gni_cq_entry_t *, void *), void * a6, gni_cq_handle_t * a7) ;
extern gni_return_t  GNI_CqCreate(gni_nic_handle_t a1, uint32_t a2, uint32_t a3, gni_cq_mode_t a4, void (*a5)(gni_cq_entry_t *, void *), void * a6, gni_cq_handle_t * a7)  {

  static gni_return_t (*GNI_CqCreate_h) (gni_nic_handle_t, uint32_t, uint32_t, gni_cq_mode_t, void (*)(gni_cq_entry_t *, void *), void *, gni_cq_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqCreate(gni_nic_handle_t, uint32_t, uint32_t, gni_cq_mode_t, void (*)(gni_cq_entry_t *, void *), void *, gni_cq_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqCreate_h == NULL)
      GNI_CqCreate_h = dlsym(tau_handle,"GNI_CqCreate"); 
    if (GNI_CqCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqCreate_h) ( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqDestroy
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqDestroy(gni_cq_handle_t a1) ;
extern gni_return_t  GNI_CqDestroy(gni_cq_handle_t a1)  {

  static gni_return_t (*GNI_CqDestroy_h) (gni_cq_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqDestroy(gni_cq_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqDestroy_h == NULL)
      GNI_CqDestroy_h = dlsym(tau_handle,"GNI_CqDestroy"); 
    if (GNI_CqDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqDestroy_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_PostRdma
 **********************************************************/

extern gni_return_t  __wrap_GNI_PostRdma(gni_ep_handle_t a1, gni_post_descriptor_t * a2) ;
extern gni_return_t  GNI_PostRdma(gni_ep_handle_t a1, gni_post_descriptor_t * a2)  {

  static gni_return_t (*GNI_PostRdma_h) (gni_ep_handle_t, gni_post_descriptor_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_PostRdma(gni_ep_handle_t, gni_post_descriptor_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_PostRdma_h == NULL)
      GNI_PostRdma_h = dlsym(tau_handle,"GNI_PostRdma"); 
    if (GNI_PostRdma_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_PostRdma_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_PostFma
 **********************************************************/

extern gni_return_t  __wrap_GNI_PostFma(gni_ep_handle_t a1, gni_post_descriptor_t * a2) ;
extern gni_return_t  GNI_PostFma(gni_ep_handle_t a1, gni_post_descriptor_t * a2)  {

  static gni_return_t (*GNI_PostFma_h) (gni_ep_handle_t, gni_post_descriptor_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_PostFma(gni_ep_handle_t, gni_post_descriptor_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_PostFma_h == NULL)
      GNI_PostFma_h = dlsym(tau_handle,"GNI_PostFma"); 
    if (GNI_PostFma_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_PostFma_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CtPostFma
 **********************************************************/

extern gni_return_t  __wrap_GNI_CtPostFma(gni_ep_handle_t a1, gni_post_descriptor_t * a2) ;
extern gni_return_t  GNI_CtPostFma(gni_ep_handle_t a1, gni_post_descriptor_t * a2)  {

  static gni_return_t (*GNI_CtPostFma_h) (gni_ep_handle_t, gni_post_descriptor_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CtPostFma(gni_ep_handle_t, gni_post_descriptor_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CtPostFma_h == NULL)
      GNI_CtPostFma_h = dlsym(tau_handle,"GNI_CtPostFma"); 
    if (GNI_CtPostFma_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CtPostFma_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_PostCqWrite
 **********************************************************/

extern gni_return_t  __wrap_GNI_PostCqWrite(gni_ep_handle_t a1, gni_post_descriptor_t * a2) ;
extern gni_return_t  GNI_PostCqWrite(gni_ep_handle_t a1, gni_post_descriptor_t * a2)  {

  static gni_return_t (*GNI_PostCqWrite_h) (gni_ep_handle_t, gni_post_descriptor_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_PostCqWrite(gni_ep_handle_t, gni_post_descriptor_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_PostCqWrite_h == NULL)
      GNI_PostCqWrite_h = dlsym(tau_handle,"GNI_PostCqWrite"); 
    if (GNI_PostCqWrite_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_PostCqWrite_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CtPostCqWrite
 **********************************************************/

extern gni_return_t  __wrap_GNI_CtPostCqWrite(gni_ep_handle_t a1, gni_post_descriptor_t * a2) ;
extern gni_return_t  GNI_CtPostCqWrite(gni_ep_handle_t a1, gni_post_descriptor_t * a2)  {

  static gni_return_t (*GNI_CtPostCqWrite_h) (gni_ep_handle_t, gni_post_descriptor_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CtPostCqWrite(gni_ep_handle_t, gni_post_descriptor_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CtPostCqWrite_h == NULL)
      GNI_CtPostCqWrite_h = dlsym(tau_handle,"GNI_CtPostCqWrite"); 
    if (GNI_CtPostCqWrite_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CtPostCqWrite_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetCompleted
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetCompleted(gni_cq_handle_t a1, gni_cq_entry_t a2, gni_post_descriptor_t ** a3) ;
extern gni_return_t  GNI_GetCompleted(gni_cq_handle_t a1, gni_cq_entry_t a2, gni_post_descriptor_t ** a3)  {

  static gni_return_t (*GNI_GetCompleted_h) (gni_cq_handle_t, gni_cq_entry_t, gni_post_descriptor_t **) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetCompleted(gni_cq_handle_t, gni_cq_entry_t, gni_post_descriptor_t **) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetCompleted_h == NULL)
      GNI_GetCompleted_h = dlsym(tau_handle,"GNI_GetCompleted"); 
    if (GNI_GetCompleted_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetCompleted_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqGetEvent
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqGetEvent(gni_cq_handle_t a1, gni_cq_entry_t * a2) ;
extern gni_return_t  GNI_CqGetEvent(gni_cq_handle_t a1, gni_cq_entry_t * a2)  {

  static gni_return_t (*GNI_CqGetEvent_h) (gni_cq_handle_t, gni_cq_entry_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqGetEvent(gni_cq_handle_t, gni_cq_entry_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqGetEvent_h == NULL)
      GNI_CqGetEvent_h = dlsym(tau_handle,"GNI_CqGetEvent"); 
    if (GNI_CqGetEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqGetEvent_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqWaitEvent
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqWaitEvent(gni_cq_handle_t a1, uint64_t a2, gni_cq_entry_t * a3) ;
extern gni_return_t  GNI_CqWaitEvent(gni_cq_handle_t a1, uint64_t a2, gni_cq_entry_t * a3)  {

  static gni_return_t (*GNI_CqWaitEvent_h) (gni_cq_handle_t, uint64_t, gni_cq_entry_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqWaitEvent(gni_cq_handle_t, uint64_t, gni_cq_entry_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqWaitEvent_h == NULL)
      GNI_CqWaitEvent_h = dlsym(tau_handle,"GNI_CqWaitEvent"); 
    if (GNI_CqWaitEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqWaitEvent_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqVectorWaitEvent
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqVectorWaitEvent(gni_cq_handle_t * a1, uint32_t a2, uint64_t a3, gni_cq_entry_t * a4, uint32_t * a5) ;
extern gni_return_t  GNI_CqVectorWaitEvent(gni_cq_handle_t * a1, uint32_t a2, uint64_t a3, gni_cq_entry_t * a4, uint32_t * a5)  {

  static gni_return_t (*GNI_CqVectorWaitEvent_h) (gni_cq_handle_t *, uint32_t, uint64_t, gni_cq_entry_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqVectorWaitEvent(gni_cq_handle_t *, uint32_t, uint64_t, gni_cq_entry_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqVectorWaitEvent_h == NULL)
      GNI_CqVectorWaitEvent_h = dlsym(tau_handle,"GNI_CqVectorWaitEvent"); 
    if (GNI_CqVectorWaitEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqVectorWaitEvent_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqVectorMonitor
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqVectorMonitor(gni_cq_handle_t * a1, uint32_t a2, uint64_t a3, uint32_t * a4) ;
extern gni_return_t  GNI_CqVectorMonitor(gni_cq_handle_t * a1, uint32_t a2, uint64_t a3, uint32_t * a4)  {

  static gni_return_t (*GNI_CqVectorMonitor_h) (gni_cq_handle_t *, uint32_t, uint64_t, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqVectorMonitor(gni_cq_handle_t *, uint32_t, uint64_t, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqVectorMonitor_h == NULL)
      GNI_CqVectorMonitor_h = dlsym(tau_handle,"GNI_CqVectorMonitor"); 
    if (GNI_CqVectorMonitor_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqVectorMonitor_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqInterruptMask
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqInterruptMask(gni_cq_handle_t a1) ;
extern gni_return_t  GNI_CqInterruptMask(gni_cq_handle_t a1)  {

  static gni_return_t (*GNI_CqInterruptMask_h) (gni_cq_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqInterruptMask(gni_cq_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqInterruptMask_h == NULL)
      GNI_CqInterruptMask_h = dlsym(tau_handle,"GNI_CqInterruptMask"); 
    if (GNI_CqInterruptMask_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqInterruptMask_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqInterruptUnmask
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqInterruptUnmask(gni_cq_handle_t a1) ;
extern gni_return_t  GNI_CqInterruptUnmask(gni_cq_handle_t a1)  {

  static gni_return_t (*GNI_CqInterruptUnmask_h) (gni_cq_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqInterruptUnmask(gni_cq_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqInterruptUnmask_h == NULL)
      GNI_CqInterruptUnmask_h = dlsym(tau_handle,"GNI_CqInterruptUnmask"); 
    if (GNI_CqInterruptUnmask_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqInterruptUnmask_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqTestEvent
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqTestEvent(gni_cq_handle_t a1) ;
extern gni_return_t  GNI_CqTestEvent(gni_cq_handle_t a1)  {

  static gni_return_t (*GNI_CqTestEvent_h) (gni_cq_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqTestEvent(gni_cq_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqTestEvent_h == NULL)
      GNI_CqTestEvent_h = dlsym(tau_handle,"GNI_CqTestEvent"); 
    if (GNI_CqTestEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqTestEvent_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqErrorStr
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqErrorStr(gni_cq_entry_t a1, void * a2, uint32_t a3) ;
extern gni_return_t  GNI_CqErrorStr(gni_cq_entry_t a1, void * a2, uint32_t a3)  {

  static gni_return_t (*GNI_CqErrorStr_h) (gni_cq_entry_t, void *, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqErrorStr(gni_cq_entry_t, void *, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqErrorStr_h == NULL)
      GNI_CqErrorStr_h = dlsym(tau_handle,"GNI_CqErrorStr"); 
    if (GNI_CqErrorStr_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqErrorStr_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqErrorRecoverable
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqErrorRecoverable(gni_cq_entry_t a1, uint32_t * a2) ;
extern gni_return_t  GNI_CqErrorRecoverable(gni_cq_entry_t a1, uint32_t * a2)  {

  static gni_return_t (*GNI_CqErrorRecoverable_h) (gni_cq_entry_t, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqErrorRecoverable(gni_cq_entry_t, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqErrorRecoverable_h == NULL)
      GNI_CqErrorRecoverable_h = dlsym(tau_handle,"GNI_CqErrorRecoverable"); 
    if (GNI_CqErrorRecoverable_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqErrorRecoverable_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgBufferSizeNeeded
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgBufferSizeNeeded(gni_smsg_attr_t * a1, unsigned int * a2) ;
extern gni_return_t  GNI_SmsgBufferSizeNeeded(gni_smsg_attr_t * a1, unsigned int * a2)  {

  static gni_return_t (*GNI_SmsgBufferSizeNeeded_h) (gni_smsg_attr_t *, unsigned int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgBufferSizeNeeded(gni_smsg_attr_t *, unsigned int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgBufferSizeNeeded_h == NULL)
      GNI_SmsgBufferSizeNeeded_h = dlsym(tau_handle,"GNI_SmsgBufferSizeNeeded"); 
    if (GNI_SmsgBufferSizeNeeded_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgBufferSizeNeeded_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgInit
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgInit(gni_ep_handle_t a1, gni_smsg_attr_t * a2, gni_smsg_attr_t * a3) ;
extern gni_return_t  GNI_SmsgInit(gni_ep_handle_t a1, gni_smsg_attr_t * a2, gni_smsg_attr_t * a3)  {

  static gni_return_t (*GNI_SmsgInit_h) (gni_ep_handle_t, gni_smsg_attr_t *, gni_smsg_attr_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgInit(gni_ep_handle_t, gni_smsg_attr_t *, gni_smsg_attr_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgInit_h == NULL)
      GNI_SmsgInit_h = dlsym(tau_handle,"GNI_SmsgInit"); 
    if (GNI_SmsgInit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgInit_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgSetDeliveryMode
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgSetDeliveryMode(gni_nic_handle_t a1, uint16_t a2) ;
extern gni_return_t  GNI_SmsgSetDeliveryMode(gni_nic_handle_t a1, uint16_t a2)  {

  static gni_return_t (*GNI_SmsgSetDeliveryMode_h) (gni_nic_handle_t, uint16_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgSetDeliveryMode(gni_nic_handle_t, uint16_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgSetDeliveryMode_h == NULL)
      GNI_SmsgSetDeliveryMode_h = dlsym(tau_handle,"GNI_SmsgSetDeliveryMode"); 
    if (GNI_SmsgSetDeliveryMode_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgSetDeliveryMode_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgSend
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgSend(gni_ep_handle_t a1, void * a2, uint32_t a3, void * a4, uint32_t a5, uint32_t a6) ;
extern gni_return_t  GNI_SmsgSend(gni_ep_handle_t a1, void * a2, uint32_t a3, void * a4, uint32_t a5, uint32_t a6)  {

  static gni_return_t (*GNI_SmsgSend_h) (gni_ep_handle_t, void *, uint32_t, void *, uint32_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgSend(gni_ep_handle_t, void *, uint32_t, void *, uint32_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgSend_h == NULL)
      GNI_SmsgSend_h = dlsym(tau_handle,"GNI_SmsgSend"); 
    if (GNI_SmsgSend_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgSend_h) ( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgSendWTag
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgSendWTag(gni_ep_handle_t a1, void * a2, uint32_t a3, void * a4, uint32_t a5, uint32_t a6, uint8_t a7) ;
extern gni_return_t  GNI_SmsgSendWTag(gni_ep_handle_t a1, void * a2, uint32_t a3, void * a4, uint32_t a5, uint32_t a6, uint8_t a7)  {

  static gni_return_t (*GNI_SmsgSendWTag_h) (gni_ep_handle_t, void *, uint32_t, void *, uint32_t, uint32_t, uint8_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgSendWTag(gni_ep_handle_t, void *, uint32_t, void *, uint32_t, uint32_t, uint8_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgSendWTag_h == NULL)
      GNI_SmsgSendWTag_h = dlsym(tau_handle,"GNI_SmsgSendWTag"); 
    if (GNI_SmsgSendWTag_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgSendWTag_h) ( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgGetNext
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgGetNext(gni_ep_handle_t a1, void ** a2) ;
extern gni_return_t  GNI_SmsgGetNext(gni_ep_handle_t a1, void ** a2)  {

  static gni_return_t (*GNI_SmsgGetNext_h) (gni_ep_handle_t, void **) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgGetNext(gni_ep_handle_t, void **) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgGetNext_h == NULL)
      GNI_SmsgGetNext_h = dlsym(tau_handle,"GNI_SmsgGetNext"); 
    if (GNI_SmsgGetNext_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgGetNext_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgGetNextWTag
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgGetNextWTag(gni_ep_handle_t a1, void ** a2, uint8_t * a3) ;
extern gni_return_t  GNI_SmsgGetNextWTag(gni_ep_handle_t a1, void ** a2, uint8_t * a3)  {

  static gni_return_t (*GNI_SmsgGetNextWTag_h) (gni_ep_handle_t, void **, uint8_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgGetNextWTag(gni_ep_handle_t, void **, uint8_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgGetNextWTag_h == NULL)
      GNI_SmsgGetNextWTag_h = dlsym(tau_handle,"GNI_SmsgGetNextWTag"); 
    if (GNI_SmsgGetNextWTag_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgGetNextWTag_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgRelease
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgRelease(gni_ep_handle_t a1) ;
extern gni_return_t  GNI_SmsgRelease(gni_ep_handle_t a1)  {

  static gni_return_t (*GNI_SmsgRelease_h) (gni_ep_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgRelease(gni_ep_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgRelease_h == NULL)
      GNI_SmsgRelease_h = dlsym(tau_handle,"GNI_SmsgRelease"); 
    if (GNI_SmsgRelease_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgRelease_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqInit
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqInit(gni_nic_handle_t a1, gni_msgq_rcv_cb_func * a2, void * a3, gni_cq_handle_t a4, gni_msgq_attr_t * a5, gni_msgq_handle_t * a6) ;
extern gni_return_t  GNI_MsgqInit(gni_nic_handle_t a1, gni_msgq_rcv_cb_func * a2, void * a3, gni_cq_handle_t a4, gni_msgq_attr_t * a5, gni_msgq_handle_t * a6)  {

  static gni_return_t (*GNI_MsgqInit_h) (gni_nic_handle_t, gni_msgq_rcv_cb_func *, void *, gni_cq_handle_t, gni_msgq_attr_t *, gni_msgq_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqInit(gni_nic_handle_t, gni_msgq_rcv_cb_func *, void *, gni_cq_handle_t, gni_msgq_attr_t *, gni_msgq_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqInit_h == NULL)
      GNI_MsgqInit_h = dlsym(tau_handle,"GNI_MsgqInit"); 
    if (GNI_MsgqInit_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqInit_h) ( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqRelease
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqRelease(gni_msgq_handle_t a1) ;
extern gni_return_t  GNI_MsgqRelease(gni_msgq_handle_t a1)  {

  static gni_return_t (*GNI_MsgqRelease_h) (gni_msgq_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqRelease(gni_msgq_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqRelease_h == NULL)
      GNI_MsgqRelease_h = dlsym(tau_handle,"GNI_MsgqRelease"); 
    if (GNI_MsgqRelease_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqRelease_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqIdle
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqIdle(gni_msgq_handle_t a1) ;
extern gni_return_t  GNI_MsgqIdle(gni_msgq_handle_t a1)  {

  static gni_return_t (*GNI_MsgqIdle_h) (gni_msgq_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqIdle(gni_msgq_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqIdle_h == NULL)
      GNI_MsgqIdle_h = dlsym(tau_handle,"GNI_MsgqIdle"); 
    if (GNI_MsgqIdle_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqIdle_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqGetConnAttrs
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqGetConnAttrs(gni_msgq_handle_t a1, uint32_t a2, gni_msgq_ep_attr_t * a3, uint32_t * a4) ;
extern gni_return_t  GNI_MsgqGetConnAttrs(gni_msgq_handle_t a1, uint32_t a2, gni_msgq_ep_attr_t * a3, uint32_t * a4)  {

  static gni_return_t (*GNI_MsgqGetConnAttrs_h) (gni_msgq_handle_t, uint32_t, gni_msgq_ep_attr_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqGetConnAttrs(gni_msgq_handle_t, uint32_t, gni_msgq_ep_attr_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqGetConnAttrs_h == NULL)
      GNI_MsgqGetConnAttrs_h = dlsym(tau_handle,"GNI_MsgqGetConnAttrs"); 
    if (GNI_MsgqGetConnAttrs_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqGetConnAttrs_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqConnect
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqConnect(gni_msgq_handle_t a1, uint32_t a2, gni_msgq_ep_attr_t * a3) ;
extern gni_return_t  GNI_MsgqConnect(gni_msgq_handle_t a1, uint32_t a2, gni_msgq_ep_attr_t * a3)  {

  static gni_return_t (*GNI_MsgqConnect_h) (gni_msgq_handle_t, uint32_t, gni_msgq_ep_attr_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqConnect(gni_msgq_handle_t, uint32_t, gni_msgq_ep_attr_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqConnect_h == NULL)
      GNI_MsgqConnect_h = dlsym(tau_handle,"GNI_MsgqConnect"); 
    if (GNI_MsgqConnect_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqConnect_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqConnRelease
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqConnRelease(gni_msgq_handle_t a1, uint32_t a2) ;
extern gni_return_t  GNI_MsgqConnRelease(gni_msgq_handle_t a1, uint32_t a2)  {

  static gni_return_t (*GNI_MsgqConnRelease_h) (gni_msgq_handle_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqConnRelease(gni_msgq_handle_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqConnRelease_h == NULL)
      GNI_MsgqConnRelease_h = dlsym(tau_handle,"GNI_MsgqConnRelease"); 
    if (GNI_MsgqConnRelease_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqConnRelease_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqSend
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqSend(gni_msgq_handle_t a1, gni_ep_handle_t a2, void * a3, uint32_t a4, void * a5, uint32_t a6, uint32_t a7, uint8_t a8) ;
extern gni_return_t  GNI_MsgqSend(gni_msgq_handle_t a1, gni_ep_handle_t a2, void * a3, uint32_t a4, void * a5, uint32_t a6, uint32_t a7, uint8_t a8)  {

  static gni_return_t (*GNI_MsgqSend_h) (gni_msgq_handle_t, gni_ep_handle_t, void *, uint32_t, void *, uint32_t, uint32_t, uint8_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqSend(gni_msgq_handle_t, gni_ep_handle_t, void *, uint32_t, void *, uint32_t, uint32_t, uint8_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqSend_h == NULL)
      GNI_MsgqSend_h = dlsym(tau_handle,"GNI_MsgqSend"); 
    if (GNI_MsgqSend_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqSend_h) ( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqProgress
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqProgress(gni_msgq_handle_t a1, uint32_t a2) ;
extern gni_return_t  GNI_MsgqProgress(gni_msgq_handle_t a1, uint32_t a2)  {

  static gni_return_t (*GNI_MsgqProgress_h) (gni_msgq_handle_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqProgress(gni_msgq_handle_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqProgress_h == NULL)
      GNI_MsgqProgress_h = dlsym(tau_handle,"GNI_MsgqProgress"); 
    if (GNI_MsgqProgress_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqProgress_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_MsgqSize
 **********************************************************/

extern gni_return_t  __wrap_GNI_MsgqSize(gni_msgq_attr_t * a1, uint32_t * a2) ;
extern gni_return_t  GNI_MsgqSize(gni_msgq_attr_t * a1, uint32_t * a2)  {

  static gni_return_t (*GNI_MsgqSize_h) (gni_msgq_attr_t *, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_MsgqSize(gni_msgq_attr_t *, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_MsgqSize_h == NULL)
      GNI_MsgqSize_h = dlsym(tau_handle,"GNI_MsgqSize"); 
    if (GNI_MsgqSize_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_MsgqSize_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SmsgSetMaxRetrans
 **********************************************************/

extern gni_return_t  __wrap_GNI_SmsgSetMaxRetrans(gni_nic_handle_t a1, uint16_t a2) ;
extern gni_return_t  GNI_SmsgSetMaxRetrans(gni_nic_handle_t a1, uint16_t a2)  {

  static gni_return_t (*GNI_SmsgSetMaxRetrans_h) (gni_nic_handle_t, uint16_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SmsgSetMaxRetrans(gni_nic_handle_t, uint16_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SmsgSetMaxRetrans_h == NULL)
      GNI_SmsgSetMaxRetrans_h = dlsym(tau_handle,"GNI_SmsgSetMaxRetrans"); 
    if (GNI_SmsgSetMaxRetrans_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SmsgSetMaxRetrans_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SubscribeErrors
 **********************************************************/

extern gni_return_t  __wrap_GNI_SubscribeErrors(gni_nic_handle_t a1, uint32_t a2, gni_error_mask_t a3, uint32_t a4, gni_err_handle_t * a5) ;
extern gni_return_t  GNI_SubscribeErrors(gni_nic_handle_t a1, uint32_t a2, gni_error_mask_t a3, uint32_t a4, gni_err_handle_t * a5)  {

  static gni_return_t (*GNI_SubscribeErrors_h) (gni_nic_handle_t, uint32_t, gni_error_mask_t, uint32_t, gni_err_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SubscribeErrors(gni_nic_handle_t, uint32_t, gni_error_mask_t, uint32_t, gni_err_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SubscribeErrors_h == NULL)
      GNI_SubscribeErrors_h = dlsym(tau_handle,"GNI_SubscribeErrors"); 
    if (GNI_SubscribeErrors_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SubscribeErrors_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ReleaseErrors
 **********************************************************/

extern gni_return_t  __wrap_GNI_ReleaseErrors(gni_err_handle_t a1) ;
extern gni_return_t  GNI_ReleaseErrors(gni_err_handle_t a1)  {

  static gni_return_t (*GNI_ReleaseErrors_h) (gni_err_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ReleaseErrors(gni_err_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ReleaseErrors_h == NULL)
      GNI_ReleaseErrors_h = dlsym(tau_handle,"GNI_ReleaseErrors"); 
    if (GNI_ReleaseErrors_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ReleaseErrors_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetErrorMask
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetErrorMask(gni_err_handle_t a1, gni_error_mask_t * a2) ;
extern gni_return_t  GNI_GetErrorMask(gni_err_handle_t a1, gni_error_mask_t * a2)  {

  static gni_return_t (*GNI_GetErrorMask_h) (gni_err_handle_t, gni_error_mask_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetErrorMask(gni_err_handle_t, gni_error_mask_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetErrorMask_h == NULL)
      GNI_GetErrorMask_h = dlsym(tau_handle,"GNI_GetErrorMask"); 
    if (GNI_GetErrorMask_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetErrorMask_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SetErrorMask
 **********************************************************/

extern gni_return_t  __wrap_GNI_SetErrorMask(gni_err_handle_t a1, gni_error_mask_t a2, gni_error_mask_t * a3) ;
extern gni_return_t  GNI_SetErrorMask(gni_err_handle_t a1, gni_error_mask_t a2, gni_error_mask_t * a3)  {

  static gni_return_t (*GNI_SetErrorMask_h) (gni_err_handle_t, gni_error_mask_t, gni_error_mask_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SetErrorMask(gni_err_handle_t, gni_error_mask_t, gni_error_mask_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SetErrorMask_h == NULL)
      GNI_SetErrorMask_h = dlsym(tau_handle,"GNI_SetErrorMask"); 
    if (GNI_SetErrorMask_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SetErrorMask_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetErrorEvent
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetErrorEvent(gni_err_handle_t a1, gni_error_event_t * a2) ;
extern gni_return_t  GNI_GetErrorEvent(gni_err_handle_t a1, gni_error_event_t * a2)  {

  static gni_return_t (*GNI_GetErrorEvent_h) (gni_err_handle_t, gni_error_event_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetErrorEvent(gni_err_handle_t, gni_error_event_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetErrorEvent_h == NULL)
      GNI_GetErrorEvent_h = dlsym(tau_handle,"GNI_GetErrorEvent"); 
    if (GNI_GetErrorEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetErrorEvent_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_WaitErrorEvents
 **********************************************************/

extern gni_return_t  __wrap_GNI_WaitErrorEvents(gni_err_handle_t a1, gni_error_event_t * a2, uint32_t a3, uint32_t a4, uint32_t * a5) ;
extern gni_return_t  GNI_WaitErrorEvents(gni_err_handle_t a1, gni_error_event_t * a2, uint32_t a3, uint32_t a4, uint32_t * a5)  {

  static gni_return_t (*GNI_WaitErrorEvents_h) (gni_err_handle_t, gni_error_event_t *, uint32_t, uint32_t, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_WaitErrorEvents(gni_err_handle_t, gni_error_event_t *, uint32_t, uint32_t, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_WaitErrorEvents_h == NULL)
      GNI_WaitErrorEvents_h = dlsym(tau_handle,"GNI_WaitErrorEvents"); 
    if (GNI_WaitErrorEvents_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_WaitErrorEvents_h) ( a1,  a2,  a3,  a4,  a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SetErrorPtag
 **********************************************************/

extern gni_return_t  __wrap_GNI_SetErrorPtag(gni_err_handle_t a1, uint8_t a2) ;
extern gni_return_t  GNI_SetErrorPtag(gni_err_handle_t a1, uint8_t a2)  {

  static gni_return_t (*GNI_SetErrorPtag_h) (gni_err_handle_t, uint8_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SetErrorPtag(gni_err_handle_t, uint8_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SetErrorPtag_h == NULL)
      GNI_SetErrorPtag_h = dlsym(tau_handle,"GNI_SetErrorPtag"); 
    if (GNI_SetErrorPtag_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SetErrorPtag_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetNumLocalDevices
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetNumLocalDevices(int * a1) ;
extern gni_return_t  GNI_GetNumLocalDevices(int * a1)  {

  static gni_return_t (*GNI_GetNumLocalDevices_h) (int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetNumLocalDevices(int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetNumLocalDevices_h == NULL)
      GNI_GetNumLocalDevices_h = dlsym(tau_handle,"GNI_GetNumLocalDevices"); 
    if (GNI_GetNumLocalDevices_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetNumLocalDevices_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetLocalDeviceIds
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetLocalDeviceIds(int a1, int * a2) ;
extern gni_return_t  GNI_GetLocalDeviceIds(int a1, int * a2)  {

  static gni_return_t (*GNI_GetLocalDeviceIds_h) (int, int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetLocalDeviceIds(int, int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetLocalDeviceIds_h == NULL)
      GNI_GetLocalDeviceIds_h = dlsym(tau_handle,"GNI_GetLocalDeviceIds"); 
    if (GNI_GetLocalDeviceIds_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetLocalDeviceIds_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetVersion
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetVersion(uint32_t * a1) ;
extern gni_return_t  GNI_GetVersion(uint32_t * a1)  {

  static gni_return_t (*GNI_GetVersion_h) (uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetVersion(uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetVersion_h == NULL)
      GNI_GetVersion_h = dlsym(tau_handle,"GNI_GetVersion"); 
    if (GNI_GetVersion_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetVersion_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetVersionInformation
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetVersionInformation(gni_version_info_t * a1) ;
extern gni_return_t  GNI_GetVersionInformation(gni_version_info_t * a1)  {

  static gni_return_t (*GNI_GetVersionInformation_h) (gni_version_info_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetVersionInformation(gni_version_info_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetVersionInformation_h == NULL)
      GNI_GetVersionInformation_h = dlsym(tau_handle,"GNI_GetVersionInformation"); 
    if (GNI_GetVersionInformation_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetVersionInformation_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetDeviceType
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetDeviceType(gni_nic_device_t * a1) ;
extern gni_return_t  GNI_GetDeviceType(gni_nic_device_t * a1)  {

  static gni_return_t (*GNI_GetDeviceType_h) (gni_nic_device_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetDeviceType(gni_nic_device_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetDeviceType_h == NULL)
      GNI_GetDeviceType_h = dlsym(tau_handle,"GNI_GetDeviceType"); 
    if (GNI_GetDeviceType_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetDeviceType_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetDevResInfo
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetDevResInfo(uint32_t a1, gni_dev_res_t a2, gni_dev_res_desc_t * a3) ;
extern gni_return_t  GNI_GetDevResInfo(uint32_t a1, gni_dev_res_t a2, gni_dev_res_desc_t * a3)  {

  static gni_return_t (*GNI_GetDevResInfo_h) (uint32_t, gni_dev_res_t, gni_dev_res_desc_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetDevResInfo(uint32_t, gni_dev_res_t, gni_dev_res_desc_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetDevResInfo_h == NULL)
      GNI_GetDevResInfo_h = dlsym(tau_handle,"GNI_GetDevResInfo"); 
    if (GNI_GetDevResInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetDevResInfo_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetJobResInfo
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetJobResInfo(uint32_t a1, uint8_t a2, gni_job_res_t a3, gni_job_res_desc_t * a4) ;
extern gni_return_t  GNI_GetJobResInfo(uint32_t a1, uint8_t a2, gni_job_res_t a3, gni_job_res_desc_t * a4)  {

  static gni_return_t (*GNI_GetJobResInfo_h) (uint32_t, uint8_t, gni_job_res_t, gni_job_res_desc_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetJobResInfo(uint32_t, uint8_t, gni_job_res_t, gni_job_res_desc_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetJobResInfo_h == NULL)
      GNI_GetJobResInfo_h = dlsym(tau_handle,"GNI_GetJobResInfo"); 
    if (GNI_GetJobResInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetJobResInfo_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SetJobResInfo
 **********************************************************/

extern gni_return_t  __wrap_GNI_SetJobResInfo(uint32_t a1, uint8_t a2, gni_job_res_t a3, uint64_t a4) ;
extern gni_return_t  GNI_SetJobResInfo(uint32_t a1, uint8_t a2, gni_job_res_t a3, uint64_t a4)  {

  static gni_return_t (*GNI_SetJobResInfo_h) (uint32_t, uint8_t, gni_job_res_t, uint64_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SetJobResInfo(uint32_t, uint8_t, gni_job_res_t, uint64_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SetJobResInfo_h == NULL)
      GNI_SetJobResInfo_h = dlsym(tau_handle,"GNI_SetJobResInfo"); 
    if (GNI_SetJobResInfo_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SetJobResInfo_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetNttGran
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetNttGran(uint32_t a1, uint32_t * a2) ;
extern gni_return_t  GNI_GetNttGran(uint32_t a1, uint32_t * a2)  {

  static gni_return_t (*GNI_GetNttGran_h) (uint32_t, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetNttGran(uint32_t, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetNttGran_h == NULL)
      GNI_GetNttGran_h = dlsym(tau_handle,"GNI_GetNttGran"); 
    if (GNI_GetNttGran_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetNttGran_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetPtag
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetPtag(uint32_t a1, uint32_t a2, uint8_t * a3) ;
extern gni_return_t  GNI_GetPtag(uint32_t a1, uint32_t a2, uint8_t * a3)  {

  static gni_return_t (*GNI_GetPtag_h) (uint32_t, uint32_t, uint8_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetPtag(uint32_t, uint32_t, uint8_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetPtag_h == NULL)
      GNI_GetPtag_h = dlsym(tau_handle,"GNI_GetPtag"); 
    if (GNI_GetPtag_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetPtag_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CeCreate
 **********************************************************/

extern gni_return_t  __wrap_GNI_CeCreate(gni_nic_handle_t a1, gni_ce_handle_t * a2) ;
extern gni_return_t  GNI_CeCreate(gni_nic_handle_t a1, gni_ce_handle_t * a2)  {

  static gni_return_t (*GNI_CeCreate_h) (gni_nic_handle_t, gni_ce_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CeCreate(gni_nic_handle_t, gni_ce_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CeCreate_h == NULL)
      GNI_CeCreate_h = dlsym(tau_handle,"GNI_CeCreate"); 
    if (GNI_CeCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CeCreate_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CeGetId
 **********************************************************/

extern gni_return_t  __wrap_GNI_CeGetId(gni_ce_handle_t a1, uint32_t * a2) ;
extern gni_return_t  GNI_CeGetId(gni_ce_handle_t a1, uint32_t * a2)  {

  static gni_return_t (*GNI_CeGetId_h) (gni_ce_handle_t, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CeGetId(gni_ce_handle_t, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CeGetId_h == NULL)
      GNI_CeGetId_h = dlsym(tau_handle,"GNI_CeGetId"); 
    if (GNI_CeGetId_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CeGetId_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_EpSetCeAttr
 **********************************************************/

extern gni_return_t  __wrap_GNI_EpSetCeAttr(gni_ep_handle_t a1, uint32_t a2, uint32_t a3, gni_ce_child_t a4) ;
extern gni_return_t  GNI_EpSetCeAttr(gni_ep_handle_t a1, uint32_t a2, uint32_t a3, gni_ce_child_t a4)  {

  static gni_return_t (*GNI_EpSetCeAttr_h) (gni_ep_handle_t, uint32_t, uint32_t, gni_ce_child_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_EpSetCeAttr(gni_ep_handle_t, uint32_t, uint32_t, gni_ce_child_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_EpSetCeAttr_h == NULL)
      GNI_EpSetCeAttr_h = dlsym(tau_handle,"GNI_EpSetCeAttr"); 
    if (GNI_EpSetCeAttr_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_EpSetCeAttr_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CeConfigure
 **********************************************************/

extern gni_return_t  __wrap_GNI_CeConfigure(gni_ce_handle_t a1, gni_ep_handle_t * a2, uint32_t a3, gni_ep_handle_t a4, gni_cq_handle_t a5, uint32_t a6) ;
extern gni_return_t  GNI_CeConfigure(gni_ce_handle_t a1, gni_ep_handle_t * a2, uint32_t a3, gni_ep_handle_t a4, gni_cq_handle_t a5, uint32_t a6)  {

  static gni_return_t (*GNI_CeConfigure_h) (gni_ce_handle_t, gni_ep_handle_t *, uint32_t, gni_ep_handle_t, gni_cq_handle_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CeConfigure(gni_ce_handle_t, gni_ep_handle_t *, uint32_t, gni_ep_handle_t, gni_cq_handle_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CeConfigure_h == NULL)
      GNI_CeConfigure_h = dlsym(tau_handle,"GNI_CeConfigure"); 
    if (GNI_CeConfigure_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CeConfigure_h) ( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CeCheckResult
 **********************************************************/

extern gni_return_t  __wrap_GNI_CeCheckResult(gni_ce_result_t * a1, uint32_t a2) ;
extern gni_return_t  GNI_CeCheckResult(gni_ce_result_t * a1, uint32_t a2)  {

  static gni_return_t (*GNI_CeCheckResult_h) (gni_ce_result_t *, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CeCheckResult(gni_ce_result_t *, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CeCheckResult_h == NULL)
      GNI_CeCheckResult_h = dlsym(tau_handle,"GNI_CeCheckResult"); 
    if (GNI_CeCheckResult_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CeCheckResult_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CeDestroy
 **********************************************************/

extern gni_return_t  __wrap_GNI_CeDestroy(gni_ce_handle_t a1) ;
extern gni_return_t  GNI_CeDestroy(gni_ce_handle_t a1)  {

  static gni_return_t (*GNI_CeDestroy_h) (gni_ce_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CeDestroy(gni_ce_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CeDestroy_h == NULL)
      GNI_CeDestroy_h = dlsym(tau_handle,"GNI_CeDestroy"); 
    if (GNI_CeDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CeDestroy_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SetBIConfig
 **********************************************************/

extern gni_return_t  __wrap_GNI_SetBIConfig(uint32_t a1, uint16_t a2, uint16_t a3, uint16_t a4) ;
extern gni_return_t  GNI_SetBIConfig(uint32_t a1, uint16_t a2, uint16_t a3, uint16_t a4)  {

  static gni_return_t (*GNI_SetBIConfig_h) (uint32_t, uint16_t, uint16_t, uint16_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SetBIConfig(uint32_t, uint16_t, uint16_t, uint16_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SetBIConfig_h == NULL)
      GNI_SetBIConfig_h = dlsym(tau_handle,"GNI_SetBIConfig"); 
    if (GNI_SetBIConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SetBIConfig_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetBIConfig
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetBIConfig(uint32_t a1, gni_bi_desc_t * a2) ;
extern gni_return_t  GNI_GetBIConfig(uint32_t a1, gni_bi_desc_t * a2)  {

  static gni_return_t (*GNI_GetBIConfig_h) (uint32_t, gni_bi_desc_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetBIConfig(uint32_t, gni_bi_desc_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetBIConfig_h == NULL)
      GNI_GetBIConfig_h = dlsym(tau_handle,"GNI_GetBIConfig"); 
    if (GNI_GetBIConfig_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetBIConfig_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_BISyncWait
 **********************************************************/

extern gni_return_t  __wrap_GNI_BISyncWait(uint32_t a1, uint32_t a2) ;
extern gni_return_t  GNI_BISyncWait(uint32_t a1, uint32_t a2)  {

  static gni_return_t (*GNI_BISyncWait_h) (uint32_t, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_BISyncWait(uint32_t, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_BISyncWait_h == NULL)
      GNI_BISyncWait_h = dlsym(tau_handle,"GNI_BISyncWait"); 
    if (GNI_BISyncWait_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_BISyncWait_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_GetNicStat
 **********************************************************/

extern gni_return_t  __wrap_GNI_GetNicStat(gni_nic_handle_t a1, gni_statistic_t a2, uint32_t * a3) ;
extern gni_return_t  GNI_GetNicStat(gni_nic_handle_t a1, gni_statistic_t a2, uint32_t * a3)  {

  static gni_return_t (*GNI_GetNicStat_h) (gni_nic_handle_t, gni_statistic_t, uint32_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_GetNicStat(gni_nic_handle_t, gni_statistic_t, uint32_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_GetNicStat_h == NULL)
      GNI_GetNicStat_h = dlsym(tau_handle,"GNI_GetNicStat"); 
    if (GNI_GetNicStat_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_GetNicStat_h) ( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_ResetNicStat
 **********************************************************/

extern gni_return_t  __wrap_GNI_ResetNicStat(gni_nic_handle_t a1, gni_statistic_t a2) ;
extern gni_return_t  GNI_ResetNicStat(gni_nic_handle_t a1, gni_statistic_t a2)  {

  static gni_return_t (*GNI_ResetNicStat_h) (gni_nic_handle_t, gni_statistic_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_ResetNicStat(gni_nic_handle_t, gni_statistic_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_ResetNicStat_h == NULL)
      GNI_ResetNicStat_h = dlsym(tau_handle,"GNI_ResetNicStat"); 
    if (GNI_ResetNicStat_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_ResetNicStat_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CompChanCreate
 **********************************************************/

extern gni_return_t  __wrap_GNI_CompChanCreate(gni_nic_handle_t a1, gni_comp_chan_handle_t * a2) ;
extern gni_return_t  GNI_CompChanCreate(gni_nic_handle_t a1, gni_comp_chan_handle_t * a2)  {

  static gni_return_t (*GNI_CompChanCreate_h) (gni_nic_handle_t, gni_comp_chan_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CompChanCreate(gni_nic_handle_t, gni_comp_chan_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CompChanCreate_h == NULL)
      GNI_CompChanCreate_h = dlsym(tau_handle,"GNI_CompChanCreate"); 
    if (GNI_CompChanCreate_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CompChanCreate_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CompChanDestroy
 **********************************************************/

extern gni_return_t  __wrap_GNI_CompChanDestroy(gni_comp_chan_handle_t a1) ;
extern gni_return_t  GNI_CompChanDestroy(gni_comp_chan_handle_t a1)  {

  static gni_return_t (*GNI_CompChanDestroy_h) (gni_comp_chan_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CompChanDestroy(gni_comp_chan_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CompChanDestroy_h == NULL)
      GNI_CompChanDestroy_h = dlsym(tau_handle,"GNI_CompChanDestroy"); 
    if (GNI_CompChanDestroy_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CompChanDestroy_h) ( a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CompChanFd
 **********************************************************/

extern gni_return_t  __wrap_GNI_CompChanFd(gni_comp_chan_handle_t a1, int * a2) ;
extern gni_return_t  GNI_CompChanFd(gni_comp_chan_handle_t a1, int * a2)  {

  static gni_return_t (*GNI_CompChanFd_h) (gni_comp_chan_handle_t, int *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CompChanFd(gni_comp_chan_handle_t, int *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CompChanFd_h == NULL)
      GNI_CompChanFd_h = dlsym(tau_handle,"GNI_CompChanFd"); 
    if (GNI_CompChanFd_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CompChanFd_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CompChanGetEvent
 **********************************************************/

extern gni_return_t  __wrap_GNI_CompChanGetEvent(gni_comp_chan_handle_t a1, gni_cq_handle_t * a2) ;
extern gni_return_t  GNI_CompChanGetEvent(gni_comp_chan_handle_t a1, gni_cq_handle_t * a2)  {

  static gni_return_t (*GNI_CompChanGetEvent_h) (gni_comp_chan_handle_t, gni_cq_handle_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CompChanGetEvent(gni_comp_chan_handle_t, gni_cq_handle_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CompChanGetEvent_h == NULL)
      GNI_CompChanGetEvent_h = dlsym(tau_handle,"GNI_CompChanGetEvent"); 
    if (GNI_CompChanGetEvent_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CompChanGetEvent_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqAttachCompChan
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqAttachCompChan(gni_cq_handle_t a1, gni_comp_chan_handle_t a2) ;
extern gni_return_t  GNI_CqAttachCompChan(gni_cq_handle_t a1, gni_comp_chan_handle_t a2)  {

  static gni_return_t (*GNI_CqAttachCompChan_h) (gni_cq_handle_t, gni_comp_chan_handle_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqAttachCompChan(gni_cq_handle_t, gni_comp_chan_handle_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqAttachCompChan_h == NULL)
      GNI_CqAttachCompChan_h = dlsym(tau_handle,"GNI_CqAttachCompChan"); 
    if (GNI_CqAttachCompChan_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqAttachCompChan_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_CqArmCompChan
 **********************************************************/

extern gni_return_t  __wrap_GNI_CqArmCompChan(gni_cq_handle_t * a1, uint32_t a2) ;
extern gni_return_t  GNI_CqArmCompChan(gni_cq_handle_t * a1, uint32_t a2)  {

  static gni_return_t (*GNI_CqArmCompChan_h) (gni_cq_handle_t *, uint32_t) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_CqArmCompChan(gni_cq_handle_t *, uint32_t) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_CqArmCompChan_h == NULL)
      GNI_CqArmCompChan_h = dlsym(tau_handle,"GNI_CqArmCompChan"); 
    if (GNI_CqArmCompChan_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_CqArmCompChan_h) ( a1,  a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   GNI_SetDeviceOrbMMR
 **********************************************************/

extern gni_return_t  __wrap_GNI_SetDeviceOrbMMR(gni_nic_handle_t a1, gni_dev_orb_mmr_t a2, uint64_t a3, uint64_t * a4) ;
extern gni_return_t  GNI_SetDeviceOrbMMR(gni_nic_handle_t a1, gni_dev_orb_mmr_t a2, uint64_t a3, uint64_t * a4)  {

  static gni_return_t (*GNI_SetDeviceOrbMMR_h) (gni_nic_handle_t, gni_dev_orb_mmr_t, uint64_t, uint64_t *) = NULL;
  gni_return_t retval;
  TAU_PROFILE_TIMER(t,"gni_return_t GNI_SetDeviceOrbMMR(gni_nic_handle_t, gni_dev_orb_mmr_t, uint64_t, uint64_t *) C", "", TAU_GROUP_TAU_GNI);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return retval;
  } else { 
    if (GNI_SetDeviceOrbMMR_h == NULL)
      GNI_SetDeviceOrbMMR_h = dlsym(tau_handle,"GNI_SetDeviceOrbMMR"); 
    if (GNI_SetDeviceOrbMMR_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return retval;
    }
  }
  TAU_PROFILE_START(t);
  retval  =  (*GNI_SetDeviceOrbMMR_h) ( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  return retval;

}

