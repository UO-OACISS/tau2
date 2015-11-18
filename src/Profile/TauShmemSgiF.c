#include <stdio.h>
#include <Profile/Profiler.h>
int TAUDECL tau_totalnodes(int set_or_get, int value);
static int tau_shmem_tagid_f=0 ; 
#define TAU_SHMEM_TAGID tau_shmem_tagid_f=tau_shmem_tagid_f%250
#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid_f) % 250 
#include <dlfcn.h>

const char * tau_orig_libname = "libsma.so";
static void *tau_handle = NULL;



/**********************************************************
   shmem_addr_accessible_
 **********************************************************/

void shmem_addr_accessible_(void * a1, int * a2) {

  typedef void (*shmem_addr_accessible__p_h) (void *, int *);
  static shmem_addr_accessible__p_h shmem_addr_accessible__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_addr_accessible_(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_addr_accessible__h == NULL)
	shmem_addr_accessible__h = (shmem_addr_accessible__p_h) dlsym(tau_handle,"shmem_addr_accessible_"); 
    if (shmem_addr_accessible__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_addr_accessible__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_barrier_
 **********************************************************/

void shmem_barrier_(int * a1, int * a2, int * a3, long * a4) {

  typedef void (*shmem_barrier__p_h) (int *, int *, int *, long *);
  static shmem_barrier__p_h shmem_barrier__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_barrier_(int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_barrier__h == NULL)
	shmem_barrier__h = (shmem_barrier__p_h) dlsym(tau_handle,"shmem_barrier_"); 
    if (shmem_barrier__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_barrier__h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_barrier_all_
 **********************************************************/

void shmem_barrier_all_() {

  typedef void (*shmem_barrier_all__p_h) ();
  static shmem_barrier_all__p_h shmem_barrier_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_barrier_all_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_barrier_all__h == NULL)
	shmem_barrier_all__h = (shmem_barrier_all__p_h) dlsym(tau_handle,"shmem_barrier_all_"); 
    if (shmem_barrier_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_barrier_all__h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_barrier_ps_
 **********************************************************/

void shmem_barrier_ps_(int * a1, int * a2, int * a3, long * a4) {

  typedef void (*shmem_barrier_ps__p_h) (int *, int *, int *, long *);
  static shmem_barrier_ps__p_h shmem_barrier_ps__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_barrier_ps_(int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_barrier_ps__h == NULL)
	shmem_barrier_ps__h = (shmem_barrier_ps__p_h) dlsym(tau_handle,"shmem_barrier_ps_"); 
    if (shmem_barrier_ps__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_barrier_ps__h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast32_
 **********************************************************/

void shmem_broadcast32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast32__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast32__p_h shmem_broadcast32__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast32_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast32__h == NULL)
	shmem_broadcast32__h = (shmem_broadcast32__p_h) dlsym(tau_handle,"shmem_broadcast32_"); 
    if (shmem_broadcast32__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast32__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast4_
 **********************************************************/

void shmem_broadcast4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast4__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast4__p_h shmem_broadcast4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast4_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast4__h == NULL)
	shmem_broadcast4__h = (shmem_broadcast4__p_h) dlsym(tau_handle,"shmem_broadcast4_"); 
    if (shmem_broadcast4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast4__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast64_
 **********************************************************/

void shmem_broadcast64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast64__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast64__p_h shmem_broadcast64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast64_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast64__h == NULL)
	shmem_broadcast64__h = (shmem_broadcast64__p_h) dlsym(tau_handle,"shmem_broadcast64_"); 
    if (shmem_broadcast64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast64__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast8_
 **********************************************************/

void shmem_broadcast8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast8__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast8__p_h shmem_broadcast8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast8_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast8__h == NULL)
	shmem_broadcast8__h = (shmem_broadcast8__p_h) dlsym(tau_handle,"shmem_broadcast8_"); 
    if (shmem_broadcast8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast8__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_character_get_
 **********************************************************/

void shmem_character_get_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_character_get__p_h) (void *, void *, int *, int *);
  static shmem_character_get__p_h shmem_character_get__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_character_get_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_character_get__h == NULL)
	shmem_character_get__h = (shmem_character_get__p_h) dlsym(tau_handle,"shmem_character_get_"); 
    if (shmem_character_get__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)* (*a3), (*a4));
  (*shmem_character_get__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(char)* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_character_put_
 **********************************************************/

void shmem_character_put_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_character_put__p_h) (void *, void *, int *, int *);
  static shmem_character_put__p_h shmem_character_put__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_character_put_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_character_put__h == NULL)
	shmem_character_put__h = (shmem_character_put__p_h) dlsym(tau_handle,"shmem_character_put_"); 
    if (shmem_character_put__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(char)* (*a3));
  (*shmem_character_put__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_cache_inv_
 **********************************************************/

void shmem_clear_cache_inv_() {

  typedef void (*shmem_clear_cache_inv__p_h) ();
  static shmem_clear_cache_inv__p_h shmem_clear_cache_inv__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_inv_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_cache_inv__h == NULL)
	shmem_clear_cache_inv__h = (shmem_clear_cache_inv__p_h) dlsym(tau_handle,"shmem_clear_cache_inv_"); 
    if (shmem_clear_cache_inv__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_cache_inv__h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_cache_line_inv_
 **********************************************************/

void shmem_clear_cache_line_inv_(void * a1) {

  typedef void (*shmem_clear_cache_line_inv__p_h) (void *);
  static shmem_clear_cache_line_inv__p_h shmem_clear_cache_line_inv__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_line_inv_(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_cache_line_inv__h == NULL)
	shmem_clear_cache_line_inv__h = (shmem_clear_cache_line_inv__p_h) dlsym(tau_handle,"shmem_clear_cache_line_inv_"); 
    if (shmem_clear_cache_line_inv__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_cache_line_inv__h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_lock_
 **********************************************************/

void shmem_clear_lock_(long * a1) {

  typedef void (*shmem_clear_lock__p_h) (long *);
  static shmem_clear_lock__p_h shmem_clear_lock__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_lock_(long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_lock__h == NULL)
	shmem_clear_lock__h = (shmem_clear_lock__p_h) dlsym(tau_handle,"shmem_clear_lock_"); 
    if (shmem_clear_lock__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_lock__h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_collect4_
 **********************************************************/

void shmem_collect4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_collect4__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_collect4__p_h shmem_collect4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_collect4_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_collect4__h == NULL)
	shmem_collect4__h = (shmem_collect4__p_h) dlsym(tau_handle,"shmem_collect4_"); 
    if (shmem_collect4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_collect4__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_collect64_
 **********************************************************/

void shmem_collect64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_collect64__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_collect64__p_h shmem_collect64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_collect64_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_collect64__h == NULL)
	shmem_collect64__h = (shmem_collect64__p_h) dlsym(tau_handle,"shmem_collect64_"); 
    if (shmem_collect64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_collect64__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_collect8_
 **********************************************************/

void shmem_collect8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_collect8__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_collect8__p_h shmem_collect8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_collect8_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_collect8__h == NULL)
	shmem_collect8__h = (shmem_collect8__p_h) dlsym(tau_handle,"shmem_collect8_"); 
    if (shmem_collect8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_collect8__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp4_prod_to_all_
 **********************************************************/

void shmem_comp4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_comp4_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_comp4_prod_to_all__p_h shmem_comp4_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp4_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp4_prod_to_all__h == NULL)
	shmem_comp4_prod_to_all__h = (shmem_comp4_prod_to_all__p_h) dlsym(tau_handle,"shmem_comp4_prod_to_all_"); 
    if (shmem_comp4_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp4_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp4_sum_to_all_
 **********************************************************/

void shmem_comp4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_comp4_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_comp4_sum_to_all__p_h shmem_comp4_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp4_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp4_sum_to_all__h == NULL)
	shmem_comp4_sum_to_all__h = (shmem_comp4_sum_to_all__p_h) dlsym(tau_handle,"shmem_comp4_sum_to_all_"); 
    if (shmem_comp4_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp4_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp8_prod_to_all_
 **********************************************************/

void shmem_comp8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_comp8_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_comp8_prod_to_all__p_h shmem_comp8_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp8_prod_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp8_prod_to_all__h == NULL)
	shmem_comp8_prod_to_all__h = (shmem_comp8_prod_to_all__p_h) dlsym(tau_handle,"shmem_comp8_prod_to_all_"); 
    if (shmem_comp8_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp8_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp8_sum_to_all_
 **********************************************************/

void shmem_comp8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_comp8_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_comp8_sum_to_all__p_h shmem_comp8_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp8_sum_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp8_sum_to_all__h == NULL)
	shmem_comp8_sum_to_all__h = (shmem_comp8_sum_to_all__p_h) dlsym(tau_handle,"shmem_comp8_sum_to_all_"); 
    if (shmem_comp8_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp8_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_get_
 **********************************************************/

void shmem_complex_get_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_complex_get__p_h) (void *, void *, int *, int *);
  static shmem_complex_get__p_h shmem_complex_get__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_get_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_get__h == NULL)
	shmem_complex_get__h = (shmem_complex_get__p_h) dlsym(tau_handle,"shmem_complex_get_"); 
    if (shmem_complex_get__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_complex_get__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_iget_
 **********************************************************/

void shmem_complex_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_complex_iget__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_complex_iget__p_h shmem_complex_iget__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_iget_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_iget__h == NULL)
	shmem_complex_iget__h = (shmem_complex_iget__p_h) dlsym(tau_handle,"shmem_complex_iget_"); 
    if (shmem_complex_iget__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
  (*shmem_complex_iget__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_iput_
 **********************************************************/

void shmem_complex_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_complex_iput__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_complex_iput__p_h shmem_complex_iput__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_iput_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_iput__h == NULL)
	shmem_complex_iput__h = (shmem_complex_iput__p_h) dlsym(tau_handle,"shmem_complex_iput_"); 
    if (shmem_complex_iput__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
  (*shmem_complex_iput__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_put_
 **********************************************************/

void shmem_complex_put_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_complex_put__p_h) (void *, void *, int *, int *);
  static shmem_complex_put__p_h shmem_complex_put__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_put_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_put__h == NULL)
	shmem_complex_put__h = (shmem_complex_put__p_h) dlsym(tau_handle,"shmem_complex_put_"); 
    if (shmem_complex_put__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_complex_put__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_get_
 **********************************************************/

void shmem_double_get_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_double_get__p_h) (void *, void *, int *, int *);
  static shmem_double_get__p_h shmem_double_get__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_get_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_get__h == NULL)
	shmem_double_get__h = (shmem_double_get__p_h) dlsym(tau_handle,"shmem_double_get_"); 
    if (shmem_double_get__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  (*shmem_double_get__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(double)* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_iget_
 **********************************************************/

void shmem_double_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_double_iget__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_double_iget__p_h shmem_double_iget__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_iget_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_iget__h == NULL)
	shmem_double_iget__h = (shmem_double_iget__p_h) dlsym(tau_handle,"shmem_double_iget_"); 
    if (shmem_double_iget__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a5), (*a6));
  (*shmem_double_iget__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), sizeof(double)* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_iput_
 **********************************************************/

void shmem_double_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_double_iput__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_double_iput__p_h shmem_double_iput__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_iput_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_iput__h == NULL)
	shmem_double_iput__h = (shmem_double_iput__p_h) dlsym(tau_handle,"shmem_double_iput_"); 
    if (shmem_double_iput__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), sizeof(double)* (*a5));
  (*shmem_double_iput__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_put_
 **********************************************************/

void shmem_double_put_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_double_put__p_h) (void *, void *, int *, int *);
  static shmem_double_put__p_h shmem_double_put__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_put_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_put__h == NULL)
	shmem_double_put__h = (shmem_double_put__p_h) dlsym(tau_handle,"shmem_double_put_"); 
    if (shmem_double_put__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(double)* (*a3));
  (*shmem_double_put__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect32_
 **********************************************************/

void shmem_fcollect32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect32__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect32__p_h shmem_fcollect32__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect32_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect32__h == NULL)
	shmem_fcollect32__h = (shmem_fcollect32__p_h) dlsym(tau_handle,"shmem_fcollect32_"); 
    if (shmem_fcollect32__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect32__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect4_
 **********************************************************/

void shmem_fcollect4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect4__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect4__p_h shmem_fcollect4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect4_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect4__h == NULL)
	shmem_fcollect4__h = (shmem_fcollect4__p_h) dlsym(tau_handle,"shmem_fcollect4_"); 
    if (shmem_fcollect4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect4__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect64_
 **********************************************************/

void shmem_fcollect64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect64__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect64__p_h shmem_fcollect64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect64_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect64__h == NULL)
	shmem_fcollect64__h = (shmem_fcollect64__p_h) dlsym(tau_handle,"shmem_fcollect64_"); 
    if (shmem_fcollect64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect64__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect8_
 **********************************************************/

void shmem_fcollect8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect8__p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect8__p_h shmem_fcollect8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect8_(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect8__h == NULL)
	shmem_fcollect8__h = (shmem_fcollect8__p_h) dlsym(tau_handle,"shmem_fcollect8_"); 
    if (shmem_fcollect8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect8__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fence_
 **********************************************************/

void shmem_fence_() {

  typedef void (*shmem_fence__p_h) ();
  static shmem_fence__p_h shmem_fence__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fence_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fence__h == NULL)
	shmem_fence__h = (shmem_fence__p_h) dlsym(tau_handle,"shmem_fence_"); 
    if (shmem_fence__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fence__h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get128_
 **********************************************************/

void shmem_get128_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get128__p_h) (void *, void *, int *, int *);
  static shmem_get128__p_h shmem_get128__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get128_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get128__h == NULL)
	shmem_get128__h = (shmem_get128__p_h) dlsym(tau_handle,"shmem_get128_"); 
    if (shmem_get128__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a3), (*a4));
  (*shmem_get128__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 16* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get16_
 **********************************************************/

void shmem_get16_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get16__p_h) (void *, void *, int *, int *);
  static shmem_get16__p_h shmem_get16__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get16_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get16__h == NULL)
	shmem_get16__h = (shmem_get16__p_h) dlsym(tau_handle,"shmem_get16_"); 
    if (shmem_get16__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a3), (*a4));
  (*shmem_get16__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 2* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get32_
 **********************************************************/

void shmem_get32_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get32__p_h) (void *, void *, int *, int *);
  static shmem_get32__p_h shmem_get32__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get32_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get32__h == NULL)
	shmem_get32__h = (shmem_get32__p_h) dlsym(tau_handle,"shmem_get32_"); 
    if (shmem_get32__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
  (*shmem_get32__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get4_
 **********************************************************/

void shmem_get4_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get4__p_h) (void *, void *, int *, int *);
  static shmem_get4__p_h shmem_get4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get4_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get4__h == NULL)
	shmem_get4__h = (shmem_get4__p_h) dlsym(tau_handle,"shmem_get4_"); 
    if (shmem_get4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
  (*shmem_get4__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get64_
 **********************************************************/

void shmem_get64_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get64__p_h) (void *, void *, int *, int *);
  static shmem_get64__p_h shmem_get64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get64_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get64__h == NULL)
	shmem_get64__h = (shmem_get64__p_h) dlsym(tau_handle,"shmem_get64_"); 
    if (shmem_get64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
  (*shmem_get64__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get8_
 **********************************************************/

void shmem_get8_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get8__p_h) (void *, void *, int *, int *);
  static shmem_get8__p_h shmem_get8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get8_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get8__h == NULL)
	shmem_get8__h = (shmem_get8__p_h) dlsym(tau_handle,"shmem_get8_"); 
    if (shmem_get8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
  (*shmem_get8__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_getmem_
 **********************************************************/

void shmem_getmem_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_getmem__p_h) (void *, void *, int *, int *);
  static shmem_getmem__p_h shmem_getmem__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_getmem_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_getmem__h == NULL)
	shmem_getmem__h = (shmem_getmem__p_h) dlsym(tau_handle,"shmem_getmem_"); 
    if (shmem_getmem__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_getmem__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_group_create_strided_
 **********************************************************/

void shmem_group_create_strided_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_group_create_strided__p_h) (int *, int *, int *, int *, int *, int *);
  static shmem_group_create_strided__p_h shmem_group_create_strided__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_group_create_strided_(int *, int *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_group_create_strided__h == NULL)
	shmem_group_create_strided__h = (shmem_group_create_strided__p_h) dlsym(tau_handle,"shmem_group_create_strided_"); 
    if (shmem_group_create_strided__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_group_create_strided__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_group_delete_
 **********************************************************/

void shmem_group_delete_(int * a1) {

  typedef void (*shmem_group_delete__p_h) (int *);
  static shmem_group_delete__p_h shmem_group_delete__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_group_delete_(int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_group_delete__h == NULL)
	shmem_group_delete__h = (shmem_group_delete__p_h) dlsym(tau_handle,"shmem_group_delete_"); 
    if (shmem_group_delete__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_group_delete__h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget128_
 **********************************************************/

void shmem_iget128_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget128__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget128__p_h shmem_iget128__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget128_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget128__h == NULL)
	shmem_iget128__h = (shmem_iget128__p_h) dlsym(tau_handle,"shmem_iget128_"); 
    if (shmem_iget128__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a5), (*a6));
  (*shmem_iget128__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 16* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget16_
 **********************************************************/

void shmem_iget16_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget16__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget16__p_h shmem_iget16__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget16_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget16__h == NULL)
	shmem_iget16__h = (shmem_iget16__p_h) dlsym(tau_handle,"shmem_iget16_"); 
    if (shmem_iget16__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a5), (*a6));
  (*shmem_iget16__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 2* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget32_
 **********************************************************/

void shmem_iget32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget32__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget32__p_h shmem_iget32__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget32_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget32__h == NULL)
	shmem_iget32__h = (shmem_iget32__p_h) dlsym(tau_handle,"shmem_iget32_"); 
    if (shmem_iget32__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
  (*shmem_iget32__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget4_
 **********************************************************/

void shmem_iget4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget4__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget4__p_h shmem_iget4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget4_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget4__h == NULL)
	shmem_iget4__h = (shmem_iget4__p_h) dlsym(tau_handle,"shmem_iget4_"); 
    if (shmem_iget4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
  (*shmem_iget4__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget64_
 **********************************************************/

void shmem_iget64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget64__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget64__p_h shmem_iget64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget64_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget64__h == NULL)
	shmem_iget64__h = (shmem_iget64__p_h) dlsym(tau_handle,"shmem_iget64_"); 
    if (shmem_iget64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
  (*shmem_iget64__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget8_
 **********************************************************/

void shmem_iget8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget8__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget8__p_h shmem_iget8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget8_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget8__h == NULL)
	shmem_iget8__h = (shmem_iget8__p_h) dlsym(tau_handle,"shmem_iget8_"); 
    if (shmem_iget8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
  (*shmem_iget8__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_and_to_all_
 **********************************************************/

void shmem_int2_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_and_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_and_to_all__p_h shmem_int2_and_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_and_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_and_to_all__h == NULL)
	shmem_int2_and_to_all__h = (shmem_int2_and_to_all__p_h) dlsym(tau_handle,"shmem_int2_and_to_all_"); 
    if (shmem_int2_and_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_and_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_max_to_all_
 **********************************************************/

void shmem_int2_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_max_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_max_to_all__p_h shmem_int2_max_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_max_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_max_to_all__h == NULL)
	shmem_int2_max_to_all__h = (shmem_int2_max_to_all__p_h) dlsym(tau_handle,"shmem_int2_max_to_all_"); 
    if (shmem_int2_max_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_max_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_min_to_all_
 **********************************************************/

void shmem_int2_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_min_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_min_to_all__p_h shmem_int2_min_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_min_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_min_to_all__h == NULL)
	shmem_int2_min_to_all__h = (shmem_int2_min_to_all__p_h) dlsym(tau_handle,"shmem_int2_min_to_all_"); 
    if (shmem_int2_min_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_min_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_or_to_all_
 **********************************************************/

void shmem_int2_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_or_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_or_to_all__p_h shmem_int2_or_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_or_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_or_to_all__h == NULL)
	shmem_int2_or_to_all__h = (shmem_int2_or_to_all__p_h) dlsym(tau_handle,"shmem_int2_or_to_all_"); 
    if (shmem_int2_or_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_or_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_prod_to_all_
 **********************************************************/

void shmem_int2_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_prod_to_all__p_h shmem_int2_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_prod_to_all__h == NULL)
	shmem_int2_prod_to_all__h = (shmem_int2_prod_to_all__p_h) dlsym(tau_handle,"shmem_int2_prod_to_all_"); 
    if (shmem_int2_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_sum_to_all_
 **********************************************************/

void shmem_int2_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_sum_to_all__p_h shmem_int2_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_sum_to_all__h == NULL)
	shmem_int2_sum_to_all__h = (shmem_int2_sum_to_all__p_h) dlsym(tau_handle,"shmem_int2_sum_to_all_"); 
    if (shmem_int2_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_xor_to_all_
 **********************************************************/

void shmem_int2_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_xor_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_xor_to_all__p_h shmem_int2_xor_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_xor_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_xor_to_all__h == NULL)
	shmem_int2_xor_to_all__h = (shmem_int2_xor_to_all__p_h) dlsym(tau_handle,"shmem_int2_xor_to_all_"); 
    if (shmem_int2_xor_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_xor_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_add_
 **********************************************************/

void shmem_int4_add_(void * a1, int * a2, int * a3) {

  typedef void (*shmem_int4_add__p_h) (void *, int *, int *);
  static shmem_int4_add__p_h shmem_int4_add__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_add_(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_add__h == NULL)
	shmem_int4_add__h = (shmem_int4_add__p_h) dlsym(tau_handle,"shmem_int4_add_"); 
    if (shmem_int4_add__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_add__h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_and_to_all_
 **********************************************************/

void shmem_int4_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_and_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_and_to_all__p_h shmem_int4_and_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_and_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_and_to_all__h == NULL)
	shmem_int4_and_to_all__h = (shmem_int4_and_to_all__p_h) dlsym(tau_handle,"shmem_int4_and_to_all_"); 
    if (shmem_int4_and_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_and_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_cswap_
 **********************************************************/

int shmem_int4_cswap_(int * a1, int * a2, int * a3, int * a4) {

  typedef int (*shmem_int4_cswap__p_h) (int *, int *, int *, int *);
  static shmem_int4_cswap__p_h shmem_int4_cswap__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_cswap_(int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_cswap__h == NULL)
	shmem_int4_cswap__h = (shmem_int4_cswap__p_h) dlsym(tau_handle,"shmem_int4_cswap_"); 
    if (shmem_int4_cswap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a4));
  retval  =  (*shmem_int4_cswap__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)*1);
  if (retval == (*a2)) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a4));
  }
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_fadd_
 **********************************************************/

int shmem_int4_fadd_(void * a1, int * a2, int * a3) {

  typedef int (*shmem_int4_fadd__p_h) (void *, int *, int *);
  static shmem_int4_fadd__p_h shmem_int4_fadd__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_fadd_(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_fadd__h == NULL)
	shmem_int4_fadd__h = (shmem_int4_fadd__p_h) dlsym(tau_handle,"shmem_int4_fadd_"); 
    if (shmem_int4_fadd__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int4_fadd__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_finc_
 **********************************************************/

int shmem_int4_finc_(void * a1, int * a2) {

  typedef int (*shmem_int4_finc__p_h) (void *, int *);
  static shmem_int4_finc__p_h shmem_int4_finc__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_finc_(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_finc__h == NULL)
	shmem_int4_finc__h = (shmem_int4_finc__p_h) dlsym(tau_handle,"shmem_int4_finc_"); 
    if (shmem_int4_finc__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a2));
  retval  =  (*shmem_int4_finc__h)( a1,  a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a2), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a2), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a2));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_inc_
 **********************************************************/

void shmem_int4_inc_(void * a1, int * a2) {

  typedef void (*shmem_int4_inc__p_h) (void *, int *);
  static shmem_int4_inc__p_h shmem_int4_inc__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_inc_(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_inc__h == NULL)
	shmem_int4_inc__h = (shmem_int4_inc__p_h) dlsym(tau_handle,"shmem_int4_inc_"); 
    if (shmem_int4_inc__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_inc__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_max_to_all_
 **********************************************************/

void shmem_int4_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_max_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_max_to_all__p_h shmem_int4_max_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_max_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_max_to_all__h == NULL)
	shmem_int4_max_to_all__h = (shmem_int4_max_to_all__p_h) dlsym(tau_handle,"shmem_int4_max_to_all_"); 
    if (shmem_int4_max_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_max_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_min_to_all_
 **********************************************************/

void shmem_int4_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_min_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_min_to_all__p_h shmem_int4_min_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_min_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_min_to_all__h == NULL)
	shmem_int4_min_to_all__h = (shmem_int4_min_to_all__p_h) dlsym(tau_handle,"shmem_int4_min_to_all_"); 
    if (shmem_int4_min_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_min_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_or_to_all_
 **********************************************************/

void shmem_int4_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_or_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_or_to_all__p_h shmem_int4_or_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_or_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_or_to_all__h == NULL)
	shmem_int4_or_to_all__h = (shmem_int4_or_to_all__p_h) dlsym(tau_handle,"shmem_int4_or_to_all_"); 
    if (shmem_int4_or_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_or_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_prod_to_all_
 **********************************************************/

void shmem_int4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_prod_to_all__p_h shmem_int4_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_prod_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_prod_to_all__h == NULL)
	shmem_int4_prod_to_all__h = (shmem_int4_prod_to_all__p_h) dlsym(tau_handle,"shmem_int4_prod_to_all_"); 
    if (shmem_int4_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_sum_to_all_
 **********************************************************/

void shmem_int4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_sum_to_all__p_h shmem_int4_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_sum_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_sum_to_all__h == NULL)
	shmem_int4_sum_to_all__h = (shmem_int4_sum_to_all__p_h) dlsym(tau_handle,"shmem_int4_sum_to_all_"); 
    if (shmem_int4_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_swap_
 **********************************************************/

int shmem_int4_swap_(void * a1, int * a2, int * a3) {

  typedef int (*shmem_int4_swap__p_h) (void *, int *, int *);
  static shmem_int4_swap__p_h shmem_int4_swap__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_swap_(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_swap__h == NULL)
	shmem_int4_swap__h = (shmem_int4_swap__p_h) dlsym(tau_handle,"shmem_int4_swap_"); 
    if (shmem_int4_swap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int4_swap__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_wait_
 **********************************************************/

void shmem_int4_wait_(int * a1, int * a2) {

  typedef void (*shmem_int4_wait__p_h) (int *, int *);
  static shmem_int4_wait__p_h shmem_int4_wait__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_(int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_wait__h == NULL)
	shmem_int4_wait__h = (shmem_int4_wait__p_h) dlsym(tau_handle,"shmem_int4_wait_"); 
    if (shmem_int4_wait__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_wait__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_wait_until_
 **********************************************************/

void shmem_int4_wait_until_(int * a1, int * a2, int * a3) {

  typedef void (*shmem_int4_wait_until__p_h) (int *, int *, int *);
  static shmem_int4_wait_until__p_h shmem_int4_wait_until__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_until_(int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_wait_until__h == NULL)
	shmem_int4_wait_until__h = (shmem_int4_wait_until__p_h) dlsym(tau_handle,"shmem_int4_wait_until_"); 
    if (shmem_int4_wait_until__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_wait_until__h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_xor_to_all_
 **********************************************************/

void shmem_int4_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_xor_to_all__p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_xor_to_all__p_h shmem_int4_xor_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_xor_to_all_(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_xor_to_all__h == NULL)
	shmem_int4_xor_to_all__h = (shmem_int4_xor_to_all__p_h) dlsym(tau_handle,"shmem_int4_xor_to_all_"); 
    if (shmem_int4_xor_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_xor_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_add_
 **********************************************************/

void shmem_int8_add_(void * a1, long * a2, int * a3) {

  typedef void (*shmem_int8_add__p_h) (void *, long *, int *);
  static shmem_int8_add__p_h shmem_int8_add__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_add_(void *, long *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_add__h == NULL)
	shmem_int8_add__h = (shmem_int8_add__p_h) dlsym(tau_handle,"shmem_int8_add_"); 
    if (shmem_int8_add__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_add__h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_and_to_all_
 **********************************************************/

void shmem_int8_and_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_and_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_and_to_all__p_h shmem_int8_and_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_and_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_and_to_all__h == NULL)
	shmem_int8_and_to_all__h = (shmem_int8_and_to_all__p_h) dlsym(tau_handle,"shmem_int8_and_to_all_"); 
    if (shmem_int8_and_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_and_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_cswap_
 **********************************************************/

long shmem_int8_cswap_(long * a1, long * a2, long * a3, int * a4) {

  typedef long (*shmem_int8_cswap__p_h) (long *, long *, long *, int *);
  static shmem_int8_cswap__p_h shmem_int8_cswap__h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_cswap_(long *, long *, long *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_cswap__h == NULL)
	shmem_int8_cswap__h = (shmem_int8_cswap__p_h) dlsym(tau_handle,"shmem_int8_cswap_"); 
    if (shmem_int8_cswap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a4));
  retval  =  (*shmem_int8_cswap__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)*1);
  if (retval == (*a2)) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a4));
  }
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_fadd_
 **********************************************************/

long shmem_int8_fadd_(void * a1, int * a2, int * a3) {

  typedef long (*shmem_int8_fadd__p_h) (void *, int *, int *);
  static shmem_int8_fadd__p_h shmem_int8_fadd__h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_fadd_(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_fadd__h == NULL)
	shmem_int8_fadd__h = (shmem_int8_fadd__p_h) dlsym(tau_handle,"shmem_int8_fadd_"); 
    if (shmem_int8_fadd__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int8_fadd__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_finc_
 **********************************************************/

long shmem_int8_finc_(void * a1, int * a2) {

  typedef long (*shmem_int8_finc__p_h) (void *, int *);
  static shmem_int8_finc__p_h shmem_int8_finc__h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_finc_(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_finc__h == NULL)
	shmem_int8_finc__h = (shmem_int8_finc__p_h) dlsym(tau_handle,"shmem_int8_finc_"); 
    if (shmem_int8_finc__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a2));
  retval  =  (*shmem_int8_finc__h)( a1,  a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a2), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a2), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a2));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_inc_
 **********************************************************/

void shmem_int8_inc_(void * a1, int * a2) {

  typedef void (*shmem_int8_inc__p_h) (void *, int *);
  static shmem_int8_inc__p_h shmem_int8_inc__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_inc_(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_inc__h == NULL)
	shmem_int8_inc__h = (shmem_int8_inc__p_h) dlsym(tau_handle,"shmem_int8_inc_"); 
    if (shmem_int8_inc__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_inc__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_max_to_all_
 **********************************************************/

void shmem_int8_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_max_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_max_to_all__p_h shmem_int8_max_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_max_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_max_to_all__h == NULL)
	shmem_int8_max_to_all__h = (shmem_int8_max_to_all__p_h) dlsym(tau_handle,"shmem_int8_max_to_all_"); 
    if (shmem_int8_max_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_max_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_min_to_all_
 **********************************************************/

void shmem_int8_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_min_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_min_to_all__p_h shmem_int8_min_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_min_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_min_to_all__h == NULL)
	shmem_int8_min_to_all__h = (shmem_int8_min_to_all__p_h) dlsym(tau_handle,"shmem_int8_min_to_all_"); 
    if (shmem_int8_min_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_min_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_or_to_all_
 **********************************************************/

void shmem_int8_or_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_or_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_or_to_all__p_h shmem_int8_or_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_or_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_or_to_all__h == NULL)
	shmem_int8_or_to_all__h = (shmem_int8_or_to_all__p_h) dlsym(tau_handle,"shmem_int8_or_to_all_"); 
    if (shmem_int8_or_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_or_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_prod_to_all_
 **********************************************************/

void shmem_int8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_prod_to_all__p_h shmem_int8_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_prod_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_prod_to_all__h == NULL)
	shmem_int8_prod_to_all__h = (shmem_int8_prod_to_all__p_h) dlsym(tau_handle,"shmem_int8_prod_to_all_"); 
    if (shmem_int8_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_sum_to_all_
 **********************************************************/

void shmem_int8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_sum_to_all__p_h shmem_int8_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_sum_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_sum_to_all__h == NULL)
	shmem_int8_sum_to_all__h = (shmem_int8_sum_to_all__p_h) dlsym(tau_handle,"shmem_int8_sum_to_all_"); 
    if (shmem_int8_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_swap_
 **********************************************************/

long shmem_int8_swap_(void * a1, long * a2, int * a3) {

  typedef long (*shmem_int8_swap__p_h) (void *, long *, int *);
  static shmem_int8_swap__p_h shmem_int8_swap__h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_swap_(void *, long *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_swap__h == NULL)
	shmem_int8_swap__h = (shmem_int8_swap__p_h) dlsym(tau_handle,"shmem_int8_swap_"); 
    if (shmem_int8_swap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int8_swap__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_wait_
 **********************************************************/

void shmem_int8_wait_(long * a1, long * a2) {

  typedef void (*shmem_int8_wait__p_h) (long *, long *);
  static shmem_int8_wait__p_h shmem_int8_wait__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_(long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_wait__h == NULL)
	shmem_int8_wait__h = (shmem_int8_wait__p_h) dlsym(tau_handle,"shmem_int8_wait_"); 
    if (shmem_int8_wait__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_wait__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_wait_until_
 **********************************************************/

void shmem_int8_wait_until_(long * a1, int * a2, long * a3) {

  typedef void (*shmem_int8_wait_until__p_h) (long *, int *, long *);
  static shmem_int8_wait_until__p_h shmem_int8_wait_until__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_until_(long *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_wait_until__h == NULL)
	shmem_int8_wait_until__h = (shmem_int8_wait_until__p_h) dlsym(tau_handle,"shmem_int8_wait_until_"); 
    if (shmem_int8_wait_until__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_wait_until__h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_xor_to_all_
 **********************************************************/

void shmem_int8_xor_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_xor_to_all__p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_xor_to_all__p_h shmem_int8_xor_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_xor_to_all_(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_xor_to_all__h == NULL)
	shmem_int8_xor_to_all__h = (shmem_int8_xor_to_all__p_h) dlsym(tau_handle,"shmem_int8_xor_to_all_"); 
    if (shmem_int8_xor_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_xor_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_get_
 **********************************************************/

void shmem_integer_get_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_integer_get__p_h) (void *, void *, int *, int *);
  static shmem_integer_get__p_h shmem_integer_get__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_get_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_get__h == NULL)
	shmem_integer_get__h = (shmem_integer_get__p_h) dlsym(tau_handle,"shmem_integer_get_"); 
    if (shmem_integer_get__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  (*shmem_integer_get__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_iget_
 **********************************************************/

void shmem_integer_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_integer_iget__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_integer_iget__p_h shmem_integer_iget__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_iget_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_iget__h == NULL)
	shmem_integer_iget__h = (shmem_integer_iget__p_h) dlsym(tau_handle,"shmem_integer_iget_"); 
    if (shmem_integer_iget__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a5), (*a6));
  (*shmem_integer_iget__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), sizeof(int)* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_iput_
 **********************************************************/

void shmem_integer_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_integer_iput__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_integer_iput__p_h shmem_integer_iput__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_iput_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_iput__h == NULL)
	shmem_integer_iput__h = (shmem_integer_iput__p_h) dlsym(tau_handle,"shmem_integer_iput_"); 
    if (shmem_integer_iput__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), sizeof(int)* (*a5));
  (*shmem_integer_iput__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_put_
 **********************************************************/

void shmem_integer_put_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_integer_put__p_h) (void *, void *, int *, int *);
  static shmem_integer_put__p_h shmem_integer_put__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_put_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_put__h == NULL)
	shmem_integer_put__h = (shmem_integer_put__p_h) dlsym(tau_handle,"shmem_integer_put_"); 
    if (shmem_integer_put__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)* (*a3));
  (*shmem_integer_put__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput128_
 **********************************************************/

void shmem_iput128_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput128__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput128__p_h shmem_iput128__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput128_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput128__h == NULL)
	shmem_iput128__h = (shmem_iput128__p_h) dlsym(tau_handle,"shmem_iput128_"); 
    if (shmem_iput128__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 16* (*a5));
  (*shmem_iput128__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput16_
 **********************************************************/

void shmem_iput16_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput16__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput16__p_h shmem_iput16__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput16_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput16__h == NULL)
	shmem_iput16__h = (shmem_iput16__p_h) dlsym(tau_handle,"shmem_iput16_"); 
    if (shmem_iput16__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 2* (*a5));
  (*shmem_iput16__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput32_
 **********************************************************/

void shmem_iput32_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput32__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput32__p_h shmem_iput32__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput32_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput32__h == NULL)
	shmem_iput32__h = (shmem_iput32__p_h) dlsym(tau_handle,"shmem_iput32_"); 
    if (shmem_iput32__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
  (*shmem_iput32__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput4_
 **********************************************************/

void shmem_iput4_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput4__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput4__p_h shmem_iput4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput4_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput4__h == NULL)
	shmem_iput4__h = (shmem_iput4__p_h) dlsym(tau_handle,"shmem_iput4_"); 
    if (shmem_iput4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
  (*shmem_iput4__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput64_
 **********************************************************/

void shmem_iput64_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput64__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput64__p_h shmem_iput64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput64_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput64__h == NULL)
	shmem_iput64__h = (shmem_iput64__p_h) dlsym(tau_handle,"shmem_iput64_"); 
    if (shmem_iput64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
  (*shmem_iput64__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput8_
 **********************************************************/

void shmem_iput8_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput8__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput8__p_h shmem_iput8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput8_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput8__h == NULL)
	shmem_iput8__h = (shmem_iput8__p_h) dlsym(tau_handle,"shmem_iput8_"); 
    if (shmem_iput8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
  (*shmem_iput8__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_get_
 **********************************************************/

void shmem_logical_get_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_logical_get__p_h) (void *, void *, int *, int *);
  static shmem_logical_get__p_h shmem_logical_get__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_get_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_get__h == NULL)
	shmem_logical_get__h = (shmem_logical_get__p_h) dlsym(tau_handle,"shmem_logical_get_"); 
    if (shmem_logical_get__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_logical_get__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_iget_
 **********************************************************/

void shmem_logical_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_logical_iget__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_logical_iget__p_h shmem_logical_iget__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_iget_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_iget__h == NULL)
	shmem_logical_iget__h = (shmem_logical_iget__p_h) dlsym(tau_handle,"shmem_logical_iget_"); 
    if (shmem_logical_iget__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
  (*shmem_logical_iget__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_iput_
 **********************************************************/

void shmem_logical_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_logical_iput__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_logical_iput__p_h shmem_logical_iput__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_iput_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_iput__h == NULL)
	shmem_logical_iput__h = (shmem_logical_iput__p_h) dlsym(tau_handle,"shmem_logical_iput_"); 
    if (shmem_logical_iput__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
  (*shmem_logical_iput__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_put_
 **********************************************************/

void shmem_logical_put_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_logical_put__p_h) (void *, void *, int *, int *);
  static shmem_logical_put__p_h shmem_logical_put__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_put_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_put__h == NULL)
	shmem_logical_put__h = (shmem_logical_put__p_h) dlsym(tau_handle,"shmem_logical_put_"); 
    if (shmem_logical_put__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_logical_put__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_my_pe_
 **********************************************************/

int shmem_my_pe_() {

  typedef int (*shmem_my_pe__p_h) ();
  static shmem_my_pe__p_h shmem_my_pe__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_my_pe_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_my_pe__h == NULL)
	shmem_my_pe__h = (shmem_my_pe__p_h) dlsym(tau_handle,"shmem_my_pe_"); 
    if (shmem_my_pe__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_my_pe__h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_n_pes_
 **********************************************************/

int shmem_n_pes_() {

  typedef int (*shmem_n_pes__p_h) ();
  static shmem_n_pes__p_h shmem_n_pes__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_n_pes_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_n_pes__h == NULL)
	shmem_n_pes__h = (shmem_n_pes__p_h) dlsym(tau_handle,"shmem_n_pes_"); 
    if (shmem_n_pes__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_n_pes__h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_pe_accessible_
 **********************************************************/

int shmem_pe_accessible_(int * a1) {

  typedef int (*shmem_pe_accessible__p_h) (int *);
  static shmem_pe_accessible__p_h shmem_pe_accessible__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_pe_accessible_(int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_pe_accessible__h == NULL)
	shmem_pe_accessible__h = (shmem_pe_accessible__p_h) dlsym(tau_handle,"shmem_pe_accessible_"); 
    if (shmem_pe_accessible__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_pe_accessible__h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_ptr_
 **********************************************************/

void shmem_ptr_(void * a1, int * a2) {

  typedef void (*shmem_ptr__p_h) (void *, int *);
  static shmem_ptr__p_h shmem_ptr__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_ptr_(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_ptr__h == NULL)
	shmem_ptr__h = (shmem_ptr__p_h) dlsym(tau_handle,"shmem_ptr_"); 
    if (shmem_ptr__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_ptr__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put128_
 **********************************************************/

void shmem_put128_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put128__p_h) (void *, void *, int *, int *);
  static shmem_put128__p_h shmem_put128__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put128_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put128__h == NULL)
	shmem_put128__h = (shmem_put128__p_h) dlsym(tau_handle,"shmem_put128_"); 
    if (shmem_put128__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 16* (*a3));
  (*shmem_put128__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put16_
 **********************************************************/

void shmem_put16_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put16__p_h) (void *, void *, int *, int *);
  static shmem_put16__p_h shmem_put16__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put16_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put16__h == NULL)
	shmem_put16__h = (shmem_put16__p_h) dlsym(tau_handle,"shmem_put16_"); 
    if (shmem_put16__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 2* (*a3));
  (*shmem_put16__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put32_
 **********************************************************/

void shmem_put32_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put32__p_h) (void *, void *, int *, int *);
  static shmem_put32__p_h shmem_put32__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put32_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put32__h == NULL)
	shmem_put32__h = (shmem_put32__p_h) dlsym(tau_handle,"shmem_put32_"); 
    if (shmem_put32__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
  (*shmem_put32__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put4_
 **********************************************************/

void shmem_put4_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put4__p_h) (void *, void *, int *, int *);
  static shmem_put4__p_h shmem_put4__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put4_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put4__h == NULL)
	shmem_put4__h = (shmem_put4__p_h) dlsym(tau_handle,"shmem_put4_"); 
    if (shmem_put4__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
  (*shmem_put4__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put64_
 **********************************************************/

void shmem_put64_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put64__p_h) (void *, void *, int *, int *);
  static shmem_put64__p_h shmem_put64__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put64_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put64__h == NULL)
	shmem_put64__h = (shmem_put64__p_h) dlsym(tau_handle,"shmem_put64_"); 
    if (shmem_put64__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
  (*shmem_put64__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put8_
 **********************************************************/

void shmem_put8_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put8__p_h) (void *, void *, int *, int *);
  static shmem_put8__p_h shmem_put8__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put8_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put8__h == NULL)
	shmem_put8__h = (shmem_put8__p_h) dlsym(tau_handle,"shmem_put8_"); 
    if (shmem_put8__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
  (*shmem_put8__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_putmem_
 **********************************************************/

void shmem_putmem_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_putmem__p_h) (void *, void *, int *, int *);
  static shmem_putmem__p_h shmem_putmem__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_putmem_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_putmem__h == NULL)
	shmem_putmem__h = (shmem_putmem__p_h) dlsym(tau_handle,"shmem_putmem_"); 
    if (shmem_putmem__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_putmem__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_quiet_
 **********************************************************/

void shmem_quiet_() {

  typedef void (*shmem_quiet__p_h) ();
  static shmem_quiet__p_h shmem_quiet__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_quiet_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_quiet__h == NULL)
	shmem_quiet__h = (shmem_quiet__p_h) dlsym(tau_handle,"shmem_quiet_"); 
    if (shmem_quiet__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_quiet__h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_max_to_all_
 **********************************************************/

void shmem_real16_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_max_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_max_to_all__p_h shmem_real16_max_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_max_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_max_to_all__h == NULL)
	shmem_real16_max_to_all__h = (shmem_real16_max_to_all__p_h) dlsym(tau_handle,"shmem_real16_max_to_all_"); 
    if (shmem_real16_max_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_max_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_min_to_all_
 **********************************************************/

void shmem_real16_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_min_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_min_to_all__p_h shmem_real16_min_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_min_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_min_to_all__h == NULL)
	shmem_real16_min_to_all__h = (shmem_real16_min_to_all__p_h) dlsym(tau_handle,"shmem_real16_min_to_all_"); 
    if (shmem_real16_min_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_min_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_prod_to_all_
 **********************************************************/

void shmem_real16_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_prod_to_all__p_h shmem_real16_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_prod_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_prod_to_all__h == NULL)
	shmem_real16_prod_to_all__h = (shmem_real16_prod_to_all__p_h) dlsym(tau_handle,"shmem_real16_prod_to_all_"); 
    if (shmem_real16_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_sum_to_all_
 **********************************************************/

void shmem_real16_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_sum_to_all__p_h shmem_real16_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_sum_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_sum_to_all__h == NULL)
	shmem_real16_sum_to_all__h = (shmem_real16_sum_to_all__p_h) dlsym(tau_handle,"shmem_real16_sum_to_all_"); 
    if (shmem_real16_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_max_to_all_
 **********************************************************/

void shmem_real4_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_max_to_all__p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_max_to_all__p_h shmem_real4_max_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_max_to_all_(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_max_to_all__h == NULL)
	shmem_real4_max_to_all__h = (shmem_real4_max_to_all__p_h) dlsym(tau_handle,"shmem_real4_max_to_all_"); 
    if (shmem_real4_max_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_max_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_min_to_all_
 **********************************************************/

void shmem_real4_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_min_to_all__p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_min_to_all__p_h shmem_real4_min_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_min_to_all_(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_min_to_all__h == NULL)
	shmem_real4_min_to_all__h = (shmem_real4_min_to_all__p_h) dlsym(tau_handle,"shmem_real4_min_to_all_"); 
    if (shmem_real4_min_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_min_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_prod_to_all_
 **********************************************************/

void shmem_real4_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_prod_to_all__p_h shmem_real4_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_prod_to_all_(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_prod_to_all__h == NULL)
	shmem_real4_prod_to_all__h = (shmem_real4_prod_to_all__p_h) dlsym(tau_handle,"shmem_real4_prod_to_all_"); 
    if (shmem_real4_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_sum_to_all_
 **********************************************************/

void shmem_real4_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_sum_to_all__p_h shmem_real4_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_sum_to_all_(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_sum_to_all__h == NULL)
	shmem_real4_sum_to_all__h = (shmem_real4_sum_to_all__p_h) dlsym(tau_handle,"shmem_real4_sum_to_all_"); 
    if (shmem_real4_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_swap_
 **********************************************************/

float shmem_real4_swap_(void * a1, float * a2, int * a3) {

  typedef float (*shmem_real4_swap__p_h) (void *, float *, int *);
  static shmem_real4_swap__p_h shmem_real4_swap__h = NULL;
  float retval = 0;
  TAU_PROFILE_TIMER(t,"float shmem_real4_swap_(void *, float *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_real4_swap__h == NULL)
	shmem_real4_swap__h = (shmem_real4_swap__p_h) dlsym(tau_handle,"shmem_real4_swap_"); 
    if (shmem_real4_swap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*1, (*a3));
  retval  =  (*shmem_real4_swap__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 4*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 4*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_real8_max_to_all_
 **********************************************************/

void shmem_real8_max_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_max_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_max_to_all__p_h shmem_real8_max_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_max_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_max_to_all__h == NULL)
	shmem_real8_max_to_all__h = (shmem_real8_max_to_all__p_h) dlsym(tau_handle,"shmem_real8_max_to_all_"); 
    if (shmem_real8_max_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_max_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_min_to_all_
 **********************************************************/

void shmem_real8_min_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_min_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_min_to_all__p_h shmem_real8_min_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_min_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_min_to_all__h == NULL)
	shmem_real8_min_to_all__h = (shmem_real8_min_to_all__p_h) dlsym(tau_handle,"shmem_real8_min_to_all_"); 
    if (shmem_real8_min_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_min_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_prod_to_all_
 **********************************************************/

void shmem_real8_prod_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_prod_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_prod_to_all__p_h shmem_real8_prod_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_prod_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_prod_to_all__h == NULL)
	shmem_real8_prod_to_all__h = (shmem_real8_prod_to_all__p_h) dlsym(tau_handle,"shmem_real8_prod_to_all_"); 
    if (shmem_real8_prod_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_prod_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_sum_to_all_
 **********************************************************/

void shmem_real8_sum_to_all_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_sum_to_all__p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_sum_to_all__p_h shmem_real8_sum_to_all__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_sum_to_all_(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_sum_to_all__h == NULL)
	shmem_real8_sum_to_all__h = (shmem_real8_sum_to_all__p_h) dlsym(tau_handle,"shmem_real8_sum_to_all_"); 
    if (shmem_real8_sum_to_all__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_sum_to_all__h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_swap_
 **********************************************************/

double shmem_real8_swap_(void * a1, double * a2, int * a3) {

  typedef double (*shmem_real8_swap__p_h) (void *, double *, int *);
  static shmem_real8_swap__p_h shmem_real8_swap__h = NULL;
  double retval = 0;
  TAU_PROFILE_TIMER(t,"double shmem_real8_swap_(void *, double *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_real8_swap__h == NULL)
	shmem_real8_swap__h = (shmem_real8_swap__p_h) dlsym(tau_handle,"shmem_real8_swap_"); 
    if (shmem_real8_swap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*1, (*a3));
  retval  =  (*shmem_real8_swap__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 8*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 8*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_real_get_
 **********************************************************/

void shmem_real_get_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_real_get__p_h) (void *, void *, int *, int *);
  static shmem_real_get__p_h shmem_real_get__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_get_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_get__h == NULL)
	shmem_real_get__h = (shmem_real_get__p_h) dlsym(tau_handle,"shmem_real_get_"); 
    if (shmem_real_get__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_real_get__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real_iget_
 **********************************************************/

void shmem_real_iget_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_real_iget__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_real_iget__p_h shmem_real_iget__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_iget_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_iget__h == NULL)
	shmem_real_iget__h = (shmem_real_iget__p_h) dlsym(tau_handle,"shmem_real_iget_"); 
    if (shmem_real_iget__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
  (*shmem_real_iget__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real_iput_
 **********************************************************/

void shmem_real_iput_(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_real_iput__p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_real_iput__p_h shmem_real_iput__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_iput_(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_iput__h == NULL)
	shmem_real_iput__h = (shmem_real_iput__p_h) dlsym(tau_handle,"shmem_real_iput_"); 
    if (shmem_real_iput__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
  (*shmem_real_iput__h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real_put_
 **********************************************************/

void shmem_real_put_(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_real_put__p_h) (void *, void *, int *, int *);
  static shmem_real_put__p_h shmem_real_put__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_put_(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_put__h == NULL)
	shmem_real_put__h = (shmem_real_put__p_h) dlsym(tau_handle,"shmem_real_put_"); 
    if (shmem_real_put__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_real_put__h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_cache_inv_
 **********************************************************/

void shmem_set_cache_inv_() {

  typedef void (*shmem_set_cache_inv__p_h) ();
  static shmem_set_cache_inv__p_h shmem_set_cache_inv__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_cache_inv_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_cache_inv__h == NULL)
	shmem_set_cache_inv__h = (shmem_set_cache_inv__p_h) dlsym(tau_handle,"shmem_set_cache_inv_"); 
    if (shmem_set_cache_inv__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_cache_inv__h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_cache_line_inv_
 **********************************************************/

void shmem_set_cache_line_inv_(void * a1) {

  typedef void (*shmem_set_cache_line_inv__p_h) (void *);
  static shmem_set_cache_line_inv__p_h shmem_set_cache_line_inv__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_cache_line_inv_(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_cache_line_inv__h == NULL)
	shmem_set_cache_line_inv__h = (shmem_set_cache_line_inv__p_h) dlsym(tau_handle,"shmem_set_cache_line_inv_"); 
    if (shmem_set_cache_line_inv__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_cache_line_inv__h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_lock_
 **********************************************************/

void shmem_set_lock_(long * a1) {

  typedef void (*shmem_set_lock__p_h) (long *);
  static shmem_set_lock__p_h shmem_set_lock__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_lock_(long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_lock__h == NULL)
	shmem_set_lock__h = (shmem_set_lock__p_h) dlsym(tau_handle,"shmem_set_lock_"); 
    if (shmem_set_lock__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_lock__h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_swap_
 **********************************************************/

int shmem_swap_(void * a1, int * a2, int * a3) {

  typedef int (*shmem_swap__p_h) (void *, int *, int *);
  static shmem_swap__p_h shmem_swap__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_swap_(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_swap__h == NULL)
	shmem_swap__h = (shmem_swap__p_h) dlsym(tau_handle,"shmem_swap_"); 
    if (shmem_swap__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 1, (*a3));
  retval  =  (*shmem_swap__h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_test_lock_
 **********************************************************/

int shmem_test_lock_(long * a1) {

  typedef int (*shmem_test_lock__p_h) (long *);
  static shmem_test_lock__p_h shmem_test_lock__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock_(long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_test_lock__h == NULL)
	shmem_test_lock__h = (shmem_test_lock__p_h) dlsym(tau_handle,"shmem_test_lock_"); 
    if (shmem_test_lock__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_test_lock__h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_udcflush_
 **********************************************************/

void shmem_udcflush_() {

  typedef void (*shmem_udcflush__p_h) ();
  static shmem_udcflush__p_h shmem_udcflush__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_udcflush_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_udcflush__h == NULL)
	shmem_udcflush__h = (shmem_udcflush__p_h) dlsym(tau_handle,"shmem_udcflush_"); 
    if (shmem_udcflush__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_udcflush__h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_udcflush_line_
 **********************************************************/

void shmem_udcflush_line_(void * a1) {

  typedef void (*shmem_udcflush_line__p_h) (void *);
  static shmem_udcflush_line__p_h shmem_udcflush_line__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_udcflush_line_(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_udcflush_line__h == NULL)
	shmem_udcflush_line__h = (shmem_udcflush_line__p_h) dlsym(tau_handle,"shmem_udcflush_line_"); 
    if (shmem_udcflush_line__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_udcflush_line__h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_wait_
 **********************************************************/

void shmem_wait_(long * a1, long * a2) {

  typedef void (*shmem_wait__p_h) (long *, long *);
  static shmem_wait__p_h shmem_wait__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_wait_(long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_wait__h == NULL)
	shmem_wait__h = (shmem_wait__p_h) dlsym(tau_handle,"shmem_wait_"); 
    if (shmem_wait__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_wait__h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_wait_until_
 **********************************************************/

void shmem_wait_until_(int * a1, int * a2, int * a3) {

  typedef void (*shmem_wait_until__p_h) (int *, int *, int *);
  static shmem_wait_until__p_h shmem_wait_until__h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_wait_until_(int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_wait_until__h == NULL)
	shmem_wait_until__h = (shmem_wait_until__p_h) dlsym(tau_handle,"shmem_wait_until_"); 
    if (shmem_wait_until__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_wait_until__h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   start_pes_
 **********************************************************/

void start_pes_(int * a1) {

  typedef void (*start_pes__p_h) (int *);
  static start_pes__p_h start_pes__h = NULL;
  TAU_PROFILE_TIMER(t,"void start_pes_(int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (start_pes__h == NULL)
	start_pes__h = (start_pes__p_h) dlsym(tau_handle,"start_pes_"); 
    if (start_pes__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*start_pes__h)( a1);

#ifdef TAU_PSHMEM_SGI_MPT
  tau_totalnodes(1,pshmem_n_pes());
  TAU_PROFILE_SET_NODE(pshmem_my_pe());
#else /* TAU_PSHMEM_SGI_MPT */
  tau_totalnodes(1,_shmem_n_pes());
  TAU_PROFILE_SET_NODE(_shmem_my_pe());
#endif /* TAU_PSHMEM_SGI_MPT */

  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_addr_accessible__
 **********************************************************/

void shmem_addr_accessible__(void * a1, int * a2) {

  typedef void (*shmem_addr_accessible___p_h) (void *, int *);
  static shmem_addr_accessible___p_h shmem_addr_accessible___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_addr_accessible__(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_addr_accessible___h == NULL)
	shmem_addr_accessible___h = (shmem_addr_accessible___p_h) dlsym(tau_handle,"shmem_addr_accessible__"); 
    if (shmem_addr_accessible___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_addr_accessible___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_barrier__
 **********************************************************/

void shmem_barrier__(int * a1, int * a2, int * a3, long * a4) {

  typedef void (*shmem_barrier___p_h) (int *, int *, int *, long *);
  static shmem_barrier___p_h shmem_barrier___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_barrier__(int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_barrier___h == NULL)
	shmem_barrier___h = (shmem_barrier___p_h) dlsym(tau_handle,"shmem_barrier__"); 
    if (shmem_barrier___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_barrier___h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_barrier_all__
 **********************************************************/

void shmem_barrier_all__() {

  typedef void (*shmem_barrier_all___p_h) ();
  static shmem_barrier_all___p_h shmem_barrier_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_barrier_all__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_barrier_all___h == NULL)
	shmem_barrier_all___h = (shmem_barrier_all___p_h) dlsym(tau_handle,"shmem_barrier_all__"); 
    if (shmem_barrier_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_barrier_all___h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_barrier_ps__
 **********************************************************/

void shmem_barrier_ps__(int * a1, int * a2, int * a3, long * a4) {

  typedef void (*shmem_barrier_ps___p_h) (int *, int *, int *, long *);
  static shmem_barrier_ps___p_h shmem_barrier_ps___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_barrier_ps__(int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_barrier_ps___h == NULL)
	shmem_barrier_ps___h = (shmem_barrier_ps___p_h) dlsym(tau_handle,"shmem_barrier_ps__"); 
    if (shmem_barrier_ps___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_barrier_ps___h)( a1,  a2,  a3,  a4);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast32__
 **********************************************************/

void shmem_broadcast32__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast32___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast32___p_h shmem_broadcast32___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast32__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast32___h == NULL)
	shmem_broadcast32___h = (shmem_broadcast32___p_h) dlsym(tau_handle,"shmem_broadcast32__"); 
    if (shmem_broadcast32___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast32___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast4__
 **********************************************************/

void shmem_broadcast4__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast4___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast4___p_h shmem_broadcast4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast4__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast4___h == NULL)
	shmem_broadcast4___h = (shmem_broadcast4___p_h) dlsym(tau_handle,"shmem_broadcast4__"); 
    if (shmem_broadcast4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast4___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast64__
 **********************************************************/

void shmem_broadcast64__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast64___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast64___p_h shmem_broadcast64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast64__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast64___h == NULL)
	shmem_broadcast64___h = (shmem_broadcast64___p_h) dlsym(tau_handle,"shmem_broadcast64__"); 
    if (shmem_broadcast64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast64___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_broadcast8__
 **********************************************************/

void shmem_broadcast8__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_broadcast8___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_broadcast8___p_h shmem_broadcast8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_broadcast8__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_broadcast8___h == NULL)
	shmem_broadcast8___h = (shmem_broadcast8___p_h) dlsym(tau_handle,"shmem_broadcast8__"); 
    if (shmem_broadcast8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_broadcast8___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_character_get__
 **********************************************************/

void shmem_character_get__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_character_get___p_h) (void *, void *, int *, int *);
  static shmem_character_get___p_h shmem_character_get___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_character_get__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_character_get___h == NULL)
	shmem_character_get___h = (shmem_character_get___p_h) dlsym(tau_handle,"shmem_character_get__"); 
    if (shmem_character_get___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)* (*a3), (*a4));
  (*shmem_character_get___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(char)* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_character_put__
 **********************************************************/

void shmem_character_put__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_character_put___p_h) (void *, void *, int *, int *);
  static shmem_character_put___p_h shmem_character_put___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_character_put__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_character_put___h == NULL)
	shmem_character_put___h = (shmem_character_put___p_h) dlsym(tau_handle,"shmem_character_put__"); 
    if (shmem_character_put___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(char)* (*a3));
  (*shmem_character_put___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_cache_inv__
 **********************************************************/

void shmem_clear_cache_inv__() {

  typedef void (*shmem_clear_cache_inv___p_h) ();
  static shmem_clear_cache_inv___p_h shmem_clear_cache_inv___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_inv__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_cache_inv___h == NULL)
	shmem_clear_cache_inv___h = (shmem_clear_cache_inv___p_h) dlsym(tau_handle,"shmem_clear_cache_inv__"); 
    if (shmem_clear_cache_inv___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_cache_inv___h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_cache_line_inv__
 **********************************************************/

void shmem_clear_cache_line_inv__(void * a1) {

  typedef void (*shmem_clear_cache_line_inv___p_h) (void *);
  static shmem_clear_cache_line_inv___p_h shmem_clear_cache_line_inv___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_line_inv__(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_cache_line_inv___h == NULL)
	shmem_clear_cache_line_inv___h = (shmem_clear_cache_line_inv___p_h) dlsym(tau_handle,"shmem_clear_cache_line_inv__"); 
    if (shmem_clear_cache_line_inv___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_cache_line_inv___h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_lock__
 **********************************************************/

void shmem_clear_lock__(long * a1) {

  typedef void (*shmem_clear_lock___p_h) (long *);
  static shmem_clear_lock___p_h shmem_clear_lock___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_lock__(long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_lock___h == NULL)
	shmem_clear_lock___h = (shmem_clear_lock___p_h) dlsym(tau_handle,"shmem_clear_lock__"); 
    if (shmem_clear_lock___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_lock___h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_collect4__
 **********************************************************/

void shmem_collect4__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_collect4___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_collect4___p_h shmem_collect4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_collect4__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_collect4___h == NULL)
	shmem_collect4___h = (shmem_collect4___p_h) dlsym(tau_handle,"shmem_collect4__"); 
    if (shmem_collect4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_collect4___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_collect64__
 **********************************************************/

void shmem_collect64__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_collect64___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_collect64___p_h shmem_collect64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_collect64__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_collect64___h == NULL)
	shmem_collect64___h = (shmem_collect64___p_h) dlsym(tau_handle,"shmem_collect64__"); 
    if (shmem_collect64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_collect64___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_collect8__
 **********************************************************/

void shmem_collect8__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_collect8___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_collect8___p_h shmem_collect8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_collect8__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_collect8___h == NULL)
	shmem_collect8___h = (shmem_collect8___p_h) dlsym(tau_handle,"shmem_collect8__"); 
    if (shmem_collect8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_collect8___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp4_prod_to_all__
 **********************************************************/

void shmem_comp4_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_comp4_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_comp4_prod_to_all___p_h shmem_comp4_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp4_prod_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp4_prod_to_all___h == NULL)
	shmem_comp4_prod_to_all___h = (shmem_comp4_prod_to_all___p_h) dlsym(tau_handle,"shmem_comp4_prod_to_all__"); 
    if (shmem_comp4_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp4_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp4_sum_to_all__
 **********************************************************/

void shmem_comp4_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_comp4_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_comp4_sum_to_all___p_h shmem_comp4_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp4_sum_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp4_sum_to_all___h == NULL)
	shmem_comp4_sum_to_all___h = (shmem_comp4_sum_to_all___p_h) dlsym(tau_handle,"shmem_comp4_sum_to_all__"); 
    if (shmem_comp4_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp4_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp8_prod_to_all__
 **********************************************************/

void shmem_comp8_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_comp8_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_comp8_prod_to_all___p_h shmem_comp8_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp8_prod_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp8_prod_to_all___h == NULL)
	shmem_comp8_prod_to_all___h = (shmem_comp8_prod_to_all___p_h) dlsym(tau_handle,"shmem_comp8_prod_to_all__"); 
    if (shmem_comp8_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp8_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_comp8_sum_to_all__
 **********************************************************/

void shmem_comp8_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_comp8_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_comp8_sum_to_all___p_h shmem_comp8_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_comp8_sum_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_comp8_sum_to_all___h == NULL)
	shmem_comp8_sum_to_all___h = (shmem_comp8_sum_to_all___p_h) dlsym(tau_handle,"shmem_comp8_sum_to_all__"); 
    if (shmem_comp8_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_comp8_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_get__
 **********************************************************/

void shmem_complex_get__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_complex_get___p_h) (void *, void *, int *, int *);
  static shmem_complex_get___p_h shmem_complex_get___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_get__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_get___h == NULL)
	shmem_complex_get___h = (shmem_complex_get___p_h) dlsym(tau_handle,"shmem_complex_get__"); 
    if (shmem_complex_get___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_complex_get___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_iget__
 **********************************************************/

void shmem_complex_iget__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_complex_iget___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_complex_iget___p_h shmem_complex_iget___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_iget__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_iget___h == NULL)
	shmem_complex_iget___h = (shmem_complex_iget___p_h) dlsym(tau_handle,"shmem_complex_iget__"); 
    if (shmem_complex_iget___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
  (*shmem_complex_iget___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_iput__
 **********************************************************/

void shmem_complex_iput__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_complex_iput___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_complex_iput___p_h shmem_complex_iput___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_iput__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_iput___h == NULL)
	shmem_complex_iput___h = (shmem_complex_iput___p_h) dlsym(tau_handle,"shmem_complex_iput__"); 
    if (shmem_complex_iput___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
  (*shmem_complex_iput___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_complex_put__
 **********************************************************/

void shmem_complex_put__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_complex_put___p_h) (void *, void *, int *, int *);
  static shmem_complex_put___p_h shmem_complex_put___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_complex_put__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_complex_put___h == NULL)
	shmem_complex_put___h = (shmem_complex_put___p_h) dlsym(tau_handle,"shmem_complex_put__"); 
    if (shmem_complex_put___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_complex_put___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_get__
 **********************************************************/

void shmem_double_get__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_double_get___p_h) (void *, void *, int *, int *);
  static shmem_double_get___p_h shmem_double_get___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_get__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_get___h == NULL)
	shmem_double_get___h = (shmem_double_get___p_h) dlsym(tau_handle,"shmem_double_get__"); 
    if (shmem_double_get___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  (*shmem_double_get___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(double)* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_iget__
 **********************************************************/

void shmem_double_iget__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_double_iget___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_double_iget___p_h shmem_double_iget___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_iget__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_iget___h == NULL)
	shmem_double_iget___h = (shmem_double_iget___p_h) dlsym(tau_handle,"shmem_double_iget__"); 
    if (shmem_double_iget___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)* (*a5), (*a6));
  (*shmem_double_iget___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), sizeof(double)* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_iput__
 **********************************************************/

void shmem_double_iput__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_double_iput___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_double_iput___p_h shmem_double_iput___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_iput__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_iput___h == NULL)
	shmem_double_iput___h = (shmem_double_iput___p_h) dlsym(tau_handle,"shmem_double_iput__"); 
    if (shmem_double_iput___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), sizeof(double)* (*a5));
  (*shmem_double_iput___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_double_put__
 **********************************************************/

void shmem_double_put__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_double_put___p_h) (void *, void *, int *, int *);
  static shmem_double_put___p_h shmem_double_put___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_double_put__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_double_put___h == NULL)
	shmem_double_put___h = (shmem_double_put___p_h) dlsym(tau_handle,"shmem_double_put__"); 
    if (shmem_double_put___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(double)* (*a3));
  (*shmem_double_put___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect32__
 **********************************************************/

void shmem_fcollect32__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect32___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect32___p_h shmem_fcollect32___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect32__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect32___h == NULL)
	shmem_fcollect32___h = (shmem_fcollect32___p_h) dlsym(tau_handle,"shmem_fcollect32__"); 
    if (shmem_fcollect32___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect32___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect4__
 **********************************************************/

void shmem_fcollect4__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect4___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect4___p_h shmem_fcollect4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect4__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect4___h == NULL)
	shmem_fcollect4___h = (shmem_fcollect4___p_h) dlsym(tau_handle,"shmem_fcollect4__"); 
    if (shmem_fcollect4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect4___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect64__
 **********************************************************/

void shmem_fcollect64__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect64___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect64___p_h shmem_fcollect64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect64__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect64___h == NULL)
	shmem_fcollect64___h = (shmem_fcollect64___p_h) dlsym(tau_handle,"shmem_fcollect64__"); 
    if (shmem_fcollect64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect64___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fcollect8__
 **********************************************************/

void shmem_fcollect8__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7) {

  typedef void (*shmem_fcollect8___p_h) (void *, void *, int *, int *, int *, int *, long *);
  static shmem_fcollect8___p_h shmem_fcollect8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fcollect8__(void *, void *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fcollect8___h == NULL)
	shmem_fcollect8___h = (shmem_fcollect8___p_h) dlsym(tau_handle,"shmem_fcollect8__"); 
    if (shmem_fcollect8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fcollect8___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_fence__
 **********************************************************/

void shmem_fence__() {

  typedef void (*shmem_fence___p_h) ();
  static shmem_fence___p_h shmem_fence___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_fence__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_fence___h == NULL)
	shmem_fence___h = (shmem_fence___p_h) dlsym(tau_handle,"shmem_fence__"); 
    if (shmem_fence___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_fence___h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get128__
 **********************************************************/

void shmem_get128__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get128___p_h) (void *, void *, int *, int *);
  static shmem_get128___p_h shmem_get128___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get128__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get128___h == NULL)
	shmem_get128___h = (shmem_get128___p_h) dlsym(tau_handle,"shmem_get128__"); 
    if (shmem_get128___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a3), (*a4));
  (*shmem_get128___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 16* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get16__
 **********************************************************/

void shmem_get16__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get16___p_h) (void *, void *, int *, int *);
  static shmem_get16___p_h shmem_get16___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get16__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get16___h == NULL)
	shmem_get16___h = (shmem_get16___p_h) dlsym(tau_handle,"shmem_get16__"); 
    if (shmem_get16___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a3), (*a4));
  (*shmem_get16___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 2* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get32__
 **********************************************************/

void shmem_get32__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get32___p_h) (void *, void *, int *, int *);
  static shmem_get32___p_h shmem_get32___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get32__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get32___h == NULL)
	shmem_get32___h = (shmem_get32___p_h) dlsym(tau_handle,"shmem_get32__"); 
    if (shmem_get32___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
  (*shmem_get32___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get4__
 **********************************************************/

void shmem_get4__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get4___p_h) (void *, void *, int *, int *);
  static shmem_get4___p_h shmem_get4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get4__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get4___h == NULL)
	shmem_get4___h = (shmem_get4___p_h) dlsym(tau_handle,"shmem_get4__"); 
    if (shmem_get4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a3), (*a4));
  (*shmem_get4___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 4* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get64__
 **********************************************************/

void shmem_get64__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get64___p_h) (void *, void *, int *, int *);
  static shmem_get64___p_h shmem_get64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get64__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get64___h == NULL)
	shmem_get64___h = (shmem_get64___p_h) dlsym(tau_handle,"shmem_get64__"); 
    if (shmem_get64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
  (*shmem_get64___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_get8__
 **********************************************************/

void shmem_get8__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_get8___p_h) (void *, void *, int *, int *);
  static shmem_get8___p_h shmem_get8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_get8__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_get8___h == NULL)
	shmem_get8___h = (shmem_get8___p_h) dlsym(tau_handle,"shmem_get8__"); 
    if (shmem_get8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a3), (*a4));
  (*shmem_get8___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), 8* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_getmem__
 **********************************************************/

void shmem_getmem__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_getmem___p_h) (void *, void *, int *, int *);
  static shmem_getmem___p_h shmem_getmem___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_getmem__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_getmem___h == NULL)
	shmem_getmem___h = (shmem_getmem___p_h) dlsym(tau_handle,"shmem_getmem__"); 
    if (shmem_getmem___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_getmem___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_group_create_strided__
 **********************************************************/

void shmem_group_create_strided__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_group_create_strided___p_h) (int *, int *, int *, int *, int *, int *);
  static shmem_group_create_strided___p_h shmem_group_create_strided___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_group_create_strided__(int *, int *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_group_create_strided___h == NULL)
	shmem_group_create_strided___h = (shmem_group_create_strided___p_h) dlsym(tau_handle,"shmem_group_create_strided__"); 
    if (shmem_group_create_strided___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_group_create_strided___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_group_delete__
 **********************************************************/

void shmem_group_delete__(int * a1) {

  typedef void (*shmem_group_delete___p_h) (int *);
  static shmem_group_delete___p_h shmem_group_delete___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_group_delete__(int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_group_delete___h == NULL)
	shmem_group_delete___h = (shmem_group_delete___p_h) dlsym(tau_handle,"shmem_group_delete__"); 
    if (shmem_group_delete___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_group_delete___h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget128__
 **********************************************************/

void shmem_iget128__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget128___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget128___p_h shmem_iget128___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget128__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget128___h == NULL)
	shmem_iget128___h = (shmem_iget128___p_h) dlsym(tau_handle,"shmem_iget128__"); 
    if (shmem_iget128___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16* (*a5), (*a6));
  (*shmem_iget128___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 16* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget16__
 **********************************************************/

void shmem_iget16__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget16___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget16___p_h shmem_iget16___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget16__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget16___h == NULL)
	shmem_iget16___h = (shmem_iget16___p_h) dlsym(tau_handle,"shmem_iget16__"); 
    if (shmem_iget16___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2* (*a5), (*a6));
  (*shmem_iget16___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 2* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget32__
 **********************************************************/

void shmem_iget32__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget32___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget32___p_h shmem_iget32___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget32__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget32___h == NULL)
	shmem_iget32___h = (shmem_iget32___p_h) dlsym(tau_handle,"shmem_iget32__"); 
    if (shmem_iget32___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
  (*shmem_iget32___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget4__
 **********************************************************/

void shmem_iget4__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget4___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget4___p_h shmem_iget4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget4__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget4___h == NULL)
	shmem_iget4___h = (shmem_iget4___p_h) dlsym(tau_handle,"shmem_iget4__"); 
    if (shmem_iget4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4* (*a5), (*a6));
  (*shmem_iget4___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 4* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget64__
 **********************************************************/

void shmem_iget64__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget64___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget64___p_h shmem_iget64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget64__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget64___h == NULL)
	shmem_iget64___h = (shmem_iget64___p_h) dlsym(tau_handle,"shmem_iget64__"); 
    if (shmem_iget64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
  (*shmem_iget64___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iget8__
 **********************************************************/

void shmem_iget8__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iget8___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iget8___p_h shmem_iget8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iget8__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iget8___h == NULL)
	shmem_iget8___h = (shmem_iget8___p_h) dlsym(tau_handle,"shmem_iget8__"); 
    if (shmem_iget8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8* (*a5), (*a6));
  (*shmem_iget8___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), 8* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_and_to_all__
 **********************************************************/

void shmem_int2_and_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_and_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_and_to_all___p_h shmem_int2_and_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_and_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_and_to_all___h == NULL)
	shmem_int2_and_to_all___h = (shmem_int2_and_to_all___p_h) dlsym(tau_handle,"shmem_int2_and_to_all__"); 
    if (shmem_int2_and_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_and_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_max_to_all__
 **********************************************************/

void shmem_int2_max_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_max_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_max_to_all___p_h shmem_int2_max_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_max_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_max_to_all___h == NULL)
	shmem_int2_max_to_all___h = (shmem_int2_max_to_all___p_h) dlsym(tau_handle,"shmem_int2_max_to_all__"); 
    if (shmem_int2_max_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_max_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_min_to_all__
 **********************************************************/

void shmem_int2_min_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_min_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_min_to_all___p_h shmem_int2_min_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_min_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_min_to_all___h == NULL)
	shmem_int2_min_to_all___h = (shmem_int2_min_to_all___p_h) dlsym(tau_handle,"shmem_int2_min_to_all__"); 
    if (shmem_int2_min_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_min_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_or_to_all__
 **********************************************************/

void shmem_int2_or_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_or_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_or_to_all___p_h shmem_int2_or_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_or_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_or_to_all___h == NULL)
	shmem_int2_or_to_all___h = (shmem_int2_or_to_all___p_h) dlsym(tau_handle,"shmem_int2_or_to_all__"); 
    if (shmem_int2_or_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_or_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_prod_to_all__
 **********************************************************/

void shmem_int2_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_prod_to_all___p_h shmem_int2_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_prod_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_prod_to_all___h == NULL)
	shmem_int2_prod_to_all___h = (shmem_int2_prod_to_all___p_h) dlsym(tau_handle,"shmem_int2_prod_to_all__"); 
    if (shmem_int2_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_sum_to_all__
 **********************************************************/

void shmem_int2_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_sum_to_all___p_h shmem_int2_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_sum_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_sum_to_all___h == NULL)
	shmem_int2_sum_to_all___h = (shmem_int2_sum_to_all___p_h) dlsym(tau_handle,"shmem_int2_sum_to_all__"); 
    if (shmem_int2_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int2_xor_to_all__
 **********************************************************/

void shmem_int2_xor_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int2_xor_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int2_xor_to_all___p_h shmem_int2_xor_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int2_xor_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int2_xor_to_all___h == NULL)
	shmem_int2_xor_to_all___h = (shmem_int2_xor_to_all___p_h) dlsym(tau_handle,"shmem_int2_xor_to_all__"); 
    if (shmem_int2_xor_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int2_xor_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_add__
 **********************************************************/

void shmem_int4_add__(void * a1, int * a2, int * a3) {

  typedef void (*shmem_int4_add___p_h) (void *, int *, int *);
  static shmem_int4_add___p_h shmem_int4_add___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_add__(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_add___h == NULL)
	shmem_int4_add___h = (shmem_int4_add___p_h) dlsym(tau_handle,"shmem_int4_add__"); 
    if (shmem_int4_add___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_add___h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_and_to_all__
 **********************************************************/

void shmem_int4_and_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_and_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_and_to_all___p_h shmem_int4_and_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_and_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_and_to_all___h == NULL)
	shmem_int4_and_to_all___h = (shmem_int4_and_to_all___p_h) dlsym(tau_handle,"shmem_int4_and_to_all__"); 
    if (shmem_int4_and_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_and_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_cswap__
 **********************************************************/

int shmem_int4_cswap__(int * a1, int * a2, int * a3, int * a4) {

  typedef int (*shmem_int4_cswap___p_h) (int *, int *, int *, int *);
  static shmem_int4_cswap___p_h shmem_int4_cswap___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_cswap__(int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_cswap___h == NULL)
	shmem_int4_cswap___h = (shmem_int4_cswap___p_h) dlsym(tau_handle,"shmem_int4_cswap__"); 
    if (shmem_int4_cswap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a4));
  retval  =  (*shmem_int4_cswap___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)*1);
  if (retval == (*a2)) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a4));
  }
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_fadd__
 **********************************************************/

int shmem_int4_fadd__(void * a1, int * a2, int * a3) {

  typedef int (*shmem_int4_fadd___p_h) (void *, int *, int *);
  static shmem_int4_fadd___p_h shmem_int4_fadd___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_fadd__(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_fadd___h == NULL)
	shmem_int4_fadd___h = (shmem_int4_fadd___p_h) dlsym(tau_handle,"shmem_int4_fadd__"); 
    if (shmem_int4_fadd___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int4_fadd___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_finc__
 **********************************************************/

int shmem_int4_finc__(void * a1, int * a2) {

  typedef int (*shmem_int4_finc___p_h) (void *, int *);
  static shmem_int4_finc___p_h shmem_int4_finc___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_finc__(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_finc___h == NULL)
	shmem_int4_finc___h = (shmem_int4_finc___p_h) dlsym(tau_handle,"shmem_int4_finc__"); 
    if (shmem_int4_finc___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a2));
  retval  =  (*shmem_int4_finc___h)( a1,  a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a2), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a2), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a2));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_inc__
 **********************************************************/

void shmem_int4_inc__(void * a1, int * a2) {

  typedef void (*shmem_int4_inc___p_h) (void *, int *);
  static shmem_int4_inc___p_h shmem_int4_inc___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_inc__(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_inc___h == NULL)
	shmem_int4_inc___h = (shmem_int4_inc___p_h) dlsym(tau_handle,"shmem_int4_inc__"); 
    if (shmem_int4_inc___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_inc___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_max_to_all__
 **********************************************************/

void shmem_int4_max_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_max_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_max_to_all___p_h shmem_int4_max_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_max_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_max_to_all___h == NULL)
	shmem_int4_max_to_all___h = (shmem_int4_max_to_all___p_h) dlsym(tau_handle,"shmem_int4_max_to_all__"); 
    if (shmem_int4_max_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_max_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_min_to_all__
 **********************************************************/

void shmem_int4_min_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_min_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_min_to_all___p_h shmem_int4_min_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_min_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_min_to_all___h == NULL)
	shmem_int4_min_to_all___h = (shmem_int4_min_to_all___p_h) dlsym(tau_handle,"shmem_int4_min_to_all__"); 
    if (shmem_int4_min_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_min_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_or_to_all__
 **********************************************************/

void shmem_int4_or_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_or_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_or_to_all___p_h shmem_int4_or_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_or_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_or_to_all___h == NULL)
	shmem_int4_or_to_all___h = (shmem_int4_or_to_all___p_h) dlsym(tau_handle,"shmem_int4_or_to_all__"); 
    if (shmem_int4_or_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_or_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_prod_to_all__
 **********************************************************/

void shmem_int4_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_prod_to_all___p_h shmem_int4_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_prod_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_prod_to_all___h == NULL)
	shmem_int4_prod_to_all___h = (shmem_int4_prod_to_all___p_h) dlsym(tau_handle,"shmem_int4_prod_to_all__"); 
    if (shmem_int4_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_sum_to_all__
 **********************************************************/

void shmem_int4_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_sum_to_all___p_h shmem_int4_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_sum_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_sum_to_all___h == NULL)
	shmem_int4_sum_to_all___h = (shmem_int4_sum_to_all___p_h) dlsym(tau_handle,"shmem_int4_sum_to_all__"); 
    if (shmem_int4_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_swap__
 **********************************************************/

int shmem_int4_swap__(void * a1, int * a2, int * a3) {

  typedef int (*shmem_int4_swap___p_h) (void *, int *, int *);
  static shmem_int4_swap___p_h shmem_int4_swap___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_int4_swap__(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int4_swap___h == NULL)
	shmem_int4_swap___h = (shmem_int4_swap___p_h) dlsym(tau_handle,"shmem_int4_swap__"); 
    if (shmem_int4_swap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int4_swap___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int4_wait__
 **********************************************************/

void shmem_int4_wait__(int * a1, int * a2) {

  typedef void (*shmem_int4_wait___p_h) (int *, int *);
  static shmem_int4_wait___p_h shmem_int4_wait___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_wait__(int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_wait___h == NULL)
	shmem_int4_wait___h = (shmem_int4_wait___p_h) dlsym(tau_handle,"shmem_int4_wait__"); 
    if (shmem_int4_wait___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_wait___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_wait_until__
 **********************************************************/

void shmem_int4_wait_until__(int * a1, int * a2, int * a3) {

  typedef void (*shmem_int4_wait_until___p_h) (int *, int *, int *);
  static shmem_int4_wait_until___p_h shmem_int4_wait_until___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_wait_until__(int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_wait_until___h == NULL)
	shmem_int4_wait_until___h = (shmem_int4_wait_until___p_h) dlsym(tau_handle,"shmem_int4_wait_until__"); 
    if (shmem_int4_wait_until___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_wait_until___h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int4_xor_to_all__
 **********************************************************/

void shmem_int4_xor_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8) {

  typedef void (*shmem_int4_xor_to_all___p_h) (void *, void *, int *, int *, int *, int *, int *, long *);
  static shmem_int4_xor_to_all___p_h shmem_int4_xor_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int4_xor_to_all__(void *, void *, int *, int *, int *, int *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int4_xor_to_all___h == NULL)
	shmem_int4_xor_to_all___h = (shmem_int4_xor_to_all___p_h) dlsym(tau_handle,"shmem_int4_xor_to_all__"); 
    if (shmem_int4_xor_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int4_xor_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_add__
 **********************************************************/

void shmem_int8_add__(void * a1, long * a2, int * a3) {

  typedef void (*shmem_int8_add___p_h) (void *, long *, int *);
  static shmem_int8_add___p_h shmem_int8_add___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_add__(void *, long *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_add___h == NULL)
	shmem_int8_add___h = (shmem_int8_add___p_h) dlsym(tau_handle,"shmem_int8_add__"); 
    if (shmem_int8_add___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_add___h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_and_to_all__
 **********************************************************/

void shmem_int8_and_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_and_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_and_to_all___p_h shmem_int8_and_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_and_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_and_to_all___h == NULL)
	shmem_int8_and_to_all___h = (shmem_int8_and_to_all___p_h) dlsym(tau_handle,"shmem_int8_and_to_all__"); 
    if (shmem_int8_and_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_and_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_cswap__
 **********************************************************/

long shmem_int8_cswap__(long * a1, long * a2, long * a3, int * a4) {

  typedef long (*shmem_int8_cswap___p_h) (long *, long *, long *, int *);
  static shmem_int8_cswap___p_h shmem_int8_cswap___h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_cswap__(long *, long *, long *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_cswap___h == NULL)
	shmem_int8_cswap___h = (shmem_int8_cswap___p_h) dlsym(tau_handle,"shmem_int8_cswap__"); 
    if (shmem_int8_cswap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a4));
  retval  =  (*shmem_int8_cswap___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)*1);
  if (retval == (*a2)) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a4));
  }
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_fadd__
 **********************************************************/

long shmem_int8_fadd__(void * a1, int * a2, int * a3) {

  typedef long (*shmem_int8_fadd___p_h) (void *, int *, int *);
  static shmem_int8_fadd___p_h shmem_int8_fadd___h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_fadd__(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_fadd___h == NULL)
	shmem_int8_fadd___h = (shmem_int8_fadd___p_h) dlsym(tau_handle,"shmem_int8_fadd__"); 
    if (shmem_int8_fadd___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int8_fadd___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_finc__
 **********************************************************/

long shmem_int8_finc__(void * a1, int * a2) {

  typedef long (*shmem_int8_finc___p_h) (void *, int *);
  static shmem_int8_finc___p_h shmem_int8_finc___h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_finc__(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_finc___h == NULL)
	shmem_int8_finc___h = (shmem_int8_finc___p_h) dlsym(tau_handle,"shmem_int8_finc__"); 
    if (shmem_int8_finc___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a2));
  retval  =  (*shmem_int8_finc___h)( a1,  a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a2), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a2), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a2));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_inc__
 **********************************************************/

void shmem_int8_inc__(void * a1, int * a2) {

  typedef void (*shmem_int8_inc___p_h) (void *, int *);
  static shmem_int8_inc___p_h shmem_int8_inc___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_inc__(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_inc___h == NULL)
	shmem_int8_inc___h = (shmem_int8_inc___p_h) dlsym(tau_handle,"shmem_int8_inc__"); 
    if (shmem_int8_inc___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_inc___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_max_to_all__
 **********************************************************/

void shmem_int8_max_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_max_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_max_to_all___p_h shmem_int8_max_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_max_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_max_to_all___h == NULL)
	shmem_int8_max_to_all___h = (shmem_int8_max_to_all___p_h) dlsym(tau_handle,"shmem_int8_max_to_all__"); 
    if (shmem_int8_max_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_max_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_min_to_all__
 **********************************************************/

void shmem_int8_min_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_min_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_min_to_all___p_h shmem_int8_min_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_min_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_min_to_all___h == NULL)
	shmem_int8_min_to_all___h = (shmem_int8_min_to_all___p_h) dlsym(tau_handle,"shmem_int8_min_to_all__"); 
    if (shmem_int8_min_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_min_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_or_to_all__
 **********************************************************/

void shmem_int8_or_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_or_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_or_to_all___p_h shmem_int8_or_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_or_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_or_to_all___h == NULL)
	shmem_int8_or_to_all___h = (shmem_int8_or_to_all___p_h) dlsym(tau_handle,"shmem_int8_or_to_all__"); 
    if (shmem_int8_or_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_or_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_prod_to_all__
 **********************************************************/

void shmem_int8_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_prod_to_all___p_h shmem_int8_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_prod_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_prod_to_all___h == NULL)
	shmem_int8_prod_to_all___h = (shmem_int8_prod_to_all___p_h) dlsym(tau_handle,"shmem_int8_prod_to_all__"); 
    if (shmem_int8_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_sum_to_all__
 **********************************************************/

void shmem_int8_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_sum_to_all___p_h shmem_int8_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_sum_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_sum_to_all___h == NULL)
	shmem_int8_sum_to_all___h = (shmem_int8_sum_to_all___p_h) dlsym(tau_handle,"shmem_int8_sum_to_all__"); 
    if (shmem_int8_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_swap__
 **********************************************************/

long shmem_int8_swap__(void * a1, long * a2, int * a3) {

  typedef long (*shmem_int8_swap___p_h) (void *, long *, int *);
  static shmem_int8_swap___p_h shmem_int8_swap___h = NULL;
  long retval = 0;
  TAU_PROFILE_TIMER(t,"long shmem_int8_swap__(void *, long *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_int8_swap___h == NULL)
	shmem_int8_swap___h = (shmem_int8_swap___p_h) dlsym(tau_handle,"shmem_int8_swap__"); 
    if (shmem_int8_swap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, (*a3));
  retval  =  (*shmem_int8_swap___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_int8_wait__
 **********************************************************/

void shmem_int8_wait__(long * a1, long * a2) {

  typedef void (*shmem_int8_wait___p_h) (long *, long *);
  static shmem_int8_wait___p_h shmem_int8_wait___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_wait__(long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_wait___h == NULL)
	shmem_int8_wait___h = (shmem_int8_wait___p_h) dlsym(tau_handle,"shmem_int8_wait__"); 
    if (shmem_int8_wait___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_wait___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_wait_until__
 **********************************************************/

void shmem_int8_wait_until__(long * a1, int * a2, long * a3) {

  typedef void (*shmem_int8_wait_until___p_h) (long *, int *, long *);
  static shmem_int8_wait_until___p_h shmem_int8_wait_until___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_wait_until__(long *, int *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_wait_until___h == NULL)
	shmem_int8_wait_until___h = (shmem_int8_wait_until___p_h) dlsym(tau_handle,"shmem_int8_wait_until__"); 
    if (shmem_int8_wait_until___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_wait_until___h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_int8_xor_to_all__
 **********************************************************/

void shmem_int8_xor_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8) {

  typedef void (*shmem_int8_xor_to_all___p_h) (void *, void *, int *, int *, int *, int *, long *, long *);
  static shmem_int8_xor_to_all___p_h shmem_int8_xor_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_int8_xor_to_all__(void *, void *, int *, int *, int *, int *, long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_int8_xor_to_all___h == NULL)
	shmem_int8_xor_to_all___h = (shmem_int8_xor_to_all___p_h) dlsym(tau_handle,"shmem_int8_xor_to_all__"); 
    if (shmem_int8_xor_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_int8_xor_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_get__
 **********************************************************/

void shmem_integer_get__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_integer_get___p_h) (void *, void *, int *, int *);
  static shmem_integer_get___p_h shmem_integer_get___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_get__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_get___h == NULL)
	shmem_integer_get___h = (shmem_integer_get___p_h) dlsym(tau_handle,"shmem_integer_get__"); 
    if (shmem_integer_get___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  (*shmem_integer_get___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4), sizeof(int)* (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_iget__
 **********************************************************/

void shmem_integer_iget__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_integer_iget___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_integer_iget___p_h shmem_integer_iget___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_iget__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_iget___h == NULL)
	shmem_integer_iget___h = (shmem_integer_iget___p_h) dlsym(tau_handle,"shmem_integer_iget__"); 
    if (shmem_integer_iget___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)* (*a5), (*a6));
  (*shmem_integer_iget___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6), sizeof(int)* (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_iput__
 **********************************************************/

void shmem_integer_iput__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_integer_iput___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_integer_iput___p_h shmem_integer_iput___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_iput__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_iput___h == NULL)
	shmem_integer_iput___h = (shmem_integer_iput___p_h) dlsym(tau_handle,"shmem_integer_iput__"); 
    if (shmem_integer_iput___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), sizeof(int)* (*a5));
  (*shmem_integer_iput___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_integer_put__
 **********************************************************/

void shmem_integer_put__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_integer_put___p_h) (void *, void *, int *, int *);
  static shmem_integer_put___p_h shmem_integer_put___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_integer_put__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_integer_put___h == NULL)
	shmem_integer_put___h = (shmem_integer_put___p_h) dlsym(tau_handle,"shmem_integer_put__"); 
    if (shmem_integer_put___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), sizeof(int)* (*a3));
  (*shmem_integer_put___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput128__
 **********************************************************/

void shmem_iput128__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput128___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput128___p_h shmem_iput128___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput128__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput128___h == NULL)
	shmem_iput128___h = (shmem_iput128___p_h) dlsym(tau_handle,"shmem_iput128__"); 
    if (shmem_iput128___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 16* (*a5));
  (*shmem_iput128___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput16__
 **********************************************************/

void shmem_iput16__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput16___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput16___p_h shmem_iput16___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput16__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput16___h == NULL)
	shmem_iput16___h = (shmem_iput16___p_h) dlsym(tau_handle,"shmem_iput16__"); 
    if (shmem_iput16___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 2* (*a5));
  (*shmem_iput16___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput32__
 **********************************************************/

void shmem_iput32__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput32___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput32___p_h shmem_iput32___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput32__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput32___h == NULL)
	shmem_iput32___h = (shmem_iput32___p_h) dlsym(tau_handle,"shmem_iput32__"); 
    if (shmem_iput32___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
  (*shmem_iput32___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput4__
 **********************************************************/

void shmem_iput4__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput4___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput4___p_h shmem_iput4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput4__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput4___h == NULL)
	shmem_iput4___h = (shmem_iput4___p_h) dlsym(tau_handle,"shmem_iput4__"); 
    if (shmem_iput4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 4* (*a5));
  (*shmem_iput4___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput64__
 **********************************************************/

void shmem_iput64__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput64___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput64___p_h shmem_iput64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput64__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput64___h == NULL)
	shmem_iput64___h = (shmem_iput64___p_h) dlsym(tau_handle,"shmem_iput64__"); 
    if (shmem_iput64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
  (*shmem_iput64___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_iput8__
 **********************************************************/

void shmem_iput8__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_iput8___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_iput8___p_h shmem_iput8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_iput8__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_iput8___h == NULL)
	shmem_iput8___h = (shmem_iput8___p_h) dlsym(tau_handle,"shmem_iput8__"); 
    if (shmem_iput8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6), 8* (*a5));
  (*shmem_iput8___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_get__
 **********************************************************/

void shmem_logical_get__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_logical_get___p_h) (void *, void *, int *, int *);
  static shmem_logical_get___p_h shmem_logical_get___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_get__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_get___h == NULL)
	shmem_logical_get___h = (shmem_logical_get___p_h) dlsym(tau_handle,"shmem_logical_get__"); 
    if (shmem_logical_get___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_logical_get___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_iget__
 **********************************************************/

void shmem_logical_iget__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_logical_iget___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_logical_iget___p_h shmem_logical_iget___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_iget__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_iget___h == NULL)
	shmem_logical_iget___h = (shmem_logical_iget___p_h) dlsym(tau_handle,"shmem_logical_iget__"); 
    if (shmem_logical_iget___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
  (*shmem_logical_iget___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_iput__
 **********************************************************/

void shmem_logical_iput__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_logical_iput___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_logical_iput___p_h shmem_logical_iput___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_iput__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_iput___h == NULL)
	shmem_logical_iput___h = (shmem_logical_iput___p_h) dlsym(tau_handle,"shmem_logical_iput__"); 
    if (shmem_logical_iput___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
  (*shmem_logical_iput___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_logical_put__
 **********************************************************/

void shmem_logical_put__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_logical_put___p_h) (void *, void *, int *, int *);
  static shmem_logical_put___p_h shmem_logical_put___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_logical_put__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_logical_put___h == NULL)
	shmem_logical_put___h = (shmem_logical_put___p_h) dlsym(tau_handle,"shmem_logical_put__"); 
    if (shmem_logical_put___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_logical_put___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_my_pe__
 **********************************************************/

int shmem_my_pe__() {

  typedef int (*shmem_my_pe___p_h) ();
  static shmem_my_pe___p_h shmem_my_pe___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_my_pe__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_my_pe___h == NULL)
	shmem_my_pe___h = (shmem_my_pe___p_h) dlsym(tau_handle,"shmem_my_pe__"); 
    if (shmem_my_pe___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_my_pe___h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_n_pes__
 **********************************************************/

int shmem_n_pes__() {

  typedef int (*shmem_n_pes___p_h) ();
  static shmem_n_pes___p_h shmem_n_pes___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_n_pes__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_n_pes___h == NULL)
	shmem_n_pes___h = (shmem_n_pes___p_h) dlsym(tau_handle,"shmem_n_pes__"); 
    if (shmem_n_pes___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_n_pes___h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_pe_accessible__
 **********************************************************/

int shmem_pe_accessible__(int * a1) {

  typedef int (*shmem_pe_accessible___p_h) (int *);
  static shmem_pe_accessible___p_h shmem_pe_accessible___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_pe_accessible__(int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_pe_accessible___h == NULL)
	shmem_pe_accessible___h = (shmem_pe_accessible___p_h) dlsym(tau_handle,"shmem_pe_accessible__"); 
    if (shmem_pe_accessible___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_pe_accessible___h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_ptr__
 **********************************************************/

void shmem_ptr__(void * a1, int * a2) {

  typedef void (*shmem_ptr___p_h) (void *, int *);
  static shmem_ptr___p_h shmem_ptr___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_ptr__(void *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_ptr___h == NULL)
	shmem_ptr___h = (shmem_ptr___p_h) dlsym(tau_handle,"shmem_ptr__"); 
    if (shmem_ptr___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_ptr___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put128__
 **********************************************************/

void shmem_put128__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put128___p_h) (void *, void *, int *, int *);
  static shmem_put128___p_h shmem_put128___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put128__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put128___h == NULL)
	shmem_put128___h = (shmem_put128___p_h) dlsym(tau_handle,"shmem_put128__"); 
    if (shmem_put128___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 16* (*a3));
  (*shmem_put128___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put16__
 **********************************************************/

void shmem_put16__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put16___p_h) (void *, void *, int *, int *);
  static shmem_put16___p_h shmem_put16___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put16__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put16___h == NULL)
	shmem_put16___h = (shmem_put16___p_h) dlsym(tau_handle,"shmem_put16__"); 
    if (shmem_put16___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 2* (*a3));
  (*shmem_put16___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put32__
 **********************************************************/

void shmem_put32__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put32___p_h) (void *, void *, int *, int *);
  static shmem_put32___p_h shmem_put32___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put32__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put32___h == NULL)
	shmem_put32___h = (shmem_put32___p_h) dlsym(tau_handle,"shmem_put32__"); 
    if (shmem_put32___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
  (*shmem_put32___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put4__
 **********************************************************/

void shmem_put4__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put4___p_h) (void *, void *, int *, int *);
  static shmem_put4___p_h shmem_put4___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put4__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put4___h == NULL)
	shmem_put4___h = (shmem_put4___p_h) dlsym(tau_handle,"shmem_put4__"); 
    if (shmem_put4___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 4* (*a3));
  (*shmem_put4___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put64__
 **********************************************************/

void shmem_put64__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put64___p_h) (void *, void *, int *, int *);
  static shmem_put64___p_h shmem_put64___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put64__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put64___h == NULL)
	shmem_put64___h = (shmem_put64___p_h) dlsym(tau_handle,"shmem_put64__"); 
    if (shmem_put64___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
  (*shmem_put64___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_put8__
 **********************************************************/

void shmem_put8__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_put8___p_h) (void *, void *, int *, int *);
  static shmem_put8___p_h shmem_put8___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_put8__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_put8___h == NULL)
	shmem_put8___h = (shmem_put8___p_h) dlsym(tau_handle,"shmem_put8__"); 
    if (shmem_put8___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4), 8* (*a3));
  (*shmem_put8___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8* (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_putmem__
 **********************************************************/

void shmem_putmem__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_putmem___p_h) (void *, void *, int *, int *);
  static shmem_putmem___p_h shmem_putmem___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_putmem__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_putmem___h == NULL)
	shmem_putmem___h = (shmem_putmem___p_h) dlsym(tau_handle,"shmem_putmem__"); 
    if (shmem_putmem___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_putmem___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_quiet__
 **********************************************************/

void shmem_quiet__() {

  typedef void (*shmem_quiet___p_h) ();
  static shmem_quiet___p_h shmem_quiet___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_quiet__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_quiet___h == NULL)
	shmem_quiet___h = (shmem_quiet___p_h) dlsym(tau_handle,"shmem_quiet__"); 
    if (shmem_quiet___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_quiet___h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_max_to_all__
 **********************************************************/

void shmem_real16_max_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_max_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_max_to_all___p_h shmem_real16_max_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_max_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_max_to_all___h == NULL)
	shmem_real16_max_to_all___h = (shmem_real16_max_to_all___p_h) dlsym(tau_handle,"shmem_real16_max_to_all__"); 
    if (shmem_real16_max_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_max_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_min_to_all__
 **********************************************************/

void shmem_real16_min_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_min_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_min_to_all___p_h shmem_real16_min_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_min_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_min_to_all___h == NULL)
	shmem_real16_min_to_all___h = (shmem_real16_min_to_all___p_h) dlsym(tau_handle,"shmem_real16_min_to_all__"); 
    if (shmem_real16_min_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_min_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_prod_to_all__
 **********************************************************/

void shmem_real16_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_prod_to_all___p_h shmem_real16_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_prod_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_prod_to_all___h == NULL)
	shmem_real16_prod_to_all___h = (shmem_real16_prod_to_all___p_h) dlsym(tau_handle,"shmem_real16_prod_to_all__"); 
    if (shmem_real16_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real16_sum_to_all__
 **********************************************************/

void shmem_real16_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real16_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real16_sum_to_all___p_h shmem_real16_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real16_sum_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real16_sum_to_all___h == NULL)
	shmem_real16_sum_to_all___h = (shmem_real16_sum_to_all___p_h) dlsym(tau_handle,"shmem_real16_sum_to_all__"); 
    if (shmem_real16_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real16_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_max_to_all__
 **********************************************************/

void shmem_real4_max_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_max_to_all___p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_max_to_all___p_h shmem_real4_max_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_max_to_all__(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_max_to_all___h == NULL)
	shmem_real4_max_to_all___h = (shmem_real4_max_to_all___p_h) dlsym(tau_handle,"shmem_real4_max_to_all__"); 
    if (shmem_real4_max_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_max_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_min_to_all__
 **********************************************************/

void shmem_real4_min_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_min_to_all___p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_min_to_all___p_h shmem_real4_min_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_min_to_all__(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_min_to_all___h == NULL)
	shmem_real4_min_to_all___h = (shmem_real4_min_to_all___p_h) dlsym(tau_handle,"shmem_real4_min_to_all__"); 
    if (shmem_real4_min_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_min_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_prod_to_all__
 **********************************************************/

void shmem_real4_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_prod_to_all___p_h shmem_real4_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_prod_to_all__(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_prod_to_all___h == NULL)
	shmem_real4_prod_to_all___h = (shmem_real4_prod_to_all___p_h) dlsym(tau_handle,"shmem_real4_prod_to_all__"); 
    if (shmem_real4_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_sum_to_all__
 **********************************************************/

void shmem_real4_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8) {

  typedef void (*shmem_real4_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, float *, long *);
  static shmem_real4_sum_to_all___p_h shmem_real4_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real4_sum_to_all__(void *, void *, int *, int *, int *, int *, float *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real4_sum_to_all___h == NULL)
	shmem_real4_sum_to_all___h = (shmem_real4_sum_to_all___p_h) dlsym(tau_handle,"shmem_real4_sum_to_all__"); 
    if (shmem_real4_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real4_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real4_swap__
 **********************************************************/

float shmem_real4_swap__(void * a1, float * a2, int * a3) {

  typedef float (*shmem_real4_swap___p_h) (void *, float *, int *);
  static shmem_real4_swap___p_h shmem_real4_swap___h = NULL;
  float retval = 0;
  TAU_PROFILE_TIMER(t,"float shmem_real4_swap__(void *, float *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_real4_swap___h == NULL)
	shmem_real4_swap___h = (shmem_real4_swap___p_h) dlsym(tau_handle,"shmem_real4_swap__"); 
    if (shmem_real4_swap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*1, (*a3));
  retval  =  (*shmem_real4_swap___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 4*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 4*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_real8_max_to_all__
 **********************************************************/

void shmem_real8_max_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_max_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_max_to_all___p_h shmem_real8_max_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_max_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_max_to_all___h == NULL)
	shmem_real8_max_to_all___h = (shmem_real8_max_to_all___p_h) dlsym(tau_handle,"shmem_real8_max_to_all__"); 
    if (shmem_real8_max_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_max_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_min_to_all__
 **********************************************************/

void shmem_real8_min_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_min_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_min_to_all___p_h shmem_real8_min_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_min_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_min_to_all___h == NULL)
	shmem_real8_min_to_all___h = (shmem_real8_min_to_all___p_h) dlsym(tau_handle,"shmem_real8_min_to_all__"); 
    if (shmem_real8_min_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_min_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_prod_to_all__
 **********************************************************/

void shmem_real8_prod_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_prod_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_prod_to_all___p_h shmem_real8_prod_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_prod_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_prod_to_all___h == NULL)
	shmem_real8_prod_to_all___h = (shmem_real8_prod_to_all___p_h) dlsym(tau_handle,"shmem_real8_prod_to_all__"); 
    if (shmem_real8_prod_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_prod_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_sum_to_all__
 **********************************************************/

void shmem_real8_sum_to_all__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8) {

  typedef void (*shmem_real8_sum_to_all___p_h) (void *, void *, int *, int *, int *, int *, double *, long *);
  static shmem_real8_sum_to_all___p_h shmem_real8_sum_to_all___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real8_sum_to_all__(void *, void *, int *, int *, int *, int *, double *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real8_sum_to_all___h == NULL)
	shmem_real8_sum_to_all___h = (shmem_real8_sum_to_all___p_h) dlsym(tau_handle,"shmem_real8_sum_to_all__"); 
    if (shmem_real8_sum_to_all___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_real8_sum_to_all___h)( a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real8_swap__
 **********************************************************/

double shmem_real8_swap__(void * a1, double * a2, int * a3) {

  typedef double (*shmem_real8_swap___p_h) (void *, double *, int *);
  static shmem_real8_swap___p_h shmem_real8_swap___h = NULL;
  double retval = 0;
  TAU_PROFILE_TIMER(t,"double shmem_real8_swap__(void *, double *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_real8_swap___h == NULL)
	shmem_real8_swap___h = (shmem_real8_swap___p_h) dlsym(tau_handle,"shmem_real8_swap__"); 
    if (shmem_real8_swap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*1, (*a3));
  retval  =  (*shmem_real8_swap___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 8*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 8*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_real_get__
 **********************************************************/

void shmem_real_get__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_real_get___p_h) (void *, void *, int *, int *);
  static shmem_real_get___p_h shmem_real_get___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_get__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_get___h == NULL)
	shmem_real_get___h = (shmem_real_get___p_h) dlsym(tau_handle,"shmem_real_get__"); 
    if (shmem_real_get___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a3), (*a4));
  (*shmem_real_get___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a4),  (*a3));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real_iget__
 **********************************************************/

void shmem_real_iget__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_real_iget___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_real_iget___p_h shmem_real_iget___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_iget__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_iget___h == NULL)
	shmem_real_iget___h = (shmem_real_iget___p_h) dlsym(tau_handle,"shmem_real_iget__"); 
    if (shmem_real_iget___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(),  (*a5), (*a6));
  (*shmem_real_iget___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a6),  (*a5));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real_iput__
 **********************************************************/

void shmem_real_iput__(void * a1, void * a2, int * a3, int * a4, int * a5, int * a6) {

  typedef void (*shmem_real_iput___p_h) (void *, void *, int *, int *, int *, int *);
  static shmem_real_iput___p_h shmem_real_iput___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_iput__(void *, void *, int *, int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_iput___h == NULL)
	shmem_real_iput___h = (shmem_real_iput___p_h) dlsym(tau_handle,"shmem_real_iput__"); 
    if (shmem_real_iput___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a6),  (*a5));
  (*shmem_real_iput___h)( a1,  a2,  a3,  a4,  a5,  a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a5), (*a6));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_real_put__
 **********************************************************/

void shmem_real_put__(void * a1, void * a2, int * a3, int * a4) {

  typedef void (*shmem_real_put___p_h) (void *, void *, int *, int *);
  static shmem_real_put___p_h shmem_real_put___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_real_put__(void *, void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_real_put___h == NULL)
	shmem_real_put___h = (shmem_real_put___p_h) dlsym(tau_handle,"shmem_real_put__"); 
    if (shmem_real_put___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a4),  (*a3));
  (*shmem_real_put___h)( a1,  a2,  a3,  a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(),  (*a3), (*a4));
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_cache_inv__
 **********************************************************/

void shmem_set_cache_inv__() {

  typedef void (*shmem_set_cache_inv___p_h) ();
  static shmem_set_cache_inv___p_h shmem_set_cache_inv___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_cache_inv__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_cache_inv___h == NULL)
	shmem_set_cache_inv___h = (shmem_set_cache_inv___p_h) dlsym(tau_handle,"shmem_set_cache_inv__"); 
    if (shmem_set_cache_inv___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_cache_inv___h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_cache_line_inv__
 **********************************************************/

void shmem_set_cache_line_inv__(void * a1) {

  typedef void (*shmem_set_cache_line_inv___p_h) (void *);
  static shmem_set_cache_line_inv___p_h shmem_set_cache_line_inv___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_cache_line_inv__(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_cache_line_inv___h == NULL)
	shmem_set_cache_line_inv___h = (shmem_set_cache_line_inv___p_h) dlsym(tau_handle,"shmem_set_cache_line_inv__"); 
    if (shmem_set_cache_line_inv___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_cache_line_inv___h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_lock__
 **********************************************************/

void shmem_set_lock__(long * a1) {

  typedef void (*shmem_set_lock___p_h) (long *);
  static shmem_set_lock___p_h shmem_set_lock___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_lock__(long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_lock___h == NULL)
	shmem_set_lock___h = (shmem_set_lock___p_h) dlsym(tau_handle,"shmem_set_lock__"); 
    if (shmem_set_lock___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_lock___h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_swap__
 **********************************************************/

int shmem_swap__(void * a1, int * a2, int * a3) {

  typedef int (*shmem_swap___p_h) (void *, int *, int *);
  static shmem_swap___p_h shmem_swap___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_swap__(void *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_swap___h == NULL)
	shmem_swap___h = (shmem_swap___p_h) dlsym(tau_handle,"shmem_swap__"); 
    if (shmem_swap___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 1, (*a3));
  retval  =  (*shmem_swap___h)( a1,  a2,  a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, (*a3), 1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, (*a3), 1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 1, (*a3));
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_test_lock__
 **********************************************************/

int shmem_test_lock__(long * a1) {

  typedef int (*shmem_test_lock___p_h) (long *);
  static shmem_test_lock___p_h shmem_test_lock___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock__(long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (shmem_test_lock___h == NULL)
	shmem_test_lock___h = (shmem_test_lock___p_h) dlsym(tau_handle,"shmem_test_lock__"); 
    if (shmem_test_lock___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*shmem_test_lock___h)( a1);
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   shmem_udcflush__
 **********************************************************/

void shmem_udcflush__() {

  typedef void (*shmem_udcflush___p_h) ();
  static shmem_udcflush___p_h shmem_udcflush___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_udcflush__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_udcflush___h == NULL)
	shmem_udcflush___h = (shmem_udcflush___p_h) dlsym(tau_handle,"shmem_udcflush__"); 
    if (shmem_udcflush___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_udcflush___h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_udcflush_line__
 **********************************************************/

void shmem_udcflush_line__(void * a1) {

  typedef void (*shmem_udcflush_line___p_h) (void *);
  static shmem_udcflush_line___p_h shmem_udcflush_line___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_udcflush_line__(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_udcflush_line___h == NULL)
	shmem_udcflush_line___h = (shmem_udcflush_line___p_h) dlsym(tau_handle,"shmem_udcflush_line__"); 
    if (shmem_udcflush_line___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_udcflush_line___h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_wait__
 **********************************************************/

void shmem_wait__(long * a1, long * a2) {

  typedef void (*shmem_wait___p_h) (long *, long *);
  static shmem_wait___p_h shmem_wait___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_wait__(long *, long *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_wait___h == NULL)
	shmem_wait___h = (shmem_wait___p_h) dlsym(tau_handle,"shmem_wait__"); 
    if (shmem_wait___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_wait___h)( a1,  a2);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_wait_until__
 **********************************************************/

void shmem_wait_until__(int * a1, int * a2, int * a3) {

  typedef void (*shmem_wait_until___p_h) (int *, int *, int *);
  static shmem_wait_until___p_h shmem_wait_until___h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_wait_until__(int *, int *, int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_wait_until___h == NULL)
	shmem_wait_until___h = (shmem_wait_until___p_h) dlsym(tau_handle,"shmem_wait_until__"); 
    if (shmem_wait_until___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_wait_until___h)( a1,  a2,  a3);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   start_pes__
 **********************************************************/

void start_pes__(int * a1) {

  typedef void (*start_pes___p_h) (int *);
  static start_pes___p_h start_pes___h = NULL;
  TAU_PROFILE_TIMER(t,"void start_pes__(int *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (start_pes___h == NULL)
	start_pes___h = (start_pes___p_h) dlsym(tau_handle,"start_pes__"); 
    if (start_pes___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*start_pes___h)( a1);
#ifdef TAU_PSHMEM_SGI_MPT
  tau_totalnodes(1,pshmem_n_pes());
  TAU_PROFILE_SET_NODE(pshmem_my_pe());
#else
  tau_totalnodes(1,_shmem_n_pes());
  TAU_PROFILE_SET_NODE(_shmem_my_pe());
#endif
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_cache_inv
 **********************************************************/

void shmem_clear_cache_inv() {

  typedef void (*shmem_clear_cache_inv_p_h) ();
  static shmem_clear_cache_inv_p_h shmem_clear_cache_inv_h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_inv()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_cache_inv_h == NULL)
	shmem_clear_cache_inv_h = (shmem_clear_cache_inv_p_h) dlsym(tau_handle,"shmem_clear_cache_inv"); 
    if (shmem_clear_cache_inv_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_cache_inv_h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_cache_inv
 **********************************************************/

void shmem_set_cache_inv() {

  typedef void (*shmem_set_cache_inv_p_h) ();
  static shmem_set_cache_inv_p_h shmem_set_cache_inv_h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_cache_inv()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_cache_inv_h == NULL)
	shmem_set_cache_inv_h = (shmem_set_cache_inv_p_h) dlsym(tau_handle,"shmem_set_cache_inv"); 
    if (shmem_set_cache_inv_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_cache_inv_h)();
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_clear_cache_line_inv
 **********************************************************/

void shmem_clear_cache_line_inv(void * a1) {

  typedef void (*shmem_clear_cache_line_inv_p_h) (void *);
  static shmem_clear_cache_line_inv_p_h shmem_clear_cache_line_inv_h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_line_inv(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_clear_cache_line_inv_h == NULL)
	shmem_clear_cache_line_inv_h = (shmem_clear_cache_line_inv_p_h) dlsym(tau_handle,"shmem_clear_cache_line_inv"); 
    if (shmem_clear_cache_line_inv_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_clear_cache_line_inv_h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   shmem_set_cache_line_inv
 **********************************************************/

void shmem_set_cache_line_inv(void * a1) {

  typedef void (*shmem_set_cache_line_inv_p_h) (void *);
  static shmem_set_cache_line_inv_p_h shmem_set_cache_line_inv_h = NULL;
  TAU_PROFILE_TIMER(t,"void shmem_set_cache_line_inv(void *)", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return;
  } 
  else { 
    if (shmem_set_cache_line_inv_h == NULL)
	shmem_set_cache_line_inv_h = (shmem_set_cache_line_inv_p_h) dlsym(tau_handle,"shmem_set_cache_line_inv"); 
    if (shmem_set_cache_line_inv_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return;
    }
  TAU_PROFILE_START(t);
  (*shmem_set_cache_line_inv_h)( a1);
  TAU_PROFILE_STOP(t);
  }

}


/**********************************************************
   my_pe_
 **********************************************************/

int my_pe_() {

  typedef int (*my_pe__p_h) ();
  static my_pe__p_h my_pe__h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int my_pe_()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (my_pe__h == NULL)
	my_pe__h = (my_pe__p_h) dlsym(tau_handle,"my_pe_"); 
    if (my_pe__h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*my_pe__h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}


/**********************************************************
   my_pe__
 **********************************************************/

int my_pe__() {

  typedef int (*my_pe___p_h) ();
  static my_pe___p_h my_pe___h = NULL;
  int retval = 0;
  TAU_PROFILE_TIMER(t,"int my_pe__()", "", TAU_USER);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
    return retval;
  } 
  else { 
    if (my_pe___h == NULL)
	my_pe___h = (my_pe___p_h) dlsym(tau_handle,"my_pe__"); 
    if (my_pe___h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
      return retval;
    }
  TAU_PROFILE_START(t);
  retval  =  (*my_pe___h)();
  TAU_PROFILE_STOP(t);
  }
  return retval;

}

