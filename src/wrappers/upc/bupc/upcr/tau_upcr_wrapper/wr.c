#include <tau_upcr.h>
#include <Profile/Profiler.h>
#include <stdio.h>

#pragma pupc off

#ifdef __BERKELEY_UPC__
#pragma UPCR NO_SRCPOS 
#endif

static int tau_upc_node = -1;
static int tau_upc_tagid_f = 0;
#define TAU_UPC_TAGID (tau_upc_tagid_f = (tau_upc_tagid_f & 255))
#define TAU_UPC_TAGID_NEXT ((++tau_upc_tagid_f) & 255)

void tau_totalnodes(int, int);

#warning "TAU: Not generating wrapper for function upcri_errno"
#warning "TAU: Not generating wrapper for function upcri_gaserr"
#warning "TAU: Not generating wrapper for function upcri_append_srcloc"
#warning "TAU: Not generating wrapper for function upcri_err"
#warning "TAU: Not generating wrapper for function upcri_early_err"
#warning "TAU: Not generating wrapper for function upcri_warn"
#warning "TAU: Not generating wrapper for function upcri_barprintf"
#warning "TAU: Not generating wrapper for function upcri_sleepprintf"

/**********************************************************
   _bupc_thread_distance
 **********************************************************/

unsigned int  __real__bupc_thread_distance(int a1, int a2) ;
unsigned int  __wrap__bupc_thread_distance(int a1, int a2)  {

  unsigned int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_thread_distance(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"unsigned int _bupc_thread_distance(int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_thread_distance(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcri_rand
 **********************************************************/

int  __real__upcri_rand() ;
int  __wrap__upcri_rand()  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcri_rand();
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcri_rand(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcri_rand();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcri_srand
 **********************************************************/

void  __real__upcri_srand(unsigned int a1) ;
void  __wrap__upcri_srand(unsigned int a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcri_srand(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcri_srand(unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcri_srand(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcri_rand_init
 **********************************************************/

void  __real__upcri_rand_init() ;
void  __wrap__upcri_rand_init()  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcri_rand_init();
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcri_rand_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcri_rand_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcri_clock_init
 **********************************************************/

void  __real_upcri_clock_init() ;
void  __wrap_upcri_clock_init()  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcri_clock_init();
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcri_clock_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcri_clock_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcri_clock
 **********************************************************/

clock_t  __real_upcri_clock() ;
clock_t  __wrap_upcri_clock()  {

  clock_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upcri_clock();
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"clock_t upcri_clock(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upcri_clock();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcri_isvalid_shared
 **********************************************************/

int  __real__upcri_isvalid_shared(upcr_shared_ptr_t a1) ;
int  __wrap__upcri_isvalid_shared(upcr_shared_ptr_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcri_isvalid_shared(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcri_isvalid_shared(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcri_isvalid_shared(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcri_isvalid_pshared
 **********************************************************/

int  __real__upcri_isvalid_pshared(upcr_pshared_ptr_t a1) ;
int  __wrap__upcri_isvalid_pshared(upcr_pshared_ptr_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcri_isvalid_pshared(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcri_isvalid_pshared(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcri_isvalid_pshared(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upcri_print_shared
 **********************************************************/

void  __real_upcri_print_shared(upcr_shared_ptr_t a1) ;
void  __wrap_upcri_print_shared(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcri_print_shared(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcri_print_shared(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcri_print_shared(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcri_print_pshared
 **********************************************************/

void  __real_upcri_print_pshared(upcr_pshared_ptr_t a1) ;
void  __wrap_upcri_print_pshared(upcr_pshared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcri_print_pshared(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcri_print_pshared(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcri_print_pshared(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_dump_shared
 **********************************************************/

int  __real__bupc_dump_shared(upcr_shared_ptr_t a1, char * a2, int a3) ;
int  __wrap__bupc_dump_shared(upcr_shared_ptr_t a1, char * a2, int a3)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_dump_shared(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _bupc_dump_shared(upcr_shared_ptr_t, char *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_dump_shared(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_local_to_shared
 **********************************************************/

upcr_shared_ptr_t  __real__bupc_local_to_shared(void * a1, int a2, int a3) ;
upcr_shared_ptr_t  __wrap__bupc_local_to_shared(void * a1, int a2, int a3)  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_local_to_shared(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _bupc_local_to_shared(void *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_local_to_shared(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_inverse_cast
 **********************************************************/

upcr_shared_ptr_t  __real__bupc_inverse_cast(void * a1) ;
upcr_shared_ptr_t  __wrap__bupc_inverse_cast(void * a1)  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_inverse_cast(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _bupc_inverse_cast(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_inverse_cast(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcri_locksystem_init
 **********************************************************/

void  __real__upcri_locksystem_init() ;
void  __wrap__upcri_locksystem_init()  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcri_locksystem_init();
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcri_locksystem_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcri_locksystem_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_global_lock_alloc
 **********************************************************/

upcr_shared_ptr_t  __real__upcr_global_lock_alloc() ;
upcr_shared_ptr_t  __wrap__upcr_global_lock_alloc()  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_global_lock_alloc();
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _upcr_global_lock_alloc(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_global_lock_alloc();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_lock_alloc
 **********************************************************/

upcr_shared_ptr_t  __real__upcr_all_lock_alloc() ;
upcr_shared_ptr_t  __wrap__upcr_all_lock_alloc()  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_lock_alloc();
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _upcr_all_lock_alloc(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_lock_alloc();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_lock
 **********************************************************/

void  __real__upcr_lock(upcr_shared_ptr_t a1) ;
void  __wrap__upcr_lock(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_lock(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_lock(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_lock_attempt
 **********************************************************/

int  __real__upcr_lock_attempt(upcr_shared_ptr_t a1) ;
int  __wrap__upcr_lock_attempt(upcr_shared_ptr_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_lock_attempt(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_lock_attempt(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_lock_attempt(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_unlock
 **********************************************************/

void  __real__upcr_unlock(upcr_shared_ptr_t a1) ;
void  __wrap__upcr_unlock(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_unlock(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_unlock(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_unlock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_lock_free
 **********************************************************/

void  __real__upcr_lock_free(upcr_shared_ptr_t a1) ;
void  __wrap__upcr_lock_free(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_lock_free(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_lock_free(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_lock_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_lock_free
 **********************************************************/

void  __real__upcr_all_lock_free(upcr_shared_ptr_t a1) ;
void  __wrap__upcr_all_lock_free(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_lock_free(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_lock_free(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_lock_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_sem_alloc
 **********************************************************/

upcr_pshared_ptr_t  __real__bupc_sem_alloc(int a1) ;
upcr_pshared_ptr_t  __wrap__bupc_sem_alloc(int a1)  {

  upcr_pshared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_sem_alloc(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_pshared_ptr_t _bupc_sem_alloc(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_sem_alloc(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_sem_free
 **********************************************************/

void  __real__bupc_sem_free(upcr_pshared_ptr_t a1) ;
void  __wrap__bupc_sem_free(upcr_pshared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_sem_free(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_sem_free(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_sem_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_sem_post
 **********************************************************/

void  __real__bupc_sem_post(upcr_pshared_ptr_t a1) ;
void  __wrap__bupc_sem_post(upcr_pshared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_sem_post(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_sem_post(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_sem_post(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_sem_postN
 **********************************************************/

void  __real__bupc_sem_postN(upcr_pshared_ptr_t a1, size_t a2) ;
void  __wrap__bupc_sem_postN(upcr_pshared_ptr_t a1, size_t a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_sem_postN(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_sem_postN(upcr_pshared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_sem_postN(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_sem_wait
 **********************************************************/

void  __real__bupc_sem_wait(upcr_pshared_ptr_t a1) ;
void  __wrap__bupc_sem_wait(upcr_pshared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_sem_wait(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_sem_wait(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_sem_wait(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_sem_waitN
 **********************************************************/

void  __real__bupc_sem_waitN(upcr_pshared_ptr_t a1, size_t a2) ;
void  __wrap__bupc_sem_waitN(upcr_pshared_ptr_t a1, size_t a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_sem_waitN(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_sem_waitN(upcr_pshared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_sem_waitN(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_sem_try
 **********************************************************/

int  __real__bupc_sem_try(upcr_pshared_ptr_t a1) ;
int  __wrap__bupc_sem_try(upcr_pshared_ptr_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_sem_try(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _bupc_sem_try(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_sem_try(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_sem_tryN
 **********************************************************/

int  __real__bupc_sem_tryN(upcr_pshared_ptr_t a1, size_t a2) ;
int  __wrap__bupc_sem_tryN(upcr_pshared_ptr_t a1, size_t a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_sem_tryN(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _bupc_sem_tryN(upcr_pshared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_sem_tryN(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memput_signal
 **********************************************************/

void  __real__bupc_memput_signal(upcr_shared_ptr_t a1, const void * a2, size_t a3, upcr_pshared_ptr_t a4, size_t a5) ;
void  __wrap__bupc_memput_signal(upcr_shared_ptr_t a1, const void * a2, size_t a3, upcr_pshared_ptr_t a4, size_t a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_memput_signal(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_memput_signal(upcr_shared_ptr_t, const void *, size_t, upcr_pshared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_memput_signal(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_memput_signal_async
 **********************************************************/

void  __real__bupc_memput_signal_async(upcr_shared_ptr_t a1, const void * a2, size_t a3, upcr_pshared_ptr_t a4, size_t a5) ;
void  __wrap__bupc_memput_signal_async(upcr_shared_ptr_t a1, const void * a2, size_t a3, upcr_pshared_ptr_t a4, size_t a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_memput_signal_async(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_memput_signal_async(upcr_shared_ptr_t, const void *, size_t, upcr_pshared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_memput_signal_async(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}

#warning "TAU: Not generating wrapper for function upcri_barrier_init"

/**********************************************************
   _upcr_notify
 **********************************************************/

void  __real__upcr_notify(int a1, int a2) ;
void  __wrap__upcr_notify(int a1, int a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_notify(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_notify(int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_notify(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_wait
 **********************************************************/

void  __real__upcr_wait(int a1, int a2) ;
void  __wrap__upcr_wait(int a1, int a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_wait(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_wait(int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_try_wait
 **********************************************************/

int  __real__upcr_try_wait(int a1, int a2) ;
int  __wrap__upcr_try_wait(int a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_try_wait(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_try_wait(int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_try_wait(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upcri_coll_init
 **********************************************************/

void  __real_upcri_coll_init() ;
void  __wrap_upcri_coll_init()  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcri_coll_init();
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcri_coll_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcri_coll_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcri_coll_init_thread
 **********************************************************/

void  __real__upcri_coll_init_thread() ;
void  __wrap__upcri_coll_init_thread()  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcri_coll_init_thread();
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcri_coll_init_thread(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcri_coll_init_thread();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_broadcast
 **********************************************************/

void  __real__upcr_all_broadcast(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4) ;
void  __wrap__upcr_all_broadcast(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_broadcast(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_broadcast(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_broadcast(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_team_broadcast
 **********************************************************/

bupc_coll_handle_t  __real__upcr_team_broadcast(bupc_team_t a1, upcr_shared_ptr_t a2, upcr_shared_ptr_t a3, size_t a4, upc_flag_t a5) ;
bupc_coll_handle_t  __wrap__upcr_team_broadcast(bupc_team_t a1, upcr_shared_ptr_t a2, upcr_shared_ptr_t a3, size_t a4, upc_flag_t a5)  {

  bupc_coll_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_team_broadcast(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_coll_handle_t _upcr_team_broadcast(bupc_team_t, upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_team_broadcast(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_scatter
 **********************************************************/

void  __real__upcr_all_scatter(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4) ;
void  __wrap__upcr_all_scatter(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_scatter(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_scatter(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_scatter(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_gather
 **********************************************************/

void  __real__upcr_all_gather(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4) ;
void  __wrap__upcr_all_gather(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_gather(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_gather(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_gather(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_gather_all
 **********************************************************/

void  __real__upcr_all_gather_all(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4) ;
void  __wrap__upcr_all_gather_all(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_gather_all(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_gather_all(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_gather_all(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_exchange
 **********************************************************/

void  __real__upcr_all_exchange(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4) ;
void  __wrap__upcr_all_exchange(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_exchange(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_exchange(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_exchange(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_team_exchange
 **********************************************************/

bupc_coll_handle_t  __real__upcr_team_exchange(bupc_team_t a1, upcr_shared_ptr_t a2, upcr_shared_ptr_t a3, size_t a4, upc_flag_t a5) ;
bupc_coll_handle_t  __wrap__upcr_team_exchange(bupc_team_t a1, upcr_shared_ptr_t a2, upcr_shared_ptr_t a3, size_t a4, upc_flag_t a5)  {

  bupc_coll_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_team_exchange(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_coll_handle_t _upcr_team_exchange(bupc_team_t, upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_team_exchange(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_permute
 **********************************************************/

void  __real__upcr_all_permute(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upcr_pshared_ptr_t a3, size_t a4, upc_flag_t a5) ;
void  __wrap__upcr_all_permute(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upcr_pshared_ptr_t a3, size_t a4, upc_flag_t a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_permute(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_permute(upcr_shared_ptr_t, upcr_shared_ptr_t, upcr_pshared_ptr_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_permute(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_team_split
 **********************************************************/

bupc_team_t  __real__upcr_team_split(bupc_team_t a1, int a2, int a3) ;
bupc_team_t  __wrap__upcr_team_split(bupc_team_t a1, int a2, int a3)  {

  bupc_team_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_team_split(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_team_t _upcr_team_split(bupc_team_t, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_team_split(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_reduceC
 **********************************************************/

void  __real__upcr_all_reduceC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceC(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceC(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed char (*)(signed char, signed char), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceC(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceC
 **********************************************************/

void  __real__upcr_all_prefix_reduceC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceC(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed char (*)(signed char, signed char), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceUC
 **********************************************************/

void  __real__upcr_all_reduceUC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceUC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceUC(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceUC(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned char (*)(unsigned char, unsigned char), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceUC(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceUC
 **********************************************************/

void  __real__upcr_all_prefix_reduceUC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceUC(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceUC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceUC(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned char (*)(unsigned char, unsigned char), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceUC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceS
 **********************************************************/

void  __real__upcr_all_reduceS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceS(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceS(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed short (*)(signed short, signed short), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceS(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceS
 **********************************************************/

void  __real__upcr_all_prefix_reduceS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceS(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed short (*)(signed short, signed short), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceUS
 **********************************************************/

void  __real__upcr_all_reduceUS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceUS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceUS(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceUS(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned short (*)(unsigned short, unsigned short), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceUS(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceUS
 **********************************************************/

void  __real__upcr_all_prefix_reduceUS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceUS(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceUS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceUS(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned short (*)(unsigned short, unsigned short), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceUS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceI
 **********************************************************/

void  __real__upcr_all_reduceI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceI(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceI(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed int (*)(signed int, signed int), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceI(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceI
 **********************************************************/

void  __real__upcr_all_prefix_reduceI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceI(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed int (*)(signed int, signed int), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceUI
 **********************************************************/

void  __real__upcr_all_reduceUI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceUI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceUI(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceUI(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned int (*)(unsigned int, unsigned int), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceUI(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceUI
 **********************************************************/

void  __real__upcr_all_prefix_reduceUI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceUI(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceUI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceUI(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned int (*)(unsigned int, unsigned int), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceUI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceL
 **********************************************************/

void  __real__upcr_all_reduceL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceL(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceL(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed long (*)(signed long, signed long), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceL(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceL
 **********************************************************/

void  __real__upcr_all_prefix_reduceL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceL(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, signed long (*)(signed long, signed long), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceUL
 **********************************************************/

void  __real__upcr_all_reduceUL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceUL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceUL(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceUL(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned long (*)(unsigned long, unsigned long), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceUL(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceUL
 **********************************************************/

void  __real__upcr_all_prefix_reduceUL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceUL(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceUL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceUL(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, unsigned long (*)(unsigned long, unsigned long), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceUL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceF
 **********************************************************/

void  __real__upcr_all_reduceF(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceF(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceF(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceF(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, float (*)(float, float), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceF(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceF
 **********************************************************/

void  __real__upcr_all_prefix_reduceF(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceF(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceF(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceF(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, float (*)(float, float), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceF(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceD
 **********************************************************/

void  __real__upcr_all_reduceD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceD(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceD(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, double (*)(double, double), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceD(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceD
 **********************************************************/

void  __real__upcr_all_prefix_reduceD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceD(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, double (*)(double, double), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_reduceLD
 **********************************************************/

void  __real__upcr_all_reduceLD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7, int a8) ;
void  __wrap__upcr_all_reduceLD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7, int a8)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_reduceLD(a1, a2, a3, a4, a5, a6, a7, a8);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_reduceLD(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, long double (*)(long double, long double), upc_flag_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_reduceLD(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_prefix_reduceLD
 **********************************************************/

void  __real__upcr_all_prefix_reduceLD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7) ;
void  __wrap__upcr_all_prefix_reduceLD(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_prefix_reduceLD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_prefix_reduceLD(upcr_shared_ptr_t, upcr_shared_ptr_t, upc_op_t, size_t, size_t, long double (*)(long double, long double), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_prefix_reduceLD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fopen
 **********************************************************/

upcr_pshared_ptr_t  __real__upcr_all_fopen(const char * a1, int a2, size_t a3, const upc_hint_t * a4) ;
upcr_pshared_ptr_t  __wrap__upcr_all_fopen(const char * a1, int a2, size_t a3, const upc_hint_t * a4)  {

  upcr_pshared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fopen(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_pshared_ptr_t _upcr_all_fopen(const char *, int, size_t, const upc_hint_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fopen(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fclose
 **********************************************************/

int  __real__upcr_all_fclose(upcr_pshared_ptr_t a1) ;
int  __wrap__upcr_all_fclose(upcr_pshared_ptr_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fclose(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_all_fclose(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fclose(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fsync
 **********************************************************/

int  __real__upcr_all_fsync(upcr_pshared_ptr_t a1) ;
int  __wrap__upcr_all_fsync(upcr_pshared_ptr_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fsync(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_all_fsync(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fsync(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fseek
 **********************************************************/

upc_off_t  __real__upcr_all_fseek(upcr_pshared_ptr_t a1, upc_off_t a2, int a3) ;
upc_off_t  __wrap__upcr_all_fseek(upcr_pshared_ptr_t a1, upc_off_t a2, int a3)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fseek(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fseek(upcr_pshared_ptr_t, upc_off_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fseek(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fset_size
 **********************************************************/

int  __real__upcr_all_fset_size(upcr_pshared_ptr_t a1, upc_off_t a2) ;
int  __wrap__upcr_all_fset_size(upcr_pshared_ptr_t a1, upc_off_t a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fset_size(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_all_fset_size(upcr_pshared_ptr_t, upc_off_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fset_size(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fget_size
 **********************************************************/

upc_off_t  __real__upcr_all_fget_size(upcr_pshared_ptr_t a1) ;
upc_off_t  __wrap__upcr_all_fget_size(upcr_pshared_ptr_t a1)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fget_size(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fget_size(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fget_size(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fpreallocate
 **********************************************************/

int  __real__upcr_all_fpreallocate(upcr_pshared_ptr_t a1, upc_off_t a2) ;
int  __wrap__upcr_all_fpreallocate(upcr_pshared_ptr_t a1, upc_off_t a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fpreallocate(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_all_fpreallocate(upcr_pshared_ptr_t, upc_off_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fpreallocate(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fcntl
 **********************************************************/

int  __real__upcr_all_fcntl(upcr_pshared_ptr_t a1, int a2, void * a3) ;
int  __wrap__upcr_all_fcntl(upcr_pshared_ptr_t a1, int a2, void * a3)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fcntl(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _upcr_all_fcntl(upcr_pshared_ptr_t, int, void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fcntl(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fread_local
 **********************************************************/

upc_off_t  __real__upcr_all_fread_local(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5) ;
upc_off_t  __wrap__upcr_all_fread_local(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fread_local(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fread_local(upcr_pshared_ptr_t, void *, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fread_local(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fwrite_local
 **********************************************************/

upc_off_t  __real__upcr_all_fwrite_local(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5) ;
upc_off_t  __wrap__upcr_all_fwrite_local(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fwrite_local(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fwrite_local(upcr_pshared_ptr_t, void *, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fwrite_local(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fread_shared
 **********************************************************/

upc_off_t  __real__upcr_all_fread_shared(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6) ;
upc_off_t  __wrap__upcr_all_fread_shared(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fread_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fread_shared(upcr_pshared_ptr_t, bupc_sharedptr_t, size_t, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fread_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fwrite_shared
 **********************************************************/

upc_off_t  __real__upcr_all_fwrite_shared(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6) ;
upc_off_t  __wrap__upcr_all_fwrite_shared(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fwrite_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fwrite_shared(upcr_pshared_ptr_t, bupc_sharedptr_t, size_t, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fwrite_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fread_list_local
 **********************************************************/

upc_off_t  __real__upcr_all_fread_list_local(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
upc_off_t  __wrap__upcr_all_fread_list_local(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fread_list_local(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fread_list_local(upcr_pshared_ptr_t, size_t, const upc_local_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fread_list_local(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fread_list_shared
 **********************************************************/

upc_off_t  __real__upcr_all_fread_list_shared(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
upc_off_t  __wrap__upcr_all_fread_list_shared(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fread_list_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fread_list_shared(upcr_pshared_ptr_t, size_t, const upc_shared_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fread_list_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fwrite_list_local
 **********************************************************/

upc_off_t  __real__upcr_all_fwrite_list_local(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
upc_off_t  __wrap__upcr_all_fwrite_list_local(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fwrite_list_local(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fwrite_list_local(upcr_pshared_ptr_t, size_t, const upc_local_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fwrite_list_local(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fwrite_list_shared
 **********************************************************/

upc_off_t  __real__upcr_all_fwrite_list_shared(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
upc_off_t  __wrap__upcr_all_fwrite_list_shared(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fwrite_list_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fwrite_list_shared(upcr_pshared_ptr_t, size_t, const upc_shared_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fwrite_list_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_fread_local_async
 **********************************************************/

void  __real__upcr_all_fread_local_async(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5) ;
void  __wrap__upcr_all_fread_local_async(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fread_local_async(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fread_local_async(upcr_pshared_ptr_t, void *, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fread_local_async(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fwrite_local_async
 **********************************************************/

void  __real__upcr_all_fwrite_local_async(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5) ;
void  __wrap__upcr_all_fwrite_local_async(upcr_pshared_ptr_t a1, void * a2, size_t a3, size_t a4, upc_flag_t a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fwrite_local_async(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fwrite_local_async(upcr_pshared_ptr_t, void *, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fwrite_local_async(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fread_shared_async
 **********************************************************/

void  __real__upcr_all_fread_shared_async(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6) ;
void  __wrap__upcr_all_fread_shared_async(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fread_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fread_shared_async(upcr_pshared_ptr_t, bupc_sharedptr_t, size_t, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fread_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fwrite_shared_async
 **********************************************************/

void  __real__upcr_all_fwrite_shared_async(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6) ;
void  __wrap__upcr_all_fwrite_shared_async(upcr_pshared_ptr_t a1, bupc_sharedptr_t a2, size_t a3, size_t a4, size_t a5, upc_flag_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fwrite_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fwrite_shared_async(upcr_pshared_ptr_t, bupc_sharedptr_t, size_t, size_t, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fwrite_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fread_list_local_async
 **********************************************************/

void  __real__upcr_all_fread_list_local_async(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
void  __wrap__upcr_all_fread_list_local_async(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fread_list_local_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fread_list_local_async(upcr_pshared_ptr_t, size_t, const upc_local_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fread_list_local_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fread_list_shared_async
 **********************************************************/

void  __real__upcr_all_fread_list_shared_async(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
void  __wrap__upcr_all_fread_list_shared_async(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fread_list_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fread_list_shared_async(upcr_pshared_ptr_t, size_t, const upc_shared_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fread_list_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fwrite_list_local_async
 **********************************************************/

void  __real__upcr_all_fwrite_list_local_async(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
void  __wrap__upcr_all_fwrite_list_local_async(upcr_pshared_ptr_t a1, size_t a2, const upc_local_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fwrite_list_local_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fwrite_list_local_async(upcr_pshared_ptr_t, size_t, const upc_local_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fwrite_list_local_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fwrite_list_shared_async
 **********************************************************/

void  __real__upcr_all_fwrite_list_shared_async(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6) ;
void  __wrap__upcr_all_fwrite_list_shared_async(upcr_pshared_ptr_t a1, size_t a2, const upc_shared_memvec_t * a3, size_t a4, const upc_filevec_t * a5, upc_flag_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_fwrite_list_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_fwrite_list_shared_async(upcr_pshared_ptr_t, size_t, const upc_shared_memvec_t *, size_t, const upc_filevec_t *, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_fwrite_list_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_fwait_async
 **********************************************************/

upc_off_t  __real__upcr_all_fwait_async(upcr_pshared_ptr_t a1) ;
upc_off_t  __wrap__upcr_all_fwait_async(upcr_pshared_ptr_t a1)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_fwait_async(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_fwait_async(upcr_pshared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_fwait_async(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_ftest_async
 **********************************************************/

upc_off_t  __real__upcr_all_ftest_async(upcr_pshared_ptr_t a1, int * a2) ;
upc_off_t  __wrap__upcr_all_ftest_async(upcr_pshared_ptr_t a1, int * a2)  {

  upc_off_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_ftest_async(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t _upcr_all_ftest_async(upcr_pshared_ptr_t, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_ftest_async(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

#warning "TAU: Not generating wrapper for function _upcr_alloc"

/**********************************************************
   _upcr_local_alloc
 **********************************************************/

upcr_shared_ptr_t  __real__upcr_local_alloc(size_t a1, size_t a2) ;
upcr_shared_ptr_t  __wrap__upcr_local_alloc(size_t a1, size_t a2)  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_local_alloc(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _upcr_local_alloc(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_local_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_global_alloc
 **********************************************************/

upcr_shared_ptr_t  __real__upcr_global_alloc(size_t a1, size_t a2) ;
upcr_shared_ptr_t  __wrap__upcr_global_alloc(size_t a1, size_t a2)  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_global_alloc(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _upcr_global_alloc(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_global_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_all_alloc
 **********************************************************/

upcr_shared_ptr_t  __real__upcr_all_alloc(size_t a1, size_t a2) ;
upcr_shared_ptr_t  __wrap__upcr_all_alloc(size_t a1, size_t a2)  {

  upcr_shared_ptr_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_all_alloc(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_shared_ptr_t _upcr_all_alloc(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__upcr_all_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_free
 **********************************************************/

void  __real__upcr_free(upcr_shared_ptr_t a1) ;
void  __wrap__upcr_free(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_free(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_free(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_all_free
 **********************************************************/

void  __real__upcr_all_free(upcr_shared_ptr_t a1) ;
void  __wrap__upcr_all_free(upcr_shared_ptr_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_all_free(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_all_free(upcr_shared_ptr_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_all_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcri_getheapstats
 **********************************************************/

void  __real_upcri_getheapstats(const char * a1, char * a2, size_t a3) ;
void  __wrap_upcri_getheapstats(const char * a1, char * a2, size_t a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcri_getheapstats(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcri_getheapstats(const char *, char *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcri_getheapstats(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcri_get_handlertable_count
 **********************************************************/

size_t  __real_upcri_get_handlertable_count() ;
size_t  __wrap_upcri_get_handlertable_count()  {

  size_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upcri_get_handlertable_count();
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upcri_get_handlertable_count(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upcri_get_handlertable_count();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _upcr_do_memcpy
 **********************************************************/

upcr_handle_t  __real__upcr_do_memcpy(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, int a4) ;
upcr_handle_t  __wrap__upcr_do_memcpy(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3, int a4)  {

  upcr_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__upcr_do_memcpy(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"upcr_handle_t _upcr_do_memcpy(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  size_t dst_thread = upcr_threadof_shared(a1);
  size_t src_thread = upcr_threadof_shared(a2);
  size_t my_thread = upcr_mythread();
  if (my_thread == src_thread) {
    TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_thread, a3);
  } else {
    TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_thread, a3, src_thread);
  }

  retval  =  __real__upcr_do_memcpy(a1, a2, a3, a4);
  if (my_thread == src_thread) {
    TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, my_thread, a3, dst_thread);
  } else {
    TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_thread, a3);
  }

  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_waitsync
 **********************************************************/

void  __real__bupc_waitsync(bupc_handle_t a1) ;
void  __wrap__bupc_waitsync(bupc_handle_t a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_waitsync(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_waitsync(bupc_handle_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_waitsync(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_memcpy_vlist_async
 **********************************************************/

bupc_handle_t  __real__bupc_memcpy_vlist_async(size_t a1, const bupc_smemvec_t * a2, size_t a3, const bupc_smemvec_t * a4) ;
bupc_handle_t  __wrap__bupc_memcpy_vlist_async(size_t a1, const bupc_smemvec_t * a2, size_t a3, const bupc_smemvec_t * a4)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memcpy_vlist_async(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memcpy_vlist_async(size_t, const bupc_smemvec_t *, size_t, const bupc_smemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memcpy_vlist_async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memput_vlist_async
 **********************************************************/

bupc_handle_t  __real__bupc_memput_vlist_async(size_t a1, const bupc_smemvec_t * a2, size_t a3, const bupc_pmemvec_t * a4) ;
bupc_handle_t  __wrap__bupc_memput_vlist_async(size_t a1, const bupc_smemvec_t * a2, size_t a3, const bupc_pmemvec_t * a4)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memput_vlist_async(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memput_vlist_async(size_t, const bupc_smemvec_t *, size_t, const bupc_pmemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memput_vlist_async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memget_vlist_async
 **********************************************************/

bupc_handle_t  __real__bupc_memget_vlist_async(size_t a1, const bupc_pmemvec_t * a2, size_t a3, const bupc_smemvec_t * a4) ;
bupc_handle_t  __wrap__bupc_memget_vlist_async(size_t a1, const bupc_pmemvec_t * a2, size_t a3, const bupc_smemvec_t * a4)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memget_vlist_async(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memget_vlist_async(size_t, const bupc_pmemvec_t *, size_t, const bupc_smemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memget_vlist_async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memcpy_ilist_async
 **********************************************************/

bupc_handle_t  __real__bupc_memcpy_ilist_async(size_t a1, const upcr_shared_ptr_t * a2, size_t a3, size_t a4, const upcr_shared_ptr_t * a5, size_t a6) ;
bupc_handle_t  __wrap__bupc_memcpy_ilist_async(size_t a1, const upcr_shared_ptr_t * a2, size_t a3, size_t a4, const upcr_shared_ptr_t * a5, size_t a6)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memcpy_ilist_async(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memcpy_ilist_async(size_t, const upcr_shared_ptr_t *, size_t, size_t, const upcr_shared_ptr_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memcpy_ilist_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memput_ilist_async
 **********************************************************/

bupc_handle_t  __real__bupc_memput_ilist_async(size_t a1, const upcr_shared_ptr_t * a2, size_t a3, size_t a4, const void *const * a5, size_t a6) ;
bupc_handle_t  __wrap__bupc_memput_ilist_async(size_t a1, const upcr_shared_ptr_t * a2, size_t a3, size_t a4, const void *const * a5, size_t a6)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memput_ilist_async(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memput_ilist_async(size_t, const upcr_shared_ptr_t *, size_t, size_t, const void *const *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memput_ilist_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memget_ilist_async
 **********************************************************/

bupc_handle_t  __real__bupc_memget_ilist_async(size_t a1, void *const * a2, size_t a3, size_t a4, const upcr_shared_ptr_t * a5, size_t a6) ;
bupc_handle_t  __wrap__bupc_memget_ilist_async(size_t a1, void *const * a2, size_t a3, size_t a4, const upcr_shared_ptr_t * a5, size_t a6)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memget_ilist_async(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memget_ilist_async(size_t, void *const *, size_t, size_t, const upcr_shared_ptr_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memget_ilist_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memcpy_fstrided_async
 **********************************************************/

bupc_handle_t  __real__bupc_memcpy_fstrided_async(upcr_shared_ptr_t a1, size_t a2, size_t a3, size_t a4, upcr_shared_ptr_t a5, size_t a6, size_t a7, size_t a8) ;
bupc_handle_t  __wrap__bupc_memcpy_fstrided_async(upcr_shared_ptr_t a1, size_t a2, size_t a3, size_t a4, upcr_shared_ptr_t a5, size_t a6, size_t a7, size_t a8)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memcpy_fstrided_async(a1, a2, a3, a4, a5, a6, a7, a8);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memcpy_fstrided_async(upcr_shared_ptr_t, size_t, size_t, size_t, upcr_shared_ptr_t, size_t, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memcpy_fstrided_async(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memput_fstrided_async
 **********************************************************/

bupc_handle_t  __real__bupc_memput_fstrided_async(upcr_shared_ptr_t a1, size_t a2, size_t a3, size_t a4, void * a5, size_t a6, size_t a7, size_t a8) ;
bupc_handle_t  __wrap__bupc_memput_fstrided_async(upcr_shared_ptr_t a1, size_t a2, size_t a3, size_t a4, void * a5, size_t a6, size_t a7, size_t a8)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memput_fstrided_async(a1, a2, a3, a4, a5, a6, a7, a8);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memput_fstrided_async(upcr_shared_ptr_t, size_t, size_t, size_t, void *, size_t, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memput_fstrided_async(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memget_fstrided_async
 **********************************************************/

bupc_handle_t  __real__bupc_memget_fstrided_async(void * a1, size_t a2, size_t a3, size_t a4, upcr_shared_ptr_t a5, size_t a6, size_t a7, size_t a8) ;
bupc_handle_t  __wrap__bupc_memget_fstrided_async(void * a1, size_t a2, size_t a3, size_t a4, upcr_shared_ptr_t a5, size_t a6, size_t a7, size_t a8)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memget_fstrided_async(a1, a2, a3, a4, a5, a6, a7, a8);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memget_fstrided_async(void *, size_t, size_t, size_t, upcr_shared_ptr_t, size_t, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memget_fstrided_async(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memcpy_strided_async
 **********************************************************/

bupc_handle_t  __real__bupc_memcpy_strided_async(upcr_shared_ptr_t a1, const size_t * a2, upcr_shared_ptr_t a3, const size_t * a4, const size_t * a5, size_t a6) ;
bupc_handle_t  __wrap__bupc_memcpy_strided_async(upcr_shared_ptr_t a1, const size_t * a2, upcr_shared_ptr_t a3, const size_t * a4, const size_t * a5, size_t a6)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memcpy_strided_async(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memcpy_strided_async(upcr_shared_ptr_t, const size_t *, upcr_shared_ptr_t, const size_t *, const size_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memcpy_strided_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memput_strided_async
 **********************************************************/

bupc_handle_t  __real__bupc_memput_strided_async(upcr_shared_ptr_t a1, const size_t * a2, const void * a3, const size_t * a4, const size_t * a5, size_t a6) ;
bupc_handle_t  __wrap__bupc_memput_strided_async(upcr_shared_ptr_t a1, const size_t * a2, const void * a3, const size_t * a4, const size_t * a5, size_t a6)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memput_strided_async(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memput_strided_async(upcr_shared_ptr_t, const size_t *, const void *, const size_t *, const size_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memput_strided_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memget_strided_async
 **********************************************************/

bupc_handle_t  __real__bupc_memget_strided_async(void * a1, const size_t * a2, upcr_shared_ptr_t a3, const size_t * a4, const size_t * a5, size_t a6) ;
bupc_handle_t  __wrap__bupc_memget_strided_async(void * a1, const size_t * a2, upcr_shared_ptr_t a3, const size_t * a4, const size_t * a5, size_t a6)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memget_strided_async(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memget_strided_async(void *, const size_t *, upcr_shared_ptr_t, const size_t *, const size_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_memget_strided_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_trysync
 **********************************************************/

int  __real__bupc_trysync(bupc_handle_t a1) ;
int  __wrap__bupc_trysync(bupc_handle_t a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_trysync(a1);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _bupc_trysync(bupc_handle_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_trysync(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_waitsync_all
 **********************************************************/

void  __real__bupc_waitsync_all(bupc_handle_t * a1, size_t a2) ;
void  __wrap__bupc_waitsync_all(bupc_handle_t * a1, size_t a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_waitsync_all(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_waitsync_all(bupc_handle_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_waitsync_all(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_trysync_all
 **********************************************************/

int  __real__bupc_trysync_all(bupc_handle_t * a1, size_t a2) ;
int  __wrap__bupc_trysync_all(bupc_handle_t * a1, size_t a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_trysync_all(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _bupc_trysync_all(bupc_handle_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_trysync_all(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_waitsync_some
 **********************************************************/

void  __real__bupc_waitsync_some(bupc_handle_t * a1, size_t a2) ;
void  __wrap__bupc_waitsync_some(bupc_handle_t * a1, size_t a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__bupc_waitsync_some(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _bupc_waitsync_some(bupc_handle_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__bupc_waitsync_some(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _bupc_trysync_some
 **********************************************************/

int  __real__bupc_trysync_some(bupc_handle_t * a1, size_t a2) ;
int  __wrap__bupc_trysync_some(bupc_handle_t * a1, size_t a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_trysync_some(a1, a2);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"int _bupc_trysync_some(bupc_handle_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_trysync_some(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memcpy_async
 **********************************************************/

bupc_handle_t  __real__bupc_memcpy_async(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3) ;
bupc_handle_t  __wrap__bupc_memcpy_async(upcr_shared_ptr_t a1, upcr_shared_ptr_t a2, size_t a3)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memcpy_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memcpy_async(upcr_shared_ptr_t, upcr_shared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  size_t dst_thread = upcr_threadof_shared(a1);
  size_t src_thread = upcr_threadof_shared(a2);
  size_t my_thread = upcr_mythread();
  if (my_thread == src_thread) {
    TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_thread, a3);
  } else {
    TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_thread, a3, src_thread);
  }

  retval  =  __real__bupc_memcpy_async(a1, a2, a3);
  if (my_thread == src_thread) {
    TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, my_thread, a3, dst_thread);
  } else {
    TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_thread, a3);
  }

  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memget_async
 **********************************************************/

bupc_handle_t  __real__bupc_memget_async(void * a1, upcr_shared_ptr_t a2, size_t a3) ;
bupc_handle_t  __wrap__bupc_memget_async(void * a1, upcr_shared_ptr_t a2, size_t a3)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memget_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memget_async(void *, upcr_shared_ptr_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, upcr_mythread(), a3, upcr_threadof_shared(a2));
  retval  =  __real__bupc_memget_async(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, upcr_threadof_shared(a2), a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memput_async
 **********************************************************/

bupc_handle_t  __real__bupc_memput_async(upcr_shared_ptr_t a1, const void * a2, size_t a3) ;
bupc_handle_t  __wrap__bupc_memput_async(upcr_shared_ptr_t a1, const void * a2, size_t a3)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memput_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memput_async(upcr_shared_ptr_t, const void *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, upcr_threadof_shared(a1), a3);
  retval  =  __real__bupc_memput_async(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, upcr_mythread(), a3, upcr_threadof_shared(a1));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_memset_async
 **********************************************************/

bupc_handle_t  __real__bupc_memset_async(upcr_shared_ptr_t a1, int a2, size_t a3) ;
bupc_handle_t  __wrap__bupc_memset_async(upcr_shared_ptr_t a1, int a2, size_t a3)  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_memset_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_memset_async(upcr_shared_ptr_t, int, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, upcr_threadof_shared(a1), a3);
  retval  =  __real__bupc_memset_async(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, upcr_mythread(), a3, upcr_threadof_shared(a1));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   _bupc_end_accessregion
 **********************************************************/

bupc_handle_t  __real__bupc_end_accessregion() ;
bupc_handle_t  __wrap__bupc_end_accessregion()  {

  bupc_handle_t retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real__bupc_end_accessregion();
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"bupc_handle_t _bupc_end_accessregion(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__bupc_end_accessregion();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upcr_startup_init
 **********************************************************/

void  __real_upcr_startup_init(int * a1, char *** a2, upcr_thread_t a3, upcr_thread_t a4, const char * a5) ;
void  __wrap_upcr_startup_init(int * a1, char *** a2, upcr_thread_t a3, upcr_thread_t a4, const char * a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcr_startup_init(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcr_startup_init(int *, char ***, upcr_thread_t, upcr_thread_t, const char *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcr_startup_init(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcr_startup_attach
 **********************************************************/

void  __real_upcr_startup_attach(uintptr_t a1, uintptr_t a2, int a3) ;
void  __wrap_upcr_startup_attach(uintptr_t a1, uintptr_t a2, int a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcr_startup_attach(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcr_startup_attach(uintptr_t, uintptr_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcr_startup_attach(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcr_startup_spawn
 **********************************************************/

void  __real_upcr_startup_spawn(int * a1, char *** a2, uintptr_t a3, uintptr_t a4, struct upcr_startup_spawnfuncs * a5) ;
void  __wrap_upcr_startup_spawn(int * a1, char *** a2, uintptr_t a3, uintptr_t a4, struct upcr_startup_spawnfuncs * a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcr_startup_spawn(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcr_startup_spawn(int *, char ***, uintptr_t, uintptr_t, struct upcr_startup_spawnfuncs *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcr_startup_spawn(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcr_exit
 **********************************************************/

void  __real_upcr_exit(int a1) ;
void  __wrap_upcr_exit(int a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcr_exit(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcr_exit(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcr_exit(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcr_global_exit
 **********************************************************/

void  __real_upcr_global_exit(int a1) ;
void  __wrap_upcr_global_exit(int a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcr_global_exit(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcr_global_exit(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcr_global_exit(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upcri_do_exit
 **********************************************************/

void  __real_upcri_do_exit(int a1) ;
void  __wrap_upcri_do_exit(int a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upcri_do_exit(a1);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void upcri_do_exit(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upcri_do_exit(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_startup_shalloc
 **********************************************************/

void  __real__upcr_startup_shalloc(upcr_startup_shalloc_t * a1, size_t a2) ;
void  __wrap__upcr_startup_shalloc(upcr_startup_shalloc_t * a1, size_t a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_startup_shalloc(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_startup_shalloc(upcr_startup_shalloc_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_startup_shalloc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_startup_pshalloc
 **********************************************************/

void  __real__upcr_startup_pshalloc(upcr_startup_pshalloc_t * a1, size_t a2) ;
void  __wrap__upcr_startup_pshalloc(upcr_startup_pshalloc_t * a1, size_t a2)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_startup_pshalloc(a1, a2);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_startup_pshalloc(upcr_startup_pshalloc_t *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_startup_pshalloc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_startup_initarray
 **********************************************************/

void  __real__upcr_startup_initarray(upcr_shared_ptr_t a1, void * a2, upcr_startup_arrayinit_diminfo_t * a3, size_t a4, size_t a5, size_t a6) ;
void  __wrap__upcr_startup_initarray(upcr_shared_ptr_t a1, void * a2, upcr_startup_arrayinit_diminfo_t * a3, size_t a4, size_t a5, size_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_startup_initarray(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_startup_initarray(upcr_shared_ptr_t, void *, upcr_startup_arrayinit_diminfo_t *, size_t, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_startup_initarray(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   _upcr_startup_initparray
 **********************************************************/

void  __real__upcr_startup_initparray(upcr_pshared_ptr_t a1, void * a2, upcr_startup_arrayinit_diminfo_t * a3, size_t a4, size_t a5, size_t a6) ;
void  __wrap__upcr_startup_initparray(upcr_pshared_ptr_t a1, void * a2, upcr_startup_arrayinit_diminfo_t * a3, size_t a4, size_t a5, size_t a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real__upcr_startup_initparray(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,upcr_threads());
    }
  }

  TAU_PROFILE_TIMER(t,"void _upcr_startup_initparray(upcr_pshared_ptr_t, void *, upcr_startup_arrayinit_diminfo_t *, size_t, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real__upcr_startup_initparray(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}

