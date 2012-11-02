#include <cray_upc_runtime.h>
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


/**********************************************************
   upc_global_exit
 **********************************************************/

void   __real_upc_global_exit(int  a1) ;
void   __wrap_upc_global_exit(int  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_global_exit(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_global_exit(int)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_global_exit(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_global_alloc
 **********************************************************/

shared   void *   __real_upc_global_alloc(size_t  a1, size_t  a2) ;
shared   void *   __wrap_upc_global_alloc(size_t  a1, size_t  a2)  {

  shared   void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_global_alloc(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"shared[1] void * upc_global_alloc(size_t, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_global_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_alloc
 **********************************************************/

shared   void *   __real_upc_all_alloc(size_t  a1, size_t  a2) ;
shared   void *   __wrap_upc_all_alloc(size_t  a1, size_t  a2)  {

  shared   void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_alloc(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"shared[1] void * upc_all_alloc(size_t, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_alloc
 **********************************************************/

shared   void *   __real_upc_alloc(size_t  a1) ;
shared   void *   __wrap_upc_alloc(size_t  a1)  {

  shared   void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_alloc(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"shared[1] void * upc_alloc(size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_alloc(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_free
 **********************************************************/

void   __real_upc_free(shared[1] void *  a1) ;
void   __wrap_upc_free(shared[1] void *  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_free(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_free(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_threadof
 **********************************************************/

size_t   __real_upc_threadof(shared[1] const void *  a1) ;
size_t   __wrap_upc_threadof(shared[1] const void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_threadof(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_threadof(shared[1] const void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_threadof(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_phaseof
 **********************************************************/

size_t   __real_upc_phaseof(shared[1] const void *  a1) ;
size_t   __wrap_upc_phaseof(shared[1] const void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_phaseof(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_phaseof(shared[1] const void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_phaseof(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_resetphase
 **********************************************************/

shared   void *   __real_upc_resetphase(shared[1] const void *  a1) ;
shared   void *   __wrap_upc_resetphase(shared[1] const void *  a1)  {

  shared   void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_resetphase(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"shared[1] void * upc_resetphase(shared[1] const void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_resetphase(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_addrfield
 **********************************************************/

size_t   __real_upc_addrfield(shared[1] const void *  a1) ;
size_t   __wrap_upc_addrfield(shared[1] const void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_addrfield(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_addrfield(shared[1] const void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_addrfield(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_affinitysize
 **********************************************************/

size_t   __real_upc_affinitysize(size_t  a1, size_t  a2, size_t  a3) ;
size_t   __wrap_upc_affinitysize(size_t  a1, size_t  a2, size_t  a3)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_affinitysize(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_affinitysize(size_t, size_t, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_affinitysize(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_global_lock_alloc
 **********************************************************/

upc_lock_t *   __real_upc_global_lock_alloc() ;
upc_lock_t *   __wrap_upc_global_lock_alloc()  {

  upc_lock_t *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_global_lock_alloc();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_lock_t * upc_global_lock_alloc()  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_global_lock_alloc();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_lock_alloc
 **********************************************************/

upc_lock_t *   __real_upc_all_lock_alloc() ;
upc_lock_t *   __wrap_upc_all_lock_alloc()  {

  upc_lock_t *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_lock_alloc();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_lock_t * upc_all_lock_alloc()  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_lock_alloc();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_lock_free
 **********************************************************/

void   __real_upc_lock_free(upc_lock_t *  a1) ;
void   __wrap_upc_lock_free(upc_lock_t *  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_lock_free(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_lock_free(upc_lock_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_lock_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_lock
 **********************************************************/

void   __real_upc_lock(upc_lock_t *  a1) ;
void   __wrap_upc_lock(upc_lock_t *  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_lock(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_lock(upc_lock_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_lock_attempt
 **********************************************************/

int   __real_upc_lock_attempt(upc_lock_t *  a1) ;
int   __wrap_upc_lock_attempt(upc_lock_t *  a1)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_lock_attempt(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_lock_attempt(upc_lock_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_lock_attempt(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_unlock
 **********************************************************/

void   __real_upc_unlock(upc_lock_t *  a1) ;
void   __wrap_upc_unlock(upc_lock_t *  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_unlock(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_unlock(upc_lock_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_unlock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memcpy
 **********************************************************/

void   __real_upc_memcpy(shared[1] void *restrict  a1, shared[1] const void *restrict  a2, size_t  a3) ;
void   __wrap_upc_memcpy(shared[1] void *restrict  a1, shared[1] const void *restrict  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memcpy(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memcpy(shared[1] void *restrict, shared[1] const void *restrict, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  size_t dst_thread = __real_upc_threadof(a1);
  size_t src_thread = __real_upc_threadof(a2);
  size_t my_thread = MYTHREAD;
  if (my_thread == src_thread) {
    TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_thread, a3);
  } else {
    TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_thread, a3, src_thread);
  }

  __real_upc_memcpy(a1, a2, a3);
  if (my_thread == src_thread) {
    TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, my_thread, a3, dst_thread);
  } else {
    TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_thread, a3);
  }

  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memget
 **********************************************************/

void   __real_upc_memget(void *restrict  a1, shared[1] const void *restrict  a2, size_t  a3) ;
void   __wrap_upc_memget(void *restrict  a1, shared[1] const void *restrict  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memget(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memget(void *restrict, shared[1] const void *restrict, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, MYTHREAD, a3, __real_upc_threadof(a2));
  __real_upc_memget(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, __real_upc_threadof(a2), a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memput
 **********************************************************/

void   __real_upc_memput(shared[1] void *restrict  a1, const void *restrict  a2, size_t  a3) ;
void   __wrap_upc_memput(shared[1] void *restrict  a1, const void *restrict  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memput(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memput(shared[1] void *restrict, const void *restrict, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real_upc_memput(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memset
 **********************************************************/

void   __real_upc_memset(shared[1] void *  a1, int  a2, size_t  a3) ;
void   __wrap_upc_memset(shared[1] void *  a1, int  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memset(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memset(shared[1] void *, int, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real_upc_memset(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fopen
 **********************************************************/

upc_file_t *   __real_upc_all_fopen(const char *  a1, int  a2, size_t  a3, const struct upc_hint *  a4) ;
upc_file_t *   __wrap_upc_all_fopen(const char *  a1, int  a2, size_t  a3, const struct upc_hint *  a4)  {

  upc_file_t *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fopen(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_file_t * upc_all_fopen(const char *, int, size_t, const struct upc_hint *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fopen(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fclose
 **********************************************************/

int   __real_upc_all_fclose(upc_file_t *  a1) ;
int   __wrap_upc_all_fclose(upc_file_t *  a1)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fclose(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_all_fclose(upc_file_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fclose(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fsync
 **********************************************************/

int   __real_upc_all_fsync(upc_file_t *  a1) ;
int   __wrap_upc_all_fsync(upc_file_t *  a1)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fsync(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_all_fsync(upc_file_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fsync(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fseek
 **********************************************************/

upc_off_t   __real_upc_all_fseek(upc_file_t *  a1, upc_off_t  a2, int  a3) ;
upc_off_t   __wrap_upc_all_fseek(upc_file_t *  a1, upc_off_t  a2, int  a3)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fseek(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fseek(upc_file_t *, upc_off_t, int)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fseek(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fset_size
 **********************************************************/

int   __real_upc_all_fset_size(upc_file_t *  a1, upc_off_t  a2) ;
int   __wrap_upc_all_fset_size(upc_file_t *  a1, upc_off_t  a2)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fset_size(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_all_fset_size(upc_file_t *, upc_off_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fset_size(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fget_size
 **********************************************************/

upc_off_t   __real_upc_all_fget_size(upc_file_t *  a1) ;
upc_off_t   __wrap_upc_all_fget_size(upc_file_t *  a1)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fget_size(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fget_size(upc_file_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fget_size(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fpreallocate
 **********************************************************/

int   __real_upc_all_fpreallocate(upc_file_t *  a1, upc_off_t  a2) ;
int   __wrap_upc_all_fpreallocate(upc_file_t *  a1, upc_off_t  a2)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fpreallocate(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_all_fpreallocate(upc_file_t *, upc_off_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fpreallocate(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fcntl
 **********************************************************/

int   __real_upc_all_fcntl(upc_file_t *  a1, int  a2, void *  a3) ;
int   __wrap_upc_all_fcntl(upc_file_t *  a1, int  a2, void *  a3)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fcntl(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_all_fcntl(upc_file_t *, int, void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fcntl(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fread_local
 **********************************************************/

upc_off_t   __real_upc_all_fread_local(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5) ;
upc_off_t   __wrap_upc_all_fread_local(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fread_local(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fread_local(upc_file_t *, void *, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fread_local(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fread_local_async
 **********************************************************/

void   __real_upc_all_fread_local_async(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5) ;
void   __wrap_upc_all_fread_local_async(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fread_local_async(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fread_local_async(upc_file_t *, void *, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fread_local_async(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fread_shared
 **********************************************************/

upc_off_t   __real_upc_all_fread_shared(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6) ;
upc_off_t   __wrap_upc_all_fread_shared(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fread_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fread_shared(upc_file_t *, shared[1] void *, size_t, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fread_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fread_shared_async
 **********************************************************/

void   __real_upc_all_fread_shared_async(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6) ;
void   __wrap_upc_all_fread_shared_async(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fread_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fread_shared_async(upc_file_t *, shared[1] void *, size_t, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fread_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fread_list_local
 **********************************************************/

upc_off_t   __real_upc_all_fread_list_local(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
upc_off_t   __wrap_upc_all_fread_list_local(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fread_list_local(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fread_list_local(upc_file_t *, size_t, const struct upc_local_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fread_list_local(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fread_list_local_async
 **********************************************************/

void   __real_upc_all_fread_list_local_async(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
void   __wrap_upc_all_fread_list_local_async(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fread_list_local_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fread_list_local_async(upc_file_t *, size_t, const struct upc_local_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fread_list_local_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fread_list_shared
 **********************************************************/

upc_off_t   __real_upc_all_fread_list_shared(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
upc_off_t   __wrap_upc_all_fread_list_shared(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fread_list_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fread_list_shared(upc_file_t *, size_t, const struct upc_shared_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fread_list_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fread_list_shared_async
 **********************************************************/

void   __real_upc_all_fread_list_shared_async(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
void   __wrap_upc_all_fread_list_shared_async(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fread_list_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fread_list_shared_async(upc_file_t *, size_t, const struct upc_shared_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fread_list_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fwrite_local
 **********************************************************/

upc_off_t   __real_upc_all_fwrite_local(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5) ;
upc_off_t   __wrap_upc_all_fwrite_local(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fwrite_local(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fwrite_local(upc_file_t *, void *, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fwrite_local(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fwrite_local_async
 **********************************************************/

void   __real_upc_all_fwrite_local_async(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5) ;
void   __wrap_upc_all_fwrite_local_async(upc_file_t *  a1, void *  a2, size_t  a3, size_t  a4, upc_flag_t  a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fwrite_local_async(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fwrite_local_async(upc_file_t *, void *, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fwrite_local_async(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fwrite_shared
 **********************************************************/

upc_off_t   __real_upc_all_fwrite_shared(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6) ;
upc_off_t   __wrap_upc_all_fwrite_shared(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fwrite_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fwrite_shared(upc_file_t *, shared[1] void *, size_t, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fwrite_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fwrite_shared_async
 **********************************************************/

void   __real_upc_all_fwrite_shared_async(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6) ;
void   __wrap_upc_all_fwrite_shared_async(upc_file_t *  a1, shared[1] void *  a2, size_t  a3, size_t  a4, size_t  a5, upc_flag_t  a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fwrite_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fwrite_shared_async(upc_file_t *, shared[1] void *, size_t, size_t, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fwrite_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fwrite_list_local
 **********************************************************/

upc_off_t   __real_upc_all_fwrite_list_local(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
upc_off_t   __wrap_upc_all_fwrite_list_local(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fwrite_list_local(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fwrite_list_local(upc_file_t *, size_t, const struct upc_local_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fwrite_list_local(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fwrite_list_local_async
 **********************************************************/

void   __real_upc_all_fwrite_list_local_async(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
void   __wrap_upc_all_fwrite_list_local_async(upc_file_t *  a1, size_t  a2, const struct upc_local_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fwrite_list_local_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fwrite_list_local_async(upc_file_t *, size_t, const struct upc_local_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fwrite_list_local_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fwrite_list_shared
 **********************************************************/

upc_off_t   __real_upc_all_fwrite_list_shared(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
upc_off_t   __wrap_upc_all_fwrite_list_shared(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fwrite_list_shared(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fwrite_list_shared(upc_file_t *, size_t, const struct upc_shared_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fwrite_list_shared(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_fwrite_list_shared_async
 **********************************************************/

void   __real_upc_all_fwrite_list_shared_async(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6) ;
void   __wrap_upc_all_fwrite_list_shared_async(upc_file_t *  a1, size_t  a2, const struct upc_shared_memvec *  a3, size_t  a4, const struct upc_filevec *  a5, upc_flag_t  a6)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_fwrite_list_shared_async(a1, a2, a3, a4, a5, a6);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_fwrite_list_shared_async(upc_file_t *, size_t, const struct upc_shared_memvec *, size_t, const struct upc_filevec *, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_fwrite_list_shared_async(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_fwait_async
 **********************************************************/

upc_off_t   __real_upc_all_fwait_async(upc_file_t *  a1) ;
upc_off_t   __wrap_upc_all_fwait_async(upc_file_t *  a1)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_fwait_async(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_fwait_async(upc_file_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_fwait_async(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_ftest_async
 **********************************************************/

upc_off_t   __real_upc_all_ftest_async(upc_file_t *  a1, int *  a2) ;
upc_off_t   __wrap_upc_all_ftest_async(upc_file_t *  a1, int *  a2)  {

  upc_off_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_ftest_async(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_off_t upc_all_ftest_async(upc_file_t *, int *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_ftest_async(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_broadcast
 **********************************************************/

void   __real_upc_all_broadcast(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4) ;
void   __wrap_upc_all_broadcast(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_broadcast(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_broadcast(shared[1] void *, shared[1] const void *, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_broadcast(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_scatter
 **********************************************************/

void   __real_upc_all_scatter(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4) ;
void   __wrap_upc_all_scatter(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_scatter(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_scatter(shared[1] void *, shared[1] const void *, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_scatter(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_gather
 **********************************************************/

void   __real_upc_all_gather(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4) ;
void   __wrap_upc_all_gather(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_gather(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_gather(shared[1] void *, shared[1] const void *, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_gather(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_gather_all
 **********************************************************/

void   __real_upc_all_gather_all(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4) ;
void   __wrap_upc_all_gather_all(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_gather_all(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_gather_all(shared[1] void *, shared[1] const void *, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_gather_all(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_exchange
 **********************************************************/

void   __real_upc_all_exchange(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4) ;
void   __wrap_upc_all_exchange(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3, upc_flag_t  a4)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_exchange(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_exchange(shared[1] void *, shared[1] const void *, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_exchange(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_permute
 **********************************************************/

void   __real_upc_all_permute(shared[1] void *  a1, shared[1] const void *  a2, shared[1] const int *  a3, size_t  a4, upc_flag_t  a5) ;
void   __wrap_upc_all_permute(shared[1] void *  a1, shared[1] const void *  a2, shared[1] const int *  a3, size_t  a4, upc_flag_t  a5)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_permute(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_permute(shared[1] void *, shared[1] const void *, shared[1] const int *, size_t, upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_permute(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceC
 **********************************************************/

void   __real_upc_all_reduceC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, signed char (*a6) (signed char, signed char) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, signed char (*a6) (signed char, signed char) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceC(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, signed char (*) (signed char, signed char), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUC
 **********************************************************/

void   __real_upc_all_reduceUC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned char (*a6) (unsigned char, unsigned char) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceUC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned char (*a6) (unsigned char, unsigned char) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUC(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned char (*) (unsigned char, unsigned char), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceS
 **********************************************************/

void   __real_upc_all_reduceS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, short (*a6) (short, short) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, short (*a6) (short, short) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceS(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, short (*) (short, short), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUS
 **********************************************************/

void   __real_upc_all_reduceUS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned short (*a6) (unsigned short, unsigned short) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceUS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned short (*a6) (unsigned short, unsigned short) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUS(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned short (*) (unsigned short, unsigned short), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceI
 **********************************************************/

void   __real_upc_all_reduceI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, int (*a6) (int, int) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, int (*a6) (int, int) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceI(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, int (*) (int, int), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUI
 **********************************************************/

void   __real_upc_all_reduceUI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned int (*a6) (unsigned int, unsigned int) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceUI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned int (*a6) (unsigned int, unsigned int) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUI(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned int (*) (unsigned int, unsigned int), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceL
 **********************************************************/

void   __real_upc_all_reduceL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long (*a6) (long, long) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long (*a6) (long, long) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceL(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, long (*) (long, long), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUL
 **********************************************************/

void   __real_upc_all_reduceUL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned long (*a6) (unsigned long, unsigned long) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceUL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned long (*a6) (unsigned long, unsigned long) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUL(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned long (*) (unsigned long, unsigned long), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceF
 **********************************************************/

void   __real_upc_all_reduceF(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, float (*a6) (float, float) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceF(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, float (*a6) (float, float) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceF(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceF(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, float (*) (float, float), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceF(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceD
 **********************************************************/

void   __real_upc_all_reduceD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, double (*a6) (double, double) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, double (*a6) (double, double) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceD(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, double (*) (double, double), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceLD
 **********************************************************/

void   __real_upc_all_reduceLD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long double (*a6) (long double, long double) , upc_flag_t  a7) ;
void   __wrap_upc_all_reduceLD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long double (*a6) (long double, long double) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceLD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceLD(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, long double (*) (long double, long double), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceLD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceC
 **********************************************************/

void   __real_upc_all_prefix_reduceC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, signed char (*a6) (signed char, signed char) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, signed char (*a6) (signed char, signed char) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceC(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, signed char (*) (signed char, signed char), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUC
 **********************************************************/

void   __real_upc_all_prefix_reduceUC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned char (*a6) (unsigned char, unsigned char) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceUC(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned char (*a6) (unsigned char, unsigned char) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUC(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned char (*) (unsigned char, unsigned char), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceS
 **********************************************************/

void   __real_upc_all_prefix_reduceS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, short (*a6) (short, short) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, short (*a6) (short, short) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceS(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, short (*) (short, short), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUS
 **********************************************************/

void   __real_upc_all_prefix_reduceUS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned short (*a6) (unsigned short, unsigned short) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceUS(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned short (*a6) (unsigned short, unsigned short) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUS(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned short (*) (unsigned short, unsigned short), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceI
 **********************************************************/

void   __real_upc_all_prefix_reduceI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, int (*a6) (int, int) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, int (*a6) (int, int) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceI(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, int (*) (int, int), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUI
 **********************************************************/

void   __real_upc_all_prefix_reduceUI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned int (*a6) (unsigned int, unsigned int) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceUI(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned int (*a6) (unsigned int, unsigned int) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUI(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned int (*) (unsigned int, unsigned int), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceL
 **********************************************************/

void   __real_upc_all_prefix_reduceL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long (*a6) (long, long) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long (*a6) (long, long) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceL(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, long (*) (long, long), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUL
 **********************************************************/

void   __real_upc_all_prefix_reduceUL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned long (*a6) (unsigned long, unsigned long) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceUL(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, unsigned long (*a6) (unsigned long, unsigned long) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUL(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, unsigned long (*) (unsigned long, unsigned long), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceF
 **********************************************************/

void   __real_upc_all_prefix_reduceF(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, float (*a6) (float, float) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceF(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, float (*a6) (float, float) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceF(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceF(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, float (*) (float, float), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceF(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceD
 **********************************************************/

void   __real_upc_all_prefix_reduceD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, double (*a6) (double, double) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, double (*a6) (double, double) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceD(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, double (*) (double, double), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceLD
 **********************************************************/

void   __real_upc_all_prefix_reduceLD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long double (*a6) (long double, long double) , upc_flag_t  a7) ;
void   __wrap_upc_all_prefix_reduceLD(shared[1] void *  a1, shared[1] const void *  a2, upc_op_t  a3, size_t  a4, size_t  a5, long double (*a6) (long double, long double) , upc_flag_t  a7)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceLD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceLD(shared[1] void *, shared[1] const void *, upc_op_t, size_t, size_t, long double (*) (long double, long double), upc_flag_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceLD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_free
 **********************************************************/

void   __real_upc_all_free(shared[1] void *  a1) ;
void   __wrap_upc_all_free(shared[1] void *  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_free(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_free(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_lock_free
 **********************************************************/

void   __real_upc_all_lock_free(upc_lock_t *  a1) ;
void   __wrap_upc_all_lock_free(upc_lock_t *  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_lock_free(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_lock_free(upc_lock_t *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_lock_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_ticks_now
 **********************************************************/

upc_tick_t   __real_upc_ticks_now() ;
upc_tick_t   __wrap_upc_ticks_now()  {

  upc_tick_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_ticks_now();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_tick_t upc_ticks_now()  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_ticks_now();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_ticks_to_ns
 **********************************************************/

unsigned long   __real_upc_ticks_to_ns(upc_tick_t  a1) ;
unsigned long   __wrap_upc_ticks_to_ns(upc_tick_t  a1)  {

  unsigned long  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_ticks_to_ns(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"unsigned long upc_ticks_to_ns(upc_tick_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_ticks_to_ns(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_cast
 **********************************************************/

void *   __real_upc_cast(shared[1] void *  a1) ;
void *   __wrap_upc_cast(shared[1] void *  a1)  {

  void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_cast(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void * upc_cast(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_cast(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_castable
 **********************************************************/

int   __real_upc_castable(shared[1] void *  a1) ;
int   __wrap_upc_castable(shared[1] void *  a1)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_castable(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_castable(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_castable(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_thread_castable
 **********************************************************/

int   __real_upc_thread_castable(unsigned int  a1) ;
int   __wrap_upc_thread_castable(unsigned int  a1)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_thread_castable(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_thread_castable(unsigned int)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_thread_castable(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_memcpy_nb
 **********************************************************/

upc_handle_t   __real_upc_memcpy_nb(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3) ;
upc_handle_t   __wrap_upc_memcpy_nb(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3)  {

  upc_handle_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_memcpy_nb(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_handle_t upc_memcpy_nb(shared[1] void *, shared[1] const void *, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  size_t dst_thread = __real_upc_threadof(a1);
  size_t src_thread = __real_upc_threadof(a2);
  size_t my_thread = MYTHREAD;
  if (my_thread == src_thread) {
    TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_thread, a3);
  } else {
    TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_thread, a3, src_thread);
  }

  retval  =  __real_upc_memcpy_nb(a1, a2, a3);
  if (my_thread == src_thread) {
    TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, my_thread, a3, dst_thread);
  } else {
    TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_thread, a3);
  }

  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_memget_nb
 **********************************************************/

upc_handle_t   __real_upc_memget_nb(void *  a1, shared[1] const void *  a2, size_t  a3) ;
upc_handle_t   __wrap_upc_memget_nb(void *  a1, shared[1] const void *  a2, size_t  a3)  {

  upc_handle_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_memget_nb(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_handle_t upc_memget_nb(void *, shared[1] const void *, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, MYTHREAD, a3, __real_upc_threadof(a2));
  retval  =  __real_upc_memget_nb(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, __real_upc_threadof(a2), a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_memput_nb
 **********************************************************/

upc_handle_t   __real_upc_memput_nb(shared[1] void *  a1, const void *  a2, size_t  a3) ;
upc_handle_t   __wrap_upc_memput_nb(shared[1] void *  a1, const void *  a2, size_t  a3)  {

  upc_handle_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_memput_nb(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_handle_t upc_memput_nb(shared[1] void *, const void *, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  retval  =  __real_upc_memput_nb(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_sync_nb
 **********************************************************/

void   __real_upc_sync_nb(upc_handle_t  a1) ;
void   __wrap_upc_sync_nb(upc_handle_t  a1)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_sync_nb(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_sync_nb(upc_handle_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_sync_nb(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_test_nb
 **********************************************************/

int   __real_upc_test_nb(upc_handle_t  a1) ;
int   __wrap_upc_test_nb(upc_handle_t  a1)  {

  int  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_test_nb(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_test_nb(upc_handle_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_test_nb(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_memcpy_nbi
 **********************************************************/

void   __real_upc_memcpy_nbi(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3) ;
void   __wrap_upc_memcpy_nbi(shared[1] void *  a1, shared[1] const void *  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memcpy_nbi(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memcpy_nbi(shared[1] void *, shared[1] const void *, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  size_t dst_thread = __real_upc_threadof(a1);
  size_t src_thread = __real_upc_threadof(a2);
  size_t my_thread = MYTHREAD;
  if (my_thread == src_thread) {
    TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_thread, a3);
  } else {
    TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_thread, a3, src_thread);
  }

  __real_upc_memcpy_nbi(a1, a2, a3);
  if (my_thread == src_thread) {
    TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, my_thread, a3, dst_thread);
  } else {
    TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_thread, a3);
  }

  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memget_nbi
 **********************************************************/

void   __real_upc_memget_nbi(void *  a1, shared[1] const void *  a2, size_t  a3) ;
void   __wrap_upc_memget_nbi(void *  a1, shared[1] const void *  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memget_nbi(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memget_nbi(void *, shared[1] const void *, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, MYTHREAD, a3, __real_upc_threadof(a2));
  __real_upc_memget_nbi(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, __real_upc_threadof(a2), a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memput_nbi
 **********************************************************/

void   __real_upc_memput_nbi(shared[1] void *  a1, const void *  a2, size_t  a3) ;
void   __wrap_upc_memput_nbi(shared[1] void *  a1, const void *  a2, size_t  a3)  {

  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memput_nbi(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memput_nbi(shared[1] void *, const void *, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real_upc_memput_nbi(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_nodeof
 **********************************************************/

size_t   __real_upc_nodeof(shared[1] void *  a1) ;
size_t   __wrap_upc_nodeof(shared[1] void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_nodeof(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_nodeof(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_nodeof(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}

