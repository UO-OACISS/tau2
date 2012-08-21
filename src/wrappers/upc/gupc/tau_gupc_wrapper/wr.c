#include <tau_gupc.h>
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
   upc_local_alloc
 **********************************************************/

shared   void *   __real_upc_local_alloc(size_t  a1, size_t  a2) ;
shared   void *   __wrap_upc_local_alloc(size_t  a1, size_t  a2)  {

  shared   void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_local_alloc(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"shared[1] void * upc_local_alloc(size_t, size_t)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_local_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_free
 **********************************************************/

void   __real_upc_free(shared void *  a1) ;
void   __wrap_upc_free(shared void *  a1)  {

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

size_t   __real_upc_threadof(shared void *  a1) ;
size_t   __wrap_upc_threadof(shared void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_threadof(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_threadof(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_threadof(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_phaseof
 **********************************************************/

size_t   __real_upc_phaseof(shared void *  a1) ;
size_t   __wrap_upc_phaseof(shared void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_phaseof(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_phaseof(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_phaseof(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_resetphase
 **********************************************************/

shared   void *   __real_upc_resetphase(shared void *  a1) ;
shared   void *   __wrap_upc_resetphase(shared void *  a1)  {

  shared   void *  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_resetphase(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"shared[1] void * upc_resetphase(shared[1] void *)  C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_resetphase(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_addrfield
 **********************************************************/

size_t   __real_upc_addrfield(shared void *  a1) ;
size_t   __wrap_upc_addrfield(shared void *  a1)  {

  size_t  retval = 0;
  if (tau_upc_node == -1) {
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_addrfield(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_addrfield(shared[1] void *)  C", "", TAU_USER);
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

void   __real_upc_memcpy(shared void *restrict  a1, shared const void *restrict  a2, size_t  a3) ;
void   __wrap_upc_memcpy(shared void *restrict  a1, shared const void *restrict  a2, size_t  a3)  {

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

void   __real_upc_memget(void *restrict  a1, shared const void *restrict  a2, size_t  a3) ;
void   __wrap_upc_memget(void *restrict  a1, shared const void *restrict  a2, size_t  a3)  {

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

void   __real_upc_memput(shared void *restrict  a1, const void *restrict  a2, size_t  a3) ;
void   __wrap_upc_memput(shared void *restrict  a1, const void *restrict  a2, size_t  a3)  {

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

void   __real_upc_memset(shared void *  a1, int  a2, size_t  a3) ;
void   __wrap_upc_memset(shared void *  a1, int  a2, size_t  a3)  {

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

