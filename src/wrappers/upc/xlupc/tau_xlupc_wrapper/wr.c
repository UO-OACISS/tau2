#include <tau_xlupc.h>
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
   select
 **********************************************************/

int  __real_select(int a1, fd_set * a2, fd_set * a3, fd_set * a4, struct timeval * a5) ;
int  __wrap_select(int a1, fd_set * a2, fd_set * a3, fd_set * a4, struct timeval * a5)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_select(a1, a2, a3, a4, a5);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int select(int, fd_set *, fd_set *, fd_set *, struct timeval *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_select(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   pselect
 **********************************************************/

int  __real_pselect(int a1, fd_set * a2, fd_set * a3, fd_set * a4, const struct timespec * a5, const __sigset_t * a6) ;
int  __wrap_pselect(int a1, fd_set * a2, fd_set * a3, fd_set * a4, const struct timespec * a5, const __sigset_t * a6)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_pselect(a1, a2, a3, a4, a5, a6);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int pselect(int, fd_set *, fd_set *, fd_set *, const struct timespec *, const __sigset_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_pselect(a1, a2, a3, a4, a5, a6);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   gnu_dev_major
 **********************************************************/

unsigned int  __real_gnu_dev_major(unsigned long long a1) ;
unsigned int  __wrap_gnu_dev_major(unsigned long long a1)  {

  unsigned int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_gnu_dev_major(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"unsigned int gnu_dev_major(unsigned long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_gnu_dev_major(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   gnu_dev_minor
 **********************************************************/

unsigned int  __real_gnu_dev_minor(unsigned long long a1) ;
unsigned int  __wrap_gnu_dev_minor(unsigned long long a1)  {

  unsigned int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_gnu_dev_minor(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"unsigned int gnu_dev_minor(unsigned long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_gnu_dev_minor(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   gnu_dev_makedev
 **********************************************************/

unsigned long long  __real_gnu_dev_makedev(unsigned int a1, unsigned int a2) ;
unsigned long long  __wrap_gnu_dev_makedev(unsigned int a1, unsigned int a2)  {

  unsigned long long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_gnu_dev_makedev(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"unsigned long long gnu_dev_makedev(unsigned int, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_gnu_dev_makedev(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_global_exit
 **********************************************************/

void  __real_upc_global_exit(int a1) ;
void  __wrap_upc_global_exit(int a1)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_global_exit(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_global_exit(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_global_exit(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_global_alloc
 **********************************************************/

void *  __real_upc_global_alloc(size_t a1, size_t a2) ;
void *  __wrap_upc_global_alloc(size_t a1, size_t a2)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_global_alloc(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *upc_global_alloc(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_global_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_alloc
 **********************************************************/

void *  __real_upc_all_alloc(size_t a1, size_t a2) ;
void *  __wrap_upc_all_alloc(size_t a1, size_t a2)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_alloc(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *upc_all_alloc(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_alloc
 **********************************************************/

void *  __real_upc_alloc(size_t a1) ;
void *  __wrap_upc_alloc(size_t a1)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_alloc(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *upc_alloc(size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_alloc(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_local_alloc
 **********************************************************/

void *  __real_upc_local_alloc(size_t a1, size_t a2) ;
void *  __wrap_upc_local_alloc(size_t a1, size_t a2)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_local_alloc(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *upc_local_alloc(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_local_alloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_free
 **********************************************************/

void  __real_upc_free(void * a1) ;
void  __wrap_upc_free(void * a1)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_free(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_free(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_threadof
 **********************************************************/

size_t  __real_upc_threadof(void * a1) ;
size_t  __wrap_upc_threadof(void * a1)  {

  size_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_threadof(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_threadof(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_threadof(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_addrfield
 **********************************************************/

size_t  __real_upc_addrfield(void * a1) ;
size_t  __wrap_upc_addrfield(void * a1)  {

  size_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_addrfield(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_addrfield(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_addrfield(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_affinitysize
 **********************************************************/

size_t  __real_upc_affinitysize(size_t a1, size_t a2, size_t a3) ;
size_t  __wrap_upc_affinitysize(size_t a1, size_t a2, size_t a3)  {

  size_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_affinitysize(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"size_t upc_affinitysize(size_t, size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_affinitysize(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_global_lock_alloc
 **********************************************************/

upc_lock_t *  __real_upc_global_lock_alloc() ;
upc_lock_t *  __wrap_upc_global_lock_alloc()  {

  upc_lock_t * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_global_lock_alloc();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_lock_t *upc_global_lock_alloc(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_global_lock_alloc();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_lock_alloc
 **********************************************************/

upc_lock_t *  __real_upc_all_lock_alloc() ;
upc_lock_t *  __wrap_upc_all_lock_alloc()  {

  upc_lock_t * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_all_lock_alloc();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_lock_t *upc_all_lock_alloc(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_all_lock_alloc();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_lock_free
 **********************************************************/

void  __real_upc_lock_free(upc_lock_t * a1) ;
void  __wrap_upc_lock_free(upc_lock_t * a1)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_lock_free(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_lock_free(upc_lock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_lock_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_lock
 **********************************************************/

void  __real_upc_lock(upc_lock_t * a1) ;
void  __wrap_upc_lock(upc_lock_t * a1)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_lock(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_lock(upc_lock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_lock_attempt
 **********************************************************/

int  __real_upc_lock_attempt(upc_lock_t * a1) ;
int  __wrap_upc_lock_attempt(upc_lock_t * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_lock_attempt(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int upc_lock_attempt(upc_lock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_lock_attempt(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_unlock
 **********************************************************/

void  __real_upc_unlock(upc_lock_t * a1) ;
void  __wrap_upc_unlock(upc_lock_t * a1)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_unlock(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_unlock(upc_lock_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_unlock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memcpy
 **********************************************************/

void  __real_upc_memcpy(void *restrict a1, const void *restrict a2, size_t a3) ;
void  __wrap_upc_memcpy(void *restrict a1, const void *restrict a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memcpy(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memcpy(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
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

void  __real_upc_memget(void *restrict a1, const void *restrict a2, size_t a3) ;
void  __wrap_upc_memget(void *restrict a1, const void *restrict a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memget(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memget(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, MYTHREAD, a3, __real_upc_threadof(a2));
  __real_upc_memget(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, __real_upc_threadof(a2), a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memput
 **********************************************************/

void  __real_upc_memput(void *restrict a1, const void *restrict a2, size_t a3) ;
void  __wrap_upc_memput(void *restrict a1, const void *restrict a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memput(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memput(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real_upc_memput(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_memset
 **********************************************************/

void  __real_upc_memset(void * a1, int a2, size_t a3) ;
void  __wrap_upc_memset(void * a1, int a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_memset(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_memset(void *, int, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real_upc_memset(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   xlupc_thread_affinity
 **********************************************************/

int  __real_xlupc_thread_affinity() ;
int  __wrap_xlupc_thread_affinity()  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_xlupc_thread_affinity();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int xlupc_thread_affinity(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_xlupc_thread_affinity();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   xlupc_stacksize
 **********************************************************/

uint64_t  __real_xlupc_stacksize() ;
uint64_t  __wrap_xlupc_stacksize()  {

  uint64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_xlupc_stacksize();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"uint64_t xlupc_stacksize(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_xlupc_stacksize();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_memget_async
 **********************************************************/

void *  __real___xlupc_memget_async(void *restrict a1, const void *restrict a2, size_t a3) ;
void *  __wrap___xlupc_memget_async(void *restrict a1, const void *restrict a2, size_t a3)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_memget_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *__xlupc_memget_async(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, MYTHREAD, a3, __real_upc_threadof(a2));
  retval  =  __real___xlupc_memget_async(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, __real_upc_threadof(a2), a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_memput_async
 **********************************************************/

void *  __real___xlupc_memput_async(void *restrict a1, const void *restrict a2, size_t a3) ;
void *  __wrap___xlupc_memput_async(void *restrict a1, const void *restrict a2, size_t a3)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_memput_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *__xlupc_memput_async(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  retval  =  __real___xlupc_memput_async(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_memset_async
 **********************************************************/

void *  __real___xlupc_memset_async(void * a1, int a2, size_t a3) ;
void *  __wrap___xlupc_memset_async(void * a1, int a2, size_t a3)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_memset_async(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *__xlupc_memset_async(void *, int, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  retval  =  __real___xlupc_memset_async(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_waitsync
 **********************************************************/

void  __real___xlupc_waitsync(void * a1) ;
void  __wrap___xlupc_waitsync(void * a1)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_waitsync(a1);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_waitsync(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_waitsync(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_trysync
 **********************************************************/

int  __real___xlupc_trysync(void * a1) ;
int  __wrap___xlupc_trysync(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_trysync(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_trysync(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_trysync(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_waitsync_multi
 **********************************************************/

void  __real___xlupc_waitsync_multi(void ** a1, unsigned int a2) ;
void  __wrap___xlupc_waitsync_multi(void ** a1, unsigned int a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_waitsync_multi(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_waitsync_multi(void **, unsigned int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_waitsync_multi(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_memget_asynci
 **********************************************************/

void  __real___xlupc_memget_asynci(void *restrict a1, const void *restrict a2, size_t a3) ;
void  __wrap___xlupc_memget_asynci(void *restrict a1, const void *restrict a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_memget_asynci(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_memget_asynci(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, MYTHREAD, a3, __real_upc_threadof(a2));
  __real___xlupc_memget_asynci(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_UPC_TAGID, __real_upc_threadof(a2), a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_memput_asynci
 **********************************************************/

void  __real___xlupc_memput_asynci(void *restrict a1, const void *restrict a2, size_t a3) ;
void  __wrap___xlupc_memput_asynci(void *restrict a1, const void *restrict a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_memput_asynci(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_memput_asynci(void *restrict, const void *restrict, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real___xlupc_memput_asynci(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_memset_asynci
 **********************************************************/

void  __real___xlupc_memset_asynci(void * a1, int a2, size_t a3) ;
void  __wrap___xlupc_memset_asynci(void * a1, int a2, size_t a3)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_memset_asynci(a1, a2, a3);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_memset_asynci(void *, int, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, __real_upc_threadof(a1), a3);
  __real___xlupc_memset_asynci(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, MYTHREAD, a3, __real_upc_threadof(a1));
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_waitsynci
 **********************************************************/

void  __real___xlupc_waitsynci() ;
void  __wrap___xlupc_waitsynci()  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_waitsynci();
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_waitsynci(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_waitsynci();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_memget_vlist
 **********************************************************/

void  __real___xlupc_memget_vlist(size_t a1, __xlupc_pmemvec_t * a2, size_t a3, __xlupc_smemvec_t * a4) ;
void  __wrap___xlupc_memget_vlist(size_t a1, __xlupc_pmemvec_t * a2, size_t a3, __xlupc_smemvec_t * a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_memget_vlist(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_memget_vlist(size_t, __xlupc_pmemvec_t *, size_t, __xlupc_smemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_memget_vlist(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_memput_vlist
 **********************************************************/

void  __real___xlupc_memput_vlist(size_t a1, __xlupc_smemvec_t * a2, size_t a3, __xlupc_pmemvec_t * a4) ;
void  __wrap___xlupc_memput_vlist(size_t a1, __xlupc_smemvec_t * a2, size_t a3, __xlupc_pmemvec_t * a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_memput_vlist(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_memput_vlist(size_t, __xlupc_smemvec_t *, size_t, __xlupc_pmemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_memput_vlist(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_memget_vlist_async
 **********************************************************/

void *  __real___xlupc_memget_vlist_async(size_t a1, __xlupc_pmemvec_t * a2, size_t a3, __xlupc_smemvec_t * a4) ;
void *  __wrap___xlupc_memget_vlist_async(size_t a1, __xlupc_pmemvec_t * a2, size_t a3, __xlupc_smemvec_t * a4)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_memget_vlist_async(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *__xlupc_memget_vlist_async(size_t, __xlupc_pmemvec_t *, size_t, __xlupc_smemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_memget_vlist_async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_memput_vlist_async
 **********************************************************/

void *  __real___xlupc_memput_vlist_async(size_t a1, __xlupc_smemvec_t * a2, size_t a3, __xlupc_pmemvec_t * a4) ;
void *  __wrap___xlupc_memput_vlist_async(size_t a1, __xlupc_smemvec_t * a2, size_t a3, __xlupc_pmemvec_t * a4)  {

  void * retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_memput_vlist_async(a1, a2, a3, a4);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void *__xlupc_memput_vlist_async(size_t, __xlupc_smemvec_t *, size_t, __xlupc_pmemvec_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_memput_vlist_async(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_read_R
 **********************************************************/

int  __real___xlupc_atomicI_read_R(void * a1) ;
int  __wrap___xlupc_atomicI_read_R(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_read_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_read_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_read_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_set_R
 **********************************************************/

void  __real___xlupc_atomicI_set_R(void * a1, int a2) ;
void  __wrap___xlupc_atomicI_set_R(void * a1, int a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI_set_R(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI_set_R(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI_set_R(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI_swap_R
 **********************************************************/

int  __real___xlupc_atomicI_swap_R(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_swap_R(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_swap_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_swap_R(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_swap_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_cswap_R
 **********************************************************/

int  __real___xlupc_atomicI_cswap_R(void * a1, int a2, int a3) ;
int  __wrap___xlupc_atomicI_cswap_R(void * a1, int a2, int a3)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_cswap_R(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_cswap_R(void *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_cswap_R(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchadd_R
 **********************************************************/

int  __real___xlupc_atomicI_fetchadd_R(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchadd_R(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchadd_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchadd_R(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchadd_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchand_R
 **********************************************************/

int  __real___xlupc_atomicI_fetchand_R(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchand_R(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchand_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchand_R(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchand_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchor_R
 **********************************************************/

int  __real___xlupc_atomicI_fetchor_R(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchor_R(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchor_R(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchxor_R
 **********************************************************/

int  __real___xlupc_atomicI_fetchxor_R(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchxor_R(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchxor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchxor_R(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchxor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchnot_R
 **********************************************************/

int  __real___xlupc_atomicI_fetchnot_R(void * a1) ;
int  __wrap___xlupc_atomicI_fetchnot_R(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchnot_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchnot_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchnot_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_read_S
 **********************************************************/

int  __real___xlupc_atomicI_read_S(void * a1) ;
int  __wrap___xlupc_atomicI_read_S(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_read_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_read_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_read_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_set_S
 **********************************************************/

void  __real___xlupc_atomicI_set_S(void * a1, int a2) ;
void  __wrap___xlupc_atomicI_set_S(void * a1, int a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI_set_S(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI_set_S(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI_set_S(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI_swap_S
 **********************************************************/

int  __real___xlupc_atomicI_swap_S(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_swap_S(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_swap_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_swap_S(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_swap_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_cswap_S
 **********************************************************/

int  __real___xlupc_atomicI_cswap_S(void * a1, int a2, int a3) ;
int  __wrap___xlupc_atomicI_cswap_S(void * a1, int a2, int a3)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_cswap_S(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_cswap_S(void *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_cswap_S(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchadd_S
 **********************************************************/

int  __real___xlupc_atomicI_fetchadd_S(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchadd_S(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchadd_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchadd_S(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchadd_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchand_S
 **********************************************************/

int  __real___xlupc_atomicI_fetchand_S(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchand_S(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchand_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchand_S(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchand_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchor_S
 **********************************************************/

int  __real___xlupc_atomicI_fetchor_S(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchor_S(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchor_S(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchxor_S
 **********************************************************/

int  __real___xlupc_atomicI_fetchxor_S(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchxor_S(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchxor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchxor_S(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchxor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchnot_S
 **********************************************************/

int  __real___xlupc_atomicI_fetchnot_S(void * a1) ;
int  __wrap___xlupc_atomicI_fetchnot_S(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchnot_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchnot_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchnot_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_read_private
 **********************************************************/

int  __real___xlupc_atomicI_read_private(void * a1) ;
int  __wrap___xlupc_atomicI_read_private(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_read_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_read_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_read_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_set_private
 **********************************************************/

void  __real___xlupc_atomicI_set_private(void * a1, int a2) ;
void  __wrap___xlupc_atomicI_set_private(void * a1, int a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI_set_private(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI_set_private(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI_set_private(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI_swap_private
 **********************************************************/

int  __real___xlupc_atomicI_swap_private(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_swap_private(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_swap_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_swap_private(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_swap_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_cswap_private
 **********************************************************/

int  __real___xlupc_atomicI_cswap_private(void * a1, int a2, int a3) ;
int  __wrap___xlupc_atomicI_cswap_private(void * a1, int a2, int a3)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_cswap_private(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_cswap_private(void *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_cswap_private(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchadd_private
 **********************************************************/

int  __real___xlupc_atomicI_fetchadd_private(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchadd_private(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchadd_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchadd_private(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchadd_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchand_private
 **********************************************************/

int  __real___xlupc_atomicI_fetchand_private(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchand_private(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchand_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchand_private(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchand_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchor_private
 **********************************************************/

int  __real___xlupc_atomicI_fetchor_private(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchor_private(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchor_private(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchxor_private
 **********************************************************/

int  __real___xlupc_atomicI_fetchxor_private(void * a1, int a2) ;
int  __wrap___xlupc_atomicI_fetchxor_private(void * a1, int a2)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchxor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchxor_private(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchxor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI_fetchnot_private
 **********************************************************/

int  __real___xlupc_atomicI_fetchnot_private(void * a1) ;
int  __wrap___xlupc_atomicI_fetchnot_private(void * a1)  {

  int retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI_fetchnot_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int __xlupc_atomicI_fetchnot_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI_fetchnot_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_read_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_read_R(void * a1) ;
int32_t  __wrap___xlupc_atomicI32_read_R(void * a1)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_read_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_read_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_read_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_set_R
 **********************************************************/

void  __real___xlupc_atomicI32_set_R(void * a1, int32_t a2) ;
void  __wrap___xlupc_atomicI32_set_R(void * a1, int32_t a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI32_set_R(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI32_set_R(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI32_set_R(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI32_swap_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_swap_R(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_swap_R(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_swap_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_swap_R(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_swap_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_cswap_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_cswap_R(void * a1, int32_t a2, int32_t a3) ;
int32_t  __wrap___xlupc_atomicI32_cswap_R(void * a1, int32_t a2, int32_t a3)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_cswap_R(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_cswap_R(void *, int32_t, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_cswap_R(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchadd_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchadd_R(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchadd_R(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchadd_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchadd_R(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchadd_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchand_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchand_R(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchand_R(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchand_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchand_R(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchand_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchor_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchor_R(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchor_R(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchor_R(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchxor_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchxor_R(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchxor_R(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchxor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchxor_R(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchxor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchnot_R
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchnot_R(void * a1) ;
int32_t  __wrap___xlupc_atomicI32_fetchnot_R(void * a1)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchnot_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchnot_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchnot_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_read_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_read_S(void * a1) ;
int32_t  __wrap___xlupc_atomicI32_read_S(void * a1)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_read_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_read_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_read_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_set_S
 **********************************************************/

void  __real___xlupc_atomicI32_set_S(void * a1, int32_t a2) ;
void  __wrap___xlupc_atomicI32_set_S(void * a1, int32_t a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI32_set_S(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI32_set_S(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI32_set_S(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI32_swap_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_swap_S(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_swap_S(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_swap_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_swap_S(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_swap_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_cswap_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_cswap_S(void * a1, int32_t a2, int32_t a3) ;
int32_t  __wrap___xlupc_atomicI32_cswap_S(void * a1, int32_t a2, int32_t a3)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_cswap_S(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_cswap_S(void *, int32_t, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_cswap_S(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchadd_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchadd_S(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchadd_S(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchadd_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchadd_S(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchadd_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchand_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchand_S(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchand_S(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchand_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchand_S(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchand_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchor_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchor_S(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchor_S(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchor_S(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchxor_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchxor_S(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchxor_S(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchxor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchxor_S(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchxor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchnot_S
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchnot_S(void * a1) ;
int32_t  __wrap___xlupc_atomicI32_fetchnot_S(void * a1)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchnot_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchnot_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchnot_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_read_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_read_private(void * a1) ;
int32_t  __wrap___xlupc_atomicI32_read_private(void * a1)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_read_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_read_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_read_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_set_private
 **********************************************************/

void  __real___xlupc_atomicI32_set_private(void * a1, int32_t a2) ;
void  __wrap___xlupc_atomicI32_set_private(void * a1, int32_t a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI32_set_private(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI32_set_private(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI32_set_private(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI32_swap_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_swap_private(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_swap_private(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_swap_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_swap_private(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_swap_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_cswap_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_cswap_private(void * a1, int32_t a2, int32_t a3) ;
int32_t  __wrap___xlupc_atomicI32_cswap_private(void * a1, int32_t a2, int32_t a3)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_cswap_private(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_cswap_private(void *, int32_t, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_cswap_private(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchadd_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchadd_private(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchadd_private(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchadd_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchadd_private(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchadd_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchand_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchand_private(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchand_private(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchand_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchand_private(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchand_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchor_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchor_private(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchor_private(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchor_private(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchxor_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchxor_private(void * a1, int32_t a2) ;
int32_t  __wrap___xlupc_atomicI32_fetchxor_private(void * a1, int32_t a2)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchxor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchxor_private(void *, int32_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchxor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI32_fetchnot_private
 **********************************************************/

int32_t  __real___xlupc_atomicI32_fetchnot_private(void * a1) ;
int32_t  __wrap___xlupc_atomicI32_fetchnot_private(void * a1)  {

  int32_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI32_fetchnot_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int32_t __xlupc_atomicI32_fetchnot_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI32_fetchnot_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_read_R
 **********************************************************/

long  __real___xlupc_atomicL_read_R(void * a1) ;
long  __wrap___xlupc_atomicL_read_R(void * a1)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_read_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_read_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_read_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_set_R
 **********************************************************/

void  __real___xlupc_atomicL_set_R(void * a1, long a2) ;
void  __wrap___xlupc_atomicL_set_R(void * a1, long a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicL_set_R(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicL_set_R(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicL_set_R(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicL_swap_R
 **********************************************************/

long  __real___xlupc_atomicL_swap_R(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_swap_R(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_swap_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_swap_R(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_swap_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_cswap_R
 **********************************************************/

long  __real___xlupc_atomicL_cswap_R(void * a1, long a2, long a3) ;
long  __wrap___xlupc_atomicL_cswap_R(void * a1, long a2, long a3)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_cswap_R(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_cswap_R(void *, long, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_cswap_R(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchadd_R
 **********************************************************/

long  __real___xlupc_atomicL_fetchadd_R(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchadd_R(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchadd_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchadd_R(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchadd_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchand_R
 **********************************************************/

long  __real___xlupc_atomicL_fetchand_R(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchand_R(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchand_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchand_R(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchand_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchor_R
 **********************************************************/

long  __real___xlupc_atomicL_fetchor_R(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchor_R(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchor_R(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchxor_R
 **********************************************************/

long  __real___xlupc_atomicL_fetchxor_R(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchxor_R(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchxor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchxor_R(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchxor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchnot_R
 **********************************************************/

long  __real___xlupc_atomicL_fetchnot_R(void * a1) ;
long  __wrap___xlupc_atomicL_fetchnot_R(void * a1)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchnot_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchnot_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchnot_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_read_S
 **********************************************************/

long  __real___xlupc_atomicL_read_S(void * a1) ;
long  __wrap___xlupc_atomicL_read_S(void * a1)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_read_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_read_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_read_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_set_S
 **********************************************************/

void  __real___xlupc_atomicL_set_S(void * a1, long a2) ;
void  __wrap___xlupc_atomicL_set_S(void * a1, long a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicL_set_S(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicL_set_S(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicL_set_S(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicL_swap_S
 **********************************************************/

long  __real___xlupc_atomicL_swap_S(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_swap_S(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_swap_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_swap_S(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_swap_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_cswap_S
 **********************************************************/

long  __real___xlupc_atomicL_cswap_S(void * a1, long a2, long a3) ;
long  __wrap___xlupc_atomicL_cswap_S(void * a1, long a2, long a3)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_cswap_S(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_cswap_S(void *, long, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_cswap_S(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchadd_S
 **********************************************************/

long  __real___xlupc_atomicL_fetchadd_S(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchadd_S(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchadd_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchadd_S(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchadd_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchand_S
 **********************************************************/

long  __real___xlupc_atomicL_fetchand_S(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchand_S(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchand_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchand_S(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchand_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchor_S
 **********************************************************/

long  __real___xlupc_atomicL_fetchor_S(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchor_S(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchor_S(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchxor_S
 **********************************************************/

long  __real___xlupc_atomicL_fetchxor_S(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchxor_S(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchxor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchxor_S(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchxor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchnot_S
 **********************************************************/

long  __real___xlupc_atomicL_fetchnot_S(void * a1) ;
long  __wrap___xlupc_atomicL_fetchnot_S(void * a1)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchnot_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchnot_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchnot_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_read_private
 **********************************************************/

long  __real___xlupc_atomicL_read_private(void * a1) ;
long  __wrap___xlupc_atomicL_read_private(void * a1)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_read_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_read_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_read_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_set_private
 **********************************************************/

void  __real___xlupc_atomicL_set_private(void * a1, long a2) ;
void  __wrap___xlupc_atomicL_set_private(void * a1, long a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicL_set_private(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicL_set_private(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicL_set_private(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicL_swap_private
 **********************************************************/

long  __real___xlupc_atomicL_swap_private(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_swap_private(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_swap_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_swap_private(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_swap_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_cswap_private
 **********************************************************/

long  __real___xlupc_atomicL_cswap_private(void * a1, long a2, long a3) ;
long  __wrap___xlupc_atomicL_cswap_private(void * a1, long a2, long a3)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_cswap_private(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_cswap_private(void *, long, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_cswap_private(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchadd_private
 **********************************************************/

long  __real___xlupc_atomicL_fetchadd_private(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchadd_private(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchadd_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchadd_private(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchadd_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchand_private
 **********************************************************/

long  __real___xlupc_atomicL_fetchand_private(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchand_private(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchand_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchand_private(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchand_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchor_private
 **********************************************************/

long  __real___xlupc_atomicL_fetchor_private(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchor_private(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchor_private(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchxor_private
 **********************************************************/

long  __real___xlupc_atomicL_fetchxor_private(void * a1, long a2) ;
long  __wrap___xlupc_atomicL_fetchxor_private(void * a1, long a2)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchxor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchxor_private(void *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchxor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicL_fetchnot_private
 **********************************************************/

long  __real___xlupc_atomicL_fetchnot_private(void * a1) ;
long  __wrap___xlupc_atomicL_fetchnot_private(void * a1)  {

  long retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicL_fetchnot_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"long __xlupc_atomicL_fetchnot_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicL_fetchnot_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_read_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_read_R(void * a1) ;
int64_t  __wrap___xlupc_atomicI64_read_R(void * a1)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_read_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_read_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_read_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_set_R
 **********************************************************/

void  __real___xlupc_atomicI64_set_R(void * a1, int64_t a2) ;
void  __wrap___xlupc_atomicI64_set_R(void * a1, int64_t a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI64_set_R(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI64_set_R(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI64_set_R(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI64_swap_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_swap_R(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_swap_R(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_swap_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_swap_R(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_swap_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_cswap_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_cswap_R(void * a1, int64_t a2, int64_t a3) ;
int64_t  __wrap___xlupc_atomicI64_cswap_R(void * a1, int64_t a2, int64_t a3)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_cswap_R(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_cswap_R(void *, int64_t, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_cswap_R(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchadd_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchadd_R(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchadd_R(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchadd_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchadd_R(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchadd_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchand_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchand_R(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchand_R(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchand_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchand_R(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchand_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchor_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchor_R(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchor_R(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchor_R(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchxor_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchxor_R(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchxor_R(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchxor_R(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchxor_R(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchxor_R(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchnot_R
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchnot_R(void * a1) ;
int64_t  __wrap___xlupc_atomicI64_fetchnot_R(void * a1)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchnot_R(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchnot_R(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchnot_R(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_read_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_read_S(void * a1) ;
int64_t  __wrap___xlupc_atomicI64_read_S(void * a1)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_read_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_read_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_read_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_set_S
 **********************************************************/

void  __real___xlupc_atomicI64_set_S(void * a1, int64_t a2) ;
void  __wrap___xlupc_atomicI64_set_S(void * a1, int64_t a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI64_set_S(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI64_set_S(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI64_set_S(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI64_swap_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_swap_S(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_swap_S(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_swap_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_swap_S(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_swap_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_cswap_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_cswap_S(void * a1, int64_t a2, int64_t a3) ;
int64_t  __wrap___xlupc_atomicI64_cswap_S(void * a1, int64_t a2, int64_t a3)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_cswap_S(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_cswap_S(void *, int64_t, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_cswap_S(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchadd_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchadd_S(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchadd_S(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchadd_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchadd_S(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchadd_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchand_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchand_S(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchand_S(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchand_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchand_S(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchand_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchor_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchor_S(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchor_S(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchor_S(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchxor_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchxor_S(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchxor_S(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchxor_S(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchxor_S(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchxor_S(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchnot_S
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchnot_S(void * a1) ;
int64_t  __wrap___xlupc_atomicI64_fetchnot_S(void * a1)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchnot_S(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchnot_S(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchnot_S(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_read_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_read_private(void * a1) ;
int64_t  __wrap___xlupc_atomicI64_read_private(void * a1)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_read_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_read_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_read_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_set_private
 **********************************************************/

void  __real___xlupc_atomicI64_set_private(void * a1, int64_t a2) ;
void  __wrap___xlupc_atomicI64_set_private(void * a1, int64_t a2)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real___xlupc_atomicI64_set_private(a1, a2);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void __xlupc_atomicI64_set_private(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real___xlupc_atomicI64_set_private(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   __xlupc_atomicI64_swap_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_swap_private(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_swap_private(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_swap_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_swap_private(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_swap_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_cswap_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_cswap_private(void * a1, int64_t a2, int64_t a3) ;
int64_t  __wrap___xlupc_atomicI64_cswap_private(void * a1, int64_t a2, int64_t a3)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_cswap_private(a1, a2, a3);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_cswap_private(void *, int64_t, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_cswap_private(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchadd_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchadd_private(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchadd_private(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchadd_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchadd_private(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchadd_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchand_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchand_private(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchand_private(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchand_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchand_private(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchand_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchor_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchor_private(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchor_private(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchor_private(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchxor_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchxor_private(void * a1, int64_t a2) ;
int64_t  __wrap___xlupc_atomicI64_fetchxor_private(void * a1, int64_t a2)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchxor_private(a1, a2);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchxor_private(void *, int64_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchxor_private(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   __xlupc_atomicI64_fetchnot_private
 **********************************************************/

int64_t  __real___xlupc_atomicI64_fetchnot_private(void * a1) ;
int64_t  __wrap___xlupc_atomicI64_fetchnot_private(void * a1)  {

  int64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real___xlupc_atomicI64_fetchnot_private(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"int64_t __xlupc_atomicI64_fetchnot_private(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real___xlupc_atomicI64_fetchnot_private(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_ticks_now
 **********************************************************/

upc_tick_t  __real_upc_ticks_now() ;
upc_tick_t  __wrap_upc_ticks_now()  {

  upc_tick_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_ticks_now();
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"upc_tick_t upc_ticks_now(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_ticks_now();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_ticks_to_ns
 **********************************************************/

uint64_t  __real_upc_ticks_to_ns(upc_tick_t a1) ;
uint64_t  __wrap_upc_ticks_to_ns(upc_tick_t a1)  {

  uint64_t retval = 0;
  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      return __real_upc_ticks_to_ns(a1);
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"uint64_t upc_ticks_to_ns(upc_tick_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_upc_ticks_to_ns(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   upc_all_broadcast
 **********************************************************/

void  __real_upc_all_broadcast(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4) ;
void  __wrap_upc_all_broadcast(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_broadcast(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_broadcast(void *restrict, const void *restrict, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_broadcast(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_scatter
 **********************************************************/

void  __real_upc_all_scatter(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4) ;
void  __wrap_upc_all_scatter(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_scatter(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_scatter(void *restrict, const void *restrict, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_scatter(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_gather
 **********************************************************/

void  __real_upc_all_gather(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4) ;
void  __wrap_upc_all_gather(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_gather(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_gather(void *restrict, const void *restrict, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_gather(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_gather_all
 **********************************************************/

void  __real_upc_all_gather_all(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4) ;
void  __wrap_upc_all_gather_all(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_gather_all(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_gather_all(void *restrict, const void *restrict, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_gather_all(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_exchange
 **********************************************************/

void  __real_upc_all_exchange(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4) ;
void  __wrap_upc_all_exchange(void *restrict a1, const void *restrict a2, size_t a3, upc_flag_t a4)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_exchange(a1, a2, a3, a4);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_exchange(void *restrict, const void *restrict, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_exchange(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_permute
 **********************************************************/

void  __real_upc_all_permute(void *restrict a1, const void *restrict a2, const int *restrict a3, size_t a4, upc_flag_t a5) ;
void  __wrap_upc_all_permute(void *restrict a1, const void *restrict a2, const int *restrict a3, size_t a4, upc_flag_t a5)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_permute(a1, a2, a3, a4, a5);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_permute(void *restrict, const void *restrict, const int *restrict, size_t, upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_permute(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceC
 **********************************************************/

void  __real_upc_all_reduceC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7) ;
void  __wrap_upc_all_reduceC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceC(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed char (*)(signed char, signed char), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUC
 **********************************************************/

void  __real_upc_all_reduceUC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7) ;
void  __wrap_upc_all_reduceUC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUC(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned char (*)(unsigned char, unsigned char), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceS
 **********************************************************/

void  __real_upc_all_reduceS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7) ;
void  __wrap_upc_all_reduceS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceS(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed short (*)(signed short, signed short), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUS
 **********************************************************/

void  __real_upc_all_reduceUS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7) ;
void  __wrap_upc_all_reduceUS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUS(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned short (*)(unsigned short, unsigned short), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceI
 **********************************************************/

void  __real_upc_all_reduceI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7) ;
void  __wrap_upc_all_reduceI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceI(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed int (*)(signed int, signed int), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUI
 **********************************************************/

void  __real_upc_all_reduceUI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7) ;
void  __wrap_upc_all_reduceUI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUI(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned int (*)(unsigned int, unsigned int), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceL
 **********************************************************/

void  __real_upc_all_reduceL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7) ;
void  __wrap_upc_all_reduceL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceL(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed long (*)(signed long, signed long), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceUL
 **********************************************************/

void  __real_upc_all_reduceUL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7) ;
void  __wrap_upc_all_reduceUL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceUL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceUL(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned long (*)(unsigned long, unsigned long), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceUL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceF
 **********************************************************/

void  __real_upc_all_reduceF(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7) ;
void  __wrap_upc_all_reduceF(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceF(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceF(void *restrict, const void *restrict, upc_op_t, size_t, size_t, float (*)(float, float), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceF(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceD
 **********************************************************/

void  __real_upc_all_reduceD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7) ;
void  __wrap_upc_all_reduceD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceD(void *restrict, const void *restrict, upc_op_t, size_t, size_t, double (*)(double, double), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_reduceLD
 **********************************************************/

void  __real_upc_all_reduceLD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7) ;
void  __wrap_upc_all_reduceLD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_reduceLD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_reduceLD(void *restrict, const void *restrict, upc_op_t, size_t, size_t, long double (*)(long double, long double), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_reduceLD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceC
 **********************************************************/

void  __real_upc_all_prefix_reduceC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed char (*a6)(signed char, signed char), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceC(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed char (*)(signed char, signed char), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUC
 **********************************************************/

void  __real_upc_all_prefix_reduceUC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceUC(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned char (*a6)(unsigned char, unsigned char), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUC(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUC(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned char (*)(unsigned char, unsigned char), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUC(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceS
 **********************************************************/

void  __real_upc_all_prefix_reduceS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed short (*a6)(signed short, signed short), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceS(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed short (*)(signed short, signed short), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUS
 **********************************************************/

void  __real_upc_all_prefix_reduceUS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceUS(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned short (*a6)(unsigned short, unsigned short), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUS(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUS(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned short (*)(unsigned short, unsigned short), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUS(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceI
 **********************************************************/

void  __real_upc_all_prefix_reduceI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed int (*a6)(signed int, signed int), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceI(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed int (*)(signed int, signed int), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUI
 **********************************************************/

void  __real_upc_all_prefix_reduceUI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceUI(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned int (*a6)(unsigned int, unsigned int), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUI(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUI(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned int (*)(unsigned int, unsigned int), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUI(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceL
 **********************************************************/

void  __real_upc_all_prefix_reduceL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, signed long (*a6)(signed long, signed long), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceL(void *restrict, const void *restrict, upc_op_t, size_t, size_t, signed long (*)(signed long, signed long), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceUL
 **********************************************************/

void  __real_upc_all_prefix_reduceUL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceUL(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, unsigned long (*a6)(unsigned long, unsigned long), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceUL(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceUL(void *restrict, const void *restrict, upc_op_t, size_t, size_t, unsigned long (*)(unsigned long, unsigned long), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceUL(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceF
 **********************************************************/

void  __real_upc_all_prefix_reduceF(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceF(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, float (*a6)(float, float), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceF(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceF(void *restrict, const void *restrict, upc_op_t, size_t, size_t, float (*)(float, float), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceF(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceD
 **********************************************************/

void  __real_upc_all_prefix_reduceD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, double (*a6)(double, double), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceD(void *restrict, const void *restrict, upc_op_t, size_t, size_t, double (*)(double, double), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   upc_all_prefix_reduceLD
 **********************************************************/

void  __real_upc_all_prefix_reduceLD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7) ;
void  __wrap_upc_all_prefix_reduceLD(void *restrict a1, const void *restrict a2, upc_op_t a3, size_t a4, size_t a5, long double (*a6)(long double, long double), upc_flag_t a7)  {

  if (tau_upc_node == -1) {
    TAU_PROFILE_SET_NODE(MYTHREAD); 
    tau_upc_node = TAU_PROFILE_GET_NODE();
    if (tau_upc_node == -1) {
      __real_upc_all_prefix_reduceLD(a1, a2, a3, a4, a5, a6, a7);
      return;
    } else {
      tau_totalnodes(1,THREADS);
    }
  }

  TAU_PROFILE_TIMER(t,"void upc_all_prefix_reduceLD(void *restrict, const void *restrict, upc_op_t, size_t, size_t, long double (*)(long double, long double), upc_flag_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_upc_all_prefix_reduceLD(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}

