#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <tau_shmem.h>
#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>

int TAUDECL tau_totalnodes(int set_or_get, int value);
static int tau_shmem_tagid_f=0;
#define TAU_SHMEM_TAGID (tau_shmem_tagid_f = (tau_shmem_tagid_f & 255))
#define TAU_SHMEM_TAGID_NEXT ((++tau_shmem_tagid_f) & 255)


/**********************************************************
   shmem_init
 **********************************************************/

extern void  __real_shmem_init() ;
extern void  __wrap_shmem_init()  {

  TAU_PROFILE_TIMER(t,"void shmem_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_finalize
 **********************************************************/

extern void  __real_shmem_finalize() ;
extern void  __wrap_shmem_finalize()  {

  TAU_PROFILE_TIMER(t,"void shmem_finalize(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_finalize();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_global_exit
 **********************************************************/

extern void  __real_shmem_global_exit(int a1) ;
extern void  __wrap_shmem_global_exit(int a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_global_exit(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_global_exit(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_my_pe
 **********************************************************/

extern int  __real_shmem_my_pe() ;
extern int  __wrap_shmem_my_pe()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_my_pe(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_my_pe();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_n_pes
 **********************************************************/

extern int  __real_shmem_n_pes() ;
extern int  __wrap_shmem_n_pes()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_n_pes(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_n_pes();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_put
 **********************************************************/

extern void  __real_shmem_short_put(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_short_put(short * a1, const short * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_put(short *, const short *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*a3);
  __real_shmem_short_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_put
 **********************************************************/

extern void  __real_shmem_int_put(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_int_put(int * a1, const int * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_put(int *, const int *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*a3);
  __real_shmem_int_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_put
 **********************************************************/

extern void  __real_shmem_long_put(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_long_put(long * a1, const long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_put(long *, const long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*a3);
  __real_shmem_long_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_put
 **********************************************************/

extern void  __real_shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put(long long *, const long long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*a3);
  __real_shmem_longlong_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_put
 **********************************************************/

extern void  __real_shmem_longdouble_put(long double * a1, const long double * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_longdouble_put(long double * a1, const long double * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_put(long double *, const long double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long double)*a3);
  __real_shmem_longdouble_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put
 **********************************************************/

extern void  __real_shmem_double_put(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_double_put(double * a1, const double * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_put(double *, const double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*a3);
  __real_shmem_double_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_put
 **********************************************************/

extern void  __real_shmem_float_put(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float_put(float * a1, const float * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_put(float *, const float *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmem_float_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_putmem
 **********************************************************/

extern void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_putmem(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_putmem(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, a3);
  __real_shmem_putmem(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put32
 **********************************************************/

extern void  __real_shmem_put32(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put32(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put32(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
  __real_shmem_put32(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put64
 **********************************************************/

extern void  __real_shmem_put64(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put64(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put64(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
  __real_shmem_put64(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put128
 **********************************************************/

extern void  __real_shmem_put128(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put128(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put128(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 16*a3);
  __real_shmem_put128(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_get
 **********************************************************/

extern void  __real_shmem_short_get(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_short_get(short * a1, const short * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_get(short *, const short *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a3, a4);
  __real_shmem_short_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_get
 **********************************************************/

extern void  __real_shmem_int_get(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_int_get(int * a1, const int * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_get(int *, const int *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a3, a4);
  __real_shmem_int_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_get
 **********************************************************/

extern void  __real_shmem_long_get(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_long_get(long * a1, const long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_get(long *, const long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a3, a4);
  __real_shmem_long_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_get
 **********************************************************/

extern void  __real_shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_get(long long *, const long long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a3, a4);
  __real_shmem_longlong_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_get
 **********************************************************/

extern void  __real_shmem_longdouble_get(long double * a1, const long double * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_longdouble_get(long double * a1, const long double * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_get(long double *, const long double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*a3, a4);
  __real_shmem_longdouble_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_get
 **********************************************************/

extern void  __real_shmem_double_get(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_double_get(double * a1, const double * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_get(double *, const double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a3, a4);
  __real_shmem_double_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_get
 **********************************************************/

extern void  __real_shmem_float_get(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float_get(float * a1, const float * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_get(float *, const float *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmem_float_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_getmem
 **********************************************************/

extern void  __real_shmem_getmem(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_getmem(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_getmem(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), a3, a4);
  __real_shmem_getmem(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get32
 **********************************************************/

extern void  __real_shmem_get32(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get32(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get32(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a3, a4);
  __real_shmem_get32(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 4*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get64
 **********************************************************/

extern void  __real_shmem_get64(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get64(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get64(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
  __real_shmem_get64(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get128
 **********************************************************/

extern void  __real_shmem_get128(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get128(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get128(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a3, a4);
  __real_shmem_get128(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 16*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_char_p
 **********************************************************/

extern void  __real_shmem_char_p(char * a1, char a2, int a3) ;
extern void  __wrap_shmem_char_p(char * a1, char a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_p(char *, char, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(char)*1);
  __real_shmem_char_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_p
 **********************************************************/

extern void  __real_shmem_short_p(short * a1, short a2, int a3) ;
extern void  __wrap_shmem_short_p(short * a1, short a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_p(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
  __real_shmem_short_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_p
 **********************************************************/

extern void  __real_shmem_int_p(int * a1, int a2, int a3) ;
extern void  __wrap_shmem_int_p(int * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_p(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  __real_shmem_int_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_p
 **********************************************************/

extern void  __real_shmem_long_p(long * a1, long a2, int a3) ;
extern void  __wrap_shmem_long_p(long * a1, long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_p(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  __real_shmem_long_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_p
 **********************************************************/

extern void  __real_shmem_longlong_p(long long * a1, long long a2, int a3) ;
extern void  __wrap_shmem_longlong_p(long long * a1, long long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_p(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  __real_shmem_longlong_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_p
 **********************************************************/

extern void  __real_shmem_float_p(float * a1, float a2, int a3) ;
extern void  __wrap_shmem_float_p(float * a1, float a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_p(float *, float, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(float)*1);
  __real_shmem_float_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_p
 **********************************************************/

extern void  __real_shmem_double_p(double * a1, double a2, int a3) ;
extern void  __wrap_shmem_double_p(double * a1, double a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_p(double *, double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(double)*1);
  __real_shmem_double_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_p
 **********************************************************/

extern void  __real_shmem_longdouble_p(long double * a1, long double a2, int a3) ;
extern void  __wrap_shmem_longdouble_p(long double * a1, long double a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_p(long double *, long double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long double)*1);
  __real_shmem_longdouble_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_char_g
 **********************************************************/

extern char  __real_shmem_char_g(char * a1, int a2) ;
extern char  __wrap_shmem_char_g(char * a1, int a2)  {

  char retval;
  TAU_PROFILE_TIMER(t,"char shmem_char_g(char *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)*1, a2);
  retval  =  __real_shmem_char_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(char)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_g
 **********************************************************/

extern short  __real_shmem_short_g(short * a1, int a2) ;
extern short  __wrap_shmem_short_g(short * a1, int a2)  {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_g(short *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a2);
  retval  =  __real_shmem_short_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(short)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_g
 **********************************************************/

extern int  __real_shmem_int_g(int * a1, int a2) ;
extern int  __wrap_shmem_int_g(int * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_g(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a2);
  retval  =  __real_shmem_int_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(int)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_g
 **********************************************************/

extern long  __real_shmem_long_g(long * a1, int a2) ;
extern long  __wrap_shmem_long_g(long * a1, int a2)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_g(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a2);
  retval  =  __real_shmem_long_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_g
 **********************************************************/

extern long long  __real_shmem_longlong_g(long long * a1, int a2) ;
extern long long  __wrap_shmem_longlong_g(long long * a1, int a2)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_g(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a2);
  retval  =  __real_shmem_longlong_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long long)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_float_g
 **********************************************************/

extern float  __real_shmem_float_g(float * a1, int a2) ;
extern float  __wrap_shmem_float_g(float * a1, int a2)  {

  float retval;
  TAU_PROFILE_TIMER(t,"float shmem_float_g(float *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*1, a2);
  retval  =  __real_shmem_float_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(float)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_double_g
 **********************************************************/

extern double  __real_shmem_double_g(double * a1, int a2) ;
extern double  __wrap_shmem_double_g(double * a1, int a2)  {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_double_g(double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*1, a2);
  retval  =  __real_shmem_double_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(double)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longdouble_g
 **********************************************************/

extern long double  __real_shmem_longdouble_g(long double * a1, int a2) ;
extern long double  __wrap_shmem_longdouble_g(long double * a1, int a2)  {

  long double retval;
  TAU_PROFILE_TIMER(t,"long double shmem_longdouble_g(long double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*1, a2);
  retval  =  __real_shmem_longdouble_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long double)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_double_iput
 **********************************************************/

extern void  __real_shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_iput(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(double)*a5);
  __real_shmem_double_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_iput
 **********************************************************/

extern void  __real_shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_iput(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(float)*a5);
  __real_shmem_float_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_iput
 **********************************************************/

extern void  __real_shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_iput(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(int)*a5);
  __real_shmem_int_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput32
 **********************************************************/

extern void  __real_shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 4*a5);
  __real_shmem_iput32(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput64
 **********************************************************/

extern void  __real_shmem_iput64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iput64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 8*a5);
  __real_shmem_iput64(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput128
 **********************************************************/

extern void  __real_shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 16*a5);
  __real_shmem_iput128(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_iput
 **********************************************************/

extern void  __real_shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_iput(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long)*a5);
  __real_shmem_long_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_iput
 **********************************************************/

extern void  __real_shmem_longdouble_iput(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_longdouble_iput(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_iput(long double *, const long double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long double)*a5);
  __real_shmem_longdouble_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_iput
 **********************************************************/

extern void  __real_shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_iput(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long long)*a5);
  __real_shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_iput
 **********************************************************/

extern void  __real_shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_iput(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(short)*a5);
  __real_shmem_short_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_iget
 **********************************************************/

extern void  __real_shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_iget(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a5, a6);
  __real_shmem_double_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(double)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_iget
 **********************************************************/

extern void  __real_shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_iget(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a5, a6);
  __real_shmem_float_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(float)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_iget
 **********************************************************/

extern void  __real_shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_iget(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a5, a6);
  __real_shmem_int_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(int)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget32
 **********************************************************/

extern void  __real_shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a5, a6);
  __real_shmem_iget32(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 4*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget64
 **********************************************************/

extern void  __real_shmem_iget64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iget64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a5, a6);
  __real_shmem_iget64(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 8*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget128
 **********************************************************/

extern void  __real_shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a5, a6);
  __real_shmem_iget128(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 16*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_iget
 **********************************************************/

extern void  __real_shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_iget(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a5, a6);
  __real_shmem_long_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(long)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_iget
 **********************************************************/

extern void  __real_shmem_longdouble_iget(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_longdouble_iget(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_iget(long double *, const long double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*a5, a6);
  __real_shmem_longdouble_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(long double)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_iget
 **********************************************************/

extern void  __real_shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_iget(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a5, a6);
  __real_shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(long long)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_iget
 **********************************************************/

extern void  __real_shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_iget(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a5, a6);
  __real_shmem_short_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(short)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_all
 **********************************************************/

extern void  __real_shmem_barrier_all() ;
extern void  __wrap_shmem_barrier_all()  {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_all(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_barrier_all();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier
 **********************************************************/

extern void  __real_shmem_barrier(int a1, int a2, int a3, long * a4) ;
extern void  __wrap_shmem_barrier(int a1, int a2, int a3, long * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_barrier(int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_barrier(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fence
 **********************************************************/

extern void  __real_shmem_fence() ;
extern void  __wrap_shmem_fence()  {

  TAU_PROFILE_TIMER(t,"void shmem_fence(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_fence();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_quiet
 **********************************************************/

extern void  __real_shmem_quiet() ;
extern void  __wrap_shmem_quiet()  {

  TAU_PROFILE_TIMER(t,"void shmem_quiet(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_quiet();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_pe_accessible
 **********************************************************/

extern int  __real_shmem_pe_accessible(int a1) ;
extern int  __wrap_shmem_pe_accessible(int a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_pe_accessible(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_pe_accessible(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_addr_accessible
 **********************************************************/

extern int  __real_shmem_addr_accessible(const void * a1, int a2) ;
extern int  __wrap_shmem_addr_accessible(const void * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_addr_accessible(const void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_addr_accessible(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_free
 **********************************************************/

extern void  __real_shmem_free(void * a1) ;
extern void  __wrap_shmem_free(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_free(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_wait_until
 **********************************************************/

extern void  __real_shmem_long_wait_until(long * a1, int a2, long a3) ;
extern void  __wrap_shmem_long_wait_until(long * a1, int a2, long a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_wait_until(long *, int, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_wait_until
 **********************************************************/

extern void  __real_shmem_short_wait_until(short * a1, int a2, short a3) ;
extern void  __wrap_shmem_short_wait_until(short * a1, int a2, short a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_wait_until(short *, int, short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_wait_until
 **********************************************************/

extern void  __real_shmem_int_wait_until(int * a1, int a2, int a3) ;
extern void  __wrap_shmem_int_wait_until(int * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_wait_until(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_wait_until
 **********************************************************/

extern void  __real_shmem_longlong_wait_until(long long * a1, int a2, long long a3) ;
extern void  __wrap_shmem_longlong_wait_until(long long * a1, int a2, long long a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_wait_until(long long *, int, long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait_until
 **********************************************************/

extern void  __real_shmem_wait_until(long * a1, int a2, long a3) ;
extern void  __wrap_shmem_wait_until(long * a1, int a2, long a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_wait_until(long *, int, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_wait
 **********************************************************/

extern void  __real_shmem_long_wait(long * a1, long a2) ;
extern void  __wrap_shmem_long_wait(long * a1, long a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_wait(long *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_wait
 **********************************************************/

extern void  __real_shmem_short_wait(short * a1, short a2) ;
extern void  __wrap_shmem_short_wait(short * a1, short a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_wait(short *, short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_wait
 **********************************************************/

extern void  __real_shmem_int_wait(int * a1, int a2) ;
extern void  __wrap_shmem_int_wait(int * a1, int a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_wait(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_wait
 **********************************************************/

extern void  __real_shmem_longlong_wait(long long * a1, long long a2) ;
extern void  __wrap_shmem_longlong_wait(long long * a1, long long a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_wait(long long *, long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait
 **********************************************************/

extern void  __real_shmem_wait(long * a1, long a2) ;
extern void  __wrap_shmem_wait(long * a1, long a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_wait(long *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_swap
 **********************************************************/

extern long  __real_shmem_long_swap(long * a1, long a2, int a3) ;
extern long  __wrap_shmem_long_swap(long * a1, long a2, int a3)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_swap(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a3);
  retval  =  __real_shmem_long_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_swap
 **********************************************************/

extern int  __real_shmem_int_swap(int * a1, int a2, int a3) ;
extern int  __wrap_shmem_int_swap(int * a1, int a2, int a3)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_swap(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a3);
  retval  =  __real_shmem_int_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_swap
 **********************************************************/

extern long long  __real_shmem_longlong_swap(long long * a1, long long a2, int a3) ;
extern long long  __wrap_shmem_longlong_swap(long long * a1, long long a2, int a3)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_swap(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a3);
  retval  =  __real_shmem_longlong_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_float_swap
 **********************************************************/

extern float  __real_shmem_float_swap(float * a1, float a2, int a3) ;
extern float  __wrap_shmem_float_swap(float * a1, float a2, int a3)  {

  float retval;
  TAU_PROFILE_TIMER(t,"float shmem_float_swap(float *, float, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*1, a3);
  retval  =  __real_shmem_float_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(float)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(float)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_double_swap
 **********************************************************/

extern double  __real_shmem_double_swap(double * a1, double a2, int a3) ;
extern double  __wrap_shmem_double_swap(double * a1, double a2, int a3)  {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_double_swap(double *, double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*1, a3);
  retval  =  __real_shmem_double_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(double)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(double)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_swap
 **********************************************************/

extern long  __real_shmem_swap(long * a1, long a2, int a3) ;
extern long  __wrap_shmem_swap(long * a1, long a2, int a3)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_swap(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 1, a3);
  retval  =  __real_shmem_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, 1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, 1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_cswap
 **********************************************************/

extern long  __real_shmem_long_cswap(long * a1, long a2, long a3, int a4) ;
extern long  __wrap_shmem_long_cswap(long * a1, long a2, long a3, int a4)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_cswap(long *, long, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a4);
  retval  =  __real_shmem_long_cswap(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*1);
  if (retval == a2) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a4);
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_cswap
 **********************************************************/

extern int  __real_shmem_int_cswap(int * a1, int a2, int a3, int a4) ;
extern int  __wrap_shmem_int_cswap(int * a1, int a2, int a3, int a4)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_cswap(int *, int, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a4);
  retval  =  __real_shmem_int_cswap(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*1);
  if (retval == a2) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a4);
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_cswap
 **********************************************************/

extern long long  __real_shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4) ;
extern long long  __wrap_shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_cswap(long long *, long long, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a4);
  retval  =  __real_shmem_longlong_cswap(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*1);
  if (retval == a2) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a4);
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_fadd
 **********************************************************/

extern long  __real_shmem_long_fadd(long * a1, long a2, int a3) ;
extern long  __wrap_shmem_long_fadd(long * a1, long a2, int a3)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_fadd(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a3);
  retval  =  __real_shmem_long_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_fadd
 **********************************************************/

extern int  __real_shmem_int_fadd(int * a1, int a2, int a3) ;
extern int  __wrap_shmem_int_fadd(int * a1, int a2, int a3)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_fadd(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a3);
  retval  =  __real_shmem_int_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_fadd
 **********************************************************/

extern long long  __real_shmem_longlong_fadd(long long * a1, long long a2, int a3) ;
extern long long  __wrap_shmem_longlong_fadd(long long * a1, long long a2, int a3)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_fadd(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a3);
  retval  =  __real_shmem_longlong_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_finc
 **********************************************************/

extern long  __real_shmem_long_finc(long * a1, int a2) ;
extern long  __wrap_shmem_long_finc(long * a1, int a2)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_finc(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a2);
  retval  =  __real_shmem_long_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_finc
 **********************************************************/

extern int  __real_shmem_int_finc(int * a1, int a2) ;
extern int  __wrap_shmem_int_finc(int * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_finc(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a2);
  retval  =  __real_shmem_int_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_finc
 **********************************************************/

extern long long  __real_shmem_longlong_finc(long long * a1, int a2) ;
extern long long  __wrap_shmem_longlong_finc(long long * a1, int a2)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_finc(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a2);
  retval  =  __real_shmem_longlong_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_add
 **********************************************************/

extern void  __real_shmem_long_add(long * a1, long a2, int a3) ;
extern void  __wrap_shmem_long_add(long * a1, long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_add(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_add(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_add
 **********************************************************/

extern void  __real_shmem_int_add(int * a1, int a2, int a3) ;
extern void  __wrap_shmem_int_add(int * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_add(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_add(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_add
 **********************************************************/

extern void  __real_shmem_longlong_add(long long * a1, long long a2, int a3) ;
extern void  __wrap_shmem_longlong_add(long long * a1, long long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_add(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_add(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_inc
 **********************************************************/

extern void  __real_shmem_long_inc(long * a1, int a2) ;
extern void  __wrap_shmem_long_inc(long * a1, int a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_inc(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_inc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_inc
 **********************************************************/

extern void  __real_shmem_int_inc(int * a1, int a2) ;
extern void  __wrap_shmem_int_inc(int * a1, int a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_inc(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_inc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_inc
 **********************************************************/

extern void  __real_shmem_longlong_inc(long long * a1, int a2) ;
extern void  __wrap_shmem_longlong_inc(long long * a1, int a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_inc(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_inc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_cache_inv
 **********************************************************/

extern void  __real_shmem_clear_cache_inv() ;
extern void  __wrap_shmem_clear_cache_inv()  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_inv(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_clear_cache_inv();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_cache_inv
 **********************************************************/

extern void  __real_shmem_set_cache_inv() ;
extern void  __wrap_shmem_set_cache_inv()  {

  TAU_PROFILE_TIMER(t,"void shmem_set_cache_inv(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_set_cache_inv();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_cache_line_inv
 **********************************************************/

extern void  __real_shmem_clear_cache_line_inv(void * a1) ;
extern void  __wrap_shmem_clear_cache_line_inv(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_cache_line_inv(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_clear_cache_line_inv(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_cache_line_inv
 **********************************************************/

extern void  __real_shmem_set_cache_line_inv(void * a1) ;
extern void  __wrap_shmem_set_cache_line_inv(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_set_cache_line_inv(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_set_cache_line_inv(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_udcflush
 **********************************************************/

extern void  __real_shmem_udcflush() ;
extern void  __wrap_shmem_udcflush()  {

  TAU_PROFILE_TIMER(t,"void shmem_udcflush(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_udcflush();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_udcflush_line
 **********************************************************/

extern void  __real_shmem_udcflush_line(void * a1) ;
extern void  __wrap_shmem_udcflush_line(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_udcflush_line(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_udcflush_line(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

extern void  __real_shmem_long_sum_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_sum_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_sum_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complexd_sum_to_all
 **********************************************************/

extern void  __real_shmem_complexd_sum_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8) ;
extern void  __wrap_shmem_complexd_sum_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_complexd_sum_to_all(double _Complex *, double _Complex *, int, int, int, int, double _Complex *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_complexd_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complexf_sum_to_all
 **********************************************************/

extern void  __real_shmem_complexf_sum_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8) ;
extern void  __wrap_shmem_complexf_sum_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_complexf_sum_to_all(float _Complex *, float _Complex *, int, int, int, int, float _Complex *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_complexf_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

extern void  __real_shmem_double_sum_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_sum_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_sum_to_all(double *, double *, int, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

extern void  __real_shmem_float_sum_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_sum_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_sum_to_all(float *, float *, int, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

extern void  __real_shmem_int_sum_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_sum_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_sum_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_sum_to_all
 **********************************************************/

extern void  __real_shmem_longdouble_sum_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_longdouble_sum_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_sum_to_all(long double *, long double *, int, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longdouble_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

extern void  __real_shmem_longlong_sum_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_sum_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_sum_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_sum_to_all
 **********************************************************/

extern void  __real_shmem_short_sum_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_sum_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_sum_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complexd_prod_to_all
 **********************************************************/

extern void  __real_shmem_complexd_prod_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8) ;
extern void  __wrap_shmem_complexd_prod_to_all(double _Complex * a1, double _Complex * a2, int a3, int a4, int a5, int a6, double _Complex * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_complexd_prod_to_all(double _Complex *, double _Complex *, int, int, int, int, double _Complex *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_complexd_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_complexf_prod_to_all
 **********************************************************/

extern void  __real_shmem_complexf_prod_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8) ;
extern void  __wrap_shmem_complexf_prod_to_all(float _Complex * a1, float _Complex * a2, int a3, int a4, int a5, int a6, float _Complex * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_complexf_prod_to_all(float _Complex *, float _Complex *, int, int, int, int, float _Complex *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_complexf_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

extern void  __real_shmem_double_prod_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_prod_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_prod_to_all(double *, double *, int, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

extern void  __real_shmem_float_prod_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_prod_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_prod_to_all(float *, float *, int, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

extern void  __real_shmem_int_prod_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_prod_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_prod_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

extern void  __real_shmem_long_prod_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_prod_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_prod_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_prod_to_all
 **********************************************************/

extern void  __real_shmem_longdouble_prod_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_longdouble_prod_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_prod_to_all(long double *, long double *, int, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longdouble_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

extern void  __real_shmem_longlong_prod_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_prod_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_prod_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

extern void  __real_shmem_short_prod_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_prod_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_prod_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

extern void  __real_shmem_int_and_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_and_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_and_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

extern void  __real_shmem_long_and_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_and_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_and_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

extern void  __real_shmem_longlong_and_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_and_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_and_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

extern void  __real_shmem_short_and_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_and_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_and_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

extern void  __real_shmem_int_or_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_or_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_or_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

extern void  __real_shmem_long_or_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_or_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_or_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

extern void  __real_shmem_longlong_or_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_or_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_or_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

extern void  __real_shmem_short_or_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_or_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_or_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

extern void  __real_shmem_int_xor_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_xor_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_xor_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

extern void  __real_shmem_long_xor_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_xor_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_xor_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

extern void  __real_shmem_longlong_xor_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_xor_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_xor_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

extern void  __real_shmem_short_xor_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_xor_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_xor_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

extern void  __real_shmem_int_max_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_max_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_max_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

extern void  __real_shmem_long_max_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_max_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_max_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

extern void  __real_shmem_longlong_max_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_max_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_max_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

extern void  __real_shmem_short_max_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_max_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_max_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_max_to_all
 **********************************************************/

extern void  __real_shmem_longdouble_max_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_longdouble_max_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_max_to_all(long double *, long double *, int, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longdouble_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

extern void  __real_shmem_float_max_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_max_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_max_to_all(float *, float *, int, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

extern void  __real_shmem_double_max_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_max_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_max_to_all(double *, double *, int, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

extern void  __real_shmem_int_min_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_min_to_all(int * a1, int * a2, int a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_min_to_all(int *, int *, int, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

extern void  __real_shmem_long_min_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_min_to_all(long * a1, long * a2, int a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_min_to_all(long *, long *, int, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

extern void  __real_shmem_longlong_min_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_min_to_all(long long * a1, long long * a2, int a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_min_to_all(long long *, long long *, int, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

extern void  __real_shmem_short_min_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_min_to_all(short * a1, short * a2, int a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_min_to_all(short *, short *, int, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_min_to_all
 **********************************************************/

extern void  __real_shmem_longdouble_min_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_longdouble_min_to_all(long double * a1, long double * a2, int a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_min_to_all(long double *, long double *, int, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longdouble_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

extern void  __real_shmem_float_min_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_min_to_all(float * a1, float * a2, int a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_min_to_all(float *, float *, int, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

extern void  __real_shmem_double_min_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_min_to_all(double * a1, double * a2, int a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_min_to_all(double *, double *, int, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast64
 **********************************************************/

extern void  __real_shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) ;
extern void  __wrap_shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast64(void *, const void *, size_t, int, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_broadcast64(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast32
 **********************************************************/

extern void  __real_shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) ;
extern void  __wrap_shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast32(void *, const void *, size_t, int, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_broadcast32(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect64
 **********************************************************/

extern void  __real_shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect64(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect32
 **********************************************************/

extern void  __real_shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect32(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect64
 **********************************************************/

extern void  __real_shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_collect64(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect32
 **********************************************************/

extern void  __real_shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_collect32(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_lock
 **********************************************************/

extern void  __real_shmem_set_lock(long * a1) ;
extern void  __wrap_shmem_set_lock(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_set_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_set_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_lock
 **********************************************************/

extern void  __real_shmem_clear_lock(long * a1) ;
extern void  __wrap_shmem_clear_lock(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_clear_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_test_lock
 **********************************************************/

extern int  __real_shmem_test_lock(long * a1) ;
extern int  __wrap_shmem_test_lock(long * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_test_lock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemx_short_put_nb
 **********************************************************/

extern void  __real_shmemx_short_put_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_short_put_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_short_put_nb(short *, const short *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*a3);
  __real_shmemx_short_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_int_put_nb
 **********************************************************/

extern void  __real_shmemx_int_put_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_int_put_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_int_put_nb(int *, const int *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*a3);
  __real_shmemx_int_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_long_put_nb
 **********************************************************/

extern void  __real_shmemx_long_put_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_long_put_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_long_put_nb(long *, const long *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*a3);
  __real_shmemx_long_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_longlong_put_nb
 **********************************************************/

extern void  __real_shmemx_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_longlong_put_nb(long long *, const long long *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*a3);
  __real_shmemx_longlong_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_longdouble_put_nb
 **********************************************************/

extern void  __real_shmemx_longdouble_put_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_longdouble_put_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_longdouble_put_nb(long double *, const long double *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long double)*a3);
  __real_shmemx_longdouble_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_double_put_nb
 **********************************************************/

extern void  __real_shmemx_double_put_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_double_put_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_double_put_nb(double *, const double *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*a3);
  __real_shmemx_double_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_float_put_nb
 **********************************************************/

extern void  __real_shmemx_float_put_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_float_put_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_float_put_nb(float *, const float *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmemx_float_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_putmem_nb
 **********************************************************/

extern void  __real_shmemx_putmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_putmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_putmem_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, a3);
  __real_shmemx_putmem_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_put32_nb
 **********************************************************/

extern void  __real_shmemx_put32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_put32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_put32_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
  __real_shmemx_put32_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_put64_nb
 **********************************************************/

extern void  __real_shmemx_put64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_put64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_put64_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
  __real_shmemx_put64_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_put128_nb
 **********************************************************/

extern void  __real_shmemx_put128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_put128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_put128_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 16*a3);
  __real_shmemx_put128_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_short_get_nb
 **********************************************************/

extern void  __real_shmemx_short_get_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_short_get_nb(short * a1, const short * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_short_get_nb(short *, const short *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a3, a4);
  __real_shmemx_short_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_int_get_nb
 **********************************************************/

extern void  __real_shmemx_int_get_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_int_get_nb(int * a1, const int * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_int_get_nb(int *, const int *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a3, a4);
  __real_shmemx_int_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_long_get_nb
 **********************************************************/

extern void  __real_shmemx_long_get_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_long_get_nb(long * a1, const long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_long_get_nb(long *, const long *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a3, a4);
  __real_shmemx_long_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_longlong_get_nb
 **********************************************************/

extern void  __real_shmemx_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_longlong_get_nb(long long *, const long long *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a3, a4);
  __real_shmemx_longlong_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_longdouble_get_nb
 **********************************************************/

extern void  __real_shmemx_longdouble_get_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_longdouble_get_nb(long double * a1, const long double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_longdouble_get_nb(long double *, const long double *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*a3, a4);
  __real_shmemx_longdouble_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_double_get_nb
 **********************************************************/

extern void  __real_shmemx_double_get_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_double_get_nb(double * a1, const double * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_double_get_nb(double *, const double *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a3, a4);
  __real_shmemx_double_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_float_get_nb
 **********************************************************/

extern void  __real_shmemx_float_get_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_float_get_nb(float * a1, const float * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_float_get_nb(float *, const float *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmemx_float_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_getmem_nb
 **********************************************************/

extern void  __real_shmemx_getmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_getmem_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_getmem_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), a3, a4);
  __real_shmemx_getmem_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_get32_nb
 **********************************************************/

extern void  __real_shmemx_get32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_get32_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_get32_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a3, a4);
  __real_shmemx_get32_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 4*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_get64_nb
 **********************************************************/

extern void  __real_shmemx_get64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_get64_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_get64_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
  __real_shmemx_get64_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_get128_nb
 **********************************************************/

extern void  __real_shmemx_get128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5) ;
extern void  __wrap_shmemx_get128_nb(void * a1, const void * a2, size_t a3, int a4, shmemx_request_handle_t * a5)  {

  TAU_PROFILE_TIMER(t,"void shmemx_get128_nb(void *, const void *, size_t, int, shmemx_request_handle_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a3, a4);
  __real_shmemx_get128_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 16*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_wait_req
 **********************************************************/

extern void  __real_shmemx_wait_req(shmemx_request_handle_t a1) ;
extern void  __wrap_shmemx_wait_req(shmemx_request_handle_t a1)  {

  TAU_PROFILE_TIMER(t,"void shmemx_wait_req(shmemx_request_handle_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_wait_req(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_test_req
 **********************************************************/

extern void  __real_shmemx_test_req(shmemx_request_handle_t a1, int * a2) ;
extern void  __wrap_shmemx_test_req(shmemx_request_handle_t a1, int * a2)  {

  TAU_PROFILE_TIMER(t,"void shmemx_test_req(shmemx_request_handle_t, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_test_req(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_int_xor
 **********************************************************/

extern void  __real_shmemx_int_xor(int * a1, int a2, int a3) ;
extern void  __wrap_shmemx_int_xor(int * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmemx_int_xor(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_int_xor(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_long_xor
 **********************************************************/

extern void  __real_shmemx_long_xor(long * a1, long a2, int a3) ;
extern void  __wrap_shmemx_long_xor(long * a1, long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmemx_long_xor(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_long_xor(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_longlong_xor
 **********************************************************/

extern void  __real_shmemx_longlong_xor(long long * a1, long long a2, int a3) ;
extern void  __wrap_shmemx_longlong_xor(long long * a1, long long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmemx_longlong_xor(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_longlong_xor(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_int_fetch
 **********************************************************/

extern int  __real_shmemx_int_fetch(int * a1, int a2) ;
extern int  __wrap_shmemx_int_fetch(int * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmemx_int_fetch(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmemx_int_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemx_long_fetch
 **********************************************************/

extern long  __real_shmemx_long_fetch(long * a1, int a2) ;
extern long  __wrap_shmemx_long_fetch(long * a1, int a2)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmemx_long_fetch(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmemx_long_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemx_longlong_fetch
 **********************************************************/

extern long long  __real_shmemx_longlong_fetch(long long * a1, int a2) ;
extern long long  __wrap_shmemx_longlong_fetch(long long * a1, int a2)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmemx_longlong_fetch(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmemx_longlong_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemx_int_set
 **********************************************************/

extern void  __real_shmemx_int_set(int * a1, int a2, int a3) ;
extern void  __wrap_shmemx_int_set(int * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmemx_int_set(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_int_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_long_set
 **********************************************************/

extern void  __real_shmemx_long_set(long * a1, long a2, int a3) ;
extern void  __wrap_shmemx_long_set(long * a1, long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmemx_long_set(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_long_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_longlong_set
 **********************************************************/

extern void  __real_shmemx_longlong_set(long long * a1, long long a2, int a3) ;
extern void  __wrap_shmemx_longlong_set(long long * a1, long long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmemx_longlong_set(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmemx_longlong_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmemx_wtime
 **********************************************************/

extern double  __real_shmemx_wtime() ;
extern double  __wrap_shmemx_wtime()  {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmemx_wtime(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmemx_wtime();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemx_fence_test
 **********************************************************/

extern int  __real_shmemx_fence_test() ;
extern int  __wrap_shmemx_fence_test()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmemx_fence_test(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmemx_fence_test();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemx_quiet_test
 **********************************************************/

extern int  __real_shmemx_quiet_test() ;
extern int  __wrap_shmemx_quiet_test()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmemx_quiet_test(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmemx_quiet_test();
  TAU_PROFILE_STOP(t);
  return retval;

}

