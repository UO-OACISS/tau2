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
   shmem_float128_get
 **********************************************************/

extern void  __real_shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float128_get(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_get(__float128 *, const __float128 *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmem_float128_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

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
   shmem_float128_put
 **********************************************************/

extern void  __real_shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float128_put(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_put(__float128 *, const __float128 *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmem_float128_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_put_signal
 **********************************************************/

extern void  __real_shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_short_put_signal(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_put_signal(short *, const short *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(short)*a3);
  __real_shmem_short_put_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_put_signal
 **********************************************************/

extern void  __real_shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_int_put_signal(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_put_signal(int *, const int *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(int)*a3);
  __real_shmem_int_put_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_put_signal
 **********************************************************/

extern void  __real_shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_long_put_signal(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_put_signal(long *, const long *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long)*a3);
  __real_shmem_long_put_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_put_signal
 **********************************************************/

extern void  __real_shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_longlong_put_signal(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put_signal(long long *, const long long *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long long)*a3);
  __real_shmem_longlong_put_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_put_signal
 **********************************************************/

extern void  __real_shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_float_put_signal(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_put_signal(float *, const float *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(float)*a3);
  __real_shmem_float_put_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_signal
 **********************************************************/

extern void  __real_shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_double_put_signal(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_signal(double *, const double *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(double)*a3);
  __real_shmem_double_put_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_get_nb
 **********************************************************/

extern void  __real_shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_short_get_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_get_nb(short *, const short *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a3, a4);
  __real_shmem_short_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_get_nb
 **********************************************************/

extern void  __real_shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_int_get_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_get_nb(int *, const int *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a3, a4);
  __real_shmem_int_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_get_nb
 **********************************************************/

extern void  __real_shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_long_get_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_get_nb(long *, const long *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a3, a4);
  __real_shmem_long_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_get_nb
 **********************************************************/

extern void  __real_shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_longlong_get_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_get_nb(long long *, const long long *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a3, a4);
  __real_shmem_longlong_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_get_nb
 **********************************************************/

extern void  __real_shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_float_get_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_get_nb(float *, const float *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmem_float_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_get_nb
 **********************************************************/

extern void  __real_shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_double_get_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_get_nb(double *, const double *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a3, a4);
  __real_shmem_double_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_get_nb
 **********************************************************/

extern void  __real_shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_float128_get_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_get_nb(__float128 *, const __float128 *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmem_float128_get_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_put_nb
 **********************************************************/

extern void  __real_shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_put_nb(short *, const short *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*a3);
  __real_shmem_short_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_put_nb
 **********************************************************/

extern void  __real_shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_put_nb(int *, const int *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*a3);
  __real_shmem_int_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_put_nb
 **********************************************************/

extern void  __real_shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_put_nb(long *, const long *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*a3);
  __real_shmem_long_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_put_nb
 **********************************************************/

extern void  __real_shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put_nb(long long *, const long long *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*a3);
  __real_shmem_longlong_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_put_nb
 **********************************************************/

extern void  __real_shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_put_nb(float *, const float *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmem_float_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_nb
 **********************************************************/

extern void  __real_shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_nb(double *, const double *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*a3);
  __real_shmem_double_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_put_nb
 **********************************************************/

extern void  __real_shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_float128_put_nb(__float128 * a1, const __float128 * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_put_nb(__float128 *, const __float128 *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmem_float128_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_put_signal_nb
 **********************************************************/

extern void  __real_shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_short_put_signal_nb(short * a1, const short * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_put_signal_nb(short *, const short *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(short)*a3);
  __real_shmem_short_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_put_signal_nb
 **********************************************************/

extern void  __real_shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_int_put_signal_nb(int * a1, const int * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_put_signal_nb(int *, const int *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(int)*a3);
  __real_shmem_int_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_put_signal_nb
 **********************************************************/

extern void  __real_shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_long_put_signal_nb(long * a1, const long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_put_signal_nb(long *, const long *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long)*a3);
  __real_shmem_long_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_put_signal_nb
 **********************************************************/

extern void  __real_shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_longlong_put_signal_nb(long long * a1, const long long * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put_signal_nb(long long *, const long long *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long long)*a3);
  __real_shmem_longlong_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_put_signal_nb
 **********************************************************/

extern void  __real_shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_float_put_signal_nb(float * a1, const float * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_put_signal_nb(float *, const float *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(float)*a3);
  __real_shmem_float_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_signal_nb
 **********************************************************/

extern void  __real_shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_double_put_signal_nb(double * a1, const double * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_signal_nb(double *, const double *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(double)*a3);
  __real_shmem_double_put_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a6);
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
   shmem_float128_iget
 **********************************************************/

extern void  __real_shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_float128_iget(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_iget(__float128 *, const __float128 *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a5, a6);
  __real_shmem_float128_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(float)*a5);
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
   shmem_float128_iput
 **********************************************************/

extern void  __real_shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_float128_iput(__float128 * a1, const __float128 * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_iput(__float128 *, const __float128 *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(float)*a5);
  __real_shmem_float128_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_char_g
 **********************************************************/

extern char  __real_shmem_char_g(const char * a1, int a2) ;
extern char  __wrap_shmem_char_g(const char * a1, int a2)  {

  char retval;
  TAU_PROFILE_TIMER(t,"char shmem_char_g(const char *, int) C", "", TAU_USER);
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

extern short  __real_shmem_short_g(const short * a1, int a2) ;
extern short  __wrap_shmem_short_g(const short * a1, int a2)  {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_g(const short *, int) C", "", TAU_USER);
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

extern int  __real_shmem_int_g(const int * a1, int a2) ;
extern int  __wrap_shmem_int_g(const int * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_g(const int *, int) C", "", TAU_USER);
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

extern long  __real_shmem_long_g(const long * a1, int a2) ;
extern long  __wrap_shmem_long_g(const long * a1, int a2)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_g(const long *, int) C", "", TAU_USER);
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

extern long long  __real_shmem_longlong_g(const long long * a1, int a2) ;
extern long long  __wrap_shmem_longlong_g(const long long * a1, int a2)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_g(const long long *, int) C", "", TAU_USER);
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

extern float  __real_shmem_float_g(const float * a1, int a2) ;
extern float  __wrap_shmem_float_g(const float * a1, int a2)  {

  float retval;
  TAU_PROFILE_TIMER(t,"float shmem_float_g(const float *, int) C", "", TAU_USER);
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

extern double  __real_shmem_double_g(const double * a1, int a2) ;
extern double  __wrap_shmem_double_g(const double * a1, int a2)  {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_double_g(const double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*1, a2);
  retval  =  __real_shmem_double_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(double)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_ld80_g
 **********************************************************/

extern long double  __real_shmem_ld80_g(const long double * a1, int a2) ;
extern long double  __wrap_shmem_ld80_g(const long double * a1, int a2)  {

  long double retval;
  TAU_PROFILE_TIMER(t,"long double shmem_ld80_g(const long double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*1, a2);
  retval  =  __real_shmem_ld80_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, 8*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_float128_g
 **********************************************************/

extern __float128  __real_shmem_float128_g(const __float128 * a1, int a2) ;
extern __float128  __wrap_shmem_float128_g(const __float128 * a1, int a2)  {

  __float128 retval;
  TAU_PROFILE_TIMER(t,"__float128 shmem_float128_g(const __float128 *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*1, a2);
  retval  =  __real_shmem_float128_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(float)*1);
  TAU_PROFILE_STOP(t);
  return retval;

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
   shmem_ld80_p
 **********************************************************/

extern void  __real_shmem_ld80_p(long double * a1, long double a2, int a3) ;
extern void  __wrap_shmem_ld80_p(long double * a1, long double a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_ld80_p(long double *, long double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, 8*1);
  __real_shmem_ld80_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_p
 **********************************************************/

extern void  __real_shmem_float128_p(__float128 * a1, __float128 a2, int a3) ;
extern void  __wrap_shmem_float128_p(__float128 * a1, __float128 a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_p(__float128 *, __float128, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(float)*1);
  __real_shmem_float128_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_swap
 **********************************************************/

extern short  __real_shmem_short_swap(short * a1, short a2, int a3) ;
extern short  __wrap_shmem_short_swap(short * a1, short a2, int a3)  {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_swap(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a3);
  retval  =  __real_shmem_short_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
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
   shmem_short_swap_nb
 **********************************************************/

extern void  __real_shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5) ;
extern void  __wrap_shmem_short_swap_nb(short * a1, short * a2, short a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_swap_nb(short *, short *, short, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a4);
  __real_shmem_short_swap_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_swap_nb
 **********************************************************/

extern void  __real_shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5) ;
extern void  __wrap_shmem_int_swap_nb(int * a1, int * a2, int a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_swap_nb(int *, int *, int, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a4);
  __real_shmem_int_swap_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_swap_nb
 **********************************************************/

extern void  __real_shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5) ;
extern void  __wrap_shmem_long_swap_nb(long * a1, long * a2, long a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_swap_nb(long *, long *, long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a4);
  __real_shmem_long_swap_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_swap_nb
 **********************************************************/

extern void  __real_shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) ;
extern void  __wrap_shmem_longlong_swap_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_swap_nb(long long *, long long *, long long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a4);
  __real_shmem_longlong_swap_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_swap_nb
 **********************************************************/

extern void  __real_shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5) ;
extern void  __wrap_shmem_float_swap_nb(float * a1, float * a2, float a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_swap_nb(float *, float *, float, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*1, a4);
  __real_shmem_float_swap_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_swap_nb
 **********************************************************/

extern void  __real_shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5) ;
extern void  __wrap_shmem_double_swap_nb(double * a1, double * a2, double a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_swap_nb(double *, double *, double, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*1, a4);
  __real_shmem_double_swap_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_cswap
 **********************************************************/

extern short  __real_shmem_short_cswap(short * a1, short a2, short a3, int a4) ;
extern short  __wrap_shmem_short_cswap(short * a1, short a2, short a3, int a4)  {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_cswap(short *, short, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a4);
  retval  =  __real_shmem_short_cswap(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*1);
  if (retval == a2) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a4);
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
   shmem_short_cswap_nb
 **********************************************************/

extern void  __real_shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6) ;
extern void  __wrap_shmem_short_cswap_nb(short * a1, short * a2, short a3, short a4, int a5, void ** a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_cswap_nb(short *, short *, short, short, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a5);
  __real_shmem_short_cswap_nb(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a5, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a5, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_cswap_nb
 **********************************************************/

extern void  __real_shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6) ;
extern void  __wrap_shmem_int_cswap_nb(int * a1, int * a2, int a3, int a4, int a5, void ** a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_cswap_nb(int *, int *, int, int, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a5);
  __real_shmem_int_cswap_nb(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a5, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a5, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_cswap_nb
 **********************************************************/

extern void  __real_shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6) ;
extern void  __wrap_shmem_long_cswap_nb(long * a1, long * a2, long a3, long a4, int a5, void ** a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_cswap_nb(long *, long *, long, long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a5);
  __real_shmem_long_cswap_nb(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a5, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a5, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_cswap_nb
 **********************************************************/

extern void  __real_shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6) ;
extern void  __wrap_shmem_longlong_cswap_nb(long long * a1, long long * a2, long long a3, long long a4, int a5, void ** a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_cswap_nb(long long *, long long *, long long, long long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a5);
  __real_shmem_longlong_cswap_nb(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a5, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a5, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_finc
 **********************************************************/

extern short  __real_shmem_short_finc(short * a1, int a2) ;
extern short  __wrap_shmem_short_finc(short * a1, int a2)  {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_finc(short *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a2);
  retval  =  __real_shmem_short_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a2);
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
   shmem_short_finc_nb
 **********************************************************/

extern void  __real_shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4) ;
extern void  __wrap_shmem_short_finc_nb(short * a1, short * a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_finc_nb(short *, short *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a3);
  __real_shmem_short_finc_nb(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_finc_nb
 **********************************************************/

extern void  __real_shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4) ;
extern void  __wrap_shmem_int_finc_nb(int * a1, int * a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_finc_nb(int *, int *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a3);
  __real_shmem_int_finc_nb(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_finc_nb
 **********************************************************/

extern void  __real_shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4) ;
extern void  __wrap_shmem_long_finc_nb(long * a1, long * a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_finc_nb(long *, long *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a3);
  __real_shmem_long_finc_nb(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_finc_nb
 **********************************************************/

extern void  __real_shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4) ;
extern void  __wrap_shmem_longlong_finc_nb(long long * a1, long long * a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_finc_nb(long long *, long long *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a3);
  __real_shmem_longlong_finc_nb(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_inc
 **********************************************************/

extern void  __real_shmem_short_inc(short * a1, int a2) ;
extern void  __wrap_shmem_short_inc(short * a1, int a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_inc(short *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_inc(a1, a2);
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
   shmem_short_inc_nb
 **********************************************************/

extern void  __real_shmem_short_inc_nb(short * a1, int a2, void ** a3) ;
extern void  __wrap_shmem_short_inc_nb(short * a1, int a2, void ** a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_inc_nb(short *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_inc_nb(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_inc_nb
 **********************************************************/

extern void  __real_shmem_int_inc_nb(int * a1, int a2, void ** a3) ;
extern void  __wrap_shmem_int_inc_nb(int * a1, int a2, void ** a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_inc_nb(int *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_inc_nb(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_inc_nb
 **********************************************************/

extern void  __real_shmem_long_inc_nb(long * a1, int a2, void ** a3) ;
extern void  __wrap_shmem_long_inc_nb(long * a1, int a2, void ** a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_inc_nb(long *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_inc_nb(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_inc_nb
 **********************************************************/

extern void  __real_shmem_longlong_inc_nb(long long * a1, int a2, void ** a3) ;
extern void  __wrap_shmem_longlong_inc_nb(long long * a1, int a2, void ** a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_inc_nb(long long *, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_inc_nb(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_fadd
 **********************************************************/

extern short  __real_shmem_short_fadd(short * a1, short a2, int a3) ;
extern short  __wrap_shmem_short_fadd(short * a1, short a2, int a3)  {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_fadd(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a3);
  retval  =  __real_shmem_short_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
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
   shmem_short_fadd_nb
 **********************************************************/

extern void  __real_shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5) ;
extern void  __wrap_shmem_short_fadd_nb(short * a1, short * a2, short a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_fadd_nb(short *, short *, short, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a4);
  __real_shmem_short_fadd_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_fadd_nb
 **********************************************************/

extern void  __real_shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5) ;
extern void  __wrap_shmem_int_fadd_nb(int * a1, int * a2, int a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_fadd_nb(int *, int *, int, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a4);
  __real_shmem_int_fadd_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_fadd_nb
 **********************************************************/

extern void  __real_shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5) ;
extern void  __wrap_shmem_long_fadd_nb(long * a1, long * a2, long a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_fadd_nb(long *, long *, long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a4);
  __real_shmem_long_fadd_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_fadd_nb
 **********************************************************/

extern void  __real_shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5) ;
extern void  __wrap_shmem_longlong_fadd_nb(long long * a1, long long * a2, long long a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_fadd_nb(long long *, long long *, long long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a4);
  __real_shmem_longlong_fadd_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_add
 **********************************************************/

extern void  __real_shmem_short_add(short * a1, short a2, int a3) ;
extern void  __wrap_shmem_short_add(short * a1, short a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_add(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_add(a1, a2, a3);
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
   shmem_short_add_nb
 **********************************************************/

extern void  __real_shmem_short_add_nb(short * a1, short a2, int a3, void ** a4) ;
extern void  __wrap_shmem_short_add_nb(short * a1, short a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_add_nb(short *, short, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_add_nb(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_add_nb
 **********************************************************/

extern void  __real_shmem_int_add_nb(int * a1, int a2, int a3, void ** a4) ;
extern void  __wrap_shmem_int_add_nb(int * a1, int a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_add_nb(int *, int, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_add_nb(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_add_nb
 **********************************************************/

extern void  __real_shmem_long_add_nb(long * a1, long a2, int a3, void ** a4) ;
extern void  __wrap_shmem_long_add_nb(long * a1, long a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_add_nb(long *, long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_add_nb(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_add_nb
 **********************************************************/

extern void  __real_shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4) ;
extern void  __wrap_shmem_longlong_add_nb(long long * a1, long long a2, int a3, void ** a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_add_nb(long long *, long long, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_add_nb(a1, a2, a3, a4);
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
   shmem_team_barrier
 **********************************************************/

extern void  __real_shmem_team_barrier(shmem_team_t a1, long * a2) ;
extern void  __wrap_shmem_team_barrier(shmem_team_t a1, long * a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_barrier(shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_barrier(a1, a2);
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
   shmem_clear_event
 **********************************************************/

extern void  __real_shmem_clear_event(long * a1) ;
extern void  __wrap_shmem_clear_event(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_clear_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_clear_event(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_event
 **********************************************************/

extern void  __real_shmem_set_event(long * a1) ;
extern void  __wrap_shmem_set_event(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_set_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_set_event(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_test_event
 **********************************************************/

extern int  __real_shmem_test_event(long * a1) ;
extern int  __wrap_shmem_test_event(long * a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_test_event(a1);
  TAU_PROFILE_STOP(t);
  return retval;

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
   shmem_short_sum_to_all
 **********************************************************/

extern void  __real_shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_sum_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

extern void  __real_shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_max_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

extern void  __real_shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_min_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

extern void  __real_shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_prod_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

extern void  __real_shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_and_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

extern void  __real_shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_or_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

extern void  __real_shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) ;
extern void  __wrap_shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_xor_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

extern void  __real_shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_sum_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

extern void  __real_shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_max_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

extern void  __real_shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_min_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

extern void  __real_shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_prod_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

extern void  __real_shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_and_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

extern void  __real_shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_or_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

extern void  __real_shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) ;
extern void  __wrap_shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_xor_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

extern void  __real_shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_sum_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

extern void  __real_shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_max_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

extern void  __real_shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_min_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

extern void  __real_shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_prod_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

extern void  __real_shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_and_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

extern void  __real_shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_or_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

extern void  __real_shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) ;
extern void  __wrap_shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_xor_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

extern void  __real_shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_sum_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

extern void  __real_shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_max_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

extern void  __real_shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_min_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

extern void  __real_shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_prod_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

extern void  __real_shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_and_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

extern void  __real_shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_or_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

extern void  __real_shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) ;
extern void  __wrap_shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_xor_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

extern void  __real_shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_sum_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

extern void  __real_shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_max_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

extern void  __real_shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_min_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

extern void  __real_shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) ;
extern void  __wrap_shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_prod_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

extern void  __real_shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_sum_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

extern void  __real_shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_max_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

extern void  __real_shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_min_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

extern void  __real_shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) ;
extern void  __wrap_shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_prod_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_ld80_sum_to_all
 **********************************************************/

extern void  __real_shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_ld80_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_ld80_sum_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_ld80_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_ld80_max_to_all
 **********************************************************/

extern void  __real_shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_ld80_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_ld80_max_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_ld80_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_ld80_min_to_all
 **********************************************************/

extern void  __real_shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_ld80_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_ld80_min_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_ld80_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_ld80_prod_to_all
 **********************************************************/

extern void  __real_shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) ;
extern void  __wrap_shmem_ld80_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_ld80_prod_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_ld80_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_sum_to_all
 **********************************************************/

extern void  __real_shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __wrap_shmem_float128_sum_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_sum_to_all(__float128 *, const __float128 *, size_t, int, int, int, __float128 *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float128_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_max_to_all
 **********************************************************/

extern void  __real_shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __wrap_shmem_float128_max_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_max_to_all(__float128 *, const __float128 *, size_t, int, int, int, __float128 *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float128_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_min_to_all
 **********************************************************/

extern void  __real_shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __wrap_shmem_float128_min_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_min_to_all(__float128 *, const __float128 *, size_t, int, int, int, __float128 *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float128_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float128_prod_to_all
 **********************************************************/

extern void  __real_shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8) ;
extern void  __wrap_shmem_float128_prod_to_all(__float128 * a1, const __float128 * a2, size_t a3, int a4, int a5, int a6, __float128 * a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_prod_to_all(__float128 *, const __float128 *, size_t, int, int, int, __float128 *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float128_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_alltoall
 **********************************************************/

extern void  __real_shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_alltoall(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoall(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoall(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_team_alltoall
 **********************************************************/

extern void  __real_shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) ;
extern void  __wrap_shmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_alltoall(void *, const void *, size_t, shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_alltoall(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   pshmem_team_alltoall
 **********************************************************/

extern void  __real_pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5) ;
extern void  __wrap_pshmem_team_alltoall(void * a1, const void * a2, size_t a3, shmem_team_t a4, long * a5)  {

  TAU_PROFILE_TIMER(t,"void pshmem_team_alltoall(void *, const void *, size_t, shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_pshmem_team_alltoall(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_alltoallv
 **********************************************************/

extern void  __real_shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) ;
extern void  __wrap_shmem_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoallv(void *, size_t *, size_t *, const void *, size_t *, size_t *, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_team_alltoallv
 **********************************************************/

extern void  __real_shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __wrap_shmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_alltoallv(void *, size_t *, size_t *, const void *, size_t *, size_t *, shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   pshmem_team_alltoallv
 **********************************************************/

extern void  __real_pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __wrap_pshmem_team_alltoallv(void * a1, size_t * a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void pshmem_team_alltoallv(void *, size_t *, size_t *, const void *, size_t *, size_t *, shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_alltoallv_packed
 **********************************************************/

extern void  __real_shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10) ;
extern void  __wrap_shmem_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, int a7, int a8, int a9, long * a10)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoallv_packed(void *, size_t, size_t *, const void *, size_t *, size_t *, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_team_alltoallv_packed
 **********************************************************/

extern void  __real_shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __wrap_shmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_alltoallv_packed(void *, size_t, size_t *, const void *, size_t *, size_t *, shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   pshmem_team_alltoallv_packed
 **********************************************************/

extern void  __real_pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8) ;
extern void  __wrap_pshmem_team_alltoallv_packed(void * a1, size_t a2, size_t * a3, const void * a4, size_t * a5, size_t * a6, shmem_team_t a7, long * a8)  {

  TAU_PROFILE_TIMER(t,"void pshmem_team_alltoallv_packed(void *, size_t, size_t *, const void *, size_t *, size_t *, shmem_team_t, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_pshmem_team_alltoallv_packed(a1, a2, a3, a4, a5, a6, a7, a8);
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
   shmem_team_split
 **********************************************************/

extern void  __real_shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __wrap_shmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_split(shmem_team_t, int, int, shmem_team_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_split(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   pshmem_team_split
 **********************************************************/

extern void  __real_pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __wrap_pshmem_team_split(shmem_team_t a1, int a2, int a3, shmem_team_t * a4)  {

  TAU_PROFILE_TIMER(t,"void pshmem_team_split(shmem_team_t, int, int, shmem_team_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_pshmem_team_split(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_team_create_strided
 **********************************************************/

extern void  __real_shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __wrap_shmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_create_strided(int, int, int, shmem_team_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_create_strided(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   pshmem_team_create_strided
 **********************************************************/

extern void  __real_pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4) ;
extern void  __wrap_pshmem_team_create_strided(int a1, int a2, int a3, shmem_team_t * a4)  {

  TAU_PROFILE_TIMER(t,"void pshmem_team_create_strided(int, int, int, shmem_team_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_pshmem_team_create_strided(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_team_free
 **********************************************************/

extern void  __real_shmem_team_free(shmem_team_t * a1) ;
extern void  __wrap_shmem_team_free(shmem_team_t * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_team_free(shmem_team_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_team_free(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_team_npes
 **********************************************************/

extern int  __real_shmem_team_npes(shmem_team_t a1) ;
extern int  __wrap_shmem_team_npes(shmem_team_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_team_npes(shmem_team_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_team_npes(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_team_mype
 **********************************************************/

extern int  __real_shmem_team_mype(shmem_team_t a1) ;
extern int  __wrap_shmem_team_mype(shmem_team_t a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_team_mype(shmem_team_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_team_mype(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_team_translate_pe
 **********************************************************/

extern int  __real_shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) ;
extern int  __wrap_shmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_team_translate_pe(shmem_team_t, int, shmem_team_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_team_translate_pe(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   pshmem_team_translate_pe
 **********************************************************/

extern int  __real_pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3) ;
extern int  __wrap_pshmem_team_translate_pe(shmem_team_t a1, int a2, shmem_team_t a3)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int pshmem_team_translate_pe(shmem_team_t, int, shmem_team_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_pshmem_team_translate_pe(a1, a2, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_init
 **********************************************************/

extern void  __real_shmem_init() ;
extern void  __wrap_shmem_init()  {

  TAU_PROFILE_TIMER(t,"void shmem_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_init();
  tau_totalnodes(1,__real_shmem_n_pes());
  TAU_PROFILE_SET_NODE(__real_shmem_my_pe());
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

extern int  __real_shmem_addr_accessible(void * a1, int a2) ;
extern int  __wrap_shmem_addr_accessible(void * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_addr_accessible(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_addr_accessible(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_init_thread
 **********************************************************/

extern int  __real_shmem_init_thread(int a1) ;
extern int  __wrap_shmem_init_thread(int a1)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_init_thread(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_init_thread(a1);
  tau_totalnodes(1,__real_shmem_n_pes());
  TAU_PROFILE_SET_NODE(__real_shmem_my_pe());
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_query_thread
 **********************************************************/

extern int  __real_shmem_query_thread() ;
extern int  __wrap_shmem_query_thread()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_query_thread(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_query_thread();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_thread_register
 **********************************************************/

extern void  __real_shmem_thread_register() ;
extern void  __wrap_shmem_thread_register()  {

  TAU_PROFILE_TIMER(t,"void shmem_thread_register(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_thread_register();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_thread_unregister
 **********************************************************/

extern void  __real_shmem_thread_unregister() ;
extern void  __wrap_shmem_thread_unregister()  {

  TAU_PROFILE_TIMER(t,"void shmem_thread_unregister(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_thread_unregister();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_thread_fence
 **********************************************************/

extern void  __real_shmem_thread_fence() ;
extern void  __wrap_shmem_thread_fence()  {

  TAU_PROFILE_TIMER(t,"void shmem_thread_fence(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_thread_fence();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_thread_quiet
 **********************************************************/

extern void  __real_shmem_thread_quiet() ;
extern void  __wrap_shmem_thread_quiet()  {

  TAU_PROFILE_TIMER(t,"void shmem_thread_quiet(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_thread_quiet();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_local_npes
 **********************************************************/

extern int  __real_shmem_local_npes() ;
extern int  __wrap_shmem_local_npes()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_local_npes(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_local_npes();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_local_pes
 **********************************************************/

extern void  __real_shmem_local_pes(int * a1, int a2) ;
extern void  __wrap_shmem_local_pes(int * a1, int a2)  {

  TAU_PROFILE_TIMER(t,"void shmem_local_pes(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_local_pes(a1, a2);
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

