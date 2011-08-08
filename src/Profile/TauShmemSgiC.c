#include <shmem.h>
#include <Profile/Profiler.h>
int TAUDECL tau_totalnodes(int set_or_get, int value);
int tau_shmem_tagid=0 ; 
#define TAU_SHMEM_TAGID tau_shmem_tagid
#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid) % 256 

/**********************************************************
   shmem_get32
 **********************************************************/

void shmem_get32(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get32(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a3, a4);
   _shmem_get32(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 4*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get64
 **********************************************************/

void shmem_get64(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get64(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
   _shmem_get64(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get128
 **********************************************************/

void shmem_get128(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get128(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a3, a4);
   _shmem_get128(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 16*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_getmem
 **********************************************************/

void shmem_getmem(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_getmem(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), a3, a4);
   _shmem_getmem(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_get
 **********************************************************/

void shmem_short_get(short * a1, const short * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_short_get(short *, const short *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a3, a4);
   _shmem_short_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_get
 **********************************************************/

void shmem_int_get(int * a1, const int * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_int_get(int *, const int *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a3, a4);
   _shmem_int_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_get
 **********************************************************/

void shmem_long_get(long * a1, const long * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_long_get(long *, const long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a3, a4);
   _shmem_long_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_get
 **********************************************************/

void shmem_longlong_get(long long * a1, const long long * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_get(long long *, const long long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a3, a4);
   _shmem_longlong_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_get
 **********************************************************/

void shmem_float_get(float * a1, const float * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_float_get(float *, const float *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
   _shmem_float_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_get
 **********************************************************/

void shmem_double_get(double * a1, const double * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_double_get(double *, const double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a3, a4);
   _shmem_double_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_get
 **********************************************************/

void shmem_longdouble_get(long double * a1, const long double * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_get(long double *, const long double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*a3, a4);
   _shmem_longdouble_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put32
 **********************************************************/

void shmem_put32(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put32(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
   _shmem_put32(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put64
 **********************************************************/

void shmem_put64(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put64(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
   _shmem_put64(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put128
 **********************************************************/

void shmem_put128(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put128(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 16*a3);
   _shmem_put128(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_putmem
 **********************************************************/

void shmem_putmem(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_putmem(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, a3);
   _shmem_putmem(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_put
 **********************************************************/

void shmem_short_put(short * a1, const short * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_short_put(short *, const short *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*a3);
   _shmem_short_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_put
 **********************************************************/

void shmem_int_put(int * a1, const int * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_int_put(int *, const int *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*a3);
   _shmem_int_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_put
 **********************************************************/

void shmem_long_put(long * a1, const long * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_long_put(long *, const long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*a3);
   _shmem_long_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_put
 **********************************************************/

void shmem_longlong_put(long long * a1, const long long * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put(long long *, const long long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*a3);
   _shmem_longlong_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_put
 **********************************************************/

void shmem_float_put(float * a1, const float * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_float_put(float *, const float *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
   _shmem_float_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put
 **********************************************************/

void shmem_double_put(double * a1, const double * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_double_put(double *, const double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*a3);
   _shmem_double_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_put
 **********************************************************/

void shmem_longdouble_put(long double * a1, const long double * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_put(long double *, const long double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long double)*a3);
   _shmem_longdouble_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget32
 **********************************************************/

void shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a5, a6);
   _shmem_iget32(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 4*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget64
 **********************************************************/

void shmem_iget64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a5, a6);
   _shmem_iget64(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 8*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget128
 **********************************************************/

void shmem_iget128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a5, a6);
   _shmem_iget128(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 16*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_iget
 **********************************************************/

void shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_short_iget(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a5, a6);
   _shmem_short_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(short)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_iget
 **********************************************************/

void shmem_int_iget(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_int_iget(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a5, a6);
   _shmem_int_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(int)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_iget
 **********************************************************/

void shmem_long_iget(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_long_iget(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a5, a6);
   _shmem_long_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(long)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_iget
 **********************************************************/

void shmem_longlong_iget(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_iget(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a5, a6);
   _shmem_longlong_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(long long)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_iget
 **********************************************************/

void shmem_float_iget(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_float_iget(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a5, a6);
   _shmem_float_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(float)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_iget
 **********************************************************/

void shmem_double_iget(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_double_iget(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a5, a6);
   _shmem_double_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(double)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_iget
 **********************************************************/

void shmem_longdouble_iget(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_iget(long double *, const long double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*a5, a6);
   _shmem_longdouble_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(long double)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput32
 **********************************************************/

void shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 4*a5);
   _shmem_iput32(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput64
 **********************************************************/

void shmem_iput64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 8*a5);
   _shmem_iput64(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput128
 **********************************************************/

void shmem_iput128(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput128(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 16*a5);
   _shmem_iput128(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_iput
 **********************************************************/

void shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_short_iput(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(short)*a5);
   _shmem_short_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_iput
 **********************************************************/

void shmem_int_iput(int * a1, const int * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_int_iput(int *, const int *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(int)*a5);
   _shmem_int_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_iput
 **********************************************************/

void shmem_long_iput(long * a1, const long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_long_iput(long *, const long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long)*a5);
   _shmem_long_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_iput
 **********************************************************/

void shmem_longlong_iput(long long * a1, const long long * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_iput(long long *, const long long *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long long)*a5);
   _shmem_longlong_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_iput
 **********************************************************/

void shmem_float_iput(float * a1, const float * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_float_iput(float *, const float *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(float)*a5);
   _shmem_float_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_iput
 **********************************************************/

void shmem_double_iput(double * a1, const double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_double_iput(double *, const double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(double)*a5);
   _shmem_double_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_iput
 **********************************************************/

void shmem_longdouble_iput(long double * a1, const long double * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_iput(long double *, const long double *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(long double)*a5);
   _shmem_longdouble_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_char_g
 **********************************************************/

char shmem_char_g(const char * a1, int a2) {

  char retval;
  TAU_PROFILE_TIMER(t,"char shmem_char_g(const char *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)*1, a2);
  retval  =   _shmem_char_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(char)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_g
 **********************************************************/

short shmem_short_g(const short * a1, int a2) {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_g(const short *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a2);
  retval  =   _shmem_short_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(short)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_g
 **********************************************************/

int shmem_int_g(const int * a1, int a2) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_g(const int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a2);
  retval  =   _shmem_int_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(int)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_g
 **********************************************************/

long shmem_long_g(const long * a1, int a2) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_g(const long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a2);
  retval  =   _shmem_long_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_g
 **********************************************************/

long long shmem_longlong_g(const long long * a1, int a2) {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_g(const long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a2);
  retval  =   _shmem_longlong_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long long)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_float_g
 **********************************************************/

float shmem_float_g(const float * a1, int a2) {

  float retval;
  TAU_PROFILE_TIMER(t,"float shmem_float_g(const float *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*1, a2);
  retval  =   _shmem_float_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(float)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_double_g
 **********************************************************/

double shmem_double_g(const double * a1, int a2) {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_double_g(const double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*1, a2);
  retval  =   _shmem_double_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(double)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longdouble_g
 **********************************************************/

long double shmem_longdouble_g(const long double * a1, int a2) {

  long double retval;
  TAU_PROFILE_TIMER(t,"long double shmem_longdouble_g(const long double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long double)*1, a2);
  retval  =   _shmem_longdouble_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long double)*1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_char_p
 **********************************************************/

void shmem_char_p(char * a1, char a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_char_p(char *, char, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(char)*1);
   _shmem_char_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_p
 **********************************************************/

void shmem_short_p(short * a1, short a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_short_p(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
   _shmem_short_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_p
 **********************************************************/

void shmem_int_p(int * a1, int a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_int_p(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
   _shmem_int_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_p
 **********************************************************/

void shmem_long_p(long * a1, long a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_long_p(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
   _shmem_long_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_p
 **********************************************************/

void shmem_longlong_p(long long * a1, long long a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_p(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
   _shmem_longlong_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_p
 **********************************************************/

void shmem_float_p(float * a1, float a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_float_p(float *, float, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(float)*1);
   _shmem_float_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_p
 **********************************************************/

void shmem_double_p(double * a1, double a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_double_p(double *, double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(double)*1);
   _shmem_double_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_p
 **********************************************************/

void shmem_longdouble_p(long double * a1, long double a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_p(long double *, long double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long double)*1);
   _shmem_longdouble_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long double)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_swap
 **********************************************************/

long shmem_swap(long * a1, long a2, int a3) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_swap(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 1, a3);
  retval  =   _shmem_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, 1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, 1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_swap
 **********************************************************/

int shmem_int_swap(int * a1, int a2, int a3) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_swap(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a3);
  retval  =   _shmem_int_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_swap
 **********************************************************/

long shmem_long_swap(long * a1, long a2, int a3) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_swap(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a3);
  retval  =   _shmem_long_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_swap
 **********************************************************/

long long shmem_longlong_swap(long long * a1, long long a2, int a3) {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_swap(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a3);
  retval  =   _shmem_longlong_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_float_swap
 **********************************************************/

float shmem_float_swap(float * a1, float a2, int a3) {

  float retval;
  TAU_PROFILE_TIMER(t,"float shmem_float_swap(float *, float, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*1, a3);
  retval  =   _shmem_float_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(float)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(float)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_double_swap
 **********************************************************/

double shmem_double_swap(double * a1, double a2, int a3) {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_double_swap(double *, double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*1, a3);
  retval  =   _shmem_double_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(double)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(double)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_cswap
 **********************************************************/

int shmem_int_cswap(int * a1, int a2, int a3, int a4) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_cswap(int *, int, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a4);
  retval  =   _shmem_int_cswap(a1, a2, a3, a4);
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

long shmem_long_cswap(long * a1, long a2, long a3, int a4) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_cswap(long *, long, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a4);
  retval  =   _shmem_long_cswap(a1, a2, a3, a4);
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

long long shmem_longlong_cswap(long long * a1, long long a2, long long a3, int a4) {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_cswap(long long *, long long, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a4);
  retval  =   _shmem_longlong_cswap(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*1);
  if (retval == a2) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a4);
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_finc
 **********************************************************/

int shmem_int_finc(int * a1, int a2) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_finc(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a2);
  retval  =   _shmem_int_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_fadd
 **********************************************************/

int shmem_int_fadd(int * a1, int a2, int a3) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_fadd(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a3);
  retval  =   _shmem_int_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_int_inc
 **********************************************************/

void shmem_int_inc(int * a1, int a2) {

  TAU_PROFILE_TIMER(t,"void shmem_int_inc(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_inc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_add
 **********************************************************/

void shmem_int_add(int * a1, int a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_int_add(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_add(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_finc
 **********************************************************/

long shmem_long_finc(long * a1, int a2) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_finc(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a2);
  retval  =   _shmem_long_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_fadd
 **********************************************************/

long shmem_long_fadd(long * a1, long a2, int a3) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_fadd(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a3);
  retval  =   _shmem_long_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_inc
 **********************************************************/

void shmem_long_inc(long * a1, int a2) {

  TAU_PROFILE_TIMER(t,"void shmem_long_inc(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_inc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_add
 **********************************************************/

void shmem_long_add(long * a1, long a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_long_add(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_add(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_finc
 **********************************************************/

long long shmem_longlong_finc(long long * a1, int a2) {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_finc(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a2);
  retval  =   _shmem_longlong_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_fadd
 **********************************************************/

long long shmem_longlong_fadd(long long * a1, long long a2, int a3) {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_fadd(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a3);
  retval  =   _shmem_longlong_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_inc
 **********************************************************/

void shmem_longlong_inc(long long * a1, int a2) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_inc(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_inc(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_add
 **********************************************************/

void shmem_longlong_add(long long * a1, long long a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_add(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_add(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_ps
 **********************************************************/

void shmem_barrier_ps(int a1, int a2, int a3, long * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_ps(int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_barrier_ps(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier_all
 **********************************************************/

void shmem_barrier_all() {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_all() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_barrier_all();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier
 **********************************************************/

void shmem_barrier(int a1, int a2, int a3, long * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_barrier(int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_barrier(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_lock
 **********************************************************/

void shmem_set_lock(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_set_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_set_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_lock
 **********************************************************/

void shmem_clear_lock(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_clear_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_clear_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_test_lock
 **********************************************************/

int shmem_test_lock(long * a1) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_test_lock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_wait
 **********************************************************/

void shmem_short_wait(short * a1, short a2) {

  TAU_PROFILE_TIMER(t,"void shmem_short_wait(short *, short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_wait
 **********************************************************/

void shmem_int_wait(int * a1, int a2) {

  TAU_PROFILE_TIMER(t,"void shmem_int_wait(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_wait
 **********************************************************/

void shmem_long_wait(long * a1, long a2) {

  TAU_PROFILE_TIMER(t,"void shmem_long_wait(long *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_wait
 **********************************************************/

void shmem_longlong_wait(long long * a1, long long a2) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_wait(long long *, long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait
 **********************************************************/

void shmem_wait(long * a1, long a2) {

  TAU_PROFILE_TIMER(t,"void shmem_wait(long *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_wait_until
 **********************************************************/

void shmem_short_wait_until(short * a1, int a2, short a3) {

  TAU_PROFILE_TIMER(t,"void shmem_short_wait_until(short *, int, short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_wait_until
 **********************************************************/

void shmem_int_wait_until(int * a1, int a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_int_wait_until(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_wait_until
 **********************************************************/

void shmem_long_wait_until(long * a1, int a2, long a3) {

  TAU_PROFILE_TIMER(t,"void shmem_long_wait_until(long *, int, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_wait_until
 **********************************************************/

void shmem_longlong_wait_until(long long * a1, int a2, long long a3) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_wait_until(long long *, int, long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_wait_until
 **********************************************************/

void shmem_wait_until(long * a1, int a2, long a3) {

  TAU_PROFILE_TIMER(t,"void shmem_wait_until(long *, int, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_sum_to_all
 **********************************************************/

void shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_sum_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

void shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_max_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

void shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_min_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

void shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_prod_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

void shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_and_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

void shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_or_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

void shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_xor_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

void shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_sum_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

void shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_max_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

void shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_min_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

void shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_prod_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

void shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_and_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

void shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_or_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

void shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_xor_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

void shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_sum_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

void shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_max_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

void shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_min_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

void shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_prod_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

void shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_and_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

void shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_or_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

void shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_xor_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

void shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_sum_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

void shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_max_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

void shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_min_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

void shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_prod_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

void shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_and_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

void shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_or_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

void shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_xor_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

void shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_sum_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

void shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_max_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

void shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_min_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

void shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_prod_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

void shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_sum_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

void shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_max_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

void shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_min_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

void shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_prod_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_sum_to_all
 **********************************************************/

void shmem_longdouble_sum_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_sum_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longdouble_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_max_to_all
 **********************************************************/

void shmem_longdouble_max_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_max_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longdouble_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_min_to_all
 **********************************************************/

void shmem_longdouble_min_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_min_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longdouble_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longdouble_prod_to_all
 **********************************************************/

void shmem_longdouble_prod_to_all(long double * a1, const long double * a2, size_t a3, int a4, int a5, int a6, long double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longdouble_prod_to_all(long double *, const long double *, size_t, int, int, int, long double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_longdouble_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect32
 **********************************************************/

void shmem_collect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {

  TAU_PROFILE_TIMER(t,"void shmem_collect32(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_collect32(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect32
 **********************************************************/

void shmem_fcollect32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect32(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_fcollect32(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_collect64
 **********************************************************/

void shmem_collect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {

  TAU_PROFILE_TIMER(t,"void shmem_collect64(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_collect64(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_fcollect64
 **********************************************************/

void shmem_fcollect64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) {

  TAU_PROFILE_TIMER(t,"void shmem_fcollect64(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_fcollect64(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast32
 **********************************************************/

void shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast32(void *, const void *, size_t, int, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_broadcast32(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast64
 **********************************************************/

void shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast64(void *, const void *, size_t, int, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_broadcast64(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   start_pes
 **********************************************************/

void start_pes(int a1) {

  TAU_PROFILE_TIMER(t,"void start_pes(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _start_pes(a1);
  tau_totalnodes(1,_shmem_n_pes());
  TAU_PROFILE_SET_NODE(_shmem_my_pe());
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_udcflush
 **********************************************************/

void shmem_udcflush() {

  TAU_PROFILE_TIMER(t,"void shmem_udcflush() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_udcflush();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_udcflush_line
 **********************************************************/

void shmem_udcflush_line(void * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_udcflush_line(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_udcflush_line(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_n_pes
 **********************************************************/

int shmem_n_pes() {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_n_pes() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_n_pes();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_my_pe
 **********************************************************/

int shmem_my_pe() {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_my_pe() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_my_pe();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_fence
 **********************************************************/

void shmem_fence() {

  TAU_PROFILE_TIMER(t,"void shmem_fence() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_fence();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_quiet
 **********************************************************/

void shmem_quiet() {

  TAU_PROFILE_TIMER(t,"void shmem_quiet() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_quiet();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_group_create_strided
 **********************************************************/

int shmem_group_create_strided(int a1, int a2, int a3, int * a4, int * a5) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_group_create_strided(int, int, int, int *, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_group_create_strided(a1, a2, a3, a4, a5);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_group_delete
 **********************************************************/

void shmem_group_delete(int a1) {

  TAU_PROFILE_TIMER(t,"void shmem_group_delete(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_group_delete(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_group_inquire
 **********************************************************/

void shmem_group_inquire(int a1, shmem_group_t * a2) {

  TAU_PROFILE_TIMER(t,"void shmem_group_inquire(int, shmem_group_t *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shmem_group_inquire(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmalloc
 **********************************************************/

void * shmalloc(size_t a1) {

  void * retval;
  TAU_PROFILE_TIMER(t,"void *shmalloc(size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmalloc(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shfree
 **********************************************************/

void shfree(void * a1) {

  TAU_PROFILE_TIMER(t,"void shfree(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   _shfree(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shrealloc
 **********************************************************/

void * shrealloc(void * a1, size_t a2) {

  void * retval;
  TAU_PROFILE_TIMER(t,"void *shrealloc(void *, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shrealloc(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmemalign
 **********************************************************/

void * shmemalign(size_t a1, size_t a2) {

  void * retval;
  TAU_PROFILE_TIMER(t,"void *shmemalign(size_t, size_t) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmemalign(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_ptr
 **********************************************************/

void * shmem_ptr(void * a1, int a2) {

  void * retval;
  TAU_PROFILE_TIMER(t,"void *shmem_ptr(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_ptr(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_pe_accessible
 **********************************************************/

int shmem_pe_accessible(int a1) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_pe_accessible(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_pe_accessible(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_addr_accessible
 **********************************************************/

int shmem_addr_accessible(void * a1, int a2) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_addr_accessible(void *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   _shmem_addr_accessible(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

