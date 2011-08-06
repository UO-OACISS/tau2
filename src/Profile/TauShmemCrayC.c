#include <shmem.h>
#include <Profile/Profiler.h>
int TAUDECL tau_totalnodes(int set_or_get, int value);
int tau_shmem_tagid=0 ; 
#define TAU_SHMEM_TAGID tau_shmem_tagid
#define TAU_SHMEM_TAGID_NEXT (++tau_shmem_tagid) % 256 

/**********************************************************
   shmem_get16
 **********************************************************/

void shmem_get16(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get16(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2*a3, a4);
   pshmem_get16(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 2*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_get32
 **********************************************************/

void shmem_get32(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_get32(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a3, a4);
   pshmem_get32(a1, a2, a3, a4);
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
   pshmem_get64(a1, a2, a3, a4);
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
   pshmem_get128(a1, a2, a3, a4);
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
   pshmem_getmem(a1, a2, a3, a4);
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
   pshmem_short_get(a1, a2, a3, a4);
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
   pshmem_int_get(a1, a2, a3, a4);
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
   pshmem_long_get(a1, a2, a3, a4);
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
   pshmem_longlong_get(a1, a2, a3, a4);
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
   pshmem_float_get(a1, a2, a3, a4);
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
   pshmem_double_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put16
 **********************************************************/

void shmem_put16(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put16(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 2*a3);
   pshmem_put16(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put32
 **********************************************************/

void shmem_put32(void * a1, const void * a2, size_t a3, int a4) {

  TAU_PROFILE_TIMER(t,"void shmem_put32(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
   pshmem_put32(a1, a2, a3, a4);
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
   pshmem_put64(a1, a2, a3, a4);
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
   pshmem_put128(a1, a2, a3, a4);
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
   pshmem_putmem(a1, a2, a3, a4);
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
   pshmem_short_put(a1, a2, a3, a4);
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
   pshmem_int_put(a1, a2, a3, a4);
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
   pshmem_long_put(a1, a2, a3, a4);
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
   pshmem_longlong_put(a1, a2, a3, a4);
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
   pshmem_float_put(a1, a2, a3, a4);
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
   pshmem_double_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put16_nb
 **********************************************************/

void shmem_put16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put16_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 2*a3);
   pshmem_put16_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put32_nb
 **********************************************************/

void shmem_put32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put32_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
   pshmem_put32_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put64_nb
 **********************************************************/

void shmem_put64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put64_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
   pshmem_put64_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_put128_nb
 **********************************************************/

void shmem_put128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_put128_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 16*a3);
   pshmem_put128_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_putmem_nb
 **********************************************************/

void shmem_putmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, a3);
   pshmem_putmem_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_put_nb
 **********************************************************/

void shmem_short_put_nb(short * a1, const short * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_short_put_nb(short *, const short *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*a3);
   pshmem_short_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_put_nb
 **********************************************************/

void shmem_int_put_nb(int * a1, const int * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_int_put_nb(int *, const int *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*a3);
   pshmem_int_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_put_nb
 **********************************************************/

void shmem_long_put_nb(long * a1, const long * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_long_put_nb(long *, const long *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*a3);
   pshmem_long_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_put_nb
 **********************************************************/

void shmem_longlong_put_nb(long long * a1, const long long * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put_nb(long long *, const long long *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*a3);
   pshmem_longlong_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_put_nb
 **********************************************************/

void shmem_float_put_nb(float * a1, const float * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_float_put_nb(float *, const float *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
   pshmem_float_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_put_nb
 **********************************************************/

void shmem_double_put_nb(double * a1, const double * a2, size_t a3, int a4, void ** a5) {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_nb(double *, const double *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*a3);
   pshmem_double_put_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_iget
 **********************************************************/

void shmem_short_iget(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_short_iget(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a5, a6);
   pshmem_short_iget(a1, a2, a3, a4, a5, a6);
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
   pshmem_int_iget(a1, a2, a3, a4, a5, a6);
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
   pshmem_long_iget(a1, a2, a3, a4, a5, a6);
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
   pshmem_longlong_iget(a1, a2, a3, a4, a5, a6);
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
   pshmem_float_iget(a1, a2, a3, a4, a5, a6);
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
   pshmem_double_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(double)*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget16
 **********************************************************/

void shmem_iget16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2*a5, a6);
   pshmem_iget16(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 2*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iget32
 **********************************************************/

void shmem_iget32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iget32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a5, a6);
   pshmem_iget32(a1, a2, a3, a4, a5, a6);
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
   pshmem_iget64(a1, a2, a3, a4, a5, a6);
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
   pshmem_iget128(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 16*a5);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_iput
 **********************************************************/

void shmem_short_iput(short * a1, const short * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_short_iput(short *, const short *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(short)*a5);
   pshmem_short_iput(a1, a2, a3, a4, a5, a6);
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
   pshmem_int_iput(a1, a2, a3, a4, a5, a6);
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
   pshmem_long_iput(a1, a2, a3, a4, a5, a6);
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
   pshmem_longlong_iput(a1, a2, a3, a4, a5, a6);
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
   pshmem_float_iput(a1, a2, a3, a4, a5, a6);
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
   pshmem_double_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput16
 **********************************************************/

void shmem_iput16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 2*a5);
   pshmem_iput16(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a5, a6);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_iput32
 **********************************************************/

void shmem_iput32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) {

  TAU_PROFILE_TIMER(t,"void shmem_iput32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 4*a5);
   pshmem_iput32(a1, a2, a3, a4, a5, a6);
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
   pshmem_iput64(a1, a2, a3, a4, a5, a6);
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
   pshmem_iput128(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a5, a6);
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
  retval  =   pshmem_char_g(a1, a2);
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
  retval  =   pshmem_short_g(a1, a2);
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
  retval  =   pshmem_int_g(a1, a2);
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
  retval  =   pshmem_long_g(a1, a2);
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
  retval  =   pshmem_longlong_g(a1, a2);
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
  retval  =   pshmem_float_g(a1, a2);
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
  retval  =   pshmem_double_g(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(double)*1);
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
   pshmem_char_p(a1, a2, a3);
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
   pshmem_short_p(a1, a2, a3);
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
   pshmem_int_p(a1, a2, a3);
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
   pshmem_long_p(a1, a2, a3);
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
   pshmem_longlong_p(a1, a2, a3);
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
   pshmem_float_p(a1, a2, a3);
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
   pshmem_double_p(a1, a2, a3);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_swap
 **********************************************************/

short shmem_short_swap(short * a1, short a2, int a3) {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_swap(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a3);
  retval  =   pshmem_short_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
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
  retval  =   pshmem_int_swap(a1, a2, a3);
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
  retval  =   pshmem_long_swap(a1, a2, a3);
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
  retval  =   pshmem_longlong_swap(a1, a2, a3);
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
  retval  =   pshmem_float_swap(a1, a2, a3);
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
  retval  =   pshmem_double_swap(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(double)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(double)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_cswap
 **********************************************************/

short shmem_short_cswap(short * a1, short a2, short a3, int a4) {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_cswap(short *, short, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a4);
  retval  =   pshmem_short_cswap(a1, a2, a3, a4);
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

int shmem_int_cswap(int * a1, int a2, int a3, int a4) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_cswap(int *, int, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*1, a4);
  retval  =   pshmem_int_cswap(a1, a2, a3, a4);
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
  retval  =   pshmem_long_cswap(a1, a2, a3, a4);
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
  retval  =   pshmem_longlong_cswap(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*1);
  if (retval == a2) { 
    TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*1);
    TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a4);
  }
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_finc
 **********************************************************/

short shmem_short_finc(short * a1, int a2) {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_finc(short *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a2);
  retval  =   pshmem_short_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a2);
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
  retval  =   pshmem_int_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_long_finc
 **********************************************************/

long shmem_long_finc(long * a1, int a2) {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_finc(long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*1, a2);
  retval  =   pshmem_long_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_longlong_finc
 **********************************************************/

long long shmem_longlong_finc(long long * a1, int a2) {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_finc(long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*1, a2);
  retval  =   pshmem_longlong_finc(a1, a2);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a2, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a2, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_short_fadd
 **********************************************************/

short shmem_short_fadd(short * a1, short a2, int a3) {

  short retval;
  TAU_PROFILE_TIMER(t,"short shmem_short_fadd(short *, short, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*1, a3);
  retval  =   pshmem_short_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(short)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(short)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*1, a3);
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
  retval  =   pshmem_int_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(int)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(int)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*1, a3);
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
  retval  =   pshmem_long_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*1, a3);
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
  retval  =   pshmem_longlong_fadd(a1, a2, a3);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a3, sizeof(long long)*1);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a3, sizeof(long long)*1);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*1, a3);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_barrier_all
 **********************************************************/

void shmem_barrier_all() {

  TAU_PROFILE_TIMER(t,"void shmem_barrier_all() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier_all();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_barrier
 **********************************************************/

void shmem_barrier(int a1, int a2, int a3, long * a4) {

  TAU_PROFILE_TIMER(t,"void shmem_barrier(int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_barrier(a1, a2, a3, a4);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_quiet
 **********************************************************/

void shmem_quiet() {

  TAU_PROFILE_TIMER(t,"void shmem_quiet() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_quiet();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_lock
 **********************************************************/

void shmem_set_lock(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_set_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_clear_lock
 **********************************************************/

void shmem_clear_lock(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_clear_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_lock(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_test_lock
 **********************************************************/

int shmem_test_lock(long * a1) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_lock(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_test_lock(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_clear_event
 **********************************************************/

void shmem_clear_event(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_clear_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_clear_event(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_set_event
 **********************************************************/

void shmem_set_event(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_set_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_set_event(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_test_event
 **********************************************************/

int shmem_test_event(long * a1) {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_test_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_test_event(a1);
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_wait_event
 **********************************************************/

void shmem_wait_event(long * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_wait_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_wait_event(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_wait
 **********************************************************/

void shmem_short_wait(short * a1, short a2) {

  TAU_PROFILE_TIMER(t,"void shmem_short_wait(short *, short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_wait
 **********************************************************/

void shmem_int_wait(int * a1, int a2) {

  TAU_PROFILE_TIMER(t,"void shmem_int_wait(int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_wait
 **********************************************************/

void shmem_long_wait(long * a1, long a2) {

  TAU_PROFILE_TIMER(t,"void shmem_long_wait(long *, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_wait
 **********************************************************/

void shmem_longlong_wait(long long * a1, long long a2) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_wait(long long *, long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_wait(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_wait_until
 **********************************************************/

void shmem_short_wait_until(short * a1, int a2, short a3) {

  TAU_PROFILE_TIMER(t,"void shmem_short_wait_until(short *, int, short) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_wait_until
 **********************************************************/

void shmem_int_wait_until(int * a1, int a2, int a3) {

  TAU_PROFILE_TIMER(t,"void shmem_int_wait_until(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_wait_until
 **********************************************************/

void shmem_long_wait_until(long * a1, int a2, long a3) {

  TAU_PROFILE_TIMER(t,"void shmem_long_wait_until(long *, int, long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_wait_until
 **********************************************************/

void shmem_longlong_wait_until(long long * a1, int a2, long long a3) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_wait_until(long long *, int, long long) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_wait_until(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_sum_to_all
 **********************************************************/

void shmem_short_sum_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_sum_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_max_to_all
 **********************************************************/

void shmem_short_max_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_max_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_min_to_all
 **********************************************************/

void shmem_short_min_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_min_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_prod_to_all
 **********************************************************/

void shmem_short_prod_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_prod_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_and_to_all
 **********************************************************/

void shmem_short_and_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_and_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_or_to_all
 **********************************************************/

void shmem_short_or_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_or_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_short_xor_to_all
 **********************************************************/

void shmem_short_xor_to_all(short * a1, const short * a2, size_t a3, int a4, int a5, int a6, short * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_short_xor_to_all(short *, const short *, size_t, int, int, int, short *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_short_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_sum_to_all
 **********************************************************/

void shmem_int_sum_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_sum_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_max_to_all
 **********************************************************/

void shmem_int_max_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_max_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_min_to_all
 **********************************************************/

void shmem_int_min_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_min_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_prod_to_all
 **********************************************************/

void shmem_int_prod_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_prod_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_and_to_all
 **********************************************************/

void shmem_int_and_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_and_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_or_to_all
 **********************************************************/

void shmem_int_or_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_or_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_int_xor_to_all
 **********************************************************/

void shmem_int_xor_to_all(int * a1, const int * a2, size_t a3, int a4, int a5, int a6, int * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_int_xor_to_all(int *, const int *, size_t, int, int, int, int *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_int_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_sum_to_all
 **********************************************************/

void shmem_long_sum_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_sum_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_max_to_all
 **********************************************************/

void shmem_long_max_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_max_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_min_to_all
 **********************************************************/

void shmem_long_min_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_min_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_prod_to_all
 **********************************************************/

void shmem_long_prod_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_prod_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_and_to_all
 **********************************************************/

void shmem_long_and_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_and_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_or_to_all
 **********************************************************/

void shmem_long_or_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_or_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_long_xor_to_all
 **********************************************************/

void shmem_long_xor_to_all(long * a1, const long * a2, size_t a3, int a4, int a5, int a6, long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_long_xor_to_all(long *, const long *, size_t, int, int, int, long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_long_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_sum_to_all
 **********************************************************/

void shmem_longlong_sum_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_sum_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_max_to_all
 **********************************************************/

void shmem_longlong_max_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_max_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_min_to_all
 **********************************************************/

void shmem_longlong_min_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_min_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_prod_to_all
 **********************************************************/

void shmem_longlong_prod_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_prod_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_and_to_all
 **********************************************************/

void shmem_longlong_and_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_and_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_and_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_or_to_all
 **********************************************************/

void shmem_longlong_or_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_or_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_or_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_longlong_xor_to_all
 **********************************************************/

void shmem_longlong_xor_to_all(long long * a1, const long long * a2, size_t a3, int a4, int a5, int a6, long long * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_xor_to_all(long long *, const long long *, size_t, int, int, int, long long *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_longlong_xor_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_sum_to_all
 **********************************************************/

void shmem_float_sum_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_sum_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_float_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_max_to_all
 **********************************************************/

void shmem_float_max_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_max_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_float_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_min_to_all
 **********************************************************/

void shmem_float_min_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_min_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_float_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_float_prod_to_all
 **********************************************************/

void shmem_float_prod_to_all(float * a1, const float * a2, size_t a3, int a4, int a5, int a6, float * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_float_prod_to_all(float *, const float *, size_t, int, int, int, float *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_float_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_sum_to_all
 **********************************************************/

void shmem_double_sum_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_sum_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_double_sum_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_max_to_all
 **********************************************************/

void shmem_double_max_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_max_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_double_max_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_min_to_all
 **********************************************************/

void shmem_double_min_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_min_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_double_min_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_double_prod_to_all
 **********************************************************/

void shmem_double_prod_to_all(double * a1, const double * a2, size_t a3, int a4, int a5, int a6, double * a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_double_prod_to_all(double *, const double *, size_t, int, int, int, double *, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_double_prod_to_all(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast32
 **********************************************************/

void shmem_broadcast32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast32(void *, const void *, size_t, int, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast32(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_broadcast64
 **********************************************************/

void shmem_broadcast64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, int a7, long * a8) {

  TAU_PROFILE_TIMER(t,"void shmem_broadcast64(void *, const void *, size_t, int, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_broadcast64(a1, a2, a3, a4, a5, a6, a7, a8);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   start_pes
 **********************************************************/

void start_pes(int a1) {

  TAU_PROFILE_TIMER(t,"void start_pes(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pstart_pes(a1);
  tau_totalnodes(1,pshmem_n_pes());
  TAU_PROFILE_SET_NODE(pshmem_my_pe());
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_init
 **********************************************************/

void shmem_init() {

  TAU_PROFILE_TIMER(t,"void shmem_init() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_init();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_finalize
 **********************************************************/

void shmem_finalize() {

  TAU_PROFILE_TIMER(t,"void shmem_finalize() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_finalize();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   num_pes
 **********************************************************/

int num_pes() {

  int retval;
  TAU_PROFILE_TIMER(t,"int num_pes() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pnum_pes();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_n_pes
 **********************************************************/

int shmem_n_pes() {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_n_pes() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pshmem_n_pes();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   my_pe
 **********************************************************/

int my_pe() {

  int retval;
  TAU_PROFILE_TIMER(t,"int my_pe() C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =   pmy_pe();
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
  retval  =   pshmem_my_pe();
  TAU_PROFILE_STOP(t);
  return retval;

}


/**********************************************************
   shmem_fence
 **********************************************************/

void shmem_fence() {

  TAU_PROFILE_TIMER(t,"void shmem_fence() C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_fence();
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shfree
 **********************************************************/

void shfree(void * a1) {

  TAU_PROFILE_TIMER(t,"void shfree(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshfree(a1);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_init_thread
 **********************************************************/

void shmem_init_thread(int a1, int * a2) {

  TAU_PROFILE_TIMER(t,"void shmem_init_thread(int, int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_init_thread(a1, a2);
  TAU_PROFILE_STOP(t);

}


/**********************************************************
   shmem_query_thread
 **********************************************************/

void shmem_query_thread(int * a1) {

  TAU_PROFILE_TIMER(t,"void shmem_query_thread(int *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
   pshmem_query_thread(a1);
  TAU_PROFILE_STOP(t);

}

