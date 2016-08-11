#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef SHMEM_FINT
#define SHMEM_FINT int
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
   shmem_get8
 **********************************************************/

extern void  __real_shmem_get8(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get8(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get8(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
  __real_shmem_get8(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get8_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get8__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET8_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET8__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_get16
 **********************************************************/

extern void  __real_shmem_get16(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get16(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get16(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2*a3, a4);
  __real_shmem_get16(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 2*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get16_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get16__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET16_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET16__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_get32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_get64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_get128_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get128__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET128_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET128__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_getmem_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_getmem__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GETMEM_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GETMEM__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_char_get
 **********************************************************/

extern void  __real_shmem_char_get(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_char_get(char * a1, const char * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_get(char *, const char *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)*a3, a4);
  __real_shmem_char_get(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(char)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_char_get_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_char_get__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_GET_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_GET__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_short_get_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_short_get__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_GET_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_GET__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_int_get_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_int_get__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_GET_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_GET__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_long_get_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_long_get__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_GET_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_GET__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_longlong_get_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_longlong_get__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_GET_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_GET__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_float_get_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float_get__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_GET_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_GET__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_double_get_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_double_get__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_GET_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_GET__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_float128_get_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float128_get__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_GET_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_GET__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put8
 **********************************************************/

extern void  __real_shmem_put8(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put8(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put8(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
  __real_shmem_put8(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put8_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put8__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT8_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT8__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put16
 **********************************************************/

extern void  __real_shmem_put16(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put16(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put16(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 2*a3);
  __real_shmem_put16(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put16_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put16__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT16_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT16__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_put32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_put64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_put128_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put128__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT128_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT128__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_putmem_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_putmem__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUTMEM_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUTMEM__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_char_put
 **********************************************************/

extern void  __real_shmem_char_put(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_char_put(char * a1, const char * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_put(char *, const char *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(char)*a3);
  __real_shmem_char_put(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_char_put_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_char_put__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_PUT_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_PUT__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_short_put_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_short_put__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_PUT_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_PUT__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_int_put_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_int_put__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_PUT_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_PUT__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_long_put_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_long_put__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_PUT_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_PUT__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_longlong_put_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_longlong_put__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_PUT_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_PUT__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_float_put_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float_put__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_PUT_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_PUT__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_double_put_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_double_put__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_PUT_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_PUT__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
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

extern void __wrap_shmem_float128_put_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float128_put__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_PUT_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_PUT__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put16_signal
 **********************************************************/

extern void  __real_shmem_put16_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_put16_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_put16_signal(void *, const void *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 2*a3);
  __real_shmem_put16_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put16_signal_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put16_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_put16_signal__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put16_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT16_SIGNAL_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put16_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT16_SIGNAL__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put16_signal(a1, a2, *a3, a4, *a5, *a6);
}


/**********************************************************
   shmem_put32_signal
 **********************************************************/

extern void  __real_shmem_put32_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_put32_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_put32_signal(void *, const void *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 4*a3);
  __real_shmem_put32_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put32_signal_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put32_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_put32_signal__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put32_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT32_SIGNAL_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put32_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT32_SIGNAL__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put32_signal(a1, a2, *a3, a4, *a5, *a6);
}


/**********************************************************
   shmem_put64_signal
 **********************************************************/

extern void  __real_shmem_put64_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_put64_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_put64_signal(void *, const void *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 8*a3);
  __real_shmem_put64_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put64_signal_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put64_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_put64_signal__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put64_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT64_SIGNAL_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put64_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT64_SIGNAL__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put64_signal(a1, a2, *a3, a4, *a5, *a6);
}


/**********************************************************
   shmem_put128_signal
 **********************************************************/

extern void  __real_shmem_put128_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_put128_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_put128_signal(void *, const void *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 16*a3);
  __real_shmem_put128_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put128_signal_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put128_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_put128_signal__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put128_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT128_SIGNAL_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put128_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUT128_SIGNAL__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_put128_signal(a1, a2, *a3, a4, *a5, *a6);
}


/**********************************************************
   shmem_putmem_signal
 **********************************************************/

extern void  __real_shmem_putmem_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6) ;
extern void  __wrap_shmem_putmem_signal(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_signal(void *, const void *, size_t, uint64_t *, uint64_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, a3);
  __real_shmem_putmem_signal(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_putmem_signal_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_putmem_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_putmem_signal__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_putmem_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUTMEM_SIGNAL_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_putmem_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_PUTMEM_SIGNAL__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_putmem_signal(a1, a2, *a3, a4, *a5, *a6);
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

extern void __wrap_shmem_short_put_signal_(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_short_put_signal__(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_SHORT_PUT_SIGNAL_(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_SHORT_PUT_SIGNAL__(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_put_signal(a1, a2, *a3, a4, *a5, *a6);
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

extern void __wrap_shmem_int_put_signal_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_int_put_signal__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_INT_PUT_SIGNAL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_INT_PUT_SIGNAL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_put_signal(a1, a2, *a3, a4, *a5, *a6);
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

extern void __wrap_shmem_long_put_signal_(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_long_put_signal__(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONG_PUT_SIGNAL_(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONG_PUT_SIGNAL__(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_put_signal(a1, a2, *a3, a4, *a5, *a6);
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

extern void __wrap_shmem_longlong_put_signal_(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_longlong_put_signal__(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONGLONG_PUT_SIGNAL_(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONGLONG_PUT_SIGNAL__(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_put_signal(a1, a2, *a3, a4, *a5, *a6);
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

extern void __wrap_shmem_float_put_signal_(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_float_put_signal__(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT_PUT_SIGNAL_(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT_PUT_SIGNAL__(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_put_signal(a1, a2, *a3, a4, *a5, *a6);
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

extern void __wrap_shmem_double_put_signal_(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_shmem_double_put_signal__(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_DOUBLE_PUT_SIGNAL_(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_put_signal(a1, a2, *a3, a4, *a5, *a6);
}

extern void __wrap_SHMEM_DOUBLE_PUT_SIGNAL__(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_put_signal(a1, a2, *a3, a4, *a5, *a6);
}


/**********************************************************
   shmem_get16_nb
 **********************************************************/

extern void  __real_shmem_get16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_get16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_get16_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2*a3, a4);
  __real_shmem_get16_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 2*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get16_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get16_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_get16_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get16_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET16_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get16_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET16_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get16_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_get32_nb
 **********************************************************/

extern void  __real_shmem_get32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_get32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_get32_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a3, a4);
  __real_shmem_get32_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 4*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get32_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get32_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_get32_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get32_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET32_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get32_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET32_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get32_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_get64_nb
 **********************************************************/

extern void  __real_shmem_get64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_get64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_get64_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
  __real_shmem_get64_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get64_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get64_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_get64_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get64_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET64_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get64_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET64_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get64_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_get128_nb
 **********************************************************/

extern void  __real_shmem_get128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_get128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_get128_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a3, a4);
  __real_shmem_get128_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 16*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get128_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get128_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_get128_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get128_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET128_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get128_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GET128_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_get128_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_getmem_nb
 **********************************************************/

extern void  __real_shmem_getmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_getmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_getmem_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), a3, a4);
  __real_shmem_getmem_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_getmem_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_getmem_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_getmem_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_getmem_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GETMEM_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_getmem_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_GETMEM_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_getmem_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_short_get_nb_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_short_get_nb__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_GET_NB_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_GET_NB__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_get_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_int_get_nb_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_int_get_nb__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_GET_NB_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_GET_NB__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_get_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_long_get_nb_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_long_get_nb__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_GET_NB_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_GET_NB__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_get_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_longlong_get_nb_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_longlong_get_nb__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_GET_NB_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_GET_NB__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_get_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_float_get_nb_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_float_get_nb__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT_GET_NB_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT_GET_NB__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_get_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_double_get_nb_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_double_get_nb__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_DOUBLE_GET_NB_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_DOUBLE_GET_NB__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_get_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_float128_get_nb_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_float128_get_nb__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT128_GET_NB_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT128_GET_NB__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_get_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_put16_nb
 **********************************************************/

extern void  __real_shmem_put16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_put16_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_put16_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 2*a3);
  __real_shmem_put16_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put16_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put16_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_put16_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put16_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT16_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put16_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT16_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put16_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_put32_nb
 **********************************************************/

extern void  __real_shmem_put32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_put32_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_put32_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
  __real_shmem_put32_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put32_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put32_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_put32_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put32_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT32_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put32_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT32_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put32_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_put64_nb
 **********************************************************/

extern void  __real_shmem_put64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_put64_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_put64_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
  __real_shmem_put64_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put64_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put64_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_put64_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put64_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT64_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put64_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT64_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put64_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_put128_nb
 **********************************************************/

extern void  __real_shmem_put128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_put128_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_put128_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 16*a3);
  __real_shmem_put128_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put128_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put128_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_put128_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put128_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT128_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put128_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUT128_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_put128_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_putmem_nb
 **********************************************************/

extern void  __real_shmem_putmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5) ;
extern void  __wrap_shmem_putmem_nb(void * a1, const void * a2, size_t a3, int a4, void ** a5)  {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_nb(void *, const void *, size_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, a3);
  __real_shmem_putmem_nb(a1, a2, a3, a4, a5);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_putmem_nb_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_putmem_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_putmem_nb__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_putmem_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUTMEM_NB_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_putmem_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_PUTMEM_NB__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_putmem_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_short_put_nb_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_short_put_nb__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_PUT_NB_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_PUT_NB__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_put_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_int_put_nb_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_int_put_nb__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_PUT_NB_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_PUT_NB__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_put_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_long_put_nb_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_long_put_nb__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_PUT_NB_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_PUT_NB__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_put_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_longlong_put_nb_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_longlong_put_nb__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_PUT_NB_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_PUT_NB__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_put_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_float_put_nb_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_float_put_nb__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT_PUT_NB_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT_PUT_NB__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_put_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_double_put_nb_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_double_put_nb__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_DOUBLE_PUT_NB_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_DOUBLE_PUT_NB__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_put_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_float128_put_nb_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_float128_put_nb__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT128_PUT_NB_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT128_PUT_NB__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float128_put_nb(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_get8_nbi
 **********************************************************/

extern void  __real_shmem_get8_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get8_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get8_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
  __real_shmem_get8_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get8_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get8_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET8_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET8_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get8_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_get16_nbi
 **********************************************************/

extern void  __real_shmem_get16_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get16_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get16_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2*a3, a4);
  __real_shmem_get16_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 2*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get16_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get16_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET16_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET16_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get16_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_get32_nbi
 **********************************************************/

extern void  __real_shmem_get32_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get32_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get32_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 4*a3, a4);
  __real_shmem_get32_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 4*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get32_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get32_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET32_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET32_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get32_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_get64_nbi
 **********************************************************/

extern void  __real_shmem_get64_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get64_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get64_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a3, a4);
  __real_shmem_get64_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 8*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get64_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get64_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET64_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET64_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get64_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_get128_nbi
 **********************************************************/

extern void  __real_shmem_get128_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_get128_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_get128_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 16*a3, a4);
  __real_shmem_get128_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, 16*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_get128_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_get128_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET128_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GET128_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_get128_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_getmem_nbi
 **********************************************************/

extern void  __real_shmem_getmem_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_getmem_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_getmem_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), a3, a4);
  __real_shmem_getmem_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_getmem_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_getmem_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GETMEM_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_GETMEM_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_getmem_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_char_get_nbi
 **********************************************************/

extern void  __real_shmem_char_get_nbi(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_char_get_nbi(char * a1, const char * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_get_nbi(char *, const char *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)*a3, a4);
  __real_shmem_char_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(char)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_char_get_nbi_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_char_get_nbi__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_GET_NBI_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_GET_NBI__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_short_get_nbi
 **********************************************************/

extern void  __real_shmem_short_get_nbi(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_short_get_nbi(short * a1, const short * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_get_nbi(short *, const short *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(short)*a3, a4);
  __real_shmem_short_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(short)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_short_get_nbi_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_short_get_nbi__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_GET_NBI_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_GET_NBI__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_int_get_nbi
 **********************************************************/

extern void  __real_shmem_int_get_nbi(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_int_get_nbi(int * a1, const int * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_get_nbi(int *, const int *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(int)*a3, a4);
  __real_shmem_int_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(int)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_int_get_nbi_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_int_get_nbi__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_GET_NBI_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_GET_NBI__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_long_get_nbi
 **********************************************************/

extern void  __real_shmem_long_get_nbi(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_long_get_nbi(long * a1, const long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_get_nbi(long *, const long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long)*a3, a4);
  __real_shmem_long_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_long_get_nbi_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_long_get_nbi__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_GET_NBI_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_GET_NBI__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_longlong_get_nbi
 **********************************************************/

extern void  __real_shmem_longlong_get_nbi(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_longlong_get_nbi(long long * a1, const long long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_get_nbi(long long *, const long long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(long long)*a3, a4);
  __real_shmem_longlong_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(long long)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_longlong_get_nbi_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_longlong_get_nbi__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_GET_NBI_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_GET_NBI__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_float_get_nbi
 **********************************************************/

extern void  __real_shmem_float_get_nbi(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float_get_nbi(float * a1, const float * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_get_nbi(float *, const float *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmem_float_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_float_get_nbi_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float_get_nbi__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_GET_NBI_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_GET_NBI__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_double_get_nbi
 **********************************************************/

extern void  __real_shmem_double_get_nbi(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_double_get_nbi(double * a1, const double * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_get_nbi(double *, const double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(double)*a3, a4);
  __real_shmem_double_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(double)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_double_get_nbi_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_double_get_nbi__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_GET_NBI_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_GET_NBI__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_float128_get_nbi
 **********************************************************/

extern void  __real_shmem_float128_get_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float128_get_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_get_nbi(__float128 *, const __float128 *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(float)*a3, a4);
  __real_shmem_float128_get_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a4, sizeof(float)*a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_float128_get_nbi_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float128_get_nbi__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_GET_NBI_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_GET_NBI__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_get_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put8_nbi
 **********************************************************/

extern void  __real_shmem_put8_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put8_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put8_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
  __real_shmem_put8_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put8_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put8_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT8_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT8_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put8_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put16_nbi
 **********************************************************/

extern void  __real_shmem_put16_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put16_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put16_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 2*a3);
  __real_shmem_put16_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put16_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put16_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT16_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT16_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put16_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put32_nbi
 **********************************************************/

extern void  __real_shmem_put32_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put32_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put32_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 4*a3);
  __real_shmem_put32_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put32_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put32_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT32_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT32_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put32_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put64_nbi
 **********************************************************/

extern void  __real_shmem_put64_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put64_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put64_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 8*a3);
  __real_shmem_put64_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put64_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put64_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT64_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT64_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put64_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put128_nbi
 **********************************************************/

extern void  __real_shmem_put128_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_put128_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_put128_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, 16*a3);
  __real_shmem_put128_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put128_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_put128_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT128_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUT128_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_put128_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_putmem_nbi
 **********************************************************/

extern void  __real_shmem_putmem_nbi(void * a1, const void * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_putmem_nbi(void * a1, const void * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_nbi(void *, const void *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, a3);
  __real_shmem_putmem_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_putmem_nbi_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_putmem_nbi__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUTMEM_NBI_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_PUTMEM_NBI__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_putmem_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_char_put_nbi
 **********************************************************/

extern void  __real_shmem_char_put_nbi(char * a1, const char * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_char_put_nbi(char * a1, const char * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_put_nbi(char *, const char *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(char)*a3);
  __real_shmem_char_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_char_put_nbi_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_char_put_nbi__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_PUT_NBI_(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_CHAR_PUT_NBI__(char * a1, const char * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_char_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_short_put_nbi
 **********************************************************/

extern void  __real_shmem_short_put_nbi(short * a1, const short * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_short_put_nbi(short * a1, const short * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_short_put_nbi(short *, const short *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(short)*a3);
  __real_shmem_short_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(short)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_short_put_nbi_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_short_put_nbi__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_PUT_NBI_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_SHORT_PUT_NBI__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_int_put_nbi
 **********************************************************/

extern void  __real_shmem_int_put_nbi(int * a1, const int * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_int_put_nbi(int * a1, const int * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_put_nbi(int *, const int *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(int)*a3);
  __real_shmem_int_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(int)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_int_put_nbi_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_int_put_nbi__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_PUT_NBI_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_INT_PUT_NBI__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_long_put_nbi
 **********************************************************/

extern void  __real_shmem_long_put_nbi(long * a1, const long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_long_put_nbi(long * a1, const long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_put_nbi(long *, const long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long)*a3);
  __real_shmem_long_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_long_put_nbi_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_long_put_nbi__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_PUT_NBI_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONG_PUT_NBI__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_longlong_put_nbi
 **********************************************************/

extern void  __real_shmem_longlong_put_nbi(long long * a1, const long long * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_longlong_put_nbi(long long * a1, const long long * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_put_nbi(long long *, const long long *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(long long)*a3);
  __real_shmem_longlong_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(long long)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_longlong_put_nbi_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_longlong_put_nbi__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_PUT_NBI_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_LONGLONG_PUT_NBI__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_float_put_nbi
 **********************************************************/

extern void  __real_shmem_float_put_nbi(float * a1, const float * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float_put_nbi(float * a1, const float * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_put_nbi(float *, const float *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmem_float_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_float_put_nbi_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float_put_nbi__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_PUT_NBI_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT_PUT_NBI__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_double_put_nbi
 **********************************************************/

extern void  __real_shmem_double_put_nbi(double * a1, const double * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_double_put_nbi(double * a1, const double * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_put_nbi(double *, const double *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(double)*a3);
  __real_shmem_double_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(double)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_double_put_nbi_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_double_put_nbi__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_PUT_NBI_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_DOUBLE_PUT_NBI__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_double_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_float128_put_nbi
 **********************************************************/

extern void  __real_shmem_float128_put_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4) ;
extern void  __wrap_shmem_float128_put_nbi(__float128 * a1, const __float128 * a2, size_t a3, int a4)  {

  TAU_PROFILE_TIMER(t,"void shmem_float128_put_nbi(__float128 *, const __float128 *, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a4, sizeof(float)*a3);
  __real_shmem_float128_put_nbi(a1, a2, a3, a4);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(float)*a3, a4);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_float128_put_nbi_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_shmem_float128_put_nbi__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_PUT_NBI_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put_nbi(a1, a2, *a3, *a4);
}

extern void __wrap_SHMEM_FLOAT128_PUT_NBI__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_float128_put_nbi(a1, a2, *a3, *a4);
}


/**********************************************************
   shmem_put16_signal_nb
 **********************************************************/

extern void  __real_shmem_put16_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_put16_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_put16_signal_nb(void *, const void *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 2*a3);
  __real_shmem_put16_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put16_signal_nb_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put16_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_put16_signal_nb__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put16_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT16_SIGNAL_NB_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put16_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT16_SIGNAL_NB__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put16_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_put32_signal_nb
 **********************************************************/

extern void  __real_shmem_put32_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_put32_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_put32_signal_nb(void *, const void *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 4*a3);
  __real_shmem_put32_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 4*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put32_signal_nb_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put32_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_put32_signal_nb__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put32_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT32_SIGNAL_NB_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put32_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT32_SIGNAL_NB__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put32_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_put64_signal_nb
 **********************************************************/

extern void  __real_shmem_put64_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_put64_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_put64_signal_nb(void *, const void *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 8*a3);
  __real_shmem_put64_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put64_signal_nb_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put64_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_put64_signal_nb__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put64_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT64_SIGNAL_NB_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put64_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT64_SIGNAL_NB__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put64_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_put128_signal_nb
 **********************************************************/

extern void  __real_shmem_put128_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_put128_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_put128_signal_nb(void *, const void *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 16*a3);
  __real_shmem_put128_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 16*a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_put128_signal_nb_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put128_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_put128_signal_nb__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put128_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT128_SIGNAL_NB_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put128_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUT128_SIGNAL_NB__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_put128_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_putmem_signal_nb
 **********************************************************/

extern void  __real_shmem_putmem_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7) ;
extern void  __wrap_shmem_putmem_signal_nb(void * a1, const void * a2, size_t a3, uint64_t * a4, uint64_t a5, int a6, void ** a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_putmem_signal_nb(void *, const void *, size_t, uint64_t *, uint64_t, int, void **) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, a3);
  __real_shmem_putmem_signal_nb(a1, a2, a3, a4, a5, a6, a7);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), a3, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_putmem_signal_nb_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_putmem_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_putmem_signal_nb__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_putmem_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUTMEM_SIGNAL_NB_(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_putmem_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_PUTMEM_SIGNAL_NB__(void * a1, const void * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_putmem_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
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

extern void __wrap_shmem_short_put_signal_nb_(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_short_put_signal_nb__(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_SHORT_PUT_SIGNAL_NB_(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_SHORT_PUT_SIGNAL_NB__(short * a1, const short * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_short_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
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

extern void __wrap_shmem_int_put_signal_nb_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_int_put_signal_nb__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_INT_PUT_SIGNAL_NB_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_INT_PUT_SIGNAL_NB__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_int_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
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

extern void __wrap_shmem_long_put_signal_nb_(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_long_put_signal_nb__(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_LONG_PUT_SIGNAL_NB_(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_LONG_PUT_SIGNAL_NB__(long * a1, const long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_long_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
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

extern void __wrap_shmem_longlong_put_signal_nb_(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_longlong_put_signal_nb__(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_LONGLONG_PUT_SIGNAL_NB_(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_LONGLONG_PUT_SIGNAL_NB__(long long * a1, const long long * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_longlong_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
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

extern void __wrap_shmem_float_put_signal_nb_(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_float_put_signal_nb__(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_FLOAT_PUT_SIGNAL_NB_(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_FLOAT_PUT_SIGNAL_NB__(float * a1, const float * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_float_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
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

extern void __wrap_shmem_double_put_signal_nb_(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_shmem_double_put_signal_nb__(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_DOUBLE_PUT_SIGNAL_NB_(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_DOUBLE_PUT_SIGNAL_NB__(double * a1, const double * a2, SHMEM_FINT * a3, uint64_t * a4, uint64_t * a5, SHMEM_FINT * a6, void ** a7)
{
   __wrap_shmem_double_put_signal_nb(a1, a2, *a3, a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_char_iget
 **********************************************************/

extern void  __real_shmem_char_iget(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_char_iget(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_iget(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), sizeof(char)*a5, a6);
  __real_shmem_char_iget(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, sizeof(char)*a5);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_char_iget_(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_char_iget__(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_CHAR_IGET_(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_CHAR_IGET__(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_short_iget_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_short_iget__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_SHORT_IGET_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_SHORT_IGET__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_int_iget_(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_int_iget__(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_INT_IGET_(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_INT_IGET__(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_long_iget_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_long_iget__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONG_IGET_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONG_IGET__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_longlong_iget_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_longlong_iget__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONGLONG_IGET_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONGLONG_IGET__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_float_iget_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_float_iget__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT_IGET_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT_IGET__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_double_iget_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_double_iget__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_DOUBLE_IGET_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_DOUBLE_IGET__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_float128_iget_(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_float128_iget__(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT128_IGET_(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT128_IGET__(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iget(a1, a2, *a3, *a4, *a5, *a6);
}


/**********************************************************
   shmem_iget8
 **********************************************************/

extern void  __real_shmem_iget8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iget8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget8(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 8*a5, a6);
  __real_shmem_iget8(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 8*a5);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_iget8_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget8(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iget8__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget8(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET8_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget8(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET8__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget8(a1, a2, *a3, *a4, *a5, *a6);
}


/**********************************************************
   shmem_iget16
 **********************************************************/

extern void  __real_shmem_iget16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iget16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iget16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG_REMOTE(TAU_SHMEM_TAGID_NEXT, Tau_get_node(), 2*a5, a6);
  __real_shmem_iget16(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG(TAU_SHMEM_TAGID, a6, 2*a5);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_iget16_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget16(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iget16__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget16(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET16_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget16(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET16__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget16(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_iget32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iget32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_iget64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iget64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_iget128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iget128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IGET128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}


/**********************************************************
   shmem_char_iput
 **********************************************************/

extern void  __real_shmem_char_iput(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_char_iput(char * a1, const char * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_char_iput(char *, const char *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, sizeof(char)*a5);
  __real_shmem_char_iput(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), sizeof(char)*a5, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_char_iput_(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_char_iput__(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_CHAR_IPUT_(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_CHAR_IPUT__(char * a1, const char * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_char_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_short_iput_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_short_iput__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_SHORT_IPUT_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_SHORT_IPUT__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_int_iput_(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_int_iput__(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_INT_IPUT_(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_INT_IPUT__(SHMEM_FINT * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_long_iput_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_long_iput__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONG_IPUT_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONG_IPUT__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_longlong_iput_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_longlong_iput__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONGLONG_IPUT_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_LONGLONG_IPUT__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_float_iput_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_float_iput__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT_IPUT_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT_IPUT__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_double_iput_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_double_iput__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_DOUBLE_IPUT_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_DOUBLE_IPUT__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_float128_iput_(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_float128_iput__(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT128_IPUT_(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_FLOAT128_IPUT__(__float128 * a1, const __float128 * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_float128_iput(a1, a2, *a3, *a4, *a5, *a6);
}


/**********************************************************
   shmem_iput8
 **********************************************************/

extern void  __real_shmem_iput8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iput8(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput8(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 8*a5);
  __real_shmem_iput8(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 8*a5, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_iput8_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput8(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iput8__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput8(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT8_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput8(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT8__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput8(a1, a2, *a3, *a4, *a5, *a6);
}


/**********************************************************
   shmem_iput16
 **********************************************************/

extern void  __real_shmem_iput16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6) ;
extern void  __wrap_shmem_iput16(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6)  {

  TAU_PROFILE_TIMER(t,"void shmem_iput16(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  TAU_TRACE_SENDMSG(TAU_SHMEM_TAGID_NEXT, a6, 2*a5);
  __real_shmem_iput16(a1, a2, a3, a4, a5, a6);
  TAU_TRACE_RECVMSG_REMOTE(TAU_SHMEM_TAGID, Tau_get_node(), 2*a5, a6);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_iput16_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput16(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iput16__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput16(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT16_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput16(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT16__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput16(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_iput32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iput32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_iput64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iput64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
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

extern void __wrap_shmem_iput128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_shmem_iput128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void __wrap_SHMEM_IPUT128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
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

extern char __wrap_shmem_char_g_(const char * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_char_g(a1, *a2);
}

extern char __wrap_shmem_char_g__(const char * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_char_g(a1, *a2);
}

extern char __wrap_SHMEM_CHAR_G_(const char * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_char_g(a1, *a2);
}

extern char __wrap_SHMEM_CHAR_G__(const char * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_char_g(a1, *a2);
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

extern short __wrap_shmem_short_g_(const short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_g(a1, *a2);
}

extern short __wrap_shmem_short_g__(const short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_g(a1, *a2);
}

extern short __wrap_SHMEM_SHORT_G_(const short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_g(a1, *a2);
}

extern short __wrap_SHMEM_SHORT_G__(const short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_g(a1, *a2);
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

extern int __wrap_shmem_int_g_(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_g(a1, *a2);
}

extern int __wrap_shmem_int_g__(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_g(a1, *a2);
}

extern int __wrap_SHMEM_INT_G_(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_g(a1, *a2);
}

extern int __wrap_SHMEM_INT_G__(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_g(a1, *a2);
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

extern long __wrap_shmem_long_g_(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_g(a1, *a2);
}

extern long __wrap_shmem_long_g__(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_g(a1, *a2);
}

extern long __wrap_SHMEM_LONG_G_(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_g(a1, *a2);
}

extern long __wrap_SHMEM_LONG_G__(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_g(a1, *a2);
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

extern long long __wrap_shmem_longlong_g_(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
}

extern long long __wrap_shmem_longlong_g__(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
}

extern long long __wrap_SHMEM_LONGLONG_G_(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
}

extern long long __wrap_SHMEM_LONGLONG_G__(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
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

extern float __wrap_shmem_float_g_(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_g(a1, *a2);
}

extern float __wrap_shmem_float_g__(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_g(a1, *a2);
}

extern float __wrap_SHMEM_FLOAT_G_(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_g(a1, *a2);
}

extern float __wrap_SHMEM_FLOAT_G__(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_g(a1, *a2);
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

extern double __wrap_shmem_double_g_(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_g(a1, *a2);
}

extern double __wrap_shmem_double_g__(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_g(a1, *a2);
}

extern double __wrap_SHMEM_DOUBLE_G_(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_g(a1, *a2);
}

extern double __wrap_SHMEM_DOUBLE_G__(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_g(a1, *a2);
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

extern long double __wrap_shmem_ld80_g_(const long double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_ld80_g(a1, *a2);
}

extern long double __wrap_shmem_ld80_g__(const long double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_ld80_g(a1, *a2);
}

extern long double __wrap_SHMEM_LD80_G_(const long double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_ld80_g(a1, *a2);
}

extern long double __wrap_SHMEM_LD80_G__(const long double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_ld80_g(a1, *a2);
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

extern __float128 __wrap_shmem_float128_g_(const __float128 * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float128_g(a1, *a2);
}

extern __float128 __wrap_shmem_float128_g__(const __float128 * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float128_g(a1, *a2);
}

extern __float128 __wrap_SHMEM_FLOAT128_G_(const __float128 * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float128_g(a1, *a2);
}

extern __float128 __wrap_SHMEM_FLOAT128_G__(const __float128 * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float128_g(a1, *a2);
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

extern void __wrap_shmem_char_p_(char * a1, char * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
}

extern void __wrap_shmem_char_p__(char * a1, char * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_CHAR_P_(char * a1, char * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_CHAR_P__(char * a1, char * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
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

extern void __wrap_shmem_short_p_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
}

extern void __wrap_shmem_short_p__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_SHORT_P_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_SHORT_P__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
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

extern void __wrap_shmem_int_p_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
}

extern void __wrap_shmem_int_p__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_P_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_P__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
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

extern void __wrap_shmem_long_p_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
}

extern void __wrap_shmem_long_p__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_P_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_P__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
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

extern void __wrap_shmem_longlong_p_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
}

extern void __wrap_shmem_longlong_p__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_P_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_P__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
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

extern void __wrap_shmem_float_p_(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
}

extern void __wrap_shmem_float_p__(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_FLOAT_P_(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_FLOAT_P__(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
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

extern void __wrap_shmem_double_p_(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
}

extern void __wrap_shmem_double_p__(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_DOUBLE_P_(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_DOUBLE_P__(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
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

extern void __wrap_shmem_ld80_p_(long double * a1, long double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_ld80_p(a1, *a2, *a3);
}

extern void __wrap_shmem_ld80_p__(long double * a1, long double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_ld80_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LD80_P_(long double * a1, long double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_ld80_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LD80_P__(long double * a1, long double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_ld80_p(a1, *a2, *a3);
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

extern void __wrap_shmem_float128_p_(__float128 * a1, __float128 * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float128_p(a1, *a2, *a3);
}

extern void __wrap_shmem_float128_p__(__float128 * a1, __float128 * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float128_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_FLOAT128_P_(__float128 * a1, __float128 * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float128_p(a1, *a2, *a3);
}

extern void __wrap_SHMEM_FLOAT128_P__(__float128 * a1, __float128 * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float128_p(a1, *a2, *a3);
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

extern short __wrap_shmem_short_swap_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_swap(a1, *a2, *a3);
}

extern short __wrap_shmem_short_swap__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_swap(a1, *a2, *a3);
}

extern short __wrap_SHMEM_SHORT_SWAP_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_swap(a1, *a2, *a3);
}

extern short __wrap_SHMEM_SHORT_SWAP__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_swap(a1, *a2, *a3);
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

extern int __wrap_shmem_int_swap_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
}

extern int __wrap_shmem_int_swap__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
}

extern int __wrap_SHMEM_INT_SWAP_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
}

extern int __wrap_SHMEM_INT_SWAP__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
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

extern long __wrap_shmem_long_swap_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
}

extern long __wrap_shmem_long_swap__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
}

extern long __wrap_SHMEM_LONG_SWAP_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
}

extern long __wrap_SHMEM_LONG_SWAP__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
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

extern long long __wrap_shmem_longlong_swap_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
}

extern long long __wrap_shmem_longlong_swap__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
}

extern long long __wrap_SHMEM_LONGLONG_SWAP_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
}

extern long long __wrap_SHMEM_LONGLONG_SWAP__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
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

extern float __wrap_shmem_float_swap_(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
}

extern float __wrap_shmem_float_swap__(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
}

extern float __wrap_SHMEM_FLOAT_SWAP_(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
}

extern float __wrap_SHMEM_FLOAT_SWAP__(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
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

extern double __wrap_shmem_double_swap_(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
}

extern double __wrap_shmem_double_swap__(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
}

extern double __wrap_SHMEM_DOUBLE_SWAP_(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
}

extern double __wrap_SHMEM_DOUBLE_SWAP__(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
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

extern void __wrap_shmem_short_swap_nb_(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_short_swap_nb__(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_SWAP_NB_(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_SWAP_NB__(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_swap_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_int_swap_nb_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_int_swap_nb__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_SWAP_NB_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_SWAP_NB__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_swap_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_long_swap_nb_(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_long_swap_nb__(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_SWAP_NB_(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_SWAP_NB__(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_swap_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_longlong_swap_nb_(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_longlong_swap_nb__(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_SWAP_NB_(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_SWAP_NB__(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_swap_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_float_swap_nb_(float * a1, float * a2, float * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_float_swap_nb__(float * a1, float * a2, float * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT_SWAP_NB_(float * a1, float * a2, float * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_FLOAT_SWAP_NB__(float * a1, float * a2, float * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_float_swap_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_double_swap_nb_(double * a1, double * a2, double * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_double_swap_nb__(double * a1, double * a2, double * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_DOUBLE_SWAP_NB_(double * a1, double * a2, double * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_DOUBLE_SWAP_NB__(double * a1, double * a2, double * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_double_swap_nb(a1, a2, *a3, *a4, a5);
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

extern short __wrap_shmem_short_cswap_(short * a1, short * a2, short * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_cswap(a1, *a2, *a3, *a4);
}

extern short __wrap_shmem_short_cswap__(short * a1, short * a2, short * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_cswap(a1, *a2, *a3, *a4);
}

extern short __wrap_SHMEM_SHORT_CSWAP_(short * a1, short * a2, short * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_cswap(a1, *a2, *a3, *a4);
}

extern short __wrap_SHMEM_SHORT_CSWAP__(short * a1, short * a2, short * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_short_cswap(a1, *a2, *a3, *a4);
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

extern int __wrap_shmem_int_cswap_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
}

extern int __wrap_shmem_int_cswap__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
}

extern int __wrap_SHMEM_INT_CSWAP_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
}

extern int __wrap_SHMEM_INT_CSWAP__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
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

extern long __wrap_shmem_long_cswap_(long * a1, long * a2, long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
}

extern long __wrap_shmem_long_cswap__(long * a1, long * a2, long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
}

extern long __wrap_SHMEM_LONG_CSWAP_(long * a1, long * a2, long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
}

extern long __wrap_SHMEM_LONG_CSWAP__(long * a1, long * a2, long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
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

extern long long __wrap_shmem_longlong_cswap_(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
}

extern long long __wrap_shmem_longlong_cswap__(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
}

extern long long __wrap_SHMEM_LONGLONG_CSWAP_(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
}

extern long long __wrap_SHMEM_LONGLONG_CSWAP__(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
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

extern void __wrap_shmem_short_cswap_nb_(short * a1, short * a2, short * a3, short * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_shmem_short_cswap_nb__(short * a1, short * a2, short * a3, short * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_SHORT_CSWAP_NB_(short * a1, short * a2, short * a3, short * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_SHORT_CSWAP_NB__(short * a1, short * a2, short * a3, short * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_short_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
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

extern void __wrap_shmem_int_cswap_nb_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_shmem_int_cswap_nb__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_INT_CSWAP_NB_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_INT_CSWAP_NB__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_int_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
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

extern void __wrap_shmem_long_cswap_nb_(long * a1, long * a2, long * a3, long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_shmem_long_cswap_nb__(long * a1, long * a2, long * a3, long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_LONG_CSWAP_NB_(long * a1, long * a2, long * a3, long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_LONG_CSWAP_NB__(long * a1, long * a2, long * a3, long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_long_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
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

extern void __wrap_shmem_longlong_cswap_nb_(long long * a1, long long * a2, long long * a3, long long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_shmem_longlong_cswap_nb__(long long * a1, long long * a2, long long * a3, long long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_LONGLONG_CSWAP_NB_(long long * a1, long long * a2, long long * a3, long long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
}

extern void __wrap_SHMEM_LONGLONG_CSWAP_NB__(long long * a1, long long * a2, long long * a3, long long * a4, SHMEM_FINT * a5, void ** a6)
{
   __wrap_shmem_longlong_cswap_nb(a1, a2, *a3, *a4, *a5, a6);
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

extern short __wrap_shmem_short_finc_(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_finc(a1, *a2);
}

extern short __wrap_shmem_short_finc__(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_finc(a1, *a2);
}

extern short __wrap_SHMEM_SHORT_FINC_(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_finc(a1, *a2);
}

extern short __wrap_SHMEM_SHORT_FINC__(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_finc(a1, *a2);
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

extern int __wrap_shmem_int_finc_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
}

extern int __wrap_shmem_int_finc__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
}

extern int __wrap_SHMEM_INT_FINC_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
}

extern int __wrap_SHMEM_INT_FINC__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
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

extern long __wrap_shmem_long_finc_(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
}

extern long __wrap_shmem_long_finc__(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
}

extern long __wrap_SHMEM_LONG_FINC_(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
}

extern long __wrap_SHMEM_LONG_FINC__(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
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

extern long long __wrap_shmem_longlong_finc_(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
}

extern long long __wrap_shmem_longlong_finc__(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
}

extern long long __wrap_SHMEM_LONGLONG_FINC_(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
}

extern long long __wrap_SHMEM_LONGLONG_FINC__(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
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

extern void __wrap_shmem_short_finc_nb_(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_shmem_short_finc_nb__(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_SHORT_FINC_NB_(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_SHORT_FINC_NB__(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_finc_nb(a1, a2, *a3, a4);
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

extern void __wrap_shmem_int_finc_nb_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_shmem_int_finc_nb__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_INT_FINC_NB_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_INT_FINC_NB__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_finc_nb(a1, a2, *a3, a4);
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

extern void __wrap_shmem_long_finc_nb_(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_shmem_long_finc_nb__(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_LONG_FINC_NB_(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_LONG_FINC_NB__(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_finc_nb(a1, a2, *a3, a4);
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

extern void __wrap_shmem_longlong_finc_nb_(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_shmem_longlong_finc_nb__(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_LONGLONG_FINC_NB_(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, *a3, a4);
}

extern void __wrap_SHMEM_LONGLONG_FINC_NB__(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_finc_nb(a1, a2, *a3, a4);
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

extern void __wrap_shmem_short_inc_(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_inc(a1, *a2);
}

extern void __wrap_shmem_short_inc__(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_inc(a1, *a2);
}

extern void __wrap_SHMEM_SHORT_INC_(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_inc(a1, *a2);
}

extern void __wrap_SHMEM_SHORT_INC__(short * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_short_inc(a1, *a2);
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

extern void __wrap_shmem_int_inc_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
}

extern void __wrap_shmem_int_inc__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
}

extern void __wrap_SHMEM_INT_INC_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
}

extern void __wrap_SHMEM_INT_INC__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
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

extern void __wrap_shmem_long_inc_(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
}

extern void __wrap_shmem_long_inc__(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
}

extern void __wrap_SHMEM_LONG_INC_(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
}

extern void __wrap_SHMEM_LONG_INC__(long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
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

extern void __wrap_shmem_longlong_inc_(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
}

extern void __wrap_shmem_longlong_inc__(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
}

extern void __wrap_SHMEM_LONGLONG_INC_(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
}

extern void __wrap_SHMEM_LONGLONG_INC__(long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
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

extern void __wrap_shmem_short_inc_nb_(short * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, *a2, a3);
}

extern void __wrap_shmem_short_inc_nb__(short * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_SHORT_INC_NB_(short * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_SHORT_INC_NB__(short * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_short_inc_nb(a1, *a2, a3);
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

extern void __wrap_shmem_int_inc_nb_(SHMEM_FINT * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, *a2, a3);
}

extern void __wrap_shmem_int_inc_nb__(SHMEM_FINT * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_INT_INC_NB_(SHMEM_FINT * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_INT_INC_NB__(SHMEM_FINT * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_int_inc_nb(a1, *a2, a3);
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

extern void __wrap_shmem_long_inc_nb_(long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, *a2, a3);
}

extern void __wrap_shmem_long_inc_nb__(long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_LONG_INC_NB_(long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_LONG_INC_NB__(long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_long_inc_nb(a1, *a2, a3);
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

extern void __wrap_shmem_longlong_inc_nb_(long long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, *a2, a3);
}

extern void __wrap_shmem_longlong_inc_nb__(long long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_LONGLONG_INC_NB_(long long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, *a2, a3);
}

extern void __wrap_SHMEM_LONGLONG_INC_NB__(long long * a1, SHMEM_FINT * a2, void ** a3)
{
   __wrap_shmem_longlong_inc_nb(a1, *a2, a3);
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

extern short __wrap_shmem_short_fadd_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_fadd(a1, *a2, *a3);
}

extern short __wrap_shmem_short_fadd__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_fadd(a1, *a2, *a3);
}

extern short __wrap_SHMEM_SHORT_FADD_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_fadd(a1, *a2, *a3);
}

extern short __wrap_SHMEM_SHORT_FADD__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_fadd(a1, *a2, *a3);
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

extern int __wrap_shmem_int_fadd_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
}

extern int __wrap_shmem_int_fadd__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
}

extern int __wrap_SHMEM_INT_FADD_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
}

extern int __wrap_SHMEM_INT_FADD__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
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

extern long __wrap_shmem_long_fadd_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
}

extern long __wrap_shmem_long_fadd__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
}

extern long __wrap_SHMEM_LONG_FADD_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
}

extern long __wrap_SHMEM_LONG_FADD__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
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

extern long long __wrap_shmem_longlong_fadd_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
}

extern long long __wrap_shmem_longlong_fadd__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
}

extern long long __wrap_SHMEM_LONGLONG_FADD_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
}

extern long long __wrap_SHMEM_LONGLONG_FADD__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
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

extern void __wrap_shmem_short_fadd_nb_(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_short_fadd_nb__(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_FADD_NB_(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_SHORT_FADD_NB__(short * a1, short * a2, short * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_short_fadd_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_int_fadd_nb_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_int_fadd_nb__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_FADD_NB_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_INT_FADD_NB__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_int_fadd_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_long_fadd_nb_(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_long_fadd_nb__(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_FADD_NB_(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONG_FADD_NB__(long * a1, long * a2, long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_long_fadd_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_longlong_fadd_nb_(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_longlong_fadd_nb__(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_FADD_NB_(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_LONGLONG_FADD_NB__(long long * a1, long long * a2, long long * a3, SHMEM_FINT * a4, void ** a5)
{
   __wrap_shmem_longlong_fadd_nb(a1, a2, *a3, *a4, a5);
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

extern void __wrap_shmem_short_add_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_add(a1, *a2, *a3);
}

extern void __wrap_shmem_short_add__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_SHORT_ADD_(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_SHORT_ADD__(short * a1, short * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_short_add(a1, *a2, *a3);
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

extern void __wrap_shmem_int_add_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
}

extern void __wrap_shmem_int_add__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_ADD_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_ADD__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
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

extern void __wrap_shmem_long_add_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
}

extern void __wrap_shmem_long_add__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_ADD_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_ADD__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
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

extern void __wrap_shmem_longlong_add_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
}

extern void __wrap_shmem_longlong_add__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_ADD_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_ADD__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
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

extern void __wrap_shmem_short_add_nb_(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_shmem_short_add_nb__(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_SHORT_ADD_NB_(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_SHORT_ADD_NB__(short * a1, short * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_short_add_nb(a1, *a2, *a3, a4);
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

extern void __wrap_shmem_int_add_nb_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_shmem_int_add_nb__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_INT_ADD_NB_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_INT_ADD_NB__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_int_add_nb(a1, *a2, *a3, a4);
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

extern void __wrap_shmem_long_add_nb_(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_shmem_long_add_nb__(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_LONG_ADD_NB_(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_LONG_ADD_NB__(long * a1, long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_long_add_nb(a1, *a2, *a3, a4);
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

extern void __wrap_shmem_longlong_add_nb_(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_shmem_longlong_add_nb__(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_LONGLONG_ADD_NB_(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_LONGLONG_ADD_NB__(long long * a1, long long * a2, SHMEM_FINT * a3, void ** a4)
{
   __wrap_shmem_longlong_add_nb(a1, *a2, *a3, a4);
}


/**********************************************************
   shmem_int_fetch
 **********************************************************/

extern int  __real_shmem_int_fetch(const int * a1, int a2) ;
extern int  __wrap_shmem_int_fetch(const int * a1, int a2)  {

  int retval;
  TAU_PROFILE_TIMER(t,"int shmem_int_fetch(const int *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_int_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

extern int __wrap_shmem_int_fetch_(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_fetch(a1, *a2);
}

extern int __wrap_shmem_int_fetch__(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_fetch(a1, *a2);
}

extern int __wrap_SHMEM_INT_FETCH_(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_fetch(a1, *a2);
}

extern int __wrap_SHMEM_INT_FETCH__(const int * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_fetch(a1, *a2);
}


/**********************************************************
   shmem_long_fetch
 **********************************************************/

extern long  __real_shmem_long_fetch(const long * a1, int a2) ;
extern long  __wrap_shmem_long_fetch(const long * a1, int a2)  {

  long retval;
  TAU_PROFILE_TIMER(t,"long shmem_long_fetch(const long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_long_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

extern long __wrap_shmem_long_fetch_(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_fetch(a1, *a2);
}

extern long __wrap_shmem_long_fetch__(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_fetch(a1, *a2);
}

extern long __wrap_SHMEM_LONG_FETCH_(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_fetch(a1, *a2);
}

extern long __wrap_SHMEM_LONG_FETCH__(const long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_long_fetch(a1, *a2);
}


/**********************************************************
   shmem_float_fetch
 **********************************************************/

extern float  __real_shmem_float_fetch(const float * a1, int a2) ;
extern float  __wrap_shmem_float_fetch(const float * a1, int a2)  {

  float retval;
  TAU_PROFILE_TIMER(t,"float shmem_float_fetch(const float *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_float_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

extern float __wrap_shmem_float_fetch_(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_fetch(a1, *a2);
}

extern float __wrap_shmem_float_fetch__(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_fetch(a1, *a2);
}

extern float __wrap_SHMEM_FLOAT_FETCH_(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_fetch(a1, *a2);
}

extern float __wrap_SHMEM_FLOAT_FETCH__(const float * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_float_fetch(a1, *a2);
}


/**********************************************************
   shmem_double_fetch
 **********************************************************/

extern double  __real_shmem_double_fetch(const double * a1, int a2) ;
extern double  __wrap_shmem_double_fetch(const double * a1, int a2)  {

  double retval;
  TAU_PROFILE_TIMER(t,"double shmem_double_fetch(const double *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_double_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

extern double __wrap_shmem_double_fetch_(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_fetch(a1, *a2);
}

extern double __wrap_shmem_double_fetch__(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_fetch(a1, *a2);
}

extern double __wrap_SHMEM_DOUBLE_FETCH_(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_fetch(a1, *a2);
}

extern double __wrap_SHMEM_DOUBLE_FETCH__(const double * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_double_fetch(a1, *a2);
}


/**********************************************************
   shmem_longlong_fetch
 **********************************************************/

extern long long  __real_shmem_longlong_fetch(const long long * a1, int a2) ;
extern long long  __wrap_shmem_longlong_fetch(const long long * a1, int a2)  {

  long long retval;
  TAU_PROFILE_TIMER(t,"long long shmem_longlong_fetch(const long long *, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real_shmem_longlong_fetch(a1, a2);
  TAU_PROFILE_STOP(t);
  return retval;

}

extern long long __wrap_shmem_longlong_fetch_(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_fetch(a1, *a2);
}

extern long long __wrap_shmem_longlong_fetch__(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_fetch(a1, *a2);
}

extern long long __wrap_SHMEM_LONGLONG_FETCH_(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_fetch(a1, *a2);
}

extern long long __wrap_SHMEM_LONGLONG_FETCH__(const long long * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_longlong_fetch(a1, *a2);
}


/**********************************************************
   shmem_int_set
 **********************************************************/

extern void  __real_shmem_int_set(int * a1, int a2, int a3) ;
extern void  __wrap_shmem_int_set(int * a1, int a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_int_set(int *, int, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_int_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_int_set_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_set(a1, *a2, *a3);
}

extern void __wrap_shmem_int_set__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_SET_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_SET__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_set(a1, *a2, *a3);
}


/**********************************************************
   shmem_long_set
 **********************************************************/

extern void  __real_shmem_long_set(long * a1, long a2, int a3) ;
extern void  __wrap_shmem_long_set(long * a1, long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_long_set(long *, long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_long_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_long_set_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_set(a1, *a2, *a3);
}

extern void __wrap_shmem_long_set__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_SET_(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_SET__(long * a1, long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_long_set(a1, *a2, *a3);
}


/**********************************************************
   shmem_float_set
 **********************************************************/

extern void  __real_shmem_float_set(float * a1, float a2, int a3) ;
extern void  __wrap_shmem_float_set(float * a1, float a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_float_set(float *, float, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_float_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_float_set_(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_set(a1, *a2, *a3);
}

extern void __wrap_shmem_float_set__(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_FLOAT_SET_(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_FLOAT_SET__(float * a1, float * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_float_set(a1, *a2, *a3);
}


/**********************************************************
   shmem_double_set
 **********************************************************/

extern void  __real_shmem_double_set(double * a1, double a2, int a3) ;
extern void  __wrap_shmem_double_set(double * a1, double a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_double_set(double *, double, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_double_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_double_set_(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_set(a1, *a2, *a3);
}

extern void __wrap_shmem_double_set__(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_DOUBLE_SET_(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_DOUBLE_SET__(double * a1, double * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_double_set(a1, *a2, *a3);
}


/**********************************************************
   shmem_longlong_set
 **********************************************************/

extern void  __real_shmem_longlong_set(long long * a1, long long a2, int a3) ;
extern void  __wrap_shmem_longlong_set(long long * a1, long long a2, int a3)  {

  TAU_PROFILE_TIMER(t,"void shmem_longlong_set(long long *, long long, int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_longlong_set(a1, a2, a3);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_longlong_set_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_set(a1, *a2, *a3);
}

extern void __wrap_shmem_longlong_set__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_SET_(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_set(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_SET__(long long * a1, long long * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_longlong_set(a1, *a2, *a3);
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

extern void __wrap_shmem_barrier_all_()
{
   __wrap_shmem_barrier_all();
}

extern void __wrap_shmem_barrier_all__()
{
   __wrap_shmem_barrier_all();
}

extern void __wrap_SHMEM_BARRIER_ALL_()
{
   __wrap_shmem_barrier_all();
}

extern void __wrap_SHMEM_BARRIER_ALL__()
{
   __wrap_shmem_barrier_all();
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

extern void __wrap_shmem_barrier_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
}

extern void __wrap_shmem_barrier__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_BARRIER_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_BARRIER__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
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

extern void __wrap_shmem_team_barrier_(shmem_team_t * a1, long * a2)
{
   __wrap_shmem_team_barrier(*a1, a2);
}

extern void __wrap_shmem_team_barrier__(shmem_team_t * a1, long * a2)
{
   __wrap_shmem_team_barrier(*a1, a2);
}

extern void __wrap_SHMEM_TEAM_BARRIER_(shmem_team_t * a1, long * a2)
{
   __wrap_shmem_team_barrier(*a1, a2);
}

extern void __wrap_SHMEM_TEAM_BARRIER__(shmem_team_t * a1, long * a2)
{
   __wrap_shmem_team_barrier(*a1, a2);
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

extern void __wrap_shmem_fence_()
{
   __wrap_shmem_fence();
}

extern void __wrap_shmem_fence__()
{
   __wrap_shmem_fence();
}

extern void __wrap_SHMEM_FENCE_()
{
   __wrap_shmem_fence();
}

extern void __wrap_SHMEM_FENCE__()
{
   __wrap_shmem_fence();
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

extern void __wrap_shmem_quiet_()
{
   __wrap_shmem_quiet();
}

extern void __wrap_shmem_quiet__()
{
   __wrap_shmem_quiet();
}

extern void __wrap_SHMEM_QUIET_()
{
   __wrap_shmem_quiet();
}

extern void __wrap_SHMEM_QUIET__()
{
   __wrap_shmem_quiet();
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

extern void __wrap_shmem_set_lock_(long * a1)
{
   __wrap_shmem_set_lock(a1);
}

extern void __wrap_shmem_set_lock__(long * a1)
{
   __wrap_shmem_set_lock(a1);
}

extern void __wrap_SHMEM_SET_LOCK_(long * a1)
{
   __wrap_shmem_set_lock(a1);
}

extern void __wrap_SHMEM_SET_LOCK__(long * a1)
{
   __wrap_shmem_set_lock(a1);
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

extern void __wrap_shmem_clear_lock_(long * a1)
{
   __wrap_shmem_clear_lock(a1);
}

extern void __wrap_shmem_clear_lock__(long * a1)
{
   __wrap_shmem_clear_lock(a1);
}

extern void __wrap_SHMEM_CLEAR_LOCK_(long * a1)
{
   __wrap_shmem_clear_lock(a1);
}

extern void __wrap_SHMEM_CLEAR_LOCK__(long * a1)
{
   __wrap_shmem_clear_lock(a1);
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

extern int __wrap_shmem_test_lock_(long * a1)
{
   __wrap_shmem_test_lock(a1);
}

extern int __wrap_shmem_test_lock__(long * a1)
{
   __wrap_shmem_test_lock(a1);
}

extern int __wrap_SHMEM_TEST_LOCK_(long * a1)
{
   __wrap_shmem_test_lock(a1);
}

extern int __wrap_SHMEM_TEST_LOCK__(long * a1)
{
   __wrap_shmem_test_lock(a1);
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

extern void __wrap_shmem_clear_event_(long * a1)
{
   __wrap_shmem_clear_event(a1);
}

extern void __wrap_shmem_clear_event__(long * a1)
{
   __wrap_shmem_clear_event(a1);
}

extern void __wrap_SHMEM_CLEAR_EVENT_(long * a1)
{
   __wrap_shmem_clear_event(a1);
}

extern void __wrap_SHMEM_CLEAR_EVENT__(long * a1)
{
   __wrap_shmem_clear_event(a1);
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

extern void __wrap_shmem_set_event_(long * a1)
{
   __wrap_shmem_set_event(a1);
}

extern void __wrap_shmem_set_event__(long * a1)
{
   __wrap_shmem_set_event(a1);
}

extern void __wrap_SHMEM_SET_EVENT_(long * a1)
{
   __wrap_shmem_set_event(a1);
}

extern void __wrap_SHMEM_SET_EVENT__(long * a1)
{
   __wrap_shmem_set_event(a1);
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

extern int __wrap_shmem_test_event_(long * a1)
{
   __wrap_shmem_test_event(a1);
}

extern int __wrap_shmem_test_event__(long * a1)
{
   __wrap_shmem_test_event(a1);
}

extern int __wrap_SHMEM_TEST_EVENT_(long * a1)
{
   __wrap_shmem_test_event(a1);
}

extern int __wrap_SHMEM_TEST_EVENT__(long * a1)
{
   __wrap_shmem_test_event(a1);
}


/**********************************************************
   shmem_wait_event
 **********************************************************/

extern void  __real_shmem_wait_event(long * a1) ;
extern void  __wrap_shmem_wait_event(long * a1)  {

  TAU_PROFILE_TIMER(t,"void shmem_wait_event(long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_wait_event(a1);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_wait_event_(long * a1)
{
   __wrap_shmem_wait_event(a1);
}

extern void __wrap_shmem_wait_event__(long * a1)
{
   __wrap_shmem_wait_event(a1);
}

extern void __wrap_SHMEM_WAIT_EVENT_(long * a1)
{
   __wrap_shmem_wait_event(a1);
}

extern void __wrap_SHMEM_WAIT_EVENT__(long * a1)
{
   __wrap_shmem_wait_event(a1);
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

extern void __wrap_shmem_short_wait_(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
}

extern void __wrap_shmem_short_wait__(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
}

extern void __wrap_SHMEM_SHORT_WAIT_(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
}

extern void __wrap_SHMEM_SHORT_WAIT__(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
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

extern void __wrap_shmem_int_wait_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
}

extern void __wrap_shmem_int_wait__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
}

extern void __wrap_SHMEM_INT_WAIT_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
}

extern void __wrap_SHMEM_INT_WAIT__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
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

extern void __wrap_shmem_long_wait_(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
}

extern void __wrap_shmem_long_wait__(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
}

extern void __wrap_SHMEM_LONG_WAIT_(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
}

extern void __wrap_SHMEM_LONG_WAIT__(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
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

extern void __wrap_shmem_longlong_wait_(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
}

extern void __wrap_shmem_longlong_wait__(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
}

extern void __wrap_SHMEM_LONGLONG_WAIT_(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
}

extern void __wrap_SHMEM_LONGLONG_WAIT__(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
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

extern void __wrap_shmem_short_wait_until_(short * a1, SHMEM_FINT * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
}

extern void __wrap_shmem_short_wait_until__(short * a1, SHMEM_FINT * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_SHORT_WAIT_UNTIL_(short * a1, SHMEM_FINT * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_SHORT_WAIT_UNTIL__(short * a1, SHMEM_FINT * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
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

extern void __wrap_shmem_int_wait_until_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
}

extern void __wrap_shmem_int_wait_until__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_WAIT_UNTIL_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_INT_WAIT_UNTIL__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
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

extern void __wrap_shmem_long_wait_until_(long * a1, SHMEM_FINT * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
}

extern void __wrap_shmem_long_wait_until__(long * a1, SHMEM_FINT * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_WAIT_UNTIL_(long * a1, SHMEM_FINT * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONG_WAIT_UNTIL__(long * a1, SHMEM_FINT * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
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

extern void __wrap_shmem_longlong_wait_until_(long long * a1, SHMEM_FINT * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
}

extern void __wrap_shmem_longlong_wait_until__(long long * a1, SHMEM_FINT * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_WAIT_UNTIL_(long long * a1, SHMEM_FINT * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
}

extern void __wrap_SHMEM_LONGLONG_WAIT_UNTIL__(long long * a1, SHMEM_FINT * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
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

extern void __wrap_shmem_short_sum_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_sum_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_SUM_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_SUM_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_short_max_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_max_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_MAX_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_MAX_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_short_min_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_min_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_MIN_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_MIN_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_short_prod_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_prod_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_PROD_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_PROD_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_short_and_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_and_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_AND_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_AND_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_short_or_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_or_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_OR_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_OR_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_short_xor_to_all_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_short_xor_to_all__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_XOR_TO_ALL_(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_SHORT_XOR_TO_ALL__(short * a1, const short * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_sum_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_sum_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_SUM_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_SUM_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_max_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_max_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_MAX_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_MAX_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_min_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_min_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_MIN_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_MIN_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_prod_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_prod_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_PROD_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_PROD_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_and_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_and_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_AND_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_AND_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_or_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_or_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_OR_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_OR_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_int_xor_to_all_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_int_xor_to_all__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_XOR_TO_ALL_(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_INT_XOR_TO_ALL__(SHMEM_FINT * a1, const int * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_sum_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_sum_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_SUM_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_SUM_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_max_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_max_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_MAX_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_MAX_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_min_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_min_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_MIN_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_MIN_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_prod_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_prod_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_PROD_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_PROD_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_and_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_and_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_AND_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_AND_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_or_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_or_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_OR_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_OR_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_long_xor_to_all_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_long_xor_to_all__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_XOR_TO_ALL_(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONG_XOR_TO_ALL__(long * a1, const long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_sum_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_sum_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_SUM_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_SUM_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_max_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_max_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_MAX_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_MAX_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_min_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_min_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_MIN_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_MIN_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_prod_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_prod_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_PROD_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_PROD_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_and_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_and_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_AND_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_AND_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_or_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_or_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_OR_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_OR_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_longlong_xor_to_all_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_longlong_xor_to_all__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_XOR_TO_ALL_(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LONGLONG_XOR_TO_ALL__(long long * a1, const long long * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float_sum_to_all_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float_sum_to_all__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_SUM_TO_ALL_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_SUM_TO_ALL__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float_max_to_all_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float_max_to_all__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_MAX_TO_ALL_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_MAX_TO_ALL__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float_min_to_all_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float_min_to_all__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_MIN_TO_ALL_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_MIN_TO_ALL__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float_prod_to_all_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float_prod_to_all__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_PROD_TO_ALL_(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT_PROD_TO_ALL__(float * a1, const float * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_double_sum_to_all_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_double_sum_to_all__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_SUM_TO_ALL_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_SUM_TO_ALL__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_double_max_to_all_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_double_max_to_all__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_MAX_TO_ALL_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_MAX_TO_ALL__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_double_min_to_all_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_double_min_to_all__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_MIN_TO_ALL_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_MIN_TO_ALL__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_double_prod_to_all_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_double_prod_to_all__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_PROD_TO_ALL_(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_DOUBLE_PROD_TO_ALL__(double * a1, const double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_ld80_sum_to_all_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_ld80_sum_to_all__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_SUM_TO_ALL_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_SUM_TO_ALL__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_ld80_max_to_all_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_ld80_max_to_all__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_MAX_TO_ALL_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_MAX_TO_ALL__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_ld80_min_to_all_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_ld80_min_to_all__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_MIN_TO_ALL_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_MIN_TO_ALL__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_ld80_prod_to_all_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_ld80_prod_to_all__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_PROD_TO_ALL_(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_LD80_PROD_TO_ALL__(long double * a1, const long double * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long double * a7, long * a8)
{
   __wrap_shmem_ld80_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float128_sum_to_all_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float128_sum_to_all__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_SUM_TO_ALL_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_SUM_TO_ALL__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float128_max_to_all_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float128_max_to_all__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_MAX_TO_ALL_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_MAX_TO_ALL__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float128_min_to_all_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float128_min_to_all__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_MIN_TO_ALL_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_MIN_TO_ALL__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_float128_prod_to_all_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_shmem_float128_prod_to_all__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_PROD_TO_ALL_(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void __wrap_SHMEM_FLOAT128_PROD_TO_ALL__(__float128 * a1, const __float128 * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, __float128 * a7, long * a8)
{
   __wrap_shmem_float128_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void __wrap_shmem_broadcast32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void __wrap_shmem_broadcast32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void __wrap_SHMEM_BROADCAST32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void __wrap_SHMEM_BROADCAST32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
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

extern void __wrap_shmem_broadcast64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void __wrap_shmem_broadcast64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void __wrap_SHMEM_BROADCAST64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void __wrap_SHMEM_BROADCAST64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
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

extern void __wrap_shmem_alltoall_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_alltoall__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_ALLTOALL_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_ALLTOALL__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall(a1, a2, *a3, *a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_alltoall32
 **********************************************************/

extern void  __real_shmem_alltoall32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_alltoall32(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoall32(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoall32(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_alltoall32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_alltoall32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_ALLTOALL32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_ALLTOALL32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall32(a1, a2, *a3, *a4, *a5, *a6, a7);
}


/**********************************************************
   shmem_alltoall64
 **********************************************************/

extern void  __real_shmem_alltoall64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7) ;
extern void  __wrap_shmem_alltoall64(void * a1, const void * a2, size_t a3, int a4, int a5, int a6, long * a7)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoall64(void *, const void *, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoall64(a1, a2, a3, a4, a5, a6, a7);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_alltoall64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_alltoall64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_ALLTOALL64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_ALLTOALL64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_alltoall64(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void __wrap_shmem_team_alltoall_(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, *a3, *a4, a5);
}

extern void __wrap_shmem_team_alltoall__(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_TEAM_ALLTOALL_(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, *a3, *a4, a5);
}

extern void __wrap_SHMEM_TEAM_ALLTOALL__(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_shmem_team_alltoall(a1, a2, *a3, *a4, a5);
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

extern void __wrap_pshmem_team_alltoall_(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, *a3, *a4, a5);
}

extern void __wrap_pshmem_team_alltoall__(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, *a3, *a4, a5);
}

extern void __wrap_PSHMEM_TEAM_ALLTOALL_(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, *a3, *a4, a5);
}

extern void __wrap_PSHMEM_TEAM_ALLTOALL__(void * a1, const void * a2, SHMEM_FINT * a3, shmem_team_t * a4, long * a5)
{
   __wrap_pshmem_team_alltoall(a1, a2, *a3, *a4, a5);
}


/**********************************************************
   shmem_alltoalls32
 **********************************************************/

extern void  __real_shmem_alltoalls32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9) ;
extern void  __wrap_shmem_alltoalls32(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoalls32(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoalls32(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_alltoalls32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls32(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}

extern void __wrap_shmem_alltoalls32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls32(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}

extern void __wrap_SHMEM_ALLTOALLS32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls32(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}

extern void __wrap_SHMEM_ALLTOALLS32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls32(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}


/**********************************************************
   shmem_alltoalls64
 **********************************************************/

extern void  __real_shmem_alltoalls64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9) ;
extern void  __wrap_shmem_alltoalls64(void * a1, const void * a2, ptrdiff_t a3, ptrdiff_t a4, size_t a5, int a6, int a7, int a8, long * a9)  {

  TAU_PROFILE_TIMER(t,"void shmem_alltoalls64(void *, const void *, ptrdiff_t, ptrdiff_t, size_t, int, int, int, long *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_alltoalls64(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_alltoalls64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls64(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}

extern void __wrap_shmem_alltoalls64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls64(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}

extern void __wrap_SHMEM_ALLTOALLS64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls64(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
}

extern void __wrap_SHMEM_ALLTOALLS64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, long * a9)
{
   __wrap_shmem_alltoalls64(a1, a2, *a3, *a4, *a5, *a6, *a7, *a8, a9);
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

extern void __wrap_shmem_alltoallv_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
}

extern void __wrap_shmem_alltoallv__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
}

extern void __wrap_SHMEM_ALLTOALLV_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
}

extern void __wrap_SHMEM_ALLTOALLV__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv(a1, a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
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

extern void __wrap_shmem_team_alltoallv_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_shmem_team_alltoallv__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_SHMEM_TEAM_ALLTOALLV_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_SHMEM_TEAM_ALLTOALLV__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
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

extern void __wrap_pshmem_team_alltoallv_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_pshmem_team_alltoallv__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_PSHMEM_TEAM_ALLTOALLV_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_PSHMEM_TEAM_ALLTOALLV__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv(a1, a2, a3, a4, a5, a6, *a7, a8);
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

extern void __wrap_shmem_alltoallv_packed_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
}

extern void __wrap_shmem_alltoallv_packed__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
}

extern void __wrap_SHMEM_ALLTOALLV_PACKED_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
}

extern void __wrap_SHMEM_ALLTOALLV_PACKED__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, SHMEM_FINT * a7, SHMEM_FINT * a8, SHMEM_FINT * a9, long * a10)
{
   __wrap_shmem_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, *a8, *a9, a10);
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

extern void __wrap_shmem_team_alltoallv_packed_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_shmem_team_alltoallv_packed__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_SHMEM_TEAM_ALLTOALLV_PACKED_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_SHMEM_TEAM_ALLTOALLV_PACKED__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_shmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
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

extern void __wrap_pshmem_team_alltoallv_packed_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_pshmem_team_alltoallv_packed__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_PSHMEM_TEAM_ALLTOALLV_PACKED_(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
}

extern void __wrap_PSHMEM_TEAM_ALLTOALLV_PACKED__(void * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, const void * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, shmem_team_t * a7, long * a8)
{
   __wrap_pshmem_team_alltoallv_packed(a1, *a2, a3, a4, a5, a6, *a7, a8);
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

extern void __wrap_shmem_collect32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_collect32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_COLLECT32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_COLLECT32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void __wrap_shmem_collect64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_collect64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_COLLECT64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_COLLECT64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void __wrap_shmem_fcollect32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_fcollect32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_FCOLLECT32_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_FCOLLECT32__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void __wrap_shmem_fcollect64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_shmem_fcollect64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_FCOLLECT64_(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void __wrap_SHMEM_FCOLLECT64__(void * a1, const void * a2, SHMEM_FINT * a3, SHMEM_FINT * a4, SHMEM_FINT * a5, SHMEM_FINT * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void __wrap_shmem_team_split_(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(*a1, *a2, *a3, a4);
}

extern void __wrap_shmem_team_split__(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(*a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_TEAM_SPLIT_(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(*a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_TEAM_SPLIT__(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_split(*a1, *a2, *a3, a4);
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

extern void __wrap_pshmem_team_split_(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(*a1, *a2, *a3, a4);
}

extern void __wrap_pshmem_team_split__(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(*a1, *a2, *a3, a4);
}

extern void __wrap_PSHMEM_TEAM_SPLIT_(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(*a1, *a2, *a3, a4);
}

extern void __wrap_PSHMEM_TEAM_SPLIT__(shmem_team_t * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_split(*a1, *a2, *a3, a4);
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

extern void __wrap_shmem_team_create_strided_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(*a1, *a2, *a3, a4);
}

extern void __wrap_shmem_team_create_strided__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(*a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_TEAM_CREATE_STRIDED_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(*a1, *a2, *a3, a4);
}

extern void __wrap_SHMEM_TEAM_CREATE_STRIDED__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_shmem_team_create_strided(*a1, *a2, *a3, a4);
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

extern void __wrap_pshmem_team_create_strided_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(*a1, *a2, *a3, a4);
}

extern void __wrap_pshmem_team_create_strided__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(*a1, *a2, *a3, a4);
}

extern void __wrap_PSHMEM_TEAM_CREATE_STRIDED_(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(*a1, *a2, *a3, a4);
}

extern void __wrap_PSHMEM_TEAM_CREATE_STRIDED__(SHMEM_FINT * a1, SHMEM_FINT * a2, SHMEM_FINT * a3, shmem_team_t * a4)
{
   __wrap_pshmem_team_create_strided(*a1, *a2, *a3, a4);
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

extern void __wrap_shmem_team_free_(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}

extern void __wrap_shmem_team_free__(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}

extern void __wrap_SHMEM_TEAM_FREE_(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
}

extern void __wrap_SHMEM_TEAM_FREE__(shmem_team_t * a1)
{
   __wrap_shmem_team_free(a1);
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

extern int __wrap_shmem_team_npes_(shmem_team_t * a1)
{
   __wrap_shmem_team_npes(*a1);
}

extern int __wrap_shmem_team_npes__(shmem_team_t * a1)
{
   __wrap_shmem_team_npes(*a1);
}

extern int __wrap_SHMEM_TEAM_NPES_(shmem_team_t * a1)
{
   __wrap_shmem_team_npes(*a1);
}

extern int __wrap_SHMEM_TEAM_NPES__(shmem_team_t * a1)
{
   __wrap_shmem_team_npes(*a1);
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

extern int __wrap_shmem_team_mype_(shmem_team_t * a1)
{
   __wrap_shmem_team_mype(*a1);
}

extern int __wrap_shmem_team_mype__(shmem_team_t * a1)
{
   __wrap_shmem_team_mype(*a1);
}

extern int __wrap_SHMEM_TEAM_MYPE_(shmem_team_t * a1)
{
   __wrap_shmem_team_mype(*a1);
}

extern int __wrap_SHMEM_TEAM_MYPE__(shmem_team_t * a1)
{
   __wrap_shmem_team_mype(*a1);
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

extern int __wrap_shmem_team_translate_pe_(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_shmem_team_translate_pe(*a1, *a2, *a3);
}

extern int __wrap_shmem_team_translate_pe__(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_shmem_team_translate_pe(*a1, *a2, *a3);
}

extern int __wrap_SHMEM_TEAM_TRANSLATE_PE_(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_shmem_team_translate_pe(*a1, *a2, *a3);
}

extern int __wrap_SHMEM_TEAM_TRANSLATE_PE__(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_shmem_team_translate_pe(*a1, *a2, *a3);
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

extern int __wrap_pshmem_team_translate_pe_(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_pshmem_team_translate_pe(*a1, *a2, *a3);
}

extern int __wrap_pshmem_team_translate_pe__(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_pshmem_team_translate_pe(*a1, *a2, *a3);
}

extern int __wrap_PSHMEM_TEAM_TRANSLATE_PE_(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_pshmem_team_translate_pe(*a1, *a2, *a3);
}

extern int __wrap_PSHMEM_TEAM_TRANSLATE_PE__(shmem_team_t * a1, SHMEM_FINT * a2, shmem_team_t * a3)
{
   __wrap_pshmem_team_translate_pe(*a1, *a2, *a3);
}


/**********************************************************
   start_pes
 **********************************************************/

extern void  __real_start_pes(int a1) ;
extern void  __wrap_start_pes(int a1)  {

  MPI_Init();
  TAU_PROFILE_TIMER(t,"void start_pes(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_start_pes(a1);
  tau_totalnodes(1,__real_shmem_n_pes());
  TAU_PROFILE_SET_NODE(__real_shmem_my_pe());
  TAU_PROFILE_STOP(t);

}

extern void __wrap_start_pes_(SHMEM_FINT * a1)
{
   __wrap_start_pes(*a1);
}

extern void __wrap_start_pes__(SHMEM_FINT * a1)
{
   __wrap_start_pes(*a1);
}

extern void __wrap_START_PES_(SHMEM_FINT * a1)
{
   __wrap_start_pes(*a1);
}

extern void __wrap_START_PES__(SHMEM_FINT * a1)
{
   __wrap_start_pes(*a1);
}


/**********************************************************
   shmem_init
 **********************************************************/

extern void  __real_shmem_init() ;
extern void  __wrap_shmem_init()  {

  MPI_Init();
  TAU_PROFILE_TIMER(t,"void shmem_init(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shmem_init();
  tau_totalnodes(1,__real_shmem_n_pes());
  TAU_PROFILE_SET_NODE(__real_shmem_my_pe());
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shmem_init_()
{
   __wrap_shmem_init();
}

extern void __wrap_shmem_init__()
{
   __wrap_shmem_init();
}

extern void __wrap_SHMEM_INIT_()
{
   __wrap_shmem_init();
}

extern void __wrap_SHMEM_INIT__()
{
   __wrap_shmem_init();
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

extern void __wrap_shmem_finalize_()
{
   __wrap_shmem_finalize();
}

extern void __wrap_shmem_finalize__()
{
   __wrap_shmem_finalize();
}

extern void __wrap_SHMEM_FINALIZE_()
{
   __wrap_shmem_finalize();
}

extern void __wrap_SHMEM_FINALIZE__()
{
   __wrap_shmem_finalize();
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

extern void __wrap_shmem_global_exit_(SHMEM_FINT * a1)
{
   __wrap_shmem_global_exit(*a1);
}

extern void __wrap_shmem_global_exit__(SHMEM_FINT * a1)
{
   __wrap_shmem_global_exit(*a1);
}

extern void __wrap_SHMEM_GLOBAL_EXIT_(SHMEM_FINT * a1)
{
   __wrap_shmem_global_exit(*a1);
}

extern void __wrap_SHMEM_GLOBAL_EXIT__(SHMEM_FINT * a1)
{
   __wrap_shmem_global_exit(*a1);
}


/**********************************************************
   _num_pes
 **********************************************************/

extern int  __real__num_pes() ;
extern int  __wrap__num_pes()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int _num_pes(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__num_pes();
  TAU_PROFILE_STOP(t);
  return retval;

}

extern int __wrap__num_pes_()
{
   __wrap__num_pes();
}

extern int __wrap__num_pes__()
{
   __wrap__num_pes();
}

extern int __wrap__NUM_PES_()
{
   __wrap__num_pes();
}

extern int __wrap__NUM_PES__()
{
   __wrap__num_pes();
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

extern int __wrap_shmem_n_pes_()
{
   __wrap_shmem_n_pes();
}

extern int __wrap_shmem_n_pes__()
{
   __wrap_shmem_n_pes();
}

extern int __wrap_SHMEM_N_PES_()
{
   __wrap_shmem_n_pes();
}

extern int __wrap_SHMEM_N_PES__()
{
   __wrap_shmem_n_pes();
}


/**********************************************************
   _my_pe
 **********************************************************/

extern int  __real__my_pe() ;
extern int  __wrap__my_pe()  {

  int retval;
  TAU_PROFILE_TIMER(t,"int _my_pe(void) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  retval  =  __real__my_pe();
  TAU_PROFILE_STOP(t);
  return retval;

}

extern int __wrap__my_pe_()
{
   __wrap__my_pe();
}

extern int __wrap__my_pe__()
{
   __wrap__my_pe();
}

extern int __wrap__MY_PE_()
{
   __wrap__my_pe();
}

extern int __wrap__MY_PE__()
{
   __wrap__my_pe();
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

extern int __wrap_shmem_my_pe_()
{
   __wrap_shmem_my_pe();
}

extern int __wrap_shmem_my_pe__()
{
   __wrap_shmem_my_pe();
}

extern int __wrap_SHMEM_MY_PE_()
{
   __wrap_shmem_my_pe();
}

extern int __wrap_SHMEM_MY_PE__()
{
   __wrap_shmem_my_pe();
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

extern int __wrap_shmem_pe_accessible_(SHMEM_FINT * a1)
{
   __wrap_shmem_pe_accessible(*a1);
}

extern int __wrap_shmem_pe_accessible__(SHMEM_FINT * a1)
{
   __wrap_shmem_pe_accessible(*a1);
}

extern int __wrap_SHMEM_PE_ACCESSIBLE_(SHMEM_FINT * a1)
{
   __wrap_shmem_pe_accessible(*a1);
}

extern int __wrap_SHMEM_PE_ACCESSIBLE__(SHMEM_FINT * a1)
{
   __wrap_shmem_pe_accessible(*a1);
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

extern int __wrap_shmem_addr_accessible_(void * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
}

extern int __wrap_shmem_addr_accessible__(void * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
}

extern int __wrap_SHMEM_ADDR_ACCESSIBLE_(void * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
}

extern int __wrap_SHMEM_ADDR_ACCESSIBLE__(void * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
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

extern int __wrap_shmem_init_thread_(SHMEM_FINT * a1)
{
   __wrap_shmem_init_thread(*a1);
}

extern int __wrap_shmem_init_thread__(SHMEM_FINT * a1)
{
   __wrap_shmem_init_thread(*a1);
}

extern int __wrap_SHMEM_INIT_THREAD_(SHMEM_FINT * a1)
{
   __wrap_shmem_init_thread(*a1);
}

extern int __wrap_SHMEM_INIT_THREAD__(SHMEM_FINT * a1)
{
   __wrap_shmem_init_thread(*a1);
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

extern int __wrap_shmem_query_thread_()
{
   __wrap_shmem_query_thread();
}

extern int __wrap_shmem_query_thread__()
{
   __wrap_shmem_query_thread();
}

extern int __wrap_SHMEM_QUERY_THREAD_()
{
   __wrap_shmem_query_thread();
}

extern int __wrap_SHMEM_QUERY_THREAD__()
{
   __wrap_shmem_query_thread();
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

extern void __wrap_shmem_thread_register_()
{
   __wrap_shmem_thread_register();
}

extern void __wrap_shmem_thread_register__()
{
   __wrap_shmem_thread_register();
}

extern void __wrap_SHMEM_THREAD_REGISTER_()
{
   __wrap_shmem_thread_register();
}

extern void __wrap_SHMEM_THREAD_REGISTER__()
{
   __wrap_shmem_thread_register();
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

extern void __wrap_shmem_thread_unregister_()
{
   __wrap_shmem_thread_unregister();
}

extern void __wrap_shmem_thread_unregister__()
{
   __wrap_shmem_thread_unregister();
}

extern void __wrap_SHMEM_THREAD_UNREGISTER_()
{
   __wrap_shmem_thread_unregister();
}

extern void __wrap_SHMEM_THREAD_UNREGISTER__()
{
   __wrap_shmem_thread_unregister();
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

extern void __wrap_shmem_thread_fence_()
{
   __wrap_shmem_thread_fence();
}

extern void __wrap_shmem_thread_fence__()
{
   __wrap_shmem_thread_fence();
}

extern void __wrap_SHMEM_THREAD_FENCE_()
{
   __wrap_shmem_thread_fence();
}

extern void __wrap_SHMEM_THREAD_FENCE__()
{
   __wrap_shmem_thread_fence();
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

extern void __wrap_shmem_thread_quiet_()
{
   __wrap_shmem_thread_quiet();
}

extern void __wrap_shmem_thread_quiet__()
{
   __wrap_shmem_thread_quiet();
}

extern void __wrap_SHMEM_THREAD_QUIET_()
{
   __wrap_shmem_thread_quiet();
}

extern void __wrap_SHMEM_THREAD_QUIET__()
{
   __wrap_shmem_thread_quiet();
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

extern int __wrap_shmem_local_npes_()
{
   __wrap_shmem_local_npes();
}

extern int __wrap_shmem_local_npes__()
{
   __wrap_shmem_local_npes();
}

extern int __wrap_SHMEM_LOCAL_NPES_()
{
   __wrap_shmem_local_npes();
}

extern int __wrap_SHMEM_LOCAL_NPES__()
{
   __wrap_shmem_local_npes();
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

extern void __wrap_shmem_local_pes_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_local_pes(a1, *a2);
}

extern void __wrap_shmem_local_pes__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_local_pes(a1, *a2);
}

extern void __wrap_SHMEM_LOCAL_PES_(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_local_pes(a1, *a2);
}

extern void __wrap_SHMEM_LOCAL_PES__(SHMEM_FINT * a1, SHMEM_FINT * a2)
{
   __wrap_shmem_local_pes(a1, *a2);
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

extern void __wrap_shmem_set_cache_inv_()
{
   __wrap_shmem_set_cache_inv();
}

extern void __wrap_shmem_set_cache_inv__()
{
   __wrap_shmem_set_cache_inv();
}

extern void __wrap_SHMEM_SET_CACHE_INV_()
{
   __wrap_shmem_set_cache_inv();
}

extern void __wrap_SHMEM_SET_CACHE_INV__()
{
   __wrap_shmem_set_cache_inv();
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

extern void __wrap_shmem_set_cache_line_inv_(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
}

extern void __wrap_shmem_set_cache_line_inv__(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
}

extern void __wrap_SHMEM_SET_CACHE_LINE_INV_(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
}

extern void __wrap_SHMEM_SET_CACHE_LINE_INV__(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
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

extern void __wrap_shmem_clear_cache_inv_()
{
   __wrap_shmem_clear_cache_inv();
}

extern void __wrap_shmem_clear_cache_inv__()
{
   __wrap_shmem_clear_cache_inv();
}

extern void __wrap_SHMEM_CLEAR_CACHE_INV_()
{
   __wrap_shmem_clear_cache_inv();
}

extern void __wrap_SHMEM_CLEAR_CACHE_INV__()
{
   __wrap_shmem_clear_cache_inv();
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

extern void __wrap_shmem_clear_cache_line_inv_(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
}

extern void __wrap_shmem_clear_cache_line_inv__(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
}

extern void __wrap_SHMEM_CLEAR_CACHE_LINE_INV_(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
}

extern void __wrap_SHMEM_CLEAR_CACHE_LINE_INV__(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
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

extern void __wrap_shmem_udcflush_()
{
   __wrap_shmem_udcflush();
}

extern void __wrap_shmem_udcflush__()
{
   __wrap_shmem_udcflush();
}

extern void __wrap_SHMEM_UDCFLUSH_()
{
   __wrap_shmem_udcflush();
}

extern void __wrap_SHMEM_UDCFLUSH__()
{
   __wrap_shmem_udcflush();
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

extern void __wrap_shmem_udcflush_line_(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}

extern void __wrap_shmem_udcflush_line__(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}

extern void __wrap_SHMEM_UDCFLUSH_LINE_(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}

extern void __wrap_SHMEM_UDCFLUSH_LINE__(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}


/**********************************************************
   shfree
 **********************************************************/

extern void  __real_shfree(void * a1) ;
extern void  __wrap_shfree(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shfree(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shfree(a1);
  TAU_PROFILE_STOP(t);

}

extern void __wrap_shfree_(void * a1)
{
   __wrap_shfree(a1);
}

extern void __wrap_shfree__(void * a1)
{
   __wrap_shfree(a1);
}

extern void __wrap_SHFREE_(void * a1)
{
   __wrap_shfree(a1);
}

extern void __wrap_SHFREE__(void * a1)
{
   __wrap_shfree(a1);
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

extern void __wrap_shmem_free_(void * a1)
{
   __wrap_shmem_free(a1);
}

extern void __wrap_shmem_free__(void * a1)
{
   __wrap_shmem_free(a1);
}

extern void __wrap_SHMEM_FREE_(void * a1)
{
   __wrap_shmem_free(a1);
}

extern void __wrap_SHMEM_FREE__(void * a1)
{
   __wrap_shmem_free(a1);
}

