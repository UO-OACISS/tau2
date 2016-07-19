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
   start_pes
 **********************************************************/

extern void  __real_start_pes(int a1) ;
extern void  __wrap_start_pes(int a1)  {

  TAU_PROFILE_TIMER(t,"void start_pes(int) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_start_pes(a1);
  tau_totalnodes(1,__real_shmem_n_pes());
  TAU_PROFILE_SET_NODE(__real_shmem_my_pe());
  TAU_PROFILE_STOP(t);

}

extern void start_pes_(int * a1)
{
   __wrap_start_pes(*a1);
}

extern void start_pes__(int * a1)
{
   __wrap_start_pes(*a1);
}

extern void START_PES_(int * a1)
{
   __wrap_start_pes(*a1);
}

extern void START_PES__(int * a1)
{
   __wrap_start_pes(*a1);
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

extern void shmem_init_()
{
   __wrap_shmem_init();
}

extern void shmem_init__()
{
   __wrap_shmem_init();
}

extern void SHMEM_INIT_()
{
   __wrap_shmem_init();
}

extern void SHMEM_INIT__()
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

extern void shmem_finalize_()
{
   __wrap_shmem_finalize();
}

extern void shmem_finalize__()
{
   __wrap_shmem_finalize();
}

extern void SHMEM_FINALIZE_()
{
   __wrap_shmem_finalize();
}

extern void SHMEM_FINALIZE__()
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

extern void shmem_global_exit_(int * a1)
{
   __wrap_shmem_global_exit(*a1);
}

extern void shmem_global_exit__(int * a1)
{
   __wrap_shmem_global_exit(*a1);
}

extern void SHMEM_GLOBAL_EXIT_(int * a1)
{
   __wrap_shmem_global_exit(*a1);
}

extern void SHMEM_GLOBAL_EXIT__(int * a1)
{
   __wrap_shmem_global_exit(*a1);
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

extern int _my_pe_()
{
   __wrap__my_pe();
}

extern int _my_pe__()
{
   __wrap__my_pe();
}

extern int _MY_PE_()
{
   __wrap__my_pe();
}

extern int _MY_PE__()
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

extern int shmem_my_pe_()
{
   __wrap_shmem_my_pe();
}

extern int shmem_my_pe__()
{
   __wrap_shmem_my_pe();
}

extern int SHMEM_MY_PE_()
{
   __wrap_shmem_my_pe();
}

extern int SHMEM_MY_PE__()
{
   __wrap_shmem_my_pe();
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

extern int _num_pes_()
{
   __wrap__num_pes();
}

extern int _num_pes__()
{
   __wrap__num_pes();
}

extern int _NUM_PES_()
{
   __wrap__num_pes();
}

extern int _NUM_PES__()
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

extern int shmem_n_pes_()
{
   __wrap_shmem_n_pes();
}

extern int shmem_n_pes__()
{
   __wrap_shmem_n_pes();
}

extern int SHMEM_N_PES_()
{
   __wrap_shmem_n_pes();
}

extern int SHMEM_N_PES__()
{
   __wrap_shmem_n_pes();
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

extern void shmem_short_put_(short * a1, const short * a2, size_t * a3, int * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
}

extern void shmem_short_put__(short * a1, const short * a2, size_t * a3, int * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
}

extern void SHMEM_SHORT_PUT_(short * a1, const short * a2, size_t * a3, int * a4)
{
   __wrap_shmem_short_put(a1, a2, *a3, *a4);
}

extern void SHMEM_SHORT_PUT__(short * a1, const short * a2, size_t * a3, int * a4)
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

extern void shmem_int_put_(int * a1, const int * a2, size_t * a3, int * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
}

extern void shmem_int_put__(int * a1, const int * a2, size_t * a3, int * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
}

extern void SHMEM_INT_PUT_(int * a1, const int * a2, size_t * a3, int * a4)
{
   __wrap_shmem_int_put(a1, a2, *a3, *a4);
}

extern void SHMEM_INT_PUT__(int * a1, const int * a2, size_t * a3, int * a4)
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

extern void shmem_long_put_(long * a1, const long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
}

extern void shmem_long_put__(long * a1, const long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
}

extern void SHMEM_LONG_PUT_(long * a1, const long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_long_put(a1, a2, *a3, *a4);
}

extern void SHMEM_LONG_PUT__(long * a1, const long * a2, size_t * a3, int * a4)
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

extern void shmem_longlong_put_(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
}

extern void shmem_longlong_put__(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGLONG_PUT_(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGLONG_PUT__(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_put(a1, a2, *a3, *a4);
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

extern void shmem_longdouble_put_(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_put(a1, a2, *a3, *a4);
}

extern void shmem_longdouble_put__(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_put(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGDOUBLE_PUT_(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_put(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGDOUBLE_PUT__(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_put(a1, a2, *a3, *a4);
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

extern void shmem_double_put_(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
}

extern void shmem_double_put__(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
}

extern void SHMEM_DOUBLE_PUT_(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
}

extern void SHMEM_DOUBLE_PUT__(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_put(a1, a2, *a3, *a4);
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

extern void shmem_float_put_(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
}

extern void shmem_float_put__(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
}

extern void SHMEM_FLOAT_PUT_(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
}

extern void SHMEM_FLOAT_PUT__(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_put(a1, a2, *a3, *a4);
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

extern void shmem_putmem_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}

extern void shmem_putmem__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}

extern void SHMEM_PUTMEM_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
}

extern void SHMEM_PUTMEM__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_putmem(a1, a2, *a3, *a4);
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

extern void shmem_put32_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
}

extern void shmem_put32__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
}

extern void SHMEM_PUT32_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put32(a1, a2, *a3, *a4);
}

extern void SHMEM_PUT32__(void * a1, const void * a2, size_t * a3, int * a4)
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

extern void shmem_put64_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
}

extern void shmem_put64__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
}

extern void SHMEM_PUT64_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put64(a1, a2, *a3, *a4);
}

extern void SHMEM_PUT64__(void * a1, const void * a2, size_t * a3, int * a4)
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

extern void shmem_put128_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
}

extern void shmem_put128__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
}

extern void SHMEM_PUT128_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
}

extern void SHMEM_PUT128__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_put128(a1, a2, *a3, *a4);
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

extern void shmem_short_get_(short * a1, const short * a2, size_t * a3, int * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
}

extern void shmem_short_get__(short * a1, const short * a2, size_t * a3, int * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
}

extern void SHMEM_SHORT_GET_(short * a1, const short * a2, size_t * a3, int * a4)
{
   __wrap_shmem_short_get(a1, a2, *a3, *a4);
}

extern void SHMEM_SHORT_GET__(short * a1, const short * a2, size_t * a3, int * a4)
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

extern void shmem_int_get_(int * a1, const int * a2, size_t * a3, int * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
}

extern void shmem_int_get__(int * a1, const int * a2, size_t * a3, int * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
}

extern void SHMEM_INT_GET_(int * a1, const int * a2, size_t * a3, int * a4)
{
   __wrap_shmem_int_get(a1, a2, *a3, *a4);
}

extern void SHMEM_INT_GET__(int * a1, const int * a2, size_t * a3, int * a4)
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

extern void shmem_long_get_(long * a1, const long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
}

extern void shmem_long_get__(long * a1, const long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
}

extern void SHMEM_LONG_GET_(long * a1, const long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_long_get(a1, a2, *a3, *a4);
}

extern void SHMEM_LONG_GET__(long * a1, const long * a2, size_t * a3, int * a4)
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

extern void shmem_longlong_get_(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
}

extern void shmem_longlong_get__(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGLONG_GET_(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGLONG_GET__(long long * a1, const long long * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longlong_get(a1, a2, *a3, *a4);
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

extern void shmem_longdouble_get_(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_get(a1, a2, *a3, *a4);
}

extern void shmem_longdouble_get__(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_get(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGDOUBLE_GET_(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_get(a1, a2, *a3, *a4);
}

extern void SHMEM_LONGDOUBLE_GET__(long double * a1, const long double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_longdouble_get(a1, a2, *a3, *a4);
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

extern void shmem_double_get_(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
}

extern void shmem_double_get__(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
}

extern void SHMEM_DOUBLE_GET_(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
}

extern void SHMEM_DOUBLE_GET__(double * a1, const double * a2, size_t * a3, int * a4)
{
   __wrap_shmem_double_get(a1, a2, *a3, *a4);
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

extern void shmem_float_get_(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
}

extern void shmem_float_get__(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
}

extern void SHMEM_FLOAT_GET_(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
}

extern void SHMEM_FLOAT_GET__(float * a1, const float * a2, size_t * a3, int * a4)
{
   __wrap_shmem_float_get(a1, a2, *a3, *a4);
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

extern void shmem_getmem_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}

extern void shmem_getmem__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}

extern void SHMEM_GETMEM_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
}

extern void SHMEM_GETMEM__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_getmem(a1, a2, *a3, *a4);
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

extern void shmem_get32_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
}

extern void shmem_get32__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
}

extern void SHMEM_GET32_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get32(a1, a2, *a3, *a4);
}

extern void SHMEM_GET32__(void * a1, const void * a2, size_t * a3, int * a4)
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

extern void shmem_get64_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
}

extern void shmem_get64__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
}

extern void SHMEM_GET64_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get64(a1, a2, *a3, *a4);
}

extern void SHMEM_GET64__(void * a1, const void * a2, size_t * a3, int * a4)
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

extern void shmem_get128_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
}

extern void shmem_get128__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
}

extern void SHMEM_GET128_(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
}

extern void SHMEM_GET128__(void * a1, const void * a2, size_t * a3, int * a4)
{
   __wrap_shmem_get128(a1, a2, *a3, *a4);
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

extern void shmem_char_p_(char * a1, char * a2, int * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
}

extern void shmem_char_p__(char * a1, char * a2, int * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
}

extern void SHMEM_CHAR_P_(char * a1, char * a2, int * a3)
{
   __wrap_shmem_char_p(a1, *a2, *a3);
}

extern void SHMEM_CHAR_P__(char * a1, char * a2, int * a3)
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

extern void shmem_short_p_(short * a1, short * a2, int * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
}

extern void shmem_short_p__(short * a1, short * a2, int * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
}

extern void SHMEM_SHORT_P_(short * a1, short * a2, int * a3)
{
   __wrap_shmem_short_p(a1, *a2, *a3);
}

extern void SHMEM_SHORT_P__(short * a1, short * a2, int * a3)
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

extern void shmem_int_p_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
}

extern void shmem_int_p__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
}

extern void SHMEM_INT_P_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_p(a1, *a2, *a3);
}

extern void SHMEM_INT_P__(int * a1, int * a2, int * a3)
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

extern void shmem_long_p_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
}

extern void shmem_long_p__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
}

extern void SHMEM_LONG_P_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_p(a1, *a2, *a3);
}

extern void SHMEM_LONG_P__(long * a1, long * a2, int * a3)
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

extern void shmem_longlong_p_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
}

extern void shmem_longlong_p__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
}

extern void SHMEM_LONGLONG_P_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_p(a1, *a2, *a3);
}

extern void SHMEM_LONGLONG_P__(long long * a1, long long * a2, int * a3)
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

extern void shmem_float_p_(float * a1, float * a2, int * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
}

extern void shmem_float_p__(float * a1, float * a2, int * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
}

extern void SHMEM_FLOAT_P_(float * a1, float * a2, int * a3)
{
   __wrap_shmem_float_p(a1, *a2, *a3);
}

extern void SHMEM_FLOAT_P__(float * a1, float * a2, int * a3)
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

extern void shmem_double_p_(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
}

extern void shmem_double_p__(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
}

extern void SHMEM_DOUBLE_P_(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
}

extern void SHMEM_DOUBLE_P__(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_p(a1, *a2, *a3);
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

extern void shmem_longdouble_p_(long double * a1, long double * a2, int * a3)
{
   __wrap_shmem_longdouble_p(a1, *a2, *a3);
}

extern void shmem_longdouble_p__(long double * a1, long double * a2, int * a3)
{
   __wrap_shmem_longdouble_p(a1, *a2, *a3);
}

extern void SHMEM_LONGDOUBLE_P_(long double * a1, long double * a2, int * a3)
{
   __wrap_shmem_longdouble_p(a1, *a2, *a3);
}

extern void SHMEM_LONGDOUBLE_P__(long double * a1, long double * a2, int * a3)
{
   __wrap_shmem_longdouble_p(a1, *a2, *a3);
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

extern char shmem_char_g_(char * a1, int * a2)
{
   __wrap_shmem_char_g(a1, *a2);
}

extern char shmem_char_g__(char * a1, int * a2)
{
   __wrap_shmem_char_g(a1, *a2);
}

extern char SHMEM_CHAR_G_(char * a1, int * a2)
{
   __wrap_shmem_char_g(a1, *a2);
}

extern char SHMEM_CHAR_G__(char * a1, int * a2)
{
   __wrap_shmem_char_g(a1, *a2);
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

extern short shmem_short_g_(short * a1, int * a2)
{
   __wrap_shmem_short_g(a1, *a2);
}

extern short shmem_short_g__(short * a1, int * a2)
{
   __wrap_shmem_short_g(a1, *a2);
}

extern short SHMEM_SHORT_G_(short * a1, int * a2)
{
   __wrap_shmem_short_g(a1, *a2);
}

extern short SHMEM_SHORT_G__(short * a1, int * a2)
{
   __wrap_shmem_short_g(a1, *a2);
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

extern int shmem_int_g_(int * a1, int * a2)
{
   __wrap_shmem_int_g(a1, *a2);
}

extern int shmem_int_g__(int * a1, int * a2)
{
   __wrap_shmem_int_g(a1, *a2);
}

extern int SHMEM_INT_G_(int * a1, int * a2)
{
   __wrap_shmem_int_g(a1, *a2);
}

extern int SHMEM_INT_G__(int * a1, int * a2)
{
   __wrap_shmem_int_g(a1, *a2);
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

extern long shmem_long_g_(long * a1, int * a2)
{
   __wrap_shmem_long_g(a1, *a2);
}

extern long shmem_long_g__(long * a1, int * a2)
{
   __wrap_shmem_long_g(a1, *a2);
}

extern long SHMEM_LONG_G_(long * a1, int * a2)
{
   __wrap_shmem_long_g(a1, *a2);
}

extern long SHMEM_LONG_G__(long * a1, int * a2)
{
   __wrap_shmem_long_g(a1, *a2);
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

extern long long shmem_longlong_g_(long long * a1, int * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
}

extern long long shmem_longlong_g__(long long * a1, int * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
}

extern long long SHMEM_LONGLONG_G_(long long * a1, int * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
}

extern long long SHMEM_LONGLONG_G__(long long * a1, int * a2)
{
   __wrap_shmem_longlong_g(a1, *a2);
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

extern float shmem_float_g_(float * a1, int * a2)
{
   __wrap_shmem_float_g(a1, *a2);
}

extern float shmem_float_g__(float * a1, int * a2)
{
   __wrap_shmem_float_g(a1, *a2);
}

extern float SHMEM_FLOAT_G_(float * a1, int * a2)
{
   __wrap_shmem_float_g(a1, *a2);
}

extern float SHMEM_FLOAT_G__(float * a1, int * a2)
{
   __wrap_shmem_float_g(a1, *a2);
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

extern double shmem_double_g_(double * a1, int * a2)
{
   __wrap_shmem_double_g(a1, *a2);
}

extern double shmem_double_g__(double * a1, int * a2)
{
   __wrap_shmem_double_g(a1, *a2);
}

extern double SHMEM_DOUBLE_G_(double * a1, int * a2)
{
   __wrap_shmem_double_g(a1, *a2);
}

extern double SHMEM_DOUBLE_G__(double * a1, int * a2)
{
   __wrap_shmem_double_g(a1, *a2);
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

extern long double shmem_longdouble_g_(long double * a1, int * a2)
{
   __wrap_shmem_longdouble_g(a1, *a2);
}

extern long double shmem_longdouble_g__(long double * a1, int * a2)
{
   __wrap_shmem_longdouble_g(a1, *a2);
}

extern long double SHMEM_LONGDOUBLE_G_(long double * a1, int * a2)
{
   __wrap_shmem_longdouble_g(a1, *a2);
}

extern long double SHMEM_LONGDOUBLE_G__(long double * a1, int * a2)
{
   __wrap_shmem_longdouble_g(a1, *a2);
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

extern void shmem_double_iput_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_double_iput__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_DOUBLE_IPUT_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_DOUBLE_IPUT__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_float_iput_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_float_iput__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_FLOAT_IPUT_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_FLOAT_IPUT__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_int_iput_(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_int_iput__(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_INT_IPUT_(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_INT_IPUT__(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_iput32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_iput32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IPUT32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IPUT32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
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

extern void shmem_iput64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_iput64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IPUT64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IPUT64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
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

extern void shmem_iput128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_iput128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IPUT128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IPUT128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iput128(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_long_iput_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_long_iput__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONG_IPUT_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONG_IPUT__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_longdouble_iput_(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_longdouble_iput__(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGDOUBLE_IPUT_(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGDOUBLE_IPUT__(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_longlong_iput_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_longlong_iput__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGLONG_IPUT_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGLONG_IPUT__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_short_iput_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_short_iput__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_SHORT_IPUT_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_SHORT_IPUT__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iput(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_double_iget_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_double_iget__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_DOUBLE_IGET_(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_DOUBLE_IGET__(double * a1, const double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_double_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_float_iget_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_float_iget__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_FLOAT_IGET_(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_FLOAT_IGET__(float * a1, const float * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_float_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_int_iget_(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_int_iget__(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_INT_IGET_(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_INT_IGET__(int * a1, const int * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_int_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_iget32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_iget32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IGET32_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget32(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IGET32__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
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

extern void shmem_iget64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_iget64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IGET64_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget64(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IGET64__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
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

extern void shmem_iget128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_iget128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IGET128_(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_IGET128__(void * a1, const void * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_iget128(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_long_iget_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_long_iget__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONG_IGET_(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONG_IGET__(long * a1, const long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_long_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_longdouble_iget_(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_longdouble_iget__(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGDOUBLE_IGET_(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGDOUBLE_IGET__(long double * a1, const long double * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longdouble_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_longlong_iget_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_longlong_iget__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGLONG_IGET_(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_LONGLONG_IGET__(long long * a1, const long long * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_longlong_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_short_iget_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void shmem_short_iget__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_SHORT_IGET_(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
}

extern void SHMEM_SHORT_IGET__(short * a1, const short * a2, ptrdiff_t * a3, ptrdiff_t * a4, size_t * a5, int * a6)
{
   __wrap_shmem_short_iget(a1, a2, *a3, *a4, *a5, *a6);
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

extern void shmem_barrier_all_()
{
   __wrap_shmem_barrier_all();
}

extern void shmem_barrier_all__()
{
   __wrap_shmem_barrier_all();
}

extern void SHMEM_BARRIER_ALL_()
{
   __wrap_shmem_barrier_all();
}

extern void SHMEM_BARRIER_ALL__()
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

extern void shmem_barrier_(int * a1, int * a2, int * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
}

extern void shmem_barrier__(int * a1, int * a2, int * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
}

extern void SHMEM_BARRIER_(int * a1, int * a2, int * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
}

extern void SHMEM_BARRIER__(int * a1, int * a2, int * a3, long * a4)
{
   __wrap_shmem_barrier(*a1, *a2, *a3, a4);
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

extern void shmem_fence_()
{
   __wrap_shmem_fence();
}

extern void shmem_fence__()
{
   __wrap_shmem_fence();
}

extern void SHMEM_FENCE_()
{
   __wrap_shmem_fence();
}

extern void SHMEM_FENCE__()
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

extern void shmem_quiet_()
{
   __wrap_shmem_quiet();
}

extern void shmem_quiet__()
{
   __wrap_shmem_quiet();
}

extern void SHMEM_QUIET_()
{
   __wrap_shmem_quiet();
}

extern void SHMEM_QUIET__()
{
   __wrap_shmem_quiet();
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

extern int shmem_pe_accessible_(int * a1)
{
   __wrap_shmem_pe_accessible(*a1);
}

extern int shmem_pe_accessible__(int * a1)
{
   __wrap_shmem_pe_accessible(*a1);
}

extern int SHMEM_PE_ACCESSIBLE_(int * a1)
{
   __wrap_shmem_pe_accessible(*a1);
}

extern int SHMEM_PE_ACCESSIBLE__(int * a1)
{
   __wrap_shmem_pe_accessible(*a1);
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

extern int shmem_addr_accessible_(const void * a1, int * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
}

extern int shmem_addr_accessible__(const void * a1, int * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
}

extern int SHMEM_ADDR_ACCESSIBLE_(const void * a1, int * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
}

extern int SHMEM_ADDR_ACCESSIBLE__(const void * a1, int * a2)
{
   __wrap_shmem_addr_accessible(a1, *a2);
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

extern void shfree_(void * a1)
{
   __wrap_shfree(a1);
}

extern void shfree__(void * a1)
{
   __wrap_shfree(a1);
}

extern void SHFREE_(void * a1)
{
   __wrap_shfree(a1);
}

extern void SHFREE__(void * a1)
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

extern void shmem_free_(void * a1)
{
   __wrap_shmem_free(a1);
}

extern void shmem_free__(void * a1)
{
   __wrap_shmem_free(a1);
}

extern void SHMEM_FREE_(void * a1)
{
   __wrap_shmem_free(a1);
}

extern void SHMEM_FREE__(void * a1)
{
   __wrap_shmem_free(a1);
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

extern void shmem_long_wait_until_(long * a1, int * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
}

extern void shmem_long_wait_until__(long * a1, int * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
}

extern void SHMEM_LONG_WAIT_UNTIL_(long * a1, int * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
}

extern void SHMEM_LONG_WAIT_UNTIL__(long * a1, int * a2, long * a3)
{
   __wrap_shmem_long_wait_until(a1, *a2, *a3);
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

extern void shmem_short_wait_until_(short * a1, int * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
}

extern void shmem_short_wait_until__(short * a1, int * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
}

extern void SHMEM_SHORT_WAIT_UNTIL_(short * a1, int * a2, short * a3)
{
   __wrap_shmem_short_wait_until(a1, *a2, *a3);
}

extern void SHMEM_SHORT_WAIT_UNTIL__(short * a1, int * a2, short * a3)
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

extern void shmem_int_wait_until_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
}

extern void shmem_int_wait_until__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
}

extern void SHMEM_INT_WAIT_UNTIL_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
}

extern void SHMEM_INT_WAIT_UNTIL__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_wait_until(a1, *a2, *a3);
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

extern void shmem_longlong_wait_until_(long long * a1, int * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
}

extern void shmem_longlong_wait_until__(long long * a1, int * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
}

extern void SHMEM_LONGLONG_WAIT_UNTIL_(long long * a1, int * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
}

extern void SHMEM_LONGLONG_WAIT_UNTIL__(long long * a1, int * a2, long long * a3)
{
   __wrap_shmem_longlong_wait_until(a1, *a2, *a3);
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

extern void shmem_wait_until_(long * a1, int * a2, long * a3)
{
   __wrap_shmem_wait_until(a1, *a2, *a3);
}

extern void shmem_wait_until__(long * a1, int * a2, long * a3)
{
   __wrap_shmem_wait_until(a1, *a2, *a3);
}

extern void SHMEM_WAIT_UNTIL_(long * a1, int * a2, long * a3)
{
   __wrap_shmem_wait_until(a1, *a2, *a3);
}

extern void SHMEM_WAIT_UNTIL__(long * a1, int * a2, long * a3)
{
   __wrap_shmem_wait_until(a1, *a2, *a3);
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

extern void shmem_long_wait_(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
}

extern void shmem_long_wait__(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
}

extern void SHMEM_LONG_WAIT_(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
}

extern void SHMEM_LONG_WAIT__(long * a1, long * a2)
{
   __wrap_shmem_long_wait(a1, *a2);
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

extern void shmem_short_wait_(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
}

extern void shmem_short_wait__(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
}

extern void SHMEM_SHORT_WAIT_(short * a1, short * a2)
{
   __wrap_shmem_short_wait(a1, *a2);
}

extern void SHMEM_SHORT_WAIT__(short * a1, short * a2)
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

extern void shmem_int_wait_(int * a1, int * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
}

extern void shmem_int_wait__(int * a1, int * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
}

extern void SHMEM_INT_WAIT_(int * a1, int * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
}

extern void SHMEM_INT_WAIT__(int * a1, int * a2)
{
   __wrap_shmem_int_wait(a1, *a2);
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

extern void shmem_longlong_wait_(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
}

extern void shmem_longlong_wait__(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
}

extern void SHMEM_LONGLONG_WAIT_(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
}

extern void SHMEM_LONGLONG_WAIT__(long long * a1, long long * a2)
{
   __wrap_shmem_longlong_wait(a1, *a2);
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

extern void shmem_wait_(long * a1, long * a2)
{
   __wrap_shmem_wait(a1, *a2);
}

extern void shmem_wait__(long * a1, long * a2)
{
   __wrap_shmem_wait(a1, *a2);
}

extern void SHMEM_WAIT_(long * a1, long * a2)
{
   __wrap_shmem_wait(a1, *a2);
}

extern void SHMEM_WAIT__(long * a1, long * a2)
{
   __wrap_shmem_wait(a1, *a2);
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

extern long shmem_long_swap_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
}

extern long shmem_long_swap__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
}

extern long SHMEM_LONG_SWAP_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
}

extern long SHMEM_LONG_SWAP__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_swap(a1, *a2, *a3);
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

extern int shmem_int_swap_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
}

extern int shmem_int_swap__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
}

extern int SHMEM_INT_SWAP_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
}

extern int SHMEM_INT_SWAP__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_swap(a1, *a2, *a3);
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

extern long long shmem_longlong_swap_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
}

extern long long shmem_longlong_swap__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
}

extern long long SHMEM_LONGLONG_SWAP_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_swap(a1, *a2, *a3);
}

extern long long SHMEM_LONGLONG_SWAP__(long long * a1, long long * a2, int * a3)
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

extern float shmem_float_swap_(float * a1, float * a2, int * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
}

extern float shmem_float_swap__(float * a1, float * a2, int * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
}

extern float SHMEM_FLOAT_SWAP_(float * a1, float * a2, int * a3)
{
   __wrap_shmem_float_swap(a1, *a2, *a3);
}

extern float SHMEM_FLOAT_SWAP__(float * a1, float * a2, int * a3)
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

extern double shmem_double_swap_(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
}

extern double shmem_double_swap__(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
}

extern double SHMEM_DOUBLE_SWAP_(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
}

extern double SHMEM_DOUBLE_SWAP__(double * a1, double * a2, int * a3)
{
   __wrap_shmem_double_swap(a1, *a2, *a3);
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

extern long shmem_swap_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_swap(a1, *a2, *a3);
}

extern long shmem_swap__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_swap(a1, *a2, *a3);
}

extern long SHMEM_SWAP_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_swap(a1, *a2, *a3);
}

extern long SHMEM_SWAP__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_swap(a1, *a2, *a3);
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

extern long shmem_long_cswap_(long * a1, long * a2, long * a3, int * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
}

extern long shmem_long_cswap__(long * a1, long * a2, long * a3, int * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
}

extern long SHMEM_LONG_CSWAP_(long * a1, long * a2, long * a3, int * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
}

extern long SHMEM_LONG_CSWAP__(long * a1, long * a2, long * a3, int * a4)
{
   __wrap_shmem_long_cswap(a1, *a2, *a3, *a4);
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

extern int shmem_int_cswap_(int * a1, int * a2, int * a3, int * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
}

extern int shmem_int_cswap__(int * a1, int * a2, int * a3, int * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
}

extern int SHMEM_INT_CSWAP_(int * a1, int * a2, int * a3, int * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
}

extern int SHMEM_INT_CSWAP__(int * a1, int * a2, int * a3, int * a4)
{
   __wrap_shmem_int_cswap(a1, *a2, *a3, *a4);
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

extern long long shmem_longlong_cswap_(long long * a1, long long * a2, long long * a3, int * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
}

extern long long shmem_longlong_cswap__(long long * a1, long long * a2, long long * a3, int * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
}

extern long long SHMEM_LONGLONG_CSWAP_(long long * a1, long long * a2, long long * a3, int * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
}

extern long long SHMEM_LONGLONG_CSWAP__(long long * a1, long long * a2, long long * a3, int * a4)
{
   __wrap_shmem_longlong_cswap(a1, *a2, *a3, *a4);
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

extern long shmem_long_fadd_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
}

extern long shmem_long_fadd__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
}

extern long SHMEM_LONG_FADD_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
}

extern long SHMEM_LONG_FADD__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_fadd(a1, *a2, *a3);
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

extern int shmem_int_fadd_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
}

extern int shmem_int_fadd__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
}

extern int SHMEM_INT_FADD_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
}

extern int SHMEM_INT_FADD__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_fadd(a1, *a2, *a3);
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

extern long long shmem_longlong_fadd_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
}

extern long long shmem_longlong_fadd__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
}

extern long long SHMEM_LONGLONG_FADD_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
}

extern long long SHMEM_LONGLONG_FADD__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_fadd(a1, *a2, *a3);
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

extern long shmem_long_finc_(long * a1, int * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
}

extern long shmem_long_finc__(long * a1, int * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
}

extern long SHMEM_LONG_FINC_(long * a1, int * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
}

extern long SHMEM_LONG_FINC__(long * a1, int * a2)
{
   __wrap_shmem_long_finc(a1, *a2);
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

extern int shmem_int_finc_(int * a1, int * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
}

extern int shmem_int_finc__(int * a1, int * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
}

extern int SHMEM_INT_FINC_(int * a1, int * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
}

extern int SHMEM_INT_FINC__(int * a1, int * a2)
{
   __wrap_shmem_int_finc(a1, *a2);
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

extern long long shmem_longlong_finc_(long long * a1, int * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
}

extern long long shmem_longlong_finc__(long long * a1, int * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
}

extern long long SHMEM_LONGLONG_FINC_(long long * a1, int * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
}

extern long long SHMEM_LONGLONG_FINC__(long long * a1, int * a2)
{
   __wrap_shmem_longlong_finc(a1, *a2);
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

extern void shmem_long_add_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
}

extern void shmem_long_add__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
}

extern void SHMEM_LONG_ADD_(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
}

extern void SHMEM_LONG_ADD__(long * a1, long * a2, int * a3)
{
   __wrap_shmem_long_add(a1, *a2, *a3);
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

extern void shmem_int_add_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
}

extern void shmem_int_add__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
}

extern void SHMEM_INT_ADD_(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
}

extern void SHMEM_INT_ADD__(int * a1, int * a2, int * a3)
{
   __wrap_shmem_int_add(a1, *a2, *a3);
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

extern void shmem_longlong_add_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
}

extern void shmem_longlong_add__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
}

extern void SHMEM_LONGLONG_ADD_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
}

extern void SHMEM_LONGLONG_ADD__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmem_longlong_add(a1, *a2, *a3);
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

extern void shmem_long_inc_(long * a1, int * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
}

extern void shmem_long_inc__(long * a1, int * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
}

extern void SHMEM_LONG_INC_(long * a1, int * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
}

extern void SHMEM_LONG_INC__(long * a1, int * a2)
{
   __wrap_shmem_long_inc(a1, *a2);
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

extern void shmem_int_inc_(int * a1, int * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
}

extern void shmem_int_inc__(int * a1, int * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
}

extern void SHMEM_INT_INC_(int * a1, int * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
}

extern void SHMEM_INT_INC__(int * a1, int * a2)
{
   __wrap_shmem_int_inc(a1, *a2);
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

extern void shmem_longlong_inc_(long long * a1, int * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
}

extern void shmem_longlong_inc__(long long * a1, int * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
}

extern void SHMEM_LONGLONG_INC_(long long * a1, int * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
}

extern void SHMEM_LONGLONG_INC__(long long * a1, int * a2)
{
   __wrap_shmem_longlong_inc(a1, *a2);
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

extern void shmem_clear_cache_inv_()
{
   __wrap_shmem_clear_cache_inv();
}

extern void shmem_clear_cache_inv__()
{
   __wrap_shmem_clear_cache_inv();
}

extern void SHMEM_CLEAR_CACHE_INV_()
{
   __wrap_shmem_clear_cache_inv();
}

extern void SHMEM_CLEAR_CACHE_INV__()
{
   __wrap_shmem_clear_cache_inv();
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

extern void shmem_set_cache_inv_()
{
   __wrap_shmem_set_cache_inv();
}

extern void shmem_set_cache_inv__()
{
   __wrap_shmem_set_cache_inv();
}

extern void SHMEM_SET_CACHE_INV_()
{
   __wrap_shmem_set_cache_inv();
}

extern void SHMEM_SET_CACHE_INV__()
{
   __wrap_shmem_set_cache_inv();
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

extern void shmem_clear_cache_line_inv_(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
}

extern void shmem_clear_cache_line_inv__(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
}

extern void SHMEM_CLEAR_CACHE_LINE_INV_(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
}

extern void SHMEM_CLEAR_CACHE_LINE_INV__(void * a1)
{
   __wrap_shmem_clear_cache_line_inv(a1);
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

extern void shmem_set_cache_line_inv_(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
}

extern void shmem_set_cache_line_inv__(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
}

extern void SHMEM_SET_CACHE_LINE_INV_(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
}

extern void SHMEM_SET_CACHE_LINE_INV__(void * a1)
{
   __wrap_shmem_set_cache_line_inv(a1);
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

extern void shmem_udcflush_()
{
   __wrap_shmem_udcflush();
}

extern void shmem_udcflush__()
{
   __wrap_shmem_udcflush();
}

extern void SHMEM_UDCFLUSH_()
{
   __wrap_shmem_udcflush();
}

extern void SHMEM_UDCFLUSH__()
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

extern void shmem_udcflush_line_(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}

extern void shmem_udcflush_line__(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}

extern void SHMEM_UDCFLUSH_LINE_(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
}

extern void SHMEM_UDCFLUSH_LINE__(void * a1)
{
   __wrap_shmem_udcflush_line(a1);
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

extern void shmem_long_sum_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_sum_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_SUM_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_SUM_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_complexd_sum_to_all_(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_complexd_sum_to_all__(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXD_SUM_TO_ALL_(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXD_SUM_TO_ALL__(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_complexf_sum_to_all_(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_complexf_sum_to_all__(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXF_SUM_TO_ALL_(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXF_SUM_TO_ALL__(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_double_sum_to_all_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_double_sum_to_all__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_SUM_TO_ALL_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_SUM_TO_ALL__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_float_sum_to_all_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_float_sum_to_all__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_SUM_TO_ALL_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_SUM_TO_ALL__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_sum_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_sum_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_SUM_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_SUM_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longdouble_sum_to_all_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longdouble_sum_to_all__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_SUM_TO_ALL_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_SUM_TO_ALL__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_sum_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_sum_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_SUM_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_SUM_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_sum_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_sum_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_SUM_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_SUM_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_sum_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_complexd_prod_to_all_(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_complexd_prod_to_all__(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXD_PROD_TO_ALL_(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXD_PROD_TO_ALL__(double _Complex * a1, double _Complex * a2, int * a3, int * a4, int * a5, int * a6, double _Complex * a7, long * a8)
{
   __wrap_shmem_complexd_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_complexf_prod_to_all_(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_complexf_prod_to_all__(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXF_PROD_TO_ALL_(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_COMPLEXF_PROD_TO_ALL__(float _Complex * a1, float _Complex * a2, int * a3, int * a4, int * a5, int * a6, float _Complex * a7, long * a8)
{
   __wrap_shmem_complexf_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_double_prod_to_all_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_double_prod_to_all__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_PROD_TO_ALL_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_PROD_TO_ALL__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_float_prod_to_all_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_float_prod_to_all__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_PROD_TO_ALL_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_PROD_TO_ALL__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_prod_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_prod_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_PROD_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_PROD_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_long_prod_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_prod_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_PROD_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_PROD_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longdouble_prod_to_all_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longdouble_prod_to_all__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_PROD_TO_ALL_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_PROD_TO_ALL__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_prod_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_prod_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_PROD_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_PROD_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_prod_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_prod_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_PROD_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_PROD_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_prod_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_and_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_and_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_AND_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_AND_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_long_and_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_and_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_AND_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_AND_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_and_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_and_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_AND_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_AND_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_and_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_and_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_AND_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_AND_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_and_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_or_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_or_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_OR_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_OR_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_long_or_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_or_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_OR_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_OR_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_or_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_or_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_OR_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_OR_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_or_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_or_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_OR_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_OR_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_or_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_xor_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_xor_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_XOR_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_XOR_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_long_xor_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_xor_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_XOR_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_XOR_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_xor_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_xor_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_XOR_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_XOR_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_xor_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_xor_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_XOR_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_XOR_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_xor_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_max_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_max_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_MAX_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_MAX_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_long_max_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_max_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_MAX_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_MAX_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_max_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_max_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_MAX_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_MAX_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_max_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_max_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_MAX_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_MAX_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longdouble_max_to_all_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longdouble_max_to_all__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_MAX_TO_ALL_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_MAX_TO_ALL__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_float_max_to_all_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_float_max_to_all__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_MAX_TO_ALL_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_MAX_TO_ALL__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_double_max_to_all_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_double_max_to_all__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_MAX_TO_ALL_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_MAX_TO_ALL__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_max_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_int_min_to_all_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_int_min_to_all__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_MIN_TO_ALL_(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_INT_MIN_TO_ALL__(int * a1, int * a2, int * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_int_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_long_min_to_all_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_long_min_to_all__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_MIN_TO_ALL_(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONG_MIN_TO_ALL__(long * a1, long * a2, int * a3, int * a4, int * a5, int * a6, long * a7, long * a8)
{
   __wrap_shmem_long_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longlong_min_to_all_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longlong_min_to_all__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_MIN_TO_ALL_(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGLONG_MIN_TO_ALL__(long long * a1, long long * a2, int * a3, int * a4, int * a5, int * a6, long long * a7, long * a8)
{
   __wrap_shmem_longlong_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_short_min_to_all_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_short_min_to_all__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_MIN_TO_ALL_(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_SHORT_MIN_TO_ALL__(short * a1, short * a2, int * a3, int * a4, int * a5, int * a6, short * a7, long * a8)
{
   __wrap_shmem_short_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_longdouble_min_to_all_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_longdouble_min_to_all__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_MIN_TO_ALL_(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_LONGDOUBLE_MIN_TO_ALL__(long double * a1, long double * a2, int * a3, int * a4, int * a5, int * a6, long double * a7, long * a8)
{
   __wrap_shmem_longdouble_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_float_min_to_all_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_float_min_to_all__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_MIN_TO_ALL_(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_FLOAT_MIN_TO_ALL__(float * a1, float * a2, int * a3, int * a4, int * a5, int * a6, float * a7, long * a8)
{
   __wrap_shmem_float_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_double_min_to_all_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void shmem_double_min_to_all__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_MIN_TO_ALL_(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
}

extern void SHMEM_DOUBLE_MIN_TO_ALL__(double * a1, double * a2, int * a3, int * a4, int * a5, int * a6, double * a7, long * a8)
{
   __wrap_shmem_double_min_to_all(a1, a2, *a3, *a4, *a5, *a6, a7, a8);
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

extern void shmem_broadcast64_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void shmem_broadcast64__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void SHMEM_BROADCAST64_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void SHMEM_BROADCAST64__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast64(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
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

extern void shmem_broadcast32_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void shmem_broadcast32__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void SHMEM_BROADCAST32_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
}

extern void SHMEM_BROADCAST32__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, int * a7, long * a8)
{
   __wrap_shmem_broadcast32(a1, a2, *a3, *a4, *a5, *a6, *a7, a8);
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

extern void shmem_fcollect64_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void shmem_fcollect64__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_FCOLLECT64_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_FCOLLECT64__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect64(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void shmem_fcollect32_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void shmem_fcollect32__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_FCOLLECT32_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_FCOLLECT32__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_fcollect32(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void shmem_collect64_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void shmem_collect64__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_COLLECT64_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_COLLECT64__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect64(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void shmem_collect32_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void shmem_collect32__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_COLLECT32_(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
}

extern void SHMEM_COLLECT32__(void * a1, const void * a2, size_t * a3, int * a4, int * a5, int * a6, long * a7)
{
   __wrap_shmem_collect32(a1, a2, *a3, *a4, *a5, *a6, a7);
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

extern void shmem_set_lock_(long * a1)
{
   __wrap_shmem_set_lock(a1);
}

extern void shmem_set_lock__(long * a1)
{
   __wrap_shmem_set_lock(a1);
}

extern void SHMEM_SET_LOCK_(long * a1)
{
   __wrap_shmem_set_lock(a1);
}

extern void SHMEM_SET_LOCK__(long * a1)
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

extern void shmem_clear_lock_(long * a1)
{
   __wrap_shmem_clear_lock(a1);
}

extern void shmem_clear_lock__(long * a1)
{
   __wrap_shmem_clear_lock(a1);
}

extern void SHMEM_CLEAR_LOCK_(long * a1)
{
   __wrap_shmem_clear_lock(a1);
}

extern void SHMEM_CLEAR_LOCK__(long * a1)
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

extern int shmem_test_lock_(long * a1)
{
   __wrap_shmem_test_lock(a1);
}

extern int shmem_test_lock__(long * a1)
{
   __wrap_shmem_test_lock(a1);
}

extern int SHMEM_TEST_LOCK_(long * a1)
{
   __wrap_shmem_test_lock(a1);
}

extern int SHMEM_TEST_LOCK__(long * a1)
{
   __wrap_shmem_test_lock(a1);
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

extern void shmemx_short_put_nb_(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_short_put_nb__(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_SHORT_PUT_NB_(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_SHORT_PUT_NB__(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_int_put_nb_(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_int_put_nb__(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_INT_PUT_NB_(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_INT_PUT_NB__(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_long_put_nb_(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_long_put_nb__(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONG_PUT_NB_(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONG_PUT_NB__(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_longlong_put_nb_(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_longlong_put_nb__(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGLONG_PUT_NB_(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGLONG_PUT_NB__(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_longdouble_put_nb_(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_longdouble_put_nb__(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGDOUBLE_PUT_NB_(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGDOUBLE_PUT_NB__(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_double_put_nb_(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_double_put_nb__(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_DOUBLE_PUT_NB_(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_DOUBLE_PUT_NB__(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_float_put_nb_(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_put_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_float_put_nb__(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_FLOAT_PUT_NB_(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_put_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_FLOAT_PUT_NB__(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_put_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_putmem_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_putmem_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_putmem_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_putmem_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUTMEM_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_putmem_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUTMEM_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_putmem_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_put32_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put32_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_put32_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put32_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUT32_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put32_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUT32_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put32_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_put64_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put64_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_put64_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put64_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUT64_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put64_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUT64_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put64_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_put128_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put128_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_put128_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put128_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUT128_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put128_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_PUT128_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_put128_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_short_get_nb_(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_short_get_nb__(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_SHORT_GET_NB_(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_SHORT_GET_NB__(short * a1, const short * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_short_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_int_get_nb_(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_int_get_nb__(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_INT_GET_NB_(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_INT_GET_NB__(int * a1, const int * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_int_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_long_get_nb_(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_long_get_nb__(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONG_GET_NB_(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONG_GET_NB__(long * a1, const long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_long_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_longlong_get_nb_(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_longlong_get_nb__(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGLONG_GET_NB_(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGLONG_GET_NB__(long long * a1, const long long * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longlong_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_longdouble_get_nb_(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_longdouble_get_nb__(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGDOUBLE_GET_NB_(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_LONGDOUBLE_GET_NB__(long double * a1, const long double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_longdouble_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_double_get_nb_(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_double_get_nb__(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_DOUBLE_GET_NB_(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_DOUBLE_GET_NB__(double * a1, const double * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_double_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_float_get_nb_(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_get_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_float_get_nb__(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_FLOAT_GET_NB_(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_get_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_FLOAT_GET_NB__(float * a1, const float * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_float_get_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_getmem_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_getmem_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_getmem_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_getmem_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GETMEM_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_getmem_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GETMEM_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_getmem_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_get32_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get32_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_get32_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get32_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GET32_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get32_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GET32_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get32_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_get64_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get64_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_get64_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get64_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GET64_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get64_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GET64_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get64_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_get128_nb_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get128_nb(a1, a2, *a3, *a4, a5);
}

extern void shmemx_get128_nb__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get128_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GET128_NB_(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get128_nb(a1, a2, *a3, *a4, a5);
}

extern void SHMEMX_GET128_NB__(void * a1, const void * a2, size_t * a3, int * a4, shmemx_request_handle_t * a5)
{
   __wrap_shmemx_get128_nb(a1, a2, *a3, *a4, a5);
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

extern void shmemx_wait_req_(shmemx_request_handle_t * a1)
{
   __wrap_shmemx_wait_req(*a1);
}

extern void shmemx_wait_req__(shmemx_request_handle_t * a1)
{
   __wrap_shmemx_wait_req(*a1);
}

extern void SHMEMX_WAIT_REQ_(shmemx_request_handle_t * a1)
{
   __wrap_shmemx_wait_req(*a1);
}

extern void SHMEMX_WAIT_REQ__(shmemx_request_handle_t * a1)
{
   __wrap_shmemx_wait_req(*a1);
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

extern void shmemx_test_req_(shmemx_request_handle_t * a1, int * a2)
{
   __wrap_shmemx_test_req(*a1, a2);
}

extern void shmemx_test_req__(shmemx_request_handle_t * a1, int * a2)
{
   __wrap_shmemx_test_req(*a1, a2);
}

extern void SHMEMX_TEST_REQ_(shmemx_request_handle_t * a1, int * a2)
{
   __wrap_shmemx_test_req(*a1, a2);
}

extern void SHMEMX_TEST_REQ__(shmemx_request_handle_t * a1, int * a2)
{
   __wrap_shmemx_test_req(*a1, a2);
}


/**********************************************************
   shfree_nb
 **********************************************************/

extern void  __real_shfree_nb(void * a1) ;
extern void  __wrap_shfree_nb(void * a1)  {

  TAU_PROFILE_TIMER(t,"void shfree_nb(void *) C", "", TAU_USER);
  TAU_PROFILE_START(t);
  __real_shfree_nb(a1);
  TAU_PROFILE_STOP(t);

}

extern void shfree_nb_(void * a1)
{
   __wrap_shfree_nb(a1);
}

extern void shfree_nb__(void * a1)
{
   __wrap_shfree_nb(a1);
}

extern void SHFREE_NB_(void * a1)
{
   __wrap_shfree_nb(a1);
}

extern void SHFREE_NB__(void * a1)
{
   __wrap_shfree_nb(a1);
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

extern void shmemx_int_xor_(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_xor(a1, *a2, *a3);
}

extern void shmemx_int_xor__(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_xor(a1, *a2, *a3);
}

extern void SHMEMX_INT_XOR_(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_xor(a1, *a2, *a3);
}

extern void SHMEMX_INT_XOR__(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_xor(a1, *a2, *a3);
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

extern void shmemx_long_xor_(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_xor(a1, *a2, *a3);
}

extern void shmemx_long_xor__(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_xor(a1, *a2, *a3);
}

extern void SHMEMX_LONG_XOR_(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_xor(a1, *a2, *a3);
}

extern void SHMEMX_LONG_XOR__(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_xor(a1, *a2, *a3);
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

extern void shmemx_longlong_xor_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_xor(a1, *a2, *a3);
}

extern void shmemx_longlong_xor__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_xor(a1, *a2, *a3);
}

extern void SHMEMX_LONGLONG_XOR_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_xor(a1, *a2, *a3);
}

extern void SHMEMX_LONGLONG_XOR__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_xor(a1, *a2, *a3);
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

extern int shmemx_int_fetch_(int * a1, int * a2)
{
   __wrap_shmemx_int_fetch(a1, *a2);
}

extern int shmemx_int_fetch__(int * a1, int * a2)
{
   __wrap_shmemx_int_fetch(a1, *a2);
}

extern int SHMEMX_INT_FETCH_(int * a1, int * a2)
{
   __wrap_shmemx_int_fetch(a1, *a2);
}

extern int SHMEMX_INT_FETCH__(int * a1, int * a2)
{
   __wrap_shmemx_int_fetch(a1, *a2);
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

extern long shmemx_long_fetch_(long * a1, int * a2)
{
   __wrap_shmemx_long_fetch(a1, *a2);
}

extern long shmemx_long_fetch__(long * a1, int * a2)
{
   __wrap_shmemx_long_fetch(a1, *a2);
}

extern long SHMEMX_LONG_FETCH_(long * a1, int * a2)
{
   __wrap_shmemx_long_fetch(a1, *a2);
}

extern long SHMEMX_LONG_FETCH__(long * a1, int * a2)
{
   __wrap_shmemx_long_fetch(a1, *a2);
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

extern long long shmemx_longlong_fetch_(long long * a1, int * a2)
{
   __wrap_shmemx_longlong_fetch(a1, *a2);
}

extern long long shmemx_longlong_fetch__(long long * a1, int * a2)
{
   __wrap_shmemx_longlong_fetch(a1, *a2);
}

extern long long SHMEMX_LONGLONG_FETCH_(long long * a1, int * a2)
{
   __wrap_shmemx_longlong_fetch(a1, *a2);
}

extern long long SHMEMX_LONGLONG_FETCH__(long long * a1, int * a2)
{
   __wrap_shmemx_longlong_fetch(a1, *a2);
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

extern void shmemx_int_set_(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_set(a1, *a2, *a3);
}

extern void shmemx_int_set__(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_set(a1, *a2, *a3);
}

extern void SHMEMX_INT_SET_(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_set(a1, *a2, *a3);
}

extern void SHMEMX_INT_SET__(int * a1, int * a2, int * a3)
{
   __wrap_shmemx_int_set(a1, *a2, *a3);
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

extern void shmemx_long_set_(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_set(a1, *a2, *a3);
}

extern void shmemx_long_set__(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_set(a1, *a2, *a3);
}

extern void SHMEMX_LONG_SET_(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_set(a1, *a2, *a3);
}

extern void SHMEMX_LONG_SET__(long * a1, long * a2, int * a3)
{
   __wrap_shmemx_long_set(a1, *a2, *a3);
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

extern void shmemx_longlong_set_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_set(a1, *a2, *a3);
}

extern void shmemx_longlong_set__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_set(a1, *a2, *a3);
}

extern void SHMEMX_LONGLONG_SET_(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_set(a1, *a2, *a3);
}

extern void SHMEMX_LONGLONG_SET__(long long * a1, long long * a2, int * a3)
{
   __wrap_shmemx_longlong_set(a1, *a2, *a3);
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

extern double shmemx_wtime_()
{
   __wrap_shmemx_wtime();
}

extern double shmemx_wtime__()
{
   __wrap_shmemx_wtime();
}

extern double SHMEMX_WTIME_()
{
   __wrap_shmemx_wtime();
}

extern double SHMEMX_WTIME__()
{
   __wrap_shmemx_wtime();
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

extern int shmemx_fence_test_()
{
   __wrap_shmemx_fence_test();
}

extern int shmemx_fence_test__()
{
   __wrap_shmemx_fence_test();
}

extern int SHMEMX_FENCE_TEST_()
{
   __wrap_shmemx_fence_test();
}

extern int SHMEMX_FENCE_TEST__()
{
   __wrap_shmemx_fence_test();
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

extern int shmemx_quiet_test_()
{
   __wrap_shmemx_quiet_test();
}

extern int shmemx_quiet_test__()
{
   __wrap_shmemx_quiet_test();
}

extern int SHMEMX_QUIET_TEST_()
{
   __wrap_shmemx_quiet_test();
}

extern int SHMEMX_QUIET_TEST__()
{
   __wrap_shmemx_quiet_test();
}

